"""Budget-Aware Evaluation（Stage 1.6）：统一 reasoning budget 下公平比较 Base/L0/L2。

把「完整 CoT + EOS」的隐式必要条件重构为 Accuracy(B)——B=max reasoning token budget。
只改 evaluation pipeline；不改训练/loss/cache/L2。复用 eval_aime 的 verifier。

- extract_final_answer(text)：统一答案提取（boxed → Final Answer marker → benchmark parser → fallback）。
- BudgetEvaluator(AimeEvaluator)：预算感知生成（逐位 EOS 判定）+ 双指标（outcome/prefix）+ token 记账。
- run_matrix：Base/L0/L2 × B 矩阵聚合 → md 报告 + 4 图（matplotlib）。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .eval_aime import (
    AimeEvaluator, _extract_boxed_answer, extract_answer, normalize_answer,
    format_prompt,
)
from .exceptions import DataError, ModelError

# 独立、固定、不可修改的 answer-completion evaluation prompt（不计入 reasoning budget B）
ANSWER_COMPLETION_PROMPT = "Based on the reasoning above, provide only the final answer."
# 默认 reasoning budget 档位
DEFAULT_BUDGETS = (256, 512, 1024, 2048, 4096)
# completion 生成上限（answer_completion_tokens 用自己的小预算，与 reasoning budget 分离）
DEFAULT_COMPLETION_MAX_TOKENS = 64
# Final Answer marker 正则（boxed 之后兜底）
_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s*answer|Final Answer|Answer)\s*[:：]\s*([^\n]+)", re.IGNORECASE)


def extract_final_answer(text: str) -> str | None:
    """统一答案提取：\\boxed{} → Final Answer marker → benchmark parser → fallback。

    输入模型在预算内的输出文本（可能 EOS 终止或截断），返回最终答案字符串或 None。
    依 eval_aime 既有 verifier 复用，不新造一套。
    """
    if not text:
        return None
    # 1) boxed 级联（论文级提取，长 CoT 中途草稿不污染）
    b = _extract_boxed_answer(text)
    if b:
        return b
    # 2) Final Answer / Answer marker
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip() or None
    # 3) benchmark parser（eval_aime.extract_answer 的 boxed→数字→末尾数字回退）
    fallback = extract_answer(text, "boxed")
    return fallback or None


# --------------------------- 数据集注册表（GSM8K 基础泛化 / MATH-500 主结果 / AIME 补充） ----
@dataclass(frozen=True)
class DatasetSpec:
    """数据集加载规格。

    - hf：HuggingFace 数据集名
    - split：加载哪个切分（GSM8K/MATH-500 用 test；AIME 用 train）
    - prob / ans：problem/answer 候选列名（大小写兼容，逐条取第一个非 None）
    - gt_extract：ground_truth 规整函数（GSM8K 剥 "#### xxx"；其余默认保留原样）
    """
    hf: str
    split: str = "train"
    prob: tuple[str, ...] = ("problem", "question", "prompt", "Problem", "Question", "Prompt")
    ans: tuple[str, ...] = ("answer", "solution", "Answer", "Solution")
    gt_extract: callable = lambda a: _extract_boxed_answer(a) or str(a).strip()


def _gsm8k_gt(a) -> str:
    """GSM8K answer 形如 "#### 42" / "#### 5.5" → 提取 "#### " 后的规整答案。"""
    s = str(a).split("####")[-1].strip()
    return _extract_boxed_answer(s) or s


# 键（大小写不敏感）→ 加载规格。未知键按直接 HF 数据集名对待（test 切分、默认列）。
DATASET_REGISTRY = {
    "GSM8K":   DatasetSpec("openai/gsm8k", "test",
                           prob=("question", "problem", "Question", "Problem"),
                           ans=("answer", "Answer"),
                           gt_extract=_gsm8k_gt),
    "MATH500": DatasetSpec("HuggingFaceH4/MATH-500", "test"),
    "AIME24":  DatasetSpec("Maxwell-Jia/AIME_2024", "train"),
    "AIME25":  DatasetSpec("yentinglin/aime_2025", "train"),
}


class BudgetEvaluator(AimeEvaluator):
    """预算感知评估器。复用 AimeEvaluator 的模型加载/verifier/load_problems。

    新增：
      - generate_budget(prompts, budget) → (text, status, reasoning_tokens)：逐位 EOS 判定，
        status ∈ {'eos','budget_stop'}；reasoning_tokens = eos 位置（不含 eos）或 budget。
      - evaluate_budget(dataset_ref, budget) → 双指标（outcome/prefix）+ token 记账。
      - _completion(prefix, x)：固定提示 answer-completion（Prefix Evaluation，不计入 budget）。
    """

    def __init__(self, *args, seed: int = 42,
                 completion_max_tokens: int = DEFAULT_COMPLETION_MAX_TOKENS, **kw):
        super().__init__(*args, **kw)
        self.seed = int(seed)
        self.completion_max_tokens = max(1, int(completion_max_tokens))

    def _verify(self, pred: str | None, gt: str) -> bool:
        """verifier：scoring='sympy' 用论文数学等价判定；否则整数精确匹配。复用 eval_aime。"""
        if pred is None or not str(pred).strip():
            return False
        if self.scoring == "sympy":
            return self._grade_sympy(str(pred), gt)
        return normalize_answer(pred) == normalize_answer(gt)

    def resolve_dataset(self, dataset_ref: str) -> DatasetSpec:
        """数据集解析：注册表命中（大小写不敏感）返回规格；否则按直接 HF 名对待。"""
        spec = DATASET_REGISTRY.get(dataset_ref.upper())
        return spec or DatasetSpec(dataset_ref)

    def load_problems(self, dataset_ref: str) -> list[tuple[str, str]]:
        """加载题目 → [(problem, ground_truth)]。支持 GSM8K/MATH-500（test）/AIME（train）。

        ground_truth 经 gt_extract 规整（GSM8K 剥 "#### "），保证进 verifier 可比。
        """
        try:
            from datasets import load_dataset
        except Exception as e:                       # pragma: no cover
            raise ModelError(f"评估需要 datasets：{e}") from e
        spec = self.resolve_dataset(dataset_ref)
        try:
            ds = load_dataset(spec.hf, split=spec.split)
        except Exception as e:
            raise DataError(f"加载数据集 {spec.hf!r} 失败：{e}") from e
        rows = []
        for row in ds:
            prob = next((row.get(k) for k in spec.prob if row.get(k) is not None), None)
            ans = next((row.get(k) for k in spec.ans if row.get(k) is not None), None)
            if prob is None or ans is None:
                raise DataError(
                    f"数据集 {spec.hf!r} 缺 problem/answer 列（实际列：{list(row.keys())}）")
            rows.append((str(prob), spec.gt_extract(ans)))
        if not rows:
            raise DataError(f"数据集 {spec.hf!r} 为空")
        return rows

    def generate_budget(self, prompts: list[str], budget: int) -> list[tuple[str, str, int]]:
        """批量预算感知生成：每 prompt 产出 n_samples 条 (text, status, reasoning_tokens)。

        status='eos'：新 token 含 eos_token_id，reasoning_tokens = eos 位置（不含 eos）。
        status='budget_stop'：撞 budget cap，reasoning_tokens = budget。
        复用 generation_smoke.py 的逐位 EOS 判定；左填充沿用 AimeEvaluator.__init__。
        """
        import torch
        import math
        n = self.n_samples
        do_sample = self.temperature > 0 and n > 1
        chunk = n if n <= 1 else max(1, min(self.batch_size, n))
        step = self.batch_size if n <= 1 else max(1, self.batch_size // chunk)
        n_chunks = max(1, math.ceil(n / chunk)) if n > 1 else 1
        # 固定 seed 保证各 budget/模型可复现（相同 seed 复用）
        if n <= 1:
            torch.manual_seed(self.seed)
        eos = self.tok.eos_token_id
        out_rows: list[tuple[str, str, int]] = []
        for i in range(0, len(prompts), step):
            batch = prompts[i:i + step]
            if self.chat_template:
                batch = [self.tok.apply_chat_template(
                    [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                    for p in batch]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=max(1, self.max_ctx - budget))
            enc = {k: v.to(self.device) for k, v in enc.items()}
            seq_len = enc["input_ids"].size(1)
            for _c in range(n_chunks):
                with torch.no_grad():
                    gen_kwargs = dict(
                        max_new_tokens=budget, do_sample=do_sample,
                        num_return_sequences=chunk,
                        temperature=self.temperature if do_sample else 1.0,
                        pad_token_id=self.tok.pad_token_id, eos_token_id=eos)
                    if do_sample and self.top_p is not None:
                        gen_kwargs["top_p"] = self.top_p
                    gen = self.model.generate(**enc, **gen_kwargs)
                for o in gen:
                    new = o[seq_len:].tolist()
                    if eos in new:
                        status = "eos"
                        rt = new.index(eos)                 # 不含 eos
                        text = self.tok.decode(new[:rt], skip_special_tokens=True)
                    else:
                        status = "budget_stop"
                        rt = len(new)
                        text = self.tok.decode(new, skip_special_tokens=True)
                    out_rows.append((text, status, rt))
        return out_rows

    def _completion(self, prefix: str, problem_x: str) -> tuple[str, int]:
        """对预算内 reasoning prefix 跑独立 answer-completion（Prefix Evaluation）。

        输入 = 原始 prompt + prefix_B + 固定 completion 提示；生成短答案。
        返回 (completion_text, answer_completion_tokens)。completion 不计入 reasoning budget B。
        """
        import torch
        torch.manual_seed(self.seed)          # 与 reasoning 相同 seed，可复现
        # 复用原始 prompt 文本（chat_template 可选包裹），再拼 prefix + 固定 completion 提示
        prompt = format_prompt(problem_x, self.prompt_style)
        if self.chat_template:
            prompt = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        full = prompt + "\n" + prefix + "\n\n" + ANSWER_COMPLETION_PROMPT
        enc = self.tok([full], return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_ctx)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        seq_len = enc["input_ids"].size(1)
        with torch.no_grad():
            gen = self.model.generate(
                **enc, max_new_tokens=self.completion_max_tokens, do_sample=False,
                temperature=1.0, pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id)
        new = gen[0][seq_len:].tolist()
        eos = self.tok.eos_token_id
        if eos in new:
            new = new[:new.index(eos)]
        return self.tok.decode(new, skip_special_tokens=True), len(new)

    def evaluate_budget(self, dataset_ref: str, budget: int) -> dict:
        """预算感知评估单数据集单预算。返回聚合字典（含每样本 rows 供落盘/审计）。

        指标：
          - outcome_correct：verifier(extract_final_answer(预算内文本)) 正确 → Accuracy@B
          - prefix_correct：无 final answer 样本的 completion 正确 → PrefixAccuracy@B
          - status/eos、budget_stop、reasoning_tokens、answer_completion_tokens、total_tokens
        """
        problems = self.load_problems(dataset_ref)
        prompts = [format_prompt(p, self.prompt_style) for p, _ in problems]
        n = self.n_samples
        rows = []
        n_outcome = n_prefix_denominator = n_prefix_correct = n_eos = 0
        rt_sum = 0
        for i, (problem, gt) in enumerate(problems):
            group = self.generate_budget([prompts[i]], budget)   # → n 条 (text,status,rt)
            for (text, status, rt) in group:
                fa = extract_final_answer(text)
                outcome_ok = self._verify(fa, gt)
                n_outcome += int(outcome_ok)
                n_eos += int(status == "eos")
                rt_sum += rt
                # Prefix：仅无 final answer 样本跑 completion
                prefix_ok = None
                completion_text = ""
                act = 0
                if fa is None:
                    n_prefix_denominator += 1
                    completion_text, act = self._completion(text, problem)
                    cfa = extract_final_answer(completion_text)
                    prefix_ok = self._verify(cfa, gt)
                    n_prefix_correct += int(prefix_ok)
                rows.append({
                    "problem_id": i, "dataset": dataset_ref, "budget": budget,
                    "ground_truth": gt, "status": status,
                    "reasoning_tokens": rt, "answer_completion_tokens": act,
                    "total_tokens": rt + act,
                    "has_final_answer": fa is not None, "final_answer": fa,
                    "outcome_correct": outcome_ok, "prefix_correct": prefix_ok,
                    "response": text, "completion": completion_text,
                })
        Nn = len(rows)
        return {
            "dataset": dataset_ref, "budget": budget, "model_path": self.model_path,
            "n_samples": n, "n": Nn,
            "accuracy": n_outcome / Nn if Nn else 0.0,
            "prefix_accuracy": (n_prefix_correct / n_prefix_denominator)
            if n_prefix_denominator else None,
            "no_answer_rate": n_prefix_denominator / Nn if Nn else 0.0,
            "eos_rate": n_eos / Nn if Nn else 0.0,
            "budget_stop_rate": (Nn - n_eos) / Nn if Nn else 0.0,
            "avg_reasoning_tokens": rt_sum / Nn if Nn else 0.0,
            "rows": rows,
        }


def run_matrix(models: list[tuple[str, str]], budgets: list[int],
               datasets: list[str], out_dir: str,
               device: str = "cpu", temperature: float = 0.0, top_p: float | None = None,
               n_samples: int = 1, seed: int = 42, scoring: str = "sympy",
               prompt_style: str = "boxed", chat_template: bool = False,
               attn_implementation: str | None = None, batch_size: int = 8,
               dtype: str = "auto",
               completion_max_tokens: int = DEFAULT_COMPLETION_MAX_TOKENS,
               ) -> list[dict]:
    """对 (label, model_path) × budget 跑矩阵，逐 (model, budget) 落盘累加 jsonl + 聚合结果。

    models：如 [("Base", "/path/Qwen3-1.7B"), ("L0", ""), ("L2", "")]——label 的空路径表示占位跳过。
    每模型加载一次 BudgetEvaluator，跑全部 budgets（省重复加载）。
    返回所有 budget-results（含 rows 的聚合摘要），并写 <out_dir>/<label>__<dataset>__B<budget>.jsonl。
    """
    os.makedirs(out_dir, exist_ok=True)
    all_results: list[dict] = []
    for label, path in models:
        if not path:
            print(f"[budget-eval] {label}: 无模型路径，占位跳过")
            continue
        with BudgetEvaluator(
                path, device=device, batch_size=batch_size, n_samples=n_samples,
                temperature=temperature, top_p=top_p, scoring=scoring,
                prompt_style=prompt_style, chat_template=chat_template,
                attn_implementation=attn_implementation, dtype=dtype,
                seed=seed, completion_max_tokens=completion_max_tokens) as ev:
            for ds in datasets:
                for B in budgets:
                    res = ev.evaluate_budget(ds, B)
                    res["label"] = label
                    all_results.append(res)
                    # 落盘每样本
                    out_path = os.path.join(out_dir, f"{label}__{ds}__B{B}.jsonl")
                    with open(out_path, "w", encoding="utf-8") as f:
                        for r in res["rows"]:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(f"[budget-eval] {label} {ds} B{B}: "
                          f"Acc={res['accuracy']:.3f} Prefix={res['prefix_accuracy']} "
                          f"EOS={res['eos_rate']:.3f} avgRT={res['avg_reasoning_tokens']:.0f}")
    return all_results


def write_report(all_results: list[dict], report_path: str) -> str:
    """写 Budget-Aware Evaluation 决策报告（md 表 + 4 图），返回报告 markdown 文本。"""
    base = os.path.dirname(report_path) or "."
    os.makedirs(base, exist_ok=True)
    plots = _write_plots(all_results, base)
    lines = ["# Stage 1.6 Budget-Aware Evaluation 决策报告", ""]
    lines.append("> 协议：统一 reasoning budget B∈{256,512,1024,2048,4096} 下公平比较 Base/L0/L2。")
    lines.append("> `Accuracy@B`=outcome（预算内自然产出正确最终答案）；`PrefixAccuracy@B`=solvability"
                 "（仅预算内无 final answer 样本经固定提示 answer-completion 得正确）。")
    lines.append("> `status`∈{eos,budget_stop} 显式区分；`reasoning_tokens` 与 `answer_completion_tokens` 分离。")
    lines.append("")
    lines.append("| Model | Budget | Accuracy | PrefixAccuracy | EOS | BudgetStop | AvgReasoningTokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(all_results, key=lambda r: (r.get("label", ""), r["budget"])):
        pa = "-" if r["prefix_accuracy"] is None else f"{r['prefix_accuracy']:.3f}"
        lines.append(f"{r.get('label','')}|{r['budget']}|{r['accuracy']:.3f}|{pa}|"
                     f"{r['eos_rate']:.3f}|{r['budget_stop_rate']:.3f}|{r['avg_reasoning_tokens']:.0f}")
    lines.append("")
    if plots:
        lines.append("## 图")
        lines.append("")
        captions = {
            "accuracy_vs_budget.png": "1. Accuracy vs Reasoning Budget",
            "prefix_accuracy_vs_budget.png": "2. PrefixAccuracy vs Reasoning Budget",
            "eos_rate_vs_budget.png": "3. EOS Rate vs Reasoning Budget",
            "avg_rt_vs_accuracy.png": "4. Average Reasoning Tokens vs Accuracy",
        }
        for fname, cap in captions.items():
            if fname in plots:
                lines.append(f"![{cap}]({fname})  \n*{cap}*")
        lines.append("")
    else:
        lines.append("> matplotlib 未装，图跳过。")
        lines.append("")
    md = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return md


def _write_plots(all_results: list[dict], out_dir: str) -> list[str]:
    """生成 4 张 matplotlib 图到 out_dir，返回成功生成的 PNG 文件名列表。"""
    if not all_results:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                       # pragma: no cover
        return []
    # 按 label 分组、budget 排序
    labels = sorted({r["label"] for r in all_results})
    budgets = sorted({r["budget"] for r in all_results})
    series = {lab: {r["budget"]: r for r in all_results if r["label"] == lab}
              for lab in labels}
    written: list[str] = []

    def _line(ax, getter, fname, ylabel, title, skip_none=False):
        for lab in labels:
            xs, ys = [], []
            for B in budgets:
                r = series[lab].get(B)
                if r is None:
                    continue
                v = getter(r)
                if skip_none and v is None:
                    continue
                xs.append(B)
                ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", label=lab)
        ax.set_xlabel("Reasoning Budget B")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = os.path.join(out_dir, fname)
        plt.savefig(p, dpi=110, bbox_inches="tight")
        plt.clf()
        written.append(fname)

    _line(plt.gca(), lambda r: r["accuracy"], "accuracy_vs_budget.png",
          "Accuracy@B", "1. Accuracy vs Reasoning Budget")
    _line(plt.gca(), lambda r: r["prefix_accuracy"], "prefix_accuracy_vs_budget.png",
          "PrefixAccuracy@B", "2. PrefixAccuracy vs Reasoning Budget", skip_none=True)
    _line(plt.gca(), lambda r: r["eos_rate"], "eos_rate_vs_budget.png",
          "EOS Rate", "3. EOS Rate vs Reasoning Budget")
    # 4) Average Reasoning Tokens vs Accuracy（散点，每 label 一色）
    for lab in labels:
        xs, ys = [], []
        for B in budgets:
            r = series[lab].get(B)
            if r is None:
                continue
            xs.append(r["avg_reasoning_tokens"])
            ys.append(r["accuracy"])
        if xs:
            plt.plot(xs, ys, marker="o", label=lab)
    plt.xlabel("Average Reasoning Tokens")
    plt.ylabel("Accuracy@B")
    plt.title("4. Average Reasoning Tokens vs Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "avg_rt_vs_accuracy.png"), dpi=110,
                bbox_inches="tight")
    plt.clf()
    written.append("avg_rt_vs_accuracy.png")
    return written


__all__ = ["BudgetEvaluator", "extract_final_answer", "run_matrix", "write_report",
           "DatasetSpec", "DATASET_REGISTRY",
           "ANSWER_COMPLETION_PROMPT", "DEFAULT_BUDGETS", "DEFAULT_COMPLETION_MAX_TOKENS"]