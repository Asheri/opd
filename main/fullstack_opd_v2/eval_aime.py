"""AIME 评估器（main/ 一等能力，替代原 benchmark 对 async-opd opd.cli.eval 的依赖）。

main/ 是真正主项目：真实模型（HF 权重）在 AIME24/AIME25 上的评估在此自包含实现——
模型加载走 transformers（AutoModelForCausalLM + AutoTokenizer，本地路径 / HF id），
数据集走 huggingface datasets，答案提取用 \boxed{} 级联 → 整数，评分精确匹配。

- `extract_answer(text)` / `normalize_answer(a)`：纯函数，无模型依赖，可单测。
- `AimeEvaluator`：真实评估器（model load + dataset + generate + score + jsonl 落盘）。
- CLI 入口见 cli.py 的 `eval-aime` 子命令；`--run-dir` 桥接读 run_dir/config.yaml
  的 `eval.*` 配置（model_path / max_new_tokens / n_samples / temperature）。
"""

from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import dataclass

from .exceptions import ConfigError, DataError, ModelError, TrainingError

# AIME 数据集别名 → HF dataset 名（列：problem + answer，答案整数）
AIME_DATASETS = {
    "AIME24": "Maxwell-Jia/AIME_2024",
    "AIME25": "yentinglin/aime_2025",
}
DEFAULT_DATASETS = ("AIME24", "AIME25")
# 生成侧上下文上限（prompt + max_new 之和不得超过；与 transformers 默认 4096 对齐）
_MAX_CONTEXT = 4096

# 标准推理提示（答案必须放 \boxed{}）
PROMPT_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
# DAPO 风格模板（Direct-OPD 论文附录 A：训练 rollouts 与评估 prompts 同用）——
# 要求最后一行 "Answer: <数字>"。对齐论文评估协议时用 prompt_style="dapo"。
PROMPT_TEMPLATE_DAPO = (
    "Solve the following math problem step by step.\n"
    "The last line of your response should be of the form\n"
    "Answer:\n"
    "$Answer (without quotes) where $Answer is the answer to the problem.\n"
    "{problem}\n"
    "Remember to put your answer on its own line after \"Answer:\"."
)


# --------------------------- 纯函数（可单测） ---------------------------
def format_prompt(problem: str, style: str = "boxed") -> str:
    """把 AIME 题目格式化为推理 prompt。

    style="boxed"（默认）→ \boxed{} 模板；style="dapo" → Direct-OPD 论文附录 A 的
    DAPO 模板（"Answer:" 结尾行），对齐论文评估协议。
    """
    tpl = PROMPT_TEMPLATE_DAPO if style == "dapo" else PROMPT_TEMPLATE
    return tpl.format(problem=str(problem).strip())


def extract_answer(text: str, style: str = "boxed") -> str:
    """从模型输出提取数值答案。

    style="dapo"：优先取 "Answer:" 行后的数字（论文 DAPO 模板的落点）；
    否则级联 \boxed{...} → 其中第一个数字；再回退最后一个数字。
    返回原始字符串（含可能的负号/千分位），未找到返回 ""。
    """
    if style == "dapo":
        m = re.search(r"[Aa]nswer\s*:\s*([^\n]+)", text)
        if m:
            mm = re.search(r"-?\d[\d,]*", m.group(1))
            if mm:
                return mm.group(0)
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        for cand in reversed(boxed):
            m = re.search(r"-?\d[\d,]*", cand)
            if m:
                return m.group(0)
        return boxed[-1].strip()
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1] if nums else ""


def normalize_answer(a) -> int | None:
    """把答案规范化成整数（AIME 答案为 3 位整数，005 与 5 等价）。"""
    if a is None:
        return None
    s = str(a).strip().replace(",", "")
    if not re.fullmatch(r"-?\d+", s):
        return None
    return int(s)


# --------------------------- 评估器 ---------------------------
@dataclass
class AimeResult:
    dataset: str
    model_path: str
    correct: int
    total: int
    rows: list[dict]
    # metric=ave 时：每题 n 采样中答对比例的均值（对齐论文 ave@32 口径）。
    # None → pass@1 口径（correct/total）。
    ave_accuracy: float | None = None

    @property
    def accuracy(self) -> float:
        if self.ave_accuracy is not None:
            return self.ave_accuracy
        return self.correct / self.total if self.total else 0.0


class AimeEvaluator:
    """真实模型 AIME 评估器（transformers 后端）。

    n_samples>1 时对每题采样 N 条，`correct` 记 pass@1（任一采样答对即对）。
    """

    def __init__(self, model_path: str, device: str = "cpu",
                 max_new_tokens: int = 2048, batch_size: int = 8,
                 n_samples: int = 1, temperature: float = 0.0,
                 trust_remote_code: bool = False, dtype: str = "auto",
                 top_p: float | None = None,
                 metric: str = "pass1",
                 prompt_style: str = "boxed"):
        # P2：参数校验前置（transformers 导入/模型加载之前），配置错快速失败、零副作用。
        # 上下文上限按模型 config 动态取（Qwen3=40960，对齐论文 MAX_VAL_RESP_LENGTH 31744）；
        # 模型加载后才得知，故保守前置校验用 _MAX_CONTEXT（历史默认 4096）挡明显非法值，
        # 模型加载后 _resolve_max_context 按实际 config 复核（可放宽）。
        if int(max_new_tokens) >= _MAX_CONTEXT and int(max_new_tokens) > 32768:
            raise ConfigError(
                f"max_new_tokens={max_new_tokens} 异常大（>32768）；请检查")
        self.n_samples = max(1, int(n_samples))
        self.temperature = float(temperature)
        self.top_p = float(top_p) if top_p is not None else None
        if metric not in ("pass1", "ave"):
            raise ConfigError(f"metric={metric!r} 非法：须 pass1 | ave（ave=论文 ave@32 平均正确率）")
        self.metric = metric
        if prompt_style not in ("boxed", "dapo"):
            raise ConfigError(f"prompt_style={prompt_style!r} 非法：须 boxed | dapo（论文模板）")
        self.prompt_style = prompt_style
        # P2（R2 审查）：greedy + 多采样会退化为 num_return_sequences>1 的重复/崩溃
        # （temperature<=0 → do_sample=False → 同种子多序列逐字重复，pass@1 被污染）。
        # 采样模式由温度驱动：n>1 且 T>0 才合法；要么 T>0（真采样），要么 n==1（贪心）。
        if self.n_samples > 1 and self.temperature <= 0:
            raise ConfigError(
                f"n_samples={self.n_samples}>1 但 temperature={self.temperature}<=0："
                "贪心解码下多序列逐字重复，pass@1 无意义。请设 temperature>0 或 n_samples=1")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:                       # pragma: no cover
            raise ModelError(f"AIME 评估需要 transformers：{e}") from e
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = max(1, int(batch_size))
        # 上下文上限：模型 config 的 max_position_embeddings（Qwen3=40960，对齐论文长生成）。
        # 模型加载后从 config 取；缺省回退 _MAX_CONTEXT。
        self.max_ctx = _MAX_CONTEXT
        try:
            self.tok = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            raise ModelError(
                f"加载 tokenizer {model_path!r} 失败（路径/HF id 无效？）：{e}") from e
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # dtype：显式 bf16/float16 用对应精度；'auto' 在 CUDA 上默认 bf16（现代卡），CPU 用 fp32。
        if dtype in ("bf16", "float16"):
            torch_dtype = {"bf16": "bfloat16", "float16": "float16"}[dtype]
        elif dtype == "auto" and str(device).startswith("cuda"):
            torch_dtype = "bfloat16"
        else:
            torch_dtype = None
        kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **kwargs).to(device).eval()
        except Exception as e:
            raise ModelError(
                f"加载模型 {model_path!r} 失败（路径/HF id 无效？）：{e}") from e
        # 模型加载后按真实 config 复核上下文上限（可放宽到 max_position_embeddings，
        # 对齐论文 MAX_VAL_RESP_LENGTH=31744；缺省回退 4096）。
        # ⚠️ 只接受真实 int（mock/None 跳过）：测试的 fake model config 是 Mock 对象。
        mpe = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(mpe, int) and mpe > 1:
            self.max_ctx = mpe
        if self.max_new_tokens >= self.max_ctx:
            raise ConfigError(
                f"max_new_tokens={self.max_new_tokens} ≥ 模型上下文上限 {self.max_ctx}；请调小")

    def close(self):
        """释放模型/tokenizer 与 GPU 显存。"""
        if hasattr(self, "model"):
            try:
                self.model.to("cpu")
            except Exception:
                pass
            del self.model
        if hasattr(self, "tok"):
            del self.tok
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --------------------------- 数据 ---------------------------
    def resolve_dataset(self, dataset_ref: str) -> str:
        return AIME_DATASETS.get(dataset_ref, dataset_ref)

    def load_problems(self, dataset_ref: str) -> list[tuple[str, str]]:
        """加载 AIME 题目 → [(problem, answer)]。"""
        try:
            from datasets import load_dataset
        except Exception as e:                       # pragma: no cover
            raise ModelError(f"AIME 评估需要 datasets：{e}") from e
        name = self.resolve_dataset(dataset_ref)
        try:
            ds = load_dataset(name, split="train")
        except Exception as e:
            raise DataError(f"加载 AIME 数据集 {name!r} 失败：{e}") from e
        # 列名兼容：不同 AIME 数据集列名不一（problem/Problem/Question/Prompt、
        # answer/Answer/Solution）——大小写不敏感 + 候选键匹配（部署实测
        # Maxwell-Jia/AIME_2024 是大写 Problem/Solution/Answer）。
        prob_keys = ("problem", "question", "prompt", "Problem", "Question", "Prompt")
        ans_keys = ("answer", "solution", "Answer", "Solution")
        rows = []
        for row in ds:
            prob = next((row.get(k) for k in prob_keys if row.get(k) is not None), None)
            ans = next((row.get(k) for k in ans_keys if row.get(k) is not None), None)
            if prob is None or ans is None:
                raise DataError(
                    f"数据集 {name!r} 缺 problem/answer 列（实际列：{list(row.keys())}）")
            rows.append((str(prob), str(ans)))
        if not rows:
            raise DataError(f"数据集 {name!r} 为空")
        return rows

    # --------------------------- 生成 ---------------------------
    def generate(self, prompts: list[str]) -> list[str]:
        """批量贪心/采样生成；每个 prompt 产出 n_samples 条响应（拍平返回）。

        批量 tokenize + 一次 model.generate（num_return_sequences=n_samples），
        避免逐 prompt 多次模型调用（R1 性能修复）。
        """
        import torch
        responses: list[str] = []
        n = self.n_samples
        do_sample = self.temperature > 0 and n > 1
        # P2（R2 审查）：num_return_sequences=n 把每批序列数放大 n 倍，峰值 KV/显存
        # 随 batch×n 线性涨。按 n 收缩批内 prompt 数，把每批生成序列数压回 batch_size
        # 量级（n 感知批缩放；batch_size<n 时退化为逐 prompt，仍受 n 保护）。
        step = self.batch_size if n <= 1 else max(1, self.batch_size // n)
        try:
            for i in range(0, len(prompts), step):
                batch = prompts[i:i + step]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True,
                               max_length=max(1, self.max_ctx - self.max_new_tokens))
                enc = {k: v.to(self.device) for k, v in enc.items()}
                seq_len = enc["input_ids"].size(1)
                with torch.no_grad():
                    gen_kwargs = dict(
                        max_new_tokens=self.max_new_tokens,
                        do_sample=do_sample, num_return_sequences=n,
                        temperature=self.temperature if do_sample else 1.0,
                        pad_token_id=self.tok.pad_token_id)
                    if do_sample and self.top_p is not None:
                        gen_kwargs["top_p"] = self.top_p
                    out = self.model.generate(**enc, **gen_kwargs)
                for o in out:
                    responses.append(self.tok.decode(o[seq_len:],
                                                     skip_special_tokens=True))
        except Exception as e:
            raise TrainingError(f"AIME 生成失败：{e}") from e
        return responses

    # --------------------------- 评估 ---------------------------
    def evaluate(self, dataset_ref: str) -> AimeResult:
        """在单个 AIME 数据集上评估。

        - metric="pass1"（默认）：n_samples>1 记 pass@1（任一采样答对即对）。
        - metric="ave"（对齐论文 ave@32）：accuracy = 每题 n 采样中答对比例的均值。
        """
        problems = self.load_problems(dataset_ref)
        prompts = [format_prompt(p, self.prompt_style) for p, _ in problems]
        responses = self.generate(prompts)          # 拍平：N × n_samples 条
        n = self.n_samples
        correct = 0
        fracs: list[float] = []
        rows = []
        for i, (problem, gt) in enumerate(problems):
            group = responses[i * n:(i + 1) * n]
            preds = [extract_answer(r, self.prompt_style) for r in group]
            gt_n = normalize_answer(gt)
            per = [normalize_answer(p) == gt_n for p in preds]
            ok = any(per)
            correct += int(ok)
            if self.metric == "ave" and n:
                fracs.append(sum(per) / n)
            rows.append({
                "problem_id": i, "dataset": dataset_ref,
                "ground_truth": gt, "predicted": preds[0],
                "correct": ok, "response": group[0],
                "n_samples": n,
            })
        ave = (sum(fracs) / len(fracs)) if fracs else None
        return AimeResult(dataset=dataset_ref, model_path=self.model_path,
                          correct=correct, total=len(problems), rows=rows,
                          ave_accuracy=ave)

    def evaluate_to_jsonl(self, dataset_ref: str, out_path: str) -> AimeResult:
        """评估并落盘每样本 jsonl，返回结果。"""
        res = self.evaluate(dataset_ref)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in res.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return res


__all__ = ["AimeEvaluator", "AimeResult", "extract_answer", "normalize_answer",
           "format_prompt", "AIME_DATASETS", "DEFAULT_DATASETS",
           "PROMPT_TEMPLATE", "PROMPT_TEMPLATE_DAPO"]