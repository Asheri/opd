"""AIME 评估器（main/ 一等能力，替代原 benchmark 对 async-opd opd.cli.eval 的依赖）。

main/ 是真正主项目：真实模型（HF 权重）在 AIME24/AIME25 上的评估在此自包含实现——
模型加载走 transformers（AutoModelForCausalLM + AutoTokenizer，本地路径 / HF id），
数据集走 huggingface datasets，答案提取用 \boxed{} 级联 → 整数，评分精确匹配。

- `extract_answer(text)` / `normalize_answer(a)`：纯函数，无模型依赖，可单测。
- `AimeEvaluator`：真实评估器（model load + dataset + generate + score + jsonl 落盘）。
- CLI 入口见 cli.py 的 `eval-aime` 子命令；`--run-dir` 桥接读 run_dir/config.yaml
  的真实模型路径（eval.model_path），toy run 目录抛 DataError 明确提示。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .exceptions import DataError, ModelError

# AIME 数据集别名 → HF dataset 名（列：problem + answer，答案整数）
AIME_DATASETS = {
    "AIME24": "Maxwell-Jia/AIME_2024",
    "AIME25": "yentinglin/aime_2025",
}
DEFAULT_DATASETS = ("AIME24", "AIME25")

# 标准推理提示（与 async-opd eval 一致；答案必须放 \boxed{}）
PROMPT_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


# --------------------------- 纯函数（可单测） ---------------------------
def format_prompt(problem: str) -> str:
    """把 AIME 题目格式化为推理 prompt。"""
    return PROMPT_TEMPLATE.format(problem=str(problem).strip())


def extract_answer(text: str) -> str:
    """从模型输出提取数值答案。

    级联：\boxed{...}（支持嵌套括号）→ 其中第一个数字；否则回退最后一个数字。
    返回原始字符串（含可能的负号/千分位），未找到返回 ""。
    """
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

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


class AimeEvaluator:
    """真实模型 AIME 评估器（transformers 后端）。"""

    def __init__(self, model_path: str, device: str = "cpu",
                 max_new_tokens: int = 2048, batch_size: int = 8,
                 n_samples: int = 1, temperature: float = 0.0,
                 trust_remote_code: bool = False, dtype: str = "auto"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:                       # pragma: no cover
            raise ModelError(f"AIME 评估需要 transformers：{e}") from e
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.n_samples = int(n_samples)
        self.temperature = float(temperature)
        try:
            self.tok = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            raise ModelError(
                f"加载 tokenizer {model_path!r} 失败（路径/HF id 无效？）：{e}") from e
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {}
        if dtype in ("bf16", "float16"):
            kwargs["torch_dtype"] = {"bf16": "bfloat16", "float16": "float16"}[dtype]
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **kwargs).to(device).eval()
        except Exception as e:
            raise ModelError(
                f"加载模型 {model_path!r} 失败（路径/HF id 无效？）：{e}") from e

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
        rows = []
        for row in ds:
            prob = row.get("problem") or row.get("question") or row.get("prompt")
            ans = row.get("answer")
            if prob is None or ans is None:
                raise DataError(
                    f"数据集 {name!r} 缺 problem/answer 列（实际列：{list(row.keys())}）")
            rows.append((str(prob), str(ans)))
        if not rows:
            raise DataError(f"数据集 {name!r} 为空")
        return rows

    # --------------------------- 生成 ---------------------------
    def generate(self, prompts: list[str]) -> list[str]:
        """批量贪心/采样生成 response 文本。"""
        import torch
        responses: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096 - self.max_new_tokens)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            do_sample = self.temperature > 0
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.temperature if do_sample else 1.0,
                    pad_token_id=self.tok.pad_token_id)
            for o in out:
                resp = self.tok.decode(o[enc["input_ids"].size(1):],
                                       skip_special_tokens=True)
                responses.append(resp)
        return responses

    # --------------------------- 评估 ---------------------------
    def evaluate(self, dataset_ref: str) -> AimeResult:
        """在单个 AIME 数据集上评估，返回 AimeResult。"""
        problems = self.load_problems(dataset_ref)
        correct = 0
        rows = []
        for i, (problem, gt) in enumerate(problems):
            resp = self.generate([format_prompt(problem)])[0]
            pred = extract_answer(resp)
            ok = normalize_answer(pred) == normalize_answer(gt)
            correct += int(ok)
            rows.append({
                "problem_id": i, "dataset": dataset_ref,
                "ground_truth": gt, "predicted": pred,
                "correct": ok, "response": resp,
            })
        return AimeResult(dataset=dataset_ref, model_path=self.model_path,
                          correct=correct, total=len(problems), rows=rows)

    def evaluate_to_jsonl(self, dataset_ref: str, out_path: str) -> AimeResult:
        """评估并落盘每样本 jsonl，返回结果。"""
        res = self.evaluate(dataset_ref)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in res.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return res


__all__ = ["AimeEvaluator", "AimeResult", "extract_answer", "normalize_answer",
           "format_prompt", "AIME_DATASETS", "DEFAULT_DATASETS"]