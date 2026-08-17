"""可插拔数据接口：把 pipeline 里硬编码的 `_make_toy_data` 抽成 DataLoader。

- `DataLoader`(ABC)：统一 `load()` 契约，返回 `(prompts, responses, reward_fn)`。
  - prompts/responses: (N,P)/(N,T) 设备常驻张量（与 v2 其余部分一致）。
  - reward_fn: `(B,T) token` -> `(B,T) 奖励` 的可调用对象（Stage 0 REINFORCE 用）。
- `ToyDataLoader`：默认实现，与旧 `_make_toy_data` 数值完全同源（randint 同流 + 查找表）。
- `JsonLinesDataLoader`：真实数据接口——读 jsonl（prompt/response 文本）→ tokenizer
  编码成定长 id 张量（供 model_kind='hf' 真实训练）。
- `build_data_loader`：按 `cfg["dataset"]["type"]` 分发，未知类型抛 `DataError`。
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import torch

from .exceptions import DataError


class DataLoader(ABC):
    """数据加载契约。子类必须实现 `load()`。"""

    @abstractmethod
    def load(self):
        """返回 (prompts, responses, reward_fn)。"""


class ToyDataLoader(DataLoader):
    """toy 默认实现：随机生成 prompt/response + 偶数 token 奖励查找表。

    与旧 `FullStackOPDv2._make_toy_data` 完全同源（同 seed 0 的 RNG 流、同查找表），
    保证重构后 Stage 0 数值不变。
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.vocab = cfg["vocab_size"]
        self.n_prompts = cfg["n_prompts"]
        self.prompt_len = cfg["prompt_len"]
        self.resp_len = cfg["resp_len"]
        self.device = device
        self._cache = None   # C4：首次 load 后缓存，避免重复 randint 重建

    def load(self):
        if self._cache is not None:
            return self._cache
        rng = torch.Generator().manual_seed(0)
        prompts = torch.randint(0, self.vocab,
                                (self.n_prompts, self.prompt_len), generator=rng)
        responses = torch.randint(0, self.vocab,
                                  (self.n_prompts, self.resp_len), generator=rng)
        lut = torch.full((self.vocab,), -0.2, dtype=torch.float32)
        lut[0::2] = 1.0
        lut = lut.to(self.device)
        self._cache = (prompts.to(self.device), responses.to(self.device),
                       lambda r: lut[r])
        return self._cache


class JsonLinesDataLoader(DataLoader):
    """真实数据接口：从 jsonl 读 `prompt`/`response` 文本 → tokenizer 编码定长 id 张量。

    供 model_kind='hf' 真实训练（Stage 0 跳过，teacher 为预下载对，见 pipeline.
    _stage0_teachers）。实现文本→token：
    - tokenizer 来自 `dataset.tokenizer_path`（默认回退顶层 `student_path`——
      学生与数据需同词表，与 teacher 一致性语义一致）；
    - prompt 截断/右 pad 到 `max_prompt_len`，response 截断/右 pad 到 `max_response_len`；
    - reward_fn：HF 路径不用（Stage 0 跳过），占位返回 0。

    ⚠️ 已知简化（v2 scheduler 假设 responses 等长、mask=None 恒全 1）：右 pad 的
    pad token 位置会计入损失——Δ_T 在 pad token 上近似中性、且数学数据多数接近定长，
    影响小；严格做法需回传 padding mask（scheduler._train_step 注释已提示）。
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        ds = cfg["dataset"]
        self.path = ds.get("path")
        self.prompt_key = ds.get("prompt_key", "prompt")
        self.response_key = ds.get("response_key", "response")
        self.max_prompt_len = int(ds.get("max_prompt_len", 256))
        self.max_response_len = int(ds.get("max_response_len", 384))
        self.tokenizer_path = ds.get("tokenizer_path") or cfg.get("student_path")
        # 2026-08-17 根因（rollout 质量）：Qwen3 是 chat 模型，裸数学题 prompt 不套
        # <|im_start|> 模板会生成乱码+loop（实测裸 prompt "*. 202951173." vs 套模板
        # "We are given a system of six linear equations..."）。apply_chat_template=true
        # 时把 prompt 包成 user 角色再编码。⚠️ 会改变 prompt token → 预建 teacher cache
        # 必须用同配置重建（否则 Δ_T 错位）。默认 false 保护现有 cache。
        self.apply_chat_template = bool(ds.get("apply_chat_template", False))
        self.device = device
        self._cache = None

    def load(self):
        if self._cache is not None:
            return self._cache
        if not self.path or not os.path.isfile(self.path):
            raise DataError(f"dataset.path 不存在或未配置: {self.path!r}")
        try:
            from transformers import AutoTokenizer
        except Exception as e:                       # pragma: no cover
            raise DataError(f"jsonl 数据加载需要 transformers：{e}") from e
        if not self.tokenizer_path:
            raise DataError(
                "dataset.tokenizer_path 或顶层 student_path 未配置（tokenizer 需与词表对齐）")
        tok = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None:
            raise DataError("tokenizer 无 pad_token，无法定长 pad")
        P, T = self.max_prompt_len, self.max_response_len
        prompts_ids: list[list[int]] = []
        responses_ids: list[list[int]] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # 跳过损坏行
                p_text = str(row.get(self.prompt_key, ""))
                r_text = str(row.get(self.response_key, ""))
                if not p_text or not r_text:
                    continue
                if self.apply_chat_template:
                    # Qwen chat 格式：user 角色 + generation prompt（模型才有推理上下文）
                    p_text = tok.apply_chat_template(
                        [{"role": "user", "content": p_text}],
                        tokenize=False, add_generation_prompt=True)
                p_ids = tok.encode(p_text, add_special_tokens=False, truncation=True,
                                   max_length=P)
                r_ids = tok.encode(r_text, add_special_tokens=False, truncation=True,
                                   max_length=T)
                p_ids = p_ids + [tok.pad_token_id] * max(0, P - len(p_ids))
                r_ids = r_ids + [tok.pad_token_id] * max(0, T - len(r_ids))
                prompts_ids.append(p_ids[:P])
                responses_ids.append(r_ids[:T])
        if not prompts_ids:
            raise DataError(f"dataset.path 无有效行: {self.path!r}")
        prompts = torch.tensor(prompts_ids, dtype=torch.long).to(self.device)
        responses = torch.tensor(responses_ids, dtype=torch.long).to(self.device)
        self._cache = (prompts, responses,
                       lambda r: torch.zeros_like(r, dtype=torch.float32))
        return self._cache


def build_data_loader(cfg: dict, device: str = "cpu") -> DataLoader:
    """按 `cfg["dataset"]["type"]` 构造数据加载器。未知类型抛 `DataError`。"""
    dtype = (cfg.get("dataset") or {}).get("type", "toy")
    if dtype == "toy":
        return ToyDataLoader(cfg, device)
    if dtype == "jsonl":
        return JsonLinesDataLoader(cfg, device)
    raise DataError(f"未知 dataset.type={dtype!r}（支持 toy|jsonl）")


__all__ = ["DataLoader", "ToyDataLoader", "JsonLinesDataLoader", "build_data_loader"]