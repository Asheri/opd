"""可插拔数据接口：把 pipeline 里硬编码的 `_make_toy_data` 抽成 DataLoader。

- `DataLoader`(ABC)：统一 `load()` 契约，返回 `(prompts, responses, reward_fn)`。
  - prompts/responses: (N,P)/(N,T) 设备常驻张量（与 v2 其余部分一致）。
  - reward_fn: `(B,T) token` -> `(B,T) 奖励` 的可调用对象（Stage 0 REINFORCE 用）。
- `ToyDataLoader`：默认实现，与旧 `_make_toy_data` 数值完全同源（randint 同流 + 查找表）。
- `JsonLinesDataLoader`：预留真实数据接口（需 tokenizer 文本→token，见类注释）。
- `build_data_loader`：按 `cfg["dataset"]["type"]` 分发，未知类型抛 `DataError`。
"""

from __future__ import annotations

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
    """真实数据接口（占位）：从 jsonl 读 `prompt`/`response` 文本。

    ⚠️ 文本→token 需要 tokenizer（与模型词表对齐），当前范围不实现——
    加载时抛 `DataError` 明确提示。接入真实数据时补 tokenize 即可。
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        ds = cfg["dataset"]
        self.path = ds.get("path")
        self.prompt_key = ds.get("prompt_key", "prompt")
        self.response_key = ds.get("response_key", "response")
        self.device = device

    def load(self):
        raise DataError(
            "JsonLinesDataLoader 需 tokenizer 实现文本→token（与模型词表对齐），"
            "当前为预留接口。请接入真实 tokenizer 后使用，或先用 dataset.type='toy'。")


def build_data_loader(cfg: dict, device: str = "cpu") -> DataLoader:
    """按 `cfg["dataset"]["type"]` 构造数据加载器。未知类型抛 `DataError`。"""
    dtype = (cfg.get("dataset") or {}).get("type", "toy")
    if dtype == "toy":
        return ToyDataLoader(cfg, device)
    if dtype == "jsonl":
        return JsonLinesDataLoader(cfg, device)
    raise DataError(f"未知 dataset.type={dtype!r}（支持 toy|jsonl）")


__all__ = ["DataLoader", "ToyDataLoader", "JsonLinesDataLoader", "build_data_loader"]