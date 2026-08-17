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
        self._raw_prompt_texts: list[str] = []      # C3：原始 prompt 文本（未套模板），供教师各自模板重编码
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
                self._raw_prompt_texts.append(p_text)   # C3：存原始文本（未套模板）
                if self.apply_chat_template:
                    # Qwen chat 格式：user 角色 + generation prompt（模型才有推理上下文）。
                    # C3 截断防护：先估模板串长度，超 P 时先截题干再套模板，保证
                    # assistant 生成标记不被 max_prompt_len 右截断切掉（否则模型无
                    # assistant 上下文 → 退化生成，2026-08-18 设计）。
                    _tpl = tok.apply_chat_template(
                        [{"role": "user", "content": p_text}],
                        tokenize=False, add_generation_prompt=True)
                    if len(tok.encode(_tpl, add_special_tokens=False)) > P:
                        _cnt_ids = tok.encode(p_text, add_special_tokens=False)
                        _ovh = (len(tok.encode(_tpl, add_special_tokens=False))
                                - len(_cnt_ids))
                        _keep = max(8, P - _ovh)
                        p_text = tok.decode(_cnt_ids[:_keep])
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

    @property
    def raw_prompt_texts(self) -> list[str]:
        """C3：已加载行的原始 prompt 文本（与 prompts 行对齐，未套任何模板）。"""
        if self._cache is None:
            raise DataError("raw_prompt_texts 需先 load()")
        return list(self._raw_prompt_texts)


def build_data_loader(cfg: dict, device: str = "cpu") -> DataLoader:
    """按 `cfg["dataset"]["type"]` 构造数据加载器。未知类型抛 `DataError`。"""
    dtype = (cfg.get("dataset") or {}).get("type", "toy")
    if dtype == "toy":
        return ToyDataLoader(cfg, device)
    if dtype == "jsonl":
        return JsonLinesDataLoader(cfg, device)
    raise DataError(f"未知 dataset.type={dtype!r}（支持 toy|jsonl）")


__all__ = ["DataLoader", "ToyDataLoader", "JsonLinesDataLoader", "build_data_loader"]


# ---------------------------------------------------------------------------
# C3（2026-08-18）：教师各自模板格式的 prompt 编码
# ---------------------------------------------------------------------------
def build_teacher_prompts(raw_texts, tokenizer_path, P: int, device: str = "cpu",
                          role: str = "user",
                          vocab_size: int | None = None) -> torch.Tensor:
    """用教师自己的 tokenizer + chat template 把原始 prompt 文本编码为 (N,P) 定长。

    C3 语义：student prompt 套 Qwen3 模板后，教师（JustRL/R1-Distill 等）不应看到
    学生格式的 token——用各自原生模板包裹 user 角色 + generation prompt，使教师
    Δ_T 的上下文匹配各自训练分布。返回 (N,P) long；截断右 pad（教师 tokenizer 的
    pad_token_id，缺省 0）。vocab_size：教师模型词表；提供时校验 tokenizer 词表
    兼容（跨词表 id 喂教师模型 = 垃圾上下文）。
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if vocab_size is not None and getattr(tok, "vocab_size", None) and             int(tok.vocab_size) > int(vocab_size):
        raise DataError(
            f"教师 tokenizer 词表 {tok.vocab_size} > 教师模型词表 {vocab_size}："
            f"{tokenizer_path} tokenizer 与模型不匹配，教师格式 prompt 会是跨词表 "
            "垃圾 id——请检查 teacher 路径/tokenizer 配置。")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows: list[list[int]] = []
    for t in raw_texts:
        s = tok.apply_chat_template([{"role": role, "content": t}],
                                    tokenize=False, add_generation_prompt=True)
        # 截断防护（同数据层）：超长时先截题干再套模板，保住生成标记尾部。
        if len(tok.encode(s, add_special_tokens=False)) > P:
            _cnt_ids = tok.encode(t, add_special_tokens=False)
            _ovh = len(tok.encode(s, add_special_tokens=False)) - len(_cnt_ids)
            t = tok.decode(_cnt_ids[:max(8, P - _ovh)])
            s = tok.apply_chat_template([{"role": role, "content": t}],
                                        tokenize=False, add_generation_prompt=True)
        ids = tok.encode(s, add_special_tokens=False, truncation=True,
                         max_length=P)
        ids = ids[:P] + [pad] * max(0, P - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long).to(device)
