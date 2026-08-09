"""可插拔模型工厂：把 pipeline 里硬编码的 `CausalToyLM(...)` 构造集中起来。

- `build_model(cfg, device, role)` 按 `cfg["model_kind"]` 构造学生/教师模型。
- 默认 `model_kind="toy"` → `CausalToyLM`（单元可测的内核）。
- 预留 `model_kind="hf|megatron|vllm"` 分支——真实模型由 async-opd/vLLM 承担，
  此处抛 `ModelError` 明确"待实现"，避免静默走错路径。
- `role` 用于区分 student/teacher，便于未来按角色差异化构建（如 teacher 用更大模型）。
"""

from __future__ import annotations

from .exceptions import ModelError
from .model import CausalToyLM


def build_model(cfg: dict, device: str = "cpu", role: str = "student"):
    """按 model_kind 构建模型。默认 toy；未知/未实现 kind 抛 ModelError。"""
    kind = cfg.get("model_kind", "toy")
    if kind == "toy":
        return CausalToyLM(
            vocab=cfg["vocab_size"], d_model=cfg["d_model"],
            n_layers=cfg["n_layers"]).to(device)
    if kind in ("hf", "megatron", "vllm"):
        raise ModelError(
            f"model_kind={kind!r} 的真实模型集成尚未实现（由 async-opd/vLLM 承担）。"
            f"当前用 model_kind='toy' 走单元可测内核。role={role}")
    raise ModelError(
        f"未知 model_kind={kind!r}（支持 toy；hf/megatron/vllm 待实现）role={role}")


__all__ = ["build_model"]