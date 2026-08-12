"""可插拔模型工厂：把 pipeline 里硬编码的 `CausalToyLM(...)` 构造集中起来。

- `build_model(cfg, device, role)` 按 `cfg["model_kind"]` 构造学生/教师模型。
- 默认 `model_kind="toy"` → `CausalToyLM`（单元可测的内核）。
- `model_kind="hf"` → `HFCausalLM`（真实 HF 模型适配器，**骨架**：代码完整但需 GPU/真实
  模型验证；本地无 GPU/模型无法实测）。megatron/vllm 仍抛 `ModelError`（由 async-opd/vLLM 承担）。
- `role` 用于区分 student/teacher：hf 下 student→`cfg["student_path"]`、teacher→`cfg["teacher_rl_path"]`。
"""

from __future__ import annotations

import torch

from .exceptions import ModelError
from .model import CausalToyLM

# transformers 可选导入（HF 骨架；未装时仅 HF 分支报错，toy 不受影响）
try:
    from transformers import AutoModelForCausalLM as _HF_AutoModelForCausalLM
    _HF_AVAILABLE = True
except Exception:                                     # pragma: no cover
    _HF_AutoModelForCausalLM = None
    _HF_AVAILABLE = False


class HFCausalLM:
    """真实 HF 模型适配器（⚠️ 骨架：需 GPU/真实模型验证）。

    包 `transformers.AutoModelForCausalLM`，暴露与 `CausalToyLM` 兼容接口，使
    pipeline/scheduler/losses 内核零改动：
      `__call__(ids) -> logits (B,L,V)`（供 `response_dists` / `generate_batch` 用）、
      `response_dists(prompts, responses) -> (B,T,V)`、`vocab / d_model / max_len`、
      `train / eval / to / training / state_dict / load_state_dict / parameters / zero_grad`。

    ⚠️ 骨架边界（GPU 验证时需逐项确认）：
    - 输入是【已 tokenize 的 id 张量】（等长、无 padding mask）——toy 流水线假设等长；
      真实变长序列需在 data 层补 `attention_mask`（forward 目前只传 input_ids）。
    - 未实现 KV cache：`generate_batch` 对增长序列 O(T²)（真实规模应走 vLLM rollout）。
    - 权重同步/断点走 `state_dict`（含 tied embedding/lm_head，load_state_dict 正常）。
    """

    def __init__(self, path: str, device: str = "cpu", dtype: str = "auto",
                 vocab: int | None = None, d_model: int | None = None,
                 max_len: int | None = None):
        if not _HF_AVAILABLE:
            raise ModelError("model_kind='hf' 需要 transformers（统一 GPU 环境应含）")
        td = {"bf16": torch.bfloat16, "float16": torch.float16,
              "fp16": torch.float16}.get(str(dtype).lower(), None)
        try:
            self.model = _HF_AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=td).to(device).eval()
        except Exception as e:
            raise ModelError(f"HF 模型 {path!r} 加载失败（路径/HF id 无效？）：{e}") from e
        self.path = path
        self.device = device
        # 尺寸以真实模型 config 为准（toy 的 cfg["vocab_size"]=64 会错）
        self.vocab = vocab or self.model.config.vocab_size
        self.d_model = d_model or getattr(self.model.config, "hidden_size", None)
        self.max_len = max_len or getattr(self.model.config,
                                          "max_position_embeddings", 2048)

    # ---- 内核用接口：forward / response_dists / generate（委托模块级实现）----
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B,L) -> logits (B,L,V)，与 CausalToyLM.forward 对齐（供 response_dists 用）。"""
        return self.model(input_ids).logits

    def response_dists(self, prompts: torch.Tensor, responses: torch.Tensor):
        from .model import response_dists
        return response_dists(self, prompts, responses)

    # ---- 训练/权重接口：委托给 HF 模块 ----
    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    @property
    def training(self) -> bool:
        return self.model.training

    def state_dict(self, *a, **k):
        return self.model.state_dict(*a, **k)

    def load_state_dict(self, *a, **k):
        return self.model.load_state_dict(*a, **k)

    def parameters(self, *a, **k):
        return self.model.parameters(*a, **k)

    def named_parameters(self, *a, **k):
        return self.model.named_parameters(*a, **k)

    def zero_grad(self, *a, **k):
        self.model.zero_grad(*a, **k)


def _hf_model_path(cfg: dict, role: str) -> str:
    """按角色取 HF 模型路径：student→student_path；teacher→teacher_rl_path。"""
    key = {"student": "student_path", "teacher": "teacher_rl_path"}.get(role)
    if key is None:
        raise ModelError(f"model_kind='hf' 不支持 role={role!r}（student|teacher）")
    path = cfg.get(key)
    if not path:
        raise ModelError(f"model_kind='hf' 但未配置 {key}（role={role}）")
    return path


def build_model(cfg: dict, device: str = "cpu", role: str = "student"):
    """按 model_kind 构建模型。默认 toy；hf 走 HFCausalLM 骨架；未知/未实现 kind 抛 ModelError。"""
    kind = cfg.get("model_kind", "toy")
    if kind == "toy":
        return CausalToyLM(
            vocab=cfg["vocab_size"], d_model=cfg["d_model"],
            n_layers=cfg["n_layers"]).to(device)
    if kind == "hf":
        # ⚠️ 骨架：本地无法 GPU 实测；需真实模型 + GPU 验证。
        return HFCausalLM(_hf_model_path(cfg, role), device,
                          dtype=cfg.get("dtype", "auto"))
    if kind in ("megatron", "vllm"):
        raise ModelError(
            f"model_kind={kind!r} 的真实模型集成尚未实现（由 async-opd/vLLM 承担）。"
            f"当前用 model_kind='toy'/'hf'。role={role}")
    raise ModelError(f"未知 model_kind={kind!r}（支持 toy/hf；megatron/vllm 待实现）role={role}")


__all__ = ["build_model", "HFCausalLM"]
