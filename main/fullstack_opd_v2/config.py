"""配置加载 + pydantic schema 校验（P0 工程化）。

动机：此前 `configs/fullstack_opd.yaml` 只作"文档参考"、代码实际用硬编码
`DEFAULT_CONFIG_V2`——且 dict-of-dicts 合并存在"顶层部署键被 stage 子字典静默忽略"的
隐患（正是 CLOUD_CONFIG 那次 P0 配置作用域 bug 的根源）。

本模块把 YAML 变成**唯一真源**：
- pydantic schema 强校验：`extra="forbid"` 拒绝任何未知/拼错的键（静默忽略→显式报错）；
  `Literal` 限制枚举取值（dtype/cache_mode/scheduling_mode/warmup_source/...）；
- `load_config()`：YAML → 合并到内置默认 → 应用点分 CLI 覆盖 → 顶层部署键下渗
  （到 stage1/stage2）→ 校验 → 返回嵌套 dict，可直接传给 `FullStackOPDv2(cfg)`。
"""
from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .pipeline import DEFAULT_CONFIG_V2


# --------------------------- 子阶段 schema ---------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 未知键 → ValidationError


class Stage0Cfg(_Strict):
    lr: float = 1e-3
    n_rl_steps: int = 40
    max_new_tokens: int = 8
    batch_size: int = 8
    grad_clip: float = 1.0


class Stage1Cfg(_Strict):
    enforce_teacher_consistency: bool = True
    cache_path: str = "fullstack_opd_cache_v2.pt"
    build_batch_size: int = 16
    warmup_M: int = 0
    warmup_source: Literal["none", "student_init", "teacher_perturbed", "mix"] = "none"
    warmup_temperature: float = 1.0
    # ---- 顶层部署键下渗槽位（A5 解法 + T2 死槽位清理）----
    # load_config 在下渗后才校验，这些键须在 stage schema 有合法位置，否则
    # extra="forbid" 会把下渗结果当未知键拒掉。只保留 stage1 真正消费的键
    # （stage1_build_cache：cache_mode/top_k_teacher），其余为死槽位不设。
    cache_mode: Literal["dense", "topk"] = "dense"
    top_k_teacher: int = 0


class Stage2Cfg(_Strict):
    scheduling_mode: Literal["fully_async", "n_step_off", "fused_hybrid_sync"] = "fully_async"
    staleness_threshold: int = 4
    queue_size: int = 8
    kl_reg_coef: float = 0.05
    clip_eps: float = 0.2
    grad_clip: float = 1.0
    lr: float = 1e-3
    n_steps: int = 30
    batch_size: int = 8
    distributed: bool = False
    n_gpus: int = 2
    prefetch: int = 4
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    tp_size: int = 1
    sequence_parallel: bool = True
    rollout_engine: Literal["toy", "vllm"] = "toy"
    rollout_tp_size: int = 2
    rollout_model: str = "Qwen/Qwen2.5-7B"
    rollout_dtype: Literal["auto", "bf16", "fp8"] = "auto"
    rollout_logprobs_cap: int = 4096
    # ---- 顶层部署键下渗槽位（A5 解法 + T2 死槽位清理）----
    # load_config 在下渗后才校验，这些键须在 stage schema 有合法位置，否则
    # extra="forbid" 会把下渗结果当未知键拒掉。只保留 stage2 真正消费的键
    # （scheduler：dtype/top_k_student/offload_to_cpu）。cache_mode/top_k_teacher
    # 由 cache 对象读取（stage2 无槽位）；ref_topk 由 pipeline 读顶层（不下渗）。
    dtype: Literal["fp32", "bf16", "float32", "bfloat16"] = "fp32"
    top_k_student: int = 0
    offload_to_cpu: bool = False


# --------------------------- 工程化新增段 ---------------------------
class RunCfg(_Strict):
    """run 目录与断点续跑配置。"""
    seed: int = 42
    run_dir: str | None = None
    checkpoint_every: int = 10


class LoggingCfg(_Strict):
    """结构化日志配置。"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str = "train.log"


class MetricsCfg(_Strict):
    """指标追踪配置。"""
    backend: Literal["csv", "wandb", "none"] = "csv"
    csv_path: str | None = None
    wandb_project: str | None = None


class DatasetCfg(_Strict):
    """数据加载配置（可插拔接口，见 data.py）。"""
    type: Literal["toy", "jsonl"] = "toy"
    path: str | None = None
    prompt_key: str = "prompt"
    response_key: str = "response"


class OPDConfig(_Strict):
    vocab_size: int = 64
    d_model: int = 48
    n_layers: int = 2
    prompt_len: int = 6
    resp_len: int = 8
    n_prompts: int = 16
    seed: int = 42
    batch_size: int = 8
    dtype: Literal["fp32", "bf16", "float32", "bfloat16"] = "fp32"
    cache_mode: Literal["dense", "topk"] = "dense"
    top_k_teacher: int = 0
    top_k_student: int = 0
    ref_topk: int = 0
    offload_to_cpu: bool = False
    model_kind: Literal["toy", "hf", "megatron", "vllm"] = "toy"
    stage0: Stage0Cfg = Field(default_factory=Stage0Cfg)
    stage1: Stage1Cfg = Field(default_factory=Stage1Cfg)
    stage2: Stage2Cfg = Field(default_factory=Stage2Cfg)
    run: RunCfg = Field(default_factory=RunCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    metrics: MetricsCfg = Field(default_factory=MetricsCfg)
    dataset: DatasetCfg = Field(default_factory=DatasetCfg)


# --------------------------- 顶层部署键下渗 ---------------------------
# 顶层部署键（CLOUD_CONFIG 风格）会在 pydantic 校验前按 stage 分流下渗，
# 使校验后的 cfg 与 config.yaml 快照天然含下渗结果（A4/A5/B4）——不再由
# pipeline/cli 在运行时各自复制一份下渗循环（否则快照≠有效配置，且 stage
# 子 dict 在 extra="forbid" 下没有 dtype 等键的合法位置）。
# 按消费端分流（T2 死槽位清理）：stage1 只消费稀疏缓存键，stage2 只消费
# 训练/部署键；ref_topk 保持纯顶层（pipeline 读 self.cfg.get("ref_topk")）。
_STAGE1_SEEP_KEYS = ("cache_mode", "top_k_teacher")
_STAGE2_SEEP_KEYS = ("dtype", "top_k_student", "offload_to_cpu")


def _seep_deployment_keys(d: dict) -> dict:
    """顶层部署键按 stage 分流下渗（stage 子键优先）。

    在 pydantic 校验前调用，使校验后的 cfg 与 config.yaml 快照天然含下渗结果
    （A4/A5/B4）。只补 stage 里没有的键：stage 子键显式给出的值不会被顶掉。
    - stage1 ← {cache_mode, top_k_teacher}（stage1_build_cache 消费）
    - stage2 ← {dtype, top_k_student, offload_to_cpu}（scheduler 消费）
    - ref_topk 保持纯顶层（pipeline 读顶层），不下渗、stage 无槽位
    """
    for k in _STAGE1_SEEP_KEYS:
        if k in d:
            sd = d.setdefault("stage1", {})
            if k not in sd:
                sd[k] = d[k]
    for k in _STAGE2_SEEP_KEYS:
        if k in d:
            sd = d.setdefault("stage2", {})
            if k not in sd:
                sd[k] = d[k]
    return d


def _parse_scalar(text: str) -> Any:
    """把 CLI 字符串覆盖值解析成 bool/int/float/str。"""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    """`stage2.lr=1e-4` → d['stage2']['lr']=1e-4（点分路径）。"""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_config(path: str | None = None,
                overrides: list[str] | None = None) -> dict:
    """YAML → 合并默认 → CLI 点分覆盖 → pydantic 校验 → 嵌套 dict。

    - path=None 时仅用内置默认（等价 DEFAULT_CONFIG_V2）。
    - overrides: ["stage2.lr=1e-4", "n_steps=50", "stage1.warmup_source=mix"]。
    - 任何未知键 / 非法枚举值 / 类型不符 → 抛 ValidationError（静默忽略→显式报错）。
    """
    data: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    for item in (overrides or []):
        if "=" not in item:
            raise ValueError(f"覆盖项需形如 key=value，收到 {item!r}")
        k, v = item.split("=", 1)
        _set_dotted(data, k.strip(), _parse_scalar(v))
    # 顶层部署键按 stage 分流下渗（stage 子键优先，见 _seep_deployment_keys）。
    # 须在 pydantic 校验前做：下渗键（如 stage2.dtype）在 extra="forbid" 的
    # stage schema 无合法位置，校验后补会破坏"快照=有效配置"（A4/A5/B4）。
    # --set 点分覆盖已在 _set_dotted 应用过，下渗只补 stage 里没有的键。
    data = _seep_deployment_keys(data)
    cfg = OPDConfig(**data)                      # 校验（未知键/非法值在此报错）
    # 合并到内置默认（pydantic 已用默认补全所有键，model_dump 即为完整配置）
    merged = {**DEFAULT_CONFIG_V2, **cfg.model_dump()}
    for stage in ("stage0", "stage1", "stage2"):
        merged[stage] = {**DEFAULT_CONFIG_V2[stage], **cfg.model_dump()[stage]}
    return merged


__all__ = ["OPDConfig", "Stage0Cfg", "Stage1Cfg", "Stage2Cfg",
           "RunCfg", "LoggingCfg", "MetricsCfg", "DatasetCfg",
           "load_config", "ValidationError"]
