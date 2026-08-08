"""全栈 OPD 叠加 demo v2 —— 批量化重构版。

算法内核与 v1（审阅修复后）一致：因果 LM、π_old 加权 PG、k3 KL、staleness 双截断。
执行底座重构：批次向量化 / 设备常驻张量 / 权重按需加载 / 奖励查找表。
"""

from .model import CausalToyLM
from .model_megatron import MegatronCausalToyLM
from .cache import TensorTeacherCache
from .scheduler import (
    AsyncBatchedScheduler,
    DistAsyncScheduler,
    WeightBroadcaster,
    parallelize_learner_tp2,
    launch_distributed_scheduler,
)
from .rollout_vllm import VLLMRolloutEngine
from .pipeline import FullStackOPDv2, DEFAULT_CONFIG_V2
from .config import OPDConfig, load_config

__all__ = [
    "CausalToyLM",
    "MegatronCausalToyLM",
    "TensorTeacherCache",
    "AsyncBatchedScheduler",
    "DistAsyncScheduler",
    "WeightBroadcaster",
    "parallelize_learner_tp2",
    "launch_distributed_scheduler",
    "VLLMRolloutEngine",
    "FullStackOPDv2",
    "DEFAULT_CONFIG_V2",
    "OPDConfig",
    "load_config",
]
