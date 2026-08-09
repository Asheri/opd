"""全栈 OPD 叠加 —— 工程化版（原 v2 demo 改造）。

算法内核不变：因果 LM、π_old 加权 PG、k3 KL、staleness 双截断。
工程化底座：run 目录 / 结构化日志 / checkpoint 断点续跑 / 指标追踪（CSV/WandB）/
可插拔数据与模型接口 / CLI 子命令。
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
from .data import DataLoader, ToyDataLoader, JsonLinesDataLoader, build_data_loader
from .model_factory import build_model
from .run import RunManager
from .checkpoint import CheckpointManager
from .metrics import MetricsRecorder
from .logging import setup_logging, get_logger, close_logging
from .eval_aime import AimeEvaluator, AimeResult, extract_answer, normalize_answer, format_prompt
from .exceptions import (
    OPDError, ConfigError, DataError, ModelError, CheckpointError, TrainingError,
)

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
    "DataLoader",
    "ToyDataLoader",
    "JsonLinesDataLoader",
    "build_data_loader",
    "build_model",
    "RunManager",
    "CheckpointManager",
    "MetricsRecorder",
    "setup_logging",
    "get_logger",
    "close_logging",
    "AimeEvaluator",
    "AimeResult",
    "extract_answer",
    "normalize_answer",
    "format_prompt",
    "OPDError",
    "ConfigError",
    "DataError",
    "ModelError",
    "CheckpointError",
    "TrainingError",
]
