"""全栈 OPD 叠加 demo 包。

把 Lightning-OPD（离线教师缓存）+ Direct-OPD（迁移对象=RL 策略偏移）+ AsyncOPD
（异步调度器）三篇论文叠加成一个端到端可运行的流水线。

详见 README.md。
"""

# ★ 修复：原先只声明 __all__ 却没有 import，`from fullstack_opd import ToyModel`
# 会直接 ImportError。
from .models import ToyModel
from .lightning_cache import OfflineTeacherPairCache
from .async_scheduler import AsyncOPDScheduler
from .pipeline import FullStackOPD, DEFAULT_CONFIG

__all__ = [
    "ToyModel",
    "OfflineTeacherPairCache",
    "AsyncOPDScheduler",
    "FullStackOPD",
    "DEFAULT_CONFIG",
]
