"""类型化异常层级：全栈 OPD 统一的异常体系。

把散落的 RuntimeError/ValueError 收敛为可精确捕获的 OPDError 子类，
调用方可按阶段（配置/数据/模型/checkpoint/训练）分别处理或统一捕获。
"""


class OPDError(Exception):
    """全栈 OPD 所有异常的基类。"""


class ConfigError(OPDError):
    """配置层面的错误（加载/校验/合并失败）。"""


class DataError(OPDError):
    """数据层面的错误（加载/解析/形状不符）。"""


class ModelError(OPDError):
    """模型层面的错误（构建/加载/架构不符）。"""


class CheckpointError(OPDError):
    """checkpoint 层面的错误（保存/加载/续跑失败）。"""


class TrainingError(OPDError):
    """训练层面的错误（训练循环/调度器异常）。"""


__all__ = ["OPDError", "ConfigError", "DataError", "ModelError",
           "CheckpointError", "TrainingError"]