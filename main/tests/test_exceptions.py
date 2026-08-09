"""exceptions.py 单测：类型化异常层级。"""
from fullstack_opd_v2.exceptions import (
    OPDError, ConfigError, DataError, ModelError, CheckpointError, TrainingError,
)


def test_hierarchy():
    for e in (ConfigError, DataError, ModelError, CheckpointError, TrainingError):
        assert issubclass(e, OPDError)
    assert issubclass(OPDError, Exception)


def test_all_exceptions_catchable_as_opderror():
    for e in (ConfigError("c"), DataError("d"), ModelError("m"),
              CheckpointError("k"), TrainingError("t")):
        assert isinstance(e, OPDError)