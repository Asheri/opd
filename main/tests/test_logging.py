"""logging.py 单测：结构化日志初始化、级别生效、文件落盘、防重复 handler。"""
import logging

from fullstack_opd_v2.logging import setup_logging, get_logger


def test_setup_logging_returns_logger():
    lg = setup_logging(level="INFO", name="opd_test_1")
    assert isinstance(lg, logging.Logger)
    assert lg.level == logging.INFO


def test_setup_logging_writes_file(tmp_path):
    logf = tmp_path / "test.log"
    lg = setup_logging(level="DEBUG", log_file=str(logf), name="opd_test_2")
    lg.info("hello 工程化日志")
    for h in lg.handlers:
        h.flush()
    content = logf.read_text(encoding="utf-8")
    assert "hello 工程化日志" in content
    assert "INFO" in content


def test_setup_logging_no_duplicate_handlers(tmp_path):
    logf = tmp_path / "dup.log"
    lg = setup_logging(level="INFO", log_file=str(logf), name="opd_test_3")
    n1 = len(lg.handlers)
    lg2 = setup_logging(level="INFO", log_file=str(logf), name="opd_test_3")
    assert len(lg2.handlers) == n1, "重复调用不应叠加 handler"


def test_get_logger_returns_same():
    a = get_logger("opd_get")
    b = get_logger("opd_get")
    assert a is b