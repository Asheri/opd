"""结构化日志：时间戳 / 级别 / 控制台 + 文件，防重复 handler。

工程化替代散落的 print()：统一走 logging，训练既有控制台实时输出，
又落盘到 run 目录的 train.log，便于事后排查与跨 run 对比。
"""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO", log_file: str | None = None,
                  name: str = "opd") -> logging.Logger:
    """初始化结构化日志（控制台 + 可选文件），返回命名的 logger。

    - 幂等：同一 name 重复调用不叠加 handler（防重复输出）。
    - level: "DEBUG"/"INFO"/"WARNING"/"ERROR" 或 logging 级别整数。
    - log_file: 非空时同时落盘到该文件（utf-8）。
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    def _has_handler(stream) -> bool:
        return any(getattr(h, "stream", None) is stream for h in logger.handlers)

    if not _has_handler(sys.stdout):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    if log_file and not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == log_file
            for h in logger.handlers):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_logger(name: str = "opd") -> logging.Logger:
    """取命名 logger（未初始化时返回默认配置的 logger）。"""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]