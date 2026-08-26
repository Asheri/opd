"""checkpoint._release_cpu_memory 单测：malloc_trim 路径不抛异常（E1 SIGKILL 缓解）。"""
from __future__ import annotations

import ctypes

import pytest

from fullstack_opd_v2.checkpoint import _release_cpu_memory


def test_release_cpu_memory_runs():
    """正常调用不抛异常（gc.collect + malloc_trim 尽力而为）。"""
    _release_cpu_memory()   # 不抛即通过（非 glibc 平台内部静默跳过）


def test_release_cpu_memory_malloc_trim_fails_silently(monkeypatch):
    """ctypes.CDLL 抛异常（如 libc 不可用）→ 静默跳过，不传播。"""
    calls = []

    class _Fake:
        def malloc_trim(self, *a, **k):
            raise OSError("fake libc failure")

    def _fake_cdll(name):
        calls.append(name)
        return _Fake()

    monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)
    _release_cpu_memory()   # 必须不抛
    assert calls == ["libc.so.6"]
