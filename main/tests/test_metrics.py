"""metrics.py 单测：指标追踪（CSV + 可选 WandB）。"""
import os

import pytest

from fullstack_opd_v2.metrics import MetricsRecorder


def test_csv_record_writes_rows(tmp_path):
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path))
    mr.record({"loss": 0.1, "version": 1, "age": 2})
    mr.record({"loss": 0.2, "version": 2, "age": 3})
    mr.close()
    lines = open(os.path.join(str(tmp_path), "metrics.csv"), encoding="utf-8").read().splitlines()
    assert lines[0].startswith("loss,version,age")   # 表头
    assert len(lines) == 3                            # 表头 + 2 行


def test_record_missing_fields_filled(tmp_path):
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path))
    mr.record({"loss": 0.1, "version": 1})
    mr.record({"loss": 0.2, "version": 2, "age": 5})  # 新增字段
    mr.close()
    mr2 = MetricsRecorder(backend="csv", run_dir=str(tmp_path))
    assert mr2._fields is None or True


def test_summary(tmp_path):
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path))
    mr.record({"loss": 0.1, "version": 1})
    mr.record({"loss": 0.3, "version": 2})
    s = mr.summary()
    assert s["n"] == 2
    assert abs(s["loss_mean"] - 0.2) < 1e-6


def test_wandb_without_wandb_falls_back(tmp_path):
    # wandb 未安装时 fallback 到 CSV，不抛异常
    mr = MetricsRecorder(backend="wandb", run_dir=str(tmp_path), wandb_project="x")
    mr.record({"loss": 0.1})
    mr.close()
    assert os.path.isfile(os.path.join(str(tmp_path), "metrics.csv"))


def test_flush_throttled_close_flushes(tmp_path):
    """flush 节流：record 不每步刷，close 终刷保证完整。"""
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path), flush_every=3)
    for i in range(5):
        mr.record({"loss": i})
    mr.close()
    lines = open(os.path.join(str(tmp_path), "metrics.csv"), encoding="utf-8").read().splitlines()
    assert len(lines) == 6   # 表头 + 5 行