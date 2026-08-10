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


def test_metrics_content_correct(tmp_path):
    """D2：表头必须包含全部已出现字段（含后到的 age），行内容逐字段正确。"""
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path))
    mr.record({"loss": 0.1, "version": 1})
    mr.record({"loss": 0.2, "version": 2, "age": 5})
    mr.close()
    lines = open(os.path.join(str(tmp_path), "metrics.csv"), encoding="utf-8").read().splitlines()
    hdr = lines[0].split(",")
    assert {"loss", "version", "age"} <= set(hdr)
    row2 = dict(zip(hdr, lines[2].split(",")))
    assert row2["loss"] == "0.2" and row2["age"] == "5"


def test_flush_count_throttled(tmp_path):
    """C2 spy：flush_every=3 时 5 条记录只触发 2 次 flush（第3条 + close 终刷）。"""
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path), flush_every=3)
    flushes = []
    orig = mr._file.flush
    mr._file.flush = lambda: (flushes.append(1), orig())[1]
    for i in range(5):
        mr.record({"loss": i})
    mr.close()
    assert len(flushes) == 2   # 第3条1次 + close终刷1次

def test_append_mode_keeps_history(tmp_path):
    """L1：append=True 时续写 metrics.csv 保留历史（resume 场景）。"""
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path))
    mr.record({"loss": 0.1, "version": 1})
    mr.close()
    mr2 = MetricsRecorder(backend="csv", run_dir=str(tmp_path), append=True)
    mr2.record({"loss": 0.2, "version": 2})
    mr2.close()
    lines = open(os.path.join(str(tmp_path), "metrics.csv"), encoding="utf-8").read().splitlines()
    assert len(lines) == 3          # 表头 + 2 行（历史保留）
    assert lines[0].startswith("loss,version")
    assert lines[1].endswith("0.1,1")
    assert lines[2].endswith("0.2,2")


def test_append_heals_half_line(tmp_path):
    """P3（R2 审查）：上次崩溃留下无换行的半行 → append 先补换行再续写，首行不畸形。

    模拟：旧文件以『表头 + 完整行 + 半行』结束（半行无 \n），append 续写后
    半行被换行终止、新行独立成行；splitlines 不应出现表头被污染的行。
    """
    path = os.path.join(str(tmp_path), "metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("loss,version\r\n0.1,1\r\n0.2,2")     # 末行无终止符（模拟崩溃半行）
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path), append=True)
    mr.record({"loss": 0.3, "version": 3})
    mr.close()
    raw = open(path, encoding="utf-8").read()
    # 半行被终止：完整行数 = 表头 + 2 旧行 + 1 新行
    lines = [l for l in raw.replace("\r\n", "\n").split("\n") if l]
    assert len(lines) == 4
    assert lines[2] == "0.2,2"        # 半行内容保留、未被后续拼接
    assert lines[3] == "0.3,3"        # 新行独立成行
    assert not lines[0].startswith("0.2,2,")   # 表头未被半行污染


def test_append_new_field_keeps_history_warns(tmp_path):
    """P3（R2 审查）：append 续写时晚到新列 → 旧行留空、新行带新列、不整表重写、告警。"""
    path = os.path.join(str(tmp_path), "metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("loss,version\r\n0.1,1\r\n")
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path), append=True)
    with pytest.warns(UserWarning, match="新字段"):
        mr.record({"loss": 0.2, "version": 2, "age": 5})
    mr.close()
    raw = open(path, encoding="utf-8").read()
    lines = [l for l in raw.replace("\r\n", "\n").split("\n") if l]
    assert lines[0].startswith("loss,version")   # 旧表头不变（不截断）
    assert "0.1,1" in lines                      # 历史保留
    # 新行带新列（DictWriter 按扩展后的 fieldnames 写，age 有值）
    assert any("0.2,2" in l and ",5" in l for l in lines)
