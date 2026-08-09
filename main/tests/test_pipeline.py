"""pipeline.py 端到端冒烟 + L1 暖缓存（fat_N 计数、向后兼容、非法 source 抛错）、奖励趋势。"""
from __future__ import annotations

import pytest

from fullstack_opd_v2.pipeline import FullStackOPDv2, DEFAULT_CONFIG_V2


def _cfg(tmp_path, **stage1_over):
    s1 = dict(DEFAULT_CONFIG_V2["stage1"])
    s1["cache_path"] = str(tmp_path / "c.pt")
    s1.update(stage1_over)
    return {
        "n_prompts": 12,
        "stage0": {**DEFAULT_CONFIG_V2["stage0"], "n_rl_steps": 5},
        "stage1": s1,
        "stage2": {**DEFAULT_CONFIG_V2["stage2"], "n_steps": 10, "batch_size": 4},
    }


def test_end_to_end_smoke(tmp_path):
    out = FullStackOPDv2(_cfg(tmp_path), device="cpu").run()
    metrics = out["metrics"]
    assert len(metrics) == 10
    assert "reward" in metrics[-1]
    assert out["cache"].delta.shape[0] == 12        # L0：fat_N == n_prompts


def test_warmup_off_backward_compatible(tmp_path):
    out = FullStackOPDv2(_cfg(tmp_path, warmup_M=0, warmup_source="none"),
                         device="cpu").run()
    assert out["cache"].delta.shape[0] == 12        # 不胖


def test_warmup_student_init_fat_count(tmp_path):
    out = FullStackOPDv2(_cfg(tmp_path, warmup_M=4, warmup_source="student_init"),
                         device="cpu").run()
    assert out["cache"].delta.shape[0] == 12 * 5    # 1 + 4


def test_warmup_mix_fat_count(tmp_path):
    out = FullStackOPDv2(_cfg(tmp_path, warmup_M=4, warmup_source="mix"),
                         device="cpu").run()
    assert out["cache"].delta.shape[0] == 12 * 9    # 1 + 4(student) + 4(teacher)


def test_warmup_invalid_source_raises(tmp_path):
    with pytest.raises(ValueError):
        FullStackOPDv2(_cfg(tmp_path, warmup_M=4, warmup_source="bogus"),
                       device="cpu").run()


def test_reward_trends_up(tmp_path):
    """E[Δ_T]（Direct-OPD 密集奖励）应随训练上升（学习信号正确）。"""
    cfg = _cfg(tmp_path)
    cfg["stage2"]["n_steps"] = 25
    out = FullStackOPDv2(cfg, device="cpu").run()
    r = [m["reward"] for m in out["metrics"]]
    first = sum(r[:3]) / 3
    last = sum(r[-3:]) / 3
    assert last > first, f"E[Δ_T] 未上升: first3={first:.3f} last3={last:.3f}"


def test_run_produces_run_dir_artifacts():
    """T10：run() 落盘完整 run 目录（config/日志/checkpoint/metrics/计时）。"""
    import json
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg(type("T", (), {"__truediv__": lambda self, o: os.path.join(td, o)})())
        cfg["run"] = {"run_dir": os.path.join(td, "exp"), "checkpoint_every": 2}
        out = FullStackOPDv2(cfg, device="cpu").run()
        run_dir = out["run_dir"]
        assert os.path.isfile(os.path.join(run_dir, "config.yaml"))
        assert os.path.isfile(os.path.join(run_dir, "metrics.csv"))
        assert os.path.isfile(os.path.join(run_dir, "logs", "train.log"))
        assert os.path.isfile(os.path.join(run_dir, "timings.json"))
        timings = json.load(open(os.path.join(run_dir, "timings.json"), encoding="utf-8"))
        assert set(("stage0_rl", "stage1_cache", "stage2_train", "total")) <= set(timings)
        ckpts = os.listdir(os.path.join(run_dir, "checkpoints"))
        assert any(f.startswith("step_") for f in ckpts)    # final force 保存


def test_distributed_branch_no_unbound_local(tmp_path, monkeypatch):
    """分布式分支不应引用未定义的 scheduler（mock launch_distributed_scheduler）。"""
    import fullstack_opd_v2.pipeline as P
    fake = lambda *a, **k: [{"step": 0, "version": 1, "age": 0, "batch": 4,
                             "loss": 0.1, "pg_loss": 0.1, "kl_loss": 0.0,
                             "adv_mean": 0.0, "reward": 0.1}]
    monkeypatch.setattr(P, "launch_distributed_scheduler", fake)
    cfg = _cfg(tmp_path)
    cfg["stage2"]["distributed"] = True
    cfg["stage2"]["n_steps"] = 1
    out = FullStackOPDv2(cfg, device="cpu").run()
    assert len(out["metrics"]) == 1
