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
