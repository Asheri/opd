"""inspect_delta_cache.py 单测：Δ_T 信号体检统计正确性 + 判据三态。

覆盖：分布统计（正比例/均值/分位/clip）、位置分解三段、判据 PASS/FAIL/BOUNDARY、
磁盘 cache 加载全流程、student top-K 重合率。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fullstack_opd_v2.cache import TensorTeacherCache
from fullstack_opd_v2.cache_store import write_cache_disk
from fullstack_opd_v2.model import CausalToyLM
from scripts.inspect_delta_cache import (
    analyze_delta_distribution, analyze_position_breakdown,
    compute_student_overlap, judge, load_cache, main,
    PASS_ABS_MEAN, PASS_POS_RATIO,
)


def _make_cache(N=6, P=4, T=6, V=24, d=16, L=1, K=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(1, V, (N, P), generator=g)
    responses = torch.randint(1, V, (N, T), generator=g)
    rl = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    ref = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, rl, ref)
    return cache, prompts, responses


def test_analyze_distribution_basic():
    """正比例/均值/分位统计正确（含 padding 掩码）。"""
    delta = np.zeros((2, 4, 3), dtype=np.float32)
    delta[0, :, 0] = 1.0     # 样本0 前3 token 的 top1 为正
    delta[0, :, 1] = -0.5
    delta[1, :, 2] = -2.0
    mask = np.zeros((2, 4), dtype=bool)
    mask[0, :3] = True       # 样本0 有效 3 token
    mask[1, :2] = True       # 样本1 有效 2 token
    out = analyze_delta_distribution(delta, mask)
    # 有效元素：样本0 的 3*3=9 + 样本1 的 2*3=6 = 15
    assert out["n"] == 15
    assert abs(out["pos_ratio"] - 3 / 15) < 1e-6
    assert abs(out["dist"]["mean"] - delta[mask].mean()) < 1e-6
    # clip：|Δ|>2.0 的只有样本1 一个 -2.0 的 top3（|−2|=2 不>2）→ 0
    assert out["clip_ratio"] == 0.0


def test_analyze_distribution_mask_none():
    """mask=None 时统计全部元素。"""
    delta = np.ones((2, 2, 2), dtype=np.float32)
    out = analyze_delta_distribution(delta, None)
    assert out["n"] == 8
    assert out["pos_ratio"] == 1.0
    assert out["dist"]["mean"] == 1.0


def test_position_breakdown_three_segments():
    """位置分解：前25%/中50%/后25% 按有效长度划分。"""
    delta = np.zeros((1, 8, 1), dtype=np.float32)
    lengths = np.array([8])
    # 全置 1 后按段清零观察段掩码
    delta[:] = 1.0
    out = analyze_position_breakdown(delta, lengths, T=8)
    assert set(out.keys()) == {"early_0_25", "mid_25_75", "late_75_1"}
    # 前 25% = token 0,1（frac<0.25），8*0.25=2 个 token
    assert out["early_0_25"]["n"] == 2
    assert out["early_0_25"]["pos_ratio"] == 1.0
    # 中 50% = token 2..5（4 个）
    assert out["mid_25_75"]["n"] == 4
    # 后 25% = token 6,7（2 个）
    assert out["late_75_1"]["n"] == 2


def test_judge_three_verdicts():
    """判据三态：PASS / FAIL / BOUNDARY。"""
    # PASS：正比例高且均值在 ±1.0
    pos = np.full((10, 1), 0.5, dtype=np.float32)
    mask = np.ones((10, 1), dtype=bool)
    j = judge(analyze_delta_distribution(pos, mask))
    assert j["verdict"] == "PASS" and j["passed"] is True
    # FAIL：均值 < -1.0
    neg = np.full((10, 1), -2.0, dtype=np.float32)
    j = judge(analyze_delta_distribution(neg, mask))
    assert j["verdict"] == "FAIL" and j["passed"] is False
    # FAIL：正比例 < 5%
    sparse = np.zeros((100, 1), dtype=np.float32); mask100 = np.ones((100, 1), dtype=bool)
    sparse[:4] = 1.0       # 4% 正
    j = judge(analyze_delta_distribution(sparse, mask100))
    assert j["verdict"] == "FAIL" and j["passed"] is False
    # BOUNDARY：正比例 10%（5%~15% 之间），均值 0
    bd = np.zeros((100, 1), dtype=np.float32)
    bd[:10] = 1.0
    j = judge(analyze_delta_distribution(bd, mask100))
    assert j["verdict"] == "BOUNDARY" and j["passed"] is False


def test_load_cache_and_full_run(tmp_path, capsys):
    """磁盘 cache 加载 + main() 全流程（构造小 cache）。"""
    cache, prompts, responses = _make_cache()
    prefix = str(tmp_path / "c")
    write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    rc = main(["--prefix", prefix, "--out", str(tmp_path / "r.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[D3] 判据:" in out
    import json
    with open(str(tmp_path / "r.json"), encoding="utf-8") as f:
        report = json.load(f)
    assert report["full_distribution"]["n"] > 0
    assert report["verdict"]["verdict"] in ("PASS", "FAIL", "BOUNDARY")


def test_student_overlap():
    """student top-K 重合率：teacher 排序 ids 上的 searchsorted 命中。"""
    teacher_ids = np.array([[[1, 5, 9], [2, 6, 10]]])   # (1,2,3)
    student = {"0": [1, 9, 99, 100]}                    # 命中 2/4
    out = compute_student_overlap(teacher_ids, student)
    assert out["status"] == "ok"
    assert abs(out["overall_hit_ratio"] - 0.5) < 1e-6
    assert out["n_samples"] == 1


def test_student_overlap_no_data():
    out = compute_student_overlap(np.zeros((1, 2, 3)), {})
    assert out["status"] == "no-data"
