"""scheduler.py 单测：异步调度器跑满步数、字段有限、staleness 年龄有上界、版本递增。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fullstack_opd_v2.model import CausalToyLM, response_dists
from fullstack_opd_v2.cache import TensorTeacherCache
from fullstack_opd_v2.scheduler import AsyncBatchedScheduler


def _setup(N=8, P=4, T=6, V=24, d=16, L=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(0, V, (N, P), generator=g)
    responses = torch.randint(0, V, (N, T), generator=g)
    teacher_rl = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    teacher_ref = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    cache = TensorTeacherCache(True, 0).build(prompts, responses, teacher_rl, teacher_ref)
    student = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    with torch.no_grad():
        ref_dists = response_dists(student, prompts, responses)
    return student, cache, prompts, responses, ref_dists


def _cfg(**over):
    cfg = dict(batch_size=4, staleness_threshold=4, queue_size=8,
               kl_reg_coef=0.05, clip_eps=0.2, grad_clip=1.0, lr=1e-3,
               n_steps=8, dtype="fp32", cache_mode="dense", top_k_student=0)
    cfg.update(over)
    return cfg


def test_scheduler_runs_all_steps_and_fields_finite():
    student, cache, prompts, responses, ref_dists = _setup()
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  ref_dists, None, None, _cfg(), "cpu")
    metrics = sched.run(8)
    assert len(metrics) == 8
    for m in metrics:
        for k in ("loss", "pg_loss", "kl_loss", "adv_mean", "reward"):
            assert math.isfinite(m[k]), f"{k} 非有限: {m[k]}"
        assert m["batch"] == 4


def test_scheduler_version_strictly_increasing():
    student, cache, prompts, responses, ref_dists = _setup(seed=1)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  ref_dists, None, None, _cfg(), "cpu")
    metrics = sched.run(8)
    versions = [m["version"] for m in metrics]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)          # 严格递增
    assert versions[0] >= 1


def test_scheduler_staleness_age_bounded_by_threshold():
    threshold = 3
    student, cache, prompts, responses, ref_dists = _setup(seed=2)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  ref_dists, None, None,
                                  _cfg(staleness_threshold=threshold), "cpu")
    metrics = sched.run(10)
    # 消费侧截断保证 age = 新版本 − 样本版本 ≤ threshold + 1（publish 自增那一下）
    for m in metrics:
        assert 0 <= m["age"] <= threshold + 1, f"age 越界: {m['age']}"


def test_scheduler_reward_is_real_scalar():
    student, cache, prompts, responses, ref_dists = _setup(seed=3)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  ref_dists, None, None, _cfg(), "cpu")
    metrics = sched.run(6)
    rewards = [m["reward"] for m in metrics]
    assert all(isinstance(r, float) for r in rewards)


def test_scheduler_summary_reports_waste():
    student, cache, prompts, responses, ref_dists = _setup(seed=4)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  ref_dists, None, None, _cfg(n_steps=6), "cpu")
    sched.run(6)
    s = sched.summary
    assert s["trained_steps"] == 6
    assert s["rollout_forwards"] >= 6
    assert 0.0 <= s["waste_ratio"] <= 1.0
    assert set(("rollout_forwards", "dropped_at_put", "dropped_at_consume",
                "trained_steps", "waste_ratio", "rollout_idle_s", "scorer_idle_s")) <= set(s)
