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


def _setup_topk(N=6, P=4, T=5, V=24, d=16, L=1, K=6, seed=0):
    """稀疏 topk 模式：topk 缓存 + 稀疏 ref 锚点（M6：P1-1 优化主场的端到端覆盖）。

    返回 (student, cache, prompts, responses, ref_ids, ref_logp, cfg)。
    """
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(0, V, (N, P), generator=g)
    responses = torch.randint(0, V, (N, T), generator=g)
    teacher_rl = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    teacher_ref = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, teacher_rl, teacher_ref)
    student = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    with torch.no_grad():
        full = response_dists(student, prompts, responses)
    ref_ids, ref_logp = full.topk(K, dim=-1).indices, full.topk(K, dim=-1).values
    cfg = _cfg(cache_mode="topk", top_k_student=K, ref_topk=K)
    return student, cache, prompts, responses, ref_ids, ref_logp, cfg


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
                "trained_steps", "waste_ratio", "rollout_idle_s", "scorer_idle_s",
                "age_histogram")) <= set(s)
    assert isinstance(s["age_histogram"], dict)
    assert sum(s["age_histogram"].values()) == 6   # 每步一个 age
    # M5：waste 拆解为 陈旧(put+consume) / 队满 / 停机尾 三源，且口径封闭（恒等式成立）
    assert 0.0 <= s["stale_discard_ratio"] <= 1.0
    assert s["dropped_queue_full"] >= 0
    assert s["shutdown_tail"] >= 0
    assert s["rollout_forwards"] == (s["trained_steps"] + s["dropped_at_put"]
                                     + s["dropped_at_consume"] + s["dropped_queue_full"]
                                     + s["shutdown_tail"])


def test_train_step_dense_fetches_delta_when_none():
    """M2 回归：dense 模式 `_train_step` 传 delta=None（分布式路径由 worker 回传 idxs/s_old、
    不送 Δ_T）必须现场从缓存零拷贝取 Δ_T，而非 ratio*None 崩溃。"""
    student, cache, prompts, responses, ref_dists = _setup(seed=5)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  ref_dists, None, None, _cfg(), "cpu")
    idxs = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        s_old = response_dists(student, prompts[idxs], responses[idxs])   # CPU 张量（模拟 worker 回传）
    m = sched._train_step(0, idxs, s_old, None, 0)                        # delta=None（M2 现场取）
    assert m is not None
    assert m["batch"] == 4
    for k in ("loss", "pg_loss", "kl_loss"):
        assert math.isfinite(m[k]), f"{k} 非有限: {m[k]}"


def test_scheduler_topk_mode_runs_end_to_end():
    """稀疏 topk 训练分支真实端到端跑通（M6）。

    回归：此前 test_scheduler 全跑 dense 模式，P1-1 的 searchsorted/_ref_logp
    二分主场（_train_step use_topk 分支）从未被覆盖。本测试构造 topk 缓存 +
    稀疏 ref 锚点，跑满步数并断言损失有限。
    """
    student, cache, prompts, responses, ref_ids, ref_logp, cfg = _setup_topk()
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  None, ref_ids, ref_logp, cfg, "cpu")
    metrics = sched.run(6)
    assert len(metrics) == 6
    assert sched.use_topk is True
    assert sched.kl_mode == "topk"
    for m in metrics:
        for k in ("loss", "pg_loss", "kl_loss"):
            assert math.isfinite(m[k]), f"{k} 非有限: {m[k]}"


def test_scheduler_on_step_callback_invoked_per_step():
    """T8：run(n_steps, on_step=cb) 每成功一步调一次 cb，次数 === n_steps。"""
    student, cache, prompts, responses, ref_dists = _setup(seed=7)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  ref_dists, None, None, _cfg(n_steps=5), "cpu")
    calls = []
    sched.run(5, on_step=lambda m: calls.append(m["step"]))
    assert len(calls) == 5
    assert calls == [0, 1, 2, 3, 4]


def test_train_step_metrics_finite_collected():
    """C3：.item() 收集后一次同步，指标仍有限。"""
    student, cache, prompts, responses, ref_dists = _setup(seed=9)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses, ref_dists,
                                  None, None, _cfg(n_steps=4), "cpu")
    ms = sched.run(4)
    for m in ms:
        for k in ("loss", "pg_loss", "kl_loss", "adv_mean", "reward"):
            assert math.isfinite(m[k])
