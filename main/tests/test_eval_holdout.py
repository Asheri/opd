"""D1 固定评估集单测：cache.slice 切分、_eval_holdout 计算、默认关闭零回归。

覆盖：TensorTeacherCache/DiskTeacherCache slice 行数；topk 模式 _eval_holdout 返回
合理标量；eval_every=0 时默认全关（_eval_holdout 返回 None、metrics 无评估步）。
"""
from __future__ import annotations

import torch

from fullstack_opd_v2.cache import TensorTeacherCache
from fullstack_opd_v2.cache_store import DiskTeacherCache, write_cache_disk
from fullstack_opd_v2.model import CausalToyLM, response_dists
from fullstack_opd_v2.scheduler import AsyncBatchedScheduler


def _setup_topk(N=8, P=4, T=6, V=24, d=16, L=1, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(1, V, (N, P), generator=g)
    responses = torch.randint(1, V, (N, T), generator=g)
    teacher_rl = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    teacher_ref = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, teacher_rl, teacher_ref)
    student = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    with torch.no_grad():
        full = response_dists(student, prompts, responses)
    ref_ids, ref_logp = full.topk(K, dim=-1).indices, full.topk(K, dim=-1).values
    cfg = dict(batch_size=4, staleness_threshold=4, queue_size=8,
               kl_reg_coef=0.05, clip_eps=0.2, grad_clip=1.0, lr=1e-3,
               n_steps=8, dtype="fp32", cache_mode="topk", top_k_student=K,
               ref_topk=K)
    return student, cache, prompts, responses, ref_ids, ref_logp, cfg


def _make_sched(student, cache, prompts, responses, ref_ids, ref_logp, cfg,
                eval_cache=None, eval_prompts=None, eval_responses=None):
    return AsyncBatchedScheduler(student, cache, prompts, responses,
                                 None, ref_ids, ref_logp, cfg, "cpu",
                                 eval_cache=eval_cache, eval_prompts=eval_prompts,
                                 eval_responses=eval_responses)


def test_tensor_cache_slice_rows():
    """TensorTeacherCache.slice：行数切分 + topk 数据正确。"""
    _, cache, _, _, _, _, _ = _setup_topk(N=8)
    train = cache.slice(0, 6)
    holdout = cache.slice(6)
    assert train.ids.shape[0] == 6
    assert holdout.ids.shape[0] == 2
    # 切片后数据与源一致（共享底层张量）
    assert torch.equal(holdout.ids, cache.ids[6:])


def test_disk_cache_slice_rows(tmp_path):
    """DiskTeacherCache.slice：memmap 视图切分 + num_samples 收缩。"""
    _, cache, prompts, responses, _, _, _ = _setup_topk(N=8)
    prefix = str(tmp_path / "c")
    write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    disk = DiskTeacherCache(prefix, device="cpu", top_k=cache.top_k, vocab=cache.vocab)
    train = disk.slice(0, 6)
    holdout = disk.slice(6)
    assert train.num_samples == 6
    assert holdout.num_samples == 2
    assert train._delta.shape[0] == 6 and holdout._delta.shape[0] == 2


def test_eval_holdout_default_off_returns_none():
    """默认 eval_every=0：_eval_holdout 返回 None，run 的 metrics 无 eval 步。"""
    student, cache, prompts, responses, ref_ids, ref_logp, cfg = _setup_topk()
    sched = _make_sched(student, cache, prompts, responses, ref_ids, ref_logp, cfg)
    assert sched._eval_holdout() is None
    metrics = sched.run(4)
    assert all("eval_reward" not in m or m.get("eval_reward") is None for m in metrics)


def test_eval_holdout_topk_returns_scalar():
    """eval_every>0 + holdout：_eval_holdout 返回有限标量（topk 支撑口径）。"""
    student, cache, prompts, responses, ref_ids, ref_logp, cfg = _setup_topk(N=8)
    train_n = 6
    eval_cache = cache.slice(train_n)
    cfg = dict(cfg, eval_every=2, eval_chunk=4)
    sched = _make_sched(student, cache.slice(0, train_n), prompts[:train_n],
                        responses[:train_n], ref_ids[:train_n], ref_logp[:train_n],
                        cfg, eval_cache=eval_cache, eval_prompts=prompts[train_n:],
                        eval_responses=responses[train_n:])
    v = sched._eval_holdout()
    assert v is not None
    assert v == v  # 非 NaN
    assert abs(v) < 1e6


def test_eval_holdout_every_n_steps_records_metric():
    """eval_every=2：run 后恰好在第 1/3/5 个成功步的 metric 上出现 eval_reward。"""
    student, cache, prompts, responses, ref_ids, ref_logp, cfg = _setup_topk(N=8)
    train_n = 6
    eval_cache = cache.slice(train_n)
    cfg = dict(cfg, eval_every=2, eval_chunk=4)
    sched = _make_sched(student, cache.slice(0, train_n), prompts[:train_n],
                        responses[:train_n], ref_ids[:train_n], ref_logp[:train_n],
                        cfg, eval_cache=eval_cache, eval_prompts=prompts[train_n:],
                        eval_responses=responses[train_n:])
    metrics = sched.run(6)
    eval_steps = [m["step"] for m in metrics if m.get("eval_reward") is not None]
    assert set(eval_steps) == {1, 3, 5}
