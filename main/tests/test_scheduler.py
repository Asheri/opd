"""scheduler.py 单测：异步调度器跑满步数、字段有限、staleness 年龄有上界、版本递增。"""
from __future__ import annotations

import math
import sys

import pytest
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


def test_scheduler_worker_hf_branch_uses_factory(monkeypatch):
    """P1-B（二次审查）：model_kind='hf' 注入 s2cfg 后，worker 走 build_model 而非
    CausalToyLM（旧版恒走 toy 分支、对无 n_layers 的 HFCausalLM 取 n_layers → AttributeError）。"""
    import unittest.mock as mock
    import fullstack_opd_v2.model_factory as MF
    from fullstack_opd_v2.scheduler import AsyncBatchedScheduler

    fake_worker = mock.Mock()
    monkeypatch.setattr(MF, "build_model", mock.Mock(return_value=fake_worker))

    # 模拟 HFCausalLM student：故意不设 n_layers，验证 hf 分支不触碰它
    student = mock.Mock()
    student.vocab = 24
    student.d_model = 16
    student.state_dict.return_value = {}
    param = torch.nn.Parameter(torch.zeros(4))
    student.parameters.return_value = [param]         # Adam 需要非空参数列表
    student.response_dists = mock.Mock(return_value=torch.zeros(2, 4, 24))
    cache = mock.Mock()
    cache.mode = "dense"
    cache.top_k = 0
    prompts = torch.zeros(6, 4, dtype=torch.long)
    responses = torch.zeros(6, 5, dtype=torch.long)
    cfg = _cfg(model_kind="hf", student_path="X")     # 顶层注入后的 s2cfg 形态
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  None, None, None, cfg, "cpu")
    assert sched.worker is fake_worker
    MF.build_model.assert_called_once_with(cfg, "cpu", role="student")


def test_scheduler_topk_renormalize_wires_through(monkeypatch):
    """P2（二次审查·测试缺口）：renormalize_topk_support=true 必须经 scheduler 传到
    pg_loss（renormalize_support=True + 显式 student top-K support 掩码）与
    low_var_kl_support（renormalize_support=True）——PG 与 KL 同支撑同步开。"""
    import fullstack_opd_v2.scheduler as SCH

    pg_kwargs = {}
    kl_kwargs = {}
    # ⚠️ 必须先保存原实现再 patch——否则 spy 内部 SCH.pg_loss 指向自己 → 无限递归挂起。
    orig_pg = SCH.pg_loss
    orig_kl = SCH.low_var_kl_support

    def spy_pg(s_cur, s_old, delta, mask=None, clip_eps=0.2, p_old=None,
               log_ratio_max=None, log_ratio_clip=None,
               renormalize_support=False, support=None,
               delta_clip=None):
        pg_kwargs.update(renormalize_support=renormalize_support, support=support,
                         has_support=support is not None)
        # 代理真实实现（不 monkeypatch 掉数学）
        return orig_pg(s_cur=s_cur, s_old=s_old, delta=delta, mask=mask,
                       clip_eps=clip_eps, p_old=p_old,
                       log_ratio_max=log_ratio_max, log_ratio_clip=log_ratio_clip,
                       renormalize_support=renormalize_support, support=support,
                       delta_clip=delta_clip)
    def spy_kl(s_topk_logp, ref_logp_at_support, mask=None, renormalize_support=False):
        kl_kwargs.update(renormalize_support=renormalize_support)
        return orig_kl(s_topk_logp, ref_logp_at_support, mask, renormalize_support)
    # scheduler 在模块顶部 from .losses import pg_loss / low_var_kl_support——
    # 绑定的是 scheduler 命名空间的引用，patch scheduler 层才生效。
    monkeypatch.setattr(SCH, "pg_loss", spy_pg)
    monkeypatch.setattr(SCH, "low_var_kl_support", spy_kl)

    student, cache, prompts, responses, ref_ids, ref_logp, cfg = _setup_topk()
    cfg["renormalize_topk_support"] = True
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  None, ref_ids, ref_logp, cfg, "cpu")
    metrics = sched.run(6)
    assert len(metrics) == 6
    # PG 收到 renormalize 且带显式 support 掩码；KL 同步收到 renormalize
    assert pg_kwargs.get("renormalize_support") is True
    assert pg_kwargs.get("has_support") is True
    assert kl_kwargs.get("renormalize_support") is True
    for m in metrics:
        assert math.isfinite(m["loss"])


def test_scheduler_adamw_8bit_optimizer(monkeypatch):
    """多学生并发：optimizer=adamw_8bit → 用 bnb AdamW8bit（mock，CPU 无法真跑 bnb）。"""
    import unittest.mock as mock
    import fullstack_opd_v2.scheduler as SCH

    fake_opt = mock.Mock()
    # bnb 在 scheduler._build_optimizer 内 from bitsandbytes.optim import AdamW8bit（局部导入）——
    # 本地测试 CPU 无法真装 bnb，monkeypatch bitsandbytes 模块模拟可用（scheduler 模块本身
    # 无 AdamW8bit 属性，不能 setattr SCH.AdamW8bit）。
    import types
    bnb = types.ModuleType("bitsandbytes")
    bnb_optim = types.ModuleType("bitsandbytes.optim")
    bnb_optim.AdamW8bit = mock.Mock(return_value=fake_opt)
    bnb.optim = bnb_optim
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.optim", bnb_optim)

    student, cache, prompts, responses, ref_dists = _setup(seed=11)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses, ref_dists,
                                  None, None, _cfg(optimizer="adamw_8bit"), "cpu")
    assert sched.opt is fake_opt


def test_scheduler_adamw_8bit_without_bnb_raises(monkeypatch):
    """adamw_8bit 但 bnb 缺失 → 显式报错（不静默回退 fp32 导致 OOM）。"""
    import unittest.mock as mock
    import fullstack_opd_v2.scheduler as SCH
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "bitsandbytes.optim":
            raise ImportError("no bnb")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    student, cache, prompts, responses, ref_dists = _setup(seed=12)
    with pytest.raises(RuntimeError):
        AsyncBatchedScheduler(student, cache, prompts, responses, ref_dists,
                              None, None, _cfg(optimizer="adamw_8bit"), "cpu")


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


def test_scheduler_cross_vocab_student_topk_train():
    """方案 A（对齐 Direct-OPD）：student vocab > teacher vocab 的 7B 风格端到端训练。

    teacher vocab=20、student vocab=32（模拟 7B 152064 vs teacher 151936）。student top-K
    含超出 teacher 词表的 id（20..31）→ Δ_T=0（未命中）；训练必须跑通、loss 有限。
    """
    N, P, T = 8, 4, 6
    Vt, Vs, K = 20, 32, 6
    g = torch.Generator().manual_seed(0)
    prompts = torch.randint(0, Vt, (N, P), generator=g)
    responses = torch.randint(0, Vt, (N, T), generator=g)
    teacher_rl = CausalToyLM(vocab=Vt, d_model=16, n_layers=1)
    teacher_ref = CausalToyLM(vocab=Vt, d_model=16, n_layers=1)
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, teacher_rl, teacher_ref)
    # student：vocab=32，但数据 token 只在 [0,20)（teacher 词表内），prompt/response 复用
    student = CausalToyLM(vocab=Vs, d_model=16, n_layers=1)
    with torch.no_grad():
        full = response_dists(student, prompts, responses)   # (N,T,32) student 前向
    # ref 锚点 = 初始 student 分布的 top-K（与 teacher 词表无关，scheduler 只用其值）
    ref_ids, ref_logp = full.topk(K, dim=-1).indices, full.topk(K, dim=-1).values
    # student 前向得到 (B,T,32)，top-K 里可能有 ≥20 的 id（低概率垃圾 token 也可能进 top-K）
    cfg = _cfg(cache_mode="topk", top_k_student=K, ref_topk=K, n_steps=6, batch_size=4)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  None, ref_ids, ref_logp, cfg, "cpu")
    metrics = sched.run(6)
    assert len(metrics) == 6
    for m in metrics:
        for k in ("loss", "pg_loss", "kl_loss"):
            assert math.isfinite(m[k]), f"{k} 非有限: {m[k]}"


def test_g5_base_skips_staleness_drop():
    """G5（§2 Q4）：staleness_drop_base=False 时 base 样本跳过陈旧度截断（恒新不误伤）。
    默认 True 保持原双截断语义；L2 交替相位把 base 置 False 显式落实契约。"""
    student, cache, prompts, responses, ref_dists = _setup(seed=21)
    idxs = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        s_old = response_dists(student, prompts[idxs], responses[idxs])
    delta = cache.get_delta(idxs)
    # 对照组：默认 True → 陈旧样本被截断（返回 None）
    cfg2 = _cfg()
    sched2 = AsyncBatchedScheduler(student, cache, prompts, responses,
                                   ref_dists, None, None, cfg2, "cpu")
    for _ in range(5):
        sched2.staleness_q.advance_version()
    assert sched2._train_step(0, idxs, s_old, delta, 0) is None, \
        "默认 base 陈旧样本仍按原语义截断"
    # G5：staleness_drop_base=False → base 陈旧样本仍被训练
    cfg = _cfg(staleness_drop_base=False)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  ref_dists, None, None, cfg, "cpu")
    for _ in range(5):
        sched.staleness_q.advance_version()
    m = sched._train_step(0, idxs, s_old, delta, 0)
    assert m is not None, "staleness_drop_base=False 时 base 陈旧样本仍应被训练（G5 契约）"
    assert math.isfinite(m["loss"])
