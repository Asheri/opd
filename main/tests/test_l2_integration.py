"""L2 集成测试：DisagreementComputer + rollout 相位（任务 2.2）。

覆盖 §3.3 rollout 相位端到端：selective 选 prompt -> student 生成
-> 4 个 chosen logp -> D_i^abs -> append_refresh（teacher 前向在此，_train_step 保持 teacher-free）。
"""
import pytest
import torch

from fullstack_opd_v2.adaptive_cache import (
    RefreshRingBuffer, DisagreementComputer, run_refresh_phase)
from fullstack_opd_v2.model import CausalToyLM


def _make_toy(vocab=8, d_model=8, n_layers=1):
    return CausalToyLM(vocab=vocab, d_model=d_model, n_layers=n_layers)


def test_refresh_phase_produces_disagreement():
    """rollout 相位：student 生成 -> 4 logp -> D_i^abs -> append_refresh，disagreement 非负。"""
    torch.manual_seed(0)
    V = 8
    stu = _make_toy(V)
    t_rl = _make_toy(V)
    t_ref = _make_toy(V)
    s_ref = _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (4, 5))
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=4,
                                max_resp_len=6, top_k=3, device="cpu")
    assert summary["n_total"] == 4
    assert summary["n_appended"] == 4
    assert rb.size == 4
    # D_i^abs 为绝对值聚合，必非负
    assert all(d >= 0.0 for d in rb._disagreements)


def test_refresh_phase_padding_mask_excludes_pad():
    """rollout 相位 mask 只统计有效 token（§3.4），刷新样本 response_length 有界。"""
    torch.manual_seed(1)
    V = 8
    stu = _make_toy(V)
    t_rl = _make_toy(V)
    t_ref = _make_toy(V)
    s_ref = _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (4, 5))
    run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                      prompts, step=1, version=1, m_selected=4,
                      max_resp_len=6, top_k=3, device="cpu")
    # response_length 取自 mask 有效 token 数，应 ≤ 生成长度且 > 0
    for l in rb._resp_lens:
        assert 1 <= l <= 6

# ============================================================================
# 任务 6.1：pipeline 交替相位循环接入 + 工程检查（§13.7）
# ============================================================================

def test_alternating_phase_loop(tmp_path, monkeypatch):
    """L2 交替相位：训练 T_train 步 ↔ rollout 刷新循环（§1/§12）。toy 模型。"""
    import fullstack_opd_v2.adaptive_cache as ac
    from fullstack_opd_v2.config import load_config
    from fullstack_opd_v2.pipeline import FullStackOPDv2
    # spy run_refresh_phase 计数，验证 rollout 相位确实发生（委托真实实现）
    calls = {"n": 0}
    orig = ac.run_refresh_phase
    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(ac, "run_refresh_phase", spy)
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=5", "stage2.n_steps=12",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4",
        # G4：refresh_min_interval 调低，使相位边界 5/10 都触发刷新（验证多次刷新）
        "l2.cache.refresh_min_interval=3",
        # IMP-1d：本测试验证双池闭环（训练必须发生）→ 关冷启动门槛（池小也要训练）
        "l2.cache.min_refresh_pool=0"])
    opd = FullStackOPDv2(cfg, device="cpu")
    out = opd.run(run_dir=str(tmp_path))
    # G1 闭环：base 12 步 + refresh 补充步（refresh 样本真正进训练），故 ≥ 12
    assert len(out["metrics"]) >= 12
    # G4：refresh_min_interval=3 → 相位边界 5/10 各触发一次 rollout 刷新（至少 2 轮）
    assert calls["n"] >= 2
    # G1 核心：refresh 样本必须进入训练（存在 pool=="refresh" 的训练步），
    # 否则 L2 就是"装配不消费"脚手架。这是本轮修复的关键断言。
    assert any(m.get("pool") == "refresh" for m in out["metrics"]), \
        "refresh 样本未进入训练（双池 feeder 未闭环）"


def test_l2_disabled_regression(tmp_path):
    """l2.enabled=false 行为与原 L0/L1 完全一致（回归，§13.7）。"""
    from fullstack_opd_v2.config import load_config
    from fullstack_opd_v2.pipeline import FullStackOPDv2
    base_cfg = load_config(overrides=["stage2.n_steps=5", "stage2.batch_size=4"])
    out_base = FullStackOPDv2(base_cfg, device="cpu").run(run_dir=str(tmp_path / "base"))
    off_cfg = load_config(overrides=["stage2.n_steps=5", "stage2.batch_size=4",
                                     "l2.enabled=false"])
    out_off = FullStackOPDv2(off_cfg, device="cpu").run(run_dir=str(tmp_path / "off"))
    assert len(out_base["metrics"]) == len(out_off["metrics"]) == 5
    # 同 seed 确定性：l2 关闭时完全走原路径。⚠️ 不逐位断言——异步调度器 4 线程并发消费
    # torch 全局 RNG，PromptFeeder 的 torch.randint 与 learner 前向交错顺序因线程调度而异，
    # 两次独立 run 的随机批次序天然不同（实测同代码 1/3 概率逐位通过）。改为结构等价 +
    # 统计近似：指标结构一致、loss 有限、均值量级近似（真回归会是 NaN/结构变化/量级爆炸）。
    import math
    for a, b in zip(out_base["metrics"], out_off["metrics"]):
        assert set(a) == set(b)
        for k in ("loss", "pg_loss", "kl_loss"):
            assert math.isfinite(a[k]) and math.isfinite(b[k])
    base_mean = sum(m["loss"] for m in out_base["metrics"]) / len(out_base["metrics"])
    off_mean = sum(m["loss"] for m in out_off["metrics"]) / len(out_off["metrics"])
    assert base_mean == pytest.approx(off_mean, rel=0.2)


def test_no_teacher_forward_in_train_step():
    """§13.7：_train_step 内无 teacher 前向（teacher-free 内核不变式）。"""
    from fullstack_opd_v2.scheduler import AsyncBatchedScheduler
    from fullstack_opd_v2.cache import TensorTeacherCache
    from fullstack_opd_v2.model import CausalToyLM, response_dists
    V = 8
    teacher_rl = _make_toy(V)
    teacher_ref = _make_toy(V)
    # spy：包装 teacher.forward 计数
    calls = {"n": 0}
    for t in (teacher_rl, teacher_ref):
        orig_forward = t.forward
        def _spy(*a, _orig=orig_forward, **kw):
            calls["n"] += 1
            return _orig(*a, **kw)
        t.forward = _spy
    student = _make_toy(V)
    prompts = torch.randint(0, V, (4, 5))
    responses = torch.randint(0, V, (4, 6))
    cache = TensorTeacherCache(top_k=0)
    cache.build(prompts, responses, teacher_rl, teacher_ref, batch_size=4)
    # cache.build 会调用一次 teacher 前向（离线 build 阶段）——清零后仅统计 _train_step
    calls_before = calls["n"]
    ref_dists = response_dists(student, prompts, responses)
    cfg = {"batch_size": 4, "staleness_threshold": 4, "kl_reg_coef": 0.05,
           "clip_eps": 0.2, "grad_clip": 1.0, "lr": 1e-3, "optimizer": "adam",
           "queue_size": 4, "dtype": "fp32", "cache_mode": "dense", "n_layers": 1}
    sched = AsyncBatchedScheduler(student, cache, prompts, responses,
                                  ref_dists, None, None, cfg, "cpu")
    idxs = torch.tensor([0, 1, 2, 3])
    s_old = response_dists(_make_toy(V), prompts[idxs], responses[idxs]).detach()
    delta = cache.get_delta(idxs)
    m = sched._train_step(0, idxs, s_old, delta, 0)
    assert m is not None
    assert calls["n"] == calls_before   # _train_step 一步内未触碰任何 teacher 前向


def test_no_gpu_memory_growth_l2(tmp_path):
    """L2 交替循环不泄漏线程/资源（GPU 显存增长的 CPU 代理：无线程堆积）。"""
    import threading
    from fullstack_opd_v2.config import load_config
    from fullstack_opd_v2.pipeline import FullStackOPDv2
    before = {t.name for t in threading.enumerate()}
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=3", "stage2.n_steps=6",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4"])
    FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    after = {t.name for t in threading.enumerate()}
    # 无守护线程泄漏（onstep-consumer 等应已退出）
    leaked = after - before - {"MainThread"}
    # 容忍 pytest/logger 自启线程，只断言无 opd 前缀线程残留
    assert not any(n.startswith("opd-") for n in leaked)


def test_no_unbounded_metadata_growth():
    """PromptStateStore 固定形状 O(n_prompts)；reuse_count 有界（§6.1/§4.6）。"""
    from fullstack_opd_v2.adaptive_cache import PromptStateStore, CacheHealthMonitor
    ps = PromptStateStore(n_prompts=50)
    for i in range(1000):
        ps.update(prompt_id=i % 50, reward=float(i % 7), disagreement=0.1,
                  resp_len=8, step=i)
    assert ps.times_seen.shape == (50,)
    assert ps.reward_ema.shape == (50,)
    assert ps.times_seen.sum() == 1000
    hm = CacheHealthMonitor(health={"hit_rate": {"warning": 0.995, "critical": 0.98}})
    for i in range(500):
        hm.record_reuse(sample_id=i % 50)
    assert len(hm._reuse_counts) <= 50   # reuse 计数按 sample_id 去重，有界


# ============================================================================
# 任务 6.2：E0-E6 实验矩阵配置生成
# ============================================================================

def test_e0_e6_matrix_configs_valid():
    """E0-E6 每个实验都能生成合法配置，且各模块开关状态符合矩阵定义（§10 累积构建）。"""
    from fullstack_opd_v2.experiment import (
        EXPERIMENT_MATRIX, build_config)
    for name in EXPERIMENT_MATRIX:
        cfg = build_config(name, n_steps=10)
        assert cfg["l2"]["enabled"] is (name != "E0_base_only")
        assert cfg["stage2"]["batch_size"] == 4   # toy/CPU 友好默认注入
    # 未知实验名须报错
    with pytest.raises(KeyError):
        build_config("E99_unknown")


def test_e0_e6_matrix_off_configs():
    """E1-E6 的累积构建开关反映到 l2 子配置（§10：E1 只加 fixed refresh，逐步叠加）。"""
    from fullstack_opd_v2.experiment import build_config
    # E1：仅 fixed refresh，无 disagreement/health/selective（累积起点）
    c1 = build_config("E1_base_fixed_refresh")
    assert c1["l2"]["refresh_ratio"]["mode"] == "fixed"
    assert c1["l2"]["disagreement"]["enabled"] is False
    assert c1["l2"]["health_monitor"]["enabled"] is False
    assert c1["l2"]["selective_rollout"]["enabled"] is False
    # E3：+health monitor（Oberserve 模块叠加）
    c3 = build_config("E3_add_health_monitor")
    assert c3["l2"]["health_monitor"]["enabled"] is True
    assert c3["l2"]["selective_rollout"]["enabled"] is False
    # E4：+dynamic ratio（fixed → adaptive）
    c4 = build_config("E4_add_dynamic_ratio")
    assert c4["l2"]["refresh_ratio"]["mode"] == "adaptive"
    assert c4["l2"]["selective_rollout"]["enabled"] is False
    # E5：+selective rollout（selector 开启）
    c5 = build_config("E5_add_selective_rollout")
    assert c5["l2"]["selective_rollout"]["enabled"] is True
    # E6：random rollout（all-on 但 selective 关闭，对照 E5 验证 selective 贡献）
    c6 = build_config("E6_random_rollout")
    assert c6["l2"]["selective_rollout"]["enabled"] is False
    assert c6["l2"]["refresh_ratio"]["mode"] == "adaptive"


def test_l2_refresh_fires_without_selective_rollout(tmp_path):
    """回归：刷新相位不得被 selective_rollout.enabled=false（selector=None）误跳过。

    selector=None 表示均匀随机选 prompt（run_refresh_phase 已支持）——E1-E4/E6 与
    S2_E1-E3（selective 关）仍必须执行刷新；否则那些实验只是普通 base 训练（此前
    `and selector is not None` 门控使 S2_E1-E3/E0-E6 的 E1-E4/E6 全部跳过刷新）。
    """
    from fullstack_opd_v2.config import load_config
    from fullstack_opd_v2.pipeline import FullStackOPDv2
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=5", "stage2.n_steps=10",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4",
        "l2.cache.refresh_min_interval=3",
        "l2.selective_rollout.enabled=false",   # selector=None，均匀随机
        # IMP-1d：验证双池闭环（训练必须发生）→ 关冷启动门槛
        "l2.cache.min_refresh_pool=0",
    ])
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    rollout_rows = [m for m in out["metrics"]
                    if isinstance(m, dict) and m.get("phase") == "rollout"]
    assert rollout_rows, "selective 关闭时刷新相位被跳过（门控 bug 回归）"
    assert all(r.get("rollout/n_appended", 0) > 0 for r in rollout_rows)
    # refresh 训练步存在（双池 feeder 闭环，refresh 样本真正进训练）
    assert any(m.get("pool") == "refresh" for m in out["metrics"])
