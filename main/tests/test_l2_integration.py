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
    n = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                          prompts, step=1, version=1, m_selected=4,
                          max_resp_len=6, top_k=3, device="cpu")
    assert n == 4
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
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4"])
    opd = FullStackOPDv2(cfg, device="cpu")
    out = opd.run(run_dir=str(tmp_path))
    assert len(out["metrics"]) == 12
    # n_steps=12, t_train=5 → 相位边界 5/10 各触发一次 rollout 刷新（至少 2 轮）
    assert calls["n"] >= 2


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
    # 同 seed 确定性：l2 关闭时完全走原路径，loss 逐步一致
    for a, b in zip(out_base["metrics"], out_off["metrics"]):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-5)


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
    """E0-E6 每个实验都能生成合法配置，且各模块开关状态符合矩阵定义（§10）。"""
    from fullstack_opd_v2.experiment import (
        EXPERIMENT_MATRIX, build_config)
    from fullstack_opd_v2.config import ConfigError
    for name in EXPERIMENT_MATRIX:
        cfg = build_config(name, n_steps=10)
        assert cfg["l2"]["enabled"] is (name != "E0_baseline_off")
        assert cfg["stage2"]["batch_size"] == 4   # toy/CPU 友好默认注入
    # 未知实验名须报错
    with pytest.raises(KeyError):
        build_config("E99_unknown")


def test_e0_e6_matrix_off_configs():
    """E2-E6 的单项 ablation 开关反映到 l2 子配置（每模块可独立关闭）。"""
    from fullstack_opd_v2.experiment import build_config
    c2 = build_config("E2_no_selective_rollout")
    assert c2["l2"]["selective_rollout"]["enabled"] is False
    c3 = build_config("E3_no_health_monitor")
    assert c3["l2"]["health_monitor"]["enabled"] is False
    c4 = build_config("E4_fixed_refresh_ratio")
    assert c4["l2"]["refresh_ratio"]["mode"] == "fixed"
    c5 = build_config("E5_no_disagreement")
    assert c5["l2"]["disagreement"]["enabled"] is False
    c6 = build_config("E6_no_value_protect")
    assert c6["l2"]["cache"]["value_protect_quantile"] == 1.0
