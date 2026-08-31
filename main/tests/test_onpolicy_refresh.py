"""P-OPD v2 纯 on-policy 改造测试（2026-08-31，v2 d81b8c0 基线重建）。

v2 基础上删除教师预计算（stage1）+ base 池（固定 D），训练 = 纯 on-policy 交替相位
（run_refresh_phase ↔ train_refresh_phase，α=1.0），教师 Δ 用 only_stu 口径（学生 top-K
完整支撑，无交集稀释）。

覆盖：
1. `_rl_ref_delta_only_stu` 与官方公式（gather−logsumexp）数学一致 + 跨词表 clamp；
2. run_refresh_phase 的 ring buffer ids = 学生 top-K 支撑（only_stu 语义）；
3. pure_refresh pipeline：无 base 池（scheduler.run 不调用）、训练步 100% refresh；
4. RefreshRingBuffer 断点 round-trip（防服务器关机续跑）。
"""
import pytest
import torch

from fullstack_opd_v2 import adaptive_cache as ac
from fullstack_opd_v2.adaptive_cache import (
    RefreshRingBuffer, DisagreementComputer, run_refresh_phase,
    _rl_ref_delta_only_stu)
from fullstack_opd_v2.model import CausalToyLM, response_dists


def _make_toy(vocab=8, d_model=8, n_layers=1):
    return CausalToyLM(vocab=vocab, d_model=d_model, n_layers=n_layers)


# ============================================================================
# 1. only_stu Δ 数学一致性（vs 官方 _compute_teacher_top_k_log_probs）
# ============================================================================

def test_only_stu_matches_official_formula():
    """_rl_ref_delta_only_stu（输入已归一 log-softmax 分布）与官方公式
    gather(raw_logits, ids) − logsumexp(raw_logits) 逐元素一致。"""
    torch.manual_seed(0)
    B, T, V, Ks = 4, 10, 32, 8
    s_old_ids = torch.randint(0, V, (B, T, Ks))
    rl_logits = torch.randn(B, T, V)
    ref_logits = torch.randn(B, T, V)
    delta_ours = _rl_ref_delta_only_stu(
        s_old_ids, rl_logits.log_softmax(-1), ref_logits.log_softmax(-1))
    # 官方 only_stu：on_student_log_probs = gather(logits, student_ids) − logsumexp(logits)
    rl_le = torch.logsumexp(rl_logits, dim=-1, keepdim=True)
    ref_le = torch.logsumexp(ref_logits, dim=-1, keepdim=True)
    rl_logp = rl_logits.gather(-1, s_old_ids) - rl_le
    ref_logp = ref_logits.gather(-1, s_old_ids) - ref_le
    assert torch.allclose(delta_ours, rl_logp - ref_logp, atol=1e-5)


def test_only_stu_delta_on_student_support():
    """only_stu Δ 定义在【学生 top-K 完整支撑】上：每个学生 token 都有值（非交集稀释）。"""
    torch.manual_seed(1)
    B, T, V, Ks = 3, 6, 16, 5
    s_old_ids = torch.randint(0, V, (B, T, Ks))
    rl_logits = torch.randn(B, T, V)
    ref_logits = torch.randn(B, T, V)
    delta = _rl_ref_delta_only_stu(
        s_old_ids, rl_logits.log_softmax(-1), ref_logits.log_softmax(-1))
    assert delta.shape == (B, T, Ks)
    assert torch.isfinite(delta).all()


def test_only_stu_cross_vocab_clamp():
    """F2：学生 vocab > 教师 vocab 时学生 top-K 超界 → Δ 置 0 不越界。"""
    torch.manual_seed(4)
    B, T, V_s, V_t, Ks = 3, 6, 8, 5, 4
    s_old_ids = torch.randint(0, V_s, (B, T, Ks))   # 含 5-7 超教师词表
    rl = torch.randn(B, T, V_t).log_softmax(-1)
    ref = torch.randn(B, T, V_t).log_softmax(-1)
    delta = _rl_ref_delta_only_stu(s_old_ids, rl, ref)
    assert delta.shape == (B, T, Ks)
    assert torch.isfinite(delta).all()
    oob = s_old_ids >= V_t
    if oob.any():
        assert torch.all(delta[oob] == 0)
    inb = ~oob
    expect = rl.gather(-1, s_old_ids.clamp(max=V_t - 1)) - ref.gather(-1, s_old_ids.clamp(max=V_t - 1))
    assert torch.allclose(delta[inb], expect[inb], atol=1e-6)


def test_refresh_phase_rejects_ks_neq_topk():
    """F1：student_top_k != top_k 时 run_refresh_phase 抛 ValueError（only_stu 槽位宽一致）。"""
    torch.manual_seed(5)
    V = 8
    stu = _make_toy(V)
    t_rl = _make_toy(V)
    t_ref = _make_toy(V)
    s_ref = _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V, student_top_k=5)  # Ks=5 ≠ Kt=3
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (4, 5))
    with pytest.raises(ValueError, match="student_top_k"):
        run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                          prompts, step=1, version=1, m_selected=4,
                          max_resp_len=6, top_k=3, device="cpu")


# ============================================================================
# 2. run_refresh_phase 的 ring buffer 存学生支撑（only_stu 语义）
# ============================================================================

def test_refresh_phase_ring_buffer_ids_are_student_support():
    """run_refresh_phase 后 ring buffer 的 ids = 行为学生在该响应上的 top-K 支撑
    （only_stu），而非教师 rl 的 top-K。delta_k 在【学生支撑】上有值。"""
    torch.manual_seed(0)
    V = 8
    stu = _make_toy(V)
    t_rl = _make_toy(V)
    t_ref = _make_toy(V)
    s_ref = _make_toy(V)
    K = 3
    rb = RefreshRingBuffer(capacity=8, top_k=K, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (4, 5))
    run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                      prompts, step=1, version=1, m_selected=4,
                      max_resp_len=6, top_k=K, device="cpu")
    assert rb.size == 4
    for k in range(rb.size):
        resp = rb._response[k]                          # (T,)
        p = prompts[rb._prompt_idx[k]].unsqueeze(0)     # (1,P)
        d = response_dists(stu, p, resp.unsqueeze(0))   # (1,T,V)
        expect = d.topk(K, -1).indices[0]               # (T,K) 学生 top-K
        assert torch.equal(rb.ids[k], expect), \
            f"槽 {k}：ring buffer ids 非学生 top-K 支撑（only_stu 语义破坏）"
        assert rb.delta_k[k].numel() == rb.ids[k].numel()


def test_refresh_phase_only_stu_delta_finite():
    """only_stu 教师前向的 Δ 数值有限且非全零（教师 rl≠ref 前向有区分度）。"""
    torch.manual_seed(2)
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
    dk = rb.delta_k[:rb.size]
    assert torch.isfinite(dk).all()
    assert dk.abs().sum() > 0


# ============================================================================
# 3. pure_refresh pipeline：无 base 池、100% on-policy
# ============================================================================

def test_pure_refresh_loop_no_base_pool(tmp_path, monkeypatch):
    """l2.pure_refresh=true + stage1.skip=true：
    - scheduler.run（base 池训练）完全不调用；
    - 训练步全部 pool=="refresh"（100% on-policy）；
    - rollout 相位发生且样本 append 成功。"""
    from fullstack_opd_v2.scheduler import AsyncBatchedScheduler
    from fullstack_opd_v2.config import load_config
    from fullstack_opd_v2.pipeline import FullStackOPDv2

    base_calls = {"n": 0}
    orig_run = AsyncBatchedScheduler.run
    def spy_run(self, *a, **k):
        base_calls["n"] += 1
        return orig_run(self, *a, **k)
    monkeypatch.setattr(AsyncBatchedScheduler, "run", spy_run)

    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.pure_refresh=true", "stage1.skip=true",
        "l2.t_train=2", "stage2.n_steps=6",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4",
        "l2.cache.min_refresh_pool=0",
        "l2.rollout.temperature=1.0"])
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    assert base_calls["n"] == 0, "pure_refresh 下 base 池训练被调用（应跳过）"
    train_rows = [m for m in out["metrics"]
                  if isinstance(m, dict) and "pool" in m]
    assert train_rows, "无任何 refresh 训练步（纯 on-policy 未闭环）"
    assert all(m["pool"] == "refresh" for m in train_rows)
    assert len(train_rows) == 6
    rollout_rows = [m for m in out["metrics"]
                    if isinstance(m, dict) and m.get("phase") == "rollout"]
    assert rollout_rows, "pure_refresh 下 rollout 相位未触发"
    assert all(r.get("rollout/n_appended", 0) > 0 for r in rollout_rows)


def test_pure_refresh_respects_n_steps(tmp_path):
    """pure_refresh 训练步数精确到 n_steps（n_steps=5, t_train=2 → 3 相位 5 步）。"""
    from fullstack_opd_v2.config import load_config
    from fullstack_opd_v2.pipeline import FullStackOPDv2
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.pure_refresh=true", "stage1.skip=true",
        "l2.t_train=2", "stage2.n_steps=5",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4",
        "l2.cache.min_refresh_pool=0"])
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    train_rows = [m for m in out["metrics"]
                  if isinstance(m, dict) and "pool" in m]
    assert len(train_rows) == 5


# ============================================================================
# 4. RefreshRingBuffer 断点 round-trip（防服务器关机续跑）
# ============================================================================

def test_refresh_ring_buffer_checkpoint_roundtrip():
    """ring buffer state_dict → load_state_dict 无损（容量/top_k 以构造为准，内容恢复）。"""
    torch.manual_seed(3)
    V = 8
    stu = _make_toy(V)
    t_rl = _make_toy(V)
    t_ref = _make_toy(V)
    s_ref = _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (6, 5))
    run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                      prompts, step=1, version=1, m_selected=6,
                      max_resp_len=6, top_k=3, device="cpu")
    sd = rb.state_dict()
    rb2 = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    rb2.load_state_dict(sd)
    assert rb2.size == rb.size == 6
    assert rb2._write_pos == rb._write_pos
    assert torch.equal(rb2.ids[:rb2.size], rb.ids[:rb.size])
    assert torch.equal(rb2.delta_k[:rb2.size], rb.delta_k[:rb.size])
    assert rb2._prompt_idx == rb._prompt_idx
    assert rb2._resp_lens == rb._resp_lens
