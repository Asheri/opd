"""L2 集成测试：DisagreementComputer + rollout 相位（任务 2.2）。

覆盖 §3.3 rollout 相位端到端：selective 选 prompt -> student 生成
-> 4 个 chosen logp -> D_i^abs -> append_refresh（teacher 前向在此，_train_step 保持 teacher-free）。
"""
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