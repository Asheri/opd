"""性能优化的数值等价性回归：任何优化不得改变不可回退的算法内核。

断言把「优化没改数学」变成 CI 可验证的事实：
  1. pg_loss 加 log_ratio_max=80 后，正常 dense 输入下逐位等于 None（原路径）。
  2. log_ratio_max=80 在支撑外场景恢复「π_old=0 处贡献为 0」的数学真值。
  3. pg_loss 传 p_old 版等于内部计算 s_old.exp() 版。

（searchsorted 支撑匹配的等价性测试见测试文件末尾，属任务 3 的 P1-1。）
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fullstack_opd_v2.losses import pg_loss


def _logp(B, T, V, seed):
    g = torch.Generator().manual_seed(seed)
    return F.log_softmax(torch.randn(B, T, V, generator=g), dim=-1)


def test_pg_loss_log_ratio_max_is_identity_on_normal_input():
    """正常 dense 输入下，log_ratio_max=80 必须逐位等于 None（原路径）。"""
    s_cur = _logp(4, 6, 64, seed=0)
    s_old = _logp(4, 6, 64, seed=1)
    delta = torch.randn(4, 6, 64, generator=torch.Generator().manual_seed(2))
    a = pg_loss(s_cur, s_old, delta)
    b = pg_loss(s_cur, s_old, delta, log_ratio_max=80.0)
    assert torch.equal(a, b), "clamp 必须在正常输入下逐位无影响"


def test_pg_loss_log_ratio_max_equals_support_only_truth():
    """支撑外 s_old=-1e4 + delta=0 下，原路径 NaN；clamp 版恢复「仅支撑内求和」真值。"""
    B, T, V = 2, 3, 2000
    s_cur = _logp(B, T, V, seed=0)
    s_old = torch.full((B, T, V), -1e4)
    s_old[..., :64] = _logp(B, T, 64, seed=1)
    delta = torch.zeros(B, T, V)
    delta[..., :32] = torch.randn(B, T, 32, generator=torch.Generator().manual_seed(2))
    # 原路径 NaN（回归：这是要修的 bug）
    assert torch.isnan(pg_loss(s_cur, s_old, delta)).item()
    # clamp 版有限
    clamped = pg_loss(s_cur, s_old, delta, log_ratio_max=80.0)
    assert torch.isfinite(clamped).item()
    # 且精确等于「仅在支撑内按 π_old 加权求和」的真值
    sc, so, dd = s_cur[..., :64], s_old[..., :64], delta[..., :64]
    ratio = (sc - so).clamp(max=80.0).exp()
    pw = torch.min(ratio * dd, torch.clamp(ratio, 0.8, 1.2) * dd)
    truth = -(so.exp() * pw).sum(-1).mean()
    assert torch.allclose(clamped, truth, atol=1e-6)


def test_pg_loss_p_old_equals_internal_exp():
    """传入 p_old 与内部计算 s_old.exp() 必须逐位相等。"""
    s_cur = _logp(3, 5, 32, seed=0)
    s_old = _logp(3, 5, 32, seed=1)
    delta = torch.randn(3, 5, 32, generator=torch.Generator().manual_seed(2))
    a = pg_loss(s_cur, s_old, delta)
    b = pg_loss(s_cur, s_old, delta, p_old=s_old.exp())
    assert torch.equal(a, b)


def test_response_dists_topk_shape_and_keys():
    """response_dists_topk 把 prompt_logprobs 的 [P:P+T] 响应段拍平成 (B,T,K) 的 (ids,logps)，
    形状、索引与落回设备均与手工输入一致（logprob 以 float32 容差比较）。"""
    import unittest.mock as mock
    from fullstack_opd_v2.rollout_vllm import VLLMRolloutEngine

    B, P, T, V = 2, 3, 4, 6
    full_cap = 6
    device = "cpu"
    k = min(V, full_cap)

    class _LP:
        """vLLM 的 Logprob 对象：行为上只需要 .logprob 属性。"""

        def __init__(self, logprob):
            self.logprob = logprob

    def make_plp(seq_off):
        """构造长度为 P+T 的 prompt_logprobs：prompt 段为 None，响应段 [P:P+T] 为稀疏 dict。
        第 t 个响应 token 的键为 seq_off*100 + t*10 + i（i ∈ [0,V)），值 logprob = t + i*0.1。"""
        plp = [None] * P
        for t in range(T):
            plp.append({seq_off * 100 + t * 10 + i: _LP(float(t) + i * 0.1)
                        for i in range(V)})
        return plp

    plp0, plp1 = make_plp(0), make_plp(1)

    # mock self.llm.generate：返回带 prompt_logprobs 属性的假输出对象。
    fake_llm = mock.MagicMock()
    fake_llm.generate.return_value = [
        mock.Mock(prompt_logprobs=plp0),
        mock.Mock(prompt_logprobs=plp1),
    ]

    # 绕过真实 __init__（本地无 vLLM 会抛 RuntimeError），手工设好方法所需属性。
    eng = object.__new__(VLLMRolloutEngine)
    eng.vocab_size = V
    eng.full_cap = full_cap
    eng.device = device
    eng.llm = fake_llm

    prompts = torch.zeros(B, P, dtype=torch.long)
    responses = torch.zeros(B, T, dtype=torch.long)

    # _VLLM_AVAILABLE=False 时模块级 SamplingParams 是 None，方法体内 SamplingParams(...) 会失败，
    # 直接 patch 成假类（只关心它被调用，不关心真实构造）。
    with mock.patch("fullstack_opd_v2.rollout_vllm.SamplingParams",
                    new=lambda **kw: mock.sentinel.sampling):
        ids, logps = eng.response_dists_topk(prompts, responses)

    # 1) 形状：各 (B,T,K)，K = min(vocab, full_cap)
    assert ids.shape == (B, T, k), ids.shape
    assert logps.shape == (B, T, k), logps.shape
    # 3) 落回 self.device
    assert ids.device.type == device
    assert logps.device.type == device

    # 2) 索引与 logprob 与手工 prompt_logprobs 输入逐位一致
    for b, plp in ((0, plp0), (1, plp1)):
        for t in range(T):
            d = plp[P + t]
            items = list(d.items())
            for j, (tid, lp) in enumerate(items):
                assert ids[b, t, j].item() == tid, (b, t, j)
                # float32 落盘有舍入误差，比较用容差
                assert abs(logps[b, t, j].item() - lp.logprob) < 1e-6, (b, t, j)


def test_searchsorted_match_equals_full_compare():
    """searchsorted 支撑匹配必须等于原 O(K²) 全对比较（含重复 student id 边界）。"""
    B, T, Kt, Ks, V = 3, 4, 6, 5, 40
    g = torch.Generator().manual_seed(0)
    teacher_ids = torch.stack([torch.stack(
        [torch.randperm(V, generator=g)[:Kt] for _ in range(T)]) for _ in range(B)])
    teacher_delta = torch.randn(B, T, Kt, generator=torch.Generator().manual_seed(1))
    student_ids = torch.stack([torch.stack(
        [torch.randperm(V, generator=g)[:Ks] for _ in range(T)]) for _ in range(B)])
    student_ids[..., 1] = student_ids[..., 0]          # 造重复 id 边界

    m = (student_ids.unsqueeze(-1) == teacher_ids.unsqueeze(-2)).to(teacher_delta.dtype)
    old = (m * teacher_delta.unsqueeze(-2)).sum(-1)
    sids_srt, order = teacher_ids.sort(-1)
    vals_srt = teacher_delta.gather(-1, order)
    pos = torch.searchsorted(sids_srt, student_ids.contiguous()).clamp(max=Kt - 1)
    found = sids_srt.gather(-1, pos) == student_ids
    new = vals_srt.gather(-1, pos) * found
    assert torch.equal(old, new)
