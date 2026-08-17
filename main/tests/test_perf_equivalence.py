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

from fullstack_opd_v2.losses import expected_reward, pg_loss
from fullstack_opd_v2.scheduler import AsyncBatchedScheduler


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


def test_expected_reward_p_dists_equals_internal_exp():
    """expected_reward 传 p_dists 与内部 dists.exp() 必须逐位相等。"""
    dists = _logp(3, 5, 32, seed=0)
    delta = torch.randn(3, 5, 32, generator=torch.Generator().manual_seed(2))
    a = expected_reward(dists, delta)
    b = expected_reward(dists, delta, p_dists=dists.exp())
    assert torch.equal(a, b)


def test_pg_loss_log_ratio_max_suppresses_support_mismatch():
    """失配屏蔽：s_old 是 log0 近似（-30）且 delta≠0 时，贡献必须=0（M1 伪梯度修复）。

    回归：之前 clamp 语义下，负 delta 支撑失配产生符号相关伪梯度（0.184，应为 0）。
    """
    B, T, V = 1, 1, 64
    s_old = torch.full((B, T, V), -30.0)          # 支撑外填充 _LOG_ZERO
    s_old[..., :5] = _logp(B, T, 5, seed=1)
    s_cur = torch.full((B, T, V), -30.0)
    s_cur[..., :5] = _logp(B, T, 5, seed=0)
    s_cur[..., 5] = -1.0                          # student 在支撑外高概率
    for sign in (-0.5, 0.5):
        delta = torch.zeros(B, T, V)
        delta[..., 5] = sign
        out = pg_loss(s_cur, s_old, delta, torch.ones(B, T), log_ratio_max=20.0)
        assert out.item() == 0.0, f"sign={sign} 失配屏蔽应=0，实际 {out.item()}"
    # 不屏蔽时确实有伪梯度（证明 bug 曾在）
    no_mask = pg_loss(s_cur, s_old, delta, torch.ones(B, T))
    assert no_mask.item() != 0.0


def test_pg_loss_mask_ones_equals_none():
    """全 1 mask 分支必须逐位等于 mask=None 快路径（P2-2 去冗余的前提）。"""
    s_cur = _logp(3, 5, 32, seed=0)
    s_old = _logp(3, 5, 32, seed=1)
    delta = torch.randn(3, 5, 32, generator=torch.Generator().manual_seed(2))
    ones = torch.ones(3, 5)
    a = pg_loss(s_cur, s_old, delta, mask=ones)
    b = pg_loss(s_cur, s_old, delta, mask=None)
    assert torch.equal(a, b)


def test_low_var_kl_mask_ones_equals_none():
    """低方差 KL 的全 1 mask 分支必须逐位等于 mask=None 快路径。"""
    from fullstack_opd_v2.losses import low_var_kl
    s = _logp(3, 5, 32, seed=0)
    ref = _logp(3, 5, 32, seed=1)
    ones = torch.ones(3, 5)
    a = low_var_kl(s, ref, mask=ones)
    b = low_var_kl(s, ref, mask=None)
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
    #    ⚠️ 实现按 logprob 降序排序（152142e：vLLM dict 迭代顺序不保证有序，必须显式
    #    排序否则 K 截断/searchsorted 支撑错乱）——测试按同一契约排序后比对。
    for b, plp in ((0, plp0), (1, plp1)):
        for t in range(T):
            d = plp[P + t]
            items = list(d.items())
            items.sort(key=lambda kv: kv[1].logprob, reverse=True)
            for j, (tid, lp) in enumerate(items):
                assert ids[b, t, j].item() == tid, (b, t, j)
                # float32 落盘有舍入误差，比较用容差
                assert abs(logps[b, t, j].item() - lp.logprob) < 1e-6, (b, t, j)


def test_response_dists_topk_padding_is_log_zero():
    """M3 回归：vLLM 只返回部分 logprob（异常/部分返回）时，未填充槽位 logp 必须是
    _LOG_ZERO（≈log 0）而非 0.0——0.0 是合法高概率，会把 padding 槽位当成「token id=0
    处 logp=0.0」污染稀疏支撑匹配。"""
    import unittest.mock as mock
    from fullstack_opd_v2.rollout_vllm import VLLMRolloutEngine, _LOG_ZERO

    B, P, T, V = 1, 2, 2, 6
    k = 6                                        # V<=full_cap → k=V，但只返回 3 项
    device = "cpu"

    class _LP:
        def __init__(self, logprob):
            self.logprob = logprob

    def make_plp():
        plp = [None] * P
        for t in range(T):
            plp.append({100 + t * 10 + i: _LP(float(t) + i * 0.1) for i in range(3)})
        return plp

    fake_llm = mock.MagicMock()
    fake_llm.generate.return_value = [mock.Mock(prompt_logprobs=make_plp())]

    eng = object.__new__(VLLMRolloutEngine)
    eng.vocab_size = V
    eng.full_cap = 100
    eng.device = device
    eng.llm = fake_llm

    prompts = torch.zeros(B, P, dtype=torch.long)
    responses = torch.zeros(B, T, dtype=torch.long)
    with mock.patch("fullstack_opd_v2.rollout_vllm.SamplingParams",
                    new=lambda **kw: mock.sentinel.sampling):
        ids, lps = eng.response_dists_topk(prompts, responses)

    assert ids.shape == (B, T, k) and lps.shape == (B, T, k)
    # 每位置只填 3 项，剩余 k-3 槽位 logp 必须 = _LOG_ZERO（非 0.0）
    assert (lps[..., 3:] == _LOG_ZERO).all(), "padding logp 应为 _LOG_ZERO"
    assert (lps[..., :3] != _LOG_ZERO).all(), "真实项的 logp 不应被填充覆盖"
    # 实现按 logprob 降序排序（152142e）：真实项顺序 = i=2,1,0（t+0.2,t+0.1,t+0.0）
    for t in range(T):
        for j in range(3):
            assert abs(lps[0, t, j].item() - (float(t) + (2 - j) * 0.1)) < 1e-6


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


def test_ref_logp_at_student_topk_searchsorted_equals_full_compare():
    """真实方法 `AsyncBatchedScheduler._ref_logp_at_student_topk` 的 searchsorted 二分必须
    等于原 O(Ks×Kr) 全对比较参考（含重复 student id 边界）。

    该方法是 GPU 部署防 OOM 的稀疏 KL 锚点路径，但 test_scheduler 全跑 dense 模式
    （top_k_student=0），从未触发。这里用 `object.__new__` 绕过完整 scheduler 构造
    （避免模型前向/缓存 build），只设它依赖的三个稀疏字段，直接调用真实方法。
    """
    N, B, T, Kr, Ks, V = 5, 3, 4, 6, 5, 40
    g = torch.Generator().manual_seed(0)
    # 每位置 top-K token id 唯一（randperm），对应真实 topk 输出
    ref_ids = torch.stack([torch.stack(
        [torch.randperm(V, generator=g)[:Kr] for _ in range(T)]) for _ in range(N)])
    ref_logp = torch.randn(N, T, Kr, generator=torch.Generator().manual_seed(1))
    student_ids = torch.stack([torch.stack(
        [torch.randperm(V, generator=g)[:Ks] for _ in range(T)]) for _ in range(B)])
    student_ids[..., 1] = student_ids[..., 0]          # 重复 student id 边界
    idxs = torch.tensor([2, 0, 4])                     # (B,) 批次索引
    ref_tail_logp = -1e2

    # 参考实现（旧逻辑）：O(Ks×Kr) 全对比较，用未排序的原始 ref 张量
    rids = ref_ids[idxs]                               # (B,T,Kr)
    rlogp = ref_logp[idxs]
    match = (student_ids.unsqueeze(-1) == rids.unsqueeze(-2)).to(ref_logp.dtype)
    gathered = (match * rlogp.unsqueeze(-2)).sum(-1)   # (B,T,Ks)
    has = match.max(-1).values
    old = gathered.where(has > 0.0,
                         torch.full_like(gathered, ref_tail_logp))

    # 二分方法：只设方法依赖的三个稀疏字段（__init__ 会预排序，这里手动等价处之）
    sched = object.__new__(AsyncBatchedScheduler)
    sched.ref_ids_sorted, order = ref_ids.sort(dim=-1)
    sched.ref_logp_sorted = ref_logp.gather(-1, order)
    sched.ref_tail_logp = ref_tail_logp

    out = sched._ref_logp_at_student_topk(idxs, student_ids)
    assert out.shape == old.shape
    assert torch.equal(out, old), "searchsorted 二分必须逐位等于 O(K²) 全对比较"
