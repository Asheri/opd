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