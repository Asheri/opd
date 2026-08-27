"""AdaptiveKLController（Direct-OPD 论文 §2.4 Eq.16）纯函数单测。

α_{m+1} = clip(α_m·(1 + ε·sgn(r̄_m)), α_min, α_max), sgn(0)=0。
不依赖 torch/vLLM/GPU。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fullstack_opd_v2.losses import AdaptiveKLController  # noqa: E402


def test_positive_delta_increases_alpha():
    """r̄>0 → α 按 ε 比例放大（教师 RL 平均提升保留 token → 增大 KL 抑制过放大）。"""
    c = AdaptiveKLController(alpha0=1.0, eps=0.01)
    a = c.step(0.5)
    assert abs(a - 1.01) < 1e-9


def test_negative_delta_decreases_alpha():
    """r̄<0 → α 按 ε 比例缩小（削弱锚点让梯度移离被抑制 token）。"""
    c = AdaptiveKLController(alpha0=1.0, eps=0.01)
    a = c.step(-0.5)
    assert abs(a - 0.99) < 1e-9


def test_zero_delta_no_change():
    """sgn(0)=0 → α 不变。"""
    c = AdaptiveKLController(alpha0=1.0, eps=0.01)
    assert c.step(0.0) == 1.0
    assert c.step(-0.0) == 1.0


def test_clip_upper_bound():
    """α 放大不越 α_max。"""
    c = AdaptiveKLController(alpha0=2.4, eps=0.01, alpha_min=0.5, alpha_max=2.5)
    a = c.step(1.0)          # 2.4*1.01=2.424 < 2.5
    a = c.step(1.0)          # 2.448 < 2.5
    a = c.step(1.0)          # 2.472
    a = c.step(1.0)          # 2.497
    a = c.step(1.0)          # 2.522 → clip 2.5
    assert a == 2.5
    assert c.step(1.0) == 2.5   # 持续钳制


def test_clip_lower_bound():
    """α 缩小不越 α_min。"""
    c = AdaptiveKLController(alpha0=0.51, eps=0.01, alpha_min=0.5, alpha_max=2.5)
    a = c.step(-1.0)         # 0.505
    a = c.step(-1.0)         # 0.500 → 0.49995 → clip 0.5
    assert a == 0.5
    assert c.step(-1.0) == 0.5


def test_alternating_signs():
    """正负交替：α 在 1.0 附近小幅振荡，不单调发散。"""
    c = AdaptiveKLController(alpha0=1.0, eps=0.01)
    for _ in range(50):
        c.step(1.0)
        c.step(-1.0)
    assert 0.99 <= c.alpha <= 1.01


def test_defaults_match_paper_range():
    """默认参数在论文区间内（α0=1.0、ε=0.01、[0.5,2.5]）。"""
    c = AdaptiveKLController()
    assert c.alpha == 1.0
    assert c.eps == 0.01
    assert c.alpha_min == 0.5
    assert c.alpha_max == 2.5


def test_custom_alpha0_from_kl_config():
    """α0 可由 kl_reg_coef 传入（scheduler 用 kl_reg_coef 作初始值）。"""
    c = AdaptiveKLController(alpha0=0.02)   # 旧实验值也合法（虽然不推荐）
    assert c.alpha == 0.02
