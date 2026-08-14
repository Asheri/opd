"""adaptive_cache.py 单元测试：L2 四能力（本文件覆盖任务 2.1 DisagreementComputer）。"""
import torch

from fullstack_opd_v2.adaptive_cache import DisagreementComputer


def test_disagreement_identical_zero():
    """identical teacher/student 时 disagreement≈0（§3 测试6）。"""
    T, mask = 3, torch.ones(2, 3)
    logp = torch.zeros(2, 3)
    d = DisagreementComputer()
    D = d.compute(teacher_rl_logp=logp, teacher_ref_logp=logp,
                  student_logp=logp, student_ref_logp=logp, mask=mask)
    assert torch.allclose(D["abs"], torch.zeros(2), atol=1e-6)


def test_disagreement_monotonic_with_gap():
    """teacher/student 差异放大时 disagreement 单调增加（§3 测试7）。"""
    d = DisagreementComputer()
    mask = torch.ones(1, 3)
    base = torch.zeros(1, 3)
    D1 = d.compute(base, base, base + 0.5, base, mask)["abs"]
    D2 = d.compute(base, base, base + 2.0, base, mask)["abs"]
    assert D2.item() > D1.item()


def test_disagreement_mask_excludes_padding():
    """padding 不计入（§3 测试2/4）。"""
    d = DisagreementComputer()
    logp = torch.zeros(2, 4)
    logp[:, 2:] = 5.0   # padding 位置有值但应被 mask 排除
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    D = d.compute(logp, logp, logp, logp, mask)  # 全同应 0，padding 被排除
    assert torch.allclose(D["abs"], torch.zeros(2), atol=1e-6)


def test_gather_chosen_logp():
    """gather_chosen_logp 从 (B,T,V) log-softmax 取 chosen-token logp（§3.4）。"""
    import torch.nn.functional as F
    d = DisagreementComputer()
    logits = torch.randn(2, 3, 8)
    log_dists = F.log_softmax(logits, dim=-1)   # 真实 log-softmax 分布
    responses = torch.tensor([[1, 0, 2], [7, 6, 5]])
    logp = d.gather_chosen_logp(log_dists, responses)
    expected = log_dists.gather(2, responses.unsqueeze(-1)).squeeze(-1)
    assert logp.shape == (2, 3)
    assert torch.allclose(logp, expected, atol=1e-6)


def test_health_score_thresholds():
    """rule-based HEALTHY/WARNING/CRITICAL 分类（§4.3）。"""
    from fullstack_opd_v2.adaptive_cache import CacheHealthMonitor
    hm = CacheHealthMonitor(health={"hit_rate": {"warning": 0.995, "critical": 0.98},
                                    "refresh_age_p95": {"warning": 5, "critical": 10}})
    assert hm.classify(hit_rate=0.999) == "HEALTHY"
    assert hm.classify(hit_rate=0.99) == "WARNING"
    assert hm.classify(hit_rate=0.97) == "CRITICAL"


def test_health_alert_cooldown():
    """同一 warning cooldown 内不重复（§4.4）。"""
    from fullstack_opd_v2.adaptive_cache import CacheHealthMonitor
    hm = CacheHealthMonitor(health={"hit_rate": {"warning": 0.995, "critical": 0.98}},
                            alert_cooldown=5)
    hm.record(step=1, hit_rate=0.99)   # WARNING
    assert hm.last_status == "WARNING"
    hm.record(step=2, hit_rate=0.99)   # 同 warning，cooldown 内不重复
    assert hm._alert_count == 1


def test_ratio_bounds():
    """α ∈ [min, max]，α_max<1（§5.4）。"""
    from fullstack_opd_v2.adaptive_cache import DynamicRatioController
    c = DynamicRatioController(initial=0.3, min=0.1, max=0.6, mode="adaptive")
    for _ in range(100):
        a = c.update(base_age=100, policy_drift=0, refresh_quality=100)
        assert 0.1 <= a <= 0.6


def test_ratio_fixed_mode():
    """fixed 模式 α 恒定（§5.8）。"""
    from fullstack_opd_v2.adaptive_cache import DynamicRatioController
    c = DynamicRatioController(initial=0.3, min=0.1, max=0.6, mode="fixed")
    assert c.update(100, 0, 100) == 0.3


def test_ratio_cold_start():
    """refresh 不足 fallback base（§5.5）。"""
    from fullstack_opd_v2.adaptive_cache import DynamicRatioController
    c = DynamicRatioController(initial=0.3, min=0.1, max=0.6, mode="adaptive")
    assert c.cold_start_adjust(alpha=0.3, n_refresh=2, n_batch=8) == 2 / 8


def test_ratio_max_step_change():
    """|α_t−α_{t-1}| ≤ max_step_change（§5.4）。"""
    from fullstack_opd_v2.adaptive_cache import DynamicRatioController
    c = DynamicRatioController(initial=0.3, min=0.1, max=0.6, mode="adaptive",
                               max_step_change=0.05)
    a1 = c.update(0, 0, 0)
    a2 = c.update(100, 0, 100)
    assert abs(a2 - a1) <= 0.05 + 1e-6