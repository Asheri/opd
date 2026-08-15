"""Stage 3：Budget-Aware Selective Rollout 配置单测。

- L2RolloutCfg.token_budget_per_refresh：每轮刷新全局 rollout token 预算（None=无上限）。
- L2SelectiveRolloutCfg：budget_mode/fixed_budget/budget_set/budget_quantiles +
  token_aware/token_weight + value_weights 补 reward 项。
- 默认 budget_mode="fixed" 保持现有行为零回归。
"""
import pytest
import torch

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.exceptions import ConfigError
from fullstack_opd_v2.adaptive_cache import (
    assign_budgets, enforce_budget, compute_rollout_metrics, group_by_budget)


def test_l2_budget_defaults():
    cfg = load_config(overrides=["l2.enabled=true"])
    assert cfg["l2"]["rollout"]["token_budget_per_refresh"] is None
    sr = cfg["l2"]["selective_rollout"]
    assert sr["budget_mode"] == "fixed"
    assert sr["fixed_budget"] == 1024
    assert sr["budget_set"] == (256, 512, 1024, 2048)
    assert sr["budget_quantiles"] == (0.25, 0.5, 0.75)
    assert sr["token_aware"] is False
    assert "reward" in sr["value_weights"]


def test_l2_budget_overrides():
    cfg = load_config(overrides=[
        "l2.selective_rollout.budget_mode=adaptive",
        "l2.selective_rollout.fixed_budget=512",
        "l2.rollout.token_budget_per_refresh=4096",
        "l2.selective_rollout.token_aware=true"])
    sr = cfg["l2"]["selective_rollout"]
    assert sr["budget_mode"] == "adaptive"
    assert sr["fixed_budget"] == 512
    assert cfg["l2"]["rollout"]["token_budget_per_refresh"] == 4096
    assert sr["token_aware"] is True


def test_l2_budget_unknown_key_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["l2.selective_rollout.unknown=1"])


# ---- Stage 3 纯函数：assign_budgets ----


def test_assign_budgets_quantiles():
    v = torch.tensor([0.1, 0.3, 0.6, 0.9])
    budgets = assign_budgets(v)
    assert budgets.dtype == torch.long
    b = budgets.tolist()
    assert b[0] < b[-1]          # 低价值→低档，高价值→高档
    assert b[0] <= b[1] <= b[2] <= b[3]   # 单调
    for x in b:
        assert x in (256, 512, 1024, 2048)


def test_assign_budgets_all_equal():
    v = torch.full((4,), 0.5)
    budgets = assign_budgets(v)
    assert budgets.tolist() == [1024, 1024, 1024, 1024]  # 中档 budget_set[2]


def test_assign_budgets_4_buckets():
    v = torch.linspace(0.1, 0.9, 8)
    budgets = assign_budgets(v)
    assert len(set(budgets.tolist())) == 4   # 4 档都被覆盖


# ---- Stage 3 纯函数：enforce_budget ----


def test_enforce_budget_within():
    indices = torch.tensor([0, 1, 2, 3])
    budgets = torch.tensor([256, 512, 256, 256])
    v = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out_i, out_b = enforce_budget(indices, budgets, v, budget_t=2000)
    assert out_i.tolist() == indices.tolist()
    assert out_b.tolist() == budgets.tolist()   # 未超预算原样返回


def test_enforce_budget_downgrade():
    indices = torch.tensor([0, 1, 2, 3])
    budgets = torch.tensor([2048, 2048, 2048, 2048])
    v = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out_i, out_b = enforce_budget(indices, budgets, v, budget_t=3000)
    assert out_b.sum().item() <= 3000
    for x in out_b.tolist():
        assert x in (256, 512, 1024, 2048)   # 预算来自 budget_set


def test_enforce_budget_none():
    indices = torch.tensor([0, 1])
    budgets = torch.tensor([2048, 2048])
    v = torch.tensor([0.1, 0.2])
    out_i, out_b = enforce_budget(indices, budgets, v, budget_t=None)
    assert out_i.tolist() == indices.tolist()
    assert out_b.tolist() == budgets.tolist()


# ---- Stage 3 纯函数：compute_rollout_metrics ----


def test_compute_rollout_metrics():
    summary = dict(n_total=100, n_appended=40, n_eos=30, n_budget=20,
                   n_loop=10, n_invalid=5, rollout_tokens=200)
    m = compute_rollout_metrics(summary, budget_t=200)
    assert m['rollout/rollout_tokens'] == 200
    assert m['rollout/budget_utilization'] == pytest.approx(1.0)
    assert m['rollout/truncation_rate'] == pytest.approx(0.2)
    assert m['rollout/loop_rate'] == pytest.approx(0.1)
    assert m['rollout/eos_rate'] == pytest.approx(0.3)
    assert m['rollout/accuracy_proxy'] == pytest.approx(0.4)
    assert m['rollout/useful_per_token'] == pytest.approx(0.2)  # 40/200


def test_compute_rollout_metrics_divzero():
    summary = dict(n_total=0, n_appended=0, n_eos=0, n_budget=0,
                   n_loop=0, n_invalid=0, rollout_tokens=0)
    m = compute_rollout_metrics(summary, budget_t=200)  # rollout_tokens=0 → utilization 0.0
    for val in m.values():
        assert val == 0.0


# ---- Stage 3 纯函数：group_by_budget ----


def test_group_by_budget():
    cand = torch.tensor([0, 1, 2, 3])
    budgets = torch.tensor([256, 1024, 256, 512])
    buckets = group_by_budget(cand, budgets)
    assert buckets == {256: [0, 2], 1024: [1], 512: [3]}