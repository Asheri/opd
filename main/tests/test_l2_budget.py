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
    assign_budgets, enforce_budget, compute_rollout_metrics, group_by_budget,
    PromptStateStore, RefreshSelector)


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


# ---- Stage 3：RefreshSelector.select_with_budget + _value 补 reward ----


def _make_selector(n_prompts: int, updates: dict) -> PromptStateStore:
    """构造带历史的 PromptStateStore + 默认 RefreshSelector。

    updates: {prompt_id: (reward, disagreement, resp_len)} 逐 prompt 喂历史。
    """
    ps = PromptStateStore(n_prompts)
    for pid, (reward, disagreement, resp_len) in updates.items():
        ps.update(pid, reward, disagreement, resp_len, step=1)
    return ps


def test_select_with_budget_fixed():
    ps = _make_selector(20, {0: (1.0, 0.5, 100), 1: (0.2, 0.1, 50)})
    sel = RefreshSelector(ps)
    indices, budgets = sel.select_with_budget(n_selected=8, n_prompts=20)
    assert indices.shape == (8,)
    assert budgets.shape == (8,)
    assert budgets.dtype == torch.long
    assert (budgets == 1024).all().item()      # fixed 默认单预算 1024


def test_select_with_budget_adaptive_4buckets():
    # 喂足历史，reward/disagreement 差异明显 → 选中集内 V 有分位数区分
    updates = {i: (float(i) / 19, float(i % 5) / 5, 100 + i * 10) for i in range(20)}
    ps = _make_selector(20, updates)
    sel = RefreshSelector(ps)
    indices, budgets = sel.select_with_budget(
        n_selected=8, n_prompts=20, budget_mode="adaptive")
    assert indices.shape == (8,)
    assert budgets.shape == (8,)
    assert budgets.dtype == torch.long
    for b in budgets.tolist():
        assert b in (256, 512, 1024, 2048)      # 都来自 budget_set


def test_select_with_budget_cold_start():
    # times_seen 全 0 → select() 走 uniform，_value() 全 0（无历史）
    # → assign_budgets 全等 → 中档 1024 fallback
    ps = PromptStateStore(n_prompts=20)
    sel = RefreshSelector(ps)
    indices, budgets = sel.select_with_budget(
        n_selected=6, n_prompts=20, budget_mode="adaptive")
    assert indices.shape == (6,)
    assert budgets.shape == (6,)
    assert (budgets == 1024).all().item()       # 全等 v → 中档 1024


def test_value_includes_reward():
    # 构造 reward_ema 差异：高 reward prompt vs 低 reward（其余信号相同）。
    # 直接设 store 字段，保证 reward_var/disagreement/times_seen 完全一致（否则无 reward
    # 权重时两 prompt 也会因 uncertainty 差异而不同，无法隔离 reward 项）。
    ps = PromptStateStore(n_prompts=2)
    ps.times_seen[:] = 1
    ps.reward_var[:] = 0.0
    ps.disagreement_ema[:] = 0.0
    ps.reward_ema[0] = 0.9      # 高 reward
    ps.reward_ema[1] = 0.1      # 低 reward
    # 不含 reward 权重（旧 config）→ 两 prompt 值相同
    sel_no_r = RefreshSelector(ps)   # 默认无 reward 键
    v_no_r = sel_no_r._value()
    assert v_no_r[0].item() == pytest.approx(v_no_r[1].item(), abs=1e-6)
    # 含 reward 权重 → 高 reward prompt 值更高
    sel_r = RefreshSelector(ps, value_weights={
        "uncertainty": 0.4, "disagreement": 0.4, "novelty": 0.2, "reward": 0.5})
    v_r = sel_r._value()
    assert v_r[0].item() > v_r[1].item()