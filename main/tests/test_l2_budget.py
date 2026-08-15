"""Stage 3：Budget-Aware Selective Rollout 配置单测。

- L2RolloutCfg.token_budget_per_refresh：每轮刷新全局 rollout token 预算（None=无上限）。
- L2SelectiveRolloutCfg：budget_mode/fixed_budget/budget_set/budget_quantiles +
  token_aware/token_weight + value_weights 补 reward 项。
- 默认 budget_mode="fixed" 保持现有行为零回归。
"""
import pytest

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.exceptions import ConfigError


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