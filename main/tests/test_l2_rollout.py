"""Stage 2：短 Rollout OPD 训练协议单测。

- config：L2RolloutCfg 默认/覆盖/拒未知键 + workload 下渗。
- model：detect_loop / generate_with_status / build_length_mask（不动 generate_batch）。
- rollout_vllm：parse_vllm_outputs 纯函数（eos/budget_stop/loop）。
- adaptive_cache：run_refresh_phase 注入 generator + loop 跳过 + ring buffer status 往返。
- pipeline：消费 l2.rollout.max_new_tokens + fallback + status 指标。
- experiment：STAGE2_ROLLOUT_MATRIX + build_config/run_matrix 泛化。
- report_stage2：Q1-Q4 报告。
"""
import pytest

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.exceptions import ConfigError


# --------------------------- 任务1：config L2RolloutCfg ---------------------------
def test_l2_rollout_defaults():
    cfg = load_config(overrides=["l2.enabled=true"])
    rollout = cfg["l2"]["rollout"]
    assert rollout["max_new_tokens"] == 1024
    assert rollout["allow_budget_stop"] is True
    assert rollout["eos_token_id"] is None        # 默认不判 EOS
    assert rollout["loop_detection"] is True
    assert rollout["pad_id"] == 0


def test_l2_rollout_overrides():
    cfg = load_config(overrides=["l2.rollout.max_new_tokens=2048",
                                 "l2.rollout.eos_token_id=0"])
    assert cfg["l2"]["rollout"]["max_new_tokens"] == 2048
    assert cfg["l2"]["rollout"]["eos_token_id"] == 0


def test_l2_rollout_unknown_key_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["l2.rollout.unknown=1"])


def test_l2_rollout_disabled_default():
    # l2 默认全关时 rollout 子段仍存在（默认值 1024）
    cfg = load_config()
    assert cfg["l2"]["enabled"] is False
    assert cfg["l2"]["rollout"]["max_new_tokens"] == 1024