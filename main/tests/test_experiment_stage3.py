"""Stage 3：Budget-Aware Selective Rollout 实验矩阵 + 比较函数（任务 7）。

- STAGE3_MATRIX：S3_E0 random 单预算对照 / S3_E1 selective 单预算 /
  S3_E2 selective + adaptive 预算，三实验统一经 pipeline 无条件 compute_rollout_metrics
  产 rollout/* 指标，可同口径对比 Performance / RolloutTokens / Eff。
- build_config 泛化：matrix 参数化到 STAGE3_MATRIX。
- run_matrix 跑通 S3_E0（selective 关、单预算）→ 断言 metrics 含 rollout/ 键。
- aggregate_stage3：从每实验 metrics 聚合 rollout/* 均值。
"""
import pytest

from fullstack_opd_v2.experiment import (
    STAGE3_MATRIX, build_config, run_matrix, aggregate_stage3)


def test_stage3_matrix_builds():
    """S3 三实验都能建出合法配置，且各预算开关符合矩阵定义（§八）。"""
    # S3_E0：random 单预算对照（selective 关、统一 1024，无分桶）
    c0 = build_config("S3_E0_random_fixed1024", n_steps=4, matrix=STAGE3_MATRIX)
    assert c0["l2"]["enabled"] is True
    assert c0["l2"]["selective_rollout"]["enabled"] is False
    assert c0["l2"]["refresh_ratio"]["mode"] == "adaptive"
    assert c0["l2"]["rollout"]["max_new_tokens"] == 1024
    assert c0["l2"]["selective_rollout"]["budget_mode"] == "fixed"
    assert c0["l2"]["rollout"]["token_budget_per_refresh"] is None
    # S3_E1：selective 单预算（价值选择 + fixed 单档，无全局 token 预算）
    c1 = build_config("S3_E1_selective_fixed1024", n_steps=4, matrix=STAGE3_MATRIX)
    assert c1["l2"]["selective_rollout"]["enabled"] is True
    assert c1["l2"]["selective_rollout"]["budget_mode"] == "fixed"
    assert c1["l2"]["rollout"]["max_new_tokens"] == 1024
    assert c1["l2"]["rollout"]["token_budget_per_refresh"] is None
    # S3_E2：selective + adaptive 预算（价值选择 + 分桶 + 全局 token 记账）
    c2 = build_config("S3_E2_selective_adaptive", n_steps=4, matrix=STAGE3_MATRIX)
    assert c2["l2"]["selective_rollout"]["enabled"] is True
    assert c2["l2"]["selective_rollout"]["budget_mode"] == "adaptive"
    assert c2["l2"]["selective_rollout"]["compute_aware"] is True
    assert c2["l2"]["rollout"]["token_budget_per_refresh"] == 4096
    # 未知实验名须报错
    with pytest.raises(KeyError):
        build_config("S3_E9_unknown", matrix=STAGE3_MATRIX)


def test_stage3_matrix_runs_and_metrics(tmp_path):
    """S3_E0（selective 关、单预算）经 pipeline 无条件 rollout 指标落进 metrics。

    ⚠️ 只跑 S3_E0（random 对照）：toy 下 S3_E2 的 adaptive budget 档位（256/512/1024/2048）
    远超 toy max_len，逐桶生成会位置编码越界——本测试只验证「selective 关也能产 rollout/*」
    的协议抽象，不跑 adaptive 分桶（toy 不追求分桶区分度）。
    """
    res = run_matrix(str(tmp_path), n_steps=4, device="cpu",
                     names=["S3_E0_random_fixed1024"], matrix=STAGE3_MATRIX)
    assert len(res) == 1
    r = res[0]
    assert r["name"] == "S3_E0_random_fixed1024"
    # 刷新相位那步会落 rollout/ 指标（pipeline 改造后无条件并入 metrics）
    rollout_steps = [m for m in r["metrics"] if isinstance(m, dict)
                     and m.get("rollout/accuracy_proxy") is not None]
    assert rollout_steps, "S3_E0 未产出 rollout/ 指标（refresh 相位未跑或未并入 metrics）"
    for key in ("rollout/accuracy_proxy", "rollout/rollout_tokens",
                "rollout/useful_per_token"):
        assert any(key in m for m in rollout_steps), f"metrics 缺 {key}"


def test_aggregate_stage3():
    """aggregate_stage3：聚合 rollout/* 均值为 Performance/RolloutTokens/Eff。"""
    results = [
        {"name": "S3_E0_random_fixed1024", "metrics": [
            {"step": 0, "rollout/accuracy_proxy": 0.5,
             "rollout/rollout_tokens": 100, "rollout/useful_per_token": 0.005},
            {"step": 8, "rollout/accuracy_proxy": 0.7,
             "rollout/rollout_tokens": 200, "rollout/useful_per_token": 0.0035},
            # 缺 rollout/ 键的普通训练步应被跳过
            {"step": 1, "reward": 1.0},
        ]},
        {"name": "S3_E1_selective_fixed1024", "metrics": [
            {"step": 0, "rollout/accuracy_proxy": 0.6,
             "rollout/rollout_tokens": 150, "rollout/useful_per_token": 0.004},
        ]},
    ]
    out = aggregate_stage3(results)
    assert set(out) == {"S3_E0_random_fixed1024", "S3_E1_selective_fixed1024"}
    # Performance = accuracy_proxy 均值（跳过无键 step）
    assert out["S3_E0_random_fixed1024"]["Performance"] == pytest.approx(0.6)
    assert out["S3_E0_random_fixed1024"]["RolloutTokens"] == pytest.approx(150.0)
    # Eff = useful_per_token 均值
    assert out["S3_E0_random_fixed1024"]["Eff"] == pytest.approx(0.00425)
    # 单条实验
    assert out["S3_E1_selective_fixed1024"]["Performance"] == pytest.approx(0.6)
    assert out["S3_E1_selective_fixed1024"]["Eff"] == pytest.approx(0.004)


def test_aggregate_stage3_empty_metrics():
    """空 metrics / 无 rollout 键 → Performance=0.0，不崩。"""
    out = aggregate_stage3([{"name": "X", "metrics": []},
                            {"name": "Y", "metrics": [{"step": 0, "reward": 1.0}]}])
    assert out["X"] == {"Performance": 0.0, "RolloutTokens": 0.0, "Eff": 0.0}
    assert out["Y"]["Performance"] == 0.0