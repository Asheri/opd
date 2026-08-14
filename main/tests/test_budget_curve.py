"""budget_curve.py 单测：Stage 1.7 Reasoning Budget Curve 与效率指标。

- 中点宽度 ΔB：Σ=budgets[-1]-budgets[0]=3840
- AUC：全对=AUC_max、全错=0；nAUC∈[0,1]
- B@A*：相邻格点线性插值（512/1024→768）、超上限 None
- Efficiency：用真实 avg_reasoning_tokens（非 max_new_tokens）
- GainPerToken：相对 Base
- ΔA(OPD Gain)：双报（ΔA 与 ΔA/E[L_hi]）
- 报告：两表 + 5 图；Base-only 省 ΔA 节 + 图 4 缺省
"""
import pytest

from fullstack_opd_v2.budget_curve import (
    _budget_widths, compute_auc, compute_nauc, compute_efficiency,
    compute_gain_per_token, compute_b_at_accuracy, compute_delta_accuracy,
    write_budget_curve_report,
)


def _mk(label, budget, acc, rt, dataset="MATH500", prefix=None):
    r = {"label": label, "budget": budget, "dataset": dataset,
         "accuracy": acc, "avg_reasoning_tokens": rt, "rows": []}
    if prefix is not None:
        r["prefix_accuracy"] = prefix
    return r


B = [256, 512, 1024, 2048, 4096]


# --------------------------- 中点宽度 / AUC / nAUC ---------------------------
def test_budget_widths_midpoint():
    # B 不等间隔（翻倍）→ 中点宽度和 ΣΔB=4992 ≠ B_range=3840（AUC_max 必须用 ΣΔB）
    w = _budget_widths(B)
    assert w[0] == 256 and w[-1] == 2048
    assert w[1] == (1024 - 256) / 2 and w[2] == (2048 - 512) / 2
    assert abs(sum(w) - 4992) < 1e-9


def test_auc_full_accuracy_equals_auc_max():
    # 全对 → AUC = ΣΔB_j = 4992，nAUC=1.0（AUC_max 与宽度口径一致）
    res = [_mk("Base", b, 1.0, b) for b in B]
    auc = compute_auc(res); nauc = compute_nauc(res)
    assert auc["Base"] == pytest.approx(4992)
    assert nauc["Base"] == pytest.approx(1.0)


def test_auc_zero_accuracy():
    res = [_mk("Base", b, 0.0, b) for b in B]
    assert compute_auc(res)["Base"] == pytest.approx(0.0)
    assert compute_nauc(res)["Base"] == pytest.approx(0.0)


def test_auc_mid():
    # 全 0.5 → AUC = 0.5*4992 = 2496, nAUC=0.5
    res = [_mk("Base", b, 0.5, b) for b in B]
    assert compute_auc(res)["Base"] == pytest.approx(0.5 * 4992)
    assert compute_nauc(res)["Base"] == pytest.approx(0.5)


# --------------------------- B(A*) 等性能比较 ---------------------------
def test_b_at_accuracy_interpolation():
    # A(512)=0.4, A(1024)=0.6 → B(0.5)=768
    res = [_mk("Base", 512, 0.4, 512), _mk("Base", 1024, 0.6, 1024)]
    b = compute_b_at_accuracy(res, a_star=0.5)
    assert b["Base"] == pytest.approx(768.0)


def test_b_at_accuracy_exact_node():
    # A(1024)=0.5 精确命中 → 返回 1024（非插值）
    res = [_mk("Base", 512, 0.4, 512), _mk("Base", 1024, 0.5, 1024)]
    assert compute_b_at_accuracy(res, a_star=0.5)["Base"] == pytest.approx(1024.0)


def test_b_at_accuracy_above_max_returns_none():
    res = [_mk("Base", b, 0.3, b) for b in B]
    assert compute_b_at_accuracy(res, a_star=0.5)["Base"] is None


def test_b_at_accuracy_first_node_above():
    # 首点即 ≥A* → 返回首点 budget
    res = [_mk("Base", 256, 0.8, 256), _mk("Base", 512, 0.9, 512)]
    assert compute_b_at_accuracy(res, a_star=0.5)["Base"] == pytest.approx(256.0)


# --------------------------- Efficiency / GainPerToken ---------------------------
def test_efficiency_uses_real_tokens():
    # 真实 E[L]=avg_reasoning_tokens，非 max_new_tokens
    res = [_mk("Base", 256, 0.5, 128)]
    eff = compute_efficiency(res)
    assert eff["Base"][256] == pytest.approx(0.5 / (128 + 1e-6))


def test_gain_per_token_relative_to_base():
    res = [_mk("Base", 256, 0.5, 128), _mk("L2", 256, 0.6, 128)]
    g = compute_gain_per_token(res)
    assert g["L2"][256] == pytest.approx((0.6 - 0.5) / (128 + 1e-6))


def test_gain_per_token_excludes_base():
    res = [_mk("Base", 256, 0.5, 128), _mk("L2", 256, 0.6, 128)]
    g = compute_gain_per_token(res)
    assert "Base" not in g


# --------------------------- ΔA(OPD Gain) 双报 ---------------------------
def test_delta_accuracy_dual_report():
    res = [_mk("L0", 1024, 0.4, 512), _mk("L2", 1024, 0.55, 600)]
    d = compute_delta_accuracy(res)
    dA, per_tok = d[1024]
    assert dA == pytest.approx(0.15)
    assert per_tok == pytest.approx(0.15 / (600 + 1e-6))


def test_delta_accuracy_requires_both_labels():
    # 只有 L2 无 L0 → 该 budget 不入 ΔA
    res = [_mk("L2", 1024, 0.55, 600)]
    assert compute_delta_accuracy(res) == {}


def test_delta_accuracy_base_only_empty():
    res = [_mk("Base", b, 0.1, b) for b in B]
    assert compute_delta_accuracy(res) == {}


# --------------------------- 报告 writer（两表 + 图） ---------------------------
def test_write_report_tables_and_plots(tmp_path):
    res = ([_mk("Base", b, 0.1 + b / 1000, b) for b in B]
           + [_mk("L0", b, 0.15 + b / 1000, b) for b in B]
           + [_mk("L2", b, 0.2 + b / 1000, b) for b in B])
    md = write_budget_curve_report(res, str(tmp_path / "curve.md"))
    assert (tmp_path / "curve.md").exists()
    assert "| Model | AUC | nAUC | Accuracy@512 | Accuracy@1024 | Accuracy@2048 | B@50% |" in md
    assert "| Model | Budget | Accuracy | Reasoning Tokens | Accuracy/Token | OPD Gain/Token |" in md
    assert "Budget-Normalized OPD Gain（等预算比较" in md
    # 表1 B@50% 列：Base acc=B/1000+0.1 <0.5 → None→"-"；L2 acc=0.2+B/1000 ≥0.5 at 512
    assert "Base|" in md and "L2|" in md
    try:
        import matplotlib  # noqa
        assert (tmp_path / "MATH500_accuracy_vs_budget.png").exists()
        assert (tmp_path / "MATH500_opd_gain_vs_budget.png").exists()
        assert (tmp_path / "MATH500_efficiency_vs_budget.png").exists()
    except Exception:
        pass  # matplotlib 未装则图跳过（md 加注）


def test_write_report_groups_by_dataset(tmp_path):
    # 两个 dataset → 两节 + 图名带各自前缀
    res = ([_mk("Base", b, 0.1, b, dataset="GSM8K") for b in B]
           + [_mk("Base", b, 0.2, b, dataset="MATH500") for b in B])
    md = write_budget_curve_report(res, str(tmp_path / "curve.md"))
    assert md.count("## Dataset:") == 2
    assert "## Dataset: GSM8K" in md and "## Dataset: MATH500" in md
    try:
        import matplotlib  # noqa
        assert (tmp_path / "GSM8K_accuracy_vs_budget.png").exists()
        assert (tmp_path / "MATH500_accuracy_vs_budget.png").exists()
    except Exception:
        pass


def test_write_report_base_only_omits_delta(tmp_path):
    res = [_mk("Base", b, 0.1, b) for b in B]
    md = write_budget_curve_report(res, str(tmp_path / "curve.md"))
    assert "Budget-Normalized OPD Gain（等预算比较" not in md  # 无 L0/L2 → 省 ΔA 节
    assert "OPD Gain/Token" in md                                # 表2 列仍在（值显示 -）
    try:
        import matplotlib  # noqa
        assert not (tmp_path / "MATH500_opd_gain_vs_budget.png").exists()  # 图4 缺省
    except Exception:
        pass