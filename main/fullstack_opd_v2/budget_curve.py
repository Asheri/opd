"""Stage 1.7：Reasoning Budget Curve 与 Compute-Efficiency Metrics。

在 Stage 1.6 Budget-Aware 的 all_results（label/budget/accuracy/avg_reasoning_tokens）之上，
加效率维度：Budget Curve / AUC / nAUC / Efficiency / GainPerToken / ΔA(OPD Gain) / B(A*)。
零改动 evaluation 内核；指标全部纯函数、可单测。

核心伦理：效率一律用**真实 generated reasoning tokens**（EOS 位置或 budget cap，即
`avg_reasoning_tokens`），绝不拿 `max_new_tokens` 当 E[L]；曲线/效率刻画能力-成本权衡，
不以「CoT length ↑」为单独能力结论。报告按 dataset 分节（不同 benchmark accuracy 量纲不同，
曲线/AUC/效率必须在 benchmark 内比较，nAUC 才消除规模差）。
"""
from __future__ import annotations

import os

from .budget_eval import DEFAULT_BUDGETS

# 效率除数防零
_EPS = 1e-6


def _series_by_label(results, labels, budgets, key="accuracy"):
    """元素 → {label: {budget: value}}；缺失 budget 跳过。"""
    series = {lab: {} for lab in labels}
    for r in results:
        lab = r.get("label")
        if lab in series and r["budget"] in budgets:
            series[lab][r["budget"]] = r.get(key)
    return series


def _budget_widths(budgets) -> list[float]:
    """中点代表宽度：端点半宽、中间点两侧中点。Σ=budgets[-1]-budgets[0]。"""
    n = len(budgets)
    widths = [0.0] * n
    for j in range(n):
        if n == 1:
            widths[j] = 1.0
        elif j == 0:
            widths[j] = float(budgets[1] - budgets[0])
        elif j == n - 1:
            widths[j] = float(budgets[n - 1] - budgets[n - 2])
        else:
            widths[j] = (budgets[j + 1] - budgets[j - 1]) / 2
    return widths


def compute_auc(results, labels=None, budgets=None, key="accuracy") -> dict[str, float]:
    """AUC_M = Σ_j A_M(B_j)·ΔB_j（中点代表宽度）。"""
    labels = labels or sorted({r["label"] for r in results})
    budgets = budgets or list(DEFAULT_BUDGETS)
    widths = _budget_widths([int(b) for b in budgets])
    series = _series_by_label(results, labels, budgets, key)
    out = {}
    for lab in labels:
        out[lab] = sum(series[lab].get(B, 0.0) * w
                       for B, w in zip(budgets, widths))
    return out


def compute_nauc(results, labels=None, budgets=None, key="accuracy") -> dict[str, float]:
    """nAUC_M = AUC_M / AUC_max。

    AUC_max = 中点宽度总和 ΣΔB_j，即全对时的理论最大 AUC（nAUC∈[0,1]）。
    注意 B 不等间隔（256..4096 翻倍）时 ΣΔB_j（=4992）≠ budgets[-1]-budgets[0]（=3840）；
    AUC_max 必须与 compute_auc 的宽度口径一致，否则全对 nAUC≠1。
    """
    labels = labels or sorted({r["label"] for r in results})
    budgets = list(budgets or DEFAULT_BUDGETS)
    auc = compute_auc(results, labels, budgets, key)
    auc_max = sum(_budget_widths([int(b) for b in budgets]))
    return {lab: (auc[lab] / auc_max if auc_max else 0.0) for lab in labels}


def compute_efficiency(results, labels=None, budgets=None, eps=_EPS) -> dict[str, dict[int, float]]:
    """Eff_M(B) = Acc_M(B)/(E[L_M|B]+eps)。E[L]=avg_reasoning_tokens（真实，非 max_new_tokens）。"""
    labels = labels or sorted({r["label"] for r in results})
    budgets = list(budgets or DEFAULT_BUDGETS)
    out = {lab: {} for lab in labels}
    for r in results:
        if r.get("label") in labels and r["budget"] in budgets:
            el = r.get("avg_reasoning_tokens") or 0.0
            out[r["label"]][r["budget"]] = r["accuracy"] / (el + eps)
    return out


def compute_gain_per_token(results, labels=None, budgets=None, base="Base",
                           eps=_EPS) -> dict[str, dict[int, float]]:
    """GainPerToken_M(B) = (Acc_M(B)-Acc_Base(B))/(E[L_M|B]+eps)。Base 自身=0。"""
    labels = labels or sorted({l for l in {r["label"] for r in results} if l != base})
    budgets = list(budgets or DEFAULT_BUDGETS)
    base_series = {r["budget"]: r["accuracy"] for r in results if r.get("label") == base}
    out = {lab: {} for lab in labels}
    for r in results:
        if r.get("label") in labels and r["budget"] in budgets:
            el = r.get("avg_reasoning_tokens") or 0.0
            out[r["label"]][r["budget"]] = (r["accuracy"] - base_series.get(r["budget"], 0.0)) / (el + eps)
    return out


def compute_b_at_accuracy(results, labels=None, budgets=None, a_star=0.5,
                          key="accuracy") -> dict[str, float | None]:
    """B_M(A*)：达到目标 A* 的最小 reasoning budget（相邻格点线性插值）。全曲 <A* → None。"""
    labels = labels or sorted({r["label"] for r in results})
    budgets = sorted(list(budgets or DEFAULT_BUDGETS))
    series = _series_by_label(results, labels, budgets, key)
    out = {}
    for lab in labels:
        sv = series[lab]
        if not sv:
            out[lab] = None
            continue
        prev_b, prev_a = None, None
        b_at = None
        for B in budgets:
            a = sv.get(B)
            if a is None:
                continue
            if a >= a_star:
                if prev_a is not None and prev_a < a_star < a:
                    # 线性插值
                    b_at = prev_b + (a_star - prev_a) / (a - prev_a) * (B - prev_b)
                else:
                    b_at = float(B)
                break
            prev_b, prev_a = B, a
        out[lab] = b_at
    return out


def compute_delta_accuracy(results, labels=None, budgets=None, hi="L2", lo="L0",
                           eps=_EPS) -> dict[int, tuple[float, float]]:
    """ΔA(B) = A_hi(B)-A_lo(B) 与 ΔA(B)/E[L_hi(B)]，两者同时报告。"""
    budgets = list(budgets or DEFAULT_BUDGETS)
    out = {}
    by = {}
    for r in results:
        if r.get("label") in (hi, lo):
            by.setdefault(r["budget"], {})[r["label"]] = r
    for B in budgets:
        rh, rl = by.get(B, {}).get(hi), by.get(B, {}).get(lo)
        if rh is None or rl is None:
            continue
        dA = rh["accuracy"] - rl["accuracy"]
        el = rh.get("avg_reasoning_tokens") or 0.0
        out[B] = (dA, dA / (el + eps))
    return out


# --------------------------- matplotlib 5 图 ---------------------------

def _write_curve_plots(results, ds, out_dir) -> list[str]:
    """5 张图到 out_dir，返回生成的 PNG 名列表（以 {ds}_ 前缀隔离多 benchmark）。"""
    if not results:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    labels = sorted({r["label"] for r in results})
    budgets = sorted({int(r["budget"]) for r in results})
    series = {lab: {int(r["budget"]): r for r in results if r["label"] == lab}
              for lab in labels}
    written = []

    def _line(ax, getter, fname, ylabel, title, xlabel="Reasoning Budget B", skip_none=False):
        for lab in labels:
            xs, ys = [], []
            for B in budgets:
                r = series[lab].get(B)
                if r is None:
                    continue
                v = getter(r)
                if skip_none and v is None:
                    continue
                xs.append(B); ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", label=lab)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, fname), dpi=110, bbox_inches="tight")
        plt.clf(); written.append(fname)

    p = f"{ds}_"
    # Fig1 Accuracy vs Budget（复用 Stage 1.6 口径）
    _line(plt.gca(), lambda r: r["accuracy"], p + "accuracy_vs_budget.png",
          "Accuracy@B", "1. Accuracy vs Reasoning Budget")
    # Fig2 PrefixAccuracy vs Budget（仅无答案样本）
    _line(plt.gca(), lambda r: r.get("prefix_accuracy"), p + "prefix_accuracy_vs_budget.png",
          "PrefixAccuracy@B", "2. PrefixAccuracy vs Reasoning Budget", skip_none=True)
    # Fig3 Accuracy vs Actual Reasoning Tokens（x=E[L|B]，真实 tokens）
    _line(plt.gca(), lambda r: r["avg_reasoning_tokens"], p + "accuracy_vs_actual_tokens.png",
          "Accuracy@B", "3. Accuracy vs Actual Reasoning Tokens",
          xlabel="Average Reasoning Tokens")
    # Fig4 OPD Gain vs Budget：ΔA(B)=A_L2-A_L0 单线（跨 label；Base-only 无 L0/L2 → 跳过）
    delta = compute_delta_accuracy(results, labels, budgets)
    if delta:
        xs = sorted(int(b) for b in delta)
        ys = [delta[x][0] for x in xs]
        plt.plot(xs, ys, marker="o", label="ΔA = A_L2 - A_L0")
        plt.axhline(0, color="gray", lw=0.8, alpha=0.6)
        plt.xlabel("Reasoning Budget B"); plt.ylabel("ΔA = A_L2 - A_L0")
        plt.title("4. OPD Gain vs Reasoning Budget")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, p + "opd_gain_vs_budget.png"),
                    dpi=110, bbox_inches="tight")
        plt.clf(); written.append(p + "opd_gain_vs_budget.png")
    # Fig5 Accuracy/Token vs Budget（Efficiency=Acc/(E[L]+eps)）
    for lab in labels:
        xs, ys = [], []
        for B in budgets:
            r = series[lab].get(B)
            if r is None:
                continue
            el = r.get("avg_reasoning_tokens") or 0.0
            xs.append(B); ys.append(r["accuracy"] / (el + _EPS))
        if xs:
            plt.plot(xs, ys, marker="o", label=lab)
    plt.xlabel("Reasoning Budget B"); plt.ylabel("Accuracy / Reasoning Token")
    plt.title("5. Accuracy per Reasoning Token vs Budget")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, p + "efficiency_vs_budget.png"),
                dpi=110, bbox_inches="tight")
    plt.clf(); written.append(p + "efficiency_vs_budget.png")
    return written


# --------------------------- 报告 writer ---------------------------

def write_budget_curve_report(results, report_path) -> str:
    """写 Stage 1.7 决策报告（每 dataset 两表 + 5 图），返回 markdown 文本。

    按 results 的 dataset 字段分组：不同 benchmark 的 accuracy 量纲不同，曲线/AUC/效率
    必须在单 benchmark 内比较（nAUC 才消除规模差）。图名带 {dataset}_ 前缀避免覆盖。
    """
    base = os.path.dirname(report_path) or "."
    os.makedirs(base, exist_ok=True)
    datasets = sorted({r.get("dataset", "?") for r in results}) or ["?"]
    sections = []
    for ds in datasets:
        ds_res = [r for r in results if r.get("dataset", "?") == ds]
        sections.append(_render_dataset(ds_res, ds, base))
    lines = ["# Stage 1.7 Reasoning Budget Curve 与效率指标报告", "",
             "> 基于 Stage 1.6 Budget-Aware 的 all_results。效率用**真实 reasoning tokens**"
             "（EOS 位置或 budget cap，非 max_new_tokens）。曲线刻画能力-成本权衡，"
             "**不以 CoT length 为单独能力结论**。按 dataset 分节（benchmark 内比较）。", "",
             *sections]
    md = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return md


def _render_dataset(results, ds, base) -> str:
    """单个 dataset：两核心表 + Budget-Normalized OPD Gain + 5 图 + 诚实解读，返回 md 段。"""
    labels = sorted({r["label"] for r in results})
    budgets = sorted({int(r["budget"]) for r in results})
    p = f"{ds}_"
    plots = _write_curve_plots(results, ds, base)
    auc = compute_auc(results, labels, budgets)
    nauc = compute_nauc(results, labels, budgets)
    b50 = compute_b_at_accuracy(results, labels, budgets, a_star=0.5)
    gain = compute_gain_per_token(results, labels, budgets)
    delta = compute_delta_accuracy(results, labels, budgets)
    lines = [f"## Dataset: {ds}", ""]

    # 表1：Model | AUC | nAUC | Accuracy@512 | Accuracy@1024 | Accuracy@2048 | B@50%
    lines.append("| Model | AUC | nAUC | Accuracy@512 | Accuracy@1024 | Accuracy@2048 | B@50% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for lab in labels:
        a512 = _acc_at(results, lab, 512); a1024 = _acc_at(results, lab, 1024)
        a2048 = _acc_at(results, lab, 2048)
        b50s = "-" if b50.get(lab) is None else f"{b50[lab]:.0f}"
        lines.append(f"{lab}|{auc[lab]:.1f}|{nauc[lab]:.3f}|"
                     f"{_fmt(a512)}|{_fmt(a1024)}|{_fmt(a2048)}|{b50s}")
    lines.append("")

    # 表2：Model | Budget | Accuracy | Reasoning Tokens | Accuracy/Token | OPD Gain/Token
    lines.append("| Model | Budget | Accuracy | Reasoning Tokens | Accuracy/Token | OPD Gain/Token |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda r: (r["label"], r["budget"])):
        lab = r["label"]; B = r["budget"]
        acc = r["accuracy"]; el = r.get("avg_reasoning_tokens") or 0.0
        at = acc / (el + _EPS)
        gt = gain.get(lab, {}).get(B, None)
        gt_s = "-" if gt is None else f"{gt:.2e}"
        lines.append(f"{lab}|{B}|{acc:.3f}|{el:.0f}|{at:.2e}|{gt_s}")
    lines.append("")

    # Budget-Normalized OPD Gain（ΔA 与 ΔA/E[L_L2] 同时报告；等预算比较严格同一 B）
    if delta:
        lines.append("### Budget-Normalized OPD Gain（等预算比较：同一 B 下 ΔA=A_L2-A_L0）")
        lines.append("")
        lines.append("| Budget | ΔA = A_L2 - A_L0 | ΔA / E[L_L2] |")
        lines.append("|---|---:|---:|")
        for B in sorted(delta):
            dA, per_tok = delta[B]
            lines.append(f"|{B}|{dA:+.3f}|{per_tok:+.2e}|")
        lines.append("")

    # 图（带 dataset 前缀文件名）
    lines.append("### 图")
    lines.append("")
    captions = {
        "accuracy_vs_budget.png": "1. Accuracy vs Reasoning Budget",
        "prefix_accuracy_vs_budget.png": "2. PrefixAccuracy vs Reasoning Budget",
        "accuracy_vs_actual_tokens.png": "3. Accuracy vs Actual Reasoning Tokens",
        "opd_gain_vs_budget.png": "4. OPD Gain (ΔA) vs Reasoning Budget",
        "efficiency_vs_budget.png": "5. Accuracy per Reasoning Token vs Budget",
    }
    if plots:
        for suffix, cap in captions.items():
            if p + suffix in plots:
                lines.append(f"![{cap}]({p + suffix})  \n*{cap}*")
        lines.append("")
    else:
        lines.append("> matplotlib 未装，图跳过。")
        lines.append("")

    # 诚实解读
    lines.append("### 解读")
    lines.append("")
    lines.append("- 曲线/AUC/nAUC/Efficiency 为**全模型全 budget 可比**口径；表1 B@50%、表2 "
                 "OPD Gain/Token、ΔA 依赖 L0/L2。当前仅 Base（L0/L2 占位待 checkpoint）→ "
                 "这些行显示 `-`/占位，图 4 待 L0/L2 数据补画。")
    lines.append("- 等预算比较**必须是同一 B**（`A_L2(B)-A_L0(B)`），禁止不同模型不同 budget 作主结论。")
    lines.append("- 等性能比较 `B_M(A*)`：达到目标 A* 需更小 budget 的模型更高效。")
    if delta:
        b_lo = min(delta, key=lambda B: abs(delta[B][0]))
        dA, _ = delta[b_lo]
        lines.append(f"- ΔA 最接近 0 的 budget 为 {b_lo}（ΔA={dA:+.3f}）——L2/L0 交叉点参考。")
    return "\n".join(lines)


def _acc_at(results, label, budget):
    for r in results:
        if r.get("label") == label and r["budget"] == budget:
            return r["accuracy"]
    return None


def _fmt(v):
    return "-" if v is None else f"{v:.3f}"