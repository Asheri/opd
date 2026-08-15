"""Stage 2 短 Rollout 报告生成（Q1-Q4，§9）。

消费 S2 训练结果（run_matrix(STAGE2_ROLLOUT_MATRIX) 的 summary）与
budget_eval 长预算评估结果（all_results，B∈{256..4096}），产出 markdown 报告：
表格 + 4 段 Q 解读。无数据时优雅降级（占位段落，如实标注待服务器实跑）。
纯函数，可 CPU 单测。
"""
from __future__ import annotations

import os


def _fmt(v, ndigits=4) -> str:
    """数值格式化；None/非数值 → '—'（无数据占位）。"""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return "—"


def _table_rows(train_results: list[dict]) -> list[list[str]]:
    """S2 训练 summary → 表格行（实验名 / reward / pg_loss / kl_loss / n_steps）。"""
    rows = []
    for r in train_results:
        s = r.get("summary", {})
        rows.append([
            s.get("experiment", r.get("name", "—")),
            _fmt(s.get("reward_mean")),
            _fmt(s.get("pg_loss_mean")),
            _fmt(s.get("kl_loss_mean")),
            str(s.get("n_steps", "—")),
        ])
    return rows


def _eval_table(eval_results: list[dict]) -> list[list[str]]:
    """budget_eval 长预算评估 → 表格行（实验名 / B / 指标均值）。"""
    rows = []
    for r in eval_results:
        # 兼容两种传入：dict(metrics) 或 ({name, metrics}) 包装
        name = r.get("name", "—") if isinstance(r, dict) else "—"
        m = r.get("metrics", r) if isinstance(r, dict) else r
        budget = m.get("budget", m.get("B", m.get("max_tokens", "—")))
        acc = m.get("accuracy", m.get("acc", m.get("pass@1")))
        rows.append([name, str(budget), _fmt(acc)])
    return rows


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    """简单 markdown 表格渲染。"""
    if not rows:
        return "（无数据）"
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


_Q_TEMPLATES = {
    "Q1": """**Q1 · 短 rollout 能否稳定产生有效 OPD learning signal？**

训练端（S2_E1/E2/E3，短预算 rollout）的 `reward_mean` 与 `pg_loss_mean` 相对
`S2_E0_static`（静态基线）的区分度，以及 `rollout/*` 状态分布（n_eos/n_budget/
n_loop/n_invalid）是验收代理。短 rollout 需在预算内产出非零 PG 梯度（有效
on-policy 注入），且 n_loop 占比有限（否则频繁 loop 拦截削弱信号）。

_{table_}""",
    "Q2": """**Q2 · 1024 训练预算能否提升长预算（4096）评估？**

对比 `S2_E2_opd1024`（训练短预算 1024）在 eval 长预算 B=4096 下的指标 vs
`S2_E0_static`（离线固定 D 基线）。若短 rollout 训练注入的 on-policy 信号确有
迁移价值，1024 训练应带来 4096 评估的提升，而非被曝光偏差拖累。

_{eval_table_}""",
    "Q3": """**Q3 · 训练预算的边际收益（512→1024→2048）如何？**

对比 `S2_E1_opd512 / S2_E2_opd1024 / S2_E3_opd2048` 三者：更多训练预算是否单调
提升信号质量，还是边际递减（1024 后趋平）？据此给出训练预算的性价比拐点。

_{table_}""",
    "Q4": """**Q4 · 训练短预算、评估长预算的迁移是否存在？**

训练 rollout 短预算（≤2048）下学到的能力在 eval 长预算（B=4096，接近论文
32K 的短生成片段）上的表现，验证「训练短、评估长」协议是否成立。若成立，
训练吞吐可大幅提升（短 rollout 生成耗时远低于长预算），是本协议的核心价值。

_{eval_table_}""",
}


def write_stage2_report(train_results: list[dict], eval_results: list[dict],
                        report_path: str) -> str:
    """Q1-Q4 报告。train_results: run_matrix(S2_E0-E3) 返回；eval_results:
    budget_eval 长预算评估的 all_results。无数据时占位（如实标注待服务器实跑）。
    返回报告 markdown 全文。
    """
    header = ["实验", "reward_mean", "pg_loss_mean", "kl_loss_mean", "n_steps"]
    table = _md_table(header, _table_rows(train_results))
    eval_header = ["实验", "预算B", "accuracy"]
    eval_table = _md_table(eval_header, _eval_table(eval_results))

    sections = []
    for q, tpl in _Q_TEMPLATES.items():
        t = tpl
        if "{table_}" in t:
            t = t.replace("{table_}", table if q in ("Q1", "Q3") else eval_table)
        if "{eval_table_}" in t:
            t = t.replace("{eval_table_}", eval_table)
        sections.append("## " + q + "\n\n" + t)

    md = (
        "# Stage 2 短 Rollout OPD 训练协议报告\n\n"
        "> Q1-Q4 发布于 2026-08-15。训练端为 toy/CPU 或服务器真实模型实跑；"
        "评估端协议见 `budget_eval`（B∈{256..4096}）。无数据字段以 '—' 占位，"
        "如实标注待服务器实跑。\n\n"
        "## 训练矩阵（S2_E0-E3）\n\n" + table + "\n\n"
        "## 长预算评估矩阵\n\n" + eval_table + "\n\n" +
        "\n\n".join(sections) + "\n"
    )
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return md


__all__ = ["write_stage2_report"]