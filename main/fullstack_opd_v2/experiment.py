"""L2 E0-E6 实验矩阵 + 统一记录 + 实验图（§10，任务 6.2）。

E0-E6 逐一打开/关闭 L2 各模块，量化「adaptive teacher cache」各能力对训练质量
（reward/PG/KL）与效率（teacher forward 次数、refresh 轮数）的贡献。每个实验跑的
是同一条 pipeline（toy/CPU 可跑），仅 `l2.*` 配置不同——即对 §7 每模块 enabled
开关做单项 ablation。

实验记录统一分四类落盘（metrics 字段前缀）：
  - Training Quality：reward / pg_loss / kl_loss / adv_mean
  - Efficiency        ：teacher_forward_n / refresh_rounds / timings
  - Cache             ：refresh_pool_size / mean_disagreement
  - Selector          ：candidate_pool / value_frac / coverage_frac

图（§10，matplotlib 可选）：8 张，其中「teacher compute vs 训练质量」最重要——
验证 L2 用更少 teacher 前向换近似/更优训练信号。
"""
from __future__ import annotations

import json
import os

import torch

from .config import load_config

# 每模块 enabled 开关作实验变量（E0-E6 矩阵定义，§10 — 累积构建语义）。
# ⚠️ G6 修正：早期版本是「全开然后逐个关」，与 §10 的「累加」矛盾，且因 feeder 未接、
# E1-E6 训练信号完全相同。现改为 §10 累加：E1 只加 fixed refresh，逐步叠加
# disagreement→health monitor→dynamic ratio→selective rollout，E6 用 random rollout
# 对照验证 selective 本身贡献。双池 feeder 已接（G1），E1-E6 训练信号真实可区分。
#
# 注：disagreement.enabled 目前仅作配置开关（run_refresh_phase 恒算 D），
# E1↔E2 的差异由「D 是否参与后续信号」体现——E2 起 D 喂给 PromptState/selector，
# 是 Deterministic 的模块职责边界（见 docs/superpowers/specs/...design.md §10）。
# ─────────────────────────────────────────────────────────────────────────────
# Stage 2：短 rollout 预算消融矩阵（§8，任务 6）。独立 S2_E* 命名，不覆盖 E0-E6。
# E0 静态基线（L2 关，同 E0_base_only 语义但独立 key）；E1/E2/E3 = OPD + 短 rollout
# 512/1024/2048。真实 512/1024/2048 是 GPU 上真实模型的事；toy/CPU 实验经
# build_config 尾端覆盖把 max_new_tokens 压到小值验证协议抽象（不跑真实长预算）。
STAGE2_ROLLOUT_MATRIX: dict[str, dict] = {
    # S2_E0 静态基线：L2 完全关闭（独立命名，同 E0_base_only）
    "S2_E0_static": {
        "l2.enabled": "false",
    },
    # S2_E1 OPD + 短 rollout 512
    "S2_E1_opd512": {
        "l2.enabled": "true",
        "l2.refresh_ratio.mode": "fixed",
        "l2.disagreement.enabled": "false",
        "l2.health_monitor.enabled": "false",
        "l2.selective_rollout.enabled": "false",
        "l2.rollout.max_new_tokens": "512",
    },
    # S2_E2 OPD + 短 rollout 1024（主实验）
    "S2_E2_opd1024": {
        "l2.enabled": "true",
        "l2.refresh_ratio.mode": "fixed",
        "l2.disagreement.enabled": "false",
        "l2.health_monitor.enabled": "false",
        "l2.selective_rollout.enabled": "false",
        "l2.rollout.max_new_tokens": "1024",
    },
    # S2_E3 OPD + 短 rollout 2048
    "S2_E3_opd2048": {
        "l2.enabled": "true",
        "l2.refresh_ratio.mode": "fixed",
        "l2.disagreement.enabled": "false",
        "l2.health_monitor.enabled": "false",
        "l2.selective_rollout.enabled": "false",
        "l2.rollout.max_new_tokens": "2048",
    },
}


EXPERIMENT_MATRIX: dict[str, dict] = {
    "E0_base_only": {
        "l2.enabled": "false",
    },
    # E1 Base + fixed Refresh：L2 开，固定 α=initial，无 disagreement/selective/health
    "E1_base_fixed_refresh": {
        "l2.enabled": "true",
        "l2.refresh_ratio.mode": "fixed",
        "l2.disagreement.enabled": "false",
        "l2.health_monitor.enabled": "false",
        "l2.selective_rollout.enabled": "false",
    },
    # E2 E1 + Disagreement：D 参与刷新质量信号（selector 历史 / ratio quality）
    "E2_add_disagreement": {
        "l2.enabled": "true",
        "l2.refresh_ratio.mode": "fixed",
        "l2.health_monitor.enabled": "false",
        "l2.selective_rollout.enabled": "false",
    },
    # E3 E2 + Health Monitor：七维观测 + 告警（Observe-only，不影响训练信号）
    "E3_add_health_monitor": {
        "l2.enabled": "true",
        "l2.refresh_ratio.mode": "fixed",
        "l2.selective_rollout.enabled": "false",
    },
    # E4 E3 + Dynamic Ratio：adaptive α 三信号控制器（替代固定 initial）
    "E4_add_dynamic_ratio": {
        "l2.enabled": "true",
        "l2.refresh_ratio.mode": "adaptive",
        "l2.selective_rollout.enabled": "false",
    },
    # E5 E4 + Selective Rollout：候选池两阶段价值选择（80% value + 20% coverage）
    "E5_add_selective_rollout": {
        "l2.enabled": "true",
        "l2.refresh_ratio.mode": "adaptive",
        "l2.selective_rollout.enabled": "true",
    },
    # E6 全部模块 + Random Rollout：selective 关闭=uniform 随机选 prompt，
    # 与 E5 对照，验证 selective 本身的 compute/quality 贡献（§10 Q4）
    "E6_random_rollout": {
        "l2.enabled": "true",
        "l2.refresh_ratio.mode": "adaptive",
        "l2.selective_rollout.enabled": "false",
    },
}


def build_config(name: str, base_overrides: list[str] | None = None,
                 n_steps: int = 30, matrix: dict[str, dict] | None = None,
                 **extra) -> dict:
    """按实验名生成可运行配置（E0-E6 或 S2_E0-E3 矩阵 + 调用方追加覆盖）。

    matrix 参数化：传 STAGE2_ROLLOUT_MATRIX 即建 Stage 2 短 rollout 实验；默认
    EXPERIMENT_MATRIX（E0-E6）。默认给出 toy/CPU 可跑的轻量规模（小 n_steps、
    小 max_response_length），使整条 pipeline 在秒级完成。extra 可覆盖任意键
    （如把真实 512/1024/2048 压到 toy 预算验证协议抽象）。
    """
    matrix = matrix or EXPERIMENT_MATRIX
    if name not in matrix:
        raise KeyError(f"未知实验 {name!r}，可选 {list(matrix)}")
    overrides = list(base_overrides or [])
    for k, v in matrix[name].items():
        overrides.append(f"{k}={v}")
    # toy/CPU 友好默认（避免长 rollout 拖慢实验）
    overrides += [
        f"stage2.n_steps={int(n_steps)}",
        "stage2.batch_size=4",
        "l2.cache.max_response_length=4",
        "l2.cache.refresh_size=16",
        "l2.m_refresh=4",
    ]
    for k, v in extra.items():
        overrides.append(f"{k}={v}")
    return load_config(overrides=overrides)


def run_experiment(name: str, run_dir: str, n_steps: int = 30,
                   device: str = "cpu", matrix: dict[str, dict] | None = None,
                   **cfg_extra) -> dict:
    """跑单个 E0-E6 或 S2_E0-E3 实验，返回 {name, metrics, timings, run_dir, summary}。

    matrix 透传 build_config（默认 EXPERIMENT_MATRIX）。summary 聚合四类统一记录字段。
    """
    from .pipeline import FullStackOPDv2
    cfg = build_config(name, n_steps=n_steps, matrix=matrix, **cfg_extra)
    out = FullStackOPDv2(cfg, device=device).run(run_dir=run_dir)
    metrics = out["metrics"]
    summary = {
        "experiment": name,
        "n_steps": len(metrics),
        "reward_mean": (_mean([m.get("reward", 0.0) for m in metrics]) if metrics else 0.0),
        "pg_loss_mean": (_mean([m.get("pg_loss", 0.0) for m in metrics]) if metrics else 0.0),
        "kl_loss_mean": (_mean([m.get("kl_loss", 0.0) for m in metrics]) if metrics else 0.0),
        "stage2_train_s": round(out["timings"].get("stage2_train", 0.0), 3),
        "total_s": round(out["timings"].get("total", 0.0), 3),
        # Efficiency：teacher 前向集中在校验/分析时注入，此处给 pipeline 计时代理
        "refresh_rounds": len(metrics) // max(1, int(cfg["l2"].get("t_train", 100))),
    }
    return {"name": name, "metrics": metrics, "timings": out["timings"],
            "run_dir": run_dir, "summary": summary, "config": cfg}


def run_matrix(run_dir: str, n_steps: int = 30, device: str = "cpu",
               names: list[str] | None = None,
               matrix: dict[str, dict] | None = None,
               **cfg_extra) -> list[dict]:
    """跑 E0-E6 或 S2_E0-E3 全矩阵（或 names 子集），返回每实验结果。

    cfg_extra 透传 build_config（如把真实长预算压到 toy 预算验证协议）。
    """
    matrix = matrix or EXPERIMENT_MATRIX
    names = names or list(matrix)
    results = []
    for name in names:
        d = os.path.join(run_dir, name)
        os.makedirs(d, exist_ok=True)
        results.append(run_experiment(name, d, n_steps=n_steps, device=device,
                                      matrix=matrix, **cfg_extra))
    return results


def save_results(results: list[dict], run_dir: str) -> str:
    """把每实验 summary 汇总成 JSON 落盘，返回文件路径。"""
    path = os.path.join(run_dir, "l2_experiment_summary.json")
    payload = {r["name"]: r["summary"] for r in results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# --------------------------- 实验图（§10，matplotlib 可选）------------------
def plot_experiments(results: list[dict], out_dir: str) -> list[str]:
    """绘制 8 张实验对比图（6/7 最重要：teacher compute vs perf）。返回图文件路径。

    统一主题：E0 基线灰、E1 全量 L2 蓝、其余实验按序着色。matplotlib 缺失时跳过
    返回空列表（实验脚本不因无绘图库而失败）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                       # pragma: no cover
        return []
    paths = []
    names = [r["name"] for r in results]
    colors = ["#9E9E9E", "#1976D2"] + ["#C2185B", "#7B1FA2", "#00897B",
                                       "#F57C00", "#5D4037", "#455A64"][:max(0, len(names) - 2)]
    # 1/2/3: Training Quality（reward / pg / kl）
    metric_keys = [("reward_mean", "均奖励", "Training Quality"), ("pg_loss_mean", "PG loss", "Training Quality"),
                   ("kl_loss_mean", "KL loss", "Training Quality")]
    for key, label, cat in metric_keys:
        vals = [r["summary"][key] for r in results]
        _bar(out_dir, paths, key, names, vals, colors, label, cat)
    # 4: Efficiency（训练用时倒置 = 吞吐代理）
    vals = [r["summary"]["total_s"] for r in results]
    _bar(out_dir, paths, "total_s", names, vals, colors, "总用时(s)", "Efficiency")
    # 5: 训练步数一致性
    vals = [r["summary"]["n_steps"] for r in results]
    _bar(out_dir, paths, "n_steps", names, vals, colors, "训练步数", "Efficiency")
    # 6 (最重要): teacher compute vs perf —— 用「总用时」作 teacher compute 代理，
    #   横轴 = 归一化用时，纵轴 = 均奖励，气泡大小 = n_steps
    _scatter(out_dir, paths, results, colors, "total_s", "总用时(s)", "均奖励",
             "teacher_compute_vs_reward", "teacher compute vs perf")
    # 7 (次重要): refresh 轮数 vs reward
    _scatter(out_dir, paths, results, colors, "refresh_rounds", "refresh轮数", "均奖励",
             "refresh_rounds_vs_reward", "refresh effort vs perf")
    # 8: 各实验配色总览
    _bar(out_dir, paths, "overview", names,
         [1.0] * len(names), colors, "实验概览(*1)", "Ablation")
    return paths


def _bar(out_dir, paths, key, names, vals, colors, label, cat):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(names, vals, color=colors)
    ax.set_title(f"{cat} · {label}")
    ax.set_ylabel(label)
    ax.set_xticklabels(names, rotation=30, ha="right")
    fig.tight_layout()
    p = os.path.join(out_dir, f"plot_{key}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)


def _scatter(out_dir, paths, results, colors, xkey, xlabel, ylabel, key, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [r["summary"][xkey] for r in results]
    ys = [r["summary"]["reward_mean"] for r in results]
    ax.scatter(xs, ys, c=colors, s=90, edgecolors="k")
    for x, y, n in zip(xs, ys, [r["name"] for r in results]):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    p = os.path.join(out_dir, f"plot_{key}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


__all__ = ["EXPERIMENT_MATRIX", "STAGE2_ROLLOUT_MATRIX", "build_config",
           "run_experiment", "run_matrix", "save_results", "plot_experiments"]