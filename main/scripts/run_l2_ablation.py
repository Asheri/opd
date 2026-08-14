"""E0-E6 实验矩阵运行脚本（§10，任务 6.2）。

用法（在 main/ 下）：
    python scripts/run_l2_ablation.py --run-dir out/l2_exp --n-steps 30
    python scripts/run_l2_ablation.py --only E0_baseline_off,E1_full_l2 --n-steps 20

跑完把每实验 summary 聚合为 l2_experiment_summary.json，并绘制 8 张实验图
（teacher compute vs perf 最重要，见 experiment.py plot_experiments）。matplotlib
缺失时仅落 JSON、跳过绘图。
"""
from __future__ import annotations

import argparse
import os
import sys

# 脚本位于 main/scripts/ 下：把仓库根（main/）加进 sys.path 以便 import fullstack_opd_v2
_MAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN_ROOT not in sys.path:
    sys.path.insert(0, _MAIN_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="L2 E0-E6 实验矩阵")
    ap.add_argument("--run-dir", default="l2_experiments", help="输出目录")
    ap.add_argument("--n-steps", type=int, default=30, help="每实验训练步数")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, help="逗号分隔的实验名子集（默认全矩阵）")
    args = ap.parse_args()

    from fullstack_opd_v2.experiment import (
        EXPERIMENT_MATRIX, run_matrix, save_results, plot_experiments)

    os.makedirs(args.run_dir, exist_ok=True)
    names = None
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    print(f"跑 E0-E6 实验矩阵: {names or list(EXPERIMENT_MATRIX)} "
          f"(n_steps={args.n_steps}, device={args.device})")
    results = run_matrix(args.run_dir, n_steps=args.n_steps, device=args.device,
                         names=names)
    summary_path = save_results(results, args.run_dir)
    print(f"实验汇总已落盘: {summary_path}")
    for r in results:
        s = r["summary"]
        print(f"  [{(s['n_steps'])}步] {r['name']:<28} "
              f"reward={s['reward_mean']:+.4f}  pg={s['pg_loss_mean']:.4f}  "
              f"kl={s['kl_loss_mean']:.4f}  total={s['total_s']:.2f}s")
    plots = plot_experiments(results, args.run_dir)
    if plots:
        print(f"已绘制 {len(plots)} 张实验图:")
        for p in plots:
            print(f"  - {p}")
    else:
        print("matplotlib 未安装，跳过绘图（JSON 已落盘）")


if __name__ == "__main__":
    main()