#!/usr/bin/env python3
"""L2 E0-E6 真实 GPU 实验：skywork_17b.yaml 基座 + EXPERIMENT_MATRIX 覆盖。

镜像 run_s2_real.py（同一执行底座），但矩阵用 E0-E6（L2 四能力累加 ablation）：
真实 1.7B 学生 + 真实教师对（JustRL/R1-Distill）+ pilot 缓存（500 条）上量化各
模块贡献（Training Quality / Teacher Compute / Rollout Tokens），验证 E5(selective)
优于 E6(random) 等方向性结论在真实规模是否成立。

用法：
  python scripts/run_l2_real.py --config configs/skywork_17b.yaml \
      --run-dir /root/autodl-tmp/l2_real --device cuda:0 --n-steps 20 \
      --eos-id 151645 --materialized 500 --load-cache \
      --cache-path /root/autodl-tmp/cache_skywork_17b.pt
"""
from __future__ import annotations

import argparse
import json
import os

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.experiment import EXPERIMENT_MATRIX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="基座 YAML（skywork_17b.yaml）")
    p.add_argument("--run-dir", required=True, help="输出根目录")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--names", nargs="+", default=None, help="实验名（默认全矩阵）")
    p.add_argument("--eos-id", type=int, default=None,
                   help="l2.rollout.eos_token_id（校准值 151645）")
    p.add_argument("--materialized", type=int, default=0,
                   help="base.materialized_size（预生成 response 锚点数）")
    p.add_argument("--m-refresh", type=int, default=8,
                   help="l2.m_refresh（每刷新相位 rollout 条数）")
    p.add_argument("--refresh-min", type=int, default=5,
                   help="l2.cache.refresh_min_interval（步数间隔触发刷新）")
    p.add_argument("--cache-path", default=None, help="教师缓存路径（覆盖 YAML）")
    p.add_argument("--load-cache", action="store_true", help="复用已建缓存")
    p.add_argument("--batch-size", type=int, default=None,
                   help="覆盖 stage2.batch_size（真实长序列防 OOM 调小）")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="覆盖 l2.rollout.max_new_tokens（真实预算，默认继承矩阵）")
    p.add_argument("--refresh-size", type=int, default=None,
                   help="覆盖 l2.cache.refresh_size（默认 5000×T×K 预分配 GPU OOM，pilot 用小值）")
    return p.parse_args()


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def main() -> None:
    args = parse_args()
    names = args.names or list(EXPERIMENT_MATRIX)
    os.makedirs(args.run_dir, exist_ok=True)
    results = []

    for name in names:
        if name not in EXPERIMENT_MATRIX:
            raise SystemExit(f"未知实验 {name!r}，可选 {list(EXPERIMENT_MATRIX)}")
        overrides = [f"{k}={v}" for k, v in EXPERIMENT_MATRIX[name].items()]
        overrides += [
            f"stage2.n_steps={args.n_steps}",
            f"l2.m_refresh={args.m_refresh}",
            f"l2.cache.refresh_min_interval={args.refresh_min}",
            f"l2.cache.refresh_max_interval={args.refresh_min + args.n_steps}",
        ]
        if args.batch_size:
            overrides.append(f"stage2.batch_size={args.batch_size}")
        if args.refresh_size:
            overrides.append(f"l2.cache.refresh_size={args.refresh_size}")
        if args.eos_id is not None:
            overrides.append(f"l2.rollout.eos_token_id={args.eos_id}")
        if args.max_new_tokens is not None:
            overrides.append(f"l2.rollout.max_new_tokens={args.max_new_tokens}")
        if args.materialized:
            overrides.append(f"base.materialized_size={args.materialized}")
        if args.cache_path:
            overrides.append(f"stage1.cache_path={args.cache_path}")
        overrides.append(f"stage1.load_cache={'true' if args.load_cache else 'false'}")
        cfg = load_config(path=args.config, overrides=overrides)

        d = os.path.join(args.run_dir, name)
        os.makedirs(d, exist_ok=True)
        print(f"\n===== 实验 {name} =====  (n_steps={args.n_steps}, "
              f"materialized={args.materialized}, load_cache={args.load_cache})", flush=True)
        try:
            from fullstack_opd_v2.pipeline import FullStackOPDv2
            out = FullStackOPDv2(cfg, device=args.device).run(run_dir=d)
            metrics = out["metrics"]
            train_metrics = [m for m in metrics
                             if isinstance(m, dict) and m.get("phase") != "rollout"]
            summary = {
                "experiment": name,
                "n_steps": len(train_metrics),
                "reward_mean": _mean([m.get("reward", 0.0) for m in train_metrics]),
                "pg_loss_mean": _mean([m.get("pg_loss", 0.0) for m in train_metrics]),
                "kl_loss_mean": _mean([m.get("kl_loss", 0.0) for m in train_metrics]),
                "total_s": round(out["timings"].get("total", 0.0), 3),
                "stage2_train_s": round(out["timings"].get("stage2_train", 0.0), 3),
            }
            # Efficiency / rollout 指标（取最后刷新相位）
            for col, key in [("rollout/teacher_forward_tokens", "teacher_forward_tokens"),
                             ("rollout/rollout_tokens", "rollout_tokens"),
                             ("rollout/n_appended", "rollout_n_appended"),
                             ("rollout/n_loop", "rollout_n_loop"),
                             ("rollout/mean_disagreement", "mean_disagreement")]:
                for m in reversed(metrics):
                    if isinstance(m, dict) and col in m:
                        summary.setdefault(key, m[col])
                        break
            print(f"  summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)
            results.append({"name": name, "summary": summary, "run_dir": d})
        except Exception as e:
            import traceback
            print(f"  ❌ 实验 {name} 失败: {e}", flush=True)
            traceback.print_exc()
            results.append({"name": name, "summary": {"experiment": name, "error": str(e)},
                            "run_dir": d})

    out_path = os.path.join(args.run_dir, "l2_experiment_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({r["name"]: r["summary"] for r in results}, f, indent=2,
                  ensure_ascii=False)
    print(f"\n✅ 汇总写入 {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
