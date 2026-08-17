#!/usr/bin/env python3
"""S2 真实 GPU 实验：加载 skywork_17b.yaml 基座 + STAGE2_ROLLOUT_MATRIX 覆盖。

跑真实模型 S2_E0/E1/E2/E3（真实 512/1024/2048 rollout），产出训练 summary
（reward/pg_loss/kl_loss + rollout/* 状态计数），供 report_stage2 生成 Q1-Q4。

用法：
  python run_s2_real.py --config configs/skywork_17b.yaml --run-dir <dir> \
      --device cuda:0 --n-steps 30 \
      [--names S2_E0_static S2_E1_opd512 S2_E2_opd1024 S2_E3_opd2048] \
      [--eos-id 151645] [--materialized 500]

与 toy 端 run_matrix 的关键差异：
  - 基座是真实 YAML（model_kind=hf + 真实模型/数据/教师对），非 DEFAULT_CONFIG_V2。
  - 每个实验独立 run 目录；共享同一份预建教师缓存（首实验 load_cache=false 建，
    其后实验 load_cache=true 复用，避免重复 GPU 建缓存）。
  - 单实验 try/except 隔离：一个失败不中断矩阵。
  - 产出每实验 {name, summary, run_dir} 并入 l2_experiment_summary.json。
"""
from __future__ import annotations

import argparse
import json
import os

import os
import sys

# 脚本位于 main/scripts/ 下；直接运行（python scripts/xxx.py）时 sys.path[0] 是 scripts/
# 而非 repo 根，导致 `from fullstack_opd_v2 ...` 失败。显式把 main/ 加入 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.experiment import STAGE2_ROLLOUT_MATRIX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="基座 YAML（skywork_17b.yaml）")
    p.add_argument("--run-dir", required=True, help="输出根目录")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--names", nargs="+", default=None,
                   help="实验名（默认全矩阵）")
    p.add_argument("--eos-id", type=int, default=None,
                   help="l2.rollout.eos_token_id（校准值 151645）")
    p.add_argument("--materialized", type=int, default=0,
                   help="base.materialized_size（预生成 response 的锚点数）")
    p.add_argument("--m-refresh", type=int, default=8,
                   help="l2.m_refresh（每刷新相位的 rollout 条数）")
    p.add_argument("--refresh-min", type=int, default=10,
                   help="l2.cache.refresh_min_interval（步数间隔触发刷新）")
    p.add_argument("--cache-path", default=None,
                   help="教师缓存路径（覆盖 YAML）")
    p.add_argument("--load-cache", action="store_true",
                   help="复用已建缓存（首实验建后置 true）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="覆盖 stage2.batch_size（真实 (4,3072,151936) 序列 flash 后仍 ~87GB，"
                        "batch 4→2 把训练激活减半防 OOM；默认继承 config）")
    p.add_argument("--refresh-size", type=int, default=None,
                   help="覆盖 l2.cache.refresh_size（默认 5000×T×K 预分配 GPU OOM，pilot 用 ~64）")
    return p.parse_args()


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def main() -> None:
    args = parse_args()
    names = args.names or list(STAGE2_ROLLOUT_MATRIX)
    os.makedirs(args.run_dir, exist_ok=True)
    results = []

    for name in names:
        if name not in STAGE2_ROLLOUT_MATRIX:
            raise SystemExit(f"未知实验 {name!r}，可选 {list(STAGE2_ROLLOUT_MATRIX)}")
        overrides = [f"{k}={v}" for k, v in STAGE2_ROLLOUT_MATRIX[name].items()]
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
            summary = {
                "experiment": name,
                "n_steps": sum(1 for m in metrics
                               if isinstance(m, dict) and m.get("phase") != "rollout"),
                "reward_mean": (_mean([m.get("reward", 0.0) for m in metrics]) if metrics else 0.0),
                "pg_loss_mean": (_mean([m.get("pg_loss", 0.0) for m in metrics]) if metrics else 0.0),
                "kl_loss_mean": (_mean([m.get("kl_loss", 0.0) for m in metrics]) if metrics else 0.0),
                "total_s": round(out["timings"].get("total", 0.0), 3),
            }
            # rollout 状态计数（最后一个 refresh 相位）
            for col, key in [("rollout/n_appended", "rollout_n_appended"),
                             ("rollout/n_eos", "rollout_n_eos"),
                             ("rollout/n_loop", "rollout_n_loop")]:
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

    # 决策报告 JSON
    out_path = os.path.join(args.run_dir, "l2_experiment_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({r["name"]: r["summary"] for r in results}, f, indent=2,
                  ensure_ascii=False)
    print(f"\n✅ 汇总写入 {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()