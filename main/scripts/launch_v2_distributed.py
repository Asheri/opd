"""2 卡分布式骨架启动器：ray.init + 驱动 FullStackOPDv2（stage2.distributed=true）。

背景
----
`AsyncBatchedScheduler` 的线程版在单进程内跑四阶段流水线，GPU 上无法利用多卡。
`DistAsyncScheduler`（`scheduler.py`）把 rank1..W 派作 Ray rollout worker（每卡一个），
权重经 `WeightBroadcaster`（NCCL P2P）异步广播；learner 复用 `_train_step` 内核（π_old
加权 PG + PPO clip + k3 KL + staleness 双截断一行不动）。

⚠️ 骨架 demo 说明
----------------
- 模型仍是 CausalToyLM（toy），本脚本验证的是 **GPU 调度 / 稀疏缓存 / 分布式权重同步
  这条路径跑通**，不是真实 7B 训练（真实尺度走 async-opd 医疗 OPD）。
- `ray.init()` 是分布式路径前置条件（`DistAsyncScheduler.__init__` 里直接
  `RayRolloutWorker.remote(...)`，未自带 init）。
- 分布式路径需要对 torch.distributed(NCCL) 已建组；本脚本默认本地单节点 2 卡。

用法
----
    python scripts/launch_v2_distributed.py                       # 30 步（读 gpu_skeleton_2gpu.yaml）
    python scripts/launch_v2_distributed.py --set stage2.n_steps=3   # smoke（3 步）
"""
from __future__ import annotations

import argparse

import torch

import ray

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.pipeline import FullStackOPDv2


def main() -> None:
    ap = argparse.ArgumentParser(description="2 卡分布式骨架启动器")
    ap.add_argument("--config", default="configs/gpu_skeleton_2gpu.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="点分覆盖，如 --set stage2.n_steps=3")
    ap.add_argument("--num-gpus", type=int, default=2,
                    help="ray.init 暴露给 worker 的 GPU 数（默认 2）")
    args = ap.parse_args()

    # 校验 torch.distributed + CUDA
    assert torch.cuda.is_available(), "分布式骨架需要 CUDA"
    assert torch.cuda.device_count() >= 2, f"需 ≥2 GPU，当前 {torch.cuda.device_count()}"
    assert torch.distributed.is_available(), "需要 torch.distributed(NCCL)"

    # ray.init：本地单节点 2 卡；NCCL_DEBUG 便于首跑排障（可关）
    ray.init(num_gpus=args.num_gpus,
             runtime_env={"env_vars": {"NCCL_DEBUG": "INFO"}} if args.num_gpus <= 2 else None)
    print(f"[launch] ray 已初始化（num_gpus={args.num_gpus}）")

    cfg = load_config(path=args.config,
                      overrides=args.overrides + ["stage2.distributed=true"])
    print(f"[launch] device={args.device}  n_steps={cfg['stage2']['n_steps']}  "
          f"topk(stu={cfg['stage2']['top_k_student']})  rollout={cfg['stage2']['rollout_engine']}\n")

    opd = FullStackOPDv2(cfg, device=args.device)
    out = opd.run()
    metrics = out["metrics"]
    timings = out["timings"]

    print("\n=== 2 卡分布式骨架结果 ===")
    print(f"训练步数            : {len(metrics)}")
    if metrics:
        last = metrics[-1]
        print(f"最后一步 version    : {last['version']}")
        print(f"最后一步 staleness  : age={last['age']} (>0 ⇒ 跨进程在消费陈旧样本)")
        print(f"最后一步 loss       : {last['loss']:.4f} (pg={last['pg_loss']:.4f}, kl={last['kl_loss']:.4f})")
        print(f"最后一步 E[Δ_T]     : {last['reward']:+.4f}")
    print("逐 stage 耗时:", {k: round(v, 2) for k, v in timings.items()})
    print("\n✓ 分布式骨架跑通（算法内核与线程版一致，Staleness 双截断 / NCCL 广播在档）")


if __name__ == "__main__":
    main()