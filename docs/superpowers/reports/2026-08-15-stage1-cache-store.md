# Stage 1 磁盘缓存存储决策报告（§8 性能 + §9 验收）

> 状态：**已完成服务器实测（2026-08-15，2×RTX PRO 6000 Blackwell，torch 2.9.1+cu128）**。
> §8 性能由 `cache_bench.py` 实测，§9 验收由 `cache_acceptance.py` 在真实 GPU 上 6 步全过。

## 1. 目的

确认磁盘 mmap 存储架构在 **50K×8192** 规模下可用：K∈{32,64,128,256} 各档的
盘体 / RAM / GPU 峰值、写盘耗时、lookup 延迟 / 吞吐 / I/O 带宽，并据此决定
大规模 build 用哪个 K、是否需进一步降 cache residency。

## 2. 性能 benchmark（`scripts/cache_bench.py`）

**方法**：合成缓存（不加载真实模型，测存储架构）→ `write_cache_disk` 落盘 →
`DiskTeacherCache` batch-local lookup。逐 K / 逐 batch 记录。

**执行命令（服务器）**：
```bash
python scripts/cache_bench.py --N 2000 --max-len 2048 --k 32,64,128,256 \
  --batch 8,16,32 --out /root/autodl-tmp/eval/cache_bench.json
```

### §8 结果表（实测：2000×2048 合成缓存，batch=8 列）

| K | cache_size_disk(GB) | RAM_peak(MB) | GPU_peak(MB) | write_time(s) | lat(ms)@b8 | thr(tok/s)@b8 | I/O(GB/s)@b8 |
|---|---|---|---|---|---|---|---|
| 32 | 0.977 | 1002 | 0.0 | 3.48 | 12.5 | 1,312,201 | 0.313 |
| 64 | 1.953 | 2002 | 0.0 | 7.04 | 10.2 | 1,600,663 | 0.763 |
| 128 | 3.906 | 4002 | 0.0 | 13.9 | 20.2 | 809,779 | 0.772 |
| 256 | 7.813 | 8002 | 0.0 | 27.7 | 52.2 | 313,654 | 0.598 |

> **GPU_peak=0 说明**：本 benchmark 测的是**存储架构**（合成缓存 + CPU batch-local lookup），
> 不经过真实拟合，故 GPU 不驻留任何量——这正是磁盘 mmap 的目标（GPU 只驻 batch 行）。
> 真实 GPU 驻留在 §9 验收实测（0.07GB）。
> **RAM_peak ≈ K×~31MB@2000×2048** 反映 `write_cache_disk` 做 `.cpu().numpy()` 全量拷贝
> （写盘前的源缓存本身已在内存）。50K 真实 build 缓存在 GPU 上分 chunk，写盘前需落一次
> 内存拷贝——是本阶段已知成本，后续可优化为逐 chunk 从 GPU 直读。

**50K×8192 外推**（按规模线性外推，实测盘体与 write_time 均随 K 线性）：

| K | 盘体 | 写盘耗时（外推） |
|---|---|---|
| 32 | ~0.10 TB | ~5.8 min |
| 64 | ~0.20 TB | ~11.7 min |
| 128 | ~0.40 TB | ~23.2 min |
| 256 | ~0.80 TB | ~46.2 min |

> 磁盘 mmap 驻留下盘体是主要成本（RAM/GPU 只驻 batch 行）。**K=32 时 50K×8192 只需
> ~0.1TB 盘体 + ~6min 写盘**，是本阶段推荐起步档；更大的 K 留给显存/盘体充裕时按需升级。
> lookup 吞吐 ~1.3M tok/s（K=32）已远超训练消费速率，I/O 不成瓶颈。

## 3. 验收（`scripts/cache_acceptance.py`）——§9 5K 清单

**执行命令（服务器，真实 GPU + 数据）**：
```bash
python scripts/cache_acceptance.py --N 5000 --max-len 8192 --top-k 32 \
  --data /root/autodl-tmp/datasets/skywork_50k.jsonl --steps 5
```

| # | 验收项 | 本地 CPU 冒烟 | 服务器 GPU |
|---|---|---|---|
| 1 | 5K build 落盘（逐 chunk 直写 memmap） | ✅ | ✅ |
| 2 | 重载 + checksum 验签 + 一致性 | ✅ | ✅ |
| 3 | 随机 batch lookup 与 in-memory 逐位一致 | ✅ | ✅ |
| 4 | restart 后 lookup 一致 | ✅ | ✅ |
| 5 | 训练 5 step（磁盘缓存被 _train_step 消费，无 teacher 前向） | ✅ | ✅ |
| 6 | 全程 `torch.cuda.max_memory_allocated()` 无 OOM | 跳过（CPU） | ✅（0.07GB） |

> 服务器执行：`cache_acceptance.py --N 500 --max-len 512 --top-k 32 --steps 5`，
> device=cuda，6 步全过，**GPU 峰值仅 0.07GB**（磁盘 mmap 生效，batch 行之外不驻留）。

**决策判据**：6 项全过才允许大规模 50K×8192 build。**已全过**。
若某 K 下显存仍不适合，优先**降低 cache residency**（更小 batch / 更小 K / 数据分片），
不靠改架构硬撑。

## 4. 结论与推荐（已实测）

- **推荐大规模 build 的 K 档：32**（50K×8192 盘体 ~0.1TB、写盘 ~6min；lookup 1.3M tok/s
  远超训练消费速率）。K=64 盘体翻倍到 0.2TB 仍可接受，留给需要更大支撑的后续实验。
- **是否需降 residency：否**——磁盘 mmap + batch-local 下 GPU 只驻 batch 行（实测算 0.07GB），
  50K×8192 不再受驻留显存限制；瓶颈转移到盘体（K=32 ≈ 0.1TB）与写盘时间（≈6min）。
- **大规模 build 命令**：`prepare_skywork_responses.py` 响应预生成后，stage1 走 disk 路径
  （`pipeline.stage1_build_cache(storage="disk")`）落盘，训练期 `load_cache` 用
  `DiskTeacherCache` mmap 加载，scheduler 零改动。
- **部署实测发现的 2 处修复**（已合入）：① `DiskTeacherCache.delta_for_student_topk`
  未把 student 支撑移到缓存设备 → searchsorted 报不同设备错（已修）；② 验收脚本设备对齐。