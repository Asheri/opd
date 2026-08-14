# Stage 1 磁盘缓存存储决策报告（§8 性能 + §9 验收）

> 状态：**模板，待服务器实测回填**。本文件给出 §8 性能 benchmark 与 §9 验收的
> 执行方法、结果表与是否需降 residency 的决策判据。标「⏳待实测」处由服务器
> `cache_bench.py` / `cache_acceptance.py` 回填真实数字。

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

### §8 结果表（⏳待实测回填）

| K | cache_size_disk(GB) | RAM_peak(MB) | GPU_peak(MB) | write_time(s) | lat(ms) | thr(tok/s) | I/O(GB/s) |
|---|---|---|---|---|---|---|---|
| 32 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 64 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 128 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 256 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

**50K×8192 盘体预估**（N×T×K×8 字节，直接按公式）：

| K | 盘体 |
|---|---|
| 32 | 50K×8192×32×8 ≈ 104.9 GB |
| 64 | ≈ 209.7 GB |
| 128 | ≈ 419.4 GB |
| 256 | ≈ 838.9 GB |

> 磁盘 mmap 驻留下盘体是主要成本（RAM/GPU 只驻 batch 行）。K=32 时 ~105GB 盘体
> 是本阶段推荐起步档；更大的 K 留给显存/盘体充裕时按需升级。

## 3. 验收（`scripts/cache_acceptance.py`）——§9 5K 清单

**执行命令（服务器，真实 GPU + 数据）**：
```bash
python scripts/cache_acceptance.py --N 5000 --max-len 8192 --top-k 32 \
  --data /root/autodl-tmp/datasets/skywork_50k.jsonl --steps 5
```

| # | 验收项 | 本地 CPU 冒烟 | 服务器 5K GPU |
|---|---|---|---|
| 1 | 5K build 落盘（逐 chunk 直写 memmap） | ✅ | ⏳ |
| 2 | 重载 + checksum 验签 + 一致性 | ✅ | ⏳ |
| 3 | 随机 batch lookup 与 in-memory 逐位一致 | ✅ | ⏳ |
| 4 | restart 后 lookup 一致 | ✅ | ⏳ |
| 5 | 训练 5 step（磁盘缓存被 _train_step 消费，无 teacher 前向） | ✅ | ⏳ |
| 6 | 全程 `torch.cuda.max_memory_allocated()` 无 OOM | 跳过（CPU） | ⏳ |

**决策判据**：6 项全过才允许大规模 50K×8192 build。若某 K 下显存仍不适合，
优先**降低 cache residency**（更小 batch / 更小 K / 数据分片），不靠改架构硬撑。

## 4. 结论与推荐（⏳待回填）

- 推荐大规模 build 的 K 档：⏳（默认 32）
- 是否需降 residency：⏳
- 大规模 build 命令：见 `prepare_skywork_responses.py` 与 stage1 磁盘路径