# GPU 显存占用分析与并行训练方案（2×RTX PRO 6000 · 覆盖 1.7B/4B/7B 三档）

> 硬件：**2×RTX PRO 6000 Blackwell，96GB×2 + NVLink**（CUDA 12.8，统一环境 `requirements-unified.txt`）。
> 算法内核：`main/fullstack_opd_v2/` 全栈 OPD（Stage 0 教师 RL → Stage 1 离线 Δ_T 缓存 → Stage 2 异步 Direct-OPD）。
> 项目真实模型：教师对 JustRL-1.5B / DeepSeek-R1-Distill-Qwen-1.5B；学生 Qwen3-1.7B / Qwen3-4B / R1-Distill-Qwen-7B；词表 V≈152k。
> 相关：`OPTIMIZATION_PLAN_2xRTXPRO6000.md`（性能方案，尺寸上限账）、`RUNBOOK_2xPRO6000.md`（运维）、`ENGINEERING_IMPLEMENTATION.md`（实现）。

> ⚠️ **诚实声明**：本文数字为**估算账**（每参数常量精确、激活/缓存按规模推算），上线以
> `nvidia-smi` / `nsys` 实测为准；NVLink 带宽、激活占用随 batch 变，文中已标注。

---

## 0. 结论速览（先看这个）

1. **三档学生都装得下 2×96GB**。唯一临界点是 **7B + fp32 Adam = 122GB > 单卡 96GB**；
   切 **8-bit Adam → 76GB 单卡可装**，且把另一卡空出来做 scorer——这是整份方案的关键决策。
2. **最大化 GPU 利用的正确形态 = 2 卡异步流水线（rank0 learner ∥ rank1 scorer）**，
   **不是 learner TP=2**：TP=2 会把 rank1 占成 learner 分片，失去"算 s_old 与训练更新"两个重活
   的解耦重叠（且与现骨架 `tp_size=1` 护栏冲突）。这正好对齐现有 `DistAsyncScheduler` 设计。
3. **topk 稀疏在 Stage 2 是硬约束、不是可选项**：真实词表下 (B,T,V) 前向中间张量 ~10GB/批
   （B=8,T=2048），dense 队列必爆；topk（K=64）降到 6MB/批。
4. **Δ_T/ref 锚点缓存走 mmap + 按批载入，不常驻**：L1 胖 D 后 topk 缓存可到 ~79GB
   （N=10k,T=2048），常驻会吃掉整卡——mmap + 按 idx 切片 H2D 是正解。
5. **1.7B/4B 档 GPU 严重过剩**（4B fp32-Adam 仅 62GB）：最大化利用 = 加大 batch（激活头寸）
   或 DP=2 双卡并行训练（见 §5.3 选项 C）。

---

## 1. 显存占用模型（每参数常量，精确）

| 项 | 字节/参数 | 说明 |
|---|---|---|
| 权重（bf16） | 2 B | 训练 + 推理 |
| 梯度（bf16） | 2 B | 仅训练 |
| fp32 Adam（两矩 4B×2 + fp32 master 4B） | 12 B | 优化器（**天花板**） |
| 8-bit Adam（两矩 fp8 1B×2 + fp32 master 4B） | 6 B | 优化器省一半 |
| **fp32-Adam 训练合计** | **16 B** | 2+2+12 |
| **8-bit-Adam 训练合计** | **10 B** | 2+2+6 |

> 🔴 **优化器是天花板，TP/fp8 只省权重不省优化器**（`OPTIMIZATION_PLAN_2xRTXPRO6000.md` §0.1 已修正的 OOM 教训）。
> fp8 训练（Transformer Engine）保留 fp32 master，**不减少优化器内存**。

---

## 2. 项目真实模型尺寸（benchmarks/aime24_25 三组合）

| 角色 | 模型 | 参数 | bf16 权重 | fp32-Adam 训练 | 8-bit-Adam 训练 |
|---|---|---|---|---|---|
| teacher_rl | JustRL-DeepSeek-1.5B | 1.5B | 3.0GB | 24.0GB | 15.0GB |
| teacher_ref | DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 3.0GB | 24.0GB | 15.0GB |
| student 档 1 | Qwen3-1.7B | 1.7B | 3.5GB | 27.8GB | 17.4GB |
| student 档 2 | Qwen3-4B | 3.9B | 7.7GB | 61.8GB | 38.6GB |
| student 档 3 | R1-Distill-Qwen-7B | 7.6B | 15.2GB | **121.6GB ❌** | **76.0GB ✓** |
| 词表 | Qwen3 / Qwen2.5 | V≈152k | — | — | — |

> 7B 档参数取 Qwen2.5-7B 实际 7.6B；4B 档取 Qwen3-4B 实际 3.86B。

---

## 3. 分阶段显存账

### 3.1 Stage 0 · 教师 RL（1.5B，微）
- 训练 fp32-Adam ≈ 24GB（8-bit 15GB）+ 激活 → **单卡轻松**，另一卡可并行预跑无依赖的
  Stage 1 数据预处理。耗时占比可忽略（模型只有 1.5B）。

### 3.2 Stage 1 · 离线缓存 build（教师对 1.5B 推理）
- 双教师 bf16 权重共 6.2GB + vLLM KV → **<10GB**；用 **vLLM TP=2 吃满两卡**缩短一次性 build
  时间（输出 topk 缓存见 §4）。教师算完即释放，训练期零教师前向。

### 3.3 Stage 2 · Direct-OPD（learner ∥ scorer）—— 主账
| 卡 | 角色 | 内容 | 7B 占用（估算） |
|---|---|---|---|
| rank0 | **learner**（训练） | 权重 15.2 + 梯度 15.2 + 8-bit Adam 45.6 = **76GB** + 激活（梯度检查点后 <1GB）+ 缓存切片 | ~80GB / 96GB |
| rank1 | **scorer**（算 s_old） | 学生旧快照 bf16 **15.2GB**（前向模型，无优化器）+ 权重广播暂存 + 缓存切片 | ~20GB / 96GB |

**三档 student 的 learner 单卡预算（不加并行）**：

| 档 | 权重+梯度 | fp32-Adam | 8-bit-Adam | 单卡 96GB 结论 |
|---|---|---|---|---|
| 1.7B | 6.8GB | 27.8GB | 17.4GB | fp32 宽松（激活头寸大） |
| 4B | 15.4GB | 61.8GB | 38.6GB | fp32 可（+激活略紧）；8-bit 更稳 |
| 7B | 30.4GB | **121.6GB ❌** | **76.0GB ✓** | **必须 8-bit Adam（或 TP=2，但不推荐，见 §5.2）** |

---

## 4. OPD 独有：Δ_T / ref 锚点缓存显存

这是本流水线区别于普通 LLM 训练的内存消费者，两处：

**① 离线缓存（Stage 1 产物，mmap 跨进程共享，不常驻）**——dense 在真实词表不可行：
```
dense (N,T,V) fp32：N=7.5k, T=512 → 7.5e3×512×152k×4B ≈ 2.3 TB  ❌
topk (N,T,K) 每位置 id(int16 2B)+delta(fp32 4B)=6B：
  N=7.5k, T=512,  K=64 → 1.5GB/张量；  delta+ref 锚点 ×2 = 2.9GB； L1 fat×5 = 14.7GB
  N=10k,  T=2048, K=64 → 7.9GB/张量；  delta+ref 锚点 ×2 = 15.7GB； L1 fat×5 = 78.6GB
```
**结论**：缓存要么 mmap + 按 idx 切片载入（训练期只占 ~1GB），要么小规模才考虑常驻。
L1 胖 D（默认 M=4）把缓存放大 ×5——**mmap 是唯一现实答案**。

**② 训练期中间张量 (B,T,V)——topk 是硬约束**：
```
scorer/learner 前向 logits (B,T,V)：B=8, T=2048, V=152k → 10GB/批（fp32）❌ 队列必爆
topk 学生支撑 (B,T,K=64)：id+logp = 6.3MB/批 ✓
```
真实词表下必须 `top_k_student>0`（`delta_for_student_topk` 现场展开），dense 只在 toy 词表合法。

---

## 5. 并行训练方案（2×PRO6000，最大化 GPU 利用）

### 5.1 核心方案：2 卡异步流水线（learner ∥ scorer）——与现有 `DistAsyncScheduler` 对齐

```
rank0 · learner（训练）                     rank1 · rollout scorer（算 s_old，只前向）
──────────────────────────────             ──────────────────────────────
学生 7B（训练态）                           学生 7B 旧快照（bf16，无优化器）
 8-bit Adam + 梯度检查点 ≈80GB/96GB          权重 15GB ≈20GB/96GB
──────────────────────────────             ──────────────────────────────
每步：s_cur 前向+反传+8-bit Adam 更新         每步：acquire_if_newer 收最新快照
     → publish → NCCL 广播（非阻塞 isend）      → response_dists 算 s_old（topk）
     → 消费 (idxs, s_old) 算 loss              → (idxs, s_old, ver) 推给 learner（异步队列）
```
- **两个 GPU 同时满负荷**：rank0 在训练、rank1 在打分——正是 CPU 版四线程流水线的 GPU 形态，
  把"算 s_old"与"训练更新"两个重活解耦到两张卡（`staleness` 双截断原样保留）。
- **权重同步 = NCCL 非阻塞 P2P/broadcast**（NVLink）：`WeightStore.acquire_if_newer` 的 NCCL 版，
  learner 每步 publish、scorer 按版本拉取；**双缓冲**让广播与下一步训练重叠。
- **缓存不常驻**：Δ_T / ref 锚点 mmap，训练期按 idx 切片 `non_blocking` H2D，与计算重叠。
- **scorer 无大 KV**：它做 teacher-forcing 前向（算 logprob），不是自回归解码；若将来 L2 要
  学生自回归 refresh 才需给 scorer 配 vLLM + KV（7B 也仅 ~15-20GB）。

### 5.2 为什么不用 learner TP=2（避开陷阱）

7B + fp32 Adam（122GB）会让人想切 Megatron TP=2——**但**：
1. **colocated 冲突**：rank1 被占为 learner 分片后，无法同时做 scorer → 失去 5.1 的异步重叠
   （`ENGINEERING_IMPLEMENTATION.md` §4 已记：TP 集合通信需 rank1 协同，与并发 rollout 死锁，
   `scheduler.py` 有护栏）。
2. 8-bit Adam 已让 7B 单卡装下（76GB），**TP 省的是不存在的压力**。
→ 默认 **learner 单卡 + scorer 单卡**。只有当需要 fp32 Adam 的数值精度（不用 8-bit）时才走
TP=2 + "colocated 交替相位"（未实现，L2 增量）。

### 5.3 三档学生的推荐配置（最大化利用）

| 档 | 推荐形态 | 内存头寸 | 说明 |
|---|---|---|---|
| **7B** | 5.1 异步流水线；learner 8-bit Adam + 梯度检查点；rank1 scorer | learner ~80GB；scorer ~20GB | 默认路径，双卡都忙 |
| **4B** | 同 5.1；learner 可回 fp32 Adam（62GB）或 8-bit 换更大 batch | 大量头寸 → 加大 batch 喂饱 tensor core | 头寸换吞吐 |
| **1.7B** | **选项 C：DP=2**（两卡都训练，梯度 all-reduce 同步；scorer 每卡一份前向副本） | 27.8GB×2 | 模型小 → 打分太便宜，单 scorer 喂不饱两卡；DP 翻吞吐 |

> 选项 C（DP=2）实现代价：`WeightStore`/`_publish` 改为梯度 all-reduce + 每卡 scorer 副本；
> 1.7B 打分（~15GB bf16 前向）可与训练同卡共存，无需第三份显存。属吞吐最大化的增量，非默认。

### 5.4 最大化利用的杠杆清单（按 ROI）

| 杠杆 | 作用 | 说明 |
|---|---|---|
| **bf16 autocast + TF32** | 逼近 250 TFLOPS/卡峰值 | 矩阵乘默认 TF32；精度敏感处 bf16 |
| **8-bit Adam（7B）** | 把 OOM 变可行 + 空出 rank1 | `adamw_8bit`，见 5.1 |
| **梯度检查点** | 激活从 ~6GB 压到 <1GB → 加 batch | 7B 训练必开 |
| **大 micro-batch / 梯度累积** | 激活头寸 96GB 下喂饱 tensor core | 4B/1.7B 档主杠杆 |
| **双缓冲 NCCL 广播** | 权重同步与训练重叠 | `WeightBroadcaster` 已就绪 |
| **缓存 mmap + 预取** | 不占 96GB 常驻、H2D 与计算重叠 | §4，L1 胖 D 后必选 |
| **torch.compile + CUDA Graph** | 固定 shape 的 response_dists 提速 | `dynamic=False` 谨慎 |
| **Stage 1 用 vLLM TP=2** | 一次性 cache build 吃满两卡 | 教师仅 1.5B，快 |

### 5.5 配置映射（到现有 CLI）

```bash
# Stage 2 启动（7B，5.1 异步流水线）
torchrun --nproc_per_node=2 python -m fullstack_opd_v2 \
    --device cuda:0 --set stage2.distributed=true --set stage2.n_gpus=2 \
    --set stage2.rollout_engine=vllm --set stage2.rollout_model=<student_path> \
    --set stage2.tp_size=1 --set stage2.rollout_tp_size=2 \
    --set stage2.cache_mode=topk --set stage2.top_k_teacher=64 \
    --set stage2.top_k_student=64 --set stage2.ref_topk=64 \
    --set stage2.offload_to_cpu=true \
    --set stage1.warmup_M=4 --set stage1.warmup_source=student_init
```
> `tp_size=1` 是有意的（learner 单卡，见 5.2）；`rollout_tp_size=2` 是 scorer 侧的 vLLM。
> 8-bit Adam 与 mmap 缓存是代码层待接的骨架（见 §6 风险）。

---

## 6. 验证与风险

- **先实测再信账**：`nvidia-smi` 逐步看各阶段占用、`nsys` 看两卡活跃度（理想 ≥90% 都有活）；
  `nvidia-smi topo -m` 确认 NVLink（否则退回 PCIe → 方案降级 FSDP）。
- **🔴 7B + fp32 Adam 必 OOM**（122GB）——8-bit Adam 前先确认数值可接受（对比 demo 健康信号
  `E[Δ_T]↑ + age>0`）。
- **🔴 dense 中间张量必爆**（10GB/批 + 队列）——真实词表强制 topk。
- **🔴 缓存常驻必爆**（L1 fat ×5 可达 79GB）——mmap + 切片是硬要求。
- **🟡 TP=2 learner 与并发 scorer 死锁**——护栏已挡（`tp_size=1`），走 5.1 不用碰。
- **激活账未标死**：随 batch/seq 变化，7B 梯度检查点后 <1GB 是保守估算，实测微调 batch。
- **正确性回归**：三档放大对照 v2 `E[Δ_T]` 单调上升 + `staleness age>0`，训练循环无教师前向。
