# 全栈 OPD 流水线 · 8×RTX4090 GPU 部署优化方案

> 配套仓库：`main/fullstack_opd_v2/`（已批量化重构的 demo 内核）
> 环境：单节点 8×RTX4090，统一环境 `requirements-unified.txt`（torch 2.9.1 + CUDA 12.8 + vLLM 0.16.0）
> 注：本方案针对 **RTX 4090**，与 `OPTIMIZATION_PLAN.md`（8×A100 版）并行策略不同——**4090 无 NVLink、仅 PCIe 4.0 互联、24GB/卡、无 fp8**，这是所有决策的根。

---

## 0. 硬件画像与硬约束（决策的根）

| 项 | RTX 4090（本机） | 影响 |
|---|---|---|
| 显存 | 24GB GDDR6X × 8 = 192GB 聚合 | 单卡放不下大 teacher；KV cache 是 rollout 显存主体 |
| 卡间互联 | **无 NVLink**，仅 PCIe 4.0 x16（~32 GB/s/向，经 PCIe 交换机共享） | **跨卡通信 ≈ A100 NVLink 的 1/15~1/20**，TP 每层的 all-reduce 会很痛 |
| BF16 算力 | ~160 TFLOPS/卡（tensor core，稠密），聚合 ~1.3 PFLOPS | 训练用 bf16，不要用 fp32 主权重 |
| 显存带宽 | 1008 GB/s/卡，聚合 ~8 TB/s | 计算密度足够，瓶颈多在通信与 Python 开销 |
| fp8 | **不支持**（fp8 是 H100/Blackwell） | 不要规划 fp8 量化；用 bf16 |
| 功耗/散热 | 8×450W = 3.6kW，消费卡无 ECC | 需供电与风道；ECC 缺失对研究实验可接受 |
| 驱动 | CUDA 12.8 → 驱动 ≥ **R570** | 部署前先 `nvidia-smi` 确认驱动版本 |

**第一动作（上线前必做）**：`nvidia-smi topo -m` 看 PCIe 拓扑，确认 8 卡是挂在同一个 PCIe 交换机（PIX）还是跨 CPU 插槽（NODE）。拓扑直接决定用 FSDP 还是 TP（见 L2）。同时跑一次 `nccl-tests` 测实际 all-reduce 带宽。

---

## 1. 优化总览（按 ROI 排序）

| 层 | 优化 | 预期收益 | 改动量 | 依赖 |
|---|---|---|---|---|
| **L4** | 教师缓存 dense→**稀疏 top-K** | 缓存体积 ↓1000×（V=128k→K=128），否则根本存不下 | 中 | 直接改 `cache.py` |
| **L2** | 并行策略：learner 用 **FSDP2**（非 TP） | PCIe 下比 TP 快数倍（通信相位 2 vs L） | 中 | 替换 WeightStore 广播 |
| **L3** | rollout 用 **vLLM 数据并行**（8×TP=1） | 解码期零跨卡通信，最适配 PCIe | 低 | 接 vLLM |
| **L1** | **bf16 + torch.compile + CUDA Graph** | 训练步耗时 ↓20~40%，小模型更明显 | 低 | 框架能力 |
| **L0** | ToyModel → **真实 LLM**（1B~7B student / 7B~34B teacher） | 从玩具到可发表实验 | 高 | 模型接入 |
| **L5** | 异步调度线程→**Ray/多进程 worker + NCCL 权重广播** | GPU 利用率↑，消除 GIL/Python 调度开销 | 高 | 架构改造 |
| **L6** | 陈旧 rollout **offload 到 CPU pinned + 异步 H2D** | 显存压力↓，陈旧样本不占卡 | 低 | 数据管线 |

**核心原则**：在 PCIe-only 的 4090 上，**把跨卡通信降到最低**＝FSDP（每步一次 all-reduce）而非 TP（每层一次）。rollout 用数据并行彻底避免解码期通信。

---

## 2. 逐层方案

### L0 · 模型规模化（从 ToyModel 到真实 LLM）
- **Student（Direct-OPD 被训练方）**：1B~7B 最舒适（FSDP 跨 8 卡，bf16 权重 14GB/8≈1.75GB/卡）；13B 可行。
- **Teacher（Lightning 离线、一次性）**：7B~34B 用 TP=8 舒适；70B 在 8×4090 上**边界**（140GB/8≈17.5GB/卡 + KV + 开销，需压 KV/seq 或开激活重算）。
- **Rollout 模型 = Student 自身**（AsyncOPD 是 on-policy，rollout 即 student 当前策略）。
- 迁移：用 `main/fullstack_opd_v2/model.py` 的接口形态作为 wrapper，内部换成 HuggingFace `AutoModelForCausalLM` + vLLM/Megatron 后端；causal mask 逻辑已由 v2 正确实现，真实模型自带因果注意力，无需重写。

### L1 · 精度与编译
- 全栈 **bf16**（4090 有 bf16 tensor core；不要用 fp8）。训练用 `torch.amp.autocast(bf16)` + `GradScaler` 不需要（bf16 无溢出风险）。
- 矩阵乘开 **TF32**（`torch.backends.cuda.matmul.allow_tf32=True`）提升吞吐。
- learner 前向+反向包进 **`torch.compile(mode="max-autotune", fullgraph=True)`**，再套 **CUDA Graph** 包住整个 train step（消除 Python + kernel launch 开销，小/中模型收益最大）。
- **梯度检查点 / activation recompute**：用显存换算力，让 7B 在 24GB 上宽松跑。

### L2 · 并行策略（PCIe 决策核心）
- **Learner：FSDP2（PyTorch 原生）/ ZeRO-3**（每步一次 all-gather 参数 + 一次 reduce-scatter 梯度），而非 Megatron TP。原因：TP 每层都 all-reduce，在 PCIe 上通信成本爆炸；FSDP 只在 step 边界通信，最适配 4090。
  - 单节点用 `HSDP`（只用 intra-node sharding 即可）。
  - 若 `topo -m` 显示跨 CPU 插槽（NODE），FSDP 的跨插槽通信更慢 → 考虑把 8 卡按插槽分组或接受一次通信。
- **Teacher 离线预计算（一次性）**：可短暂用 **Megatron TP=8** 把最大 teacher 塞进去（慢但只跑一次）。或直接 7B~34B TP=8（轻量）。
- **megatron-core（统一环境已含）保留作备选**：仅当 model 大到 FSDP 单步显存不够时切 TP；默认 FSDP。

### L3 · 推理引擎（vLLM 数据并行）
- **AsyncOPD 的 rollout 用 vLLM，8 卡数据并行（每卡一个 TP=1 实例）**，各吃一批 prompt，解码期**零跨卡通信**——这是 4090 上 rollout 的最优形态。
- 对应 AsyncOPD 原仓库的 `rollout_parallelism="data_parallel"` / fused-hybrid colocate 模式。
- KV cache 按 24GB 预算分配（每卡留 ~4GB 给 learner 共享），用 vLLM 的 `gpu_memory_utilization` 控。
- **colocated 部署**：同一张卡既跑 vLLM rollout shard 又跑 FSDP learner 切片（AsyncOPD fused-hybrid 即此），避免卡空闲。

### L4 · 离线教师缓存升级（最高价值的代码改动）
v2 的 `cache.py` 存的是稠密 `(N, T, V)`。真实词表 V=32k~150k → 体积爆炸，**不可能存下**。
**Direct-OPD 本就用 student top-K 做 softmax**（`delta_opd_reward_topk`），所以只需存**稀疏 top-K 教师 log 比**：
- 每个 (prompt, position) 存 `K` 个 `(token_id, logp_rl, logp_ref)`，K=64~256。
- 体积：稠密 `N·T·V·4B` → 稀疏 `N·T·K·(4+4+4+4)B`。V=128k、K=128 时 **↓约 1000×**。
- 存为 **mmap 连续数组**（落 NVMe，跨进程共享，不占 RAM）。
- 迁移：把 `cache.py::get_delta` 改为返回 `(idx[K], rl_logp[K], ref_logp[K])` 稀疏三元组；`direct_opd.py` 的 topk 奖励函数已经支持稀疏输入，几乎无缝对接。
- 这是「无 NVLink 下能跑真实词表」的成败手。

### L5 · 异步调度放大（线程→进程）
v2 用 `threading + Queue`，在 GPU 上**无法并行计算**（GIL + CUDA 串行）。放大做法：
- **Rollout worker = Ray actor / multiprocessing**，每 worker 持一个 vLLM 实例（DP），产出 rollout → offload token 到 CPU pinned / NVMe。
- **Learner = FSDP 进程组**，消费 rollout，按当前权重重算 ratio（保留 v2 的 staleness 双截断）。
- **权重同步 = NCCL broadcast**，learner rank0 → rollout worker 每 K 步广播一次（替换 v2 `WeightStore.load_state_dict` 逐样本加载 → `acquire_if_newer` 的 NCCL 版）。
- **陈旧丢弃**：age > threshold 的样本直接丢（v2 `StalenessQueue` 双侧截断已就绪），不进 GPU。

### L6 · 数据 / I-O
- 陈旧 rollout 张量 **offload 到 CPU pinned memory**，消费时 **异步 H2D**（`non_blocking=True` + 流同步），不在 GPU 上囤陈旧样本。
- 教师 Δ_T 缓存走 **mmap**，避免全量载入 RAM。
- 日志/指标用 **WandB / TensorBoard**（异步写盘），不阻塞训练循环。

---

## 3. 代码迁移映射（v2 模块 → 真实框架）

| v2 模块（已验证正确） | 保留 | 改造为 |
|---|---|---|
| `model.py`（causal mask / 批次前向） | ✅ 因果注意力逻辑 | 内部换 HF/vLLM/Megatron 模型；ToyModel 仅作单测 |
| `losses.py`（`policy_gradient_kl` π_old 加权 / `low_var_kl` k3） | ✅ **原样保留**（上轮审阅战果） | 直接复用，作用在真实分布上 |
| `cache.py`（设备常驻堆叠张量） | ✅ 索引/版本逻辑 | dense `(N,T,V)` → **稀疏 top-K mmap**（L4） |
| `buffer.py`（`WeightStore.acquire_if_newer`） | ✅ 版本推进才加载 | `load_state_dict` → **NCCL broadcast** |
| `scheduler.py`（四阶段异步 + staleness 双截断） | ✅ **异步结构 + 双截断原样保留** | 线程+Queue → **Ray/mp worker + NCCL 权重广播**（L5） |
| `pipeline.py`（toy 数据 / 奖励查找表 / 计时） | ✅ 计时与编排 | toy 数据 → 真实数据集；奖励查找表 → 真实 reward |

**不要动的内核**：因果 mask、π_old 加权 PG、k3 KL、staleness 双侧截断——这四项经上轮数学推导 + 回归验证正确，是 demo 的算法正确性保证。

---

## 4. 落地里程碑（建议顺序）

**P0（上线即做，低风险高收益）**
1. `nvidia-smi topo -m` + `nccl-tests` 测拓扑与通信带宽，定 FSDP vs TP。
2. 统一环境装好（`requirements-unified.txt`），`torch.cuda.is_available()` + bf16 冒烟测试。
3. L4：教师缓存改稀疏 top-K（否则真实词表直接 OOM）。
4. L1：bf16 + `torch.compile` + CUDA Graph 包 train step。
5. 小模型（1B student / 7B teacher）smoke test：跑通 Stage0→1→2，指标同 v2 收敛。

**P1（规模放大）**
6. L3：vLLM 数据并行 rollout 接入 AsyncOPD。
7. L2：learner 切 FSDP2，colocated 部署。
8. L5：Ray/多进程 worker + NCCL 权重广播，替换线程。
9. 监控：staleness 直方图 + 自适应 KL 系数 + WandB。

**P2（压榨）**
10. 7B~13B student 全量训练；teacher 上 34B（TP=8）。
11. L6：陈旧 rollout CPU offload + 异步 H2D。
12. 双缓冲（weight 广播与训练重叠）+ CUDA Graph 深调 + 必要时切 Megatron TP。

---

## 5. 风险与验证

- **先测后调**：上线先用 `torch.profiler` + `nvidia-smi dmon` 定位瓶颈（PCIe 通信？H2D？Python？），再针对性优化——不要盲调。
- **PCIe 是天花板**：若 `topo -m` 显示跨插槽 NODE，FSDP 跨插槽通信慢，考虑模型放单插槽 4 卡或接受一次通信。
- **24GB 显存**：7B student + KV 在 colocated 下需精算 `gpu_memory_utilization`；teacher 70B 边界，优先 13B~34B。
- **正确性不退化**：每步放大都对照 v2 的 `E[Δ_T]` 单调上升 + `staleness age>0` 两个信号回归，确保三重限制仍被打破。
- **驱动/电源**：确认 R570+ 驱动、3.6kW 供电与风道；消费卡无 ECC，定期 checkpoint 防单点故障。

---

## 6. 一句话总结

**4090 优化的命门是「压通信」**：learner 用 FSDP2（每步一次 all-reduce）、rollout 用 vLLM 数据并行（解码零通信）、教师缓存改稀疏 top-K（真实词表才存得下）；算法内核（因果 mask / π_old 加权 PG / k3 KL / staleness 双截断）原封不动，只把 ToyModel 与线程调度换成真实框架 + 进程级异步。
