# 全栈 OPD 流水线 · 2×RTX PRO 6000 部署优化方案（二次审阅修订版）

> 配套仓库：`main/fullstack_opd_v2/`（已批量化重构的 demo 内核）
> 环境：单节点 2×RTX PRO 6000 Blackwell，统一环境 `requirements-unified.txt`（torch 2.9.1 + CUDA 12.8 + vLLM 0.16.0）
> 注：本方案针对 **2×RTX PRO 6000**，与 `OPTIMIZATION_PLAN_8x4090.md`（8×4090 版）决策**整体相反**——PRO 6000 有 **NVLink、96GB/卡**，通信不再是瓶颈，瓶颈变成「只有 2 卡 / 每卡 96GB」。
>
> ⚠️ **二次审阅修订点（2026-08-06）**：初版有 3 处事实错误 + 2 处需验证项，已修正——(a) **FP8 并非 PRO 6000 独有**（RTX 4090/Ada 的 4 代 Tensor Core 也支持 FP8，本机独有是 FP4 + 更高带宽 + NVLink + 96GB + ECC）；(b) **70B「训练」在 2×96GB 上会 OOM**（优化器内存 840GB 远超 192GB 总量，TP/fp8 只省权重不省优化器）；(c) **colocated 不能同卡同时驻留**（需 CPU offload 换入换出）。详见文末「二次审阅结论」。

---

## 0. 硬件画像与硬约束（决策的根）

| 项 | RTX PRO 6000（本机，2 卡） | 对比 8×4090 |
|---|---|---|
| 显存 | **96GB GDDR7 + ECC** × 2 = 192GB 聚合（同总量，但集中到 2 卡） | 4090 是 24GB×8=192GB 但分散 |
| 卡间互联 | **NVLink 桥（可选配件！）**；带宽见下方注 | 4090 无 NVLink，仅 PCIe 4.0 ~32GB/s |
| BF16 算力 | ~250 TFLOPS/卡（稠密），2 卡 ~500 TFLOPS | 4090 ~160/卡，8 卡 ~1.3 PFLOPS（**原始算力 4090 高 ~2.6×**） |
| FP8 算力 | ~500 TFLOPS/卡（稠密） | **4090 (Ada) 也支持 FP8**（4 代 Tensor Core），非本机独有 |
| 显存带宽 | **1.8 TB/s/卡**，2 卡 3.6 TB/s | 4090 1.0 TB/s/卡 |
| 功耗 | 2×600W = 1.2kW，ECC 内存 | 4090 8×450W = 3.6kW，无 ECC |
| 驱动 | CUDA 12.8 → 驱动 ≥ R570（Blackwell 需新驱动） | 同 |

> **NVLink 带宽注（需验证）**：本机显存带宽为 1.8 TB/s（确定）。但 **NVLink 桥的实际互联带宽可能低于数据中心的 NVLink 5（1.8 TB/s）**——工作站双卡桥的 NVLink 带宽请以官方桥规格 / `nvidia-smi topo -m` + `nccl-tests` 实测为准。即便低于该值，NVLink 仍 **远快于 PCIe**，TP 依然可行，只是 TP vs FSDP 的性能差会缩小。**关键：NVLink 桥是单独选购/安装的配件**，未装则静默退回 PCIe 4.0（性能悬崖）。

**结论（与 4090 版相反）**：
- PRO 6000 **通信不再是瓶颈**（有 NVLink），所以 **TP（张量并行）可用且快**——这正好和 4090「用 FSDP 压通信」相反。
- 但**只有 2 卡**，并行度天花板低；**原始算力仅为 8×4090 的 ~38%**——小模型高吞吐场景不如 4090，但**单卡 96GB 能塞下大模型（推理）**是 4090 做不到的。
- **相对 4090 的真正差异化优势 = NVLink + 96GB/卡 + FP4 + ECC + 更高带宽**，不是 FP8（4090 也有 FP8）。

---

## 0.1 ★ 尺寸可行性（推理 vs 训练）—— 初版最大 OOM 漏洞的修正

直接训练（Direct-OPD learner）的内存 = **权重 + 梯度 + 优化器状态**，其中**优化器是天花板**，TP/fp8 只能压权重，压不了优化器。算笔账（bf16 权重 2B/param，Adam fp32 优化器 12B/param）：

| 模型 | 权重 bf16 | Adam 优化器(全量) | 2×96GB 能否**训练** | 2×96GB 能否**推理**(teacher/rollout) |
|---|---|---|---|---|
| 7B | 14GB | 84GB | ✅ 宽松（TP=2 每卡 ~42GB 优化器） | ✅ 单卡 |
| 13B | 26GB | 156GB | ⚠️ 紧（TP=2 每卡 78GB 优化器 + 权重 + 梯度 ≈ 91GB；需 **8-bit Adam + 梯度检查点**） | ✅ 单卡 / TP=2 |
| 34B | 68GB | 408GB | ❌ 优化器 408GB > 192GB 总量，**不可能** | ✅ TP=2（34GB/卡） |
| 70B | 140GB | 840GB | ❌ 优化器 840GB >> 192GB，**绝不可能**（TP/fp8 都救不了） | ⚠️ TP=2 70GB/卡（紧，KV 受限）；**fp8 35GB/卡 舒适** |

> **修正结论**：
> - **Learner（被训练的 student）上限 ≈ 13B**（8-bit Adam + 梯度检查点 + 可能 CPU offload），7B 宽松；**70B 训练在 2×96GB 上不可行**。
> - **Teacher（Lightning 离线，仅推理）/ Rollout（vLLM，仅推理）= 70B 可行**（TP=2；fp8 更舒适）。
> - 初版把「70B student（训练）」列为可行是 **OOM 错误**——只算了权重没算优化器。
> - fp8 **训练**经 Transformer Engine 仍保留 fp32 master 权重，**不减少优化器内存**，故无法让 70B 训练变可行。

---

## 1. 优化总览（按 ROI 排序）

| 层 | 优化 | 预期收益 | 改动量 |
|---|---|---|---|
| **L4** | 教师缓存 dense→**稀疏 top-K**（同 4090 版） | 真实词表才存得下，与硬件无关 | 中 |
| **L2** | 并行策略：**Megatron TP=2 + SP**（NVLink 加持） | 2 卡通信极快，TP 优于 FSDP 的复杂度 | 中 |
| **L3** | rollout：**vLLM TP=2 + FP8 量化** | 解码快、KV cache 减半、批更大（4090 也能 fp8，但本机 NVLink 无 PCIe 瓶颈） | 低 |
| **L1** | **bf16 训练 + 可选 fp8 训练（TE，不省优化器）+ torch.compile/Graph** | 吞吐↑ | 低~中 |
| **L0** | ToyModel→真实 LLM：**student 训练 7B~13B；teacher/rollout 推理 70B**（TP=2/fp8） | 区分推理/训练尺寸 | 高 |
| **L5** | 异步调度：2 worker + NCCL 权重广播（NVLink） | 2 卡下仍解耦 rollout/learner | 高 |
| **L6** | 陈旧 rollout CPU offload + 异步 H2D；colocated 用 CPU offload 换入换出 | 显存压力↓ | 低 |

**核心原则**：NVLink 让通信变便宜 → **用 TP=2 把模型纵向切开**（而非 FSDP 横向分片）；**FP8 留给 rollout 量化（KV 减半）**；96GB/卡 → **推理直接上大模型（70B）**；但**训练受优化器内存限制，learner 上限 ~13B**。

---

## 2. 逐层方案

### L0 · 模型规模化（区分推理与训练）
- **Student（Direct-OPD 被训练方）**：**训练**受优化器内存限制 → **7B 宽松 / 13B（8-bit Adam + 梯度检查点，紧）**；**70B 训练不可行**（见 0.1）。
- **Teacher（Lightning 离线、仅推理）**：**70B 用 TP=2（bf16 紧，fp8 舒适）**；比 4090 的 34B 推理上限高一截。
- **Rollout 模型 = Student 自身（仅推理，vLLM）**：尺寸同 teacher，可上 70B（TP=2 / fp8）。
- 迁移：同 4090 版，`model.py` 接口作 wrapper，内部换 HF/vLLM/Megatron；因果注意力逻辑已由 v2 正确实现，真实模型自带。

### L1 · 精度与编译
- 训练主用 **bf16**（`torch.amp.autocast(bf16)`）；可选 **fp8 训练**经 **Transformer Engine / Megatron 的 fp8 路径**（Blackwell 5 代 Tensor Core，FP4 也支持）。
  - ⚠️ **fp8 训练保留 fp32 master 权重，优化器状态不减少** → 不能用来让 70B 训练变可行；仅省激活/前向显存与加速计算。
  - 精度敏感处保留 bf16；fp8 训练需 TE 延迟缩放 / 混合精度监控。
- 矩阵乘开 **TF32**。
- learner 包进 **`torch.compile(mode="max-autotune")` + CUDA Graph** 包 train step（变长序列需固定 shape 或 `dynamic=False` 谨慎处理）。
- **梯度检查点** 让 13B 训练在 96GB 上宽松。

### L2 · 并行策略（NVLink 让 TP 复活）
- **Learner：Megatron TP=2 + Sequence Parallel（SP）**，2 卡经 NVLink 通信（带宽以实测为准，但 >> PCIe），每层 all-reduce 快——比 4090 的 FSDP 更简洁。
  - 统一环境已含 `megatron-core 0.16.1`，直接可用。
  - 若嫌 Megatron 重，**FSDP2 跨 2 卡（NVLink）** 也很快（每步一次通信），二选一；TP=2 是首选（2 卡天然 2 路）。
  - 优化器内存：用 **8-bit Adam / ZeRO-2 风格分片** 把 13B 训练压进 96GB（见 0.1）。
- **Pipeline Parallel 不必要**：只有 2 卡，PP=2 收益低且气泡大，不推荐。

### L3 · 推理引擎（vLLM + FP8）
- **AsyncOPD rollout 用 vLLM TP=2**（NVLink 加持，注意力 TP all-reduce 快），而非 4090 版的「8×TP=1 数据并行」（那是为规避 PCIe 通信，本机无需）。
- **FP8 权重量化**：vLLM 支持 fp8（W8A8）——**需预量化权重或动态量化路径**（非纯开箱；确认模型有 fp8 checkpoint 或走 online W8A8）。Blackwell fp8 tensor core 加速解码，**KV cache 与权重显存减半** → 更大 batch / 更长 seq。
  - 注：4090 (Ada) 硬件也支持 fp8 推理，本机相对 4090 的 fp8 优势是**更高 fp8 吞吐 + 无 PCIe 瓶颈 + 独有 FP4**，而非「能不能用 fp8」。
- **colocated 部署（必须 CPU offload 换入换出）**：2 卡同时承载 vLLM(rollout) + learner(训练) 时，**二者不能同卡同时驻留**——13B 训练优化器 ~78GB/卡 + 权重 + KV 已超 96GB。正确做法：rollout 阶段把 learner 优化器/权重 **offload 到 CPU**，rollout 结束再 reload 训练（AsyncOPD 的 fused-hybrid 即此模式），不是同时占满。

### L4 · 离线教师缓存升级（与硬件无关，仍 P0）
同 4090 版：v2 `cache.py` 的 dense `(N,T,V)` 在真实词表 V=32k~150k 下存不下 → 改存稀疏 **top-K (token_id, logp_rl, logp_ref)**，K=64~256，落 **mmap** 跨进程共享，体积 ↓约 1000×。`direct_opd.py::delta_opd_reward_topk` 已支持稀疏输入，无缝对接。
- PRO 6000 额外好处：96GB/卡可酌情存更密的 top-K 或更大的 (N,T)；但稀疏仍是默认最佳实践。

### L5 · 异步调度（2 worker + NCCL 广播）
v2 用 `threading + Queue`，GPU 上无法并行。2 卡下：
- **2 个 Ray actor / 进程**各持 vLLM(TP=2) 切片，产出 rollout → offload 到 CPU pinned / NVMe。
- Learner(TP=2) 消费，按当前权重重算 ratio（保留 v2 staleness 双截断）。
- **权重同步 = NCCL broadcast**（经 NVLink，极快），learner→rollout 每 K 步广播（`WeightStore.acquire_if_newer` 的 NCCL 版，替换 load_state_dict）。
- 陈旧样本 age>threshold 直接丢（v2 `StalenessQueue` 双侧截断就绪）。
- 2 卡下「解耦 rollout/learner 占用」的收益比 8 卡小，但 AsyncOPD 的 stale-recompute + 无 teacher 价值不变。

### L6 · 数据 / I-O
- 陈旧 rollout **offload CPU pinned + 异步 H2D**（`non_blocking` + 流同步）。
- **colocated 的 learner↔rollout 权重换入换出走 CPU offload**（L3 所述），避免同卡同时驻留 OOM。
- 教师 Δ_T 缓存走 **mmap**，不全量载入。
- 指标用 **WandB / TensorBoard** 异步写。

---

## 3. 代码迁移映射（同 4090 版）

| v2 模块（已验证正确） | 保留 | 改造为 |
|---|---|---|
| `model.py`（causal mask / 批次前向） | ✅ 因果注意力 | 内部换 HF/vLLM/Megatron |
| `losses.py`（π_old 加权 PG / k3 KL） | ✅ **原样保留** | 作用在真实分布 |
| `cache.py`（设备常驻堆叠张量） | ✅ 索引/版本 | dense→**稀疏 top-K mmap**（L4） |
| `buffer.py`（`WeightStore.acquire_if_newer`） | ✅ 版本推进才加载 | `load_state_dict`→**NCCL broadcast** |
| `scheduler.py`（四阶段异步 + staleness 双截断） | ✅ **异步结构 + 双截断保留** | 线程→**2 Ray worker + NCCL 广播** |
| `pipeline.py` | ✅ 计时/编排 | toy 数据→真实数据集 |

**不动的内核**：因果 mask / π_old 加权 PG / k3 KL / staleness 双截断。

---

## 4. 落地里程碑（已按尺寸可行性修正）

**P0（上线即做）**
1. `nvidia-smi topo -m` 确认 2 卡走 NVLink（应显示 `NV#`）——**先确认 NVLink 桥已物理安装**，否则是 PCIe；`nccl-tests` 实测带宽。
2. 装统一环境，`torch.cuda.is_available()` + bf16 + fp8(TE) 冒烟；确认驱动 R570+（Blackwell）。
3. L4：教师缓存改稀疏 top-K。
4. L1：bf16 + `torch.compile` + CUDA Graph。
5. 小模型（**7B student 训练 / 7B teacher 推理**）smoke test，指标同 v2 收敛（E[Δ_T]↑ + staleness age>0）。

**P1（规模放大）**
6. L3：vLLM TP=2 + **FP8 量化** rollout（确认 fp8 权重/动态量化路径）。
7. L2：learner 切 **Megatron TP=2 + SP**（NVLink）+ **8-bit Adam** 把训练压进 96GB。
8. L5：2 Ray worker + NCCL 权重广播；L6 colocated CPU offload 换入换出。
9. 监控：staleness 直方图 + 自适应 KL + WandB。

**P2（大模型推理 + 训练上限）**
10. **teacher/rollout 上 70B（TP=2 / fp8 推理）**；**student 训练上限 ~13B（8-bit Adam + 梯度检查点）**——**不承诺 70B 训练**。
11. L1 进阶：fp8 训练经 Transformer Engine（精度敏感层回退 bf16；明确不解决优化器 OOM）。
12. L6：双缓冲（权重广播与训练重叠）。

---

## 5. 风险与验证

- **先测拓扑 + 确认物理桥**：`nvidia-smi topo -m` 必须显示 NVLink 互联（PRO 6000 双卡桥，**需确认桥已安装**）；若显示 PIX/NODE 即走 PCIe，性能断崖。同时 `nccl-tests` 实测 NVLink 带宽（可能低于数据中心的 1.8 TB/s，以实测为准）。
- **驱动**：Blackwell 需 **R570+**（CUDA 12.8）；统一环境已匹配。
- **🔴 训练 OOM 是头号风险**：70B/34B **训练**在 2×96GB 不可行（优化器 >192GB 总量）；learner 上限 ~13B（8-bit Adam + 梯度检查点）。**初版「70B student 训练」是错误的**。
- **🔴 colocated 不能同卡同时驻留**：13B 训练优化器 ~78GB/卡 + 权重 + KV > 96GB，必须用 CPU offload 换入换出，而非同时占满。
- **FP8 精度**：rollout fp8 量化几乎无损；fp8 训练需监控精度（TE 延迟缩放 / 混合），且**不减少优化器内存**。
- **FP8 非本机独有**：4090 (Ada) 也支持 FP8 推理；本机独有优势是 NVLink + 96GB + FP4 + ECC + 更高带宽。
- **ECC 占用**：ECC 开启会占用少量可用显存（个位数 %），尺寸估算已留余量。
- **正确性不退化**：每步放大对照 v2 的 `E[Δ_T]` 单调上升 + `staleness age>0` 回归。
- **功耗友好**：1.2kW 远小于 4090 的 3.6kW，供电/风道压力小；ECC 内存对长训练更稳。

---

## 6. 与 8×4090 方案的关键差异（决策对照，已修正）

| 维度 | 8×RTX4090 | 2×RTX PRO 6000 |
|---|---|---|
| 卡间互联 | 无 NVLink，PCIe 4.0 | **NVLink（需确认桥已装；带宽以实测为准）** |
| 并行策略 | **FSDP2**（压通信） | **Megatron TP=2 + SP**（通信便宜） |
| rollout | vLLM **数据并行** 8×TP=1（规避 PCIe） | vLLM **TP=2 + FP8 量化** |
| FP8 | **也支持**（Ada 4 代 Tensor Core） | 支持（更高吞吐 + 独有 FP4） |
| 单卡显存 | 24GB | **96GB** |
| 最大 **推理**模型 | 34B（TP=8） | **70B（TP=2 / fp8）** |
| 最大 **训练**模型 | ~13B（8×4090，优化器分片） | **~13B（8-bit Adam，同上限；非 70B）** |
| 原始 BF16 算力 | ~1.3 PFLOPS（高） | ~500 TFLOPS（较低） |
| 适合场景 | 小模型高吞吐、卡多扩展 | **大模型推理 + 单卡大显存 + 低功耗** |
| 功耗 | 3.6kW | **1.2kW** |

---

## 7. 一句话总结（修订）

**PRO 6000 优化的命门 =「用 NVLink 做 TP + 吃满 96GB 大模型推理 + FP8 减半 KV + CPU offload 换入换出」**：rollout/teacher 用 vLLM/Megatron TP=2+fp8 直接上 **70B 推理**；learner 受优化器内存限制**训练上限 ~13B**（8-bit Adam + 梯度检查点）；colocated 必须 CPU offload 换入换出而非同卡同时驻留。算法内核原封不动，只换模型与调度底座。相对 4090，本机**算力更低但单卡显存/通信带宽/NVLink 碾压**，是「大模型推理 + 低功耗」路线——但**FP8 并非独有、70B 训练不可行**这两点初版有误，已修正。

---

## 附录：二次审阅结论（2026-08-06）

初版 3 处错误 + 2 处需验证项，均已修订：

1. **🔴 FP8 非 PRO 6000 独有**（事实错误）：RTX 4090/Ada 的 4 代 Tensor Core 已支持 FP8（vLLM fp8 W8A8 可跑）。本机独有 = NVLink + 96GB + FP4 + ECC + 更高带宽。全文中「FP8 是 PRO 6000 独有红利 / 4090 用不了」均已改正。
2. **🔴 70B 训练 OOM**（最大 OOM 漏洞）：训练内存 = 权重+梯度+优化器，70B Adam 优化器 840GB >> 2×96GB=192GB 总量，TP/fp8 只省权重不省优化器 → **70B 训练不可行**。初版把「70B student（训练）」列为可行是错误的。修正：learner 上限 ~13B；70B 仅推理（teacher/rollout）。
3. **🔴 colocated 同卡同时驻留 OOM**（架构错误）：13B 训练优化器 78GB/卡 + 权重 + KV > 96GB，vLLM 与 learner 不能同时占满 2 卡；必须 CPU offload 换入换出（fused-hybrid 模式）。已注明。
4. **🟡 NVLink 带宽 1.8 TB/s 可能混淆**（需验证）：显存带宽 1.8 TB/s 确定；NVLink 桥实际带宽可能低于数据中心 NVLink 5 的 1.8 TB/s，以实测为准（仍 >> PCIe）。且 **NVLink 桥是可选配件**，未装则静默退回 PCIe → 性能悬崖。已加验证步骤。
5. **🟡 fp8 训练不省优化器**（兼容性/精度）：TE fp8 保留 fp32 master，优化器内存不变，不能让 70B 训练变可行；仅加速计算+省激活。已注明。
6. **🟡 连带提醒**：8×4090 版（`OPTIMIZATION_PLAN_8x4090.md`）同样高估了训练尺寸（其「34B teacher」应明确为推理上限，34B 训练在 8×4090 也因优化器 >192GB 不可行），建议同步按 0.1 的尺寸账修订。
