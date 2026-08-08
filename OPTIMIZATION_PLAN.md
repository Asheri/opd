# 全栈 OPD 流水线 · 8×A100 部署优化方案

> 目标：把 `main/` 的 v2 demo（已验证正确的算法内核：causal mask、π_old 加权 PG、k3 KL、staleness 双截断）搬到 **单节点 8×A100** 真实规模执行，并最大化吞吐与稳定性。
> 算法正确性不动，只优化「执行底座」。方案按优先级标注 P0（必做·低风险）/ P1（架构级）/ P2（进阶）。

---

## 0. 关键事实（来自三份真实仓库的 8 卡配置）

| 来源 | 真实拓扑 | 关键参数 |
|---|---|---|
| Lightning-OPD (8B) | 8 卡全给 actor，`rollout-num-gpus 0`（colocated） | `max-tokens-per-gpu 16384`，Megatron-core 后端 |
| AsyncOPD | FSDP actor + 每 rank colocate 一个 TP=1 vLLM | `rollout_parallelism="data_parallel"`，fused-hybrid |
| Direct-OPD (verl) | FSDP/FSDP2 或 Megatron，Ray 编排 | `verl/trainer/ppo/ray_trainer.py` |
| 统一环境（requirements-unified） | torch 2.9.1 / CUDA 12.8 / vLLM 0.16 / megatron-core 0.16.1 / flash-attn 2.8.3 / ray 2.54.1 | — |

**A100 精度结论**：支持 **bf16**（~312 TFLOPS），**不支持 fp8**（那是 H100）。所以全栈走 bf16 + AMP，matmul 可开 tf32。

---

## 1. 部署拓扑（P0，决定一切）

**推荐：colocated FSDP(v2/FSDP2) actor + vLLM data-parallel rollout，8 卡同节点。**
（不要 rollout/learner 分卡池——那是 8B 规模下的浪费，Lightning/Async 的真实配置都选 colocated。）

```
┌──────────────── 单节点 8×A100 (NVLink ~600GB/s) ────────────────┐
│  GPU0..7 : FSDP actor shard_i  ─┐                               │
│           + colocated vLLM DP   │  AsyncScheduler 解耦          │
│             (TP=1 副本)          ┘  rollout-gen ⇄ learner-step  │
│                                 ┌→ NCCL broadcast weights ──┐  │
└─────────────────────────────────┴──────────────────────────┴──┘
        Lightning 离线 Δ_T 缓存（device-pinned fp16 张量）预加载
```

- **同节点 NVLink** → TP/FSDP all-gather 走高带宽，权重同步用 NCCL broadcast 而非文件/state_dict 序列化。
- **8B bf16 权重 ≈ 16GB**，TP=8 或 FSDP 分片后每卡 ~2GB，余量巨大，可放 KV cache + 优化器状态（AdamW ~32GB bf16 分片后 ~4GB/卡）。

---

## 2. 计算 / 精度优化（P0）

| 项 | 做法 | 备注 |
|---|---|---|
| 精度 | `torch.cuda.amp` + `dtype=bf16`，matmul 开 tf32 | A100 不支持 fp8 |
| 注意力 | FlashAttention-2（`flash-attn==2.8.3` 已 pin） | 显存/速度双收益 |
| 增量解码 | vLLM **PagedAttention** KV cache（替换 naive 逐 token 生成） | rollout 阶段核心加速 |
| 编译 | `torch.compile` 包住 learner 的 forward/backward | 消 Python 开销（A100 稳定） |
| 梯度检查点 | `torch.utils.checkpoint` 包 transformer 层 | 换算力换 batch size |
| RMSNorm/fused | 用 Megatron 的 fused RMSNorm + 融合交叉熵 | 省 HBM 往返 |

> demo 里 `ToyModel.generate` 的逐 token 循环 → 上线后由 **vLLM generate** 整体接管，本项自动获得。

---

## 3. 并行与权重同步（P1）

- **并行策略二选一**（统一环境都支持）：
  - **FSDP2（verl 默认）**：实现简单，8B 轻松塞下，推荐起步。
  - **Megatron TP=8 + 可选 PP**：吞吐略高、显存更省，但配置复杂（Lightning 默认走这个）。
- **权重同步（替代 demo 的 `WeightStore.load_state_dict`）**：
  - learner 更新后，通过 **NCCL broadcast** 把新 state_dict 推到所有 colocated vLLM 副本（vLLM `update_weights` / sleep-wake API）。
  - 把 demo `WeightStore.publish` 的 pickle→广播改为 **直接 in-place 张量 broadcast**；`acquire_if_newer`（v2 已加）保留作版本闸门。
- **梯度累积 micro-batch**：设大 global batch（如 256），按 `micro_batch=32` × `grad_accum=8` 摊到 8 卡，提升数据并行效率。

---

## 4. AsyncOPD 异步专项（P1，本项目的灵魂）

| 项 | 做法 | 对应 demo 现状 |
|---|---|---|
| Off-policy IS 重算 | learner 时刻用**当前** student 重算 ratio（v2 `policy_gradient_kl` 已是 π_old 加权） | ✅ 已正确 |
| Staleness 双截断 | 入队 + 消费双侧截断 + **记录 staleness 直方图** | ✅ 逻辑在，缺监控 |
| 双缓冲预取 | 2× replay buffer，learner 训练一个、rollout 填充另一个，重叠 H2D | ❌ demo 单缓冲 |
| 自适应 KL 闸门 | `kl_coef` 随实测 KL 升降（高 KL → 提 coef / 降 lr），替代固定 0.05 | ❌ 固定值 |
| 权重延迟容忍 | staleness_threshold 按 step 时间标定（而非固定 4），避免 GPU 空等 | ⚠️ 固定值 |

> 异步语义已在 v2 验证（staleness age=5，同步等待已破）；上线只需补**监控 + 自适应**。

---

## 5. Lightning-OPD 离线专项（P0，只跑一次）

预计算 teacher log-prob 是**一次性**成本，之后 teacher 完全下线：

1. **TP/vLLM 并行预计算**：用 vLLM `tensor_parallel_size=8` 或 Megatron TP 批量算 SFT rollout 的 log-prob（demo `stage1_build_cache` 的逐样本循环 → 整批 vLLM）。
2. **Δ_T 缓存 fp16**：`rl - ref` 存 bf16/fp16（而非 fp32），缓存体积减半；device-pinned memory + 异步 H2D。
3. **教师一致性强校验**：SFT 与 OPD 必须同一 teacher（demo `TeacherConsistencyError` 已抛）；上线加**权重哈希比对**防误用。
4. **缓存分片落盘**：N 大时按 prompt 分片 `torch.save(shard)`，避免单文件 100GB+。

---

## 6. IO / 数据（P1）

- **流式数据加载**：真实 RLHF（math/web）用 `IterableDataset` 流式喂，避免全量驻留。
- **device-resident 默认**：v2 缓存已是设备常驻 (N,T,V) 张量零拷贝索引 → 保留并扩到真实 vocab（150k）。
- **collate 预填充**：prompt/response 在 CPU 端 pad 到统一长度再整批 H2D，减少小传输。

---

## 7. 数值 / 稳定性（P0）

- **梯度裁剪**（v2 已加 `clip_grad_norm_`）保留，阈值随 global batch 调。
- **AdamW + bf16 主权重**：Megatron/FSDP 默认；主权重 fp32 防崩。
- **EMA student**：对 student 做指数滑动平均作更稳的 target / 评估，抑制异步抖动。
- **Δ_T 裁剪**：teacher log-prob 差超过 ±K 的 token 截断，防极端值。

---

## 8. 监控（P1）

- **WandB/TensorBoard** 记录：`loss / pg_loss / kl_loss / E[Δ_T] / staleness_age 直方图 / gpu_util / rollout_latency / weight_sync_ms`。
- **三重限制仪表盘**：常驻教师(teacher 前向次数=0) / 同步等待(learner-rollout 重叠率) / 迁移终态(E[Δ_T] 单调上升) 各自一条曲线，直接对应论文 claim。
- **告警**：staleness age 持续触顶（说明 rollout 跟不上）→ 扩 rollout 并发或降 learner 步频。

---

## 9. 渐进上线 & 消融（P0 流程）

1. **Smoke（1 GPU）**：小模型 + 小数据，验证端到端不崩、E[Δ_T] 上升。
2. **Scaling（1→8 GPU）**：测 FSDP all-gather 带宽、线性度（理想 ~7x 提速）。
3. **Ablation**：
   - 有/无 learner-time ratio 重算（验证 staleness 鲁棒性）。
   - staleness_threshold = 0（退化同步）vs 4 vs 8（验证异步收益）。
   - Δ_T fp16 vs fp32（验证缓存压缩无损）。
4. **全量 run**：8×A100，按 §1 拓扑，开 WandB。

---

## 10. 与 v2 demo 的衔接（哪些已就绪 / 待替换）

| 模块 | v2 demo 状态 | 上线动作 |
|---|---|---|
| causal mask / π_old 加权 PG / k3 KL | ✅ 正确 | **原样保留** |
| StalenessQueue 双截断 | ✅ 逻辑在 | 加直方图 + 自适应阈值 |
| 设备常驻张量缓存 | ✅ (N,T,V) | 扩 vocab + fp16 + 分片 |
| WeightStore | ✅ acquire_if_newer | load_state_dict → NCCL broadcast |
| 逐样本前向/生成 | ❌ 已批量化 | ToyModel → 真实模型(vLLM/Megatron) |
| 四阶段异步结构 | ✅ 队列传 batch | Ray 多进程 rollout worker |

---

## 优先级速查

- **P0（上线前必做）**：§1 拓扑定稿 · §2 bf16+FA2+vLLM · §5 离线预计算 fp16 · §7 数值稳定 · §9 smoke test
- **P1（首轮全量前）**：§3 NCCL 权重同步 · §4 监控+自适应 KL · §6 流式数据 · §8 WandB
- **P2（调优期）**：§3 Megatron TP 切换 · §4 双缓冲 · torch.compile 深度调参

---

## 预期收益（相对 v2 CPU demo）

- **rollout**：vLLM PagedAttention + A100 bf16 → 比 naive 生成快 **1–2 个数量级**。
- **learner**：FSDP + FA2 + bf16 → 8B 模型单步 ~秒级，8 卡接近线性加速。
- **异步**：rollout-gen 与 learner-step 重叠 → 消除 v1「同步等待」，GPU 利用率从 ~40% 提至 ~85%+。
- **离线**：teacher 完全下线后训练循环零 teacher 前向（常驻教师已破）。

> 注：v2 在 CPU 上 stage2 已实现 8.04x 吞吐（批量化摊掉 Python/权重加载开销）；到 GPU 后瓶颈转为显存带宽与跨卡通信，上述 §2–§3 即针对这些。
