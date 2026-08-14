# Adaptive Staleness-Aware Teacher Cache 设计规格

> 日期：2026-08-14
> 范围：1.7B 学生 Skywork 重训练，双卡 FSDP + colocated 交替相位 + L2 周期刷新
> 状态：待实现（本规格经多轮 brainstorming 确认）

## 0. 背景与目标

将 L2 周期刷新机制升级为 **Adaptive Staleness-Aware Teacher Cache**，闭环：

```
Selective Rollout ──► Teacher-Student Disagreement ──► Cache Quality
      ▲                                                   │
      │                                                   ▼
  Next Rollout ◄── Dynamic Refresh Ratio ◄── (mix_ratio / 触发时机)
```

四个能力：
1. **Teacher-Student Disagreement**（§3，本规格核心，按独立提示词的权威定义）
2. **Cache Health Monitor**（§4）
3. **Dynamic Refresh Ratio**（§5）
4. **Selective Rollout**（§6）

### 总体要求（硬性）

1. 不破坏现有 Base Pool 静态 anchor 语义（KL 锚点同源不变式）。
2. `_train_step` 继续 teacher-free，teacher 信号优先走 cache lookup。
3. 所有新增功能默认可配置、可关闭，不强耦合训练逻辑。
4. 兼容分布式训练（双卡 FSDP）。
5. 无隐式数据 leakage（disagreement 用当前 batch 缓存 Δ_T + 当前 π_cur，不引未来/测试数据）。
6. 正确处理 variable-length response、padding、EOS、token mask。
7. 记录 generation version / checkpoint step，使 sample staleness 可计算。
8. 新增指标经现有 `MetricsRecorder`（CSV/WandB）落盘，不造新 logger。
9. 最小侵入式修改，不大规模重构。
10. 不为创新堆算法；第一版优先：可解释、可监控、compute overhead 可控、可 ablation、不破坏训练。

---

## 1. 整体架构（colocated 交替相位 + FSDP + L2）

**否决单卡同步数据并行**。双卡分时交替：

```
┌─ 训练相位（T_train 步）─────────────────────────────────┐
│  FSDP all-gather student 权重 -> GPU                     │
│  rollout/vLLM 引擎 + teacher -> CPU offload              │
│  双卡 FSDP 数据并行：teacher-forcing 算 s_old/s_cur      │
│  -> PG+KL 反向 -> reduce-scatter 梯度 -> 优化器 step     │
│  _train_step 内核【一行不动】，纯查表，无 teacher 前向    │
│  训练完 -> student 权重 CPUOffload                        │
└────────────────────────┬────────────────────────────────┘
                         ▼ 相位切换（CPU offload 换入换出）
┌─ rollout 相位（L2 刷新）────────────────────────────────┐
│  student 权重搬回 GPU                                    │
│  Selective Rollout：按难度采 M_refresh prompt            │
│  student 自回归生成 response（8192 max）+ 收集 logp      │
│  teacher_rl/ref 前向算 top-k + chosen-token logp        │
│  初始 student（ref）前向算 student_ref logp              │
│  算 disagreement（§3）-> append 进 ring buffer           │
│  混合 feeder：训练按 mix_ratio 采 {base + refresh}       │
│  rollout 完 -> student offload，切回训练相位             │
└────────────────────────┬────────────────────────────────┘
                         ▼ 循环 ~10 轮（1000 步）
```

**FSDP 细节**：`FullyShardedDataParallel` + `ShardingStrategy.FULL_SHARD`，训练相位分片权重常驻 GPU（不用每步 `CPUOffload`），rollout 相位 `summon_full_params`/手动 `.cpu()` offload 腾 GPU。`_train_step` 调用点不变，FSDP 自动 all-gather。

**算法性质**：L0/L1（离线固定 D）-> L2（半在线，周期注入 on-policy 样本缓解曝光偏差），对齐 AsyncOPD fused-hybrid。

---

## 2. Ring Buffer 工程优化

### 双池结构（cache 扩展，最小侵入）
- **Base Pool**：初始 50K 离线预计算，静态不变（保 L1 锚点 + KL 同源不变式）。
- **Refresh Pool**：动态 ring buffer，`cache.append_refresh(...)` 进新样本，满后淘汰。

### Q1 何时刷新（Cache Health 信号驱动，与 §5 的 α 控制分离）
- 自适应触发：`E[Δ_T]` 滑动斜率 < ε（收敛放缓）或 `refresh_age_p95` 超 §4 阈值，且过 `min_interval`；或过 `max_interval` 兜底。

### Q2 刷新哪些（Selective Rollout，详见 §6）
- 难度加权温度采样 `p(i) ∝ exp(d(i)/τ)`，`d(i)` 来自 disagreement EMA。

### Q3 保留哪些（淘汰策略）
- 主策略 FIFO（按入队序淘汰最旧，保新鲜度）。
- 价值保护：`disagreement_abs` 高于全局 90 分位的样本免淘汰一轮。
- O(1) 淘汰不破坏。

### Q4 训练时信哪些（信任度，最关键）
- **算法层（已有，零改动）**：`pg_loss` 按 π_old 重要性采样 `ratio=π_cur/π_old` + `clip_eps=0.2`。refresh 样本 on-policy（ratio≈1，信任度高）；base 样本 off-policy（ratio 偏离 1，被 clip 自动降权）。
- **feeder 层**：`mix_ratio` 控两池采样比例，渐进提高。
- **关键 bug 修正**：现有 `_train_step:291` staleness 截断 `current_version - ver > threshold` 对 base 样本是隐患（base ver=0，训练后期全被截断丢弃）。修正：**base 样本不参与版本号截断**（其 staleness 是离线固有，靠 ratio+clip 降权）；**仅 refresh 样本**带版本号、受 threshold 截断。

---

## 3. Teacher-Student Disagreement（权威定义）

> 本节为独立提示词的权威规格，覆盖原设计中能力2的粗略定义。

### 3.1 数学定义

teacher 已有（缓存）：
```
Δ_T^{(t)} = logπ_T^RL(y_t|x,y_<t) − logπ_T^Ref(y_t|x,y_<t)
```

student 在 rollout 阶段计算：
```
Δ_S^{(t)} = logπ_S(y_t|x,y_<t) − logπ_Ref(y_t|x,y_<t)
```

若 teacher Ref 与 student reference 完全同源，token-level：
```
D_t = Δ_T^{(t)} − Δ_S^{(t)}
```

**同源性检查结果（必须先验证）**：teacher_ref = R1-Distill-Qwen-1.5B，student ref = 初始 Qwen3-1.7B，**不同源**。故 `D_t ≠ Δ_T−Δ_S` 简化式，须分别存 4 个 logp：
```
D_t = [logπ_T^RL(y_t) − logπ_T^Ref(y_t)] − [logπ_S(y_t) − logπ_Ref(y_t)]
```

response-level（**优先用 abs**）：
```
D_i^abs  = Σ_t m_{i,t}|D_{i,t}| / Σ_t m_{i,t}
D_i^mean = Σ_t m_{i,t} D_{i,t}  / Σ_t m_{i,t}
```
- `m_{i,t}` 有效 token mask；padding、EOS 之后不参与。
- response length 可变，**不除固定 8192**。

### 3.2 数据结构扩展（不破坏原有 rl_k/ref_k）

每条 refresh sample 携带：
```
sample_id, prompt_id, generation_step, response_length, token_mask,
rl_k, ref_k,                         # 原 top-k（训练查表用，不动）
student_logp,                        # logπ_S(y_t) per token（chosen）
student_ref_logp,                    # logπ_Ref(y_t) per token（chosen，初始 student）
teacher_rl_chosen_logp,              # logπ_T^RL(y_t) per token（chosen，新增）
teacher_ref_chosen_logp,             # logπ_T^Ref(y_t) per token（chosen，新增）
disagreement,                        # D_i^mean
disagreement_abs,                    # D_i^abs（主指标）
```

**去重检查**：若 student_ref 与已有 reference（KL 锚点 ref_ids/ref_logp）完全相同可省 student_ref_logp。但 KL 锚点是初始 student 的 **top-k**，refresh response 的 chosen token 不一定在锚点 top-k 内 -> **不能从锚点恢复 chosen-token ref logp**，须独立存储（或 rollout 阶段初始 student 前向现场 gather）。

### 3.3 rollout 阶段计算流程

1. student 生成 response y（autoregressive），收集 `logπ_S(y_t)`（generate 时 logits log_softmax + gather chosen）。
2. 初始 student（ref，P1-4 保留的独立实例）对 (prompt, y) 前向，gather `logπ_Ref(y_t)`。
3. teacher_rl / teacher_ref 对 (prompt, y) 前向：
   - top-k 存 cache（rl_k/ref_k，训练查表）；
   - gather chosen-token `logπ_T^RL(y_t)` / `logπ_T^Ref(y_t)`（**新增**，不能只存 top-k）。
4. 对有效 token 算 `D_t`，mask 加权聚合成 `D_i^abs` / `D_i^mean`。
5. disagreement + metadata 一起 `append_refresh` 进 ring buffer。

**`_train_step` 不重新 forward teacher**（teacher-free 保持）。

### 3.4 Numerical Stability 检查项

- log probability 是否来自同一 token shift（teacher/student 的 logits 对齐同一 response token）。
- student 与 teacher 针对完全相同 response token 对齐。
- padding token 不进入 disagreement（mask 置 0）。
- EOS token 计入有效 token（真实生成，mask=1）；EOS 之后的 padding 不计入（mask=0）。第一版 refresh 变长 response 必须传真实 mask，不再用 `_train_step` 的全 1 快路径。
- **top-k teacher representation 与 student exact chosen-token logp 的可比较性**：teacher cache 只存 top-k，student 生成 token 不一定在 top-k。
  - **必须**：rollout 阶段 teacher 前向时 gather chosen-token logp 独立存储。
  - **禁止**：把"未进入 top-k"当成概率为 0。

### 3.5 Sample Utility

```
U_i = λ_D · D_i + λ_R · R̂_i − λ_A · A_i
```
- `D_i`：disagreement（用 D_i^abs）
- `R̂_i`：归一化 reward = per-sample `E[Δ_T]`（`Σ_t m_t Δ_T(i,t) / Σ_t m_t`）经跨样本 z-score 归一化
- `A_i`：sample age（当前 version − generation_step）

第一版默认（全部可配置）：
```yaml
utility:
  disagreement_weight: 0.5
  reward_weight: 0.3
  age_penalty: 0.2
```

### 3.6 与现有 loss 的关系（重要）

**Disagreement 是辅助决策信号，不直接改核心训练 loss。** 第一阶段只用于：
1. monitoring
2. sample ranking（selective rollout）
3. cache eviction（价值保护）
4. dynamic refresh ratio 输入

为后续"disagreement 进 loss"实验保留接口，但未经验证不启用。

---

## 4. Cache Health Monitor（权威定义）

> 目标不是加日志，而是构建能解释训练异常的数据质量与 cache 状态监控系统。
> 经现有 `MetricsRecorder`（CSV/WandB）落盘，不造新 logger、不增 dashboard 依赖。
> **Health Monitor 只 Observe→Diagnose，不自动改训练**（避免难 debug 闭环；Dynamic Refresh Ratio 是独立消费方，非 Monitor 内部闭环）。

### 4.1 七类监控维度（Base/Refresh 分开统计）

**A. Freshness**：`Age_i = t − v_i`（t=当前 training step，v_i=generation step）。记录 `age/{mean,std,p50,p90,p95,max,over_threshold_ratio}`。
> 口径区分：Age 用 **step**（数据新鲜度，本节）；`_train_step` 截断用 **version**（权重陈旧度，§2 Q4）。sample 同时记 `generation_step` 与 `generation_version`（req7）。

**B. Pool Composition**：`r_B=N_B/N`、`r_R=N_R/N` + 实际 batch 采样比例 `pool/{base,refresh}_ratio_{requested,actual}`（发现 cache shortage / filtering / sampler bug）。

**C. Cache Lookup Health**：`lookup/{total,hit,miss,hit_rate,invalid,duplicate}`。miss 记有限量 debug 信息（不无限打印）。

**D. Sample Reuse**：`R_i`=sample i 被训练读取次数。记录 `reuse/{mean,p50,p95,max,high_ratio}`（发现少量样本过度 oversampling）。

**E. Teacher-Student Disagreement**：接入 §3，`disagreement/{mean,p95,high_ratio}`，Base/Refresh 分开。

**F. Reward / Quality**：`reward/{mean,std,p50,p95,high_ratio}` + `ΔR = E[R_refresh] − E[R_base]`。

**G. Response Length**：`length/{mean,std,p50,p95,max,eos_rate,max_length_ratio}`。特别监控 `P(L=8192)`（揭示 EOS 学坏 / reasoning loop / reward length bias / rollout straggler）。

### 4.2 Coverage

`coverage/{unique_prompt_ratio, unique_response_ratio, category_entropy, repeated_prompt_ratio}`。无 prompt category 不硬编码，先实现 unique ratio + 重复率。

### 4.3 Cache Health Score（rule-based，阈值 configurable）

独立 health evaluator：`H_t = f(HitRate, Freshness, Coverage, Disagreement, Reuse, Length)`。第一版**不用 ML**，基于阈值的 `HEALTHY / WARNING / CRITICAL`：
```yaml
health:
  hit_rate:        {warning: 0.995, critical: 0.98}
  refresh_age_p95: {warning: 5,     critical: 10}
  reuse_p95:       {warning: 8,     critical: 20}
  max_length_ratio:{warning: 0.10,  critical: 0.25}
```

### 4.4 Alert 机制

不刷屏：同一 warning 用 **cooldown**。记录 `cache_health/status`（HEALTHY/WARNING/CRITICAL）+ `cache_health/reason`（如 `refresh_age_p95 too high`）。

### 4.5 Dashboard（现有 WandB/CSV）

时间序列：refresh ratio / refresh age / disagreement / reward / hit rate / response length / cache health score。分布直方图：age / disagreement / reward / reuse / response length（采样统计）。

### 4.6 Performance Requirement（不许降吞吐）

- batch-level aggregation，**不逐 token loop**，**不全量扫描 50K Base Pool**。
- 用 counters / EMA / reservoir statistics；histogram 只采样统计。
- `health_monitor: false` 时性能不应明显退化。

---

## 5. Dynamic Refresh Ratio（权威定义）

> 本节管 α（训练 batch 中 refresh 占比，**连续调整**）；**刷新触发时机**（何时进 rollout 相位生成新样本，**离散事件**）见 §2 Q1。两者是不同控制。

feeder：`P_train = (1−α)P_base + α·P_refresh`，α 从固定 0.3 升级为动态。

### 5.1 设计目标（三信号）
- Base 太 stale（`Age_base↑`）-> α↑
- Student 变化太快（`Drift_t↑`）-> α↓（防 on-policy 样本过快主导）
- Refresh 质量高（`Quality_refresh↑`）-> α↑

### 5.2 输入信号
- `A_B = E[Age_base]`、`A_R = E[Age_refresh]`（来自 §4 Health Monitor）
- policy drift：`D_t^policy ≈ E[logπ_θt(y) − logπ_θ(t-1)(y)]`，**优先复用已有 k3 KL / log ratio 信号，不重复 forward**
- refresh quality：`Q_t = E[U_i | i∈Refresh]`，`U_i` 来自 §3.5 disagreement+reward utility

### 5.3 Dynamic controller
```
α_t = clip(α_0 + λ_A·Ã_B − λ_D·D̃_t^policy + λ_Q·Q̃_t, α_min, α_max)
```
所有信号 normalize；EMA（`EMA_t = β·EMA_{t-1} + (1−β)·x_t`）防高频震荡。

### 5.4 Safety
- `α_min ≤ α_t ≤ α_max`，且 **`α_max < 1`**（保留 Base 作 stationary anchor，禁止 refresh 100% 占据）。
- `max_step_change`：`|α_t − α_{t-1}|` 限幅，防 feeder ratio 突变。
- `warmup_steps`：前 N 步用 α_0 不动态调整。

### 5.5 Cold Start
Refresh Pool 不足（`N_R < N_required`）时：`α_actual = min(α_t, N_R/N_batch)`，自动 fallback Base。

### 5.6 与 Health Monitor 关系（职责分离）
第一阶段：Health Monitor **observe only**；Dynamic Ratio Controller **consume selected metrics**。不让 Monitor 自己改 ratio。未来再考虑 Health->Controller->Ratio 闭环。

### 5.7 Logging
`refresh_ratio/{requested,actual,base,reason}` + `controller/{base_age,refresh_quality,policy_drift,raw_alpha,clipped_alpha}`。**特别记录 ratio 调整原因**（如 `alpha decreased: policy_drift > threshold`）。

### 5.8 Ablation hooks（三模式共享同一 feeder）
- `fixed`：α=0.3
- `linear`：0.1→0.5 线性
- `adaptive`：完整 controller

---

## 6. Selective Rollout

- 每样本难度分 `d(i) = EMA(D_i^abs)`（来自缓存 disagreement）。
- 刷新时温度采样 `p(i) ∝ exp(d(i)/τ)`。
- `selective_rollout: false` 退回随机。

---

## 7. 配置（新增 `L2Cfg`，默认全关）

```yaml
l2:
  enabled: false
  t_train: 100
  m_refresh: 1000
  ring_buffer_capacity: 5000
  refresh:
    ratio:
      mode: adaptive          # fixed | linear | adaptive（§5.8 ablation）
      initial: 0.30
      min: 0.10
      max: 0.60               # <1，保留 base anchor（§5.4）
      age_weight: 0.25
      drift_weight: 0.50
      quality_weight: 0.25
      ema_beta: 0.9
      warmup_steps: 500
      max_step_change: 0.05
  selective_rollout: true
  selection_temperature: 1.0
  health_monitor: true
  health:                     # §4.3 rule-based 阈值（configurable）
    hit_rate:         {warning: 0.995, critical: 0.98}
    refresh_age_p95:  {warning: 5,     critical: 10}
    reuse_p95:        {warning: 8,     critical: 20}
    max_length_ratio: {warning: 0.10,  critical: 0.25}
  alert_cooldown: 50          # §4.4 同一 warning 冷却步数
  refresh_min_interval: 50
  refresh_max_interval: 150
  delta_slope_eps: 0.001
  value_protect_quantile: 0.9
  utility:
    disagreement_weight: 0.5
    reward_weight: 0.3
    age_penalty: 0.2
```

`extra="forbid"`：须在 `config.py` 加 `L2Cfg` schema + `OPDConfig.l2` 槽位。`l2.enabled: false` 时全部退回现有 L0/L1 静态路径，零行为变化。

---

## 8. 最小侵入式文件映射

| 能力 | 文件 | 理由 |
|------|------|------|
| 全部新增逻辑 | 新 `adaptive_cache.py` | `RefreshSelector`/`CacheHealthMonitor`/`DynamicRatioController`/`DisagreementComputer` 独立类，不污染 cache.py/scheduler.py |
| Ring Buffer | `cache.py` 加 `append_refresh()` | base 池张量不动，refresh 池独立 + FIFO+价值保护 |
| 双池 Feeder | `scheduler.py::_rand_idxs` 包一层 | base/refresh 按 mix_ratio 采样，base 不受版本截断 |
| staleness bug 修 | `scheduler.py::_train_step:291` | base 样本跳过版本截断 |
| Disagreement 计算 | rollout 相位（新 `adaptive_cache.py`） | _train_step 不动 |
| 配置 | `config.py` 新 `L2Cfg` + seep | 默认关 |
| 指标 | `metrics.py` 无改动 | `record()` 加字段即可 |
| Lookup/Reuse 计数器 | `cache.py`/`scheduler.py` 埋点 | 最小侵入：lookup hit/miss + reuse count 仅 increment 计数器，聚合逻辑在 `CacheHealthMonitor` |

---

## 9. 测试

### Disagreement 单元测试（§3 权威要求）
1. 数学公式单元测试
2. padding mask 测试
3. variable-length 测试
4. EOS 测试
5. token alignment 测试
6. identical teacher/student 时 disagreement≈0
7. teacher/student 差异放大时 disagreement 单调增加
8. distributed batch 下结果一致

### 集成测试
- ring buffer append/淘汰/价值保护
- 双池 feeder mix_ratio 采样比例
- base 样本不被版本截断
- `l2.enabled: false` 退回原行为（回归）

### Cache Health Monitor 测试（§4 权威要求）
- empty refresh pool / ring buffer rollover
- cache miss / duplicate / invalid entry
- age calculation / reuse counting
- threshold classification（HEALTHY/WARNING/CRITICAL）
- distributed aggregation（跨 rank 一致）
- `health_monitor: false` 时性能不退化

### Dynamic Refresh Ratio 测试（§5 权威要求）
- fixed / linear / adaptive 三模式
- bounds（`α_min ≤ α ≤ α_max`，`α_max<1`）
- EMA 平滑 / max_step_change 限幅
- empty refresh pool cold start（fallback base）
- distributed consistency
- extreme age / drift / quality（信号极值不崩溃）

---

## 10. Ablation

- **Dynamic Ratio**（§5.8）：`mode=fixed/linear/adaptive` 三组共享同一 feeder，对比 adaptive 是否优于 fixed。
- **四能力组合**：关 selective / 关 dynamic / 仅 health-monitor 记录 / 全开。
对比 `E[Δ_T]` 收敛曲线 + AIME 分数，验证各模块贡献。

---

## 11. 潜在兼容性问题

- **chosen-token logp 存储**：§3.2 数据结构定义完整字段供后续扩展；第一版实现取舍：rollout 阶段算完 `D_i` 标量后，4 个 per-token chosen logp **算完即弃**，只持久化 `disagreement`/`disagreement_abs` 标量 + `generation_step` + `response_length` + `token_mask` + 原 top-k（`rl_k`/`ref_k`）。per-token logp 按需在后续版本补存，避免 refresh sample 显存/磁盘膨胀。
- **student_ref 前向开销**：rollout 阶段多一个初始 student 前向（算 ref logp）。可用 P1-4 保留的独立实例，CPU offload 复用。
- **top-k vs chosen-token**：_train_step 的 `delta_for_student_topk` 按 student top-k 支撑展开（训练用），与 disagreement 的 chosen-token logp（监控用）是两套数据，不混淆。
- **FSDP + refresh 相位**：student 生成需完整权重，FSDP 下用 `summon_full_params` gather；生成是 no_grad，不触发反向通信。

---

## 12. 数据流变化

```
Skywork 50K parquet -> jsonl -> 初始 response 预生成(8192) -> 初始 topk cache(base pool)
   -> 【训练 T_train 步 ↔ rollout 刷新(disagreement 计算)-> append ring buffer】循环 ~10 轮
   -> checkpoint(FSDP state+opt+RNG+ring buffer 状态) -> eval
```

refresh sample 数据流：
```
selective prompt -> student 生成 y + logπ_S
   -> 初始 student 前向 logπ_Ref(y)
   -> teacher 前向 top-k + chosen logπ_T^RL/Ref(y)
   -> D_t = [π_T^RL−π_T^Ref]−[π_S−π_Ref] -> mask 聚合 D_i^abs
   -> append_refresh(ring buffer, 带 generation_step/version)
   -> feeder 按 mix_ratio 采 -> _train_step(纯查表, teacher-free)
```
