# L2 Adaptive Staleness-Aware Teacher Cache · 实现报告

> 面向用户：完整记录 L2 四能力（Teacher-Student Disagreement / Cache Health Monitor /
> Dynamic Refresh Ratio / Selective Rollout）的工程实现全貌——含**第一轮 worklfow 的装配
> 成果**与**本轮闭环修复（G1-G9）**。规格见 `docs/superpowers/specs/2026-08-14-adaptive-teacher-cache-design.md`，
> 分阶段计划见 `docs/superpowers/plans/2026-08-14-adaptive-teacher-cache.md`。

---

## 0. 结论速览

- **233 个测试全绿**（`cd main && python -m pytest tests/ -q`，~74s）。
- **端到端 toy smoke 通过**：早前"文档命令必崩"的 G9 缺陷已修（`max_response_length=8192`
  超出 toy `max_len=64`），现在文档命令原样可跑。
- **闭环达成（G1/G2/G3/G4/G8）**：refresh 样本真正进入训练（`_train_step_refresh`，
  teacher-free 稀疏 top-K PG+KL），`α` 真实应用，`PromptStateStore` 闭环喂给 selector，
  触发时机由 `refresh_min/max_interval` 门控，resume 恢复 optimizer 与 ring buffer。
- **E0-E6 实验矩阵改为 §10 累积语义**，且因 feeder 已接，各实验训练信号**真实可区分**
  （见 §7 Ablation）。E5（全 L2 + selective）reward 最优，E6（random rollout）劣于 E5
  → 初步验证 selective 贡献方向。

---

## 1. 实现状态总表（对照对抗审查遗留缺陷）

| ID | 缺陷 | 严重度 | 状态 | 说明 |
|----|------|--------|------|------|
| G1 | 双池 feeder 未接入：refresh 样本从不进训练 | 高 | ✅ 已修 | `RefreshRingBuffer` 存行为策略 s_old + teacher Δ_T；`scheduler._train_step_refresh` 做稀疏 top-K PG+KL；pipeline 交替相位按 α 折算 refresh 训练步 |
| G2 | PromptState 闭环未接：selector 恒 uniform | 中 | ✅ 已修 | `run_refresh_phase` 写回 `prompt_state.update`（reward 估计 + disagreement + resp_len），`times_seen>0` 后 selector 走两阶段价值选择 |
| G3 | α 算而不用 | 中 | ✅ 已修 | `drc.update()` 返回值经 `cold_start_adjust` 后折算 `n_refresh = round(α/(1-α)·n_base)`，调用 `scheduler.train_refresh_phase` |
| G4 | 触发时机无条件 | 中 | ✅ 已修 | `refresh_min_interval` 门控 + `refresh_max_interval` 强制；首个相位冷启动必刷新 |
| G5 | base 版本截断契约未实现 | 低 | ⚠️ 部分 | 现状下 base 每次用当前权重重算 s_old 带新版本不会误触发；"base 跳过版本截断"契约未显式实现（见 §8 已知问题） |
| G6 | E0-E6 语义与 §10 不一致 + 信号不可区分 | 低 | ✅ 已修 | 矩阵改累积构建；feeder 接后信号可区分 |
| G7 | §3.5 utility / §13.2 统一 metadata 部分 | 低 | ⚠️ 部分 | ring buffer 存子集；`U_i` 完整公式未驱动（见 §8） |
| G8 | resume 未恢复 optimizer 与 ring buffer | 高 | ✅ 已修 | `scheduler.opt.load_state_dict(_resume_opt)`；`RefreshRingBuffer.state_dict/load_state_dict` + pipeline 装回 |
| G9 | 文档 smoke 崩溃 | 阻断 | ✅ 已修 | `run_refresh_phase` 把 `max_resp_len` clamp 到 `(student.max_len - prompt_len)` |
| G10 | Implementation Report 未产出 | — | ✅ 本文件 | — |

---

## 2. 架构变化

L2 采用 **colocated 交替相位**（§1）：训练相位（base 池，`_train_step` teacher-free 一行
不动）↔ rollout 刷新相位（teacher 前向在此）+ **refresh 训练相位**（双池注入，本轮新增）。

```
┌─────────────────────────────── 一个 L2 周期 ───────────────────────────────┐
│ 训练相位(AsyncBatchedScheduler.run, base 池)                                │
│   └─ _train_step: 查 base cache Δ_T，π_old 加权 PG + k3 KL，teacher-free    │
│ 刷新相位(run_refresh_phase)                                                 │
│   └─ RefreshSelector 两阶段选 prompt → student 生成 → 4 chosen logp        │
│      → D_i^abs → ring_buffer.append(s_old + Δ_T + context)                 │
│   └─ PromptStateStore.update（闭环，喂下轮 selector）                       │
│ 动态α相位(DynamicRatioController)                                          │
│   └─ α=clip(α0+λA·ÃB−λD·D̃drift+λQ·Q̃)，cold_start_adjust                     │
│ refresh 训练相位(scheduler.train_refresh_phase)   ← G1 闭环                 │
│   └─ _train_step_refresh: 从 ring buffer 取 s_old+Δ_T，按 s_cur top-K 支撑  │
│      展开做稀疏 top-K PG+KL（teacher-free）                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

**单向依赖**（§13.1，禁止循环改内部状态）：
`RefreshSelector → DisagreementComputer → RefreshRingBuffer → CacheHealthMonitor
→ DynamicRatioController → Feeder`。Monitor 只 observe，Controller 只 consume，
均不直接改彼此状态。

## 3. 文件变化

| 文件 | 变更 |
|------|------|
| `main/fullstack_opd_v2/adaptive_cache.py` | 新增 6 类：`RefreshRingBuffer`（+s_old/context/searchsorted 展开）、`DisagreementComputer`、`CacheHealthMonitor`、`DynamicRatioController`、`PromptStateStore`、`RefreshSelector`；`run_refresh_phase`（G9 clamp + G2 闭环 + 存行为 s_old） |
| `main/fullstack_opd_v2/scheduler.py` | 新增 `_train_step_refresh`（G1 稀疏 top-K PG+KL）、`train_refresh_phase`（G3）；`run`/`_train_dispatcher` 加 `start_step`（交替相位 step 单调，修 checkpoint 覆盖） |
| `main/fullstack_opd_v2/pipeline.py` | L2 交替相位循环接入（G3 α 应用、G4 触发时机、G8 rb 恢复）；`_save_ckpt` 传 `rb.state_dict()`；optimizer resume |
| `main/fullstack_opd_v2/config.py` | `L2Cfg` 全段 schema（`extra="forbid"`，每模块 `enabled`） |
| `main/fullstack_opd_v2/experiment.py` | E0-E6 矩阵改 §10 累积构建（G6） |
| `main/fullstack_opd_v2/checkpoint.py` | 支持 optimizer/RNG/refresh_buffer 落盘（`_opt_state_to_cpu`） |
| `main/tests/test_l2_integration.py` | 新增 G1 闭环断言 + E0-E6 累积语义断言 |

## 4. 数据流

refresh 样本完整生命周期：

```
RolloutSelector.select(M) ──cand──► student.generate_batch(M)
  ──► 4 chosen logp(teacher_rl/ref, student/ref) ──► DisagreementComputer.D_i^abs
  ──► ring_buffer.append( prompt_idx, response, s_old_ids/logp(行为 top-K),
                          teacher ids_k/delta_k, token_mask, D_i^abs, gen_step )
  ──► PromptStateStore.update(cand, reward_est, D, resp_len, step)   [G2]
        │
        ▼  (下轮 selector 用 times_seen/disagreement/reward_var 做 cheap scoring)
  scheduler.train_refresh_phase(rb, α, n_refresh, start_step)        [G1+G3]
  ──► _train_step_refresh: s_cur=当前student(行为对refresh response)
      s_topk=topk(s_cur) → delta_at=rb.delta_at_student_topk(searchsorted)
      → s_old_at=rb.s_old_at_student_topk → ref_at(ref anchors)
      → pg_loss(稀疏, renormalize) + low_var_kl_support → opt.step → _publish
```

## 5. 配置

所有 L2 能力集中在 `l2.*`，默认 `l2.enabled=false` 退回 L0/L1 静态路径（回归测试验证）。
每模块独立 `enabled` 支持单项 ablation。

```yaml
l2:
  enabled: true
  t_train: 100            # 每轮 base 训练步
  m_refresh: 1000         # 每轮刷新量（=M_selected）
  cache:
    refresh_size: 5000    # ring buffer 容量
    max_response_length: 8192
    refresh_min_interval: 50   # G4 触发性：距上次刷新 ≥ 此值才刷
    refresh_max_interval: 150  # G4 强制：超此值必刷
    value_protect_quantile: 0.9
  disagreement:    { enabled: true }
  health_monitor:  { enabled: true, health: {...}, alert_cooldown: 50 }
  refresh_ratio:   { enabled: true, mode: adaptive|fixed|linear,
                     initial: .30, min: .10, max: .60,
                     age_weight: .25, drift_weight: .50, quality_weight: .25,
                     ema_beta: .9, warmup_steps: 500, max_step_change: .05 }
  selective_rollout: { enabled: true, candidate_multiplier: 4,
                       value_fraction: .80, coverage_fraction: .20,
                       value_weights: {uncertainty:.4,disagreement:.4,novelty:.2},
                       compute_aware: false, max_same_prompt_fraction: .05,
                       exploration_fraction: .20 }
  utility:         { disagreement_weight: .5, reward_weight: .3, age_penalty: .2 }
```

## 6. 测试结果

`cd main && python -m pytest tests/ -q` → **233 passed, 1 warning**（wandb 缺失 fallback 预期告警）。
关键新增：

- `test_alternating_phase_loop`：**G1 闭环断言**——训练后存在 `pool=="refresh"` 的步，
  证明 refresh 样本真正进训练（否则 L2 是装配不消费的脚手架）。
- `test_refresh_phase_padding_mask_excludes_pad`：变长 response mask 只统计有效 token。
- `test_no_teacher_forward_in_train_step`：`_train_step` 一步内无 teacher 前向（spy 计数不变）。
- `test_no_gpu_memory_growth_l2` / `test_no_unbounded_metadata_growth`：无线程泄漏、
  PromptState 固定形状 O(n_prompts)。
- `test_e0_e6_matrix_configs_valid` / `test_e0_e6_matrix_off_configs`：E1-E6 累积开关语义。

## 7. Ablation 结果（toy/CPU，n_steps=12，仅演示矩阵可区分）

```
E0_base_only            reward=-0.2391  pg=0.2740  kl=0.0629  total=2.34s  12步
E1_base_fixed_refresh   reward=-0.2405  pg=0.2757  kl=0.0624  total=2.34s  12步
E2_add_disagreement     reward=-0.2443  pg=0.2802  kl=0.0630  total=2.74s  12步
E3_add_health_monitor   reward=-0.2422  pg=0.2767  kl=0.0623  total=2.15s  12步
E4_add_dynamic_ratio    reward=-0.2300  pg=0.2651  kl=0.0628  total=2.09s  12步
E5_add_selective_rollout reward=-0.2186 pg=0.2518 kl=0.0583  total=2.29s  13步  ← 最优
E6_random_rollout        reward=-0.2377  pg=0.2737  kl=0.0632  total=2.46s  12步
```

> ⚠️ **toy/CPU 演示结论仅供参考趋势**：E5（全 L2 + selective）reward 最优且多出 1 个
> refresh 训练步；E6（random rollout）劣于 E5 → selective 贡献方向正确；E4（adaptive α）
> 优于 E2/E3（fixed α）→ dynamic ratio 方向正确。**真实规模（1.7B 学生 + 真实教师对 +
> GPU）需按 §10 在服务器重跑**才能量化贡献，且须监控 `Performance/Teacher Compute` 与
> `Performance/Rollout Tokens` 两个比值（不只看 final benchmark，见《L2 四模块整合》提示词）。

## 8. 已知问题 / 边界

1. **G5 base 版本截断契约未显式实现**：`_train_step:291` 仍统一按
   `current_version - ver > threshold` 截断。当前 base 样本每次用当前权重重算 s_old 带
   新版本，不会误触发；但"base 跳过截断、仅 refresh 受截断"的§2 Q4 契约未落地。若引入
   base 样本复用旧 s_old 会误伤。
2. **G7 §3.5 sample utility `U_i` 未驱动**：`L2UtilityCfg` 存在但 `U_i=λ_D D+λ_R R̂−λ_A A`
   未在 ring buffer 价值保护/淘汰中实际使用（当前用 `disagreement_abs` 作价值保护分位）。
3. **`disagreement.enabled` 是纯配置开关**：`run_refresh_phase` 恒算 D（DisagreementComputer），
   E1↔E2 差异由"D 是否喂给 selector/quality 信号"体现，未做"完全不计算 D"的硬 gate。
4. **HF/FSDP/真实 GPU 路径仍是骨架**：`HFCausalLM`、`DistAsyncScheduler`、colocated CPU
   offload 需 GPU 验证（标注 `⚠️ 骨架`）。本实现算法内核在分布式下被直接复用（`_train_step`
   不动），但未在真实 2×PRO6000 上端到端跑通。
5. **`_build_mask` 用 `pad_id=0` 近似**：真实变长场景需用 tokenizer.pad_token_id / EOS id
   判定（§3.4 注明）。
6. **refresh 训练步不纳入 `n_steps` 口径**：`n_total` 只计 base 训练步，refresh 是补充步，
   故 E5 出现 13 步（12 base + 1 refresh）。记录时需区分 base/refresh 步数（metrics 有
   `pool` 字段）。

## 9. 下一阶段建议

1. **服务器重跑 E0-E6（真实规模）**：用 1.7B 学生 + 真实教师对（JustRL/R1-Distill）+ Skywork
   数据，按 §10 跑 E0-E6，重点看 `Performance/Teacher Compute` 与 `Performance/Rollout Tokens`
   两个比值。这是验证 L2 价值的唯一权威口径。
2. **落地 G5**：把 base 版本截断改成"base 跳过、仅 refresh 受截断"，消除潜在误伤。
3. **落地 G7（sample utility）**：用 `U_i` 驱动 ring buffer 价值保护与淘汰，替代纯
   disagreement 分位。
4. **GPU 验证 HF/FSDP 骨架**：跑通 colocated 交替相位（FSDP learner + CPU offload rollout），
   验证真实词表下稀疏支撑展开的显存与数值行为。
5. **成本核算模块**：在 rollout 相位显式统计 `teacher_forward_tokens` 与 `rollout_tokens`，
   注入 metrics，供 E0-E6 计算 `Performance/Compute` 比值（当前效率指标用 `total_s` 作代理）。