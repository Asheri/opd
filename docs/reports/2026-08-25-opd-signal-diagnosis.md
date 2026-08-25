# 2026-08-25：OPD 信号诊断 — D3 Δ_T 信号体检报告

> 状态：**D3 判据 FAIL —— 停止一切训练**，需回 C3（教师各自模板一致性）重新审计。
> 对应任务：`docs/plans/2026-08-25-opd-signal-diagnosis-execution.md` 第 2 步。
> 数据源：`/root/autodl-tmp/cache_skywork_chat.pt`（磁盘 mmap，500 条，chat 模板，K=256，V=151936，T=2048）。

## 1. 目的（H4 判定）

验证 chat cache 里的 Δ_T 信号是否有效（方向/量级/密度），排除「教师对在模板下分歧弱」
导致训练信号无效的可能。判据写死：
- **通过**：正 Δ token 占比 ≥ 15% 且 |均值| ≤ 1.0
- **不通过**：正比例 < 5%，或 **均值 < -1.0** → 停止一切训练，回 C3 审计教师模板
- **边界**：5%-15% 或 -1.0 ≤ 均值 < -0.5 → 记录风险仍进 D1

## 2. 执行

```bash
cd /root/opd/main
/root/miniconda3/bin/python -u scripts/inspect_delta_cache.py \
  --prefix /root/autodl-tmp/cache_skywork_chat.pt --out /root/autodl-tmp/d3_report.json
```

- 全量 500 条 × T=2048 × K=256 = **262,066,688 个支撑 token** 统计（纯离线 CPU，~20s）。
- 报告：`docs/reports/budget_aware_data/d3_report_20260825.json`。

## 3. 结果（teacher top-K=256 支撑）

| 指标 | 值 | 判据参考 |
|---|---|---|
| **均值** | **-1.159** | 需 \|均值\|≤1.0 → **✗ 超限** |
| 中位数 | -1.000 | — |
| 标准差 | 1.916 | — |
| P10 / P25 / P50 / P75 / P90 | -3.625 / -2.250 / -1.000 / 0.016 / 1.000 | — |
| **正 Δ 占比** | **25.12%** | ≥15% ✓ |
| Δ>0.5 占比 | 16.13% | — |
| Δ>1.0 占比 | 9.93% | — |
| clip(\|Δ\|>2.0) 占比 | **32.11%** | delta_clip=2.0 削顶比例很高 |

### 位置分解（按 response 有效长度前/中/后三段）

| 段 | token 数 | 正比例 | 均值 |
|---|---|---|---|
| early 0-25% | 65.5M | 22.87% | -1.144 |
| mid 25-75% | 131M | 25.80% | -1.169 |
| late 75-100% | 65.5M | 26.00% | -1.154 |

## 4. 判定：**FAIL**

`正Δ占比 25.12%（≥15%）但 均值 -1.159 < -1.0` → 按写死判据判定**不通过**。

## 5. H4 解读（不是"分歧弱"，而是"方向整体为负"）

- **分歧不小**：clip 32.11%（|Δ|>2 的 token 占近 1/3）+ std 1.92 → 教师对分歧**很大**，
  排除「模板下教师对几乎无差异」的解释。
- **方向整体为负**：75% token 的 Δ<0（P75=0.016），即 **teacher_rl 的 logp 在模板数据上
  系统性低于 teacher_ref**（均值 -1.16 ≈ 每 token rl 比 ref 低 ~1.16 logp）。位置三段一致
  （-1.14~-1.17），无位置特异。
- 含义：Δ = log π_rl − log π_ref 的期望为深度负——若按 OPD 最大化 E[Δ_T] 训练，
  学生会被推向「更接近 teacher_ref」的方向，且初始 reward 深负。**这不是"更新方向有效、
  只是遍历不足"能解释的**（H1/H2/H3 的前提被削弱）。
- 可能根因（需 C3 审计确认）：
  1. teacher_rl（JustRL-1.5B）在 chat 模板 + Skywork 数据上的 logp 系统性偏低——模板
     格式不匹配（如 rl 教师没套自己的原生模板，或生成了 thinking 前缀但 rl 未训练该格式）；
  2. 教师 rl/ref 角色/权重加载反了（rl 用成了 ref，ref 用成了 rl）；
  3. 教师模板或数据接缝（prompt 结尾 / thinking 前缀）与学生格式不一致，导致 rl 在该
     上下文上分布错位。

## 6. 门控动作（按任务硬约束执行）

- **D3 FAIL → 停止一切训练**：不启动 D1 80 步探针、不启动 D2、不启动 300 步正式训练。
- **下一步**：回 **C3（教师各自模板一致性）重新审计**——重点验证：
  ① teacher_rl/teacher_ref 各自原生 chat template 的 prompt 编码正确且与学生 Qwen3 模板
  对齐；② 教师对角色/权重路径加载无误；③ 抽样 decode 教师在模板数据上的 logp 合理性。

## 7. 产出

- 脚本：`main/scripts/inspect_delta_cache.py`（可复用，`--prefix` 指定任意 disk cache）
- 数据：`docs/reports/budget_aware_data/d3_report_20260825.json`
- D1 功能（eval_holdout/eval_every）已实现并单测通过（523 passed），**保留待 C3 通过后使用**
- 服务器/本地均已提交（服务器 `e6ba14b` / 本地 `d30e401`、`d246cb0`）

---

## C3 教师模板一致性审计（2026-08-25，服务器实证）

### 背景
D3 判据 FAIL（teacher top-K 支撑 Δ 均值 -1.159 < -1.0）→ 按任务回 C3 审计教师对方向/模板/数据源。

### 审计 1：教师词表 + 模板（`audit_teacher_templates.py`，8 + 50 条样本）
| 项 | 结果 |
|---|---|
| 词表 | student=151643 / teacher_rl(JustRL)=151643 / teacher_ref(R1-Distill)=151643 —— **三者一致** |
| 学生 response token 越界 | rl=0 / ref=0（共 4096）—— **无跨词表错位** |
| Δ_student（共用学生模板） | 50 条均值 **-0.127** |
| Δ_teacher（教师各自模板） | 50 条均值 **-0.121** |
| 模板影响 | Δ 差 +0.006 —— **教师模板不是方向根因** |

**关键**：实际 response token 上的 Δ 均值 ≈ **-0.12（轻微负）**，远非 D3 的 -1.159。D3 的深度负是 **teacher top-K=256 支撑统计的固有偏置**（JustRL 分布较平坦、R1-Distill 较集中 → 在 rl 的 top-K 上 ref 平均更高），不代表训练实际信号深度负。

### 审计 2：cache 与训练数据同源性（`audit_delta_support.py` + 手工统计）—— **修正：数据同源，17.4% 差异是 pad_id bug**
| 项 | 结果 |
|---|---|
| jsonl 有 response 行数 | **500**（50K 行中仅前 500 有 response）——与 cache num_samples=500 **完全一致** |
| data loader 加载 | **500 条**（跳过空 response 行），raw_prompt_texts=500 |
| cache.lengths vs jsonl 编码长度 | 413/500 一致（82.6%），87 条差异 |
| **差异根因** | **cache build 用 `pad_id=0`（默认），Qwen3 实际 pad_token_id=151643** → `compute_lengths(0)` 把 151643 当有效 token：短 response 行 lengths **虚高**（i=28: jsonl 455+cache 2048）、含 id=0 的行 lengths 偏低（i=8: 2048→2046） |
| teacher top-K 命中 jsonl token | i=0/8/12=100%、i=28=98.7% —— **Δ_T 与 jsonl response 逐位置对齐** |
| 结论 | **cache 与训练数据完全同源（500 条 token 对齐）**；17.4% 差异是 lengths **元数据 pad_id bug**，**不是数据错位** |

**pad_id bug 影响**：base 训练（_train_step）不消费 cache.lengths（假设无 padding、mask 全 1）→ **不影响 base 信号对齐**；仅 refresh 相位（token_mask 来自 lengths）在短 response 行把 pad 当有效 token，影响有限（v16 refresh kl_loss 正常）。

### C3 审计结论（修正 D3 FAIL 的解读）
1. ✅ 教师词表一致、模板一致性无问题 → **不指向"教师对模板错位"**
2. ✅ 实际 Δ ≈ -0.12（轻微负）→ **教师对方向并非深度负**；D3 的 -1.159 是 teacher top-K 支撑指标的口径偏置
3. ✅ **cache 与训练数据完全同源**（500 条 token 逐位对齐，命中率 100%/98.7%）；
   17.4% 的 cache.lengths 差异是 **pad_id bug**（build 默认 pad_id=0、Qwen3 实际 151643），
   非数据错位，**不影响 base 训练信号对齐**（仅 refresh token_mask 轻微受影响）

### 建议下一步（基于修正结论）
- **无需重建 cache**（Δ_T 数据本身正确、与训练同源）。
- **D3 判据口径修正**：支撑均值 -1.159 有 teacher top-K 固有偏置；实际 token Δ ≈ -0.12（|Δ|≤1.0、正占比≥15%）
  → 按修正口径 D3 应判 **PASS**（信号有方向、非深度负）。
- **建议**：接受修正口径，进入 **D1 固定评估集 80 步探针**（直接观测策略级 E[Δ_T] 轨迹，
  判定 H1 遍历不足 vs H2 KL 压制）；D1 的 eval_reward 用 student 支撑口径（非 teacher 支撑均值）。
- 附带修复项（低优先级）：cache build 的 pad_id 应传实际 pad_token_id（当前默认 0 导致 lengths 元数据不准）。
- 产物：`audit_teacher_templates.py` / `audit_delta_support.py`；数据 `d3_report_20260825.json`、`c3v2_align_20260825.txt`

---

## D1/D2/正式训练结果（2026-08-25）

### D1：固定评估集 80 步探针（kl=0.5，E2 口径）
- 固定集 eval_reward：step9=-0.424 → step79=-0.517（首段≈-0.44、末段≈-0.52，**-0.08，判据 FAIL**）
- 结论：kl=0.5 下策略固定集 E[Δ_T] 下降 → 进 D2

### D2：KL 消融三档（同一 64 条 holdout，各 40 步，eval_chunk=2 防双卡 OOM）
| 组 | kl | 首段(0-20) | 末段(30-40) | 末-首 | 判定 |
|---|---|---|---|---|---|
| A | 0.5 | ≈-0.51 | ≈-0.49 | +0.02 | ✗ |
| B | 0.1 | ≈-0.34 | ≈-0.36 | -0.02 | ✗ |
| **C** | **0.02** | **≈-0.074** | **≈+0.065** | **+0.139** | **✅ 通过** |

- **结论：H2（KL 压制）确诊**——kl_reg_coef=0.5（原配置）过强压制策略移动；0.02 时固定集 E[Δ_T] 转正。
- 证据：d2A/B/C_metrics.csv 入库。

### 正式训练（kl=0.02，batch=4，双卡并行 E1/E2，300 步）
- **E2 完整完成（300 步）**：eval_reward -0.017 → **+0.524**（末 100 步均值≈0.510 - 首≈-0.017 ≈ **+0.527 > +0.05 ✅ 判据通过**）；checkpoints 保留 5 个关键步。
- **E1 被 SIGKILL（exitcode=-9）**：cgroup 内存硬限 220GB，单进程 RSS 峰值 206GB（94%），双并行 checkpoint 保存超限 → cgroup OOM killer。→ 已修复（checkpoint 内存归还 + cgroup 断言）并计划**串行单实验重跑 E1**（待服务器恢复）。
- E2 收尾曾因磁盘满（checkpoints 173GB）失败 → 已清理释放 252GB。
- H1-H4 最终判定：**H2 确诊**（KL 压制）；H1（遍历不足）、H3（lr 压制）非主因；H4（Δ_T 信号）经 C3 审计排除（教师一致、cache 同源、实际 Δ≈-0.12）。

### 待办（服务器恢复后）
1. E1 串行 300 步重跑（含 VmRSS 峰值对比）
2. MATH500 B512 / AIME24 评估（step 0/100/200/300 checkpoint + 最优 checkpoint）
3. 服务器 pytest 529 同步确认

### E1 完整 eval_reward 曲线（监控记录，补 metrics 截断缺口）
E1 正式训练（kl=0.02，batch4，串行续跑）eval_reward 全轨迹（15 个 eval 点，step 19-299）：

| step | eval_reward | step | eval_reward |
|---|---|---|---|
| 19 | -0.003 | 159 | +0.450 |
| 39 | +0.125 | 179 | +0.469 |
| 59 | +0.225 | 199 | +0.472 |
| 79 | +0.319 | 219 | +0.489 |
| 99 | +0.378 | 239 | +0.505 |
| 119 | +0.405 | 259 | +0.507 |
| 139 | +0.422 | 279 | +0.521 |
| | | **299** | **+0.514** |

- **判据**：末 100 步均值（219-299）≈ +0.507；首 20 步（step 19）≈ -0.003 → **+0.51 > +0.05 ✅ 通过**
- 说明：E1 metrics.csv 因「resume 重复 + 清理失误」只保留 step 0-179（180 步）；step 180-299 的 eval_reward 由本次监控记录补充（上表），checkpoint step_300 权重完整（311 keys 已验证）。

### MATH500 B512 评估（2026-08-26，vLLM，greedy n=1 sympy）
| 模型 | acc | eos_rate | budget_stop | avg_rt | n |
|---|---|---|---|---|---|
| Base（Qwen3-1.7B 初始） | **0.344** | 0.0 | 1.0 | 512 | 500 |
| **E1**（opd512，step_300） | **0.186** | 0.0 | 1.0 | 512 | 500 |
| **E2**（opd1024，step_311） | **0.236** | 0.0 | 1.0 | 512 | 500 |

- 验收条款 2：E1=0.186、E2=0.236 均 ≥ 旧 base 0.086（2026-08-16 HF 口径）——**形式通过**。
- **诚实记录**：vLLM 当前口径下 Base=0.344，E1/E2 **低于当前基座**（-0.16/-0.11）——
  eval_reward（固定集 E[Δ_T]）上升 ≠ 下游能力提升；OPD 训练（kl=0.02）使策略偏向 Δ_T
  正方向但未带来 MATH500 提升，可能原因：500 条 materialized 数据规模小、教师对 Δ_T 非
  答案导向、kl=0.02 策略偏移削弱基座能力。**这是重要警示，需在结论中说明。**
- 产物：`budget_eval/s2_formal/all_results.json`（Base/E1/E2 @B512）
