# 2026-08-28：OPD 复现最终归因报告（停止结论）

> **状态：停止 OPD 方向**（用户 2026-08-28 决策，选项 C）。
> 信号有效性不足（RC4）是 OPD 复现失败的根本原因——三域独立测量交叉确认，实现已审计无 bug。

---

## 1. 执行摘要

- **目标**：复现 Direct-OPD 论文（Δ_T 作为密集隐式奖励训练学生，迁移教师 RL 增益）。
- **结论**：**信号本质弱（Δ_T 与正确性相关性不足）是复现失败的根本**——即使教师对方向正确（rl 教师最强）、on-policy、论文配方全部对齐，训练沿弱信号优化只会复现旧失败（eval_reward↑ 能力↓）。**停止 OPD 方向**。
- **新发现**：格式错位污染加剧信号负值（跳过响应开头格式 token 后 ρ 从 -0.138 转正到 +0.017），但信号本质仍弱。

## 2. 完整证据链

### 2.1 旧实验（2026-08-26/27，B2048 决定性实验 + 终验）
| 项 | 结果 |
|---|---|
| B2048 决定性（H9） | Base 0.404 > E2 0.288（H9 排除） |
| B8192@3 终验 | MATH500 Base 0.816 / E1 0.314 / E2 0.376；AIME24 Base 0.233 / E1/E2 0 |
| 符号反转 | eval_reward +0.51 但能力 -0.44（代理指标假象） |
| 学生推理 | avg_rt 4303→6069-6565（推理更长但正确性降） |

### 2.2 v2 Phase 0（2026-08-27/28，论文原样复现）
| 门控 | 结果 | 判定 |
|---|---|---|
| 学生基线 AIME24/25 ave@8 | 26.7 / 25.8 | ✅ [20,60] |
| 教师对方向 | JustRL 42.1 > 学生 26.7 > R1Distill 22.5 | ✅ 与论文一致 |
| **E-1b' 三域 ρ** | boxed +0.177 / DAPO-MATH500 -0.147 / **DAPO-Skywork -0.034** | ❌ 全部 <0.2 |

### 2.3 归因实验（2026-08-28）
| 实验 | 结果 | 结论 |
|---|---|---|
| A. token 级归因 | pos_ratio 弱正 +0.019、mean_d 弱负 -0.012 | 指标矛盾，信号弱 |
| B. Skywork 训练域 | ρ=-0.034（未转正） | 域错位部分成立，信号仍弱 |
| C. 格式 token 污染 | 响应开头 ~20 token rl_logp 深度负（-10~-17）；**skip 20 后 ρ -0.138→+0.017** | 污染加剧负值，信号本质 +0.017 |

## 3. 归因定论（双重因素）

1. **信号本质弱（RC4，根本）**：Δ_T = logπ_JustRL − logπ_R1Distill 的教师对偏移方向与正确性相关性弱（+0.017，三域交叉确认）——JustRL 的 RL 优化格式/效率而非正确性 → Δ 方向风格主导，学生沿 Δ 优化无正确性增益。
2. **格式错位污染（加剧负值）**：学生（Qwen3）响应开头的 thinking 风格 token 被 JustRL（DeepSeek 系）教师深度惩罚（logp -10~-17）→ 前 ~20 token 深度负 Δ → 使 ρ 从真实 ~+0.02 偏到 -0.14。**训练 cache Δ 同样被污染**（学生被推向"不要 Qwen3 风格开头"→ 可能是旧实验学生风格漂移的机制之一）。

## 4. 实现审计（无 bug）

| 环节 | 论文对照 | 结论 |
|---|---|---|
| pg_loss | Eq.13/15（Rao-Blackwell + stop-grad + 稀疏重归一） | ✅ |
| cache Δ | Eq.10（rl_k−ref_k token 级） | ✅ |
| E-1b' Δ | 与训练 Δ 语义一致 | ✅ |

## 5. 停止决策（用户选 C）

- 信号本质弱是三个独立测量的交叉结论（+0.177/-0.147/-0.034，skip 后 +0.017）——无一接近论文所需 ≥0.2。
- 实现已审计无 bug——不是实现偏差。
- 继续训练在 +0.017 信号下大概率复现旧失败（eval_reward↑ 能力↓）——Phase 0 门控设计的目的（不通过不烧训练）达成。
- **诚实止损**：OPD 机制在本配置（教师对 + 学生 + 信号）下不成立。

## 6. 后续方向（如重新立项）

1. **RAFT/最优-n 蒸馏**（若目标为能力提升）：用教师（JustRL）正确响应做监督——B2 的 correctness 部分，实用但非 OPD 机制。
2. **换 RL 目标明确的教师对**：Δ 方向与正确性解耦是根本——换"正确性导向 RL"的教师对可恢复信号（此前用户决策严禁换对，供参考）。
3. **格式归一化**：若继续 OPD，训练 cache Δ 需跳过前 N token 的格式偏移或格式归一化。

## 7. RB 口径复刻（2026-08-30，判别性实验）

复刻论文信号口径（Eq.11/13：token 级 + 学生 top-16 支撑 + Rao-Blackwell 加权）重测 Skywork 域（800 条，HF 前向 gather）：

| 口径 | ρ | AUC |
|---|---|---|
| 序列级 E-1b'（DAPO-Skywork） | **-0.034** | 0.479 |
| **RB 口径**（token 级 top-16 + Rao-Blackwell） | **+0.061** | 0.537 |

**最终判别（后被官方代码重算推翻，见 §8）**：
1. 序列级确实低估（-0.034 → +0.061，转正，改善 0.095）——RB 口径更贴近论文信号；
2. 但 RB 口径仍 <0.2——一度认为信号弱（RC4）；
3. **推翻**：§8 官方代码重算证明信号强（+0.54~0.60），此前低估是 **mean 聚合稀释长序列信号**（官方 sum 聚合）。

实现：`delta_correctness_corr.py --stage rb-correlate`（`_rb_weighted_delta`/`_logsoftmax_topk`/`_gather_logp_at_ids` 纯函数，606 passed）。产物：`/root/autodl-tmp/r1_eval/delta_corr_sky/rb_report.json`。

## 8. 官方代码重算（2026-08-30，发邮件前验证，颠覆归因）

用官方 Direct-OPD 仓库纯函数（`_compute_teacher_top_k_log_probs` only_stu + `_compute_delta_opd_rm_scores` Eq.13 + ttrl_math 判分；HF 前向等价替换引擎）重算序列级 Δ↔correct（800 条 samples，本地 sympy 判分 MATH500 acc 0.698 / Skywork acc 0.316）：

| 口径 | 我们之前（mean 聚合） | 官方代码（sum 聚合）MATH500 | 官方代码（sum 聚合）Skywork |
|---|---|---|---|
| unweighted Eq.10 | +0.177 / -0.147 / -0.034 | +0.280（AUC 0.676） | +0.046（AUC 0.528） |
| **weighted Eq.13** | +0.061（rb-mean） | **+0.596（AUC 0.874）** | **+0.539（AUC 0.835）** |

**颠覆性结论**：
1. **官方信号口径（Eq.13 softmax 加权 + 序列级 sum）与正确性强正相关**（Skywork +0.539 / MATH500 +0.596）——**信号不是弱的**；
2. **RC4（信号弱）推翻**：此前"信号弱"是**测量聚合方式假象**——我们用 mean（每 token 平均）稀释了长推理样本累积的正确性信号；官方用 sum（总和）保留。rb-correlate 与官方单样本 Δ 数学等价（差 <1%），非实现 bug；
3. **失败原因重新归因到训练实现**：信号好，但训练**固定 D off-policy（on-policy ~4%，RC1）+ top-K 交集稀释 + 格式污染**破坏了信号传递——**RC1（固定 D）升为主嫌**；
4. 未加权 Eq.10 仍弱（Skywork +0.046）——**加权（Eq.13）是关键**（softmax 权重聚焦学生实际可能 token），raw gap 不含权重故弱。

实现：`main/scripts/official_delta_corr.py`（复制官方纯函数 + HF 前向）。产物：`/root/autodl-tmp/r1_eval/official_corr{,_sky}/official_report.json`。

## 8. 产物

- 评估 jsonl：`/root/autodl-tmp/r1_eval/{AIME24,AIME25,teacher_AIME24,delta_corr,delta_corr_sky}/`
- 报告：本文档 + `2026-08-28-v2-phase0-results.md` + `2026-08-27-opd-final-report.md`
- 脚本：`token_level_attribution.py`、`reeval_rho_skip_prefix.py`、`delta_correctness_corr.py --stage rb-correlate`（RB 口径）
