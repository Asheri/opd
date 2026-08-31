# 2026-08-28：v2 论文复现 Phase 0 执行结果（单卡 96GB）

> 方案：`docs/plans/2026-08-27-paper-replication-r1.md`（v2 论文原样复现：原教师对 + Qwen3-1.7B + AIME ave@8）。
> 执行：单卡 RTX PRO 6000 96GB；代码 `d81b8c0`（C1-C4 + v2.1 评估 + v2.2 数据修正）。

## Phase 0 门控结果

| 门控 | 结果 | 判定 |
|---|---|---|
| **基线 AIME24 ave@8** | **0.2667**（8.0/30） | ✅ ∈[20,60] |
| **基线 AIME25 ave@8** | **0.2583**（7.8/30） | ✅ ∈[20,60] |
| 教师对 JustRL AIME24 | 0.4208（12.6/30） | 对照论文 51.3 |
| 教师对 R1Distill AIME24 | 0.225（6.8/30） | 对照论文 28.5 |
| **E-1b' DAPO 域 ρ** | **-0.1467**（AUC 0.408） | ❌ **<0.05 不通过，停止** |

## 关键结论

1. **协议系统性偏低**（~55-82% 论文值，ave@8 vs ave@32 + 实现差异），但**相对关系与论文一致**（JustRL > 学生 > R1Distill）——教师对方向正确，协议可用。
2. **E-1b' Δ↔correct 负相关**（ρ=-0.147、AUC 0.408）：DAPO 域下 Δ_T（JustRL−R1Distill logp 偏移）方向与答案正确性**负相关**——学生沿 Δ 优化会降低正确性。对比之前 B2048+boxed 域 ρ=+0.177（弱正）。
3. **信号方向问题坐实**：即使教师对能力方向正确（rl 最强），Δ_T 的相对偏移成分以风格/格式主导，DAPO 域下更明显。

## 执行链

- 基线 AIME24/25：`/root/autodl-tmp/r1_eval/{AIME24,AIME25}/BaseS__*__B8192.jsonl`（各 30 题×8 采样）
- 教师对：`/root/autodl-tmp/r1_eval/teacher_AIME24/{JustRL,R1Distill}__AIME24__B8192.jsonl`
- E-1b'：`/root/autodl-tmp/r1_eval/delta_corr/{samples,logp_rl,logp_ref}.jsonl`（各 800）+ `report.json`

## 归因实验（2026-08-28，用户分析驱动）

用户指出 ρ=-0.147 可能是"度量方式不等价 + prompt 域错位 + 二值噪声"叠加而非信号反向。执行归因：

### 实验 A：token 级归因（离线，per-token Δ 分组）
```
correct   : mean_per_token_d=-0.211  d>0比例=0.401  mean_sum_d=-353.8
incorrect : mean_per_token_d=-0.199  d>0比例=0.383  mean_sum_d=-562.1
diff      : mean_d=-0.012  pos_ratio=+0.019
```
两个指标方向矛盾（pos_ratio 弱正 +0.019、mean_d 弱负 -0.012）——信号弱且不一致（度量+噪声放大 ρ 负）。

### 实验 B：Skywork 训练域 E-1b'（DAPO）
```
n=800, spearman_rho=-0.034, AUC=0.479, acc=0.316
```
**三域完整对比**：boxed(MATH500)=+0.177 / DAPO(MATH500)=-0.147 / **DAPO(Skywork)=-0.034**。

### 归因定论
1. prompt 域错位**部分成立**（Skywork 域 ρ 从 -0.147 → -0.034，改善 0.11）；
2. **但换到训练域后 ρ 仍≈0（未转正）**——不是"域错位导致信号负"，而是 **Δ_T 在所有测量域下与正确性无强正相关**（ρ 范围 -0.03~+0.18，无一 ≥0.2）；
3. 度量方式/二值噪声放大负值，但**信号有效性不足（RC4）是根本**——即使教师对方向正确（rl 最强）、on-policy、论文配方，Δ_T 的 logp 偏移方向与正确性相关性弱 → 学生沿 Δ 优化无正确性增益。**这解释了旧实验 eval_reward↑ 但能力↓ 的根本机制**。

### 实现审计（2026-08-28，本地代码）
训练信号实现与论文对齐，**无实现 bug**：
- `pg_loss` = 论文 Eq.13/15（Rao-Blackwell：`pg=−(π_old^renorm·min(ratio·Δ,clip·Δ)).sum()`；stop-gradient 权重用 s_old；token 级 Δ；稀疏支撑重归一）✅
- cache Δ = 论文 Eq.10（`delta_k=rl_k−ref_k` token 级）✅
- E-1b' Δ（vLLM per-token logp）与训练 Δ 语义一致 ✅

**结论**：信号弱是 **Δ_T 教师对偏移的本质**（JustRL RL 优化格式/效率而非正确性 → Δ 方向风格主导），非度量/实现问题。**为信号改造（B2：`reward=α·correct+β·Δ_seq` correctness 加权）提供明确依据**。

### 格式 token 污染发现（2026-08-28，logp 抽样审计）
抽样 decode 发现：**响应开头 ~20 个 token（Qwen3 thinking 风格 `\nOkay...`/特殊标记）的 rl_logp 深度负**（JustRL-DeepSeek 教师对 Qwen3 风格概率极低，logp -10~-17），后续 token 正常（≈0）。**跳过前 20 个 token 重算 ρ：-0.138 → +0.016（转正）**。

**修正归因（双重因素）**：
1. **格式错位污染**：前 ~20 token 深度负 Δ 使 ρ 偏向负（-0.14 vs 真实 ~+0.02）——训练 cache Δ 同样在 Qwen3 风格 response 上算，**同样污染训练信号**（学生被推向"不要 Qwen3 风格开头"→ 可能是旧实验学生风格漂移的机制之一）；
2. **信号本质仍弱**：跳过污染后 ρ=+0.017（微正，远低于论文所需 ≥0.2）——RC4 成立。

**训练侧建议**：E-1b' 判据应**跳过响应开头格式 token**（~20）重测；训练信号（cache Δ）应考虑对格式 token 的偏移做处理（如跳过前 N token 的 Δ 或格式归一化）。

## 后续选项（待决策）

1. **信号改造**（B2）：Δ 加 correctness 加权（reward=α·correct+β·Δ_seq）——绕开负相关
2. **回 boxed 域重测 E-1b'**：确认是否域特异（boxed ρ=+0.177 vs dapo -0.147）
3. **换信号来源**：不同教师对（但用户之前决策"严禁更换教师对"）
4. **审计实现**：Δ 计算/域是否与论文一致（负相关可能是实现偏移）
