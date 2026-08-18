# Stage 1.6 Budget-Aware Evaluation 决策报告

> 状态：**MATH500（主结果）已实测（2026-08-16，Qwen3-1.7B，greedy，n_samples=1）。
> B1024/B2048 时间盒未跑（评估 ~50 tok/s 过慢），按截断主导外推（该模型族 EOS rate≈0、
> 数学数据撞预算 cap，见 generation_smoke）。GSM8K / AIME 补充档待跑。**

## 1. 协议定义

`Accuracy@B`=outcome（预算内自然产出正确最终答案）；`PrefixAccuracy@B`=solvability
（仅预算内无 final answer 样本经固定提示 answer-completion 得正确）。
`status`∈{eos,budget_stop} 显式区分；`reasoning_tokens` 与 `answer_completion_tokens` 分离。
生成为 greedy（n_samples=1 → do_sample=False）。

## 2. 评估矩阵（MATH500）

| Model | Budget | n | Accuracy | PrefixAccuracy | EOS | BudgetStop | AvgReasoningTokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | 256 | 500 | 0.0460 | 0.0000 | 0.0000 | 1.0000 | 256 |
| Base | 512 | 500 | 0.0860 | 0.0000 | 0.0000 | 1.0000 | 512 |
| Base | 1024 | — | — | — | — | — | 截断主导（未跑） |
| Base | 2048 | — | — | — | — | — | 截断主导（未跑） |

## 3. 解读
- Accuracy 随预算提升但幅度小（256→512），且 EOS rate=0、BudgetStop=1.0：
  **该模型族在数学数据上从不在预算内自然 EOS**，预算只是截断点。
- 无答案率 n_noans（B256）= 10/500。

## L0/L2 行（S2 checkpoint，AIME24 @ B4096，n=1）

| Model | Dataset | Budget | Accuracy | EOS | BudgetStop | 说明 |
|---|---|---:|---:|---:|---:|---|
| S2_E0（≈L0 静态基线） | AIME24 | 4096 | 0.033 | 0.000 | 1.000 | 20 步静态训练 checkpoint |
| S2_E3（≈L2·2048 rollout） | AIME24 | 4096 | 0.000 | 0.000 | 1.000 | 刷新训练小池使 kl 升、评估反降 |

> ⚠️ 20 步训练距有效 AIME 提升仍远；E3 评估低于 E0 与 S2 报告一致（tiny refresh 池噪声
> 使 checkpoint 退化）。MATH500 上的 L0/L2 行待更长训练（百步级）+ 更多刷新样本后补跑。
> 等预算 OPD gain（ΔA=A_L2−A_L0）在 AIME24 上为 **−0.033**（负值，方向性待验证）。
