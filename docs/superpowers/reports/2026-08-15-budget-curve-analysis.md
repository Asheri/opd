# Stage 1.7 Reasoning Budget Curve 与效率指标报告

> 状态：**MATH500 主结果 2 点实测（B256/B512，greedy）；B1024/B2048 未跑（时间盒）。
> 曲线仅 2 点，AUC/nAUC 为梯形近似；L0/L2 行待 checkpoint。**

## Dataset: MATH500（主结果）

| Model | Budget | Accuracy | Reasoning Tokens | Accuracy/Token |
|---|---|---:|---:|---:|
| Base | 256 | 0.0460 | 256 | 0.000180 |
| Base | 512 | 0.0860 | 512 | 0.000168 |
| Base | 1024 | （未跑·截断主导） | — | — |
| Base | 2048 | （未跑·截断主导） | — | — |

AUC(梯形近似，256-512) = 16.8960 · nAUC = 0.066000

## 解读
- 效率口径用真实 reasoning tokens（EOS 位置或 budget cap）。Base 曲线 2 点斜率平缓，
  Accuracy/Token 随预算下降（更长 CoT 边际收益递减）——符合「截断主导、无自然 EOS」。
- 等预算 OPD gain（ΔA=A_L2−A_L0）与 B@50% 依赖 L0/L2 checkpoint，当前占位。
- 完整曲线（5 档）与 GSM8K/AIME 补充档需恢复完整 budget eval（~50 tok/s 瓶颈下
  B2048 单档即 ~5h；建议后续换 vLLM 加速或加多卡并行）。
