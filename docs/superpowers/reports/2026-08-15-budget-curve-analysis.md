# Stage 1.7 Reasoning Budget Curve 与效率指标报告

> 基于 Stage 1.6 Budget-Aware 的 all_results。效率用**真实 reasoning tokens**
> （EOS 位置或 budget cap，非 max_new_tokens）。曲线刻画能力-成本权衡，
> **不以 CoT length 为单独能力结论**。按 dataset 分节（benchmark 内比较）。

> **状态：代码与 CLI 已落地（`main/fullstack_opd_v2/budget_curve.py` + `eval-budget` 追加写本报告）。
> 本表为 Base 占位 —— 真实数字待服务器实跑填充；L0/L2 行待真实 HF checkpoint。**

## Dataset: MATH500（主结果）

| Model | AUC | nAUC | Accuracy@512 | Accuracy@1024 | Accuracy@2048 | B@50% |
|---|---:|---:|---:|---:|---:|---:|
| Base | - | - | - | - | - | - |
| L0 | * | * | * | * | * | * |
| L2 | * | * | * | * | * | * |

| Model | Budget | Accuracy | Reasoning Tokens | Accuracy/Token | OPD Gain/Token |
|---|---:|---:|---:|---:|---:|
| Base | 256 | - | - | - | - |
| Base | 512 | - | - | - | - |
| Base | 1024 | - | - | - | - |
| Base | 2048 | - | - | - | - |
| Base | 4096 | - | - | - | - |

### Budget-Normalized OPD Gain（等预算比较：同一 B 下 ΔA=A_L2-A_L0）

| Budget | ΔA = A_L2 - A_L0 | ΔA / E[L_L2] |
|---|---:|---:|
|（待 L0/L2 checkpoint）| | |

### 图

（待服务器实跑生成 5 图：accuracy / prefix_accuracy / accuracy_vs_actual_tokens / opd_gain / efficiency_vs_budget）

### 解读

- 曲线/AUC/nAUC/Efficiency 为**全模型全 budget 可比**口径；表1 B@50%、表2 OPD Gain/Token、
  ΔA 依赖 L0/L2。当前仅 Base（L0/L2 占位待 checkpoint）→ 这些行显示 `-`/占位，图 4 待 L0/L2 数据补画。
- 等预算比较**必须是同一 B**（`A_L2(B)-A_L0(B)`），禁止不同模型不同 budget 作主结论。
- 等性能比较 `B_M(A*)`：达到目标 A* 需更小 budget 的模型更高效。

## Dataset: GSM8K（基础泛化）

> 同构表（AUC/nAUC/双表/5 图），Base 实跑待服务器，L0/L2 占位待 checkpoint。

## Dataset: AIME24 / AIME25（补充）

> 同构表。AIME 真实实跑可选（补充角色），Base 待服务器确认是否纳入矩阵。

## 填充方式（待服务器恢复）

```bash
ssh opd 'cd /root/opd/main && git pull'
ssh opd 'cd /root/opd/main && /root/miniconda3/bin/python -m fullstack_opd_v2 eval-budget \
  --models "Base=/root/autodl-tmp/models/Qwen__Qwen3-1.7B" \
  --budgets 256,512,1024,2048,4096 \
  --datasets GSM8K MATH500 \
  --scoring sympy --prompt-style boxed --chat-template \
  --attn-impl flash_attention_2 --batch-size 4 --device cuda:0 \
  --out /root/autodl-tmp/eval/budget_aware'
scp -r opd:/root/autodl-tmp/eval/budget_aware \
  "C:\Users\12062\OneDrive\Desktop\opd\docs\superpowers\reports\budget_aware_data"
```

CLI 会同时产出 `2026-08-15-budget-aware-eval.md`（Stage 1.6 协议）与
`2026-08-15-budget-curve-analysis.md`（Stage 1.7 效率指标）两份报告 + 各自图。