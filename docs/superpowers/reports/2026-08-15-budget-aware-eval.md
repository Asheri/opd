# Stage 1.6 Budget-Aware Evaluation 决策报告

> 状态：**协议与代码已落地（`main/fullstack_opd_v2/budget_eval.py` + `eval-budget` CLI）**。
> 矩阵表 Base 实测待服务器实跑填充；L0/L2 行占位待真实 HF checkpoint。

## 1. 协议定义

把「完整 CoT + EOS」的隐式必要条件重构为 `Accuracy(B)`——B = max reasoning token budget。

对每个 evaluation prompt `x` 与预算 `B∈{256,512,1024,2048,4096}`，模型生成 `y_{1:B}`，
允许 EOS 提前结束或撞预算截断，**显式区分**：

| status | 含义 | reasoning_tokens |
|--------|------|------------------|
| `eos` | 新 token 含 eos_token_id，自然终止 | eos 位置（不含 eos） |
| `budget_stop` | 撞预算 cap 截断 | B |

**双指标**：
- `Accuracy@B`（Outcome）：`verifier(extract_final_answer(预算内文本))` —— 模型在预算内自然产出正确最终答案的能力。
- `PrefixAccuracy@B`（Solvability）：仅对**预算内无 final answer** 的样本，保留 prefix_B 跑独立
  answer-completion（固定、不可修改提示 `Based on the reasoning above, provide only the final answer.`），
  经 verifier 判对。`PrefixAccuracy@B = 完成正确数 / 无答案数`，报告附无答案率。

**Token 记账**：`reasoning_tokens`（∈{eos 位置, B}）与 `answer_completion_tokens`（≤ completion_max_tokens=64）
严格分离，`total_tokens = reasoning + completion`；B 只约束 reasoning，completion 不计入预算。

**答案提取**（复用 eval_aime 既有 verifier，不新造）：`extract_final_answer` = `\boxed{}` 级联 →
Final Answer marker → benchmark parser → fallback。预算耗尽无 final answer **不判错**，进 prefix evaluation。

## 2. 数据集三档（用户拍板 2026-08-15）

| 档位 | 数据集 | HF | split | 角色 |
|------|--------|----|-------|------|
| 基础泛化 | GSM8K | `openai/gsm8k` | test | ground_truth 剥 `#### ` |
| **主结果** | **MATH-500** | `HuggingFaceH4/MATH-500` | test | 原样 LaTeX |
| 补充 | AIME24 | `Maxwell-Jia/AIME_2024` | train | 整数 |
| 补充 | AIME25 | `yentinglin/aime_2025` | train | 整数 |

verifier 默认 `scoring="sympy"`（论文数学等价判定 `grade_answer_mathd or grade_answer_sympy`），
支持 MATH-500 分数/LaTeX 与 GSM8K 小数，避免 `int` 精确匹配对非整数误判。

## 3. 评估矩阵

统一：相同 evaluation prompts / seed=42 / temperature=0.7 / top_p=0.95 / verifier（sympy）/
answer-completion protocol；`n_samples=1`（保留 >1 接口）。flash_attention_2、bf16、chat_template 包裹。

| Model | Budget | Accuracy | PrefixAccuracy | EOS | BudgetStop | AvgReasoningTokens |
|---|---:|---:|---:|---:|---:|---:|
| Base | 256 | - | - | - | - | - |
| Base | 512 | - | - | - | - | - |
| Base | 1024 | - | - | - | - | - |
| Base | 2048 | - | - | - | - | - |
| Base | 4096 | - | - | - | - | - |
| L0 | * | * | * | * | * | * |
| L2 | * | * | * | * | * | * |

> L0/L2 行待真实 HF checkpoint（当前无 L0/L2 checkpoint，`--models L0=` / `L2=` 空路径=占位跳过）。
> 数据集三档各一张同构表 + 4 图（Accuracy/PrefixAccuracy/EOS Rate vs Budget、AvgRT vs Accuracy）。

## 4. 服务器实跑命令（Base 先行）

```bash
# 1. 拉最新代码（含 budget_eval.py + eval-budget CLI）
ssh opd 'cd /root/opd/main && git pull'

# 2. 实跑矩阵（主结果 MATH-500 + 基础泛化 GSM8K；AIME 补充如需一并加 --datasets）
ssh opd 'cd /root/opd/main && /root/miniconda3/bin/python -m fullstack_opd_v2 eval-budget \
  --models "Base=/root/autodl-tmp/models/Qwen__Qwen3-1.7B" \
  --budgets 256,512,1024,2048,4096 \
  --datasets GSM8K MATH500 \
  --n-samples 1 --temperature 0.7 --top-p 0.95 \
  --scoring sympy --prompt-style boxed --chat-template \
  --attn-impl flash_attention_2 --batch-size 4 --device cuda:0 \
  --out /root/autodl-tmp/eval/budget_aware'

# 3. 拉回报告 + PNG + 每样本 jsonl
scp -r opd:/root/autodl-tmp/eval/budget_aware \
  "C:\Users\12062\OneDrive\Desktop\opd\docs\superpowers\reports\budget_aware_data"
```

## 5. 待办

- [ ] 服务器实跑 Base 矩阵（GSM8K + MATH-500，AIME 补充可选）→ 填 §3 表 + 4 图
- [ ] L0/L2 checkpoint 就绪后补跑对应行
- [ ] 逐样本 jsonl（`<label>__<ds>__B<budget>.jsonl`）供审计/重算