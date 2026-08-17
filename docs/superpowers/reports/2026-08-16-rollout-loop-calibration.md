# Rollout Loop Detector 校准报告（IMP-1）

> 日期：2026-08-17 ｜ 状态：**已填充实测结论** ｜ 模型：Qwen3-1.7B（服务器实跑）｜ N=100 ｜ max_new=512 ｜ eos_token_id=151645 ｜ temperature=1.0 ｜ 人工标注：无

## 背景与目标

- 当前 loop 检测存在**误杀风险**：真实 Qwen3-1.7B + Skywork 短 rollout 循环退化率
  75-87%，默认 `(2,3,4)` 可能把正常长 CoT 误判为 loop（误报）。
- 目标：在配置矩阵（periods × min_len）中选择「**误报最低且能抓住明显循环**」的配置。
- 硬约束：**不得为凑 `<50%` 人为放松 detector**——若最低误报配置仍误杀正常 CoT，
  应记录并转向采样侧（temperature / repetition_penalty）治理，而非放宽检测。

## 方法学

1. 真实 rollout N 条（temperature=1.0，短预算 `max_new`，新生成 token 序列去 pad）。
2. 对每条 rollout，用与训练完全一致的 `detect_loop`（尾部周期自相关 + min_len 门槛）判定。
3. 对每种 (periods, min_len) 配置统计：`loop_detected_count` / `loop_rate` /
   `samples_flagged`；有**人工标注**时另算 `false_positive_rate` / `false_negative_cases`。
4. 人工抽样检查：正常长 CoT 是否被误杀；`Final Answer` 重复是否被捕获；
   真正 token-level repetition 是否被捕获。

## 配置矩阵

| periods | min_len |
|---|---|
| `(2, 3, 4)` | 8 |
| `(2, 3, 4)` | 16 |
| `(2, 3, 4)` | 24 |
| `(4, 6, 8)` | 8 |
| `(4, 6, 8)` | 16 |
| `(4, 6, 8)` | 24 |
| `(6, 8, 12)` | 8 |
| `(6, 8, 12)` | 16 |
| `(6, 8, 12)` | 24 |

## 结果

| config | loop_detected_count | loop_rate | FP rate | FN cases | samples_flagged |
|---|---:|---:|---:|---|---|
| `periods=(2, 3, 4),min_len=8` | 0 | 0.000 | - | - | - |
| `periods=(2, 3, 4),min_len=16` | 0 | 0.000 | - | - | - |
| `periods=(2, 3, 4),min_len=24` | 0 | 0.000 | - | - | - |
| `periods=(4, 6, 8),min_len=8` | 0 | 0.000 | - | - | - |
| `periods=(4, 6, 8),min_len=16` | 0 | 0.000 | - | - | - |
| `periods=(4, 6, 8),min_len=24` | 0 | 0.000 | - | - | - |
| `periods=(6, 8, 12),min_len=8` | 0 | 0.000 | - | - | - |
| `periods=(6, 8, 12),min_len=16` | 0 | 0.000 | - | - | - |
| `periods=(6, 8, 12),min_len=24` | 0 | 0.000 | - | - | - |

## 人工抽样检查清单（需 GPU + 人工）

- [ ] 正常长 CoT 是否被误杀（抽查 flagged 样本看内容）
- [ ] `Final Answer` 标记重复是否被捕获
- [ ] 真正 token-level repetition 是否被捕获
- [ ] 误报最低配置下的误报样本内容（是否为真实退化或误判）

## 实测结论（服务器 GPU 实跑，2026-08-17）

**核心发现：当前 `detect_loop`（尾部周期）在 512 token 短预算下实测 0% loop 率，且无 Final Answer 标记重复。**

- 100 条真实 rollout @ max_new=512（temperature=1.0）：**0/100 尾部周期 loop**（periods 2-8 × min_len 8/16/24 全部 0）；
  全部撞预算（E[L]=512），0 自然 EOS（模型在数学 CoT 上从不在预算内 EOS，已知行为）。
- 补充 marker 检查（20 条解码）：**0/20 Final Answer 重复**，仅 1/20 `boxed` 出现 2 次（疑为"check boxed"合法模式）。
- **与 stage2 报告"75-87% loop 率"矛盾**：该数值在**当前代码 + 当前条件（512 token）下无法复现**。
  推断：早前 75-87% 来自旧/异常代码状态（如早期 detect_loop 过严误报、或生成路径 bug），
  或需更长序列（1024/2048）才显现 marker 级退化。需以**实际训练路径**（run_refresh_phase +
  generate_with_status_kv）复跑确认（见 IMP-3）。
- **valid_rate 含义**：若训练路径同样 0 loop，则 valid_rate ≈ 1.0（所有 budget_stop 样本有效），
  **远超 >= 0.50 目标**——但这是 512 token 单点结论，1024/2048 预算需补测。

## 决策

- 选「误报最低且能抓住明显循环」的配置作为 `l2.rollout.loop_periods` / `loop_min_len`。
- 若最低误报配置仍误杀正常 CoT：不放松 detector，转向采样侧治理并记录。

## GPU 验证状态

- 真实 rollout 100 条：**已完成（服务器 2×RTX PRO 6000，cuda:0，189s）**。
- 尾部周期 loop 检测 + Final Answer marker 检查：**已完成，0%**。
- 人工标注（逐条内容判定）：**待**（无 ground-truth 标签，未计算 FP/FN）。
- 1024/2048 预算补测 + 实际训练路径（run_refresh_phase）复跑：**待（IMP-3）**。
