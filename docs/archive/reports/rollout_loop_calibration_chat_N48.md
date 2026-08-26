# Rollout Loop Detector 校准报告（IMP-1）

> 日期：2026-08-18 ｜ 状态：已填充 ｜ 模型：/root/autodl-tmp/models/Qwen__Qwen3-1.7B ｜ N=48 ｜ max_new=1024 ｜ eos_token_id=151645 ｜ 人工标注：无

## 背景与目标

- 当前 loop 检测存在**误杀风险**：真实 Qwen3-1.7B + Skywork 短 rollout 循环退化率
  75-87%，默认 `(2,3,4)` 可能把正常长 CoT 误判为 loop（误报）。
- 目标：在配置矩阵（periods × min_len）中选择「**误报最低且能抓住明显循环**」的配置。
- 硬约束：**不得为凑 `<50%` 人为放松 detector**——若最低误报配置仍误杀正常 CoT，
  应记录并转向采样侧（temperature / repetition_penalty）治理，而非放宽检测。

## 方法学

1. 真实 rollout N 条（chat_template=True，temperature=0.7，repetition_penalty=1.0，短预算 `max_new`，新生成 token 序列去 pad）。
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

## 决策

- 选「误报最低且能抓住明显循环」的配置作为 `l2.rollout.loop_periods` / `loop_min_len`。
- 若最低误报配置仍误杀正常 CoT：不放松 detector，转向采样侧治理并记录。

## GPU 验证状态

- 真实 rollout 100 条：**待服务器 GPU**（本机无 GPU，不伪造通过）。
- 人工标注与抽样检查：**待**。
- 结果表填充后，本报告才可视为校准结论。
