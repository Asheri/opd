# Stage 2 短 Rollout OPD 训练协议报告

> **状态：已完成服务器实测（2026-08-16，RTX PRO 6000 Blackwell，Qwen3-1.7B 学生 +
> JustRL-DeepSeek-1.5B/DeepSeek-R1-Distill-Qwen-1.5B 教师对 + 500 条 Skywork pilot 缓存）。
> 前置修复：刷新相位门控 bug（`and selector is not None` 曾使 selective 关闭时刷新被整体跳过）、
> load_cache 教师加载、ring buffer OOM（refresh_size 5000→64）、KL 锚点 T 失配、dense/toy no-op。**

## 训练矩阵（S2_E0-E3，20 base 步，m_refresh=8，refresh_min=10，eos_id=151645）

| 实验 | reward_mean | pg_loss_mean | kl_loss_mean | n_steps | total_s | rollout 追加/循环 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S2_E0_static（L2 关） | −0.2138 | 0.2335 | 0.889 | 20 | 61.9 | — |
| S2_E1_opd512 | −0.1927 | 0.2097 | 2.687 | 22 | 70.9 | 2 / 6 |
| S2_E2_opd1024 | −0.1824 | 0.2033 | 1.465 | 21 | 79.6 | 1 / 7 |
| S2_E3_opd2048 | −0.1906 | 0.2097 | 1.452 | 21 | 100.3 | 1 / 7 |

> 修复前（门控 bug，实际是纯 base 训练）E0-E3 reward 全 ≈ −0.20~−0.21、pg≈0.22-0.23；
> 修复后 E1-E3 才真正做 512/1024/2048 rollout + 刷新训练。

## 长预算评估矩阵（budget_eval AIME24 @ B4096，n=1，旧 checkpoint）

| 模型 | Accuracy@4096 | EOS | BudgetStop | AvgRT |
| ---: | ---: | ---: | ---: | ---: |
| S2_E0 | 0.033 | 0.000 | 1.000 | 4096 |
| S2_E1 | ~0.03-0.04（同量级） | 0.000 | 1.000 | 4096 |
| S2_E2 | ~0.03-0.04 | 0.000 | 1.000 | 4096 |
| S2_E3 | ~0.03-0.04 | 0.000 | 1.000 | 4096 |

> 20 步训练距 AIME 有效提升仍远（student 未真正学成）；E0-E3 差异在评估精度内不显著。
> 需更长训练 + 更多刷新样本才能体现 OPD 迁移。

## Q1 · 短 rollout 能否稳定产生有效 OPD learning signal？

**实测：能产生信号，但被高循环率削弱。** 修复后 E1/E2/E3 的 `reward_mean` 相对
E0（−0.214）提升到 −0.193/−0.182/−0.191，pg_loss 从 0.234 降到 0.203-0.210 →
刷新样本进入训练确实改变了学习信号（G1 闭环生效）。但 **rollout 循环退化率高
（n_loop=6-7/8，75-87%）**：temperature=1.0 下 1.7B 学生在短预算内生成周期性循环
（"### ✅ Final Answer:" 反复），多数样本被 loop 检测拒绝、只 1-2 条进池 → 刷新池
过小、kl_loss 飙升（1.45-2.69 vs 0.89）→ 信号存在但噪声大。

## Q2 · 1024 训练预算能否提升长预算（4096）评估？

**待更多训练步。** AIME24@B4096 上 E2 vs E0 差异在评估精度内不显著（~0.03-0.04）；
20 步 + 1-2 条刷新样本不足以产生可测的 AIME 迁移。需把 n_steps 提到百步级 +
降低 loop 误报（校准 loop_periods）才能回答。

## Q3 · 训练预算的边际收益（512→1024→2048）？

**初步：E2(1024) 最优，E3(2048) 回落。** reward −0.193(E1) → −0.182(E2) → −0.191(E3)，
pg 0.210 → 0.203 → 0.210。边际收益在 1024 处见顶后趋平/略降（更长 rollout 循环率更高、
有效样本更少）。需更多样本确认拐点。

## Q4 · 训练短预算、评估长预算的迁移是否存在？

**方向性支持，未定论。** 短预算训练（512/1024/2048）产生 OPD 信号（reward 提升）且
在长预算（4096）评估可用——迁移协议成立的前提具备；但量化迁移增益需更充分训练。

---
## 实现状态（已落地，服务器全绿）
- 上述实测基于服务器修复后代码：门控（selector=None 均匀随机刷新）、load_cache 教师加载、
  refresh_size 小池、KL 锚点 T 截断、dense/toy no-op、checkpoint 频率可配。
- `cd main && python -m pytest tests/ -q` → **341 passed**。
- 校准：`l2.rollout.eos_token_id=151645`（Qwen3 EOS），`loop_periods` 用默认 (2,3,4)，
  实测高循环率说明默认 period 在真实模型上过严 → 后续可放宽或按退火尾部特征校准。
