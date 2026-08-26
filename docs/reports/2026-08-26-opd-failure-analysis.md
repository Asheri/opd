# 2026-08-26：OPD 失败归因分析（B2048 决定性实验后）

> **状态：分析完成；判别实验 E-0/E-1 待服务器执行（零/低成本，判据写死）**。
>
> 输入事实（2026-08-26 用户提供，B2048 + chat 统一协议）：
> - **Base − E2 = 0.116 ≥ 0.05 → H9（预算错位/截断假象）排除**；
> - E2 `no_answer` 仅 **1.2%**（截断假象不成立：模型在 2048 预算内完成推理并给出答案，就是错得多）；
> - 绝对值与 eos_rate / avg_reasoning_tokens 待回填 `/root/autodl-tmp/chat_retest/B2048/all_results.json`（不伪造，先记差值）。

---

## 1. 失败的准确表述

按论文目标（把教师 RL 诱导的改进迁移到学生**下游能力**）：**复现失败**。

| 指标 | 值 | 判定 |
|---|---|---|
| eval_reward（固定集 E[Δ_T]） | -0.017 → **+0.51**（E1/E2 双双"✅ 通过"） | 训练目标达成 |
| MATH500（B2048 + chat，对齐训练协议） | E2 比 Base **低 0.116**（B512 口径 -0.108，同向） | 真实目标反向 |

**两个指标符号相反**。这不是"优化不动"，是**优化方向与目标相反**——训练在主动损害基座能力。B2048 + chat + no_answer 1.2% 排除了所有评估侧解释（截断 H9、模板协议、budget 错位）后，原因只能在训练侧：**信号、数据、或正则化**。

---

## 2. 核心证据链（全部有出处）

| # | 事实 | 出处 |
|---|---|---|
| F1 | B2048+chat 下 E2 仍 -0.116，no_answer 1.2% | 用户报告 2026-08-26 |
| F2 | `eval_reward` = 固定 holdout（同一批 500 条的尾部子集）上 `Σ π_cur(v)·Δ(v)`（学生 top-K 支撑内），与训练目标**同分布同口径** | `scheduler.py::_eval_holdout` |
| F3 | 训练/评估数据 = **500 条 base 预生成静态 response，重放消费**（batch4×300步=1200 槽位；refresh 按 m_refresh=8/refresh_min=10 默认仅贡献 ~240 条，待 0d 核实实值） | C3 审计 + run 配置 |
| F4 | 原始 Direct-OPD：Δ_T "**作用于更强 student 自身 on-policy 状态**"（verl actor 当前 rollout 批上算 rm）；本项目 L0/L1 改为离线固定 D——这是**自述的复现偏差** | `README.md` 论文抽取表 + TECHNICAL_REPORT「离线固定 D 的 L0/L1 改动」 |
| F5 | D1：kl=0.5 下 eval_reward **下降**（-0.42→-0.52）；D2：kl=0.5/0.1 均不升，**kl=0.02 转正** → 以此选定 0.02 | 诊断报告 D1/D2 |
| F6 | 实际 response token 上 Δ 均值 ≈ **-0.12**（轻微负、25% 正）；teacher top-K 支撑口径 -1.159 是口径偏置 | C3 审计（修正 D3） |
| F7 | 三模型词表一致（151643，Qwen 系）→ 教师对是同族（JustRL-DeepSeek-1.5B ≈ R1-Distill-Qwen-1.5B + RL），跨词表错位已排除 | C3 审计 |
| F8 | kl=0.02 是**用 eval_reward 这个代理指标选出的**；"H2 确诊（KL 压制）"的判据也是它 | 诊断报告 D2 |

---

## 3. 机制分析：为什么 eval_reward↑ 同时能力↓

### 3.1 训练到底在优化什么（代码事实）

`pg_loss = -E_{π_old^renorm}[min(ratio·Δ_T, clip·Δ_T)]`，其中 ratio = π_cur/π_old，
数据是 **500 条固定的 base 生成轨迹**，Δ_T 是这两条轨迹上预计算的教师 log-ratio 缓存。
`_eval_holdout`（F2）在**同一固定分布**上度量 `E_{π_cur}[Δ_T]`。

### 3.2 关键偏差：固定 D vs 论文 on-policy（RC1，主嫌）

F4：论文里 Δ_T 作用在**学生自己的当前轨迹**上（verl actor 的 rollout 批），
策略每动一步、状态分布跟着动，Δ 在学生**实际访问的区域**起作用，自我一致。
本项目的 L0/L1 把 D 冻结在 base 的 500 条轨迹上，于是：

> 学生可以在**完全不改善（甚至损害）自由生成质量**的前提下，把固定 500 条轨迹上的
> 逐状态条件分布 sharpen 到缓存的 Δ 模式上，把 eval_reward 刷上去。

这正是观测到的符号反转的机制：**固定支撑上的条件分布靠拢 ≠ 生成分布改善**。
500 题 × ~2 epoch（F3）足够把逐状态 token 偏好记忆进去；这种记忆在新 prompt 上
不迁移，还扭曲生成分布。

### 3.3 KL 角色重估：D1/D2 的"✅"是被代理指标骗的（RC2×RC3）

- kl≥0.1 时 eval_reward 不动，当时的解读是"KL 压制了好的更新"（H2）；
  **B2048 后的重新解读**：信任域正确地**拒绝了离开数据支撑的大跳变**——
  固定集目标在 base 附近没有可用的局部上升方向（D1 里它甚至下降），
  +0.51 的"收益"只能靠大幅漂移（离开 base 分布）取得。
- kl=0.02 恰好是唯一允许这种漂移的档位，而 D2 用 eval_reward 挑中了它（F8）。
  **D2 的结论作废**；kl=0.02 不是"健康值"，是把缰绳解开的值。
- 漂移 + 500 样本重放（RC3）共同造成过拟合式损伤：B512 时 E1(0.186)<E2(0.236)
  （512 rollout 的 refresh 更少、更偏离训练分布），方向一致。

### 3.4 Δ_T 语义从未验证过与"正确性"相关（RC4，候补主嫌）

C3 证明了信号**存在且非退化**（F6/F7：词表一致、Δ≈-0.12、25% 正、无模板错位），
但**从未验证 Δ>0 的方向与答案正确性相关**。Δ>0 区 = "JustRL 的 RL 相对 R1-Distill
上调的 token 模式"——其中能力成分与风格成分（CoT 长度、格式习惯）的比例未知：

- DeepSeek-R1-Distill-Qwen-1.5B 公开 MATH500 ≈ **83.9%**（很强的推理基线）；
- JustRL-DeepSeek-1.5B 的 RL 目标**待查**（JustRL 论文）：若是能力型 RL（accuracy
  reward），Δ 应含正确性信号；若是 token 效率型 RL（缩短 CoT 保精度），Δ>0 ≈ "更短"，
  对 1.7B 学生可能是纯害——**若 Δ 的风格成分主导，任何 KL/数据/on-policy 调整都救不了**。

### 3.5 次要贡献（RC5）

delta_clip=2.0 削顶（teacher 支撑口径 32% 超 2——恰好削掉最强分歧的信号）、
top-K=256 支撑截断、refresh token_mask 的 pad_id 元数据 bug（已知，影响有限）。
不改变方向性结论。

---

## 4. 根因排序

| 根因 | 角色 | 置信度 | 判别实验 |
|---|---|---|---|
| **RC1 固定 D**（偏离论文 on-policy） | 主嫌 | 高（结构性，F2/F4 同源于它） | E-0c 拐点 + E-1b |
| **RC2 代理门控**（eval_reward 选 KL/判通过） | 方法论放大器 | 高（符号反转已证实） | 纪律修订（§6） |
| **RC3 弱 KL(0.02) + 500 样本重放** | 放大器 | 高 | E-0c 拐点 |
| **RC4 Δ_T 语义与正确性脱钩** | 候补主嫌 | 中（未测，决定性缺口） | E-1（决定性） |
| RC5 工程近似 | 噪声 | 低 | 随分支处理 |

**RC1 与 RC4 谁是主犯由 E-1b 判定**：Δ↔correct 相关性好 → RC1 主犯（on-policy 化可救）；
差 → RC4 主犯（换教师对 / 信号改造）。

---

## 5. 判别实验与修复分支（判据写死）

### E-0 零训练（先做，~2 GPU 时）

**0a 回填 B2048 全表**：抄 `all_results.json` 的 Base/E2/E1 绝对值 + eos_rate +
avg_reasoning_tokens。**E2 的 avg_rt/eos 是症状判别器**：avg_rt≈2048 且 eos≈0 →
生成变冗长（漂移症状）；短而错 → 能力损伤。

**0b 导出健全性 sanity（5 分钟，低概率但先排除）**：diff 导出目录与原始 HF 目录的
`config.json`/`generation_config.json`；抽 5 题用 HF transformers generate 对比
导出模型 vs vLLM 结果（排除导出/加载损坏）。

**0c 拐点扫描（核心）**：E2 现存 5 个关键 checkpoint（step_20/60/120/200/311）导出
（`export_student_ckpt.py --device cpu`）→ MATH500 B2048 chat（命令复用执行清单
文档一 Step 3b，预算改 2048）。判据：

| 观测 | 判定 | 含义 |
|---|---|---|
| step_20 已 ≤ Base−0.05 | RC1/RC4 主导 | 信号/静态数据从第一步就坏，早停救不了 |
| step_20 ≈ Base 且随 step 单调劣化 | RC3 主导 | 漂移/过拟合 → 早停 + 收紧 KL 可救 |
| 中途峰值（如 step_120） | 混合 | 最优步可用 + 早停纪律 |

**0d on-policy 占比核实**：E2 run 的 `metrics.csv` 按 `pool` 列统计 base/refresh
步数比（量化 F3 的推断）。

### E-1 决定性（~2-3 GPU 时）

**1a 教师对体检**：两教师各自 MATH500 B2048 chat（`vllm_budget_eval --models
"JustRL=...,R1Distill=..." --budgets 2048 --n-limit 500 --chat-template`，双卡并行）。
判据：`teacher_rl(JustRL) < teacher_ref(R1-Distill)` → Δ_T 的"改进方向"本身存疑 →
RC4 加权 + 必须查 JustRL 论文的 RL 目标。

**1b Δ_T ↔ 正确性相关性（本分析的 decisive experiment）**：MATH500 抽 200 题 ×
base 采样 4 条/题（T=1.0，chat，B2048）→ 两教师各一次 forward 算序列级 Δ → sympy
判分 → Spearman(Δ_seq, correct) + 样本级 AUC。零件全有（`load_problems`/`wrap_chat`/
`extract_final_answer`/`_grade_answer_sympy`/vLLM `prompt_logprobs`），
脚本 `delta_correctness_corr.py` 待写（~200 行，本轮未写，分支确认后补）。

| Spearman ρ | 判定 | 分支 |
|---|---|---|
| ρ ≥ 0.2 | 信号有效，RC1 主犯 | **分支 A** |
| 0.05 ≤ ρ < 0.2 | 弱信号 | **分支 B2** |
| ρ < 0.05 | 信号无效，RC4 主犯 | **分支 B1** |

### E-2 修复分支

**分支 A（信号有效，漂移主导）——回到论文的 on-policy 语义**：
- A1 **on-policy 化**：refresh 变主食（`refresh_min_interval=1~2`、`m_refresh=batch`、
  静态池只做前 N 步 warmup）。L2 全链路基建已在（vLLM rollout + weight sync +
  ring buffer + teacher Δ 存 buffer），只是配比反了——这是**离实现最近的修复**。
- A2 **KL 重调**：0.1/0.5 × 120 步（0.02 作废，F8），判据 = 下游 100 题子集每 50 步。
- A3 数据 500→2000（执行清单文档二 Step 3 已备好命令）。

**分支 B1（信号无效）——换教师对**：`Qwen3-1.7B-Base`（ref）↔ `Qwen3-1.7B`
（RLHF 后正式版）。Δ_T 定义在学生同族同代分布上（词表同源、无跨代风格差），
是最正统的 Direct-OPD 前提恢复。成本：下载 ~7GB×2（磁盘够，119GB 可用）。

**分支 B2（弱信号）——信号改造**（执行清单文档二 Step 4 的具体化）：
序列级 Δ（aggregate 到 response 级）+ correctness 混合权重
（`reward = α·correct + β·Δ_seq`），退化为教师先验加权的 RAFT/最优-n 蒸馏——
稳健但偏离论文，作为兜底。

**止损线**：任何分支先跑 120 步，下游 100 题子集不升即停，不恋战。

---

## 6. 门控纪律修订（立即生效）

1. **一切训练通过判据换成下游指标**：MATH500 B2048 chat（训练中用 100 题子集快速版
   每 50 步；终验全量 500）。eval_reward 从"通过判据"降级为"诊断量"。
2. **eval_reward 的新语义——漂移报警器**：训练中 eval_reward↑ 同时下游不升/降 =
   正在离开数据支撑的信号，立即触发人工检查（本次失败的模式特征）。
3. D2 的 kl=0.02 结论作废；KL 档位由下游指标重选。
4. 旧结论的重新标注：`2026-08-25-opd-signal-diagnosis.md` 中"H2 确诊"应读作
   "H2 在代理口径下成立"；H4（Δ_T 信号）**并未被 C3 完全排除**——C3 排除的是
   "信号不存在/方向深度负/模板错位"，"信号与正确性相关"从未验证（E-1b 补）。

---

## 7. 本文档之前的执行记录

- B2048 决定性实验由用户执行（2026-08-26），结果见顶部输入事实。
- H9 判定表（执行清单文档二 Step 0）按写死判据走"排除"行：停下回查训练/信号——
  即本文档。后续 Step 1-4 的原门控冻结，按本文档 §5 顺序执行。
