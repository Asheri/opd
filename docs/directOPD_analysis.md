# Direct-OPD 论文结构化总结（方法论 + 实验设计重点）

> 论文：*Weak-to-Strong Generalization via Direct On-Policy Distillation* (Direct-OPD)
> 作者：Shiyuan Feng, Huan-ang Gao, Haohan Chi, Hanlin Wu, Zhilong Zhang, Zheng Jiang, Bingxiang He, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou（Tsinghua AIR & ByteDance Seed）
> 版本：arXiv:2607.05394v2 (v2: 2026-07-08; paper Date: 2026-07-09) ｜ DOI: 10.48550/arXiv.2607.05394
> 适用场景：作为上下文参考直接提供给其他 agent 使用。
> 来源标注约定（4 级）：`[原文明确]` / `[基于论文推测]` / `[未提供]` / `[分析者补充]`

---

## 0. 结论先行（TL;DR）

- **核心贡献**：Direct-OPD 把"在弱小模型上跑 RL"产生的**策略偏移（policy shift）** `ΔT(y|x) = log π_T(y|x) − log π_Tref(y|x)` 当作**稠密隐式奖励（dense implicit reward）**，在**更强学生自己的 on-policy 状态**上复用，而**不**去模仿弱教师的最终策略，也**不**在目标模型上跑稀疏奖励 RL。`[原文明确]`
- **关键优势**：弱到强（weak-to-strong）可迁移——即使学生初始能力**已超过** post-RL 教师，仍被提升；跨教师对、跨模型族成立；并支持多策略偏移的顺序组合。`[原文明确]`
- **效率**：Qwen3-1.7B 在 **8×A100、约 4 小时** 内 AIME 2024 从 **48.3% → 58.3%**，效果可对标在 32×A100 上直接 RL 一周的 Polaris，但算力低约一个数量级。`[原文明确]`
- **失败条件（局限）**：当且仅当教师/参考的改进信号在**学生实际访问到的状态**上无意义时会失效；最佳 response length 与 KL 强度仍依赖具体的 teacher–student 配对。`[原文明确]`

---

## 1. 论文元信息与类型分类（Module: 分类）

| 字段 | 内容 |
|---|---|
| 标题 | Weak-to-Strong Generalization via Direct On-Policy Distillation |
| 主题领域 | LLM post-training / RLVR / 知识蒸馏 / weak-to-strong generalization |
| **论文类型** | **hybrid（理论推导 + 大规模实验验证）** `[原文明确]`：含 policy-as-reward 恒等式的解析推导（§2.2–2.3），以及 RQ1–RQ3 + §4 分析的系统性实验。 |
| 理论部分 | KL 正则 RL 的最优闭式解 → policy-shift = implicit reward（Eq.6–7, 9） |
| 实验部分 | 2 个教师对 × 至多 3 个学生族，AIME24/25 评测，含 matched-step 与顺序组合对照 |
| 关键配方 | 在弱模型（1.5B）上跑 GRPO RL，用 `(π_T, π_Tref)` checkpoint 对生成隐式奖励，喂给 1.7B–7B 学生做 on-policy 蒸馏 |

---

## 2. 研究假设与核心研究问题（Module: Hypotheses / RQs）

### 2.1 中心假设（Central Hypothesis）
> 弱模型 RL 跑出的**最终策略**混杂了"RL 带来的有用改变"与"小模型自身能力上限"；正确迁移对象应是**RL 引起的策略偏移本身**（即 `log π_T − log π_Tref`），而非 `π_T` 的绝对分布。`[原文明确，§1, §2.2]`

支撑该假设的两个论点：
1. **数学等价性**：在 KL 正则 RL 下，policy/reference 对数比在常数意义下恢复 reward（Eq.6–7），故一对弱模型 checkpoint 直接把 RL 监督信号存进了 policy 空间。`[原文明确，§2.2]`
2. **解耦能力上限**：减法丢弃了弱模型 RL 前就偏好的部分，只保留 RL 改变的部分，因此不会把小模型的能力天花板一起蒸馏进学生。`[原文明确，§1]`

### 2.2 三个研究问题（RQ）
| RQ | 研究问题 | 对应章节 |
|---|---|---|
| **RQ1** | 小教师的 RL 策略偏移能否提升**已匹配或超越**教师能力的学生？是否跨不同教师对、不同学生族成立？ | §3.1, Fig.2, Tab.1 |
| **RQ2** | 在**固定 RL 步数预算**下，小模型跑 RL + Direct-OPD 迁移，是否在**精度与算力**上都优于直接在大模型上跑 RL？ | §3.2, Fig.3 |
| **RQ3** | 多个独立学到的策略偏移能否**顺序组合**到同一学生上以累积增益？ | §3.3, Fig.4 |

---

## 3. 符号体系与关键定义（Module: Notation）

> 全部符号以论文 §2.1–2.3 为准。约定：**标量斜体** `x`、`y`；**策略（分布）** 用 `π(·)`；前缀 `s_t = (x, y_{<t})`；序列长度 `T`。`[原文明确]`

| 符号 | 含义 | 来源 |
|---|---|---|
| `x ∼ D` | prompt，从数据分布 `D` 采样 | §2.1 |
| `y = (y_1,…,y_T)` | 生成的 response（token 序列） | §2.1 |
| `s_t = (x, y_{<t})` | 第 `t` 步前缀（prefix） | §2.1 |
| `π_θ` | 正在训练的学生策略 | §2.1 |
| `π_S` | 学生初始化策略（KL 锚点 reference） | §2.1 |
| `π_Tref` | 弱模型 **RL 前** 的参考策略（pre-RL reference） | §2.1 |
| `π_T` | 弱模型 **RL 后** 的教师策略（post-RL teacher） | §2.1 |
| `Δ_T(y\|x)` | 序列级策略偏移：`log π_T(y\|x) − log π_Tref(y\|x)`（Eq.5） | §2.2 |
| `r_t(v)` | token 级即时偏移：`log π_T(v\|s_t) − log π_Tref(v\|s_t)`（Eq.10） | §2.3 |
| `S_t = TopK_v π_θ(v\|s_t)` | 学生在该前缀的 top-k 支撑集 | §2.1, Eq.11 |
| `p̄^S_t_t(v)` | 学生在 `S_t` 上的 renormalized 分布（Eq.11） | §2.3 |
| `α > 0` | 学生侧 KL 惩罚系数（Direct-OPD 的显式 actor KL） | §2.3 |
| `β > 0` | 教师侧 KL 正则惩罚（KL-regularized RL 所用） | §2.2 |
| `r_T(x,y)` | 隐含在 `(π_T, π_Tref)` 背后的 latent reward（Eq.7） | §2.2 |
| `Z_T(x)` | per-prompt 配分函数常数（Eq.6–7） | §2.2 |

---

## 4. 方法论与算法架构（Module: Methodology）【重点详细】

### 4.1 核心思想：policy shift 即 implicit reward（§2.2）
**从基础定义出发的 zero-skip-step 推导** `[原文明确，Eq.6]`：

对 reward `r`、参考策略 `π_ref`、惩罚 `β>0`，KL 正则目标为
$$ \max_{\pi}\; \mathbb{E}_{y\sim\pi}\!\left[\,r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\,\right]. $$
其对 `π(y|x)` 的解是闭式最优：`π* ∝ π_ref · exp(r/β)`。两边同除以 `π_ref` 后取对数得
$$ \log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} = \frac{1}{\beta}r(x,y) - \log Z(x). \tag{6} $$
由于 `log Z(x)` 对同一 prompt 的所有 response 为常数，policy/reference 对数比在"正尺度 + 每 prompt 常数"意义下**恢复 reward**——这正是 DPO 的同一恒等式，本文**反向使用**：已知 `π_T` 与 `π_Tref`，反读出类 reward 信号：
$$ Δ_T(y\mid x) = \frac{1}{\beta}r_T(x,y) - \log Z_T(x). \tag{7} $$
即 `Δ_T` 可解释为教师的 **implicit reward**（up to 正尺度与每 prompt 常数）。`[原文明确]`

**为什么对（why it's correct）**：该恒等式不依赖任何具体 reward model，只依赖"教师是某 latent 奖励下 KL 正则 RL 的最优解"这一假设；因此一对 checkpoint 把 RL 监督信号**原样**编码在 policy 空间中，学生无需 verifiable reward、无需在目标上跑稀疏奖励 RL 即可获得该信号。`[原文明确，§2.2]`

### 4.2 形式化目标函数与最优解（§2.3）
**序列级目标（Eq.8）**：以学生初始化 `π_S` 为锚点，对策略偏移 `Δ_T` 做优化并正则化回自身初始化：
$$ J_{\text{Direct-OPD}}(\theta) = \mathbb{E}_{x\sim D}\!\left[\,\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\!\left[Δ_T(y\mid x)\right] - \alpha\, D_{\text{KL}}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_S(\cdot\mid x)\right)\,\right]. \tag{8} $$
其最优解为（Eq.9，代入 Eq.7 得 `π* ∝ π_S·exp(r_T/(αβ))`）：
$$ \pi^*(y\mid x) \propto \pi_S(y\mid x)\exp\!\left(\frac{1}{\alpha}Δ_T(y\mid x)\right) = \pi_S(y\mid x)\left(\frac{\pi_T(y\mid x)}{\pi_{T\text{ref}}(y\mid x)}\right)^{1/\alpha}. \tag{9} $$
**解读**：学生在数学上等价于"以自身初始化 `π_S` 为 reference、以小教师 implicit reward 做 KL 正则 RL"。`[原文明确]`

**token 级分解（Eq.10）**：因两个教师都在相同前缀上自回归分解，序列偏移可精确拆为 token 级偏移之和：
$$ Δ_T(y\mid x) = \sum_t r_t(y_t\mid s_t),\qquad r_t(v) = \log\pi_T(v\mid s_t) - \log\pi_{T\text{ref}}(v\mid s_t). \tag{10} $$
`r_t(v)>0` 表示教师 RL 鼓励该 token，`<0` 表示被抑制；故 `r_t(v)` 可视为**稠密、即时的 per-token reward**。`[原文明确]`

### 4.3 top-k 动作约束 + Rao–Blackwellized 梯度（§2.3）
- **top-k 限制（Eq.11）**：在每个访问前缀，仅保留学生 top-k 支撑 `S_t`，并 renormalize：`p̄^S_t_t(v) = π_θ(v\|s_t)/Σ_{u∈S_t}π_θ(u\|s_t)`。这复用了 §2.1 的 on-policy top-k 接口。`[原文明确]`
- **朴素 MC 梯度（Eq.12）**：`∇_θ J_MC = E_{x,y}[Σ_t r_t(y_t)∇_θ log π_θ(y_t\|s_t)]`，单 token 估计、方差高。`[原文明确]`
- **Rao–Blackwellized 梯度（Eq.13）**：用受限分布 `p̄_t` 对每步动作求期望（仍在轨迹 `y∼π_θ` 上采样），消除 token 采样方差、保留轨迹分布不变：
  $$ ∇_θ J_{\text{analytical}} = \mathbb{E}_{x,y\sim\pi_\theta}\!\left[\sum_t\sum_{v\in S_t}\underbrace{\bar p_t(v)}_{\text{weight}}\;\underbrace{r_t(v)}_{\text{reward}}\;∇_θ\log\pi_\theta(v\mid s_t)\right]. \tag{13} $$
  `[原文明确]`

### 4.4 Stop-gradient 系数（§2.3）
权重 `p̄_t(v)` 经 softmax 依赖 `θ`，若对其求导会注入 top-k 分布的额外 Jacobian 项，破坏 policy-gradient 形式。因此对加权 reward **detach**：
$$ A^w_t(v) = \text{stop\_gradient}\!\left(\bar p_t(v)\cdot r_t(v)\right). \tag{14} $$
最终局部 top-k 代理目标（Eq.15）：
$$ ∇_θ J_{\text{Direct-OPD}} \approx \mathbb{E}_{x\sim D, y\sim\pi_\theta}\!\left[\sum_t\sum_{v\in S_t} A^w_t(v)\,∇_θ\log\pi_\theta(v\mid s_t)\right] - \alpha\,∇_θ D_{\text{KL}}\!\left(\pi_\theta\|\pi_S\right). \tag{15} $$
KL 项实践中用 **verl** 的标准 KL-penalty 实现锚定 `π_θ` 到 `π_S`，不做解析式微分。`[原文明确]`

### 4.5 自适应 KL 控制（§2.4）
**问题**：`Δ_T` 的尺度由教师训练时的 `r_T/β` 决定，二者都固化在 checkpoint 对中且**不可恢复**；而 `α` 是学生侧 KL 系数，二者尺度无先验换算，单一固定 `α` 无法跨配对校准。`[原文明确]`

**控制器（Eq.16）**：令 `r̄_m` 为第 `m` 步 batch 内（访问前缀 × top-k 候选）学生加权偏移的均值（即梯度实际优化的稠密 reward），actor 更新前：
$$ \alpha_{m+1} = \text{clip}\!\left(\alpha_m\big(1+\epsilon\,\text{sgn}(\bar r_m)\big),\,\alpha_{\min},\alpha_{\max}\right),\quad \text{sgn}(0)=0. \tag{16} $$
默认 `ϵ=0.01`，`[α_min, α_max] = [0.5, 2.5]`。`[原文明确]`
**语义**：`r̄_m>0`（教师 RL 平均提升了保留 token）→ 增大 `α` 抑制过放大；`r̄_m<0` → 减小 `α` 削弱锚点，让梯度把概率移离被教师抑制的 token。这与"朝目标 KL 值收敛"的标准 in-reward 自适应控制器**不同**——它由稠密 reward 的**符号**驱动、作用在显式 actor KL 系数上。`[原文明确]`

---

## 5. 实验设计（Module: Experimental Design）【重点详细】

### 5.1 数据来源与预处理（§3, Appendix A）
| 项目 | 设置 | 来源 |
|---|---|---|
| 训练数据 | **Skywork-OR1-RL-Data 的数学子集**（随 Skywork-OR1 [15] 发布）；并验证用 **DAPO-Math-17K** 替换时趋势一致 | `[原文明确，Appendix A]` |
| 教师 RL 数据 | 小教师 RL 在 **DAPO dataset** [14] 上训练（§3.2 的 R1-Distill-1.5B RL） | `[原文明确，§3.2]` |
| Prompt 模板 | **DAPO-style** 数学模板（要求 "Answer: $Answer" 末行）；与教师 RL 用的 boxed-answer 模板不同，文中称 DAPO 模板迁移略好 | `[原文明确，Appendix A]` |
| 预处理 | 未提供额外清洗；仅数据来源/模板替换的消融（"result not specific to a single training dataset"） | `[基于论文推测：仅做了数据集替换对照]` |

### 5.2 模型与教师/学生对（实体索引）
| 角色 | 模型 | 备注 |
|---|---|---|
| **Teacher ref (Pair1)** | R1-Distill-1.5B | pre-RL 参考 |
| **Post-RL Teacher (Pair1)** | JustRL-1.5B [2] | R1-Distill-1.5B 经 JustRL 配方 RL 后 |
| **Teacher ref (Pair2)** | Nemotron-1.5B | pre-RL 参考 |
| **Post-RL Teacher (Pair2)** | QuestA-Nemotron-1.5B [7] | 不同训练管线/数据源（robustness 检查） |
| **Student** | R1-Distill-7B | 初始已超 post-RL 教师（weak-to-strong 硬测试） |
| **Student** | Qwen3-1.7B | 主学生 |
| **Student** | Qwen3-4B | 初始已超 post-RL 教师 |
| **Student (RQ2)** | Qwen3-1.7B-nonthinking / Qwen3-4B-nonthinking | 用于 matched-step RL 对照 |

### 5.3 评估指标与协议（Table 2）`[原文明确，Appendix A]`
| 设置 | 值 |
|---|---|
| Benchmarks | **AIME 2024, AIME 2025** |
| Samples per problem | **32**（报告 `ave@32`） |
| Sampling temperature | **0.7** |
| Top-p | **0.95** |
| Maximum generation length | **31,744** |

> 主指标为 AIME24/25 的 `ave@32` 准确率；训练曲线横坐标为 Training Step。

### 5.4 训练超参设置
**Direct-OPD 训练超参（Table 3）** `[原文明确]`：
| Hyperparameter | Value |
|---|---|
| 训练框架 | **verl** |
| Global batch size | 64 |
| Mini batch size | 64 |
| Rollout n | 4 |
| Max prompt length | 1,024 |
| Max response length | **2,048**（short-horizon；§4.2 证明可泛化到长 rollout） |
| Sampling temperature | 1.0 |
| Top-p | 1.0 |
| Learning rate | **1×10⁻⁶** |
| Training steps | **300** |
| KL coefficient | **[0.8, 2]**（不同 student–teacher 对；否则用 adaptive KL） |
| Student top-k support | **16** |
| Top-k strategy | Student top-k |

**RL 教师 / 直接 RL 基线设置（Table 4，GRPO）** `[原文明确]`：
| Hyperparameter | R1-Distill RL (1.5B & 7B) | Qwen3-nonthinking RL (1.7B & 4B) |
|---|---|---|
| Algorithm | GRPO | GRPO |
| Train batch size | 512 | 128 |
| PPO mini batch size | 128 | 128 |
| Rollout n | 8 | 8 |
| Max prompt length | 2,048 | 2,048 |
| Max response length | 16,384 | 16,384 |
| Temperature | 1.0 | 1.0 |
| Learning rate | 1×10⁻⁶ | 1×10⁻⁶ |
| KL coefficient | **0.0** | **0.0** |
| Clip high / low | 0.28 / 0.2 | （未列 clip，[未提供]） |

### 5.5 计算资源（§3.2）`[原文明确]`
| 阶段 | 算力 | 时长 |
|---|---|---|
| R1-Distill-1.5B RL（1500 步） | 32×A100 | ≈160 h |
| R1-Distill-7B 直接 RL | 32×A100 | ≈320 h |
| Direct-OPD 迁移阶段 | **8×A100** | **≈4 h**（相对 RL 成本可忽略） |

### 5.6 实验矩阵（RQ → 图/表 映射）
| RQ | 设置 | 教师对 | 学生 | 主图/表 |
|---|---|---|---|---|
| RQ1 | 弱到强迁移 | Pair1 + Pair2 | R1-Distill-7B, Qwen3-1.7B, Qwen3-4B | Fig.2, Tab.1 |
| RQ2 | matched-step 对照 | Pair1（小教师 step 300–1500） | R1-Distill-7B / Qwen3-4B-nonthinking | Fig.3 |
| RQ3 | 顺序组合 | Pair1 → Pair2 | Qwen3-1.7B | Fig.4 |
| 分析 | top-k overlap | Pair1 + Pair2 | R1-Distill-7B, Qwen3-1.7B | Fig.5, Fig.6/10 |
| 分析 | response length | Pair1（512/2k/4k） | Qwen3-1.7B, R1-Distill-7B | Fig.7, Fig.8 |
| 分析 | KL 敏感性 | Pair1 + Pair2（fixed vs adaptive） | R1-Distill-7B, Qwen3-1.7B | Fig.9 |

---

## 6. 实验结果（Module: Results）

### 6.1 RQ1 — 弱到强泛化（matrix-style result table）`[原文明确，Tab.1]`
> 列说明：**Init**=学生初始准确率；**+DOPD**=Direct-OPD 后；**Δ**=提升；**T-ref**=教师 RL 前；**T-RL**=post-RL 教师。所有值为 `ave@32` 准确率（百分比 %，按原表 Tab.1 保留）。

| Teacher Pair | Student | Benchmark | Init | +Direct-OPD | Δ | T-ref | T-RL |
|---|---|---|---|---|---|---|---|
| **JustRL** (R1-1.5B→JustRL-1.5B) | Qwen3-1.7B | AIME24 | 48.3 | **58.3** | **+10.0** | 28.5 | 51.3 |
| | Qwen3-1.7B | AIME25 | 36.8 | **43.2** | **+6.4** | 24.0 | 37.5 |
| | Qwen3-4B | AIME24 | 72.5 | **77.6** | **+5.1** | 28.5 | 51.3 |
| | Qwen3-4B | AIME25 | 65.6 | **68.8** | **+3.2** | 24.0 | 37.5 |
| | R1-Distill-7B | AIME24 | 56.7 | **63.1** | **+6.4** | 28.5 | 51.3 |
| | R1-Distill-7B | AIME25 | 40.5 | **48.8** | **+8.3** | 24.0 | 37.5 |
| **QuestA** (Nemotron-1.5B→QuestA-Nemotron-1.5B) | Qwen3-1.7B | AIME24 | 48.3 | **59.0** | **+10.7** | 61.77 | 72.50 |
| | Qwen3-1.7B | AIME25 | 36.8 | **43.1** | **+6.3** | 49.50 | 62.29 |
| | R1-Distill-7B | AIME24 | 56.3 | **61.2** | **+4.9** | 61.77 | 72.50 |
| | R1-Distill-7B | AIME25 | 39.5 | **44.0** | **+4.5** | 49.50 | 62.29 |

**要点**：R1-Distill-7B（56.7/56.3）与 Qwen3-4B（72.5）初始**已高于** JustRL 教师（51.3），仍被提升 → 证明迁移的是"RL 方向"而非"模仿教师最终策略"。QuestA 对来自不同管线/数据，进一步证明效应非单一教师族专属。`[原文明确，§3.1]`

### 6.2 RQ2 — 超直接 RL（matched-step）`[原文明确，§3.2, Fig.3]`
- 在**相同 RL 步数**下：先在小模型（R1-Distill-1.5B）跑 RL（checkpoint step = 300/600/900/1200/1500），再迁移到 R1-Distill-7B，其 AIME25 表现**优于**直接在 R1-Distill-7B 上跑 RL。
- 算力上：1500 步 1.5B RL ≈160h/32A100，7B RL ≈320h；Direct-OPD 迁移仅 +4h/8A100，可忽略。
- Qwen3 非思考模型验证：1.7B 跑 100 步 RL 后迁移到 Qwen3-4B-nonthinking，达到直接 4B RL 的 0.635 水平（AIME25）。`[原文明确]`

### 6.3 RQ3 — 顺序组合（Fig.4）`[原文明确，§3.3]`
| Stage | AIME24 | AIME25 |
|---|---|---|
| Initial (Qwen3-1.7B) | 48.3 | 36.8 |
| After JustRL shift | 58.3 (**+10.0**) | 43.2 (**+6.4**) |
| After QuestA shift (sequential) | **63.8 (+15.5)** | **46.8 (+10.0)** |

> 第二阶段（QuestA 信号）从 JustRL 阶段 checkpoint 继续训练，全局步 300–600；stage 边界的小不连续来自独立采样的评估方差。

### 6.4 分析结论速览
- **§4.1 跨模式迁移**：Direct-OPD **不要求** teacher–student top-k overlap 渐进式提升（Fig.5）；cross-pattern 迁移时 overlap 仍低，且 actor 熵不坍塌（Fig.6/10）→ 迁移的是"学生自身支撑集内的局部策略偏移方向"，而非整份教师策略。`[原文明确]`
- **§4.2 短视界泛化**：2k response length 在 Qwen3-1.7B 与 R1-Distill-7B 上验证最稳（Fig.7）；40 步 2k 训练后，actor 在 ~16k 长 rollout 上已朝教师偏移方向移动（Fig.8）；6k 在晚期不可靠前缀上过驱动、验证反而更差（45.6 vs 2k 的 48.8）。`[原文明确]`
- **§4.3 KL 可靠性**：最佳固定 KL **依赖配对**（Fig.9）；稠密 token reward 不能脱离 rollout 分布独立最大化（大的正 reward 可能对应更差验证）；adaptive KL 在初始校正后把均值 reward 拉向 ~0 的平衡区。`[原文明确]`

---

## 7. 局限与结论（Module: Limitations / Conclusion）

- **结论**：可迁移对象不是 post-RL 教师策略，而是其相对自身 pre-RL 参考的**对数比**，在**学生访问到的 token** 上评估——这正是更小更弱的教师仍能提升更强学生的原因。该方向跨教师对/学生族成立、在远低于直接 RL 算力的成本下优于 step-matched 直接 RL、且可跨教师组合。`[原文明确，§6]`
- **局限**：信号是**条件性**的——当教师/参考的改进在学生访问状态上无意义时 Direct-OPD 会失效；最佳 response length 与 KL 强度仍依赖具体 teacher–student 配对。`[原文明确，§6]`
- **与 OPD 家族的差异**：标准 OPD 蒸馏教师最终策略（弱教师会带入能力天花板）；Direct-OPD 只蒸馏 `log π_T − log π_Tref` 的偏移，丢弃教师绝对策略。`[原文明确，§5]`

---

## 8. Bilingual 术语表（Glossary）

| 中文 | English | 定义/备注 |
|---|---|---|
| 策略偏移 | policy shift | `Δ_T = log π_T − log π_Tref`，RL 引起的改变量 |
| 隐式奖励 | implicit reward | 由 policy/reference 对数比恢复的 reward（Eq.7） |
| 弱到强泛化 | weak-to-strong generalization | 弱监督者激发更强模型能力 |
| 在线策略蒸馏 | on-policy distillation (OPD) | 学生在自身采样的状态上受教师监督 |
| KL 正则强化学习 | KL-regularized RL | `max E[r] − β KL(π‖π_ref)` |
| 稠密奖励 | dense reward | per-token 即时信用分配（此处即 `r_t(v)`） |
| 参考策略 | reference policy | 此处指学生初始化 `π_S` 与教师 pre-RL `π_Tref` |
| 自适应 KL 控制 | adaptive KL control | 按稠密 reward 符号动态调整 `α`（Eq.16） |
| Rao–Blackwell 化梯度 | Rao–Blackwellized gradient | 对受限动作分布求期望降方差（Eq.13） |
| top-k 支撑集 | top-k support | `S_t = TopK_v π_θ(v\|s_t)`，限制信号作用域 |
| 停止梯度系数 | stop-gradient coefficient | `A^w_t(v)=sg(p̄_t(v)·r_t(v))`（Eq.14） |
| 顺序组合 | sequential composition | 多策略偏移依次施加（RQ3） |

---

## 9. 引用索引（§ / Table / Fig → 内容）

| 标记 | 内容 |
|---|---|
| §1 | 引言、中心问题与贡献列表 |
| §2.1 | OPD 预备知识与四策略定义、top-k KL 估计 |
| §2.2 | policy shift as implicit reward（Eq.5–7） |
| §2.3 | Direct-OPD 目标、token 分解、top-k、Rao-Blackwell 梯度、stop-gradient（Eq.8–15） |
| §2.4 | 自适应 KL 控制（Eq.16） |
| §3.1 / Fig.2 / Tab.1 | RQ1 弱到强迁移 |
| §3.2 / Fig.3 | RQ2 matched-step 超直接 RL |
| §3.3 / Fig.4 | RQ3 顺序组合 |
| §4.1 / Fig.5,6,10 | 跨模式迁移与熵诊断 |
| §4.2 / Fig.7,8 | 短视界训练泛化 |
| §4.3 / Fig.9 | KL 控制与 reward 可靠性 |
| §5 / §6 | 相关工作 / 结论与局限 |
| Appendix A / Tab.2,3,4 | 数据、评估协议、训练与 RL 超参 |
| Appendix B,C / Fig.10,11 | 额外熵诊断与 QuestA AIME25 曲线 |

---
*来源标注约定：本文件所有 `[原文明确]` 均对应论文正文/附录的 §、Table、Fig 引用；`[基于论文推测]` 为基于文本的推断；`[未提供]` 表示论文未给出；`[分析者补充]` 为跨文档整合注记。*
