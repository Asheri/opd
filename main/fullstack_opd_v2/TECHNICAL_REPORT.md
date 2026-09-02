# 全栈 OPD 叠加 · 技术文档与训练分析报告

> **本文档 = 工程实现 + 训练分析 唯一权威说明**（由原 `ENGINEERING_IMPLEMENTATION.md` 扩展而来）。
> 满足项目 CLAUDE.md「文档要求」节的六点：①工程实现（按原始论文修改后）②训练前后 benchmark
> 分数与协议 ③显存占用 ④用时 ⑤数据构成 ⑥其他必要信息。
>
> **第一部分**（§0–§4 + 附 1/2）讲**代码是怎么搭的**：线程、队列、版本号、缓存张量、张量形状、
> 函数名。不推导公式（π_old 加权、k3 恒等式等只作为张量名出现）。带 🔍 的小节是「通俗理解 +
> 具体例子」，给不熟悉异步/分布式训练的同学。适用代码：`main/fullstack_opd_v2/`。
>
> **第二部分**（§5–§9）是**实验分析报告**：benchmark 分数与协议、显存、用时、数据构成、已知
> 边界与复现。所有实验数字必须标注所用协议（本项目的教训：pass@1 与 ave@32 混报会产生误导）。

| 章节 | 内容 | 对应 CLAUDE.md 要求 |
|---|---|---|
| **第一部分** §0–§4 | 工程实现（模型/分布、端到端时序、异步、教师离线、信用分配、边界） | ① 工程实现 |
| **§5** | 训练前后 benchmark 分数 + 完整评估协议声明 | ② 训练分析 |
| **§6** | 训练与评估的显存占用分析 | ③ 显存 |
| **§7** | 训练与评估的用时分析 | ④ 用时 |
| **§8** | 训练数据构成分析 | ⑤ 数据 |
| **§9** | 已知边界、长生成失败教训、复现步骤 | ⑥ 其他 |

---

# ⚠️ P-OPD 重建（2026-08-31）：阅读本文档前的必读声明

**2026-08-31 起，训练 = 纯 on-policy 交替相位**（commit `c744c32`）：
- **删除**：stage1 预计算教师 Δ 缓存（Lightning-OPD 离线缓存）、base 池（固定 D）训练、warmup 胖 D、`cache_store.py` 磁盘缓存、v1 `fullstack_opd/` 包、`cli cache` 子命令。
- **当前架构**：学生每相位新鲜 rollout → 教师 rl/ref **only_stu 实时前向**算 Δ（学生 top-K 完整支撑 `gather−logsumexp`，无交集稀释）→ ring buffer（学生支撑）→ `_train_step_refresh` 训练（teacher-free，α 冻结 1.0 = 100% on-policy）。vLLM 权重同步走 `rollout_weight_sync=off` 逃生舱（`apply_model(load_weights)` 直拷，仅 tp=1）。

> **本文档 §0–§3 及 §4.6.1–4.6.4 描述的是"重建前"的离线缓存 + base 池架构**（历史，供理解演进）。
> **当前架构的唯一权威描述在 §4.6.5**。§5–§9 实验数字为历史产物（重建前 L0/L1/静态路径），
> P-OPD 实测待服务器跑完后补充。

---

# 第一部分 · 工程实现（按原始论文修改后的版本）

> 本文只讲**代码是怎么搭的**：线程、队列、版本号、缓存张量、张量形状、函数名。
> 不推导公式（π_old 加权、k3 恒等式等只作为张量名出现）。所有名词都能在下表文件里直接搜到。
> 带 🔍 的小节是「通俗理解 + 具体例子」，给不熟悉异步/分布式训练的同学。
>
> 适用代码：`main/fullstack_opd_v2/`。

| 工程问题 | 落地机制 | 主要文件 |
|---|---|---|
| 学生 rollout 与训练**交替相位** | rollout 相位（学生生成 + 教师 only_stu 前向）↔ 训练相位（`_train_step_refresh`），`while step_done` 驱动，α 冻结 1.0 | `pipeline.py`、`adaptive_cache.py`、`scheduler.py` |
| 教师**实时打分** | 每相位 only_stu 前向（教师对学生 top-K 完整支撑 logp 差），CPU offload 省显存 | `adaptive_cache.py`（`_rl_ref_delta_only_stu`）、`pipeline.py` |
| Direct-OPD **信用分配** | 实时 Δ_T（学生支撑）+ 分布级 PG 的逐张量操作 | `adaptive_cache.py`、`losses.py`、`scheduler.py` |
| **工程化底座（P0）** | pydantic 配置强校验 + CLI 子命令 + run 目录 + 断点续跑 + 指标追踪 + 结构化日志 + 类型异常 + 可插拔数据/模型 + AIME 评估 | `config.py`、`cli.py`、`run.py`、`checkpoint.py`、`metrics.py`、`logging.py`、`exceptions.py`、`data.py`、`model_factory.py`、`eval_aime.py` |
| **P-OPD 纯 on-policy** | 纯 refresh 交替相位（无 base 池/无预计算缓存）、ring buffer 学生支撑、空相位防护、vLLM off 逃生舱 | `adaptive_cache.py`、`scheduler.py`、`pipeline.py`、`rollout_vllm.py` |
| **预算感知评估** | 统一 reasoning token 预算 B 下公平比较（Accuracy(B) + 预算曲线/AUC/nAUC/效率） | `budget_eval.py`、`budget_curve.py` |

---

## 0. 先认清：本代码里到底有哪几种「模型 / 分布」？

（这一节防止把"教师 / 学生 / 参考分布 / 旧快照"混为一谈——它们角色完全不同。本节为 §1–§3 的实体速查。）

| 实体 | 是什么 | 出现在哪 | 在训练里干什么 |
|---|---|---|---|
| **teacher_rl / teacher_ref** | 两个**教师**模型（Stage 0 产出 post-RL 弱教师 + 其训练前副本） | 仅 Stage 1 | 跑一次 `response_dists` 算 `Δ_T = logπ_rl − logπ_ref`，之后**立刻释放**，训练期零前向 |
| **student（`self.student`）** | 被训练的学生 = **new 模型** | 全程（learner） | 当前权重，训练时前向算 `s_cur`（带梯度） |
| **rollout 快照（`self.worker` / vLLM 引擎）** | **old 模型** = student 的某一旧版本（从 `WeightStore` 加载） | RolloutCollector / Ray worker | 用旧权重前向算 `s_old`（无梯度），随样本入队 |
| **ref_dists / ref_ids / ref_logp** | **初始 student 的分布**（Stage 2 入口冻结一次的张量，**不是模型、不执行前向**） | Stage 2 入口算一次 | 仅作 **KL 锚点**（信任域），不参与算 `s_cur/s_old` |
| **Δ_T 缓存** | 教师偏好差 `(N,T,V)` 或 top-K，训练期常量 | Stage 1 构建 | 信用分配输入（= RL 里的奖励信号） |

> 🔑 **三个常被误解的点**：
> 1. **没有"ref 学生模型在 rollout"这回事**。`ref_dists` 是初始 student 分布被**冻成一张表**，训练时只拿它和 `s_cur` 算 KL；它从不做前向、不产出 `s_old`。Stage 1 里出现的 `teacher_ref` 是个**教师**，不是 student 的参考模型。
> 2. **`s_cur` 和 `s_old` 的输入完全一样**：都是 `cat([prompts[idxs], responses[idxs]])` 喂模型（`response_dists` 内部拼接）。两者的**唯一区别是权重**（当前 student vs 旧快照）。`prompts` 是上下文前缀，`responses` 是被打分的目标序列——不是"ref 的 rollout 当 y"。
> 3. **Δ_T 来自教师，不是任何学生模型**——它在 OPD 里扮演的角色等价于 RL 的"环境奖励"。

> 📌 **upshot**：本代码的"学生"只有**两种权重形态**（new = `self.student`，old = 从 `WeightStore` 加载的快照），外加一个**冻结的初始分布张量**（KL 锚点）。没有第三种"student 模型"。

### 0.5 端到端时序（按真实代码顺序，纠正"ref 先 rollout 存 D"的误解）

很多人会本能地写成「先用 ref 在 D 上 rollout、把 prompt+response 存下来」，但**本代码完全不是这个顺序**。下面按 `pipeline.py` 真实执行顺序列出每一步由谁、对谁、产出什么：

```
[构造期 · __init__]
  _make_toy_data() 一次性生成 固定数据集 D = (prompts (N,P), responses (N,T))，设备常驻
  → D 从一开始就是「输入」，不是任何模型 rollout 出来的「结果」。

[Stage 0 · stage0_small_rl]  产出两个教师
  weak  = CausalToyLM + 批量 REINFORCE 训练 n_rl_steps 步  → 教师rl（post-RL）
  ref   = 拷贝 weak 初始权重、冻结                   → 教师base（pre-RL）
  → 这一步在「学生」还没出生时就把教师备好了。

[Stage 1 · stage1_build_cache]  教师的 rollout（⚠️ 是教师，不是学生 ref）
  对【全体 D】跑 teacher_rl.response_dists / teacher_ref.response_dists
  → Δ_T = logπ_rl − logπ_ref  形状 (N,T,V)（topk 模式则 (N,T,K)）
  cache.save(...) 落盘；teacher_rl / teacher_ref 立即释放，训练期零前向
  → 这里的 rollout 产出是 Δ_T，不是 prompt+response；prompt+response 是上面 D 里现成的。

[Stage 2 入口 · 算 ref_dists]  学生 ref（初始分布，仅用于 KL）
  student = CausalToyLM（刚初始化，还没训）
  ref_dists = response_dists(student, prompts, responses) 一次前向、冻成表
  → 它「在 D 上前向一次」是为了得到初始分布做 KL 锚点，不是 rollout、不产出也不保存 (prompt,response)。
  → 此时 student 即「当前 new」；old 还不存在，要等第一步 _publish 才有 v=1 快照。

[Stage 2 · 异步训练循环]  4 线程流水线（细节见 §1）
  PromptFeeder   : 随机抽 idxs (B,)
  RolloutCollector: 从 WeightStore 取【最新 old 快照】→ 前向算 s_old（log-prob-old）→ 连同 idxs+版本号入队
  TeacherScorer  : 按 idxs 从 Δ_T 缓存 get_delta(idxs) → 连同 (idxs, s_old) 入 staleness_q
  TrainDispatcher: 取 FIFO 首条；若 age=当前版本−样本版本 > 阈值 → 丢弃；
                   否则 student(new) 按 idxs 前向算 s_cur（log-prob-new）；
                   ratio = exp(s_cur − s_old) 进 pg_loss 裁剪；
                   KL = low_var_kl(s_cur, ref_dists[idxs]) 进 loss；
                   step → _publish 推进版本（old 更新）。
```

> ✅ **一句话对照你的描述**：
> - 「学生 ref 在 D 中选 idx rollout、存 prompt+response」→ ❌。`prompt+response` 是构造期就有的固定 D；`ref_dists` 仅在 Stage 2 入口对**全 D** 前向一次做 KL 锚点；真正"对 D rollout 出 Δ_T"的是 **Stage 1 的教师**，且产出的结果是 Δ_T 而非 (prompt,response)。
> - 「教师 base/rl 算 token 级 Δ_T」→ ✅ 正确，但 Δ_T 在缓存里**始终是按 token 的 (N,T,V)**，没有被"求和成序列级标量"；序列级聚合发生在 `pg_loss` 内部（`s_old.exp()·逐词元信用` 对词表维 `sum(-1)` 得 (B,T)），不是缓存阶段做的。
> - 「actor 加载最新快照算 s_old、连同 idx+版本号缓存、buffer 双截断防死锁」→ ✅ 完全正确（即 RolloutCollector + StalenessQueue）。
> - 「scorer 按 idx 取 Δ_T 给 learner / learner FIFO 取、太旧弃用 / new 前向算 s_cur 与 s_old 比 ratio / ratio 管新旧差、KL 管偏离」→ ✅ 全部正确。

---

### 0.6 优化目标的数学模型（逐行对齐 losses.py / cache.py）

> （这一节应要求给出可被公式化的最小目标，方便二次审阅；其余章节仍只讲工程实现。符号全部对得上代码。）

**符号表**

| 符号 | 含义 | 代码 |
|---|---|---|
| $s_t=(x,a_{<t})$ | 第 $t$ 个位置的上下文（prefix） | `response_dists` 内部 `cat([prompts, responses[:, :t]])` |
| $\pi_\theta(\cdot\mid s_t)$ | 学生当前策略（下一词分布） | `s_cur`（带梯度） |
| $\pi_{\text{old}}(\cdot\mid s_t)$ | 学生行为/快照策略（算 $s_{\text{old}}$ 时的旧权重） | `s_old`（无梯度） |
| $\pi_{\theta_0}(\cdot\mid s_t)$ | **初始学生**分布（Stage 2 入口冻结） | `ref_dists` |
| $\pi_{\text{rl}},\ \pi_{\text{ref}}$ | **RL 后教师** / **RL 前教师**分布 | Stage 0 产出 |
| $\Delta_T(t,v)$ | 稠密奖励 $\log\pi_{\text{rl}}(v\mid s_t)-\log\pi_{\text{ref}}(v\mid s_t)$ | `cache.delta_k`（逐位置×逐词表） |
| $\rho_\theta(v)$ | 重要性比 $\pi_\theta/\pi_{\text{old}}$ | `ratio=(s_cur-s_old).exp()` |
| $\varepsilon=0.2,\ \beta=0.05$ | 裁剪 / KL 系数 | `clip_eps` / `kl_reg_coef` |

**目标函数（梯度期实际优化）**

$$
\min_\theta\;
\underbrace{-\frac{1}{BT}\sum_{i=1}^{B}\sum_{t=1}^{T}
\mathbb{E}_{v\sim\pi_{\text{old}}}\!\Big[\min\!\big(\rho_\theta(v)\,\Delta_T(t,v),\ \mathrm{clip}(\rho_\theta(v),1{-}\varepsilon,1{+}\varepsilon)\,\Delta_T(t,v)\big)\Big]}_{\text{IS 校正的分布级 PG}}
\;+\;
\underbrace{\beta\,\frac{1}{BT}\sum_{i,t}\mathrm{KL}\!\big(\pi_\theta(\cdot\mid s_t)\,\|\,\pi_{\theta_0}(\cdot\mid s_t)\big)}_{\text{信任域}}
$$

> **⚠️ 关于 $\mathbb{E}_{v\sim\pi_{\text{old}}}$ 的常见误读：动作 $v$ 到底从哪来？**
>
> 把公式写成 $\mathbb{E}_{v\sim\pi_{\text{old}}}[\cdots]$ 容易让人以为"批里的动作 $v$ 是从 $\pi_{\text{old}}$ 采样出来的"。在**真实工程里这句要拆成两层期望**来看：
>
> $$
> \underbrace{\mathbb{E}_{(s_t,t)\sim D}}_{\text{外层：上下文来自固定数据集 }D}\;
> \underbrace{\mathbb{E}_{v\sim\pi_{\text{old}}(\cdot\mid s_t)}[\cdots]}_{\text{内层：动作空间按旧学生分布求期望}}
> $$
>
> - **外层（上下文 $s_t$）来自固定数据集 $D$，不是从 $\pi_{\text{old}}$ 的 rollout 采样来的。** 训练时我们 teacher-force 数据集里给定的 $(prompt, response)$ 前缀，`response_dists` 只算"给定前缀下学生自己的下一词分布"，**从不自己采样生成 token**。这一点你说得对：**数据集中那个 token 确实不是 $\pi_{\text{old}}$ 采的。**
> - **内层（动作 $v$）是对"全词表"的解析期望，不是采样。** `losses.py:26` 是 `pg = -(s_old.exp() * pointwise).sum(-1)`——把 `pointwise` 在**整个词表 $V$ 上求和、用 $\pi_{\text{old}}(v)$ 当权重**，等价于 $\sum_v \pi_{\text{old}}(v)[\cdots]$。这是分类动作空间上的**闭式期望**，和 RL 里"从 $\pi_{\text{old}}$ 抽一条轨迹"完全是两回事：我们既不抽动作、也不抽上下文，而是把代理损失在整张词表上摊开。
> - **$\Delta_T(t,v)$ 覆盖全词表，所以"来自 $\pi_{\text{old}}$ 的动作"都能取到。** 教师预加载的是**稠密** $\Delta_T$：`cache.build` 里 `self.delta = rl_full - ref_full`，是完整 $(N,T,V)$ 张量（$\Delta_T(t,v)$ 对**每个** $v$ 都有值）；稀疏模式则是 top-K 支撑（支撑外填 0）。因此 $v$ 取词表里任何值——包括 $\pi_{\text{old}}$ 支持内的任何 token——$\Delta_T(t,v)$ 都现成可用。**恰恰是教师把 $\Delta_T$ 摊到全词表，才让内层"按 $\pi_{\text{old}}$ 对全词表取期望"成为可能。** "预加载后 $\Delta_T$ 只能对数据集那个 token 取值"的直觉是反的：预加载给的是全词表，受限的只是模式（稠密全给 / 稀疏给 top-K）。
> - **为什么"π_old 支撑 ⊆ V ⇒ Δ_T 必然存在"（dense 模式）：** 学生的词表本就是 $V$，故 $\pi_{\text{old}}$ 的支撑恒为 $V$ 的子集；而 dense teacher 的 $\Delta_T$ 覆盖恰为整个 $V$，于是 teacher 的覆盖**必然包含** $\pi_{\text{old}}$ 会触及的任何 $v$。这正是"不需要采样动作去查 $\Delta_T$"的根因——$\Delta_T$ 已为整词表备好，π_old 训练途中质量怎么重分布都不缺值。稀疏模式则破此保证：teacher 只存 top-K（⊂ $V$），真实信号仅存在于 `student-top-K ∩ teacher-top-K` 交集，交集外填 0（中性），属有界近似（见下方"稠密 vs 稀疏"边界）。
> - **IS 无偏性的边界**：内层恒等式 $\mathbb{E}_{v\sim\pi_{\text{old}}}[\rho_\theta(v)\Delta_T]=\mathbb{E}_{v\sim\pi_\theta}[\Delta_T]$（因 $\rho\cdot\pi_{\text{old}}=\pi_\theta$）对**内层（动作）期望**严格成立，与上下文来源无关；唯一"离线"的是**外层（上下文）期望**——RL PPO 外层是 $\pi_{\text{old}}$ rollout 诱导的状态分布，OPD 外层是固定数据集 $D$。
>
> **一句话**：$\pi_{\text{old}}$ 在这个公式里**只是内层期望的权重分布（旧学生快照对全词表的概率）**，不是"生成动作/上下文的行为策略"。数据集 $D$ 提供上下文，教师 $\Delta_T$ 提供全词表的逐 token 奖励，旧/新学生分布提供权重与重要性比——三者在词表上做加权求和，全程零采样。
>
> **稠密 vs 稀疏的精度边界**：上述"全词表期望"在 **dense 模式**（demo 默认）下严格成立；**稀疏模式**下 $\Delta_T$ 只在学生当前 top-K 支撑上非零（`delta_for_student_topk` 把 `delta` scatter 回 $(B,T,V)$，支撑外=0），所以内层实际只在 top-K 支撑上按 $\pi_{\text{old}}$ 加权求和，是受支撑限制的近似（与 `low_var_kl_support` 的"略低估"同源，均为稀疏锚点的有界近似，非恒等）。
> - **⚠️ 稀疏模式未做支撑重归一化（即问题 1 所指）**：`pg_loss` 仍对全 $V$ 求和（支撑外因 $\Delta_T=0$ 贡献为 0），等价于 $\sum_{v\in\text{top-K}}\pi_{\text{old}}(v)[\cdots]$，而**不是** $\frac{1}{\sum_{v\in\text{top-K}}\pi_{\text{old}}(v)}\sum_{v\in\text{top-K}}\pi_{\text{old}}(v)[\cdots]$。top-K 之外的尾部质量被直接丢弃、不归一；`low_var_kl_support` 同理。这是为省显存（Δ_T 只存 top-K 才不爆内存）的**有意近似**：若 top-K 已捕获 ~99% 概率质量，偏差约 1%，且 PG 与 KL 两项同尺度缩放、相对权衡不变，自适应优化器会吸收该常数因子；**dense 模式（默认）严格无此问题，是真正归一化的全词表期望**。若要更干净，可在稀疏路径分子母同除以 top-K 质量 $\sum_{v\in\text{top-K}}\pi_{\text{old}}(v)$ 恢复条件期望，但实践中尾质量可忽略。

等价地（只看优化方向）最大化

$$
\max_\theta\;\frac{1}{BT}\sum_{i,t}\mathbb{E}_{v\sim\pi_{\text{old}}}\!\Big[\min\!\big(\rho_\theta(v)\Delta_T,\ \mathrm{clip}(\rho_\theta,1{\pm}\varepsilon)\Delta_T\big)\Big]
\;-\;\beta\,\frac{1}{BT}\sum_{i,t}\mathrm{KL}(\pi_\theta\|\pi_{\theta_0})
$$

KL 用低方差、且对稠密情形**无偏**的 $k_3$ 估计：

$$
\mathrm{KL}(\pi_\theta\|\pi_{\theta_0})
=\mathbb{E}_{v\sim\pi_\theta}\!\big[e^{x}-x-1\big],\qquad x=\log\pi_{\theta_0}(v)-\log\pi_\theta(v)
$$

（稠密实现里 $k_3$ 在 $\pi_\theta$ 下取期望**恒等于**真 KL；稀疏 `low_var_kl_support` 只在 top-K 支撑上求和，才略低估——见 `losses.py` 注释。）

**核心分解（数据新鲜、$\pi_{\text{old}}=\pi_\theta$、$\rho=1$）**

此时 $\min(1\cdot\Delta_T,1\cdot\Delta_T)=\Delta_T$，单位置目标退化为

$$
J_{i,t}(\theta)=\mathbb{E}_{v\sim\pi_\theta(\cdot\mid s_t)}\!\big[\Delta_T(t,v)\big]
=\mathrm{KL}\!\big(\pi_\theta\|\pi_{\text{ref}}\big)\;-\;\mathrm{KL}\!\big(\pi_\theta\|\pi_{\text{rl}}\big)
$$

因为 $\mathbb{E}_{\pi_\theta}[\log\pi_{\text{rl}}-\log\pi_{\text{ref}}]
=\mathrm{KL}(\theta\|\pi_{\text{ref}})-\mathrm{KL}(\theta\|\pi_{\text{rl}})$（$\pi_\theta$ 的熵项两相抵消）。

于是**整个目标在 on-policy 下可写成三个 KL 的权衡**：

$$
\max_\theta\;
\underbrace{\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})}_{\text{别离 base 教师太远}}
\;-\;
\underbrace{\mathrm{KL}(\pi_\theta\|\pi_{\text{rl}})}_{\text{去贴近 RL 教师}}
\;-\;
\underbrace{\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\theta_0})}_{\text{信任域锚住起点}}
$$

直觉：让学生**比贴近 base 教师更贴近 RL 教师**——即"获取教师经过 RL 训练后的那部分改进"，这就是 Direct-OPD 名字的由来。

**各部分的工程含义（与代码一一对应）**

- **$\Delta_T(t,v)$ 是稠密奖励，且是逐词表（对 $v$）的**：它给每个候选 token 一个带方向的标量。因为对**全词表取闭式期望（不是对采样到的 action）**——详见上方 ⚠️"关于 $\mathbb{E}_{v\sim\pi_{\text{old}}}$ 的误读"框——所以 `pg_loss` 是**确定性**的、无 REINFORCE 采样方差；训练信号是"分布形状"而非"一条轨迹"。
- **$\rho_\theta$（重要性比）**：异步的数学代价。$s_{\text{old}}$ 是用旧 $\theta$ 算的快照，必须用 $\rho$ 把它的梯度折算成当前 $\theta$ 下可用；且 $\mathbb{E}_{\pi_{\text{old}}}[\rho_\theta(v)\Delta_T]= \mathbb{E}_{\pi_\theta}[\Delta_T]$（因 $\rho\cdot\pi_{\text{old}}=\pi_\theta$），所以 IS 校正**无偏**地把离策略数据还原成 on-policy 期望。
- **$\mathrm{clip}(\rho,1\pm\varepsilon)$**：PPO 对**行为策略**的信任域——限制单步更新幅度，防止一个异步步把 $\theta$ 推得离 $\pi_{\text{old}}$ 太远。对带符号的 $\Delta_T$ 取 `min` 是标准的悲观下界（A>0 裁掉上溢比、A<0 裁掉下溢比）。
- **$\beta\cdot\mathrm{KL}(\pi_\theta\|\pi_{\theta_0})$**：对**初始学生分布**的信任域，防训练崩塌；同时把学生拴在数据生成分布附近，使固定/离线数据保持有效（见 §0.5 末尾关于固定数据的讨论）。
- **staleness 双截断**：**不在可微目标里**，是数据管线的硬约束——`age=当前版本−样本版本>阈值` 直接丢弃，保证 $\rho$ 不会过旧（让上面的 IS 校正保持近似无偏）。
- **🔍 关于"IS 方差爆炸"的澄清（数据集 D 极稀疏、与学生随机策略支撑几乎零质量——即问题 2 所指）**：经典 IS 方差上界（**采样分布须覆盖目标分布支撑**）针对的是**对动作/状态做蒙特卡洛采样**的 IS 估计器。OPD 的损失里**没有这种 MC IS**：①动作维 $\Delta_T$ 是对全词表的**闭式求和**（`-(s_old.exp()*pointwise).sum(-1)`），**确定性、零 MC 方差**，$\rho_\theta$ 是在每个 token 上**解析求值**而非采样权重；②上下文维直接从固定数据集 $D$ 取样，而 $D$ **就是外层期望的目标分布**（我们在 $D$ 上优化，并非用 $D$ 去 IS 重加权到学生的自回归 rollout 分布），故上下文维也**没有 IS 比**。因此"$D$ 在学生 rollout 支撑上近零质量"这一事实**不会引发 IS 方差爆炸**——恰恰因为 OPD 从不尝试用 $D$ 去估计学生 rollout 分布的期望，也就无需 $D$ 覆盖该支撑。系统里**唯一出现的 IS 是陈旧校正**（$\rho_\theta=\pi_\theta/\pi_{\text{old}}$ 对快照间漂移），它由 **PPO clip**（夹到 $[1-\varepsilon,1+\varepsilon]$）与 **staleness 双截断**（丢弃 age 过大样本、保持 $\pi_{\text{old}}\approx\pi_\theta$）共同约束方差。该稀疏性带来的真实局限是下面的曝光偏差——它**不是** IS 方差问题，要单列说明。

- **🔍 关于曝光偏差（exposure bias）：是的，它真实存在——这是固定数据集 $D$ 的固有代价（即问题 2 所指）**。你的判断完全正确：训练时学生 **teacher-force** 在数据集 $D$ 的上下文上，推理时却**自回归生成**自己的上下文（可能不在 $D$ 中），那里 $\Delta_T$ 无定义、策略可能不迁移。这正是 teacher-forcing / 离线蒸馏的共性局限，OPD 无法回避，且**随"离线程度"加重**。
  - **为什么本工程仍可接受**：Direct-OPD 的"预加载"本质就是用**离线固定-D 训练**换掉 Lightning-OPD 的"学生实时 rollout + 教师实时打分"闭环；预加载牺牲的正是**上下文覆盖率**，换取零 rollout 开销、可整批并行打分。所以曝光偏差**不是 bug，而是该 trade-off 的已知代价**。
  - **本工程的两道缓解**：① **KL 锚 $\mathrm{KL}(\pi_\theta\|\pi_{\theta_0})$** 把学生拴在初始（近 $D$）分布上，限制它偏离 $D$ 太远、压低自回归时掉出教师支撑的概率；② **`ref_dists`（初始学生分布张量）** 作为 KL 正则锚点，提供"不漂移太远"的硬约束（详见 §0 实体速查）。
  - **若要根治**：必须回到"教师对新上下文实时打分"——即保留 Lightning-OPD 的 rollout 环节（或在线 / 近似在线 OPD），那是预加载刻意放松掉的一环；换言之，曝光偏差的严重度与"离线程度"正相关，Direct-OPD 处于离线端、Lightning-OPD 处于近似在线端。

**与标准 RL/PPO 的三点不同（避免误读）**

1. **没有价值网络、也没有 baseline**：`Δ_T` 本身是 teacher RL 前后分布的**相对差**，天然就是"逐 token 的相对优势"，所以无需像 Stage 0 的 REINFORCE 那样减均值基线。Stage 0 用 REINFORCE+基线产教师，Stage 2 直接用 Δ_T 蒸馏——两者奖励来源一致（都是同一份 teacher）。
2. **没有 entropy 奖励项**：纯蒸馏目标，探索性由教师分布本身提供；代码未加 entropy bonus。
3. 序列级聚合 = 对 $T$ 取 `mean`（常数因子不影响优化）；padding 位置可通过 `mask` 屏蔽（demo 合成数据未用）。

**两个必须区分的 "ref"**：`π_ref`（教师 base）只出现在 $\Delta_T$ 内部，是"奖励的来源之一"；`π_{\theta_0}` 是 KL 正则的锚，是"学生自己的起点"。两者**不是同一个实体**（上轮已澄清 `ref_dists` 是初始学生分布张量、不是教师）。

### 🔧 关于"保留 Lightning-OPD rollout 环节"的可行设计（问题 2 的落地路径）

上一框确认了：固定 $D$ 必然产生曝光偏差。Lightning-OPD 的**根治**就是让学生**自回归 rollout** 自己的响应、再由教师实时打分——使训练上下文来自学生当前（漂移中）策略，而非固定 $D$。问题是：Direct-OPD 的"预加载"恰恰是为了**省掉教师实时打分**。两者存在成本张力，因此真正可落地的是一条**谱**，按"训练期教师推理开销"与"上下文新鲜度"权衡：

| 级别 | 上下文来源 | 教师打分时机 | 训练期教师开销 | 曝光偏差 | 状态 / 对应代码 |
|---|---|---|---|---|---|
| **L0（Direct-OPD）** | 固定 $D$ 的 $(p_i,r_i)$ | 仅 Stage 1 离线一次 | 零 | 最大（永久固定） | 默认静态路径 |
| **L1（离线 rollout 暖缓存）** | Stage 1 用学生/教师分布**采样多条** $(p_i, r'_i)$ 拼胖 D | 仅 Stage 1 离线一次（但上下文变丰富） | 零（训练期仍预加载） | 大幅降低（上下文覆盖学生/教师支撑） | **默认开启**（`warmup_M=4, warmup_source=student_init`） |
| **L2（Adaptive Teacher Cache）** | 训练期每 $K$ 步用**当前学生** rollout 一批 $(p,r')$ | 每 $K$ 步一次（教师常驻） | 摊销后小 | 有界（新鲜度 ≤ $K$ 步漂移） | **已实现并接线**（`adaptive_cache.py` + `scheduler.py` + `pipeline.py`）；`l2.enabled=false`（默认）退回 L0/L1 |
| **L3（全在线 Lightning-OPD）** | 每批都来自当前学生 rollout | 每批实时 | 最大（回到 live teacher） | 近零 | 骨架（`VLLMRolloutEngine` + 实时 Δ_T 路径），未完整跑通 |

**关键洞察**：rollout 环节**不必在训练期在线**——它可以是 **Stage 1 的离线增强**（`generate_batch` / vLLM `generate` 在 Stage 1 即可生成多条轨迹并预缓存 $\Delta_T$）。这样**零训练期教师开销**就把上下文从"每 prompt 一条固定 $r_i$"扩成"多条学生/教师分布采样轨迹"，直接压低曝光偏差，且**调度器 `_train_step` 一行不动**（仍是消费预加载缓存）。这是性价比最高、风险最低的第一步。

**各方案需要的代码改动（已对照当前实现）：**

- **L1（已实现，默认开启）**：只动 `pipeline.py` 的 `stage1_build_cache` —— 在 `cache.build` 前，对每个 prompt 用**初始学生**或**温度扰动的教师** `generate_batch` 出 $M$ 条响应，拼成 $(N\cdot M, T)$ 的"胖 $D$"，再统一 `cache.build`。`AsyncBatchedScheduler` 完全不变，因为它只读 `cache` + `prompts/responses` 索引。KL 锚 `ref_dists` 在 Stage 2 入口对所有胖 $D$ 上下文重算一次即可（仍是一次性、离线）。
  - ⚠️ 注意：L1 的胖 $D$ 用的是 **Stage 1 时的学生**（初始/随机）和教师分布，训练期学生还会继续漂；所以 L1 把曝光偏差降一个量级，但**不保证**训练全程上下文都在学生当前支撑内——仍需 L2 收尾。

- **L2（真正流式适应）**：需在 `scheduler.py` 加一个**动态数据集 + 动态缓存**：
  1. 保留 `teacher_rl` / `teacher_ref` 常驻（当前它们只在 Stage 0/1 用，没传进调度器——`pipeline.py:243` 只把 `student`+`cache` 注入 `AsyncBatchedScheduler`）；
  2. 新增 `_rollout_refresh` 线程（或 `_train_dispatcher` 内每 $K$ 步回调）：取一批 prompt → 用当前 `rollout_engine.generate` / `worker` 自回归采样 `r'` → 用 `teacher_rl`/`teacher_ref` 现算 $\Delta_T$ → 写进**动态缓存**（`cache.py` 需加 `append(idxs, prompts, responses, teacher_rl, teacher_ref)` 支持就地写；dense 模式 in-place 写 `self.delta` 分块，topk 模式用预分配 ring buffer）；同时用**冻结的初始学生** `response_dists` 现算 fresh 上下文的 `ref_dists`（KL 锚也要随 fresh 上下文刷新）；
  3. `_prompt_feeder` 改为**混合采样**：从固定 `prompts` 与动态缓冲按 `refresh_mix_ratio` 取；`_rollout_collector`/`_train_step` 按 idxs 指向的缓冲决定读固定张量还是动态张量。
  - 新鲜度由 $K$ 限死：刷新间隔内的学生漂移 ≤ $K$ 步，比 L0 的"永久固定"小得多，曝光偏差有界。

- **L3**：即完整 Lightning-OPD，每批 rollout+实时打分——直接复用 Stage 0 的 `generate_batch` + Stage 1 的 `cache.build` 逻辑，但搬到训练循环内逐批跑，相当于放弃预加载。

#### L1 详细机制（离线 rollout 暖缓存）

- **动机**：曝光偏差的根因是"训练上下文只有每 prompt 一条固定 $r_i$"。L1 在 **Stage 1（教师/学生冻结时）** 用采样把这条扩成多条，使缓存覆盖学生/教师分布支撑——训练期仍消费预加载缓存，调度器内核零改动。
- **具体数据流**（`pipeline.stage1_build_cache` + `run()` 内）：
  1. 对每 prompt $p_i$，按 `warmup_source` 用**初始学生**（`student_init`）或温度扰动的 `teacher_rl`（`teacher_perturbed`）或两者（`mix`）各 $\times M$ 次 `generate_batch(p_i, max_new_tokens=T, temperature=warmup_temperature)` → 得到响应 $r'_{i,m}$；
  2. 拼成**胖 $D$**：`fat_prompts=(N·(1+K),P)`、`fat_responses=(N·(1+K),T)`，其中 $K=M$（student_init/teacher_perturbed）或 $K=2M$（mix）；
  3. `cache.build(fat_prompts, fat_responses, teacher_rl, teacher_ref, …)` → `self.delta=(N·(1+K),T,V)`；
  4. `stage1_build_cache` **返回 `(cache, fat_prompts, fat_responses)`**；`run()` 把 `student` **提前到 Stage 1 前创建**并传入（使 warmup 分布与 KL 锚点同源），Stage 2 入口 `ref_dists = response_dists(student, fat_prompts, fat_responses)` 与调度器均改用 `fat_*`。
- **代码改动（已实现）**：**只动 `pipeline.py`**——`stage1_build_cache` 增加 warmup 采样循环 + `cat` 拼胖 D、返回值增为三元组；`run()` 提前创建 `student` 并传入、Stage 2 入口与调度器改用返回的 `fat_*` 张量。`AsyncBatchedScheduler` / `_train_step` / `cache.py` **内核不动**（仍只读 `cache`+索引，`n_prompts` 自动随胖 N 变大）。新增配置 `stage1.warmup_M`(**默认 4=开启，与 DEFAULT_CONFIG_V2/Stage1Cfg 一致**)、`stage1.warmup_source∈{none, student_init, teacher_perturbed, mix}`（**默认 student_init**）、`stage1.warmup_temperature`。`warmup_M=0` + `warmup_source=none` 时完全退化为 L0（fat=原 D，行为零变化）。
  - ⚠️ **resume 与 warmup 的同源不变式（P1-4 修复）**：warmup 采样分布必须与 KL 锚点（初始 student 分布）同源。resume 会把断点权重 load 进 `student`，因此 `_run_body` 用**独立新建的 `warmup_student`**（不被 `load_state_dict` 覆盖）传给 `stage1_build_cache`——即使断点续跑，warmup 上下文仍是初始分布，与从断点恢复的 `ref` 锚点一致。
- **代价**：教师推理开销**仍只在 Stage 1 一次**（`1+M` 倍于 L0，但离线、可批处理）；训练期零教师开销。内存：`delta`/`ref_dists` 张量 $\times(1+M)$，demo 词表 64 下可忽略，真实词表须配 topk 缓存。
- **残余缺口**：胖 $D$ 用的是 **Stage 1 时刻**的策略（初始学生 / 扰动教师）。训练期学生还在漂，所以 L1 **不保证**训练全程上下文都在学生当前支撑内——它把曝光偏差降一个量级（从"单点固定"到"分布覆盖"），但未消除。要消除需 L2。

#### L2 详细机制（周期性在线刷新）【已实现并接线，详见 §4.6】

> ⚠️ **状态修正**：下文原设计稿中的 `append` 接口、`_rollout_refresh` 线程、混合 `_prompt_feeder` 均已落地——分别由 `adaptive_cache.RefreshRingBuffer`、`scheduler` 的 `train_refresh_phase` / `_train_step_refresh`、`pipeline` 的 L2 交替相位循环实现。默认 `l2.enabled=false` 退回 L0/L1，零回归风险。

- **动机**：让训练上下文来自**当前学生**策略（而非 Stage 1 的旧策略），把曝光偏差的"新鲜度"有界化到 $K$ 步漂移内。
- **具体数据流**（`scheduler.py` 加动态缓冲 + refresh 线程）：
  1. **注入教师**：`pipeline.py` 把 `teacher_rl`/`teacher_ref` + **冻结的初始学生 `theta0`**（Stage 2 入口 `student` 的快照）一并传进 `AsyncBatchedScheduler`（`pipeline.py:243` 现在只传了 `student`+`cache`）；
  2. **动态缓冲**：`self.dyn_prompts / self.dyn_responses / self.dyn_delta`（`cache.py` 加 `append(idxs, p, r, teacher_rl, teacher_ref)`——dense 模式 in-place 写 `self.delta` 分块、topk 模式用预分配 **ring buffer** 容量 `N + dyn_cap`）+ 对应的 `dyn_ref`（用冻结 `theta0.response_dists(p,r)` 现算）；
  3. **`_rollout_refresh` 线程**（或 `_train_dispatcher` 内每 $K$ 步回调）：取一批 prompt → 用**当前学生** `self.rollout_engine.generate(p)` / `self.worker` 自回归采样 `r'` → `teacher_rl.response_dists(p,r')` − `teacher_ref.response_dists(p,r')` 现算 $\Delta_T$ → 写 `dyn_delta`；同步用 `theta0` 现算 `dyn_ref`；
  4. **`_prompt_feeder` 改混合采样**：按 `refresh_mix_ratio` 从固定 `[0,N)` 与动态 `[N, N+dyn)` 取 `idxs`，并携带"源标记"；
  5. **`_rollout_collector`/`_train_step`** 按源标记决定读 `self.prompts` 还是 `self.dyn_prompts`（其余 `s_old`/`Δ_T`/`ref_dists` 取值逻辑完全复用）。
- **一个额外好处**：动态样本 $(p,r')$ 的 `s_old` 是用**当前学生快照**算的（worker 刚 `acquire_if_newer` 到最新），与 `s_cur` 几乎同步 → 这些样本 **ρ≈1、近乎 on-policy**，IS 校正几乎不需要。曝光偏差与陈旧偏差在此同时被压低。
- **代码改动（已实现）**：`adaptive_cache.py`（`RefreshRingBuffer` / `DisagreementComputer` / `CacheHealthMonitor` / `DynamicRatioController` / `PromptStateStore` / `RefreshSelector` / `run_refresh_phase`）、`scheduler.py`（`_train_step_refresh` / `train_refresh_phase` + 双池 feeder）、`pipeline.py`（交替相位循环 + 保留教师供 refresh）。详见 §4.6。默认 `l2.enabled=false` 退回 L0/L1，零回归。
- **代价**：教师推理开销被摊销到每 $K$ 步一次（而非每步），`K` 越大越省；`K` 越小新鲜度越高。教师常驻显存（与 learner 同卡需 colocated offload L6）。
- **残余缺口**：刷新间隔内学生仍会漂 $<K$ 步，动态样本的 Δ_T 也有微小 staleness——但已由 $K$ 显式上界，且被 `StalenessQueue` 双截断兜底。

#### L3 详细机制（全在线 Lightning-OPD）

- **动机**：彻底消除曝光偏差——每个训练样本都来自**当前学生** rollout、且 Δ_T 由教师**实时**打分。
- **具体数据流**：退化为 Lightning-OPD 本身——`PromptFeeder` 触发当前学生 `generate` → `TeacherScorer` 用 `teacher_rl/teacher_ref` 现算 Δ_T（**不再读预加载缓存**）→ `TrainDispatcher` 训练。直接复用 `generate_batch`（Stage 0）+ `cache.build` 的单次逻辑，但搬到训练循环内逐批跑。
- **代价**：教师推理开销 = **每批一次**（Direct-OPD 预加载想省掉的正是最这笔）。真实 7B 教师下这是训练主瓶颈——这正是 L0/L1/L2 存在的理由。
- **与谱的关系**：L3 是 Live-Teacher 上限；L0–L2 是用"预加载/摊销"逐步逼近它的省钱版，代价是曝光偏差从"近零"退化到"有界/较大"。

> **💡 直觉小结（demo 张量示例，vocab=64, P=6, T=8, N=16, M=4, K=5）**
> - L0：`delta=(16,8,64)`、`ref_dists=(16,8,64)`，每 prompt **1** 条固定 $r_i$。
> - L1（M=4, `student_init`）：`delta=(80,8,64)`（×5）、`ref_dists=(80,8,64)`，每 prompt **5** 条上下文（1 原始 + 4 学生采样）；若 `warmup_source=mix` 则另加 4 条教师采样 → 每 prompt **9** 条、`delta=(144,8,64)`。训练期零教师开销。
> - L2（dyn_cap=64, K=5）：固定 `delta` 16 条 + 动态 ring buffer 64 条；教师每 **5** 步推理一批（≈ 1/5 批次成本）；动态样本 ρ≈1 近 on-policy。
> - L3：每步都教师推理 8 条，成本 ×5 vs L2（K=5），但曝光偏差近零。

**结论**：可以保留 rollout 环节，且不必一步退回到全在线。建议路径 **L1（零改动调度器暖缓存，默认开启）→ L2（Adaptive Teacher Cache 动态刷新，已实现并接线）**——既保留 Direct-OPD 的"零训练期教师开销"优势，又把曝光偏差从无界压到有界。L2 子系统的 `append`/refresh 接口与调度器接线已完成（见 §4.6）；配合 **P-OPD 纯 on-policy 交替相位**，可进一步把 OPD 推到谱的更"在线"端。

---

## 1. 学生 rollout 与更新异步是怎么实现的

### 1.0 先搞懂：什么叫"异步"？为什么要异步？

**同步（串行）**训练是这样的一条直线：

```
取一批数据 → 用当前模型算一遍 s_old → 算 Δ_T → 训练更新一步 → 再取下一批 …
```

每一步都必须等上一步彻底做完。问题是：**"算 s_old"（一次模型前向）和"训练更新"（前向+反向+优化器）都是重活**，串行时 GPU 要么在算 s_old、要么在训练，总有一半时间空着。

**异步**就是把它们拆成几条**同时跑的流水线**：一条线程专门不停算 s_old，主线程专门不停训练。算 s_old 的线程算好的结果放进一个"传送带"（队列），训练线程从传送带上取来就用。两边谁也不等谁，GPU 一直有事干。

代价是：训练线程拿到的 `s_old`，可能是用**几步之前的旧权重**算的（因为算 s_old 的线程还没追上训练线程的最新权重）。这就是"陈旧（stale）"。整套异步机制，说白了就是——**既要并行提速，又要想办法控制"旧到什么程度还能用"**。

### 1.1 线程拓扑（`AsyncBatchedScheduler.run`）

`run()` 在主线程跑训练调度，另起 3 个守护线程，共 4 个解耦阶段，靠 3 条队列串联：

```
PromptFeeder线程          RolloutCollector线程         TeacherScorer线程          主线程 TrainDispatcher
┌──────────────┐  (B,)   ┌──────────────────┐ (B,T,V) ┌────────────────┐ 批次  ┌─────────────────────┐
│ 随机抽批次    │─ _pq ─▶│ 按版本加载权重快照 │─ _rq ──▶│ 查缓存取 Δ_T    │──────▶│ _train_step:        │
│ 索引 idxs     │ Queue  │ 前向算 s_old       │ Queue  │ (离线,见§2)     │staleness_q│  重算 s_cur→损失 │
└──────────────┘         └──────────────────┘         └────────────────┘       │  →反向→step→_publish│
                                                                               └─────────────────────┘
```

- `_pq`：PromptFeeder → RolloutCollector，只传**批次索引** `idxs`，形状 `(B,)`（B=批大小，demo 默认 8）。
- `_rq`：RolloutCollector → TeacherScorer，传 `(idxs, s_old, 版本号)`，`s_old` 形状 `(B,T,V)`。
- `staleness_q`：TeacherScorer → TrainDispatcher，传 `(idxs, s_old, Δ_T)` + 版本号（见 §1.4）。
- `_pq`/`_rq` 是普通 `queue.Queue(maxsize=cfg["queue_size"])`（默认 8，即最多囤 8 个批次）。
- `stop = threading.Event()`：训练步数达标后置位，3 个线程在各自 `get(timeout=...)` 处超时醒来、看到 stop 就退出。

### 1.2 权重快照 & 版本号 🔍

这是整套异步的"定锚点"，务必看懂。

#### 🔍 快照（snapshot）是什么？

**快照 = 把模型当前的全部参数复制一份、冻住。**

为什么需要复制一份？因为训练线程在**不停地改**模型的权重（每训一步，参数就变一点）。如果 rollout 线程**直接读**那个正在被改的模型，就会读到"撕裂"的权重——比如第 3 层是第 10 步的新值、第 5 层还是第 9 步的旧值（训练线程改到一半被它读了）。用这种半成品权重算出来的 `s_old` 是错的。

所以正确做法是：训练线程每改完一步，就**存一份完整的、冻住的副本**到 `WeightStore`；rollout 线程要算 `s_old` 时，只读这份冻住的副本，绝不碰正在被改的原模型。这样它读到的永远是"某一完整时刻"的一致权重。

> 类比：你在改一份共享文档，别人要引用它。你不是让对方边看你改边抄（会抄到半成品），而是每改完一版就"另存为 v1 / v2 / v3…"，对方引用某个**已存好的整版**。

#### 🔍 版本号（version）是什么？

**版本号 = 一个从 0 开始、每发一次新快照就 +1 的整数计数器**（`WeightStore._version`）。它只是给每份快照编个号：第 1 份快照是 v1，第 2 份是 v2，以此类推。

它的作用是**给"新旧"一个可比较的刻度**。光有快照不够——你还得知道"这份 s_old 是用第几版权重算的"，才能判断它落后当前多少步。

#### 🔍 版本号怎么"随样本走"？

两个动作各推一次版本（`scheduler.py`）：

```python
# learner 每训完一步：
v = self.weight_store.publish(self.student.state_dict())  # 存新快照，_version +1，返回新版本号
self.staleness_q.advance_version()                        # 队列的"当前版本"也 +1

# rollout 线程算完一个批次的 s_old，把它【连同当时用的版本号】一起入队：
self._rq.put((idxs, s_old, self._loaded_ver), timeout=0.5)
#                              ^^^^^^^^^^^^^^^ 这个批次是用第 _loaded_ver 版权重算的
```

于是**每个在队列里流动的样本，都自带一个小标签："我是用第几版权重算的"**。训练端取出样本时，拿"当前最新版本号"减去"样本标签上的版本号"，就得到它的**陈旧度（age）**——落后了几版。

#### 🔢 举个具体例子

> 📌 **口语词 ↔ 正式线程名对照**（与 §1.1 对齐）：
> - **「learner / 训练线程」= 主线程 `TrainDispatcher`**——跑 `_train_step`、调 `_publish()` 发快照、推进版本号。它既是被训练的学生模型，也是执行训练的主线程，是**同一个东西**。
> - **「rollout 线程」= `RolloutCollector` 线程**——按版本加载快照、算 `s_old`。
> - `PromptFeeder` / `TeacherScorer` 是辅助线程（喂索引、查 Δ_T 缓存），本例只聚焦"rollout → 训练"的陈旧路径，故未在例子中出现。

设 `staleness_threshold = 4`：

```
时刻 A：TrainDispatcher（训练线程，即 learner 角色）发布初始权重 → 版本 v=1。
        RolloutCollector（rollout 线程）加载之，_loaded_ver=1。
时刻 B：RolloutCollector 用 v=1 算好批次甲的 s_old，把 (甲, s_old, ver=1) 入队。
        与此同时，TrainDispatcher 已经咔咔训了 3 步 → 当前版本变成 v=4。
时刻 C：TrainDispatcher 取出批次甲，看到标签 ver=1，当前 v=4 → age = 4-1 = 3。
        3 ≤ 4，没超龄 → 正常用它训练。
时刻 D：另一个也是用 v=1 算的批次乙，在路上耽搁了更久，
        等它被取出时 TrainDispatcher 已训到 v=6 → age = 6-1 = 5 > 4 → 丢弃，不训。
```

> 注：例子中「TrainDispatcher 训了 3 步」这个动作发生在**时刻 B 期间**——因为 RolloutCollector 和 TrainDispatcher 是两条**并行**的线，RolloutCollector 算甲的同时 TrainDispatcher 在疯狂训练，所以甲一算完就"已经落后 3 版"了。这正是异步提速的来源，也是陈旧度要被管住的原因。

### 1.3 WeightStore：按需加载，不每步搬权重 🔍

`acquire_if_newer`（`buffer.py`）是"按需加载"的核心：

```python
def acquire_if_newer(self, last_ver):
    if self._version == last_ver:
        return None, self._version      # 没有更新的版本 → 不克隆、不加载，直接复用现有权重
    snap = {k: v.clone() for k, v in self._snapshot.items()}
    return snap, self._version          # 有新版 → 克隆一份新快照返回
```

#### 🔢 为什么这么做？（v1 的教训）

v1 的毛病是：**每来一个样本就 `load_state_dict` 一次**。`load_state_dict` 要把整套参数从快照拷进 rollout 模型，是很重的 IO。但 learner 可能训 1 步、rollout 线程已经算了 5 个批次——这 5 个批次用的是**同一版权重**，却做了 5 次重复拷贝，纯浪费。

v2 的做法：rollout 线程记住自己**当前加载的是第几版**（`self._loaded_ver`）。每次要算 `s_old` 前先问一句"有比我手上更新的吗？"

- 没有（`_version == _loaded_ver`）→ 返回 `None`，**直接用已加载的权重接着算**，一次拷贝都省；
- 有（`_version > _loaded_ver`）→ 才克隆新快照、加载、更新 `_loaded_ver`。

#### 🔢 例子

```
learner 训到 v=3（已发布 v3 快照）。rollout 线程 _loaded_ver=3。
它连续算批次 1、2、3：
  批次1：acquire_if_newer(3) → _version=3 == 3 → 返回 None → 复用 v3 权重，零拷贝。
  批次2：同上，复用。
  批次3：同上，复用。
  （这 3 个批次一次 load_state_dict 都没做）
learner 又训一步到 v=4。rollout 线程算批次4：
  acquire_if_newer(3) → _version=4 > 3 → 克隆 v4 快照，加载，_loaded_ver=4。
```

拿到新快照后，按部署形态加载：
- toy 路径：`self.worker.load_state_dict(snap)`；
- vLLM 路径（L3）：`self.rollout_engine.update_weights(snap)`；
- 开了 colocated offload（L6，`offload_to_cpu=True`）：快照平时存 CPU 省显存，加载前 `{k: v.to(self.device)}` 搬回 GPU。

### 1.4 StalenessQueue：双截断防"太旧" 🔍

#### 🔍 "太旧"到底是什么意思？为什么怕它？

回忆：`s_old` 是**行为策略**的分布，在 PG 损失里当重要性比 `ratio = π_cur / π_old` 的**分母来源**。PPO 这套"旧策略校正"有个隐含假设：**s_old 离当前策略不能太远**。

直观地想：如果 `s_old` 是用 10 步之前的权重算的，那时候的策略和现在已经差别很大，用它算出的 `ratio` 是在"校正一个早已不存在的旧策略"，梯度方向就会被带偏——你以为在微调，其实在按一张过期地图开车。

**"太旧"就是：算这份 `s_old` 的权重版本，落后当前权重版本太多步**（`age > threshold`）。所以必须给异步设一个"最多能容忍旧几版"的闸口，超了就扔，宁可这批次不训，也不拿过期梯度污染模型。

#### 🔍 为什么要截"两"次？

因为**时间在流动**：一个样本入队时是"新鲜"的，但它在有界队列里**排队等训练**的这段时间里，learner 可能又训了好几版——等它被取出来的时候，已经变"太旧"了。所以要在**入队时**和**取出时**各查一次：

```python
# ① 入队侧（StalenessQueue.put，buffer.py）：太旧直接拒收，连队列都不进
def put(self, item, version, timeout=None):
    age = self._cur_version - version
    if age > self.threshold:
        return False                          # 太旧 → 丢弃
    self._q.put((item, version, age), timeout=timeout)
```

```python
# ② 消费侧（_train_step，scheduler.py）：取出来时再查一次
if self.staleness_q.current_version - ver > threshold:
    return None                               # 在队列里等太久，变太旧了 → 这批次不训
```

- **入队侧**的意义：挡住"出生就太旧"的样本，**别让它们占队列位置、浪费训练算力**。
- **消费侧**的意义：接住"入队时还行、排队期间变旧"的样本。

少任何一道都会漏：只入队截，拦不住"排队变旧"；只消费截，队列会被一堆早就过期的样本塞满、挤掉新鲜样本。

#### 🔢 例子（threshold=4）

```
样本丙，ver=2：
  入队时：当前 v=3 → age=1 ≤ 4 → 放行入队。✓（第①道通过）
  队列有点满，训练偏慢，丙在队里等了一会儿……
  等训练线程取出来时：当前已 v=7 → age=5 > 4 → 丢弃。✗（第②道拦下）
```

如果没有第②道：丙会被拿去训练，用一张落后 5 版的过期 `s_old` 算梯度，把模型往错误方向推一把。
如果没有第①道：一个 ver=1、当前 v=6（age=5）的样本也会挤进队列占坑，把后面新鲜的样本挤出去。

### 1.5 一次 batch 的完整生命周期（时序）

```
t0  PromptFeeder:   idxs=(B,) ─_pq──────────────────────────────┐
t1  RolloutCollector: 取 idxs；acquire_if_newer(版本 v) → s_old(B,T,V) ─_rq─┐
t2  TeacherScorer: 取 (idxs,s_old,v)；查缓存取 Δ_T；staleness_q.put(...,版本v)（入队截断）
t3  TrainDispatcher: staleness_q.get() → _train_step（消费截断）：
       s_cur = student.response_dists(p_b, r_b)   # 用【当前】权重现算（带梯度）
       loss = pg_loss(s_cur, s_old, Δ_T) + kl_coef*kl
       backward → clip_grad_norm_ → opt.step() → _publish() → 版本 v+1
```

**关键**：`s_cur` 永远用**当下**的 student 现算（带梯度，是要更新的对象），只有 `s_old` 是允许陈旧的 rollout 快照（只当重要性比的分母，不反传）。陈旧度只影响"分母是哪一版"，不影响梯度算在谁身上——梯度永远打在最新的 `s_cur` 上。

### 1.6 背压 & 不死锁 🔍

#### 🔍 什么是"背压"（backpressure）？

流水线各阶段速度天然不一样：rollout 前向可能比训练快，训练又可能比 scorer 慢。当下游慢、上游快时，中间的队列会被**塞满**。

"背压"就是：**下游的慢，顺着队列往回"顶"上游，逼上游放慢**。本代码用**有界队列 + put 超时**实现：

- 队列设了 `maxsize`（默认 8），最多囤 8 个批次，不会无限堆积 → **内存有上界**。
- 上游 `put(item, timeout=0.5)`：队列满了就等 0.5 秒，还满就抛 `queue.Full`，代码 `except Full: continue`——**丢弃本次、下一轮再来**，而不是死等。

效果：下游一慢，队列一满，上游就被迫"丢帧/歇一下"，整条线的速度自动向最慢的一环看齐，且任何一环都不会把内存撑爆。

#### 🔍 什么是"不死锁"？为什么这里不会死锁？

**死锁** = 两个（或多个）线程互相等对方，谁都动不了，程序卡死。比如：线程 A 拿着锁 1 等锁 2，线程 B 拿着锁 2 等锁 1。

本代码不会死锁，靠的是**所有 put/get 都带 timeout**：

- `get(timeout=1)` 取不到（队列空）→ 抛 `queue.Empty` → `continue` 空转重试，**不会永远卡着等数据**，而且每次醒来都会看一眼 `stop` 事件，该退出就退出。
- `put(timeout=0.5)` 放不进（队列满）→ 抛 `queue.Full` → `continue` 丢弃重试，**不会永远卡着等空位**。

因为**没有任何一个地方会无限期阻塞**，每个线程都会周期性"醒来重试"，自然不存在"永远互相等"的局面 → 无死锁。

#### 🔢 反例：如果没有 timeout 会怎样？

假设 `put` 用不带超时的死等版本：

```
训练线程卡住（比如在等一个永远不会来的资源）→ staleness_q 一直满
  → TeacherScorer 的 put 死等 → 它不消费 _rq → _rq 满
  → RolloutCollector 的 put 死等 → 它不消费 _pq → _pq 满
  → PromptFeeder 的 put 死等
四个线程全部冻结，程序挂死，只能强杀。
```

加了 timeout 后，同样的情况只是大家各自"丢帧重试"，训练线程一旦恢复，流水线立刻继续流动。

### 1.7 分布式形态（L5，仅云 GPU）

线程 + Queue 换成「Ray actor + NCCL 权重广播」（`scheduler.py` 下半部），**算法内核 `_train_step` 一行不动**：

- `_RayRolloutWorkerImpl`：每卡一个 Ray actor，跑原来的 rollout+scoring 两段；
- `WeightBroadcaster`：learner↔worker 用**非阻塞 P2P（isend/irecv）**推权重。

#### 🔍 为什么分布式下用 P2P，不用集体 broadcast？

**集体 broadcast**（`torch.distributed.broadcast`）要求**所有 rank 在同一时刻一起调用**同一个通信操作——它是"集合动作"，少一个 rank 加入就集体卡住。而 AsyncOPD 的 rollout 和 learner **步数天然不对齐**（learner 训 3 步的功夫，worker 可能算 5 个批次），它们根本没法"约好同一时刻一起 broadcast"。硬用 broadcast → 死锁。

**非阻塞 P2P**（`isend`/`irecv`）是"点对点、发完就走"：learner 每步把新权重 `isend` 给各 worker（fire-and-forget，不等接收方），worker 在**自己准备算 rollout 时**才 `irecv` 拉取。两边各按各的节奏，不需要对齐 → 不会死锁。

`DistAsyncScheduler._publish()` 覆盖为 `broadcaster.push_async(...)`，把"发快照"从"写内存"换成"NCCL 推给各卡"。

> ⚠️ 边界（见 §4）：L2（learner Megatron TP=2）与本并发模型**互斥**——TP 集合通信需 rank1 协同，而 rank1 被派作 rollout worker 会死锁。代码里已加护栏报错，L2 需另写 colocated 交替相位调度。

### 1.8 论文里的"缓存 (前缀 s, 动作 a, rollout 阶段 log prob)"，本实现落在哪？🔍

> ⚠️ **先厘清一个最容易被搞混的点：我们在讲 OPD，为什么反复提到 RL / PPO？**
>
> 因为 **OPD 本身就是一个「策略优化（policy optimization）」问题**——它的训练损失和 PPO 长得一模一样（重要性比 `ratio` + 裁剪 + KL 正则，见 §3），唯一的区别是把 PPO 里的「环境奖励」换成了蒸馏信号 `Δ_T`。更进一步，**异步 OPD 遇到的「陈旧行为策略」难题，和异步 PPO 遇到的完全相同**：都是用一套滞后于当前策略的旧策略去算行为概率、需要重要性采样（IS）校正、并对"太旧"的样本做退役。
>
> 所以 asyncOPD 论文（以及本文）借用 RL/PPO 的术语和工程方案，**不是话题切换成 RL，而是 OPD 的实现地基就是 RL/PPO 那套策略梯度机器**。下面这张表把"哪些是共享、哪些是 OPD 独有"列清楚：

| 维度 | 标准 RL / PPO | OPD（本实现） | 是否同一套 |
|---|---|---|---|
| **奖赏来源** | 环境真实奖励 `r(s,a)` | 教师偏好差 `Δ_T = logπ_rl − logπ_ref` | 不同，但都扮演"奖赏信号"角色 |
| **优化目标** | 最大化 `E[环境回报]` | 让 `E[Δ_T]`（当前策略下）最大，对齐教师 | **都是"最大化某期望奖赏"** |
| **训练损失** | PG + PPO 裁剪 + KL 正则 | `pg_loss`（ratio + min 裁剪 + KL 锚点，见 §3.2） | **同一套数学** |
| **异步陈旧处理** | 行为策略 `π_old` 滞后 → IS 校正 + 拒旧样本 | `s_old` 滞后 → 版本号 + 双截断（§1.2–1.4） | **同一套工程** |
| **交互对象** | 真实环境（在线采样 trajectory） | 固定的 `(prompt, response)` 数据集 | OPD 是离线蒸馏，**不与环境交互** |

> 一句话总结：**OPD ≠ RL 在"和环境交互采样"这个意义上；但在"用 PPO 式损失 + 异步 IS 校正去优化一个策略"这个意义上，OPD 就是 RL/PPO 的一个特例（奖赏来自教师而非环境）。** 因此本节标题与其说在讲 RL，不如说在讲"asyncOPD 复用了 async PPO 的哪部分轨迹缓存方案"。

下面把论文的意图对齐，再看本代码怎么对应——结论是**三项都在，只是"动作 a"因数据固定而改成"按索引取"而非"持久缓存"，rollout log prob 就是 `s_old` 且必须随批次携带**。

#### 🔍 论文为什么要把 (s, a, logp_old) 一起缓存？

标准异步 PPO 里（asyncOPD 论文直接复用了这套缓存方案），动作**在 rollout 阶段由"行为策略"当场采样**，并当场给每个动作打分得到 `logp_old = log π_old(a|s)`。而 learner 训练时要用"**当前策略**"重算 `s_cur = log π_cur(a|s)`，做重要性比 `ratio = π_cur/π_old`。

问题在于：这两个 log 概率**必须落在同一条轨迹 (s, a) 上**才有可比性。而 `π_old` 是"几步之前的旧策略"，learner 手里只有新权重，**没法用新权重反推出旧策略在那条 (s,a) 上的 log prob**。所以 `(s, a, logp_old)` 必须在 rollout 时一次性算好、随样本一起送到 learner——learner 自己补一个 `s_cur` 即可。

一句话：**`logp_old` 是整条数据里唯一"learner 重算不了"的字段，必须带来；`s` 和 `a` 则是定位这条轨迹的坐标。**

#### 🔍 本实现逐项落位（对照 `_rq` / `_train_step`）

RolloutCollector 算完 `s_old` 后入队的元组是：

```python
# scheduler.py · _rollout_collector
self._rq.put((idxs, s_old, self._loaded_ver), timeout=0.5)
#              ^^^^^  ^^^^^  ^^^^^^^^^^^^^
#              批次索引   rollout log prob   权重版本号
```

TrainDispatcher 取出后，在 `_train_step` 里：

```python
idxs_dev = idxs.to(self.device)
p_b = self.prompts[idxs_dev]          # 前缀 s：用索引现取（prompt）
r_b = self.responses[idxs_dev]        # 动作 a：用索引现取（response 就是动作！）
s_cur = response_dists(self.student, p_b, r_b)   # 当前策略的 s_cur（带梯度）
# s_old 已从队列元组带来（无梯度）；ratio = (s_cur - s_old).exp()
```

所以论文三项对应如下：

| 论文的缓存项 | 本实现里的载体 | 来源 |
|---|---|---|
| **rollout 阶段 log prob** | **`s_old`**，`(B,T,V)`，就是元组第二项 | RolloutCollector 用（按需加载的）旧权重前向算好，随批次入队。✅ 必须带 |
| **动作 `a`** | **`r_b = self.responses[idxs_dev]`** | 固定离线数据集，learner 端按 `idxs` 索引现取，不进队列 |
| **前缀 `s`** | **`p_b = self.prompts[idxs_dev]`** + `response_dists` 内部拼 `prompt+response[:t-1]` | 同上，`response_dists` 把前缀构造出来 |

#### 🔍 为什么"动作 a"可以不缓存、只存索引？这是和在线 RL 的本质区别

- **在线 asyncPPO**：动作是 rollout 时用行为策略**采样**出来的，每条样本的动作可能不同、且**无法从固定数据恢复** → 必须把 `a` 存进 replay buffer，否则 learner 不知道当时采了哪个动作。
- **本 OPD（Lightning-OPD 设定）**：训练数据是一份**固定的 `(prompt, response)` 数据集**，`response` 就是动作，是常量。既然动作永不变，它就没必要"缓存进轨迹"，用 `idxs` 去 `responses[idxs_dev]` 取即可——**省掉"搬运/存储轨迹大张量"这一步**。

> 这也点出了本实现"异步"的真正含义：**异步的是"算 s_old 这件重活"与"训练更新"的解耦 + 对陈旧权重的 IS 校正**，而不是"在线采样"。数据本身是 on-policy 固定的，动作不会被行为策略反复重采样。陈旧度（age）描述的是"s_old 的权重落后几版"，而不是"复用多样化样本"。

#### 🔢 一个元组例子

设 `idxs = [3, 7, 1, 5]`（批量 4 个样本），队列里那条是 `([3,7,1,5], s_old, ver=2)`：

```
learner 取出后补全整条轨迹：
  动作 a   = responses[[3,7,1,5]]          # 4 个 response 序列
  前缀 s   = prompts[[3,7,1,5]]            # 4 个 prompt
  logp_old = s_old                         # 那 4 个样本的 rollout 分布 (4,T,V)
  再补：s_cur = student(prompts, responses) # 当前权重现算
  → ratio = (s_cur - s_old).exp()          # 同一条 (s,a) 上对比，重要性校正
```

三者齐备，轨迹结构与标准异步 PPO 等价（注意本 OPD 的 `a` 是固定的 `response`，不是行为策略采样的；差异见下一段）。

#### ⚠️ 诚实的边界：本实现是"有界在途队列"，不是"持久重放缓冲"

论文/工业实现（Sample Factory、Apex、verl、slime）用的是**大容量持久 replay buffer**：`(s, a, logp, r)` 存进去后，learner 会**反复抽样同一样本多次**（样本复用是它们提速的关键之一），陈旧靠"存着的大量样本 + IS"消化。

本实现用的是 **`maxsize=8` 的有界在途队列**（`_pq/_rq/staleness_q`），它**不是 replay buffer**：
- 一个样本只走一遍流水线，超期（age > threshold）就被双截断**丢弃**，不复用；
- 目的只是"让 rollout 重活和训练重活并行、且有界内存"，不是"攒样本反复训"。

这是 toy/内核 demo 的合理简化——固定数据下样本复用增益有限，且省去 replay buffer 的采样/优先级/去重复杂度。生产上云时，应换成真正的 replay buffer + 在线采样（verl/slime 的路子），那时"动作 a 和 rollout log prob"就会像论文那样**显式持久存入 buffer**，而不是只靠 `idxs` 索引 + `s_old` 携带。

---

## 2. 教师离线是怎么实现的

### 2.0 先搞懂：什么叫"教师离线"？为什么要离线？

OPD 需要一个"教师"来告诉学生"每个动作有多好"。**在线（live）**做法是：训练时每一步都把教师模型跑一遍前向，现算现用。问题是教师通常比学生大得多（比如学生 7B、教师 70B），**每步都跑一遍 70B 前向，训练会慢到没法用**，而且教师一直占着显存。

**离线**做法：既然训练用的 `(prompts, responses)` 数据是**固定的**（Lightning-OPD 的设定），那教师对这些数据的输出也是**固定的**——完全可以**在训练开始之前，把教师对所有数据的输出一次性算完、存成一张大表**。训练时要用，直接**查表**，一次教师前向都不用跑。教师算完即释放，显存也省了。

### 2.1 离线一次性预计算（`TensorTeacherCache.build`，Stage 1）

训练前，对全部 `(prompts, responses)` 用两个教师各做一次 `response_dists`：
- `teacher_rl`：Stage 0 小模型 RL 之后的 post-RL 弱教师；
- `teacher_ref`：RL 之前的参考副本。

按 `build_batch_size` 分块前向，拼成 `(N, T, V)`，再**预计算差值** `Δ_T = rl − ref`。这就是训练期唯一需要教师的地方，之后两个教师对象即可释放。

### 2.2 两种缓存形态（`top_k` 决定）

`TensorTeacherCache(mode)`：

- **dense（`top_k<=0`，demo 默认）**：存完整 `rl/ref/delta`，形状 `(N, T, V)`。
- **top-K 稀疏（`top_k>0`，GPU/真实词表用）**：每个 `(n, t)` 位置只存**教师自己的 top-K** 个 `(token_id, logp_rl, logp_ref)`：
  ```python
  tk = rl_c.topk(Kt, dim=-1)          # 逐 chunk 取（修复后绝不先拼出完整稠密）
  ids_l.append(tk.indices)            # (N,T,Kt)
  rlk_l.append(tk.values)
  refk_l.append(ref_c.gather(-1, tk.indices))   # ref 在教师 top-K 上的 logp
  self.delta_k = self.rl_k - self.ref_k         # (N,T,Kt)
  ```

  #### 🔢 为什么稀疏能省 1000×？
  真实词表 `V=128000`。dense 要存 `(N,T,128000)`；稀疏只存每位置概率最高的 `K=256` 个 → `(N,T,256)`。`128000 / 256 = 500`，加上每张量只存 id+两个 logp，**体积约 ↓1000×**。原本存不下的，现在能 `torch.save` 落盘 / mmap 跨进程共享（L4）。
  为什么不直接存全词表？因为教师的概率质量几乎都集中在那 top 几百个 token 上，剩下的 12 万多个 token 概率≈0，存了也是浪费。

### 2.3 训练期零 live teacher

`_teacher_scorer` 线程**不做任何教师前向**，只查缓存：

```python
if self.cache.mode == "topk":
    delta_payload = None                     # 稀疏：只透传 idxs，Δ_T 在 learner 现场展开（省得搬 (B,T,V)）
else:
    delta_payload = self.cache.get_delta(idxs)   # dense：零拷贝索引 (B,T,V)
```

- dense 的 `get_delta` 就是 `self.delta[idxs]`——**一次张量索引**，取出对应批次那几行，零拷贝。
- 稀疏模式**不在 scorer 展开**，只把 `idxs` 透传给训练端（展开放 learner 做，见 §3.3），避免在队列里搬 `(B,T,V)` 大张量。

### 2.4 teacher 一致性校验的实现逻辑 🔍

#### 🔍 为什么必须保证"同一教师"？

`Δ_T = logπ_rl − logπ_ref` 这个差值，**只有在 `teacher_rl` 和 `teacher_ref` 是同一个模型（只是训练阶段不同）时才有意义**——它表达的是"同一个教师，RL 前后对同一动作的偏好变化"。

如果两个"教师"根本不是同一个架构：
- **词表不同**（一个 V=64、一个 V=128）：两个 `(N,T,V)` 张量最后一维都对不齐，相减直接报错或错位；
- **隐藏维度 d_model 不同**：即使词表凑巧相同，两者 logit 的数值尺度也不同，相减得到的是**没有物理意义的垃圾**；
- 这样的 Δ_T 喂进 PG，会给学生一个**怎么训练都消不掉的系统性错误方向**（所谓"不可约梯度偏差"）。

所以 `build()` 在最开头做一次**结构校验**，先确认两个教师"长得是同一个模型"，再开始算。

#### 🔍 校验了哪几项？（逐行）

`build()` 开头，当 `enforce_consistency=True`（默认开）：

```python
ok = (
    type(teacher_rl) is type(teacher_ref)                          # ① 必须是同一个类（同架构代码）
    and teacher_rl.vocab == teacher_ref.vocab                      # ② 词表大小一致
    and getattr(teacher_rl, "d_model", None) == getattr(teacher_ref, "d_model", None)  # ③ 隐藏维度一致
    and getattr(teacher_rl, "max_len", None) == getattr(teacher_ref, "max_len", None)  # ④ 上下文长度一致
)
if not ok:
    raise TeacherConsistencyError("teacher_rl 与 teacher_ref 必须共享架构/词表/隐藏维度/上下文长度")
```

四条全过才算"同一教师"，任一不过立刻抛 `TeacherConsistencyError`，**宁可报错也不产出错误的 Δ_T**。

#### 🔢 例子

```
情况 A：teacher_rl 和 teacher_ref 都是 CausalToyLM(vocab=64, d_model=48, max_len=64)
        → 四条全过 → 正常 build。
情况 B：teacher_rl.vocab=64，但 teacher_ref.vocab=128（有人传错了模型）
        → 第②条不过 → build 一开始直接抛 TeacherConsistencyError，
          不会等算完才发现 Δ_T 对不上。
```

#### ⚠️ 这个校验的局限

它只查**架构尺寸**（类名、词表、隐藏维度、上下文长度），**查不了"权重血缘"**——即它不能保证 `teacher_ref` 真的是 `teacher_rl` 训练前的那个版本。真实部署应进一步比对 `config.json`、tokenizer 哈希，甚至 checkpoint 的血缘/版本号。本 demo 用的是 toy 模型，结构校验已够。

---

## 3. Direct-OPD 的信用分配是怎么实现的

### 3.0 先搞懂：什么叫"信用分配"？

强化学习里，一个 response 由很多个 token 组成，最后得到一个总的好坏评价。**信用分配**就是回答："这个总评价，应该**算在哪些 token 头上**？每个 token 该记多少功 / 背多少锅？"

常见做法是学一个**价值网络**或算**标量 advantage**（每个样本一个数）。Direct-OPD 不一样——它**不学价值网络**，而是直接用一张**预先算好的逐 token 奖励张量 Δ_T**。

### 3.1 信用对象 Δ_T 是什么张量

`Δ_T = logπ_rl − logπ_ref`，形状 `(N, T, V)`：
- 对每个样本 `n`、每个 response 位置 `t`、每个**候选 token `v`**，给出"RL 教师比参考教师多偏好 `v` 多少"（log 概率差）。
- 它在 §2 的离线缓存里**一次性算好并预存**（dense 存全量 / 稀疏存 top-K 切片），训练期是**常量**，不反传。

关键：信用是**逐 (t, v)** 的分布级对象——它告诉你"在状态 `s_t` 下，**整个候选词表里每个动作**有多好"，而不只是"实际采的那一个动作有多好"。这比标量 advantage 信息量大得多。

### 3.2 分布级 PG 怎么把 Δ_T 分给当前策略（`losses.pg_loss`）

训练端 `_train_step` 现算 `s_cur`（当前 student 分布），与队列里带来的 `s_old`（rollout 快照）、缓存里的 `delta=Δ_T` 一起喂给 `pg_loss`，逐步张量操作：

```python
ratio     = (s_cur - s_old).exp()                      # (B,T,V) 逐 vocab 重要性比
unclipped = ratio * delta                              # 未裁剪
clipped   = torch.clamp(ratio, 1-eps, 1+eps) * delta   # PPO 裁剪
pointwise = torch.min(unclipped, clipped)              # 悲观下界
pg        = -(s_old.exp() * pointwise).sum(-1)         # 按 π_old 加权，对词表求和 → (B,T)
```

工程读法：
- `ratio` 把"当前策略 vs rollout 快照"的偏离**逐 vocab** 量化；
- `min(ratio·Δ, clip·Δ)` 是 PPO 的悲观裁剪，**逐 vocab 独立**做（防止某个 token 的概率被一次更新推得太猛）；
- 最后 `s_old.exp() * pointwise` 再 `.sum(-1)`，把整张词表按**行为策略 π_old 的概率**加权聚合成每个位置一个数。

于是一个 `(B,T,V)` 的 Δ_T，经"逐 vocab 裁剪 + π_old 加权求和"，被分配成 `(B,T)` 的逐位置信用，再被 mask 平均成标量损失。**整个过程没有任何显式的 advantage 估计器或价值网络**——信用完全来自离线 Δ_T 张量。

#### 🔢 一个极小的数字例子（T=1，V=4）

设某位置：
- 旧策略 `π_old = [0.1, 0.2, 0.3, 0.4]`（4 个 token 的概率）；
- 教师偏好 `Δ_T = [+1.0, -0.5, +0.2, 0.0]`（token0 很好 +1，token1 差 -0.5，token2 略好 +0.2，token3 中性）；
- 当前策略和旧策略一样（`ratio=1`，未裁剪）。

则该位置信用 = `Σ_v π_old(v)·Δ(v)` = `0.1×1.0 + 0.2×(-0.5) + 0.3×0.2 + 0.4×0.0` = `0.1 - 0.1 + 0.06 + 0` = **+0.06**。

PG 损失是 `-信用`，所以这一位贡献 `-0.06`（负损失=要增大概率的方向）。训练会让当前策略**把概率质量往 Δ 为正的 token（token0、token2）挪**，远离 Δ 为负的 token1——这就是"信用被分配给各 token"的实际效果。`min/clip` 的作用是：如果 `ratio` 已经很大（某个 token 概率被推过头），就改用裁剪值，**踩一脚刹车**。

### 3.3 稀疏支撑的工程衔接（真实词表）

当缓存是 top-K 稀疏时，训练端不存稠密 Δ_T，而是**按 student 自身的 top-K 支撑现场展开**（`_train_step` + `cache.delta_for_student_topk`）：

```python
s_topk  = torch.topk(s_cur, self.top_k_student, dim=-1)         # student 当前 top-K 支撑
delta_d = self.cache.delta_for_student_topk(idxs_dev, s_topk.indices)  # (B,T,V)，支撑外=0
loss_pg = pg_loss(s_cur, s_old, delta_d, mask, self.clip_eps)   # 内核不变，仍吃 (B,T,V)
```

`delta_for_student_topk`（`cache.py`）做的事：把缓存里教师的 top-K Δ 与 student 的 top-K token **按 token id 匹配**，匹配上的取教师 Δ、不匹配的取 0，再 `scatter_(-1, student_topk_ids, matched)` 回填成 `(B,T,V)`。这样**分布级 PG 内核零改动**，又能避开存储/搬运完整 `(N,T,V)`。

> 工程注记：支撑外置 0 是 Direct-OPD 的近似——"student 高概率支撑之外，教师偏移可忽略"。

### 3.4 KL 锚点：把"不漂移"也变成缓存查找

为防止更新跑偏，加一个对**初始 student 分布**的 KL 正则。它同样被工程化成"预先存 + 训练时查"：

- demo（小词表）：直接把初始 student 分布存成稠密 `ref_dists (N,T,V)`，用 `low_var_kl(s_cur, ref_dists[idxs], mask)`。
- 真实词表（防 `(N,T,V)` OOM）：只存初始分布的 top-K `(ref_ids, ref_logp) (N,T,Kr)`，训练时 `_ref_logp_at_student_topk` 按当前 student 支撑取回，用 `low_var_kl_support(s_topk.values, ref_at, mask)` 在支撑上求和。
  - 对"student 高概率但初始分布≈0"的 token 填 `ref_tail_logp≈-1e2`，给出强漂移惩罚。
  - ⚠️ 该支撑版是**有界近似**（省略支撑外非负尾部，系统性略低估真 KL），不是恒等替换。

### 3.5 监控

`expected_reward(dists, delta) = (dists.exp() * delta).sum(-1)` 给出 `E[Δ_T]`，训练日志里的 `reward`（当前策略）与 `adv_mean`（旧策略）就是它，用来确认信用分配确实在往正方向推（demo 里单调上升到 +0.7+）。

---

## 4. 关键工程取舍与已知边界

- **配置层级必须对齐**：部署开关（`dtype/cache_mode/top_k_*/offload_to_cpu`）的"定义层级"要和"读取层级"一致。曾在顶层定义却在 stage 子字典读取，导致云配置被静默忽略——已修（`pipeline.run` 把顶层键回填进 stage 配置，stage 子键优先）。
- **稀疏 KL 是近似**：top-K 支撑求和会略低估真 KL（方向安全，正则偏弱）。
- **分布级 PG vs 工业 token 级**：本代码损失是分布级（全词表算 ratio），要求完整 `π_old`。vLLM 只返 top-K，因此 `VLLMRolloutEngine` 在小词表（`vocab≤full_logprobs_cap`）走**精确重建**、大词表应改走 verl/slime 的 **token-level PPO**（引擎已另提供 `response_logprobs`）。大词表下把截断分布喂给 `pg_loss` 会得到 nan（`exp(s_cur+1e4)` 溢出），不是"不精确"而是数值崩。
- **L2/L5 编排互斥**：learner Megatron TP=2 与"rank1 当 rollout worker"的并发模型互斥（TP 集合通信死锁），已加护栏；L2 需 colocated 交替相位调度（learner TP=2 ↔ rollout vLLM TP=2 同驻双卡 + CPU offload）。
- **异步的陈旧度有硬上界**：`staleness_threshold`（默认 4）双截断，是"异步能多旧"的显式旋钮。
- **P0 工程化已落地**：`config.py` 用 pydantic `extra="forbid"` 把"未知键静默忽略"变"显式报错"；`main/tests/` 共 **39 个 pytest 文件**覆盖异步 / L2 子系统 / 配置 / 断点 / 指标 / 评估协议；`pyproject.toml` 注册 `fullstack-opd-v2` console script 并声明 `[test]`/`[gpu]` 可选依赖（详见 §4.5）。

---

## 4.5 工程化底座（P0）：可复现、可测、可打包

> 这一节补全前文未覆盖的**工程化层**——它们不改动算法内核，却是把 demo 变成"可交付工程"的关键。所有模块在 `fullstack_opd_v2/` 包内，已通过 **39 个 pytest 文件**覆盖（见 §4.5.9）。

### 4.5.1 配置：pydantic schema + 点分 CLI 覆盖（`config.py`）

`config.py`（516 行）用 **pydantic v2 + `extra="forbid"`** 强校验：任何拼错/多余的键直接 `ValidationError`，而非静默忽略。Schema 分层：

- `Stage0Cfg / Stage1Cfg / Stage2Cfg`：三阶段各自消费的键；
- `L2Cfg`（含 `L2CacheCfg / L2RolloutCfg / L2DisagreementCfg / L2HealthMonitorCfg / L2RefreshRatioCfg / L2SelectiveRolloutCfg / L2UtilityCfg`）：L2 子系统完整 schema；
- `RunCfg / LoggingCfg / MetricsCfg / DatasetCfg / EvalCfg / BaseCfg / CacheCfg`：`OPDConfig` 的其余顶层块。

`load_config()` 的流水线（已实测）：① 读 YAML → ② 与 `DEFAULT_CONFIG_V2` 深合并 → ③ 解析 `--set stage2.lr=1e-4` 这种**点分路径**覆盖 → ④ 顶层部署键（`dtype/top_k_*/offload_to_cpu`）**下渗**进对应 stage 子字典（修复早期"顶层定义、stage 子字典读取、云配置被静默忽略"的坑）→ ⑤ pydantic 校验。

> ⚠️ 下渗是 `extra="forbid"` 兼容的关键：`stage1` 只消费 `cache_mode/top_k_teacher`、`stage2` 只消费 `dtype/top_k_student/offload_to_cpu`，下渗在校验前完成，使 `config.yaml` 快照天然含下渗结果。

### 4.5.2 CLI 子命令（`cli.py`）

`cli.py`（386 行）提供 **6 个子命令**，统一经 `load_config` 解析：

- `train`：跑全栈（Stage 0/1/2），支持 `--resume` 续跑；
- `cache`：只建 Lightning 离线缓存（Stage 1）；
- `eval`：评 checkpoint 健康信号；
- `info`：打印解析后完整配置（调试点分覆盖结果）；
- `eval-aime`：真实模型 AIME24/25 评估（`--model` 直评 / `--run-dir` 桥接读 `config.yaml` 的 `eval.model_path`）；
- `eval-holdout`：holdout 集评估（与 `_eval_holdout` 线程呼应）。

`pyproject.toml` 注册 console script `fullstack-opd-v2 = fullstack_opd_v2.cli:main`，`pip install -e .` 后可直接 `fullstack-opd-v2 train ...`。

### 4.5.3 可复现 run 目录（`run.py`）

`RunManager` 把"一次训练"变成可复现单元：`runs/<timestamp>/` 下快照 `config.yaml`（解析后完整配置）、`metrics.csv`、`logs/train.log`、`checkpoints/step_<N>.pt`。无 `--run-dir` 时自动时间戳建目录，`--resume` 时复用并续跑。

### 4.5.4 断点续跑（`checkpoint.py`）

`CheckpointManager.save/load/resume` 存 `state_dict + weight_store_version + step + cfg + metrics`，支持从最新断点续跑。针对多学生并发显存峰值，提供 `_release_cpu_memory`（调用 `malloc_trim` 释放 RSS）与 `_opt_state_to_cpu`（优化器状态卸 CPU）。

### 4.5.5 指标追踪（`metrics.py`）

`MetricsRecorder` 后端 `csv`（默认）或 `wandb`；**wandb 缺失自动 fallback CSV**（不崩）；append 续写（resume 不丢历史）、`flush_every`、线程锁保证多线程安全。每步落 `loss/pg/kl/adv/reward/age/version` 等。

### 4.5.6 结构化日志 / 类型异常（`logging.py` / `exceptions.py`）

`setup_logging` 统一时间戳+级别+控制台+文件，`name` 幂等（重复调用不叠 handler）。`exceptions.py` 把散落的 `RuntimeError/ValueError` 收敛为 `OPDError` 子类（`ConfigError/DataError/ModelError/CheckpointError/TrainingError`），调用方可按阶段精确捕获。

### 4.5.7 可插拔数据 / 模型 / 缓存接口（`data.py` / `model_factory.py` / `cache_store.py`）

- `data.py`：`DataLoader`(ABC) + `ToyDataLoader` + `JsonLinesDataLoader` + `build_data_loader`——Skywork-OR1 论文数据经 `dataset.type=jsonl` 零代码改动接入（见 §8.4 接口就绪）。
- `model_factory.py`：`build_model(cfg, device, role)` + `HFCausalLM` 适配器（接真实 HF 模型；骨架，需 GPU 验证）。
- `cache_store.py`：`DiskTeacherCache`（mmap 磁盘教师缓存，与 `TensorTeacherCache` 同接口，解决 50K×8192 显存墙；metadata 13 键一致性校验）。

### 4.5.8 AIME 评估（`eval_aime.py`）

`AimeEvaluator` + `extract_answer` / `normalize_answer` / `format_prompt`：支持 AIME24/25，boxed 提取 + sympy 等价判定，`--n-samples`（`ave@N`）+ 逐题落盘 + **resume 跳过已完成题**（修复长生成中断全丢）。

### 4.5.9 测试与打包（P0 落地）

- **测试**：`main/tests/` 共 **39 个 pytest 文件**（含 `test_adaptive_cache`、`test_l2_integration`、`test_l2_rollout`、`test_vllm_dist`、`test_run_s2_real_parallel`、`test_budget_eval`、`test_pipeline`、`test_config`、`test_perf_equivalence`、`test_checkpoint`、`test_metrics`、`test_eval_aime`、`test_data`、`test_model_factory`、`test_cache_store` 等），覆盖异步、L2 子系统、配置、断点、指标、评估协议。
- **打包**：`pyproject.toml`（`name=fullstack-opd v2.0.0`）声明 `torch/pydantic/pyyaml` 依赖、`[test]=pytest`、`[gpu]=vllm/megatron-core/ray` 可选依赖、`[project.scripts] fullstack-opd-v2`、`packages=[fullstack_opd, fullstack_opd_v2]`、`pytest.testpaths=tests`。

---

## 4.6 L2 Adaptive Teacher Cache 子系统（已实现并接线）

> ⚠️ **状态修正**：报告旧版称"L2 未实现"。实际 `adaptive_cache.py`（1291 行）已落地完整子系统，且已在 `scheduler.py`（`_train_step_refresh` / `train_refresh_phase`）与 `pipeline.py`（L2 交替相位循环）**接线**。`l2.enabled=False`（默认）时退回 L0/L1 静态路径，故默认行为不变、零回归风险。

### 4.6.1 设计目标

L2 把"训练期用**当前学生** rollout 一批 $(p,r')$、教师实时算 $\Delta_T$、写动态缓存"从设计稿变成可运行子系统，并把"什么时候 refresh、refresh 多少、refresh 哪些 prompt"交给**自适应控制器**而非硬编码。默认 `l2.enabled=false` 退回 L0/L1。

### 4.6.2 核心组件（`adaptive_cache.py`）

| 类 | 职责 |
|---|---|
| `RefreshRingBuffer` | 动态缓存环形缓冲（capacity/top_k/vocab）；`append` 写教师 top-K Δ + 行为 `s_old` 支撑 + ref 锚点支撑；`delta_at_student_topk` / `s_old_at_student_topk` / `ref_anchor_at_student_topk` 按当前学生支撑取回；`_utility`（衰减效用）/ `_value_threshold`（价值阈值）；`state_dict` / `load_state_dict` 支持续跑；`mean_disagreement` |
| `DisagreementComputer` | 算教师 RL/ref 分布对学生的分歧 `D_i^abs`，作为刷新质量信号（`compute` / `gather_chosen_logp`） |
| `CacheHealthMonitor` | 七维观测（命中率/失效/重复/刷新年龄 p95 等）+ `classify` 状态机 + `record` + `_reason` 文本诊断；observe-only，不影响训练信号 |
| `DynamicRatioController` | 三信号（base_age / policy_drift / refresh_quality）→ 自适应 $\alpha$；`mode="adaptive"/"fixed"`；`cold_start_adjust` 冷启动调比 |
| `PromptStateStore` | 每 prompt 状态（reward / disagreement / resp_len / 复用计数 → `novelty`） |
| `RefreshSelector` | 候选池两阶段价值选择（80% value + 20% coverage）：`select` / `select_with_budget`（Budget-Aware 分桶） |
| `run_refresh_phase` | 学生短 rollout → 4 次 `response_dists` 取 logp → 算 `D_i^abs` → `append` 进 ring buffer |
| `compute_rollout_metrics` | 刷新相位产出 `rollout/accuracy_proxy` / `rollout_tokens` / `useful_per_token`（Performance / RolloutTokens / Eff） |
| `assign_budgets` / `enforce_budget` / `group_by_budget` | Budget-Aware 分桶与 token 预算执行（§4.7） |

### 4.6.3 调度器接线（`scheduler.py`）

- `train_refresh_phase(rb, alpha, n_refresh_steps)`：从 ring buffer 采 `n_refresh_steps` 批做 refresh 训练；$\alpha$ 由 `DynamicRatioController` 决定。
- `_train_step_refresh(done, rb_idxs, rb)`：refresh 池样本的 teacher-free 稀疏 top-K PG + KL（双池 feeder，G1 闭环）；`REFRESH_LOG_RATIO_MAX=3.0` 对 log-ratio 全局 clamp（IS 上界）+ 纵深防御屏蔽支撑外 `log0`；`_REFRESH_CHUNK=4` 把整批 `(M,T,V)` 拆 chunk 累积，规避双卡并行 OOM。
- 双池契约：**base 池 s_old 恒用 HF/toy worker**（vLLM 只负责 L2 生成，IMP-2/P1）；**仅 refresh 池**（存 rollout 时刻行为 `s_old`）受陈旧度约束（IMP-2/P3）。

### 4.6.4 管线交替相位（`pipeline.py`）

`pipeline.py` 的 L2 分支（任务 6.1，§13 整合）：启用时**保留** `teacher_rl/teacher_ref` + `warmup_student` 供 refresh rollout（非 L2 仍释放，L0/L1 静态路径零开销）；`l2.enabled` 时进入**交替相位循环**——base 训练相位与 refresh 训练相位交替，`refresh_buffer` 的 `state_dict` 参与 checkpoint 续跑。`async`（默认）与 L2 交替相位互斥时由护栏报错。

### 4.6.5 P-OPD：纯 on-policy 交替相位（2026-08-31，当前唯一训练路径）

**动机**：官方代码重算证明 Eq.13 only_stu 口径信号强（Skywork +0.539 / MATH500 +0.596，RC4 推翻）；
失败归因转 **RC1（固定 D off-policy 主嫌）** → 纯 on-policy 化（用户决策，回滚 v2 基线后重建）。

**架构**（`l2.pure_refresh=true` + `stage1.skip=true`）：
```
[构造] 静态 prompt 集 → 加载教师对（_stage0_teachers，跳过 Stage0 RL）→ 占位 cache（仅 top_k/vocab）
[循环] while step_done < n_total:
  [rollout 相位] 当前学生 rollout（m_refresh × n_rollout 条）
     → 教师 rl/ref only_stu 前向（_rl_ref_delta_only_stu）算 Δ
     → append ring buffer（ids=学生 top-K，delta，行为 s_old，ref 锚点）
  [训练相位] train_refresh_phase：从 ring buffer 采样 _train_step_refresh（teacher-free，α 冻结 1.0）
     → step_done 推进；连续空相位 > max_empty_phases 明确失败
```

**only_stu 教师 Δ**（`_rl_ref_delta_only_stu`，adaptive_cache.py）：
$$\Delta_T = \log\pi_{\text{rl}}(y|s)\big|_{\text{学生 top-}K} - \log\pi_{\text{ref}}(y|s)\big|_{\text{学生 top-}K}$$
- 教师对**学生 top-K 完整支撑**取 logp（`gather(logits, ids) − logsumexp`，与官方 `_compute_teacher_top_k_log_probs` 数学一致），**无交集稀释**（对比旧 `_rl_ref_delta_k`：教师 rl 自身 top-K + ref gather，交集才有效）。
- 跨词表：学生 id 超教师词表 → Δ 置 0（clamp + mask）；vLLM prompt_logprobs 无法取任意 ids logp → 教师前向一律 HF（per-chunk 省显存）。
- **F1 约束**：`student_top_k == cache.top_k`（only_stu Δ 写宽与 ring buffer 槽位宽一致，否则 append shape 崩）。

**ring buffer 学生支撑**：`RefreshRingBuffer.ids = s_old_ids`（行为学生 top-K），训练时
`delta_at_student_topk` 在 s_cur top-K 上匹配（refresh 紧跟 rollout，匹配率近 100%）。

**vLLM 逃生舱**（`rollout_weight_sync="off"`，rollout_vllm.py）：vLLM 0.16 NCCL WeightTransferEngine
在 Blackwell/Ada 均报 `Expected ... got:cuda`（E18）→ `LLM.apply_model(lambda m: m.load_weights(...))`
直拷学生权重（仅 `tp_size=1`），保 on-policy 且不依赖 NCCL。

**预算模式 T 恒定**：`max_b = max(budget_set)`（非每相位 `budgets.max()`）——冷启动 v 全等 vs
后续分位数展开会导致 ring buffer 槽位 T 变化 shape 崩。

**数据量**：每相位 `m_refresh × n_rollout` 条 on-policy 样本入 ring buffer（容量 `refresh_size`）；
300 步 / `t_train=2` = 150 相位 × 32 = 4800 条 on-policy（默认配置）。

**验证**：`test_onpolicy_refresh.py` 9 项（only_stu 与官方公式一致/跨词表 clamp/学生支撑语义/
纯 refresh 无 base/断点 round-trip）；全量 599 passed（重建时）。

> 💡 纯 on-policy 下 IS 几乎不需要（refresh 样本 `s_old` 用最新快照算，ρ≈1）；`l2.enabled=false`
> 明确报错（base 池已删）。

---

## 4.7 预算感知评估（Budget-Aware Evaluation，Stage 1.6 / 1.7）

把"完整 CoT + EOS"的隐式必要条件重构为 **Accuracy(B)**——$B$ = max reasoning token budget，在 Base/L0/L2 间**统一预算**公平比较。

- `budget_eval.py`（468 行）：`extract_final_answer`（boxed → Final Answer marker → benchmark parser → fallback）+ `BudgetEvaluator(AimeEvaluator)`（逐位 EOS 判定 + 双指标 outcome/prefix + token 记账）+ `run_matrix` 跑 `Base/L0/L2 × B∈{256,512,1024,2048,4096}` → md 报告 + 4 图。
- `budget_curve.py`（344 行）：在 `budget_eval.all_results` 上加效率维度——Budget Curve / AUC / **nAUC**（消除 benchmark 量纲差）/ Efficiency / GainPerToken / ΔA(OPD Gain) / B(A*)。伦理约束：**效率一律用真实 generated reasoning tokens，绝不拿 `max_new_tokens` 当 E[L]**。

---

## 4.8 L2 消融实验矩阵（`experiment.py`）

`experiment.py`（349 行）把"L2 各模块贡献"做成可重跑的 ablation：

- `EXPERIMENT_MATRIX`（**E0-E6**，§10 累加语义）：E0 关 L2 基线 → E1 加 fixed refresh → E2 加 Disagreement → E3 加 Health Monitor → E4 加 Dynamic Ratio → E5 加 Selective Rollout → E6 全开 + Random Rollout 对照。
- `STAGE2_ROLLOUT_MATRIX`（**S2_E0-E3**）：P-OPD 纯 on-policy 对照 + OPD + 短 rollout 512/1024/2048。
- `STAGE3_MATRIX`（**S3_E0-E2**）：Budget-Aware Selective Rollout 对比（random 单预算 / selective 单预算 / selective + adaptive 预算）。
- 工具：`build_config`（点分覆盖生成 toy/CPU 可跑配置）、`run_experiment` / `run_matrix` / `save_results`、`plot_experiments`（8 图，含"teacher compute vs perf"最重要一张）、`aggregate_stage3`（Performance/RolloutTokens/Eff 聚合）。
- `report_stage2.py`：消费 S2 训练 summary + budget_eval 长预算结果，产出 Q1-Q4 markdown 报告（无数据字段以 `—` 占位，如实标注待服务器实跑）。

> 这些是**实验驱动验证 L2 子系统**的工程骨架，当前 toy/CPU 可跑协议抽象、真实长预算待服务器实跑（见 §5.3 / §9.2）。

---

## 附 1：机制 ↔ 代码 速查表

| 机制 | 类/函数 | 文件 |
|---|---|---|
| 4 阶段异步线程 | `AsyncBatchedScheduler.run / _prompt_feeder / _rollout_collector / _teacher_scorer / _train_dispatcher` | `scheduler.py` |
| 单步训练内核（两版调度共用） | `AsyncBatchedScheduler._train_step` | `scheduler.py` |
| 版本号权重快照 / 按需加载 | `WeightStore.publish / acquire_if_newer` | `buffer.py` |
| 陈旧度双截断 | `StalenessQueue.put / get / advance_version` | `buffer.py` |
| 离线教师缓存（dense/top-K） | `TensorTeacherCache.build / get_delta / delta_for_student_topk` | `cache.py` |
| teacher 一致性校验 | `TensorTeacherCache.build` 开头 + `TeacherConsistencyError` | `cache.py` |
| 信用分配 PG | `pg_loss` | `losses.py` |
| KL 锚点（稠密/稀疏） | `low_var_kl / low_var_kl_support` | `losses.py` |
| 监控 E[Δ_T] | `expected_reward` | `losses.py` |
| NCCL 权重广播（L5） | `WeightBroadcaster.push_async / pull_async` | `scheduler.py` |
| Ray rollout worker（L5） | `_RayRolloutWorkerImpl / DistAsyncScheduler` | `scheduler.py` |
| vLLM rollout（L3） | `VLLMRolloutEngine.response_dists / update_weights` | `rollout_vllm.py` |
| Megatron TP=2 learner（L2） | `MegatronCausalToyLM / parallelize_learner_tp2` | `model_megatron.py` |
| 异步轨迹 (s,a,logp_old) 的承载 | `s_old` 随 `_rq` 元组携带；`a`/`s` 由 `idxs` 索引现取（见 §1.8） | `scheduler.py` |
| 配置 schema（pydantic, extra=forbid） | `OPDConfig` / `load_config` | `config.py` |
| CLI 子命令 | `cli:main`（train/cache/eval/info/eval-aime/eval-holdout） | `cli.py` |
| 可复现 run 目录 | `RunManager` | `run.py` |
| 断点续跑 | `CheckpointManager`（state_dict+version+step+cfg+metrics） | `checkpoint.py` |
| 指标追踪 | `MetricsRecorder`（csv/wandb fallback） | `metrics.py` |
| 结构化日志 / 类型异常 | `setup_logging` / `OPDError` 子类 | `logging.py` / `exceptions.py` |
| 可插拔数据 / 模型 / 磁盘缓存 | `build_data_loader` / `build_model`(HFCausalLM) / `DiskTeacherCache` | `data.py` / `model_factory.py` / `cache_store.py` |
| AIME 评估 | `AimeEvaluator` / `extract_answer` | `eval_aime.py` |
| L2 Adaptive Teacher Cache | `RefreshRingBuffer` / `DisagreementComputer` / `CacheHealthMonitor` / `DynamicRatioController` / `PromptStateStore` / `RefreshSelector` / `run_refresh_phase` | `adaptive_cache.py` |
| L2 调度器接线 | `_train_step_refresh` / `train_refresh_phase` | `scheduler.py` |
| 预算感知评估 | `BudgetEvaluator` / `run_matrix` / `budget_curve` 指标 | `budget_eval.py` / `budget_curve.py` |
| L2 消融矩阵 | `EXPERIMENT_MATRIX`(E0-E6) / `STAGE2_ROLLOUT_MATRIX` / `STAGE3_MATRIX` | `experiment.py` |

---

## 附 2：术语白话速查

| 术语 | 一句话白话 |
|---|---|
| **快照 snapshot** | 把当前模型参数整个复制一份冻住，供别人读"某一完整时刻"的一致权重 |
| **版本号 version** | 每发一次新快照就 +1 的计数器，给"新旧"一个可比较的刻度 |
| **陈旧 staleness / age** | 某份 s_old 的权重版本，落后当前最新版本多少步 |
| **双截断** | 入队时查一次、取出时再查一次陈旧度，超阈值就丢弃 |
| **背压 backpressure** | 下游慢→队列满→上游被迫丢帧/放慢，速度自动向最慢环节看齐 |
| **不死锁** | 所有 put/get 带 timeout，没有无限期阻塞，就不会互相卡死 |
| **教师离线** | 训练前把教师对所有数据的输出一次算完存成表，训练时只查表 |
| **Δ_T** | 逐 (位置, token) 的"RL 教师 vs 参考教师"偏好差，是信用分配的原料 |
| **s_old / s_cur** | s_old=rollout 时刻的旧分布（可当分母、允许陈旧）；s_cur=当前分布（带梯度、要更新） |
| **稀疏 top-K 支撑** | 只存/只用概率最高的 K 个 token，省显存，其余置 0 |
| **动作 a** | 本 OPD 里 = `responses`（固定离线数据），按 `idxs` 索引现取，不持久缓存 |
| **rollout 阶段 log prob** | = `s_old`，随批次入队，learner 重算不了（旧权重已消失），必须携带 |
| **重放缓冲 replay buffer** | 论文/工业用大容量持久缓冲反复复用样本；本实现是有界在途队列（只走一遍、超期即丢），非 replay buffer |
| **L2 Adaptive Teacher Cache** | 训练期自适应刷新教师缓存的子系统：ring buffer + 分歧/健康/价值/选择器控制器，把"何时刷、刷多少、刷哪些"自动化 |
| **P-OPD** | 纯 on-policy 交替相位 OPD：跳过预计算教师得分，refresh 相位现算信号，比标准 OPD 更"在线" |
| **Budget-Aware Evaluation** | 统一 reasoning token 预算 B 下公平比较 Base/L0/L2 的评估框架（Accuracy(B) + 预算曲线/AUC/nAUC/效率） |
| **run 目录** | 一次训练的独立可复现目录（config.yaml 快照 + metrics.csv + train.log + checkpoints/） |

---

# 第二部分 · 训练分析与实验报告

> 本部分回答"这个 OPD 流水线实际训练效果如何、花多少资源、数据是什么"。
> **每个分数必须注明协议**：本项目踩过 pass@1 与 ave@32 混报导致数字误导的坑（见 §5.1）。
> 未实测的项如实标注【待补】，不编造。

## 5. 训练前后 benchmark 分数（含评估协议声明）

### 5.1 评估协议（权威口径，与 Direct-OPD 论文对齐）

论文 `train_justrl_qwen.sh` 的验证协议（launch 脚本原样）：

| 设置 | 论文值 | 本项目 eval-aime 对应参数 |
|---|---|---|
| 每 prompt 采样数 | **32**（`VAL_N=32`） | `--n-samples 32` |
| 采样温度 | **0.7**（`val_kwargs.temperature=0.7`） | `--temperature 0.7` |
| Top-p | **0.95**（`val_kwargs.top_p=0.95`） | `--top-p 0.95` |
| 最大生成长度 | **31,744**（`MAX_VAL_RESP_LENGTH`；Qwen3 config 上限 40960） | `--max-new-tokens 31744` |
| prompt 模板 | boxed（"reason step by step… within \\boxed{}"）+ **chat_template 包裹**（verl `apply_chat_template`） | `--prompt-style boxed --chat-template` |
| 评分 | `\boxed{}` 级联提取 + **sympy/mathd 数学等价判定**（`ttrl_math.grade`） | `--scoring sympy` |
| 指标 | **ave@32**（每题 32 采样中答对比例的平均） | `--metric ave` |
| 数据集 | aime24 / aime25 / hmmt_feb | `--datasets AIME24 AIME25` |

**两个必须区分的口径**：
- **ave@N**：`accuracy = 每题 N 采样中答对比例的均值`（论文口径）。n=32 → ave@32；n=8 → ave@8。
- **pass@1**：`correct/total`（任一采样答对即算对）。本项目早期误把 pass@1 当 ave@32 报告，
  导致"33.3 / 23.3 / 30.0"这类规整数字被质疑像假数据——实为 pass@1 的凑整巧合。

### 5.2 已实测分数（短生成协议，pass@1）

> ⚠️ 以下为**短生成**（max_new_tokens≈2048）+ boxed 模板测得的 **pass@1**，**非论文 ave@32 协议**。
> 数值会系统性低于论文协议（短生成截断在 CoT 中途、模型未写出 `\boxed{}`）。来源：
> 来源：原 `docs/multistudent_cloud_training_report.md`（已并入本文档并删除，避免重复维护）。

| 学生 | 基座 AIME24 | 基座 AIME25 | 训练后 AIME24 | 训练后 AIME25 | 训练步数/时长 |
|---|---|---|---|---|---|
| Qwen3-1.7B | 6.7% (2/30) | 10.0% (3/30) | 6.7% | **13.3%** (+3.3pp) | 60 步 / ~137s |
| Qwen3-4B | — | — | **16.7%** | **16.7%** | 60 步 / ~137s |
| Qwen3-7B | — | — | —（词表不匹配，未蒸馏） | | |

**1.7B 中间断点曲线**（200 步，短生成 pass@1，验证过训）：

| 断点 | AIME24 | AIME25 |
|---|---|---|
| step40 | 10.0% | 6.7% |
| step80 | 3.3% | 13.3% |
| **step120 ⭐** | **13.3%** | **16.7%** |
| step160 | 3.3% | 16.7% |
| step199（终） | 3.3% | 6.7% |

**结论**：step120 是最佳检查点；step199 大幅回落 → **200 步过训**，建议早停 ~120 步。
> ⚠️ 30 题小样本下 3.3%=1/30、13.3%=4/30，差距仅 3 题——短生成 pass@1 方差极大，
> 需 ave@32 长生成协议重验（见 §5.3）。

### 5.3 论文协议 ave@32 长生成评估【待补】

**为什么待补**：论文协议（32768 token 长生成）在单卡 + transformers 原生生成下每个采样
需 30-60 分钟，960 采样 × 多模型需数百小时，**现实不可行**。已两次尝试后终止（详见 §9.2）。

**已确认的工程事实（供后续重跑）**：
- 短生成（2048）下 **30/30 题全部无 `\boxed{}`**——模型截断在 CoT 中途，从未写出最终答案；
  `preds_all` 非空是 `extract_answer` 的"最后数字回退"抓的 CoT 中间数，是噪声不是能力。
- 因此"2048 短生成分数"不是模型真实水平，而是**截断噪声**；长生成才是有效协议。

**待补表格**（服务器恢复后用 ave@32 长生成协议填充）：

| 模型 | AIME24 ave@32 | AIME25 ave@32 | 备注 |
|---|---|---|---|
| 1.7B 基座（Qwen3-1.7B） | 【待补】 | 【待补】 | 对齐论文 Qwen3-1.7B 起点 |
| 1.7B ms_step120 | 【待补】 | 【待补】 | 验证是否真最优断点 |
| 1.7B ms_step80 / step160 | 【待补】 | 【待补】 | 验证 3.3% 是否噪声 |
| 4B v3_step59 | 【待补】 | 【待补】 | 对齐论文 Qwen3-4B |
| 7B 基座 / 7B 学生 | 【待补】 | 【待补】 | DeepSeek-R1-Distill-7B |

**协议调整建议（用户 2026-08-14 提出，权衡单卡耗时）**：
- 采样数降为 **n=8 或 n=16**（ave@8 / ave@16，仍保"平均正确率"口径，方差略大）；
- **max_new_tokens 下调**（如 8192，覆盖模型自然 CoT 收尾、比 32768 快 ~4×）；
- 或两者结合（n=8 + 8192，估算单模型单数据集约 3 小时，若启用 flash_attn）。
- **关键加速**：eval-aime 应传 `attn_implementation="flash_attention_2"`（服务器已装 flash_attn 2.8.3），
  当前未传默认走 SDPA，长序列开销大（见 §6.4）。

## 6. 训练与评估的显存占用分析

### 6.1 每参数显存常量（真实模型，RTX PRO 6000 96GB）

| 项 | 1.7B (Qwen3-1.7B) | 4B (Qwen3-4B) | 7B (DeepSeek-R1-Distill-7B) |
|---|---|---|---|
| 权重 bf16 | ~3.4 GB | ~8 GB | ~14 GB |
| KV cache / token（GQA） | 112 KB（8 KV 头） | — | — |
| 满长 32768 KV cache | ~3.6 GB/序列 | ~8 GB/序列 | ~12 GB/序列 |

> 细节见 `docs/GPU_MEMORY_AND_PARALLEL_PLAN.md`（§1 每参数常量 + §3 分阶段显存账）。

### 6.2 训练阶段显存账（Stage 0/1/2）

| 阶段 | 内容 | 显存特点 |
|---|---|---|
| Stage 0 教师 RL | 1.5B 弱教师 REINFORCE | 微（batch 小） |
| Stage 1 离线缓存 | 教师对全 D 推理 Δ_T | **Δ_T 大张量 (N,T,V)**；dense 在真实词表下 233GB 超限 → 用 **top-K 稀疏**（`top_k=256`，↓1000×） |
| Stage 2 Direct-OPD | learner + scorer 同卡 colocate | 学生权重 + 快照 + Δ_T 缓存 + 优化器 |

**实测踩坑（多学生并发）**：
- 每进程驻留 student+worker+teacher+优化器，7B 峰值 **94.7GB 满** → warmup_student 条件建 + batch 降 + `expandable_segments`；
- 4B 用 bnb `adamw_8bit` 反而把权重转 fp32 更占（94.3GB 满）→ **改 `optimizer=adam`（fp32）+ batch=2**。

### 6.3 评估阶段显存（短生成 2048）

- 单模型短生成：权重 + 短 KV + logits ≈ 20-30GB，可双模型同卡并行（1.7B 两进程 ~14-19GB 稳定）。

### 6.4 评估阶段显存（长生成 32768）—— 教训

实测双模型并行长生成：基座 49GB + 学生 45.7GB = **94.7GB / 97.9GB，余量仅 3.2GB**。

**组成分析（为什么远超"权重+KV"预估 ~14GB）**：

| 项 | 估算 | 说明 |
|---|---|---|
| 权重 bf16 | 3.4 GB | — |
| KV cache（满长, batch2） | ~7.5 GB/模型 | 28×8×128×2B ×2 序列 ×32768 |
| **logits 张量** | **~20-25 GB** | batch×seq×vocab(151936)×bf16——**长序列下是隐形显存杀手，最易被低估** |
| prefill 激活 | ~10-30 GB | 未显式用 flash_attn，SDPA 长 prefill 中间缓冲大 |

**关键教训**：
1. **长序列（32K）× 大 vocab（152K）的 logits 是最大隐形开销**，远超 KV cache；
2. eval-aime **未显式传 `attn_implementation="flash_attention_2"`** → 默认 SDPA，长序列开销大；
   flash_attn 已装（2.8.3）却未启用，是浪费的 3-10× 加速潜力；
3. batch=2 让 logits/KV 翻倍，是 94.7GB 逼近上限的直接原因；batch=1 则余量充足（~30GB）。

## 7. 训练与评估的用时分析

### 7.1 训练阶段耗时（实测，2×RTX PRO 6000）

| 阶段 | 1.7B | 4B |
|---|---|---|
| Stage 0（小模型 RL） | 1.9s | 2.0s |
| Stage 1（离线缓存 Δ_T） | 42.0s | 47.6s |
| Stage 2（异步训练） | 168.3s（200 步） | 87.5s（60 步） |
| **总计** | **212.2s** | **137.1s** |

- 端到端（含缓存构建）≈ **2.3~3.5 分钟**完成一个学生的完整 OPD 蒸馏。
- 时间大头是 Stage 1 离线缓存（~42-48s）——**"离线教师对"方案的收益点**（Stage 2 无 live teacher 前向）。

### 7.2 评估耗时（实测与估算）

| 协议 | 单数据集耗时 | 说明 |
|---|---|---|
| 短生成 2048, n=1（贪心） | ~2-4 分钟 | 快，但分数是截断噪声 |
| 短生成 2048, n=32 | ~75 分钟（4B）/ ~70 分钟（7B） | batch=8 |
| **长生成 32768, n=32** | **20-40 小时/模型（实测单采样 ~50 分钟）** | 单卡不可行，已终止 |

**实测 decode 速度**（1.7B，短序列）：~27.7 tok/s。长序列（32K KV 压力 + 双进程共享卡）显著更慢。

### 7.3 吞吐与 batch 加速

- batch=1 → batch=2：序列级吞吐提升 ~50-80%，但显存翻倍（logits/KV）；
- 长生成 batch=1 单卡余量充足，batch=2 逼近上限（94.7GB）；
- **未启用 flash_attn 是最大浪费**：启用后长序列 decode 预计提速 3-10×。

## 8. 训练数据构成分析

### 8.1 数据来源与划分

| 用途 | 数据集 | 规模 | 说明 |
|---|---|---|---|
| **训练（当前）** | 默认/GSM8K 数学 | 学生训练集 | **⚠️ 非论文数据**。实际训练用了默认数据路径（未接 Skywork） |
| **训练（论文要求，待落地）** | Skywork-OR1-RL-Data math split | 105,055 条（计划子集 10K） | **【待办】** 规格已设计、未下载未转换（见下方 §8.4） |
| **评估** | AIME24（Maxwell-Jia/AIME_2024） | 30 题 | 竞赛数学，答案 3 位整数 |
| **评估** | AIME25（yentinglin/aime_2025） | 30 题 | 同上 |
| **评估** | HMMT Feb（论文数据集） | — | 论文有，本项目未跑 |

> ⚠️ **训练/评估数据不同源**：本项目当前训练用默认/GSM8K + AIME 评估；论文用 Skywork-OR1 + DAPO 模板。
> 即便评估协议完全对齐（ave@32），**当前**绝对分数与论文仍会有差距——预期"量级对齐"而非"逐点相等"。

### 8.4 Skywork 训练数据落地【待办，未完成】

**状态：规格已设计（`docs/specs/2026-08-13-align-directopd-experiment-design.md` §四），
但数据下载与转换从未执行**——训练实际未用 Skywork。用户 2026-08-14 指出该疏漏。

**计划管道**（服务器恢复后执行，数据接口已就绪零代码改动）：
1. **下载**：Skywork-OR1-RL-Data `data/math-00000-of-00001.parquet`（105,055 条，服务器学术代理）；
2. **转换**：跑 `Direct-OPD/scripts/prepare_skywork_math.py` → DAPO 模板 prompt + ground_truth；
3. **子集采样**：10.5 万条随机采样 10K → 转 jsonl（`{"prompt": <DAPO 包装>, "response": <student 生成>}`）；
4. **response 标签**：用初始 student `generate_batch` 对每个 DAPO prompt 生成响应（on-policy，
   复用 `stage1.warmup_source=student_init` 机制）；
5. **加载**：config 加 `dataset.type=jsonl` + 路径，走现有 `JsonLinesDataLoader`
   （`fullstack_opd_v2/data.py` 已实现，无需改代码）。

### 8.2 prompt 模板与长度配置

| 配置 | 值 |
|---|---|
| 训练 prompt 模板 | 数据集自带（GSM8K 数学题） |
| 评估 prompt 模板 | boxed（`"reason step by step… within \\boxed{}"`）；论文另用 DAPO 模板 |
| `max_prompt_length`（论文） | 1024 |
| `max_response_length`（论文训练） | 2048 |
| `MAX_VAL_RESP_LENGTH`（论文验证） | 32768 |
| 本项目评估 `max_new_tokens` | 短生成 2048 / 长生成 32768（实测单卡不可行，见 §5.3） |

### 8.3 Δ_T 缓存模式与数据量影响

| 配置 | 值 | 影响 |
|---|---|---|
| `stage1.cache_mode` | topk（GPU/真实词表） | Δ_T 只存 top-K，体积 ↓1000×（vs dense 233GB OOM） |
| `top_k_teacher` | 256 | 每 (n,t) 位置存 256 个 token 的教师偏好 |
| `top_k_student` | 256 | 训练时按学生 top-K 展开支撑 |
| `warmup_M` / `warmup_source` | 0 / none（ms 系列） | 未启用 L1 暖缓存（L1 开启时数据量 N×(1+M)） |

> L1 暖缓存：`warmup_M=4` + `warmup_source=student_init` 时，每 prompt 的缓存数据从 1 条
> 扩到 5 条（N×(1+M)）；`mix` 则 9 条（N×(1+2M)）。本项目 ms 系列用 0/none（L0）。

## 9. 已知边界、长生成教训与复现

### 9.1 未实现项 / 已知边界（工程）

- **L2 周期刷新**已实现并接线（`adaptive_cache.py` + `scheduler.py` `_train_step_refresh` / `train_refresh_phase` + `pipeline.py` 交替相位），默认 `l2.enabled=false` 退回 L0/L1；真实长预算消融（E0-E6 / S2 / S3）待服务器实跑（见 §4.6 / §4.8）。
- **L3 全在线**为骨架（`VLLMRolloutEngine` + 实时 Δ_T 路径），未完整跑通；
- **训练数据未用论文 Skywork**：当前训练走默认数据，Skywork-OR1 math 仅完成规格设计未落地
  （见 §8.4【待办】）——这使当前分数无法与论文直接逐点比对；
- **7B 未蒸馏**：词表硬约束（OPD 要求学生=教师同词表；student=152064 vs teacher=151936）→
  非显存问题，是 OPD 机制硬约束；
- **分布式 L5** 是带护栏骨架（ray/megatron/vllm 缺失时报错），未完整跑通；
- **重放缓冲**：本实现是有界在途队列，非论文的持久 replay buffer（样本只走一遍）。

### 9.2 长生成评估失败教训（32768 单卡不可行）

**背景**：论文协议（32768 token 长生成）在单卡 + transformers 原生生成下不可行，两次尝试后终止。

**失败时间线**（2026-08-13/14）：
1. 4B 长生成（batch=1）→ OOM 历史；chunk 修复后跑通短生成；
2. 1.7B 基座+学生双并行长生成（batch=2）→ 显存峰值 94.7GB/97.9GB 逼近上限；
3. 单采样实测 ~50 分钟（超长 CoT 到 32768 上限），960 采样需数百小时 → 终止。

**根因（三层）**：
1. **decode 慢**：1.7B 短序列仅 27.7 tok/s；长序列（32K KV）显著更慢；
2. **logits 显存大**：长序列 × 151936 vocab 的 logits 张量 ~20-25GB（被低估的隐形杀手）；
3. **未启用 flash_attn**：`attn_implementation` 未显式设，默认 SDPA，浪费 3-10× 加速。

**改进方案**（配合用户 2026-08-14 提出的协议调整）：
- eval-aime 加 `attn_implementation="flash_attention_2"`（服务器已装 flash_attn 2.8.3）；
- 采样数 n=8/16 + max_new_tokens 下调（如 8192）；
- **逐题即时落盘 + resume**（本次已实现，见 §5.3 底部与代码）——中断可续跑，不再全丢。

### 9.3 与原始论文的主要差异（对齐"按原始论文修改后"要求）

| 维度 | 原始论文（Direct-OPD） | 本项目 |
|---|---|---|
| 训练数据 | Skywork-OR1 math + DAPO 模板 | 当前默认/GSM8K；**Skywork 待落地**（§8.4 待办，数据接口已就绪） |
| 评估协议 | ave@32 / T=0.7 / top_p=0.95 / 31744 | 已对齐（eval-aime 支持），长生成待跑 |
| 教师对 | 论文用弱教师（RL 前后） | 同（Stage 0 产出 post-RL 弱教师 + pre-RL 副本） |
| Δ_T 缓存 | 论文逐 token 稠密 | dense（demo）/ top-K 稀疏（真实词表，↓1000×） |
| 学生更新 | 同步（论文基线） | **异步**（AsyncBatchedScheduler 四线程流水线） |
| 信用分配 | 分布级 PG | 同 + 稀疏支撑 `delta_for_student_topk` |
| KL 锚 | 对初始学生分布 | 同 + `low_var_kl_support` 稀疏版（有界近似） |
| L2 流式适应 | 论文无（离线固定 D） | **已实现**：`adaptive_cache` 子系统 + 双池 feeder 交替相位；`l2.enabled=false` 默认退回 L0/L1 |
| P-OPD 纯 on-policy | 论文无 | **新增**：`p_oped` / `l2.pure_refresh` 跳过预计算教师得分、纯 on-policy 交替相位 |

### 9.4 复现步骤（P-OPD 纯 on-policy，2026-08-31）

**训练**（服务器 2×RTX PRO 6000 Blackwell；vLLM 0.16 用 off 逃生舱）：

```bash
cd /root/opd/main
# 冒烟（n_steps=20 / m_refresh=8）
python -m fullstack_opd_v2 train --config configs/qwen3_r1_onpolicy.yaml \
  --run-dir /root/autodl-tmp/runs_r1p/smoke --set stage2.n_steps=20 \
  --set stage2.rollout_engine=vllm --set stage2.gradient_checkpointing=true \
  --set stage2.rollout_gpu_mem=0.55 --set stage2.rollout_max_model_len=4096 \
  --set stage2.teacher_offload=true --set l2.m_refresh=8 \
  --set l2.rollout.eos_token_id=151645
# 冒烟通过后全量 300 步（去掉 --set l2.m_refresh=8，用配置默认 32）
```

**评估**（论文协议对齐，逐题落盘 + resume）：

```bash
python -m fullstack_opd_v2 eval-aime \
  --model <HF 模型路径> \
  --datasets AIME24 AIME25 \
  --n-samples 8 --temperature 0.7 --top-p 0.95 \
  --metric ave --prompt-style dapo --scoring sympy --chat-template \
  --batch-size 1 --max-new-tokens 8192 --dtype bf16 \
  --out <输出目录> --device cuda:0
# 中断后重跑同 --out：自动跳过已完成题（resume）
```

> **判据**：AIME24/25 ave@32 ≥ 基线 26.7 + 5（论文同口径）；对照 v2 混合方案
> （E2 0.376 vs Base 0.816 已失败）。
