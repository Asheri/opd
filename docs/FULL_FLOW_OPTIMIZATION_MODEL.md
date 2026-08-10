# 全栈 OPD 全流程（数据流动层）+ 优化目标数学模型

> 对应 `main/fullstack_opd_v2/`（main/ 为真正主项目，自包含，无 async-opd 依赖）。
> 三阶段流水线：**小模型 RL → 离线缓存教师对 Δ_T → 异步 Direct-OPD 训练**，
> 打破「常驻教师 / 同步等待 / 迁移终态」三重限制。
> 衡量目标：**① 训练时间**（异步+预加载教师的时间优化）**② AIME 蒸馏前后得分**（效果保持）。

**文档分工**：本文件讲「**数据怎么流动 + 优化目标怎么度量**」；工程实现细节（线程/队列/
版本号/缓存张量的代码级说明、逐行对齐的数学）见
`main/fullstack_opd_v2/ENGINEERING_IMPLEMENTATION.md`（标 🔍 的地方为通俗例子）。本文的
数学记号与 ENGINEERING §0.6 完全对齐，两文互相引用不重复。

---

# 第一部分：全流程（数据流动层）

## 0. 先认清实体：本代码里到底有哪几种「模型 / 分布」？

（防止把「教师 / 学生 / 旧快照 / 参考锚点」混为一谈。详见 ENGINEERING §0 实体速查。）

| 实体 | 是什么 | 出现在哪 | 训练里干什么 |
|---|---|---|---|
| `teacher_rl / teacher_ref` | 两个**教师**（Stage 0 产出的 post-RL 弱教师 + 其训练前副本） | 仅 Stage 1 | 一次 `response_dists` 算 `Δ_T`，之后**立即释放**，训练期零前向 |
| `student`（new） | 被训练的**当前学生** | 全程（learner） | 训练时前向算 `s_cur`（带梯度，被更新） |
| 快照（old） | student 的某一**旧版本**（`WeightStore` 加载） | RolloutCollector | 旧权重前向算 `s_old`（无梯度，只当重要性比分母），随样本入队 |
| `ref_dists` | **初始 student 的分布**（Stage 2 入口冻结成张量，**不是模型**） | Stage 2 入口算一次 | 只作 **KL 锚点**（信任域），不参与 `s_cur/s_old` |
| `cache.delta` | 教师偏好差 `(N,T,V)` 或 `(N,T,K)`，训练期常量 | Stage 1 构建 | 信用分配输入（= RL 里的奖励信号） |

> 🔑 三个常被误解的点（详见 ENGINEERING §0 注释）：
> 1. **没有「ref 学生模型在 rollout」这回事**：`ref_dists` 是初始分布被**冻成一张表**，
>    只和 `s_cur` 算 KL，从不做前向；Stage 1 里的 `teacher_ref` 是**教师**，不是 student 的参考模型。
> 2. **`s_cur` 与 `s_old` 的输入完全一样**（都是 `cat([prompts, responses])` 喂模型），
>    **唯一区别是权重**（当前 student vs 旧快照）。
> 3. **Δ_T 来自教师，不是任何学生模型**——在 OPD 里等价于 RL 的「环境奖励」。

### 0.1 全局数据流总览

```
┌─ Stage 0 ────────────────┐   ┌─ Stage 1 ─────────────────┐   ┌─ Stage 2 ──────────────────────────┐
│ 小模型 RL (REINFORCE)     │   │ Lightning 离线缓存 (Δ_T)    │   │ 异步 Direct-OPD (四线程流水线)       │
│                          │   │                            │   │                                    │
│ D=(x_i)  ──► π_T^RL (post)│   │ (x,y) ──► π_RL 前向 ─► logp_rl│   │ idxs ─► s_old ─► +Δ_T ─► π_old 加权PG │
│        ──► π_T^ref (pre)  │   │        ──► π_ref 前向 ─► logp_ref│   │        ──► s_cur ─► k3 KL ─► 更新    │
│                          │   │        ──► Δ_T=rl−ref 落盘 │   │ 权重：student ─► weight_store ─► worker │
└──────────────────────────┘   └───────────┬────────────────┘   └────────────────────────────────────┘
                                           ▼
                              cache.delta (N,T,V) 或 (N,T,K)   ← 训练期零教师前向，零拷贝索引
```

**三重限制突破（数据流层面的体现）**：
- **常驻教师** → 教师前向只在 Stage 1 离线做一次，Δ_T 落缓存；Stage 2 训练循环内**无任何教师 forward**。
- **同步等待** → Stage 2 四线程生产者-消费者流水线，批次在队列流动，不需每步同步等教师/rollout。
- **迁移终态** → 迁移对象是 RL 策略偏移 Δ_T（log-ratio），不是教师终态策略，作用于 student 自身 on-policy 状态。

---

## 1. Stage 0 · 小模型 RL → 弱教师对

**目标**：用便宜的弱模型 RL 产出教师对 `(π_T^RL, π_T^ref)`——这是 Δ_T 的来源。

**数据流（`pipeline.stage0_small_rl`，详见 ENGINEERING §0.5 时序）**：
```
输入:   prompts (N,P) 设备常驻 · reward_fn（DataLoader 提供，toy 为查找表 lut(V,)）
第 k 步:
  1. idxs = randint(N, B)                 # 抽批次索引 (B,)
  2. p_b = prompts[idxs]                  # (B,P)
  3. r = generate_batch(weak, p_b)        # 自回归解码 → responses (B,T)，无梯度
  4. weak.train()
  5. logp = token_logprobs(weak, p_b, r)  # 带梯度的逐 token logπ (B,T)
  6. reward = reward_fn(r)                # (B,T) 规则奖励（向量化查找）
  7. loss = −mean(logp · (reward − mean(reward)))   # REINFORCE + 均值基线
  8. backward → clip_grad → optimizer.step
输出:  weak.eval() → π_T^RL (post-RL) 和 π_T^ref = 训练前副本 (pre-RL)
```

**关键张量形状**：`(N,P)` prompt、`(N,T)` response、`(B,T,V)` logp、`(B,T)` reward。

> 注：Stage 0 的 REINFORCE 用「均值基线」减方差，因为它是**学教师**（奖励=规则奖励、需探索）；
> Stage 2 的 Direct-OPD **不需要价值网络/基线**——Δ_T 本身就是教师 RL 前后的相对差（逐 token
> 相对优势），奖励来源直接是同一份教师。两者奖励来源一致（详见第二部分 §1.6 与 ENGINEERING §0.6）。

---

## 2. Stage 1 · Lightning 离线缓存（预加载教师）

**目标**：对固定 rollout 集合 `D = {(x_i, y_i)}` 离线预计算教师对 log-ratio，**此后训练不再启动教师**。

**数据流（`pipeline.stage1_build_cache` → `cache.py`，详见 ENGINEERING §2）**：
```
输入:  prompts (N,P) · responses (N,T) · teacher_rl=π_T^RL · teacher_ref=π_T^ref
      （L1 warmup：用初始 student / 教师分布对每 prompt 额外采样 M 条响应，拼「胖 D」，
        默认开启 warmup_M=4/warmup_source=student_init → N×(1+M)，mix 则 N×(1+2M)）
for 每批 (P_b, R_b):
  logp_rl  = π_T^RL 的前向 log-softmax (b,T,V)
  logp_ref = π_T^ref 的前向 log-softmax (b,T,V)
  Δ_T(y|x) = logp_rl − logp_ref          # 逐位置、逐 token、逐词表
cache.delta = Δ_T 张量                     # dense (N,T,V) 或 topk (N,T,K)（L4 稀疏）
cache.save(path)                          # 落盘
```

**⚠️ Δ_T 覆盖全词表（这是「按 π_old 对全词表取期望」的前提，详见 ENGINEERING §0.6 ⚠️ 框）**：
教师预加载的是**稠密** Δ_T——`cache.build` 里 `self.delta = rl_full - ref_full` 是完整 `(N,T,V)`
张量，对**每个**候选 token v 都有值。因此训练时内层 `E_{v∼π_old}[·]` 不必「从 π_old 采样动作去查
Δ_T」——Δ_T 已为整词表备好，学生分布怎么重分布都不缺值。**稀疏模式（topk）**则破坏此保证：
只存 teacher top-K（⊂V），真实信号仅在 `student-top-K ∩ teacher-top-K` 交集，交集外填 0（中性），
属有界近似（第二部分 §1.5 详述）。

**稀疏缓存（L4，真实词表 V=128k 必需）**：dense `(N,T,V)` 存不下 → 每位置只存
teacher 的 top-K `(token_id, logp)`，训练期用 `searchsorted` 二分匹配（O(K) 替代 O(K²)）。
- `cache.ids_sorted` / `cache.delta_k_sorted` 在 build 期预排序；
- 一致性校验：`teacher_rl` 与 `teacher_ref` 必须同架构/词表/d_model/max_len，否则抛 `TeacherConsistencyError`（详见 ENGINEERING §2.4）。

**L1 warmup（离线 rollout 暖缓存，默认开启）**——曝光偏差缓解的第一级（完整谱见第二部分 §3.2）：
- 动机：曝光偏差的根因是「训练上下文只有每 prompt 一条固定 r_i」。L1 在 Stage 1（教师/学生冻结时）
  用**初始 student**（`student_init`）或温度扰动教师（`teacher_perturbed`）或两者（`mix`）
  `generate_batch` 各 ×M 条响应，拼成胖 D：`fat_prompts=(N·(1+K),P)`、`fat_responses=(N·(1+K),T)`，
  K=M 或 2M。调度器内核零改动（仍只读 `cache` + 索引）。
- **同源不变式（P1-4 修复）**：warmup 采样必须与 KL 锚点同分布——两者都是**初始 student 分布**。
  resume 时 `warmup_student` 用**独立新建的初始 student**（不被断点 `load_state_dict` 覆盖），
  保证「warmup 上下文分布 = KL 锚点分布」恒成立（详见 §3.3 与 `pipeline.py:_run_body`）。
- 计数：默认 N=16、M=4 → `delta=(80,T,V)`（×5）；`mix` → `(144,T,V)`（×9）。

#### 胖 D 的具体组成（`stage1_build_cache` 的拼接顺序，逐行对齐代码）

`fat_prompts` 与 `fat_responses` 都是**块结构拼接**（`torch.cat` 沿 dim 0）：

```
fat_prompts   = cat([ prompts, prompts, …, prompts ])          # 同一份 prompts 重复 (1+K) 次
fat_responses = cat([ responses, rp₁ … rp_M(, rt₁ … rt_M) ])   # 块 0 = 原固定 D
```

- 每个块 = 一个 `(N,T)` 响应集；**prompt 行在所有块里逐行相同**（第 i 行永远是同一个 x_i），
  各块的差异全在响应 y 上——即「**同一 prompt x_i 的 1+M 条不同上下文 y**」。
- `rp_m = generate_batch(warmup_student, prompts, max_new=T, temperature=warmup_temperature)`
  ——初始 student 分布采样（`warmup_temperature=1.0` 即原分布）；
- `rt_m = generate_batch(teacher_rl, prompts, max_new=T, temperature=…)`
  ——**post-RL 教师** `teacher_rl` 加温度扰动采样（注意不是 `teacher_ref`）；
- 顺序：`student_init`/`teacher_perturbed` → 只有对应一组 M 块；`mix` → **先 student×M 块、再 teacher×M 块**；
- `generate_batch` 始终是采样（`softmax(logits/temperature)` + `multinomial`），非贪心。

**示例（student_init，N=2，M=2，P=6，T=8，V=64）**：

```
原 D：    prompts=[x₁,x₂] (2,6)   responses=[y₁,y₂] (2,8)
2 轮采样：rp₁=[y₁'₁,y₂'₁] (2,8)   rp₂=[y₁'₂,y₂'₂] (2,8)     # 每轮对全部 2 个 prompt 各采一条

fat_prompts   = [x₁,x₂,  x₁,x₂,  x₁,x₂]        (6,6)
fat_responses = [y₁,y₂,  y₁'₁,y₂'₁,  y₁'₂,y₂'₂]  (6,8)
```

| 行区间 | 块 | prompt | response | 来源 |
|---|---|---|---|---|
| 0–1 | 块 0 | x₁, x₂ | y₁, y₂ | 原固定 D |
| 2–3 | 块 1 | x₁, x₂ | y₁'₁, y₂'₁ | 初始学生第 1 轮采样 |
| 4–5 | 块 2 | x₁, x₂ | y₁'₂, y₂'₂ | 初始学生第 2 轮采样 |

→ `cache.delta=(6,8,64)`：第 i 行 = 该行响应在对应 prompt 下的 Δ_T；每个 prompt 从 1 条
上下文扩成 1+M=3 条。`mix` 则在块 2 后追加 `rt₁, rt₂` 两块（教师采样），`delta=(10,8,64)`。
`fat_prompts` 的 prompt 逐块重复是**有意的冗余**（保持「行号 = (块, prompt) 双索引」最简单、
`cache` 按行索引即可），代价是 prompt 张量 ×(1+K) 常驻；`delta`/`ref_dists` 同理随块结构
放大（真实词表须配 topk 缓存）。

**Δ_T 的语义**：`Δ_T(y|x) > 0` ⇔ RL 使教师更可能产生 y（RL 学到的改进方向）；
`< 0` ⇔ RL 抑制了 y。相减丢弃了教师 RL 前已有的偏好，只保留 RL 诱导的偏移。

---

## 3. Stage 2 · 异步 Direct-OPD 训练（数据流动核心）

**四线程流水线（`AsyncBatchedScheduler`，详见 ENGINEERING §1）**，队列里流动的是**批次**而非样本：

```
PromptFeeder ──(B,) 索引──► RolloutCollector ──(idxs, s_old, ver)──► TeacherScorer ──贴 Δ_T──► TrainDispatcher
```

### 3.1 每个阶段的张量/权重流

**(a) PromptFeeder**：无限产生随机批次索引 `idxs (B,)` → `_pq` 队列（容量 `queue_size`）。

**(b) RolloutCollector**（⚠️ 名字误导：不自回归采样，对固定 rollout 做 teacher-forcing）：
```
1. 权重同步：weight_store.acquire_if_newer(self._loaded_ver)
   ── 仅当 student 权重版本推进时才 load_state_dict（v1 每样本加载一次，v2 只在版本变化时）
2. s_old = response_dists(worker, prompts[idxs], responses[idxs])   # (B,T,V) 无梯度
   ── 用【可能陈旧的】student 快照算 π_old(y|x)（Lightning 设定）
3. 入队 (idxs, s_old, loaded_ver) → _rq
```

**(c) TeacherScorer**：
```
1. 弹 (idxs, s_old, ver)
2. dense: delta = cache.get_delta(idxs)          # 零拷贝 (B,T,V)（★无 teacher 前向）
   topk : delta 透传 None（learner 现场按 student 支撑展开）
3. staleness_q.put((idxs, s_old, delta), version=ver)
   ── 入队侧截断：age = v_cur − v_sample > threshold → 拒收（n_rejected）
```

**(d) TrainDispatcher**（learner 核心，`_train_step`）：
```
1. 消费侧二次截断：age > threshold → 丢弃（n_dropped_consume）
2. idxs_dev = idxs.to(device)
3. s_cur = student.response_dists(prompts[idxs], responses[idxs])  # (B,T,V) 带梯度
   ── learner 时刻用【当前】student 重算分布（recompute 代理）
4. p_old = s_old.exp()                          # π_old 缓存（省一次 exp）
5. 损失（详见第二部分数学）：
     L_pg = −Σ_v π_old(v)·min(ratio·Δ_T, clip(ratio)·Δ_T)      # π_old 加权 PG + PPO clip
     L_kl = KL(π_cur‖π_ref) via k3 估计量                       # 低方差 KL 正则
     L = L_pg + λ_kl·L_kl
6. backward → clip_grad_norm → optimizer.step → weight_store.publish → 版本 +1
7. on_step 回调（异步后台线程）：metrics 落 CSV + 按 checkpoint_every 存 student 断点
```

> **s_cur / s_old 与「两个 ref」的厘清（详见 ENGINEERING §0）**：
> - `s_cur` 和 `s_old` 喂给模型的**输入完全相同**（`cat([prompts[idxs], responses[idxs]])`），
>   唯一差别是**权重**（当前 student vs 旧快照）。「prompts 是上下文前缀、responses 是被打分的
>   目标序列」——不是「ref 的 rollout 当 y」。
> - 两个必须区分的 "ref"：`π_ref`（**教师** base）只出现在 Δ_T 内部，是「奖励的来源之一」；
>   `π_θ₀`（= `ref_dists`，**初始 student 分布**）是 KL 正则的锚，是「学生自己的起点」。
>   两者**不是同一个实体**。

### 3.2 权重 / 版本 / 陈旧度流动

```
student.state_dict ──publish──► weight_store._snapshot（版本 +1）
                                    │ acquire_if_newer
                                    ▼
                        RolloutCollector 加载快照 → 算 s_old（带 ver 标签）
                                    │
staleness_q.current_version（每次 publish 推进）┌─ 入队侧：age=v_cur−v_sample>threshold 拒收
      ▼                                        └─ 消费侧：再查一次，过旧丢弃（双截断）
_train_step 用 age 校验 → 训练 → publish → 版本再推进
```

**陈旧度 age 的语义**：`age = 当前版本 − 样本版本`，`age > 0` 证明异步确实在消费
陈旧样本（论文 AsyncOPD 的核心信号）。`M5` 修复后 waste 拆解：
`rollouts = trained + 陈旧(put+consume) + 队满 + 停机尾`（即不是所有产出的 rollout 都被训练）。

> 更完整的异步机制（快照/版本号为什么必须存在、双截断两道各拦什么、背压与不死锁、
> 有界在途队列 vs 持久 replay buffer 的边界）见 ENGINEERING §1。

### 3.3 run 目录 / 产物数据流（工程化）

```
run_dir/<timestamp>/
├── config.yaml      ← load_config 输出快照（含下渗后部署键）
├── metrics.csv      ← 每步 loss/pg/kl/adv/reward/age/version（resume 续写保留历史）
├── timings.json     ← stage0_rl/stage1_cache/stage2_train/total 逐段计时
├── train.log        ← 结构化日志
└── checkpoints/     ← step_N.pt（state_dict + version + cfg + ref 锚点；final 无条件保存）
```

**断点续跑（resume）数据流**：`resume` 加载断点的 `state_dict` + `version` + **`ref` 锚点**
（初始 student 在 fat D 上的 KL 锚点）→ 恢复版本号 → Stage 2 从该版本继续；
KL 锚点从断点恢复（不重算），保持「KL 锚点 = 初始 student 分布」不变式。
**L1 warmup 与 resume 的交互（P1-4）**：resume 不会把断点权重污染到 warmup 采样——`warmup_student`
是**独立新建的初始 student**，故「warmup 分布 = KL 锚点分布」在 resume 下仍然成立。

### 3.4 健康信号（效果与异步的实证，连接第二部分度量）

| 信号 | 含义 | 健康判据 |
|---|---|---|
| `reward`（`expected_reward` = `E_{π_cur}[Δ_T]`） | 当前策略下的期望教师偏移 | 随训练**单调上升**（demo 修复后 −0.18 → +0.72） |
| `age` | 样本陈旧度 | `age > 0` 证明异步在消费陈旧样本（论文核心信号） |
| `waste` 拆解 | trained / 陈旧 / 队满 / 停机尾 | 陈旧+队满占比不过高（双截断生效的体现） |
| `timings.json` | 逐段耗时 | Stage 2 ≈ 流水线最慢阶段（第二部分 §2 的实证） |

---

# 第二部分：优化目标的数学模型

> 记号与 ENGINEERING §0.6 逐行对齐（那里有代码对照列与完整的「对 E_{v∼π_old} 的误读」澄清）。

## 目标 0 · 记号

| 符号 | 含义 |
|---|---|
| $s_t=(x, y_{<t})$ | 第 $t$ 个位置的上下文（前缀），`response_dists` 内部 `cat([prompts, responses[:, :t]])` |
| $x \in \mathcal{X}$ | prompt |
| $y \in \mathcal{Y}$ | 响应 token 序列 |
| $\pi_\theta(\cdot\mid s_t)$ | student（被训练方）**当前**策略 = `s_cur`（带梯度） |
| $\pi_{old}(\cdot\mid s_t)$ | rollout 时刻的 student 快照（行为策略）= `s_old`（无梯度） |
| $\pi_{\theta_0}(\cdot\mid s_t)$ | **初始 student** 分布（Stage 2 入口冻结）= `ref_dists`（KL 锚点） |
| $\pi_T^{RL}, \pi_T^{ref}$ | 教师 post-RL / pre-RL（弱模型）；两者差出 Δ_T |
| $V$ | 词表大小；$T$ 响应长度；$B$ 批大小 |
| $\Delta_T(t,v)$ | 稠密奖励 $=\log\pi_T^{RL}(v\mid s_t)-\log\pi_T^{ref}(v\mid s_t)$，**逐词表** |
| $\rho_\theta(v)$ | 重要性比 $\pi_\theta(v)/\pi_{old}(v)$ = `ratio=(s_cur−s_old).exp()` |
| $\varepsilon$ / $\beta$ | 裁剪 / KL 系数（`clip_eps=0.2` / `kl_reg_coef=0.05`）|
| $\mathcal{D}$ | 离线固定 rollout 集（默认含 warmup 胖 D，L1）|

---

## 目标 1 · 效果目标：Direct-OPD 损失（保持与论文一致的训练效果）

### 1.1 迁移对象：Δ_T（不是教师终态）

$$\Delta_T(y\mid x) = \log \pi_T^{RL}(y\mid x) - \log \pi_T^{ref}(y\mid x).$$

它只保留"RL 让教师改变了什么"，丢弃教师 RL 前已有的偏好——当 student 已强于教师时，
模仿教师终态会覆盖掉更强的行为（论文图 1(a)：R1-Distill-7B 56.7 → OPD 掉到 ~50），
而迁移 $\Delta_T$ 则保留提升方向。

### 1.2 PG 损失（π_old 加权 + PPO clip）——`losses.pg_loss`

$$\min_\theta\;-\frac{1}{BT}\sum_{i,t}\mathbb{E}_{v\sim\pi_{old}}\Big[\min\!\big(\rho_\theta(v)\,\Delta_T(t,v),\ \operatorname{clip}(\rho_\theta(v),1-\varepsilon,1+\varepsilon)\,\Delta_T(t,v)\big)\Big].$$

- **为什么必须 π_old 加权**：$\sum_v \pi_{old}(v)\,\rho_\theta(v)\,\Delta_T(v)
  = \sum_v \pi_\theta(v)\,\Delta_T(v) = \mathbb{E}_{\pi_\theta}[\Delta_T]$——
  即 $\rho=1$ 时精确等于 **Direct-OPD 目标 $-\mathbb{E}_{\pi_\theta}[\Delta_T]$**（on-policy 期望奖励）。
  等权 `mean` 不是这个目标；token 级标量 advantage 形式一阶梯度恒为 0（实测验证）。
- **min(clip) 的悲观下界**：PPO 风格，限制每步更新幅度，防 ratio 爆炸（A>0 裁上溢、A<0 裁下溢）。
- **失配屏蔽**（M1 修复）：$\pi_{old}\approx 0$ 处（支撑外）贡献强制为 0，
  避免 $r=\exp(s_{cur}-s_{old})$ 放大到天文数字造成伪梯度/NaN。

> **⚠️ 关于 $\mathbb{E}_{v\sim\pi_{old}}$ 的误读（两层期望，零采样）**——详见 ENGINEERING §0.6 ⚠️ 框：
>
> 写成 $\mathbb{E}_{v\sim\pi_{old}}[\cdots]$ 容易让人以为"动作 v 是从 π_old 采样出来的"。真实工程里要拆成两层：
> $$
> \underbrace{\mathbb{E}_{(s_t,t)\sim D}}_{\text{外层：上下文来自固定数据集 }D}\quad
> \underbrace{\mathbb{E}_{v\sim\pi_{old}(\cdot\mid s_t)}[\cdots]}_{\text{内层：动作空间按旧学生分布求闭式期望}}
> $$
> - **外层（上下文）来自固定 D**，不是 π_old 的 rollout 采样：训练 teacher-force 数据集里的
>   $(prompt, response)$ 前缀，`response_dists` 只算"给定前缀下学生自己的下一词分布"，从不自己采样 token。
> - **内层（动作 v）是对全词表的解析期望，不是采样**：`pg_loss` 里
>   `-(s_old.exp() * pointwise).sum(-1)` 把 pointwise 在**整张词表 V** 上、用 π_old(v) 当权重求和，
>   等价于 $\sum_v \pi_{old}(v)[\cdots]$。这是分类动作空间上的**闭式期望**，无 MC 方差。
> - **π_old 只是内层期望的权重分布**（旧快照对全词表的概率），不是"生成动作/上下文的行为策略"。
>   D 提供上下文、教师 Δ_T 提供全词表逐 token 奖励、旧/新学生分布提供权重与重要性比——三者在
>   词表上加权求和，**全程零采样**。
> - **IS 无偏性的边界**：内层恒等式 $\mathbb{E}_{\pi_{old}}[\rho_\theta\,\Delta_T]=\mathbb{E}_{\pi_\theta}[\Delta_T]$
>   对内层（动作）期望**严格成立**；唯一"离线"的是**外层（上下文）期望**——RL/PPO 外层是
>   π_old rollout 诱导的状态分布，OPD 外层是固定数据集 D。这正是 Direct-OPD 与在线 RL 的本质区别。

### 1.3 KL 正则（k3 估计量，π_θ 下期望）——`losses.low_var_kl`

$$L_{kl} = \sum_{v} \pi_\theta(v)\; k_3\!\Big(\log\frac{\pi_{ref}(v)}{\pi_\theta(v)}\Big),\qquad
k_3(u) = e^{u} - u - 1.$$

这是 **KL$(\pi_\theta \| \pi_{ref})$ 的低方差逐点估计**（分布形式下恒等），锚定 student
到初始分布，防策略漂移。稀疏锚点版本 `low_var_kl_support` 只在 top-K 支撑求和（有界近似，
系统性略低估真 KL，方向安全——详见 §1.5）。

**总训练目标**：

$$\mathcal{L}(\theta) = \frac{1}{BT}\sum_{i,t}\Big[
 -\mathbb{E}_{v\sim\pi_{old}}\min\big(\rho(v)\Delta_T,\ \operatorname{clip}(\rho)\Delta_T\big)
 + \beta\, \mathrm{KL}(\pi_\theta\|\pi_{\theta_0})\Big].$$

### 1.4 数据新鲜的极限：on-policy 三 KL 分解（理解目标的钥匙）

当样本新鲜（$\pi_{old}=\pi_\theta,\ \rho=1$）时 $\min(1\cdot\Delta_T,1\cdot\Delta_T)=\Delta_T$，
单位置目标退化为（推导见 ENGINEERING §0.6）：

$$J_{i,t}(\theta)=\mathbb{E}_{v\sim\pi_\theta}[\Delta_T(t,v)]
=\underbrace{\mathrm{KL}(\pi_\theta\|\pi_{ref})}_{\text{别离 base 教师太远}}
\;-\;\underbrace{\mathrm{KL}(\pi_\theta\|\pi_{rl})}_{\text{去贴近 RL 教师}}
\;-\;\underbrace{\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\theta_0})}_{\text{信任域锚住起点}}$$

直觉：让学生**比贴近 base 教师更贴近 RL 教师**——即"获取教师经过 RL 训练后的那部分改进"，
这就是 **Direct-OPD 名字的由来**。整个目标在 on-policy 下是三个 KL 的权衡。

### 1.5 dense vs 稀疏：精度边界（诚实声明近似）

- **dense（demo 默认）**：Δ_T 覆盖全词表，§1.2 的"全词表闭式期望"**严格成立**，是真正归一化的期望。
- **topk 稀疏（真实词表）**：`delta_for_student_topk` 只在 **student 当前 top-K** 支撑上非零
  （与 teacher top-K 交集内取教师 Δ，交集外=0）。内层实际只在该支撑上按 π_old 加权求和：
  - **未做支撑重归一化**——等价于 $\sum_{v\in\text{top-K}}\pi_{old}(v)[\cdots]$，top-K 之外的
    尾部质量被直接丢弃，不是条件期望。这是为省显存（Δ_T 只存 top-K 才不爆内存）的**有意近似**：
    若 top-K 已捕获 ~99% 概率质量偏差约 1%，且 PG 与 KL 两项同尺度缩放、相对权衡不变；
  - **支撑交集**：真实信号仅存在于 `student-top-K ∩ teacher-top-K`，交集外 Δ_T=0（中性）。
  - 两个 `low_var_kl_support` / `pg_loss` 的稀疏路径同源，均为**有界近似、方向安全**，非恒等替换。

### 1.6 效果度量：AIME 蒸馏前后得分

**效果度量**：AIME24/25 蒸馏前后得分。论文参考（AIME24 ave@32）：ref 28.5 → 教师 51.3；
学生 pre 56.7(7B)/48.3(1.7B) → post +6.4/+10.0。**工程目标**：在 student 上复现这种
「弱到强提升」——即 AIME$_{post} >$ AIME$_{pre}$，且接近论文 Direct-OPD 的增益。

**评估方法论（`opd eval-aime`，R1/P2 修复后）**：
- `n_samples>1`（pass@1 采样）**必须** `temperature>0`——贪心 + 多序列逐字重复会让 pass@1 失真
  （构造期 ConfigError 前置校验，`eval_aime.py`）；
- `--run-dir` 桥接会读 run 目录 `config.yaml` 的 `eval.*`（max_new_tokens/n_samples/temperature）
  作为默认，CLI 显式值优先；
- 基准 harness：`benchmarks/aime24_25/`（teacher 基线 / 学生蒸馏前 / 蒸馏后对比，`aggregate.py` 汇总）。

---

## 目标 2 · 时间目标：异步流水线 + 预加载教师的时间收益

### 2.1 同步基线 vs 异步+预加载

**同步 OPD（传统）每步耗时**（教师必须在训练循环内，每步都要 teacher 前向）：

$$T_{step}^{sync} = t_{teacher} + t_{rollout} + t_{train},$$

其中 $t_{teacher}$ 是教师前向（计算 $\pi_T$ 供 $\Delta_T$），$t_{rollout}$ 是 student 采样/打分，
$t_{train}$ 是 learner 前向+损失+反传+更新。

**本工程（异步 + 预加载）**：
- **预加载教师**：$\Delta_T$ 在 Stage 1 离线算好落缓存，训练循环内 $t_{teacher}=0$。
  教师离线成本 $C_{offline}$ 一次性，摊到 $N_{steps}$ 步 → 每步摊销 $C_{offline}/N_{steps}$。
- **四线程异步流水线**：生产-消费重叠，稳态吞吐受**最慢阶段**限制：

$$T_{step}^{async} \approx \max\big(t_{rollout},\; t_{train}\big)
+ \underbrace{C_{offline}/N_{steps}}_{\text{摊销}} +
\underbrace{\tau_{fill}/N_{steps}}_{\text{管道填充}} ,$$

其中 $\tau_{fill}$ 是流水线启动填充延迟（第一批要经过四个阶段才能训练）。

> **L1 warmup 的成本归属**：warmup 采样（M×N 条 `generate_batch`）计入 $C_{offline}$——它是
> Stage 1 的**一次性**离线成本（demo 下是 cache.build 的 ~17×，但绝对量 <1s），摊到整个训练后
> 每步可忽略；代价是 `delta`/`ref_dists` 张量 ×(1+M) 常驻显存（真实词表须配 topk 缓存）。
> 这是「消曝光偏差」与「时间/显存」的显式交换。

### 2.2 吞吐 / 加速比 / 陈旧度权衡

**吞吐模型**：流水线是四个生产者-消费者（$\lambda = 1/\max_i \tau_i$ 稳态），
$N_{batches}$ 批的总时间：

$$T_{total} \approx N_{batches}\cdot \max(\tau_{rollout},\tau_{train}) + \tau_{fill}
+ C_{offline},$$

其中（稳态下）$N_{batches} = N_{steps}$（每 batch 训练一步，陈旧样本被截断则略多——见 waste 拆解）。

**加速比（相对同步基线）**：

$$\text{Speedup} = \frac{T^{sync}}{T^{async}} \approx
\frac{t_{teacher} + t_{rollout} + t_{train}}{\max(t_{rollout},t_{train})}
= 1 + \frac{t_{teacher}}{\max(\cdot)} + \frac{\min(\cdot)}{\max(\cdot)}.$$

- 当 $t_{teacher}$ 不可忽略时，**预加载教师直接砍掉一项**（这是 Stage 1 的核心收益）。
- 当 $t_{rollout}\approx t_{train}$ 时异步重叠约省一半（$\min/\max\approx 1$ → 接近 2×）。

**陈旧度-质量-时间权衡**：异步以「消费陈旧样本」为代价换吞吐。每样本的年龄
$a = v_{cur} - v_{sample}$。更宽松的 threshold $\theta$ 允许更旧样本 → 更高吞吐，
但陈旧梯度降低质量：

$$\underbrace{\mathbb{E}[\text{质量损失}]}_{\text{随 }\theta\text{ 增}} =
\sum_{a>\theta} p(a)\cdot \delta(a),\qquad
\underbrace{\text{吞吐}}_{\text{随 }\theta\text{ 增}} =
\frac{N_{steps}}{\sum_a \min(1,\; \mathbb{1}[a\le\theta]) \cdot \tau_{step}} ,$$

其中 $\delta(a)$ 是年龄 $a$ 样本的梯度偏差（相对当前策略）。**双截断**在入队/消费两侧
把 $a$ 钉在 $[0,\theta+1]$，保证 $\delta(a)$ 有界。健康信号 `staleness age>0` 正是
「异步确实在消费陈旧样本」的实证。

> **为什么双截断是质量-时间权衡的硬约束**：陈旧样本的 IS 校正（ρ）由 **PPO clip**（夹到
> $[1-\varepsilon,1+\varepsilon]$）+ **staleness 双截断**（丢 age>θ）共同限幅——这让
> $\delta(a)$ 有上界、ρ 不爆炸（详见 ENGINEERING §0.6「IS 方差澄清」：OPD 没有 MC IS，
> 动作维是闭式求和、上下文维直接在 D 上优化，不存在"D 覆盖不足→IS 方差爆炸"）。

### 2.3 工程化时间度量

`timings.json` 记录逐段：$T_{stage0}, T_{stage1}, T_{stage2}, T_{total}$。
**时间优化目标**即：在固定 $N_{steps}$ 与模型下最小化 $T_{total}$，且
$T_{stage2}$ 遵循 2.2 的流水线模型（$N_{steps}\cdot\max(\tau_{rollout},\tau_{train})$）；
$T_{stage1}$ 含 L1 warmup 的一次性采样成本（§2.1 注）。

---

## 目标 3 · 综合优化（效果 × 时间的联合目标）

### 3.1 带约束优化

把两个目标写成带约束的优化（或以 Pareto 前沿理解）：

$$\min_{\theta,\;\text{架构}} \;\; T_{total}(\theta)
\quad \text{s.t.} \;\;
\Delta\text{AIME} := \text{AIME}_{post}(\theta) - \text{AIME}_{pre} \;\ge\; \epsilon,$$

其中 $\epsilon$ 是期望的弱到强增益下限（论文参考：+6.4 / +10.0 / +5.1）。$T_{total}$ 由
预加载教师（$C_{offline}$ 摊销）与异步流水线（$\max(\tau_{rollout},\tau_{train})$）主导。

### 3.2 曝光偏差谱 L0–L3：联合优化旋钮

效果-时间权衡里最根本的旋钮是**离线程度**——曝光偏差随"离线程度"加重（不是 bug，是离线蒸馏的
固有代价，详见 ENGINEERING §0.6「曝光偏差」框）。谱系：

| 级别 | 上下文来源 | 教师打分时机 | 训练期教师开销 | 曝光偏差 | 状态 |
|---|---|---|---|---|---|
| **L0** | 固定 D 每 prompt 1 条 | 仅 Stage 1 离线一次 | 零 | 最大（永久固定） | 可关（warmup_M=0+none） |
| **L1** | Stage 1 用学生/教师分布采样多条拼**胖 D** | 仅 Stage 1 离线一次 | 零（上下文变丰富） | 大幅降低 | **已实现，默认开启**（warmup_M=4/student_init） |
| **L2** | 训练期每 K 步用**当前学生** rollout 一批 | 每 K 步一次（教师常驻） | 摊销后小 | 有界（新鲜度 ≤K 步漂移） | 未实现（refresh 线程 + 动态缓存） |
| **L3** | 每批都来自当前学生 rollout | 每批实时 | 最大（回到 live teacher） | 近零 | = 原版 Lightning-OPD |

**与优化目标的关系**：L1→L2 是在**不引入训练期教师开销**（或摊销到可忽略）的前提下压曝光偏差
的最优路径——它直接把约束条件 $\Delta\text{AIME}\ge\epsilon$ 里的"效果上限"抬高，同时只增加
一次性/摊销成本。**结论路径**：L1（已实现、默认开）→ L2（动态刷新收尾，`cache.append` +
`_rollout_refresh` 两个增量，见 ENGINEERING §0.6 L2 设计）。

### 3.3 可调杠杆

**可调杠杆**（在效果与时间之间权衡）：
| 杠杆 | 时间影响 | 效果影响 |
|---|---|---|
| 缓存稀疏度 K（topk） | 缓存小、索引快 | 支撑截断 → 有界低估 Δ_T/KL（§1.5） |
| staleness 阈值 θ | ↑θ 吞吐↑ | ↑θ 梯度偏差↑（§2.2 有界） |
| 批次/队列大小 | ↑ 吞吐 | 稳定性（有界队列+背压） |
| warmup_M / source（L1 档位） | Stage1 一次性 M×N 采样 | 缓解曝光偏差 → 效果↑（§3.2） |
| L2 refresh 间隔 K | 教师推理摊销 K 分之一 | 上下文新鲜度 → 曝光偏差有界 |
| λ_kl | — | 正则强弱 → 漂移控制 |

**一句话**：全栈在**固定效果目标**（$\Delta\text{AIME} \ge \epsilon$）下，用
「离线预加载教师（砍掉 $t_{teacher}$）+ 异步流水线（$T\propto\max(\tau_{rollout},\tau_{train})$）」
把训练时间压到流水线最慢阶段，而非三者之和；曝光偏差这条剩余代价用 **L0–L3 谱**按
"一次性/摊销成本"换"效果上限"——这正是 `timings.json`（时间）与
`benchmarks/aime24_25/`（效果）要共同实证的优化收益。

---

## 附 · 数学解释关键点速查（与 ENGINEERING_IMPLEMENTATION.md 交叉索引）

1. **为什么迁移 Δ_T 而非教师终态**：Δ_T = 减法去掉教师 RL 前偏好，只留 RL 改进方向；
   当 student 已强于教师，模仿终态覆盖强行为，迁移偏移则保留提升。（ENGINEERING §0.6）
2. **为什么 π_old 加权**：Σ_v π_old·r·Δ = Σ_v π_θ·Δ = E_{π_θ}[Δ_T]（on-policy 期望）；
   **π_old 只是内层期望的权重，不是采样行为策略**（两层期望拆解见 §1.2 ⚠️ 框 / ENGINEERING §0.6）。
3. **为什么 k3**：k3(u)=e^u−u−1 是 KL 的逐点被积函数，π_θ 下期望即真 KL，且低方差（§1.3）。
4. **为什么异步快**：生产者-消费者重叠 → 总时间 ≈ 最慢阶段 × 批数，而非各阶段之和（§2.2 / ENGINEERING §1）。
5. **为什么预加载教师有效**：教师前向从每步移除，离线成本一次摊销（C_offline/N_steps）（§2.1 / ENGINEERING §2）。
6. **为什么要有 staleness 截断**：age 越界样本的梯度偏差 δ(a) 无界 → 双截断 + PPO clip 钉住
   （§2.2 / ENGINEERING §1.4）。
7. **为什么 on-policy 下目标是三 KL**：J = KL(θ‖π_ref) − KL(θ‖π_rl) − β·KL(θ‖π_θ0)，
   即"比贴近 base 教师更贴近 RL 教师"——Direct-OPD 名字的由来（§1.4 / ENGINEERING §0.6）。
8. **稀疏是近似不是恒等**：topk 不重归一、交集支撑、方向安全（§1.5 / ENGINEERING §0.6 边界框）。
9. **曝光偏差是离线程度谱的代价**：L0–L3 是"一次性/摊销成本 vs 效果上限"的旋钮，L1 已实现默认开
   （§3.2 / ENGINEERING §0.6 L0–L3 表）。
