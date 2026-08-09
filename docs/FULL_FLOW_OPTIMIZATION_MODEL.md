# 全栈 OPD 全流程（数据流动层）+ 优化目标数学模型

> 对应 `main/fullstack_opd_v2/`（main/ 为真正主项目，自包含，无 async-opd 依赖）。
> 三阶段流水线：**小模型 RL → 离线缓存教师对 Δ_T → 异步 Direct-OPD 训练**，
> 打破「常驻教师 / 同步等待 / 迁移终态」三重限制。
> 衡量目标：**① 训练时间**（异步+预加载教师的时间优化）**② AIME 蒸馏前后得分**（效果保持）。

---

# 第一部分：全流程（精细到数据流动层）

## 0. 全局数据流总览

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

**数据流（`pipeline.stage0_small_rl`）**：
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

---

## 2. Stage 1 · Lightning 离线缓存（预加载教师）

**目标**：对固定 rollout 集合 `D = {(x_i, y_i)}` 离线预计算教师对 log-ratio，**此后训练不再启动教师**。

**数据流（`pipeline.stage1_build_cache` → `cache.py`）**：
```
输入:  prompts (N,P) · responses (N,T) · teacher_rl=π_T^RL · teacher_ref=π_T^ref
      （可选 warmup：用初始 student / 教师分布对每 prompt 额外采样 M 条响应，拼「胖 D」：
        N×(1+M) 或 N×(1+2M)，缓解曝光偏差 L1）
for 每批 (P_b, R_b):
  logp_rl  = π_T^RL 的前向 log-softmax (b,T,V)
  logp_ref = π_T^ref 的前向 log-softmax (b,T,V)
  Δ_T(y|x) = logp_rl − logp_ref          # 逐位置、逐 token、逐词表
cache.delta = Δ_T 张量                     # dense (N,T,V) 或 topk (N,T,K)（L4 稀疏）
cache.save(path)                          # 落盘
```

**稀疏缓存（L4，真实词表 V=128k 必需）**：dense `(N,T,V)` 存不下 → 每位置只存
teacher 的 top-K `(token_id, logp)`，训练期用 `searchsorted` 二分匹配（O(K) 替代 O(K²)）。
- `cache.ids_sorted` / `cache.delta_k_sorted` 在 build 期预排序；
- 一致性校验：`teacher_rl` 与 `teacher_ref` 必须同架构/词表/d_model/max_len，否则抛 `TeacherConsistencyError`。

**Δ_T 的语义**：`Δ_T(y|x) > 0` ⇔ RL 使教师更可能产生 y（RL 学到的改进方向）；
`< 0` ⇔ RL 抑制了 y。相减丢弃了教师 RL 前已有的偏好，只保留 RL 诱导的偏移。

---

## 3. Stage 2 · 异步 Direct-OPD 训练（数据流动核心）

**四线程流水线（`AsyncBatchedScheduler`）**，队列里流动的是**批次**而非样本：

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
`rollouts = trained + 陈旧(put+consume) + 队满 + 停机尾`。

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

---

# 第二部分：优化目标的数学模型

## 目标 0 · 记号

| 符号 | 含义 |
|---|---|
| $x \in \mathcal{X}$ | prompt；$y \in \mathcal{Y}$ | 响应 token 序列 |
| $\pi_\theta$ | student（被训练方）策略 |
| $\pi_T^{RL}, \pi_T^{ref}$ | 教师 post-RL / pre-RL（弱模型）|
| $\pi_{old}$ | rollout 时刻的 student 快照（行为策略）|
| $V$ | 词表大小；$T$ 响应长度 |
| $\Delta_T(y \mid x)$ | 教师策略偏移 = $\log\pi_T^{RL}(y\mid x) - \log\pi_T^{ref}(y\mid x)$ |
| $\lambda_{kl}$ | KL 正则系数（`kl_reg_coef`）|
| $\mathcal{D}$ | 离线固定 rollout 集（可能含 warmup 胖 D）|

---

## 目标 1 · 效果目标：Direct-OPD 损失（保持与论文一致的训练效果）

**迁移对象**是教师 RL 诱导的策略偏移，不是教师终态：

$$\Delta_T(y\mid x) = \log \pi_T^{RL}(y\mid x) - \log \pi_T^{ref}(y\mid x).$$

它只保留"RL 让教师改变了什么"，丢弃教师 RL 前已有的偏好——当 student 已强于教师时，
模仿教师终态会覆盖掉更强的行为（论文图 1(a)：R1-Distill-7B 56.7 → OPD 掉到 ~50），
而迁移 $\Delta_T$ 则保留提升方向。

**PG 损失（π_old 加权 + PPO clip）**——本工程 `losses.pg_loss`：

$$L_{pg} = - \sum_{v=1}^{V} \pi_{old}(v)\; \min\!\Big( r(v)\,\Delta_T(v),\;\operatorname{clip}(r(v),1-\epsilon,1+\epsilon)\,\Delta_T(v) \Big),$$

其中 $r(v) = \pi_\theta(v)/\pi_{old}(v)$ 是逐 vocab 重要性比。

- **为什么必须 π_old 加权**：在分布形式下，$\sum_v \pi_{old}(v)\,r(v)\,\Delta_T(v)
  = \sum_v \pi_\theta(v)\,\Delta_T(v) = \mathbb{E}_{\pi_\theta}[\Delta_T]$——
  即 $r=1$ 时精确等于 **Direct-OPD 目标 $-\mathbb{E}_{\pi_\theta}[\Delta_T]$**（on-policy 期望奖励）。
  等权 `mean` 不是这个目标；token 级标量 advantage 形式一阶梯度恒为 0（实测验证）。
- **min(clip) 的悲观下界**：PPO 风格，限制每步更新幅度，防 ratio 爆炸。
- **失配屏蔽**（M1 修复）：$\pi_{old}\approx 0$ 处（支撑外）贡献强制为 0，
  避免 $r=\exp(s_{cur}-s_{old})$ 放大到天文数字造成伪梯度/NaN。

**KL 正则（k3 估计量，π_θ 下期望）**——`losses.low_var_kl`：

$$L_{kl} = \sum_{v} \pi_\theta(v)\; k_3\!\Big(\log\frac{\pi_{ref}(v)}{\pi_\theta(v)}\Big),\qquad
k_3(u) = e^{u} - u - 1.$$

这是 **KL$(\pi_\theta \| \pi_{ref})$ 的低方差逐点估计**（分布形式下恒等），锚定 student
到初始分布，防策略漂移。稀疏锚点版本 `low_var_kl_support` 只在 top-K 支撑求和（有界近似，
系统性略低估真 KL，方向安全）。

**总训练目标**：

$$\mathcal{L}(\theta) = \mathbb{E}_{\substack{x\sim\mathcal{D}\\y\sim\pi_{old}}} \Big[
 - \sum_v \pi_{old}(v)\min\big(r(v)\Delta_T, \operatorname{clip}(r)\Delta_T\big)
 + \lambda_{kl}\, KL(\pi_\theta\|\pi_{ref}) \Big].$$

**效果度量**：AIME24/25 蒸馏前后得分。论文参考（AIME24 ave@32）：ref 28.5 → 教师 51.3；
学生 pre 56.7(7B)/48.3(1.7B) → post +6.4/+10.0。**工程目标**：在 student 上复现这种
「弱到强提升」——即 AIME$_{post} >$ AIME$_{pre}$，且接近论文 Direct-OPD 的增益。

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

### 2.2 吞吐 / 加速比 / 陈旧度权衡

**吞吐模型**：流水线是四个生产者-消费者（$\lambda = 1/\max_i \tau_i$ 稳态），
$N_{batches}$ 批的总时间：

$$T_{total} \approx N_{batches}\cdot \max(\tau_{rollout},\tau_{train}) + \tau_{fill}
+ C_{offline},$$

其中（稳态下）$N_{batches} = N_{steps}$（每 batch 训练一步，陈旧样本被截断则略多）。

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

### 2.3 工程化时间度量

`timings.json` 记录逐段：$T_{stage0}, T_{stage1}, T_{stage2}, T_{total}$。
**时间优化目标**即：在固定 $N_{steps}$ 与模型下最小化 $T_{total}$，且
$T_{stage2}$ 遵循 2.2 的流水线模型（$N_{steps}\cdot\max(\tau_{rollout},\tau_{train})$）。

---

## 目标 3 · 综合优化（效果 × 时间的联合目标）

把两个目标写成带约束的优化（或以 Pareto 前沿理解）：

$$\min_{\theta,\;\text{架构}} \;\; T_{total}(\theta)
\quad \text{s.t.} \;\;
\Delta\text{AIME} := \text{AIME}_{post}(\theta) - \text{AIME}_{pre} \;\ge\; \epsilon,$$

其中 $\epsilon$ 是期望的弱到强增益下限（论文参考：+6.4 / +10.0 / +5.1）。$T_{total}$ 由
预加载教师（$C_{offline}$ 摊销）与异步流水线（$\max(\tau_{rollout},\tau_{train})$）主导。

**可调杠杆**（在两者之间权衡）：
| 杠杆 | 时间影响 | 效果影响 |
|---|---|---|
| 缓存稀疏度 K（topk） | 缓存小、索引快 | 支撑截断 → 有界低估 Δ_T/KL |
| staleness 阈值 θ | ↑θ 吞吐↑ | ↑θ 梯度偏差↑ |
| 批次/队列大小 | ↑ 吞吐 | 稳定性 |
| warmup_M（L1 胖 D） | Stage1 多一次采样 | 缓解曝光偏差 → 效果↑ |
| λ_kl | — | 正则强弱 → 漂移控制 |

**一句话**：全栈在**固定效果目标**（$\Delta\text{AIME} \ge \epsilon$）下，用
「离线预加载教师（砍掉 $t_{teacher}$）+ 异步流水线（$T\propto\max(\tau_{rollout},\tau_{train})$）」
把训练时间压到流水线最慢阶段，而非三者之和——这正是 `timings.json` 要实证的优化收益。

---

## 附 · 数学解释关键点速查

1. **为什么迁移 Δ_T 而非教师终态**：Δ_T = 减法去掉教师 RL 前偏好，只留 RL 改进方向；
   当 student 已强于教师，模仿终态覆盖强行为，迁移偏移则保留提升。
2. **为什么 π_old 加权**：Σ_v π_old·r·Δ = Σ_v π_θ·Δ = E_{π_θ}[Δ_T]（on-policy 期望）。
3. **为什么 k3**：k3(u)=e^u−u−1 是 KL 的逐点被积函数，π_θ 下期望即真 KL，且低方差。
4. **为什么异步快**：生产者-消费者重叠 → 总时间 ≈ 最慢阶段 × 批数，而非各阶段之和。
5. **为什么预加载教师有效**：教师前向从每步移除，离线成本一次摊销（C_offline/N_steps）。
6. **为什么要有 staleness 截断**：age 越界样本的梯度偏差 δ(a) 无界 → 双截断钉住质量。
