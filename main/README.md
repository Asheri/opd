# 全栈 OPD 叠加 Demo（Direct-OPD ▢ Lightning-OPD ▢ AsyncOPD）

把三篇 OPD 论文**全栈叠加**成一个可运行的流水线 demo：

```
小模型 RL  ──►  离线缓存「教师对」log-ratio (Δ_T)  ──►  Direct-OPD 训练跑在 AsyncOPD 调度器上
 (Stage 0)        (Stage 1 · Lightning-OPD)              (Stage 2 · Direct + Async)
```

目标：**同时摆脱三重限制**——

| 限制 | 由谁打破 | 论文 / 真实代码 |
|------|----------|----------------|
| 常驻教师（resident teacher） | **Lightning-OPD** 离线缓存教师 log-prob，训练期不启 teacher server | `Lightning-OPD/data_curation/prepare_lightning_opd.py::phase2_logprobs`、`slime/rollout/on_policy_distillation.py::post_process_rewards` |
| 同步等待（synchronous waiting） | **AsyncOPD** 异步调度器，rollout 与 learner 解耦，陈旧样本 learner 时刻重算 | `async-opd/opd/coordinator/streaming.py`、`opd/utils/staleness_queue.py`、`opd/trainer/teacher_artifact_buffer.py` |
| 迁移终态（migration terminal state） | **Direct-OPD** 迁移对象改为「RL 诱导的策略偏移」Δ_T，作用于更强 student 自身 on-policy 状态 | `Direct-OPD/verl/verl/workers/actor/dp_actor.py::_compute_delta_opd_rm_scores`、`trainer/ppo/core_algos.py::compute_token_reward_direct_advantage` |

---

## 1. 三篇论文的核心抽取

### Lightning-OPD（离线 OPD，消除常驻教师）
- 预计算 teacher log-probs **一次**（在 SFT rollouts 上），训练期直接复用 → 不再需要 live teacher server（4.0× 效率）。
- **Teacher consistency**：SFT 与 OPD 必须用同一 teacher（同 tokenizer/词表/架构），否则引入不可约梯度偏差。
- 本 demo 把该思想扩展到 Direct-OPD 所需的「**教师对**」：缓存 post-RL teacher 与 pre-RL reference 在每条 (prompt, response) 上的 next-token 分布，并预计算 Δ_T。

### Direct-OPD（迁移对象 = RL 策略偏移，消除迁移终态）
- 不直接蒸馏 post-RL teacher 的**最终策略**（那会混入小模型的局限 = 迁移终态限制）。
- 蒸馏 teacher 的 **RL 诱导策略偏移**：`Δ_T(x,a) = log π_T^RL(a|x) − log π_T^ref(a|x)`。
- 该 log-ratio 告诉我们在 student 自己的 on-policy 状态上，RL 让 weak teacher 更/更不可能采取哪些 action；把它作为**密集隐式奖励**施加于 student。
- 真实实现：`delta = teacher_rl_logp − teacher_ref_logp`；`weights = softmax(student_topk_logp)`；`rm = (weights * delta).sum(-1)`。

### AsyncOPD（异步调度，消除同步等待）
- rollout 生成与 learner 更新**解耦**，用有界 FIFO + 版本号做陈旧度管理。
- 关键发现：reverse-KL 对陈旧 rollout 脆弱，但其脆弱性可由 **learner 时刻用当前 student 重算**这一 OPD 专用代理缓解。
- Direct-OPD 把 Δ_T 当 reward（PG 形式）本身即「用当前 student 重算」的稳健估计量 → 全栈叠加天然用上稳健估计量。

---

## 2. 代码地图（与真实 repo 的对应关系）

```
main/
├── run_fullstack.py            # 入口：python run_fullstack.py
├── configs/fullstack_opd.yaml  # 三阶段配置（仅文档用，代码用内置 DEFAULT_CONFIG）
└── fullstack_opd/
    ├── models.py            # ToyModel（可运行的极小 transformer，替代真实 LLM）
    │                         #   对应三个 repo 的 LogProbModel / ActorRollout
    ├── data.py              # RolloutSample / ScoredSample 数据结构
    ├── lightning_cache.py   # ★Stage1 Lightning-OPD 离线教师对 Δ_T 缓存 + teacher consistency
    ├── direct_opd.py        # ★Direct-OPD 迁移对象 Δ_T → 密集奖励 + token_reward_direct 优势
    ├── buffer.py            # ★AsyncOPD StalenessQueue / WeightStore / TeacherArtifactBuffer
    ├── losses.py            # ★AsyncOPD forward/reverse KL + learner 时刻重算 (PPO clip)
    ├── async_scheduler.py   # ★Stage2 AsyncOPD 全异步调度器 (PromptFeeder/RolloutCollector/
    │                         #   TeacherScorer/TrainDispatcher 四线程)
    ├── stages.py            # Stage0 小模型 RL；Stage1 建缓存；Stage2 训练 student
    └── pipeline.py          # FullStackOPD 编排器 + DEFAULT_CONFIG
```

> 标注 ★ 的是本 demo 的核心新增代码；其余为让 demo 能端到端跑起来的可运行替身（真实场景替换为三个 clone 下来的 repo 的对应模块即可）。

---

## 3. 运行

```bash
cd C:/Users/12062/OneDrive/Desktop/opd/main
python run_fullstack.py
```

> demo 仅依赖 `torch`（CPU、极小词表即可跑通端到端，证明全栈叠加逻辑正确）。真实训练时把 `ToyModel` 换成 `async-opd` 的 rollout/trainer、`Direct-OPD/verl` 的 actor、`Lightning-OPD/slime` 的 teacher 缓存即可。

---

## 4. 流水线在代码里的样子（伪代码）

```python
# Stage 0 —— 小模型 RL：产生 post-RL weak teacher，pre-RL reference 为其训练前副本
teacher_rl, teacher_ref = stage0_small_rl(prompts)

# Stage 1 —— Lightning-OPD：离线缓存「教师对」Δ_T（无 live teacher）
cache = OfflineTeacherPairCache(enforce_consistency=True)
cache.build(prompts, responses, teacher_rl, teacher_ref)   # 一次性预计算，落盘
#   cache 内部：Δ_T[p] = logπ_rl(·|prompt_p) − logπ_ref(·|prompt_p)

# Stage 2 —— Direct-OPD 训练跑在 AsyncOPD 调度器上
student = AsyncOPDScheduler(student_init, cache, prompts, responses).run(n_steps)
#   RolloutCollector: 用（可能陈旧的）student 快照生成 on-policy rollout，打版本号
#   TeacherScorer:   从离线 cache 取 Δ_T（★无 live teacher）
#   TrainDispatcher: 用【当前】student 重算 logp → PPO clip 处理陈旧 →
#                    Direct-OPD 奖励 = E_student[Δ_T]（★迁移对象是策略偏移，不是终态）
#                    + low-var KL 正则防止策略漂移（Lightning 隐式正则）
```

跑完后 `run_fullstack.py` 会打印每一步的 `version / staleness_age / loss / E[Δ_T]`，可直接看到：
- `staleness_age > 0` → 异步调度确实在消费陈旧样本（同步等待被打破）；
- `E[Δ_T]` 作为 student 的密集奖励被优化（迁移终态被打破）；
- 训练循环里不出现任何 teacher 前向（常驻教师被打破）。

---

## 5. v2：批量化重构版（fullstack_opd_v2/）

v1 验证算法正确性后，v2 **从底层重写执行底座**（算法内核不变：因果 LM、π_old 加权 PG、
k3 KL、staleness 双截断）：

| 重构点 | v1 | v2 |
|--------|----|----|
| 前向粒度 | 每样本 unsqueeze(0) 单独前向 | 原生 (B,...) 批次形状，一次前向覆盖整批 |
| 解码 | 逐样本逐 token | `generate_batch` 整批同步自回归 |
| 教师缓存 | dict[int, CPU tensor]，逐步搬运 | 设备常驻 (N,T,V) 张量 + 预计算 Δ_T，零拷贝索引 |
| 权重同步 | 每样本 acquire + load_state_dict | `acquire_if_newer` 版本推进才加载 |
| 规则奖励 | 逐 token python 循环 | (V,) 查找表向量化 |
| 队列载荷 | 单样本 | mini-batch（四阶段结构不变） |
| 因果 mask | 每步重建 | 按 (长度, 设备) 缓存 |

```
main/
├── run_fullstack.py          # v1 入口（算法基线，已审阅修复）
├── run_fullstack_v2.py       # v2 入口（批量化重构）
├── benchmark.py              # v1 vs v2 基准对比
├── fullstack_opd/            # v1 包
└── fullstack_opd_v2/         # v2 包：model / losses / cache / buffer / scheduler / pipeline
```

实测（CPU，torch 2.13，30 步，batch=8）：

```
                     v1 (逐样本)   v2 (批量化)
总墙钟时间 (s)            2.97         1.80    → 1.65x
stage2 吞吐 (样本/s)      29.0        233.2    → 8.04x
最终 E[Δ_T]             +0.66       +0.72    （v2 每步 8 样本，收敛更快）
staleness age               5            5    （异步语义保持一致）
```

```bash
python run_fullstack_v2.py train   # 跑 v2
python benchmark.py          # v1 vs v2 对比
```

### 5.1 工程化（P0：测试 / 配置 / 打包）

**安装为包**（去掉 `sys.path` hack，可任意目录导入/运行）：
```bash
pip install -e .                 # 见 pyproject.toml（deps: torch / pydantic / pyyaml）
python -m fullstack_opd_v2 train # 模块入口（子命令见 §7）
fullstack-opd-v2 train           # console script
```

**YAML 配置真加载 + schema 校验**（`fullstack_opd_v2/config.py`，pydantic `extra=forbid` 拒绝未知/拼错键）：
```bash
python -m fullstack_opd_v2 train --config configs/fullstack_opd.yaml
python -m fullstack_opd_v2 train --set stage2.n_steps=50 --set stage1.warmup_source=mix --set stage1.warmup_M=4
```
任何未知键 / 非法枚举值（如 `dtype: fp16`）都会显式报错，不再静默忽略。

**测试**（pytest，`tests/`：losses/cache/buffer/scheduler/pipeline/config）：
```bash
pytest tests/ -q                 # 42 passed
```

> 进一步加速的空间（demo 未做，真实规模由对应组件接管）：KV cache 增量解码（→ vLLM）、
> CUDA 图 / torch.compile、多进程 rollout worker（→ Ray）、混合精度（→ megatron-core）。

---

## 6. 审阅修复记录（v1，2026-08-06）

对 v1 做完整审阅并修复 11 处：P0 因果 mask 缺失（双向注意力偷看未来 token）、
P0 `__init__.py` 未 import；P1 PG 损失两种错误形式（等权 mean ≠ 目标；token 级标量 adv
一阶梯度恒为 0）→ 修为 π_old 加权逐 vocab 重要性采样；P1 `low_var_kl` 等权 mean 不是 KL →
π_student 加权 k3 期望；P1 StalenessQueue 队列本体闲置 → 兼任 scored 队列；
P2 梯度裁剪 / 播种 / dropout=0 / teacher 一致性加强 / weights_only / 导入位置 / 负索引 / REINFORCE baseline。
修复后 E[Δ_T] −0.18 → +0.72 单调上升。

---

## 7. 工程化改造（demo → 真正工程项目，2026-08-09）

把 v2 demo 改造成工程化项目：**算法内核一行不动**，新增可复现、可续跑、可对比的运行底座。

### CLI 子命令（`python -m fullstack_opd_v2`）

```bash
# train：跑全栈流水线（Stage 0/1/2），落盘 run 目录
python -m fullstack_opd_v2 train --config configs/fullstack_opd.yaml --run-dir runs/exp1
python -m fullstack_opd_v2 train --config configs/fullstack_opd.yaml --run-dir runs/exp1 --resume   # 断点续跑
# cache：只建 Lightning 离线缓存（Stage 1）
python -m fullstack_opd_v2 cache --config configs/fullstack_opd.yaml --out /shared/cache.pt
# eval：载入 checkpoint 学生（健康信号 / AIME 蒸馏后评估的入口）
python -m fullstack_opd_v2 eval --checkpoint runs/exp1/checkpoints/step_30.pt --config configs/fullstack_opd.yaml
# info：打印解析后完整配置（校验 YAML/覆盖合法）
python -m fullstack_opd_v2 info --config configs/fullstack_opd.yaml --set stage2.n_steps=50
```

### run 目录结构（每次训练的可复现单元）

```
runs/<timestamp>/
├── config.yaml      ← 解析后配置快照（可复现）
├── metrics.csv      ← 每步指标（loss/pg/kl/adv/reward/age/version）
├── timings.json     ← 逐 stage 计时（衡量「异步+预加载」的时间优化）
├── train.log        ← 结构化日志
└── checkpoints/     ← step_<N>.pt（断点续跑 + AIME 蒸馏后评估）
```

### 新增模块

| 模块 | 职责 |
|---|---|
| `cli.py` | 子命令 train/cache/eval/info |
| `logging.py` | 结构化日志（控制台+文件） |
| `data.py` | 可插拔数据接口（Toy 默认 / JsonLines 预留） |
| `model_factory.py` | 可插拔模型工厂（toy 默认） |
| `run.py` | run 目录管理 |
| `checkpoint.py` | 断点保存/加载/续跑 |
| `metrics.py` | 指标 CSV/WandB |
| `exceptions.py` | 类型化异常层级 |

### 与实验指标对齐

- **训练时间（优化指标）**：`timings.json` 记录 stage0/stage1/stage2/total，跨 run 对比异步+预加载教师的时间收益。
- **AIME 蒸馏前后（效果指标）**：checkpoint 落 `checkpoints/step_<N>.pt`，配合 `benchmarks/aime24_25/` 评估蒸馏前后得分。

### 配置新增段

```yaml
model_kind: toy                 # 可插拔模型
run: {seed: 42, run_dir: null, checkpoint_every: 10}
logging: {level: INFO, file: train.log}
metrics: {backend: csv, csv_path: null, wandb_project: null}
dataset: {type: toy, path: null, prompt_key: prompt, response_key: response}
```
