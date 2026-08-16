# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库定位

本目录是「全栈 OPD 叠加」研究工作区：把三篇 OPD 论文的机制叠加成一条可运行流水线，
同时打破 **常驻教师 / 同步等待 / 迁移终态** 三重限制。

```
小模型 RL ──► 离线缓存「教师对」log-ratio Δ_T ──► Direct-OPD 训练跑在 AsyncOPD 调度器上
 (Stage 0)      (Stage 1 · Lightning-OPD)          (Stage 2 · Direct + Async)
```

| 目录 | 性质 | 说明 |
|------|------|------|
| `main/` | **本项目自研代码** | 全栈叠加 demo（纯 torch，CPU 可跑）。绝大多数改动应发生在这里 |
| `async-opd/` | 上游 clone（独立 git repo） | 调度器基座：vLLM rollout + FSDP/Megatron 训练 + NCCL 权重同步 |
| `Direct-OPD/` | 上游 clone（独立 git repo） | patched `verl`；Δ_T 隐式奖励的原始实现 |
| `Lightning-OPD/` | 上游 clone（独立 git repo） | `slime` + sglang；离线教师 logprob 预计算 |
| `.workbuddy/memory/` | 会话日志 | 按日期记录任务决策与审阅结论；改动前值得翻最近几天 |

顶层 `opd/` 本身**不是** git 仓库；三个子目录各自是独立 clone（有自己的 remote）。
不要在三个上游 clone 里做实验性改动——它们是参照实现；需要的机制应 vendor 进 `main/`。

## 开发命令（`main/`）

Python 解释器注意：顶层 `.venv/` 是空壳（无 torch/pytest）。实际可用环境是
`C:/Users/12062/AppData/Local/Programs/Python/Python311/python.exe`
（torch 2.11.0+cpu / pytest 9.0.3 / pydantic 2.13.3），且 `main/` 已 `pip install -e .`。

```bash
cd main

# 测试（42 个，全 CPU，~16s）
python -m pytest tests/ -q
python -m pytest tests/test_losses.py -q                      # 单文件
python -m pytest tests/test_scheduler.py::test_ages_bounded -q # 单测试

# 运行 v2（推荐入口，三者等价；子命令 train 见 cli.py）
python -m fullstack_opd_v2 train
fullstack-opd-v2 train
python run_fullstack_v2.py train

# YAML 配置 + 点分 CLI 覆盖
python -m fullstack_opd_v2 train --config configs/fullstack_opd.yaml
python -m fullstack_opd_v2 train --set stage2.n_steps=50 --set stage1.warmup_source=mix --set stage1.warmup_M=4
python -m fullstack_opd_v2 train --device cpu

# v1 基线 / v1-vs-v2 基准
python run_fullstack.py
python benchmark.py
```

上游 repo 的命令（仅在需要对照原版时）：

```bash
cd async-opd && make public-check   # CPU-safe 控制面 smoke test
python -m opd.cli.train --config configs/examples/opd_gsm8k_0.5b_4gpu.yaml --overwrite   # 需 GPU
```

## `main/` 架构

**两个并存的包，算法内核相同、执行底座不同。新工作走 v2。**

- `fullstack_opd/`（v1）：逐样本执行底座，作为算法正确性基线保留。已完整审阅修复 11 处。
- `fullstack_opd_v2/`（v2）：批量化重构 + GPU 部署骨架。**默认工作目标。**

v2 模块职责：

| 模块 | 角色 |
|------|------|
| `pipeline.py` | 编排器 `FullStackOPDv2` + `DEFAULT_CONFIG_V2` + `CLOUD_CONFIG`（2×RTX PRO 6000 预设）+ `stage0_small_rl` / `stage1_build_cache` |
| `config.py` | pydantic schema（`extra="forbid"`）+ `load_config()`：YAML → 默认合并 → 点分覆盖 → 校验 |
| `cache.py` | `TensorTeacherCache`：设备常驻 Δ_T 张量，dense `(N,T,V)` 或 topk `(N,T,K)`；teacher 一致性校验 |
| `losses.py` | `pg_loss` / `low_var_kl` / `low_var_kl_support` / `expected_reward` |
| `buffer.py` | `StalenessQueue`（版本号 + 双截断）、`WeightStore`（`acquire_if_newer` 按需加载 + 可选 CPU offload） |
| `scheduler.py` | `AsyncBatchedScheduler` 四线程流水线；`DistAsyncScheduler` / `WeightBroadcaster` / `parallelize_learner_tp2`（GPU 骨架） |
| `model.py` / `model_megatron.py` | `CausalToyLM`（占位小 transformer）/ Megatron TP+SP 版 |
| `rollout_vllm.py` | `VLLMRolloutEngine`：与 `response_dists` 接口对齐的 vLLM 替换 |
| `demo.py` / `__main__.py` | CLI 入口 |

### Stage 2 四线程流水线（`AsyncBatchedScheduler`）

```
PromptFeeder ──(B,)索引──► RolloutCollector ──(idxs,s_old,ver)──► TeacherScorer ──贴 Δ_T──► TrainDispatcher
```

- `RolloutCollector` **名字有误导性**：它不自回归采样，只对固定 `(prompts, responses)`
  做 teacher-forcing 算 `s_old`。真正的 `generate_batch` 只在 Stage 0 和 Stage 1 warmup 用。
- 权重只在版本推进时加载（`acquire_if_newer`），不是每样本 `load_state_dict`。
- 陈旧度**双截断**：入队侧 `StalenessQueue.put` 拒收过旧，消费侧 `_train_step` 再查一次。
- `TrainDispatcher` 用**当前** student 一次批量前向重算 `s_cur`（learner-side recompute 代理）。

## 不可回退的算法约束

这些是审阅修复的结论，改动损失/缓存时必须守住（`losses.py` 顶部注释也有记录）：

- **PG 必须按行为策略 `π_old` 加权**逐 vocab 重要性采样：
  `−Σ_v π_old(v)·min(ratio(v)·Δ(v), clip(ratio)·Δ(v))`。`ratio=1` 时精确等于
  `−E_{π_cur}[Δ_T]`。等权 `mean` 目标错误；token 级标量 advantage 形式一阶梯度恒为 0。
- **KL 正则用 k3 估计量在 `π_student` 下取期望**，分布形式下恒等真 `KL(π‖π_ref)`。等权 mean 不是 KL。
- **因果 mask 必须存在**（曾漏掉导致双向注意力偷看未来 token）。
- **`low_var_kl_support` 是有界近似，不是恒等替换**：只对 top-K 支撑求和，系统性略低估真 KL
  （方向安全）；支撑内但不在 ref top-K 的 token 填极负 `ref_tail_logp` 给出强漂移惩罚。
- **稀疏 top-K 不重归一化**是有意的省显存近似；`§0.6` 的数学是 dense 形式严格成立，demo 默认 dense。
- **teacher 一致性**：`teacher_rl` 与 `teacher_ref` 必须同架构/词表/`d_model`/`max_len`，
  否则引入不可约梯度偏差（`TensorTeacherCache` 会抛 `TeacherConsistencyError`）。

健康信号：`E[Δ_T]` 应随训练单调上升（修复后 −0.18 → +0.72），`staleness age > 0`
证明异步确实在消费陈旧样本，训练循环内不应出现任何 teacher 前向。

## 配置约定

`configs/fullstack_opd.yaml` **现在真的被加载**（早期版本仅作文档、代码用硬编码默认）。
pydantic `extra="forbid"` + `Literal` 使未知键/拼错键/非法枚举值显式报错，不再静默忽略。

**「顶层部署键被 stage 子字典静默忽略」是本项目踩过的 P0 bug 类型。**
`CLOUD_CONFIG` 风格把 `dtype` / `cache_mode` / `top_k_teacher` / `top_k_student` /
`offload_to_cpu` 放在顶层；`load_config()`（config.py 的 `_seep_deployment_keys`）在
pydantic 校验前把顶层部署键**按消费端分流**下渗到 `stage1` / `stage2`（stage 子键优先）：
stage1 ← {cache_mode, top_k_teacher}，stage2 ← {dtype, top_k_student, offload_to_cpu}。
`ref_topk` 保持**纯顶层**（pipeline 读 `self.cfg.get("ref_topk")`），不下渗。
新增顶层部署键时**必须同步**：在 `_STAGE1_SEEP_KEYS` / `_STAGE2_SEEP_KEYS` 分流表加键 +
在对应 stage schema 加下渗槽位，否则该开关会被静默忽略、退回默认路径。

## 曝光偏差与 L0–L3 谱

离线固定 `D` 的 OPD 有固有曝光偏差（**不是 bug**，是离线程度的代价）。缓解路径成谱：

- **L0** 固定 `D` 永久固定（偏差最大）
- **L1 已实现**：Stage 1 用初始 student / 教师分布额外采样 M 条拼「胖 D」再 `cache.build`。
  由 `stage1.warmup_M` / `warmup_source`（`none` | `student_init` | `teacher_perturbed` | `mix`）
  / `warmup_temperature` 控制，**默认开启（= L1：学生 ref 一次性 rollout 拼胖 D）**，
  关闭则 `warmup_M=0`+`warmup_source=none`。调度器与缓存内核**零改动**。
  `stage1_build_cache` 返回 `(cache, fat_prompts, fat_responses)`，Stage 2 的 KL 锚点与调度器
  都用 `fat_*`——warmup 分布与 KL 锚点必须同源（`student` 因此提前到 Stage 1 前创建）。
  计数：`student_init` M → `N×(1+M)`；`mix` M → `N×(1+2M)`。
- **L2 未实现**：周期刷新（常驻 teacher + `_rollout_refresh` 线程 + `cache.append` ring buffer + 混合 feeder）
- **L3** = 全在线，即退回原版 Lightning-OPD

## GPU 部署骨架状态

`distributed` / `tp_size>1` / `rollout_engine: vllm` 是**带护栏的骨架**（ray / megatron-core /
vllm 为可选导入，缺失时报错）。本地 CPU demo 默认全关。L2 的 colocated 交替相位仍待实现。
算法内核在分布式路径下被直接复用（`_train_step` 不动）。

服务器部署方案见 `DEPLOY.md`（统一环境：Python 3.12 / torch 2.9.1 / CUDA 12.8 / vLLM 0.16.0，
不装完整 verl、裁掉 sglang）与 `OPTIMIZATION_PLAN*.md`（8×A100 / 8×4090 / 2×RTX PRO 6000 三套）。

## 全局资源利用约束（GPU ≥ 2）

**当可用 GPU 数量 ≥ 2 时，必须考虑使用所有 GPU 以最大化利用**，不得无理由串行单卡：

- **多模型评估**：如有 3 个待评估模型且 2 张 GPU，应把模型并行分到两张卡上（如 2+1 分片），
  而非逐个串行跑在同一张卡上（并行总耗时 ≈ 串行/卡数）。
- **rollout / 数据生成**：如需 rollout 200 条且 2 张 GPU，应每张卡并行 rollout 100 条
  （分片 + 结果合并；`run_s2_real.py` / `run_l2_real.py` / `calibrate_rollout.py` 等按
  `--device cuda:0/1` 分别起进程），而非单卡串行 200 条。
- **训练 / 建缓存**：batch/样本可分片时优先多卡并行（FSDP / 数据并行）；规模超过单卡内存时
  用多卡分载（如 2×RTX PRO 6000 96GB 双卡）。
- 决策顺序：先判断任务是否可分片 → 可分片则**优先并行**；只有任务本身有顺序依赖 / 显存或
  通信瓶颈使并行无收益 / 用户显式指定单卡时，才回退单卡，且应说明原因。

## 深入阅读

- `main/README.md` — 三篇论文核心抽取、代码地图、v1→v2 重构对照表、审阅修复记录
- `main/fullstack_opd_v2/TECHNICAL_REPORT.md` — **技术文档与训练分析报告（唯一权威）**：
  §0–§4 工程实现（端到端时序、数学模型、异步机制、教师离线、信用分配、边界）、
  §5 benchmark 分数与评估协议、§6 显存、§7 用时、§8 数据构成、§9 已知边界与复现
- `DEPLOY.md` — 依赖冲突的架构裁剪方案与安装步骤

## 文档要求（工程实现技术文档 + 训练分析报告）

**本项目的长期硬性要求**：需要把目前为止的详细工程实现（**按照原始论文修改后的版本**，
即 Direct-OPD / Lightning-OPD / AsyncOPD 三篇叠加 + 本项目的落地改动）写成一份
**详细技术文档**（建议 `main/fullstack_opd_v2/TECHNICAL_REPORT.md`，或按需分章节）。
该文档**必须包含**以下部分，缺一不可：

1. **工程实现（按原始论文修改后）**：完整描述三篇论文机制如何叠加、每一步相对原始论文
   的改动及其理由（例如：离线固定 `D` 的 L0/L1 改动、`renormalize_topk_support` 对齐原始
   Direct-OPD、跨词表 `delta_for_student_topk`、async 调度器解耦等）。可参考/继承
   `TECHNICAL_REPORT.md` §0–§4 的数学对齐写法，但要按「当前代码的真实状态」更新。

2. **训练分析（必须含 benchmark 分数与协议）**：
   - **训练前后（pre/post）的 benchmark 分数对比**：基座 vs 学生（如 1.7B/7B/4B 三档），
     短生成与长生成两套都要记录。
   - **benchmark 方式必须写全协议**：论文对齐口径为
     `avg@32, n=32, T=0.7, top_p=0.95, max_new_tokens=32768, boxed 模板, sympy 评分`
     （以及 chat_template 包裹、batch_size、dtype 等实测参数）。每个数字都要注明用哪套
     协议测出，绝不混用（本项目踩过 pass@1 vs ave@32 混报的坑）。
   - 若某数字是短生成（如 2048）测的，必须显式标注"短生成，非论文协议"。

3. **训练与评估的显存占用分析**：逐阶段（stage0 RL / stage1 cache build / stage2 训练 /
   AIME 评估）实测或推算显存峰值，说明构成（权重 / KV cache / logits / 激活 / 中间张量）。
   记录关键教训，例如：长序列（32K）× 大 vocab（151936）下 **logits 张量是隐形显存杀手**；
   `attn_implementation` 未显式设 flash_attn 时 SDPA 开销大；batch_size 与峰值显存的关系。

4. **训练与评估的用时分析**：各阶段 wall-clock 耗时、每数据集/每采样平均耗时、吞吐
   （token/s）、batch_size 对用时的加速比（如 batch 1→2）、长生成 vs 短生成的时间放大倍数。

5. **训练数据构成分析**：数据集来源（Skywork/DAPO/AIME）、训练集与评估集划分、每条样本
   prompt 模板、`max_prompt_length` / `max_response_length` / `MAX_VAL_RESP_LENGTH` 等长度
   配置、教师对 Δ_T 的缓存模式（dense/topk、top_k）、warmup 拼接对数据量的影响（`N×(1+M)`）。

6. **其他必要信息**：参考原始论文的机制速查、已知边界与未实现项（如 L2 周期刷新）、
   复现步骤（命令 + 配置）、与本文件其它节（架构/算法约束/配置约定）的交叉引用。

> 写作时遵循「代码注释、文档、提交信息均用中文」的全局要求。文档随代码演进持续维护，
> 不要让它过期（当前代码的真实状态为准，不沿用旧描述）。

## 语言

代码注释、文档、提交信息均用中文（与现有代码风格一致）。
