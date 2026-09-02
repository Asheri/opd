# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库定位

本目录是「全栈 OPD 叠加」研究工作区：把三篇 OPD 论文的机制叠加成一条可运行流水线，
同时打破 **常驻教师 / 同步等待 / 迁移终态** 三重限制。

```
学生 rollout ──► 教师 only_stu 实时打分 Δ_T ──► ring buffer ──► Direct-OPD 训练
 (rollout 相位)   (每相位现算, 非预计算)          (学生支撑)   (训练相位, α 冻结 1.0)
```

> **2026-08-31 P-OPD 重建**：stage1 预计算缓存（Lightning-OPD 离线 Δ_T）与 base 池（固定 D）
> **已删除**。训练 = 纯 on-policy 交替相位（rollout ↔ 训练），教师 Δ 用 only_stu 口径
> （教师对学生 top-K 完整支撑 logp 差，官方重算信号强 Eq.13 +0.539/+0.596）。

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

# 测试（569 个，全 CPU，~2min）
python -m pytest tests/ -q
python -m pytest tests/test_losses.py -q                      # 单文件
python -m pytest tests/test_scheduler.py::test_ages_bounded -q # 单测试

# 运行 P-OPD（推荐入口；子命令 train 见 cli.py）
python -m fullstack_opd_v2 train
fullstack-opd-v2 train
python run_fullstack_v2.py train

# P-OPD 主配置（纯 on-policy） + 点分 CLI 覆盖
python -m fullstack_opd_v2 train --config configs/qwen3_r1_onpolicy.yaml
python -m fullstack_opd_v2 train --config configs/qwen3_r1_onpolicy.yaml \
  --set stage2.n_steps=300 --set l2.rollout.eos_token_id=151645
python -m fullstack_opd_v2 train --device cpu

# toy demo（历史路径，仅供 CPU 冒烟）
python -m fullstack_opd_v2 train --config configs/fullstack_opd.yaml
```

上游 repo 的命令（仅在需要对照原版时）：

```bash
cd async-opd && make public-check   # CPU-safe 控制面 smoke test
python -m opd.cli.train --config configs/examples/opd_gsm8k_0.5b_4gpu.yaml --overwrite   # 需 GPU
```

## `main/` 架构

**仅 `fullstack_opd_v2/`（v1 `fullstack_opd/` 与 precompute 子系统已于 2026-08-31 删除）。**

v2 模块职责（P-OPD 纯 on-policy）：

| 模块 | 角色 |
|------|------|
| `pipeline.py` | 编排器 `FullStackOPDv2`：加载教师对（`_stage0_teachers`）+ 占位 cache → 纯 refresh 交替相位循环（`run_refresh_phase` ↔ `train_refresh_phase`，`while step_done` 驱动，α 冻结 1.0） |
| `adaptive_cache.py` | `run_refresh_phase`（学生 rollout → 教师 only_stu 前向算 Δ → ring buffer）、`RefreshRingBuffer`、`_rl_ref_delta_only_stu`（教师对学生 top-K 完整支撑 logp 差） |
| `config.py` | pydantic schema（`extra="forbid"`）+ `load_config()`；`l2.pure_refresh` / `stage1.skip` / `l2.cache.max_empty_phases` |
| `cache.py` | `TensorTeacherCache` **占位**（仅 mode/top_k/vocab，无 Δ 数据；base 池已删，训练不消费缓存） |
| `scheduler.py` | `AsyncBatchedScheduler`（`_train_step_refresh` teacher-free 训练 + `train_refresh_phase`；base 池 `_train_step` 保留标注废弃）；GPU 骨架 `DistAsyncScheduler` / `WeightBroadcaster` |
| `losses.py` / `buffer.py` | `pg_loss` / `low_var_kl` / `low_var_kl_support`；`StalenessQueue` / `WeightStore` |
| `model.py` / `model_factory.py` | `CausalToyLM`（toy）/ `HFCausalLM`（真实 HF 模型） |
| `rollout_vllm.py` | `VLLMRolloutEngine`：vLLM 替换；`rollout_weight_sync=off` 时 `apply_model(load_weights)` 直拷逃生舱（tp=1） |
| `eval_aime.py` / `budget_eval.py` | 论文对齐评估（ave@32 / DAPO 模板） |

### P-OPD 交替相位（rollout ↔ 训练）

```
[rollout 相位] 当前学生 rollout（m_refresh×n_rollout 条）
   → 教师 rl/ref only_stu 前向算 Δ（学生 top-K 完整支撑，gather−logsumexp）
   → append ring buffer（ids=学生 top-K，行为 s_old，ref 锚点）
[训练相位]   `_train_step_refresh` 从 ring buffer 采样稀疏 top-K PG + KL（teacher-free）
   → step_done 推进；α 冻结 1.0（100% on-policy）
```

- **无 base 池**：`l2.enabled=false` 明确报错（纯 on-policy 唯一路径）；`scheduler.run`（旧 base 池）不再被调用。
- 空相位防护：冷启动/池空/rollout 全无效连续超过 `l2.cache.max_empty_phases` 次 → 明确失败（防死循环/静默空跑）。
- 断点：checkpoint 含 ring buffer（`refresh_buffer`）+ optimizer + RNG，resume 可续跑。

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

健康信号：`E[Δ_T]` 应随训练单调上升（旧实验 −0.18 → +0.72 参考）；纯 on-policy 下
训练相位（`_train_step_refresh`）**无 teacher 前向**（teacher 只在 rollout 相位
`run_refresh_phase` 算 Δ），`l2.rollout.require_weight_sync` 保证 vLLM 权重同步 fail-closed。

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

## 曝光偏差与 on-policy 化

**2026-08-31 P-OPD：已彻底 on-policy**——stage1 预计算缓存（L0/L1 静态路径）与 base 池固定 `D`
**已删除**。训练样本全部来自**当前学生每相位新鲜 rollout**（rollout 相位 ↔ 训练相位交替），
教师 Δ 用 only_stu 口径（学生 top-K 完整支撑）实时计算，无曝光偏差、无固定 D 的离线代价。

历史谱（供理解演进）：
- **L0** 固定 `D` 永久固定（旧，已删）
- **L1** warmup 拼胖 D（旧，已删——`stage1.warmup_M` / `warmup_source` 不再消费）
- **L2（现唯一路径）** 周期刷新 → 已演进为**纯 on-policy 交替相位**（无 base 池，
  `run_refresh_phase` ↔ `train_refresh_phase`，α 冻结 1.0）
- **L3** = 全在线（概念保留；当前 P-OPD 即全在线变体）

## GPU 部署骨架状态

`distributed` / `tp_size>1` / `rollout_engine: vllm` 是**带护栏的骨架**（ray / megatron-core /
vllm 为可选导入，缺失时报错）。本地 CPU demo 默认全关。**P-OPD 交替相位已落地**
（rollout 相位学生生成 + 教师 only_stu 前向 ↔ 训练相位，teacher 在 refresh 相位 CPU offload
搬回 GPU）。算法内核在分布式路径下被直接复用（`_train_step_refresh` 不动）。

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
- **vLLM 权重同步逃生舱（2026-08-31）**：vLLM 0.16 NCCL WeightTransferEngine 在
  Blackwell(sm_120) 与 Ada(sm_89) 均报 `Expected ... got:cuda`（`init_weight_transfer_engine`
  worker 侧，engine poisoned，见 training-errors.md E18）→ **统一用 `rollout_weight_sync=off`**
  逃生舱：`LLM.apply_model(load_weights)` 直拷（仅 `tp_size=1`，不跨 TP 分发）。on-policy
  保持（每次 rollout 相位前直拷学生权重进 vLLM）。vLLM 修复后可改回 `auto`（NCCL，需
  trainer/worker 异卡交叉分卡，2026-08-17 实测 `Duplicate GPU detected` 教训）。
- 决策顺序：先判断任务是否可分片 → 可分片则**优先并行**；只有任务本身有顺序依赖 / 显存或
  通信瓶颈使并行无收益 / 用户显式指定单卡时，才回退单卡，且应说明原因。

## 训练产物不可再生约束（metrics / 断点 / jsonl）

**训练与评估产出的 metrics.csv、checkpoint、jsonl 是唯一事实来源，一旦丢失无法重建**。
教训（2026-08-26 诊断报告记录）：E1 训练 metrics.csv 因「resume 重复截断 + 清理失误」
只保留 step 0-179，step 180-299 共 120 步 eval_reward 永久丢失，只能靠 monitor.sh
监控记录补救——**永远不许再发生**。

- **resume 续跑前必须备份 metrics**：`run_s2_real.py` 的 `_truncate_metrics_csv` 是
  破坏性操作（把 metrics.csv 截断到断点前）。**代码已内置自动备份**：`--resume`
  续跑前自动 `cp metrics.csv → metrics_pre_resume_step<N>.csv`（已存在不覆盖），
  无需手工备份；任何手工清理仍需先确认备份/监控记录存在，无备份禁止删。
- **任何清理/覆盖/删除 run-dir、metrics.csv、checkpoint、评估 jsonl 前，先检查目标**：
  确认已有备份或监控记录可替代才允许删除；不确定就不删，先问。
- **监控是 metrics 的兜底**：训练期间保持 monitor.sh 抓取 eval_reward 等关键指标
  （metrics 丢失后唯一可恢复的曲线来源）；监控输出定期归档，不随手清理。
- **不伪造**：metrics 缺段时如实标注 N/A 与数据来源（监控 or csv），绝不编造数字填补。

## 全局网络资源约束（HF / GitHub 超时）

**当 HuggingFace / GitHub 等外网连接超时或不可达时，必须考虑使用学术资源加速**：

```bash
source /etc/network_turbo   # AutoDL 学术加速（代理 github/huggingface）
```

- 适用：模型/分词器下载、`huggingface-cli`、`git clone` GitHub 仓库、pip 走 HF/GitHub 源等。
- 服务器（AutoDL）默认无直连外网；直连报 `Network is unreachable` / 连接 huggingface.co
  超时时，先 `source /etc/network_turbo` 再重试，不要直接判失败。
- 优先级：**本地已有模型路径 > 加速代理下载**。本地路径可用时（如
  `/root/autodl-tmp/models/...`）一律走本地 + `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`，
  不触网；仅当本地缺失必需文件时才走加速代理。
- 注意：加速开启后访问其他资源（如 pip 常规源）会更慢，用完可在同一 shell 之外另开
  无代理会话；no_proxy 已含 modelscope/aliyuncs 等国内源。

## 深入阅读

- `main/README.md` — 三篇论文核心抽取、代码地图、P-OPD 重建说明
- `main/fullstack_opd_v2/TECHNICAL_REPORT.md` — **技术文档与训练分析报告（唯一权威）**：
  工程实现（端到端时序、数学模型、only_stu 教师 Δ、交替相位、逃生舱）、
  benchmark 分数与评估协议、显存、用时、数据构成、已知边界与复现
- `DEPLOY.md` — 依赖冲突的架构裁剪方案与安装步骤（P-OPD：教师 offload + vLLM off 逃生舱）

## 文档要求（工程实现技术文档 + 训练分析报告）

**本项目的长期硬性要求**：需要把目前为止的详细工程实现（**按照原始论文修改后的版本**，
即 Direct-OPD / Lightning-OPD / AsyncOPD 三篇叠加 + 本项目的落地改动）写成一份
**详细技术文档**（建议 `main/fullstack_opd_v2/TECHNICAL_REPORT.md`，或按需分章节）。
该文档**必须包含**以下部分，缺一不可：

1. **工程实现（按原始论文修改后）**：完整描述三篇论文机制如何叠加、每一步相对原始论文
   的改动及其理由（例如：**纯 on-policy 交替相位**（删 stage1 预计算缓存 + base 池固定 D）、
   **only_stu 教师 Δ 口径**（`_rl_ref_delta_only_stu` 对学生 top-K 完整支撑 gather−logsumexp）、
   `renormalize_topk_support` 对齐原始 Direct-OPD、vLLM `rollout_weight_sync=off` 逃生舱等）。
   可参考/继承 `TECHNICAL_REPORT.md` 的数学对齐写法，但要按「当前代码的真实状态」更新。

2. **训练分析（必须含 benchmark 分数与协议）**：
   - **训练前后（pre/post）的 benchmark 分数对比**：基座 vs 学生（如 1.7B/7B/4B 三档），
     短生成与长生成两套都要记录。
   - **benchmark 方式必须写全协议**：论文对齐口径为
     `avg@32, n=32, T=0.7, top_p=0.95, max_new_tokens=32768, boxed 模板, sympy 评分`
     （以及 chat_template 包裹、batch_size、dtype 等实测参数）。每个数字都要注明用哪套
     协议测出，绝不混用（本项目踩过 pass@1 vs ave@32 混报的坑）。
   - 若某数字是短生成（如 2048）测的，必须显式标注"短生成，非论文协议"。

3. **训练与评估的显存占用分析**：逐阶段（教师对加载 / 交替相位训练 + 教师 only_stu 前向 /
   AIME 评估）实测或推算显存峰值，说明构成（权重 / KV cache / logits / 激活 / 中间张量）。
   记录关键教训，例如：长序列（32K）× 大 vocab（151936）下 **logits 张量是隐形显存杀手**；
   `attn_implementation` 未显式设 flash_attn 时 SDPA 开销大；batch_size 与峰值显存的关系。

4. **训练与评估的用时分析**：各阶段 wall-clock 耗时、每数据集/每采样平均耗时、吞吐
   （token/s）、batch_size 对用时的加速比（如 batch 1→2）、长生成 vs 短生成的时间放大倍数。

5. **训练数据构成分析**：数据集来源（Skywork/DAPO/AIME）、训练集与评估集划分、每条样本
   prompt 模板、`max_prompt_length` / `max_response_length` / `MAX_VAL_RESP_LENGTH` 等长度
   配置、**纯 on-policy 数据量**（每相位 `m_refresh × n_rollout` × 相位数，ring buffer 容量
   `refresh_size`）、only_stu 教师 Δ 的 top_k 支撑（= `cache.top_k`，无预计算缓存）。

6. **其他必要信息**：参考原始论文的机制速查、已知边界与未实现项（如 vLLM weight transfer
   逃生舱仅 tp=1）、复现步骤（命令 + 配置）、与本文件其它节（架构/算法约束/配置约定）的交叉引用。

> 写作时遵循「代码注释、文档、提交信息均用中文」的全局要求。文档随代码演进持续维护，
> 不要让它过期（当前代码的真实状态为准，不沿用旧描述）。

## 语言

代码注释、文档、提交信息均用中文（与现有代码风格一致）。
