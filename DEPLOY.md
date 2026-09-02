# 全栈 OPD 部署指南（GPU 服务 · 单一环境）

把三篇论文的代码叠加成一个流水线：
**小模型 RL → 离线缓存教师对 log 比（Lightning-OPD）→ AsyncOPD 调度器上的 Direct-OPD 训练**，
同时摆脱「常驻教师 / 同步等待 / 迁移终态」三重限制。

| 组件 | 仓库 | 框架 | 在全栈中的角色 | 统一环境下的形态 |
|---|---|---|---|---|
| 调度器 | `async-opd/` | opd + vllm + megatron-core | 流式异步调度、staleness 截断 | **完整安装**（基座） |
| 迁移对象 | `Direct-OPD/verl/` | verl | Δ_T = logπ_T^RL − logπ_T^ref 隐式奖励 | **vendor 两个纯 torch 函数** |
| 教师缓存 | `Lightning-OPD/slime/` | slime + sglang | 离线预计算教师 log 比 | **只保留缓存产物**（vllm 生成） |
| 全栈 demo | `main/` | 纯 torch（ToyModel 占位） | 三者串联的最小示例 | 统一环境天然覆盖 |

---

## 1. 直接回答：PyTorch / TRL 版本

- **PyTorch**：**`torch==2.9.1` + CUDA 12.8**（AsyncOPD freeze 锚点，与 megatron-core 0.16.1 /
  flash-attn 2.8.3 / vllm 0.16.0 匹配）。统一环境下不再有第二个 torch 版本——
  verl sglang 路径的 `torch==2.8.0` 钉死随 sglang 一起被裁掉。

- **TRL**：**统一环境不需要 TRL**。
  `trl<=0.9.6` 只是 verl 的 `TRL_REQUIRES` extra（走 HuggingFace TRL 路径时才用）；
  Direct-OPD 的核心机制是 verl 内部的两个纯 torch 函数
  （`dp_actor.py::_compute_delta_opd_rm_scores`、`core_algos.py::compute_token_reward_direct_advantage`），
  已 vendor 为 `main/fullstack_opd/direct_opd.py`，不依赖 TRL。
  仅在回退到「完整 verl 框架」方案时才需要 `trl<=0.9.6`。

---

## 2. 硬件需求

- **GPU**：真实训练 ≥ 8× H100/H800 80GB（Lightning-OPD 论文：8B 规模约 30 GPU·h，单节点 8×H100）；
  `main/` demo 不需要 GPU（CPU 可跑）。
- **CUDA 运行时**：**12.8**（torch 2.9.1 cu128 wheel）。
- **NVIDIA 驱动**：≥ **R570**（支持 CUDA 12.8）。
- **内存/网络**：多卡 NVLink/InfiniBand；Ray 负责跨进程调度。
- **磁盘**：模型权重 + 离线教师缓存（log 比），预留数百 GB。

---

## 3. 如何装进同一个环境（原冲突 → 架构裁剪）

原先的硬冲突，以及统一方案对每一处冲突的处理：

| 依赖 | 原冲突 | 统一解法 |
|---|---|---|
| `numpy` | AsyncOPD `==2.2.6` × verl `<2.0.0` | **不装 verl**；Direct-OPD 核心 vendor 为纯 torch 模块 → numpy 2.2.6 统一 |
| `vllm` | AsyncOPD `==0.16.0` × verl `<=0.11.0` | 推理统一 vLLM 0.16.0：AsyncOPD rollout + Lightning 离线 logprob 预计算都用它（Lightning 的 curation 阶段本来就用 vllm） |
| `torch` | verl sglang 路径钉 `2.8.0` | sglang 只在 slime 训练器里用；训练器由 AsyncOPD 承担 → **sglang 整体裁掉** |
| `sglang` | verl `==0.5.2` × slime router | 同上，裁掉 |
| `megatron` | core 0.16.1 × bridge@dev_rl | 训练后端统一 **megatron-core 0.16.1**；slime 训练器（bridge/modelopt/ring_flash_attn）裁掉 |
| `llamafactory` | Lightning 的 SFT 环境 | SFT 用 **async-opd 自带 pipeline**（其 pyproject 明确含 SFT） |

一句话：**全栈架构本身只需要一个训练器 + 一个推理引擎 + 一份离线缓存**。
三套框架各自完整安装才会冲突；按全栈的职责切分后，每个组件只保留自己那一片，
依赖图自然收敛到 AsyncOPD 的 freeze（它本身就是验证过可共存的最新组合）。

```
统一环境（Python 3.12 / torch 2.9.1 / CUDA 12.8）
├─ async-opd [完整安装]        → Stage 0 RL + Stage 2 调度器/训练器（含 SFT）
├─ direct_opd.py [vendor]      → Stage 2 的 Δ_T 奖励（纯 torch，demo 已含）
└─ Lightning 离线缓存 [vllm]    → Stage 1 的 phase2 logprob 预计算
```

---

## 4. 安装步骤（单环境）

```bash
# (0) 共享数据/缓存目录
mkdir -p /shared/opd_cache

# (1) 创建统一环境（Python 3.12）
conda create -n opd-fullstack python=3.12 -y
conda activate opd-fullstack

# (2) 一次性装齐（若 pip 默认 torch wheel 非 cu12.8：
#     pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128）
pip install -r requirements-unified.txt

# (3) 安装 async-opd 本体（调度器 + 训练器 + SFT）
pip install -e ./async-opd

# (4) Direct-OPD 核心已在 main/fullstack_opd/direct_opd.py（纯 torch，免安装）

# (5) 跑通全栈 demo（CPU 即可，验证流水线逻辑）
cd main && python run_fullstack.py
```

端到端串接（真实规模，对应 demo 的三个 stage）：

1. **Stage 0 — 小模型 RL**：统一环境内用 async-opd 的 GRPO/PPO pipeline
   产出 post-RL teacher π_T^RL 与 pre-RL reference π_T^ref。
2. **Stage 1 — Lightning 离线缓存**：统一环境内用 vLLM 跑
   `Lightning-OPD/data_curation/prepare_lightning_opd.py` 风格的 phase2 预计算，
   把 `logπ_T^RL` / `logπ_T^ref` 写到缓存（parquet / pt）。**此后训练不再启动教师**。
   关键前提「教师一致性」：SFT 教师与 OPD 教师必须同一模型，否则引入不可约梯度偏置。
3. **Stage 2 — 异步 Direct-OPD 训练**：async-opd 调度器流式产出学生 rollout，
   从缓存取 Δ_T 作隐式奖励，learner 端用当前学生重算 ratio 做 PPO-clip 鲁棒代理，
   按 staleness 阈值截断过旧样本。

---

## 5. 附录 A：回退方案（保留完整三框架，分环境）

若必须使用完整 verl / slime（例如要对照原版实验），则回到多环境方案：

```
conda env: async-opd       <- requirements.txt           (调度层基座)
conda env: direct-opd      <- requirements-direct-opd.txt (verl，numpy<2，trl<=0.9.6 可选)
conda env: lightning-opd   <- requirements-lightning-opd.txt (slime 原版训练器)
共享目录: /shared/opd_cache
```

三个 split 文件已标注「备选」，内容与 2026-08-05 版本一致。

## 6. 附录 B：demo 审阅修复记录（2026-08-06）

对 `main/` demo 做了一次完整审阅，修复 11 处问题（详见 git diff / 记忆）：
P0：Transformer 缺 causal mask（双向注意力偷看未来 token）；
P0：`__init__.py` 只声明 `__all__` 未 import（ImportError）；
P1：PG 损失两种错误形式（等权 mean ≠ Direct-OPD 目标；token 级标量 adv 一阶梯度恒为 0）
   → 修为按 π_old 加权的逐 vocab 重要性采样（ratio=1 时精确等于 −E_{π_cur}[Δ_T]）；
P1：`low_var_kl` 等权 mean 不是 KL → 修为按 π_student 加权的 k3 期望（恒等真 KL）；
P1：`StalenessQueue` 队列本体闲置 → 兼任 scored 队列，入队/消费双侧截断；
P2：梯度裁剪、全局播种、dropout=0、teacher 一致性校验加强、weights_only 加载、
   torch 导入位置、P==0 负索引、REINFORCE baseline。
修复后回归：30 步 exit 0，E[Δ_T] −0.18 → **+0.72** 单调上升，staleness age=5。

---

## 7. 文件清单

- `requirements-unified.txt` — **【推荐】单一环境**（torch 2.9.1 / CUDA 12.8 / vllm 0.16.0）
- `requirements.txt` — 备选：调度层基座（分环境方案用）
- `requirements-direct-opd.txt` — 备选：完整 verl（numpy<2，trl<=0.9.6 可选）
- `requirements-lightning-opd.txt` — 备选：完整 slime 训练器
- `requirements-demo.txt` — `main/` demo 最小环境（仅 torch）
- `main/` — 全栈叠加 demo（CPU 可跑，已审阅修复，回归 exit 0）

> ⚠️ **2026-08-31 P-OPD 标注**：本指南的 Stage 1 离线缓存（Lightning-OPD 预计算教师 Δ）
> 已删除——训练 = 纯 on-policy 交替相位（only_stu 实时打分 + vLLM `rollout_weight_sync=off`
> 逃生舱仅 tp=1）。主配置 `main/configs/qwen3_r1_onpolicy.yaml`，见 AGENTS.md / TECHNICAL_REPORT §4.6.5。

