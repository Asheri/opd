# 全栈 OPD 上云运维手册（RUNBOOK · 2×RTX PRO 6000 单机双卡）

> 适用：**2×RTX PRO 6000**（96GB×2 + NVLink，CUDA 12.8，驱动 R570+）+ **完整全栈**（DEPLOY.md 统一环境）。
> 本手册覆盖「代码上云那一刻之后」的**全部流程**：环境验收 → 数据 → Stage 0/1/2 → 监控 → 收敛导出 → 运维回滚。
> 配套：`DEPLOY.md`（环境安装）、`OPTIMIZATION_PLAN_2xRTXPRO6000.md`（性能方案）、`main/`（v2 算法内核）。
> ⚠️ **README 前置**：本手册假定你在本地已把 `main/` 与文档推到 GitHub `Asheri/opd`；三个上游 clone
>   （`async-opd/` / `Direct-OPD/` / `Lightning-OPD/`）是**独立 git repo**，上云后需单独 clone（见 Phase 1）。

---

## 0. 全流程速览（一张图）

```
[Phase 0 环境验收]  →  [Phase 1 代码+数据]  →  [Stage 0 小模型 RL]
        │                    │                      │  产出 teacher π_RL + reference π_ref
        ▼                    ▼                      ▼
[Stage 2 异步 Direct-OPD 训练]  ←  [Stage 1 Lightning 离线缓存 Δ_T builds]
   learner TP=2 / rollout vLLM TP=2      vLLM phase2 预计算 → 稀疏 top-K cache
        │
        ▼
[监控·收敛判定·导出学生模型]  →  [运维：checkpoint / 回滚 / 备选环境]
```

**三条铁律贯穿全程**（算法内核，改不得）：
1. **π_old 加权 PG + PPO clip**（`losses.py:pg_loss`）+ **k3 KL**（`low_var_kl`）；
2. **staleness 双截断**（入队/消费双侧）+ **causal mask**；
3. **teacher 一致性**：SFT 教师与 OPD 教师必须**同一模型**（否则不可约梯度偏置）。

**两条硬约束**（PRO6000 特判，见 `OPTIMIZATION_PLAN_2xRTXPRO6000.md` §0.1）：
- 🔴 **learner（被训练 student）上限 ≈ 13B**（8-bit Adam + 梯度检查点；7B 宽松）。优化器内存是天花板，TP/fp8 只省权重不省优化器。
- 🔴 **colocated 不能同卡同时驻留**：vLLM(rollout) 与 learner 需 **CPU offload 换入换出**（fused-hybrid），而非同时占满。

---

## Phase 0 · 环境验收（服务器就绪确认）

**目标**：确认硬件拓扑、驱动、统一环境、冒烟一次性通过，再进入数据与训练。

### 0.1 硬件拓扑与 NVLink（**先做，否则后面全白做**）
```bash
nvidia-smi                       # 确认 2×RTX PRO 6000、驱动 R570+、CUDA 12.8
nvidia-smi topo -m               # 🔴 必须看到 2 卡间为 NV#（NVLink）。若 PIX/NODE = 走 PCIe → 性能悬崖
nccl-tests                       # 实测 NVLink 带宽（可能 < 数据中心的 1.8TB/s，以实测为准）
df -h /shared                    # 预留数百 GB（模型权重 + 教师缓存）
```
> 🔴 **NVLink 桥是可选配件**，未装则静默退回 PCIe 4.0。拓扑确认不过关 → 先装桥再继续，否则 TP 方案失效。

### 0.2 统一环境安装
```bash
mkdir -p /shared/opd_cache
conda create -n opd-fullstack python=3.12 -y && conda activate opd-fullstack
# 若 pip 默认 torch wheel 非 cu12.8：
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-unified.txt
pip install -e ./async-opd
```

### 0.3 冒烟测试（全部通过才算环境就绪）
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"   # True, 2
python -c "import torch; print(torch.backends.cuda.matmul.allow_tf32)"                    # 确认 TF32 开关
python -m pytest tests/ -q        # 在 main/ 下；58 个全绿（CPU 即可，验证内核没被云环境破坏）
python -c "import vllm, megatron.core, ray; print('L3/L2/L5 依赖 OK')"
```
**验收通过标准**：`nvidia-smi topo -m` 显示 NVLink、`torch.cuda` 双卡 OK、58 测试全绿、vllm/megatron/ray 可导入。

---

## Phase 1 · 代码与数据落位

### 1.1 拉取代码（三份）
```bash
# (a) 自研 main/ + 文档（已在 GitHub Asheri/opd）
git clone https://github.com/Asheri/opd.git /workspace/opd

# (b) 三个上游 clone（各自独立 repo，上云后单独拉）
cd /workspace
git clone <async-opd 的 remote>   async-opd
git clone <Direct-OPD 的 remote>  Direct-OPD
git clone <Lightning-OPD 的 remote> Lightning-OPD
```
> ⚠️ 三个上游是**独立 git repo**，其 remote 在上云时需从本地 `.git/config` 抄过来（或按上游官方地址）。

### 1.2 数据与权重
- **数据集**：Stage 0 的小模型 RL 需要一套 prompt 集（替换 toy 的 `n_prompts`）。真实数据放到 `/shared/opd_cache/`。
- **模型权重**：student（7B）与 teacher（7B，同模型！）从 HF 下载到 `/shared/models/`。**教师一致性**要求二者同架构/词表/`d_model`。
- **缓存产物**：Stage 1 生成的 Δ_T 缓存落 `/shared/opd_cache/cache.pt`（mmap 或单文件）。

---

## Stage 0 · 小模型 RL（产出教师对）

**目标**：用小模型 RL 产出 post-RL teacher `π_T^RL` 与 pre-RL reference `π_T^ref`。二者是 Stage 1 缓存 Δ_T 的来源。

```bash
# async-opd 的 GRPO/PPO pipeline（DEPLOY.md Stage 0）
python -m opd.cli.train --config configs/examples/opd_gsm8k_0.5b_4gpu.yaml --overwrite
```
**关键产出与校验**：
- 保存 `π_T^RL`（RL 后）与 `π_T^ref`（RL 前/初始）两个 checkpoint。
- **教师一致性校验** 在 `TensorTeacherCache.build` 里做（`TeacherConsistencyError`）——上云后首次 build 就会碰到，别慌，是护栏在档。
- **健康信号基准**：用 `main/` demo 先跑一遍，确认 `E[Δ_T] −0.18 → +0.72` 待会儿在真实缓存上能复现。

---

## Stage 1 · Lightning 离线缓存构建（消除常驻教师）

**目标**：用 vLLM 对离线固定 rollout 预计算 `logπ_T^RL − logπ_T^ref`，写稀疏 top-K 缓存。**此后训练不再启动任何教师前向**。

```bash
# vLLM phase2 logprob 预计算（Lightning-OPD/data_curation 风格）
# 真实词表 V=32k~150k：dense (N,T,V) 存不下 → 必须稀疏 top-K（L4）
python <lightning-phase2-script> \
    --model <teacher_path> --data <dataset> \
    --top_k 64 --out /shared/opd_cache/cache_topk.pt
```

**缓存形态（L4 稀疏 top-K）**：每位置存 `(token_id, logp_rl, logp_ref)`，K=64~256，落 mmap 跨进程共享，体积 ↓约 1000×。对接 `cache.py` 的 `delta_for_student_topk`（searchsorted 预排序已就绪）。

**L1 暖缓存（可选，缓解曝光偏差）**：`stage1.warmup_M` / `warmup_source`（`student_init` | `teacher_perturbed` | `mix`）额外采样 M 条响应拼「胖 D」。`warmup_source` 与 KL 锚点必须同源。

**验收**：缓存 build 通过、稀疏匹配（`test_searchsorted_match_equals_full_compare`）在真实 top-K 上正确、`warmup_*` 计数符合 `N×(1+M)` / `N×(1+2M)`。

---

## Stage 2 · 异步 Direct-OPD 训练

**目标**：`DistAsyncScheduler` 四阶段异步 + staleness 双截断 + π_old 加权 PG + k3 KL，跑在 2 卡上。

### 2.1 启动器（torchrun / ray）
```bash
# 分布式骨架入口（scheduler.py:launch_distributed_scheduler）
torchrun --nproc_per_node=2 python -m fullstack_opd_v2 \
    --device cuda:0 --set stage2.distributed=true --set stage2.n_gpus=2 \
    --set stage2.rollout_engine=vllm --set stage2.rollout_model=<teacher_path> \
    --set stage2.rollout_tp_size=2 --set stage2.tp_size=1 \
    --set stage2.cache_mode=topk --set stage2.top_k_teacher=64 \
    --set stage2.top_k_student=64 --set stage2.ref_topk=64 \
    --set stage1.warmup_source=mix --set stage1.warmup_M=4 \
    --set stage2.offload_to_cpu=true
```
> `tp_size=1` 是**有意的**：`DistAsyncScheduler` 把 rank1 用作 rollout worker，learner 若切 TP=2 会与并发 rollout 模型**死锁**（见 `scheduler.py` 的护栏 RuntimeError）。Learner 真正 TP=2 需走「colocated 交替相位」调度（尚未实现，见 OPTIMIZATION_PLAN §L2/L3）。

### 2.2 两卡分工（PRO6000 特判）
| 卡 | 角色 | 说明 |
|---|---|---|
| rank0 | **learner**（训练） | 7B 训练，bf16 + 8-bit Adam + 梯度检查点，吃 96GB 绰绰有余 |
| rank1..W | **rollout worker**（vLLM TP=2） | 7B 推理，FP8 量化（KV 减半） |

**colocated 换入换出（L6）**：`offload_to_cpu=true` 时，rollout 阶段把 learner 优化器/权重换出到 CPU，rollout 结束 reload——**不是**同卡同时占满。若 OOM，优先调 `offload_to_cpu` 与 `prefetch`。

### 2.3 训练期实时监控
```bash
# WandB / TensorBoard 异步写（产出三张图）
# 1) E[Δ_T] 应单调上升（修复后 −0.18 → +0.72）
# 2) staleness age 直方图（age>0 证明异步在消费陈旧样本；age 越平稳越好）
# 3) loss / pg_loss / kl_loss / adv_mean
```
**健康信号检查表**：
- [ ] `E[Δ_T]` 单调上升
- [ ] `staleness age > 0`（双截断在工作）
- [ ] 训练循环**无任何 teacher 前向**（缓存使命达成）
- [ ] `waste_ratio` 拆解：`rollouts = trained + 陈旧(put+consume) + 队满 + 停机尾`（M5 口径）
- [ ] GPU 利用率接近峰值 2×96GB 合理占用，无 OOM

---

## 收敛判定与模型导出

**收敛信号**（`main/` v2 已把 `E[Δ_T]` 单调上升定为健康基准）：
- `E[Δ_T]` 不再上升（平台期）→ 可考虑停止；
- KL 正则稳定、`adv_mean` 合理、`staleness age` 分布不失控。

**导出**：
```bash
# student 状态 → 生产格式（HF safetensors）
python -c "from fullstack_opd_v2.model import CausalToyLM; m=CausalToyLM(...); m.load_state_dict(<ckpt>); m.save_pretrained('/shared/models/student_final')"
```
**评估**：在 held-out 集上对照 pre-RL student 测下游指标（ppl / 下游任务），量化 OPD 收益。

---

## 运维 · 回滚与备选

### 回滚点
| 故障 | 回滚动作 |
|---|---|
| 训练发散（loss NaN / E[Δ_T] 崩溃） | 退到上一个 checkpoint 重训；调 `lr` / `kl_reg_coef` / `staleness_threshold` |
| 环境依赖冲突 | 退到附录 A 多环境方案（async-opd / direct-opd / lightning-opd 分 env） |
| NVLink 未装 / 性能悬崖 | 改 `OPTIMIZATION_PLAN_8x4090.md` 的 FSDP 路线（不依赖 NVLink） |
| 稀疏缓存数值漂移 | 用 `test_perf_equivalence.py` 回归；必要时回退 dense 缓存（小词表） |

### 备选环境（DEPLOY.md 附录 A）
必须对照原版 verl / slime 时，用三 env 分环境方案；统一环境是默认推荐。

### 训练报错档案（全局规则）
上云训练**每一个报错**都必须追加到本地 `C:\Users\12062\OneDrive\Desktop\items\training-errors.md`
（日期/阶段/症状/根因/修复/验证）。这是全项目硬性规则，云上训练尤其要记。

---

## 附 · 一页速查（命令清单）

| 阶段 | 命令 | 验证 |
|---|---|---|
| 拓扑 | `nvidia-smi topo -m` | 2 卡间 `NV#` |
| 环境 | `pip install -r requirements-unified.txt` | 58 测试全绿 |
| 冒烟 | `python -c "import torch,vllm,megatron.core,ray"` | 无 ImportError |
| Stage0 | `python -m opd.cli.train --config ...` | 产出 `π_RL`+`π_ref` |
| Stage1 | vLLM phase2 + `--top_k 64` | 稀疏 top-K 缓存 |
| Stage2 | `torchrun --nproc_per_node=2 python -m fullstack_opd_v2 ...` | `E[Δ_T]↑` + `age>0` |
| 导出 | `save_pretrained('/shared/models/student_final')` | 下游评估 |