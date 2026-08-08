# GPU 部署路径性能优化设计

日期：2026-08-08
范围：`main/fullstack_opd_v2/`（v1 包不动）
基线 commit：`7a78a91`

## 1. 目标与非目标

**目标**：提升 GPU 部署路径的真实吞吐与显存占用。

**非目标**：
- 本地 CPU demo / 测试迭代速度（只当副产品，不作验收指标）
- 背压节流（rollout 浪费本轮只做观测，不改行为）
- L2 周期刷新、Megatron colocated 交替相位
- `fullstack_opd/`（v1）的任何改动

**贯穿约束**：`CLAUDE.md` 与 `losses.py` 顶部记录的「不可回退算法约束」一条都不动。
凡触及损失/缓存内核的改动，必须以数值等价测试锁死。

## 2. 基线实测

环境：`Python311/python.exe`（torch 2.11.0+cpu），默认配置，CPU。

| 阶段 | 耗时 | 说明 |
|------|------|------|
| stage0_rl | 5.02 s | 其中 3.5 s 是 `torch.optim.Adam` 首次构造触发的 `torch._dynamo` + `sympy` import；纯计算 1.3 s |
| stage1_cache | 0.02 s | 已充分优化 |
| stage2_train | 1.34 s | 47 ms/step；纯 `_train_step` 仅 18 ms/step，61% 为线程/队列开销 |
| **总墙钟** | **6.40 s** | |

测试：42 passed in 16.55 s。
健康信号基线：`E[Δ_T]` 末值 `+0.757`，末步 `age=5`，`version=30`。

## 3. 改动清单（按实测严重度排序）

| # | 改动 | 文件 | 依据 |
|---|------|------|------|
| P0-1 | vLLM 分布重建批量化 | `rollout_vllm.py` | 6700 万次 Python 单元素写入 ≈ 10 min/批 |
| P0-2 | `_LOG_ZERO` NaN 修复 | `rollout_vllm.py`(+`losses.py` 可选) | 实测 loss = `nan` |
| P1-1 | `searchsorted` + build 期预排序 | `cache.py`, `scheduler.py` | 2 GiB 中间张量 → 8 MiB |
| P1-2 | dense 缓存只留 `delta` | `cache.py` | 3×(N,T,V) → 1× |
| P2-1 | 异步仪表（只观测） | `scheduler.py`, `buffer.py` | 实测 77% rollout 浪费 |
| P2-2 | 去冗余计算 | `scheduler.py`, `losses.py`, `buffer.py` | 见 §7 |

## 4. P0-1 · vLLM 分布重建批量化

### 问题

`VLLMRolloutEngine.response_dists` 用三重 Python 循环逐元素写 `out[b, t, tok_id] = float(lp.logprob)`。
GPU 预设（B=32, T=512, K=4096）下是 6710 万次 `Tensor.__setitem__`，
并构造 7.81 GiB 的 `(B,T,V)` fp32 **CPU** 张量后再 `.to(device)`。

实测（B=8,T=64,K=512,V=128000 后线性外推至 B=32,T=512,K=4096）：

| 方案 | 实测 | 外推耗时 | 峰值显存 |
|------|------|----------|----------|
| 现状：逐元素 `setitem` | 2.340 s | 599 s（10.0 min） | 7.81 GiB |
| 展平 + 一次 `scatter` | 0.164 s | 41.9 s（**14.3×**） | 7.81 GiB |
| 直接返回稀疏 `(ids, logps)` | 0.083 s | 21.3 s（**28.1×**） | 0.50 GiB（**16×**） |

展平 + scatter 方案已验证与现状 `torch.equal`。

### 方案

`response_dists()` 重写为展平 + 一次 `scatter`（真实落地，14.3×，数值 `torch.equal`）：

1. `response_dists()` 内部改为展平 + 一次 `scatter`，去掉逐元素 `setitem`。
2. `prompts`/`responses` 的逐样本 `.tolist()` 合并为一次 `torch.cat(...).cpu().tolist()`
   （`_prompt_seq`）：每批次一次 GPU→CPU 同步，而非 2B 次。

新增 `response_dists_topk(prompts, responses) -> (ids, logps)`（形状 `(B,T,K)`）为
**预留接口，本轮未接进训练循环（未启用）**。原因见 §10 已知边界：稀疏 `s_old`
会改变 `pg_loss` 语义（探针实测与 dense 差 77%），需单独立项做成有界近似。本接口
仅实现 + mock 测试锁定，供后续 L2 或稀疏损失项目复用。

## 5. P0-2 · `_LOG_ZERO` NaN 修复

### 根因

`_LOG_ZERO = -1e4`（`rollout_vllm.py:45`）作为支撑外填充值进入 `pg_loss`：

```
ratio = (s_cur - s_old).exp() = exp(-12 - (-1e4)) = exp(9988) = inf
inf × (delta = 0) = nan
```

稀疏模式下 student 支撑外 `delta` 恰为 0，两个近似叠加**必然**触发。
实测：`pg_loss` 返回 `nan`，`ratio.max() = inf`。

bf16 下 `_LOG_ZERO = -100` 就已达 `1.66e38`（临近溢出），故这不只是单个魔数的问题。

### 方案

**主修复**：`_LOG_ZERO` 从 `-1e4` 改为 `-30`。
实测 `-30` 下 `ratio.max() = 1.823e11`，无 `inf`；`exp(-30) ≈ 9.4e-14` 作为 log 0 足够；
bf16 下 `exp(18) = 6.55e7` 安全。此单项即根治 NaN（实测场景 3 确认）。

**可选纵深防御**：`pg_loss` 新增 `log_ratio_max: float | None = None` 参数。
默认 `None` → 逐位走原路径（**默认行为与今日完全一致**）；
显式传 `80.0` → 对 log-ratio 施加 `clamp(max=80)`。

`clamp` 上界 80 的依据：`exp(80) ≈ 5.5e34`，fp32/bf16 均不溢出；
而真实 π_cur/π_old 比值远低于此（staleness 仅数个版本，PPO clip 又把有效 ratio 压在 `1±ε`）。

等价性实测：
- 正常 dense 输入下 `clamp=80` 与 `clamp=None` **逐位相等**（`torch.equal`，绝对差 0）
- 支撑外场景下 `clamp=80` 的结果**精确等于**「仅在 π_old 支撑内求和」的数学真值
  （`allclose atol=1e-6`）——即 clamp 恢复了「π_old=0 处贡献为 0」这一应有语义

### 明确不改：`ref_tail_logp = -1e2`

`scheduler.py:81` 的 `ref_tail_logp` 进入的是 `low_var_kl_support` 的 `k3(x) = exp(x) − x − 1`。
对极负 `x`，`exp(x) → 0`，`k3 ≈ −x − 1` 是**线性**而非指数，**不溢出**。

实测：`tail=-1e2` 时 `KL = 49.85`，`× kl_coef=0.05 → 2.49`（pg 量级约 0.3）——
惩罚强但有界，正是 `CLAUDE.md` 所述「给出强漂移惩罚」的设计意图。bf16 下同样有限。

**结论：已验证安全，本次不动。** 与 `_LOG_ZERO` 属不同性质，勿混为一谈。

## 6. P1 · 显存

### P1-1 `searchsorted` + build 期预排序

`delta_for_student_topk` 与 `_ref_logp_at_student_topk` 均构造 `(B,T,Ks,Kt)` 全对比较矩阵。
GPU 预设下的峰值中间张量：

| 配置 | 元素数 | fp32 | bf16 |
|------|--------|------|------|
| B=32, T=128, K=256 | 2.68e8 | 1.00 GiB | 0.50 GiB |
| B=32, T=512, K=256 | 1.07e9 | 4.00 GiB | 2.00 GiB |
| B=32, T=1024, K=256 | 2.15e9 | 8.00 GiB | 4.00 GiB |

两处各一份，T 增长即 OOM，无法靠调参绕开。

**方案**：`cache.build` 的 topk 分支额外存 `ids_sorted` / `delta_k_sorted`（按 token id 升序）。
查表改为二分：

```python
pos   = torch.searchsorted(sorted_ids, student_ids.contiguous()).clamp(max=Kt - 1)
found = sorted_ids.gather(-1, pos) == student_ids
value = sorted_delta.gather(-1, pos) * found
```

峰值中间张量 `(B,T,Ks,Kt) → (B,T,Ks)`，降 `Kt` 倍（K=256 时 2 GiB → 8 MiB）。

**等价性已实测**：与全对比较 `torch.allclose` 最大差 0.0，`has` 掩码 `torch.equal`，
含「student 支撑内有重复 id」边界同样等价。

**持久化兼容**：`save` 增写 sorted 字段；`load` 检测字段缺失则现场排序一次
（旧 `.pt` 仍可加载，仅首次略慢）。`test_save_load_roundtrip_topk` 相应扩展。

### P1-2 dense 缓存只留 `delta`

现状存 `rl` / `ref` / `delta` 三份 `(N,T,V)`，而训练路径只读 `delta`。

| 配置 | 单份 | 现状 3 份 | 改后 1 份 |
|------|------|-----------|-----------|
| N=512, T=512, V=128000 | 125 GiB | **375 GiB** | **125 GiB** |

**方案**：`rl` / `ref` 改为 `build` 内局部变量，算出 `delta` 后即释放；`save` 只写 `delta`。

**前提已核查**：`grep` 确认 `cache.rl` / `cache.ref` 仅在 `cache.py` 内部赋值与持久化，
**无任何外部读取点**（`rl_k` / `ref_k` 是稀疏模式独立字段，保留）。
`test_dense_build_delta_shape_and_value` 断言 `cache.delta`，不受影响。

## 7. P2 · 仪表与冗余

### P2-1 异步仪表（只观测，不改行为）

实测：跑 30 步训练，`RolloutCollector` 发起 **133 次**前向，消费侧丢弃 **81** 个，
入队侧 `put` 拦截 **0** 个 → **浪费率 77%**。`age` 稳定在 5、阈值为 4，
说明生产/消费速率比约 4.4:1 —— 是速率失衡，不是阈值设小。

GPU 上这等于 vLLM 推理算力烧掉 3/4。本轮**只量化，不改行为**（用户决定）。

`AsyncBatchedScheduler.run()` 返回值附汇总：

- `rollout_forwards` — rollout 实际前向次数
- `dropped_at_put` / `dropped_at_consume` — 两道截断各自丢弃数
- `waste_ratio` — `(rollout_forwards − trained) / rollout_forwards`
- `age_histogram` — age 分布（现仅末步 age 可见）
- 各线程累计空转时间 — 定位真实瓶颈级

`StalenessQueue` 增 `n_rejected` 计数。均为只增字段，不改控制流。

### P2-2 去冗余计算

| 项 | 现状 | 改法 | 实测收益 |
|----|------|------|----------|
| `s_old.exp()` | `pg_loss` 内每步重算 `(B,T,V)` | `expected_reward` 复用 `p_old`（`p_dists` 参数） | 省掉 `expected_reward(s_old,...)` 内部那次 `s_old.exp()` |
| 全 1 mask | 仍走乘/求和/除 | 走 `mask=None` 快路径 | 已验证 `torch.equal` 完全相等 |
| `_publish` | 每步全克隆 state_dict | `WeightStore` 预分配缓冲 + `copy_` 原地覆盖 | 7B fp32 下省 28 GiB 分配/步 |

> **监控降频已移除**：原设计的「每 N 步采样」会破坏现有
> `test_scheduler_runs_all_steps_and_fields_finite` 的「每步 finite」断言，且收益
> （0.094 ms = 单步 18.35 ms 的 0.5%）不值得。每步监控保留。
> 另：`p_old` 缓存 + `expected_reward(p_dists)` 后，`s_old` 的 exp 从 2 次降到 1 次
> （`pg_loss` 与 `adv` 监控共享），已兑现该子项的去冗余。

说明：
- `p_old` 作为 `pg_loss` 的**可选**参数（不传则内部计算 `s_old.exp()`），签名向后兼容。
  验证依据：现有 `test_pg_loss_onpolicy_equals_neg_expected_delta` 等用三参数形式
  `pg_loss(s, s, delta)` 调用，新增可选参数后仍走内部计算路径，断言不变即证明兼容。
  另在 `test_perf_equivalence.py` 断言「传入 `p_old` 版 vs 内部计算版 `torch.equal`」。
- 全 1 mask 快路径等价性：`mask.sum() + 1e-8` 的分母在全 1 情形下不产生可测差异
  （实测绝对差 0.000e+00，`torch.equal` 为真）。
  ⚠️ 仅当 mask 确为全 1 时可走此快路径；调度器当前用 `torch.ones` 构造，成立。
  若未来引入真实 padding mask，此优化必须失效回落到 mask 分支。
- `_publish` 仅改 `WeightStore` 内部缓冲复用；`acquire_if_newer` 仍返回独立克隆，
  语义不变（用户明确选择不动 acquire 侧，避免所有权约定风险）。

## 8. 版本管理与回退

因触及 `losses.py` 算法内核，回退能力先于改动落地：

1. **基线 tag** — 动手前打 `perf-baseline-v0` 指向 `7a78a91`。
   单文件回退：`git checkout perf-baseline-v0 -- main/fullstack_opd_v2/losses.py`。
   查看内核全部改动：`git diff perf-baseline-v0 -- main/fullstack_opd_v2/losses.py`。
2. **一项一 commit** — 六项 = 六个 commit，各自带测试。`git revert <sha>` 可单撤任一项。
   顺序按依赖排：P0-2 先于 P0-1（P0-1 的稀疏返回喂给已加固的损失）。
3. **运行时开关** — `pg_loss(log_ratio_max=None)` 为默认，即默认逐位等于今日行为；
   出问题时显式传 `80.0` 开启加固，无需改代码即可二分定位。
4. **数值等价测试** — 新增 `tests/test_perf_equivalence.py`（9 个）：
   - `pg_loss(log_ratio_max=80)` vs `None` 在正常 dense 输入下 `torch.equal`
   - `log_ratio_max=80` 在支撑外场景恢复「仅支撑内求和」真值
   - `p_old` 传入版 vs 内部计算版 `torch.equal`
   - `expected_reward(p_dists)` vs 内部 `dists.exp()` `torch.equal`
   - 全 1 mask 快路径 vs `mask=None` `torch.equal`（`pg_loss` + `low_var_kl` 两条）
   - `searchsorted` 支撑匹配 vs 原全对比较 `torch.equal`（含重复 id 边界）
   - `_ref_logp_at_student_topk` 二分 vs O(K²) `torch.equal`
   - `response_dists_topk` mock 引擎形状/索引验证
5. **基线指标存档** — 见 §2（`E[Δ_T]=+0.757`、42 passed、6.40 s），作为改后比对依据。

## 9. 验收标准

**正确性（硬门槛）**：
- 现有 42 个测试全绿（现状 52 个，净 +10 全为追加，无一改动断言）
- **不得放宽或改写任何现有断言**。允许的唯一例外是**追加**新断言：
  `test_save_load_roundtrip_topk` 需增加 sorted 字段的 roundtrip 检查（§6.1）。
  若某项优化导致现有断言失败，视为该优化不正确 —— 回退优化，而非调整断言。
- `tests/test_perf_equivalence.py` 全绿（§8.4 九条等价性）
- `E[Δ_T]` 仍单调上升、末值与基线 `+0.757` 同量级
- `age > 0` 仍成立（异步确实在消费陈旧样本）
- 训练循环内无任何 teacher 前向

**性能（GPU 路径，按外推验证）**：
- vLLM 分布重建 ≥ 14×（dense 路径，已落地）
- 稀疏支撑匹配（searchsorted）峰值显存降 `Kt` 倍（已落地）
- dense 缓存显存 3× → 1×（已落地）
- vLLM 路径不再产生 `nan` loss（已落地）
- （`response_dists_topk` 的 ≥28× / 16× 显存收益**未启用** —— 稀疏 s_old 需单独立项，
  见 §10，不列入本轮验收）

**可观测性**：
- `run()` 返回值含 §7.1 全部字段，`waste_ratio` 可复现当前 77% 的量级

## 10. 已知边界

- 本设计不解决 rollout 77% 浪费的**根因**（需背压或速率匹配），只使其可测。
- 稀疏 top-K 不重归一化仍是有意近似（`CLAUDE.md` 既有约束），本次不触碰。
- **`response_dists_topk` 未接进训练循环**。稀疏 `s_old` 会让 `pg_loss` 变成
  「仅对 student top-K 支撑加权」的**有损近似**：探针实测（K=16/V=64 随机分布）
  与 dense 相对差 77%，误差正比于「支撑外 π_old 质量 × 支撑外 Δ 加权」，**无全局上界**，
  随策略/温度/词表形态变化。这解除了「损失与 dense 逐位等价」的承诺，须像
  `low_var_kl_support` 一样单独立项做成有界近似（支撑覆盖、方向性、适用词表形态论证）。
  故 `response_dists_topk` 本轮仅实现 + mock 测试锁定，不作 GPU 主接口。
- 真实 V=128k 下 `response_dists` 的 dense 路径仍需 7.81 GiB —— 这是已认知的边界，
  由「稀疏 s_old 未启用」直接导致；若需上线大词表，必须先完成稀疏损失立项。
- stage0 那 3.5 s 的 `torch._dynamo` import 开销属 CPU 冷启动问题，
  本轮不在范围内（用户选择只优化 GPU 路径）。
