# 2026-08-25：OPD 信号诊断 — D3 Δ_T 信号体检报告

> 状态：**D3 判据 FAIL —— 停止一切训练**，需回 C3（教师各自模板一致性）重新审计。
> 对应任务：`docs/plans/2026-08-25-opd-signal-diagnosis-execution.md` 第 2 步。
> 数据源：`/root/autodl-tmp/cache_skywork_chat.pt`（磁盘 mmap，500 条，chat 模板，K=256，V=151936，T=2048）。

## 1. 目的（H4 判定）

验证 chat cache 里的 Δ_T 信号是否有效（方向/量级/密度），排除「教师对在模板下分歧弱」
导致训练信号无效的可能。判据写死：
- **通过**：正 Δ token 占比 ≥ 15% 且 |均值| ≤ 1.0
- **不通过**：正比例 < 5%，或 **均值 < -1.0** → 停止一切训练，回 C3 审计教师模板
- **边界**：5%-15% 或 -1.0 ≤ 均值 < -0.5 → 记录风险仍进 D1

## 2. 执行

```bash
cd /root/opd/main
/root/miniconda3/bin/python -u scripts/inspect_delta_cache.py \
  --prefix /root/autodl-tmp/cache_skywork_chat.pt --out /root/autodl-tmp/d3_report.json
```

- 全量 500 条 × T=2048 × K=256 = **262,066,688 个支撑 token** 统计（纯离线 CPU，~20s）。
- 报告：`docs/reports/budget_aware_data/d3_report_20260825.json`。

## 3. 结果（teacher top-K=256 支撑）

| 指标 | 值 | 判据参考 |
|---|---|---|
| **均值** | **-1.159** | 需 \|均值\|≤1.0 → **✗ 超限** |
| 中位数 | -1.000 | — |
| 标准差 | 1.916 | — |
| P10 / P25 / P50 / P75 / P90 | -3.625 / -2.250 / -1.000 / 0.016 / 1.000 | — |
| **正 Δ 占比** | **25.12%** | ≥15% ✓ |
| Δ>0.5 占比 | 16.13% | — |
| Δ>1.0 占比 | 9.93% | — |
| clip(\|Δ\|>2.0) 占比 | **32.11%** | delta_clip=2.0 削顶比例很高 |

### 位置分解（按 response 有效长度前/中/后三段）

| 段 | token 数 | 正比例 | 均值 |
|---|---|---|---|
| early 0-25% | 65.5M | 22.87% | -1.144 |
| mid 25-75% | 131M | 25.80% | -1.169 |
| late 75-100% | 65.5M | 26.00% | -1.154 |

## 4. 判定：**FAIL**

`正Δ占比 25.12%（≥15%）但 均值 -1.159 < -1.0` → 按写死判据判定**不通过**。

## 5. H4 解读（不是"分歧弱"，而是"方向整体为负"）

- **分歧不小**：clip 32.11%（|Δ|>2 的 token 占近 1/3）+ std 1.92 → 教师对分歧**很大**，
  排除「模板下教师对几乎无差异」的解释。
- **方向整体为负**：75% token 的 Δ<0（P75=0.016），即 **teacher_rl 的 logp 在模板数据上
  系统性低于 teacher_ref**（均值 -1.16 ≈ 每 token rl 比 ref 低 ~1.16 logp）。位置三段一致
  （-1.14~-1.17），无位置特异。
- 含义：Δ = log π_rl − log π_ref 的期望为深度负——若按 OPD 最大化 E[Δ_T] 训练，
  学生会被推向「更接近 teacher_ref」的方向，且初始 reward 深负。**这不是"更新方向有效、
  只是遍历不足"能解释的**（H1/H2/H3 的前提被削弱）。
- 可能根因（需 C3 审计确认）：
  1. teacher_rl（JustRL-1.5B）在 chat 模板 + Skywork 数据上的 logp 系统性偏低——模板
     格式不匹配（如 rl 教师没套自己的原生模板，或生成了 thinking 前缀但 rl 未训练该格式）；
  2. 教师 rl/ref 角色/权重加载反了（rl 用成了 ref，ref 用成了 rl）；
  3. 教师模板或数据接缝（prompt 结尾 / thinking 前缀）与学生格式不一致，导致 rl 在该
     上下文上分布错位。

## 6. 门控动作（按任务硬约束执行）

- **D3 FAIL → 停止一切训练**：不启动 D1 80 步探针、不启动 D2、不启动 300 步正式训练。
- **下一步**：回 **C3（教师各自模板一致性）重新审计**——重点验证：
  ① teacher_rl/teacher_ref 各自原生 chat template 的 prompt 编码正确且与学生 Qwen3 模板
  对齐；② 教师对角色/权重路径加载无误；③ 抽样 decode 教师在模板数据上的 logp 合理性。

## 7. 产出

- 脚本：`main/scripts/inspect_delta_cache.py`（可复用，`--prefix` 指定任意 disk cache）
- 数据：`docs/reports/budget_aware_data/d3_report_20260825.json`
- D1 功能（eval_holdout/eval_every）已实现并单测通过（523 passed），**保留待 C3 通过后使用**
- 服务器/本地均已提交（服务器 `e6ba14b` / 本地 `d30e401`、`d246cb0`）
