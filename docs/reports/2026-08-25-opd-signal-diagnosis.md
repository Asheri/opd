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

---

## C3 教师模板一致性审计（2026-08-25，服务器实证）

### 背景
D3 判据 FAIL（teacher top-K 支撑 Δ 均值 -1.159 < -1.0）→ 按任务回 C3 审计教师对方向/模板/数据源。

### 审计 1：教师词表 + 模板（`audit_teacher_templates.py`，8 + 50 条样本）
| 项 | 结果 |
|---|---|
| 词表 | student=151643 / teacher_rl(JustRL)=151643 / teacher_ref(R1-Distill)=151643 —— **三者一致** |
| 学生 response token 越界 | rl=0 / ref=0（共 4096）—— **无跨词表错位** |
| Δ_student（共用学生模板） | 50 条均值 **-0.127** |
| Δ_teacher（教师各自模板） | 50 条均值 **-0.121** |
| 模板影响 | Δ 差 +0.006 —— **教师模板不是方向根因** |

**关键**：实际 response token 上的 Δ 均值 ≈ **-0.12（轻微负）**，远非 D3 的 -1.159。D3 的深度负是 **teacher top-K=256 支撑统计的固有偏置**（JustRL 分布较平坦、R1-Distill 较集中 → 在 rl 的 top-K 上 ref 平均更高），不代表训练实际信号深度负。

### 审计 2：cache 与训练数据同源性（`audit_delta_support.py` + 手工统计）
| 项 | 结果 |
|---|---|
| cache.lengths vs jsonl 编码长度 | **413/500 一致（82.6%）**，87 条差异 |
| 差异分布 | 20 条 cache 更长（max +1593，如 i=28: jsonl=455 vs cache=2048）；67 条 jsonl 更长（max -72） |
| teacher top-K 命中 jsonl token | i=0/8/12=100%、i=28=98.7% —— cache 的 Δ_T 位置与 jsonl response **开头对齐** |
| 结论 | cache 与 jsonl **基本同源但 17.4% 行长度/尾部错位**（cache build 时 response 与当前 jsonl 部分行不同版本） |

**含义**：训练用当前 jsonl response 前向 s_cur，cache 用构建时 response 算 Δ_T——17.4% 行的 pad 尾部位置存在**信号错位**（cache 在真实 token 上有 Δ_T、student 在 pad 上预测）。这是 reward 无收敛的**部分根因**（非全部：大部分行同源）。

### C3 审计结论（修正 D3 FAIL 的解读）
1. ✅ 教师词表一致、模板一致性无问题 → **不指向"教师对模板错位"**
2. ✅ 实际 Δ ≈ -0.12（轻微负）→ **教师对方向并非深度负**；D3 的 -1.159 是 teacher top-K 支撑指标的口径偏置
3. ⚠️ **cache 与训练 jsonl 17.4% 行错位** → 需确认 cache build 数据源；若重建 cache 须与训练用**同一 jsonl/response 版本**

### 建议下一步（需用户决策）
- **A（推荐）**：重建 cache（用当前 skywork_50k.jsonl 前 500 条，与训练完全同源），同时 D3 判据改用「实际 token 或 student 支撑 Δ」口径复评；通过后进 D1
- **B**：接受「实际 Δ -0.12 非深负」的修正结论，直接用现有 cache 进 D1（但需接受 17.4% 行错位噪声）
- 产物：`audit_teacher_templates.py` / `audit_delta_support.py`；数据 `d3_report_20260825.json`、`c3v2_align_20260825.log`
