# 7B 学生无法训练 —— 根因分析与解决方案

> 日期：2026-08-13 ｜ 状态：根因已确认（词表家族不匹配），待服务器恢复后验证选型并部署

## 一、症状（服务器 ms_7b 日志）

```
File ".../losses.py", line 58, in pg_loss
    unclipped = ratio * delta
RuntimeError: The size of tensor a (152064) must match the size of tensor b (151936)
```

- `ratio` 来自 student 前向（`s_cur - s_old` 相关），维度 = **152064**
- `delta` 来自教师对缓存 `Δ_T`，维度 = **151936**

## 二、根因：词表家族不匹配（OPD 硬约束被打破）

| 模型 | 家族 | vocab |
|------|------|-------|
| 7B 学生（服务器实际 `JustRL-R1-7B`，即 DeepSeek-R1-Distill-Qwen-7B） | **Qwen2.5** | **152064** |
| 教师对缓存 `cache_7b.pt` 的 Δ_T | **Qwen3** | **151936** |
| 4B/17B 学生（Qwen3-4B / Qwen3-1.7B） | Qwen3 | 151936 |
| 4B/17B 缓存 Δ_T（训练成功） | Qwen3 | 151936 |

**关键结论**：
- 4B/17B 成功 = 学生（Qwen3 151936）与教师对 Δ_T（Qwen3 151936）**同家族**，约束满足。
- 7B 失败 = 学生选了 **Qwen2.5 家族**（DeepSeek-R1-Distill-Qwen-7B = 基于 Qwen2.5-7B，152064），
  但教师对缓存是 **Qwen3 家族**（151936）→ 前向维度对不上。
- OPD 不可回退约束：「学生必须与教师同词表」（`TensorTeacherCache` 有
  `TeacherConsistencyError` 校验；7B 这例是训练期 student 前向直接维度崩，更早暴露）。

**论文能跑通的原因**：Direct-OPD 论文 Table 1 的 R1-Distill-7B 用的是
`R1-Distill-1.5B → JustRL-1.5B`（**全 Qwen2.5 家族**，vocab 统一 152064）——学生与教师对同家族。

## 三、解决方案（两条路，推荐 B）

### 方案 B（推荐）：7B 回到论文原配置 —— 全 Qwen2.5 家族
- 学生：`DeepSeek-R1-Distill-Qwen-7B`（152064）
- 教师对：`DeepSeek-R1-Distill-Qwen-1.5B`（pre-RL，152064）+ `JustRL-DeepSeek-1.5B`（post-RL，152064）
- **7B 单独重建 `cache_7b.pt`**（用 Qwen2.5 家族教师对），不碰 4B/17B 的 Qwen3 缓存。
  （多学生并发本就是每学生缓存各建，互不干扰。）
- 这是论文 Table 1 的原始设置，效果可预期（56.7 → 63.1 ave@32）。

### 方案 A：7B 档换成 Qwen3 家族模型（复用 Qwen3 教师对缓存）
- 学生换成 `Qwen3-R1-Distill-8B` 或 `Qwen3-8B`（151936），与现有 Qwen3 教师对缓存匹配。
- 缺点：8B 比 7B 更大，显存更紧；且偏离论文的 R1-Distill-7B 档位。

## 四、第二个坎：7B 显存（词表修好后仍需面对）

- 7B fp32-Adam 预估：权重 14.6 + 梯度 14.6 + 优化器 ~87.6 ≈ **117GB > 96GB**（单卡放不下）。
- 解决：`optimizer: adamw_8bit`（bnb）—— 但**先修 4B 暴露的 bnb 转 fp32 反占问题**；
  或 `stage2.offload_to_cpu: true`（worker 权重 CPU offload，`student_real.yaml` 已开）。
- 服务器 4B 最终用 fp32-Adam（61.8GB）成功而非 bnb——需复核 bnb 在该卡的实测峰值。

## 五、待办（服务器恢复后）

1. **诊断**：打印 `cache_7b.pt` 的 `Δ_T` 维度 + 各模型实际 vocab，确认教师对是 Qwen2.5 还是 Qwen3 家族。
2. 按方案 B 重建 `cache_7b.pt`（Qwen2.5 教师对），配置 `ms_7b.yaml` 指向它。
3. 显存复核：`adamw_8bit` 或 `offload_to_cpu`，batch 2~4 起步。
4. 跑通 60 步冒烟 → 评估 → 纳入多学生汇总。
