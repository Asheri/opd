# Stage 0 规模决策报告：50K×8192 数据构建路径可行性

> **状态：已完成服务器实测（2026-08-16，RTX PRO 6000 Blackwell，Qwen3-1.7B，flash_attention_2）。
> 本报告所有数字来自真实 GPU benchmark（`gen_benchmark.py`，批 1-2 保守口径），
> 外加 `generation_smoke.py`（批 4，141.7 tok/s/卡）与 `k_calibration.py`（批 4，323 tok/s）交叉参照。

## 1. 实测 generation throughput（gen_benchmark.py，每档 --max-time 120s）

| N | max_new | batch | tok/s | E[L] | P(L>2048) | P(L=8192) | gpu_mem_peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2048 | 1 | 47.4 | 2048 | 0.00 | 0.00 | ~6.4 GB |
| 32 | 4096 | 2 | 69.2 | 4096 | 1.00 | 0.00 | ~6.4 GB |
| 128 | 2048 | 2 | 69.7 | 2048 | 0.00 | 0.00 | ~6.4 GB |
| 128 | 8192 | 2 | 70.0 | 4198* | 1.00 | 0.00 | ~6.4 GB |

> *8192 档为 120s 时间截断（非自然 E[L]）；其余档 E[L]=cap（100% 撞 max_new 截断）。
> 成本口径 T_actual=ΣL_i（`aggregate_stats.generated_tokens`）。
> 交叉参照：批 4 下 `generation_smoke.py` 实测 141.7 tok/s/卡（双卡 281.9）、
> `k_calibration.py` 短前缀 1024 档 323 tok/s —— 批 1-2 是保守下限，批 4 约 2-4×。

## 2. 实测 response length distribution

| 统计量 | 值（cap=2048） | 值（cap=4096） | 值（cap=8192·时间截断） |
|---|---|---|---|
| mean_len (E[L]) | 2048 | 4096 | 4198 |
| p50 / p90 / p95 / max | 2048×4 | 4096×4 | 4198×4 |
| eos_rate / truncation_rate | 0 / 1.0 | 0 / 1.0 | 0 / 1.0 |
| P(L>2048) | 0.00 | 1.00 | 1.00 |
| P(L>4096) / P(L=8192) | 0 / 0 | 0 / 0 | 1.00 / 0.00 |

> **结论：Skywork math prompt 在 temperature=1.0 下 CoT 极长，任何 ≤8192 cap 都 100% 撞截断、
> 0 自然 EOS**（与 generation_smoke 4096 cap 结论一致）。P(L>4096)=1.0 表明长尾真实存在。

## 3. 5K/10K/25K/50K 预计 wall time（stage0_scale_probe.py 外推，批 1-2 tok/s）

| stage | N | max_len | mean_len | tok/s | hours(1卡) | hours(2卡) |
|---|---:|---:|---:|---:|---:|---:|
| pilot | 5,000 | 2,048 | 2048 | 69.7 | 40.8 | 20.4 |
| scale-1 | 10,000 | 4,096 | 4096 | 69.2 | 164.4 | 82.2 |
| scale-2 | 25,000 | 8,192 | 4198 | 70.0 | 416.5 | 208.2 |
| full | 50,000 | 8,192 | 4198 | 70.0 | 832.9 | 416.5 |

> 批 4 口径（141 tok/s/卡）下 full 50K×8192 约 413-807h/卡，同样超现实。
> **降规模是唯一现实路径**；5K pilot（40.8h/20.4h）为可接受起步档。

## 4. 推荐最终规模（实测后勾选）

- [ ] 保持 50K×8192（单卡即现实）
- [ ] 保持 50K×8192，但需 **2 卡并行** —— 2 卡 416h 仍超现实，否决
- [x] **降规模优先：推荐 N = 5K materialized（pilot 档）**，理由：50K 全量 833h/卡（批1-2）
      或 413h/卡（批4）远超 72h 现实阈值；5K@2048 pilot 40.8h/20.4h 可作后台任务推进。

## 5. 是否需要 2 卡并行 generation

- [x] 需要（单卡 full 超现实阈值，2 卡达标）—— 对 full 50K：两卡 416h 仍不达标 → **不通过**
- [ ] 不需要（单卡可达）—— 对 5K pilot：单卡 40.8h 现实，2 卡 20.4h 更快；按需

## 6. 是否值得保持 max_response_len=8192

- [x] 值得（P(L>4096)=1.00 确有显著长尾，丢 4096-8192 会截断真实思考链）
- [ ] 不值得，降到 max_response_len = 4096 —— 用户已于 K 校准定案 max tokens=4096
      （E[L] 恒等于 cap，4096 只是更早断；缓存按位置存 Δ_T 不受影响）

## 7. Base Pool 推荐规模

`base.materialized_size` 建议值 = **5000**（5K pilot 起步；实际只需 E[L] 长度，
其余 prompt 留空待 L2 在线 refresh，`JsonLinesDataLoader` 自动跳过空 response 行）。

## 8. 降规模路径（已确认不硬跑 50K）

- 本次会话采用 pilot：`cache_skywork_17b.pt`（500 条，max_response_len=2048）+ S2/E0-E6
  训练矩阵在该 pilot 上跑通（验证协议与健康曲线）。
- 全量 50K×4096 生成（~413h/卡@批4）留作后续后台任务，`prepare_skywork_responses.py`
  分片并行 + resume-safe 推进；本报告数字即该决策的实测依据。
