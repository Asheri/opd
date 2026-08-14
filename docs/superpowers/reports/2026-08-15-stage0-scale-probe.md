# Stage 0 规模决策报告：50K×8192 数据构建路径可行性

> **状态：模板（待服务器实测填数字）**。本报告的每个数字都来自真实 GPU benchmark
> （`gen_benchmark.py`），不沿用任何理论估算。请按下方「待填」标注，在服务器跑完
> `gen_benchmark.py` + `stage0_scale_probe.py` 后回填，并勾选结论。

## 1. 实测 generation throughput（待填）

在服务器用 `main/scripts/gen_benchmark.py --matrix ...` 实测，命令示例：

```bash
cd /root/opd/main
python scripts/gen_benchmark.py \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --prompts /root/autodl-tmp/datasets/skywork_50k.jsonl \
  --matrix "32,2048,1 32,4096,2 128,2048,2 128,8192,2 512,8192,2" \
  --max-time 120 --out /root/autodl-tmp/eval/gen_benchmark.json
```

| N | max_new | batch | tok/s | E[L] | P(L>2048) | P(L=8192) | gpu_mem_peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

> 成本权威口径 = **T_actual = Σ L_i**（`aggregate_stats.generated_tokens`），
> 绝不用 N × max_new_tokens（严重高估）。

## 2. 实测 response length distribution（待填）

| 统计量 | 值 |
|---|---|
| mean_len (E[L]) | 待填 |
| p50 / p90 / p95 / max | 待填 |
| eos_rate / truncation_rate | 待填 |
| P(L>2048) / P(L>4096) / P(L=8192) | 待填 |

## 3. 5K/10K/25K/50K 预计 wall time（stage0_scale_probe.py 输出）

```bash
python scripts/stage0_scale_probe.py \
  --benchmark /root/autodl-tmp/eval/gen_benchmark.json \
  --out /root/autodl-tmp/eval/scale_probe_report.json
```

| stage | N | max_len | mean_len | tok/s | hours(1卡) | hours(2卡) |
|---|---:|---:|---:|---:|---:|---:|
| pilot | 5,000 | 2,048 | 待填 | 待填 | 待填 | 待填 |
| scale-1 | 10,000 | 4,096 | 待填 | 待填 | 待填 | 待填 |
| scale-2 | 25,000 | 8,192 | 待填 | 待填 | 待填 | 待填 |
| full | 50,000 | 8,192 | 待填 | 待填 | 待填 | 待填 |

## 4. 推荐最终规模（待勾选）

- [ ] 保持 50K×8192（单卡即现实）
- [ ] 保持 50K×8192，但需 **2 卡并行**（`--num-shards 2 --shard-rank 0/1`）
- [ ] 降规模优先：推荐 N = 待填（理由：待填）

## 5. 是否需要 2 卡并行 generation（待勾选）

- [ ] 需要（单卡 full 超现实阈值，2 卡达标）
- [ ] 不需要（单卡可达）

## 6. 是否值得保持 max_response_len=8192（待勾选）

- [ ] 值得（有显著 P(L>4096) 长尾，丢掉会截断真实思考链）
- [ ] 不值得，降到 max_response_len = 待填（P(L>4096) 极低，E[L] 远小于 8192）

## 7. Base Pool 推荐规模（待填）

`base.materialized_size` 建议值 = 待填（5K~10K 起步；实际只需 E[L] 长度，其余 prompt
留空待 L2 在线 refresh，`JsonLinesDataLoader` 自动跳过空 response 行）。

## 8. 若 50K×8192 不现实 → 降规模路径（不硬跑）

按 §4 推荐降规模后，用分片并行补静态锚点：

```bash
cd /root/opd/main
python scripts/prepare_skywork_responses.py \
  --jsonl /root/autodl-tmp/datasets/skywork_50k.jsonl \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --max-samples 5000 --max-new-tokens 8192 --batch-size 8 \
  --num-shards 2 --shard-rank 0    # 另一卡跑 --shard-rank 1
```

> 分片并行：各 shard 独立 tmp、互不重合，完成后 flock 串行 merge；支持中断 resume。