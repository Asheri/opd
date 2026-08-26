# Stage 2：短 Rollout OPD 训练协议 — 真实 GPU 校准与实验结果

> 发布日期：2026-08-15
> 阶段：Stage 2（短预算训练 rollout：train 512/1024/2048 → eval 长预算 4096）
> 状态：校准完成（真实模型）；S2_E0-E3 实跑结果待服务器完成。

## 1. 真实模型校准（任务 152，已完成）

在真实 HF 模型 **Qwen3-1.7B**（`tokenizer_path` 与数据同词表）上跑短 rollout 校准，
产出 `l2.rollout` 的真实取值依据。

| 项 | 取值 | 说明 |
|----|------|------|
| `eos_token_id` | **151645**（`<|im_end|>`） | Qwen3 tokenizer 真实 EOS；采样时显式设此值即"采到停止符即停" |
| `loop_periods` | **空 `()`** | 真实模型在短预算下**不产生循环退化尾部**（p∈{2..8} 命中率 <5%），`detect_loop` 恒不触发 |
| 生成长度分布 | E[L]=512，P(L>2048)=0.00 | 实测 24 条 rollout 全命中预算上限，无自然 EOS |

**判定结论**：默认 `eos_token_id=None`（永不判 EOS，除非 loop）在真实模型上等价于
**全 BUDGET_STOP**。`loop_periods` 无需配置（真实推理不会退化成周期重复），
loop 拦截对真实模型是**防御性**而非必需——这符合"预算截断是常态、自然 EOS 不是必然"的协议设计。

校准脚本：`main/scripts/calibrate_rollout.py`（`--model Qwen__Qwen3-1.7B`）。

## 2. S2 实验矩阵（真实 GPU 实跑）

配置基座 `configs/skywork_17b.yaml`（model_kind=hf，学生 Qwen3-1.7B，教师对 JustRL-1.5B / R1-Distill-1.5B，
cache topk K=256，materialized=500 锚点，数据 Skywork 50K prompts + 500 预生成 response）。

| 实验 | l2 | rollout budget | 语义 |
|------|----|----------------|------|
| S2_E0_static | 关 | — | 静态基线（纯 base 训练） |
| S2_E1_opd512 | 开 | 512 | OPD + 短 rollout 512 |
| S2_E2_opd1024 | 开 | 1024 | OPD + 短 rollout 1024（主实验） |
| S2_E3_opd2048 | 开 | 2048 | OPD + 短 rollout 2048 |

> 训练端为短预算（真实 512/1024/2048 rollout）；评估端保持长预算（`budget_eval` B=4096），
> 验证"训练短、评估长"的迁移（Q4）。

## 3. Q1-Q4 解读

### Q1 · 短 rollout 能否稳定产生有效 OPD learning signal？

<!-- 待 S2 实跑填充：pg_loss_mean / reward_mean / kl_loss_mean + rollout 状态计数 -->

### Q2 · 1024 训练预算能否提升长预算（4096）评估？

<!-- 待 S2 + budget_eval B=4096 填充 -->

### Q3 · 训练预算的边际收益（512→1024→2048）如何？

<!-- 待 S2 实跑填充 -->

### Q4 · 训练短预算、评估长预算的迁移是否存在？

<!-- 待 S2 + budget_eval B=4096 填充 -->

## 4. 验收代理（当前置为待实跑）

- [ ] 训练端：`S2_E2_opd1024` 跑完 `pg_loss_mean != 0`、`reward_mean` 相对 `S2_E0_static` 有区分。
- [ ] 状态分布：真实 rollout 在 512/1024/2048 下应全 `BUDGET_STOP`（eos=None + 无循环），
      `n_eos`≈0、`n_loop`≈0。
- [ ] 评估端：短预算训练的长预算 B=4096 分数 ≥ 静态基线（迁移不损失）。

## 复现命令

```bash
# 校准（真实模型）
cd main && python scripts/calibrate_rollout.py --model Qwen__Qwen3-1.7B \
  --jsonl datasets/skywork_50k.jsonl --device cuda:0 --n 32 --max-new 512

# S2 矩阵（首实验建缓存，其后 --load-cache 复用）
python scripts/run_s2_real.py --config configs/skywork_17b.yaml --run-dir <dir> \
  --device cuda:0 --n-steps 20 --eos-id 151645 --materialized 500 \
  --cache-path /root/autodl-tmp/cache_skywork_17b.pt --names S2_E0_static
python scripts/run_s2_real.py --config configs/skywork_17b.yaml --run-dir <dir> \
  --device cuda:0 --n-steps 20 --eos-id 151645 --materialized 500 \
  --cache-path /root/autodl-tmp/cache_skywork_17b.pt --load-cache \
  --names S2_E1_opd512 S2_E2_opd1024 S2_E3_opd2048
```