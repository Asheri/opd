# 2026-08-18：chat template 三件套——服务器恢复后执行序列（零决策）

> 前置：本地已提交（HEAD=6608c5e），全量 434 passed。以下全部在服务器 /root/opd 执行。
> 目标：C3 模板一致性（Step 0 + 重生成 + 重建 Δ_T）→ C1 权重同步加强验证 → 模板 pilot 复测
> （验收：valid_rate≥0.5、refresh pool≥8、附 decode 样本）。

## 0. 同步代码（本地 → 服务器）

本地：
```
cd C:\Users\12062\OneDrive\Desktop\opd
git push origin <branch>      # 或直接 sftp 覆盖 main/ 下变更文件
```
服务器（确保 /root/opd 与本地 HEAD 一致，参照既有 sync 流程）：
```
cd /root/opd && git fetch && git reset --hard <HEAD>   # 或 sftp 覆盖变更文件
cd /root/opd/main && /root/miniconda3/bin/python -m pytest tests/ -q | tail -1
```

## 1. C3 Step 0 — Qwen3 generation prompt 结尾确认（决定 response 前缀约定）

```
/root/miniconda3/bin/python - <<'PY'
import transformers
tok = transformers.AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen__Qwen3-1.7B")
s = tok.apply_chat_template([{"role":"user","content":"X"}], tokenize=False, add_generation_prompt=True)
print("GEN_PROMPT:", repr(s))
print("HAS_THINKING:", "thinking" in s)
PY
```
- 记录结尾是否含 `  thinking`；prepare_skywork_responses --apply-chat-template 生成的
  response 自动以该前缀为起点（模型生成即该前缀后文本），无需手工拼接；仅需把结论写入报告。

## 2. C1 — 权重同步加强验证（扰动 + 分布级 + 贪心）

```
cd /root/opd/main
CUDA_VISIBLE_DEVICES=1,0 /root/miniconda3/bin/python scripts/verify_weight_sync.py
```
- 通过标准：扰动（每层注入 +0.1 后 vLLM logp 变化 >0.01、复原<0.01）；分布级
  ≥512 位置 top1≥0.99、topK logp MAE<0.03；贪心 4×128 位置一致率≥0.99。
- 通过后报告措辞才可升级为"权重加载正确"；未通过则继续排查（不静默）。

## 3. 按模板重生成 base responses（C3）

先确认 input jsonl 各行 prompt 为原始题目、response 已有但将被覆盖（--apply-chat-template
只处理 response 为空的 todo 行 → 如需整体重生成，先清空 response 列或新建副本）：

```
# 副本保护原 jsonl
cp /root/autodl-tmp/datasets/skywork_math_500.jsonl{,.raw}
/root/miniconda3/bin/python scripts/prepare_skywork_responses.py \
  --jsonl /root/autodl-tmp/datasets/skywork_math_500.jsonl \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B --device cuda:0 \
  --max-samples 500 --apply-chat-template
```
- 样本检查：抽 2-3 条 decode 前 200 字符（应为正常推理，非乱码 token soup）。

## 4. 重建 teacher cache（模板 prompt + 教师各自模板 Δ_T）

```
cd /root/opd/main
/root/miniconda3/bin/python -m fullstack_opd_v2.cli cache \
  --config configs/skywork_17b.yaml --set dataset.apply_chat_template=true \
  --set stage1.load_cache=false --out /root/autodl-tmp/cache_skywork_chat.pt 2>&1 | tail -5
```
- 检查 cache metadata 含 `prompt_format: "chat"`（C2 守卫写入）。
- 抽样 decode 校验：教师模板下 Δ_T 合理（无 NaN、支撑率正常）。

## 5. 模板 pilot 复测（L2 + vLLM，repetition_penalty 回退 1.0）

```
cd /root/opd/main
/root/miniconda3/bin/python scripts/run_s2_real.py --config configs/skywork_17b.yaml \
  --run-dir /root/autodl-tmp/runs_s2_vllm_chat --device cuda:0 --n-steps 20 \
  --names S2_E1_opd512 S2_E2_opd1024 --eos-id 151645 --materialized 500 \
  --load-cache \
  --set stage2.rollout_engine=vllm \
  --set dataset.apply_chat_template=true \
  --set stage1.cache_path=/root/autodl-tmp/cache_skywork_chat.pt \
  --set l2.rollout.repetition_penalty=1.0
```
- 用第 4 步已建的 cache（--load-cache + cache_path 指向新 cache）；load_cache 路径会校验
  prompt_format=chat（C2 守卫）——若建的是旧裸 cache 会直接 fail-fast，防静默错位。
  E1/E2 各自 20 步（交叉分卡 E1 训练@0+vLLM@1、E2 反向，见 run_s2_real 约定）。

## 6. 验收（判定达标才算完成）

| 条款 | 判据 |
|---|---|
| valid_rate | ≥ 0.5（IMP-1 原目标；模板下实测应远高于此） |
| refresh pool | ≥ 8（不再触发冷启动跳过，refresh 训练真正跑起来） |
| decode 样本 | 报告中附 2-3 条完整 rollout decode（正常推理内容） |
| C1 | verify_weight_sync.py 三关全过 |
| 回归 | 服务器 pytest 全绿 + 本地 434 passed |

- 顺带把 loop 检测器在模板 rollout 上重新抽样校准（calibrate_rollout.py，
  periods/min_len 可能可收紧）——旧校准已标注 stale。

## 7. 已知边界（不影响本序列执行）

- warmup_M=0 部署下教师模板 fat 行对齐无需处理；warmup>0 + 模板会按 1+warmup_M 倍
  cat 对齐（stage1_build_cache 已实现）。
- is_checkpoint_format=True 每步全量 load_weights；200-step 正式训练前评估耗时，必要时
  切 TP=1 merge_map+param.copy_ 直接拷贝快路径。
