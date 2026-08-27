# 2026-08-27：新教师对机制验证计划（Qwen3 Base↔Instruct，最小代价拿积极结果）

> **性质：执行计划（待批准）**。服务器恢复后执行；总计 ~6-7h 墙钟 / ~10 GPU·h。
> 依据：`2026-08-27-opd-final-report.md`（复现失败定稿）+ 本轮逐项代码核对（§1）。
>
> **审阅修正（2026-08-27，主会话审阅意见落地）**：
> - R-E1（必须）：训练 eos 用 Base 原生 `151643`（非 instruct `151645`）——Base 权重未学 `im_end` 语义，补丁不改权重；§3.1 冒烟加 im_end 生成验证后定档。
> - R-E2（必须）：D 生成后加质量统计门控（no_answer_rate ≤30%），替代"抽 3 条"弱判据。
> - R-S1：Base 下载后词表/tokenizer 校验（V9 + 下载完整性）。
> - R-S2：E-1b' sample 后检查 Base 响应长度（过短则提预算）。
> - R-S3：Phase 2 快评（100 题）与全量（500）一致性核对，防噪声选错步。
> 核心思路：旧失败主因是**赔率结构**（教师对 shift 仅 +0.05、学生起点 0.816 已≈教师上限）。
> 换大 shift 同族对（Base↔Instruct，理论迁移空间 = 0.816 − Base 实测值），回到论文
> on-policy 语义（D = 学生自身 rollout），全部基建零改动复用。

---

## 1. 逐项核对结论（假设 → 事实）

| # | 方案中的假设 | 核对结果 | 出处/动作 |
|---|---|---|---|
| V1 | R1（run_s2_real metrics 丢行）需先修 | ✅ **已修**（`metrics = out["metrics"]` 在最新 main L210） | `git show refs/heads/main:main/scripts/run_s2_real.py`；服务器 `git pull` 即得 |
| V2 | R2（all_results 覆盖）需先修 | ✅ **已修**且实战通过（实测报告 §8"R2 合并写实战通过"） | 同上 |
| V3 | B8192@3 majority 口径可用 | ✅ `--n-samples/--temperature` + majority 聚合已实现（commit 77203cf，23 测试过）；`--max-model-len` 默认已提 12288 | `vllm_budget_eval.py:52-64` |
| V4 | E-1b' 脚本支持新教师对 | ✅ `--student/--teacher-rl/--teacher-ref/--budget/--chat-template` 全有；默认指向旧对，**必须显式覆盖** | `delta_correctness_corr.py:345-363` |
| V5 | AIME24 可用 vllm_budget_eval 测 | ✅ DatasetSpec 注册表含 AIME24/AIME25 | `budget_eval.py:97-99` |
| V6 | **Base 模型自带 chat template** | ❌ **不成立**--Qwen3-1.7B-Base 纯预训练模型**不带 chat 模板**，eos=151643（endoftext）；instruct 的 eos=151645（im_end） | HF 社区资料（见文末链接）；**必须加"模板补丁"步骤**（§3.1） |
| V7 | 直接在原 skywork_50k.jsonl 上补生成 1000 条 | ❌ **会污染 D**：原 jsonl 已有 500 条 **instruct 生成的 response**，loader 会一起读入（500 instruct + 1000 Base 混合） | `prepare_skywork_responses.py` todo 逻辑排除非空行；**必须复制+清空 response**（§3.2） |
| V8 | 当前"Base"是未经训练的基座 | ❌ **澄清**：`Qwen__Qwen3-1.7B`（HF id `Qwen/Qwen3-1.7B`）是 **instruct 版**（SFT+RLHF 后）。旧报告里的"Base=0.816"实为 **instruct 分数**--这正好成为新设计中 **rl 教师的已知分数**（无需重测） | yaml `student_path` + 0.816@chat 协议 + thinking 式 avg_rt 4303 |
| V9 | 词表/模板一致性 | ✅ Base 与 Instruct 同 tokenizer（151936 词表），教师/学生/评估三侧 tokenization 逐 token 一致；`enforce_teacher_consistency` 会通过 | Qwen3 家族事实 |
| V10 | KL 锚点需调整 | ✅ 无需改动：锚点=初始学生分布，新设计中初始学生=ref 教师（同一权重），锚点与 Δ 的 ref **天然重合**（数学上即论文原始设定） | `pipeline.py` ref_dists 逻辑 |
| V11 | 正式训练命令形态 | ✅ 镜像 E1 串行重跑（已知可跑完 300 步）：train cuda:0 + vLLM refresh 引擎 rollout_device 默认 cuda:1（交叉分卡）；矩阵覆盖被 `--set` 后到先覆盖（extra_sets 最后生效） | `run_s2_real.py` `_build_overrides`（matrix 先、--set 后） |
| V12 | 服务器同步状态 | ⚠️ 服务器停在 `ce05e61`；**77203cf（多采样）在其后**--Phase 0.2 依赖它，开跑前必须 `git pull` | 终稿报告 §0 + git log |

**修正汇总（相对上一轮口头方案）**：① 新增 Base 目录 chat 模板补丁步骤（V6）；
② jsonl 复制+清空 response（V7）；③ "旧 Base=0.816"重标注为 instruct/rl 教师分数（V8）；
④ R1/R2 从"待修"改为"服务器 pull 即可"（V1/V2）；⑤ delta_corr 必须显式传新教师对路径（V4）。

---

## 2. 实验设计（新对）

| 角色 | 模型 | 路径（服务器） | 状态 |
|---|---|---|---|
| **学生**（= ref 教师） | Qwen3-1.7B-**Base**（预训练） | `/root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base` | **待下载**（~3.4GB） |
| **rl 教师** | Qwen3-1.7B（instruct，SFT+RLHF） | `/root/autodl-tmp/models/Qwen__Qwen3-1.7B` | 已在（现学生） |
| D（训练数据） | **学生自身 rollout**（1000 条，chat 模板，T=4096，T=1.0，seed 43） | `skywork_50k_base1000.jsonl`（复制+清空后生成） | 待生成 |

- **Δ_T = logπ_instruct − logπ_Base**：同一模型的 post-training 偏移（诚实标注：
  instruct = SFT+RLHF 复合，非纯 RL--**机制等价**于论文 Δ_T，叙事为"机制验证"非逐字复现）。
- **理论迁移空间 = 0.816（instruct 已实测）− Base 实测值**（Phase 0.2 测出）。
  预期 Base@chat MATH500 ≈ 0.3-0.6 → 空间 +0.2~+0.5（旧对只有 +0.056）。
- 旧对（JustRL/R1Distill）与旧 cache/旧 500 条 instruct response 全部**原地保留不动**（不可再生约束）。

**门控纪律（继承终稿报告 §7）**：判据一律下游指标（MATH500 B8192@3 majority）；
eval_reward 仅作漂移报警；120 步止损线。

---

## 3. Phase 0：预检（~1.5h 墙钟，双卡并行；不通过不训练）

### 3.0 前置（~15min）

```bash
# 服务器同步（含 77203cf 多采样 + R1 修复；主 checkout 需先 push 未推的 2 个 docs 提交）
cd /root/opd && git pull && git log --oneline -1
grep -n 'metrics = out' main/scripts/run_s2_real.py          # 判据：恰 1 行（R1 已修）
PYTHONIOENCODING=utf-8 /root/miniconda3/bin/python -m pytest main/tests/test_vllm_budget_eval.py -q | tail -1   # 判据：23 passed

# 下载 Base（AutoDL 加速）
source /etc/network_turbo
huggingface-cli download Qwen/Qwen3-1.7B-Base --local-dir /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base

# R-S1：下载后词表/tokenizer 校验（V9 一致性 + 下载完整性，防后半程崩）
python -c "
from transformers import AutoTokenizer
a=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen__Qwen3-1.7B')
b=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base')
assert a.vocab_size==b.vocab_size==151936, (a.vocab_size,b.vocab_size)
print('词表一致 151936 ✅；Base eos_token_id=', b.eos_token_id)
"
# 判据：词表一致；记录 Base 原生 eos_token_id（预期 151643 endoftext）

### 3.1 chat 模板补丁（V6 修正，~2min，必做）

```bash
# Base 无 chat 模板：从 instruct 目录复制模板文件 + 注入 tokenizer_config
cp /root/autodl-tmp/models/Qwen__Qwen3-1.7B/chat_template.json \
   /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base/ 2>/dev/null || true
python - <<'EOF'
import json
src = "/root/autodl-tmp/models/Qwen__Qwen3-1.7B/tokenizer_config.json"
dst = "/root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base/tokenizer_config.json"
a, b = json.load(open(src, encoding="utf-8")), json.load(open(dst, encoding="utf-8"))
if "chat_template" in a: b["chat_template"] = a["chat_template"]
json.dump(b, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
EOF
# 冒烟判据（写死）：输出含 <|im_start|>user 与 <|im_end|>，无异常
python -c "from transformers import AutoTokenizer as T; print(T.from_pretrained('/root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base').apply_chat_template([{'role':'user','content':'1+1='}], add_generation_prompt=True, tokenize=False))"

# R-E1：im_end 生成验证——决定训练 eos 档（Base 权重未学 im_end 语义，模板补丁不改权重）
python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base')
model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base', torch_dtype=torch.bfloat16, device_map="cuda:0")
msgs = tok.apply_chat_template([{"role": "user", "content": "Solve for x: 2x+3=7."}],
                               add_generation_prompt=True, tokenize=False)
ids = tok(msgs, return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=1.0)
text = tok.decode(out[0])
has_im_end = tok.eos_token_id in out[0].tolist()  # Base 原生 endoftext
print("生成含 im_end token(151645):", 151645 in out[0].tolist(), "| eos 触发(151643):", has_im_end)
print("样本:", text[:200].replace("\n", " | "))
EOF
# 判据：输出正常文本 + 记录 eos 行为——refresh 训练 eos 档据此定：
#   Base 能产出 im_end → 训练 --eos-id 151645（与 instruct 模板一致）
#   Base 不产 im_end → 训练 --eos-id 151643（Base 原生 endoftext；推荐默认）
```

### 3.2 学生基线测量（GPU0，~50min）

```bash
/root/miniconda3/bin/python scripts/vllm_budget_eval.py \
  --models "BaseS=/root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base" \
  --budgets 8192 --n-samples 3 --temperature 0.7 \
  --dataset MATH500 --n-limit 500 --chat-template \
  --tokenizer /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --device cuda:0 --out-dir /root/autodl-tmp/newpair/base_B8192_3
```

（`--tokenizer` 指向 instruct：即使 3.1 未做也稳；评估/训练/教师三侧模板一致性由同族 tokenizer 保证。）

### 3.3 E-1b' 信号密度（GPU1，~70min；与 3.2 并行）

```bash
O=/root/autodl-tmp/newpair/delta_corr
/root/miniconda3/bin/python scripts/delta_correctness_corr.py --stage sample \
  --student /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base \
  --dataset MATH500 --n-problems 200 --n-samples 4 --budget 4096 \
  --temperature 1.0 --chat-template --device cuda:1 --out $O
/root/miniconda3/bin/python scripts/delta_correctness_corr.py --stage logp \
  --teacher-rl /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --teacher-ref /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base \
  --teacher rl  --device cuda:1 --out $O
/root/miniconda3/bin/python scripts/delta_correctness_corr.py --stage logp \
  --teacher-rl /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --teacher-ref /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base \
  --teacher ref --device cuda:1 --out $O
/root/miniconda3/bin/python scripts/delta_correctness_corr.py --stage correlate --out $O
# R-S2：sample 后检查 Base 响应长度（B4096 下预训练续写可能过短 → Δ 域太小）
python -c "
import json
rows=[json.loads(l) for l in open('$O/samples.jsonl')]
rt=[len(r['response'].split()) for r in rows]
import statistics; print('Base 响应平均长度(tokens 近似):', round(statistics.mean(rt),1), '中位:', statistics.median(rt))
"
# 判据：平均长度 < 200 → 提高 sample --budget 或接受（on-policy 本来语义）；记录于报告
```

### 3.4 Phase 0 门控（写死）

| 指标 | 通过 | 边界（停下问用户） | 不通过（停） |
|---|---|---|---|
| BaseS acc（B8192@3） | ≤ 0.65（空间 ≥ +0.15） | 0.65 < acc ≤ 0.75 | > 0.75（无空间）或 < 0.10（chat 协议不可用） |
| E-1b' Spearman ρ | ≥ 0.20 | 0.05 ≤ ρ < 0.20 | < 0.05（机制层存疑，转实现审计） |

任一停：总代价 ≤ 1.5h（对比旧路径先烧 50+ GPU 时）。

---

## 4. Phase 1：数据 + 缓存 + 训练（~4h）

### 4.1 jsonl 复制 + 清空 response（V7 修正；原文件不动）

```bash
cp /root/autodl-tmp/datasets/skywork_50k.jsonl /root/autodl-tmp/datasets/skywork_50k_base1000.jsonl
python - <<'EOF'
import json
p = "/root/autodl-tmp/datasets/skywork_50k_base1000.jsonl"
rows = [json.loads(l) for l in open(p, encoding="utf-8")]
n = 0
for r in rows:
    if r.get("response"): r["response"] = ""; n += 1
with open(p, "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("blanked", n)     # 判据：blanked 500（恰好旧 instruct 行数）
EOF
```

### 4.2 D 生成：学生自身 rollout（双卡分片，~40min）

```bash
/root/miniconda3/bin/python scripts/prepare_skywork_responses.py \
  --jsonl /root/autodl-tmp/datasets/skywork_50k_base1000.jsonl \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base \
  --apply-chat-template --max-samples 1000 --seed 43 \
  --max-new-tokens 4096 --temperature 1.0 \
  --device cuda:0 --batch-size 8 --shard-rank 0 --num-shards 2 --resume &
/root/miniconda3/bin/python scripts/prepare_skywork_responses.py \
  --jsonl /root/autodl-tmp/datasets/skywork_50k_base1000.jsonl \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base \
  --apply-chat-template --max-samples 1000 --seed 43 \
  --max-new-tokens 4096 --temperature 1.0 \
  --device cuda:1 --batch-size 8 --shard-rank 1 --num-shards 2 --resume &
wait
# 判据（R-E2 加强）：jsonl 非空 response 行 = 1000 + 质量统计门控
python - <<'EOF'
import json, re
rows=[json.loads(l) for l in open("/root/autodl-tmp/datasets/skywork_50k_base1000.jsonl", encoding="utf-8")]
rs=[r["response"] for r in rows if r.get("response")]
n=len(rs)
noans=sum(1 for r in rs if "\\boxed{" not in r and "answer" not in r.lower())
short=sum(1 for r in rs if len(r)<100)
print(f"非空={n}/1000 无答案≈{noans/n:.2%} 过短(<100字符)={short/n:.2%} 平均长度={sum(len(r) for r in rs)//max(n,1)}")
EOF
# 门控（写死）：无答案率 ≤30% 且 过短率 ≤20% 才继续；超限 → 停，重评 D 生成策略
# 抽 3 条 decode 无乱码无 loop（保留原判据）
```

### 4.3 新配置 `configs/qwen3_base_opd.yaml`（复制 skywork_17b.yaml 改 9 处）

```yaml
student_path:     /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base
teacher_rl_path:  /root/autodl-tmp/models/Qwen__Qwen3-1.7B        # instruct = rl 教师
teacher_ref_path: /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base   # = 学生初始
dataset:
  path: /root/autodl-tmp/datasets/skywork_50k_base1000.jsonl
  apply_chat_template: true
  max_response_len: 4096        # D 长度对齐（C2 守卫）
stage1:
  cache_path: /root/autodl-tmp/cache_qwen3_base_opd.pt
  build_batch_size: 4           # T 翻倍 -> batch 减半
base:
  materialized_size: 1000
stage2:
  n_steps: 120                  # E-0c 教训：120 步门控，不盲跑 300
  kl_reg_coef: 0.1              # RC2 教训：0.02 作废；0.1 中档信任域
  batch_size: 2                 # T=4096 激活减半
run:
  checkpoint_every: 20          # 6 断点供拐点扫描
```

### 4.4 cache build（GPU0，~1.5h）

```bash
/root/miniconda3/bin/python -m fullstack_opd_v2 cache \
  --config configs/qwen3_base_opd.yaml --set stage1.load_cache=false \
  --device cuda:0 --out /root/autodl-tmp/cache_qwen3_base_opd.pt
# 判据：metadata prompt_format=chat、T=4096、num_samples=1000；verify_consistency 过
```

### 4.5 训练（GPU0 训练 + GPU1 vLLM refresh，~2-2.5h；镜像 E1 串行已知好路径）

```bash
/root/miniconda3/bin/python scripts/run_s2_real.py \
  --config configs/qwen3_base_opd.yaml \
  --run-dir /root/autodl-tmp/runs_newpair/e1_opd4096 \
  --names S2_E2_opd1024 --device cuda:0 --n-steps 120 \
  --eos-id 151643 \
  --set stage2.rollout_engine=vllm \
  --set l2.rollout.max_new_tokens=4096 \
  --set stage2.gradient_checkpointing=true \
  --set l2.rollout.repetition_penalty=1.0
# 判据：跑满 120 步；summary 正常（R1 已修）；metrics.csv 120+ 行（含 refresh 步）
# 回退：vLLM refresh 权重同步异常时去掉 rollout_engine=vllm（退 toy 引擎，慢但稳）
```

（矩阵名 S2_E2_opd1024 只为启用 L2 refresh；`--set l2.rollout.max_new_tokens=4096`
后到覆盖矩阵的 1024。**eos 档 = R-E1 冒烟验证结果**：Base 不产 im_end → 151643（默认，
上文已改）；若冒烟证明 Base 产 im_end，可改回 151645——两档均需与 D 的 response
（Base 生成）的停止 token 一致，否则 refresh 无 eos 样本。）

### 4.6 Phase 1 止损门控（训练中）

- eval_reward（每 20 步自动）**仅作漂移报警**：升而 checkpoint 快评降 → 预警但不自动停；
- 训练异常/OOM → 报错追加 training-errors.md（硬约束）。

---

## 5. Phase 2：判定（~1.5h，写死）

### 5.1 checkpoint 快评扫（100 题子集，双卡）

```bash
for S in 40 80 120; do
  /root/miniconda3/bin/python scripts/export_student_ckpt.py \
    --ckpt /root/autodl-tmp/runs_newpair/e1_opd4096/S2_E2_opd1024/checkpoints/step_${S}.pt \
    --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B-Base \
    --out /root/autodl-tmp/exported/newpair_s${S}      # 默认 --device cpu，不抢卡
done
# 双卡并行快评（GPU0: S40+S80 顺序；GPU1: S120）：
/root/miniconda3/bin/python scripts/vllm_budget_eval.py \
  --models "S40=...,S80=..." --budgets 8192 --n-samples 3 --temperature 0.7 \
  --dataset MATH500 --n-limit 100 --chat-template --tokenizer /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --device cuda:0 --out-dir /root/autodl-tmp/newpair/sweep &
# ... GPU1 同法跑 S120；wait
```

### 5.2 最优步全量终验（500 题 + AIME24）

最优 step（快评最高者）跑全量 MATH500 B8192@3（双卡可用两进程分半：`--n-limit` 不支持
分半，改一进程 500 题 ~50min，另一卡并行跑 AIME24 同口径）。

**R-S3 快评→全量一致性**：快评（100 题）选步后，全量前先核对 100 题子集在快评与全量中的
acc 是否一致（同 100 题的重跑比对）——若不一致（噪声大）说明 100 题区分度不足，改为
200 题快评或取快评 top-2 步全量对比，防噪声选错步。

### 5.3 判定表（写死）

| 结果（全量 500 题，vs 学生基线） | 判定 | 动作 |
|---|---|---|
| ≥ +0.05 | **积极结果**：Δ_T 机制验证成功 | 出报告（论文对比叙事：机制成立 + 旧失败归因闭环） |
| +0.02 ~ +0.05 | 弱阳性 | 200 步延长 + refresh 配比提高（refresh_min 10→5）再判一次 |
| < +0.02 | 失败 | 停；此时才有资格怀疑实现层（逐行审计 l2 信号链） |

---

## 6. 风险表

| 风险 | 概率 | 缓解 |
|---|---|---|
| Base 在 chat 协议下乱写（D 质量/ρ 低） | 中 | Phase 0.3 门控直接拦截（代价 1.5h）；快评 sample 首验 3 条 |
| Base 无模板导致脚本崩 | 已消除 | §3.1 补丁 + 冒烟判据；评估走 `--tokenizer` instruct |
| vLLM refresh 权重同步抽风（历史 NCCL） | 低-中 | 镜像 E1 串行路径；回退 toy 引擎 |
| T=4096 cache build 显存 | 低 | build_batch_size=4（历史 8@2048 的等比缩） |
| 下载被墙 | 低 | network_turbo / modelscope 镜像 |
| 磁盘 | 极低 | 增量 ~80GB（cache 12 + ckpt 6×11.5 + 产物），可用 617GB |

## 7. 成本与总验收

**预算**：Phase 0 ~1.5h → Phase 1 ~4h → Phase 2 ~1.5h，合计 **~7h 墙钟 / ~10 GPU·h**（旧路径已烧 60+ GPU·h）。

| # | 验收项 | 判据（写死） |
|---|---|---|
| 1 | 预检完整 | V1-V12 逐项过；Phase 0 双门控过 |
| 2 | D 纯净 | jsonl 非空 response=1000 且全部来自 Base 学生（复制文件 blanked=500 核对） |
| 3 | 训练完成 | 120 步、6 断点、summary 无 error、metrics 完整 |
| 4 | 主判据 | 最优步 MATH500 B8192@3 ≥ 学生基线 + 0.05 |
| 5 | 产物入库 | 全部 jsonl/report 落盘 + 报告回填；报错追加 training-errors.md |
| 6 | 旧数据零破坏 | 原 jsonl / 旧 cache / 旧 run-dir / 旧 exported 一律未动 |

## 参考（V6/V9 事实来源）

- [Qwen/Qwen3-1.7B-Base（HF）](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [SWE-ZERO 中对 Base 模型 eos/chat 处理的实践（marin issue）](https://github.com/marin-community/marin/issues/5611)
- [Qwen3 官方博客](https://qwen.ai/blog?id=qwen3)
