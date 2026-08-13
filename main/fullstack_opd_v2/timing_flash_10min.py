"""flash_attn 实测：10 分钟限时生成，按已生成 token 数估算 32768 全长耗时。

目的：消除两个未知（决定后续评估协议）：
  1. flash_attn 启用后的真实 decode 速度（tok/s）
  2. 据此外推一条 32768 全长需要多久 -> 判断单卡是否可行

方法：启用 flash_attention_2 加载 Qwen3-1.7B，用真实 AIME24 题 + 完整协议
（chat_template + boxed + T=0.7/top_p=0.95），max_new_tokens=32768，但用
StoppingCriteria 在 600 秒（10 分钟）时停止。记录实际生成的 token 数与耗时，
外推 32768 全长时间。

用法（服务器）：
  cd /root/opd/main && python /root/autodl-tmp/eval/timing_flash_10min.py
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset
from fullstack_opd_v2.eval_aime import format_prompt

PATH = "/root/autodl-tmp/models/Qwen__Qwen3-1.7B"
TIME_LIMIT_S = 600   # 10 分钟
TARGET_TOKENS = 32768

print(f"[{time.strftime('%H:%M:%S')}] 加载模型 (flash_attention_2, bf16)...", flush=True)
m = AutoModelForCausalLM.from_pretrained(
    PATH, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2").to("cuda:0").eval()
tok = AutoTokenizer.from_pretrained(PATH)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

# 真实 AIME24 第一题 + 完整协议
ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
problem = str(ds[0]["Problem"])
raw = format_prompt(problem, "boxed")
chat = tok.apply_chat_template([{"role": "user", "content": raw}],
                               add_generation_prompt=True, tokenize=False)
inp = tok(chat, return_tensors="pt").to("cuda:0")
prompt_len = inp.input_ids.shape[1]
print(f"[{time.strftime('%H:%M:%S')}] prompt {prompt_len} token，开始限时 {TIME_LIMIT_S}s 生成...", flush=True)


class TimeLimitCriteria(StoppingCriteria):
    """到时间上限即停（每步检查，最多多生成 1 token）。"""
    def __init__(self, limit_s):
        self.t0 = time.time()
        self.limit = limit_s

    def __call__(self, input_ids, scores, **kwargs):
        return (time.time() - self.t0) >= self.limit


t0 = time.time()
with torch.no_grad():
    out = m.generate(
        **inp, max_new_tokens=TARGET_TOKENS, do_sample=True,
        temperature=0.7, top_p=0.95, num_return_sequences=1,
        pad_token_id=tok.pad_token_id,
        stopping_criteria=StoppingCriteriaList([TimeLimitCriteria(TIME_LIMIT_S)]))
dt = time.time() - t0
gen_tokens = out.shape[1] - prompt_len
tok_per_s = gen_tokens / dt

# 外推 32768 全长耗时
full_est = TARGET_TOKENS / tok_per_s if tok_per_s > 0 else float("inf")

print("=" * 60, flush=True)
print(f"flash_attn 实测结果（限时 {TIME_LIMIT_S}s）：", flush=True)
print(f"  实际生成: {gen_tokens} token / {dt:.1f}s", flush=True)
print(f"  decode 速度: {tok_per_s:.1f} tok/s", flush=True)
print(f"  外推 32768 全长: {full_est/60:.1f} 分钟 ({full_est/3600:.2f} 小时)/条", flush=True)
print(f"  n=8 单数据集(30题)估算: {30*8*full_est/3600:.1f} 小时", flush=True)
print(f"  n=16 单数据集估算: {30*16*full_est/3600:.1f} 小时", flush=True)
print("=" * 60, flush=True)

# 看模型是否在限时内自然 EOS（gen_tokens < TARGET 说明提前停了）
if gen_tokens < TARGET_TOKENS:
    print(f"  ⚠️ 模型在 {gen_tokens} token 自然 EOS（未到 32768 上限）", flush=True)
    print(f"  -> 模型自然 CoT 长度约 {gen_tokens} token，max_new_tokens 无需 32768", flush=True)
else:
    print(f"  模型写到 32768 上限未 EOS（或被时间截断）", flush=True)

# 保存结果
import json
result = {
    "gen_tokens": int(gen_tokens),
    "elapsed_s": round(dt, 1),
    "tok_per_s": round(tok_per_s, 1),
    "full_32768_est_min": round(full_est / 60, 1),
    "n8_dataset_est_h": round(30 * 8 * full_est / 3600, 1),
    "n16_dataset_est_h": round(30 * 16 * full_est / 3600, 1),
    "natural_eos": gen_tokens < TARGET_TOKENS,
}
json.dump(result, open("/root/autodl-tmp/eval/timing_flash_result.json", "w"), indent=2)
print(f"结果已存 /root/autodl-tmp/eval/timing_flash_result.json", flush=True)
