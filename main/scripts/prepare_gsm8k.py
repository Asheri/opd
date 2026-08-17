"""准备 GSM8K 数学数据 → jsonl（prompt=question, response=answer），供 OPD 真实训练。"""
import json, os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
N = int(os.environ.get("GSM8K_N", "2000"))
out = os.environ.get("GSM8K_OUT", "/root/autodl-tmp/data/gsm8k.jsonl")
from datasets import load_dataset
print("加载 GSM8K（hf-mirror）...")
ds = load_dataset("openai/gsm8k", "main", split=f"train[:{N}]")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps({"prompt": row["question"], "response": row["answer"]}, ensure_ascii=False) + "\n")
print(f"OK: {len(ds)} 条 -> {out}")
