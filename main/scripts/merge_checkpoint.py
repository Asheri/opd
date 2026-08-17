"""把 OPD 训练断点（state_dict）合并回基座 HF 模型 → save_pretrained 目录。
用法: python merge_checkpoint.py <ckpt.pt> <base_model_dir> <out_dir>"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
base, out = sys.argv[2], sys.argv[3]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
missing, unexpected = model.load_state_dict(ck["state"], strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))
if unexpected:
    print("unexpected 前5:", unexpected[:5])
model.save_pretrained(out)
tok = AutoTokenizer.from_pretrained(base)
tok.save_pretrained(out)
print("saved ->", out)
