import torch, sys, json
sys.path.insert(0, "/root/opd/main")
from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.model_factory import build_model

def gb(b): return f"{b/2**30:.2f}"

def show(tag):
    print(f"[{tag}] alloc={gb(torch.cuda.memory_allocated())}GiB "
          f"reserved={gb(torch.cuda.memory_reserved())}GiB", flush=True)

torch.cuda.set_device(0)
cfg = load_config(path="/root/opd/main/configs/skywork_17b.yaml")
print("model_kind:", cfg.get("model_kind"), "dtype:", cfg.get("dtype"),
      "student_path:", cfg.get("student_path"), flush=True)
st = build_model(cfg, "cuda:0", role="student")
st.train()
attn = getattr(st.model.config, "_attn_implementation", "UNKNOWN")
print("attn_implementation (student.model.config):", attn, flush=True)
print("flash available:", torch.backends.cuda.flash_sdp_enabled(),
      "| model dtype:", next(st.model.parameters()).dtype, flush=True)
show("after load")

B, P, T = 2, 1024, 2048
full = torch.randint(0, st.vocab, (B, P + T), device="cuda:0")
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    logp = torch.log_softmax(st(full), dim=-1)   # (B, P+T, V) 带梯度
    show("after forward logits")
    loss = logp.logsumexp(dim=-1).mean()         # 模拟 loss
    loss.backward()
show("after backward")
print("logits shape:", tuple(logp.shape), "dtype:", logp.dtype, flush=True)