import torch, sys
sys.path.insert(0, "/root/opd/main")
from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.model_factory import build_model
from fullstack_opd_v2.cache_store import load_cache_metadata, DiskTeacherCache
from fullstack_opd_v2.data import JsonLinesDataLoader
from fullstack_opd_v2.model import response_dists
from fullstack_opd_v2.losses import pg_loss

torch.cuda.set_device(0)
cfg = load_config(path="/root/opd/main/configs/skywork_17b.yaml")
dtype = torch.bfloat16
st = build_model(cfg, "cuda:0", role="student")
st.train()
cache_path = "/root/autodl-tmp/cache_skywork_17b.pt"
meta = load_cache_metadata(cache_path)
cache = DiskTeacherCache(cache_path, device="cuda:0",
                         top_k=int(meta["top_k"]), vocab=int(meta["vocab"]))
dl = JsonLinesDataLoader(cfg)
prompts, responses, _ = dl.load()
prompts = prompts.cuda(); responses = responses.cuda()
N = prompts.size(0)
print(f"N={N} prompt={tuple(prompts.shape)} resp={tuple(responses.shape)} "
      f"top_k={meta['top_k']} model_vocab={st.vocab} dtype={st.model.dtype}", flush=True)

def show(tag):
    print(f"[{tag}] alloc={torch.cuda.memory_allocated()/2**30:.1f}GiB "
          f"reserved={torch.cuda.memory_reserved()/2**30:.1f}GiB", flush=True)

show("after load")
idxs = torch.randint(0, N, (2,))
p_b = prompts[idxs]; r_b = responses[idxs]
with torch.no_grad():
    s_old = response_dists(st, p_b, r_b)   # 模拟 worker 快照 (2,T,V) fp32
show("after s_old(worker)")

with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
    s_cur = response_dists(st, p_b, r_b)
    show("after s_cur forward (logits fp32)")
    s_cur_b = s_cur.to(dtype)
    show("after s_cur->bf16")
    s_old = s_old.to("cuda:0", dtype=s_cur_b.dtype)
    p_old = s_old.exp()
    show("after p_old")
    s_topk = torch.topk(s_cur_b, int(meta["top_k"]), dim=-1)
    delta_d = cache.delta_for_student_topk(idxs.cuda(), s_topk.indices,
                                           vocab_out=st.vocab)
    show("after delta_for_student_topk")
    delta_d = delta_d.to(dtype)
    pg_support = torch.zeros_like(delta_d, dtype=torch.bool)
    pg_support.scatter_(-1, s_topk.indices, True)
    loss_pg = pg_loss(s_cur_b, s_old, delta_d, None, 0.2, p_old=p_old,
                      log_ratio_max=30, renormalize_support=True,
                      support=pg_support, delta_clip=2.0)
    show("after pg_loss")
    loss = loss_pg
    loss.backward()
    show("after backward")
print("s_cur forward dtype:", s_cur.dtype, "shape:", tuple(s_cur.shape), flush=True)