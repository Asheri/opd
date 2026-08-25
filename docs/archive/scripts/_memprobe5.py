import torch, sys
sys.path.insert(0, "/root/opd/main")
from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.model_factory import build_model
from fullstack_opd_v2.data import JsonLinesDataLoader
from fullstack_opd_v2.cache_store import DiskTeacherCache, load_cache_metadata
from fullstack_opd_v2.scheduler import AsyncBatchedScheduler
from fullstack_opd_v2.model import response_dists

def gb(b):
    return f"{b/2**30:.1f}"

def show(tag):
    print(f"[{tag}] alloc={gb(torch.cuda.memory_allocated())}GiB "
          f"reserved={gb(torch.cuda.memory_reserved())}GiB", flush=True)

torch.cuda.set_device(0)
cfg = load_config(path="configs/skywork_17b.yaml")
st = build_model(cfg, "cuda:0", role="student")
dl = JsonLinesDataLoader(cfg)
prompts, responses, _ = dl.load()
prompts = prompts.cuda()
responses = responses.cuda()
meta = load_cache_metadata("/root/autodl-tmp/cache_skywork_17b.pt")
cache = DiskTeacherCache("/root/autodl-tmp/cache_skywork_17b.pt",
                         device="cuda:0", top_k=meta["top_k"], vocab=meta["vocab"])
s2cfg = dict(cfg["stage2"])
s2cfg["model_kind"] = "hf"
s2cfg["student_path"] = cfg["student_path"]
sched = AsyncBatchedScheduler(st, cache, prompts, responses,
                              None, None, None, s2cfg, "cuda:0")
show("built")
for i in range(6):
    idxs = torch.randint(0, 500, (4,))
    with torch.no_grad():
        s_old = response_dists(sched.worker, prompts[idxs], responses[idxs])
    show(f"after rollout itr {i}")
    del s_old
    torch.cuda.empty_cache()
show("after gc")