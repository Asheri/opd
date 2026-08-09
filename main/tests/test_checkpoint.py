"""checkpoint.py 单测：断点保存/加载/续跑。"""
import os

import torch

from fullstack_opd_v2.checkpoint import CheckpointManager
from fullstack_opd_v2.model import CausalToyLM


def _ckpt_dir(tmp_path):
    return os.path.join(str(tmp_path), "checkpoints")


def test_save_load_roundtrip(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    cm.save(10, m, version=20, cfg={"a": 1})
    ck = cm.load(cm.latest())
    assert ck["step"] == 10
    assert ck["version"] == 20
    assert ck["cfg"] == {"a": 1}
    # 恢复 state_dict
    m2 = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    m2.load_state_dict(ck["state"])
    assert torch.equal(m2.emb.weight, m.emb.weight)


def test_latest_picks_highest_step(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    cm.save(10, m, version=10, cfg={})
    cm.save(30, m, version=30, cfg={})
    latest = cm.latest()
    assert os.path.basename(latest) == "step_30.pt"


def test_every_throttle(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=3)
    for s in (1, 2, 3, 4, 5, 6):
        cm.save(s, m, version=s, cfg={})
    # 只存 step%3==0 的
    assert cm.latest() is not None
    assert os.path.basename(cm.latest()) == "step_6.pt"


def test_resume(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    cm.save(15, m, version=15, cfg={"seed": 7})
    ck = CheckpointManager(str(tmp_path), every=1).resume()
    assert ck["step"] == 15
    assert ck["cfg"] == {"seed": 7}


def test_manager_constructed_with_dir_but_loads_file(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    p = cm.save(3, m, version=3, cfg={}, force=True)
    assert p is not None
    # 镜像 cli eval 调用形态：run_dir 传 "."，checkpoint_dir 显式给文件父目录
    cm2 = CheckpointManager(".", checkpoint_dir=os.path.dirname(p))
    ck = cm2.load(p)                                 # 文件路径 load
    assert ck["step"] == 3