"""data.py 单测：可插拔数据接口，Toy 默认实现与旧 _make_toy_data 同源。"""
import torch

import pytest

from fullstack_opd_v2.data import DataLoader, ToyDataLoader, JsonLinesDataLoader, build_data_loader
from fullstack_opd_v2.exceptions import DataError


def _cfg(**over):
    cfg = dict(vocab_size=64, n_prompts=16, prompt_len=6, resp_len=8,
               dataset={"type": "toy"})
    cfg.update(over)
    return cfg


def test_toy_dataloader_shapes():
    prompts, responses, reward_fn = ToyDataLoader(_cfg(), "cpu").load()
    assert prompts.shape == (16, 6)
    assert responses.shape == (16, 8)
    assert prompts.dtype == torch.long


def test_toy_reward_fn_matches_lookup():
    _, _, reward_fn = ToyDataLoader(_cfg(), "cpu").load()
    r = torch.tensor([[0, 1, 2, 3]])
    expected = torch.full((4,), -0.2)
    expected[0::2] = 1.0
    assert torch.equal(reward_fn(r), expected.unsqueeze(0))


def test_toy_deterministic_same_stream():
    a = ToyDataLoader(_cfg(), "cpu").load()
    b = ToyDataLoader(_cfg(), "cpu").load()
    assert torch.equal(a[0], b[0])   # 同 seed 同流 → 数据一致


def test_toy_dataloader_caches_load():
    """C4：第二次 load 返回同一对象（缓存）。"""
    dl = ToyDataLoader(_cfg(), "cpu")
    a = dl.load(); b = dl.load()
    assert a[0] is b[0]      # 同一张量对象
    assert a[1] is b[1]


def test_build_loader_dispatches():
    assert isinstance(build_data_loader(_cfg(), "cpu"), ToyDataLoader)
    assert isinstance(build_data_loader(_cfg(dataset={"type": "jsonl", "path": "x.jsonl"}), "cpu"),
                      JsonLinesDataLoader)


def test_build_loader_unknown_raises():
    with pytest.raises(DataError):
        build_data_loader(_cfg(dataset={"type": "nope"}), "cpu")


def test_dataloader_is_abstract():
    with pytest.raises(TypeError):
        DataLoader()   # abstractmethod 未实现