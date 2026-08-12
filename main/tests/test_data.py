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


# --------------------------- jsonl 真实数据（tokenizer 编码）---------------------------
def _jsonl_cfg(tmp_path, **ds_over):
    import os
    path = os.path.join(str(tmp_path), "data.jsonl")
    ds = {"type": "jsonl", "path": path, "max_prompt_len": 8,
          "max_response_len": 10, "tokenizer_path": "mock-tok"}
    ds.update(ds_over)
    cfg = dict(vocab_size=64, n_prompts=16, prompt_len=6, resp_len=8,
               student_path="student-tok", dataset=ds)
    return cfg, path


def _mock_tokenizer(monkeypatch):
    """mock transformers.AutoTokenizer：from_pretrained 返回实例，encode→伪 id 列表。"""
    import transformers
    class FakeTok:
        pad_token = "<pad>"
        pad_token_id = 0
        def __init__(self, *a, **k):
            pass
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()
        def encode(self, text, add_special_tokens=False, truncation=True,
                   max_length=None):
            # 伪 token：每个字符一个 id（1..len），超长截断
            n = len(text)
            if max_length is not None:
                n = min(n, max_length)
            return list(range(1, n + 1))
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTok)
    return FakeTok


def test_jsonl_loader_encodes_to_fixed_length(tmp_path, monkeypatch):
    """jsonl → tokenizer 编码 → 定长 (N,P)/(N,T) 张量（截断 + 右 pad）。"""
    import json
    _mock_tokenizer(monkeypatch)
    cfg, path = _jsonl_cfg(tmp_path)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(3):
            f.write(json.dumps({"prompt": "abc", "response": "defghijklmno"}) + "\n")  # resp 超长
    prompts, responses, reward_fn = JsonLinesDataLoader(cfg, "cpu").load()
    assert prompts.shape == (3, 8)          # max_prompt_len=8（"abc"→3 ids + 5 pad）
    assert responses.shape == (3, 10)       # max_response_len=10（超长截断）
    assert prompts.dtype == torch.long
    # 首行 prompt: ids[1,2,3] + pad(0)*5
    assert prompts[0].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    # response 截断到 10（文本 11 字符 → 截断 10）
    assert responses[0].tolist() == list(range(1, 11))
    # reward_fn 占位（HF 路径不用）：返回 0
    assert torch.equal(reward_fn(responses),
                       torch.zeros_like(responses, dtype=torch.float32))


def test_jsonl_loader_missing_path_raises(monkeypatch):
    cfg, _ = _jsonl_cfg(type("T", (), {"__truediv__": lambda s, o: str(o)})())
    cfg["dataset"]["path"] = "/nonexistent/x.jsonl"
    with pytest.raises(DataError):
        JsonLinesDataLoader(cfg, "cpu").load()


def test_jsonl_loader_skips_bad_lines(tmp_path, monkeypatch):
    """损坏行/缺字段行跳过，有效行才编码。"""
    import json
    _mock_tokenizer(monkeypatch)
    cfg, path = _jsonl_cfg(tmp_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("not-json\n")
        f.write(json.dumps({"prompt": "ok", "response": "yes"}) + "\n")
        f.write(json.dumps({"prompt": "only"}) + "\n")      # 缺 response → 跳过
        f.write(json.dumps({"prompt": "a", "response": "b"}) + "\n")
    prompts, responses, _ = JsonLinesDataLoader(cfg, "cpu").load()
    assert prompts.shape[0] == 2   # 只保留 2 个有效行