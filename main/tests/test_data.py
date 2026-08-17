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
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            # 伪模板：包成 "||user:CONTENT||assistant:"，长度变化可被测试捕获
            content = msgs[0]["content"]
            s = "||user:" + content + "||assistant:"
            return s if not tokenize else self.encode(s)
        def decode(self, ids):
            # 伪解码：id i → 第 i 个字符（id 与 encode 的 "每字符一个 id" 伪规则一致）
            import string
            alphabet = string.printable
            return "".join(alphabet[i] if i < len(alphabet) else "?" for i in ids)
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


def test_jsonl_loader_apply_chat_template(tmp_path, monkeypatch):
    """2026-08-17 根因：apply_chat_template=true 时 prompt 先套 chat 模板再编码
    （Qwen 裸 prompt 生成乱码+loop）。验证模板包裹改变了编码输入。"""
    import json
    _mock_tokenizer(monkeypatch)
    cfg, path = _jsonl_cfg(tmp_path, apply_chat_template=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "abc", "response": "def"}) + "\n")
    prompts, _, _ = JsonLinesDataLoader(cfg, "cpu").load()
    # 模板 "||user:abc||assistant:" = 22 字符 → 22 ids，截断到 max_prompt_len=8
    # 与未套模板（"abc"=3 ids）不同 → 模板确实生效
    assert prompts.shape == (1, 8)
    assert prompts[0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]  # 前 8 个模板字符 id


def test_jsonl_loader_raw_prompt_texts(tmp_path, monkeypatch):
    """C3：JsonLinesDataLoader 暴露与 prompts 行对齐的原始 prompt 文本（未套模板）。"""
    import json
    _mock_tokenizer(monkeypatch)
    cfg, path = _jsonl_cfg(tmp_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "abc", "response": "def"}) + "\n")
        f.write(json.dumps({"prompt": "xyz", "response": "uvw"}) + "\n")
    dl = JsonLinesDataLoader(cfg, "cpu")
    prompts, _, _ = dl.load()
    assert dl.raw_prompt_texts == ["abc", "xyz"]
    assert prompts.shape[0] == 2


def test_build_teacher_prompts_applies_own_template(monkeypatch):
    """C3：build_teacher_prompts 用教师自己的 tokenizer+模板编码（区别于学生 raw）。"""
    import torch
    import transformers
    class FakeTok:
        pad_token = "<pad>"
        pad_token_id = 0
        def __init__(self, *a, **k):
            pass
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "||T:" + msgs[0]["content"] + "||A:"
        def encode(self, text, add_special_tokens=False, truncation=True,
                   max_length=None):
            n = len(text)
            if max_length is not None:
                n = min(n, max_length)
            return list(range(1, n + 1))
        def decode(self, ids):
            import string
            alphabet = string.printable
            return "".join(alphabet[i] if i < len(alphabet) else "?" for i in ids)
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTok)
    from fullstack_opd_v2.data import build_teacher_prompts
    out = build_teacher_prompts(["abc"], "teacher-path", P=8)
    assert out.shape == (1, 8)
    # "||T:abc||A:" = 11 字符 → 11 ids，截断到 8：前 8 个模板字符 id
    assert out[0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert out.dtype == torch.long


def test_jsonl_loader_template_truncation_keeps_generation_marker(tmp_path, monkeypatch):
    """C3/max_prompt_len：✓ 长题干 + 模板 == 先截题干、保留 assistant 生成标记尾部。

    裸右截断会先切模板尾部（无 assistant 上下文 → 退化生成）；内容优先截断保证
    模板结构完整。用可区分 marker 的 tokenizer 精确断言首尾 id。
    """
    import json
    import transformers
    U, A = 900, 901      # user 开启 / assistant 生成标记

    class MarkerTok:
        pad_token = "<pad>"
        pad_token_id = 0
        def __init__(self, *a, **k):
            pass
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return chr(U) + msgs[0]["content"] + chr(A)
        def encode(self, text, add_special_tokens=False, truncation=True,
                   max_length=None):
            ids = [U if ch == chr(U) else (A if ch == chr(A) else ord(ch))
                   for ch in text]
            if max_length is not None:
                ids = ids[:max_length]
            return ids
        def decode(self, ids):
            return "".join(chr(i) if i not in (U, A)
                           else (chr(U) if i == U else chr(A)) for i in ids)

    monkeypatch.setattr(transformers, "AutoTokenizer", MarkerTok)
    cfg, path = _jsonl_cfg(tmp_path, apply_chat_template=True, max_prompt_len=20)
    long_q = "abcdefghijklmnopqrstuvwxyz0123"     # 30 字符 > 截断预算
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": long_q, "response": "rs"}) + "\n")
    prompts, _, _ = JsonLinesDataLoader(cfg, "cpu").load()
    row = prompts[0].tolist()
    assert len(row) == 20
    assert row[0] == U          # 模板开头保留
    assert row[-1] == A         # assistant 生成标记保留（未被右截断切掉）
    assert row[1:9] == [ord(c) for c in "abcdefgh"]   # 题干前段保留
    assert row[1:-1] == [ord(c) for c in "abcdefghijklmnopqr"]  # 题干截到 18 字符
    assert row[1] != 0          # 非 pad


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