"""vllm_budget_eval.py 单测：_aggregate_budget 聚合正确性（不依赖 vLLM/GPU）。

协议与 budget_eval 一致：outcome=预算内自然产出正确最终答案（sympy 判定）；status 按
finish_reason（stop=eos / length=budget_stop）；reasoning_tokens=生成 token 数。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from vllm_budget_eval import _aggregate_budget, build_prompts
from fullstack_opd_v2.budget_eval import wrap_chat, format_prompt


class _FakeTok:
    """fake tokenizer：记录 apply_chat_template 调用参数，返回 Qwen 系 chat 包裹文本。"""
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        return ("<|im_start|>user\n" + messages[0]["content"]
                + "<|im_end|>\n<|im_start|>assistant\n")


def test_wrap_chat_format():
    """wrap_chat 用 apply_chat_template 把文本作为 user 消息包裹，add_generation_prompt=True。"""
    tok = _FakeTok()
    out = wrap_chat("Q?", tok)
    assert out == "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n"
    msgs, kw = tok.calls[0]
    assert msgs == [{"role": "user", "content": "Q?"}]
    assert kw["add_generation_prompt"] is True
    assert kw["tokenize"] is False


def test_build_prompts_bare():
    """build_prompts 裸路径（tok=None）：与 format_prompt 一致（零回归）。"""
    problems = [("p0", "42"), ("p1", "7")]
    got = build_prompts(problems, "boxed")
    assert got == [format_prompt(p, "boxed") for p, _ in problems]
    assert got[0].startswith("p0")
    assert got[1].startswith("p1")


def test_build_prompts_chat():
    """build_prompts 传 fake tok：每条 prompt 被 apply_chat_template 包裹（user 消息）。"""
    tok = _FakeTok()
    problems = [("p0", "42")]
    got = build_prompts(problems, "boxed", tok=tok)
    bare = format_prompt("p0", "boxed")
    assert len(got) == 1
    assert got[0] == "<|im_start|>user\n" + bare + "<|im_end|>\n<|im_start|>assistant\n"
    msgs, kw = tok.calls[0]
    assert msgs == [{"role": "user", "content": bare}]
    assert kw["add_generation_prompt"] is True
    assert kw["tokenize"] is False


class _Out:
    """mock vLLM RequestOutput（最小字段：text / token_ids / finish_reason）。"""
    class _Comp:
        def __init__(self, text, toks, fr):
            self.text = text
            self.token_ids = toks
            self.finish_reason = fr

    def __init__(self, text, toks, fr):
        self.outputs = [self._Comp(text, toks, fr)]


def test_aggregate_eos_correct():
    problems = [("p0", "42")]
    outs = [_Out("reasoning\\n\\\\boxed{42}", [1, 2, 3], "stop")]
    r = _aggregate_budget(problems, outs, 256, "E1")
    assert r["accuracy"] == 1.0
    assert r["eos_rate"] == 1.0
    assert r["budget_stop_rate"] == 0.0
    assert r["no_answer_rate"] == 0.0
    assert r["n"] == 1
    assert r["avg_reasoning_tokens"] == 3
    assert r["rows"][0]["status"] == "eos"


def test_aggregate_budget_stop_incorrect():
    problems = [("p0", "42")]
    outs = [_Out("no answer here", [1, 2, 3, 4, 5], "length")]
    r = _aggregate_budget(problems, outs, 256, "E1")
    assert r["accuracy"] == 0.0
    assert r["budget_stop_rate"] == 1.0
    assert r["no_answer_rate"] == 1.0
    assert r["avg_reasoning_tokens"] == 5
    assert r["rows"][0]["status"] == "budget_stop"


def test_aggregate_mixed():
    problems = [("p0", "42"), ("p1", "7")]
    outs = [_Out("steps\\n\\\\boxed{42}", [1, 2, 3], "stop"),
            _Out("wrong tail", [1, 2, 3, 4], "length")]
    r = _aggregate_budget(problems, outs, 512, "Base")
    assert r["accuracy"] == 0.5
    assert r["eos_rate"] == 0.5
    assert r["budget_stop_rate"] == 0.5
    assert r["no_answer_rate"] == 0.5
    assert r["avg_reasoning_tokens"] == 3.5


def test_aggregate_sympy_equivalence():
    # sympy 数学等价：\boxed{0.5} vs "1/2"（same value）
    problems = [("p0", "1/2")]
    outs = [_Out("simplify\\n\\\\boxed{0.5}", [7], "stop")]
    r = _aggregate_budget(problems, outs, 256, "E2")
    assert r["accuracy"] == 1.0


def test_aggregate_empty():
    r = _aggregate_budget([], [], 256, "X")
    assert r["n"] == 0
    assert r["accuracy"] == 0.0
