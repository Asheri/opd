"""vllm_budget_eval.py 单测：_aggregate_budget 聚合正确性（不依赖 vLLM/GPU）。

协议与 budget_eval 一致：outcome=预算内自然产出正确最终答案（sympy 判定）；status 按
finish_reason（stop=eos / length=budget_stop）；reasoning_tokens=生成 token 数。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from vllm_budget_eval import _aggregate_budget


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
