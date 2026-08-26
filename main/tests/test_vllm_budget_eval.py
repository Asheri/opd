"""vllm_budget_eval.py 单测：_aggregate_budget 聚合正确性（不依赖 vLLM/GPU）。

协议与 budget_eval 一致：outcome=预算内自然产出正确最终答案（sympy 判定）；status 按
finish_reason（stop=eos / length=budget_stop）；reasoning_tokens=生成 token 数。
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from vllm_budget_eval import (_aggregate_budget, _apply_cuda_visible, _merge_all_results,
                              build_prompts, parse_args)
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


def test_build_prompts_chat_preserves_order():
    """多条 problems 逐条包裹、顺序保持，每条 content=format_prompt 结果。"""
    tok = _FakeTok()
    problems = [("q0", "a0"), ("q1", "a1"), ("q2", "a2")]
    got = build_prompts(problems, "boxed", tok=tok)
    assert len(got) == 3
    for i, (q, _) in enumerate(problems):
        bare = format_prompt(q, "boxed")
        assert got[i] == "<|im_start|>user\n" + bare + "<|im_end|>\n<|im_start|>assistant\n"
    assert [m[0]["content"] for m, _ in tok.calls] == \
        [format_prompt(q, "boxed") for q, _ in problems]


def test_parse_args_chat_template_default_off():
    """默认 --chat-template=False、--tokenizer=None（零回归关键：默认裸 prompt）。"""
    a = parse_args(["--models", "Base=/x", "--out-dir", "/tmp/x"])
    assert a.chat_template is False
    assert a.tokenizer is None


def test_parse_args_chat_template_on():
    a = parse_args(["--models", "Base=/x", "--out-dir", "/tmp/x", "--chat-template"])
    assert a.chat_template is True


def test_parse_args_tokenizer_explicit():
    a = parse_args(["--models", "Base=/x", "--out-dir", "/tmp/x",
                    "--tokenizer", "/tok/here"])
    assert a.tokenizer == "/tok/here"


def test_apply_cuda_visible_cuda(monkeypatch):
    """--device cuda:i → CUDA_VISIBLE_DEVICES=i（vLLM 选卡唯一途径，双卡并行前提）。"""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _apply_cuda_visible("cuda:1") == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert _apply_cuda_visible("cuda:0") == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_apply_cuda_visible_non_cuda(monkeypatch):
    """非 cuda:N（"cpu"/None）不改环境，避免误伤。"""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _apply_cuda_visible("cpu") is None
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert _apply_cuda_visible(None) is None
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert _apply_cuda_visible("cuda:") is None


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


def test_aggregate_eos_no_answer():
    """eos 自然停止但未给出 \\boxed 答案：eos_rate=1 但 no_answer=1、accuracy=0。
    对照 H9：no_answer 与 eos 解耦——训练学生长推理在 B512 截断前 eos=0 才被低估。"""
    problems = [("p0", "42")]
    outs = [_Out("reasoning only, no boxed answer", [1, 2, 3], "stop")]
    r = _aggregate_budget(problems, outs, 256, "E2")
    assert r["eos_rate"] == 1.0
    assert r["no_answer_rate"] == 1.0
    assert r["accuracy"] == 0.0
    assert r["rows"][0]["status"] == "eos"


def test_aggregate_budget_stop_with_answer():
    """预算截断（length）但已产出 \\boxed 答案：budget_stop 也能得分、no_answer=0。
    对照 H9：截断 ≠ 无答案——Base 用捷径在截断前给出答案故 B512 不伤，学生被截断于推理中。"""
    problems = [("p0", "42")]
    outs = [_Out("short steps\\n\\\\boxed{42} then cut", [1, 2, 3, 4, 5], "length")]
    r = _aggregate_budget(problems, outs, 256, "E2")
    assert r["budget_stop_rate"] == 1.0
    assert r["no_answer_rate"] == 0.0
    assert r["accuracy"] == 1.0
    assert r["rows"][0]["status"] == "budget_stop"


def test_aggregate_empty():
    r = _aggregate_budget([], [], 256, "X")
    assert r["n"] == 0
    assert r["accuracy"] == 0.0


def test_merge_all_results_preserves_other_labels(tmp_path):
    """已有 S120/B512，新写 S200/B512 -> 两条共存（同 out-dir 多次调用不丢旧结果）。"""
    (tmp_path / "all_results.json").write_text(
        json.dumps([{"label": "S120", "budget": 512, "accuracy": 0.1}]),
        encoding="utf-8")
    merged = _merge_all_results(str(tmp_path),
                                [{"label": "S200", "budget": 512, "accuracy": 0.2}])
    assert [(r["label"], r["budget"]) for r in merged] == [("S120", 512), ("S200", 512)]


def test_merge_all_results_same_key_overwrite(tmp_path):
    """已有 S120/B512 acc=0.1，重跑同键 acc=0.2 -> 只剩 0.2（幂等，不累积）。"""
    (tmp_path / "all_results.json").write_text(
        json.dumps([{"label": "S120", "budget": 512, "accuracy": 0.1}]),
        encoding="utf-8")
    merged = _merge_all_results(str(tmp_path),
                                [{"label": "S120", "budget": 512, "accuracy": 0.2}])
    assert len(merged) == 1
    assert merged[0]["accuracy"] == 0.2


def test_merge_all_results_corrupted_file(tmp_path):
    """已有文件非法 JSON -> 静默视为空，返回仅含新结果（不崩评估）。"""
    (tmp_path / "all_results.json").write_text("{not valid json", encoding="utf-8")
    merged = _merge_all_results(str(tmp_path),
                                [{"label": "E1", "budget": 2048, "accuracy": 0.3}])
    assert len(merged) == 1
    assert merged[0]["label"] == "E1"
    assert merged[0]["budget"] == 2048
