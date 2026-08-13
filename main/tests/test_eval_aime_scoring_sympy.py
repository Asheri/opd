"""论文评分协议（Direct-OPD ttrl_math fast 路径）单测。

覆盖 eval_aime.py 新增的 scoring='sympy'：\boxed{} 级联提取 +
grade_answer_mathd / grade_answer_sympy 数学等价判定。
依赖 sympy + pylatexenc（论文评分依赖）。
"""
import pytest

from fullstack_opd_v2.eval_aime import (
    _extract_boxed_answer, _grade_answer_mathd, _grade_answer_sympy,
    _norm_sympy, _boxed_last,
)


@pytest.fixture(scope="module", autouse=True)
def _require_scoring_deps():
    try:
        import sympy  # noqa: F401
        import pylatexenc  # noqa: F401
    except Exception as e:      # pragma: no cover
        pytest.skip(f"论文评分依赖缺失：{e}")


# ---------- \boxed{} 级联提取 ----------
def test_boxed_last_extracts_final():
    assert _boxed_last(r"思考 \boxed{42}") == r"\boxed{42}"
    assert _boxed_last(r"第一个\boxed{12} 最后\boxed{34}") == r"\boxed{34}"
    assert _boxed_last(r"\fbox{7}") == r"\boxed{7}"          # fbox 兼容
    assert _boxed_last("无 box") is None


def test_extract_boxed_answer_content():
    assert _extract_boxed_answer(r"\boxed{42}") == "42"
    assert _extract_boxed_answer(r"\boxed{\frac{25}{8}}") == r"\frac{25}{8}"
    assert _extract_boxed_answer(r"前\boxed{1} 后\boxed{204}") == "204"
    assert _extract_boxed_answer("无答案") is None


# ---------- 数学等价判定（论文 grade_answer_sympy / mathd） ----------
def test_sympy_fraction_equals_decimal():
    # 论文真实场景：gt='25/8'，模型答 \frac{25}{8}
    assert _grade_answer_sympy(r"\frac{25}{8}", "25/8")
    assert _grade_answer_mathd(r"\frac{25}{8}", "25/8")


def test_sympy_simple_fraction_decimal():
    assert _grade_answer_sympy("3/4", "0.75")
    assert _grade_answer_sympy("0.75", "3/4")


def test_sympy_integer_exact():
    assert _grade_answer_sympy("204", "204")
    assert not _grade_answer_sympy("204", "205")


def test_sympy_mixed_number():
    # 7 3/4 = 31/4
    assert _grade_answer_sympy("7 3/4", "31/4")


def test_sympy_leading_zeros_equivalent():
    # 归一化后 005 -> 5
    assert _norm_sympy("005") == "5"
    assert _grade_answer_sympy("005", "5")


def test_sympy_mathd_string_equal():
    assert _grade_answer_mathd("1/2", "1/2")
    assert not _grade_answer_mathd("1/2", "1/3")


# ---------- chat_template 对齐论文 verl（apply_chat_template 包裹） ----------
def test_chat_template_wraps_prompt(monkeypatch):
    """chat_template=True 时 generate 用 apply_chat_template 包裹每个 prompt。"""
    import unittest.mock as mock
    from fullstack_opd_v2.eval_aime import AimeEvaluator
    ev = object.__new__(AimeEvaluator)
    ev.model_path = "fake"
    ev.device = "cpu"
    ev.max_new_tokens = 16
    ev.max_ctx = 4096
    ev.batch_size = 8
    ev.n_samples = 1
    ev.temperature = 0.0
    ev.top_p = None
    ev.metric = "pass1"
    ev.prompt_style = "boxed"
    ev.scoring = "int"
    ev.chat_template = True
    # fake tokenizer：apply_chat_template 把 user 消息包成 <|im_start|>user/assistant 标记
    ev.tok = mock.Mock()
    ev.tok.pad_token_id = 0
    ev.tok.pad_token = "<pad>"
    ev.tok.padding_side = "left"
    ev.tok.apply_chat_template.side_effect = (
        lambda msgs, **kw: "<|im_start|>user\n" + msgs[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"
    )
    # fake model.generate：验证收到的 input 已包裹
    captured = {}
    def fake_generate(**kw):
        captured["input_ids"] = kw["input_ids"]
        import torch
        return torch.zeros(1, 5, dtype=torch.long)  # 1 序列 × 5 token
    ev.model = mock.Mock()
    ev.model.generate.side_effect = fake_generate
    # fake tokenize：记录输入文本
    import torch
    def fake_tok_call(batch, **kw):
        captured["texts"] = batch
        return {"input_ids": torch.zeros(len(batch), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(batch), 4, dtype=torch.long)}
    ev.tok.side_effect = fake_tok_call
    out = ev.generate(["question?"])
    assert captured["texts"][0].startswith("<|im_start|>user\n")
    assert "question?" in captured["texts"][0]
    assert "<|im_start|>assistant" in captured["texts"][0]


def test_chat_template_off_keeps_raw(monkeypatch):
    """chat_template=False（默认）时不做包裹，保持原裸字符串。"""
    import unittest.mock as mock
    from fullstack_opd_v2.eval_aime import AimeEvaluator
    ev = object.__new__(AimeEvaluator)
    ev.model_path = "fake"
    ev.device = "cpu"
    ev.max_new_tokens = 16
    ev.max_ctx = 4096
    ev.batch_size = 8
    ev.n_samples = 1
    ev.temperature = 0.0
    ev.top_p = None
    ev.metric = "pass1"
    ev.prompt_style = "boxed"
    ev.scoring = "int"
    ev.chat_template = False
    ev.tok = mock.Mock()
    ev.tok.pad_token_id = 0
    import torch
    ev.tok.side_effect = lambda batch, **kw: {"input_ids": torch.zeros(len(batch), 4, dtype=torch.long),
                                               "attention_mask": torch.ones(len(batch), 4, dtype=torch.long)}
    captured = {}
    ev.model = mock.Mock()
    ev.model.generate.side_effect = lambda **kw: torch.zeros(1, 5, dtype=torch.long)
    ev.generate(["raw question?"])
    assert ev.tok.apply_chat_template.call_count == 0  # 未调用包裹
