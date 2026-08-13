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
