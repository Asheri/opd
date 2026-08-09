"""eval_aime.py 单测：纯函数答案提取/规范化/提示格式化 + 评估器（mock 模型/数据集）。"""
import unittest.mock as mock

import pytest

from fullstack_opd_v2.eval_aime import (
    AimeEvaluator, AimeResult, extract_answer, normalize_answer,
    format_prompt, AIME_DATASETS,
)
from fullstack_opd_v2.exceptions import DataError, ModelError


# --------------------------- 纯函数 ---------------------------
def test_format_prompt_contains_boxed():
    p = format_prompt("What is 2+2?")
    assert "What is 2+2?" in p
    assert "\\boxed{}" in p


def test_extract_answer_boxed():
    assert extract_answer("Answer: \\boxed{42}") == "42"
    assert extract_answer("思考...\\boxed{005}\\n") == "005"
    assert extract_answer("\\boxed{12} and \\boxed{34}") == "34"   # 取最后一个 boxed


def test_extract_answer_nested_boxed():
    # 嵌套括号（如 \boxed{\frac{1}{2}}）→ 取第一个数字
    assert extract_answer("\\boxed{\\frac{1}{2}}") == "1"


def test_extract_answer_last_number_fallback():
    assert extract_answer("最终答案是 7。") == "7"
    assert extract_answer("没有答案") == ""


def test_normalize_answer_handles_leading_zeros():
    assert normalize_answer("005") == 5
    assert normalize_answer(5) == 5
    assert normalize_answer("1,234") == 1234
    assert normalize_answer("abc") is None


def test_aime_dataset_aliases():
    assert AIME_DATASETS["AIME24"] == "Maxwell-Jia/AIME_2024"
    assert AIME_DATASETS["AIME25"] == "yentinglin/aime_2025"


# --------------------------- 评估器（mock 后端） ---------------------------
def _fake_evaluator(responses, problems):
    """构造不加载真实模型的 AimeEvaluator（只 stub 生成/数据）。"""
    ev = object.__new__(AimeEvaluator)
    ev.model_path = "fake"
    ev.device = "cpu"
    ev.max_new_tokens = 16
    ev.batch_size = 8
    ev.n_samples = 1
    ev.temperature = 0.0
    ev.tok = mock.Mock()
    ev.model = mock.Mock()
    it = iter(responses)
    ev.generate = lambda prompts: [next(it) for _ in prompts]
    return ev


def test_evaluate_scores_accuracy():
    ev = _fake_evaluator(
        responses=["\\boxed{42}", "\\boxed{7}", "I think 100"],
        problems=[("p0", "42"), ("p1", "7"), ("p2", "5")])
    ev.load_problems = lambda d: [("p0", "42"), ("p1", "7"), ("p2", "5")]
    res = ev.evaluate("AIME24")
    assert isinstance(res, AimeResult)
    assert res.total == 3
    assert res.correct == 2          # 42✓ 7✓ 100✗(≠5)
    assert abs(res.accuracy - 2 / 3) < 1e-9
    assert res.rows[0]["correct"] is True
    assert res.rows[2]["correct"] is False


def test_evaluate_to_jsonl_writes_rows(tmp_path):
    ev = _fake_evaluator(responses=["\\boxed{9}"], problems=[("p", "9")])
    ev.load_problems = lambda d: [("p", "9")]
    out = tmp_path / "aime.jsonl"
    res = ev.evaluate_to_jsonl("AIME24", str(out))
    assert res.correct == 1
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    import json
    assert json.loads(lines[0])["correct"] is True


def test_resolve_dataset():
    ev = object.__new__(AimeEvaluator)
    assert ev.resolve_dataset("AIME24") == "Maxwell-Jia/AIME_2024"
    assert ev.resolve_dataset("hf:custom/ds") == "hf:custom/ds"


def test_missing_model_path_raises():
    # 构造缺 tok/model 的实例，generate 抛 AttributeError 是可接受的失败面；
    # 关键：AimeEvaluator 构造需真实模型，缺失时 ModelError（由 CLI 层 catch）。
    with pytest.raises(ModelError):
        # 构造会尝试 from_pretrained（无真实模型）→ 应抛 ModelError/OSError 族
        AimeEvaluator("/nonexistent/model_xyz", device="cpu")


def test_load_problems_missing_columns(monkeypatch):
    ev = object.__new__(AimeEvaluator)
    ev.resolve_dataset = lambda d: d
    fake_ds = [{"problem": "p", "answer": "1"}, {"question": "q"}]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: fake_ds)
    with pytest.raises(DataError):
        ev.load_problems("AIME24")