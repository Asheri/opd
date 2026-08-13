"""eval_aime.py 单测：纯函数答案提取/规范化/提示格式化 + 评估器（mock 模型/数据集）。"""
import unittest.mock as mock

import pytest

from fullstack_opd_v2.eval_aime import (
    AimeEvaluator, AimeResult, extract_answer, normalize_answer,
    format_prompt, AIME_DATASETS,
)
from fullstack_opd_v2.exceptions import DataError, ModelError, ConfigError


# --------------------------- 构造参数校验 ---------------------------
def test_guard_n_samples_gt1_requires_temp_gt0(monkeypatch):
    """P2：n_samples>1 且 temperature<=0 → ConfigError（贪心多序列逐字重复）。

    参数校验已前置到 transformers 导入之前（配置错快速失败），非法组合不触模型。
    合法组合用 monkeypatch 挡掉 transformers 模块属性（`from transformers import X`
    在函数内执行时才解析，patch 后即生效），避免真实模型加载。
    """
    import transformers
    fake_tok = mock.Mock()
    fake_tok.pad_token = None
    fake_tok.eos_token = "<eos>"
    fake_model = mock.Mock()
    monkeypatch.setattr(transformers, "AutoTokenizer",
                        mock.Mock(from_pretrained=mock.Mock(return_value=fake_tok)))
    monkeypatch.setattr(transformers, "AutoModelForCausalLM",
                        mock.Mock(from_pretrained=mock.Mock(return_value=fake_model)))

    with pytest.raises(ConfigError):
        AimeEvaluator("fake-model", n_samples=2, temperature=0.0)
    with pytest.raises(ConfigError):
        AimeEvaluator("fake-model", n_samples=4, temperature=-0.5)
    # 合法组合不抛：n==1 贪心 / n>1 且 T>0
    AimeEvaluator("fake-model", n_samples=1, temperature=0.0)
    AimeEvaluator("fake-model", n_samples=3, temperature=0.7)


# --------------------------- 纯函数 ---------------------------
def test_format_prompt_contains_boxed():
    p = format_prompt("What is 2+2?")
    assert "What is 2+2?" in p
    assert "\\boxed{}" in p


def test_format_prompt_dapo():
    """Direct-OPD 论文附录 A 模板：要求 "Answer:" 结尾行。"""
    p = format_prompt("What is 2+2?", style="dapo")
    assert "What is 2+2?" in p
    assert "Answer:" in p
    assert "without quotes" in p


def test_extract_answer_dapo_line():
    """DAPO 风格：取 "Answer:" 行后的数字（论文模板的答案落点）。"""
    assert extract_answer("step by step...\nAnswer:\n42", style="dapo") == "42"
    assert extract_answer("...\nAnswer: 7\n", style="dapo") == "7"
    # 无 Answer 行 → 回退最后一个数字
    assert extract_answer("最终 3", style="dapo") == "3"
    assert extract_answer("无答案", style="dapo") == ""


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
def _fake_evaluator(responses, problems, **over):
    """构造不加载真实模型的 AimeEvaluator（只 stub 生成/数据）。"""
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
    ev.tok = mock.Mock()
    ev.model = mock.Mock()
    for k, v in over.items():
        setattr(ev, k, v)
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

def test_evaluate_ave_metric_returns_per_problem_fraction():
    """metric=ave（对齐论文 ave@32）：accuracy = 每题 n 采样中答对比例的均值。

    3 题 × 2 采样：p0 全对(1.0)、p1 半对(0.5)、p2 全错(0.0) → ave = 0.5。
    pass@1 语义（correct）仍各自记录：p0✓ p1✓(任一) p2✗ → 2/3。
    """
    ev = object.__new__(AimeEvaluator)
    ev.model_path = "fake"
    ev.device = "cpu"
    ev.max_new_tokens = 16
    ev.max_ctx = 4096
    ev.batch_size = 8
    ev.n_samples = 2
    ev.temperature = 0.7
    ev.top_p = 0.95
    ev.metric = "ave"
    ev.prompt_style = "dapo"
    ev.generate = lambda prompts: [r for _ in prompts for r in (
        "Answer:\n1", "Answer:\n1",     # p0：2/2 对
        "Answer:\n5", "Answer:\n9",     # p1：1/2 对
        "Answer:\n3", "Answer:\n4")]    # p2：0/2 对
    ev.load_problems = lambda d: [("p0", "1"), ("p1", "9"), ("p2", "7")]
    res = ev.evaluate("AIME24")
    assert res.total == 3
    assert res.correct == 2                # pass@1：p0✓ p1✓(任一9) p2✗
    assert abs(res.ave_accuracy - 0.5) < 1e-9   # (1.0+0.5+0.0)/3
    assert abs(res.accuracy - 0.5) < 1e-9       # ave 时 accuracy 用 ave_accuracy
    assert res.rows[1]["correct"] is True       # pass@1 行级语义保留


def test_metric_invalid_raises():
    with pytest.raises(ConfigError):
        AimeEvaluator("fake", n_samples=1, temperature=0.0, metric="bad")


def test_prompt_style_invalid_raises():
    with pytest.raises(ConfigError):
        AimeEvaluator("fake", n_samples=1, temperature=0.0, prompt_style="bad")


def test_generate_passes_top_p_when_sampling(monkeypatch):
    """论文评估协议：采样（do_sample=True）时 top_p 穿透到 model.generate。"""
    import torch
    ev = object.__new__(AimeEvaluator)
    ev.device = "cpu"
    ev.n_samples = 2
    ev.temperature = 0.7
    ev.top_p = 0.95
    ev.batch_size = 8
    ev.max_new_tokens = 8
    ev.max_ctx = 4096
    tok = mock.Mock()
    tok.pad_token_id = 0
    tok.decode = mock.Mock(return_value="\\boxed{1}")
    def encode(batch, **k):
        return {"input_ids": torch.zeros(len(batch), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(batch), 4, dtype=torch.long)}
    tok.side_effect = encode
    ev.tok = tok
    calls = []
    model = mock.Mock()
    def fake_generate(**kwargs):
        calls.append(kwargs)
        B = kwargs["input_ids"].size(0)
        n = kwargs["num_return_sequences"]
        return torch.zeros(B * n, 4 + ev.max_new_tokens, dtype=torch.long)
    model.generate.side_effect = fake_generate
    ev.model = model
    ev.generate(["p0"])
    assert calls[0]["do_sample"] is True
    assert calls[0]["top_p"] == 0.95
    # 贪心（n=1, T=0）不传 top_p
    ev.n_samples = 1
    ev.temperature = 0.0
    ev.generate(["p0"])
    assert "top_p" not in calls[1]


def test_n_samples_pass_at_1():
    """R1：n_samples>1 时 correct 记 pass@1（任一采样答对即对）。"""
    ev = object.__new__(AimeEvaluator)
    ev.model_path = "fake"
    ev.device = "cpu"
    ev.max_new_tokens = 16
    ev.max_ctx = 4096
    ev.batch_size = 8
    ev.n_samples = 2
    ev.temperature = 0.7
    ev.top_p = None
    ev.metric = "pass1"
    ev.prompt_style = "boxed"
    # 2 题 × 2 采样 = 4 条响应（拍平）
    ev.generate = lambda prompts: [r for _ in prompts for r in ("\boxed{1}", "\boxed{2}")]
    ev.load_problems = lambda d: [("p0", "1"), ("p1", "9")]
    res = ev.evaluate("AIME24")
    assert res.total == 2
    assert res.correct == 1      # p0: 采样含 1 ✓；p1: 采样 1,2 均 ≠9 ✗


def test_max_new_tokens_too_large_rejected():
    """R1 + P2 + 对齐论文长生成：max_new_tokens 异常大（>32768）抛 ConfigError（前置）。

    论文评估用长生成（MAX_VAL_RESP_LENGTH=31744）→ 放宽容许到 32768；
    32768 通过前置校验、进入 tokenizer 加载（路径不存在）才抛 ModelError。
    """
    import fullstack_opd_v2.eval_aime as EA
    with pytest.raises(ConfigError):                      # >32768 异常大 → 前置抛
        AimeEvaluator("/nonexistent/path", max_new_tokens=40000)
    with pytest.raises(ModelError):                       # 32768 通过前置 → 路径不存在
        AimeEvaluator("/nonexistent/path", max_new_tokens=32768)
    with pytest.raises(ModelError):                       # 4095 通过 → 路径不存在
        AimeEvaluator("/nonexistent/path", max_new_tokens=4095)
    assert EA._MAX_CONTEXT == 4096                        # 历史默认仍 4096（回退值）


def test_generate_direct_batch_n_samples(monkeypatch):
    """P2：直接测真实 generate()——批量 + num_return_sequences=n + do_sample 组合。

    用 mock tok/model 构造实例（绕过 from_pretrained），验证 R1 核心修复：
    - n_samples>1 + 温度>0 → do_sample=True、num_return_sequences=n；
    - n=1 贪心 → do_sample=False、temperature=1.0（采样温度不传给贪心）；
    - n 感知批缩放：n>1 时每批 prompt 数 = batch_size//n，峰值序列数被压回 batch_size；
    - 返回拍平列表长度 = N×n，顺序 = 逐 prompt × 逐采样。
    """
    import torch
    import fullstack_opd_v2.eval_aime as EA

    def make_ev(n_samples, temperature, batch_size, top_p=None):
        ev = object.__new__(AimeEvaluator)
        ev.device = "cpu"
        ev.n_samples = n_samples
        ev.temperature = temperature
        ev.top_p = top_p
        ev.metric = "pass1"
        ev.prompt_style = "boxed"
        ev.batch_size = batch_size
        ev.max_new_tokens = 8
        ev.max_ctx = 4096
        tok = mock.Mock()
        tok.pad_token_id = 0
        tok.decode = mock.Mock(return_value="\\boxed{1}")
        def encode(batch, **k):
            return {"input_ids": torch.zeros(len(batch), 4, dtype=torch.long),
                    "attention_mask": torch.ones(len(batch), 4, dtype=torch.long)}
        tok.side_effect = encode                       # self.tok(batch, ...) 调用
        ev.tok = tok
        model = mock.Mock()
        calls = []
        def fake_generate(**kwargs):
            calls.append(kwargs)
            B = kwargs["input_ids"].size(0)
            n = kwargs["num_return_sequences"]
            return torch.zeros(B * n, 4 + ev.max_new_tokens, dtype=torch.long)
        model.generate.side_effect = fake_generate
        ev.model = model
        return ev, calls

    # 采样路径：3 prompt、n=2、batch_size=2 → step=1 → 3 次 generate，各 2 序列
    ev, calls = make_ev(n_samples=2, temperature=0.7, batch_size=2)
    out = ev.generate(["p0", "p1", "p2"])
    assert len(out) == 6                                # N×n = 3×2
    assert len(calls) == 3                              # ceil(3/1)
    for c in calls:
        assert c["num_return_sequences"] == 2
        assert c["do_sample"] is True
        assert c["temperature"] == 0.7
        assert c["input_ids"].size(0) == 1              # n 感知批缩放：step=1

    # 贪心路径：n=1、温度>0 → do_sample 仍 False，温度传 1.0
    ev, calls = make_ev(n_samples=1, temperature=0.5, batch_size=2)
    out = ev.generate(["p0", "p1", "p2"])
    assert len(out) == 3
    assert len(calls) == 2                              # ceil(3/2)，无 n 缩放
    for c in calls:
        assert c["num_return_sequences"] == 1
        assert c["do_sample"] is False
        assert c["temperature"] == 1.0

    # n > batch_size 退化保护：batch_size=2, n=4 → step=max(1,0)=1
    ev, calls = make_ev(n_samples=4, temperature=0.7, batch_size=2)
    ev.generate(["p0", "p1"])
    assert all(c["input_ids"].size(0) == 1 for c in calls)


def test_close_frees_model():
    """R1：close() 把模型搬到 CPU 并释放。"""
    ev = object.__new__(AimeEvaluator)
    m = mock.Mock()
    m.to = mock.Mock()
    ev.model = m
    ev.tok = mock.Mock()
    ev.close()
    m.to.assert_called_once_with("cpu")
    assert not hasattr(ev, "model")   # 已释放
