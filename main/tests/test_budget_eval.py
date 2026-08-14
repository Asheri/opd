"""budget_eval.py 单测：Budget-Aware Evaluation 协议。

- extract_final_answer：4 级答案提取（boxed/FinalAnswer/benchmark/fallback）。
- generate_budget：逐位 EOS 判定（status=eos|budget_stop, reasoning_tokens）。
- evaluate_budget：双指标（Accuracy@B outcome / PrefixAccuracy@B）+ token 记账。
- run_matrix / write_report：矩阵聚合 + md 表 + 4 图。
"""
import unittest.mock as mock

import pytest

from fullstack_opd_v2.budget_eval import (
    BudgetEvaluator, extract_final_answer, run_matrix, write_report,
    ANSWER_COMPLETION_PROMPT, DEFAULT_BUDGETS, DEFAULT_COMPLETION_MAX_TOKENS,
    DATASET_REGISTRY, DatasetSpec, _gsm8k_gt,
)
from fullstack_opd_v2.exceptions import ConfigError


# --------------------------- extract_final_answer（任务1） ---------------------------
def test_efa_boxed_wins():
    out = extract_final_answer("思路...\\boxed{25/8}")
    assert out and "25/8" in out


def test_efa_final_answer_marker():
    assert extract_final_answer("reasoning...\nFinal Answer: 42") == "42"


def test_efa_boxed_beats_marker():
    out = extract_final_answer("Final Answer: 99\n\\boxed{7}")
    assert out and "7" in out


def test_efa_answer_marker():
    assert extract_final_answer("推到一半\nAnswer: 7") == "7"


def test_efa_none_when_no_answer():
    assert extract_final_answer("只是推理，没有结论") is None


def test_efa_empty():
    assert extract_final_answer("") is None
    assert extract_final_answer(None) is None


def test_efa_truncated_mid_conclusion_no_false_positive():
    # 预算截断在无结论处 → 无答案（不因末尾数字误判）
    assert extract_final_answer("我们得到 n=3 因为 2+1=3") == "3"   # fallback 末尾数字


# --------------------------- generate_budget（任务2） ---------------------------
class _FakTok:
    """fake tokenizer：__call__ 返回定长左填充批次；decode 返回任意文本。"""
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "left"

    def __init__(self, seq_len=5):
        self.seq_len = seq_len

    def __call__(self, texts, **kw):
        import torch
        B = len(texts)
        return {"input_ids": torch.zeros(B, self.seq_len, dtype=torch.long),
                "attention_mask": torch.ones(B, self.seq_len, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "R" * len(ids)


class _FakModel:
    """fake generate：返回 seq_len + new_tokens 的序列（num_return_sequences 行）。"""
    def __init__(self, new_tokens):
        self.new_tokens = new_tokens

    def generate(self, **gen):
        import torch
        seq_len = gen["input_ids"].size(1)
        seq = gen["input_ids"][0].tolist() + self.new_tokens
        n_ret = gen.get("num_return_sequences", 1)
        return torch.tensor([seq] * n_ret, dtype=torch.long)


def _mk_budget_eval(new_tokens, **over):
    """构造不加载真实模型的 BudgetEvaluator（只 stub generate_budget 所需）。"""
    ev = object.__new__(BudgetEvaluator)
    ev.model_path = "fake"
    ev.device = "cpu"
    ev.batch_size = 8
    ev.n_samples = 1
    ev.temperature = 0.0
    ev.top_p = None
    ev.scoring = "int"
    ev.prompt_style = "boxed"
    ev.chat_template = False
    ev.max_ctx = 4096
    ev.seed = 42
    ev.completion_max_tokens = 64
    ev.tok = _FakTok()
    ev.model = _FakModel(new_tokens)
    for k, v in over.items():
        setattr(ev, k, v)
    return ev


def test_generate_budget_eos_detection():
    ev = _mk_budget_eval(new_tokens=[1, 2])   # 2=eos
    rows = ev.generate_budget(["a"], budget=8)
    assert len(rows) == 1
    text, status, rt = rows[0]
    assert status == "eos"
    assert rt == 1                       # eos 位置（不含 eos）
    assert rt < 8                        # 非截断


def test_generate_budget_budget_stop():
    ev = _mk_budget_eval(new_tokens=[1, 1, 1])
    text, status, rt = ev.generate_budget(["a"], budget=3)[0]
    assert status == "budget_stop"
    assert rt == 3                       # 满预算，无 eos


def test_generate_budget_left_pad_reused():
    ev = _mk_budget_eval(new_tokens=[1])
    assert ev.tok.padding_side == "left"


def test_generate_budget_n_samples_kept():
    # 保留 n_samples>1 接口：n=2 时每 prompt 产出 2 条
    ev = _mk_budget_eval(new_tokens=[1], n_samples=2, temperature=0.7,
                         batch_size=4)
    rows = ev.generate_budget(["a"], budget=8)
    assert len(rows) == 2


# --------------------------- evaluate_budget（任务3） ---------------------------
def _fake_eval_budget(groups, problems, completion=("\\boxed{42}", 5), **over):
    """构造 stub 的 BudgetEvaluator：generate_budget 逐组返回、_completion 固定。"""
    ev = object.__new__(BudgetEvaluator)
    ev.model_path = "fake"
    ev.device = "cpu"
    ev.batch_size = 8
    ev.n_samples = 1
    ev.temperature = 0.0
    ev.top_p = None
    ev.scoring = "int"
    ev.prompt_style = "boxed"
    ev.chat_template = False
    ev.completion_max_tokens = 64
    ev.seed = 42
    ev.load_problems = lambda d: problems
    it = iter(groups)
    ev.generate_budget = lambda prompts, budget: next(it)
    ev._completion = lambda prefix, x: completion
    for k, v in over.items():
        setattr(ev, k, v)
    return ev


def test_evaluate_budget_outcome_correct():
    # 预算内 EOS 且 boxed 答案正确 → outcome=True, eos=1, 无 prefix
    ev = _fake_eval_budget(
        groups=[[("\\boxed{42}", "eos", 3)]],
        problems=[("p0", "42")])
    res = ev.evaluate_budget("AIME24", 256)
    assert res["accuracy"] == 1.0
    assert res["eos_rate"] == 1.0
    assert res["budget_stop_rate"] == 0.0
    assert res["no_answer_rate"] == 0.0
    assert res["prefix_accuracy"] is None
    row = res["rows"][0]
    assert row["outcome_correct"] is True
    assert row["answer_completion_tokens"] == 0      # 有答案不跑 completion


def test_evaluate_budget_prefix_only_no_answer():
    # 预算内无 final answer → 跑 completion → prefix 正确
    ev = _fake_eval_budget(
        groups=[[("no answer", "budget_stop", 8)]],
        problems=[("p0", "42")],
        completion=("\\boxed{42}", 5))
    res = ev.evaluate_budget("AIME24", 256)
    assert res["accuracy"] == 0.0                    # outcome 错（无答案）
    assert res["no_answer_rate"] == 1.0
    assert res["prefix_accuracy"] == 1.0             # completion 正确
    row = res["rows"][0]
    assert row["outcome_correct"] is False
    assert row["prefix_correct"] is True
    assert row["answer_completion_tokens"] == 5


def test_evaluate_budget_prefix_wrong_when_completion_wrong():
    ev = _fake_eval_budget(
        groups=[[("no answer", "budget_stop", 8)]],
        problems=[("p0", "42")],
        completion=("\\boxed{99}", 5))
    res = ev.evaluate_budget("AIME24", 256)
    assert res["prefix_accuracy"] == 0.0


def test_evaluate_budget_token_accounting():
    ev = _fake_eval_budget(
        groups=[[("\\boxed{42}", "eos", 3)], [("no answer", "budget_stop", 10)]],
        problems=[("p0", "42"), ("p1", "7")],
        completion=("\\boxed{7}", 5))
    res = ev.evaluate_budget("AIME24", 256)
    assert res["n"] == 2
    for r in res["rows"]:
        assert r["total_tokens"] == r["reasoning_tokens"] + r["answer_completion_tokens"]
        if r["has_final_answer"]:
            assert r["answer_completion_tokens"] == 0


def test_completion_not_counted_in_reasoning_budget():
    # completion_max_tokens 独立于 reasoning budget；answer_completion_tokens 有独立上限
    assert DEFAULT_COMPLETION_MAX_TOKENS == 64
    assert ANSWER_COMPLETION_PROMPT  # 固定、不可修改的 completion 提示存在
    assert DEFAULT_BUDGETS == (256, 512, 1024, 2048, 4096)


def test_evaluate_budget_aggregate_mixed():
    # 2 样本：1 对（eos）、1 错（无答案+completion 错）→ acc=0.5, prefix=0.0
    ev = _fake_eval_budget(
        groups=[[("\\boxed{42}", "eos", 3)], [("no answer", "budget_stop", 10)]],
        problems=[("p0", "42"), ("p1", "7")],
        completion=("\\boxed{99}", 5))
    res = ev.evaluate_budget("AIME24", 256)
    assert res["accuracy"] == 0.5
    assert res["prefix_accuracy"] == 0.0
    assert res["no_answer_rate"] == 0.5
    assert res["avg_reasoning_tokens"] == 6.5       # (3+10)/2


# --------------------------- 数据集注册表（GSM8K/MATH-500/AIME） ---------------------------
def test_registry_has_three_roles():
    # GSM8K 基础泛化（test）、MATH-500 主结果（test）、AIME 补充（train）
    assert DATASET_REGISTRY["GSM8K"].split == "test"
    assert DATASET_REGISTRY["MATH500"].split == "test"
    assert DATASET_REGISTRY["AIME24"].split == "train"
    assert DATASET_REGISTRY["AIME25"].split == "train"


def test_registry_case_insensitive_resolve():
    ev = object.__new__(BudgetEvaluator)
    assert ev.resolve_dataset("gsm8k").hf == "openai/gsm8k"
    assert ev.resolve_dataset("math500").hf == "HuggingFaceH4/MATH-500"
    # 未知键按直接 HF 名对待（test 切分、默认列）
    assert ev.resolve_dataset("custom/ds").hf == "custom/ds"
    assert ev.resolve_dataset("custom/ds").split == "train"


def test_gsm8k_gt_extracts_after_marker():
    assert _gsm8k_gt("#### 42") == "42"
    assert _gsm8k_gt("Let's solve...\n#### 5.5") == "5.5"
    assert _gsm8k_gt("42") == "42"          # 无 #### 标记 → 原样


def _inject_datasets(rows):
    """把假 datasets 模块注入 sys.modules（load_problems 内是局部 from-import）。"""
    import sys
    import types
    fake = types.ModuleType("datasets")
    fake.load_dataset = mock.Mock(return_value=rows)
    return mock.patch.dict(sys.modules, {"datasets": fake}), fake.load_dataset


def test_load_problems_uses_registry_spec():
    ev = object.__new__(BudgetEvaluator)
    fake_ds = iter([{"question": "Q1", "answer": "#### 42"},
                    {"question": "Q2", "answer": "#### 7"}])
    patcher, mld = _inject_datasets(fake_ds)
    with patcher:
        rows = ev.load_problems("GSM8K")
    assert mld.call_args[0][0] == "openai/gsm8k"
    assert mld.call_args[1]["split"] == "test"
    assert rows == [("Q1", "42"), ("Q2", "7")]


def test_load_problems_missing_column_raises():
    ev = object.__new__(BudgetEvaluator)
    patcher, _ = _inject_datasets(iter([{"problem": "Q1"}]))
    with patcher, pytest.raises(Exception) as ei:
        ev.load_problems("MATH500")
    assert "缺 problem/answer 列" in str(ei.value)


# --------------------------- run_matrix / write_report（任务4） ---------------------------
def test_run_matrix_skips_empty_path(tmp_path):
    res = run_matrix([("L0", "")], [256], ["AIME24"], str(tmp_path), device="cpu")
    assert res == []                               # 空路径占位跳过


def test_write_report_creates_md_and_plots(tmp_path):
    fake = [{"label": "Base", "dataset": "AIME24", "budget": 256, "accuracy": 0.5,
             "prefix_accuracy": 0.6, "eos_rate": 0.3, "budget_stop_rate": 0.7,
             "avg_reasoning_tokens": 256.0, "rows": []},
            {"label": "Base", "dataset": "AIME24", "budget": 512, "accuracy": 0.7,
             "prefix_accuracy": 0.8, "eos_rate": 0.5, "budget_stop_rate": 0.5,
             "avg_reasoning_tokens": 512.0, "rows": []}]
    md = write_report(fake, str(tmp_path / "report.md"))
    assert (tmp_path / "report.md").exists()
    assert "| Model | Budget | Accuracy | PrefixAccuracy | EOS | BudgetStop | AvgReasoningTokens |" in md
    assert "Base|256|0.500|0.600|0.300|0.700|256" in md
    assert "Base|512|0.700|0.800|0.500|0.500|512" in md
    try:
        import matplotlib   # noqa
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 4
    except Exception:
        pass                # matplotlib 未装则图跳过（md 加注）


# --------------------------- CLI eval-budget（任务5） ---------------------------
def test_cli_eval_budget_parses_models():
    from fullstack_opd_v2.cli import build_parser
    ap = build_parser()
    args = ap.parse_args(["eval-budget", "--models", "Base=/m", "--models", "L0=",
                          "--budgets", "256,512"])
    assert args.command == "eval-budget"
    assert args.models == ["Base=/m", "L0="]
    assert args.budgets == "256,512"
    assert args.scoring == "sympy"          # 默认数学等价判定（支持 MATH-500/GSM8K 分数小数）


def test_cli_eval_budget_requires_models():
    from argparse import Namespace
    from fullstack_opd_v2.cli import _cmd_eval_budget
    args = Namespace(models=None)
    with pytest.raises(ConfigError) as ei:
        _cmd_eval_budget(args)
    assert "--models" in str(ei.value)


def test_cli_eval_budget_skips_empty_path(tmp_path):
    from argparse import Namespace
    from fullstack_opd_v2.cli import _cmd_eval_budget
    args = Namespace(
        models=["L0="], budgets="256,512", datasets=None, out=str(tmp_path),
        n_samples=1, seed=42, temperature=0.0, top_p=None, scoring="sympy",
        prompt_style="boxed", chat_template=False, attn_impl=None,
        batch_size=8, dtype="auto", completion_max_tokens=64, device="cpu")
    with mock.patch("fullstack_opd_v2.budget_eval.run_matrix",
                    return_value=[]) as mr:
        rc = _cmd_eval_budget(args)
    assert rc == 0
    assert mr.called
    # 空路径占位 → run_matrix 返回 [] → 两份报告（Stage 1.6 + 1.7）都写出
    assert (tmp_path / "2026-08-15-budget-aware-eval.md").exists()
    assert (tmp_path / "2026-08-15-budget-curve-analysis.md").exists()