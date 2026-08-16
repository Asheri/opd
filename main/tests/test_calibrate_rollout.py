"""scripts/calibrate_rollout.py 纯函数单测（无 GPU / 无 HF 模型 / 无联网）。

仅测该脚本的可测试内核：tail_is_loop / analyze_rollouts / write_yaml /
CLI 路径校验（HF 模型加载只发生在 main() 内，不触达）。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "calibrate_rollout.py"


@pytest.fixture(scope="module")
def cal():
    spec = importlib.util.spec_from_file_location("calibrate_rollout", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tail_is_loop(cal):
    assert cal.tail_is_loop([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], 3)
    assert not cal.tail_is_loop([1, 2, 3, 4, 5, 6], 2)          # 太短不判
    assert not cal.tail_is_loop([1, 2, 3, 1, 2, 3], 3)          # 短于 min_len(16)


def test_analyze_rollouts_suggests_periods(cal):
    # 周期 3 与周期 5 的序列各占 1/3 → 命中率 33% > 5% → 建议含 3 和 5
    seqs = [[1, 2, 3] * 6, [1, 2, 3, 4, 5] * 4, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
    rep = cal.analyze_rollouts(seqs, eos_tok=0, eos_used=None)
    assert rep["n"] == 3
    assert 3 in rep["suggested_loop_periods"] and 5 in rep["suggested_loop_periods"]
    for p in rep["suggested_loop_periods"]:
        assert rep["loop_rate_by_period"][p] > 0.05
    assert rep["suggested_eos_token_id"] is None


def test_analyze_rollouts_empty(cal):
    rep = cal.analyze_rollouts([], eos_tok=0, eos_used=None)
    assert rep["n"] == 0
    assert rep["suggested_loop_periods"] == ()
    assert rep["lens_min"] == 0 and rep["lens_max"] == 0


def test_write_yaml(tmp_path, cal):
    seqs = [[1, 2, 3] * 6]
    rep = cal.analyze_rollouts(seqs, eos_tok=0, eos_used=151643)
    out = tmp_path / "l2_rollout_suggest.yaml"
    text = cal.write_yaml(rep, out)
    assert out.exists()
    import yaml
    data = yaml.safe_load(text)
    assert data["l2"]["rollout"]["loop_periods"] == list(rep["suggested_loop_periods"])
    assert data["l2"]["rollout"]["eos_token_id"] == 151643


def test_main_missing_jsonl_clean_error(cal, tmp_path):
    """CLI 接口：缺 jsonl 在 HF 加载前干净报错（无 GPU 环境可验，不触达模型）。"""
    sys.argv = ["calibrate_rollout", "--model", str(tmp_path / "m"),
                "--jsonl", str(tmp_path / "nope.jsonl")]
    with pytest.raises(SystemExit) as e:
        cal.main()
    assert "jsonl 不存在" in str(e.value)


def test_main_missing_model_clean_error(cal, tmp_path):
    """CLI 接口：缺模型路径在 HF 加载前干净报错（jsonl 存在但 model 缺失）。"""
    jl = tmp_path / "prompts.jsonl"
    jl.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    sys.argv = ["calibrate_rollout", "--model", str(tmp_path / "no_model"),
                "--jsonl", str(jl)]
    with pytest.raises(SystemExit) as e:
        cal.main()
    assert "HF 模型路径不存在" in str(e.value)


def test_parse_args_output_flag(tmp_path, cal):
    sys.argv = ["calibrate_rollout", "--model", str(tmp_path / "m"),
                "--jsonl", str(tmp_path / "p.jsonl"), "--output", str(tmp_path / "out.yaml")]
    ns = cal.parse_args()
    assert str(ns.output).endswith("out.yaml")
    assert ns.eos_id is None
# --------------------------- IMP-1：loop detector 校准（sweep + 报告） ---------------------------
def test_sweep_loop_configs(cal):
    """sweep 对不同 (periods, min_len) 配置给出 loop_detected_count/rate/flagged。"""
    seqs = [
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],   # 周期 3 明显 loop
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],                  # 正常
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],            # 周期 5（(2,3,4) 不抓）
    ]
    out = cal.sweep_loop_configs(seqs, [((2, 3, 4), 8)])
    r = out["periods=(2, 3, 4),min_len=8"]
    assert r["loop_detected_count"] == 1
    assert r["loop_rate"] == pytest.approx(1 / 3)
    assert r["samples_flagged"] == [0]
    # 无 labels 时 FP/FN 为 None
    assert r["false_positive_rate"] is None
    assert r["false_negative_cases"] is None


def test_sweep_loop_configs_with_labels(cal):
    """有人工标注时计算 FP rate / FN cases（误杀与漏网都被量化）。"""
    seqs = [
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],   # 0: 真周期3 loop -> flag, label T (TP)
        [7, 8, 9, 10, 11, 12, 7, 8, 7, 8],                        # 1: 尾 2-周期但正常(误判) -> flag, label F (FP)
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],            # 2: 周期5 loop 未被 (2,3,4) 抓 -> 不flag, label T (FN)
    ]
    labels = [True, False, True]
    out = cal.sweep_loop_configs(seqs, [((2, 3, 4), 8)], labels)
    r = out["periods=(2, 3, 4),min_len=8"]
    assert r["loop_detected_count"] == 2
    assert r["samples_flagged"] == [0, 1]
    assert r["false_positive_rate"] == pytest.approx(1.0)   # 1 FP / 1 label False
    assert r["false_negative_cases"] == [2]                  # seq 2 漏网


def test_write_calibration_report(cal, tmp_path):
    """生成校准 markdown：含方法学/结果表/决策/GPU 状态。"""
    seqs = [[1, 2, 3] * 6, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    sweep = cal.sweep_loop_configs(seqs, [((2, 3, 4), 8)])
    out = tmp_path / "cal.md"
    text = cal.write_calibration_report(
        sweep, out, {"n": 2, "max_new": 512, "model": "x", "eos_token_id": 1,
                     "date": "2026-08-16", "status": "已填充", "labels_available": False})
    assert out.exists()
    for h in ("## 方法学", "## 结果", "## 决策", "## GPU 验证状态", "## 人工抽样检查清单"):
        assert h in text, f"报告缺 {h}"
    assert "periods=(2, 3, 4),min_len=8" in text
    assert "待服务器 GPU" in text


def test_write_calibration_report_empty_placeholder(cal, tmp_path):
    """空 sweep 结果 → 占位报告（不伪造通过），含「无数据」标注。"""
    out = tmp_path / "cal_empty.md"
    text = cal.write_calibration_report({}, out, {"status": "待服务器真实 rollout 数据填充"})
    assert out.exists()
    assert "（无数据）" in text
    assert "待服务器 GPU" in text
