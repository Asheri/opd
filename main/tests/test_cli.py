"""cli.py 单测：子命令 train/cache/eval/info 端到端可跑。"""
import os
import re

import pytest

from fullstack_opd_v2.cli import main, build_parser


def _write_cfg(tmp_path, n_steps=3):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "n_prompts: 8\n"
        "stage0:\n  n_rl_steps: 2\n"
        "stage2:\n  n_steps: 3\n  batch_size: 4\n"
        f"stage1:\n  cache_path: {tmp_path / 'c.pt'}\n",
        encoding="utf-8")
    return cfg


def test_parser_has_subcommands():
    p = build_parser()
    sub = next(a for a in p._actions if a.dest == "command")
    assert set(sub.choices) == {"train", "cache", "eval", "info", "eval-aime"}


def test_cli_info(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    assert main(["info", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "stage2" in out and "n_steps" in out


def test_cli_train(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    run_dir = str(tmp_path / "run1")
    assert main(["train", "--config", str(cfg), "--run-dir", run_dir,
                 "--device", "cpu"]) == 0
    assert os.path.isfile(os.path.join(run_dir, "metrics.csv"))
    assert os.path.isdir(os.path.join(run_dir, "checkpoints"))
    out = capsys.readouterr().out
    assert "run 目录" in out or "完成" in out


def test_cli_train_resume(tmp_path, capsys):
    cfg = _write_cfg(tmp_path, n_steps=2)
    run_dir = str(tmp_path / "run2")
    main(["train", "--config", str(cfg), "--run-dir", run_dir, "--device", "cpu"])
    # resume 续跑
    assert main(["train", "--config", str(cfg), "--run-dir", run_dir,
                 "--resume", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "resume" in out.lower() or "续跑" in out


def test_cli_unknown_command_system_exit_2():
    # argparse 对未知子命令直接 SystemExit(2)（标准 CLI 行为）
    with pytest.raises(SystemExit) as e:
        main(["bogus"])
    assert e.value.code == 2


def test_cli_bad_override_friendly(capsys):
    rc = main(["info", "--set", "stage2.n_steps"])
    assert rc == 2
    assert "error" in capsys.readouterr().out.lower()


def _latest_ckpt(run_dir):
    """按 step 号（非词法序）取最新断点，避免 step_10 < step_3 的词法陷阱。"""
    ck_dir = os.path.join(run_dir, "checkpoints")
    names = os.listdir(ck_dir)
    assert names
    step = max(int(re.match(r"step_(\d+)\.pt$", n).group(1)) for n in names)
    return os.path.join(ck_dir, f"step_{step}.pt")


def test_cli_eval_end_to_end(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    run_dir = str(tmp_path / "r_eval")
    main(["train", "--config", str(cfg), "--run-dir", run_dir, "--device", "cpu"])
    ckpt = _latest_ckpt(run_dir)
    assert main(["eval", "--config", str(cfg), "--checkpoint", ckpt, "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "step=" in out


def test_cli_cache_end_to_end(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    out = str(tmp_path / "cache.pt")
    assert main(["cache", "--config", str(cfg), "--out", out, "--device", "cpu"]) == 0
    assert os.path.isfile(out)


def test_no_args_requires_subcommand(capsys):
    # 无参调用：add_subparsers(required=True) 由 argparse 直接 SystemExit(2)
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2

def test_cli_eval_aime_run_dir_missing_model_path(tmp_path, capsys):
    """toy run 目录无 eval.model_path → DataError → exit 2。"""
    cfg = _write_cfg(tmp_path)
    run_dir = str(tmp_path / "r_evalaime")
    main(["train", "--config", str(cfg), "--run-dir", run_dir, "--device", "cpu"])
    rc = main(["eval-aime", "--run-dir", run_dir, "--device", "cpu"])
    assert rc == 2
    assert "model_path" in capsys.readouterr().out


def test_cli_eval_aime_run_dir_bridge(tmp_path, capsys, monkeypatch):
    """run-dir 配置 eval.model_path → 桥接评估真实模型（mock AimeEvaluator）。"""
    import json
    import os
    import yaml
    import fullstack_opd_v2.eval_aime as EA

    class FakeAimeEvaluator:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self          # R1：eval-aime 用 with AimeEvaluator(...) as ev 确保 close
        def __exit__(self, *exc):
            return False
        def evaluate_to_jsonl(self, ds, out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"dataset": ds, "correct": True}) + "\n")
            return type("R", (), {"correct": 1, "total": 1, "accuracy": 1.0})()

    monkeypatch.setattr(EA, "AimeEvaluator", FakeAimeEvaluator)
    run_dir = str(tmp_path / "r2")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"eval": {"model_path": "/path/to/real-model"}}, f)
    rc = main(["eval-aime", "--run-dir", run_dir, "--datasets", "AIME24", "--device", "cpu"])
    assert rc == 0
    assert os.path.isfile(os.path.join(run_dir, "aime", "AIME24.jsonl"))


def test_cli_eval_aime_malformed_config(tmp_path, capsys):
    """R3：run-dir config.yaml 非法 YAML → ConfigError → exit 2。"""
    import os
    run_dir = str(tmp_path / "r_bad")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("eval:\n  model_path: [unclosed\n")
    rc = main(["eval-aime", "--run-dir", run_dir, "--device", "cpu"])
    assert rc == 2
    assert "config.yaml" in capsys.readouterr().out
