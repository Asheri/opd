"""cli.py 单测：子命令 train/cache/eval/info 端到端可跑。"""
import os

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
    assert set(sub.choices) == {"train", "cache", "eval", "info"}


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


def test_cli_eval_end_to_end(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    run_dir = str(tmp_path / "r_eval")
    main(["train", "--config", str(cfg), "--run-dir", run_dir, "--device", "cpu"])
    ckpts = sorted(os.listdir(os.path.join(run_dir, "checkpoints")))
    assert ckpts
    ckpt = os.path.join(run_dir, "checkpoints", ckpts[-1])
    assert main(["eval", "--config", str(cfg), "--checkpoint", ckpt, "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "step=" in out


def test_cli_cache_end_to_end(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    out = str(tmp_path / "cache.pt")
    assert main(["cache", "--config", str(cfg), "--out", out, "--device", "cpu"]) == 0
    assert os.path.isfile(out)