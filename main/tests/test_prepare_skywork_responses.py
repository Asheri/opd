# -*- coding: utf-8 -*-
"""prepare_skywork_responses.py 参数解析（C3 模板重生成选项）。"""
from __future__ import annotations

import sys

import pytest

from scripts.prepare_skywork_responses import parse_args


@pytest.mark.parametrize("argv,expect", [
    (["p", "--jsonl", "a.jsonl", "--model", "m"], {}),
    (["p", "--jsonl", "a.jsonl", "--model", "m", "--apply-chat-template"],
     {"--apply-chat-template": True}),
    (["p", "--jsonl", "a.jsonl", "--model", "m", "--force"],
     {"--force": True}),
    (["p", "--jsonl", "a.jsonl", "--model", "m", "--apply-chat-template",
      "--force"], {"--apply-chat-template": True, "--force": True}),
])
def test_parse_args_flags(monkeypatch, argv, expect):
    monkeypatch.setattr(sys, "argv", argv)
    ns = parse_args()
    assert ns.jsonl.name == "a.jsonl"
    assert ns.model.name == "m"
    for flag, val in expect.items():
        assert getattr(ns, flag.lstrip("-").replace("-", "_")) is val


def test_parse_args_r1_r5_defaults(monkeypatch):
    """R1/R5（2026-08-27 数据质量审阅）：--max-prompt-len 默认 1024（对齐训练 loader P）、
    --loop-check 默认开、--no-loop-check 显式关。"""
    monkeypatch.setattr(sys, "argv",
                        ["p", "--jsonl", "a.jsonl", "--model", "m"])
    ns = parse_args()
    assert ns.max_prompt_len == 1024
    assert ns.loop_check is True


def test_parse_args_r1_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["p", "--jsonl", "a.jsonl", "--model", "m",
                         "--max-prompt-len", "512", "--no-loop-check",
                         "--top-p", "1.0"])
    ns = parse_args()
    assert ns.max_prompt_len == 512
    assert ns.loop_check is False
    assert ns.top_p == 1.0
