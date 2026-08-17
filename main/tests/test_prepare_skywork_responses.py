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
