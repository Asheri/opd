"""checkpoint 管理：学生模型断点保存/加载/续跑。

工程化核心：训练可中断、可续跑。每 `every` 步把
`state_dict + version + step + cfg（+ 最近 metrics）` 落盘到
`run_dir/checkpoints/step_<N>.pt`，`--resume` 时加载最新断点续跑。
"""

from __future__ import annotations

import os
import re

import torch

from .exceptions import CheckpointError


class CheckpointManager:
    """管理 run_dir/checkpoints/ 下的断点，支持节流保存与最新定位。"""

    def __init__(self, run_dir: str, every: int = 10,
                 checkpoint_dir: str | None = None):
        self.checkpoint_dir = checkpoint_dir or os.path.join(run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.every = max(1, int(every))

    # --------------------------- 保存 ---------------------------
    def save(self, step: int, student, version: int, cfg: dict,
             metrics: list | None = None) -> str | None:
        """若 step 是 every 的倍数则存断点，否则跳过（节流）。"""
        if step % self.every != 0:
            return None
        path = os.path.join(self.checkpoint_dir, f"step_{step}.pt")
        tmp = path + ".tmp"
        torch.save({
            "step": step,
            "version": version,
            "state": {k: v.detach().cpu() for k, v in student.state_dict().items()},
            "cfg": cfg,
            "metrics": (metrics or [])[-1:],
        }, tmp)
        os.replace(tmp, path)                     # 原子替换，避免半写
        return path

    # --------------------------- 定位 ---------------------------
    def latest(self) -> str | None:
        """返回最新（step 最大）的断点路径。"""
        best = None
        best_step = -1
        if os.path.isdir(self.checkpoint_dir):
            for name in os.listdir(self.checkpoint_dir):
                m = re.match(r"step_(\d+)\.pt$", name)
                if m and int(m.group(1)) > best_step:
                    best_step = int(m.group(1))
                    best = os.path.join(self.checkpoint_dir, name)
        return best

    # --------------------------- 加载 ---------------------------
    def load(self, ckpt_path: str) -> dict:
        """加载断点 → {step, version, state, cfg, metrics}。"""
        if not os.path.isfile(ckpt_path):
            raise CheckpointError(f"断点不存在: {ckpt_path}")
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise CheckpointError(f"加载断点失败 {ckpt_path}: {e}") from e
        return ck

    def resume(self) -> dict | None:
        """自动定位最新断点并加载；无断点返回 None。"""
        latest = self.latest()
        if latest is None:
            return None
        return self.load(latest)


__all__ = ["CheckpointManager"]