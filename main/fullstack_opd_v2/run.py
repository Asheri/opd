"""run 目录管理：每次训练建独立 run 目录，快照配置、组织产物。

工程化核心：把"一次训练"变成可复现、可跨 run 对比的单元——
```
runs/<timestamp>/
├── config.yaml      ← 解析后的完整配置快照（可复现）
├── metrics.csv      ← 每步指标（loss/pg/kl/adv/reward/age/version）
├── train.log        ← 结构化日志
└── checkpoints/     ← step_<N>.pt 断点
```
无 run_dir 时自动生成 `runs/<timestamp>`；给定时（如 --resume）复用该目录。
"""

from __future__ import annotations

import os
from datetime import datetime

import yaml


class RunManager:
    """负责 run 目录的创建与路径组织。"""

    def __init__(self, cfg: dict, run_dir: str | None = None,
                 base_dir: str = "runs"):
        self.cfg = cfg
        self.base_dir = base_dir
        if run_dir:
            self.run_dir = run_dir
        else:                                    # 自动时间戳目录
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(base_dir, ts)
        self._built_paths: dict | None = None

    def create(self) -> dict:
        """创建目录结构并快照配置，返回路径字典。"""
        logs = os.path.join(self.run_dir, "logs")
        checkpoints = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(logs, exist_ok=True)
        os.makedirs(checkpoints, exist_ok=True)

        config_path = os.path.join(self.run_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, allow_unicode=True, sort_keys=False)

        self._built_paths = {
            "run_dir": self.run_dir,
            "logs": logs,
            "checkpoints": checkpoints,
            "config": config_path,
            "log_file": os.path.join(logs, "train.log"),
            "metrics_csv": os.path.join(self.run_dir, "metrics.csv"),
            "checkpoint_dir": checkpoints,
        }
        return self._built_paths

    def paths(self) -> dict:
        """返回路径字典（未 create 时自动 create）。"""
        if self._built_paths is None:
            return self.create()
        return self._built_paths


__all__ = ["RunManager"]