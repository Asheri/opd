"""全栈 OPD 流水线里流动的数据结构。

对应三个 repo 里的样本/批次表示：
- async-opd/opd/rollout/base.py 的 rollout batch
- Direct-OPD/verl/verl/protocol.py::DataProto
- Lightning-OPD/slime/rollout/... 的 Sample
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RolloutSample:
    """RolloutCollector 产出的样本（可能基于陈旧 student 权重生成）。"""
    prompt_id: int
    response_ids: torch.Tensor            # (T,)
    student_logp_old: torch.Tensor        # (T, V) 生成该 rollout 时 student 的分布（可能陈旧）
    student_version: int                  # 生成时所用的 student 版本号


@dataclass
class ScoredSample:
    """TeacherScorer 给 rollout 贴上离线 Δ_T 后的样本（无 live teacher）。"""
    prompt_id: int
    response_ids: torch.Tensor
    student_logp_old: torch.Tensor
    student_version: int
    teacher_rl_dists: torch.Tensor        # (T, V) 离线缓存的 post-RL teacher 分布
    teacher_ref_dists: torch.Tensor       # (T, V) 离线缓存的 pre-RL reference 分布
