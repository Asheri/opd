"""★ AsyncOPD：异步调度基础设施（有界陈旧度队列 + 权重快照 + teacher 产物缓冲）。

对应真实代码：
- async-opd/opd/utils/staleness_queue.py::StalenessQueue
- async-opd/opd/coordinator/streaming.py（4 线程解耦）
- async-opd/opd/trainer/teacher_artifact_buffer.py::TrainerTeacherArtifactBuffer
"""

from __future__ import annotations

import queue
import threading


class StalenessQueue:
    """有界 FIFO + 版本号。rollout 用旧 student 版本生成后入队，带来 age 标记。

    AsyncOPD 用 staleness_threshold 截断过旧的样本（避免陈旧梯度破坏训练）。
    """

    def __init__(self, staleness_threshold: int = 8):
        self.threshold = staleness_threshold
        self._q: "queue.Queue" = queue.Queue(maxsize=max(16, staleness_threshold * 2))
        self._cur_version = 0
        self._lock = threading.Lock()

    def advance_version(self) -> int:
        with self._lock:
            self._cur_version += 1
            return self._cur_version

    def put(self, item, version: int, timeout: float | None = None) -> bool:
        """入队（带陈旧度截断）。返回 False 表示太旧被丢弃。

        队列满时抛 queue.Full（由调用方决定重试或丢弃）——不要在这里吞掉，
        否则 shutdown 时生产者会阻塞在满队列上（曾经导致 exit code 1）。
        """
        with self._lock:                # ★ 修复：版本读取也要持锁，与 advance_version 互斥
            age = self._cur_version - version
        if age > self.threshold:        # 太旧，丢弃（陈旧度截断，入队侧）
            return False
        self._q.put((item, version, age), timeout=timeout)
        return True

    def get(self, timeout: float | None = None):
        """出队，返回 (item, version, age_at_put)。超时抛 queue.Empty。"""
        return self._q.get(timeout=timeout)

    def get_nowait(self):
        return self._q.get_nowait()

    @property
    def current_version(self) -> int:
        with self._lock:
            return self._cur_version


class WeightStore:
    """learner → rollout worker 的权重快照同步（带版本号）。

    对应 AsyncOPD 的 weight sync / ray_weight_sync：learner 更新后 publish 快照，
    rollout worker 拉取（可能滞后几个版本 → 这就是 staleness 的来源）。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.snapshot = None
        self.version = 0

    def publish(self, state_dict) -> int:
        with self._lock:
            self.snapshot = {k: v.detach().clone() for k, v in state_dict.items()}
            self.version += 1
            return self.version

    def acquire(self):
        with self._lock:
            snap = {k: v.clone() for k, v in self.snapshot.items()}
            return snap, self.version


class TeacherArtifactBuffer:
    """teacher 产物缓冲（避免 learner 时刻重算力）。

    真实 AsyncOPD 里缓存 teacher 的 top-k logps / hidden states；
    本 demo 中即 Lightning 离线缓存的 Δ_T —— 训练期不再调用 teacher。
    """

    def __init__(self, cache, max_batches: int = 3):
        self.cache = cache
        self.max_batches = max_batches

    def assemble(self, prompt_id: int):
        rl, ref = self.cache.get_dists(prompt_id)
        return rl, ref
