"""v2 异步基础设施：陈旧度队列 + 权重快照。

与 v1 相同的语义（AsyncOPD 真实形态），两处加速：
- WeightStore.acquire_if_newer：rollout worker 只在版本变化时才克隆+加载权重
  （v1 每个样本都 acquire+load_state_dict 一次，纯浪费）；
- StalenessQueue.put 持锁读版本 + timeout（沿用 v1 审阅修复）。
"""

from __future__ import annotations

import queue
import threading


class StalenessQueue:
    """有界 FIFO + 版本号。入队侧截断过旧样本，消费侧再截一次（双保险）。"""

    def __init__(self, staleness_threshold: int = 8):
        self.threshold = staleness_threshold
        self._q: "queue.Queue" = queue.Queue(maxsize=max(16, staleness_threshold * 2))
        self._cur_version = 0
        self._lock = threading.Lock()
        self.n_rejected = 0          # P2-1：入队侧因过旧拒绝的样本数（只观测）

    def advance_version(self) -> int:
        with self._lock:
            self._cur_version += 1
            return self._cur_version

    def put(self, item, version: int, timeout: float | None = None) -> bool:
        """返回 False = 太旧被丢弃；队列满抛 queue.Full（由调用方处理）。"""
        with self._lock:
            age = self._cur_version - version
        if age > self.threshold:
            self.n_rejected += 1
            return False
        self._q.put((item, version, age), timeout=timeout)
        return True

    def get(self, timeout: float | None = None):
        """-> (item, version, age_at_put)。超时抛 queue.Empty。"""
        return self._q.get(timeout=timeout)

    @property
    def current_version(self) -> int:
        with self._lock:
            return self._cur_version


class WeightStore:
    """learner → rollout worker 的权重快照同步（带版本号）。

    offload_to_cpu=True（colocated 部署 L6）：快照存 CPU，rollout 时按需搬回 GPU——
    使 learner 优化器/权重与 vLLM rollout 可「换入换出」（fused-hybrid 模式），
    避免 2×96GB 上二者同卡同时驻留 OOM（见 OPTIMIZATION_PLAN_2xRTXPRO6000.md §0.1/§L3）。
    """

    def __init__(self, offload_to_cpu: bool = False):
        self.offload_to_cpu = offload_to_cpu
        self._lock = threading.Lock()
        self._snapshot = None
        self._version = 0

    def publish(self, state_dict) -> int:
        with self._lock:
            if self.offload_to_cpu:
                self._snapshot = {k: v.detach().cpu() for k, v in state_dict.items()}
            else:
                self._snapshot = {k: v.detach().clone() for k, v in state_dict.items()}
            self._version += 1
            return self._version

    def acquire_if_newer(self, last_ver: int):
        """仅当版本推进时才克隆快照，否则返回 (None, 当前版本)——避免重复搬运。

        若 offload_to_cpu，快照为 CPU 张量，调用方需 .to(device) 后 load_state_dict。
        """
        with self._lock:
            if self._version == last_ver:
                return None, self._version
            snap = {k: v.clone() for k, v in self._snapshot.items()}
            return snap, self._version
