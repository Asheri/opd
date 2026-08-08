"""buffer.py 单测：StalenessQueue 双截断、WeightStore 版本推进。"""
from __future__ import annotations

import torch

from fullstack_opd_v2.buffer import StalenessQueue, WeightStore


def test_staleness_queue_accepts_fresh():
    q = StalenessQueue(staleness_threshold=4)
    assert q.put("x", version=0) is True
    item, version, age = q.get(timeout=1)
    assert item == "x" and version == 0 and age == 0


def test_staleness_queue_drops_too_old():
    q = StalenessQueue(staleness_threshold=2)
    for _ in range(5):
        q.advance_version()              # current_version = 5
    assert q.put("stale", version=0) is False   # age=5 > 2 → 丢弃
    assert q.put("fresh", version=4) is True    # age=1 ≤ 2 → 接受


def test_staleness_queue_age_reported():
    q = StalenessQueue(staleness_threshold=10)
    for _ in range(3):
        q.advance_version()              # current_version = 3
    q.put("y", version=1)
    _, version, age = q.get(timeout=1)
    assert version == 1 and age == 2


def test_staleness_queue_current_version():
    q = StalenessQueue(4)
    assert q.current_version == 0
    v = q.advance_version()
    assert v == 1 and q.current_version == 1


def test_weight_store_publish_and_acquire_newer():
    ws = WeightStore()
    sd = {"w": torch.ones(3)}
    v1 = ws.publish(sd)
    assert v1 == 1
    snap, ver = ws.acquire_if_newer(last_ver=0)
    assert ver == 1 and snap is not None
    assert torch.equal(snap["w"], sd["w"])
    # 版本未变 → 不再克隆
    snap2, ver2 = ws.acquire_if_newer(last_ver=1)
    assert snap2 is None and ver2 == 1
    # 推进后又能拿到
    sd2 = {"w": torch.zeros(3)}
    ws.publish(sd2)
    snap3, ver3 = ws.acquire_if_newer(last_ver=1)
    assert ver3 == 2 and torch.equal(snap3["w"], sd2["w"])


def test_weight_store_offload_to_cpu():
    ws = WeightStore(offload_to_cpu=True)
    sd = {"w": torch.ones(2)}
    ws.publish(sd)
    snap, _ = ws.acquire_if_newer(last_ver=0)
    assert snap["w"].device.type == "cpu"


def test_weight_store_snapshot_is_detached_copy():
    ws = WeightStore()
    sd = {"w": torch.ones(2, requires_grad=True)}
    ws.publish(sd)
    snap, _ = ws.acquire_if_newer(last_ver=0)
    assert snap["w"].requires_grad is False
    # 修改原 sd 不影响已克隆的快照
    with torch.no_grad():
        sd["w"].add_(100)
    assert torch.equal(snap["w"], torch.ones(2))
