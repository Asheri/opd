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


def test_weight_store_publish_reuses_buffer_and_returns_independent_clone():
    ws = WeightStore(offload_to_cpu=False)
    sd1 = {"a": torch.ones(3), "b": torch.zeros(2)}
    ws.publish(sd1)
    # 拿到快照引用
    snap_a_id = id(ws._snapshot["a"])
    # 第二次 publish 更新值
    sd2 = {"a": torch.full((3,), 5.0), "b": torch.full((2,), 7.0)}
    ws.publish(sd2)
    # 缓冲对象复用（id 不变），值已更新
    assert id(ws._snapshot["a"]) == snap_a_id
    assert torch.equal(ws._snapshot["a"], torch.full((3,), 5.0))
    # acquire 返回独立克隆，不受后续 publish 影响
    snap2, _ = ws.acquire_if_newer(0)
    ws.publish(sd1)
    assert torch.equal(snap2["a"], torch.full((3,), 5.0))


def test_weight_store_publish_after_init_prefill_reuses_buffer():
    """复现 scheduler.__init__ 预填 _snapshot 后首步 publish 的交互：
    预填已使 _snapshot 非 None，publish 应走 copy_ 复用分支而非重建，
    且键一致时不应 KeyError。"""
    ws = WeightStore(offload_to_cpu=False)
    # 模拟 scheduler.__init__ 直接预填（绕过 publish，版本=0）
    student = {"a": torch.ones(3), "b": torch.zeros(2)}
    ws._snapshot = {k: v.detach().clone() for k, v in student.items()}
    ws._version = 0
    snap_id = id(ws._snapshot["a"])
    # 首步 publish：键一致 → copy_ 原地覆盖，对象复用
    sd = {"a": torch.full((3,), 9.0), "b": torch.full((2,), 1.0)}
    v = ws.publish(sd)
    assert v == 1
    assert id(ws._snapshot["a"]) == snap_id
    assert torch.equal(ws._snapshot["a"], torch.full((3,), 9.0))
    # acquire 返回独立克隆
    snap, ver = ws.acquire_if_newer(last_ver=0)
    assert ver == 1 and torch.equal(snap["a"], torch.full((3,), 9.0))
