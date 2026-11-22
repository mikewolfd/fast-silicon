"""Unit tests for fast_shared.prefix_tree multi-chunk matching, rollback, LRU eviction, and shedding."""

from __future__ import annotations

import numpy as np

from fast_shared.prefix_tree import PrefixTree, snapshot_boundaries


def test_prefix_tree_multi_chunk_lcp_and_snapshot_positions_cpu() -> None:
    pt = PrefixTree(chunk_tokens=4, byte_budget=64 * 1024)
    tokens_a = list(range(16))  # 4 full chunks of 4 tokens
    assert snapshot_boundaries(0, len(tokens_a), chunk_tokens=4, snapshot_stride=1) == [4, 8, 12, 16]
    assert snapshot_boundaries(4, len(tokens_a), chunk_tokens=4, snapshot_stride=2, include_deepest=True) == [8, 16]

    keys: list[str] = []
    parent: str | None = None
    for depth in range(1, 5):
        chunk_toks = tokens_a[(depth - 1) * 4 : depth * 4]
        k_arr = np.full((1, 4), float(depth), dtype=np.float32)
        ins_key = pt.insert(parent=parent, tokens=chunk_toks, depth=depth, chunk={"k": k_arr})
        assert ins_key is not None
        keys.append(ins_key)
        parent = ins_key

    snap12 = np.full((1, 8), 12.0, dtype=np.float32)
    assert pt.attach_snapshot(keys[2], {"gdn": snap12}) is True

    # Query sharing first 3 chunks (tokens 0..11) + diverging on chunk 3
    query = tokens_a[:12] + [99, 100, 101, 102]
    m = pt.match(query)
    assert m.reused_tokens == 12
    assert m.replayed_tokens == 0
    assert len(m.chunks) == 3
    assert np.allclose(m.snapshot["gdn"], snap12)
    pt.record(m, restored=True)
    assert pt.stats()["hits"] == 1

    # Strided snapshot at depth=2 (token 8) with chunk depth=3 matched -> 8 reused, 4 replayed
    pt_stride = PrefixTree(chunk_tokens=4, byte_budget=64 * 1024)
    p2: str | None = None
    s_keys: list[str] = []
    for depth in range(1, 5):
        chunk_toks = tokens_a[(depth - 1) * 4 : depth * 4]
        p2 = pt_stride.insert(parent=p2, tokens=chunk_toks, depth=depth, chunk=np.ones((8,), dtype=np.float32))
        assert p2 is not None
        s_keys.append(p2)
    assert pt_stride.attach_snapshot(s_keys[1], np.ones((4,), dtype=np.float32)) is True
    m2 = pt_stride.match(tokens_a[:13])
    assert m2.reused_tokens == 8
    assert m2.replayed_tokens == 4


def test_prefix_tree_snapshot_rollback_and_budget_eviction_cpu() -> None:
    chunk_bytes = 256
    # Budget fits two chunks (512B) + small snapshot (128B), NOT a giant 1024B snapshot
    pt = PrefixTree(chunk_tokens=4, byte_budget=chunk_bytes * 3)
    c0 = np.zeros((chunk_bytes // 4,), dtype=np.float32)
    c1 = np.ones((chunk_bytes // 4,), dtype=np.float32)

    k0 = pt.insert(parent=None, tokens=[0, 1, 2, 3], depth=1, chunk=c0)
    k1 = pt.insert(parent=k0, tokens=[4, 5, 6, 7], depth=2, chunk=c1)
    assert k0 is not None and k1 is not None
    assert pt.stats()["bytes_held"] == chunk_bytes * 2

    # Attaching an oversized snapshot with rollback_keys_on_fail cleanly rolls back orphan chunks
    oversized_snap = np.zeros((chunk_bytes * 4 // 4,), dtype=np.float32)
    ok = pt.attach_snapshot(k1, oversized_snap, rollback_keys_on_fail=[k0, k1])
    assert ok is False
    assert pt.stats()["node_count"] == 0
    assert pt.stats()["bytes_held"] == 0

    # Normal LRU eviction when inserting disjoint branch
    k0_a = pt.insert(parent=None, tokens=[0, 1, 2, 3], depth=1, chunk=c0)
    assert k0_a is not None
    assert pt.attach_snapshot(k0_a, np.ones((chunk_bytes // 4,), dtype=np.float32)) is True

    k0_b = pt.insert(parent=None, tokens=[100, 101, 102, 103], depth=1, chunk=c0)
    k1_b = pt.insert(parent=k0_b, tokens=[104, 105, 106, 107], depth=2, chunk=c1)
    assert k0_b is not None and k1_b is not None
    assert pt.stats()["bytes_held"] <= pt.byte_budget
    assert pt.match([0, 1, 2, 3]).is_empty

    freed = pt.shed()
    assert freed == chunk_bytes * 2
    assert pt.stats()["node_count"] == 0
    assert pt.stats()["bytes_held"] == 0
