"""Pure-CPU unit tests for `fast_decision_models.prefix_tree` and `Kev08bMlxBackend` PrefixTree invariants."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fast_decision_models.prefix_tree as pt_mod
from fast_decision_models.prefix_tree import (
    ROOT_KEY,
    PrefixTree,
    chain_key,
    payload_nbytes,
    recommended_prefix_budget_bytes,
    snapshot_boundaries,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_prefix_tree_stdlib_only_import_cpu() -> None:
    """Importing `fast_decision_models.prefix_tree` stays 100% Python stdlib under `-B -I -S`."""
    src_dir = _repo_root() / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(src_dir)!r}); "
        "import fast_decision_models, fast_decision_models.prefix_tree; "
        "banned = {'numpy', 'mlx', 'tokenizers', 'fast_shared', 'huggingface_hub', 'coremltools', 'torch'}; "
        "leaked = sorted(banned.intersection(sys.modules)); "
        "assert not leaked, f'Unexpected non-stdlib imports: {leaked}'; "
        "print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "OK"


def test_prefix_tree_chain_hash_and_collision_guard_cpu(monkeypatch) -> None:
    """Verify chained SHA-256 determinism, int32 validation, and exact-token collision protection."""
    c0 = list(range(1, 5))
    c1 = list(range(5, 9))
    c1_alt = [5, 6, 7, 99]

    k0 = chain_key(ROOT_KEY, c0)
    assert k0 == chain_key(ROOT_KEY, c0)
    assert len(k0) == 64
    assert chain_key(k0, c1) != chain_key(ROOT_KEY, c1)
    assert chain_key(k0, c1) != chain_key(k0, c1_alt)

    with pytest.raises(ValueError, match="int32"):
        chain_key(ROOT_KEY, [1 << 33])

    # Force an artificial SHA-256 key collision and prove token-tuple verification rejects both insert and match!
    tree = PrefixTree(chunk_tokens=4, byte_budget=4096)
    monkeypatch.setattr(pt_mod, "chain_key", lambda _parent, _toks: "deadbeef" * 8)

    k_first = tree.insert(parent=None, tokens=[1, 2, 3, 4], depth=1, chunk=b"aaaa")
    assert k_first == "deadbeef" * 8
    assert tree.attach_snapshot(k_first, b"ssss") is True

    # Different tokens mapping to same forced digest must be rejected on insert and miss on match.
    assert tree.insert(parent=None, tokens=[9, 9, 9, 9], depth=1, chunk=b"bbbb") is None
    m_bad = tree.match([9, 9, 9, 9, 10], needs_snapshot=True)
    assert m_bad.is_empty
    m_good = tree.match([1, 2, 3, 4, 10], needs_snapshot=True)
    assert m_good.reused_tokens == 4
    assert m_good.snapshot == b"ssss"


def test_prefix_tree_hybrid_snapshot_depth_and_replay_accounting_cpu() -> None:
    """Hybrid (`needs_snapshot=True`) reuse must cap at the deepest snapshot along the matched path."""
    tree = PrefixTree(chunk_tokens=4, byte_budget=1024)
    t0 = [10, 11, 12, 13]
    t1 = [20, 21, 22, 23]
    t2 = [30, 31, 32, 33]
    seq = t0 + t1 + t2 + [99]

    k0 = tree.insert(parent=None, tokens=t0, depth=1, chunk=b"kv00")
    k1 = tree.insert(parent=k0, tokens=t1, depth=2, chunk=b"kv11")
    k2 = tree.insert(parent=k1, tokens=t2, depth=3, chunk=b"kv22")
    assert k0 is not None and k1 is not None and k2 is not None

    # No snapshots yet: pure attention matches 12 tokens, hybrid matches 0 with 12 replayed.
    m_attn = tree.match(seq, needs_snapshot=False)
    assert m_attn.reused_tokens == 12 and m_attn.replayed_tokens == 0 and m_attn.snapshot is None

    m_hyb0 = tree.match(seq, needs_snapshot=True)
    assert m_hyb0.is_empty and m_hyb0.replayed_tokens == 12
    tree.record(m_hyb0, restored=False)
    assert tree.misses == 1 and tree.matched_but_replayed_tokens == 12

    # Attach snapshot at depth 1 only -> matching 3 chunks caps at 4 reused + 8 replayed.
    assert tree.attach_snapshot(k0, (b"conv0", b"rec0"), keeping_chain_to=k2) is True
    assert tree.has_snapshot(k0) is True and tree.has_snapshot(k1) is False
    m_hyb1 = tree.match(seq, needs_snapshot=True)
    assert m_hyb1.reused_tokens == 4 and m_hyb1.replayed_tokens == 8
    assert m_hyb1.keys == (k0,)
    assert m_hyb1.chunks == (b"kv00",)
    assert m_hyb1.snapshot == (b"conv0", b"rec0")

    # Attach snapshot at depth 3 -> matching all 3 chunks reuses 12 tokens; matching 2 chunks still caps at 4!
    assert tree.attach_snapshot(k2, (b"conv2", b"rec2"), keeping_chain_to=k2) is True
    m_hyb2 = tree.match(t0 + t1 + [99], needs_snapshot=True)
    assert m_hyb2.reused_tokens == 4 and m_hyb2.replayed_tokens == 4

    m_hyb3 = tree.match(seq, needs_snapshot=True)
    assert m_hyb3.reused_tokens == 12 and m_hyb3.replayed_tokens == 0
    assert m_hyb3.keys == (k0, k1, k2)
    assert m_hyb3.chunks == (b"kv00", b"kv11", b"kv22")
    assert m_hyb3.snapshot == (b"conv2", b"rec2")
    tree.record(m_hyb3, restored=True)
    assert tree.hits == 1 and tree.reused_tokens == 12


def test_prefix_tree_hard_ram_budget_and_leaf_only_lru_eviction_cpu() -> None:
    """Verify strict `bytes_held <= byte_budget`, leaf-only LRU eviction, and protected active chains."""
    # Each chunk is 100 bytes; budget fits at most 3 chunks (300 bytes).
    tree = PrefixTree(chunk_tokens=2, byte_budget=300)

    # Branch A: rootA (depth 1, 100B) -> childA (depth 2, 100B)
    k_ra = tree.insert(parent=None, tokens=[1, 2], depth=1, chunk=b"a" * 100)
    k_ca = tree.insert(parent=k_ra, tokens=[3, 4], depth=2, chunk=b"b" * 100)
    # Branch B: rootB (depth 1, 100B)
    k_rb = tree.insert(parent=None, tokens=[5, 6], depth=1, chunk=b"c" * 100)
    assert k_ra and k_ca and k_rb
    assert tree.bytes_held == 300 and tree.node_count == 3

    # Touch childA via match+record so rootB becomes the oldest leaf (last_used: rootB < childA).
    m_a = tree.match([1, 2, 3, 4], needs_snapshot=False)
    tree.record(m_a, restored=True)

    # Insert Branch C (100B): must evict rootB (oldest leaf), keeping rootA + childA!
    k_rc = tree.insert(parent=None, tokens=[7, 8], depth=1, chunk=b"d" * 100)
    assert k_rc is not None
    assert tree.bytes_held == 300 and tree.evicted_nodes == 1
    assert tree.match([5, 6], needs_snapshot=False).is_empty
    assert tree.match([1, 2, 3, 4], needs_snapshot=False).reused_tokens == 4

    # Insert child of rootC (100B): rootC is protected! Oldest unprotected leaf is childA (NOT rootA, which has children=1).
    k_cc = tree.insert(parent=k_rc, tokens=[9, 10], depth=2, chunk=b"e" * 100)
    assert k_cc is not None
    assert tree.bytes_held == 300
    assert tree.match([1, 2, 3, 4], needs_snapshot=False).reused_tokens == 2  # childA evicted, rootA kept!
    assert tree.match([7, 8, 9, 10], needs_snapshot=False).reused_tokens == 4

    # Oversized single payload (> byte_budget) must fail WITHOUT evicting any existing node!
    factory_called = False

    def _boom_factory() -> bytes:
        nonlocal factory_called
        factory_called = True
        return b"x" * 350

    evicted_before = tree.evicted_nodes
    assert tree.insert(parent=None, tokens=[11, 12], depth=1, chunk=_boom_factory, byte_size=350) is None
    assert not factory_called
    assert tree.evicted_nodes == evicted_before
    assert tree.bytes_held == 300

    # Active-chain protection: attaching a 150B snapshot to k_cc requires 100(rootC)+100(childC)+150 = 350 > 300 -> rejected without eviction!
    assert tree.attach_snapshot(k_cc, b"s" * 150, keeping_chain_to=k_cc) is False
    assert tree.evicted_nodes == evicted_before
    assert tree.bytes_held == 300

    # Attaching a 100B snapshot to k_cc CAN fit by evicting rootA (100B): 100+100+100 == 300!
    assert tree.attach_snapshot(k_cc, b"s" * 100, keeping_chain_to=k_cc) is True
    assert tree.bytes_held == 300
    assert tree.node_count == 2 and tree.snapshot_count == 1
    assert tree.match([1, 2], needs_snapshot=False).is_empty

    # Verify shed() frees all bytes and updates evicted count.
    freed = tree.shed()
    assert freed == 300
    assert tree.bytes_held == 0 and tree.node_count == 0 and tree.snapshot_count == 0


def test_prefix_tree_helpers_and_payload_nbytes_cpu() -> None:
    """Verify host budget scaling, snapshot boundary planning, and recursive payload byte accounting."""
    assert recommended_prefix_budget_bytes(24 << 30) == 128 << 20
    assert recommended_prefix_budget_bytes(48 << 30) == 512 << 20
    assert recommended_prefix_budget_bytes(0) == 128 << 20

    assert snapshot_boundaries(0, 1024, chunk_tokens=256, snapshot_stride=1) == [256, 512, 768, 1024]
    assert snapshot_boundaries(0, 900, chunk_tokens=256, snapshot_stride=2) == [512, 768]
    assert snapshot_boundaries(512, 900, chunk_tokens=256, snapshot_stride=2) == [768]
    assert snapshot_boundaries(0, 200, chunk_tokens=256, snapshot_stride=1) == []

    arr_k = np.zeros((1, 2, 256, 256), dtype=np.float16)
    arr_r = np.zeros((1, 16, 128, 128), dtype=np.float32)
    assert payload_nbytes((arr_k, arr_r)) == arr_k.nbytes + arr_r.nbytes
    assert payload_nbytes({"a": b"1234", "b": bytearray(6)}) == 10
    assert payload_nbytes(None) == 0

    tree = PrefixTree(chunk_tokens=4, byte_budget=1024)
    with pytest.raises(ValueError, match="byte_size mismatch"):
        tree.insert(parent=None, tokens=[1, 2, 3, 4], depth=1, chunk=b"1234", byte_size=8)
    assert "0 node(s)" in tree.summary()
