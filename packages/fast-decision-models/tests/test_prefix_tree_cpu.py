"""Pure-CPU unit tests for `fast_decision_models.prefix_tree` and `Kev08bMlxBackend` PrefixTree invariants."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import fast_decision_models.prefix_tree as pt_mod
from fast_decision_models.kev_mlx import Kev08bMlxBackend
from fast_decision_models.prefix_tree import (
    DEFAULT_CHUNK_TOKENS,
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
    assert payload_nbytes((arr_k, arr_r, np.dtype("float32"))) == arr_k.nbytes + arr_r.nbytes
    try:
        from fast_shared.prefix_tree import payload_nbytes as shared_payload_nbytes

        assert shared_payload_nbytes((arr_k, arr_r, np.dtype("float32"))) == arr_k.nbytes + arr_r.nbytes
    except ImportError:
        shared_payload_nbytes = None
    try:
        import mlx.core as mx

        assert payload_nbytes((arr_k, arr_r, mx.float32)) == arr_k.nbytes + arr_r.nbytes
        if shared_payload_nbytes is not None:
            assert shared_payload_nbytes((arr_k, arr_r, mx.float32)) == arr_k.nbytes + arr_r.nbytes
    except ImportError:
        pass

    tree = PrefixTree(chunk_tokens=4, byte_budget=1024)
    with pytest.raises(ValueError, match="byte_size mismatch"):
        tree.insert(parent=None, tokens=[1, 2, 3, 4], depth=1, chunk=b"1234", byte_size=8)
    assert "0 node(s)" in tree.summary()
    small_chunk = np.zeros((4,), dtype=np.float16)
    small_snap = np.zeros((4,), dtype=np.float32)
    snap_key = tree.insert(parent=None, tokens=[1, 2, 3, 4], depth=1, chunk=small_chunk)
    assert snap_key is not None
    assert tree.attach_snapshot(snap_key, (small_snap, np.dtype("float32"))) is True


class _NumpyMx:
    """Minimal NumPy-backed stand-in for `mlx.core` used to exercise `_run_batched_pointer` on pure CPU."""

    int32 = np.int32
    float32 = np.float32
    bfloat16 = np.float32

    @staticmethod
    def array(x: Any, dtype: Any = None) -> np.ndarray:
        return np.array(x, dtype=dtype)

    @staticmethod
    def arange(n: int, dtype: Any = None) -> np.ndarray:
        return np.arange(n, dtype=dtype)

    @staticmethod
    def zeros(shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def where(cond: Any, a: Any, b: Any) -> np.ndarray:
        return np.where(cond, a, b)

    @staticmethod
    def maximum(a: Any, b: Any) -> np.ndarray:
        return np.maximum(a, b)

    @staticmethod
    def contiguous(arr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(arr).copy()

    @staticmethod
    def concatenate(seq: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.concatenate(list(seq), axis=axis)

    @staticmethod
    def eval(*_args: Any) -> None:
        return None

    @staticmethod
    def clear_cache() -> None:
        return None

    @staticmethod
    def get_cache_memory() -> int:
        return 0


class _FakeDeltaCache:
    def __init__(self) -> None:
        self.conv_state: np.ndarray | None = None
        self.rec_state: np.ndarray | None = None


class _FakeFullCache:
    def __init__(self) -> None:
        self.keys: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.offset: int = 0


class _FakeBackbone:
    """Deterministic CPU-only hybrid backbone recording prefill waves and validating KV+GDN state."""

    def __init__(self) -> None:
        self.prefill_waves: list[tuple[int, int]] = []

    def make_cache(self) -> list[Any]:
        return [_FakeDeltaCache(), _FakeFullCache()]

    def __call__(
        self,
        input_ids: np.ndarray,
        cache: list[Any],
        *,
        valid_token_scale: np.ndarray | None = None,
        attn_mask: Any = "causal",
        pos_2d: np.ndarray | None = None,
        owner_idx: np.ndarray | None = None,
        owner_is_identity: bool = False,
        write_state: bool = True,
    ) -> np.ndarray:
        del attn_mask, pos_2d, owner_is_identity
        bsz, seq_len = input_ids.shape
        tok_f = input_ids.astype(np.float32)
        if write_state:
            self.prefill_waves.append((int(bsz), int(seq_len)))
            mask = np.ones((bsz, seq_len), dtype=np.float32) if valid_token_scale is None else valid_token_scale[:, :, 0]
            eff_tok = tok_f * mask
            step_sum = eff_tok.sum(axis=1, keepdims=True).reshape(bsz, 1, 1, 1) * np.float32(0.01)
            prev_rec = (
                cache[0].rec_state
                if cache[0].rec_state is not None
                else np.zeros((bsz, 1, 1, 1), dtype=np.float32)
            )
            cache[0].rec_state = (prev_rec + step_sum).astype(np.float32)
            cache[0].conv_state = eff_tok[:, -1:].reshape(bsz, 1, 1).astype(np.float32) * np.float32(0.01)

            k_new = eff_tok.reshape(bsz, 1, seq_len, 1).astype(np.float32)
            v_new = (eff_tok * np.float32(0.5)).reshape(bsz, 1, seq_len, 1).astype(np.float32)
            if cache[1].keys is None:
                cache[1].keys = k_new
                cache[1].values = v_new
            else:
                cache[1].keys = np.concatenate([cache[1].keys, k_new], axis=2)
                cache[1].values = np.concatenate([cache[1].values, v_new], axis=2)
            cache[1].offset += int(seq_len)
            return np.zeros((bsz, seq_len, 4), dtype=np.float32)

        owners = np.arange(bsz, dtype=np.int32) if owner_idx is None else owner_idx
        out = np.zeros((bsz, seq_len, 4), dtype=np.float32)
        for ri in range(bsz):
            ob = int(owners[ri])
            rec_val = float(cache[0].rec_state[ob, 0, 0, 0])
            kv_val = float(cache[1].keys[ob, 0, :, 0].sum()) * 0.001
            base = rec_val + kv_val
            out[ri, :, 0] = base + tok_f[ri] * 0.1
            out[ri, :, 1] = base - tok_f[ri] * 0.05
            out[ri, :, 2] = tok_f[ri] * 0.02
            out[ri, :, 3] = 1.0
        return out


def _make_fake_kev(
    *,
    prefix_tree: PrefixTree | None,
    batch_docs: int = 8,
    prefix_snapshot_stride: int = 1,
) -> tuple[Kev08bMlxBackend, _FakeBackbone, list[list[int]]]:
    kev = Kev08bMlxBackend.__new__(Kev08bMlxBackend)
    backbone = _FakeBackbone()
    kev._mx = _NumpyMx()
    kev._np = np
    kev.backbone = backbone
    kev.batch_docs = batch_docs
    kev.max_branch_batch = 16
    kev.pad_id = 0
    kev.native_temperature = 1.0
    kev.q_w = np.eye(4, dtype=np.float32)
    kev.q_b = np.zeros((4,), dtype=np.float32)
    kev.k_w = np.array(
        [
            [0.5, -0.2, 0.1, 0.0],
            [-0.1, 0.6, 0.2, 0.1],
            [0.2, 0.1, 0.7, -0.1],
            [0.0, 0.1, -0.1, 0.4],
        ],
        dtype=np.float32,
    )
    kev.k_b = np.zeros((4,), dtype=np.float32)
    kev.ptr_scale = 0.5
    kev._linear_layer_indices = [0]
    kev._full_layer_indices = [1]
    kev.prefix_tree = prefix_tree
    kev._owns_prefix_tree = False
    kev.prefix_snapshot_stride = prefix_snapshot_stride

    def _encode(state: Any, questions: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
        state_ids = [int(t) for t in state]
        rows = []
        for q in questions:
            qid = int(q["id"])
            br = [100 + qid, 200 + qid, 300 + qid, 400 + qid]
            rows.append({"ids": br, "decide": 3, "opts": [1, 2], "k": 2})
        return state_ids, rows

    kev.encode_questions = _encode  # type: ignore[method-assign]

    prefix_calls: list[list[int]] = []
    orig_prefill = kev._prefill_state_with_prefix_tree

    def _tracked_prefill(state_ids: Sequence[int], *, eval_restored_concat: bool = False) -> list[Any]:
        prefix_calls.append(list(state_ids))
        return orig_prefill(state_ids, eval_restored_concat=eval_restored_concat)

    kev._prefill_state_with_prefix_tree = _tracked_prefill  # type: ignore[method-assign]
    return kev, backbone, prefix_calls


def test_kev_prefix_cache_default_and_opt_out_cpu() -> None:
    """`Kev08bMlxBackend` defaults to `prefix_cache_mb='auto'` while preserving `None`/`0` opt-out and custom trees."""
    sig = inspect.signature(Kev08bMlxBackend.__init__)
    assert sig.parameters["prefix_cache_mb"].default == "auto"

    t_24, owns_24 = Kev08bMlxBackend._resolve_prefix_tree(
        prefix_tree=None,
        prefix_cache_mb="auto",
        prefix_chunk_tokens=DEFAULT_CHUNK_TOKENS,
        host_mem_bytes=24 << 30,
    )
    assert t_24 is not None and owns_24 is True
    assert t_24.chunk_tokens == 256 and t_24.byte_budget == (128 << 20)

    t_48, owns_48 = Kev08bMlxBackend._resolve_prefix_tree(
        prefix_tree=None,
        prefix_cache_mb="auto",
        prefix_chunk_tokens=DEFAULT_CHUNK_TOKENS,
        host_mem_bytes=48 << 30,
    )
    assert t_48 is not None and owns_48 is True
    assert t_48.byte_budget == (512 << 20)

    for disabled_val in (None, 0, -1, False):
        t_off, owns_off = Kev08bMlxBackend._resolve_prefix_tree(
            prefix_tree=None,
            prefix_cache_mb=disabled_val,  # type: ignore[arg-type]
            prefix_chunk_tokens=DEFAULT_CHUNK_TOKENS,
            host_mem_bytes=48 << 30,
        )
        assert t_off is None and owns_off is False

    t_custom_mb, owns_custom_mb = Kev08bMlxBackend._resolve_prefix_tree(
        prefix_tree=None,
        prefix_cache_mb=64,
        prefix_chunk_tokens=128,
        host_mem_bytes=48 << 30,
    )
    assert t_custom_mb is not None and owns_custom_mb is True
    assert t_custom_mb.chunk_tokens == 128 and t_custom_mb.byte_budget == (64 << 20)

    injected = PrefixTree(chunk_tokens=16, byte_budget=4096)
    for mb_val in ("auto", None, 0, 64):
        t_inj, owns_inj = Kev08bMlxBackend._resolve_prefix_tree(
            prefix_tree=injected,
            prefix_cache_mb=mb_val,  # type: ignore[arg-type]
            prefix_chunk_tokens=DEFAULT_CHUNK_TOKENS,
            host_mem_bytes=48 << 30,
        )
        assert t_inj is injected and owns_inj is False


def test_kev_hit_and_share_aware_routing_cpu() -> None:
    """Verify hit- & share-aware routing preserves B>=2 cold batching while accelerating B=1, hits, and shared prefixes."""
    q1 = [{"id": 1}]
    q2 = [{"id": 2}]

    # 1. Cold distinct B=4 (>= chunk_sz) stays on the batched wave (zero PrefixTree detours or misses).
    tree_a = PrefixTree(chunk_tokens=4, byte_budget=1 << 20)
    kev_a, bb_a, pt_calls_a = _make_fake_kev(prefix_tree=tree_a, batch_docs=8)
    kev_ref, _, _ = _make_fake_kev(prefix_tree=None, batch_docs=8)
    distinct_items = [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], q1),
        ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], q1),
        ([21, 22, 23, 24, 25, 26, 27, 28, 29, 30], q1),
        ([31, 32, 33, 34, 35, 36, 37, 38, 39, 40], q1),
    ]
    out_distinct = kev_a.score_multi_states(distinct_items)
    ref_distinct = kev_ref.score_multi_states(distinct_items)
    assert pt_calls_a == []
    assert bb_a.prefill_waves == [(4, 10)]
    assert tree_a.node_count == 0 and tree_a.hits == 0 and tree_a.misses == 0
    for got_doc, ref_doc in zip(out_distinct, ref_distinct):
        np.testing.assert_allclose(got_doc[0][0], ref_doc[0][0], atol=1e-6)
        np.testing.assert_allclose(got_doc[0][1], ref_doc[0][1], atol=1e-6)

    # 2. Intra-batch shared prefix B=4 (cold tree) routes all 4 through PrefixTree: 1 miss populates, 3 siblings hit!
    tree_b = PrefixTree(chunk_tokens=4, byte_budget=1 << 20)
    kev_b, bb_b, pt_calls_b = _make_fake_kev(prefix_tree=tree_b, batch_docs=8)
    shared = [50, 51, 52, 53, 54, 55, 56, 57]
    shared_items = [
        (shared + [1, 2], q1),
        (shared + [3, 4], q1),
        (shared + [5, 6], q1),
        (shared + [7, 8], q1),
    ]
    out_shared = kev_b.score_multi_states(shared_items)
    ref_shared = kev_ref.score_multi_states(shared_items)
    assert len(pt_calls_b) == 4
    assert tree_b.misses == 1 and tree_b.hits == 3 and tree_b.reused_tokens == 24
    assert all(b == 1 for b, _ in bb_b.prefill_waves)
    for got_doc, ref_doc in zip(out_shared, ref_shared):
        np.testing.assert_allclose(got_doc[0][0], ref_doc[0][0], atol=1e-6)
        np.testing.assert_allclose(got_doc[0][1], ref_doc[0][1], atol=1e-6)

    # 3. B=1 populates PrefixTree; subsequent B=3 mixed batch hits warm doc while batching the 2 cold docs together!
    tree_c = PrefixTree(chunk_tokens=4, byte_budget=1 << 20)
    kev_c, bb_c, pt_calls_c = _make_fake_kev(prefix_tree=tree_c, batch_docs=8)
    warm_doc = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    cold_1 = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
    cold_2 = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    _ = kev_c.score_multi_states([(warm_doc, q1)])
    assert len(pt_calls_c) == 1
    assert tree_c.misses == 1 and tree_c.hits == 0 and tree_c.node_count == 2

    pt_calls_c.clear()
    bb_c.prefill_waves.clear()
    mixed_3 = [(cold_1, q1), (warm_doc, q2), (cold_2, q1)]
    out_m3 = kev_c.score_multi_states(mixed_3)
    ref_m3 = kev_ref.score_multi_states(mixed_3)
    assert pt_calls_c == [warm_doc]
    assert tree_c.hits == 1 and tree_c.misses == 1 and tree_c.reused_tokens == 8
    assert (2, 10) in bb_c.prefill_waves
    for got_doc, ref_doc in zip(out_m3, ref_m3):
        np.testing.assert_allclose(got_doc[0][0], ref_doc[0][0], atol=1e-6)

    # 4. Single non-empty item alongside an empty-question item counts as n_nonempty_items == 1 and populates tree.
    tree_d = PrefixTree(chunk_tokens=4, byte_budget=1 << 20)
    kev_d, _, pt_calls_d = _make_fake_kev(prefix_tree=tree_d, batch_docs=8)
    out_single_plus_empty = kev_d.score_multi_states([(warm_doc, q1), (cold_1, [])])
    assert pt_calls_d == [warm_doc]
    assert out_single_plus_empty[1] == []
    assert tree_d.misses == 1 and tree_d.node_count == 2

    # 5. Short items (< chunk_sz) stay out of PrefixTree even in B=1 or shared-prefix B=2 calls.
    tree_e = PrefixTree(chunk_tokens=4, byte_budget=1 << 20)
    kev_e, bb_e, pt_calls_e = _make_fake_kev(prefix_tree=tree_e, batch_docs=8)
    _ = kev_e.score_multi_states([([1, 2, 3], q1)])
    _ = kev_e.score_multi_states([([1, 2, 3], q1), ([1, 2, 3], q2)])
    assert pt_calls_e == []
    assert tree_e.node_count == 0 and tree_e.hits == 0 and tree_e.misses == 0
    assert (2, 3) in bb_e.prefill_waves

    # 6. Full mixed batch: shared pair + 2 distinct cold long docs + short doc + empty doc.
    tree_f = PrefixTree(chunk_tokens=4, byte_budget=1 << 20)
    kev_f, bb_f, pt_calls_f = _make_fake_kev(prefix_tree=tree_f, batch_docs=8)
    mixed_all = [
        (cold_1, q1),
        (shared + [10, 11], q1),
        ([5, 6, 7], q2),
        (cold_2, []),
        (shared + [12, 13], q2),
        ([41, 42, 43, 44, 45, 46, 47, 48, 49, 50], q1),
    ]
    out_mixed = kev_f.score_multi_states(mixed_all)
    ref_mixed = kev_ref.score_multi_states(mixed_all)
    assert len(pt_calls_f) == 2
    assert tree_f.misses == 1 and tree_f.hits == 1 and tree_f.reused_tokens == 8
    assert (3, 10) in bb_f.prefill_waves  # cold_1 + short_doc + [41..50] batched together!
    assert out_mixed[3] == []
    for idx in (0, 1, 2, 4, 5):
        np.testing.assert_allclose(out_mixed[idx][0][0], ref_mixed[idx][0][0], atol=1e-6)
        np.testing.assert_allclose(out_mixed[idx][0][1], ref_mixed[idx][0][1], atol=1e-6)

    # 7. Snapshot-stride > 1: matched KV chunk WITHOUT snapshot (`reused_tokens == 0`) must NOT divert B=2 distinct docs!
    tree_g = PrefixTree(chunk_tokens=4, byte_budget=1 << 20)
    kev_g, bb_g, pt_calls_g = _make_fake_kev(prefix_tree=tree_g, batch_docs=8, prefix_snapshot_stride=2)
    # B=1 of 8-token doc with stride=2 places snapshot only at depth 2 (token 8), NOT at depth 1 (token 4).
    _ = kev_g.score_multi_states([(shared, q1)])
    assert tree_g.snapshot_count == 1
    pt_calls_g.clear()
    bb_g.prefill_waves.clear()
    # Now pass B=2 where first doc shares only 4 tokens (no snapshot -> reused_tokens == 0) + second doc is distinct.
    partial_no_snap = shared[:4] + [99, 98, 97, 96]
    _ = kev_g.score_multi_states([(partial_no_snap, q1), (cold_1[:8], q1)])
    assert pt_calls_g == []
    assert bb_g.prefill_waves == [(2, 8)]


def test_prefix_tree_rolling_hit_rate_circuit_breaker_cpu() -> None:
    """Verify rolling `< 15%` hit-rate circuit breaker skips snapshot writes on unshared corpora while preserving hits."""
    tree = PrefixTree(
        chunk_tokens=4,
        byte_budget=1 << 20,
        rolling_window=20,
        min_hit_rate=0.15,
        circuit_breaker_min_lookups=16,
    )
    assert tree.circuit_breaker_open is False
    assert tree.should_write_snapshots(profile="backfill") is True

    # Seed 1 cached prefix first so we can verify warm hits still work even after circuit breaker opens.
    seed_toks = [901, 902, 903, 904]
    k_seed = tree.insert(parent=None, tokens=seed_toks, depth=1, chunk=b"kv_s")
    assert k_seed is not None
    assert tree.attach_snapshot(k_seed, b"snap_s") is True

    # Feed 16 consecutive misses -> rolling hit rate 0/16 = 0.0 < 0.15 -> circuit breaker opens!
    for idx in range(16):
        miss_m = tree.match([idx * 10 + 1, idx * 10 + 2, idx * 10 + 3, idx * 10 + 4], needs_snapshot=True)
        assert miss_m.is_empty
        tree.record(miss_m, restored=False)

    st = tree.stats()
    assert st["rolling_hit_rate"] == 0.0
    assert st["circuit_breaker_open"] is True
    assert tree.circuit_breaker_open is True
    # Backfill / auto skips cold snapshot writes once tripped, while interactive keeps warm session caching on.
    assert tree.should_write_snapshots(profile="backfill") is False
    assert tree.should_write_snapshots(profile="auto") is False
    assert tree.should_write_snapshots(profile="interactive") is True

    # Warm lookup on already-cached seed_toks STILL hits even while circuit breaker is open!
    hit_m = tree.match(seed_toks + [99], needs_snapshot=True)
    assert hit_m.reused_tokens == 4
    tree.record(hit_m, restored=True)

    # Sustained hits (>= 15% of window) automatically close the circuit breaker again.
    for _ in range(4):
        hm = tree.match(seed_toks + [99], needs_snapshot=True)
        tree.record(hm, restored=True)
    assert tree.circuit_breaker_open is False
    assert tree.should_write_snapshots(profile="backfill") is True


def test_kev_exact_lcp_grouping_and_batched_shared_prefix_waves_cpu() -> None:
    """Verify `(prefix_32, len)` grouping + batched B=8 prefix waves + B=8..16 candidate tails without serial B=1 drop."""
    tree = PrefixTree(chunk_tokens=32, byte_budget=1 << 20)
    kev, bb, _ = _make_fake_kev(prefix_tree=tree, batch_docs=8)
    kev._batched_lcp_enabled = True
    kev.profile = "backfill"

    kev_ref, _, _ = _make_fake_kev(prefix_tree=None, batch_docs=8)

    # Create 3 distinct shared prefixes of length >= 32 (40, 44, 48 tokens), each with 4 candidate tails,
    # interleaved in scrambled order so naive length-sorting would separate siblings and legacy PrefixTree
    # would serialize all 12 items down to B=1.
    p_a = [100 + i for i in range(40)]
    p_b = [200 + i for i in range(44)]
    p_c = [300 + i for i in range(48)]
    q1 = [{"id": 1}]

    scrambled_items = []
    for cand_idx in range(4):
        scrambled_items.append((p_a + [10 + cand_idx] * (cand_idx + 2), q1))
        scrambled_items.append((p_b + [20 + cand_idx] * (cand_idx + 3), q1))
        scrambled_items.append((p_c + [30 + cand_idx] * (cand_idx + 1), q1))

    got = kev.score_multi_states(scrambled_items, batch_docs=8)
    ref = kev_ref.score_multi_states(scrambled_items, batch_docs=8)

    # All 3 unique prefixes must be prefilled together in ONE batched wave (B=3), NOT 12 serial B=1 waves!
    assert any(bsz == 3 for bsz, _ in bb.prefill_waves), f"Expected batched prefix wave B=3, got {bb.prefill_waves}"
    assert not all(bsz == 1 for bsz, _ in bb.prefill_waves)
    # PrefixTree recorded 3 cold prefix misses and 9 sibling hits!
    assert tree.misses == 3 and tree.hits == 9
    for got_doc, ref_doc in zip(got, ref):
        np.testing.assert_allclose(got_doc[0][0], ref_doc[0][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(got_doc[0][1], ref_doc[0][1], rtol=1e-5, atol=1e-3)


def test_prefix_tree_two_hit_admission_and_kev_auto_routing_cpu() -> None:
    """Verify bounded 2-hit admission prevents cold thrashing while accelerating subsequent repeated prefixes."""
    # 1. Resolve 'auto' sets admission_min_hits=2 while interactive profile, explicit budgets, and injected trees keep 1.
    t_auto, owns_auto = Kev08bMlxBackend._resolve_prefix_tree(
        prefix_tree=None,
        prefix_cache_mb="auto",
        prefix_chunk_tokens=DEFAULT_CHUNK_TOKENS,
        host_mem_bytes=24 << 30,
    )
    assert t_auto is not None and owns_auto is True
    assert t_auto.admission_min_hits == 2

    t_inter, owns_inter = Kev08bMlxBackend._resolve_prefix_tree(
        prefix_tree=None,
        prefix_cache_mb="auto",
        prefix_chunk_tokens=DEFAULT_CHUNK_TOKENS,
        host_mem_bytes=24 << 30,
        profile="interactive",
    )
    assert t_inter is not None and owns_inter is True
    assert t_inter.admission_min_hits == 1

    t_ovr, _ = Kev08bMlxBackend._resolve_prefix_tree(
        prefix_tree=None,
        prefix_cache_mb="auto",
        prefix_chunk_tokens=DEFAULT_CHUNK_TOKENS,
        host_mem_bytes=24 << 30,
        profile="interactive",
        prefix_admission_min_hits=3,
    )
    assert t_ovr is not None and t_ovr.admission_min_hits == 3

    # 2. Pure PrefixTree unit behavior: bounded window + 1st-sight filter + 2nd-sight admit + interactive bypass.
    pt = PrefixTree(chunk_tokens=4, byte_budget=1 << 20, admission_min_hits=2, admission_window=8)
    for i in range(12):
        toks = [1000 + i * 4 + k for k in range(6)]
        assert pt.admit_prefix(toks, profile="auto", sightings=1) is False
    st_pt = pt.stats()
    assert st_pt["admission_min_hits"] == 2
    assert st_pt["seen_first_chunks"] == 8  # bounded by admission_window!
    assert pt.admit_prefix([1000 + 11 * 4 + k for k in range(6)], profile="auto", sightings=1) is True
    assert pt.admit_prefix([9001, 9002, 9003, 9004], profile="interactive", sightings=1) is True
    assert pt.admit_prefix([9101, 9102, 9103, 9104], profile="auto", sightings=2) is True

    # 3. KevMlxBackend B=1 cold burst (24 unique long docs) must write 0 snapshots and NOT trip circuit breaker,
    # then a subsequent repeated passage must admit on 2nd sight and hit on 3rd..5th!
    tree = PrefixTree(chunk_tokens=4, byte_budget=1 << 20, admission_min_hits=2)
    kev, bb, pt_calls = _make_fake_kev(prefix_tree=tree, batch_docs=8)
    kev._batched_lcp_enabled = True
    kev.profile = "auto"
    kev_ref, _, _ = _make_fake_kev(prefix_tree=None, batch_docs=8)
    kev_ref._batched_lcp_enabled = True
    q1 = [{"id": 1}]
    cold_24 = [([200 + i * 10 + j for j in range(10)], q1) for i in range(24)]
    for item in cold_24:
        got_c = kev.score_multi_states([item], batch_docs=1)
        ref_c = kev_ref.score_multi_states([item], batch_docs=1)
        np.testing.assert_allclose(got_c[0][0][0], ref_c[0][0][0], atol=1e-6)
    assert tree.node_count == 0
    assert tree.bytes_held == 0
    assert tree.circuit_breaker_open is False
    assert pt_calls == []

    warm_doc = [777, 778, 779, 780, 781, 782, 783, 784, 785, 786]
    warm_ref = kev_ref.score_multi_states([(warm_doc, q1)], batch_docs=1)
    for rep_i in range(5):
        got_w = kev.score_multi_states([(warm_doc, q1)], batch_docs=1)
        np.testing.assert_allclose(got_w[0][0][0], warm_ref[0][0][0], atol=1e-6)
    assert tree.hits == 3  # rep 0 filtered, rep 1 admitted+stored (miss), reps 2..4 hit!
    assert tree.node_count == 2 and tree.reused_tokens == 24
    assert tree.circuit_breaker_open is False

    # 4. Intra-batch shared B>=2 admits immediately on first batch without waiting for a second call.
    tree_b2 = PrefixTree(chunk_tokens=4, byte_budget=1 << 20, admission_min_hits=2)
    kev_b2, _, pt_calls_b2 = _make_fake_kev(prefix_tree=tree_b2, batch_docs=8)
    kev_b2._batched_lcp_enabled = True
    kev_b2.profile = "auto"
    shared = [500, 501, 502, 503, 504, 505, 506, 507]
    _ = kev_b2.score_multi_states([(shared + [1, 2], q1), (shared + [3, 4], q1)], batch_docs=8)
    assert tree_b2.node_count == 2
    assert tree_b2.misses == 1 and tree_b2.hits == 1 and tree_b2.reused_tokens == 8
    _ = kev_b2.score_multi_states([(shared + [5, 6], q1)], batch_docs=8)
    assert tree_b2.hits == 2 and tree_b2.reused_tokens == 16


