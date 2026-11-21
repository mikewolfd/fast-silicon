"""Pure-CPU unit tests for `fast_shared.runtime_lessons` (Lessons 1, 6, 7, GRPO ANE offload, and DynamicSiliconPolicy)."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

import fast_shared
import fast_shared.runtime_lessons as rl
from fast_shared import (
    APPLE10_FUSED_SWA_SDPA_SEQ_LENS,
    MASK_NEG_VAL_FP16,
    MIN_SHARED_PREFIX_TOKENS,
    VALID_MTP_MODES,
    VALID_WORKLOADS,
    WIRED_MEMORY_CEILING_FRACTION,
    DisaggregatedCriticBudget,
    DynamicExecutionPlan,
    DynamicSiliconPolicy,
    HostSiliconProfile,
    PackedEncoderBatch,
    PackedSegmentSpan,
    PackedSequenceBin,
    PackedSequenceResult,
    SiliconBudget,
    SpeculativePolicy,
    build_block_diagonal_mask,
    find_common_prefix_len,
    pack_encoder_batches,
    pack_sequences_ffd,
    recommend_silicon_budget,
    resolve_dynamic_execution_plan,
    select_speculative_policy,
    validate_grpo_ane_offload,
    HostMemorySnapshot,
    LiveMemoryGovernor,
    MemoryAdaptation,
    advise_nocache_paths,
    compute_memory_adaptation,
    estimate_host_memory_bytes,
    nocache_open,
    read_file_nocache,
    sample_host_memory,
    set_fd_nocache,
)

_HW_APPLE9_48GB_266 = {
    "system": "Darwin",
    "machine": "arm64",
    "hw_model": "Mac16,8",
    "brand": "Apple M4 Pro",
    "gpu_family": "Apple9",
    "supports_apple9": True,
    "supports_apple10": False,
    "supports_metal4": True,
    "mem_gb": 48,
    "macos_version": "26.6",
    "mlx_enable_tf32": True,
    "mlx_metal_no_nax": False,
}

_HW_APPLE10_24GB_2721 = {
    "system": "Darwin",
    "machine": "arm64",
    "hw_model": "Mac17,2",
    "brand": "Apple M5",
    "gpu_family": "Apple10",
    "supports_apple9": True,
    "supports_apple10": True,
    "supports_metal4": True,
    "mem_gb": 24,
    "macos_version": "27.2.1",
    "mlx_enable_tf32": True,
    "mlx_metal_no_nax": False,
}


# ---------------------------------------------------------------------------
# 0. Exports & Pure-CPU Import Isolation
# ---------------------------------------------------------------------------


def test_public_exports_and_pure_cpu_import_cpu() -> None:
    import ast
    from pathlib import Path

    for sym in rl.__all__:
        assert hasattr(fast_shared, sym), f"Missing top-level export fast_shared.{sym}"
        assert sym in fast_shared.__all__, f"Missing {sym} from fast_shared.__all__"

    # Verify runtime_lessons.py itself has zero MLX / CoreML / Torch imports.
    src_path = Path(rl.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert not (imported_roots & {"mlx", "torch", "coremltools", "coreai"})

    # Verify both fast_shared and fast_shared.runtime_lessons import & run with MLX unavailable.
    code = (
        "import sys; "
        "sys.modules['mlx'] = None; sys.modules['mlx.core'] = None; "
        "import fast_shared, fast_shared.runtime_lessons as rl; "
        "b = rl.recommend_silicon_budget(24.0, workload='qlora'); "
        "p = rl.resolve_dynamic_execution_plan(batch_size=8, host_mem_gb=24.0, apple10_nax=True, macos_version='27.2.1'); "
        "assert b.memory_limit_gb == 19.2 and p.mtp_enabled is False and p.preferred_ane_backend == 'coreml_mil'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout


# ---------------------------------------------------------------------------
# 1. Lesson 1: recommend_silicon_budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host_mem_gb", [16.0, 24.0, 32.0, 36.0, 48.0, 64.0, 128.0])
@pytest.mark.parametrize("workload", VALID_WORKLOADS)
def test_recommend_silicon_budget_wired_ceiling_invariants_cpu(host_mem_gb: float, workload: str) -> None:
    b = recommend_silicon_budget(host_mem_gb=host_mem_gb, workload=workload)
    assert isinstance(b, SiliconBudget)
    assert b.workload == workload
    assert b.wired_ceiling_gb <= WIRED_MEMORY_CEILING_FRACTION * host_mem_gb + 1e-9
    assert b.effective_cap_gb <= b.wired_ceiling_gb + 1e-9
    assert 0.0 < b.cache_limit_gb < b.memory_limit_gb <= b.effective_cap_gb + 1e-9
    assert b.batch_size >= 1
    assert b.seq_pack_len >= 512
    assert b.pack_len == b.seq_pack_len
    assert b.grad_accum >= 1
    assert b.to_dict()["workload"] == workload


def test_recommend_silicon_budget_m5_24gb_and_m4_pro_48gb_targets_cpu() -> None:
    # 24 GB Base M5: 19.2 GB cap, extraction B=128, qlora pack_len=1024 grad_accum=2
    ext_24 = recommend_silicon_budget(24.0, None, "extraction")
    assert ext_24.tier == "standard"
    assert ext_24.effective_cap_gb == pytest.approx(19.2)
    assert ext_24.memory_limit_gb == pytest.approx(19.2)
    assert ext_24.cache_limit_gb == pytest.approx(1.5)
    assert ext_24.batch_size == 128

    qlora_24 = recommend_silicon_budget(24.0, workload="qlora")
    assert qlora_24.effective_cap_gb == pytest.approx(19.2)
    assert qlora_24.memory_limit_gb == pytest.approx(19.2)
    assert qlora_24.seq_pack_len == 1024
    assert qlora_24.grad_accum == 2
    assert qlora_24.seq_pack_len * qlora_24.grad_accum == 2048
    assert qlora_24.ckpt_tokens == 512

    grpo_24 = recommend_silicon_budget(24.0, workload="grpo")
    assert grpo_24.memory_limit_gb <= 18.0
    assert grpo_24.effective_cap_gb - grpo_24.memory_limit_gb >= 1.2 - 1e-6

    # >= 36 GB (36 GB and 48 GB M4 Pro): extraction B=256, qlora pack_len=2048 grad_accum=1
    for mem in (36.0, 48.0):
        ext_hi = recommend_silicon_budget(mem, workload="extraction")
        assert ext_hi.tier == "high"
        assert ext_hi.memory_limit_gb == pytest.approx(24.0)
        assert ext_hi.cache_limit_gb == pytest.approx(2.0)
        assert ext_hi.batch_size == 256

        qlora_hi = recommend_silicon_budget(mem, workload="qlora")
        assert qlora_hi.seq_pack_len == 2048
        assert qlora_hi.grad_accum == 1
        assert qlora_hi.memory_limit_gb == pytest.approx(24.0)
        assert qlora_hi.ckpt_tokens == 4096


def test_recommend_silicon_budget_metal_working_set_clamping_cpu() -> None:
    # Oversized metal_working_set_gb never breaches the 80% UMA ceiling
    clamped = recommend_silicon_budget(24.0, metal_working_set_gb=23.5, workload="qlora")
    assert clamped.effective_cap_gb == pytest.approx(19.2)
    assert clamped.memory_limit_gb <= 19.2

    # Tight metal_working_set_gb on a 48 GB host downgrades tier safely
    throttled = recommend_silicon_budget(48.0, metal_working_set_gb=18.0, workload="qlora")
    assert throttled.effective_cap_gb == pytest.approx(18.0)
    assert throttled.memory_limit_gb == pytest.approx(18.0)
    assert throttled.tier == "standard"
    assert throttled.seq_pack_len == 1024
    assert throttled.grad_accum == 2

    # Very tight working set downgrades to tight tier
    tight = recommend_silicon_budget(48.0, metal_working_set_gb=12.0, workload="extraction")
    assert tight.tier == "tight"
    assert tight.effective_cap_gb == pytest.approx(12.0)
    assert tight.batch_size == 64

    # metal_working_set_gb=0.0 treated as unreported -> fallback to 80% host RAM
    fallback = recommend_silicon_budget(24.0, metal_working_set_gb=0.0, workload="extraction")
    assert fallback.effective_cap_gb == pytest.approx(19.2)


def test_recommend_silicon_budget_validation_errors_cpu() -> None:
    with pytest.raises(ValueError, match="host_mem_gb"):
        recommend_silicon_budget(0.0, workload="extraction")
    with pytest.raises(ValueError, match="host_mem_gb"):
        recommend_silicon_budget(float("nan"), workload="extraction")
    with pytest.raises(ValueError, match="metal_working_set_gb"):
        recommend_silicon_budget(24.0, metal_working_set_gb=-1.0, workload="extraction")
    with pytest.raises(ValueError, match="wired_ceiling_fraction"):
        recommend_silicon_budget(24.0, workload="extraction", wired_ceiling_fraction=0.90)
    with pytest.raises(ValueError, match="Unsupported workload"):
        recommend_silicon_budget(24.0, workload="unknown_job")


# ---------------------------------------------------------------------------
# 2. Lesson 6: FFD Sequence Packing & Shared-System Deduplication
# ---------------------------------------------------------------------------


def test_pack_sequences_ffd_deterministic_order_and_lossless_roundtrip_cpu() -> None:
    lengths = [6, 5, 4, 4, 3, 2]
    seqs = [list(range(100 * i + 1, 100 * i + 1 + length)) for i, length in enumerate(lengths)]

    r1 = pack_sequences_ffd(seqs, max_seq_len=10, pad_to_max=True, causal=True)
    r2 = pack_sequences_ffd(seqs, max_seq_len=10, pad_to_max=True, causal=True)
    assert isinstance(r1, PackedSequenceResult)
    assert r1.num_sequences == len(seqs)
    assert r1.num_bins == 3
    assert [b.item_indices for b in r1.bins] == [(0, 2), (1, 3), (4, 5)]
    assert r1.logical_tokens == sum(lengths)
    assert r1.packed_real_tokens == sum(lengths)
    assert r1.packed_total_tokens == 30
    assert r1.deduplicated_tokens == 0

    # Bit-identical determinism
    for b1, b2 in zip(r1.bins, r2.bins):
        assert isinstance(b1, PackedSequenceBin)
        assert np.array_equal(b1.input_ids, b2.input_ids)
        assert np.array_equal(b1.position_ids, b2.position_ids)
        assert np.array_equal(b1.segment_ids, b2.segment_ids)
        assert b1.attention_mask is not None and b2.attention_mask is not None
        assert np.array_equal(b1.attention_mask, b2.attention_mask)

    # Lossless token reconstruction + per-segment RoPE position_ids reset
    recovered: dict[int, list[int]] = {}
    for b in r1.bins:
        for sp in b.segment_spans:
            assert isinstance(sp, PackedSegmentSpan)
            assert sp.position_offset == 0
            recovered[sp.orig_index] = b.input_ids[sp.start : sp.end].tolist()
            expected_pos = list(range(sp.length))
            assert b.position_ids[sp.start : sp.end].tolist() == expected_pos
            assert np.all(b.segment_ids[sp.start : sp.end] == sp.orig_index)
    assert len(recovered) == len(seqs)
    for idx, seq in enumerate(seqs):
        assert recovered[idx] == seq


def test_pack_sequences_ffd_causal_mask_and_nan_free_pad_invariants_cpu() -> None:
    seqs = [[10, 11, 12], [20, 21]]
    res = pack_sequences_ffd(seqs, max_seq_len=8, pad_to_max=True, causal=True)
    assert res.num_bins == 1
    b = res.bins[0]
    mask = b.attention_mask
    assert mask is not None
    assert mask.shape == (8, 8)
    assert mask.dtype == np.bool_

    # Segment 0 occupies [0:3], segment 1 occupies [3:5], pad occupies [5:8]
    # Causal inside segment 0
    assert np.array_equal(mask[:3, :3], np.tril(np.ones((3, 3), dtype=np.bool_)))
    # Causal inside segment 1
    assert np.array_equal(mask[3:5, 3:5], np.tril(np.ones((2, 2), dtype=np.bool_)))
    # Zero cross-segment leakage
    assert not np.any(mask[:3, 3:5])
    assert not np.any(mask[3:5, :3])
    # Zero real<->pad leakage
    assert not np.any(mask[:5, 5:])
    assert not np.any(mask[5:, :5])
    # Pad rows retain strict diagonal self-attention (NaN-free under -inf / FP16 softmax)
    assert np.array_equal(mask[5:, 5:], np.eye(3, dtype=np.bool_))

    # Additive FP16 mask check: finite, zero NaNs in softmax
    add_mask = build_block_diagonal_mask(b.segment_ids, causal=True, additive=True, dtype=np.float16)
    assert add_mask.dtype == np.float16
    assert np.all(np.isfinite(add_mask))
    assert float(add_mask.min()) == MASK_NEG_VAL_FP16
    probs = np.exp(add_mask.astype(np.float32))
    probs /= probs.sum(axis=-1, keepdims=True)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_pack_sequences_ffd_shared_system_prefix_deduplication_cpu() -> None:
    sys_prefix = list(range(1000, 1064))  # 64 shared system-prompt tokens
    suffixes = [
        list(range(2000, 2040)),  # 40
        list(range(3000, 3050)),  # 50
        list(range(4000, 4035)),  # 35
        list(range(5000, 5045)),  # 45
    ]
    seqs = [sys_prefix + suf for suf in suffixes]
    prompt_lens = [64 + 25, 64 + 30, 64 + 20, 64 + 25]

    no_dedup = pack_sequences_ffd(seqs, max_seq_len=128, dedup_shared_prefix=False)
    dedup = pack_sequences_ffd(
        seqs,
        max_seq_len=128,
        dedup_shared_prefix=True,
        min_shared_prefix_len=MIN_SHARED_PREFIX_TOKENS,
        prompt_lens=prompt_lens,
        causal=True,
    )

    assert dedup.shared_prefix_len == 64
    assert dedup.shared_prefix_ids.tolist() == sys_prefix
    assert dedup.deduplicated_tokens == (len(seqs) - 1) * 64
    assert dedup.num_bins < no_dedup.num_bins
    assert dedup.effective_compute_tokens == 64 + sum(len(s) for s in suffixes)
    assert dedup.effective_compute_tokens < no_dedup.effective_compute_tokens

    # Check RoPE continuation positions & prefix+suffix mask geometry
    b0 = dedup.bins[0]
    assert b0.shared_prefix_len == 64
    assert b0.attention_mask is not None
    assert b0.attention_mask.shape == (b0.bin_len, 64 + b0.bin_len)
    for sp in b0.segment_spans:
        assert sp.position_offset == 64
        assert b0.position_ids[sp.start : sp.end].tolist() == list(range(64, 64 + sp.length))
        # Every real suffix token attends to all 64 prefix columns
        assert np.all(b0.attention_mask[sp.start : sp.end, :64])
    # Pad rows never attend to prefix columns
    pad_rows = np.flatnonzero(b0.segment_ids < 0)
    if pad_rows.size > 0:
        assert not np.any(b0.attention_mask[pad_rows, :64])

    # Prompt-len cap stops strictly before first answer predictor (prompt_len - 1)
    tight_prompt_lens = [40, 45, 50, 55]
    assert find_common_prefix_len(seqs, min_prefix_len=32, prompt_lens=tight_prompt_lens) == 39

    # Prefix shorter than threshold does not deduplicate
    short_seqs = [list(range(1, 20)) + [100 + i, 200 + i] for i in range(4)]
    assert find_common_prefix_len(short_seqs, min_prefix_len=32) == 0


def test_pack_sequences_ffd_oversized_tile_alignment_and_swa_mask_cpu() -> None:
    seqs = [list(range(1, 151)), list(range(200, 230))]
    res = pack_sequences_ffd(seqs, max_seq_len=128, pad_to_max=False, pad_multiple=64, window_size=32)
    assert res.num_bins == 2
    # Oversized 150-token sequence stays intact (never truncated), padded to tile64 -> 192
    b_big = res.bins[0]
    assert b_big.real_tokens == 150
    assert b_big.bin_len == 192
    assert b_big.input_ids[:150].tolist() == seqs[0]
    assert b_big.attention_mask is not None
    # SWA window=32: token 100 attends to 69..100, NOT to 68
    assert bool(b_big.attention_mask[100, 69]) is True
    assert bool(b_big.attention_mask[100, 68]) is False

    # Empty sequences return clean zero result
    empty = pack_sequences_ffd([], max_seq_len=512)
    assert empty.num_bins == 0 and empty.num_sequences == 0 and empty.bins == ()


def test_pack_encoder_batches_modernbert_layout_and_cls_gather_cpu() -> None:
    rng = np.random.default_rng(42)
    lens = rng.integers(88, 212, size=40).tolist()
    seqs = []
    for idx, length in enumerate(lens):
        arr = rng.integers(200, 50000, size=int(length), dtype=np.int32)
        arr[0] = 1000 + idx  # Unique per-item [CLS] marker to verify exact gather order
        arr[-1] = 102
        seqs.append(arr)

    batches = pack_encoder_batches(seqs, max_seq_len=512, batch_size=16, pad_to_max=True)
    assert len(batches) == 3
    assert [b.num_items for b in batches] == [16, 16, 8]
    assert all(isinstance(b, PackedEncoderBatch) for b in batches)

    for b in batches:
        assert b.bin_width == 512
        assert b.input_ids.shape == (b.num_bins, 512)
        assert b.position_ids.shape == (b.num_bins, 512)
        assert b.segment_ids.shape == (b.num_bins, 512)
        assert b.attention_mask is not None
        assert b.attention_mask.shape == (b.num_bins, 1, 512, 512)
        assert b.padded_tokens < b.unpacked_padded_tokens
        assert b.packing_efficiency > 0.60

        # Gathering flat input_ids at cls_positions recovers exact per-item [CLS] markers in order
        gathered_cls = b.input_ids.reshape(-1)[b.cls_positions].tolist()
        expected_cls = [1000 + g_idx for g_idx in b.item_indices]
        assert gathered_cls == expected_cls

        # Bidirectional symmetry inside each bin's real tokens + strict segment isolation
        for bi in range(b.num_bins):
            m = b.attention_mask[bi, 0]
            assert np.array_equal(m, m.T)
            seg = b.segment_ids[bi]
            real_pos = np.flatnonzero(seg >= 0)
            pad_pos = np.flatnonzero(seg < 0)
            if real_pos.size > 0 and pad_pos.size > 0:
                assert not np.any(m[np.ix_(real_pos, pad_pos)])
                assert not np.any(m[np.ix_(pad_pos, real_pos)])
            if pad_pos.size > 0:
                assert np.array_equal(m[np.ix_(pad_pos, pad_pos)], np.eye(pad_pos.size, dtype=np.bool_))

    assert pack_encoder_batches([], max_seq_len=512) == []


# ---------------------------------------------------------------------------
# 3. Lesson 7: select_speculative_policy
# ---------------------------------------------------------------------------


def test_select_speculative_policy_thresholds_and_chipset_regimes_cpu() -> None:
    # Conservative default (apple10_nax=None): mtp_enabled == (enable_mtp and batch_size == 1)
    p1 = select_speculative_policy(batch_size=1, enable_mtp=True)
    assert isinstance(p1, SpeculativePolicy)
    assert p1.mtp_enabled is True
    assert p1.draft_k == 4
    assert p1.decoder_mode == "mtp_speculative"

    p1_gram = select_speculative_policy(batch_size=1, enable_mtp=True, has_grammar=True)
    assert p1_gram.mtp_enabled is True
    assert p1_gram.draft_k == 3

    for bsz in (2, 4, 8, 128):
        pb = select_speculative_policy(batch_size=bsz, enable_mtp=True)
        assert pb.mtp_enabled is False
        assert pb.draft_k == 0
        assert pb.decoder_mode == "ragged_b_slots"
        assert "0.56x" in pb.reason or "saturate" in pb.reason

    # Chipset-aware 2 <= B <= 4 split: Apple9 allows draft_k=2; Apple10 NAX disables immediately
    a9_b3 = select_speculative_policy(batch_size=3, enable_mtp=True, apple10_nax=False)
    assert a9_b3.mtp_enabled is True
    assert a9_b3.draft_k == 2
    assert a9_b3.decoder_mode == "mtp_speculative"

    a10_b3 = select_speculative_policy(batch_size=3, enable_mtp=True, apple10_nax=True)
    assert a10_b3.mtp_enabled is False
    assert a10_b3.draft_k == 0
    assert a10_b3.decoder_mode == "ragged_b_slots"
    assert "NAX" in a10_b3.reason

    # B >= 5 disables MTP even on Apple9
    a9_b5 = select_speculative_policy(batch_size=5, enable_mtp=True, apple10_nax=False)
    assert a9_b5.mtp_enabled is False

    # Caller / stochastic / ANE gates
    assert select_speculative_policy(batch_size=1, enable_mtp=False).mtp_enabled is False
    assert select_speculative_policy(batch_size=1, enable_mtp=True, temperature=0.7).mtp_enabled is False
    assert select_speculative_policy(batch_size=1, enable_mtp=True, use_ane=True).mtp_enabled is False

    # Prefix-tree threshold: shared_prefix_tokens >= 32 and num_candidates >= 2
    assert select_speculative_policy(batch_size=8, shared_prefix_tokens=31, num_candidates=4).prefix_tree_enabled is False
    assert select_speculative_policy(batch_size=8, shared_prefix_tokens=32, num_candidates=1).prefix_tree_enabled is False
    pt_ok = select_speculative_policy(batch_size=8, shared_prefix_tokens=32, num_candidates=2)
    assert pt_ok.prefix_tree_enabled is True
    assert pt_ok.mtp_enabled is False
    assert "2.59x" in pt_ok.prefix_tree_reason


# ---------------------------------------------------------------------------
# 4. Disaggregated GPU Actor + ANE Critic Coordinator
# ---------------------------------------------------------------------------


def test_validate_grpo_ane_offload_valid_and_violation_cases_cpu() -> None:
    # Valid 24 GB M5 run: 11.5 GB GPU actor + 1.1 GB ANE mapped (0.0 GB Metal) <= 19.2 GB
    ok = validate_grpo_ane_offload(
        host_mem_gb=24.0,
        actor_gpu_gb=11.5,
        ane_mapped_gb=1.1,
        critic_metal_gb=0.0,
        num_prompts=4,
        group_size=4,
        ane_pipe_depth=4,
    )
    assert isinstance(ok, DisaggregatedCriticBudget)
    assert ok.valid is True
    assert ok.overlapped_critic is True
    assert ok.critic_metal_gb == 0.0
    assert ok.wired_ceiling_gb == pytest.approx(19.2)
    assert ok.combined_gb == pytest.approx(12.6)
    assert ok.actor_gpu_headroom_gb == pytest.approx(6.6)
    assert ok.unified_headroom_gb == pytest.approx(6.6)
    assert ok.total_rollouts == 16
    assert ok.ane_microbatches == (4, 4, 4, 4)
    assert ok.num_ane_microbatches == 4
    assert ok.violations == ()
    assert ok.to_dict()["valid"] is True

    # Non-multiple rollout wave produces exact remainder microbatch
    rem = validate_grpo_ane_offload(48.0, 14.0, 1.0, num_prompts=3, group_size=3, ane_pipe_depth=4)
    assert rem.valid is True
    assert rem.total_rollouts == 9
    assert rem.ane_microbatches == (4, 4, 1)

    # Violation 1: Non-zero critic Metal allocation violates strict ANE offload invariant
    bad_metal = validate_grpo_ane_offload(48.0, 12.0, 1.0, critic_metal_gb=0.5)
    assert bad_metal.valid is False
    assert bad_metal.overlapped_critic is False
    assert any("0.0 GB Metal" in v for v in bad_metal.violations)

    # Violation 2: Combined actor + ANE exceeds 80% UMA ceiling (18.5 + 1.2 = 19.7 > 19.2)
    bad_uma = validate_grpo_ane_offload(24.0, 18.5, 1.2)
    assert bad_uma.valid is False
    assert bad_uma.unified_headroom_gb < 0.0
    assert bad_uma.actor_gpu_headroom_gb < 0.0
    assert any("UMA ceiling" in v for v in bad_uma.violations)

    # Violation 3: Actor exceeds tighter Metal working set cap even when under 80% UMA
    bad_ws = validate_grpo_ane_offload(48.0, 17.0, 1.0, metal_working_set_gb=16.0)
    assert bad_ws.valid is False
    assert any("effective Metal cap" in v for v in bad_ws.violations)

    with pytest.raises(ValueError, match="UMA ceiling"):
        validate_grpo_ane_offload(24.0, 18.5, 1.2, raise_on_error=True)


# ---------------------------------------------------------------------------
# 5. Unified DynamicSiliconPolicy & resolve_dynamic_execution_plan Matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host_label", "hw_dict", "expect_nax", "expect_mem", "expect_cache", "expect_ext_b", "expect_pack", "expect_ga", "expect_2721"),
    [
        ("apple9_48gb_26_6", _HW_APPLE9_48GB_266, False, 24.0, 2.0, 256, 2048, 1, False),
        ("apple10_24gb_27_2_1", _HW_APPLE10_24GB_2721, True, 19.2, 1.5, 128, 1024, 2, True),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 3, 8, 128])
def test_dynamic_silicon_policy_full_host_by_batch_matrix_cpu(
    host_label: str,
    hw_dict: dict[str, object],
    expect_nax: bool,
    expect_mem: float,
    expect_cache: float,
    expect_ext_b: int,
    expect_pack: int,
    expect_ga: int,
    expect_2721: bool,
    batch_size: int,
) -> None:
    policy = DynamicSiliconPolicy(hw=hw_dict)
    plan = policy.resolve(
        batch_size=batch_size,
        seq_len=512,
        workload="extraction",
        shared_prefix_tokens=64 if batch_size >= 8 else 0,
        num_candidates=4 if batch_size >= 8 else 1,
        temperature=0.0,
        has_grammar=False,
        mtp_mode="auto",
        model_name="laya",
    )
    assert isinstance(plan, DynamicExecutionPlan), host_label
    assert isinstance(policy.host, HostSiliconProfile)

    # Chipset & OS gates
    assert plan.apple10_nax is expect_nax
    assert plan.has_nax is expect_nax
    assert plan.os_nax_ready is True
    assert plan.metal4_int_tensor_ops_ready is True
    assert plan.ane_core_fix_ready is expect_2721
    assert plan.os_coreai_runtime_ready is expect_2721
    assert plan.preferred_ane_backend == "coreml_mil"  # Unpromoted model stays 100% pure ANE
    assert plan.fused_swa_sdpa_enabled is expect_nax
    assert plan.use_fused_swa_sdpa is expect_nax  # seq_len=512 in {384, 512}
    assert plan.tf32_nax_gemm_enabled is expect_nax
    assert plan.gemm_schedule == ("nax_tf32" if expect_nax else "apple9_simdgroup")

    # RAM tier invariants
    assert plan.memory_limit_gb == pytest.approx(expect_mem)
    assert plan.cache_limit_gb == pytest.approx(expect_cache)
    assert plan.extraction_batch_size == expect_ext_b
    assert plan.qlora_pack_len == expect_pack
    assert plan.qlora_grad_accum == expect_ga
    assert plan.ane_ring_variant == "fp16"  # No calibrated w8 bundle signaled

    # Per-call B dispatch invariants
    if batch_size == 1:
        assert plan.mtp_enabled is True
        assert plan.draft_k == 4
        assert plan.should_load_mtp_drafter is True
        assert plan.estimated_drafter_mem_mb > 0.0
        assert plan.decoder_mode == "mtp_speculative"
        assert plan.scheduler == "direct_gpu"
        assert plan.prefix_tree_enabled is False
    elif batch_size == 3:
        assert plan.scheduler == "gpu_batched"
        if not expect_nax:
            # Apple9 M4: B=3 allows draft_k=2 (M = 3 * 3 = 9 <= 12 < 16)
            assert plan.mtp_enabled is True
            assert plan.draft_k == 2
            assert plan.should_load_mtp_drafter is True
            assert plan.decoder_mode == "mtp_speculative"
        else:
            # Apple10 M5 NAX: switches immediately to ragged_b_slots (0 MB drafter under auto)
            assert plan.mtp_enabled is False
            assert plan.draft_k == 0
            assert plan.should_load_mtp_drafter is False
            assert plan.estimated_drafter_mem_mb == 0.0
            assert plan.decoder_mode == "ragged_b_slots"
    else:
        # B=8 and B=128: ragged_b_slots, hetero2_ffd, 0 MB drafter RAM under auto, prefix_tree enabled
        assert plan.mtp_enabled is False
        assert plan.draft_k == 0
        assert plan.should_load_mtp_drafter is False
        assert plan.estimated_drafter_mem_mb == 0.0
        assert plan.decoder_mode == "ragged_b_slots"
        assert plan.scheduler == "hetero2_ffd"
        assert plan.prefix_tree_enabled is True

    # JSON serialization roundtrip
    dumped = json.dumps(plan.to_dict())
    assert json.loads(dumped)["batch_size"] == batch_size


def test_dynamic_policy_chipset_hetero2_share_and_fused_swa_seq_lens_cpu() -> None:
    p_a9 = DynamicSiliconPolicy(hw=_HW_APPLE9_48GB_266)
    p_a10 = DynamicSiliconPolicy(hw=_HW_APPLE10_24GB_2721)

    for slen in (256, 384, 512, 2048):
        r9 = p_a9.resolve(batch_size=16, seq_len=slen, workload="embedding")
        r10 = p_a10.resolve(batch_size=16, seq_len=slen, workload="embedding")
        assert r10.hetero2_gpu_share > r9.hetero2_gpu_share
        assert r9.use_fused_swa_sdpa is False
        assert r10.use_fused_swa_sdpa is (slen in APPLE10_FUSED_SWA_SDPA_SEQ_LENS)

    # B=1 with grammar picks draft_k=3 on both chipsets
    assert p_a9.resolve(batch_size=1, has_grammar=True).draft_k == 3
    assert p_a10.resolve(batch_size=1, has_grammar=True).draft_k == 3


def test_dynamic_policy_os_gates_and_coreai_promotion_guard_cpu() -> None:
    # OS < 26.2 disables NAX assumptions even on Apple10 hardware
    old_os = resolve_dynamic_execution_plan(
        batch_size=3,
        host_mem_gb=24.0,
        apple10_nax=True,
        macos_version="26.1",
    )
    assert old_os.os_nax_ready is False
    assert old_os.apple10_nax is False
    assert old_os.metal4_int_tensor_ops_ready is False
    assert old_os.ane_core_fix_ready is False
    assert old_os.os_coreai_runtime_ready is False
    assert old_os.fused_swa_sdpa_enabled is False
    assert old_os.preferred_ane_backend == "coreml_mil"

    # OS 26.2 unlocks NAX, NOT Metal 4 int tensor ops
    os_262 = resolve_dynamic_execution_plan(
        batch_size=3, host_mem_gb=24.0, apple10_nax=True, macos_version="26.2"
    )
    assert os_262.os_nax_ready is True
    assert os_262.apple10_nax is True
    assert os_262.metal4_int_tensor_ops_ready is False
    assert os_262.ane_core_fix_ready is False

    # OS 26.4 unlocks Metal 4 int tensor ops, NOT 27.2.1 ANE core fix / CoreAI runtime
    os_264 = resolve_dynamic_execution_plan(
        batch_size=8, host_mem_gb=48.0, apple10_nax=False, macos_version="26.4"
    )
    assert os_264.metal4_int_tensor_ops_ready is True
    assert os_264.ane_core_fix_ready is False
    assert os_264.os_coreai_runtime_ready is False

    # OS 27.2.0 is still below 27.2.1 gate
    os_2720 = resolve_dynamic_execution_plan(
        batch_size=1, host_mem_gb=24.0, apple10_nax=True, macos_version="27.2.0", coreai_promotion_passed=True
    )
    assert os_2720.ane_core_fix_ready is False
    assert os_2720.os_coreai_runtime_ready is False
    assert os_2720.preferred_ane_backend == "coreml_mil"

    # OS 27.2.1 keeps coreml_mil unless (model, batch) explicitly passed promotion gates
    pol_2721 = DynamicSiliconPolicy(hw=_HW_APPLE10_24GB_2721, promoted_coreai_targets=[("laya_f16", 1)])
    assert pol_2721.resolve(batch_size=1, model_name="laya_f16").preferred_ane_backend == "coreai"
    assert pol_2721.resolve(batch_size=8, model_name="laya_f16").preferred_ane_backend == "coreml_mil"
    assert pol_2721.resolve(batch_size=1, model_name="pplx").preferred_ane_backend == "coreml_mil"

    # Dynamic promotion registration unlocks specific (model, batch) pair
    pol_2721.register_coreai_promotion("laya_f16", 8)
    assert pol_2721.resolve(batch_size=8, model_name="laya_f16").preferred_ane_backend == "coreai"

    # On macOS 26.6, even coreai_promotion_passed=True stays on coreml_mil
    pol_266 = DynamicSiliconPolicy(hw=_HW_APPLE9_48GB_266, promoted_coreai_targets=[("laya_f16", 1)])
    assert pol_266.resolve(batch_size=1, model_name="laya_f16", coreai_promotion_passed=True).preferred_ane_backend == "coreml_mil"


def test_dynamic_policy_ane_ring_variant_and_mtp_modes_cpu() -> None:
    # <= 24 GB chooses "w8" ONLY when calibrated bundle is present; >= 36 GB stays "fp16"
    assert DynamicSiliconPolicy(hw=_HW_APPLE10_24GB_2721, has_calibrated_ane_w8=False).host.ane_ring_variant == "fp16"
    assert DynamicSiliconPolicy(hw=_HW_APPLE10_24GB_2721, has_calibrated_ane_w8=True).host.ane_ring_variant == "w8"
    assert DynamicSiliconPolicy(hw=_HW_APPLE9_48GB_266, has_calibrated_ane_w8=True).host.ane_ring_variant == "fp16"

    pol = DynamicSiliconPolicy(hw=_HW_APPLE10_24GB_2721)
    assert VALID_MTP_MODES == ("auto", "on", "off")
    # mtp_mode="auto": lazy load (0 MB at B=8, loaded at B=1)
    assert pol.resolve(batch_size=8, mtp_mode="auto").should_load_mtp_drafter is False
    assert pol.resolve(batch_size=8, mtp_mode="auto").estimated_drafter_mem_mb == 0.0
    assert pol.resolve(batch_size=1, mtp_mode="auto").should_load_mtp_drafter is True
    # mtp_mode="off": disabled even at B=1
    off_b1 = pol.resolve(batch_size=1, mtp_mode="off")
    assert off_b1.mtp_enabled is False
    assert off_b1.should_load_mtp_drafter is False
    # mtp_mode="on": eager load signaled even when B=8 disables active verification
    on_b8 = pol.resolve(batch_size=8, mtp_mode="on")
    assert on_b8.mtp_enabled is False
    assert on_b8.should_load_mtp_drafter is True


def test_dynamic_policy_default_probe_host_hardware_called_once_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _fake_probe() -> dict[str, object]:
        calls["n"] += 1
        return dict(_HW_APPLE10_24GB_2721)

    monkeypatch.setattr(rl, "probe_host_hardware", _fake_probe)

    # DynamicSiliconPolicy() probes once on init, then resolves per-call with 0 extra probes
    policy = DynamicSiliconPolicy()
    assert calls["n"] == 1
    p1 = policy.resolve(batch_size=1)
    p8 = policy(batch_size=8, shared_prefix_tokens=128, num_candidates=4)
    assert calls["n"] == 1
    assert p1.apple10_nax is True and p1.mtp_enabled is True
    assert p8.apple10_nax is True and p8.mtp_enabled is False and p8.prefix_tree_enabled is True

    # resolve_dynamic_execution_plan() with default host args probes automatically
    one_shot = resolve_dynamic_execution_plan(batch_size=3)
    assert calls["n"] == 2
    assert one_shot.apple10_nax is True
    assert one_shot.mtp_enabled is False
    assert one_shot.memory_limit_gb == pytest.approx(19.2)
    assert math.isfinite(one_shot.hetero2_gpu_share)


def test_mach_host_memory_sampling_and_estimate_arithmetic_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    snap = sample_host_memory()
    assert isinstance(snap, HostMemorySnapshot)
    assert snap.page_size_bytes > 0
    if sys.platform == "darwin":
        assert snap.source == "mach_host_statistics64"
        assert snap.host_mem_bytes > 0
        assert snap.available_bytes > 0
        assert snap.healthy_bytes > 0
        assert snap.phys_footprint_bytes > 0
        assert snap.compression_credit_bytes >= 0
        assert snap.pressure_level in {"normal", "warning", "critical"}
        assert math.isfinite(snap.uma_fraction) and 0.0 <= snap.uma_fraction <= 1.0
        assert math.isfinite(snap.host_headroom_fraction) and 0.0 <= snap.host_headroom_fraction <= 1.0

    # Exact Splash estimateHostMemory arithmetic checks
    ps = 16384
    avail_no_comp, healthy_no_comp, cred0 = estimate_host_memory_bytes(
        free_pages=100,
        speculative_pages=10,
        external_pages=20,
        purgeable_pages=5,
        internal_pages=80,
        compressor_pages=0,
        compressed_pages=0,
        page_size_bytes=ps,
        enable_compression_credit=True,
    )
    # reclaimable = (100 - 10) + 20 + 5 = 115; comp_occ == 0 -> credit = 80 // 2 = 40
    assert cred0 == 40 * ps
    assert avail_no_comp == (115 + 40) * ps
    assert healthy_no_comp == (115 + 20) * ps

    # High compression (comp_stored // 2 >= comp_occ) -> half anonymous pages credited
    avail_hi, _, cred_hi = estimate_host_memory_bytes(
        free_pages=50,
        speculative_pages=5,
        external_pages=10,
        purgeable_pages=0,
        internal_pages=60,
        compressor_pages=10,
        compressed_pages=24,
        page_size_bytes=ps,
        enable_compression_credit=True,
    )
    assert cred_hi == 30 * ps
    assert avail_hi == (55 + 30) * ps

    # Low/no savings (comp_stored <= comp_occ) -> 0 credit
    avail_low, healthy_low, cred_low = estimate_host_memory_bytes(
        free_pages=50,
        speculative_pages=5,
        external_pages=10,
        purgeable_pages=0,
        internal_pages=60,
        compressor_pages=10,
        compressed_pages=10,
        page_size_bytes=ps,
        enable_compression_credit=True,
    )
    assert cred_low == 0 and avail_low == 55 * ps and healthy_low == 55 * ps

    # Critical disables compression credit when called with enable_compression_credit=False
    avail_off, _, cred_off = estimate_host_memory_bytes(
        free_pages=100,
        speculative_pages=0,
        external_pages=0,
        purgeable_pages=0,
        internal_pages=80,
        compressor_pages=10,
        compressed_pages=40,
        page_size_bytes=ps,
        enable_compression_credit=False,
    )
    assert cred_off == 0 and avail_off == 100 * ps

    # Non-Darwin fallback path
    monkeypatch.setattr(rl.sys, "platform", "linux")
    fb = sample_host_memory()
    assert fb.source == "fallback" and fb.pressure_level == "unknown" and fb.host_mem_bytes == 0


def _fake_snap(
    *,
    host_mem_gb: float = 24.0,
    available_gb: float = 12.0,
    healthy_gb: float = 10.0,
    phys_gb: float = 4.0,
    file_backed_gb: float = 4.0,
    file_cache_min_gb: float = 0.5,
    pressure: str = "normal",
) -> HostMemorySnapshot:
    host_b = int(host_mem_gb * (1 << 30))
    avail_b = int(available_gb * (1 << 30))
    phys_b = int(phys_gb * (1 << 30))
    return HostMemorySnapshot(
        platform="darwin",
        page_size_bytes=16384,
        host_mem_bytes=host_b,
        available_bytes=avail_b,
        healthy_bytes=int(healthy_gb * (1 << 30)),
        phys_footprint_bytes=phys_b,
        compressed_bytes=0,
        file_backed_bytes=int(file_backed_gb * (1 << 30)),
        inactive_bytes=0,
        file_cache_minimum_bytes=int(file_cache_min_gb * (1 << 30)),
        free_pages=100,
        speculative_pages=0,
        external_pages=100,
        purgeable_pages=0,
        internal_pages=100,
        compressor_pages=0,
        compressed_pages=0,
        compression_credit_bytes=0,
        pressure_level=pressure,  # type: ignore[arg-type]
        pressure_code={"normal": 1, "warning": 2, "critical": 4}.get(pressure, 0),
        uma_fraction=phys_b / float(host_b),
        host_headroom_fraction=avail_b / float(host_b),
        source="synthetic",
    )


def test_live_memory_governor_adaptation_hysteresis_and_mlx_hooks_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Healthy headroom preserves full B_slots and cache limit
    ad_ok = compute_memory_adaptation(
        base_b_slots=128,
        base_cache_limit_mb=256,
        snapshot=_fake_snap(host_mem_gb=24.0, available_gb=14.0, healthy_gb=12.0, phys_gb=4.0),
    )
    assert isinstance(ad_ok, MemoryAdaptation)
    assert ad_ok.b_slots == 128 and ad_ok.cache_limit_mb == 256
    assert ad_ok.throttle_tier == "none" and ad_ok.should_clear_cache is False

    # 2. OS background warning with ample headroom trims cache only (preserves B_slots)
    ad_bg_warn = compute_memory_adaptation(
        base_b_slots=256,
        base_cache_limit_mb=256,
        snapshot=_fake_snap(host_mem_gb=48.0, available_gb=9.0, healthy_gb=6.5, phys_gb=0.2, pressure="warning"),
    )
    assert ad_bg_warn.b_slots == 256 and ad_bg_warn.cache_limit_mb == 128
    assert ad_bg_warn.throttle_tier == "mild" and ad_bg_warn.should_clear_cache is True

    # 3. Approaching 80% UMA (mild at >=65% = 15.6 GB on 24 GB) narrows slots to 3/4
    ad_mild = compute_memory_adaptation(
        base_b_slots=128,
        base_cache_limit_mb=256,
        snapshot=_fake_snap(host_mem_gb=24.0, available_gb=6.0, healthy_gb=5.0, phys_gb=16.0),
    )
    assert ad_mild.throttle_tier == "mild" and ad_mild.b_slots == 96 and ad_mild.cache_limit_mb == 128

    # 4. Warn (>=72% UMA or healthy < warn headroom or active OS warning) halves B_slots & clears cache
    ad_warn = compute_memory_adaptation(
        base_b_slots=128,
        base_cache_limit_mb=256,
        snapshot=_fake_snap(host_mem_gb=24.0, available_gb=4.0, healthy_gb=2.0, phys_gb=17.5, pressure="warning"),
    )
    assert ad_warn.throttle_tier == "warn" and ad_warn.b_slots == 64
    assert ad_warn.cache_limit_mb == 128 and ad_warn.should_clear_cache is True

    # 5. Critical (>=80% UMA cliff on 24 GB = 19.2 GB, or OS critical) quarters B_slots & caps cache <= 64 MB
    ad_crit = compute_memory_adaptation(
        base_b_slots=128,
        base_cache_limit_mb=256,
        min_b_slots=4,
        snapshot=_fake_snap(host_mem_gb=24.0, available_gb=2.5, healthy_gb=1.8, phys_gb=19.4, pressure="normal"),
    )
    assert ad_crit.throttle_tier == "critical" and ad_crit.b_slots == 32
    assert ad_crit.cache_limit_mb == 64 and ad_crit.should_clear_cache is True

    # 6. Stateful hysteresis: critical -> 1 healthy poll stays throttled -> 2nd healthy poll relaxes!
    cache_calls: list[int] = []
    clear_calls: list[int] = []
    monkeypatch.setattr("fast_shared.embeddings.set_mlx_cache_limit", lambda mb: cache_calls.append(int(mb)))
    monkeypatch.setattr("fast_shared.bench.clear_mlx_arena", lambda mb: clear_calls.append(int(mb)))

    gov = LiveMemoryGovernor(poll_interval_sec=0.0, clean_polls_to_relax=2)
    ad_crit_snap = _fake_snap(phys_gb=19.5)
    step1 = gov.adapt(base_b_slots=128, base_cache_limit_mb=256, apply_mlx=True, snapshot=ad_crit_snap)
    assert step1.throttle_tier == "critical" and step1.b_slots == 32
    assert cache_calls[-1] == 64 and clear_calls == [0]

    healthy_snap = _fake_snap(host_mem_gb=24.0, available_gb=14.0, healthy_gb=12.0, phys_gb=4.0)
    step2 = gov.adapt(base_b_slots=128, base_cache_limit_mb=256, apply_mlx=True, snapshot=healthy_snap)
    assert step2.throttle_tier == "critical" and step2.reason.startswith("hysteresis_hold")
    step3 = gov.adapt(base_b_slots=128, base_cache_limit_mb=256, apply_mlx=True, snapshot=healthy_snap)
    assert step3.throttle_tier == "none" and step3.b_slots == 128 and cache_calls[-1] == 256
    assert gov.stats()["adaptations_triggered"] >= 1 and gov.stats()["pressure_events"] >= 1 and gov.stats()["cache_clears"] >= 1
    assert ad_crit_snap.uma_fraction >= 0.80

    # 7. DynamicSiliconPolicy convenience wrappers propagate snapshot cleanly
    pol = DynamicSiliconPolicy(hw=_HW_APPLE10_24GB_2721)
    assert isinstance(pol.sample_host_memory(), HostMemorySnapshot)
    pol_ad = pol.adapt_runtime_limits(base_b_slots=64, snapshot=_fake_snap(phys_gb=19.5))
    assert pol_ad.b_slots == 16 and pol_ad.throttle_tier == "critical"


def test_nocache_io_helpers_parity_and_guards_cpu(tmp_path) -> None:
    payload = (b"FastSiliconNoCacheShardHeader\n" + bytes(range(256)) * 2048)[: 600 * 1024]
    p = tmp_path / "shard.bin"
    p.write_bytes(payload)

    with pytest.raises(ValueError, match="unbuffered"):
        with nocache_open(p, "r", buffering=0):
            pass

    got_read = read_file_nocache(p, chunk_size=65536)
    assert got_read == payload

    with nocache_open(p, "rb") as fh:
        assert fh.read() == payload

    with nocache_open(p, "r", encoding="utf-8", errors="ignore") as tfh:
        assert len(tfh.read()) > 0

    assert advise_nocache_paths([p, tmp_path / "missing.bin"]) == 1
    assert set_fd_nocache(-1, True) is False
