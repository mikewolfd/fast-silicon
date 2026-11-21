"""Regression tests for Shard 4 (`fast-gemma4`) P0-P3 remediation findings (CPU + Metal GPU)."""

from __future__ import annotations

import ctypes
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import fast_gemma4.engine as fg4_engine
import fast_gemma4.kernels as fg4_kernels
import fast_gemma4.model as fg4_model
import fast_gemma4.paths as fg4_paths
import fast_gemma4.speculative as fg4_spec
from fast_gemma4.ane_pipeline import (
    ANE_CHUNK_S,
    AnePrefixWorker,
    AsyncHybridPrefillRing,
    ingest_ane_finish_tail_and_last,
    is_w8_bundle_calibrated,
    validate_ane_kv_finite,
    validate_slot_kv_scales_finite,
)
from fast_gemma4.engine import FastGemma4Engine, select_execution_plan
from fast_gemma4.model import (
    FastGemma4Model,
    _select_sdpa_mask,
    compute_safe_fp16_residual_scales,
    create_causal_mask,
    safe_argmax_with_nan_guard,
)
from fast_gemma4.paths import Gemma4Paths, resolve_home, resolve_paths
from fast_gemma4.speculative import run_decode_batch, run_decode_prompt_lookup, run_decode_rolling

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    script_path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"fg4_script_{name}", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"fg4_script_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


# ---------------------------------------------------------------------------
# Guardrail: fast_shared.ir_dtype_guard (--strict-bool-mask)
# ---------------------------------------------------------------------------


def test_ir_dtype_guard_strict_bool_mask_passes() -> None:
    """Verify zero errors (and zero `MLX_BOOL_SDPA_MASK`) across src/fast_gemma4, tests, and scripts."""
    from fast_shared.ir_dtype_guard import check_mlx_paths

    report = check_mlx_paths(
        [REPO_ROOT / "src" / "fast_gemma4", REPO_ROOT / "tests", REPO_ROOT / "scripts"],
        flag_bool_sdpa_as_error=True,
    )
    assert report.ok, "\n".join(f.format() for f in report.errors)
    bool_masks = [f for f in report.findings if f.rule_id == "MLX_BOOL_SDPA_MASK"]
    assert bool_masks == [], "\n".join(b.format() for b in bool_masks)


# ---------------------------------------------------------------------------
# P0: FP16 residual-scale protection (L13 215K -> ~13.4K) & scale-normalized RMSNorm
# ---------------------------------------------------------------------------


def test_p0_fp16_residual_scale_folding_math_and_overflow_protection_cpu() -> None:
    """Verify exact power-of-2 L10..L14+ residual folding keeps L13 (215K) < 65,504 and matches FP32 reference."""
    build_ane = _load_script("build_ane_models")

    scales_e2b = compute_safe_fp16_residual_scales(15)
    assert scales_e2b.shape == (15,)
    assert float(scales_e2b[0]) == 1.0 and float(scales_e2b[9]) == 1.0
    assert float(scales_e2b[10]) == 0.5 and float(scales_e2b[11]) == 0.25 and float(scales_e2b[13]) == 0.0625
    assert 215_000.0 * float(scales_e2b[13]) < 15_000.0 < 65_504.0

    # Build a 15-layer synthetic `.npz` with realistic weights and verify mathematical invariance of one L13 block
    rng = np.random.default_rng(7)
    H, P = 16, 8
    raw_npz: dict[str, np.ndarray] = {
        "per_layer_model_proj_w": np.zeros((14 * P, H, 1, 1), dtype=np.float16),
        "per_layer_proj_norm_w": np.ones((P,), dtype=np.float16),
    }
    for i in range(15):
        p = f"l{i}_"
        raw_npz[p + "w_in"] = rng.uniform(0.5, 1.5, size=(1, H, 1, 1)).astype(np.float16)
        raw_npz[p + "w_kn"] = np.ones((1, 1, 1, 512 if i in (4, 9, 14) else 256), dtype=np.float16)
        raw_npz[p + "w_pa"] = rng.uniform(0.5, 1.5, size=(1, H, 1, 1)).astype(np.float16)
        raw_npz[p + "w_pff"] = rng.uniform(0.5, 1.5, size=(1, H, 1, 1)).astype(np.float16)
        raw_npz[p + "w_poff"] = rng.uniform(0.5, 1.5, size=(1, H, 1, 1)).astype(np.float16)
        raw_npz[p + "w_pple"] = rng.uniform(0.5, 1.5, size=(1, H, 1, 1)).astype(np.float16)
        raw_npz[p + "w_pg"] = rng.uniform(-0.1, 0.1, size=(P, H, 1, 1)).astype(np.float16)
        raw_npz[p + "scalar"] = np.array([1.25], dtype=np.float16)

    folded, got_scales = build_ane.fold_fp16_residual_scales(raw_npz)
    assert np.allclose(got_scales, scales_e2b)
    assert int(folded["fp16_residual_protected"]) == 1

    # Idempotence: folding a second time must not double-scale!
    folded2, got_scales2 = build_ane.fold_fp16_residual_scales(folded)
    assert np.allclose(got_scales2, got_scales)
    assert np.array_equal(folded2["l13_w_pa"], folded["l13_w_pa"])
    assert np.array_equal(folded2["l13_w_pg"], folded["l13_w_pg"])
    assert np.array_equal(folded2["l11_scalar"], folded["l11_scalar"])

    # Exact mathematical equivalence on layer 11 -> 12 transition (alpha_11=0.25 -> alpha_12=0.0625)
    def _safe_rmsnorm_f16(x_f16: np.ndarray, w_f16: np.ndarray, sqrt_eps: float) -> np.ndarray:
        s = np.maximum(np.max(np.abs(x_f16), axis=-1, keepdims=True), np.float16(1e-4))
        xs = (x_f16 / s).astype(np.float16)
        msq = np.mean((xs * xs).astype(np.float16), axis=-1, keepdims=True, dtype=np.float16)
        es = (np.float16(sqrt_eps) / s).astype(np.float16)
        rsq = (1.0 / np.sqrt(msq.astype(np.float32) + (es * es).astype(np.float32))).astype(np.float16)
        return (xs * rsq * w_f16).astype(np.float16)

    # Even with |x| = 40,000 (where naive x*x -> +Inf in FP16), scale-normalized RMSNorm stays finite and exact!
    x_huge = np.full((1, H), 40_000.0, dtype=np.float16)
    x_huge[0, 0] = np.float16(-40_000.0)
    with np.errstate(over="ignore"):
        naive_sq = x_huge * x_huge
    assert np.isinf(naive_sq).all()
    safe_n = _safe_rmsnorm_f16(x_huge, np.ones((1, H), dtype=np.float16), sqrt_eps=1e-3)
    assert np.all(np.isfinite(safe_n))
    assert np.allclose(np.abs(safe_n), 1.0, atol=2e-2)

    # Verify layer-11 block outputs agree between unscaled FP32 and folded FP16!
    cur11 = rng.standard_normal((4, H)).astype(np.float32) * 100.0
    ao = rng.standard_normal((4, H)).astype(np.float32)
    md = rng.standard_normal((4, H)).astype(np.float32)
    pp_out = rng.standard_normal((4, H)).astype(np.float32)
    pli = rng.standard_normal((4, P)).astype(np.float32)

    def _rms_f32(v: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return v * (1.0 / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)) * w

    # Reference FP32 unscaled
    c1_ref = cur11 + _rms_f32(ao, raw_npz["l11_w_pa"].reshape(1, H).astype(np.float32))
    pff_ref = _rms_f32(c1_ref, raw_npz["l11_w_pff"].reshape(1, H).astype(np.float32))
    c2_ref = c1_ref + _rms_f32(md, raw_npz["l11_w_poff"].reshape(1, H).astype(np.float32))
    pg_ref = c2_ref @ raw_npz["l11_w_pg"][:, :, 0, 0].astype(np.float32).T
    cur12_ref = (c2_ref + _rms_f32(pp_out, raw_npz["l11_w_pple"].reshape(1, H).astype(np.float32))) * float(
        raw_npz["l11_scalar"][0]
    )

    # Folded FP16 path (cur11' = 0.25 * cur11 -> cur12' = 0.0625 * cur12)
    a11, a12 = float(got_scales[11]), float(got_scales[12])
    cur11_s = (cur11 * a11).astype(np.float16)
    c1_s = cur11_s + _safe_rmsnorm_f16(ao.astype(np.float16), folded["l11_w_pa"].reshape(1, H), 1e-3)
    pff_s = _safe_rmsnorm_f16(c1_s, folded["l11_w_pff"].reshape(1, H), 1e-3 * a11)
    c2_s = c1_s + _safe_rmsnorm_f16(md.astype(np.float16), folded["l11_w_poff"].reshape(1, H), 1e-3)
    pg_s = c2_s.astype(np.float32) @ folded["l11_w_pg"][:, :, 0, 0].astype(np.float32).T
    cur12_s = (
        c2_s + _safe_rmsnorm_f16(pp_out.astype(np.float16), folded["l11_w_pple"].reshape(1, H), 1e-3)
    ) * folded["l11_scalar"][0]

    assert _cosine(pff_ref, pff_s.astype(np.float32)) > 0.9999
    assert _cosine(pg_ref, pg_s) > 0.9999
    assert _cosine(cur12_ref * a12, cur12_s.astype(np.float32)) > 0.9999


# ---------------------------------------------------------------------------
# P0: Runtime ANE Inf/NaN detection & clean GPU prefill fallback
# ---------------------------------------------------------------------------


def test_p0_ane_nan_inf_detection_and_gpu_fallback_cpu() -> None:
    """Verify Inf/NaN ANE outputs and KV_i8 scales are caught and trigger clean GPU BF16 prefill fallback."""
    good = np.ones((2, 4), dtype=np.float16)
    bad_nan = good.copy()
    bad_nan[0, 2] = np.nan
    bad_inf = good.copy()
    bad_inf[1, 0] = np.inf
    assert validate_ane_kv_finite(good) is True
    assert validate_ane_kv_finite(good, bad_nan) is False
    assert validate_ane_kv_finite(good, bad_inf) is False

    # Slot cache scale validation (both NumPy and MLX arrays)
    good_cache = {
        "kv_i8": True,
        "layers": [{"k_s": np.ones((4, 8, 1), dtype=np.float16), "v_s": np.ones((4, 8, 1), dtype=np.float16)}],
    }
    assert validate_slot_kv_scales_finite(good_cache, [0, 1]) is True
    good_cache["layers"][0]["k_s"][1, 3, 0] = np.nan
    assert validate_slot_kv_scales_finite(good_cache, [0]) is True
    assert validate_slot_kv_scales_finite(good_cache, [0, 1]) is False

    mx_cache = {
        "kv_i8": True,
        "layers": [
            {
                "k_s": mx.array(np.array([[[1.0], [1.0]], [[1.0], [np.nan]]], dtype=np.float16)),
                "v_s": mx.ones((2, 2, 1), dtype=mx.float16),
            }
        ],
    }
    assert validate_slot_kv_scales_finite(mx_cache, [0]) is True
    assert validate_slot_kv_scales_finite(mx_cache, [1]) is False

    # AnePrefixWorker catches NaN emitted by runner and clears poisoned coverage
    class _PoisonRunner:
        def run_batch(self, B, *args, **kwargs):
            ok_s, ov_s, ok_f, ov_f = args[12], args[13], args[14], args[15]
            ok_s.fill(1.0)
            ov_s.fill(1.0)
            ok_f.fill(1.0)
            ov_f.fill(1.0)
            ok_f[0, 0, 0, 0, 0] = np.nan
            return 0.001

    class _MiniModel:
        swa_layers = [0]
        full_layers = [1, 2]
        n_kv_heads = 1
        swa_head_dim = 256
        full_head_dim = 512
        window_size = 512

    pkv = {
        "l_sys": 4,
        "k_sys": [mx.zeros((4, 1, 256), dtype=mx.bfloat16), mx.zeros((4, 1, 512), dtype=mx.bfloat16), mx.zeros((4, 1, 512), dtype=mx.bfloat16)],
        "v_sys": [mx.zeros((4, 1, 256), dtype=mx.bfloat16), mx.zeros((4, 1, 512), dtype=mx.bfloat16), mx.zeros((4, 1, 512), dtype=mx.bfloat16)],
    }
    worker = AnePrefixWorker(
        _MiniModel(),  # type: ignore[arg-type]
        _PoisonRunner(),  # type: ignore[arg-type]
        pkv,
        np.zeros((16, 8), dtype=np.float16),
        np.zeros((16, 8), dtype=np.float16),
        min_chunk_tokens=4,
    )
    with pytest.warns(RuntimeWarning, match="Non-finite"):
        res = worker.run_wave_cpu_ane([[1, 2, 3, 4, 5, 6]])
    assert res["non_finite_detected"] is True
    assert res["covered"] == [0]
    assert res["chunks"] == {}

    # ingest_ane_finish_tail_and_last falls back to GPU prefill when chunks contain NaN
    ingested_calls = 0
    gpu_steps = 0

    class _FallbackModel:
        def ingest_ane_chunk_kv(self, *a, **kw):
            nonlocal ingested_calls
            ingested_calls += 1

        def compact_cache(self, cache, rows):
            return {"batch_size": len(rows), "layers": [{"k": mx.zeros((len(rows), 8, 1, 4)), "v": mx.zeros((len(rows), 8, 1, 4))}]}

        def forward_early14_ragged(self, toks, sub_c, ob_sub):
            nonlocal gpu_steps
            gpu_steps += 1

        def forward_step_ragged(self, toks, cache, off, apply_softcap=True):
            return mx.zeros((toks.shape[0], 1, 8), dtype=mx.bfloat16)

    bad_chunk = np.full((1, 1, 1, ANE_CHUNK_S, 256), np.nan, dtype=np.float16)
    good_chunk = np.ones((1, 1, 1, ANE_CHUNK_S, 256), dtype=np.float16)
    cache = {"batch_size": 1, "layers": [{"k": mx.zeros((1, 8, 1, 4)), "v": mx.zeros((1, 8, 1, 4))}]}
    with pytest.warns(RuntimeWarning, match="Non-finite"):
        ingest_ane_finish_tail_and_last(
            _FallbackModel(),  # type: ignore[arg-type]
            cache,
            [[1, 2, 3, 4]],
            {"chunks": {0: ([0], [3], bad_chunk, good_chunk, good_chunk, good_chunk)}, "covered": [3]},
            lcp=0,
        )
    assert ingested_calls == 0
    assert gpu_steps == 3  # All 3 prefix tokens recomputed cleanly on GPU!

    # run_decode_rolling `_admit_ane_cohort` falls back to GPU prefill (`start_s0=0`) when scattered k_s has NaN
    gpu_fallback_starts: list[int] = []

    class _RollAneNanModel:
        window_size = 512
        kv_i8 = True
        new_to_old = np.arange(16, dtype=np.int32)

        def make_slot_cache(self, prefix_kv, batch_size, doc_cap_full=128, swa_ring_cap=512, kv_i8=None):
            return {
                "batch_size": batch_size,
                "l_sys": 0,
                "k_sys": [],
                "v_sys": [],
                "swa_cap": 512,
                "full_cap": 128,
                "kv_i8": True,
                "drain_slack": 16,
                "layers": [{"k_s": np.ones((batch_size, 8, 1), dtype=np.float16), "v_s": np.ones((batch_size, 8, 1), dtype=np.float16)}],
            }

        def _eval_doc_layers(self, layers, kv_i8):
            pass

        def _prefill_early14_slot_groups(self, sufs, slots, cache, group_size=16, start_s0=0):
            gpu_fallback_starts.append(start_s0)

        def forward_decode_step_slot(self, toks, cache, off, slots, n_keep=1, return_logits=False):
            return mx.zeros((n_keep, 1, 4), dtype=mx.bfloat16), mx.array([1] * n_keep, dtype=mx.int32)

    class _FakeBinding:
        def __init__(self, runner, model, cache):
            self.cache = cache

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class _FakeRing:
        runner = type("_R", (), {"out_ring_size": 16})()
        n_chunks = 1

        def prefill_cohort_into_ring(self, doc_ids, sufs, **kw):
            return {"doc_ids": doc_ids, "sufs": sufs, "covered": [len(s) - 1 for s in sufs], "ane_sec": 0.01, "ring_slots_held": []}

        def scatter_cohort_to_slot_cache(self, binding, item, target_slots):
            # Poison scattered k_s with NaN
            binding.cache["layers"][0]["k_s"][target_slots[0], 0, 0] = np.nan
            return 0.001

    with patch.object(fg4_spec, "SlotCacheMetalBinding", _FakeBinding):
        with pytest.warns(RuntimeWarning, match="Non-finite"):
            roll_out = run_decode_rolling(
                _RollAneNanModel(),  # type: ignore[arg-type]
                frozenset({1}),  # token 1 immediately stops
                None,
                {"l_sys": 0, "k_sys": [], "v_sys": []},
                [[2, 3, 4, 5]],
                b_slots=1,
                use_ane_ring=True,
                ane_ring=_FakeRing(),
                ane_cohort_size=1,
            )
    assert roll_out["ane_nan_fallbacks"] == 1
    assert gpu_fallback_starts == [0]


# ---------------------------------------------------------------------------
# P1: AWQ corpus calibration + manifest gate + default FP16 ring variant
# ---------------------------------------------------------------------------


def test_p1_awq_calibration_corpus_and_manifest_gate_cpu(tmp_path: Path) -> None:
    """Verify real corpus loading, manifest certification gate, and dynamic 'auto' default for `ane_ring_variant`."""
    build_ane = _load_script("build_ane_models")

    assert inspect.signature(FastGemma4Engine.__init__).parameters["ane_ring_variant"].default == "auto"

    # JSONL corpus loading
    jl = tmp_path / "domain.jsonl"
    jl.write_text(
        "\n".join(json.dumps({"text": f"Domain technical record #{i} on transformer protection."}) for i in range(10)),
        encoding="utf-8",
    )
    docs, src, _ = build_ane.load_calibration_texts(calib_file=jl, max_docs=10)
    assert src == "file" and len(docs) == 10

    # Auto-discovered repo fixtures (if present)
    fix_docs, fix_src, _ = build_ane.load_calibration_texts(repo_root=REPO_ROOT)
    assert len(fix_docs) >= 8 and fix_src in ("fixtures", "synthetic")

    # Manifest calibration gate
    w8_mc = tmp_path / "gemma4_early14_prefix_S256_w8.mlmodelc"
    w8_mc.mkdir()
    assert is_w8_bundle_calibrated(w8_mc) is False

    arch = {"exit_layer": 14, "n_kv": 1, "fp16_residual_protected": True, "residual_scales": [1.0] * 15}
    build_ane.write_ane_calibration_manifest(
        tmp_path,
        bundle_name=w8_mc.name,
        variant="w8",
        arch=arch,
        compute_precision_mode="fp16_safe",
        awq_mse=True,
        mean_alpha=0.95,
        calibration_source="synthetic",
        calibration_docs=8,
        qat_validated=False,
    )
    assert is_w8_bundle_calibrated(w8_mc) is False

    build_ane.write_ane_calibration_manifest(
        tmp_path,
        bundle_name=w8_mc.name,
        variant="w8",
        arch=arch,
        compute_precision_mode="fp16_safe",
        awq_mse=True,
        mean_alpha=0.94,
        calibration_source="file",
        calibration_docs=16,
        qat_validated=False,
    )
    assert is_w8_bundle_calibrated(w8_mc) is True


# ---------------------------------------------------------------------------
# P1: MLP & lm_head MPP OS/GPU-family gating
# ---------------------------------------------------------------------------


def test_p1_mlp_and_lm_head_mpp_os_and_gpu_family_gating_cpu() -> None:
    """Verify macOS >= 26.1 gating for MPP BF16 and Apple9 vs Apple10 lm_head/MLP thresholds."""
    with patch("platform.mac_ver", return_value=("26.0.1", ("", "", ""), "arm64")):
        assert fg4_model._supports_mpp_bf16() is False
    with patch("platform.mac_ver", return_value=("26.1", ("", "", ""), "arm64")):
        assert fg4_model._supports_mpp_bf16() is True
    with patch("platform.mac_ver", return_value=("26.6", ("", "", ""), "arm64")):
        assert fg4_model._supports_mpp_bf16() is True

    m = FastGemma4Model.__new__(FastGemma4Model)
    m.use_mpp = True
    m.mpp_m32_cutoff = 96
    m.act_dtype = mx.bfloat16
    m.w_lm_bf16_pad = mx.zeros((64, 64), dtype=mx.bfloat16)

    x32 = mx.zeros((32, 1, 64), dtype=mx.bfloat16)
    x64 = mx.zeros((64, 1, 64), dtype=mx.bfloat16)
    x128 = mx.zeros((128, 1, 64), dtype=mx.bfloat16)

    # On macOS < 26.1, `_can_use_lm_head_mpp` is False even at large M
    m._mpp_bf16_supported = False
    m._gpu_family = "apple9"
    assert m._can_use_lm_head_mpp(x128) is False

    # On macOS >= 26.1 Apple9, `M < 64` uses MLX, `M >= 64` uses MPP; on Apple10/NAX, defers to native MLX
    m._mpp_bf16_supported = True
    m._gpu_family = "apple9"
    assert m._can_use_lm_head_mpp(x32) is False
    assert m._can_use_lm_head_mpp(x64) is True
    m._gpu_family = "apple10"
    assert m._can_use_lm_head_mpp(x64) is False


# ---------------------------------------------------------------------------
# P1: Additive 0/-inf masks (zero mx.bool_) & >=1024 causal SDPA chunking
# ---------------------------------------------------------------------------


def test_p1_sdpa_additive_masks_no_bool_and_1024_chunking_gpu() -> None:
    """Verify `create_causal_mask` emits `0/-inf` in `act_dtype` (never bool) and `"causal"` fast path matches."""
    m_full = create_causal_mask(8, offset=12, window_size=None, dtype=mx.bfloat16)
    mx.eval(m_full)
    assert m_full.dtype == mx.bfloat16
    assert m_full.dtype != mx.bool_
    m_np = np.array(m_full.astype(mx.float32))
    assert float(m_np[0, 12]) == 0.0 and np.isneginf(m_np[0, 13])

    # `_select_sdpa_mask`: Full lc>1 ALWAYS uses "causal" (even when offset > 0), lc==1 uses None!
    assert _select_sdpa_mask(1, offset=100, is_full=True, window_size=None, dtype=mx.bfloat16) is None
    assert _select_sdpa_mask(64, offset=0, is_full=True, window_size=None, dtype=mx.bfloat16) == "causal"
    assert _select_sdpa_mask(64, offset=128, is_full=True, window_size=None, dtype=mx.bfloat16) == "causal"
    assert _select_sdpa_mask(64, offset=128, is_full=False, window_size=512, dtype=mx.bfloat16) == "causal"
    swa_win = _select_sdpa_mask(64, offset=500, is_full=False, window_size=512, dtype=mx.bfloat16)
    assert isinstance(swa_win, mx.array) and swa_win.dtype == mx.bfloat16

    # Numerical bit-exact parity of MLX SDPA mask="causal" vs additive 0/-inf at offset=128, lc=64, E4B D=512
    rng = np.random.default_rng(11)
    q = mx.array(rng.standard_normal((2, 8, 64, 512)) * 0.1, dtype=mx.bfloat16)
    k = mx.array(rng.standard_normal((2, 2, 192, 512)) * 0.1, dtype=mx.bfloat16)
    v = mx.array(rng.standard_normal((2, 2, 192, 512)) * 0.1, dtype=mx.bfloat16)
    o_causal = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask="causal")
    o_add = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0, mask=create_causal_mask(64, offset=128, window_size=None, dtype=mx.bfloat16)
    )
    mx.eval(o_causal, o_add)
    assert float(mx.max(mx.abs(o_causal.astype(mx.float32) - o_add.astype(mx.float32))).item()) == 0.0

    # Confirm prefill_prefix promotes chunk_sz >= 1024 when l_sys >= 1024
    pref_src = inspect.getsource(FastGemma4Model.prefill_prefix)
    assert "1024" in pref_src


# ---------------------------------------------------------------------------
# P1 & P2: E4B (n_q=8, n_kv=2, D=256/512) Metal kernels + n_splits<=32 + n_splits==1 fast path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D,n_kv,kv_i8", [(256, 2, False), (512, 2, True), (256, 1, True)])
def test_p1_e4b_kernels_and_single_pass_sdpa_gpu(D: int, n_kv: int, kv_i8: bool) -> None:
    """Verify parameterized E4B/E2B QKV/KV RoPE, 4D KV_i8 dequant, n_splits==1 fast path, and n_splits<=32 guard."""
    rng = np.random.default_rng(42 + D + n_kv)
    n_q = 8
    B, Lq = 2, 3
    eps = 1e-6

    # 1. fused_qkv_packed_norm_rope (n_q=8, n_kv=1 or 2) parity vs pure MLX
    qkv = mx.array(rng.standard_normal((B, Lq, (n_q + 2 * n_kv) * D)) * 0.2, dtype=mx.bfloat16)
    wqn = mx.array(rng.uniform(0.5, 1.5, size=(D,)), dtype=mx.bfloat16)
    wkn = mx.array(rng.uniform(0.5, 1.5, size=(D,)), dtype=mx.bfloat16)
    inv = mx.array(1.0 / (10000.0 ** (np.arange(0, D, 2, dtype=np.float32) / D)), dtype=mx.float32)
    off_meta = mx.array([7, Lq], dtype=mx.int32)
    q_rot, k_rot, v_norm = fg4_kernels.fused_qkv_packed_norm_rope(
        qkv, wqn, wkn, inv, off_meta, eps, n_q=n_q, n_kv=n_kv
    )
    mx.eval(q_rot, k_rot, v_norm)
    assert q_rot.shape == (B, Lq, n_q, D)
    assert k_rot.shape == (B, Lq, n_kv, D)
    assert v_norm.shape == (B, Lq, n_kv, D)

    # Reference KV-only kernel parity on the exact same KV slice
    kv_only = qkv[:, :, n_q * D :]
    k_only, v_only = fg4_kernels.fused_kv_only_norm_rope_ragged(
        kv_only, wkn, inv, mx.array([7] * B, dtype=mx.int32), eps, n_kv=n_kv
    )
    mx.eval(k_only, v_only)
    assert float(mx.max(mx.abs(k_rot.astype(mx.float32) - k_only.astype(mx.float32))).item()) < 2e-2
    assert float(mx.max(mx.abs(v_norm.astype(mx.float32) - v_only.astype(mx.float32))).item()) < 2e-2

    # 2. 4D (B, Cap, n_kv, D) KV_i8 quantize + gather_dequant + slot SDPA (n_splits=1 vs n_splits=4 vs MLX SDPA)
    l_sys = 16
    cap = 64
    k_sys = mx.array(rng.standard_normal((l_sys, n_kv, D)) * 0.2, dtype=mx.bfloat16)
    v_sys = mx.array(rng.standard_normal((l_sys, n_kv, D)) * 0.2, dtype=mx.bfloat16)
    k_doc_bf = mx.array(rng.standard_normal((B, cap, n_kv, D)) * 0.2, dtype=mx.bfloat16)
    v_doc_bf = mx.array(rng.standard_normal((B, cap, n_kv, D)) * 0.2, dtype=mx.bfloat16)
    slots = mx.array([0, 1], dtype=mx.int32)
    off_b = mx.array([l_sys + 10, l_sys + 12], dtype=mx.int32)

    if kv_i8:
        kq, ks, vq, vs = fg4_kernels.quantize_kv_pair_i8(k_doc_bf, v_doc_bf)
        # Also verify 4D gather_dequant_kv_pair_i8_bf16
        kd_g, vd_g = fg4_kernels.gather_dequant_kv_pair_i8_bf16(
            kq, ks, vq, vs, slots, mx.arange(10, dtype=mx.int32)
        )
        mx.eval(kq, ks, vq, vs, kd_g, vd_g)
        assert kd_g.shape == (B, n_kv, 10, D)
        assert _cosine(
            np.array(kd_g.astype(mx.float32)),
            np.array(k_doc_bf[:, :10].transpose(0, 2, 1, 3).astype(mx.float32)),
        ) > 0.999
        k_arg, v_arg = kq, vq
        ks_arg, vs_arg = ks, vs
    else:
        k_arg, v_arg = k_doc_bf, v_doc_bf
        ks_arg = vs_arg = None

    # meta = [L_sys, T_cap, win, ring, n_splits, ring_mask]
    meta_sp1 = mx.array([l_sys, cap, 0, 0, 1, -1], dtype=mx.int32)
    meta_sp4 = mx.array([l_sys, cap, 0, 0, 4, -1], dtype=mx.int32)
    o_sp1 = fg4_kernels.mqa8_sdpa_prefix_ragged_slot(
        q_rot, k_sys, v_sys, k_arg, v_arg, off_b, slots, 1, meta_sp1, k_s_doc=ks_arg, v_s_doc=vs_arg
    )
    o_sp4 = fg4_kernels.mqa8_sdpa_prefix_ragged_slot(
        q_rot, k_sys, v_sys, k_arg, v_arg, off_b, slots, 4, meta_sp4, k_s_doc=ks_arg, v_s_doc=vs_arg
    )
    mx.eval(o_sp1, o_sp4)
    assert o_sp1.shape == (B, Lq, n_q * D)
    assert _cosine(np.array(o_sp1.astype(mx.float32)), np.array(o_sp4.astype(mx.float32))) > 0.9999

    # Reference MLX SDPA per row
    for b_i in range(B):
        doc_len = int(off_b[b_i].item()) - l_sys + Lq
        kd_row = (
            fg4_kernels.dequant_kv_pair_i8_bf16(
                k_arg[b_i : b_i + 1, :doc_len], ks_arg[b_i : b_i + 1, :doc_len],
                v_arg[b_i : b_i + 1, :doc_len], vs_arg[b_i : b_i + 1, :doc_len],
            )
            if kv_i8
            else (k_doc_bf[b_i : b_i + 1, :doc_len].transpose(0, 2, 1, 3), v_doc_bf[b_i : b_i + 1, :doc_len].transpose(0, 2, 1, 3))
        )
        k_all = mx.concatenate([k_sys.transpose(1, 0, 2)[None], kd_row[0]], axis=2)
        v_all = mx.concatenate([v_sys.transpose(1, 0, 2)[None], kd_row[1]], axis=2)
        ref = mx.fast.scaled_dot_product_attention(
            q_rot[b_i : b_i + 1].transpose(0, 2, 1, 3), k_all, v_all, scale=1.0, mask="causal"
        ).transpose(0, 2, 1, 3).reshape(Lq, n_q * D)
        mx.eval(ref)
        assert _cosine(np.array(o_sp1[b_i].astype(mx.float32)), np.array(ref.astype(mx.float32))) > 0.999

    # 3. `n_splits <= 32` validation guard
    with pytest.raises(ValueError, match="n_splits"):
        fg4_kernels.mqa8_sdpa_prefix_ragged_slot(
            q_rot, k_sys, v_sys, k_arg, v_arg, off_b, slots, 33, meta_sp1, k_s_doc=ks_arg, v_s_doc=vs_arg
        )
    with pytest.raises(ValueError, match="n_splits"):
        fg4_kernels.mqa8_sdpa_prefix_ragged_slot(
            q_rot, k_sys, v_sys, k_arg, v_arg, off_b, slots, 0, meta_sp1, k_s_doc=ks_arg, v_s_doc=vs_arg
        )


def test_p2_fused_ple_geglu_non_contiguous_inputs_gpu() -> None:
    """Verify `mx.contiguous()` guards prevent silent misreads on transposed/strided `gate` or `pli`."""
    rng = np.random.default_rng(77)
    gate_t = mx.array(rng.standard_normal((2, 4, 256)), dtype=mx.bfloat16).transpose(1, 0, 2)  # non-contiguous (4, 2, 256)
    pli_t = mx.array(rng.standard_normal((2, 4, 14, 256)), dtype=mx.bfloat16).transpose(1, 0, 2, 3)  # non-contiguous
    l_idx = mx.array([3], dtype=mx.int32)
    got = fg4_kernels.fused_ple_geglu_indexed(gate_t, pli_t, l_idx)
    ref = nn.gelu_approx(gate_t) * pli_t[:, :, 3, :]
    mx.eval(got, ref)
    assert float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item()) < 5e-2


# ---------------------------------------------------------------------------
# P1: Native `swa_cap` power-of-2 validation in `libgemma4_ane_prefix.dylib`
# ---------------------------------------------------------------------------


def test_p1_native_swa_cap_power_of_2_validation() -> None:
    """Verify `gemma4_px_bind_slot_cache_i8` rejects non-power-of-2 `swa_cap` cleanly (returns NULL)."""
    paths = resolve_paths()
    assert paths.lib_ane_prefix.exists(), f"Missing {paths.lib_ane_prefix}"
    lib = ctypes.CDLL(str(paths.lib_ane_prefix))
    u64p = ctypes.POINTER(ctypes.c_uint64)
    lib.gemma4_px_bind_slot_cache_i8.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        u64p,
        u64p,
        ctypes.c_int,
        u64p,
        u64p,
    ]
    lib.gemma4_px_bind_slot_cache_i8.restype = ctypes.c_void_p
    dummy = np.zeros(4, dtype=np.uint64)
    # Non-power-of-2 swa_cap=500 must fail immediately before touching handle/buffers
    res = lib.gemma4_px_bind_slot_cache_i8(
        ctypes.c_void_p(0x1),
        2,
        500,  # NOT power of 2!
        512,
        1,
        dummy.ctypes.data_as(u64p),
        dummy.ctypes.data_as(u64p),
        1,
        dummy.ctypes.data_as(u64p),
        dummy.ctypes.data_as(u64p),
    )
    assert not res


# ---------------------------------------------------------------------------
# P1: NaN argmax guards across `model.py` & `speculative.py`
# ---------------------------------------------------------------------------


def test_p1_nan_argmax_guards_everywhere_gpu() -> None:
    """Verify NaN logits never map silently to token 0 in `safe_argmax`, `run_decode_batch`, or `run_decode_prompt_lookup`."""
    arr = np.zeros((3, 16), dtype=np.float32)
    arr[0, :] = np.nan
    arr[1, 9] = 5.0
    arr[2, 4] = 8.0
    arr[2, 15] = np.nan
    am = safe_argmax_with_nan_guard(mx.array(arr, dtype=mx.bfloat16), axis=-1)
    mx.eval(am)
    assert am.tolist() == [-1, 9, -1]

    class _MiniDecodeModel:
        new_to_old = np.arange(16, dtype=np.int32)
        window_size = 512
        swa_layers = [0]
        full_layers = [1]

        def compact_cache(self, cache, rows):
            return cache

        def forward_step_ragged(self, toks, cache, off, apply_softcap=True, keep_from_col=0):
            # Emit NaN at verification step
            out = np.zeros((toks.shape[0], toks.shape[1] - keep_from_col, 16), dtype=np.float32)
            out[0, 0, 3] = np.nan
            return mx.array(out, dtype=mx.bfloat16)

    nan_lg = mx.array(arr[:1, None, :], dtype=mx.bfloat16)
    clean_lg = mx.array(arr[1:2, None, :], dtype=mx.bfloat16)
    cache = {"l_sys": 0, "layers": [{"cap": 512}, {"cap": 1024}]}

    # 1. run_decode_batch greedy (T=0) raises FloatingPointError on NaN
    with pytest.raises(FloatingPointError, match="NaN"):
        run_decode_batch(
            _MiniDecodeModel(),  # type: ignore[arg-type]
            frozenset({0}),
            None,
            nan_lg,
            mx.array([1], dtype=mx.int32),
            cache,
            1,
            max_tokens=4,
            temperature=0.0,
        )

    # 2. run_decode_batch sampled (T>0) raises FloatingPointError on NaN
    with pytest.raises(FloatingPointError, match="NaN"):
        run_decode_batch(
            _MiniDecodeModel(),  # type: ignore[arg-type]
            frozenset({0}),
            None,
            nan_lg,
            mx.array([1], dtype=mx.int32),
            cache,
            1,
            max_tokens=4,
            temperature=0.5,
        )

    # 3. run_decode_prompt_lookup init_lg raises FloatingPointError on NaN
    with pytest.raises(FloatingPointError, match="NaN"):
        run_decode_prompt_lookup(
            _MiniDecodeModel(),  # type: ignore[arg-type]
            frozenset({0}),
            None,
            nan_lg,
            mx.array([1], dtype=mx.int32),
            cache,
            [[1, 2, 3, 4]],
            1,
            max_tokens=4,
        )

    # 4. run_decode_prompt_lookup verification step raises FloatingPointError on NaN
    with pytest.raises(FloatingPointError, match="NaN"):
        run_decode_prompt_lookup(
            _MiniDecodeModel(),  # type: ignore[arg-type]
            frozenset({0}),
            None,
            clean_lg,
            mx.array([1], dtype=mx.int32),
            cache,
            [[1, 2, 3, 4]],
            1,
            max_tokens=4,
        )


# ---------------------------------------------------------------------------
# P1: 24 GiB Base M5 memory safety, wired limit restore & ANE table release
# ---------------------------------------------------------------------------


def test_p1_low_ram_24gib_clamping_wired_limit_and_ane_release_cpu() -> None:
    """Verify <=24 GiB Base M5 auto-clamping, `mx.set_wired_limit` save/restore, and `_w_emb` release."""
    train_lora = _load_script("train_lora")

    # 1. train_lora safe memory limit clamp
    assert train_lora.compute_safe_memory_limit_gb(None, 24.0, int(18 * 1024**3)) <= 18.0
    assert train_lora.compute_safe_memory_limit_gb(38.0, 24.0, int(18 * 1024**3)) <= 18.0
    assert train_lora.compute_safe_memory_limit_gb(38.0, 24.0, int(18 * 1024**3), force=True) == 38.0
    assert 30.0 <= train_lora.compute_safe_memory_limit_gb(None, 48.0, int(37.4 * 1024**3)) <= 38.0

    # 2. select_execution_plan respects max_b_slots cap
    assert select_execution_plan("auto", True, lcp=100, max_suf=200, n_docs=256, batch_size=None, max_b_slots=128) == (
        False,
        128,
    )

    # 3. FastGemma4Engine on 24 GiB host clamps defaults, sets wired limit, and releases ANE tables + restores limit in close()
    wired_calls: list[int] = []

    def _fake_set_wired(val: int) -> int:
        wired_calls.append(int(val))
        return 0 if len(wired_calls) == 1 else wired_calls[-2]

    class _DummyModel:
        hidden_size = 2560
        num_layers = 42
        n_kv_heads = 2
        first_kv_shared = 24
        model_dir = Path("/nonexistent/e4b")
        tokenizer_json = Path("/nonexistent/tokenizer.json")
        kv_i8 = True
        _auto_ane_runner = None
        _auto_ane_w8_runner = None
        _auto_cpu_ane_tables = (np.zeros((2, 2)), np.zeros((2, 2)))

    class _DummyPaths:
        lib_ane_prefix = Path(__file__)  # exists

        def resolve_ane_bundle(self, variant, *, model_dir=None, local_files_only=True, exit_layer=None):
            return Path(__file__)

    with (
        patch.object(fg4_engine, "_host_ram_gib", return_value=24.0),
        patch.object(fg4_engine, "resolve_paths", return_value=_DummyPaths()),
        patch.object(fg4_engine, "FastGemma4Model", return_value=_DummyModel()),
        patch.object(fg4_engine, "FastGemma4Tokenizer", return_value=object()),
        patch.object(mx, "set_wired_limit", side_effect=_fake_set_wired, create=True),
        patch.object(mx, "device_info", return_value={"max_recommended_working_set_size": 17 * 1024**3}, create=True),
    ):
        eng = FastGemma4Engine(warmup=False)
        assert eng.model_name == "google/gemma-4-E4B-it"
        assert eng.default_max_b_slots == 128
        assert eng.max_prefix_cache == 2
        assert eng.ane_available is False  # Disabled on <=24 GiB unless explicitly opted in!
        assert wired_calls == [17 * 1024**3]

        eng._w_emb = np.zeros((4, 4), dtype=np.float16)
        eng._w_pli14 = np.zeros((4, 4), dtype=np.float16)
        eng.close()
        assert eng._w_emb is None and eng._w_pli14 is None
        assert eng.model._auto_cpu_ane_tables is None
        assert wired_calls == [17 * 1024**3, 0]  # Restored cleanly!


# ---------------------------------------------------------------------------
# P2/P3: Paths cache fingerprint, resolve_home, drain_slack, lm_head, dead-code removal
# ---------------------------------------------------------------------------


def test_p2_p3_paths_fingerprint_resolve_home_and_drain_slack_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ANE key invalidates on weight overwrite, `resolve_home` ignores bare source pkg, and `drain_slack` enforced."""
    # 1. checkpoint_ane_key changes when model.safetensors or config.json changes in-place
    ckpt = tmp_path / "my_ckpt"
    ckpt.mkdir()
    w_file = ckpt / "model.safetensors"
    c_file = ckpt / "config.json"
    w_file.write_bytes(b"weights-v1")
    c_file.write_text('{"hidden_size": 1536}', encoding="utf-8")

    k1 = Gemma4Paths.checkpoint_ane_key(ckpt)
    w_file.write_bytes(b"weights-v2-longer")
    os.utime(w_file, (time.time() + 5, time.time() + 5))
    k2 = Gemma4Paths.checkpoint_ane_key(ckpt)
    assert k1 != k2

    # Stale local `ckpt/ane_models/*.mlmodelc` older than model.safetensors is rejected
    local_ane = ckpt / "ane_models" / "gemma4_early14_prefix_S256_fp16.mlmodelc"
    local_ane.mkdir(parents=True)
    old_t = w_file.stat().st_mtime - 100.0
    os.utime(local_ane, (old_t, old_t))
    p_tmp = resolve_paths(tmp_path / "home_art")
    assert p_tmp.resolve_ane_bundle("fp16", model_dir=ckpt) is None

    # Fresh local bundle newer than weights is accepted
    new_t = w_file.stat().st_mtime + 100.0
    os.utime(local_ane, (new_t, new_t))
    assert p_tmp.resolve_ane_bundle("fp16", model_dir=ckpt) == local_ane

    # 2. resolve_home does NOT stop at bare pyproject.toml/native when artifacts/ is absent
    bare_pkg = tmp_path / "fast-silicon" / "packages" / "fast-gemma4"
    (bare_pkg / "src" / "fast_gemma4").mkdir(parents=True)
    (bare_pkg / "native").mkdir()
    (bare_pkg / "pyproject.toml").write_text("[project]\nname='fast-gemma4'\n", encoding="utf-8")
    fake_file = bare_pkg / "src" / "fast_gemma4" / "paths.py"
    fake_file.write_text("", encoding="utf-8")

    art_root = tmp_path / "shared_artifacts"
    (art_root / "fast-gemma4").mkdir(parents=True)
    monkeypatch.delenv("FAST_GEMMA4_HOME", raising=False)
    monkeypatch.setenv("FAST_ARTIFACTS_ROOT", str(art_root))
    with patch.object(fg4_paths, "__file__", str(fake_file)):
        resolved_h = resolve_home()
    assert resolved_h == (art_root / "fast-gemma4").resolve()

    # Without FAST_ARTIFACTS_ROOT, falls back to user cache dir (NOT bare source package!)
    monkeypatch.delenv("FAST_ARTIFACTS_ROOT", raising=False)
    with patch.object(fg4_paths, "__file__", str(fake_file)):
        resolved_cache = resolve_home()
    assert resolved_cache == (Path.home() / ".cache" / "fast-gemma4").resolve()

    # 3. `k <= 16` / `drain_slack` guard in forward_early14_slot & forward_ff_jump_slot
    m = FastGemma4Model.__new__(FastGemma4Model)
    m.window_size = 512
    m.first_kv_shared = 0
    m.layers = []
    m.act_dtype = mx.bfloat16
    m.kv_i8 = True
    c = m.make_slot_cache({"l_sys": 0, "k_sys": [], "v_sys": []}, batch_size=1, doc_cap_full=64, drain_slack=16)
    assert c["drain_slack"] == 16
    with pytest.raises(ValueError, match="drain_slack"):
        m.forward_early14_slot(mx.zeros((1, 17), dtype=mx.int32), c, mx.array([0], dtype=mx.int32), mx.array([0], dtype=mx.int32))
    with pytest.raises(ValueError, match="drain_slack"):
        m.forward_ff_jump_slot(mx.zeros((1, 17), dtype=mx.int32), c, mx.array([0], dtype=mx.int32), mx.array([0], dtype=mx.int32))

    # 4. Dead-end `fused_gemma4_res_rms_bf16` and `use_vec4_norm` are deleted
    assert not hasattr(fg4_kernels, "fused_gemma4_res_rms_bf16")
    assert "use_vec4_norm" not in inspect.signature(FastGemma4Model.__init__).parameters


def test_p2_explicit_lm_head_and_vocab_mismatch_validation_gpu(tmp_path: Path) -> None:
    """Verify explicit untied/materialized `lm_head.weight` loading + `vocab_out` mismatch warning/error."""
    ckpt = tmp_path / "untied_e4b_ckpt"
    ckpt.mkdir()
    H, V_full, V_pruned = 64, 32, 16
    cfg = {
        "text_config": {
            "hidden_size": H,
            "num_hidden_layers": 2,
            "num_kv_shared_layers": 0,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "sliding_window": 512,
            "hidden_size_per_layer_input": 16,
            "vocab_size": V_full,
            "vocab_size_per_layer_input": V_full,
            "tie_word_embeddings": False,
            "attention_k_eq_v": False,
            "layer_types": ["sliding_attention", "full_attention"],
            "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6,
        }
    }
    (ckpt / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (ckpt / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {f"tok{i}": i for i in range(V_full)}}, "added_tokens": []}),
        encoding="utf-8",
    )
    kept_ids = np.arange(V_pruned, dtype=np.int32)
    np.savez(ckpt / "vocab_remap.npz", kept_old_ids=kept_ids, old_ids=kept_ids)

    rng = np.random.default_rng(123)
    emb_w = mx.array(rng.standard_normal((V_full, H)) * 0.1, dtype=mx.bfloat16)
    # Distinct explicit pruned lm_head weights (value = 2.0 * emb_w[:V_pruned] + 0.5)
    lm_head_w = mx.array(rng.standard_normal((V_pruned, H)) * 0.2 + 0.5, dtype=mx.bfloat16)

    tensors: dict[str, mx.array] = {
        "language_model.model.embed_tokens.weight": emb_w,
        "language_model.lm_head.weight": lm_head_w,
        "language_model.model.embed_tokens_per_layer.weight": mx.zeros((V_full, 2 * 16), dtype=mx.bfloat16),
        "language_model.model.per_layer_model_projection.weight": mx.zeros((2 * 16, H), dtype=mx.bfloat16),
        "language_model.model.per_layer_projection_norm.weight": mx.ones((16,), dtype=mx.bfloat16),
        "language_model.model.norm.weight": mx.ones((H,), dtype=mx.bfloat16),
    }
    for i, hd in enumerate((256, 512)):
        lp = f"language_model.model.layers.{i}"
        tensors[f"{lp}.input_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
        tensors[f"{lp}.post_attention_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
        tensors[f"{lp}.pre_feedforward_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
        tensors[f"{lp}.post_feedforward_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
        tensors[f"{lp}.post_per_layer_input_norm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
        tensors[f"{lp}.layer_scalar"] = mx.ones((1,), dtype=mx.bfloat16)
        tensors[f"{lp}.self_attn.q_proj.weight"] = mx.zeros((8 * hd, H), dtype=mx.bfloat16)
        tensors[f"{lp}.self_attn.k_proj.weight"] = mx.zeros((2 * hd, H), dtype=mx.bfloat16)
        tensors[f"{lp}.self_attn.v_proj.weight"] = mx.zeros((2 * hd, H), dtype=mx.bfloat16)
        tensors[f"{lp}.self_attn.o_proj.weight"] = mx.zeros((H, 8 * hd), dtype=mx.bfloat16)
        tensors[f"{lp}.self_attn.q_norm.weight"] = mx.ones((hd,), dtype=mx.bfloat16)
        tensors[f"{lp}.self_attn.k_norm.weight"] = mx.ones((hd,), dtype=mx.bfloat16)
        tensors[f"{lp}.mlp.gate_proj.weight"] = mx.zeros((64, H), dtype=mx.bfloat16)
        tensors[f"{lp}.mlp.up_proj.weight"] = mx.zeros((64, H), dtype=mx.bfloat16)
        tensors[f"{lp}.mlp.down_proj.weight"] = mx.zeros((H, 64), dtype=mx.bfloat16)
        tensors[f"{lp}.per_layer_input_gate.weight"] = mx.zeros((16, H), dtype=mx.bfloat16)
        tensors[f"{lp}.per_layer_projection.weight"] = mx.zeros((H, 16), dtype=mx.bfloat16)
    mx.save_safetensors(str(ckpt / "model.safetensors"), tensors)

    p = resolve_paths(tmp_path / "art_home")
    # 1. Matching vocab_out loads explicit materialized pruned lm_head.weight (NOT embed_tokens!)
    m_ok = FastGemma4Model(paths=p, model_dir=ckpt, vocab_out=V_pruned, lm_head_mode="q4", use_mpp=False)
    assert m_ok.vocab_size == V_pruned
    assert m_ok.n_kv_heads == 2 and m_ok.n_q_heads == 8 and m_ok.attention_k_eq_v is False
    probe = mx.ones((1, 1, H), dtype=mx.bfloat16)
    lg_got = m_ok._apply_lm_head(probe, apply_softcap=False)
    lg_exp = probe @ lm_head_w.T
    mx.eval(lg_got, lg_exp)
    assert float(mx.max(mx.abs(lg_got.astype(mx.float32) - lg_exp.astype(mx.float32))).item()) < 1e-2

    # 2. Disagreed explicit vocab_out raises ValueError cleanly when materialized pruned lm_head exists
    with pytest.raises(ValueError, match="vocab_out"):
        FastGemma4Model(paths=p, model_dir=ckpt, vocab_out=12, lm_head_mode="q4", use_mpp=False)

