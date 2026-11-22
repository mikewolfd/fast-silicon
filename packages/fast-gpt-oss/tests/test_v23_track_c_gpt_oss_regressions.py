"""Regression tests for Track C precision and CoT budget validation in `fast-gpt-oss`."""

from __future__ import annotations

import contextlib
import fcntl
import os
from pathlib import Path
import subprocess
from typing import Iterator

import numpy as np
import pytest

from fast_gpt_oss.engine import (
    _EFFORT_DEFAULT_BUDGET,
    _REPEAT_4GRAM_MIN_HITS,
    _REPEAT_4GRAM_PENALTY,
    _REPEAT_4GRAM_WINDOW,
    _find_repeated_4gram_penalties,
    resolve_reasoning_params,
)
from fast_gpt_oss.kernels import KernelSpec, build_kernels, render_kernel_sources

HW_LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")


def _ancestor_holds_lock(lock_path: Path) -> bool:
    try:
        anc: set[int] = set()
        pid = os.getpid()
        while pid > 1 and pid not in anc:
            anc.add(pid)
            ppid_s = subprocess.check_output(["ps", "-o", "ppid=", "-p", str(pid)], text=True).strip()
            pid = int(ppid_s) if ppid_s else 0
        holders = {
            int(x)
            for x in subprocess.check_output(["lsof", "-t", str(lock_path)], text=True).split()
            if x.strip().isdigit()
        }
        return bool((anc - {os.getpid()}) & holders)
    except Exception:
        return False


@contextlib.contextmanager
def _gpu_lock() -> Iterator[None]:
    HW_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(HW_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o666)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            if not _ancestor_holds_lock(HW_LOCK_PATH):
                fcntl.flock(fd, fcntl.LOCK_EX)
                acquired = True
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return 0.0
    return float(np.dot(af, bf) / denom)


def test_gpt_oss_hybrid_kernel_precision_contract_cpu() -> None:
    """Verify hybrid precision invariants (`f16` 10-bit intermediates + `f32` accumulators) stay intact."""
    sources = render_kernel_sources(KernelSpec())

    rms = sources["rmsnorm_bf16_to_f16"]["source"]
    assert "#pragma clang fp reassociate(off)" in rms
    assert "float ss = 0.0f;" in rms
    assert "half(float(xp[i]) * inv * float(W[i]))" in rms
    assert "bfloat(float(xp[i])" not in rms

    rms_f32 = sources["rmsnorm_bf16_to_f16_f32"]
    assert rms_f32["output_names"] == ["OutF16", "OutF32"]
    assert "device half* op16 = OutF16" in rms_f32["source"]
    assert "device float* op32 = OutF32" in rms_f32["source"]

    rope = sources["yarn_rope_split_b"]["source"]
    assert "s = sincos(theta, c);" in rope
    assert "const device half* base_qkv" in rope

    gu = sources["moe_batch_gu_unified"]
    dn = sources["moe_batch_dn_unified"]
    assert gu["output_names"] == ["MidOut"]
    assert "threadgroup half TG[" in gu["source"]
    assert "MidOut[" in gu["source"] and "half((" in gu["source"]
    assert "e8m0_to_float" in gu["header"]
    assert "bfloat" not in gu["source"]
    assert "const device uint4*)(MidOut" in dn["source"]
    assert "threadgroup half TG[" in dn["source"]
    assert "bfloat" not in dn["source"]

    for attn_key in ("hydragen_p1_mma", "hydragen_p2_varlen", "swa128_decode", "swa128_prefill_mma"):
        attn_src = sources[attn_key]["source"]
        assert "threadgroup float" in attn_src
        if "Sinks" in sources[attn_key]["input_names"]:
            assert "Sinks[" in attn_src


def test_gpt_oss_cot_budget_and_loop_breaker_defaults_cpu() -> None:
    """Verify validated CoT effort budgets and conservative 64-token 4-gram loop breaker."""
    assert _EFFORT_DEFAULT_BUDGET == {
        "skip": 0,
        "none": 0,
        "low": 48,
        "medium": 80,
        "high": 128,
    }
    assert _REPEAT_4GRAM_WINDOW == 64
    assert _REPEAT_4GRAM_MIN_HITS == 2
    assert _REPEAT_4GRAM_PENALTY == pytest.approx(5.0)

    # Constrained JSON/schema contracts: disable CoT by default unless reasoning is explicitly enabled.
    assert resolve_reasoning_params(
        has_schema=True,
        reasoning_effort=None,
        reasoning_budget=None,
        enable_thinking=False,
    ) == ("skip", 0)
    assert resolve_reasoning_params(
        has_schema=True,
        reasoning_effort=None,
        reasoning_budget=None,
        enable_thinking=True,
    ) == ("low", 48)
    assert resolve_reasoning_params(
        has_schema=True,
        reasoning_effort="low",
        reasoning_budget=64,
        enable_thinking=False,
    ) == ("low", 64)
    # Opt-in medium effort preserves its larger analysis budget (e.g. LiquidAI extraction lift).
    assert resolve_reasoning_params(
        has_schema=True,
        reasoning_effort="medium",
        reasoning_budget=None,
        enable_thinking=True,
    ) == ("medium", 80)

    # Open-ended contracts: `enable_thinking=True` maps skip/none -> "low", keeping unbounded (0) unless budgeted.
    assert resolve_reasoning_params(
        has_schema=False,
        reasoning_effort=None,
        reasoning_budget=None,
        enable_thinking=False,
    ) == ("low", 0)
    assert resolve_reasoning_params(
        has_schema=False,
        reasoning_effort="skip",
        reasoning_budget=None,
        enable_thinking=True,
    ) == ("low", 0)
    assert resolve_reasoning_params(
        has_schema=False,
        reasoning_effort="low",
        reasoning_budget=96,
        enable_thinking=True,
    ) == ("low", 96)

    with pytest.raises(ValueError, match="reasoning_effort"):
        resolve_reasoning_params(
            has_schema=False,
            reasoning_effort="invalid_tier",
            reasoning_budget=None,
            enable_thinking=True,
        )

    # Spaced 4-gram copies outside window=64 must NOT trigger false suppression.
    spaced_seq = [10, 20, 30, 40] + list(range(1000, 1065)) + [10, 20, 30, 40, 50, 10, 20, 30]
    assert _find_repeated_4gram_penalties(spaced_seq) == {}

    # Single prior hit inside window=64 stays unpenalized; >=2 prior hits triggers scaled suppression.
    one_hit = [11, 12, 13, 14, 99, 11, 12, 13]
    assert _find_repeated_4gram_penalties(one_hit) == {}

    two_hits = [11, 12, 13, 14, 98, 11, 12, 13, 14, 99, 11, 12, 13]
    assert _find_repeated_4gram_penalties(two_hits) == pytest.approx({14: 5.0})
    three_hits = [11, 12, 13, 14, 97] + two_hits
    assert _find_repeated_4gram_penalties(three_hits) == pytest.approx({14: 10.0})


def test_gpt_oss_f16_intermediates_beat_bf16_vs_fp32_ref_gpu() -> None:
    """Verify production `float16` RMSNorm + `MidOut` beat `bfloat16` intermediates vs FP32 reference."""
    import mlx.core as mx

    h_dim = 256
    d_ff = 256
    n_experts = 8
    top_k = 4
    n_tok = 8
    spec = KernelSpec(
        hidden_size=h_dim,
        intermediate_size=d_ff,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=64,
        num_local_experts=n_experts,
        sliding_window=16,
        moe_hot_threshold=4,
    )

    rng = np.random.default_rng(2026)
    h_np = (rng.standard_normal((n_tok, h_dim)) * np.linspace(0.5, 4.0, n_tok)[:, None]).astype(np.float32)
    # Include mild coordinate outliers akin to RMSNorm/residual heavy-tailed activations.
    h_np[:, 0] *= 3.5
    ln_w_np = (1.0 + 0.15 * rng.standard_normal((h_dim,))).astype(np.float32)

    with _gpu_lock():
        kernels = build_kernels(spec)

        h_bf16 = mx.array(h_np, dtype=mx.bfloat16)
        ln_w_bf16 = mx.array(ln_w_np, dtype=mx.bfloat16)
        eps_arr = mx.array([1e-5], dtype=mx.float32)  # guard: allow-fp32

        gate_f = mx.array((rng.standard_normal((n_experts, d_ff, h_dim)) * 0.08).astype(np.float16))
        up_f = mx.array((rng.standard_normal((n_experts, d_ff, h_dim)) * 0.08).astype(np.float16))
        dn_f = mx.array((rng.standard_normal((n_experts, h_dim, d_ff)) * 0.08).astype(np.float16))
        gw, gs = mx.quantize(gate_f, group_size=32, bits=4, mode="mxfp4")
        uw, us = mx.quantize(up_f, group_size=32, bits=4, mode="mxfp4")
        dn_w, dn_s = mx.quantize(dn_f, group_size=32, bits=4, mode="mxfp4")

        gu_w = mx.concatenate([gw, uw], axis=1)
        gu_s = mx.concatenate([gs, us], axis=1)
        gu_bias = mx.array((rng.standard_normal((n_experts, 2 * d_ff)) * 0.02).astype(np.float16))
        dn_bias = mx.array((rng.standard_normal((n_experts, h_dim)) * 0.02).astype(np.float16))

        # Expert 0 receives all 8 tokens (exercises hot path), experts 1..3 receive fewer (cold path).
        inds_np = np.array([[0, (t % 3) + 1, (t % 3) + 4, (t + 2) % 6 + 2] for t in range(n_tok)], dtype=np.uint32)
        for t in range(n_tok):
            row = inds_np[t]
            if len(set(row.tolist())) < top_k:
                inds_np[t] = np.array([0, 1 + (t % 2), 3 + (t % 2), 5 + (t % 3)], dtype=np.uint32)
        inds = mx.array(inds_np, dtype=mx.uint32)
        raw_scores = mx.array(rng.uniform(0.1, 1.0, (n_tok, top_k)).astype(np.float32))  # guard: allow-fp32
        scores = raw_scores / mx.sum(raw_scores, axis=-1, keepdims=True)

        # 1. Production Metal path: float16 RMSNorm + float16 MidOut + float32 accumulation.
        rms_f16 = kernels["rmsnorm_bf16_to_f16"](
            inputs=[h_bf16, ln_w_bf16, eps_arr],
            grid=(n_tok * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_tok, h_dim)],
            output_dtypes=[mx.float16],
        )[0]

        n_assign = n_tok * top_k
        max_hot = max(1, (n_assign // 5) + 2)
        tok_idx, slot_idx, rweights, hot_meta, cold_meta, meta_counts = kernels["moe_build_tiles"](  # guard: allow-fp32
            inputs=[inds, scores, mx.array([n_tok], dtype=mx.uint32)],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(n_assign,), (n_assign,), (n_assign,), (max_hot * 3,), (32 * 3,), (2,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.uint32, mx.uint32, mx.uint32],
        )
        mid_f16 = kernels["moe_batch_gu_unified"](
            inputs=[rms_f16, tok_idx, gu_w, gu_s, gu_bias, hot_meta, cold_meta, meta_counts],
            grid=((d_ff // 8) * 128, max_hot + n_experts, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_assign, d_ff)],
            output_dtypes=[mx.float16],
        )[0]
        ys_f32 = kernels["moe_batch_dn_unified"](  # guard: allow-fp32
            inputs=[mid_f16, slot_idx, rweights, dn_w, dn_s, dn_bias, hot_meta, cold_meta, meta_counts],
            grid=((h_dim // 16) * 128, max_hot + n_experts, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_tok, top_k, h_dim)],
            output_dtypes=[mx.float32],
        )[0]
        out_f16_path = mx.sum(ys_f32, axis=1)

        # 2. Exact FP32 reference using dequantized MXFP4 weights (ground truth).
        h32 = h_bf16.astype(mx.float32)  # guard: allow-fp32
        w32 = ln_w_bf16.astype(mx.float32)  # guard: allow-fp32
        rms_f32 = h32 * mx.rsqrt(mx.mean(h32 * h32, axis=-1, keepdims=True) + eps_arr) * w32

        gu_deq32 = mx.dequantize(gu_w, gu_s, group_size=32, bits=4, mode="mxfp4", dtype=mx.float32)  # guard: allow-fp32
        dn_deq32 = mx.dequantize(dn_w, dn_s, group_size=32, bits=4, mode="mxfp4", dtype=mx.float32)  # guard: allow-fp32
        gu_bias32 = gu_bias.astype(mx.float32)  # guard: allow-fp32
        dn_bias32 = dn_bias.astype(mx.float32)  # guard: allow-fp32

        flat_e = inds.reshape(-1)
        gu_w_sel = gu_deq32[flat_e].reshape(n_tok, top_k, 2 * d_ff, h_dim)
        gu_out32 = mx.einsum("th,tkmh->tkm", rms_f32, gu_w_sel) + gu_bias32[flat_e].reshape(n_tok, top_k, 2 * d_ff)
        g32, u32 = mx.split(gu_out32, 2, axis=-1)
        g32 = mx.minimum(g32, spec.swiglu_limit)
        u32 = mx.clip(u32, -spec.swiglu_limit, spec.swiglu_limit)
        mid_f32_ref = (u32 + 1.0) * (g32 * mx.sigmoid(g32 * spec.swiglu_alpha))
        dn_w_sel = dn_deq32[flat_e].reshape(n_tok, top_k, h_dim, d_ff)
        dn_out32 = mx.einsum("tkd,tkhd->tkh", mid_f32_ref, dn_w_sel) + dn_bias32[flat_e].reshape(n_tok, top_k, h_dim)
        out_fp32_ref = mx.sum(dn_out32 * scores[..., None], axis=1)

        # 3. Simulated `bfloat16` intermediate path (demonstrating 3 lost mantissa bits vs `float16`).
        rms_bf16 = rms_f32.astype(mx.bfloat16)
        ord_i = mx.argsort(flat_e)
        inv_i = mx.argsort(ord_i)
        sorted_e = flat_e[ord_i].astype(mx.uint32)
        tok_idx_bf16 = ord_i // top_k
        flat_x_bf16 = rms_bf16[tok_idx_bf16].reshape(n_assign, 1, h_dim)
        gu_bf16 = mx.gather_qmm(  # guard: allow-lhs-indices
            flat_x_bf16,
            gu_w,
            gu_s,
            rhs_indices=sorted_e,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
            sorted_indices=True,
        ).reshape(n_assign, 2 * d_ff) + gu_bias[sorted_e].astype(mx.bfloat16)
        g_b, u_b = mx.split(gu_bf16.astype(mx.float32), 2, axis=-1)  # guard: allow-fp32
        g_b = mx.minimum(g_b, spec.swiglu_limit)
        u_b = mx.clip(u_b, -spec.swiglu_limit, spec.swiglu_limit)
        mid_bf16 = ((u_b + 1.0) * (g_b * mx.sigmoid(g_b * spec.swiglu_alpha))).astype(mx.bfloat16)
        dn_bf16 = mx.gather_qmm(  # guard: allow-lhs-indices
            mid_bf16.reshape(n_assign, 1, d_ff),
            dn_w,
            dn_s,
            rhs_indices=sorted_e,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
            sorted_indices=True,
        )
        dn_bf16_unsort = (dn_bf16.reshape(n_assign, h_dim) + dn_bias[sorted_e].astype(mx.bfloat16))[inv_i].reshape(
            n_tok, top_k, h_dim
        )
        out_bf16_path = mx.sum(dn_bf16_unsort.astype(mx.float32) * scores[..., None], axis=1)  # guard: allow-fp32

        mx.eval(rms_f16, rms_f32, rms_bf16, out_f16_path, out_fp32_ref, out_bf16_path, meta_counts)

        rms_ref_np = np.asarray(rms_f32)
        rms_f16_err = float(np.max(np.abs(np.asarray(rms_f16.astype(mx.float32)) - rms_ref_np)))  # guard: allow-fp32
        rms_bf16_err = float(np.max(np.abs(np.asarray(rms_bf16.astype(mx.float32)) - rms_ref_np)))  # guard: allow-fp32
        assert rms_f16_err < rms_bf16_err

        ref_np = np.asarray(out_fp32_ref)
        f16_np = np.asarray(out_f16_path)
        bf16_np = np.asarray(out_bf16_path)

        cos_f16 = _cosine_similarity(f16_np, ref_np)
        cos_bf16 = _cosine_similarity(bf16_np, ref_np)
        err_f16 = float(np.max(np.abs(f16_np - ref_np)))
        err_bf16 = float(np.max(np.abs(bf16_np - ref_np)))

        assert cos_f16 >= 0.999
        assert cos_f16 > cos_bf16
        assert err_f16 < err_bf16
