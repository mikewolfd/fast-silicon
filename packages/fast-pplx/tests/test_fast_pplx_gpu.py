"""Real-weight GPU parity & throughput tests against reference.npz (serialized lock)."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Iterator

import mlx.core as mx
import numpy as np
import pytest

from fast_pplx.config import DEFAULT_MODEL_NAME
from fast_pplx.gpu import FastPplxGPU, _find_local_hf_snapshot
from fast_pplx.paths import resolve_paths

_LOCK_PATH = "/tmp/m4_gpu_validation.lock"
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@contextmanager
def _gpu_validation_lock() -> Iterator[None]:
    if os.environ.get("M4_GPU_LOCK_HELD") == "1":
        yield
        return
    fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _require_cached_weights() -> Path | None:
    paths = resolve_paths()
    if paths.weights_npz.exists() or (paths.model_dir / "config.json").exists():
        return None
    snap = _find_local_hf_snapshot(DEFAULT_MODEL_NAME)
    if snap is not None:
        return snap
    pytest.skip("No local pplx-embed-v2-context-0.6b weights or HF snapshot cached")
    return None


@pytest.mark.gpu_ane
def test_gpu_trace_and_all_16_reference_cases() -> None:
    snap = _require_cached_weights()
    cases = json.loads((_FIXTURES_DIR / "cases.json").read_text(encoding="utf-8"))
    ref = np.load(_FIXTURES_DIR / "reference.npz")

    with _gpu_validation_lock():
        with FastPplxGPU(model_dir=snap, dtype="float16", use_mapped_embedding=True, local_files_only=True) as gpu:
            # 1. Layer trace parity on contextual_three_chunks
            ids = ref["contextual_three_chunks/input_ids"]
            final_h, trace = gpu.forward_hidden(ids, return_trace=True)
            mx.eval([final_h, *trace])
            for idx in range(12):
                got = np.array(trace[idx], dtype=np.float32)
                exp = ref[f"trace/{idx}"]
                cos = float(np.sum(got * exp) / (np.linalg.norm(got) * np.linalg.norm(exp)))
                assert cos >= 0.9999, f"trace/{idx} cos={cos}"
            final_np = np.array(final_h, dtype=np.float32)
            exp_final = ref["trace/12"]
            final_cos = float(np.sum(final_np * exp_final) / (np.linalg.norm(final_np) * np.linalg.norm(exp_final)))
            assert final_cos >= 0.99995, f"final trace/12 cos={final_cos}"

            # 2. All 16 reference cases (float32 projection + int8 encoded outputs)
            for case in cases:
                cid = case["id"]
                if case["task"] == "query":
                    flat_q = [row[0] for row in case["documents"]]
                    proj = gpu.encode_queries(flat_q, output_dtype="float32", normalize=False, apply_tanh=False)
                    i8 = gpu.encode_queries(flat_q, output_dtype="int8", normalize=False)
                    exp_proj = ref[f"{cid}/projected"].reshape(-1, 1024)
                    f_cos = np.sum(proj * exp_proj, axis=-1) / (
                        np.linalg.norm(proj, axis=-1) * np.linalg.norm(exp_proj, axis=-1)
                    )
                    assert float(f_cos.min()) >= 0.9999, f"{cid} f32 cos={f_cos}"
                    exp_i8 = np.concatenate([ref[f"{cid}/encoded/{k}"] for k in range(len(flat_q))], axis=0)
                    i8_cos = np.sum(i8.astype(np.float32) * exp_i8.astype(np.float32), axis=-1) / (
                        np.linalg.norm(i8.astype(np.float32), axis=-1)
                        * np.linalg.norm(exp_i8.astype(np.float32), axis=-1)
                    )
                    assert float(i8_cos.min()) >= 0.9997, f"{cid} i8 cos={i8_cos}"
                    assert int(np.max(np.abs(i8.astype(np.int32) - exp_i8.astype(np.int32)))) <= 1
                else:
                    docs = case["documents"]
                    proj_list = gpu.encode_contextual_chunks(docs, output_dtype="float32", normalize=False, apply_tanh=False)
                    i8_list = gpu.encode_contextual_chunks(docs, output_dtype="int8", normalize=False)
                    exp_proj_all = ref[f"{cid}/projected"]
                    for di, doc in enumerate(docs):
                        if not doc:
                            assert proj_list[di].shape == (0, 1024)
                            assert i8_list[di].shape == (0, 1024)
                            continue
                        rp = exp_proj_all[di, : len(doc)]
                        ri = ref[f"{cid}/encoded/{di}"]
                        gp = proj_list[di]
                        gi = i8_list[di]
                        for ci, chunk_str in enumerate(doc):
                            if not chunk_str:
                                assert np.all(gi[ci] == 0)
                                continue
                            f_cos = float(np.dot(gp[ci], rp[ci]) / (np.linalg.norm(gp[ci]) * np.linalg.norm(rp[ci])))
                            i8_cos = float(
                                np.dot(gi[ci].astype(np.float32), ri[ci].astype(np.float32))
                                / (
                                    np.linalg.norm(gi[ci].astype(np.float32))
                                    * np.linalg.norm(ri[ci].astype(np.float32))
                                )
                            )
                            abs_d = int(np.max(np.abs(gi[ci].astype(np.int32) - ri[ci].astype(np.int32))))
                            assert f_cos >= 0.9999, f"{cid}[{di}][{ci}] f32 cos={f_cos}"
                            assert i8_cos >= 0.9997, f"{cid}[{di}][{ci}] i8 cos={i8_cos}"
                            assert abs_d <= 1, f"{cid}[{di}][{ci}] max_abs_diff={abs_d}"
                    if cid == "non_contextual":
                        nd = gpu.encode_documents([d[0] for d in docs], output_dtype="float32", normalize=True)
                        exp_nd = np.concatenate([ref[f"{cid}/encoded/{di}"] for di in range(len(docs))], axis=0).astype(
                            np.float32
                        )
                        exp_nd /= np.maximum(np.linalg.norm(exp_nd, axis=-1, keepdims=True), 1e-12)
                        nd_cos = np.sum(nd * exp_nd, axis=-1)
                        assert float(nd_cos.min()) >= 0.9998, f"encode_documents upstream cos={nd_cos}"


@pytest.mark.gpu_ane
def test_gpu_normalization_matryoshka_and_throughput() -> None:
    snap = _require_cached_weights()
    with _gpu_validation_lock():
        with FastPplxGPU(model_dir=snap, dtype="float16", use_mapped_embedding=True, local_files_only=True) as gpu:
            # Verify normalized unit L2 length across full 1024 and truncated 256 Matryoshka dims
            for dim in (None, 256):
                vecs = gpu.encode_queries(["How do ocean currents transfer heat?"], dimensions=dim, normalize=True)
                norms = np.linalg.norm(vecs, axis=-1)
                assert vecs.shape == (1, 1024 if dim is None else dim)
                assert np.allclose(norms, 1.0, atol=1e-4)

            # Throughput check at B=32, S=128
            ids = np.full((32, 128), 1000, dtype=np.int32)
            ids[:, 0] = 248077
            mask = np.ones((32, 128), dtype=np.bool_)
            for _ in range(2):
                _ = gpu._project_mean_pool(gpu.forward_hidden(ids), mask)
            t0 = time.perf_counter()
            for _ in range(5):
                out = gpu._project_mean_pool(gpu.forward_hidden(ids), mask)
            dt = time.perf_counter() - t0
            tok_s = (32 * 128 * 5) / dt
            assert np.all(np.isfinite(out))
            assert tok_s >= 7000.0, f"Unexpected GPU throughput regression: {tok_s:.1f} tok/s"


@pytest.mark.gpu_ane
@pytest.mark.parametrize("act_dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("use_mask", [False, True])
def test_gpu_gdn_s4096_streamed_fp32_state_parity(act_dtype: mx.Dtype, use_mask: bool) -> None:
    """Verify S=4096 chunked GDN (1024-tok slices) keeps FP32 rec_state across BOTH Metal kernels."""
    from fast_pplx.gpu import gated_delta_update, normalize_qk

    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")

    with _gpu_validation_lock():
        mx.random.seed(2026)
        B, S, Hk, Hv, Dk, Dv = 1, 4096, 4, 4, 128, 128
        q = mx.random.normal((B, S, Hk, Dk)).astype(act_dtype)
        k = mx.random.normal((B, S, Hk, Dk)).astype(act_dtype)
        v = (mx.random.normal((B, S, Hv, Dv)) * 0.25).astype(act_dtype)
        a = (mx.random.normal((B, S, Hv)) * 0.2).astype(act_dtype)
        b = (mx.random.normal((B, S, Hv)) * 0.2).astype(act_dtype)
        A_log = mx.full((Hv,), -2.0, dtype=mx.float32)
        dt_bias = mx.full((Hv,), -1.0, dtype=mx.float32)

        q, k = normalize_qk(q, k, inv_scale=float(Dk) ** -0.5, eps=1e-6)
        mask_arr = None
        if use_mask:
            m_np = np.ones((B, S), dtype=np.bool_)
            m_np[:, 3700:] = False
            mask_arr = mx.array(m_np, dtype=mx.bool_)

        # 1. Single-pass whole S=4096 execution (exercises packed kernel when unmasked, masked kernel when masked)
        y_whole, st_whole = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=mask_arr, use_kernel=True)
        mx.eval(y_whole, st_whole)
        assert st_whole.dtype == mx.float32

        # 2. Streamed 4x1024 chunk handoff with FP32 rec_state
        st = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        chunks = []
        for s0 in range(0, S, 1024):
            s1 = s0 + 1024
            m_sub = None if mask_arr is None else mask_arr[:, s0:s1]
            y_sub, st = gated_delta_update(
                q[:, s0:s1], k[:, s0:s1], v[:, s0:s1], a[:, s0:s1], b[:, s0:s1],
                A_log, dt_bias, state=st, mask=m_sub, use_kernel=True,
            )
            mx.async_eval(y_sub, st)
            assert st.dtype == mx.float32
            chunks.append(y_sub)
        y_stream = mx.concatenate(chunks, axis=1)
        mx.eval(y_stream, st)

        yw_np = np.array(y_whole.astype(mx.float32), dtype=np.float32)
        ys_np = np.array(y_stream.astype(mx.float32), dtype=np.float32)
        assert np.all(np.isfinite(ys_np))
        if use_mask:
            # Verify padded tail is cleanly zeroed by single-lane masked branch
            assert np.all(ys_np[:, 3700:] == 0.0)
        yw64 = yw_np.astype(np.float64).ravel()
        ys64 = ys_np.astype(np.float64).ravel()
        cos = float(np.dot(yw64, ys64) / max(float(np.linalg.norm(yw64) * np.linalg.norm(ys64)), 1e-12))
        max_abs = float(np.max(np.abs(yw64 - ys64)))
        assert cos >= 0.999999, f"S=4096 GDN chunked mismatch ({act_dtype}, mask={use_mask}): cos={cos}, max_abs={max_abs}"
        assert max_abs <= 1e-4, f"S=4096 GDN chunked drift ({act_dtype}, mask={use_mask}): max_abs={max_abs}"
        assert np.allclose(
            np.array(st_whole.astype(mx.float32), dtype=np.float32),
            np.array(st.astype(mx.float32), dtype=np.float32),
            atol=1e-5,
        )


@pytest.mark.gpu_ane
@pytest.mark.parametrize("dk", [64, 128])
def test_gpu_gdn_packed_kernel_generalized_dk(dk: int) -> None:
    """Verify generalized partials_per_lane tree reduction works across Dk=64 and Dk=128."""
    from fast_pplx.gpu import gated_delta_update, normalize_qk

    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")

    with _gpu_validation_lock():
        mx.random.seed(77 + dk)
        B, T, Hk, Hv, Dv = 1, 64, 2, 2, 32
        q = mx.random.normal((B, T, Hk, dk)).astype(mx.float16)
        k = mx.random.normal((B, T, Hk, dk)).astype(mx.float16)
        v = (mx.random.normal((B, T, Hv, Dv)) * 0.25).astype(mx.float16)
        a = mx.zeros((B, T, Hv), dtype=mx.float16)
        b = mx.zeros((B, T, Hv), dtype=mx.float16)
        A_log = mx.full((Hv,), -2.0, dtype=mx.float32)
        dt_bias = mx.zeros((Hv,), dtype=mx.float32)
        q, k = normalize_qk(q, k, inv_scale=float(dk) ** -0.5, eps=1e-6)

        y_kern, st_kern = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=True)
        y_ref, st_ref = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=False)
        mx.eval(y_kern, st_kern, y_ref, st_ref)
        assert st_kern.dtype == mx.float32
        np.testing.assert_allclose(np.array(y_kern, dtype=np.float32), np.array(y_ref, dtype=np.float32), atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(np.array(st_kern, dtype=np.float32), np.array(st_ref, dtype=np.float32), atol=2e-3, rtol=2e-3)


@pytest.mark.gpu_ane
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_gpu_real_model_s4096_streamed_vs_whole_parity(
    dtype_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify full 12-layer FastPplxGPU S=4096 streamed vs whole parity in both float16 and bfloat16."""
    import fast_pplx.gpu as pplx_gpu_mod

    snap = _require_cached_weights()
    with _gpu_validation_lock():
        with FastPplxGPU(model_dir=snap, dtype=dtype_name, use_mapped_embedding=True, local_files_only=True) as gpu:
            phrase = (
                "Ocean currents move thermal energy toward higher latitudes across subtropical gyres, "
                "moderating coastal climates over seasonal and decadal intervals. "
            )
            toks = gpu.tokenizer.encode_document(phrase * 220, max_length=4096)
            assert len(toks) == 4096

            monkeypatch.setattr(pplx_gpu_mod, "_STREAMED_SEQ_THRESHOLD", 10_000)
            whole_p = gpu.encode_tokens([toks], batch_size=1, output_dtype="float32", normalize=False, apply_tanh=False)
            whole_i = gpu.encode_tokens([toks], batch_size=1, output_dtype="int8", normalize=False)

            monkeypatch.setattr(pplx_gpu_mod, "_STREAMED_SEQ_THRESHOLD", 2048)
            stream_p = gpu.encode_tokens([toks], batch_size=1, output_dtype="float32", normalize=False, apply_tanh=False)
            stream_i = gpu.encode_tokens([toks], batch_size=1, output_dtype="int8", normalize=False)

            cos = float(
                np.dot(whole_p[0], stream_p[0])
                / max(float(np.linalg.norm(whole_p[0]) * np.linalg.norm(stream_p[0])), 1e-12)
            )
            i8_max_err = int(np.max(np.abs(whole_i.astype(np.int32) - stream_i.astype(np.int32))))
            assert cos >= 0.99995, f"Real model S=4096 ({dtype_name}) streamed vs whole cos={cos}"
            assert i8_max_err <= 1, f"Real model S=4096 ({dtype_name}) streamed vs whole i8_max_err={i8_max_err}"


@pytest.mark.gpu_ane
def test_gpu_varlen_packed_btree_gdn_and_mx_fast_parity() -> None:
    """M-P1.1: 1D varlen multi-doc GPU packing + btree segment-reset GDN + mx.fast.gated_delta_update parity."""
    from fast_pplx.gpu import gated_delta_ops, gated_delta_update, normalize_qk

    snap = _require_cached_weights()
    with _gpu_validation_lock():
        # 1. Direct mx.fast.gated_delta_update / packed btree parity >= 0.9998 across float16 & bfloat16
        mx.random.seed(314)
        for act_dt in (mx.float16, mx.bfloat16):
            q = mx.random.normal((2, 96, 16, 128)).astype(act_dt)
            k = mx.random.normal((2, 96, 16, 128)).astype(act_dt)
            v = (mx.random.normal((2, 96, 16, 128)) * 0.25).astype(act_dt)
            a = (mx.random.normal((2, 96, 16)) * 0.2).astype(act_dt)
            b = (mx.random.normal((2, 96, 16)) * 0.2).astype(act_dt)
            A_log = mx.full((16,), -2.0, dtype=mx.float32)
            dt_bias = mx.full((16,), -1.0, dtype=mx.float32)
            q, k = normalize_qk(q, k, inv_scale=128.0 ** -0.5, eps=1e-6)
            y_fast, st_fast = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=True)
            y_ref, st_ref = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=False)
            mx.eval(y_fast, st_fast, y_ref, st_ref)
            assert st_fast.dtype == mx.float32
            yf = np.array(y_fast.astype(mx.float32)).ravel()
            yr = np.array(y_ref.astype(mx.float32)).ravel()
            cos = float(np.dot(yf, yr) / (np.linalg.norm(yf) * np.linalg.norm(yr)))
            assert cos >= 0.9998, f"GDN fast vs ref cos={cos} for {act_dt}"

        # 2. Multi-doc unequal-length varlen GPU batch matches standalone B=1 calls >= 0.9998 (zero cross-doc leakage)
        with FastPplxGPU(model_dir=snap, dtype="float16", use_mapped_embedding=True, local_files_only=True) as gpu:
            assert hasattr(gpu, "_pack_varlen_token_bins")
            texts = [
                "Short query on ocean thermohaline circulation.",
                "Medium passage detailing solid-state lithium garnet separators and interfacial impedance. " * 4,
                "Brief note.",
                "Extended technical section analyzing bidirectional gated delta linear attention and grouped query RoPE. " * 7,
                "Another concise passage about glacial firn core oxygen isotopes.",
            ]
            toks = [gpu.tokenizer.encode_document(t) for t in texts]
            bins = gpu._pack_varlen_token_bins(toks, max_packed_tokens=1024)
            assert len(bins) < len(toks), "Expected multi-doc 1D varlen packing into fewer bins"

            batched = gpu.encode_tokens(toks, batch_size=8, output_dtype="float32", normalize=True)
            singles = np.concatenate(
                [gpu.encode_tokens([t], batch_size=1, output_dtype="float32", normalize=True) for t in toks],
                axis=0,
            )
            cos_per_doc = np.sum(batched * singles, axis=-1)
            assert float(np.min(cos_per_doc)) >= 0.9998, f"Varlen packed doc parity drift: {cos_per_doc}"

            ctx_docs = [
                ["Introduction to thermal arrays.", "Deep mooring results at 1400 meters."],
                ["Single chunk summary."],
                ["Part one of ceramic sintering.", "", "Part three after empty chunk.", "Final cycle metrics."],
            ]
            ctx_batched = gpu.encode_contextual_chunks(ctx_docs, batch_size=8, output_dtype="float32", normalize=True)
            for di, doc in enumerate(ctx_docs):
                ctx_single = gpu.encode_contextual_chunks([doc], batch_size=1, output_dtype="float32", normalize=True)[0]
                for ci, ch in enumerate(doc):
                    if not ch:
                        assert np.all(ctx_batched[di][ci] == 0.0)
                    else:
                        c_val = float(np.dot(ctx_batched[di][ci], ctx_single[ci]))
                        assert c_val >= 0.9998, f"Varlen contextual doc {di} chunk {ci} cos={c_val}"


@pytest.mark.gpu_ane
def test_gpu_gdn_staged_tiles_segmented_conv_carry_norms_and_nonfinite() -> None:
    from fast_pplx.config import PplxConfig
    from fast_pplx.gpu import (
        PplxGatedDeltaNet,
        PplxGatedRMSNorm,
        PplxMLP,
        PplxRMSNorm,
        gated_delta_update,
        normalize_qk,
    )

    with _gpu_validation_lock():
        mx.random.seed(2026)
        B, T, Hk, Hv, Dk, Dv = 2, 256, 16, 16, 128, 128
        seg_np = np.zeros((B, T), dtype=np.uint8)
        seg_np[0, 0] = 2
        seg_np[0, 1:90] = 1
        seg_np[0, 90] = 2
        seg_np[0, 91:220] = 1
        seg_np[1, 10] = 2
        seg_np[1, 11:250] = 1
        seg_mx = mx.array(seg_np)
        st0 = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32) * 0.1
        A_log = mx.full((Hv,), -2.0, dtype=mx.float32)
        dt_bias = mx.full((Hv,), -1.0, dtype=mx.float32)

        for act_dt in (mx.float16, mx.bfloat16):
            q = mx.random.normal((B, T, Hk, Dk)).astype(act_dt)
            k = mx.random.normal((B, T, Hk, Dk)).astype(act_dt)
            v = (mx.random.normal((B, T, Hv, Dv)) * 0.25).astype(act_dt)
            a = (mx.random.normal((B, T, Hv)) * 0.2).astype(act_dt)
            b = (mx.random.normal((B, T, Hv)) * 0.2).astype(act_dt)
            q, k = normalize_qk(q, k, inv_scale=Dk**-0.5, eps=1e-6)

            y_ref, s_ref = gated_delta_update(
                q, k, v, a, b, A_log, dt_bias, state=st0, seg_flags=seg_mx, use_kernel=False
            )
            mx.eval(y_ref, s_ref)
            yr = np.asarray(y_ref.astype(mx.float32)).ravel()
            sr = np.asarray(s_ref).ravel()

            for tile_rows in (8, 16, 24):
                y_k, s_k = gated_delta_update(
                    q, k, v, a, b, A_log, dt_bias, state=st0, seg_flags=seg_mx, use_kernel=True, tile_rows=tile_rows
                )
                mx.eval(y_k, s_k)
                yk = np.asarray(y_k.astype(mx.float32)).ravel()
                sk = np.asarray(s_k).ravel()
                y_cos = float(np.dot(yk, yr) / (np.linalg.norm(yk) * np.linalg.norm(yr)))
                s_cos = float(np.dot(sk, sr) / (np.linalg.norm(sk) * np.linalg.norm(sr)))
                assert y_cos >= 0.9998, f"tile_rows={tile_rows} dt={act_dt} y_cos={y_cos}"
                assert s_cos >= 0.9998, f"tile_rows={tile_rows} dt={act_dt} s_cos={s_cos}"

        # Non-finite NaN & Inf injection must propagate (never zeroed/clamped)
        v_bad = np.asarray((mx.random.normal((1, 16, Hv, Dv)) * 0.2).astype(mx.float16))
        v_bad[0, 3, 2, 7] = np.nan
        v_inf = v_bad.copy()
        v_inf[0, 3, 2, 7] = np.inf
        q1 = (mx.random.normal((1, 16, Hk, Dk)) * 0.1).astype(mx.float16)
        k1 = (mx.random.normal((1, 16, Hk, Dk)) * 0.1).astype(mx.float16)
        a1 = mx.zeros((1, 16, Hv), dtype=mx.float16)
        b1 = mx.zeros((1, 16, Hv), dtype=mx.float16)
        y_nan, _ = gated_delta_update(q1, k1, mx.array(v_bad), a1, b1, A_log, dt_bias, use_kernel=True, tile_rows=16)
        y_inf, _ = gated_delta_update(q1, k1, mx.array(v_inf), a1, b1, A_log, dt_bias, use_kernel=True, tile_rows=16)
        mx.eval(y_nan, y_inf)
        assert np.isnan(np.asarray(y_nan.astype(mx.float32))[0, 3, 2, 7])
        assert not np.isfinite(np.asarray(y_inf.astype(mx.float32))[0, 3, 2, 7])

        # Fused RMSNorm, GatedRMSNorm, SwiGLU split, and multi-slice Apple9 conv-carry parity
        cfg = PplxConfig(
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=1,
            layer_types=("linear_attention",),
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            rope_dim=32,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            vocab_size=64,
        )
        rms = PplxRMSNorm(1024)
        grms = PplxGatedRMSNorm(1024)
        rms.weight = mx.random.normal((1024,)).astype(mx.float32) * 0.3 + 1.0
        grms.weight = mx.random.normal((1024,)).astype(mx.float32) * 0.3 + 1.0
        x_in = mx.random.normal((2, 19, 1024)).astype(mx.float16)
        z_in = mx.random.normal((2, 19, 1024)).astype(mx.float16)
        y_rms_k = rms(x_in)
        y_rms_ref = mx.fast.rms_norm(x_in.astype(mx.float32), rms.weight, 1e-6).astype(mx.float16)
        y_grms_k = grms(x_in, z_in)
        y_grms_ref = (mx.fast.rms_norm(x_in.astype(mx.float32), grms.weight, 1e-6) * mx.sigmoid(z_in.astype(mx.float32)) * z_in.astype(mx.float32)).astype(mx.float16)
        mx.eval(y_rms_k, y_rms_ref, y_grms_k, y_grms_ref)
        assert float(mx.max(mx.abs(y_rms_k.astype(mx.float32) - y_rms_ref.astype(mx.float32))).item()) <= 1e-2
        assert float(mx.max(mx.abs(y_grms_k.astype(mx.float32) - y_grms_ref.astype(mx.float32))).item()) <= 2e-2

        mlp = PplxMLP(cfg)
        y_mlp_k = mlp._run_slice(x_in)
        gate_up = mlp.gate_up_proj(x_in)
        g_ref, u_ref = mx.split(gate_up, [cfg.intermediate_size], axis=-1)
        act_ref = (g_ref.astype(mx.float32) * mx.sigmoid(g_ref.astype(mx.float32)) * u_ref.astype(mx.float32)).astype(x_in.dtype)
        y_mlp_ref = mlp.down_proj(act_ref)
        mx.eval(y_mlp_k, y_mlp_ref)
        assert float(mx.max(mx.abs(y_mlp_k.astype(mx.float32) - y_mlp_ref.astype(mx.float32))).item()) <= 2e-2

        # Empty T=0 gated_delta_update and short S=1, 2, 3, 64 multi-slice conv carry vs use_kernel=False reference
        y_t0, s_t0 = gated_delta_update(
            mx.zeros((1, 0, Hk, Dk), dtype=mx.float16),
            mx.zeros((1, 0, Hk, Dk), dtype=mx.float16),
            mx.zeros((1, 0, Hv, Dv), dtype=mx.float16),
            mx.zeros((1, 0, Hv), dtype=mx.float16),
            mx.zeros((1, 0, Hv), dtype=mx.float16),
            A_log,
            dt_bias,
            use_kernel=True,
        )
        mx.eval(y_t0, s_t0)
        assert y_t0.shape == (1, 0, Hv, Dv) and s_t0.shape == (1, Hv, Dv, Dk)

        gdn_k = PplxGatedDeltaNet(cfg)
        gdn_ref = PplxGatedDeltaNet(cfg)
        gdn_ref.use_kernel = False
        gdn_ref.update(gdn_k.parameters())
        mx.eval(gdn_k.parameters(), gdn_ref.parameters())
        for slen in (1, 2, 3, 64):
            xs = mx.random.normal((2, slen, 1024)).astype(mx.float16) * 0.2
            cs0 = mx.random.normal((2, 3, gdn_k.conv_dim)).astype(mx.float16) * 0.2
            yk_s, csk, rsk = gdn_k._run_step(xs, cs0, None, None, keep_conv_state_fp32=True)
            yr_s, csr, rsr = gdn_ref._run_step(xs, cs0, None, None, keep_conv_state_fp32=True)
            mx.eval(yk_s, csk, rsk, yr_s, csr, rsr)
            assert csk.dtype == mx.float32
            assert float(mx.max(mx.abs(csk - csr)).item()) <= 2e-3
            assert float(mx.max(mx.abs(yk_s.astype(mx.float32) - yr_s.astype(mx.float32))).item()) <= 2e-3


@pytest.mark.gpu_ane
def test_gpu_block32_int8_state_codec_and_end_to_end_parity() -> None:
    from fast_pplx.gpu import dequantize_gdn_state_i8, gated_delta_update, quantize_gdn_state_i8

    snap = _require_cached_weights()
    with _gpu_validation_lock():
        # 1. Metal Block-32 INT8 state encode/decode parity + NaN/Inf isolation
        rng = np.random.default_rng(77)
        st_np = rng.normal(0.0, 0.25, (2, 16, 128, 128)).astype(np.float32)
        st_np[0, 2, 10, 35] = np.nan
        st_np[1, 5, 20, 100] = -np.inf
        codes, scales = quantize_gdn_state_i8(mx.array(st_np))
        restored = dequantize_gdn_state_i8(codes, scales, out_dtype=mx.float32)
        mx.eval(codes, scales, restored)
        assert codes.nbytes + scales.nbytes < st_np.nbytes // 3
        r_np = np.asarray(restored, dtype=np.float32)
        assert np.all(np.isnan(r_np[0, 2, 10, 32:64]))
        assert np.all(np.isnan(r_np[1, 5, 20, 96:128]))
        clean = np.ones_like(st_np, dtype=bool)
        clean[0, 2, 10, 32:64] = False
        clean[1, 5, 20, 96:128] = False
        cos = float(np.dot(r_np[clean], st_np[clean]) / (np.linalg.norm(r_np[clean]) * np.linalg.norm(st_np[clean])))
        assert cos >= 0.9999

        # 2. Metal gated_delta_update state_dtype="int8" carried chunked parity vs FP32
        mx.random.seed(99)
        q = mx.random.normal((1, 256, 16, 128)).astype(mx.float16) * 0.1
        k = mx.random.normal((1, 256, 16, 128)).astype(mx.float16) * 0.1
        v = mx.random.normal((1, 256, 16, 128)).astype(mx.float16) * 0.2
        a = mx.random.normal((1, 256, 16)).astype(mx.float16) * 0.2
        b = mx.random.normal((1, 256, 16)).astype(mx.float16) * 0.2
        A_log = mx.full((16,), -2.0, dtype=mx.float32)
        dt_bias = mx.full((16,), -1.0, dtype=mx.float32)

        yf_a, sf_a = gated_delta_update(q[:, :128], k[:, :128], v[:, :128], a[:, :128], b[:, :128], A_log, dt_bias, state_dtype="float32")
        yf_b, _ = gated_delta_update(q[:, 128:], k[:, 128:], v[:, 128:], a[:, 128:], b[:, 128:], A_log, dt_bias, state=sf_a, state_dtype="float32")
        yi_a, si_a = gated_delta_update(q[:, :128], k[:, :128], v[:, :128], a[:, :128], b[:, :128], A_log, dt_bias, state_dtype="int8")
        yi_b, si_b = gated_delta_update(q[:, 128:], k[:, 128:], v[:, 128:], a[:, 128:], b[:, 128:], A_log, dt_bias, state=si_a, state_dtype="int8")
        mx.eval(yf_a, yf_b, yi_a, yi_b, si_b[0], si_b[1])
        assert isinstance(si_b, tuple) and si_b[0].dtype == mx.int8 and si_b[1].dtype == mx.bfloat16
        yf_all = np.asarray(mx.concatenate([yf_a, yf_b], axis=1).astype(mx.float32)).ravel()
        yi_all = np.asarray(mx.concatenate([yi_a, yi_b], axis=1).astype(mx.float32)).ravel()
        assert float(np.dot(yf_all, yi_all) / (np.linalg.norm(yf_all) * np.linalg.norm(yi_all))) >= 0.9999

        # 3. End-to-end real-weight FastPplxGPU state_dtype="int8" parity (including S=4096 streamed boundary)
        docs = [
            "Short ocean current passage.",
            "Ceramic oxide ionic conductivity across garnet grain boundaries. " * 25,
            "Bidirectional gated delta recurrence maintains orthogonal state across long contexts. " * 260,
        ]
        with FastPplxGPU(model_dir=snap, dtype="float16", state_dtype="float32", use_mapped_embedding=True, local_files_only=True) as g_f32:
            v_f32 = g_f32.encode_documents(docs, batch_size=2, max_length=4096, output_dtype="float32", normalize=True)
        with FastPplxGPU(model_dir=snap, dtype="float16", state_dtype="int8", use_mapped_embedding=True, local_files_only=True) as g_i8:
            assert g_i8.state_dtype_name == "int8"
            v_i8 = g_i8.encode_documents(docs, batch_size=2, max_length=4096, output_dtype="float32", normalize=True)
        cos_e2e = np.sum(v_f32 * v_i8, axis=-1)
        assert float(np.min(cos_e2e)) >= 0.9999, f"End-to-end int8 vs float32 state drift: {cos_e2e}"


@pytest.mark.gpu_ane
def test_gpu_gdn_streamed_slice0_invariance_and_fp32_beta_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlx.utils import tree_map
    import fast_pplx.gpu as gpu_mod
    from fast_pplx.gpu import PplxConfig, PplxGatedDeltaNet, gated_delta_update, normalize_qk

    with _gpu_validation_lock():
        # Force Apple10 mx.fast GDN path so even M4 Pro exercises Base M5's slice-0 dispatch.
        monkeypatch.setattr(gpu_mod, "_CACHED_IS_APPLE10_HOST", True)
        cfg = PplxConfig(
            vocab_size=64,
            hidden_size=128,
            num_hidden_layers=1,
            layer_types=("linear_attention",),
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            rope_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        mx.random.seed(123)
        gdn = PplxGatedDeltaNet(cfg, state_dtype="float32")
        gdn.update(tree_map(lambda arr: arr.astype(mx.float16) if isinstance(arr, mx.array) else arr, gdn.parameters()))
        gdn.A_log = gdn.A_log.astype(mx.float32)
        gdn.dt_bias = gdn.dt_bias.astype(mx.float32)
        gdn.norm.weight = gdn.norm.weight.astype(mx.float32)

        x = mx.random.normal((1, 2560, 128)).astype(mx.float16) * 0.25
        y_stream = gdn(x)
        conv0 = mx.zeros((1, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=mx.float32)
        rec0 = mx.zeros((1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32)
        y_whole, _, _ = gdn._run_step(x, conv0, rec0, None, keep_conv_state_fp32=True)
        y_slice0, _, _ = gdn._run_step(x[:, :1024, :], conv0, rec0, None, keep_conv_state_fp32=True)
        mx.eval(y_stream, y_whole, y_slice0)
        assert float(mx.max(mx.abs(y_stream[:, :1024, :] - y_slice0)).item()) <= 1e-6
        assert float(mx.max(mx.abs(y_stream - y_whole)).item()) <= 1e-6

        q, k = normalize_qk(
            mx.random.normal((1, 256, 4, 128)),
            mx.random.normal((1, 256, 4, 128)),
            inv_scale=128**-0.5,
        )
        q = q.astype(mx.float16)
        k = k.astype(mx.float16)
        v = mx.random.normal((1, 256, 4, 128)).astype(mx.float16) * 0.2
        a = mx.random.normal((1, 256, 4)).astype(mx.float16) * 0.2
        b = mx.random.normal((1, 256, 4)).astype(mx.float16) * 0.2
        A_log = mx.full((4,), -2.0, dtype=mx.float32)
        dt_bias = mx.full((4,), -1.0, dtype=mx.float32)
        init_st = mx.zeros((1, 4, 128, 128), dtype=mx.float32)
        y_kern, st_kern = gated_delta_update(
            q, k, v, a, b, A_log, dt_bias, state=init_st, use_kernel=True, state_dtype="float32"
        )
        y_ref, st_ref = gated_delta_update(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            a.astype(mx.float32),
            b.astype(mx.float32),
            A_log,
            dt_bias,
            state=init_st,
            use_kernel=False,
            state_dtype="float32",
        )
        mx.eval(y_kern, st_kern, y_ref, st_ref)
        assert float(mx.max(mx.abs(st_kern - st_ref)).item()) < 5e-6
        assert float(mx.max(mx.abs(y_kern.astype(mx.float32) - y_ref)).item()) < 3e-5
