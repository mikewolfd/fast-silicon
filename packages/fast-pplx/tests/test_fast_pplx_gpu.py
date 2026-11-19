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
                    proj = gpu.encode_queries(flat_q, output_dtype="float32", normalize=False)
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
                    proj_list = gpu.encode_contextual_chunks(docs, output_dtype="float32", normalize=False)
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
            whole_p = gpu.encode_tokens([toks], batch_size=1, output_dtype="float32", normalize=False)
            whole_i = gpu.encode_tokens([toks], batch_size=1, output_dtype="int8", normalize=False)

            monkeypatch.setattr(pplx_gpu_mod, "_STREAMED_SEQ_THRESHOLD", 2048)
            stream_p = gpu.encode_tokens([toks], batch_size=1, output_dtype="float32", normalize=False)
            stream_i = gpu.encode_tokens([toks], batch_size=1, output_dtype="int8", normalize=False)

            cos = float(
                np.dot(whole_p[0], stream_p[0])
                / max(float(np.linalg.norm(whole_p[0]) * np.linalg.norm(stream_p[0])), 1e-12)
            )
            i8_max_err = int(np.max(np.abs(whole_i.astype(np.int32) - stream_i.astype(np.int32))))
            assert cos >= 0.99995, f"Real model S=4096 ({dtype_name}) streamed vs whole cos={cos}"
            assert i8_max_err <= 1, f"Real model S=4096 ({dtype_name}) streamed vs whole i8_max_err={i8_max_err}"
