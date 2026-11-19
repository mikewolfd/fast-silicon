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
