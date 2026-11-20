from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_embeddinggemma_2 import DOCUMENT_PREFIX, MAX_EMBED_DIM, MRL_DIMENSIONS, QUERY_PREFIX, FastEmbeddingGemma2, resolve_paths
from fast_embeddinggemma_2.parquet_pipeline import embed_parquet, slice_ocr_pages_utf8
from fast_embeddinggemma_2.server import FastEmbeddingGemma2Server

FIXTURE_64 = Path(__file__).resolve().parent / "fixtures" / "ref_vecs_64.npz"


@pytest.fixture(scope="module")
def engine() -> FastEmbeddingGemma2:
    return FastEmbeddingGemma2(warmup=True)


def test_ref_64_parity_all_modes(engine: FastEmbeddingGemma2) -> None:
    ref = np.load(FIXTURE_64)
    ids_64 = ref["ids"]
    lens_64 = ref["lens"]
    v_f16 = ref["v_f16"]
    v_bf16 = ref["v_stock_bf16"]
    sub_lists = [ids_64[i, : int(lens_64[i])].tolist() for i in range(64)]

    v_gpu = engine.encode_corpus(sub_lists, mode="gpu")
    v_het = engine.encode_corpus(sub_lists, mode="hetero")
    assert v_gpu.shape == (64, 768) and v_gpu.dtype == np.float32
    assert v_het.shape == (64, 768) and v_het.dtype == np.float32

    assert np.allclose(np.linalg.norm(v_gpu, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(v_het, axis=1), 1.0, atol=1e-4)

    assert float(np.min(np.sum(v_gpu * v_f16, axis=1))) >= 0.9999
    assert float(np.min(np.sum(v_gpu * v_bf16, axis=1))) >= 0.9997
    assert float(np.min(np.sum(v_het * v_f16, axis=1))) >= 0.9995

    if engine.ane_available:
        # Doc #63 has 744 tokens (>512), so strict mode='ane' refuses silent GPU overflow unless opted in
        with pytest.raises(ValueError, match="longer than 512"):
            engine.encode_corpus(sub_lists, mode="ane")
        le512 = [i for i, s in enumerate(sub_lists) if len(s) <= 512]
        v_ane = engine.encode_corpus([sub_lists[i] for i in le512], mode="ane")
        assert engine.last_stats is not None and engine.last_stats["gpu_docs"] == 0
        assert float(np.min(np.sum(v_ane * v_gpu[le512], axis=1))) >= 0.9995
        v_ane_ov = engine.encode_corpus(sub_lists, mode="ane", allow_gpu_overflow=True)
        assert float(np.min(np.sum(v_ane_ov * v_gpu, axis=1))) >= 0.9995
        v_h2 = engine.encode_corpus(sub_lists, mode="hetero2")
        assert float(np.min(np.sum(v_h2 * v_gpu, axis=1))) >= 0.9995

    if engine.cpu_sme_available:
        le512 = [i for i, s in enumerate(sub_lists) if len(s) <= 512]
        v_cpu = engine.encode_corpus([sub_lists[i] for i in le512], mode="cpu")
        assert float(np.min(np.sum(v_cpu * v_gpu[le512], axis=1))) >= 0.9995


def test_query_buckets_parity_and_edge_cases(engine: FastEmbeddingGemma2) -> None:
    assert engine.encode_texts([]).shape == (0, 768)
    assert engine.encode_corpus([]).shape == (0, 768)

    v_empty = engine.encode_texts("")
    assert v_empty.shape == (768,)
    assert abs(float(np.linalg.norm(v_empty)) - 1.0) < 1e-4
    v_empty_gpu = engine.encode_query([], prefer_ane=False)
    assert float(np.dot(v_empty, v_empty_gpu)) >= 0.9995

    assert engine.encode_texts(["single item"]).shape == (1, 768)

    base_text = (
        "The regional engineering board reviewed annual capital allocations for grid modernization, "
        "fiber transit telemetry, water treatment upgrades, and safety audits across regional facilities."
    )
    rng = np.random.default_rng(7)
    for sl in (16, 32, 64, 128, 220, 256, 360, 384, 480, 512):
        # Test realistic English subword tokens with strict >= 0.9997 threshold
        real_ids = engine.tokenizer.encode((base_text + " ") * 40, max_length=sl)
        if len(real_ids) < sl:
            real_ids = real_ids[:-1] + [real_ids[1]] * (sl - len(real_ids)) + [real_ids[-1]]
        q_ane = engine.encode_query(real_ids, prefer_ane=True)
        q_gpu = engine.encode_query(real_ids, prefer_ane=False)
        assert abs(float(np.linalg.norm(q_gpu)) - 1.0) < 1e-4
        assert float(np.dot(q_ane, q_gpu)) >= 0.9997, f"Real-text mismatch at len={sl}"

        # Also test random token IDs across every length bucket
        toks = [engine.tokenizer.bos_id] + rng.integers(1000, 30000, size=sl - 2).tolist() + [engine.tokenizer.eos_id]
        assert float(np.dot(engine.encode_query(toks, prefer_ane=True), engine.encode_query(toks, prefer_ane=False))) >= 0.999

    # >512 token query automatically runs full GPU 2048-context path
    long_toks = [engine.tokenizer.bos_id] + rng.integers(100, 200000, size=698).tolist() + [engine.tokenizer.eos_id]
    q_long = engine.encode_query(long_toks, prefer_ane=True)
    assert q_long.shape == (768,)
    assert abs(float(np.linalg.norm(q_long)) - 1.0) < 1e-4
    q_long_batch = engine.encode_corpus([long_toks], mode="gpu")[0]
    assert float(np.dot(q_long, q_long_batch)) >= 0.9999


def test_tokenizer_and_chunker(engine: FastEmbeddingGemma2) -> None:
    short = "Pacific Transmission Consortium Seasonal Inverter & Synchronous Reserve Assessment."
    short_ids = engine.tokenizer.encode(short)
    chunks = engine.tokenizer.chunk_text(short, max_body_tokens=510, overlap=51)
    assert len(chunks) == 1
    assert chunks[0][0] == short_ids
    assert chunks[0][1] == 0 and chunks[0][2] == len(short)

    q_pref = engine.tokenizer.encode(short, add_query_prefix=True)
    assert q_pref == engine.tokenizer.encode(QUERY_PREFIX + short)
    assert engine.tokenizer.encode(short, prompt_name="query") == q_pref

    d_pref = engine.tokenizer.encode(short, add_document_prefix=True)
    assert d_pref == engine.tokenizer.encode(DOCUMENT_PREFIX + short)

    long_text = " ".join(["distributed energy storage substation inverter telemetry engineering"] * 130)
    long_chunks = engine.tokenizer.chunk_text(long_text, max_body_tokens=504, overlap=51, add_document_prefix=True)
    assert len(long_chunks) >= 2
    for ids, c0, c1 in long_chunks:
        assert len(ids) <= 512
        assert ids[0] == engine.tokenizer.bos_id
        assert ids[-1] == engine.tokenizer.eos_id
        assert 0 <= c0 < c1 <= len(long_text)


def test_native_emb2_ane_cli(engine: FastEmbeddingGemma2) -> None:
    paths = resolve_paths()
    assert paths.ane_cli.exists(), f"Missing native CLI at {paths.ane_cli}"

    text = "What is the federal budget deficit?"
    ids = engine.tokenizer.encode(text, add_query_prefix=True)
    id_arg = ",".join(str(x) for x in ids)

    proc = subprocess.run(
        [str(paths.ane_cli), "--ids", id_arg, "--seq-len", "32", "--bench", "5", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert len(payload["embedding"]) == 768
    vec_cli = np.asarray(payload["embedding"], dtype=np.float32)
    vec_py = engine.encode_query(ids, prefer_ane=True)
    vec_gpu = engine.encode_query(ids, prefer_ane=False)
    assert float(np.dot(vec_cli, vec_py)) >= 0.9999
    assert float(np.dot(vec_cli, vec_gpu)) >= 0.9995

    if engine.cpu_sme_available:
        proc_cpu = subprocess.run(
            [str(paths.ane_cli), "--ids", id_arg, "--compute", "cpu", "--seq-len", "256", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        vec_cpu = np.asarray(json.loads(proc_cpu.stdout)["embedding"], dtype=np.float32)
        assert float(np.dot(vec_cpu, vec_gpu)) >= 0.9995


def test_slice_ocr_pages_utf8() -> None:
    p1 = "Intro — header page."
    p2 = "OCR scanned exhibit text."
    p3 = "Native appendix."
    full = p1 + p2 + p3
    b1 = len(p1.encode("utf-8"))
    b2 = len((p1 + p2).encode("utf-8"))
    offsets = [0, b1, b2]
    assert slice_ocr_pages_utf8(full, offsets, [2]) == p2
    assert slice_ocr_pages_utf8(full, offsets, [1, 3]) == f"{p1}\n\n{p3}"


def test_parquet_pipeline_roundtrip_and_sharded_resume(engine: FastEmbeddingGemma2, tmp_path: Path) -> None:
    in_path = tmp_path / "sample.parquet"
    rows = [
        {
            "doc_id": f"doc_{i}",
            "is_ocr": i % 2 == 0,
            "body": f"Hearing passage number {i} covering federal infrastructure allocations and oversight.",
        }
        for i in range(12)
    ]
    pq.write_table(pa.Table.from_pylist(rows), in_path)

    out_f32 = tmp_path / "out_f32.parquet"
    res = embed_parquet(engine, in_path, out_f32, text_col="body", filter_ocr=True, progress_every=0)
    assert res.rows_read == 6
    assert res.rows_emitted == 6

    tbl = pq.read_table(out_f32)
    assert tbl.num_rows == 6
    assert tbl.schema.field("vector").type == pa.list_(pa.float32(), 768)
    vecs = np.asarray(tbl["vector"].to_pylist(), dtype=np.float32)
    expected = engine.encode_texts([r["body"] for r in rows if r["is_ocr"]])
    assert float(np.min(np.sum(vecs * expected, axis=1))) >= 0.9998

    out_dir = tmp_path / "shards"
    res2 = embed_parquet(
        engine,
        in_path,
        out_dir,
        text_col="body",
        dtype="float16",
        shard_rows=4,
        chunk=True,
        max_body_tokens=504,
        add_document_prefix=True,
        drop_text_col=True,
        progress_every=0,
    )
    assert res2.rows_emitted == 12
    parts = sorted(out_dir.glob("part-*.parquet"))
    assert len(parts) == 3
    first_part = pq.read_table(parts[0])
    assert "body" not in first_part.column_names
    assert first_part.schema.field("vector").type == pa.list_(pa.float16(), 768)

    mtime_0 = parts[0].stat().st_mtime_ns
    parts[1].unlink()
    embed_parquet(
        engine,
        in_path,
        out_dir,
        text_col="body",
        dtype="float16",
        shard_rows=4,
        chunk=True,
        max_body_tokens=504,
        add_document_prefix=True,
        drop_text_col=True,
        resume=True,
        progress_every=0,
    )
    assert parts[0].stat().st_mtime_ns == mtime_0
    assert parts[1].exists() and pq.read_table(parts[1]).num_rows == 4


def test_http_server_smoke(engine: FastEmbeddingGemma2) -> None:
    texts = ["grid reliability assessment FY2025", "watershed dissolved oxygen lab survey"]
    expected = engine.encode_texts(texts)

    with FastEmbeddingGemma2Server(engine, host="127.0.0.1", port=0) as srv:
        base_url = f"http://{srv.host}:{srv.port}"
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            assert resp.status == 200
            health = json.loads(resp.read().decode("utf-8"))
            assert health["status"] == "ok" and health["dim"] == 768

        body = json.dumps({"input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data["data"]) == 2
            got = np.asarray([row["embedding"] for row in data["data"]], dtype=np.float32)
            assert float(np.min(np.sum(got * expected, axis=1))) >= 0.9998


def test_mrl_dimensions_and_cli_cpu(tmp_path: Path) -> None:
    """CPU-only verification of Matryoshka dimension selection (API, CLI, Parquet, HTTP, and ref parity)."""
    import threading
    import urllib.error

    from fast_embeddinggemma_2.cli import build_parser
    from fast_shared.embeddings import truncate_and_normalize

    assert MAX_EMBED_DIM == 768 and MRL_DIMENSIONS == (128, 256, 512, 768)

    p = build_parser()
    for cmd, extra in (
        ("embed", ["--dimensions", "256", "hello"]),
        ("parquet", ["--input", "in.parquet", "--output", "out.parquet", "--dimensions", "256"]),
        ("serve", ["--dimensions", "128"]),
        ("bench", ["--dimensions", "256"]),
    ):
        ns = p.parse_args([cmd, *extra])
        assert ns.dimensions in (128, 256)

    for bad_dim in (0, -1, 769, True, 256.5):
        with pytest.raises(ValueError):
            FastEmbeddingGemma2(dimensions=bad_dim, warmup=False)

    # Verify MRL truncation + L2 re-normalization preserves high F16/BF16 parity on real 64-doc fixture vectors
    ref = np.load(FIXTURE_64)
    v_f16 = ref["v_f16"]
    v_bf16 = ref["v_stock_bf16"]
    for d in (512, 256, 128):
        s_f16 = truncate_and_normalize(v_f16, d)
        s_bf16 = truncate_and_normalize(v_bf16, d)
        assert s_f16.shape == (64, d)
        assert np.allclose(np.linalg.norm(s_f16, axis=1), 1.0, atol=1e-6)
        assert float(np.min(np.sum(s_f16 * s_bf16, axis=1))) >= 0.9995

    # Build lightweight CPU-only FastEmbeddingGemma2 wrapper (no Metal/ANE initialization) to exercise engine + server + parquet paths
    class _CpuMockTok:
        bos_id = 2
        eos_id = 1

        def resolve_prefix(self, **kwargs):
            return kwargs.get("prefix") or ""

        def encode(self, _text, **_kwargs):
            return [2, 10, 1]

        def encode_batch(self, texts, **_kwargs):
            return [[2, 10 + i, 1] for i, _ in enumerate(texts)]

    class _CpuMockGPU:
        def encode_single(self, toks):
            idx = sum(toks) % len(v_f16)
            return v_f16[idx].astype(np.float32)

    stub = FastEmbeddingGemma2.__new__(FastEmbeddingGemma2)
    stub.dim = 256
    stub.max_dim = 768
    stub.supports_dimensions = True
    stub.mrl_dimensions = MRL_DIMENSIONS
    stub.model_name = "google/embeddinggemma-2"
    stub.max_seq_len = 2048
    stub.ane_query_max_len = 512
    stub.ane_available = False
    stub.cpu_sme_available = False
    stub.default_mode = "gpu"
    stub.tokenizer = _CpuMockTok()
    stub.gpu = _CpuMockGPU()
    stub._gpu_lock = threading.Lock()
    stub._corpus_lock = threading.Lock()
    stub.last_stats = None

    assert stub.encode_texts([]).shape == (0, 256)
    assert stub.encode_corpus([], dimensions=128).shape == (0, 128)
    assert stub.encode_query([2, 3, 1]).shape == (256,)
    assert stub.encode_query([2, 3, 1], dimensions=512).shape == (512,)
    assert stub.encode_corpus([[2, 10, 1]], mode="gpu", dimensions=128).shape == (1, 128)
    v_qbatch = stub.encode_texts(["doc a", "doc b"], is_query=True, dimensions=128)
    assert v_qbatch.shape == (2, 128)
    assert np.allclose(np.linalg.norm(v_qbatch, axis=1), 1.0, atol=1e-6)

    def _cpu_encode_corpus(token_lists, *, mode=None, dimensions=None):
        eff_dim = stub._resolve_dim(dimensions)
        if not token_lists:
            return np.empty((0, eff_dim), dtype=np.float32)
        rows = np.stack([v_f16[sum(t) % len(v_f16)] for t in token_lists], axis=0)
        return truncate_and_normalize(rows, eff_dim)

    stub.encode_corpus = _cpu_encode_corpus  # type: ignore[method-assign]
    v_batch = stub.encode_texts(["doc a", "doc b"], dimensions=128)
    assert v_batch.shape == (2, 128)
    assert np.allclose(np.linalg.norm(v_batch, axis=1), 1.0, atol=1e-6)

    in_pq = tmp_path / "in.parquet"
    out_pq = tmp_path / "out_256.parquet"
    pq.write_table(pa.Table.from_pylist([{"body": "first"}, {"body": "second"}]), in_pq)
    embed_parquet(stub, in_pq, out_pq, text_col="body", dtype="float16", progress_every=0)
    tbl = pq.read_table(out_pq)
    assert tbl.schema.field("vector").type == pa.list_(pa.float16(), 256)

    out_pq_128 = tmp_path / "out_128.parquet"
    embed_parquet(stub, in_pq, out_pq_128, text_col="body", dtype="float32", dimensions=128, progress_every=0)
    tbl_128 = pq.read_table(out_pq_128)
    assert tbl_128.schema.field("vector").type == pa.list_(pa.float32(), 128)

    with FastEmbeddingGemma2Server(stub, host="127.0.0.1", port=0) as srv:
        base = f"http://{srv.host}:{srv.port}"
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            h = json.loads(r.read().decode("utf-8"))
            assert h["dim"] == 256 and h["max_dim"] == 768 and h["mrl_dimensions"] == [128, 256, 512, 768]

        for req_dim, expected_dim, inp in (
            (None, 256, ["query one"]),
            (128, 128, ["doc one", "doc two"]),
            (512, 512, ["query one"]),
            (768, 768, ["doc one", "doc two"]),
        ):
            payload: dict[str, object] = {"input": inp}
            if req_dim is not None:
                payload["dimensions"] = req_dim
            req = urllib.request.Request(
                f"{base}/v1/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                body = json.loads(r.read().decode("utf-8"))
                assert len(body["data"]) == len(inp)
                for row in body["data"]:
                    vec = np.asarray(row["embedding"], dtype=np.float32)
                    assert vec.shape == (expected_dim,)
                    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

        bad_req = urllib.request.Request(
            f"{base}/v1/embeddings",
            data=json.dumps({"input": "x", "dimensions": 999}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad_req, timeout=5)
        assert exc.value.code == 400


def test_fused_res_rms_metal_gpu_parity_and_empty_guard() -> None:
    """Verify fused FP16 Residual+RMSNorm Metal kernels directly on M4 Pro GPU."""
    import contextlib
    import fcntl
    import os
    import mlx.core as mx
    from fast_embeddinggemma_2.gpu import fused_gemma_res_rms

    @contextlib.contextmanager
    def _gpu_lock():
        if os.environ.get("M4_GPU_LOCK_HELD") == "1":
            yield
            return
        with open("/tmp/m4_gpu_validation.lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    rng = np.random.default_rng(1337)
    with _gpu_lock():
        w_post = mx.array((1.0 + rng.normal(0.0, 0.15, (512,))).astype(np.float16))
        w_next = mx.array((1.0 + rng.normal(0.0, 0.15, (512,))).astype(np.float16))
        sc = mx.array([0.875], dtype=mx.float16)

        # 1. Empty batch m_rows == 0 guard must never launch grid=(0, 1, 1)
        for empty_shape in ((0, 512), (0, 1, 512)):
            zh = mx.zeros(empty_shape, dtype=mx.float16)
            for use_sc in (None, sc):
                yr0 = fused_gemma_res_rms(zh, zh, w_post, None, scalar=use_sc)
                assert yr0.shape == empty_shape and yr0.dtype == mx.float16
                yr1, yn1 = fused_gemma_res_rms(zh, zh, w_post, w_next, scalar=use_sc)
                assert yr1.shape == empty_shape and yn1.shape == empty_shape
                assert yr1.dtype == mx.float16 and yn1.dtype == mx.float16

        # 2. Metal kernel parity across M=1,7,8,64,128 (2D) & (2,32,512) (3D)
        for shape in ((1, 512), (7, 512), (8, 512), (64, 512), (128, 512), (2, 32, 512)):
            h = mx.array(rng.normal(0.0, 1.0, shape).astype(np.float16))
            sub = mx.array(rng.normal(0.0, 1.0, shape).astype(np.float16))

            # Unfused pure-MLX references
            yr_ref = h + mx.fast.rms_norm(sub, w_post, 1e-6)
            yn_ref = mx.fast.rms_norm(yr_ref, w_next, 1e-6)
            yr_sc_ref = yr_ref * sc[0]
            yn_sc_ref = mx.fast.rms_norm(yr_sc_ref, w_next, 1e-6)

            yr_a = fused_gemma_res_rms(h, sub, w_post, None)
            yr_b, yn_b = fused_gemma_res_rms(h, sub, w_post, w_next)
            yr_c = fused_gemma_res_rms(h, sub, w_post, None, scalar=sc)
            yr_d, yn_d = fused_gemma_res_rms(h, sub, w_post, w_next, scalar=sc)
            mx.eval(yr_ref, yn_ref, yr_sc_ref, yn_sc_ref, yr_a, yr_b, yn_b, yr_c, yr_d, yn_d)

            for got, ref in (
                (yr_a, yr_ref),
                (yr_b, yr_ref),
                (yn_b, yn_ref),
                (yr_c, yr_sc_ref),
                (yr_d, yr_sc_ref),
                (yn_d, yn_sc_ref),
            ):
                assert got.shape == shape and got.dtype == mx.float16
                g32 = np.array(got.astype(mx.float32), copy=False).reshape(-1, 512).astype(np.float64)
                r32 = np.array(ref.astype(mx.float32), copy=False).reshape(-1, 512).astype(np.float64)
                assert float(np.max(np.abs(g32 - r32))) < 1e-2
                dots = np.sum(g32 * r32, axis=-1)
                norms = np.linalg.norm(g32, axis=-1) * np.linalg.norm(r32, axis=-1) + 1e-12
                assert float(np.min(dots / norms)) > 0.99999


def test_t1_over_buckets_no_jit_and_gpu_seg_pool_mrl(engine: FastEmbeddingGemma2) -> None:
    import mlx.core as mx
    from fast_shared.embeddings import truncate_and_normalize
    from fast_embeddinggemma_2.gpu import (
        GPU_B_OPTS,
        OVER_B_OPTS,
        OVER_BUCKETS,
        build_over_chunks,
        mean_pool_segments,
    )

    assert OVER_BUCKETS == (640, 768, 1024, 1536, 2048)
    for ml, opts in OVER_B_OPTS.items():
        assert ml in OVER_BUCKETS and opts[-1] == 1
    for _l_lane, opts in GPU_B_OPTS.items():
        assert opts[-1] == 1

    over_items = [(i, [2] + [100] * (520 + i * 7) + [1]) for i in range(9)]
    chunks = build_over_chunks(over_items)
    assert sum(len(c) for _, c in chunks) == len(over_items)
    for ml, chk in chunks:
        assert ml in OVER_BUCKETS
        assert len(chk) in OVER_B_OPTS[ml]

    empty_out = mean_pool_segments(mx.zeros((0, 512), dtype=mx.float16), [], [])
    mx.eval(empty_out)
    assert empty_out.shape == (0, 512) and empty_out.dtype == mx.float32

    rng = np.random.default_rng(123)
    ho_np = rng.standard_normal((40, 512)).astype(np.float16)
    segs = [(0, 7), (10, 1), (12, 19), (33, 5)]
    starts = [s for s, _ in segs]
    lens = [l for _, l in segs]
    pooled = mean_pool_segments(mx.array(ho_np), starts, lens)
    mx.eval(pooled)
    pooled_np = np.array(pooled)
    assert pooled_np.shape == (len(segs), 512) and pooled_np.dtype == np.float32
    expected_np = np.stack([ho_np[s : s + l].astype(np.float32).mean(axis=0) for s, l in segs], axis=0)
    assert np.allclose(pooled_np, expected_np, atol=1e-3, rtol=1e-3)

    # End-to-end GPU overflow + odd-tail + MRL 256d parity
    tok_lists = [
        [2] + rng.integers(100, 30000, size=ln - 2).tolist() + [1]
        for ln in (19, 133, 512, 600, 740, 990, 1400, 1850)
    ]
    v_gpu_768 = engine.encode_corpus(tok_lists, mode="gpu")
    v_gpu_256 = engine.encode_corpus(tok_lists, mode="gpu", dimensions=256)
    assert v_gpu_768.shape == (len(tok_lists), 768)
    assert v_gpu_256.shape == (len(tok_lists), 256)
    exp_256 = truncate_and_normalize(v_gpu_768, 256)
    assert float(np.min(np.sum(v_gpu_256 * exp_256, axis=1))) >= 0.9999


def test_t2_hetero_sustained_budget_cap_cpu() -> None:
    from fast_embeddinggemma_2.engine import HETERO_CPU_MAX_DOCS, HETERO_CPU_MAX_SEC, _should_stop_hetero_cpu

    assert HETERO_CPU_MAX_DOCS > 0 and HETERO_CPU_MAX_SEC > 0.0
    assert not _should_stop_hetero_cpu(0.1, 32, 4096, eff_mode="hetero")
    assert _should_stop_hetero_cpu(HETERO_CPU_MAX_SEC + 0.05, 16, 1024, eff_mode="hetero")
    assert _should_stop_hetero_cpu(0.1, HETERO_CPU_MAX_DOCS, 4096, eff_mode="hetero")
    # Explicit cpu mode never triggers the hetero sustained cap
    assert not _should_stop_hetero_cpu(99.0, 99999, 9999999, eff_mode="cpu")


def test_t4_ane_nine_buckets_and_async_overlap(engine: FastEmbeddingGemma2) -> None:
    from fast_embeddinggemma_2.engine import _QUERY_CAPS

    expected_buckets = (32, 64, 128, 192, 256, 320, 384, 448, 512)
    assert _QUERY_CAPS == expected_buckets
    if not engine.ane_available:
        pytest.skip("ANE not available")

    assert tuple(sorted(engine.ane_handles.keys())) == expected_buckets
    assert engine._coreml is not None and engine._io is not None
    assert hasattr(engine._coreml, "emb2_pk_submit_queue")
    assert hasattr(engine._coreml, "emb2_pk_wait_queue")
    assert hasattr(engine._io, "emb2_pool_project_batch_dim")

    base_text = (
        "Pacific Transmission Consortium technical reliability assessment and audit of substation automation."
    )
    for sl in (24, 48, 96, 160, 224, 290, 350, 410, 480):
        ids = engine.tokenizer.encode((base_text + " ") * 40, max_length=sl)
        if len(ids) < sl:
            ids = ids[:-1] + [ids[1]] * (sl - len(ids)) + [ids[-1]]
        q_ane = engine.encode_query(ids, prefer_ane=True)
        q_gpu = engine.encode_query(ids, prefer_ane=False)
        assert float(np.dot(q_ane, q_gpu)) >= 0.9997, f"ANE/GPU mismatch at len={sl}"

    # Multi-bucket ANE corpus including 64/192/320/448 lanes + async double-buffer + MRL 256d
    lens = [48] * 8 + [160] * 12 + [290] * 12 + [410] * 12 + [200] * 10 + [490] * 10
    real_toks = [
        engine.tokenizer.encode((f"Section {i}: " + base_text + " ") * 40, max_length=ln)
        for i, ln in enumerate(lens)
    ]
    v_gpu = engine.encode_corpus(real_toks, mode="gpu")
    v_ane = engine.encode_corpus(real_toks, mode="ane")
    v_ane_256 = engine.encode_corpus(real_toks, mode="ane", dimensions=256)
    v_gpu_256 = engine.encode_corpus(real_toks, mode="gpu", dimensions=256)
    assert float(np.min(np.sum(v_ane * v_gpu, axis=1))) >= 0.9995
    assert float(np.min(np.sum(v_ane_256 * v_gpu_256, axis=1))) >= 0.9995

    rng = np.random.default_rng(99)
    rand_toks = [[2] + rng.integers(100, 30000, size=ln - 2).tolist() + [1] for ln in lens]
    vr_gpu = engine.encode_corpus(rand_toks, mode="gpu")
    vr_ane = engine.encode_corpus(rand_toks, mode="ane")
    assert float(np.min(np.sum(vr_ane * vr_gpu, axis=1))) >= 0.999


def test_warmup_skips_gpu_on_ane_default_and_over_by_default() -> None:
    import threading
    from types import SimpleNamespace
    from fast_embeddinggemma_2.engine import FastEmbeddingGemma2
    from fast_embeddinggemma_2.gpu import GPU_B_OPTS, OVER_B_OPTS

    calls: list[tuple[str, object]] = []

    class _FakeGPU:
        def encode_single(self, toks):
            calls.append(("encode_single", len(toks)))
            return np.ones(768, dtype=np.float32)

        def exec_packed(self, L_l, lanes):
            calls.append(("exec_packed", (L_l, len(lanes))))
            return 0, 0, [], SimpleNamespace()

        def exec_over(self, ml, items):
            calls.append(("exec_over", (ml, len(items))))
            return 0, 0, [], SimpleNamespace()

    eng = FastEmbeddingGemma2.__new__(FastEmbeddingGemma2)
    eng.ane_available = True
    eng.default_mode = "ane"
    eng._gpu_lock = threading.Lock()
    eng.gpu = _FakeGPU()  # type: ignore[assignment]
    eng.encode_query = lambda ids, prefer_ane=True, **_kw: (  # type: ignore[method-assign]
        calls.append(("encode_query", prefer_ane)) or np.ones(768, dtype=np.float32)
    )

    eng.warmup(prewarm_gpu_shapes=True)
    assert calls == [("encode_query", True)]

    # Switch to hetero2: warms encode_single + GPU_B_OPTS, skips OVER_B_OPTS unless explicitly requested
    calls.clear()
    eng.default_mode = "hetero2"
    import mlx.core as mx

    orig_eval, orig_sync = mx.eval, mx.synchronize
    mx.eval = lambda *_args, **_kw: None  # type: ignore[assignment]
    mx.synchronize = lambda: None  # type: ignore[assignment]
    try:
        eng.warmup(prewarm_gpu_shapes=True)
        packed_shapes = {arg for op, arg in calls if op == "exec_packed"}
        over_shapes = {arg for op, arg in calls if op == "exec_over"}
        expected_packed = {(L_l, B) for L_l, bs in GPU_B_OPTS.items() for B in bs}
        expected_over = {(ml, B) for ml, bs in OVER_B_OPTS.items() for B in bs}
        assert ("encode_single", 8) in calls
        assert packed_shapes == expected_packed
        assert over_shapes == set()

        calls.clear()
        eng.warmup(prewarm_gpu_shapes=True, prewarm_over_shapes=True)
        assert {arg for op, arg in calls if op == "exec_over"} == expected_over
    finally:
        mx.eval = orig_eval
        mx.synchronize = orig_sync


def test_plain_matmul_and_force_fused_sdpa_gate() -> None:
    """Findings #2 & #4: Plain projections use mx.matmul(x, w.T); force_fused strictly M5+SWA+D256+384/512."""
    import ast
    import mlx.core as mx
    from fast_embeddinggemma_2 import gpu as emb2_gpu

    src = Path(emb2_gpu.__file__).read_text(encoding="utf-8")
    assert not hasattr(emb2_gpu, "_MPP_64")
    assert not hasattr(emb2_gpu, "_MPP_128")
    assert not hasattr(emb2_gpu, "_mpp_matmul")
    assert 'epilogue="none"' not in src
    assert "_MPP_GEGLU_64" in src and "_MPP_PG_64" in src
    tree = ast.parse(src, filename=str(emb2_gpu.__file__))
    trunk_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_forward_trunk")
    trunk_src = ast.get_source_segment(src, trunk_fn) or ""
    assert "mx.matmul(" in trunk_src and "_mpp_matmul(" not in trunk_src

    # Apple10 (M5) + SWA + D256 + S in (384, 512) + FP16 + macOS >= 26.2 -> True
    for seq_l in (384, 512):
        assert emb2_gpu.should_force_fused_sdpa(
            is_apple10=True, is_swa=True, hd=256, seq_l=seq_l, dtype=mx.float16,
            sdpa_supports_force_fused=True, macos_version=(26, 2),
        )
    # NEVER on Apple9 (M4 Pro), NEVER on D512, NEVER on other seq_l, NEVER on macOS < 26.2
    assert not emb2_gpu.should_force_fused_sdpa(is_apple10=False, is_swa=True, hd=256, seq_l=512)
    assert not emb2_gpu.should_force_fused_sdpa(is_apple10=True, is_swa=False, hd=512, seq_l=512)
    assert not emb2_gpu.should_force_fused_sdpa(is_apple10=True, is_swa=True, hd=512, seq_l=512)
    for bad_s in (64, 128, 192, 256, 320, 448, 640, 1024, 2048):
        assert not emb2_gpu.should_force_fused_sdpa(is_apple10=True, is_swa=True, hd=256, seq_l=bad_s)
    assert not emb2_gpu.should_force_fused_sdpa(
        is_apple10=True, is_swa=True, hd=256, seq_l=512, macos_version=(26, 1)
    )
    assert not emb2_gpu.should_force_fused_sdpa(
        is_apple10=True, is_swa=True, hd=256, seq_l=512, sdpa_supports_force_fused=False
    )

    assert emb2_gpu.should_use_fused_mpp(m_tot=128, metal4=True, macos_version=(26, 0), dtype=mx.float16)
    assert not emb2_gpu.should_use_fused_mpp(m_tot=128, metal4=True, macos_version=(26, 0), dtype=mx.bfloat16)
    assert emb2_gpu.should_use_fused_mpp(m_tot=128, metal4=True, macos_version=(26, 1), dtype=mx.bfloat16)
    assert not emb2_gpu.should_use_fused_mpp(m_tot=32, metal4=True, macos_version=(26, 6), dtype=mx.float16)
    assert not emb2_gpu.should_use_fused_mpp(m_tot=128, metal4=False, macos_version=(26, 6), dtype=mx.float16)


def test_seg_mean_pool_f32_direct_and_over_compiled_f32_precision(engine: FastEmbeddingGemma2) -> None:
    """Finding #5: Direct FP32 segment mean-pool (no half4 truncation) and FP32 sum in _over_compiled."""
    import mlx.core as mx
    from fast_embeddinggemma_2 import gpu as emb2_gpu

    src = Path(emb2_gpu.__file__).read_text(encoding="utf-8")
    assert "*(device float4*)outp = acc / float(n_pos);" in src
    assert "ho.astype(mx.float32)" in src
    assert "msk_mx.astype(mx.float16)" not in src

    # Test high-precision mean where half4 truncation would introduce ~5e-4 error
    ho_np = np.full((3, 512), 1.0, dtype=np.float16)
    ho_np[1, :] = np.float16(1.0009765625)  # 1 + 2^-10
    ho_np[2, :] = np.float16(1.0)
    pooled = emb2_gpu.mean_pool_segments(mx.array(ho_np), [0], [3])
    mx.eval(pooled)
    pooled_np = np.array(pooled)
    assert pooled_np.dtype == np.float32
    expected = ho_np.astype(np.float32).mean(axis=0, keepdims=True)
    # Direct float4 output matches FP32 reference to ~1e-6 (would fail at ~2e-4 if truncated through half4)
    assert np.allclose(pooled_np, expected, atol=1e-6, rtol=1e-6)

    # Exercise 2048-token _over_compiled path cleanly
    rng = np.random.default_rng(2048)
    long_tok = [engine.tokenizer.bos_id] + rng.integers(100, 30000, size=2046).tolist() + [engine.tokenizer.eos_id]
    v_2048 = engine.encode_corpus([long_tok], mode="gpu")
    assert v_2048.shape == (1, 768) and np.all(np.isfinite(v_2048))
    assert np.allclose(np.linalg.norm(v_2048, axis=1), 1.0, atol=1e-4)


def test_coreml_error_propagation_and_mlcomputeplan(engine: FastEmbeddingGemma2) -> None:
    """Findings #6 & #12: NSError propagation and MLComputePlan zero-CPU verification."""
    if not engine.ane_available or engine._coreml is None:
        pytest.skip("ANE unavailable")

    plans = engine.inspect_ane_plans(warn_on_cpu_fallback=False)
    assert len(plans) == len(engine.ane_handles)
    for seq_len, counts in plans.items():
        assert counts["ane_ops"] > 0, f"Expected ane_ops > 0 for S={seq_len}, got {counts}"
        assert counts["cpu_ops"] == 0, f"Unexpected CPU fallback for S={seq_len}: {counts}"

    # Loading nonexistent model must return NULL and record non-empty error string
    bad_h = engine._coreml.emb2_pk_load_units(b"/nonexistent/emb2_missing.mlmodelc", 64, 1, 0)
    assert not bad_h
    assert len(engine._coreml.emb2_last_error().decode("utf-8")) > 0
    assert engine._coreml.emb2_pk_wait_queue(None) < 0.0


def test_cpu_sme_opt_in_bounds_topology_and_explicit_cpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding #8: CPU-SME is opt-in, bounded on small/medium batches & LLM co-exec, and topology-aware."""
    from fast_embeddinggemma_2.engine import (
        HETERO3_CPU_MIN_DOCS,
        HETERO3_CPU_MIN_TOKS,
        MAX_CPU_SME_THREADS,
        _CPU_BUNDLES,
        _allow_hetero3_cpu_worker,
        _should_stop_hetero_cpu,
        probe_cpu_topology,
    )

    assert MAX_CPU_SME_THREADS == 2
    assert all(pd <= 2 for _, _, pd in _CPU_BUNDLES)
    assert _should_stop_hetero_cpu(1.0, 10, 1000, eff_mode="hetero3")
    assert not _should_stop_hetero_cpu(0.1, 10, 1000, eff_mode="hetero3")

    monkeypatch.delenv("FAST_SILICON_LLM_COEXEC", raising=False)
    assert not _allow_hetero3_cpu_worker(64, 8192, cpu_sme_available=True)
    assert _allow_hetero3_cpu_worker(HETERO3_CPU_MIN_DOCS, 8192, cpu_sme_available=True)
    assert _allow_hetero3_cpu_worker(32, HETERO3_CPU_MIN_TOKS, cpu_sme_available=True)
    assert not _allow_hetero3_cpu_worker(1000, 200000, cpu_sme_available=True, llm_coexec=True)
    monkeypatch.setenv("FAST_SILICON_LLM_COEXEC", "1")
    assert not _allow_hetero3_cpu_worker(1000, 200000, cpu_sme_available=True)
    monkeypatch.delenv("FAST_SILICON_LLM_COEXEC", raising=False)

    topo = probe_cpu_topology()
    assert topo["max_cpu_sme_threads"] == 2
    assert isinstance(topo["has_efficiency_cores"], bool)
    assert topo["perflevel0_name"]

    # Explicit opt-in CPU-SME engine loads CPU bundles and passes parity on <=512 docs
    paths = resolve_paths()
    if paths.coreml_dylib.exists() and (paths.ane_models_dir / "emb2_cpu_s256_mix_w8.mlmodelc").exists():
        with FastEmbeddingGemma2(warmup=False, enable_ane=False, enable_cpu_sme=True, default_mode="cpu") as cpu_eng:
            assert cpu_eng.cpu_sme_available is True
            assert cpu_eng._gpu is None
            ref = np.load(FIXTURE_64)
            lens = ref["lens"]
            sub_4 = [ref["ids"][i, : int(lens[i])].tolist() for i in range(4)]
            v_cpu = cpu_eng.encode_corpus(sub_4, mode="cpu")
            assert v_cpu.shape == (4, 768) and np.all(np.isfinite(v_cpu))
            assert float(np.min(np.sum(v_cpu * ref["v_f16"][:4], axis=1))) >= 0.9990
            assert cpu_eng._gpu is None


def test_ane_lazy_gpu_strict_query_and_wired_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding #7: ANE default_mode skips GPU init/warmup; strict ANE rejects >512; close() restores wired limit."""
    import mlx.core as mx
    from fast_embeddinggemma_2.engine import _apply_mlx_wired_limit, _restore_mlx_wired_limit

    wired_calls: list[int] = []
    monkeypatch.setattr(mx, "set_wired_limit", lambda val: (wired_calls.append(int(val)) or 54321))
    prev = _apply_mlx_wired_limit()
    assert prev == 54321 and len(wired_calls) == 1 and wired_calls[0] > 0
    _restore_mlx_wired_limit(prev)
    assert wired_calls[-1] == 54321

    paths = resolve_paths()
    if paths.coreml_dylib.exists() and paths.io_dylib.exists() and paths.ane_models_dir.exists():
        with FastEmbeddingGemma2(default_mode="ane", warmup=True, verify_ane_plan=False) as ane_eng:
            assert ane_eng._gpu is None
            v = ane_eng.encode_texts("pure neural engine query", mode="ane")
            assert v.shape == (768,) and np.all(np.isfinite(v))
            assert ane_eng._gpu is None
            # Strict mode="ane" refuses >512-token single text instead of silently invoking GPU
            long_text = " ".join(["engineering telemetry"] * 400)
            with pytest.raises(ValueError, match="Strict ANE mode"):
                ane_eng.encode_texts(long_text, mode="ane")
            assert ane_eng._gpu is None


def test_paths_resolve_home_requires_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Monorepo checkouts with pyproject.toml + native/ but no artifacts/ must fall through."""
    from fast_embeddinggemma_2.paths import resolve_home

    monkeypatch.delenv("FAST_EMBEDDINGGEMMA2_HOME", raising=False)
    monkeypatch.delenv("FAST_ARTIFACTS_ROOT", raising=False)

    fake_pkg = tmp_path / "fast-silicon" / "packages" / "fast-embeddinggemma-2"
    anchor = fake_pkg / "src" / "fast_embeddinggemma_2" / "paths.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("# stub\n", encoding="utf-8")
    (fake_pkg / "pyproject.toml").write_text("[project]\nname='fast-embeddinggemma-2'\n", encoding="utf-8")
    (fake_pkg / "native").mkdir()

    assert resolve_home(anchor_file=anchor) == (Path("~/.cache") / "fast-embeddinggemma-2").expanduser().resolve()

    ar = tmp_path / "shared_artifacts"
    (ar / "fast-embeddinggemma-2" / "artifacts").mkdir(parents=True)
    monkeypatch.setenv("FAST_ARTIFACTS_ROOT", str(ar))
    assert resolve_home(anchor_file=anchor) == (ar / "fast-embeddinggemma-2").resolve()

    (fake_pkg / "artifacts").mkdir()
    assert resolve_home(anchor_file=anchor) == fake_pkg.resolve()


def test_cli_alignment_rank_metrics_and_outlier_replay(engine: FastEmbeddingGemma2) -> None:
    """Findings #9 & #14: Aligned CLI modes/flags, rank verification, and hetero outlier replay."""
    from fast_embeddinggemma_2.cli import _rank_metrics, _replay_hetero_outliers, build_parser

    assert engine.default_mode == "hetero"
    assert engine.cpu_sme_available is False

    parser = build_parser()
    ns_embed = parser.parse_args(["embed", "hello"])
    assert ns_embed.mode == "hetero"
    ns_h3 = parser.parse_args(["embed", "--mode", "hetero3", "hello"])
    assert ns_h3.mode == "hetero3"
    ns_bench = parser.parse_args(["bench", "--runs", "2", "--check-ps", "--document-prefix"])
    assert ns_bench.runs == 2 and ns_bench.check_ps is True and ns_bench.document_prefix is True
    ns_verify = parser.parse_args(["verify", "--strict"])
    assert ns_verify.strict is True

    ref = np.load(FIXTURE_64)
    lens = ref["lens"]
    sub_lists = [ref["ids"][i, : int(lens[i])].tolist() for i in range(len(lens))]
    v_gpu = engine.encode_corpus(sub_lists, mode="gpu")
    v_het = engine.encode_corpus(sub_lists, mode="hetero")
    rm = _rank_metrics(v_gpu, ref["v_f16"], top_k=5)
    assert rm["top1_match"] == 1.0 and rm["topk_overlap"] >= 0.95 and rm["spearman_mean"] >= 0.999

    rep = _replay_hetero_outliers(engine, sub_lists, v_het, v_gpu)
    assert rep["replay_fixed_gpu_min_cos"] >= 0.9998
    assert len(rep["outlier_idxs"]) == 8


def test_nchw_output_transpose_and_build_ane_nchw_cpu() -> None:
    """Verify CoreML ANE bundle builds keep NCHW out_h and NEON untiling is bit-exact for multiples and tails."""
    import ast
    import ctypes

    pkg_root = Path(__file__).resolve().parents[1]
    build_src = (pkg_root / "scripts" / "build_ane_models.py").read_text(encoding="utf-8")
    tree = ast.parse(build_src)
    ane_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_ane_packed_w8")
    ane_seg = ast.get_source_segment(build_src, ane_fn) or ""
    assert 'mb.identity(x=normed, name="out_h")' in ane_seg
    assert "oh_r" not in ane_seg

    dylib_path = pkg_root / "build" / "libemb2_coreml_runner.dylib"
    if not dylib_path.is_file():
        dylib_path = resolve_paths().coreml_dylib
    if not dylib_path.is_file():
        pytest.skip("Native dylib not built yet")
    lib = ctypes.CDLL(str(dylib_path))
    lib.emb2_transpose_512xS_to_Sx512.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.emb2_transpose_512xS_to_Sx512.restype = None

    rng = np.random.default_rng(7)
    for S in (19, 32, 64, 131):
        src = rng.integers(0, 65536, size=(512, S), dtype=np.uint16)
        dst = np.empty((S, 512), dtype=np.uint16)
        lib.emb2_transpose_512xS_to_Sx512(S, src.ctypes.data, dst.ctypes.data)
        np.testing.assert_array_equal(dst, src.T)

