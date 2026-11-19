from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_bge_small import FastBgeSmall, resolve_paths
from fast_bge_small.parquet_pipeline import embed_parquet, slice_ocr_pages_utf8
from fast_bge_small.server import FastBgeServer

FIXTURE_64 = Path(__file__).resolve().parent / "fixtures" / "ref_vecs_64.npz"


@pytest.fixture(scope="module")
def engine() -> FastBgeSmall:
    return FastBgeSmall(warmup=True)


def test_ref_64_parity_all_modes(engine: FastBgeSmall) -> None:
    ref = np.load(FIXTURE_64)
    ids_64 = ref["ids"]
    mask_64 = ref["mask"]
    v_f16 = ref["v_f16"]
    v_bf16 = ref["v_stock_bf16"]
    sub_lists = [ids_64[i, : int(mask_64[i].sum())].tolist() for i in range(64)]

    v_gpu = engine.encode_corpus(sub_lists, mode="gpu")
    v_het = engine.encode_corpus(sub_lists, mode="hetero")
    assert v_gpu.shape == (64, 384) and v_gpu.dtype == np.float32
    assert v_het.shape == (64, 384) and v_het.dtype == np.float32

    assert np.allclose(np.linalg.norm(v_gpu, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(v_het, axis=1), 1.0, atol=1e-4)

    assert float(np.min(np.sum(v_gpu * v_f16, axis=1))) >= 0.9999
    assert float(np.min(np.sum(v_gpu * v_bf16, axis=1))) >= 0.9998
    assert float(np.min(np.sum(v_het * v_f16, axis=1))) >= 0.9999

    if engine.ane_available:
        v_ane = engine.encode_corpus(sub_lists, mode="ane")
        assert float(np.min(np.sum(v_ane * v_gpu, axis=1))) >= 0.9998


def test_query_buckets_parity_and_edge_cases(engine: FastBgeSmall) -> None:
    assert engine.encode_texts([]).shape == (0, 384)
    assert engine.encode_corpus([]).shape == (0, 384)

    v_empty = engine.encode_texts("")
    assert v_empty.shape == (384,)
    assert abs(float(np.linalg.norm(v_empty)) - 1.0) < 1e-4
    v_empty_gpu = engine.encode_query([], prefer_ane=False)
    assert float(np.dot(v_empty, v_empty_gpu)) >= 0.9998

    assert engine.encode_texts(["single item"]).shape == (1, 384)

    rng = np.random.default_rng(7)
    for sl in (16, 48, 100, 180, 220, 300, 360, 420, 500):
        toks = [101] + rng.integers(1000, 25000, size=sl - 2).tolist() + [102]
        q_ane = engine.encode_query(toks, prefer_ane=True)
        q_gpu = engine.encode_query(toks, prefer_ane=False)
        assert float(np.dot(q_ane, q_gpu)) >= 0.9995, f"Mismatch at len={sl}"


def test_tokenizer_and_chunker(engine: FastBgeSmall) -> None:
    short = "Pacific Transmission Consortium Seasonal Inverter & Synchronous Reserve Assessment."
    short_ids = engine.tokenizer.encode(short)
    chunks = engine.tokenizer.chunk_text(short, max_body_tokens=510, overlap=51)
    assert len(chunks) == 1
    assert chunks[0][0] == short_ids
    assert chunks[0][1] == 0 and chunks[0][2] == len(short)

    long_text = " ".join(["distributed energy storage substation inverter telemetry engineering"] * 130)
    long_chunks = engine.tokenizer.chunk_text(long_text, max_body_tokens=510, overlap=51)
    assert len(long_chunks) >= 2
    for ids, c0, c1 in long_chunks:
        assert len(ids) <= 512
        assert ids[0] == engine.tokenizer.cls_id
        assert ids[-1] == engine.tokenizer.sep_id
        assert 0 <= c0 < c1 <= len(long_text)


def test_native_bge_ane_cli(engine: FastBgeSmall) -> None:
    paths = resolve_paths()
    assert paths.ane_cli.exists(), f"Missing native CLI at {paths.ane_cli}"

    text = "What is the federal budget deficit?"
    proc = subprocess.run(
        [str(paths.ane_cli), "--text", text, "--query-prefix", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert len(payload["embedding"]) == 384
    vec_cli = np.asarray(payload["embedding"], dtype=np.float32)
    vec_py = engine.encode_texts(text, is_query=True, add_query_prefix=True)
    assert float(np.dot(vec_cli, vec_py)) >= 0.9995


def test_slice_ocr_pages_utf8() -> None:
    # Multi-byte em-dash triggers UTF-8 byte offset != char offset
    p1 = "Intro — header page."
    p2 = "OCR scanned exhibit text."
    p3 = "Native appendix."
    full = p1 + p2 + p3
    b1 = len(p1.encode("utf-8"))
    b2 = len((p1 + p2).encode("utf-8"))
    offsets = [0, b1, b2]
    assert slice_ocr_pages_utf8(full, offsets, [2]) == p2
    assert slice_ocr_pages_utf8(full, offsets, [1, 3]) == f"{p1}\n\n{p3}"


def test_parquet_pipeline_roundtrip_and_sharded_resume(engine: FastBgeSmall, tmp_path: Path) -> None:
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
    assert tbl.schema.field("vector").type == pa.list_(pa.float32(), 384)
    vecs = np.asarray(tbl["vector"].to_pylist(), dtype=np.float32)
    expected = engine.encode_texts([r["body"] for r in rows if r["is_ocr"]])
    assert float(np.min(np.sum(vecs * expected, axis=1))) >= 0.9999

    # Test sharded directory + float16 + chunking + resume
    out_dir = tmp_path / "shards"
    res2 = embed_parquet(
        engine,
        in_path,
        out_dir,
        text_col="body",
        dtype="float16",
        shard_rows=4,
        chunk=True,
        drop_text_col=True,
        progress_every=0,
    )
    assert res2.rows_emitted == 12
    parts = sorted(out_dir.glob("part-*.parquet"))
    assert len(parts) == 3
    first_part = pq.read_table(parts[0])
    assert "body" not in first_part.column_names
    assert first_part.schema.field("vector").type == pa.list_(pa.float16(), 384)

    # Verify resume preserves untouched shards and repairs a missing shard
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
        drop_text_col=True,
        resume=True,
        progress_every=0,
    )
    assert parts[0].stat().st_mtime_ns == mtime_0
    assert parts[1].exists() and pq.read_table(parts[1]).num_rows == 4


def test_http_server_smoke(engine: FastBgeSmall) -> None:
    texts = ["grid reliability assessment FY2025", "watershed dissolved oxygen lab survey"]
    expected = engine.encode_texts(texts)

    with FastBgeServer(engine, host="127.0.0.1", port=0) as srv:
        base_url = f"http://{srv.host}:{srv.port}"
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            assert resp.status == 200
            health = json.loads(resp.read().decode("utf-8"))
            assert health["status"] == "ok" and health["dim"] == 384

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
            assert float(np.min(np.sum(got * expected, axis=1))) >= 0.9999
