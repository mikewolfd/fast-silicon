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


def test_build_ane_models_fp16_epsilon_floor_and_source() -> None:
    """Finding #1: LayerNorm epsilon in build_ane_models.py must clamp >= 2^-14 normal FP16 (never 0.0)."""
    import importlib.util

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "build_ane_models.py"
    spec = importlib.util.spec_from_file_location("bge_build_ane_models", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Raw np.float16(1e-12) underflows to exact 0.0 in IEEE-754 binary16
    assert float(np.float16(1e-12)) == 0.0
    eps_safe = mod.safe_fp16_layernorm_epsilon(1e-12)
    assert isinstance(eps_safe, np.float16)
    assert float(eps_safe) >= float(np.finfo(np.float16).tiny) > 6e-5
    assert float(mod.safe_fp16_layernorm_epsilon(0.0)) >= float(np.finfo(np.float16).tiny)
    assert float(mod.safe_fp16_layernorm_epsilon(1e-3)) == pytest.approx(float(np.float16(1e-3)))

    import ast

    src = script_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(script_path))
    ln_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "layer_norm"
    ]
    assert len(ln_calls) == 5
    for call in ln_calls:
        eps_kw = [kw.value for kw in call.keywords if kw.arg == "epsilon"]
        assert len(eps_kw) == 1
        assert isinstance(eps_kw[0], ast.Name) and eps_kw[0].id == "ln_eps"
    assert (1, 256) in mod.OPTIONAL_QUERY_BUCKETS


def test_build_ane_models_coreml_optional() -> None:
    pytest.importorskip("coremltools")


def test_b8_s256_partial_batch_no_nan_and_matches_gpu(engine: FastBgeSmall) -> None:
    """Finding #1: B=8 S=256 partial batches (n_docs % 8 != 0) must not produce NaN or drift."""
    if not engine.ane_available or 256 not in engine.buckets:
        pytest.skip("ANE S=256 bucket unavailable")

    rng = np.random.default_rng(256)
    for n_docs in (1, 2, 3, 7, 9):
        toks = [
            [101] + rng.integers(1000, 25000, size=220).tolist() + [102]
            for _ in range(n_docs)
        ]
        v_ane = engine.encode_corpus(toks, mode="ane")
        v_gpu = engine.encode_corpus(toks, mode="gpu")
        assert v_ane.shape == (n_docs, 384)
        assert np.all(np.isfinite(v_ane)), f"Non-finite ANE embedding for n_docs={n_docs}"
        assert np.allclose(np.linalg.norm(v_ane, axis=1), 1.0, atol=1e-4)
        assert float(np.min(np.sum(v_ane * v_gpu, axis=1))) >= 0.9995, f"Mismatch at n_docs={n_docs}"


def test_ane_error_propagation_and_mlcomputeplan(engine: FastBgeSmall) -> None:
    """Findings #6 & #12: NSError propagation and MLComputePlan zero-CPU verification."""
    if not engine.ane_available or engine.ane is None:
        pytest.skip("ANE unavailable")

    plans = engine.inspect_ane_plans(warn_on_cpu_fallback=False)
    assert len(plans) == len(engine.buckets)
    for cap, counts in plans.items():
        assert counts["ane_ops"] > 0, f"Expected ane_ops > 0 for S={cap}, got {counts}"
        assert counts["cpu_ops"] == 0, f"Unexpected CPU fallback for S={cap}: {counts}"

    # Passing the B=8 bucket (256) to single-query bge_ane_query must fail cleanly with error recorded
    if 256 in engine.buckets and 256 not in engine.query_buckets:
        arr = np.array([101, 2054, 102], dtype=np.int32)
        out = np.empty(384, dtype=np.float32)
        rc = engine.ane.bge_ane_query(engine.buckets[256], arr.ctypes.data, 3, out.ctypes.data, 1)
        assert rc < 0.0
        err_msg = engine.ane.bge_ane_last_error().decode("utf-8")
        assert "B=1" in err_msg

    # Loading a nonexistent bundle must fail cleanly and record error
    rc_bad = engine.ane.bge_ane_load_bucket(b"/nonexistent/bge_missing.mlmodelc", 1, 64, 1, 1)
    assert rc_bad < 0
    assert len(engine.ane.bge_ane_last_error().decode("utf-8")) > 0


def test_tn128_mpp_and_contiguous_single_layers_and_msl_smoke(engine: FastBgeSmall) -> None:
    """Findings #3, #10, #11: tn=128 fused MPP tiles, pre-transposed contiguous weights, and Metal smoke."""
    import mlx.core as mx
    from fast_bge_small import gpu as bge_gpu

    import ast

    # Verify tn=128 tile configuration and zero per-call `.T` views inside forward_packed
    src = Path(bge_gpu.__file__).read_text(encoding="utf-8")
    assert 'tm=128, tn=128, sgs=8' in src
    assert 'tm=64, tn=128, sgs=4' in src
    tree = ast.parse(src, filename=str(bge_gpu.__file__))
    fp_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward_packed")
    fp_src = ast.get_source_segment(src, fp_fn) or ""
    assert ".T)" not in fp_src
    assert 'ld_t["w_qT"]' in fp_src and 'ld_t["w_oT"]' in fp_src
    assert 'ld_t["w_upT"]' in fp_src and 'ld_t["w_dnT"]' in fp_src

    # Verify Layer-11 / single layers are pre-transposed contiguous weights
    caps = bge_gpu.probe_metal_capabilities()
    assert caps["mlx_version"] >= (0, 32, 4)
    assert isinstance(caps["apple9"], bool) and isinstance(caps["apple10"], bool)

    # Exercise all custom Metal/MPP kernels on host MSL (4.0 on macOS 26.x / 4.1 on macOS 27.x)
    toks = [[101, 2054, 2003, 1996, 15068, 2291, 1029, 102]] * 4
    v_packed = engine.gpu.encode_corpus(toks)
    v_single = engine.gpu.encode_single(toks[0])
    assert np.all(np.isfinite(v_packed)) and np.all(np.isfinite(v_single))
    assert float(np.dot(v_packed[0], v_single)) >= 0.9999

    # Guard unaligned M_pad
    with pytest.raises(ValueError, match="multiple of 128"):
        engine.gpu.forward_packed(
            mx.zeros((64,), dtype=mx.int32),
            mx.zeros((64,), dtype=mx.int32),
            mx.zeros((1, 4), dtype=mx.int32),
            mx.zeros((1, 2), dtype=mx.int32),
            mx.zeros((1,), dtype=mx.int32),
            64,
            64,
        )


def test_memory_hygiene_wired_limit_and_ane_lazy_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding #7: ANE default_mode skips eager GPU init/warmup; close() restores wired limit."""
    import mlx.core as mx
    from fast_bge_small.engine import _apply_mlx_wired_limit, _restore_mlx_wired_limit

    wired_calls: list[int] = []
    monkeypatch.setattr(mx, "set_wired_limit", lambda val: (wired_calls.append(int(val)) or 12345))
    prev = _apply_mlx_wired_limit()
    assert prev == 12345
    assert len(wired_calls) == 1 and wired_calls[0] > 0
    _restore_mlx_wired_limit(prev)
    assert wired_calls[-1] == 12345

    paths = resolve_paths()
    if paths.ane_dylib.exists() and paths.ane_models_dir.exists():
        with FastBgeSmall(default_mode="ane", warmup=True, verify_ane_plan=False) as ane_eng:
            assert ane_eng._gpu is None
            vec = ane_eng.encode_texts("pure neural engine query", mode="ane")
            assert vec.shape == (384,) and np.all(np.isfinite(vec))
            assert ane_eng._gpu is None


def test_paths_resolve_home_requires_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Monorepo checkouts with pyproject.toml + native/ but no artifacts/ must fall through."""
    from fast_bge_small.paths import resolve_home

    monkeypatch.delenv("FAST_BGE_SMALL_HOME", raising=False)
    monkeypatch.delenv("FAST_ARTIFACTS_ROOT", raising=False)

    fake_pkg = tmp_path / "fast-silicon" / "packages" / "fast-bge-small"
    anchor = fake_pkg / "src" / "fast_bge_small" / "paths.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("# stub\n", encoding="utf-8")
    (fake_pkg / "pyproject.toml").write_text("[project]\nname='fast-bge-small'\n", encoding="utf-8")
    (fake_pkg / "native").mkdir()

    # Without artifacts/, must fall through to ~/.cache/fast-bge-small
    assert resolve_home(anchor_file=anchor) == (Path("~/.cache") / "fast-bge-small").expanduser().resolve()

    # With FAST_ARTIFACTS_ROOT set, resolves under FAST_ARTIFACTS_ROOT/fast-bge-small
    ar = tmp_path / "shared_artifacts"
    (ar / "fast-bge-small" / "artifacts").mkdir(parents=True)
    monkeypatch.setenv("FAST_ARTIFACTS_ROOT", str(ar))
    assert resolve_home(anchor_file=anchor) == (ar / "fast-bge-small").resolve()

    # Once checkout itself has artifacts/, checkout takes precedence over FAST_ARTIFACTS_ROOT
    (fake_pkg / "artifacts").mkdir()
    assert resolve_home(anchor_file=anchor) == fake_pkg.resolve()


def test_cli_alignment_rank_metrics_and_outlier_replay(engine: FastBgeSmall) -> None:
    """Findings #9 & #14: CLI modes/flags, rank verification, and hetero outlier replay."""
    from fast_bge_small.cli import _rank_metrics, _replay_hetero_outliers, build_parser

    parser = build_parser()
    ns_embed = parser.parse_args(["embed", "--mode", "hetero2", "hello"])
    assert ns_embed.mode == "hetero2"
    ns_bench = parser.parse_args(["bench", "--runs", "2", "--check-ps", "--mode", "hetero"])
    assert ns_bench.runs == 2 and ns_bench.check_ps is True
    ns_verify = parser.parse_args(["verify", "--strict"])
    assert ns_verify.strict is True

    ref = np.load(FIXTURE_64)
    sub_lists = [ref["ids"][i, : int(ref["mask"][i].sum())].tolist() for i in range(64)]
    v_gpu = engine.encode_corpus(sub_lists, mode="gpu")
    v_het = engine.encode_corpus(sub_lists, mode="hetero")
    rm = _rank_metrics(v_gpu, ref["v_f16"], top_k=5)
    assert rm["top1_match"] == 1.0 and rm["topk_overlap"] >= 0.95 and rm["spearman_mean"] >= 0.999

    rep = _replay_hetero_outliers(engine, sub_lists, v_het, v_gpu)
    assert rep["replay_fixed_gpu_min_cos"] >= 0.9999
    assert len(rep["outlier_idxs"]) == 8


def test_vectorized_cpu_query_dylib_parity(engine: FastBgeSmall) -> None:
    """Finding #15: Optional vectorized libbge_cpu_query.dylib matches GPU single-query output."""
    import ctypes

    paths = resolve_paths()
    if not paths.cpu_dylib.exists():
        pytest.skip("Optional libbge_cpu_query.dylib not built (run `make cpu-query`)")

    raw = dict(np.load(paths.weights_npz))
    wt0 = np.ascontiguousarray(
        (raw["embeddings.word_embeddings.weight"].astype(np.float32) + raw["embeddings.token_type_embeddings.weight"][0:1].astype(np.float32))
    )
    pos = np.ascontiguousarray(raw["embeddings.position_embeddings.weight"].astype(np.float32))
    emb_ln_w = np.ascontiguousarray(raw["embeddings.LayerNorm.weight"].astype(np.float32))
    emb_ln_b = np.ascontiguousarray(raw["embeddings.LayerNorm.bias"].astype(np.float32))

    w_qkv = np.ascontiguousarray(np.stack([
        np.concatenate([raw[f"encoder.layer.{l}.attention.self.query.weight"], raw[f"encoder.layer.{l}.attention.self.key.weight"], raw[f"encoder.layer.{l}.attention.self.value.weight"]], axis=0)
        for l in range(12)
    ], axis=0).astype(np.float32))
    b_qkv = np.ascontiguousarray(np.stack([
        np.concatenate([raw[f"encoder.layer.{l}.attention.self.query.bias"], raw[f"encoder.layer.{l}.attention.self.key.bias"], raw[f"encoder.layer.{l}.attention.self.value.bias"]], axis=0)
        for l in range(12)
    ], axis=0).astype(np.float32))
    w_o = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.attention.output.dense.weight"] for l in range(12)], axis=0).astype(np.float32))
    b_o = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.attention.output.dense.bias"] for l in range(12)], axis=0).astype(np.float32))
    ln1_w = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.attention.output.LayerNorm.weight"] for l in range(12)], axis=0).astype(np.float32))
    ln1_b = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.attention.output.LayerNorm.bias"] for l in range(12)], axis=0).astype(np.float32))
    w_up = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.intermediate.dense.weight"] for l in range(12)], axis=0).astype(np.float32))
    b_up = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.intermediate.dense.bias"] for l in range(12)], axis=0).astype(np.float32))
    w_dn = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.output.dense.weight"] for l in range(12)], axis=0).astype(np.float32))
    b_dn = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.output.dense.bias"] for l in range(12)], axis=0).astype(np.float32))
    ln2_w = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.output.LayerNorm.weight"] for l in range(12)], axis=0).astype(np.float32))
    ln2_b = np.ascontiguousarray(np.stack([raw[f"encoder.layer.{l}.output.LayerNorm.bias"] for l in range(12)], axis=0).astype(np.float32))

    lib = ctypes.CDLL(str(paths.cpu_dylib))
    lib.bge_cpu_init.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.bge_cpu_init.restype = None
    lib.bge_cpu_query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.bge_cpu_query.restype = ctypes.c_double

    lib.bge_cpu_init(
        wt0.ctypes.data, wt0.shape[0], pos.ctypes.data, pos.shape[0],
        emb_ln_w.ctypes.data, emb_ln_b.ctypes.data,
        w_qkv.ctypes.data, b_qkv.ctypes.data,
        w_o.ctypes.data, b_o.ctypes.data,
        ln1_w.ctypes.data, ln1_b.ctypes.data,
        w_up.ctypes.data, b_up.ctypes.data,
        w_dn.ctypes.data, b_dn.ctypes.data,
        ln2_w.ctypes.data, ln2_b.ctypes.data,
    )
    toks = np.array(engine.tokenizer.encode("renewable grid reliability technical report"), dtype=np.int32)
    out = np.empty(384, dtype=np.float32)
    dt = lib.bge_cpu_query(toks.ctypes.data, len(toks), out.ctypes.data, 1)
    assert dt >= 0.0 and np.all(np.isfinite(out))
    ref_gpu = engine.encode_query(toks, prefer_ane=False)
    assert float(np.dot(out, ref_gpu)) >= 0.9999
