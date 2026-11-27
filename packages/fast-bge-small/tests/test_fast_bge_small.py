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


def test_mpp_capability_gating_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-P1.2: Metal 4 MPP must check probe_metal_capabilities() before instantiating kernels."""
    from fast_bge_small import gpu as bge_gpu

    assert hasattr(bge_gpu, "should_use_fused_mpp")
    assert bge_gpu.should_use_fused_mpp({"metal4": False, "macos_version": (26, 6)}) is False
    assert bge_gpu.should_use_fused_mpp({"metal4": True, "macos_version": (15, 5)}) is False
    assert bge_gpu.should_use_fused_mpp({"metal4": True, "macos_version": (26, 0)}) is True

    # Simulate reloading/initializing MPP kernels on a non-Metal-4 host: must never call make_mpp_matmul_kernel.
    called = False

    def _boom(*_args, **_kwargs):
        nonlocal called
        called = True
        raise RuntimeError("make_mpp_matmul_kernel must not run when metal4=False")

    monkeypatch.setattr(bge_gpu, "make_mpp_matmul_kernel", _boom)
    kernels = bge_gpu._build_mpp_kernels({"metal4": False, "macos_version": (15, 4)})
    assert called is False
    assert kernels == (None, None, None)


def test_ffd_length_sorted_packing_cpu() -> None:
    """M-P1.2: Sequence packing must length-sort descending + FFD fill while preserving orig_idx."""
    from fast_bge_small.gpu import _plan_packed_batches

    # Arrival order would split into 3 bins: [300, 100] (400), [300] (300), [100] (100) at max_tokens=400;
    # length-sorted FFD packs [300, 100] (400) and [300, 100] (400) into 2 exact bins!
    seqs = [
        [101] + [2000] * 98 + [102],   # idx 0: len 100
        [101] + [2001] * 298 + [102],  # idx 1: len 300
        [101] + [2002] * 98 + [102],   # idx 2: len 100
        [101] + [2003] * 298 + [102],  # idx 3: len 300
    ]
    bins = _plan_packed_batches(seqs, max_tokens=400)
    assert len(bins) == 2
    seen_orig = sorted(oi for b in bins for oi, _ in b)
    assert seen_orig == [0, 1, 2, 3]
    for b in bins:
        lens = [len(ids) for _, ids in b]
        assert lens == sorted(lens, reverse=True)
        assert sum(lens) == 400


def test_clean_mmap_host_tables_cpu(tmp_path: Path) -> None:
    """M-P1.2: Host wt0/pos tables materialize clean read-only memmaps shared across ANE and GPU."""
    from fast_bge_small.paths import load_or_mmap_host_tables

    rng = np.random.default_rng(42)
    word = rng.standard_normal((64, 384)).astype(np.float16)
    ttype = rng.standard_normal((2, 384)).astype(np.float16)
    pos = rng.standard_normal((512, 384)).astype(np.float16)
    npz_path = tmp_path / "bge_small_f16.npz"
    np.savez(
        npz_path,
        **{
            "embeddings.word_embeddings.weight": word,
            "embeddings.token_type_embeddings.weight": ttype,
            "embeddings.position_embeddings.weight": pos,
        },
    )

    wt0_mm, pos_mm = load_or_mmap_host_tables(npz_path)
    assert not (tmp_path / "bge_wt0_f16.bin").exists()
    assert not (tmp_path / "bge_pos_f16.bin").exists()
    assert isinstance(wt0_mm, np.memmap)
    assert isinstance(pos_mm, np.memmap)
    assert wt0_mm.flags.writeable is False
    assert pos_mm.flags.writeable is False
    expected_wt0 = (word + ttype[0:1]).astype(np.float16)
    np.testing.assert_array_equal(np.asarray(wt0_mm), expected_wt0)
    np.testing.assert_array_equal(np.asarray(pos_mm), pos)


def test_w8a16_multifunction_builder_and_mask_literal_cpu() -> None:
    """M-P1.2: ANE builder defaults include b1_s256, W8A16 symmetric quant, multifunction, and -40000.0 mask."""
    import importlib.util

    pkg_root = Path(__file__).resolve().parent.parent
    script_path = pkg_root / "scripts" / "build_ane_models.py"
    spec = importlib.util.spec_from_file_location("bge_build_ane_models_v2", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert (1, 256) in mod.PRODUCTION_BUCKETS
    assert hasattr(mod, "build_multifunction_bundle")
    assert mod.bge_function_name(1, 256) == "bge_b1_s256"
    assert mod.bge_function_name(8, 256) == "bge_b8_s256"

    src = script_path.read_text(encoding="utf-8")
    assert "linear_symmetric" in src
    assert "save_multifunction" in src

    mm_src = (pkg_root / "native" / "bge_ane_runner.mm").read_text(encoding="utf-8")
    assert "-40000.0f" in mm_src
    assert "fp16_bits(-10000.0f)" not in mm_src
    assert "@available(macOS 15.0, *)" in mm_src
    assert 'std::strcmp(function_name, "main") != 0' in mm_src
    assert "SemaphoreGuard" in mm_src
    assert "bge_ane_free_bucket" in mm_src
    assert "os.symlink(out_mlmodelc.name, link_p)" in src


def test_single_layers_dropped_and_borrowed_tables(engine: FastBgeSmall) -> None:
    """M-P1.2: GPU keeps transposed weights only for Layer 11 (-42 MB) and preserves unordered corpus order."""
    assert set(engine.gpu._single_layers.keys()) == {11}

    rng = np.random.default_rng(19)
    # Interleave short and long sequences to exercise length-sorted FFD + orig_idx scatter in both gpu and hetero
    toks = [
        [101] + rng.integers(1000, 20000, size=int(sl)).tolist() + [102]
        for sl in (22, 310, 14, 480, 95, 240, 18, 390, 60, 254, 12, 410)
    ]
    v_singles = np.stack([engine.encode_query(t, prefer_ane=False) for t in toks], axis=0)
    v_gpu = engine.encode_corpus(toks, mode="gpu", gpu_max_tokens=1024)
    assert float(np.min(np.sum(v_gpu * v_singles, axis=1))) >= 0.9998
    if engine.ane_available:
        v_het = engine.encode_corpus(toks, mode="hetero", gpu_max_tokens=1024, ane_chunk_docs=3)
        assert float(np.min(np.sum(v_het * v_singles, axis=1))) >= 0.9995

    # Out-of-bounds token IDs (< 0 or >= vocab_size) clamp safely to [UNK]=100 on GPU single & packed paths
    ref_unk = engine.encode_query([101, 100, 100, 102], prefer_ane=False)
    oob_single = engine.encode_query([101, -7, 999999, 102], prefer_ane=False)
    assert float(np.dot(ref_unk, oob_single)) >= 0.9999
    oob_corpus = engine.encode_corpus([[101, -7, 999999, 102], [101, 100, 100, 102]], mode="gpu")
    assert float(np.dot(oob_corpus[0], oob_corpus[1])) >= 0.9999

    # Lifecycle test: closing a temporary FastBgeSmall instance frees its buckets & mmaps without disturbing `engine`
    with FastBgeSmall(warmup=False, verify_ane_plan=False) as temp_eng:
        _ = temp_eng.encode_query([101, 2054, 2003, 102], prefer_ane=True)
        owned_before = set(temp_eng._owned_bucket_ids)
    assert len(temp_eng._owned_bucket_ids) == 0
    assert temp_eng._wt0_f16 is None and temp_eng._pos_f16 is None
    if engine.ane_available:
        assert owned_before.isdisjoint(engine._owned_bucket_ids)
        q_after = engine.encode_query([101, 2054, 2003, 102], prefer_ane=True)
        assert q_after.shape == (384,) and np.all(np.isfinite(q_after))


def test_lazy_neural_init_skipped_for_metal_default_mode_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`enable_ane=None` skips CoreML ANE bundle init for `default_mode='gpu'` unless `enable_ane=True`."""
    import inspect
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    import fast_bge_small.engine as bge_eng

    sig = inspect.signature(FastBgeSmall.__init__)
    assert sig.parameters["enable_ane"].default is None

    home = tmp_path / "bge_home"
    (home / "artifacts" / "weights").mkdir(parents=True)
    (home / "artifacts" / "tokenizer").mkdir(parents=True)
    (home / "artifacts" / "ane_models").mkdir(parents=True)
    (home / "build").mkdir(parents=True)
    (home / "artifacts" / "weights" / "bge_small_f16.npz").write_bytes(b"stub")
    (home / "build" / "libbge_ane_runner.dylib").write_bytes(b"stub")

    raw_tok = Tokenizer(WordLevel(vocab={"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "hello": 200}, unk_token="[UNK]"))
    raw_tok.pre_tokenizer = WhitespaceSplit()
    raw_tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 101), ("[SEP]", 102)],
    )
    raw_tok.save(str(home / "artifacts" / "tokenizer" / "tokenizer.json"))

    class _StubGPU:
        def __init__(self, weights_path: Path, *, wt0_f16: object = None, pos_f16: object = None) -> None:
            del weights_path, wt0_f16, pos_f16

        def encode_single(self, token_ids: object) -> np.ndarray:
            del token_ids
            v = np.ones((384,), dtype=np.float32)
            v /= np.linalg.norm(v)
            return v

        def encode_corpus(self, token_lists: list, *, max_tokens: int = 8192) -> np.ndarray:
            del max_tokens
            out = np.ones((len(token_lists), 384), dtype=np.float32)
            out /= np.linalg.norm(out, axis=-1, keepdims=True)
            return out

    ane_calls: list[bool] = []

    def _spy_init_ane(self: FastBgeSmall, *, verify_plan: bool = True) -> None:
        ane_calls.append(verify_plan)
        self.ane = object()  # type: ignore[assignment]
        self.ane_available = True

    monkeypatch.setattr(bge_eng, "load_or_mmap_host_tables", lambda *_a, **_kw: (np.zeros((1, 384), dtype=np.float16), np.zeros((1, 384), dtype=np.float16)))
    monkeypatch.setattr(bge_eng, "FastBgeSmallGPU", _StubGPU)
    monkeypatch.setattr(bge_eng, "set_mlx_cache_limit", lambda *_a, **_kw: None)
    monkeypatch.setattr(bge_eng, "_apply_mlx_wired_limit", lambda: None)
    monkeypatch.setattr(bge_eng, "_restore_mlx_wired_limit", lambda *_a, **_kw: None)
    monkeypatch.setattr(bge_eng, "_maybe_clear_mlx_arena", lambda *_a, **_kw: None)
    monkeypatch.setattr(FastBgeSmall, "_init_ane", _spy_init_ane)

    # 1. default_mode="gpu" skips _init_ane completely
    with FastBgeSmall(home=home, default_mode="gpu", warmup=False) as eng_gpu:
        assert ane_calls == []
        assert eng_gpu.ane_available is False
        assert eng_gpu.default_mode == "gpu"
        with pytest.raises(RuntimeError, match="not initialized"):
            eng_gpu.encode_texts("hello", mode="ane")
        with pytest.raises(RuntimeError, match="not initialized"):
            eng_gpu.encode_corpus([[101, 200, 102]], mode="ane")
        assert eng_gpu.encode_texts("hello", mode="hetero").shape == (384,)
        assert eng_gpu.encode_corpus([[101, 200, 102], [101, 200, 200, 102]], mode="hetero2").shape == (2, 384)

    # 2. Explicit enable_ane=True initializes ANE even when default_mode="gpu"
    ane_calls.clear()
    with FastBgeSmall(home=home, default_mode="gpu", enable_ane=True, warmup=False) as eng_gpu_ane:
        assert ane_calls == [True]
        assert eng_gpu_ane.ane_available is True
        assert eng_gpu_ane.default_mode == "gpu"

    # 3. Non-GPU default modes initialize ANE normally by default
    for non_gpu_mode, expected_resolved in (("ane", "ane"), ("hetero", "hetero"), ("hetero2", "hetero")):
        ane_calls.clear()
        with FastBgeSmall(home=home, default_mode=non_gpu_mode, warmup=False) as eng_m:
            assert ane_calls == [True]
            assert eng_m.ane_available is True
            assert eng_m.default_mode == expected_resolved

    # 4. Explicit enable_ane=False skips ANE in hetero and errors clearly in ane mode
    ane_calls.clear()
    with FastBgeSmall(home=home, default_mode="hetero", enable_ane=False, warmup=False) as eng_no_ane:
        assert ane_calls == []
        assert eng_no_ane.ane_available is False
        assert eng_no_ane.default_mode == "gpu"

    with pytest.raises(RuntimeError, match="enable_ane=False"):
        FastBgeSmall(home=home, default_mode="ane", enable_ane=False, warmup=False)


def test_bge_local_files_only_forwarded_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `FastBgeSmall` exposes `local_files_only=False` by default and forwards it to `ensure_prebuilt_artifacts`."""
    import inspect
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    import fast_bge_small.engine as bge_eng

    sig = inspect.signature(FastBgeSmall.__init__)
    assert sig.parameters["local_files_only"].default is False

    home = tmp_path / "bge_prebuilt_home"
    (home / "artifacts" / "weights").mkdir(parents=True)
    (home / "artifacts" / "tokenizer").mkdir(parents=True)
    (home / "artifacts" / "weights" / "bge_small_f16.npz").write_bytes(b"stub")
    raw_tok = Tokenizer(WordLevel(vocab={"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102}, unk_token="[UNK]"))
    raw_tok.pre_tokenizer = WhitespaceSplit()
    raw_tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 101), ("[SEP]", 102)],
    )
    raw_tok.save(str(home / "artifacts" / "tokenizer" / "tokenizer.json"))

    seen_calls: list[dict[str, object]] = []
    from dataclasses import replace

    stub_paths = bge_eng.resolve_paths(home)
    missing_paths = replace(
        stub_paths,
        weights_npz=home / "artifacts" / "weights" / "missing.npz",
        ane_models_dir=home / "artifacts" / "missing_ane",
    )

    monkeypatch.setattr(bge_eng, "resolve_paths", lambda _h=None: missing_paths)
    monkeypatch.setattr(bge_eng, "require_runtime_paths", lambda *_a, **_kw: stub_paths)
    monkeypatch.setattr(
        bge_eng,
        "ensure_prebuilt_artifacts",
        lambda h=None, *, include_ane=True, local_files_only=False, **_kw: (
            seen_calls.append({"home": h, "include_ane": include_ane, "local_files_only": local_files_only})
            or stub_paths
        ),
    )
    monkeypatch.setattr(bge_eng, "load_or_mmap_host_tables", lambda *_a, **_kw: (np.zeros((1, 384), dtype=np.float16), np.zeros((1, 384), dtype=np.float16)))
    monkeypatch.setattr(bge_eng, "FastBgeSmallGPU", lambda *_a, **_kw: object())

    with FastBgeSmall(default_mode="gpu", auto_prebuilt=True, local_files_only=True, warmup=False) as eng_off:
        assert eng_off.default_mode == "gpu"
        assert seen_calls[-1] == {"home": None, "include_ane": False, "local_files_only": True}


def test_bge_query_batch_routing_and_table_bind_cache_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Q>=8 batched query routing (H4a) and ANE table-bind deduplication across calls and close()."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    import fast_bge_small.engine as bge_eng

    home = tmp_path / "bge_batch_home"
    (home / "artifacts" / "weights").mkdir(parents=True)
    (home / "artifacts" / "tokenizer").mkdir(parents=True)
    (home / "artifacts" / "ane_models").mkdir(parents=True)
    (home / "build").mkdir(parents=True)
    (home / "artifacts" / "weights" / "bge_small_f16.npz").write_bytes(b"stub")
    (home / "build" / "libbge_ane_runner.dylib").write_bytes(b"stub")

    raw_tok = Tokenizer(WordLevel(vocab={"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "query": 200}, unk_token="[UNK]"))
    raw_tok.pre_tokenizer = WhitespaceSplit()
    raw_tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 101), ("[SEP]", 102)])
    raw_tok.save(str(home / "artifacts" / "tokenizer" / "tokenizer.json"))

    class _StubGPU:
        def __init__(self, weights_path: Path, *, wt0_f16: object = None, pos_f16: object = None) -> None:
            del weights_path, wt0_f16, pos_f16

    class _StubANE:
        def __init__(self) -> None:
            self.bind_calls = 0
            self.clear_calls = 0

        def bge_ane_init_tables_borrowed(self, *_args: object) -> int:
            self.bind_calls += 1
            return 0

        def bge_ane_clear_borrowed_tables(self, *_args: object) -> None:
            self.clear_calls += 1

    stub_ane = _StubANE()
    wt0 = np.zeros((4, 384), dtype=np.float16)
    pos = np.zeros((4, 384), dtype=np.float16)

    def _spy_init_ane(self: FastBgeSmall, *, verify_plan: bool = True) -> None:
        del verify_plan
        self._wt0_f16 = wt0
        self._pos_f16 = pos
        self.ane = stub_ane  # type: ignore[assignment]
        self.ane_available = True
        self._bind_ane_tables(stub_ane)

    monkeypatch.setattr(bge_eng, "load_or_mmap_host_tables", lambda *_a, **_kw: (wt0, pos))
    monkeypatch.setattr(bge_eng, "FastBgeSmallGPU", _StubGPU)
    monkeypatch.setattr(bge_eng, "set_mlx_cache_limit", lambda *_a, **_kw: None)
    monkeypatch.setattr(bge_eng, "_apply_mlx_wired_limit", lambda: None)
    monkeypatch.setattr(bge_eng, "_restore_mlx_wired_limit", lambda *_a, **_kw: None)
    monkeypatch.setattr(bge_eng, "_maybe_clear_mlx_arena", lambda *_a, **_kw: None)
    monkeypatch.setattr(FastBgeSmall, "_init_ane", _spy_init_ane)

    with FastBgeSmall(home=home, default_mode="hetero", enable_ane=True, warmup=False) as eng:
        assert stub_ane.bind_calls == 1
        # Repeated _bind_ane_tables with identical library and arrays skips redundant ctypes init
        eng._bind_ane_tables()
        eng._bind_ane_tables()
        assert stub_ane.bind_calls == 1

        eq_calls: list[bool] = []
        ec_calls: list[tuple[int, str]] = []

        def _fake_encode_query(_ids: object, *, prefer_ane: bool = True) -> np.ndarray:
            eq_calls.append(prefer_ane)
            return np.full((384,), 1.0 / np.sqrt(384.0), dtype=np.float32)

        def _fake_encode_corpus(tok_lists: object, *, mode: str = "hetero", **_kw: object) -> np.ndarray:
            n = len(tok_lists)  # type: ignore[arg-type]
            ec_calls.append((n, mode))
            return np.full((n, 384), 1.0 / np.sqrt(384.0), dtype=np.float32)

        monkeypatch.setattr(eng, "encode_query", _fake_encode_query)
        monkeypatch.setattr(eng, "encode_corpus", _fake_encode_corpus)

        # Q=2 (< 8) with ANE available stays sequential on encode_query
        out2 = eng.encode_texts(["query one", "query two"], is_query=True, mode="hetero")
        assert out2.shape == (2, 384)
        assert eq_calls == [True, True]
        assert ec_calls == []

        # Q=8 (>= 8) routes to batched encode_corpus
        eq_calls.clear()
        out8 = eng.encode_texts([f"query {i}" for i in range(8)], is_query=True, mode="hetero")
        assert out8.shape == (8, 384)
        assert eq_calls == []
        assert ec_calls == [(8, "hetero")]

        # Explicit gpu mode routes even small multi-query batch via encode_corpus
        ec_calls.clear()
        out_gpu = eng.encode_texts(["query one", "query two"], is_query=True, mode="gpu")
        assert out_gpu.shape == (2, 384)
        assert eq_calls == []
        assert ec_calls == [(2, "gpu")]

    assert stub_ane.clear_calls == 1
    # Re-opening rebinds cleanly after close() cleared the cache key
    with FastBgeSmall(home=home, default_mode="hetero", enable_ane=True, warmup=False) as eng2:
        assert eng2.ane_available is True
        assert stub_ane.bind_calls == 2


