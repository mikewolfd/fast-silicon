"""CPU-only unit tests for fast-pplx config, paths, tokenizer, weights, server, Parquet, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import threading
import urllib.request

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from fast_pplx import (
    ContextualTokenizer,
    DEFAULT_MODEL_NAME,
    DEFAULT_PPLX_CONFIG,
    FastPplxServer,
    FastPplxTokenizer,
    PplxConfig,
    RepoPaths,
    resolve_paths,
)
from fast_pplx.cli import build_parser, main as cli_main
from fast_pplx.gpu import (
    _build_sdpa_mask,
    _canonicalize_weights,
    _find_local_hf_snapshot,
)
from fast_pplx.parquet_pipeline import embed_parquet
from fast_pplx.paths import require_runtime_paths
from fast_shared.embeddings import MappedEmbedding, tile_aligned_slices


def _write_synthetic_tokenizer(path: Path) -> Path:
    vocab = {
        "<unk>": 0,
        "D": 1,
        "[": 2,
        "]": 3,
        "hello": 4,
        "world": 5,
        "chunk": 6,
        "one": 7,
        "two": 8,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<|endoftext|>", "[Q] ", "[D] "])
    tok.add_tokens(["<|chunk_sep|>"])
    out = path / "tokenizer.json"
    tok.save(str(out))
    return out


def _write_synthetic_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> Path:
    offset = 0
    header: dict[str, dict[str, object]] = {}
    payloads: list[bytes] = []
    for name, (arr, dtype_tag) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {
            "dtype": dtype_tag,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payloads.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad_len = (-len(hdr_bytes)) % 8
    hdr_bytes = hdr_bytes + b" " * pad_len
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for chunk in payloads:
            fh.write(chunk)
    return path


def test_paths_resolution_and_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_PPLX_HOME", str(tmp_path))
    p = resolve_paths()
    assert isinstance(p, RepoPaths)
    assert p.home == tmp_path.resolve()
    assert p.weights_dir == tmp_path.resolve() / "artifacts" / "weights"
    assert p.coreml_dylib == tmp_path.resolve() / "build" / "libpplx_coreml_runner.dylib"
    assert p.io_dylib == tmp_path.resolve() / "build" / "libpplx_fast_io.dylib"
    assert p.ane_cli == tmp_path.resolve() / "build" / "pplx-ane-cli"

    with pytest.raises(FileNotFoundError, match="Missing exported weights"):
        require_runtime_paths(tmp_path)

    p.weights_npz.parent.mkdir(parents=True, exist_ok=True)
    p.weights_npz.write_bytes(b"stub")
    with pytest.raises(FileNotFoundError, match="Missing tokenizer"):
        require_runtime_paths(tmp_path)

    p.tokenizer_json.parent.mkdir(parents=True, exist_ok=True)
    p.tokenizer_json.write_text("{}", encoding="utf-8")
    assert require_runtime_paths(tmp_path, require_ane=False) == p

    with pytest.raises(FileNotFoundError, match="Missing native ANE runner"):
        require_runtime_paths(tmp_path, require_ane=True)


def test_config_overrides_interval_and_handles_null_pad(tmp_path: Path) -> None:
    assert DEFAULT_PPLX_CONFIG.full_attention_interval == 2
    assert DEFAULT_PPLX_CONFIG.rope_theta == 10_000_000.0
    assert DEFAULT_PPLX_CONFIG.chunk_sep_token_id == 248079
    assert DEFAULT_PPLX_CONFIG.query_prefix_token_id == 248077

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "embedding_dim": 1024,
                "query_prefix": "[Q] ",
                "document_prefix": "[D] ",
                "boundary_marker": "<|chunk_sep|>",
                "text_config": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 12,
                    "intermediate_size": 3584,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "full_attention_interval": 4,
                    "layer_types": ["linear_attention", "full_attention"] * 6,
                    "pad_token_id": None,
                    "eos_token_id": 248044,
                    "rope_parameters": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000,
                        "mrope_section": [11, 11, 10],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = PplxConfig.from_json(cfg_path)
    assert cfg.full_attention_interval == 2
    assert cfg.pad_token_id == 248044
    assert cfg.rope_dim == 64
    assert cfg.mrope_section == (11, 11, 10)


def test_synthetic_tokenizer_special_tokens_and_spans(tmp_path: Path) -> None:
    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)
    assert FastPplxTokenizer is ContextualTokenizer
    doc_prefix_id = tok.reserved_id("[D] ")

    q_ids = tok.encode_query("hello world")
    assert q_ids == [tok.query_prefix_id, 4, 5]

    # Document prefix [D]  is special=True, so encode_special_tokens=True splits literal prefix
    # rather than emitting reserved ID; user control tokens are neutralized too.
    d_ids = tok.encode_document("hello <|endoftext|> world")
    assert doc_prefix_id not in d_ids
    assert tok.pad_id not in d_ids

    c_ids, c_spans = tok.encode_contextual_document(["hello", "", "chunk two"], return_spans=True)
    assert c_ids.count(tok.chunk_sep_id) == 2
    assert c_spans[1] == (0, 0)
    assert c_ids[c_spans[0][0] : c_spans[0][1]] == [4]
    assert c_ids[c_spans[2][0] : c_spans[2][1]] == [6, 8]

    chunked = tok.chunk_text(
        "hello world chunk one two hello world",
        max_chunk_tokens=8,
        overlap=1,
        add_document_prefix=True,
    )
    assert len(chunked) >= 2
    assert all(len(ids) <= 8 for ids, _, _ in chunked)


def test_tile_aligned_slices_and_mapped_embedding(tmp_path: Path) -> None:
    assert tile_aligned_slices(256, 512, 64) == [(0, 256)]
    assert tile_aligned_slices(520, 512, 64) == [(0, 520)]
    assert tile_aligned_slices(1087, 512, 64) == [(0, 512), (512, 1087)]
    assert tile_aligned_slices(1088, 512, 64) == [(0, 512), (512, 1024), (1024, 1088)]

    f32 = np.arange(24, dtype=np.float32).reshape(6, 4)
    bf16_u16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
    st_path = _write_synthetic_safetensors(
        tmp_path / "embed.safetensors",
        {"language_model.embed_tokens.weight": (bf16_u16, "BF16")},
    )
    with MappedEmbedding(st_path, tensor_name="language_model.embed_tokens.weight") as mm:
        gathered = mm.gather(np.array([[2, 5]], dtype=np.int32), out_dtype="float16")
        assert gathered.shape == (1, 2, 4)
        assert gathered.dtype == np.float16
        np.testing.assert_allclose(gathered[0, 0], f32[2], atol=0.1)
        np.testing.assert_allclose(gathered[0, 1], f32[5], atol=0.1)


def test_canonicalize_weights_and_additive_sdpa_mask() -> None:
    raw = {
        "visual.patch_embed.weight": mx.ones((4, 4), dtype=mx.bfloat16),
        "language_model.embed_tokens.weight": mx.ones((16, 8), dtype=mx.bfloat16),
        "language_model.norm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "contextual_projection.weight": mx.eye(8, dtype=mx.bfloat16),
        "language_model.layers.0.input_layernorm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "language_model.layers.0.post_attention_layernorm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "language_model.layers.0.linear_attn.norm.weight": mx.zeros((4,), dtype=mx.bfloat16),
        "language_model.layers.0.linear_attn.A_log": mx.ones((2,), dtype=mx.bfloat16),
        "language_model.layers.0.linear_attn.dt_bias": mx.ones((2,), dtype=mx.bfloat16),
        "language_model.layers.0.linear_attn.conv1d.weight": mx.ones((12, 1, 4), dtype=mx.bfloat16),
        "language_model.layers.0.mlp.gate_proj.weight": mx.ones((4, 8), dtype=mx.bfloat16),
        "language_model.layers.0.mlp.up_proj.weight": mx.full((4, 8), 2.0, dtype=mx.bfloat16),
        "language_model.layers.1.input_layernorm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "language_model.layers.1.post_attention_layernorm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.q_norm.weight": mx.zeros((4,), dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.k_norm.weight": mx.zeros((4,), dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.q_proj.weight": mx.ones((8, 8), dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.k_proj.weight": mx.full((4, 8), 2.0, dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.v_proj.weight": mx.full((4, 8), 3.0, dtype=mx.bfloat16),
    }
    items, head_w = _canonicalize_weights(
        raw,
        num_layers=2,
        act_dtype=mx.float16,
        include_embed=False,
        norms_already_shifted=False,
    )
    wmap = dict(items)
    assert not any(k.startswith("visual.") or k == "embed_tokens.weight" for k in wmap)
    assert head_w.dtype == mx.float32
    assert np.allclose(np.array(wmap["norm.weight"]), 1.0)
    assert np.allclose(np.array(wmap["layers.0.input_layernorm.weight"]), 1.0)
    assert np.allclose(np.array(wmap["layers.1.self_attn.q_norm.weight"]), 1.0)
    assert np.allclose(np.array(wmap["layers.0.linear_attn.norm.weight"]), 0.0)
    assert wmap["layers.0.linear_attn.norm.weight"].dtype == mx.float32
    assert wmap["layers.0.linear_attn.A_log"].dtype == mx.float32
    assert wmap["layers.0.linear_attn.conv1d.weight"].shape == (12, 4, 1)
    assert wmap["layers.0.mlp.gate_up_proj.weight"].shape == (8, 8)
    assert wmap["layers.1.self_attn.qkv_proj.weight"].shape == (16, 8)

    mask = np.array([[True, True, False]], dtype=np.bool_)
    sdpa_f16 = _build_sdpa_mask(mask, dtype_name="float16")
    assert sdpa_f16.dtype == mx.float16
    assert sdpa_f16.shape == (1, 1, 1, 3)
    assert float(sdpa_f16[0, 0, 0, 0]) == 0.0
    assert float(sdpa_f16[0, 0, 0, 2]) == -65504.0


def test_cli_hw_check_and_server_parquet_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parser = build_parser()
    for sub in ("setup-weights", "setup-ane", "embed", "bench", "parquet", "serve"):
        assert parser.parse_args([sub] if sub in ("setup-weights", "setup-ane", "bench", "serve") else (
            ["embed", "hi"] if sub == "embed" else ["parquet", "--input", "a.parquet", "--output", "b.parquet"]
        )).command == sub

    called = {"hw": 0}

    def _fake_hw_warn(*args, **kwargs) -> bool:
        called["hw"] += 1
        return False

    monkeypatch.setattr("fast_pplx.cli.warn_if_untested_hardware", _fake_hw_warn)
    src_dir = tmp_path / "src_hf"
    src_dir.mkdir()
    _write_synthetic_tokenizer(src_dir)
    (src_dir / "config.json").write_text("{}", encoding="utf-8")
    (src_dir / "model.safetensors").write_bytes(b"stub")
    home_dir = tmp_path / "home"
    rc = cli_main(["setup-weights", "--home", str(home_dir), "--model-dir", str(src_dir), "--local-files-only"])
    assert rc == 0
    assert called["hw"] == 1
    assert (home_dir / "artifacts" / "tokenizer" / "tokenizer.json").exists()
    assert (home_dir / "artifacts" / "weights" / "model" / "model.safetensors").exists()

    # Verify FastPplxServer and parquet_pipeline contract using a deterministic fake engine
    tok = ContextualTokenizer(home_dir / "artifacts" / "tokenizer" / "tokenizer.json", DEFAULT_PPLX_CONFIG)

    class _FakeEngine:
        dim = 8
        max_dim = 8
        supports_dimensions = True
        model_name = DEFAULT_MODEL_NAME
        max_seq_len = 512
        ane_available = False
        default_mode = "gpu"
        tokenizer = tok

        def encode_query(self, token_ids, *, dimensions=None, **_kwargs):
            d = dimensions or self.dim
            v = np.ones((d,), dtype=np.float32)
            v /= np.linalg.norm(v)
            return v

        def encode_corpus(self, token_lists, *, dimensions=None, **_kwargs):
            d = dimensions or self.dim
            out = np.zeros((len(token_lists), d), dtype=np.float32)
            for idx, ids in enumerate(token_lists):
                out[idx, idx % d] = float(max(1, len(ids)))
                out[idx] /= np.linalg.norm(out[idx])
            return out

    eng = _FakeEngine()
    with FastPplxServer(eng, host="127.0.0.1", port=0) as srv:
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/health") as resp:
            health = json.loads(resp.read().decode("utf-8"))
        assert health["model"] == DEFAULT_MODEL_NAME

        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/v1/embeddings",
            data=json.dumps({"input": ["hello world"], "dimensions": 4}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert len(payload["data"][0]["embedding"]) == 4

    in_pq = tmp_path / "in.parquet"
    out_pq = tmp_path / "out.parquet"
    pq.write_table(pa.table({"id": [1, 2], "text": ["hello world", "chunk one"]}), in_pq)
    summary = embed_parquet(eng, in_pq, out_pq, text_col="text", dimensions=4)
    assert summary.rows_emitted == 2


def test_fixture_tokenizer_all_16_cases_match_reference_npz() -> None:
    snap = _find_local_hf_snapshot(DEFAULT_MODEL_NAME)
    if snap is None:
        pytest.skip("Local HuggingFace snapshot not cached; skipping HF tokenizer parity test")
    fix_dir = Path(__file__).resolve().parent / "fixtures"
    cases = json.loads((fix_dir / "cases.json").read_text(encoding="utf-8"))
    ref = np.load(fix_dir / "reference.npz")

    cfg = PplxConfig.from_json(snap / "config.json")
    tok = ContextualTokenizer(snap, cfg)
    for case in cases:
        cid = case["id"]
        if case["task"] == "query":
            out = tok.encode_queries(case["documents"])
            np.testing.assert_array_equal(out["input_ids"], ref[f"{cid}/input_ids"], err_msg=cid)
        else:
            out = tok.encode_documents(case["documents"])
            np.testing.assert_array_equal(out["input_ids"], ref[f"{cid}/input_ids"], err_msg=cid)
            np.testing.assert_array_equal(out["chunk_token_spans"], ref[f"{cid}/spans"], err_msg=cid)


def test_context_window_packing_and_hetero2_cli_cpu(tmp_path: Path) -> None:
    from fast_pplx.engine import FastPplx
    from fast_pplx.parquet_pipeline import embed_contextual_parquet

    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)

    doc_text = " ".join(["hello world chunk one"] * 10)
    grouped = tok.chunk_and_group_document(doc_text, chunk_tokens=10, context_tokens=24, overlap=2)
    assert len(grouped) >= 2
    for win_chunks, spans in grouped:
        assert len(win_chunks) == len(spans) >= 1
        if len(win_chunks) > 1:
            assert len(tok.encode_contextual_document(win_chunks, return_spans=False)) <= 24

    class _MockGPU:
        max_seq_len = 32768

        def encode_tokens(self, token_lists, *, batch_size=16, dimensions=None, output_dtype="float32", normalize=True):
            dim = dimensions or 1024
            return np.ones((len(token_lists), dim), dtype=np.float32) * 0.125

        def encode_contextual_chunks(self, docs, *, batch_size=8, max_length=32768, dimensions=None, output_dtype="float32", normalize=True):
            dim = dimensions or 1024
            return [np.ones((len(d), dim), dtype=np.float32) * 0.5 for d in docs]

        def close(self):
            pass

    with FastPplx(home=tmp_path, mode="auto", _gpu_backend=_MockGPU(), _tokenizer=tok) as eng:
        assert eng.ane_available is False
        assert eng.default_mode == "gpu"
        in_pq = tmp_path / "ctx_in.parquet"
        out_pq = tmp_path / "ctx_out.parquet"
        pq.write_table(pa.table({"doc_id": ["d0"], "text": [doc_text]}), in_pq)
        summary = embed_contextual_parquet(
            eng, in_pq, out_pq, chunk_tokens=10, context_tokens=24, overlap=2, dimensions=128
        )
        assert summary.rows_emitted >= 4
        out_tbl = pq.read_table(out_pq)
        assert {"window_index", "chunk_index", "char_start", "char_end", "vector"} <= set(out_tbl.schema.names)

    ns = build_parser().parse_args(
        ["parquet", "--input", "in.parquet", "--output", "out.parquet", "--mode", "hetero2", "--context-tokens", "2048", "--chunk-tokens", "256"]
    )
    assert ns.mode == "hetero2"
    assert ns.context_tokens == 2048
    assert ns.chunk_tokens == 256


def test_streamed_gdn_sdpa_whole_in_proj_parity_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import fast_pplx.gpu as pplx_gpu_mod
    from fast_pplx.gpu import PplxModel

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(99)
        cfg = PplxConfig(
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            rope_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=("linear_attention", "full_attention"),
        )
        model = PplxModel(cfg, has_embed_module=False)
        model.layers[0].linear_attn.use_kernel = False
        mx.eval(model.parameters())

        seq_len, valid_b1 = 97, 72
        x = mx.random.normal((2, seq_len, cfg.hidden_size))
        mask_np = np.ones((2, seq_len), dtype=np.bool_)
        mask_np[1, valid_b1:] = False
        ssm_mask = mx.array(mask_np, dtype=mx.bool_)
        sdpa_mask = _build_sdpa_mask(mask_np, dtype_name="float32")

        monkeypatch.setattr(pplx_gpu_mod, "_STREAMED_SEQ_THRESHOLD", 10_000)
        whole = model.forward_hidden(x, ssm_mask=ssm_mask, sdpa_mask=sdpa_mask)

        monkeypatch.setattr(pplx_gpu_mod, "_STREAMED_SEQ_THRESHOLD", 64)
        monkeypatch.setattr(pplx_gpu_mod, "_GDN_SLICE_LEN", 32)
        monkeypatch.setattr(pplx_gpu_mod, "_SDPA_SLICE_LEN", 32)
        monkeypatch.setattr(pplx_gpu_mod, "_SDPA_MIN_TAIL", 32)
        streamed = model.forward_hidden(x, ssm_mask=ssm_mask, sdpa_mask=sdpa_mask)

        mx.eval(whole, streamed)
        assert np.allclose(np.array(whole, dtype=np.float32), np.array(streamed, dtype=np.float32), atol=2e-4, rtol=1e-4)
    finally:
        mx.set_default_device(prev_dev)


def test_resolve_home_checks_artifacts_and_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_pplx.paths import resolve_home

    monkeypatch.delenv("FAST_PPLX_HOME", raising=False)
    monkeypatch.delenv("FAST_ARTIFACTS_ROOT", raising=False)

    # Fake monorepo layout: <tmp>/work/fast-silicon/packages/fast-pplx/src/fast_pplx/paths.py
    pkg_dir = tmp_path / "work" / "fast-silicon" / "packages" / "fast-pplx"
    anchor = pkg_dir / "src" / "fast_pplx" / "paths.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("# stub\n", encoding="utf-8")
    # Put pyproject.toml + native/ + build/ inside pkg_dir, WITHOUT artifacts/
    (pkg_dir / "pyproject.toml").write_text("[project]\nname='fast-pplx'\n", encoding="utf-8")
    (pkg_dir / "native").mkdir()
    (pkg_dir / "build").mkdir()

    # 1. Must NOT stop on native/ or pyproject.toml; falls back to ~/.cache/fast-pplx when nothing has artifacts/
    fake_home = tmp_path / "user_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    assert resolve_home(_anchor_file=anchor) == (fake_home / ".cache" / "fast-pplx").resolve()

    # 2. Populated ~/.cache/fast-pplx/artifacts resolves to cache dir while build_dir falls back to pkg_dir/build
    (fake_home / ".cache" / "fast-pplx" / "artifacts").mkdir(parents=True)
    p_cache = resolve_paths(_anchor_file=anchor)
    assert p_cache.home == (fake_home / ".cache" / "fast-pplx").resolve()
    assert p_cache.build_dir == (pkg_dir / "build").resolve()

    # 3. Sibling checkout <tmp>/work/fast-pplx/artifacts beats ~/.cache
    sib_dir = tmp_path / "work" / "fast-pplx"
    (sib_dir / "artifacts").mkdir(parents=True)
    assert resolve_home(_anchor_file=anchor) == sib_dir.resolve()

    # 4. FAST_ARTIFACTS_ROOT/fast-pplx/artifacts beats sibling checkout
    art_root = tmp_path / "shared_artifacts"
    (art_root / "fast-pplx" / "artifacts").mkdir(parents=True)
    monkeypatch.setenv("FAST_ARTIFACTS_ROOT", str(art_root))
    assert resolve_home(_anchor_file=anchor) == (art_root / "fast-pplx").resolve()

    # 5. Direct artifacts/ inside pkg_dir beats FAST_ARTIFACTS_ROOT
    (pkg_dir / "artifacts").mkdir()
    assert resolve_home(_anchor_file=anchor) == pkg_dir.resolve()


def test_tf32_telemetry_chip_gated_attention_and_fp32_norms(tmp_path: Path) -> None:
    import os
    from fast_pplx.gpu import (
        FastPplxGPU,
        PplxAttention,
        PplxRMSNorm,
        _detect_gpu_family_info,
        get_mlx_runtime_telemetry,
        normalize_qk,
    )

    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    tel = get_mlx_runtime_telemetry()
    assert tel["mlx_enable_tf32"] == "0"
    assert "is_apple10" in tel and "sdpa_slice_len" in tel

    # Apple9 (M4 Pro) -> 512; Apple10 (M5 26.2+) -> 1024; M5 pre-26.2 -> 512
    m4 = _detect_gpu_family_info({"device_name": "Apple M4 Pro", "architecture": "applegpu_g16s"}, "26.6")
    assert m4["is_apple10"] is False and m4["sdpa_slice_len"] == 512
    m5 = _detect_gpu_family_info({"device_name": "Apple M5", "architecture": "applegpu_g17g"}, "26.2")
    assert m5["is_apple10"] is True and m5["nax_sdpa_eligible"] is True and m5["sdpa_slice_len"] == 1024
    m5_old = _detect_gpu_family_info({"device_name": "Apple M5", "architecture": "applegpu_g17g"}, "26.0")
    assert m5_old["is_apple10"] is True and m5_old["nax_sdpa_eligible"] is False and m5_old["sdpa_slice_len"] == 512

    cfg = PplxConfig(
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, rope_dim=8,
        layer_types=("linear_attention", "full_attention"),
    )
    attn_m5 = PplxAttention(cfg, gpu_family=m5)
    attn_m4 = PplxAttention(cfg, gpu_family=m4)
    assert attn_m5.sdpa_slice_len == 1024
    assert attn_m4.sdpa_slice_len == 512

    # Bool mask must be rejected on PplxAttention fast path
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        x_small = mx.ones((1, 4, 64), dtype=mx.float16)
        with pytest.raises(TypeError, match="SDPA mask"):
            attn_m5(x_small, sdpa_mask=mx.ones((1, 1, 4, 4), dtype=mx.bool_))
        # Causal string fast path works with zero array mask allocation
        out_causal = attn_m5(x_small, sdpa_mask="causal")
        mx.eval(out_causal)
        assert out_causal.shape == (1, 4, 64)

        # normalize_qk protects tiny FP16 inputs (where eps/128 < 2^-24) & large >256 inputs against overflow
        q_large = mx.full((1, 2, 2, 128), 512.0, dtype=mx.float16)
        k_tiny = mx.full((1, 2, 2, 128), 1e-4, dtype=mx.float16)
        qn, kn = normalize_qk(q_large, k_tiny, inv_scale=128.0 ** -0.5, eps=1e-6)
        mx.eval(qn, kn)
        assert np.all(np.isfinite(np.array(qn, dtype=np.float32)))
        assert np.all(np.isfinite(np.array(kn, dtype=np.float32)))
        assert qn.dtype == mx.float16 and kn.dtype == mx.float16

        # PplxRMSNorm protects deep FP16 residual outliers (>256) from x*x FP16 overflow
        norm = PplxRMSNorm(16, eps=1e-6)
        big_x = mx.full((1, 2, 16), 2048.0, dtype=mx.float16)
        normed = norm(big_x)
        mx.eval(normed)
        assert np.all(np.isfinite(np.array(normed, dtype=np.float32)))
        assert np.allclose(np.array(normed, dtype=np.float32), 1.0, atol=1e-2)

        # _embed_inputs gathers MappedEmbedding as float16 even for bfloat16 (no CPU float32 upcast)
        seen_dtypes: list[str] = []

        class _SpyMapped:
            def gather(self, ids, *, out_dtype="float16"):
                seen_dtypes.append(str(out_dtype))
                return np.ones((*ids.shape, 8), dtype=np.float16)

        stub = object.__new__(FastPplxGPU)
        stub._mapped_embed = _SpyMapped()
        stub.dtype_name = "bfloat16"
        stub.act_dtype = mx.bfloat16
        emb = stub._embed_inputs(np.array([[1, 2]], dtype=np.int32))
        mx.eval(emb)
        assert seen_dtypes == ["float16"]
        assert emb.dtype == mx.bfloat16
    finally:
        mx.set_default_device(prev_dev)


def test_parquet_int8_streaming_sharded_resume_and_cli_forwarding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fast_pplx.parquet_pipeline import embed_contextual_parquet, embed_parquet

    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)

    in_pq = tmp_path / "docs.parquet"
    doc_a = " ".join(["hello world chunk one two"] * 6)
    doc_b = " ".join(["chunk two hello world one"] * 6)
    pq.write_table(pa.table({"doc_id": ["a", "b", "empty"], "text": [doc_a, doc_b, "   "]}), in_pq)

    encode_calls = {"ctx": 0, "flat": 0}

    class _Int8Engine:
        dim = 16
        max_dim = 16
        max_seq_len = 512
        supports_dimensions = True
        tokenizer = tok

        def encode_contextual_chunks(self, docs, *, dimensions=None, output_dtype="float32", normalize=True, mode=None):
            encode_calls["ctx"] += 1
            d = dimensions or self.dim
            dt = {"int8": np.int8, "float16": np.float16, "float32": np.float32}[output_dtype]
            return [np.full((len(chunks), d), 42 if output_dtype == "int8" else 0.5, dtype=dt) for chunks in docs]

        def encode_corpus(self, tok_lists, *, dimensions=None, output_dtype="float32", normalize=True, mode=None):
            encode_calls["flat"] += 1
            d = dimensions or self.dim
            dt = {"int8": np.int8, "float16": np.float16, "float32": np.float32}[output_dtype]
            return np.full((len(tok_lists), d), -17 if output_dtype == "int8" else 0.25, dtype=dt)

        def close(self):
            pass

    eng = _Int8Engine()

    # 1. Contextual single-file int8 output
    out_ctx_file = tmp_path / "ctx_i8.parquet"
    s1 = embed_contextual_parquet(
        eng, in_pq, out_ctx_file, chunk_tokens=10, context_tokens=24, overlap=2,
        dimensions=8, dtype="int8", skip_empty=True,
    )
    assert s1.rows_read == 3
    assert s1.rows_emitted >= 4
    tbl = pq.read_table(out_ctx_file)
    assert tbl.schema.field("vector").type == pa.list_(pa.int8(), 8)
    assert all(v == 42 for v in tbl["vector"][0].as_py())

    # 2. Contextual sharded directory + --resume skips completed shards without re-encoding
    out_shards = tmp_path / "ctx_shards"
    encode_calls["ctx"] = 0
    s2 = embed_contextual_parquet(
        eng, in_pq, out_shards, chunk_tokens=10, context_tokens=24, overlap=2,
        dimensions=8, dtype="int8", shard_rows=2, resume=False,
    )
    shards = sorted(out_shards.glob("part-*.parquet"))
    assert len(shards) >= 2
    first_calls = encode_calls["ctx"]
    assert first_calls >= 2

    encode_calls["ctx"] = 0
    _ = embed_contextual_parquet(
        eng, in_pq, out_shards, chunk_tokens=10, context_tokens=24, overlap=2,
        dimensions=8, dtype="int8", shard_rows=2, resume=True,
    )
    assert encode_calls["ctx"] == 0  # Every existing shard matched row count and was skipped!

    # 3. Non-contextual embed_parquet supports int8 streaming
    out_flat_i8 = tmp_path / "flat_i8.parquet"
    s3 = embed_parquet(eng, in_pq, out_flat_i8, text_col="text", dimensions=8, dtype="int8", skip_empty=True)
    assert s3.rows_emitted == 2
    flat_tbl = pq.read_table(out_flat_i8)
    assert flat_tbl.schema.field("vector").type == pa.list_(pa.int8(), 8)
    assert all(v == -17 for v in flat_tbl["vector"][0].as_py())

    # 4. CLI forwards --resume, --shard-rows, --encode-batch-rows, --skip-empty, --query-prefix, --dtype int8
    captured: dict[str, object] = {}

    def _spy_ctx(*args, **kwargs):
        captured.update(kwargs)
        from fast_shared.parquet_pipeline import ParquetEmbedSummary

        return ParquetEmbedSummary(1, 1, 1, 10, 0.01, tmp_path / "out")

    monkeypatch.setattr("fast_pplx.cli.warn_if_untested_hardware", lambda *a, **k: False)
    monkeypatch.setattr("fast_pplx.cli._make_engine", lambda a, **kw: eng)
    monkeypatch.setattr("fast_pplx.cli.embed_contextual_parquet", _spy_ctx)

    rc = cli_main([
        "parquet", "--input", str(in_pq), "--output", str(tmp_path / "cli_out"),
        "--context-tokens", "2048", "--chunk-tokens", "256", "--dtype", "int8",
        "--resume", "--shard-rows", "128", "--encode-batch-rows", "64",
        "--skip-empty", "--query-prefix",
    ])
    assert rc == 0
    assert captured["resume"] is True
    assert captured["shard_rows"] == 128
    assert captured["encode_batch_rows"] == 64
    assert captured["skip_empty"] is True
    assert captured["add_query_prefix"] is True
    assert captured["dtype"] == "int8"

    # Verify new `verify` and upgraded `bench` CLI flags parse cleanly
    p = build_parser()
    assert p.parse_args(["verify", "--mode", "gpu", "--json"]).command == "verify"
    b_ns = p.parse_args(["bench", "--mode", "all", "--runs", "2", "--timing-scope", "e2e"])
    assert b_ns.mode == "all" and b_ns.runs == 2 and b_ns.timing_scope == "e2e"


def test_thread_safe_last_stats_and_hetero_vs_hetero2_deterministic(tmp_path: Path) -> None:
    import time as _time
    from fast_pplx.engine import FastPplx

    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)

    class _MockGPU:
        max_seq_len = 32768

        def encode_tokens(self, token_lists, *, batch_size=16, dimensions=None, output_dtype="float32", normalize=True):
            return np.ones((len(token_lists), dimensions or 16), dtype=np.float32)

        def close(self):
            pass

    class _MockANE:
        caps = (64, 128)
        max_cap = 128

        def run_buckets(self, buckets, *, out_dim, apply_tanh_int8, apply_l2_norm):
            _time.sleep(0.005)
            out = []
            for bkt in buckets:
                live = [t for t in bkt.targets if t.end > t.start]
                out.append((bkt, np.ones((len(live), out_dim), dtype=np.float32) * 2.0))
            return out

        def close(self):
            pass

    # 1. `hetero` is deterministic static partition (zero stealing); `hetero2` steals
    with FastPplx(home=tmp_path, mode="hetero", _gpu_backend=_MockGPU(), _ane_backend=_MockANE(), _tokenizer=tok) as eng:
        items = [[4, 5, 6] for _ in range(20)] + [[7] * 200]
        _ = eng.encode_tokens(items, dimensions=16, mode="hetero")
        st_h = eng.last_stats
        assert st_h["deterministic"] is True
        assert st_h["gpu_stolen_items"] == 0
        assert st_h["ane_items"] == 20 and st_h["gpu_items"] == 1

        # 2. Thread-local last_stats isolation across concurrent threads
        results_by_thread: dict[str, dict[str, object]] = {}

        def _worker(tag: str, mode_name: str, n_items: int) -> None:
            _ = eng.encode_tokens([[4, 5]] * n_items, dimensions=16, mode=mode_name)
            _time.sleep(0.01)
            results_by_thread[tag] = eng.last_stats

        t1 = threading.Thread(target=_worker, args=("gpu_t", "gpu", 3))
        t2 = threading.Thread(target=_worker, args=("ane_t", "ane", 5))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert results_by_thread["gpu_t"]["mode"] == "gpu"
        assert results_by_thread["gpu_t"]["gpu_items"] == 3
        assert results_by_thread["ane_t"]["mode"] == "ane"
        assert results_by_thread["ane_t"]["ane_items"] == 5

    # 3. deterministic=True routes auto / hetero2 strictly to deterministic gpu
    with FastPplx(
        home=tmp_path, mode="auto", deterministic=True,
        _gpu_backend=_MockGPU(), _ane_backend=_MockANE(), _tokenizer=tok,
    ) as eng_det:
        assert eng_det.default_mode == "gpu"
        _ = eng_det.encode_tokens([[4, 5, 6]], dimensions=16, mode="hetero2")
        assert eng_det.last_stats["mode"] == "gpu"
        assert eng_det.last_stats["deterministic"] is True
