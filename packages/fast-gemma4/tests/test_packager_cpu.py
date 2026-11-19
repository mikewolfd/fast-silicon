"""Pure-CPU unit tests for `checkpoint_packager` (zero GPU/Metal, zero `mx.eval`)."""

from __future__ import annotations

import json
from pathlib import Path
import py_compile
import struct
import sys

import numpy as np
import pytest

import fast_gemma4.packager as fg4_packager
from fast_gemma4.packager import (
    DEFAULT_KEPT_VOCAB_SIZE,
    RawTensor,
    array_to_raw_tensor,
    bf16_u16_to_f32,
    dequantize_affine_8bit_u32,
    f32_to_bf16_u16,
    inspect_safetensors_headers,
    is_multimodal_key,
    load_kept_old_ids,
    load_safetensors_dir,
    pack_checkpoint_dir,
    pack_checkpoint_tensors,
    quantize_affine_8bit_u32,
    read_safetensors_header,
    reload_and_dequantize_dir_cpu,
    strip_multimodal_tensors,
    write_safetensors_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB_REMAP_150786 = REPO_ROOT / "artifacts" / "vocab" / "vocab_remap_150786.npz"
BF16_HF_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/fb0b166bbb9a0eb4b37915bfc515a197c9122f39"
)
QAT4_HF_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-qat-4bit/snapshots/42f62737af7a9fd8c1d55d79666c1a217be4e2e2"
)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12))


def test_packager_py_compiles() -> None:
    py_compile.compile(str(Path(fg4_packager.__file__)), doraise=True)


def test_bf16_conversion_and_8bit_g64_quant_roundtrip() -> None:
    rng = np.random.default_rng(7)
    w_f32 = rng.normal(0.0, 0.05, size=(128, 256)).astype(np.float32)
    # Include a constant group to verify zero-span stability
    w_f32[0, :64] = 0.125
    w_bf16_u16 = f32_to_bf16_u16(w_f32)
    w_bf16_f32 = bf16_u16_to_f32(w_bf16_u16)

    qw, qs, qb = quantize_affine_8bit_u32(w_bf16_f32, group_size=64)
    assert qw.dtype == "U32" and qw.shape == (128, 64)
    assert qs.dtype == "BF16" and qs.shape == (128, 4)
    assert qb.dtype == "BF16" and qb.shape == (128, 4)

    recon = dequantize_affine_8bit_u32(qw, qs, qb, group_size=64)
    assert recon.shape == (128, 256)
    assert _cos_sim(recon, w_bf16_f32) > 0.99997
    assert np.allclose(recon[0, :64], 0.125, atol=1e-3)


def test_multimodal_stripper_exact_bytes_and_index(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    tensors: dict[str, RawTensor] = {
        "vision_tower.encoder.layers.0.weight": array_to_raw_tensor(rng.normal(size=(64, 64)), "BF16"),
        "audio_tower.layers.0.linear.weight": array_to_raw_tensor(rng.normal(size=(128, 64)), "BF16"),
        "embed_vision.embedding_projection.weight": array_to_raw_tensor(rng.normal(size=(64, 32)), "BF16"),
        "embed_audio.embedding_projection.weight": array_to_raw_tensor(rng.normal(size=(64, 64)), "BF16"),
        "multi_modal_projector.linear.weight": array_to_raw_tensor(rng.normal(size=(32, 64)), "BF16"),
        "language_model.model.layers.0.self_attn.q_proj.weight": array_to_raw_tensor(rng.normal(size=(64, 64)), "BF16"),
        "language_model.model.norm.weight": array_to_raw_tensor(rng.normal(size=(64,)), "BF16"),
    }
    expected_removed = (64 * 64 + 128 * 64 + 64 * 32 + 64 * 64 + 32 * 64) * 2
    expected_kept = (64 * 64 + 64) * 2

    kept, rep = strip_multimodal_tensors(tensors)
    assert rep.bytes_removed == expected_removed
    assert rep.bytes_after == expected_kept
    assert rep.bytes_before == expected_removed + expected_kept
    assert set(kept.keys()) == {
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.norm.weight",
    }
    assert all(is_multimodal_key(k) is None for k in kept)
    assert set(rep.bytes_by_prefix.keys()) == {
        "vision_tower",
        "audio_tower",
        "embed_vision",
        "embed_audio",
        "multi_modal_projector",
    }

    # Verify shard write + 8-byte uint64 header format
    shards, tot = write_safetensors_dir(tmp_path / "stripped", kept)
    assert shards == ["model.safetensors"]
    assert tot == expected_kept
    hdr, base = read_safetensors_header(tmp_path / "stripped" / "model.safetensors")
    assert base >= 8 + len(json.dumps(hdr, separators=(",", ":")))
    idx = json.loads((tmp_path / "stripped" / "model.safetensors.index.json").read_text())
    assert idx["metadata"]["total_size"] == expected_kept
    assert set(idx["weight_map"].keys()) == set(kept.keys())


def test_full_pack_strip_vocab_reload_roundtrip_synthetic_multishard(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    src_dir = tmp_path / "src_ckpt"
    dst_dir = tmp_path / "dst_ckpt"
    src_dir.mkdir()

    v_full = 512
    v_kept = 192
    kept_ids = np.sort(rng.choice(v_full, size=v_kept, replace=False)).astype(np.int32)
    remap_npz = tmp_path / "vocab_remap.npz"
    np.savez(remap_npz, old_ids=kept_ids)  # Test `old_ids` alias compatibility

    # Build synthetic multi-dtype (BF16, F16, F32) checkpoint matching Gemma-4 E2B topology
    raw_src: dict[str, RawTensor] = {
        # Unused multimodal towers
        "vision_tower.encoder.layers.0.self_attn.q_proj.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.02, size=(128, 128)), "BF16"
        ),
        "audio_tower.layers.0.self_attn.q_proj.linear.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.02, size=(128, 128)), "F16"
        ),
        "embed_vision.embedding_projection.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.02, size=(64, 64)), "F32"
        ),
        "embed_audio.embedding_projection.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.02, size=(64, 64)), "BF16"
        ),
        "multi_modal_projector.weight": array_to_raw_tensor(rng.normal(0.0, 0.02, size=(64, 64)), "BF16"),
        # Language model vocab tables
        "language_model.model.embed_tokens.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.04, size=(v_full, 128)), "BF16"
        ),
        "language_model.model.embed_tokens_per_layer.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.04, size=(v_full, 256)), "BF16"
        ),
        # Language model non-MLP linears (multi-dtype inputs -> all packed to 8-bit g64)
        "language_model.model.per_layer_model_projection.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.03, size=(256, 128)), "F16"
        ),
        "language_model.model.layers.0.self_attn.q_proj.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.03, size=(128, 128)), "BF16"
        ),
        "language_model.model.layers.0.self_attn.k_proj.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.03, size=(64, 128)), "F32"
        ),
        "language_model.model.layers.0.self_attn.v_proj.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.03, size=(64, 128)), "BF16"
        ),
        "language_model.model.layers.0.self_attn.o_proj.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.03, size=(128, 128)), "BF16"
        ),
        "language_model.model.layers.0.per_layer_input_gate.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.03, size=(64, 128)), "BF16"
        ),
        "language_model.model.layers.0.per_layer_projection.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.03, size=(128, 64)), "BF16"
        ),
        # MLP projections (MUST remain unquantized bit-exact BF16!)
        "language_model.model.layers.0.mlp.gate_proj.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.04, size=(256, 128)), "BF16"
        ),
        "language_model.model.layers.0.mlp.up_proj.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.04, size=(256, 128)), "BF16"
        ),
        "language_model.model.layers.0.mlp.down_proj.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.04, size=(128, 256)), "BF16"
        ),
        # 1D norms & layer scalar
        "language_model.model.layers.0.input_layernorm.weight": array_to_raw_tensor(
            1.0 + rng.normal(0.0, 0.1, size=(128,)), "BF16"
        ),
        "language_model.model.layers.0.layer_scalar": array_to_raw_tensor(np.array([0.875], dtype=np.float32), "BF16"),
        "language_model.model.norm.weight": array_to_raw_tensor(1.0 + rng.normal(0.0, 0.1, size=(128,)), "F32"),
    }

    # Force multi-shard input safetensors on disk
    shards_in, tot_in = write_safetensors_dir(src_dir, raw_src, max_shard_bytes=120_000)
    assert len(shards_in) >= 2
    assert tot_in == sum(t.nbytes for t in raw_src.values())

    cfg_in = {
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 128,
            "vocab_size": v_full,
            "vocab_size_per_layer_input": v_full,
        },
        "vision_config": {"hidden_size": 128},
        "audio_config": {"hidden_size": 128},
    }
    (src_dir / "config.json").write_text(json.dumps(cfg_in, indent=2), encoding="utf-8")

    rep = pack_checkpoint_dir(
        src_dir,
        dst_dir,
        vocab_remap_npz=remap_npz,
        vocab_out=v_kept,
        max_shard_bytes=80_000,
    )

    expected_mm_bytes = sum(t.nbytes for k, t in raw_src.items() if is_multimodal_key(k) is not None)
    assert rep.strip_report.bytes_removed == expected_mm_bytes
    assert rep.vocab_out_size == v_kept
    assert len(rep.output_shards) >= 2

    # Reload purely on CPU from disk & verify every tensor
    deq, raw_out, cfg_out = reload_and_dequantize_dir_cpu(dst_dir)
    idx_out = json.loads((dst_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))

    # 1. Zero multimodal keys left anywhere
    for container in (raw_out, deq, idx_out["weight_map"]):
        assert not any(is_multimodal_key(k) is not None for k in container)

    # 2. Index metadata matches exact total byte count across output shards
    assert idx_out["metadata"]["total_size"] == sum(t.nbytes for t in raw_out.values())
    assert cfg_out["quantization"]["group_size"] == 64
    assert cfg_out["quantization"]["bits"] == 8
    assert cfg_out["quantization"]["mode"] == "affine"
    for mlp_k in rep.bf16_mlp_keys:
        assert cfg_out["quantization"][mlp_k[: -len(".weight")]] is False
    for qpref in rep.quantized_modules:
        assert cfg_out["quantization"][qpref] == {"group_size": 64, "bits": 8}

    # Vocab & Tokenizer seam: full 262K/v_full input embeddings MUST remain intact!
    assert cfg_out["text_config"]["vocab_size"] == v_full
    assert cfg_out["text_config"]["vocab_size_per_layer_input"] == v_full
    assert raw_out["language_model.model.embed_tokens.weight"].shape == (v_full, 128 // 4)
    assert raw_out["language_model.model.embed_tokens_per_layer.weight"].shape == (v_full, 256 // 4)
    assert (dst_dir / "vocab_remap.npz").exists()
    assert (dst_dir / "provenance.json").exists()
    prov = json.loads((dst_dir / "provenance.json").read_text(encoding="utf-8"))
    assert prov["vocab_policy"]["input_embeddings_full_rows_preserved"] is True
    assert prov["vocab_policy"]["full_vocab_size"] == v_full
    assert prov["vocab_policy"]["lm_head_vocab_out"] == v_kept
    assert set(prov["shard_sha256"].keys()) == set(rep.output_shards)

    # 3. Bit-exact BF16 MLP exclusion (no scales/biases!)
    for mlp_k in (
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
    ):
        prefix = mlp_k[: -len(".weight")]
        assert f"{prefix}.scales" not in raw_out
        assert f"{prefix}.biases" not in raw_out
        assert raw_out[mlp_k].dtype == "BF16"
        assert np.array_equal(raw_out[mlp_k].array, raw_src[mlp_k].array), f"MLP {mlp_k} was not bit-exact!"

    # 4. 8-bit g64 MLX layout + cos > 0.9999 for all quantized non-MLP tensors
    for prefix in rep.quantized_modules:
        w_t = raw_out[f"{prefix}.weight"]
        s_t = raw_out[f"{prefix}.scales"]
        b_t = raw_out[f"{prefix}.biases"]
        assert w_t.dtype == "U32"
        assert s_t.dtype == "BF16" and b_t.dtype == "BF16"
        assert w_t.shape[1] * 4 == s_t.shape[1] * 64
        assert b_t.shape == s_t.shape

        orig_f32 = raw_src[f"{prefix}.weight"].to_float32()
        assert w_t.shape[0] == orig_f32.shape[0]
        assert _cos_sim(deq[f"{prefix}.weight"], orig_f32) > 0.9999

    # 5. Raw header check on shard 0 confirms 8-byte little-endian uint64 length prefix
    shard0 = dst_dir / rep.output_shards[0]
    with shard0.open("rb") as fp:
        hdr_len = struct.unpack("<Q", fp.read(8))[0]
        hdr_json = json.loads(fp.read(hdr_len).decode("utf-8"))
        assert "__metadata__" in hdr_json


def test_real_150786_vocab_remap_preserves_262k_inputs_and_slices_lm_head() -> None:
    if not VOCAB_REMAP_150786.exists():
        pytest.skip(f"Missing {VOCAB_REMAP_150786}")
    kept_ids = load_kept_old_ids(vocab_remap_npz=VOCAB_REMAP_150786)
    assert kept_ids is not None
    assert kept_ids.shape == (DEFAULT_KEPT_VOCAB_SIZE,)
    assert kept_ids.dtype == np.int32

    rng = np.random.default_rng(123)
    emb_u16 = f32_to_bf16_u16(rng.normal(0.0, 0.03, size=(262144, 64)).astype(np.float32))
    raw = {
        "language_model.model.embed_tokens.weight": RawTensor("BF16", (262144, 64), emb_u16),
        "language_model.model.embed_tokens_per_layer.weight": RawTensor("BF16", (262144, 64), emb_u16),
        "language_model.model.layers.0.mlp.gate_proj.weight": array_to_raw_tensor(
            rng.normal(0.0, 0.03, size=(64, 64)), "BF16"
        ),
    }
    packed, rep = pack_checkpoint_tensors(raw, kept_old_ids=kept_ids, materialize_pruned_lm_head=True)
    assert rep.vocab_out_size == DEFAULT_KEPT_VOCAB_SIZE
    assert packed["language_model.model.embed_tokens.weight"].shape == (262144, 16)
    assert packed["language_model.model.embed_tokens_per_layer.weight"].shape == (262144, 16)
    assert packed["language_model.lm_head.weight"].shape == (DEFAULT_KEPT_VOCAB_SIZE, 16)
    assert packed["language_model.lm_head.scales"].shape == (DEFAULT_KEPT_VOCAB_SIZE, 1)
    assert rep.remapped_vocab_keys == ["language_model.lm_head.weight"]
    recon = dequantize_affine_8bit_u32(
        packed["language_model.lm_head.weight"],
        packed["language_model.lm_head.scales"],
        packed["language_model.lm_head.biases"],
        group_size=64,
    )
    ref = bf16_u16_to_f32(emb_u16[kept_ids])
    assert _cos_sim(recon, ref) > 0.9999

    # Also verify explicit untied lm_head.weight is sliced while embed_tokens stays full
    raw_untied = dict(raw)
    lm_u16 = f32_to_bf16_u16(rng.normal(0.0, 0.03, size=(262144, 64)).astype(np.float32))
    raw_untied["language_model.lm_head.weight"] = RawTensor("BF16", (262144, 64), lm_u16)
    packed_u, rep_u = pack_checkpoint_tensors(raw_untied, kept_old_ids=kept_ids)
    assert packed_u["language_model.model.embed_tokens.weight"].shape == (262144, 16)
    assert packed_u["language_model.lm_head.weight"].shape == (DEFAULT_KEPT_VOCAB_SIZE, 16)
    assert rep_u.remapped_vocab_keys == ["language_model.lm_head.weight"]


def test_fail_closed_rejects_non_float_u32_and_already_quantized(tmp_path: Path) -> None:
    u32_tensor = RawTensor("U32", (4, 4), np.arange(16, dtype=np.uint32).reshape(4, 4))
    with pytest.raises(TypeError, match="non-float dtype"):
        u32_tensor.to_float32()

    with pytest.raises(TypeError, match="non-float dtype"):
        pack_checkpoint_tensors({"language_model.model.layers.0.self_attn.q_proj.weight": u32_tensor})

    with pytest.raises(ValueError, match="already contains quantized"):
        pack_checkpoint_tensors(
            {
                "language_model.model.layers.0.self_attn.q_proj.scales": array_to_raw_tensor(
                    np.ones((64, 1), dtype=np.float32), "BF16"
                )
            }
        )

    # Already-quantized config.json in src_dir must fail BEFORE mutating pre-existing dst_dir
    src_q = tmp_path / "src_quant"
    dst = tmp_path / "dst_existing"
    src_q.mkdir()
    dst.mkdir()
    sentinel = dst / "keep_me.txt"
    sentinel.write_text("intact", encoding="utf-8")
    write_safetensors_dir(
        src_q,
        {"language_model.model.norm.weight": array_to_raw_tensor(np.ones((64,), dtype=np.float32), "BF16")},
    )
    (src_q / "config.json").write_text(
        json.dumps({"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="already contains `quantization`"):
        pack_checkpoint_dir(src_q, dst)
    assert sentinel.read_text(encoding="utf-8") == "intact"
    assert not list(tmp_path.glob("*.staging.*"))


def test_bf16_scale_floor_before_encoding_and_atomic_rollback(tmp_path: Path) -> None:
    # Ultra-tiny non-zero span that underflows BF16 scale divisor (~1e-42) + constant zero-span group
    w = np.zeros((2, 128), dtype=np.float32)
    w[0, :64] = 0.0
    w[0, 63] = np.float32(1e-42)  # raw_scale = ~3.9e-45 -> flushes to BF16 0.0 unless floored before q
    w[0, 64:] = 0.25
    w[1, :] = np.linspace(-0.5, 0.5, 128, dtype=np.float32)

    qw, qs, qb = quantize_affine_8bit_u32(w, group_size=64)
    scales_f32 = qs.to_float32()
    assert np.all(np.isfinite(scales_f32))
    assert np.all(scales_f32 > 0.0)
    assert qs.array[0, 0] != 0
    assert qs.array[0, 1] != 0
    deq = dequantize_affine_8bit_u32(qw, qs, qb, group_size=64)
    assert np.all(np.isfinite(deq))
    assert np.allclose(deq[0, 64:], 0.25, atol=1e-4)
    assert np.max(np.abs(deq[1] - w[1])) < 1e-2

    # Atomic rollback if shard 2 fails validation (cannot have U32 weight sneak through)
    src_bad = tmp_path / "src_bad_shard2"
    dst_good = tmp_path / "dst_good"
    dst_good.mkdir()
    (dst_good / "existing.json").write_text("{}", encoding="utf-8")
    write_safetensors_dir(
        src_bad,
        {
            "language_model.model.norm.weight": array_to_raw_tensor(np.ones((64,), dtype=np.float32), "BF16"),
            "language_model.model.layers.0.self_attn.q_proj.weight": RawTensor(
                "U32", (64, 16), np.ones((64, 16), dtype=np.uint32)
            ),
        },
        max_shard_bytes=200,
    )
    with pytest.raises(TypeError, match="non-float dtype"):
        pack_checkpoint_dir(src_bad, dst_good)
    assert (dst_good / "existing.json").read_text(encoding="utf-8") == "{}"
    assert not list(tmp_path.glob("*.staging.*"))


def test_real_gemma4_headers_strip_accounting_and_qat_u32_parity() -> None:
    if not BF16_HF_SNAPSHOT.exists():
        pytest.skip("Local Gemma-4 BF16 snapshot not cached")
    hdrs = inspect_safetensors_headers(BF16_HF_SNAPSHOT)
    mm_bytes = sum(v["nbytes"] for k, v in hdrs.items() if is_multimodal_key(k) is not None)
    assert mm_bytes == 951_458_176

    if QAT4_HF_SNAPSHOT.exists():
        qat_hdr, qat_base = read_safetensors_header(QAT4_HF_SNAPSHOT / "model.safetensors")
        w_info = qat_hdr["language_model.model.layers.0.mlp.gate_proj.weight"]
        s_info = qat_hdr["language_model.model.layers.0.mlp.gate_proj.scales"]
        b_info = qat_hdr["language_model.model.layers.0.mlp.gate_proj.biases"]
        assert w_info["dtype"] == "U32" and w_info["shape"] == [6144, 384]
        assert s_info["dtype"] == "BF16" and s_info["shape"] == [6144, 24]
        assert b_info["dtype"] == "BF16" and b_info["shape"] == [6144, 24]
        with (QAT4_HF_SNAPSHOT / "model.safetensors").open("rb") as fp:
            fp.seek(qat_base + w_info["data_offsets"][0])
            w_row0 = np.frombuffer(fp.read(384 * 4), dtype=np.uint32).reshape(1, 384)
            fp.seek(qat_base + s_info["data_offsets"][0])
            s_row0 = np.frombuffer(fp.read(24 * 2), dtype=np.uint16).reshape(1, 24)
            fp.seek(qat_base + b_info["data_offsets"][0])
            b_row0 = np.frombuffer(fp.read(24 * 2), dtype=np.uint16).reshape(1, 24)
        deq0 = dequantize_affine_8bit_u32(
            RawTensor("U32", (1, 384), w_row0),
            RawTensor("BF16", (1, 24), s_row0),
            RawTensor("BF16", (1, 24), b_row0),
            group_size=64,
        )
        assert deq0.shape == (1, 1536) and np.all(np.isfinite(deq0))


def test_packed_checkpoint_matches_fast_gemma4_model_contract_cpu(tmp_path: Path) -> None:
    """Verify packed safetensors + config.json satisfy FastGemma4Model loader & vocab contract."""
    from fast_gemma4.model import _compute_previous_kvs
    from fast_gemma4.tokenizer import FULL_VOCAB_SIZE, VocabRemapper

    tok_json = REPO_ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
    if not tok_json.exists() and BF16_HF_SNAPSHOT.exists():
        tok_json = BF16_HF_SNAPSHOT / "tokenizer.json"

    rng = np.random.default_rng(99)
    h_dim, ple_dim, mlp_int, d_swa, d_full = 1536, 256, 256, 256, 512
    layer_types = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]
    n_layers, n_shared = len(layer_types), 1
    first_shared = n_layers - n_shared

    src_dir = tmp_path / "src_contract"
    dst_dir = tmp_path / "dst_contract"
    src_dir.mkdir()

    # Keep test light in RAM by keeping 262,144 vocab on E2B? Use 512 unless testing FULL_VOCAB_SIZE headers;
    # validate exact FastGemma4Model key set, dtypes, shapes, and per-layer quantization overrides.
    v_test = 512
    raw_src: dict[str, RawTensor] = {
        "vision_tower.encoder.layers.0.weight": array_to_raw_tensor(rng.normal(0, 0.02, (64, 64)), "BF16"),
        "language_model.model.embed_tokens.weight": array_to_raw_tensor(rng.normal(0, 0.02, (v_test, h_dim)), "BF16"),
        "language_model.model.embed_tokens_per_layer.weight": array_to_raw_tensor(
            rng.normal(0, 0.02, (v_test, n_layers * ple_dim)), "BF16"
        ),
        "language_model.model.per_layer_model_projection.weight": array_to_raw_tensor(
            rng.normal(0, 0.02, (n_layers * ple_dim, h_dim)), "BF16"
        ),
        "language_model.model.per_layer_projection_norm.weight": array_to_raw_tensor(
            np.ones((ple_dim,), dtype=np.float32), "BF16"
        ),
        "language_model.model.norm.weight": array_to_raw_tensor(np.ones((h_dim,), dtype=np.float32), "BF16"),
    }
    for i, lt in enumerate(layer_types):
        lp = f"language_model.model.layers.{i}"
        d = d_full if lt == "full_attention" else d_swa
        raw_src[f"{lp}.self_attn.q_proj.weight"] = array_to_raw_tensor(rng.normal(0, 0.02, (8 * d, h_dim)), "BF16")
        if i < first_shared:
            raw_src[f"{lp}.self_attn.k_proj.weight"] = array_to_raw_tensor(rng.normal(0, 0.02, (d, h_dim)), "BF16")
            raw_src[f"{lp}.self_attn.v_proj.weight"] = array_to_raw_tensor(rng.normal(0, 0.02, (d, h_dim)), "BF16")
            raw_src[f"{lp}.self_attn.k_norm.weight"] = array_to_raw_tensor(np.ones((d,), dtype=np.float32), "BF16")
        raw_src[f"{lp}.self_attn.o_proj.weight"] = array_to_raw_tensor(rng.normal(0, 0.02, (h_dim, 8 * d)), "BF16")
        raw_src[f"{lp}.self_attn.q_norm.weight"] = array_to_raw_tensor(np.ones((d,), dtype=np.float32), "BF16")
        raw_src[f"{lp}.mlp.gate_proj.weight"] = array_to_raw_tensor(rng.normal(0, 0.02, (mlp_int, h_dim)), "BF16")
        raw_src[f"{lp}.mlp.up_proj.weight"] = array_to_raw_tensor(rng.normal(0, 0.02, (mlp_int, h_dim)), "BF16")
        raw_src[f"{lp}.mlp.down_proj.weight"] = array_to_raw_tensor(rng.normal(0, 0.02, (h_dim, mlp_int)), "BF16")
        raw_src[f"{lp}.per_layer_input_gate.weight"] = array_to_raw_tensor(rng.normal(0, 0.02, (ple_dim, h_dim)), "BF16")
        raw_src[f"{lp}.per_layer_projection.weight"] = array_to_raw_tensor(rng.normal(0, 0.02, (h_dim, ple_dim)), "BF16")
        for norm_n in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_per_layer_input_norm",
        ):
            raw_src[f"{lp}.{norm_n}.weight"] = array_to_raw_tensor(np.ones((h_dim,), dtype=np.float32), "BF16")
        raw_src[f"{lp}.layer_scalar"] = array_to_raw_tensor(np.array([0.9], dtype=np.float32), "BF16")

    write_safetensors_dir(src_dir, raw_src)
    cfg_in = {
        "model_type": "gemma4",
        "tie_word_embeddings": True,
        "text_config": {
            "hidden_size": h_dim,
            "intermediate_size": mlp_int,
            "num_hidden_layers": n_layers,
            "num_kv_shared_layers": n_shared,
            "layer_types": layer_types,
            "sliding_window": 512,
            "head_dim": d_swa,
            "global_head_dim": d_full,
            "hidden_size_per_layer_input": ple_dim,
            "vocab_size": v_test,
            "vocab_size_per_layer_input": v_test,
        },
        "vision_config": {"hidden_size": 768},
    }
    (src_dir / "config.json").write_text(json.dumps(cfg_in), encoding="utf-8")
    kept_ids = np.arange(0, 256, dtype=np.int32)
    pack_checkpoint_dir(src_dir, dst_dir, kept_old_ids=kept_ids)

    cfg_out = json.loads((dst_dir / "config.json").read_text(encoding="utf-8"))
    hdrs = inspect_safetensors_headers(dst_dir)
    tcfg = cfg_out["text_config"]
    qcfg = cfg_out["quantization"]

    assert _compute_previous_kvs(tcfg["layer_types"], tcfg["num_kv_shared_layers"]) == [0, 1, 2, 1]
    assert tcfg["vocab_size"] == v_test
    assert "vision_config" not in cfg_out
    assert hdrs["language_model.model.embed_tokens.weight"]["dtype"] == "U32"
    assert hdrs["language_model.model.embed_tokens.weight"]["shape"] == [v_test, h_dim // 4]
    assert hdrs["language_model.model.embed_tokens.scales"]["shape"] == [v_test, h_dim // 64]
    assert hdrs["language_model.model.embed_tokens_per_layer.weight"]["shape"] == [v_test, (n_layers * ple_dim) // 4]
    assert qcfg["language_model.model.embed_tokens"] == {"group_size": 64, "bits": 8}

    for i in range(n_layers):
        for mlp_p in (
            f"language_model.model.layers.{i}.mlp.gate_proj",
            f"language_model.model.layers.{i}.mlp.up_proj",
            f"language_model.model.layers.{i}.mlp.down_proj",
        ):
            assert qcfg[mlp_p] is False
            assert hdrs[f"{mlp_p}.weight"]["dtype"] == "BF16"
            assert f"{mlp_p}.scales" not in hdrs
        assert qcfg[f"language_model.model.layers.{i}.self_attn.q_proj"] == {"group_size": 64, "bits": 8}

    if VOCAB_REMAP_150786.exists() and tok_json.exists():
        rem = VocabRemapper.load_or_build(tok_json, VOCAB_REMAP_150786, target_size=DEFAULT_KEPT_VOCAB_SIZE)
        assert rem.vocab_size == DEFAULT_KEPT_VOCAB_SIZE
        assert int(np.max(rem.kept_old_ids)) < FULL_VOCAB_SIZE


def test_latin_extended_vocab162304_covers_accented_names_cpu(tmp_path: Path) -> None:
    """Verify Latin-Extended 162,304 vocab covers accented names with 0 misses and 1-to-1 token mapping."""
    from fast_gemma4.tokenizer import FastGemma4Tokenizer, LATIN_EXTENDED_KEPT_VOCAB_SIZE, VocabRemapper

    tok_json = REPO_ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
    if not tok_json.exists():
        pytest.skip(f"Missing {tok_json}")

    tok = FastGemma4Tokenizer(tok_json)
    rem_150k = VocabRemapper(tok_json, target_size=DEFAULT_KEPT_VOCAB_SIZE)
    assert rem_150k.vocab_size == 150786
    # Verify high-rank ASCII words like ' Markup' stay covered at 150,786
    _, misses_markup_150k = rem_150k.encode_remap(tok.encode(" Markup Agreed", add_special_tokens=False))
    assert misses_markup_150k == 0

    rem_162k_p = tmp_path / "vocab_remap.npz"
    rem_162k = VocabRemapper.load_or_build(tok_json, rem_162k_p, target_size=LATIN_EXTENDED_KEPT_VOCAB_SIZE)
    assert rem_162k.vocab_size == 162304
    assert rem_162k.vocab_size % 64 == 0

    probe_names = (
        '{"ps":[{"n":"Raúl Grijalva","r":"reviewer","e":"staff"},'
        '{"n":"Ben Ray Luján","r":"lead","e":"staff"},'
        '{"n":"Nydia Velázquez","r":"author","e":"person"},'
        '{"n":"Jesús García","r":"author","e":"person"}],'
        '"ct":"Technical Review ✓"}'
    )
    old_ids = tok.encode(probe_names, add_special_tokens=False)
    new_ids, misses_162k = rem_162k.encode_remap(old_ids)
    assert misses_162k == 0, f"Expected 0 misses on accented UTF-8 names at 162,304, got {misses_162k}"
    assert len(new_ids) == len(old_ids), "Expected 1-to-1 single-token mapping without UTF-8 byte expansion"
    assert tok.decode(rem_162k.decode_to_old(new_ids), skip_special_tokens=False) == probe_names


def test_load_safetensors_file_accepts_null_metadata_cpu(tmp_path: Path) -> None:
    """`mx.save_safetensors` writes `"__metadata__": null`; the fused LoRA staging shards arrive that way."""
    from fast_gemma4.packager import load_safetensors_file

    hdr = json.dumps(
        {"__metadata__": None, "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, separators=(",", ":")
    ).encode("utf-8")
    hdr += b" " * (-len(hdr) % 8)
    p = tmp_path / "null_meta.safetensors"
    p.write_bytes(struct.pack("<Q", len(hdr)) + hdr + np.asarray([1.0, 2.0], dtype=np.float32).tobytes())

    tensors, meta = load_safetensors_file(p)
    assert meta == {}
    assert list(tensors) == ["w"]
