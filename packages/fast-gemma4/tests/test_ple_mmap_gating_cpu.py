"""CPU tests for selective INT8 PLE mmap gating and on-demand sidecar caching."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import fast_gemma4
import fast_gemma4.engine as fg4_engine
import fast_gemma4.model as fg4_model
from fast_gemma4 import FastGemma4Engine, FastGemma4Model, resolve_paths, should_use_ple_mmap
from fast_gemma4.packager import RawTensor, array_to_raw_tensor, pack_checkpoint_dir, write_safetensors_dir


def _write_mini_tokenizer(dst_dir: Path, vocab_size: int = 128) -> Path:
    vocab: dict[str, int] = {
        "<pad>": 0,
        "<eos>": 1,
        "<bos>": 2,
        "<|tool_response>": 50,
        "<turn|>": 106,
    }
    for idx in range(vocab_size):
        if idx not in vocab.values():
            vocab[f"tok_{idx}"] = idx
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<pad>"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(
        [
            AddedToken("<pad>", special=True),
            AddedToken("<eos>", special=True),
            AddedToken("<bos>", special=True),
            AddedToken("<|tool_response>", special=True),
            AddedToken("<turn|>", special=True),
        ]
    )
    out = dst_dir / "tokenizer.json"
    tok.save(str(out))
    return out


def _build_synth_e2b_ckpt(
    ckpt_dir: Path,
    *,
    vocab_size: int = 128,
    hidden_size: int = 128,
    interm_size: int = 256,
    ple_dim: int = 32,
    seed: int = 23,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _write_mini_tokenizer(ckpt_dir, vocab_size=vocab_size)
    rng = np.random.default_rng(seed)

    def _norm_w(shape: tuple[int, ...], scale: float = 0.03) -> np.ndarray:
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    num_layers = 3
    num_kv_shared = 1
    hd = 64
    n_q = 4
    n_kv = 1
    f32_map: dict[str, np.ndarray] = {
        "language_model.model.embed_tokens.weight": _norm_w((vocab_size, hidden_size), 0.04),
        "language_model.model.embed_tokens_per_layer.weight": _norm_w((vocab_size, num_layers * ple_dim), 0.04),
        "language_model.model.per_layer_model_projection.weight": _norm_w((num_layers * ple_dim, hidden_size), 0.03),
        "language_model.model.per_layer_projection_norm.weight": np.ones((ple_dim,), dtype=np.float32),
        "language_model.model.norm.weight": np.ones((hidden_size,), dtype=np.float32),
    }
    layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
    for i, ltype in enumerate(layer_types):
        lp = f"language_model.model.layers.{i}"
        is_shared = i >= (num_layers - num_kv_shared)
        eff_interm = interm_size * (2 if is_shared else 1)
        f32_map[f"{lp}.input_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)
        f32_map[f"{lp}.post_attention_layernorm.weight"] = np.full((hidden_size,), 0.5, dtype=np.float32)
        f32_map[f"{lp}.pre_feedforward_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)
        f32_map[f"{lp}.post_feedforward_layernorm.weight"] = np.full((hidden_size,), 0.5, dtype=np.float32)
        f32_map[f"{lp}.post_per_layer_input_norm.weight"] = np.full((hidden_size,), 0.5, dtype=np.float32)
        f32_map[f"{lp}.layer_scalar"] = np.array([0.95], dtype=np.float32)
        f32_map[f"{lp}.self_attn.q_norm.weight"] = np.full((hd,), 0.5, dtype=np.float32)
        f32_map[f"{lp}.self_attn.q_proj.weight"] = _norm_w((n_q * hd, hidden_size))
        if not is_shared:
            f32_map[f"{lp}.self_attn.k_norm.weight"] = np.full((hd,), 0.5, dtype=np.float32)
            f32_map[f"{lp}.self_attn.k_proj.weight"] = _norm_w((n_kv * hd, hidden_size))
            f32_map[f"{lp}.self_attn.v_proj.weight"] = _norm_w((n_kv * hd, hidden_size))
        f32_map[f"{lp}.self_attn.o_proj.weight"] = _norm_w((hidden_size, n_q * hd))
        f32_map[f"{lp}.mlp.gate_proj.weight"] = _norm_w((eff_interm, hidden_size))
        f32_map[f"{lp}.mlp.up_proj.weight"] = _norm_w((eff_interm, hidden_size))
        f32_map[f"{lp}.mlp.down_proj.weight"] = _norm_w((hidden_size, eff_interm))
        f32_map[f"{lp}.per_layer_input_gate.weight"] = _norm_w((ple_dim, hidden_size))
        f32_map[f"{lp}.per_layer_projection.weight"] = _norm_w((hidden_size, ple_dim))

    raw_tensors: dict[str, RawTensor] = {k: array_to_raw_tensor(v, "BF16") for k, v in f32_map.items()}
    write_safetensors_dir(ckpt_dir, raw_tensors)
    cfg = {
        "model_type": "gemma4",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": hidden_size,
            "intermediate_size": interm_size,
            "num_hidden_layers": num_layers,
            "num_kv_shared_layers": num_kv_shared,
            "layer_types": layer_types,
            "num_attention_heads": n_q,
            "num_key_value_heads": n_kv,
            "head_dim": hd,
            "global_head_dim": hd,
            "hidden_size_per_layer_input": ple_dim,
            "vocab_size": vocab_size,
            "vocab_size_per_layer_input": vocab_size,
            "sliding_window": 64,
            "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6,
            "use_double_wide_mlp": True,
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
            },
        },
    }
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12))


def test_should_use_ple_mmap_policy_matrix() -> None:
    assert "should_use_ple_mmap" in fast_gemma4.__all__

    # Models without PLE never enable PLE mmap.
    assert (
        should_use_ple_mmap(
            ple_mmap=True,
            ple_dim=0,
            has_sidecar=True,
            has_inline_ple_weight=False,
            host_ram_gib=16.0,
        )
        is False
    )

    # Stripped checkpoints (sidecar only) require mmap even if caller asks for auto/False.
    for req in ("auto", False, True):
        assert (
            should_use_ple_mmap(
                ple_mmap=req,
                ple_dim=256,
                has_sidecar=True,
                has_inline_ple_weight=False,
                host_ram_gib=48.0,
                profile="throughput",
            )
            is True
        )

    # Explicit bool overrides auto when inline weights exist.
    assert (
        should_use_ple_mmap(
            ple_mmap=True,
            ple_dim=256,
            has_sidecar=False,
            has_inline_ple_weight=True,
            host_ram_gib=48.0,
            profile="throughput",
        )
        is True
    )
    assert (
        should_use_ple_mmap(
            ple_mmap=False,
            ple_dim=256,
            has_sidecar=False,
            has_inline_ple_weight=True,
            host_ram_gib=16.0,
            profile="interactive",
        )
        is False
    )

    with pytest.raises(ValueError, match="Unsupported ple_mmap"):
        should_use_ple_mmap(
            ple_mmap="bogus",
            ple_dim=256,
            has_sidecar=False,
            has_inline_ple_weight=True,
            host_ram_gib=24.0,
        )

    # 16 GB host: enable mmap on E2B across profiles to protect unified RAM headroom.
    for prof in ("auto", "interactive", "throughput"):
        assert (
            should_use_ple_mmap(
                ple_mmap="auto",
                ple_dim=256,
                has_sidecar=False,
                has_inline_ple_weight=True,
                host_ram_gib=16.0,
                hidden_size=1536,
                num_layers=35,
                ple_nbytes=1_321_205_760,
                profile=prof,
            )
            is True
        )

    # 24 GB Base M5/M4 + 4-bit E2B: keep in MLX RAM for auto/throughput; enable only for interactive.
    assert (
        should_use_ple_mmap(
            ple_mmap="auto",
            ple_dim=256,
            has_sidecar=False,
            has_inline_ple_weight=True,
            host_ram_gib=24.0,
            hidden_size=1536,
            num_layers=35,
            ple_nbytes=1_321_205_760,
            profile="auto",
        )
        is False
    )
    assert (
        should_use_ple_mmap(
            ple_mmap="auto",
            ple_dim=256,
            has_sidecar=False,
            has_inline_ple_weight=True,
            host_ram_gib=24.0,
            hidden_size=1536,
            num_layers=35,
            ple_nbytes=1_321_205_760,
            profile="throughput",
        )
        is False
    )
    assert (
        should_use_ple_mmap(
            ple_mmap="auto",
            ple_dim=256,
            has_sidecar=False,
            has_inline_ple_weight=True,
            host_ram_gib=24.0,
            hidden_size=1536,
            num_layers=35,
            ple_nbytes=1_321_205_760,
            profile="interactive",
        )
        is True
    )

    # 24 GB host + E4B+ / oversized BF16 PLE: enable across all profiles.
    for kwargs in (
        {"hidden_size": 2560, "num_layers": 35, "ple_nbytes": 1_500_000_000},
        {"hidden_size": 1536, "num_layers": 42, "ple_nbytes": 1_500_000_000},
        {"hidden_size": 1536, "num_layers": 35, "ple_nbytes": 2_000_000_000},
    ):
        assert (
            should_use_ple_mmap(
                ple_mmap="auto",
                ple_dim=256,
                has_sidecar=False,
                has_inline_ple_weight=True,
                host_ram_gib=24.0,
                profile="throughput",
                **kwargs,
            )
            is True
        )

    # >= 48 GB host: keep PLE in MLX RAM across E2B & E4B unless explicitly overridden.
    assert (
        should_use_ple_mmap(
            ple_mmap="auto",
            ple_dim=256,
            has_sidecar=False,
            has_inline_ple_weight=True,
            host_ram_gib=48.0,
            hidden_size=2560,
            num_layers=42,
            ple_nbytes=3_000_000_000,
            profile="interactive",
        )
        is False
    )


def test_engine_and_model_ple_mmap_kwargs_cpu(tmp_path: Path) -> None:
    assert inspect.signature(FastGemma4Model.__init__).parameters["ple_mmap"].default == "auto"
    assert inspect.signature(FastGemma4Model.__init__).parameters["ple_profile"].default == "auto"
    assert inspect.signature(FastGemma4Engine.__init__).parameters["ple_mmap"].default == "auto"

    with pytest.raises(ValueError, match="Unsupported ple_mmap"):
        FastGemma4Engine(home=tmp_path, ple_mmap="invalid", warmup=False)

    captured: dict[str, Any] = {}

    class _DummyModel:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.model_dir = tmp_path
            self.tokenizer_json = tmp_path / "tokenizer.json"
            self.hidden_size = 1536
            self.num_layers = 35
            self.n_kv_heads = 1
            self.enable_moe = False
            self.first_kv_shared = 15
            self.remapper = None

        def close(self) -> None:
            pass

    class _DummyTok:
        def __init__(self, _path: Path) -> None:
            pass

    with (
        patch.object(fg4_engine, "FastGemma4Model", _DummyModel),
        patch.object(fg4_engine, "FastGemma4Tokenizer", _DummyTok),
        patch.object(fg4_engine, "_select_auto_ane_ring_variant", return_value=("fp16", None)),
    ):
        eng = FastGemma4Engine(
            home=tmp_path,
            profile="interactive",
            ple_mmap=True,
            ane_ring_variant="fp16",
            mtp_drafter=False,
            warmup=False,
        )
        eng.close()

    assert captured["ple_mmap"] is True
    assert captured["ple_profile"] == "interactive"


def test_on_demand_ple_mmap_cache_and_parity_cpu(tmp_path: Path) -> None:
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        bf16_dir = tmp_path / "ckpt_bf16"
        q4_dir = tmp_path / "ckpt_q4"
        home_dir = tmp_path / "fast_gemma4_home"
        _build_synth_e2b_ckpt(bf16_dir)
        pack_checkpoint_dir(
            bf16_dir,
            q4_dir,
            quant_mode="qat_q4_0",
            ple_sidecar_int8=False,
            strip_multimodal=False,
        )

        paths = resolve_paths(home_dir)
        toks = mx.array([[2, 5, 11, 29]], dtype=mx.int32)

        with patch.object(fg4_model, "_detect_host_ram_gib", return_value=24.0):
            # 1) 24 GB + E2B + auto -> stays in RAM, does NOT create sidecar cache.
            m_ram = FastGemma4Model(paths=paths, model_dir=q4_dir, mlp_dtype="auto", ple_mmap="auto")
            assert m_ram.ple_mmap_enabled is False
            assert m_ram._ple_mmap_i8 is None
            assert m_ram.embed_tokens_per_layer is not None
            cache_dir = paths.home / "ple_sidecars" / paths.checkpoint_ane_key(m_ram.model_dir)
            assert not cache_dir.exists()

            # 2) 24 GB + E2B + profile="interactive" -> builds cached sidecar under paths.home without modifying q4_dir.
            m_mmap = FastGemma4Model(
                paths=paths,
                model_dir=q4_dir,
                mlp_dtype="auto",
                ple_mmap="auto",
                ple_profile="interactive",
            )
            assert m_mmap.ple_mmap_enabled is True
            assert isinstance(m_mmap._ple_mmap_i8, np.memmap)
            assert m_mmap.embed_tokens_per_layer is None
            assert not (q4_dir / "embed_tokens_per_layer_i8.json").exists()
            assert (cache_dir / "embed_tokens_per_layer_i8.json").is_file()

            # 3) Second load must reuse the cached sidecar without invoking _build_ple_sidecar_to_dir.
            with patch.object(
                FastGemma4Model,
                "_build_ple_sidecar_to_dir",
                side_effect=AssertionError("expected cached PLE sidecar reuse"),
            ):
                m_cached = FastGemma4Model(
                    paths=paths,
                    model_dir=q4_dir,
                    mlp_dtype="auto",
                    ple_mmap="auto",
                    ple_profile="interactive",
                )
            assert m_cached.ple_mmap_enabled is True
            assert isinstance(m_cached._ple_mmap_i8, np.memmap)

            # 4) Parity on _compute_embed_and_pli between in-RAM Q4 and INT8 mmap sidecar.
            with mx.stream(mx.cpu):
                h_ram, pli_ram = m_ram._compute_embed_and_pli(toks)
                h_mm, pli_mm = m_mmap._compute_embed_and_pli(toks)
                mx.eval(h_ram, pli_ram, h_mm, pli_mm)
            assert pli_ram is not None and pli_mm is not None
            assert _cosine(np.asarray(h_ram.astype(mx.float32)), np.asarray(h_mm.astype(mx.float32))) > 0.99999
            assert _cosine(np.asarray(pli_ram.astype(mx.float32)), np.asarray(pli_mm.astype(mx.float32))) > 0.999

            # 5) BF16 (non-quantized) inline PLE checkpoint with explicit ple_mmap=True also works cleanly.
            m_bf16_ram = FastGemma4Model(paths=paths, model_dir=bf16_dir, mlp_dtype="auto", ple_mmap=False)
            m_bf16_mm = FastGemma4Model(paths=paths, model_dir=bf16_dir, mlp_dtype="auto", ple_mmap=True)
            assert m_bf16_mm.ple_mmap_enabled is True
            assert isinstance(m_bf16_mm._ple_mmap_i8, np.memmap)
            with mx.stream(mx.cpu):
                _, pli_b_ram = m_bf16_ram._compute_embed_and_pli(toks)
                _, pli_b_mm = m_bf16_mm._compute_embed_and_pli(toks)
                mx.eval(pli_b_ram, pli_b_mm)
            assert pli_b_ram is not None and pli_b_mm is not None
            assert _cosine(np.asarray(pli_b_ram.astype(mx.float32)), np.asarray(pli_b_mm.astype(mx.float32))) > 0.999
    finally:
        mx.set_default_device(prev_dev)
