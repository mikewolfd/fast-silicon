"""TDD CPU + locked GPU tests for `fast-gemma4` v2 roadmap deliverables (M-P0.1 & M-P2.1)."""

from __future__ import annotations

import contextlib
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterator
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from fast_gemma4.cli import _make_engine, build_parser
import fast_gemma4.engine as fg4_engine
from fast_gemma4.engine import FastGemma4Engine
from fast_gemma4.kernels import (
    fused_softcap_argmax_bf16,
    multi_row_gemv_exact_bf16,
    multi_row_qmv_exact_bf16,
)
from fast_gemma4.lora_accel import (
    compute_effective_ckpt_tokens,
    dequantize_prequantized_shard_tensors,
    inspect_lora_base_checkpoint,
    load_text_only_lora_base_model,
    quantize_frozen_trunk_8bit,
    strip_quantization_from_config_dict,
)
import fast_gemma4.model as fg4_model
from fast_gemma4.model import (
    FastGemma4Model,
    should_use_bf16_mpp_for_mlp,
    singleton_quantized_linear,
)
from fast_gemma4.mtp_drafter import (
    DEFAULT_MTP_DRAFTER_REPO,
    DEFAULT_QAT_MTP_DRAFTER_REPO,
    Gemma4MtpDrafter,
    resolve_mtp_drafter_dir,
    run_decode_mtp,
    select_mtp_drafter_repo,
)
from fast_gemma4.packager import (
    RawTensor,
    array_to_raw_tensor,
    pack_checkpoint_dir,
    quantize_q4_0_sym_u32,
    reload_and_dequantize_dir_cpu,
    write_safetensors_dir,
)
from fast_gemma4.paths import resolve_paths
from fast_gemma4.speculative import run_decode_rolling
from fast_gemma4.tokenizer import GEMMA4_EOS_TOKEN_IDS, FastGemma4Tokenizer

HW_LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")


def _ancestor_holds_lock(lock_path: Path) -> bool:
    try:
        anc: set[int] = set()
        pid = os.getpid()
        while pid > 1 and pid not in anc:
            anc.add(pid)
            ppid_s = subprocess.check_output(["ps", "-o", "ppid=", "-p", str(pid)], text=True).strip()
            pid = int(ppid_s) if ppid_s else 0
        holders = {
            int(x)
            for x in subprocess.check_output(["lsof", "-t", str(lock_path)], text=True).split()
            if x.strip().isdigit()
        }
        return bool((anc - {os.getpid()}) & holders)
    except Exception:
        return False


@contextlib.contextmanager
def _gpu_lock() -> Iterator[None]:
    HW_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(HW_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o666)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            if not _ancestor_holds_lock(HW_LOCK_PATH):
                fcntl.flock(fd, fcntl.LOCK_EX)
                acquired = True
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12))


def _write_mini_tokenizer(dst_dir: Path, vocab_size: int = 256) -> Path:
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


def _build_synthetic_e2b_bf16_dir(
    ckpt_dir: Path,
    *,
    vocab_size: int = 256,
    hidden_size: int = 256,
    interm_size: int = 512,
    ple_dim: int = 64,
    include_multimodal: bool = False,
    seed: int = 19,
) -> dict[str, np.ndarray]:
    """Create a tiny valid 3-layer Gemma-4 checkpoint (`[S, F, S-shared]`) on disk."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _write_mini_tokenizer(ckpt_dir, vocab_size=vocab_size)
    rng = np.random.default_rng(seed)

    # Quantize synthetic source weights onto exact q4_0 g32 grid so QAT tests can verify near-lossless roundtrip.
    def _qat_grid(shape: tuple[int, ...], scale: float = 0.02) -> np.ndarray:
        arr = rng.normal(0.0, scale, size=shape).astype(np.float32)
        if len(shape) == 2 and shape[1] % 32 == 0:
            g = arr.reshape(shape[0], shape[1] // 32, 32)
            d = np.maximum(np.max(np.abs(g), axis=-1) / 8.0, 1e-4).astype(np.float32)
            q = np.clip(np.rint(g / d[:, :, None]), -8.0, 7.0)
            return (q * d[:, :, None]).reshape(shape).astype(np.float32)
        return arr

    num_layers = 3
    num_kv_shared = 1
    swa_hd = 128
    full_hd = 128
    n_q = 4
    n_kv = 1

    f32_map: dict[str, np.ndarray] = {
        "language_model.model.embed_tokens.weight": _qat_grid((vocab_size, hidden_size), 0.03),
        "language_model.model.embed_tokens_per_layer.weight": _qat_grid((vocab_size, num_layers * ple_dim), 0.03),
        "language_model.model.per_layer_model_projection.weight": _qat_grid((num_layers * ple_dim, hidden_size), 0.02),
        "language_model.model.per_layer_projection_norm.weight": np.ones((ple_dim,), dtype=np.float32),
        "language_model.model.norm.weight": np.ones((hidden_size,), dtype=np.float32),
    }
    if include_multimodal:
        f32_map["vision_tower.encoder.layers.0.weight"] = rng.normal(0.0, 0.02, size=(64, 64)).astype(np.float32)
        f32_map["audio_tower.layers.0.weight"] = rng.normal(0.0, 0.02, size=(64, 64)).astype(np.float32)

    layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
    for i, ltype in enumerate(layer_types):
        lp = f"language_model.model.layers.{i}"
        hd = full_hd if ltype == "full_attention" else swa_hd
        is_shared = i >= (num_layers - num_kv_shared)
        eff_interm = interm_size * (2 if is_shared else 1)
        f32_map[f"{lp}.input_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)
        f32_map[f"{lp}.post_attention_layernorm.weight"] = np.full((hidden_size,), 0.5, dtype=np.float32)
        f32_map[f"{lp}.pre_feedforward_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)
        f32_map[f"{lp}.post_feedforward_layernorm.weight"] = np.full((hidden_size,), 0.5, dtype=np.float32)
        f32_map[f"{lp}.post_per_layer_input_norm.weight"] = np.full((hidden_size,), 0.5, dtype=np.float32)
        f32_map[f"{lp}.layer_scalar"] = np.array([0.95], dtype=np.float32)
        f32_map[f"{lp}.self_attn.q_norm.weight"] = np.full((hd,), 0.5, dtype=np.float32)
        f32_map[f"{lp}.self_attn.q_proj.weight"] = _qat_grid((n_q * hd, hidden_size))
        if not is_shared:
            f32_map[f"{lp}.self_attn.k_norm.weight"] = np.full((hd,), 0.5, dtype=np.float32)
            f32_map[f"{lp}.self_attn.k_proj.weight"] = _qat_grid((n_kv * hd, hidden_size))
            f32_map[f"{lp}.self_attn.v_proj.weight"] = _qat_grid((n_kv * hd, hidden_size))
        f32_map[f"{lp}.self_attn.o_proj.weight"] = _qat_grid((hidden_size, n_q * hd))
        f32_map[f"{lp}.mlp.gate_proj.weight"] = _qat_grid((eff_interm, hidden_size))
        f32_map[f"{lp}.mlp.up_proj.weight"] = _qat_grid((eff_interm, hidden_size))
        f32_map[f"{lp}.mlp.down_proj.weight"] = _qat_grid((hidden_size, eff_interm))
        f32_map[f"{lp}.per_layer_input_gate.weight"] = _qat_grid((ple_dim, hidden_size))
        f32_map[f"{lp}.per_layer_projection.weight"] = _qat_grid((hidden_size, ple_dim))

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
            "head_dim": swa_hd,
            "global_head_dim": full_hd,
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
        "vision_config": {"hidden_size": 64},
        "audio_config": {"hidden_size": 64},
    }
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return f32_map


# ---------------------------------------------------------------------------
# Deliverable 1: mlp_dtype="auto" + QAT q4_0 g32 + PLE INT8 mmap + qat-mobile
# ---------------------------------------------------------------------------


def test_mlp_dtype_auto_policy_and_defaults_cpu() -> None:
    assert inspect.signature(FastGemma4Model.__init__).parameters["mlp_dtype"].default == "auto"
    assert inspect.signature(FastGemma4Engine.__init__).parameters["mlp_dtype"].default == "auto"
    parser = build_parser()
    args = parser.parse_args(["generate", "hello"])
    assert args.mlp_dtype == "auto"

    # Apple10 NAX (M5): ALWAYS keep QuantizedLinear across B=1..128 regardless of RAM or bits.
    for b in (1, 4, 16, 32, 64, 128):
        for bits in (2, 4, 8, None):
            assert (
                should_use_bf16_mpp_for_mlp(
                    mlp_dtype="auto",
                    gpu_family="apple10",
                    host_ram_gib=48.0,
                    batch_size=b,
                    mlp_bits=bits,
                )
                is False
            )

    # Apple9 (M4): keep QuantizedLinear whenever B < 64, RAM < 48 GB, or weights are <= 4-bit (q4_0 / qat-mobile).
    for b in (1, 8, 16, 32, 63):
        assert (
            should_use_bf16_mpp_for_mlp(
                mlp_dtype="auto",
                gpu_family="apple9",
                host_ram_gib=48.0,
                batch_size=b,
                mlp_bits=8,
            )
            is False
        )
    for b in (64, 128):
        for low_bits in (2, 4):
            assert (
                should_use_bf16_mpp_for_mlp(
                    mlp_dtype="auto",
                    gpu_family="apple9",
                    host_ram_gib=48.0,
                    batch_size=b,
                    mlp_bits=low_bits,
                )
                is False
            )
        assert (
            should_use_bf16_mpp_for_mlp(
                mlp_dtype="auto",
                gpu_family="apple9",
                host_ram_gib=24.0,
                batch_size=b,
                mlp_bits=8,
            )
            is False
        )
        # Promote to BF16 MPP ONLY when B >= 64 on >= 48 GB Apple9 and weights are 8-bit / BF16-backed!
        assert (
            should_use_bf16_mpp_for_mlp(
                mlp_dtype="auto",
                gpu_family="apple9",
                host_ram_gib=48.0,
                batch_size=b,
                mlp_bits=8,
            )
            is True
        )
        assert (
            should_use_bf16_mpp_for_mlp(
                mlp_dtype="auto",
                gpu_family="apple9",
                host_ram_gib=48.0,
                batch_size=b,
                mlp_bits=None,
            )
            is True
        )


def test_packager_qat_q4_0_g32_sym_and_ple_int8_mmap_sidecar_cpu(tmp_path: Path) -> None:
    src_dir = tmp_path / "google--gemma-4-E2B-it-qat-q4_0-unquantized"
    dst_dir = tmp_path / "packed_qat_q4_0"
    f32_src = _build_synthetic_e2b_bf16_dir(src_dir, include_multimodal=True)

    # Verify standalone symmetric q4_0 g32 helper
    qw, qs, qb = quantize_q4_0_sym_u32(f32_src["language_model.model.layers.0.mlp.gate_proj.weight"], group_size=32)
    assert qw.dtype == "U32" and qw.shape == (512, 256 // 8)
    assert qs.dtype == "BF16" and qs.shape == (512, 256 // 32)
    assert qb.dtype == "BF16" and qb.shape == (512, 256 // 32)
    assert np.allclose(qb.to_float32(), -8.0 * qs.to_float32(), rtol=1e-3, atol=1e-5)

    rep = pack_checkpoint_dir(
        src_dir,
        dst_dir,
        quant_mode="qat_q4_0",
        ple_sidecar_int8=True,
        vocab_out=None,
    )
    assert rep.bf16_mlp_keys == []
    assert (dst_dir / "embed_tokens_per_layer_i8.bin").is_file()
    assert (dst_dir / "embed_tokens_per_layer_scales.bin").is_file()
    assert (dst_dir / "embed_tokens_per_layer_i8.json").is_file()

    deq, raw_out, cfg_out = reload_and_dequantize_dir_cpu(dst_dir)
    qcfg = cfg_out["quantization"]
    assert qcfg["group_size"] == 32
    assert qcfg["bits"] == 4
    # MLPs quantized to 4-bit g32 symmetric
    for l_idx in range(3):
        mlp_pref = f"language_model.model.layers.{l_idx}.mlp.gate_proj"
        assert qcfg[mlp_pref] == {"group_size": 32, "bits": 4}
        assert _cosine(deq[f"{mlp_pref}.weight"], f32_src[f"{mlp_pref}.weight"]) > 0.9995
        # PLE gate & projection kept at 8-bit!
        for ple_leaf in ("per_layer_input_gate", "per_layer_projection"):
            ple_pref = f"language_model.model.layers.{l_idx}.{ple_leaf}"
            assert qcfg[ple_pref]["bits"] == 8
            assert _cosine(deq[f"{ple_pref}.weight"], f32_src[f"{ple_pref}.weight"]) > 0.9999

    # Load packed QAT + INT8 mmap PLE sidecar on CPU stream (zero GPU dispatch)
    with mx.stream(mx.cpu):
        m_packed = FastGemma4Model(paths=resolve_paths(tmp_path), model_dir=dst_dir, mlp_dtype="auto")
        assert isinstance(m_packed.layers[0]["gate_up_proj"], nn.QuantizedLinear)
        assert m_packed.layers[0]["gate_up_proj"].bits == 4
        assert m_packed.layers[0]["gate_up_proj"].group_size == 32
        assert m_packed.layers[0]["ple_gate"].bits == 8
        assert m_packed.layers[0]["ple_proj"].bits == 8
        assert getattr(m_packed, "_ple_mmap_i8", None) is not None
        assert isinstance(m_packed._ple_mmap_i8, np.memmap)

        toks = mx.array([[2, 10, 15, 22]], dtype=mx.int32)
        h_p, pli_p = m_packed._compute_embed_and_pli(toks)
        h14_p, pli14_p = m_packed._compute_embed_and_pli_early14(toks)
        mx.eval(h_p, pli_p, h14_p, pli14_p)
        n_early = m_packed._n_early_pli_layers
        assert pli_p is not None and pli_p.shape == (1, 4, 3, 64)
        assert pli14_p is not None and pli14_p.shape == (1, 4, n_early, 64)
        assert np.allclose(
            np.asarray(pli14_p.astype(mx.float32)),
            np.asarray(pli_p[:, :, :n_early, :].astype(mx.float32)),
            atol=2e-2,
        )

        # Also verify direct load of unpacked `*-qat-q4_0-unquantized` auto-quantizes onto q4_0 g32 + 8-bit PLE!
        m_direct = FastGemma4Model(paths=resolve_paths(tmp_path), model_dir=src_dir, mlp_dtype="auto")
        assert isinstance(m_direct.layers[0]["gate_up_proj"], nn.QuantizedLinear)
        assert m_direct.layers[0]["gate_up_proj"].bits == 4
        assert m_direct.layers[0]["gate_up_proj"].group_size == 32
        assert m_direct.layers[0]["ple_gate"].bits == 8
        assert m_direct.layers[0]["ple_proj"].bits == 8


def test_load_qat_mobile_2bit_4bit_8bit_hierarchy_cpu(tmp_path: Path) -> None:
    qat_mob_dir = tmp_path / "gemma-4-E2B-it-qat-mobile"
    f32_src = _build_synthetic_e2b_bf16_dir(qat_mob_dir, include_multimodal=False)

    # Convert synthetic weights into MLX mixed 2/4/8-bit matching `mlx-community/gemma-4-E2B-it-qat-mobile`
    with mx.stream(mx.cpu):
        out_mx: dict[str, mx.array] = {}
        for k, arr in f32_src.items():
            mx_arr = mx.array(arr, dtype=mx.bfloat16)
            if not k.endswith(".weight") or arr.ndim != 2 or "per_layer_model_projection" in k or "norm" in k:
                out_mx[k] = mx_arr
                continue
            prefix = k[: -len(".weight")]
            if "embed_tokens." in k and "per_layer" not in k:
                bits = 2
            elif ".layers.2.mlp." in k:  # shared-KV layer -> 2-bit!
                bits = 2
            elif "per_layer_input_gate" in k or "per_layer_projection" in k:
                bits = 8
            else:
                bits = 4
            wq, sc, bs = mx.quantize(mx_arr, group_size=64, bits=bits, mode="affine")
            out_mx[f"{prefix}.weight"] = wq
            out_mx[f"{prefix}.scales"] = sc
            out_mx[f"{prefix}.biases"] = bs
        mx.eval(list(out_mx.values()))
        mx.save_safetensors(str(qat_mob_dir / "model.safetensors"), out_mx)

    # Remove index & top-level `quantization`, leaving ONLY `quantization_config.module_quant_configs` regexes!
    (qat_mob_dir / "model.safetensors.index.json").unlink(missing_ok=True)
    cfg = json.loads((qat_mob_dir / "config.json").read_text(encoding="utf-8"))
    cfg.pop("quantization", None)
    cfg["quantization_config"] = {
        "module_quant_configs": {
            "^lm_head$": {"num_bits": 2},
            r"language_model\.embed_tokens$": {"num_bits": 2},
            r"language_model\.embed_tokens_per_layer$": {"num_bits": 4},
            r"language_model\.layers\.[0-1]\.mlp\.": {"num_bits": 4},
            r"language_model\.layers\.\d+\.mlp\.": {"num_bits": 2},
            r"language_model\.layers\.\d+\.per_layer_input_gate$": {"num_bits": 8},
            r"language_model\.layers\.\d+\.per_layer_projection$": {"num_bits": 8},
            r"language_model\.layers\.\d+\.self_attn\.": {"num_bits": 4},
        },
        "modules_to_not_convert": ["per_layer_model_projection"],
        "num_bits": 4,
        "quant_method": "gemma",
        "quantize_embeddings": True,
    }
    (qat_mob_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    with mx.stream(mx.cpu):
        model = FastGemma4Model(paths=resolve_paths(tmp_path), model_dir=qat_mob_dir, mlp_dtype="auto")
        assert isinstance(model.embed_tokens, nn.QuantizedEmbedding) and model.embed_tokens.bits == 2
        assert isinstance(model.embed_tokens_per_layer, nn.QuantizedEmbedding) and model.embed_tokens_per_layer.bits == 4
        assert isinstance(model.layers[0]["gate_up_proj"], nn.QuantizedLinear) and model.layers[0]["gate_up_proj"].bits == 4
        assert isinstance(model.layers[2]["gate_up_proj"], nn.QuantizedLinear) and model.layers[2]["gate_up_proj"].bits == 2
        assert isinstance(model.layers[2]["down_proj"], nn.QuantizedLinear) and model.layers[2]["down_proj"].bits == 2
        assert isinstance(model.layers[0]["ple_gate"], nn.QuantizedLinear) and model.layers[0]["ple_gate"].bits == 8
        assert isinstance(model.layers[0]["ple_proj"], nn.QuantizedLinear) and model.layers[0]["ple_proj"].bits == 8


# ---------------------------------------------------------------------------
# Deliverable 2: Cache hygiene, 262K vocab default, kv_i8="auto", EOS IDs, CLI
# ---------------------------------------------------------------------------


def test_engine_cache_limit_256mb_kv_i8_auto_eos_and_cli_cpu(tmp_path: Path) -> None:
    assert GEMMA4_EOS_TOKEN_IDS == frozenset({1, 106, 50})
    tok_p = _write_mini_tokenizer(tmp_path, vocab_size=256)
    tok = FastGemma4Tokenizer(tok_p)
    assert {1, 106, 50}.issubset(tok.stop_token_ids)

    assert inspect.signature(FastGemma4Engine.__init__).parameters["kv_i8"].default == "auto"
    assert inspect.signature(FastGemma4Model.__init__).parameters["kv_i8"].default == "auto"

    cache_calls: list[int] = []

    def _fake_set_cl(limit: int) -> int:
        cache_calls.append(int(limit))
        return 999_999

    class _DummyPaths:
        lib_ane_prefix = tmp_path / "missing.dylib"
        vocab_remap_npz = tmp_path / "vocab_remap_150786.npz"

        def resolve_ane_bundle(self, *_a: object, **_kw: object) -> None:
            return None

    class _DummyModel:
        hidden_size = 1536
        num_layers = 35
        n_kv_heads = 1
        first_kv_shared = 15
        window_size = 512
        kv_i8 = False
        kv_i8_mode = "auto"
        model_dir = tmp_path / "ckpt"
        tokenizer_json = tok_p
        new_to_old = np.arange(256, dtype=np.int32)
        vocab_size = 256

    with (
        patch.object(fg4_engine, "resolve_paths", return_value=_DummyPaths()),
        patch.object(fg4_engine, "FastGemma4Model", return_value=_DummyModel()),
        patch.object(fg4_engine, "FastGemma4Tokenizer", return_value=tok),
        patch.object(mx, "set_cache_limit", side_effect=_fake_set_cl, create=True),
        patch.object(mx, "set_wired_limit", return_value=0, create=True),
        patch.object(mx, "device_info", return_value={"max_recommended_working_set_size": 0}, create=True),
    ):
        eng = FastGemma4Engine(warmup=False)
        assert cache_calls == [256 * 1024 * 1024]
        assert eng.resolve_call_kv_i8(has_schema=False) is False
        assert eng.resolve_call_kv_i8(has_schema=True) is True
        eng.close()
        assert cache_calls == [256 * 1024 * 1024, 999_999]

    # CLI `_make_engine` defaults `ane_ring_variant="auto"` and `kv_i8="auto"`!
    parser = build_parser()
    gen_args = parser.parse_args(["generate", "hi", "--no-warmup"])
    captured_kwargs: dict[str, Any] = {}

    def _capture_engine(**kw: Any) -> Any:
        captured_kwargs.update(kw)
        return object()

    with patch("fast_gemma4.cli.FastGemma4Engine", side_effect=_capture_engine):
        _make_engine(gen_args)
    assert captured_kwargs["ane_ring_variant"] == "auto"
    assert captured_kwargs["mlp_dtype"] == "auto"
    assert captured_kwargs["kv_i8"] == "auto"


def test_vocab_full_default_vs_allow_english_vocab_prune_cpu(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "stock_ckpt"
    _build_synthetic_e2b_bf16_dir(ckpt_dir, vocab_size=256)
    paths = resolve_paths(tmp_path)
    paths.vocab_remap_npz.parent.mkdir(parents=True, exist_ok=True)
    kept_global = np.arange(0, 128, dtype=np.int32)
    np.savez(paths.vocab_remap_npz, kept_old_ids=kept_global)

    with mx.stream(mx.cpu):
        # 1. Default vocab_out=None keeps full 256 vocab even when global vocab_remap_150786.npz exists!
        m_full = FastGemma4Model(paths=paths, model_dir=ckpt_dir, vocab_out=None)
        assert m_full.vocab_size == 256
        assert m_full.remapper is None

        # 2. Explicit allow_english_vocab_prune=True activates global remap when vocab_out=None!
        m_pruned = FastGemma4Model(
            paths=paths, model_dir=ckpt_dir, vocab_out=None, allow_english_vocab_prune=True
        )
        assert m_pruned.vocab_size == 128
        assert m_pruned.remapper is not None


# ---------------------------------------------------------------------------
# Deliverable 3: Zero-Copy GPU Pre-Mask Telemetry in fused_softcap_argmax_bf16
# ---------------------------------------------------------------------------


def test_fused_softcap_argmax_telemetry_and_valid_mask_gpu() -> None:
    rng = np.random.default_rng(2026)
    B, V, stride = 6, 4096, 4160
    softcap = 30.0
    raw_np = rng.normal(0.0, 18.0, size=(B, 1, stride)).astype(np.float32)
    # Row 4: exact tie on top-2 -> top2_margin == 0.0
    raw_np[4, 0, :] = -20.0
    raw_np[4, 0, 17] = 12.5
    raw_np[4, 0, 99] = 12.5
    # Row 5: NaN row -> argmax == -1, telemetry NaN
    raw_np[5, 0, 123] = np.nan

    # Random valid bitmask uint32[B, (V + 31)//32] (~15% valid tokens)
    words = (V + 31) // 32
    valid_bool = rng.random((B, V)) < 0.15
    valid_bool[:, 7] = True  # guarantee >= 1 valid token per row
    mask_u32 = np.zeros((B, words), dtype=np.uint32)
    for j in range(V):
        mask_u32[:, j >> 5] |= (valid_bool[:, j].astype(np.uint32) << np.uint32(j & 31))

    with _gpu_lock():
        raw_mx = mx.array(raw_np, dtype=mx.bfloat16)
        mask_mx = mx.array(mask_u32, dtype=mx.uint32)

        am_plain = fused_softcap_argmax_bf16(raw_mx, softcap, vocab_limit=V)
        am_telem, telem_3 = fused_softcap_argmax_bf16(
            raw_mx, softcap, vocab_limit=V, collect_telemetry=True
        )
        am_masked, telem_4 = fused_softcap_argmax_bf16(
            raw_mx, softcap, vocab_limit=V, collect_telemetry=True, valid_mask_u32=mask_mx
        )
        # Compute exact softcapped FP32 values matching the kernel's precision for FP64 ground truth
        inv_sc = 1.0 / float(softcap)
        z_fp32 = mx.tanh(raw_mx[:, 0, :V].astype(mx.float32) * inv_sc) * float(softcap)
        mx.eval(am_plain, am_telem, telem_3, am_masked, telem_4, z_fp32)

        am_p_list = am_plain.tolist()
        assert am_telem.tolist() == am_p_list
        assert am_masked.tolist() == am_p_list
        assert am_p_list[4] == 17
        assert am_p_list[5] == -1

        t3_np = np.asarray(telem_3)
        t4_np = np.asarray(telem_4)
        z_np = np.asarray(z_fp32).astype(np.float64)
        assert t3_np.shape == (B, 3)
        assert t4_np.shape == (B, 4)
        assert np.all(np.isnan(t3_np[5])) and np.all(np.isnan(t4_np[5]))

        for b in range(5):
            zb = z_np[b]
            m = np.max(zb)
            ex = np.exp(zb - m)
            s = np.sum(ex)
            lse_ref = m + math.log(s)
            p = ex / s
            h_ref = -float(np.sum(p * (zb - lse_ref)))
            top2 = np.partition(zb, -2)[-2:]
            margin_ref = float(top2[1] - top2[0])

            zb_val = zb[valid_bool[b]]
            m_v = np.max(zb_val)
            lse_val_ref = float(m_v + math.log(np.sum(np.exp(zb_val - m_v))))

            assert abs(float(t3_np[b, 0]) - lse_ref) < 2e-3
            assert abs(float(t3_np[b, 1]) - h_ref) < 2e-3
            assert abs(float(t3_np[b, 2]) - margin_ref) < 2e-3
            assert np.allclose(t4_np[b, :3], t3_np[b, :3], atol=1e-5)
            assert abs(float(t4_np[b, 3]) - lse_val_ref) < 2e-3
        assert float(t3_np[4, 2]) == 0.0


# ---------------------------------------------------------------------------
# Deliverable 4: MLX 0.32.4 SDPA, Exact MTP (4 bugs), 26B-A4B MoE, train_lora
# ---------------------------------------------------------------------------


def test_mtp_exact_singleton_qlinear_anchor_pos_and_qat_pairing_gpu(tmp_path: Path) -> None:
    # Bug (d): QAT target auto-pairs QAT assistant repo
    qat_stub = type("_T", (), {"model_dir": tmp_path / "gemma-4-E2B-it-qat-q4_0", "is_qat_checkpoint": True})()
    reg_stub = type("_T", (), {"model_dir": tmp_path / "gemma-4-E2B-it-MLX-4bit", "is_qat_checkpoint": False})()
    assert select_mtp_drafter_repo(qat_stub) == DEFAULT_QAT_MTP_DRAFTER_REPO
    assert select_mtp_drafter_repo(reg_stub) == DEFAULT_MTP_DRAFTER_REPO

    with _gpu_lock():
        # Bug (a): singleton_quantized_linear matches L_q=1 QuantizedLinear bit-for-bit (0.0 max diff)!
        mx.random.seed(77)
        x = mx.random.normal((2, 4, 256)).astype(mx.bfloat16)
        lin_dense = nn.Linear(256, 512, bias=False)
        lin_dense.weight = (mx.random.normal((512, 256)) * 0.03).astype(mx.bfloat16)
        qlin_32 = nn.QuantizedLinear.from_linear(lin_dense, group_size=32, bits=4, mode="affine")
        qlin_mxfp4 = nn.QuantizedLinear.from_linear(lin_dense, group_size=32, bits=4, mode="mxfp4")
        mx.eval(x, qlin_32.parameters(), qlin_mxfp4.parameters())

        for qlin in (qlin_32, qlin_mxfp4):
            got = singleton_quantized_linear(qlin, x)
            assert got is not None
            ref_rows = [
                mx.concatenate([qlin(x[b : b + 1, t : t + 1]) for t in range(4)], axis=1)
                for b in range(2)
            ]
            ref = mx.concatenate(ref_rows, axis=0)
            mx.eval(got, ref)
            assert np.array_equal(np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32)))

        # Bug (c): draft_k holds off_rope == off_cur across all K steps & fixes SWA off_kv window!
        # Build synthetic E2B-shaped mini target + 4-layer drafter to verify exact parity under BOTH kv_i8=False and kv_i8=True!
        ckpt_dir = tmp_path / "mini_e2b_q4"
        src_dir = tmp_path / "mini_e2b_src"
        _build_synthetic_e2b_bf16_dir(src_dir, vocab_size=256, hidden_size=256, interm_size=512, ple_dim=64)
        pack_checkpoint_dir(src_dir, ckpt_dir, quant_mode="qat_q4_0", vocab_out=None)

        # Build matching synthetic 4-layer MTP drafter for hidden_size=256
        drafter_dir = tmp_path / "mini_drafter"
        drafter_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(33)
        d_dim = 64
        dr_weights: dict[str, mx.array] = {
            "pre_projection.weight": mx.array(rng.normal(0.0, 0.02, (d_dim, 512)), dtype=mx.bfloat16),
            "post_projection.weight": mx.array(rng.normal(0.0, 0.02, (256, d_dim)), dtype=mx.bfloat16),
            "model.norm.weight": mx.ones((d_dim,), dtype=mx.bfloat16),
            "masked_embedding.centroids.weight": mx.array(rng.normal(0.0, 0.05, (16, d_dim)), dtype=mx.bfloat16),
            "masked_embedding.token_ordering": mx.arange(256, dtype=mx.int32),
            "model.embed_tokens.weight": mx.array(rng.normal(0.0, 0.05, (256, d_dim)), dtype=mx.bfloat16),
        }
        for i in range(4):
            lp = f"model.layers.{i}"
            hd = 128
            dr_weights[f"{lp}.input_layernorm.weight"] = mx.ones((d_dim,), dtype=mx.bfloat16)
            dr_weights[f"{lp}.self_attn.q_norm.weight"] = mx.full((hd,), 0.5, dtype=mx.bfloat16)
            dr_weights[f"{lp}.self_attn.q_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (4 * hd, d_dim)), dtype=mx.bfloat16)
            dr_weights[f"{lp}.self_attn.o_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (d_dim, 4 * hd)), dtype=mx.bfloat16)
            dr_weights[f"{lp}.post_attention_layernorm.weight"] = mx.full((d_dim,), 0.5, dtype=mx.bfloat16)
            dr_weights[f"{lp}.pre_feedforward_layernorm.weight"] = mx.ones((d_dim,), dtype=mx.bfloat16)
            dr_weights[f"{lp}.mlp.gate_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (128, d_dim)), dtype=mx.bfloat16)
            dr_weights[f"{lp}.mlp.up_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (128, d_dim)), dtype=mx.bfloat16)
            dr_weights[f"{lp}.mlp.down_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (d_dim, 128)), dtype=mx.bfloat16)
            dr_weights[f"{lp}.post_feedforward_layernorm.weight"] = mx.full((d_dim,), 0.5, dtype=mx.bfloat16)
            dr_weights[f"{lp}.layer_scalar"] = mx.array([0.95], dtype=mx.bfloat16)
        mx.eval(list(dr_weights.values()))
        mx.save_safetensors(str(drafter_dir / "model.safetensors"), dr_weights)
        (drafter_dir / "config.json").write_text("{}", encoding="utf-8")

        # Bug (b) + (a) + (c): Verify 100% exact greedy parity for BOTH kv_i8=False AND kv_i8=True!
        for kv_flag in (False, True):
            eng = FastGemma4Engine(
                home=tmp_path,
                model_dir=ckpt_dir,
                default_mode="gpu",
                kv_i8=kv_flag,
                mtp_drafter=drafter_dir,
                warmup=False,
            )
            try:
                rope_arrs: list[mx.array] = []
                orig_step = eng.mtp_drafter.draft_step  # type: ignore[union-attr]

                def _spy_step(last_toks, last_h, cache, off_kv, off_rope, **kw):
                    rope_arrs.append(off_rope)
                    return orig_step(last_toks, last_h, cache, off_kv, off_rope, **kw)

                with patch.object(eng.mtp_drafter, "draft_step", side_effect=_spy_step):
                    r_ar = eng.generate_batch(
                        ["tok_10 tok_11 tok_12 tok_13"],
                        max_tokens=16,
                        temperature=0.0,
                        mode="gpu",
                        speculative=False,
                        use_chat_template=False,
                    )
                    r_mtp = eng.generate_batch(
                        ["tok_10 tok_11 tok_12 tok_13"],
                        max_tokens=16,
                        temperature=0.0,
                        mode="gpu",
                        speculative=None,
                        use_chat_template=False,
                    )
                mx.eval(*rope_arrs)
                rope_seen = [r.tolist() for r in rope_arrs]
                assert r_mtp.stats.get("speculative_backend") == "mtp"
                assert r_mtp.stats["kv_i8"] is kv_flag
                assert r_ar.stats["kv_i8"] is kv_flag
                assert r_mtp.items[0].token_ids == r_ar.items[0].token_ids
                # Verify first round's K draft steps all held anchor off_rope constant!
                assert len(rope_seen) >= 3
                assert rope_seen[0] == rope_seen[1] == rope_seen[2]
            finally:
                eng.close()


def test_gemma4_26b_a4b_moe_ple0_keqv_mxfp4_and_sdpa_delegation_gpu(tmp_path: Path) -> None:
    """Verify PLE=0, attention_k_eq_v without wk duplication, Router+SwitchGLU (mxfp4), and MLX SDPA delegation."""
    moe_dir = tmp_path / "gemma-4-26b-a4b-it-mxfp4"
    moe_dir.mkdir(parents=True, exist_ok=True)
    _write_mini_tokenizer(moe_dir, vocab_size=256)

    H = 256
    interm_dense = 512
    moe_interm = 128
    num_experts = 8  # also exercises REAP pruned expert counts (<128)
    top_k = 2
    n_q = 8
    n_kv_swa = 4
    n_kv_full = 2
    swa_hd = 128
    full_hd = 128
    layer_types = ["sliding_attention", "full_attention"]

    with _gpu_lock():
        mx.random.seed(91)
        weights: dict[str, mx.array] = {}

        def _put_mxfp4(prefix: str, w_bf16: mx.array) -> None:
            wq, sc = mx.quantize(w_bf16, group_size=32, bits=4, mode="mxfp4")
            weights[f"{prefix}.weight"] = wq
            weights[f"{prefix}.scales"] = sc

        emb_w = (mx.random.normal((256, H)) * 0.03).astype(mx.bfloat16)
        _put_mxfp4("language_model.model.embed_tokens", emb_w)
        weights["language_model.model.norm.weight"] = mx.ones((H,), dtype=mx.bfloat16)

        layer_refs: list[dict[str, Any]] = []
        for i, ltype in enumerate(layer_types):
            lp = f"language_model.model.layers.{i}"
            is_full = ltype == "full_attention"
            hd = full_hd if is_full else swa_hd
            n_kv = n_kv_full if is_full else n_kv_swa

            weights[f"{lp}.input_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
            weights[f"{lp}.post_attention_layernorm.weight"] = mx.full((H,), 0.5, dtype=mx.bfloat16)
            weights[f"{lp}.pre_feedforward_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
            weights[f"{lp}.post_feedforward_layernorm_1.weight"] = mx.full((H,), 0.5, dtype=mx.bfloat16)
            weights[f"{lp}.pre_feedforward_layernorm_2.weight"] = mx.ones((H,), dtype=mx.bfloat16)
            weights[f"{lp}.post_feedforward_layernorm_2.weight"] = mx.full((H,), 0.5, dtype=mx.bfloat16)
            weights[f"{lp}.post_feedforward_layernorm.weight"] = mx.full((H,), 0.5, dtype=mx.bfloat16)
            weights[f"{lp}.layer_scalar"] = mx.array([0.9], dtype=mx.bfloat16)
            weights[f"{lp}.self_attn.q_norm.weight"] = mx.full((hd,), 0.5, dtype=mx.bfloat16)
            weights[f"{lp}.self_attn.k_norm.weight"] = mx.full((hd,), 0.5, dtype=mx.bfloat16)

            wq = (mx.random.normal((n_q * hd, H)) * 0.02).astype(mx.bfloat16)
            wk = (mx.random.normal((n_kv * hd, H)) * 0.02).astype(mx.bfloat16)
            wv = None if is_full else (mx.random.normal((n_kv * hd, H)) * 0.02).astype(mx.bfloat16)
            wo = (mx.random.normal((H, n_q * hd)) * 0.02).astype(mx.bfloat16)
            _put_mxfp4(f"{lp}.self_attn.q_proj", wq)
            _put_mxfp4(f"{lp}.self_attn.k_proj", wk)
            if wv is not None:
                _put_mxfp4(f"{lp}.self_attn.v_proj", wv)
            _put_mxfp4(f"{lp}.self_attn.o_proj", wo)

            wg = (mx.random.normal((interm_dense, H)) * 0.02).astype(mx.bfloat16)
            wu = (mx.random.normal((interm_dense, H)) * 0.02).astype(mx.bfloat16)
            wd = (mx.random.normal((H, interm_dense)) * 0.02).astype(mx.bfloat16)
            _put_mxfp4(f"{lp}.mlp.gate_proj", wg)
            _put_mxfp4(f"{lp}.mlp.up_proj", wu)
            _put_mxfp4(f"{lp}.mlp.down_proj", wd)

            # Router (8-bit affine on proj, matching mlx_lm quant_predicate!)
            r_proj = (mx.random.normal((num_experts, H)) * 0.05).astype(mx.bfloat16)
            rq, rsc, rbs = mx.quantize(r_proj, group_size=64, bits=8, mode="affine")
            weights[f"{lp}.router.proj.weight"] = rq
            weights[f"{lp}.router.proj.scales"] = rsc
            weights[f"{lp}.router.proj.biases"] = rbs
            weights[f"{lp}.router.scale"] = mx.full((H,), 1.1, dtype=mx.bfloat16)
            weights[f"{lp}.router.per_expert_scale"] = mx.full((num_experts,), 0.8, dtype=mx.bfloat16)

            # Test BOTH SwitchGLU layouts: layer 0 = `experts.switch_glu.*`, layer 1 = fused `experts.gate_up_proj`!
            eg = (mx.random.normal((num_experts, moe_interm, H)) * 0.02).astype(mx.bfloat16)
            eu = (mx.random.normal((num_experts, moe_interm, H)) * 0.02).astype(mx.bfloat16)
            ed = (mx.random.normal((num_experts, H, moe_interm)) * 0.02).astype(mx.bfloat16)
            if i == 0:
                _put_mxfp4(f"{lp}.experts.switch_glu.gate_proj", eg)
                _put_mxfp4(f"{lp}.experts.switch_glu.up_proj", eu)
                _put_mxfp4(f"{lp}.experts.switch_glu.down_proj", ed)
            else:
                egu = mx.concatenate([eg, eu], axis=1)
                _put_mxfp4(f"{lp}.experts.gate_up_proj", egu)
                _put_mxfp4(f"{lp}.experts.down_proj", ed)
            layer_refs.append({"is_full": is_full, "n_kv": n_kv, "hd": hd})

        mx.eval(list(weights.values()))
        mx.save_safetensors(str(moe_dir / "model.safetensors"), weights)

        qcfg: dict[str, Any] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        for i in range(2):
            qcfg[f"language_model.model.layers.{i}.router.proj"] = {"group_size": 64, "bits": 8, "mode": "affine"}
        cfg = {
            "model_type": "gemma4",
            "tie_word_embeddings": True,
            "quantization": qcfg,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": H,
                "intermediate_size": interm_dense,
                "num_hidden_layers": 2,
                "num_kv_shared_layers": 0,
                "layer_types": layer_types,
                "num_attention_heads": n_q,
                "num_key_value_heads": n_kv_swa,
                "num_global_key_value_heads": n_kv_full,
                "attention_k_eq_v": True,
                "head_dim": swa_hd,
                "global_head_dim": full_hd,
                "hidden_size_per_layer_input": 0,
                "enable_moe_block": True,
                "num_experts": num_experts,
                "top_k_experts": top_k,
                "moe_intermediate_size": moe_interm,
                "vocab_size": 256,
                "sliding_window": 1024,
                "final_logit_softcapping": 30.0,
                "rms_norm_eps": 1e-6,
                "use_double_wide_mlp": False,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0},
                    "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
                },
            },
        }
        (moe_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        model = FastGemma4Model(paths=resolve_paths(tmp_path), model_dir=moe_dir)
        assert model.ple_dim == 0
        assert model.enable_moe is True
        # Verify global k_eq_v layer did NOT duplicate wk inside qkv_proj!
        qkv_l1 = model.layers[1]["qkv_proj"]
        assert int(qkv_l1.weight.shape[0]) == (n_q + n_kv_full) * full_hd

        # Verify L_q > 1 routes through mx.fast.scaled_dot_product_attention AND matches sequential L_q=1!
        pkv = model.prefill_prefix([2, 5, 6])
        c_multi = model.make_batch_cache(pkv, batch_size=1, doc_cap_full=64, swa_ring_cap=1024)
        c_seq = model.make_batch_cache(pkv, batch_size=1, doc_cap_full=64, swa_ring_cap=1024)

        sdpa_calls = 0
        orig_sdpa = mx.fast.scaled_dot_product_attention

        def _count_sdpa(*a, **kw):
            nonlocal sdpa_calls
            sdpa_calls += 1
            return orig_sdpa(*a, **kw)

        suf = [10, 11, 12, 13]
        with patch.object(mx.fast, "scaled_dot_product_attention", side_effect=_count_sdpa):
            lg_multi = model.forward_step_ragged(
                mx.array([suf], dtype=mx.int32), c_multi, mx.array([3], dtype=mx.int32)
            )
            mx.eval(lg_multi)
        assert sdpa_calls == 2  # Both layers delegated L_q=4 SDPA to mx.fast.scaled_dot_product_attention!

        seq_lgs = []
        for idx_t, tok_id in enumerate(suf):
            seq_lgs.append(
                model.forward_step_ragged(
                    mx.array([[tok_id]], dtype=mx.int32), c_seq, mx.array([3 + idx_t], dtype=mx.int32)
                )
            )
        lg_seq = mx.concatenate(seq_lgs, axis=1)
        mx.eval(lg_seq)
        assert _cosine(np.asarray(lg_multi.astype(mx.float32)), np.asarray(lg_seq.astype(mx.float32))) > 0.999

        # Also verify FastGemma4Engine rolling generation works end-to-end with sliding_window=1024 & PLE=0!
        eng = FastGemma4Engine(home=tmp_path, model_dir=moe_dir, mtp_drafter=False, warmup=True)
        try:
            res = eng.generate_batch(["tok_7 tok_8", "tok_7 tok_9"], max_tokens=4, use_chat_template=False)
            assert len(res.items) == 2
        finally:
            eng.close()


def test_train_lora_ckpt_tokens_quantize_embed_and_prequantized_load_cpu(tmp_path: Path) -> None:
    # 1. Stop raising ckpt_tokens=4096 on --qlora-8bit: 512 on <=24 GB, 1280 on >=36 GB!
    assert compute_effective_ckpt_tokens(None, qlora_8bit=True, host_ram_gib=24.0) == 512
    assert compute_effective_ckpt_tokens(None, qlora_8bit=True, host_ram_gib=16.0) == 512
    assert compute_effective_ckpt_tokens(None, qlora_8bit=True, host_ram_gib=48.0) == 1280
    assert compute_effective_ckpt_tokens(None, qlora_8bit=True, host_ram_gib=36.0) == 1280
    assert compute_effective_ckpt_tokens(None, qlora_8bit=False, host_ram_gib=48.0) == 2200
    assert compute_effective_ckpt_tokens(768, qlora_8bit=True, host_ram_gib=24.0) == 768

    # 2. quantize_frozen_trunk_8bit defaults quantize_embed_tokens=True and skips already-quantized layers!
    assert inspect.signature(quantize_frozen_trunk_8bit).parameters["quantize_embed_tokens"].default is True

    with mx.stream(mx.cpu):
        class _DummyTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(64, 64)
                self.embed_tokens_per_layer = nn.Embedding(64, 128)
                self.pre_q = nn.QuantizedLinear(64, 64, bias=False, group_size=32, bits=4, mode="affine")
                self.dense = nn.Linear(64, 64, bias=False)

        tm = _DummyTM()
        stats = quantize_frozen_trunk_8bit(tm, group_size=64, bits=8)
        assert stats["embedding_quantized"] == 2
        assert stats["linear_quantized"] == 1
        assert isinstance(tm.embed_tokens, nn.QuantizedEmbedding)
        assert isinstance(tm.pre_q, nn.QuantizedLinear) and tm.pre_q.bits == 4  # never double-quantized!

    # 3. inspect_lora_base_checkpoint detects pre-stripped / pre-quantized vs multimodal BF16!
    mm_dir = tmp_path / "mm_bf16"
    pq_dir = tmp_path / "pq_text"
    _build_synthetic_e2b_bf16_dir(mm_dir, include_multimodal=True)
    pack_checkpoint_dir(mm_dir, pq_dir, quant_mode="qat_q4_0", vocab_out=None)

    info_mm = inspect_lora_base_checkpoint(mm_dir)
    assert info_mm["is_prequantized"] is False
    assert info_mm["has_multimodal_tensors"] is True

    info_pq = inspect_lora_base_checkpoint(pq_dir)
    assert info_pq["is_prequantized"] is True
    assert info_pq["has_multimodal_tensors"] is False

    # 4. load_text_only_lora_base_model strips vision/audio config before loader runs and loads prequantized directly!
    seen_cfgs: list[dict[str, Any]] = []
    seen_dirs: list[Path] = []

    def _fake_loader(model_path: Path, *, lazy: bool = False, strict: bool = False) -> tuple[nn.Module, dict[str, Any]]:
        seen_dirs.append(Path(model_path))
        cfg_d = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        seen_cfgs.append(cfg_d)
        return nn.Linear(16, 16), cfg_d

    _, info_mm_loaded = load_text_only_lora_base_model(mm_dir, load_model_fn=_fake_loader)
    assert info_mm_loaded["has_multimodal_tensors"] is True
    assert "vision_config" not in seen_cfgs[-1] and "audio_config" not in seen_cfgs[-1]
    assert seen_dirs[-1] != mm_dir

    _, info_pq_loaded = load_text_only_lora_base_model(pq_dir, load_model_fn=_fake_loader)
    assert info_pq_loaded["is_prequantized"] is True
    assert seen_dirs[-1] == pq_dir


def test_mtp_prepermuted_embed_gather_mm_and_unordered_embeddings_gpu(tmp_path: Path) -> None:
    """Verify #347 item 3: (a) load-time embed_perm_f16 / embed_perm_t + gather_mm scan; (b) use_ordered_embeddings=False."""
    ckpt_src = tmp_path / "target_src"
    ckpt_dir = tmp_path / "target_q4"
    _build_synthetic_e2b_bf16_dir(ckpt_src, vocab_size=256, hidden_size=256, interm_size=512, ple_dim=64)
    pack_checkpoint_dir(ckpt_src, ckpt_dir, quant_mode="qat_q4_0", vocab_out=None)

    rng = np.random.default_rng(108)
    d_dim = 64

    def _make_base_drafter_weights() -> dict[str, mx.array]:
        w: dict[str, mx.array] = {
            "pre_projection.weight": mx.array(rng.normal(0.0, 0.02, (d_dim, 512)), dtype=mx.bfloat16),
            "post_projection.weight": mx.array(rng.normal(0.0, 0.02, (256, d_dim)), dtype=mx.bfloat16),
            "model.norm.weight": mx.ones((d_dim,), dtype=mx.bfloat16),
            "model.embed_tokens.weight": mx.array(rng.normal(0.0, 0.05, (256, d_dim)), dtype=mx.bfloat16),
        }
        for i in range(4):
            lp = f"model.layers.{i}"
            hd = 128
            w[f"{lp}.input_layernorm.weight"] = mx.ones((d_dim,), dtype=mx.bfloat16)
            w[f"{lp}.self_attn.q_norm.weight"] = mx.full((hd,), 0.5, dtype=mx.bfloat16)
            w[f"{lp}.self_attn.q_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (4 * hd, d_dim)), dtype=mx.bfloat16)
            w[f"{lp}.self_attn.o_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (d_dim, 4 * hd)), dtype=mx.bfloat16)
            w[f"{lp}.post_attention_layernorm.weight"] = mx.full((d_dim,), 0.5, dtype=mx.bfloat16)
            w[f"{lp}.pre_feedforward_layernorm.weight"] = mx.ones((d_dim,), dtype=mx.bfloat16)
            w[f"{lp}.mlp.gate_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (128, d_dim)), dtype=mx.bfloat16)
            w[f"{lp}.mlp.up_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (128, d_dim)), dtype=mx.bfloat16)
            w[f"{lp}.mlp.down_proj.weight"] = mx.array(rng.normal(0.0, 0.02, (d_dim, 128)), dtype=mx.bfloat16)
            w[f"{lp}.post_feedforward_layernorm.weight"] = mx.full((d_dim,), 0.5, dtype=mx.bfloat16)
            w[f"{lp}.layer_scalar"] = mx.array([0.95], dtype=mx.bfloat16)
        return w

    with _gpu_lock():
        target = FastGemma4Model(paths=resolve_paths(tmp_path), model_dir=ckpt_dir, mlp_dtype="auto")

        # (a) Ordered embeddings with nontrivial permutation -> pre-permuted `embed_perm_f16` & `embed_perm_t`!
        ord_dir = tmp_path / "drafter_ordered"
        ord_dir.mkdir()
        perm = rng.permutation(256).astype(np.int32)
        w_ord = _make_base_drafter_weights()
        w_ord["masked_embedding.centroids.weight"] = mx.array(rng.normal(0.0, 0.05, (16, d_dim)), dtype=mx.bfloat16)
        w_ord["masked_embedding.token_ordering"] = mx.array(perm, dtype=mx.int32)
        mx.eval(list(w_ord.values()))
        mx.save_safetensors(str(ord_dir / "model.safetensors"), w_ord)
        (ord_dir / "config.json").write_text(json.dumps({"use_ordered_embeddings": True}), encoding="utf-8")

        dr_ord = Gemma4MtpDrafter(target, ord_dir, top_k=4, local_files_only=True)
        assert dr_ord.use_ordered_embeddings is True
        assert dr_ord.embed_perm_f16 is not None and dr_ord.embed_perm_f16.dtype == mx.float16
        assert dr_ord.embed_perm_t is not None and dr_ord.embed_perm_t.shape == (16, d_dim, 16)

        h_probe = mx.array(rng.normal(0.0, 1.0, (2, d_dim)), dtype=mx.bfloat16)
        gather_mm_calls = 0
        orig_gmm = mx.gather_mm

        def _spy_gmm(*a, **kw):
            nonlocal gather_mm_calls
            gather_mm_calls += 1
            return orig_gmm(*a, **kw)

        with patch.object(mx, "gather_mm", side_effect=_spy_gmm):
            tok_gpu = dr_ord.draft_scan_top_k(h_probe, k=4)
            mx.eval(tok_gpu)
        assert gather_mm_calls == 1
        with mx.stream(mx.cpu):
            tok_cpu = dr_ord.draft_scan_top_k(h_probe, k=4)
            mx.eval(tok_cpu)
        assert tok_gpu.tolist() == tok_cpu.tolist()

        # (b) Unordered embeddings (`26B-A4B` / `31B` assistants: `use_ordered_embeddings=False`, no centroids)!
        unord_dir = tmp_path / "drafter_unordered"
        unord_dir.mkdir()
        w_unord = _make_base_drafter_weights()
        mx.eval(list(w_unord.values()))
        mx.save_safetensors(str(unord_dir / "model.safetensors"), w_unord)
        (unord_dir / "config.json").write_text(json.dumps({"use_ordered_embeddings": False}), encoding="utf-8")

        dr_unord = Gemma4MtpDrafter(target, unord_dir, top_k=4, local_files_only=True)
        assert dr_unord.use_ordered_embeddings is False
        assert dr_unord.centroids_w is None
        tok_unord = dr_unord.draft_scan_top_k(h_probe, k=4)
        mx.eval(tok_unord)
        assert tok_unord.shape == (2,)
        assert all(0 <= int(t) < 256 for t in tok_unord.tolist())


def test_remediation_batch_b_gate_init_rollback_prequant_fuse_and_drafter_fallback_cpu(tmp_path: Path) -> None:
    """Verify #365 CPU items: B=1 L_q=128 never promotes BF16 MLP, init rollback, prequant fuse, QAT drafter fallback."""
    # 1. Pre-quantized Step-2 shard dequantization + config quantization stripping -> pack_checkpoint_dir succeeds!
    src_dir = tmp_path / "src_bf16"
    pq_dir = tmp_path / "pq_base"
    staged_dir = tmp_path / "staged_bf16"
    repacked_dir = tmp_path / "repacked_out"
    _build_synthetic_e2b_bf16_dir(src_dir, include_multimodal=True)
    pack_checkpoint_dir(src_dir, pq_dir, quant_mode="qat_q4_0", vocab_out=None)

    with mx.stream(mx.cpu):
        pq_cfg = json.loads((pq_dir / "config.json").read_text(encoding="utf-8"))
        shard_in = dict(mx.load(str(pq_dir / "model.safetensors")))
        shard_out, fused_n = dequantize_prequantized_shard_tensors(shard_in, pq_cfg)
        assert fused_n == 0
        assert not any(k.endswith(".scales") or k.endswith(".biases") for k in shard_out)
        staged_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(staged_dir / "model.safetensors"), shard_out)
        clean_cfg = strip_quantization_from_config_dict(pq_cfg)
        assert "quantization" not in clean_cfg and "quantization_config" not in clean_cfg
        (staged_dir / "config.json").write_text(json.dumps(clean_cfg, indent=2), encoding="utf-8")
        (staged_dir / "tokenizer.json").write_bytes((pq_dir / "tokenizer.json").read_bytes())
        rep = pack_checkpoint_dir(staged_dir, repacked_dir, vocab_out=None)
        assert len(rep.quantized_modules) > 0

        # 2. _run_mlp with mlp_dtype="auto" checks batch concurrency B (not M = B * L_q)!
        m = FastGemma4Model(paths=resolve_paths(tmp_path), model_dir=repacked_dir, mlp_dtype="auto")
        m._gpu_family = "apple9"
        m._host_ram_gib = 48.0
        m._mpp_bf16_supported = True
        m.use_mpp = True
        # Repacked 8bit_bf16_mlp has BF16 MLPs; test with an 8-bit QuantizedLinear MLP layer!
        lin_gu = nn.Linear(256, 1024, bias=False)
        lin_dn = nn.Linear(512, 256, bias=False)
        ld_q8: dict[str, Any] = {
            "gate_up_proj": nn.QuantizedLinear.from_linear(lin_gu, group_size=64, bits=8, mode="affine"),
            "down_proj": nn.QuantizedLinear.from_linear(lin_dn, group_size=64, bits=8, mode="affine"),
            "mlp_bits": 8,
        }
        # B=1, L_q=128 -> M=128 >= 64, BUT B=1 < 64 -> MUST NOT lazily promote!
        x_b1_l128 = mx.zeros((1, 128, 256), dtype=mx.bfloat16)
        _ = m._run_mlp(x_b1_l128, ld_q8, allow_mpp=True)
        assert "w_g" not in ld_q8
        # B=64, L_q=1 -> B=64 >= 64 on Apple9 48 GB 8-bit -> promotes, and release_lazy_bf16_mlp_weights clears it!
        with patch.object(fg4_model, "mpp_fused_geglu_bf16", side_effect=lambda x, wg, wu, wgu, **_k: mx.zeros((64, 1, 512), dtype=mx.bfloat16)):
            _ = m._run_mlp(mx.zeros((64, 1, 256), dtype=mx.bfloat16), ld_q8, allow_mpp=True)
        assert "w_g" in ld_q8
        m.layers = [ld_q8]
        assert m.release_lazy_bf16_mlp_weights() == 1
        assert "w_g" not in ld_q8 and "w_dn" not in ld_q8

        # 3. export_early14_f16 raises clear ValueError when ple_dim == 0
        m.ple_dim = 0
        with pytest.raises(ValueError, match="ple_dim > 0"):
            m.export_early14_f16(tmp_path / "unused.npz")

    # 4. FastGemma4Engine.__init__ rolls back set_cache_limit / set_wired_limit on init failure!
    cl_calls: list[int] = []
    wl_calls: list[int] = []
    with (
        patch.object(mx, "set_cache_limit", side_effect=lambda v: (cl_calls.append(int(v)), 777)[1], create=True),
        patch.object(mx, "set_wired_limit", side_effect=lambda v: (wl_calls.append(int(v)), 888)[1], create=True),
        patch.object(mx, "device_info", return_value={"max_recommended_working_set_size": 4096}, create=True),
        patch.object(fg4_engine, "FastGemma4Model", side_effect=ValueError("boom")),
    ):
        with pytest.raises(ValueError, match="boom"):
            FastGemma4Engine(home=tmp_path, warmup=False)
    assert cl_calls == [256 * 1024 * 1024, 777]
    assert wl_calls == [4096, 888]

    # 5. resolve_mtp_drafter_dir rejects empty dirs and falls back QAT -> standard assistant when local_files_only=True!
    empty_dr = tmp_path / "empty_drafter"
    empty_dr.mkdir()
    valid_dr = tmp_path / "valid_drafter"
    valid_dr.mkdir()
    (valid_dr / "model.safetensors").write_bytes(b"stub")

    seen_snaps: list[str] = []

    def _fake_snap(repo_id: str, **_kw: Any) -> str:
        seen_snaps.append(repo_id)
        if repo_id == DEFAULT_QAT_MTP_DRAFTER_REPO:
            return str(empty_dr)  # header/README-only incomplete snapshot!
        return str(valid_dr)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snap):
        got_dr = resolve_mtp_drafter_dir(DEFAULT_QAT_MTP_DRAFTER_REPO, local_files_only=True)
    assert got_dr == valid_dr
    assert seen_snaps == [DEFAULT_QAT_MTP_DRAFTER_REPO, DEFAULT_MTP_DRAFTER_REPO]

    # 6. generate / generate_batch accept reasoning_effort & reasoning_budget kwargs cleanly!
    sig_gen = inspect.signature(FastGemma4Engine.generate).parameters
    sig_gb = inspect.signature(FastGemma4Engine.generate_batch).parameters
    assert "reasoning_effort" in sig_gen and "reasoning_budget" in sig_gen
    assert "reasoning_effort" in sig_gb and "reasoning_budget" in sig_gb


def test_multi_row_qmv_and_gemv_exact_0_ulp_and_ff_jump_telemetry_gpu(tmp_path: Path) -> None:
    """Verify #365 GPU items: single-pass multi-row qmv & gemv 0.0 ULP without gather_qmm, plus FF-jump telemetry."""
    with _gpu_lock():
        mx.random.seed(2027)
        for bits in (2, 4, 8):
            for gs in (32, 64):
                for in_d, out_d in ((64, 128), (128, 256), (256, 512), (512, 1026), (1536, 3072)):
                    if in_d < gs:
                        continue
                    lin = nn.Linear(in_d, out_d, bias=False)
                    lin.weight = (mx.random.normal((out_d, in_d)) * 0.05).astype(mx.bfloat16)
                    qlin = nn.QuantizedLinear.from_linear(lin, group_size=gs, bits=bits, mode="affine")
                    x = mx.random.normal((1, 5, in_d)).astype(mx.bfloat16)
                    mx.eval(qlin.parameters(), x)
                    ref = mx.concatenate([qlin(x[:, t : t + 1]) for t in range(5)], axis=1)
                    with patch.object(mx, "gather_qmm", side_effect=AssertionError("gather_qmm should not be called!")):
                        got = singleton_quantized_linear(qlin, x)
                    assert got is not None
                    mx.eval(ref, got)
                    assert np.array_equal(np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32))), (
                        f"ULP mismatch for bits={bits} gs={gs} {in_d}->{out_d}"
                    )

        # Also verify multi_row_gemv_exact_bf16 0.0 ULP on nn.Linear!
        for in_d, out_d in ((256, 512), (1536, 3072)):
            w = (mx.random.normal((out_d, in_d)) * 0.05).astype(mx.bfloat16)
            x = mx.random.normal((1, 5, in_d)).astype(mx.bfloat16)
            mx.eval(w, x)
            ref_g = mx.concatenate([mx.matmul(x[:, t : t + 1], w.T) for t in range(5)], axis=1)
            got_g = multi_row_gemv_exact_bf16(x, w)
            assert got_g is not None
            mx.eval(ref_g, got_g)
            assert np.array_equal(np.asarray(got_g.astype(mx.float32)), np.asarray(ref_g.astype(mx.float32)))

        # Verify forward_ff_jump_slot collects telemetry when collect_telemetry=True!
        ckpt_src = tmp_path / "telem_src"
        ckpt_dir = tmp_path / "telem_q4"
        _build_synthetic_e2b_bf16_dir(ckpt_src, vocab_size=256, hidden_size=256, interm_size=512, ple_dim=64)
        pack_checkpoint_dir(ckpt_src, ckpt_dir, quant_mode="qat_q4_0", vocab_out=None)
        model = FastGemma4Model(paths=resolve_paths(tmp_path), model_dir=ckpt_dir, mlp_dtype="auto")
        pkv = model.prefill_prefix([2, 10])
        sc = model.make_slot_cache(pkv, batch_size=1, doc_cap_full=64, swa_ring_cap=512, kv_i8=True)
        x_j, am_j = model.forward_ff_jump_slot(
            mx.array([[11, 12, 13]], dtype=mx.int32),
            sc,
            mx.array([2], dtype=mx.int32),
            mx.array([0], dtype=mx.int32),
            return_logits=False,
            collect_telemetry=True,
        )
        telem_j = sc.get("_last_telemetry")
        assert telem_j is not None
        mx.eval(x_j, am_j, telem_j)
        assert telem_j.shape == (1, 3) and np.all(np.isfinite(np.asarray(telem_j)))
