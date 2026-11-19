#!/usr/bin/env python3
"""Export EmbeddingGemma-2 weights (FP16 .npz) and tokenizer files into artifacts/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np

from fast_shared.hf import copy_tokenizer_files, resolve_hf_model_dir

DEFAULT_REPO_ID = "mlx-community/embeddinggemma-2-bf16"
NUM_LAYERS = 24
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _decode_tensor_f16(fp, base: int, info: dict) -> np.ndarray:
    b0, b1 = info["data_offsets"]
    fp.seek(base + b0)
    raw = fp.read(b1 - b0)
    dtype = info["dtype"]
    shape = info["shape"]
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape).astype(np.float16)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float16)
    raise ValueError(f"Unsupported tensor dtype {dtype!r} for {shape}")


def export_from_dir(model_dir: Path, home_dir: Path) -> Path:
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors in {model_dir}")

    weights_dir = home_dir / "artifacts" / "weights"
    tok_dir = home_dir / "artifacts" / "tokenizer"
    weights_dir.mkdir(parents=True, exist_ok=True)

    d: dict[str, np.ndarray] = {}
    with safetensors_path.open("rb") as fp:
        hdr_len = struct.unpack("<Q", fp.read(8))[0]
        hdr = json.loads(fp.read(hdr_len))
        base = 8 + hdr_len

        def get(k: str) -> np.ndarray:
            if k in hdr:
                return _decode_tensor_f16(fp, base, hdr[k])
            alt = "model." + k
            if alt in hdr:
                return _decode_tensor_f16(fp, base, hdr[alt])
            raise KeyError(f"Missing tensor {k!r} in {safetensors_path}")

        d["embed_tokens"] = get("language_model.embed_tokens.weight")
        d["ple_proj"] = get("language_model.ple.per_layer_model_projection.weight")[:, :, None, None]
        d["ple_norm"] = get("language_model.ple.per_layer_projection_norm.weight")
        d["final_norm"] = get("language_model.norm.weight")[None, :, None, None]
        d["emb_proj"] = get("language_model.embedding_projection.weight")[:, :, None, None]

        for i in range(NUM_LAYERS):
            p = f"language_model.layers.{i}."
            q = f"l{i}_"
            d[q + "in_ln"] = get(p + "input_layernorm.weight")[None, :, None, None]
            wq = get(p + "self_attn.q_proj.weight")
            wk = get(p + "self_attn.k_proj.weight")
            wv = get(p + "self_attn.v_proj.weight")
            d[q + "qkv"] = np.concatenate([wq, wk, wv], axis=0)[:, :, None, None]
            d[q + "o"] = get(p + "self_attn.o_proj.weight")[:, :, None, None]
            d[q + "qn"] = get(p + "self_attn.q_norm.weight")[None, None, None, :]
            d[q + "kn"] = get(p + "self_attn.k_norm.weight")[None, None, None, :]
            d[q + "pa_ln"] = get(p + "post_attention_layernorm.weight")[None, :, None, None]
            d[q + "pff_ln"] = get(p + "pre_feedforward_layernorm.weight")[None, :, None, None]
            wg = get(p + "mlp.gate_proj.weight")
            wu = get(p + "mlp.up_proj.weight")
            d[q + "gu"] = np.concatenate([wg, wu], axis=0)[:, :, None, None]
            d[q + "dn"] = get(p + "mlp.down_proj.weight")[:, :, None, None]
            d[q + "poff_ln"] = get(p + "post_feedforward_layernorm.weight")[None, :, None, None]
            d[q + "pg"] = get(p + "ple_block.per_layer_input_gate.weight")[:, :, None, None]
            d[q + "pp"] = get(p + "ple_block.per_layer_projection.weight")[:, :, None, None]
            d[q + "pple_ln"] = get(p + "ple_block.post_per_layer_input_norm.weight")[None, :, None, None]
            d[q + "scalar"] = get(p + "layer_scalar")

    out_npz = weights_dir / "embeddinggemma_2_f16.npz"
    np.savez(out_npz, **d)
    copy_tokenizer_files(model_dir, tok_dir, TOKENIZER_FILES)
    print(f"Exported {len(d)} tensors to {out_npz} ({out_npz.stat().st_size / 1e6:.2f} MB)")
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EmbeddingGemma-2 weights and tokenizer assets.")
    parser.add_argument("--home", type=Path, default=repo_root(), help="fast-embeddinggemma-2 root directory")
    parser.add_argument("--model-dir", type=Path, default=None, help="Local HF snapshot directory containing model.safetensors")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace repo ID if --model-dir is omitted")
    parser.add_argument("--local-files-only", action="store_true", help="Do not hit network during snapshot resolution")
    args = parser.parse_args()

    model_dir = resolve_hf_model_dir(args.model_dir, args.repo_id, local_files_only=args.local_files_only)
    export_from_dir(model_dir, args.home.resolve())


if __name__ == "__main__":
    main()
