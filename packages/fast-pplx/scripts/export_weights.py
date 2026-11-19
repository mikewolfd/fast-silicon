#!/usr/bin/env python3
"""Export stripped text-only FP16 backbone + mmap FP16 embeddings for pplx-embed-v2-context-0.6b.

Reads HuggingFace safetensors shards directly using struct + NumPy (no torch required),
strips `visual.*` parameters, emits Track-2-canonical MLX/ANE weights in
`artifacts/weights/pplx_0_6b_f16.npz` (with `__norms_shifted__=1`) alongside
`artifacts/weights/embed_tokens.safetensors` (`language_model.embed_tokens.weight`, F16)
and `artifacts/tokenizer/*`.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from fast_shared.hf import copy_tokenizer_files, resolve_hf_model_dir

from fast_pplx.config import DEFAULT_MODEL_NAME, DEFAULT_PPLX_CONFIG, PplxConfig
from fast_pplx.paths import resolve_paths


def _load_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as fp:
        hdr_len = struct.unpack("<Q", fp.read(8))[0]
        hdr = json.loads(fp.read(hdr_len).decode("utf-8"))
        return hdr, 8 + hdr_len


def _read_tensor_f32(path: Path, hdr: dict[str, Any], data_base: int, key: str) -> np.ndarray:
    info = hdr[key]
    dtype = info["dtype"]
    shape = tuple(int(d) for d in info["shape"])
    b0, b1 = info["data_offsets"]
    with path.open("rb") as fp:
        fp.seek(data_base + b0)
        raw = fp.read(b1 - b0)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32, copy=True)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").reshape(shape)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"Unsupported safetensors dtype {dtype!r} for key {key!r}")


def _read_tensor_f16(path: Path, hdr: dict[str, Any], data_base: int, key: str) -> np.ndarray:
    info = hdr[key]
    if info["dtype"] == "F16":
        shape = tuple(int(d) for d in info["shape"])
        b0, b1 = info["data_offsets"]
        with path.open("rb") as fp:
            fp.seek(data_base + b0)
            raw = fp.read(b1 - b0)
        return np.frombuffer(raw, dtype="<f2").reshape(shape).copy()
    return _read_tensor_f32(path, hdr, data_base, key).astype(np.float16)


def _write_single_tensor_safetensors(out_path: Path, name: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.float16:
        dtype_str = "F16"
    elif arr.dtype == np.float32:
        dtype_str = "F32"
    else:
        raise TypeError(f"Unsupported array dtype for safetensors write: {arr.dtype}")

    payload = arr.tobytes()
    header_obj = {
        "__metadata__": {"format": "fast_pplx"},
        name: {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [0, len(payload)],
        },
    }
    hdr_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    if pad:
        hdr_bytes += b" " * pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(struct.pack("<Q", len(hdr_bytes)))
        fp.write(hdr_bytes)
        fp.write(payload)


class _ShardReader:
    """Lazy multi-shard safetensors reader supporting contextual_head.safetensors fallback."""

    def __init__(self, snapshot_dir: Path) -> None:
        self.snapshot_dir = snapshot_dir
        self._key_to_file: dict[str, Path] = {}
        self._headers: dict[Path, tuple[dict[str, Any], int]] = {}

        idx_path = snapshot_dir / "model.safetensors.index.json"
        if idx_path.exists():
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
            for key, rel in idx.get("weight_map", {}).items():
                self._key_to_file[key] = snapshot_dir / rel
        else:
            for cand_name in ("model.safetensors", "model-00001-of-00001.safetensors"):
                cand = snapshot_dir / cand_name
                if cand.exists():
                    hdr, base = _load_safetensors_header(cand)
                    self._headers[cand] = (hdr, base)
                    for key in hdr:
                        if key != "__metadata__":
                            self._key_to_file[key] = cand
                    break

        head_path = snapshot_dir / "contextual_head.safetensors"
        if head_path.exists():
            hdr, base = _load_safetensors_header(head_path)
            self._headers[head_path] = (hdr, base)
            for key in hdr:
                if key != "__metadata__":
                    self._key_to_file[key] = head_path

    def _get_header(self, path: Path) -> tuple[dict[str, Any], int]:
        cached = self._headers.get(path)
        if cached is None:
            cached = _load_safetensors_header(path)
            self._headers[path] = cached
        return cached

    def has_key(self, key: str) -> bool:
        return key in self._key_to_file

    def get_f32(self, key: str) -> np.ndarray:
        path = self._key_to_file[key]
        hdr, base = self._get_header(path)
        return _read_tensor_f32(path, hdr, base, key)

    def get_f16(self, key: str) -> np.ndarray:
        path = self._key_to_file[key]
        hdr, base = self._get_header(path)
        return _read_tensor_f16(path, hdr, base, key)


def export_model_weights(
    *,
    home: str | Path | None = None,
    model_dir: str | Path | None = None,
    repo_id: str = DEFAULT_MODEL_NAME,
    local_files_only: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    paths = resolve_paths(home)
    snap_dir = resolve_hf_model_dir(model_dir, repo_id, local_files_only=local_files_only)

    cfg_path = snap_dir / "config.json"
    cfg = PplxConfig.from_json(cfg_path) if cfg_path.exists() else DEFAULT_PPLX_CONFIG

    paths.weights_dir.mkdir(parents=True, exist_ok=True)
    copy_tokenizer_files(snap_dir, paths.tokenizer_dir)

    reader = _ShardReader(snap_dir)

    # 1. Mmap-ready FP16 embed_tokens safetensors (visual.* stripped completely)
    embed_key = (
        "language_model.embed_tokens.weight"
        if reader.has_key("language_model.embed_tokens.weight")
        else "model.embed_tokens.weight"
    )
    embed_f16 = reader.get_f16(embed_key)
    _write_single_tensor_safetensors(paths.embed_tokens_safetensors, embed_key, embed_f16)

    # 2. Canonical shared NPZ consumed by BOTH FastPplxGPU and build_ane_models.py
    prefix_root = "language_model." if reader.has_key("language_model.norm.weight") else "model."
    out_w: dict[str, np.ndarray] = {
        "__norms_shifted__": np.array(1, dtype=np.int32),
    }

    for i, layer_type in enumerate(cfg.layer_types):
        hp = f"{prefix_root}layers.{i}."
        mp = f"layers.{i}."
        out_w[f"{mp}input_layernorm.weight"] = (1.0 + reader.get_f32(f"{hp}input_layernorm.weight")).astype(np.float16)
        out_w[f"{mp}post_attention_layernorm.weight"] = (
            1.0 + reader.get_f32(f"{hp}post_attention_layernorm.weight")
        ).astype(np.float16)

        if layer_type == "linear_attention":
            out_w[f"{mp}linear_attn.in_proj_qkv.weight"] = reader.get_f16(f"{hp}linear_attn.in_proj_qkv.weight")
            out_w[f"{mp}linear_attn.in_proj_z.weight"] = reader.get_f16(f"{hp}linear_attn.in_proj_z.weight")
            out_w[f"{mp}linear_attn.in_proj_b.weight"] = reader.get_f16(f"{hp}linear_attn.in_proj_b.weight")
            out_w[f"{mp}linear_attn.in_proj_a.weight"] = reader.get_f16(f"{hp}linear_attn.in_proj_a.weight")
            # Upstream [C, 1, K] -> MLX channels-last [C, K, 1]
            out_w[f"{mp}linear_attn.conv1d.weight"] = np.ascontiguousarray(
                np.moveaxis(reader.get_f16(f"{hp}linear_attn.conv1d.weight"), 2, 1)
            )
            out_w[f"{mp}linear_attn.A_log"] = reader.get_f32(f"{hp}linear_attn.A_log")
            out_w[f"{mp}linear_attn.dt_bias"] = reader.get_f32(f"{hp}linear_attn.dt_bias")
            # Qwen3.5 GatedRMSNorm uses unshifted weights in FP32
            out_w[f"{mp}linear_attn.norm.weight"] = reader.get_f32(f"{hp}linear_attn.norm.weight")
            out_w[f"{mp}linear_attn.out_proj.weight"] = reader.get_f16(f"{hp}linear_attn.out_proj.weight")
        elif layer_type == "full_attention":
            w_q = reader.get_f16(f"{hp}self_attn.q_proj.weight")
            w_k = reader.get_f16(f"{hp}self_attn.k_proj.weight")
            w_v = reader.get_f16(f"{hp}self_attn.v_proj.weight")
            out_w[f"{mp}self_attn.qkv_proj.weight"] = np.concatenate([w_q, w_k, w_v], axis=0)
            out_w[f"{mp}self_attn.q_norm.weight"] = (1.0 + reader.get_f32(f"{hp}self_attn.q_norm.weight")).astype(
                np.float16
            )
            out_w[f"{mp}self_attn.k_norm.weight"] = (1.0 + reader.get_f32(f"{hp}self_attn.k_norm.weight")).astype(
                np.float16
            )
            out_w[f"{mp}self_attn.o_proj.weight"] = reader.get_f16(f"{hp}self_attn.o_proj.weight")
        else:
            raise ValueError(f"Unsupported layer_type {layer_type!r} at layer {i}")

        w_g = reader.get_f16(f"{hp}mlp.gate_proj.weight")
        w_u = reader.get_f16(f"{hp}mlp.up_proj.weight")
        out_w[f"{mp}mlp.gate_up_proj.weight"] = np.concatenate([w_g, w_u], axis=0)
        out_w[f"{mp}mlp.down_proj.weight"] = reader.get_f16(f"{hp}mlp.down_proj.weight")

    out_w["norm.weight"] = (1.0 + reader.get_f32(f"{prefix_root}norm.weight")).astype(np.float16)

    proj_key = (
        "contextual_projection.weight"
        if reader.has_key("contextual_projection.weight")
        else f"{prefix_root}contextual_projection.weight"
    )
    out_w["contextual_projection.weight"] = np.ascontiguousarray(reader.get_f32(proj_key), dtype=np.float32)

    np.savez(paths.weights_npz, **out_w)
    elapsed_s = time.perf_counter() - t0
    return {
        "snapshot_dir": str(snap_dir),
        "embed_tokens_safetensors": str(paths.embed_tokens_safetensors),
        "weights_npz": str(paths.weights_npz),
        "tokenizer_dir": str(paths.tokenizer_dir),
        "embed_shape": list(embed_f16.shape),
        "num_layers": len(cfg.layer_types),
        "elapsed_s": round(elapsed_s, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export text-only pplx-embed-v2-context-0.6b weights for fast-pplx")
    parser.add_argument("--home", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    info = export_model_weights(
        home=args.home,
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
