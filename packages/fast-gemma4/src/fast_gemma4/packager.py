"""Pure Python + NumPy Zero-QAT checkpoint packager and multimodal tower stripper.

Zero GPU / Metal / MLX runtime calls: operates directly on raw `safetensors`
8-byte uint64 header lengths, JSON indices, and contiguous tensor byte buffers.

Supports:
  1. Multimodal tower stripping (`vision_tower`, `audio_tower`, `embed_vision`,
     `embed_audio`, `multi_modal_projector`, etc.) with exact byte accounting
     and updated `model.safetensors.index.json`.
  2. Zero-QAT Mixed-Precision Packing (`M4_PRO_MAX_HW_LESSONS_LEARNED.md#L125-L127`):
     - Non-MLP 2D matrices & embeddings -> MLX `uint32` `8-bit g64 affine`
       (`weight` `U32 [out, in//4]`, `scales` `BF16 [out, in//64]`, `biases` `BF16 [out, in//64]`)
     - MLP projections (`.mlp.{gate,up,down}_proj`) -> unquantized `BF16`
     - 1D norms & `layer_scalar` -> unquantized `BF16`
  3. Optional `vocab_out=150,786` output-row remapping via `VocabRemapper` (`kept_old_ids` / `old_ids`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
from typing import Any, Iterable, Sequence
import uuid

import numpy as np

DEFAULT_GROUP_SIZE = 64
DEFAULT_BITS = 8
DEFAULT_KEPT_VOCAB_SIZE = 150786
DEFAULT_MAX_SHARD_BYTES = 4 * 1024 * 1024 * 1024
_FLOAT_SAFETENSORS_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
_BF16_ONE_U16 = np.uint16(0x3F80)
_BF16_MIN_POS_NORMAL_U16 = np.uint16(0x0080)

MULTIMODAL_PREFIXES: tuple[str, ...] = (
    "vision_tower",
    "audio_tower",
    "embed_vision",
    "embed_audio",
    "multi_modal_projector",
    "mm_projector",
    "vision_model",
    "audio_model",
    "vision_proj",
    "audio_proj",
)

_DTYPE_ITEMSIZE: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

_SIMPLE_NP_DTYPES: dict[str, np.dtype[Any]] = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype("<u1"),
    "I8": np.dtype("<i1"),
    "U16": np.dtype("<u2"),
    "I16": np.dtype("<i2"),
    "F16": np.dtype("<f2"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "F32": np.dtype("<f4"),
    "U64": np.dtype("<u8"),
    "I64": np.dtype("<i8"),
    "F64": np.dtype("<f8"),
}


@dataclass
class RawTensor:
    """In-memory raw tensor holding exact wire dtype (`BF16` stored as `<u2` bit-patterns)."""

    dtype: str
    shape: tuple[int, ...]
    array: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(math.prod(self.shape)) * _DTYPE_ITEMSIZE[self.dtype]

    def to_float32(self) -> np.ndarray:
        if self.dtype == "BF16":
            return bf16_u16_to_f32(self.array).reshape(self.shape)
        if self.dtype in ("F16", "F32", "F64"):
            return np.asarray(self.array, dtype=np.float32).reshape(self.shape)
        raise TypeError(
            f"Refusing implicit cast from non-float dtype {self.dtype!r} to float32 "
            "(packed/quantized integer tensors must not be decoded via to_float32)."
        )

    def to_bf16_u16(self) -> np.ndarray:
        if self.dtype == "BF16":
            return np.ascontiguousarray(self.array, dtype=np.dtype("<u2")).reshape(self.shape)
        return f32_to_bf16_u16(self.to_float32()).reshape(self.shape)

    def to_bytes(self) -> bytes:
        arr = np.ascontiguousarray(self.array)
        if self.dtype == "BF16":
            arr = arr.astype(np.dtype("<u2"), copy=False)
        elif self.dtype in _SIMPLE_NP_DTYPES:
            arr = arr.astype(_SIMPLE_NP_DTYPES[self.dtype], copy=False)
        return arr.tobytes(order="C")


@dataclass
class StripReport:
    bytes_before: int
    bytes_after: int
    bytes_removed: int
    removed_keys: list[str] = field(default_factory=list)
    kept_keys: list[str] = field(default_factory=list)
    bytes_by_prefix: dict[str, int] = field(default_factory=dict)

    @property
    def gib_removed(self) -> float:
        return self.bytes_removed / float(1024**3)


@dataclass
class PackReport:
    strip_report: StripReport
    quantized_modules: list[str] = field(default_factory=list)
    bf16_mlp_keys: list[str] = field(default_factory=list)
    unquantized_other_keys: list[str] = field(default_factory=list)
    remapped_vocab_keys: list[str] = field(default_factory=list)
    vocab_out_size: int | None = None
    output_shards: list[str] = field(default_factory=list)
    output_total_bytes: int = 0
    provenance_path: str | None = None
    quant_mode: str = "8bit_bf16_mlp"
    module_quant_meta: dict[str, dict[str, int]] = field(default_factory=dict)
    ple_sidecar_path: str | None = None


def bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(u16, dtype=np.uint16)
    return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16_u16(f32: np.ndarray) -> np.ndarray:
    """IEEE-754 round-to-nearest-even `float32 -> uint16` BF16 bit-patterns."""
    arr = np.ascontiguousarray(f32, dtype=np.float32)
    u32 = arr.view(np.uint32)
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    is_nan = (u32 & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    rounded = np.where(is_nan, u32 | np.uint32(0x00400000), u32 + bias)
    return (rounded >> np.uint32(16)).astype(np.uint16)


def array_to_raw_tensor(arr: np.ndarray, dtype: str) -> RawTensor:
    if dtype == "BF16":
        if arr.dtype == np.uint16:
            u16 = np.ascontiguousarray(arr, dtype=np.dtype("<u2"))
        else:
            u16 = f32_to_bf16_u16(arr.astype(np.float32)).astype(np.dtype("<u2"), copy=False)
        return RawTensor(dtype="BF16", shape=tuple(int(d) for d in arr.shape), array=u16)
    if dtype not in _SIMPLE_NP_DTYPES:
        raise ValueError(f"Unsupported safetensors dtype {dtype!r}")
    casted = np.ascontiguousarray(arr, dtype=_SIMPLE_NP_DTYPES[dtype])
    return RawTensor(dtype=dtype, shape=tuple(int(d) for d in casted.shape), array=casted)


# ---------------------------------------------------------------------------
# Key classification helpers
# ---------------------------------------------------------------------------

def _strip_model_prefix(key: str) -> str:
    if key.startswith("model."):
        return key[len("model.") :]
    return key


def is_multimodal_key(key: str, prefixes: Sequence[str] = MULTIMODAL_PREFIXES) -> str | None:
    """Return matching multimodal tower prefix if `key` belongs to unused vision/audio towers."""
    stripped = _strip_model_prefix(key)
    head = stripped.split(".", 1)[0]
    for p in prefixes:
        if head == p or stripped.startswith(f"{p}."):
            return p
    return None


def is_mlp_weight_key(key: str) -> bool:
    return key.endswith(
        (
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
            ".mlp.gate_up_proj.weight",
        )
    )


def is_input_embedding_key(key: str) -> bool:
    return key.endswith(
        (
            ".embed_tokens.weight",
            ".embed_tokens_per_layer.weight",
            "embed_tokens.weight",
            "embed_tokens_per_layer.weight",
        )
    )


def is_lm_head_key(key: str) -> bool:
    return key.endswith((".lm_head.weight", "lm_head.weight"))


def is_vocab_table_key(key: str) -> bool:
    return is_input_embedding_key(key) or is_lm_head_key(key)


def is_ple_table_key(key: str) -> bool:
    return key.endswith((".embed_tokens_per_layer.weight", "embed_tokens_per_layer.weight"))


def is_ple_gate_or_proj_key(key: str) -> bool:
    return key.endswith(
        (
            ".per_layer_input_gate.weight",
            ".per_layer_projection.weight",
            "per_layer_input_gate.weight",
            "per_layer_projection.weight",
        )
    )


def _list_shard_names(model_dir: Path) -> list[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        # Preserve index encounter order (deduped) so streaming writes shards predictably
        seen: dict[str, None] = {}
        for sname in idx["weight_map"].values():
            seen.setdefault(str(sname), None)
        shard_names = list(seen.keys())
    else:
        shard_names = [p.name for p in sorted(model_dir.glob("model*.safetensors"))]
        if not shard_names:
            shard_names = [p.name for p in sorted(model_dir.glob("*.safetensors"))]
    if not shard_names:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return shard_names


# ---------------------------------------------------------------------------
# Pure Python + NumPy Raw Safetensors I/O
# ---------------------------------------------------------------------------

def read_safetensors_header(path: str | Path) -> tuple[dict[str, Any], int]:
    """Read `<Q` 8-byte header length + JSON index from a `.safetensors` file."""
    p = Path(path)
    with p.open("rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Truncated safetensors header length in {p}")
        hdr_len = struct.unpack("<Q", raw_len)[0]
        hdr_bytes = fp.read(hdr_len)
        if len(hdr_bytes) != hdr_len:
            raise ValueError(f"Truncated safetensors JSON header in {p}")
        hdr = json.loads(hdr_bytes.decode("utf-8"))
    return hdr, 8 + int(hdr_len)


def inspect_safetensors_headers(model_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Inspect all tensor metadata (`dtype`, `shape`, `nbytes`, `shard`) without loading payloads."""
    mdir = Path(model_dir)
    index_path = mdir / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(idx["weight_map"].values()))
    else:
        shard_names = [p.name for p in sorted(mdir.glob("*.safetensors"))]
    if not shard_names:
        raise FileNotFoundError(f"No .safetensors files found in {mdir}")

    out: dict[str, dict[str, Any]] = {}
    for sname in shard_names:
        hdr, _ = read_safetensors_header(mdir / sname)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            b0, b1 = v["data_offsets"]
            out[k] = {
                "dtype": v["dtype"],
                "shape": list(v["shape"]),
                "nbytes": int(b1 - b0),
                "shard": sname,
            }
    return out


def load_safetensors_file(path: str | Path) -> tuple[dict[str, RawTensor], dict[str, str]]:
    p = Path(path)
    hdr, base = read_safetensors_header(p)
    meta = dict(hdr.get("__metadata__") or {})  # mx.save_safetensors writes `"__metadata__": null`
    tensors: dict[str, RawTensor] = {}
    with p.open("rb") as fp:
        for k, info in hdr.items():
            if k == "__metadata__":
                continue
            dt = str(info["dtype"])
            shape = tuple(int(d) for d in info["shape"])
            b0, b1 = int(info["data_offsets"][0]), int(info["data_offsets"][1])
            expected_bytes = int(math.prod(shape)) * _DTYPE_ITEMSIZE[dt]
            if (b1 - b0) != expected_bytes:
                raise ValueError(
                    f"Tensor {k!r} byte length mismatch in {p}: got {b1 - b0}, expected {expected_bytes}"
                )
            fp.seek(base + b0)
            buf = fp.read(b1 - b0)
            if len(buf) != b1 - b0:
                raise ValueError(f"Truncated payload for {k!r} in {p}")
            if dt == "BF16":
                arr = np.frombuffer(buf, dtype=np.dtype("<u2")).reshape(shape).copy()
            elif dt in _SIMPLE_NP_DTYPES:
                arr = np.frombuffer(buf, dtype=_SIMPLE_NP_DTYPES[dt]).reshape(shape).copy()
            else:
                raise ValueError(f"Unsupported safetensors dtype {dt!r} for {k!r}")
            tensors[k] = RawTensor(dtype=dt, shape=shape, array=arr)
    return tensors, meta


def write_safetensors_file(
    path: str | Path,
    tensors: dict[str, RawTensor],
    metadata: dict[str, str] | None = None,
) -> int:
    """Write a single `.safetensors` file with 8-byte `<Q` header length + JSON + tensor bytes."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    payload_chunks: list[bytes] = []
    offset = 0
    for k in sorted(tensors.keys()):
        t = tensors[k]
        raw = t.to_bytes()
        n = len(raw)
        header[k] = {
            "dtype": t.dtype,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + n],
        }
        payload_chunks.append(raw)
        offset += n

    hdr_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    if pad:
        hdr_bytes = hdr_bytes + (b" " * pad)

    with p.open("wb") as fp:
        fp.write(struct.pack("<Q", len(hdr_bytes)))
        fp.write(hdr_bytes)
        for chunk in payload_chunks:
            fp.write(chunk)
    return offset


def load_safetensors_dir(model_dir: str | Path) -> tuple[dict[str, RawTensor], dict[str, str], dict[str, Any] | None]:
    mdir = Path(model_dir)
    index_path = mdir / "model.safetensors.index.json"
    idx_data: dict[str, Any] | None = None
    if index_path.exists():
        idx_data = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(idx_data["weight_map"].values()))
    else:
        shard_names = [p.name for p in sorted(mdir.glob("model*.safetensors"))]
        if not shard_names:
            shard_names = [p.name for p in sorted(mdir.glob("*.safetensors"))]
    if not shard_names:
        raise FileNotFoundError(f"No .safetensors files in {mdir}")

    all_tensors: dict[str, RawTensor] = {}
    merged_meta: dict[str, str] = {}
    for sname in shard_names:
        tmap, meta = load_safetensors_file(mdir / sname)
        all_tensors.update(tmap)
        merged_meta.update(meta)
    return all_tensors, merged_meta, idx_data


def write_safetensors_dir(
    out_dir: str | Path,
    tensors: dict[str, RawTensor],
    *,
    metadata: dict[str, str] | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    write_index: bool = True,
) -> tuple[list[str], int]:
    """Write single or sharded `.safetensors` plus `model.safetensors.index.json`."""
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    keys = sorted(tensors.keys())
    total_bytes = sum(tensors[k].nbytes for k in keys)

    shards: list[dict[str, RawTensor]] = []
    if total_bytes <= max_shard_bytes or len(keys) <= 1:
        shards.append({k: tensors[k] for k in keys})
    else:
        cur: dict[str, RawTensor] = {}
        cur_bytes = 0
        for k in keys:
            kb = tensors[k].nbytes
            if cur and (cur_bytes + kb) > max_shard_bytes:
                shards.append(cur)
                cur = {}
                cur_bytes = 0
            cur[k] = tensors[k]
            cur_bytes += kb
        if cur:
            shards.append(cur)

    weight_map: dict[str, str] = {}
    shard_files: list[str] = []
    n_shards = len(shards)
    for i, smap in enumerate(shards, start=1):
        fname = "model.safetensors" if n_shards == 1 else f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        write_safetensors_file(odir / fname, smap, metadata=metadata)
        shard_files.append(fname)
        for k in smap:
            weight_map[k] = fname

    if write_index:
        idx_obj = {
            "metadata": {"total_size": int(total_bytes)},
            "weight_map": {k: weight_map[k] for k in sorted(weight_map.keys())},
        }
        (odir / "model.safetensors.index.json").write_text(
            json.dumps(idx_obj, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return shard_files, int(total_bytes)


# ---------------------------------------------------------------------------
# Multimodal Tower Stripper
# ---------------------------------------------------------------------------

def strip_multimodal_tensors(
    tensors: dict[str, RawTensor],
    *,
    prefixes: Sequence[str] = MULTIMODAL_PREFIXES,
) -> tuple[dict[str, RawTensor], StripReport]:
    """Strip unused vision/audio tower tensors and return kept tensors + exact byte report."""
    kept: dict[str, RawTensor] = {}
    removed_keys: list[str] = []
    kept_keys: list[str] = []
    by_prefix: dict[str, int] = {}
    bytes_before = 0
    bytes_removed = 0

    for k in sorted(tensors.keys()):
        t = tensors[k]
        nb = t.nbytes
        bytes_before += nb
        matched = is_multimodal_key(k, prefixes=prefixes)
        if matched is not None:
            removed_keys.append(k)
            bytes_removed += nb
            by_prefix[matched] = by_prefix.get(matched, 0) + nb
        else:
            kept[k] = t
            kept_keys.append(k)

    report = StripReport(
        bytes_before=bytes_before,
        bytes_after=bytes_before - bytes_removed,
        bytes_removed=bytes_removed,
        removed_keys=removed_keys,
        kept_keys=kept_keys,
        bytes_by_prefix=by_prefix,
    )
    return kept, report


# ---------------------------------------------------------------------------
# MLX `uint32` `8-bit g64` Affine Quantizer / Dequantizer
# ---------------------------------------------------------------------------

def quantize_affine_8bit_u32(
    weight_f32: np.ndarray,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[RawTensor, RawTensor, RawTensor]:
    """Quantize 2D `(out_f, in_f)` weights into MLX `uint32` `8-bit g64` `(weight, scales, biases)`."""
    if weight_f32.ndim != 2:
        raise ValueError(f"Expected 2D matrix for quantization, got shape {weight_f32.shape}")
    out_f, in_f = int(weight_f32.shape[0]), int(weight_f32.shape[1])
    if in_f % group_size != 0 or in_f % 4 != 0:
        raise ValueError(f"Input dim {in_f} must be divisible by group_size={group_size} and 4")

    w = np.ascontiguousarray(weight_f32, dtype=np.float32)
    n_groups = in_f // group_size
    wg = w.reshape(out_f, n_groups, group_size)

    w_min = wg.min(axis=-1)
    w_max = wg.max(axis=-1)

    biases_u16 = f32_to_bf16_u16(w_min)
    biases_f32 = bf16_u16_to_f32(biases_u16)

    span = np.maximum(w_max - biases_f32, 0.0)
    raw_scale = np.where(span > 0.0, span / 255.0, 1.0).astype(np.float32)
    scales_u16 = f32_to_bf16_u16(raw_scale)
    scales_f32 = bf16_u16_to_f32(scales_u16)
    # Apply BF16 scale floor to serialized `scales_u16` BEFORE quantizing `q` so the
    # encoded scale on disk strictly matches the divisor used to compute `q`.
    zero_mask = scales_f32 == 0.0
    if np.any(zero_mask):
        fallback_u16 = np.where(span <= 0.0, _BF16_ONE_U16, _BF16_MIN_POS_NORMAL_U16)
        scales_u16 = np.where(zero_mask, fallback_u16, scales_u16).astype(np.uint16)
        scales_f32 = bf16_u16_to_f32(scales_u16)

    q = np.clip(
        np.rint((wg - biases_f32[:, :, None]) / scales_f32[:, :, None]),
        0.0,
        255.0,
    ).astype(np.uint8)

    # Pack 4 x uint8 little-endian into 1 x uint32 -> [out_f, in_f // 4]
    q4 = q.reshape(out_f, in_f // 4, 4).astype(np.uint32)
    packed_u32 = q4[..., 0] | (q4[..., 1] << np.uint32(8)) | (q4[..., 2] << np.uint32(16)) | (q4[..., 3] << np.uint32(24))

    return (
        RawTensor(dtype="U32", shape=(out_f, in_f // 4), array=np.ascontiguousarray(packed_u32, dtype=np.dtype("<u4"))),
        RawTensor(dtype="BF16", shape=(out_f, n_groups), array=np.ascontiguousarray(scales_u16, dtype=np.dtype("<u2"))),
        RawTensor(dtype="BF16", shape=(out_f, n_groups), array=np.ascontiguousarray(biases_u16, dtype=np.dtype("<u2"))),
    )


def quantize_q4_0_sym_u32(
    weight_f32: np.ndarray,
    *,
    group_size: int = 32,
) -> tuple[RawTensor, RawTensor, RawTensor]:
    """Quantize 2D `(out_f, in_f)` weights onto `ggml q4_0` symmetric signed INT4 (`[-8, 7]`, `group_size=32`).

    Encoded identically into MLX `mode='affine'`, `bits=4`, `group_size=32` via unsigned nibbles
    `u = q_s + 8 in [0..15]`, `scales = d` (`BF16`), and `biases = -8 * d` (`BF16`, exact exponent shift).
    """
    if weight_f32.ndim != 2:
        raise ValueError(f"Expected 2D matrix for q4_0 quantization, got shape {weight_f32.shape}")
    out_f, in_f = int(weight_f32.shape[0]), int(weight_f32.shape[1])
    if in_f % group_size != 0 or in_f % 8 != 0:
        raise ValueError(f"Input dim {in_f} must be divisible by group_size={group_size} and 8")

    w = np.ascontiguousarray(weight_f32, dtype=np.float32)
    n_groups = in_f // group_size
    wg = w.reshape(out_f, n_groups, group_size)

    neg_part = np.maximum(0.0, -wg.min(axis=-1)) / 8.0
    pos_part = np.maximum(0.0, wg.max(axis=-1)) / 7.0
    d_raw = np.maximum(neg_part, pos_part).astype(np.float32)
    nonzero_span = d_raw > 0.0
    raw_scale = np.where(nonzero_span, d_raw, 1.0).astype(np.float32)
    scales_u16 = f32_to_bf16_u16(raw_scale)
    scales_f32 = bf16_u16_to_f32(scales_u16)
    zero_mask = scales_f32 == 0.0
    if np.any(zero_mask):
        fallback_u16 = np.where(~nonzero_span, _BF16_ONE_U16, _BF16_MIN_POS_NORMAL_U16)
        scales_u16 = np.where(zero_mask, fallback_u16, scales_u16).astype(np.uint16)
        scales_f32 = bf16_u16_to_f32(scales_u16)

    biases_u16 = f32_to_bf16_u16((-8.0 * scales_f32).astype(np.float32))
    q_s = np.clip(np.rint(wg / scales_f32[:, :, None]), -8.0, 7.0).astype(np.int32)
    u = (q_s + 8).astype(np.uint32)

    u8 = u.reshape(out_f, in_f // 8, 8)
    packed_u32 = np.zeros((out_f, in_f // 8), dtype=np.uint32)
    for idx_n in range(8):
        packed_u32 |= u8[..., idx_n] << np.uint32(4 * idx_n)

    return (
        RawTensor(dtype="U32", shape=(out_f, in_f // 8), array=np.ascontiguousarray(packed_u32, dtype=np.dtype("<u4"))),
        RawTensor(dtype="BF16", shape=(out_f, n_groups), array=np.ascontiguousarray(scales_u16, dtype=np.dtype("<u2"))),
        RawTensor(dtype="BF16", shape=(out_f, n_groups), array=np.ascontiguousarray(biases_u16, dtype=np.dtype("<u2"))),
    )


def dequantize_affine_u32(
    packed_u32: RawTensor | np.ndarray,
    scales: RawTensor | np.ndarray,
    biases: RawTensor | np.ndarray | None = None,
    *,
    bits: int = 8,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> np.ndarray:
    """Dequantize MLX `uint32` `bits in {2, 4, 8}` affine `(weight, scales, biases)` -> `float32 [out_f, in_f]`."""
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bits={bits}; expected 2, 4, or 8")
    w_arr = packed_u32.array if isinstance(packed_u32, RawTensor) else packed_u32
    w_u32 = np.ascontiguousarray(w_arr, dtype=np.uint32)
    out_f, packed_cols = w_u32.shape
    pack_factor = 32 // bits
    in_f = packed_cols * pack_factor
    n_groups = in_f // group_size
    mask = np.uint32((1 << bits) - 1)
    parts = [((w_u32 >> np.uint32(bits * i)) & mask).astype(np.float32) for i in range(pack_factor)]
    q = np.stack(parts, axis=-1).reshape(out_f, n_groups, group_size)

    s_f32 = (scales.to_float32() if isinstance(scales, RawTensor) else np.asarray(scales, dtype=np.float32)).reshape(
        out_f, n_groups, 1
    )
    if biases is not None:
        b_f32 = (
            biases.to_float32() if isinstance(biases, RawTensor) else np.asarray(biases, dtype=np.float32)
        ).reshape(out_f, n_groups, 1)
    else:
        b_f32 = np.zeros((out_f, n_groups, 1), dtype=np.float32)
    return (q * s_f32 + b_f32).reshape(out_f, in_f)


def dequantize_affine_8bit_u32(
    packed_u32: RawTensor | np.ndarray,
    scales: RawTensor | np.ndarray,
    biases: RawTensor | np.ndarray | None = None,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> np.ndarray:
    """Dequantize MLX `uint32` `8-bit g64` `(weight, scales, biases)` -> `float32 [out_f, in_f]`."""
    return dequantize_affine_u32(packed_u32, scales, biases, bits=8, group_size=group_size)


def write_ple_int8_sidecar(dst_dir: Path, ple_f32: np.ndarray) -> str:
    """Write clean-page mmap-ready symmetric INT8 `embed_tokens_per_layer` sidecar (`i8 + per-row fp16 scale`)."""
    if ple_f32.ndim != 2:
        raise ValueError(f"Expected 2D embed_tokens_per_layer matrix, got shape {ple_f32.shape}")
    vocab_rows, total_ple_dim = int(ple_f32.shape[0]), int(ple_f32.shape[1])
    arr = np.ascontiguousarray(ple_f32, dtype=np.float32)
    amax = np.max(np.abs(arr), axis=1)
    scales_f16 = np.maximum(amax / 127.0, 1e-6).astype(np.float16)
    scales_f32 = np.maximum(scales_f16.astype(np.float32), 1e-8)
    q_i8 = np.clip(np.rint(arr / scales_f32[:, None]), -127.0, 127.0).astype(np.int8)

    bin_name = "embed_tokens_per_layer_i8.bin"
    sc_name = "embed_tokens_per_layer_scales.bin"
    meta_name = "embed_tokens_per_layer_i8.json"
    (dst_dir / bin_name).write_bytes(q_i8.tobytes(order="C"))
    (dst_dir / sc_name).write_bytes(scales_f16.tobytes(order="C"))
    meta_obj = {
        "shape": [vocab_rows, total_ple_dim],
        "dtype": "int8",
        "data_file": bin_name,
        "scales_file": sc_name,
        "scales_dtype": "float16",
        "scale_mode": "per_row_fp16",
    }
    (dst_dir / meta_name).write_text(json.dumps(meta_obj, indent=2) + "\n", encoding="utf-8")
    return meta_name


# ---------------------------------------------------------------------------
# Vocab Remapping (`vocab_out = 150,786` via `VocabRemapper` / `.npz`)
# ---------------------------------------------------------------------------

def load_kept_old_ids(
    *,
    kept_old_ids: Sequence[int] | np.ndarray | None = None,
    vocab_remap_npz: str | Path | None = None,
    tokenizer_json: str | Path | None = None,
    vocab_out: int | None = DEFAULT_KEPT_VOCAB_SIZE,
) -> np.ndarray | None:
    if kept_old_ids is not None:
        return np.asarray(kept_old_ids, dtype=np.int32)
    if vocab_remap_npz is not None:
        data = np.load(vocab_remap_npz)
        if "kept_old_ids" in data:
            return np.asarray(data["kept_old_ids"], dtype=np.int32)
        if "old_ids" in data:
            return np.asarray(data["old_ids"], dtype=np.int32)
        raise KeyError(f"Expected 'kept_old_ids' or 'old_ids' in {vocab_remap_npz}, got {list(data.keys())}")
    if tokenizer_json is not None and vocab_out is not None:
        fg4_src = Path(__file__).resolve().parents[1]
        if fg4_src.exists() and str(fg4_src) not in sys.path:
            sys.path.insert(0, str(fg4_src))
        from fast_gemma4.tokenizer import build_kept_vocab

        kept_sorted, _, _, _, _ = build_kept_vocab(tokenizer_json, target_size=int(vocab_out))
        return np.asarray(kept_sorted, dtype=np.int32)
    return None


# ---------------------------------------------------------------------------
# Fail-Closed Validation, Per-Layer Quant Metadata & Transform Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            blk = fp.read(4 * 1024 * 1024)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _resolve_quant_mode(quant_mode: str, src_dir: str | Path | None = None) -> str:
    mode = str(quant_mode or "auto").strip().lower()
    if mode == "auto":
        hint = str(src_dir or "").lower()
        if "qat-q4_0" in hint or "qat_q4_0" in hint:
            return "qat_q4_0"
        return "8bit_bf16_mlp"
    if mode in ("8bit_bf16_mlp", "qat_q4_0"):
        return mode
    raise ValueError(f"Unsupported quant_mode={quant_mode!r}; expected 'auto', '8bit_bf16_mlp', or 'qat_q4_0'")


def _validate_source_config(cfg: dict[str, Any] | None, *, enable_quant: bool) -> None:
    if not cfg or not enable_quant:
        return
    if cfg.get("quantization") or cfg.get("quantization_config"):
        raise ValueError(
            "Source checkpoint already contains `quantization` / `quantization_config` in config.json; "
            "refusing to double-quantize an already-quantized checkpoint."
        )


def _validate_source_tensor_meta(key: str, dtype: str, *, enable_quant: bool) -> None:
    if not enable_quant:
        return
    if key.endswith((".scales", ".biases")):
        raise ValueError(
            f"Source checkpoint already contains quantized parameter `{key}`; "
            "refusing to re-pack already-quantized safetensors."
        )
    if dtype not in _FLOAT_SAFETENSORS_DTYPES:
        raise TypeError(
            f"Source weight `{key}` has non-float dtype `{dtype}` (expected one of {sorted(_FLOAT_SAFETENSORS_DTYPES)}); "
            "refusing to silently cast packed integers or non-float buffers during quantization."
        )


def _lm_head_key_for_embed(embed_key: str) -> str:
    if embed_key.startswith("language_model.model."):
        return "language_model.lm_head.weight"
    if embed_key.startswith("model."):
        return "lm_head.weight"
    return "lm_head.weight"


def _transform_kept_tensor(
    k: str,
    t: RawTensor,
    *,
    quant_mode: str,
    quantize_non_mlp_8bit: bool,
    keep_mlp_bf16: bool,
    group_size: int,
    kept_old_ids: np.ndarray | None,
    inferred_vocab: int | None,
    quantized_modules: list[str],
    module_quant_meta: dict[str, dict[str, int]],
    bf16_mlp_keys: list[str],
    unquantized_other_keys: list[str],
    remapped_vocab_keys: list[str],
) -> dict[str, RawTensor]:
    """Transform one kept source tensor into output tensor(s) (preserving full input embeddings)."""
    enable_quant = bool(quantize_non_mlp_8bit or quant_mode == "qat_q4_0")
    _validate_source_tensor_meta(k, t.dtype, enable_quant=enable_quant)
    out: dict[str, RawTensor] = {}

    # 1. Vocab row remapping: keep `embed_tokens` & `embed_tokens_per_layer` FULL (262,144 rows)
    #    matching FastGemma4Tokenizer.encode(); ONLY slice explicit `lm_head.weight`.
    if kept_old_ids is not None and is_lm_head_key(k):
        if inferred_vocab is None or (t.shape and int(t.shape[0]) == inferred_vocab):
            sliced = t.array[kept_old_ids]
            t = RawTensor(
                dtype=t.dtype,
                shape=tuple(int(d) for d in sliced.shape),
                array=np.ascontiguousarray(sliced),
            )
            remapped_vocab_keys.append(k)

    # 2. Keep MLPs unquantized in bit-exact BF16 when requested
    if keep_mlp_bf16 and is_mlp_weight_key(k):
        if t.dtype != "BF16":
            t = array_to_raw_tensor(t.to_float32(), "BF16")
        out[k] = t
        bf16_mlp_keys.append(k)
        return out

    # 3a. QAT symmetric `q4_0` `g32` mode (`bits=4, group_size=32`, with `8-bit` PLE gate/proj)
    if (
        quant_mode == "qat_q4_0"
        and enable_quant
        and k.endswith(".weight")
        and len(t.shape) == 2
        and t.shape[1] % group_size == 0
        and t.dtype in _FLOAT_SAFETENSORS_DTYPES
    ):
        prefix = k[: -len(".weight")]
        if is_ple_gate_or_proj_key(k) and t.shape[1] % 4 == 0:
            qw, qs, qb = quantize_affine_8bit_u32(t.to_float32(), group_size=group_size)
            out[f"{prefix}.weight"] = qw
            out[f"{prefix}.scales"] = qs
            out[f"{prefix}.biases"] = qb
            quantized_modules.append(prefix)
            module_quant_meta[prefix] = {"group_size": int(group_size), "bits": 8}
            return out
        if t.shape[1] % 8 == 0:
            qw, qs, qb = quantize_q4_0_sym_u32(t.to_float32(), group_size=group_size)
            out[f"{prefix}.weight"] = qw
            out[f"{prefix}.scales"] = qs
            out[f"{prefix}.biases"] = qb
            quantized_modules.append(prefix)
            module_quant_meta[prefix] = {"group_size": int(group_size), "bits": 4}
            return out

    # 3b. Quantize eligible 2D linear / embedding `.weight` matrices to 8-bit g64 affine
    is_quant_target = (
        quant_mode != "qat_q4_0"
        and quantize_non_mlp_8bit
        and k.endswith(".weight")
        and len(t.shape) == 2
        and t.shape[1] % group_size == 0
        and t.shape[1] % 4 == 0
        and t.dtype in _FLOAT_SAFETENSORS_DTYPES
    )
    if is_quant_target:
        prefix = k[: -len(".weight")]
        qw, qs, qb = quantize_affine_8bit_u32(t.to_float32(), group_size=group_size)
        out[f"{prefix}.weight"] = qw
        out[f"{prefix}.scales"] = qs
        out[f"{prefix}.biases"] = qb
        quantized_modules.append(prefix)
        module_quant_meta[prefix] = {"group_size": int(group_size), "bits": DEFAULT_BITS}
        return out

    # 4. Norms, layer_scalar, or incompatible shapes remain unquantized (BF16 for floats)
    if t.dtype in ("F16", "F32", "F64"):
        t = array_to_raw_tensor(t.to_float32(), "BF16")
    out[k] = t
    unquantized_other_keys.append(k)
    return out


def _build_quantization_config(
    *,
    group_size: int,
    bits: int,
    quantized_modules: Sequence[str],
    module_quant_meta: dict[str, dict[str, int]] | None,
    bf16_mlp_keys: Sequence[str],
    unquantized_other_keys: Sequence[str],
) -> dict[str, Any]:
    """Emit global defaults + per-layer overrides keeping BF16 MLPs unquantized (`False`)."""
    qcfg: dict[str, Any] = {
        "group_size": int(group_size),
        "bits": int(bits),
        "mode": "affine",
    }
    for k in sorted(bf16_mlp_keys):
        prefix = k[: -len(".weight")] if k.endswith(".weight") else k
        qcfg[prefix] = False
    for k in sorted(unquantized_other_keys):
        if k.endswith(".weight") and (
            ".mlp." in k
            or ".self_attn." in k
            or ".per_layer_" in k
            or ".lm_head." in k
            or "embed_tokens" in k
        ):
            qcfg[k[: -len(".weight")]] = False
    meta_lookup = module_quant_meta or {}
    for prefix in sorted(quantized_modules):
        p_meta = meta_lookup.get(prefix)
        if p_meta is not None:
            qcfg[prefix] = {
                "group_size": int(p_meta["group_size"]),
                "bits": int(p_meta["bits"]),
            }
        else:
            qcfg[prefix] = {
                "group_size": int(group_size),
                "bits": int(bits),
            }
    return qcfg


def _atomic_publish_dir(staging_dir: Path, dst_dir: Path) -> None:
    """Atomically publish `staging_dir` to `dst_dir` with rollback if target exists."""
    if not dst_dir.exists():
        os.replace(staging_dir, dst_dir)
        return
    backup_dir = dst_dir.parent / f".{dst_dir.name}.backup.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    os.replace(dst_dir, backup_dir)
    try:
        os.replace(staging_dir, dst_dir)
    except Exception:
        if not dst_dir.exists() and backup_dir.exists():
            os.replace(backup_dir, dst_dir)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)


def pack_checkpoint_tensors(
    raw_tensors: dict[str, RawTensor],
    *,
    strip_multimodal: bool = True,
    quantize_non_mlp_8bit: bool = True,
    keep_mlp_bf16: bool | None = None,
    group_size: int | None = None,
    quant_mode: str = "8bit_bf16_mlp",
    kept_old_ids: np.ndarray | None = None,
    full_vocab_size: int | None = None,
    materialize_pruned_lm_head: bool = False,
) -> tuple[dict[str, RawTensor], PackReport]:
    """Transform a raw BF16/multi-dtype tensor dict into a stripped, packed dict."""
    eff_mode = _resolve_quant_mode(quant_mode, None)
    eff_gs = int(group_size) if group_size is not None else (32 if eff_mode == "qat_q4_0" else DEFAULT_GROUP_SIZE)
    eff_keep_mlp_bf16 = (False if eff_mode == "qat_q4_0" else True) if keep_mlp_bf16 is None else bool(keep_mlp_bf16)

    if strip_multimodal:
        working, strip_rep = strip_multimodal_tensors(raw_tensors)
    else:
        working = dict(raw_tensors)
        tot = sum(t.nbytes for t in working.values())
        strip_rep = StripReport(bytes_before=tot, bytes_after=tot, bytes_removed=0, kept_keys=sorted(working.keys()))

    inferred_vocab = full_vocab_size
    if inferred_vocab is None and kept_old_ids is not None:
        for k, t in working.items():
            if is_vocab_table_key(k) and t.shape:
                inferred_vocab = int(t.shape[0])
                break

    if kept_old_ids is not None and inferred_vocab is not None and len(kept_old_ids) > 0:
        max_id = int(np.max(kept_old_ids))
        if max_id >= inferred_vocab or int(np.min(kept_old_ids)) < 0:
            raise IndexError(f"kept_old_ids out of range [0, {inferred_vocab}): min={int(np.min(kept_old_ids))}, max={max_id}")

    has_explicit_lm_head = any(is_lm_head_key(k) for k in working)
    packed: dict[str, RawTensor] = {}
    quantized_modules: list[str] = []
    module_quant_meta: dict[str, dict[str, int]] = {}
    bf16_mlp_keys: list[str] = []
    unquantized_other_keys: list[str] = []
    remapped_vocab_keys: list[str] = []

    for k in sorted(working.keys()):
        t = working[k]
        transformed = _transform_kept_tensor(
            k,
            t,
            quant_mode=eff_mode,
            quantize_non_mlp_8bit=quantize_non_mlp_8bit,
            keep_mlp_bf16=eff_keep_mlp_bf16,
            group_size=eff_gs,
            kept_old_ids=kept_old_ids,
            inferred_vocab=inferred_vocab,
            quantized_modules=quantized_modules,
            module_quant_meta=module_quant_meta,
            bf16_mlp_keys=bf16_mlp_keys,
            unquantized_other_keys=unquantized_other_keys,
            remapped_vocab_keys=remapped_vocab_keys,
        )
        packed.update(transformed)

        if materialize_pruned_lm_head and kept_old_ids is not None and not has_explicit_lm_head and k.endswith("embed_tokens.weight"):
            lm_key = _lm_head_key_for_embed(k)
            lm_items = _transform_kept_tensor(
                lm_key,
                t,
                quant_mode=eff_mode,
                quantize_non_mlp_8bit=quantize_non_mlp_8bit,
                keep_mlp_bf16=eff_keep_mlp_bf16,
                group_size=eff_gs,
                kept_old_ids=kept_old_ids,
                inferred_vocab=inferred_vocab,
                quantized_modules=quantized_modules,
                module_quant_meta=module_quant_meta,
                bf16_mlp_keys=bf16_mlp_keys,
                unquantized_other_keys=unquantized_other_keys,
                remapped_vocab_keys=remapped_vocab_keys,
            )
            packed.update(lm_items)

    report = PackReport(
        strip_report=strip_rep,
        quantized_modules=quantized_modules,
        bf16_mlp_keys=bf16_mlp_keys,
        unquantized_other_keys=unquantized_other_keys,
        remapped_vocab_keys=remapped_vocab_keys,
        vocab_out_size=(int(len(kept_old_ids)) if kept_old_ids is not None else None),
        quant_mode=eff_mode,
        module_quant_meta=module_quant_meta,
    )
    return packed, report


def pack_checkpoint_dir(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    strip_multimodal: bool = True,
    quantize_non_mlp_8bit: bool = True,
    keep_mlp_bf16: bool | None = None,
    group_size: int | None = None,
    quant_mode: str = "auto",
    ple_sidecar_int8: bool = False,
    kept_old_ids: Sequence[int] | np.ndarray | None = None,
    vocab_remap_npz: str | Path | None = None,
    vocab_out: int | None = DEFAULT_KEPT_VOCAB_SIZE,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    prune_multimodal_config: bool = True,
    materialize_pruned_lm_head: bool = False,
    extra_files: Sequence[str] = (),
) -> PackReport:
    """Stream-pack a safetensors checkpoint shard-by-shard with atomic staging & provenance manifest."""
    sdir = Path(src_dir)
    ddir = Path(dst_dir)
    ddir.parent.mkdir(parents=True, exist_ok=True)

    eff_mode = _resolve_quant_mode(quant_mode, sdir)
    eff_gs = int(group_size) if group_size is not None else (32 if eff_mode == "qat_q4_0" else DEFAULT_GROUP_SIZE)
    eff_bits = 4 if eff_mode == "qat_q4_0" else DEFAULT_BITS
    eff_keep_mlp_bf16 = (False if eff_mode == "qat_q4_0" else True) if keep_mlp_bf16 is None else bool(keep_mlp_bf16)
    enable_quant = bool(quantize_non_mlp_8bit) if eff_mode != "qat_q4_0" else True

    cfg_src = sdir / "config.json"
    cfg: dict[str, Any] | None = None
    cfg_sha256: str | None = None
    if cfg_src.exists():
        cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
        cfg_sha256 = _sha256_file(cfg_src)
        _validate_source_config(cfg, enable_quant=enable_quant)

    # Pre-validate all kept headers before allocating staging artifacts
    hdr_meta = inspect_safetensors_headers(sdir)
    kept_headers: dict[str, dict[str, Any]] = {}
    for k, info in hdr_meta.items():
        if strip_multimodal and is_multimodal_key(k) is not None:
            continue
        _validate_source_tensor_meta(k, str(info["dtype"]), enable_quant=enable_quant)
        kept_headers[k] = info

    tok_json = sdir / "tokenizer.json"
    remap_ids = load_kept_old_ids(
        kept_old_ids=kept_old_ids,
        vocab_remap_npz=vocab_remap_npz,
        tokenizer_json=(tok_json if tok_json.exists() and vocab_out is not None and vocab_remap_npz is None and kept_old_ids is None else None),
        vocab_out=vocab_out,
    )

    inferred_vocab: int | None = None
    for k, info in kept_headers.items():
        if is_vocab_table_key(k) and info["shape"]:
            inferred_vocab = int(info["shape"][0])
            break
    if remap_ids is not None and inferred_vocab is not None and len(remap_ids) > 0:
        max_id = int(np.max(remap_ids))
        if max_id >= inferred_vocab or int(np.min(remap_ids)) < 0:
            raise IndexError(f"kept_old_ids out of range [0, {inferred_vocab}): max={max_id}")

    has_explicit_lm_head = any(is_lm_head_key(k) for k in kept_headers)
    shard_names = _list_shard_names(sdir)

    staging_dir = ddir.parent / f".{ddir.name}.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Streaming bookkeeping & shard writer
        bytes_before = 0
        bytes_removed = 0
        removed_keys: list[str] = []
        kept_keys: list[str] = []
        by_prefix: dict[str, int] = {}

        quantized_modules: list[str] = []
        module_quant_meta: dict[str, dict[str, int]] = {}
        bf16_mlp_keys: list[str] = []
        unquantized_other_keys: list[str] = []
        remapped_vocab_keys: list[str] = []
        ple_sidecar_meta_name: str | None = None

        merged_meta: dict[str, str] = {}
        part_files: list[tuple[str, list[str], int]] = []
        cur_shard: dict[str, RawTensor] = {}
        cur_bytes = 0
        output_total_bytes = 0

        def _flush_cur_shard() -> None:
            nonlocal cur_shard, cur_bytes, output_total_bytes
            if not cur_shard:
                return
            part_name = f".part-{len(part_files) + 1:05d}.safetensors"
            meta_out = dict(merged_meta)
            meta_out["format"] = meta_out.get("format", "mlx")
            write_safetensors_file(staging_dir / part_name, cur_shard, metadata=meta_out)
            part_files.append((part_name, sorted(cur_shard.keys()), cur_bytes))
            output_total_bytes += cur_bytes
            cur_shard = {}
            cur_bytes = 0

        for sname in shard_names:
            shard_tensors, shard_meta = load_safetensors_file(sdir / sname)
            merged_meta.update(shard_meta)
            for k in sorted(shard_tensors.keys()):
                t = shard_tensors[k]
                nb = t.nbytes
                bytes_before += nb
                matched_mm = is_multimodal_key(k) if strip_multimodal else None
                if matched_mm is not None:
                    removed_keys.append(k)
                    bytes_removed += nb
                    by_prefix[matched_mm] = by_prefix.get(matched_mm, 0) + nb
                    continue

                if ple_sidecar_int8 and is_ple_table_key(k):
                    kept_keys.append(k)
                    ple_sidecar_meta_name = write_ple_int8_sidecar(staging_dir, t.to_float32())
                    continue

                kept_keys.append(k)
                transformed = _transform_kept_tensor(
                    k,
                    t,
                    quant_mode=eff_mode,
                    quantize_non_mlp_8bit=enable_quant,
                    keep_mlp_bf16=eff_keep_mlp_bf16,
                    group_size=eff_gs,
                    kept_old_ids=remap_ids,
                    inferred_vocab=inferred_vocab,
                    quantized_modules=quantized_modules,
                    module_quant_meta=module_quant_meta,
                    bf16_mlp_keys=bf16_mlp_keys,
                    unquantized_other_keys=unquantized_other_keys,
                    remapped_vocab_keys=remapped_vocab_keys,
                )

                items_list: list[dict[str, RawTensor]] = [transformed]
                if materialize_pruned_lm_head and remap_ids is not None and not has_explicit_lm_head and k.endswith("embed_tokens.weight"):
                    lm_key = _lm_head_key_for_embed(k)
                    items_list.append(
                        _transform_kept_tensor(
                            lm_key,
                            t,
                            quant_mode=eff_mode,
                            quantize_non_mlp_8bit=enable_quant,
                            keep_mlp_bf16=eff_keep_mlp_bf16,
                            group_size=eff_gs,
                            kept_old_ids=remap_ids,
                            inferred_vocab=inferred_vocab,
                            quantized_modules=quantized_modules,
                            module_quant_meta=module_quant_meta,
                            bf16_mlp_keys=bf16_mlp_keys,
                            unquantized_other_keys=unquantized_other_keys,
                            remapped_vocab_keys=remapped_vocab_keys,
                        )
                    )

                for item_map in items_list:
                    add_bytes = sum(vt.nbytes for vt in item_map.values())
                    if cur_shard and (cur_bytes + add_bytes) > max_shard_bytes:
                        _flush_cur_shard()
                    cur_shard.update(item_map)
                    cur_bytes += add_bytes

            del shard_tensors

        _flush_cur_shard()

        # Finalize canonical shard filenames inside staging_dir & record index + SHA-256
        n_shards = len(part_files)
        output_shards: list[str] = []
        weight_map: dict[str, str] = {}
        shard_sha256s: dict[str, str] = {}
        for idx_s, (part_name, skeys, _) in enumerate(part_files, start=1):
            final_name = "model.safetensors" if n_shards == 1 else f"model-{idx_s:05d}-of-{n_shards:05d}.safetensors"
            final_path = staging_dir / final_name
            os.replace(staging_dir / part_name, final_path)
            output_shards.append(final_name)
            shard_sha256s[final_name] = _sha256_file(final_path)
            for sk in skeys:
                weight_map[sk] = final_name

        idx_obj = {
            "metadata": {"total_size": int(output_total_bytes)},
            "weight_map": {k: weight_map[k] for k in sorted(weight_map.keys())},
        }
        (staging_dir / "model.safetensors.index.json").write_text(
            json.dumps(idx_obj, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        strip_rep = StripReport(
            bytes_before=bytes_before,
            bytes_after=bytes_before - bytes_removed,
            bytes_removed=bytes_removed,
            removed_keys=removed_keys,
            kept_keys=kept_keys,
            bytes_by_prefix=by_prefix,
        )
        report = PackReport(
            strip_report=strip_rep,
            quantized_modules=quantized_modules,
            bf16_mlp_keys=bf16_mlp_keys,
            unquantized_other_keys=unquantized_other_keys,
            remapped_vocab_keys=remapped_vocab_keys,
            vocab_out_size=(int(len(remap_ids)) if remap_ids is not None else None),
            output_shards=output_shards,
            output_total_bytes=output_total_bytes,
            provenance_path="provenance.json",
            quant_mode=eff_mode,
            module_quant_meta=module_quant_meta,
            ple_sidecar_path=ple_sidecar_meta_name,
        )

        # Write config.json preserving full input-embedding `vocab_size` (for tokenizer parity)
        # while emitting per-layer quantization metadata.
        cfg_out_obj: dict[str, Any] | None = None
        if cfg is not None:
            cfg_out_obj = json.loads(json.dumps(cfg))
            if prune_multimodal_config and strip_multimodal:
                for mk in ("vision_config", "audio_config"):
                    cfg_out_obj.pop(mk, None)
            if enable_quant:
                qcfg = _build_quantization_config(
                    group_size=eff_gs,
                    bits=eff_bits,
                    quantized_modules=quantized_modules,
                    module_quant_meta=module_quant_meta,
                    bf16_mlp_keys=bf16_mlp_keys,
                    unquantized_other_keys=unquantized_other_keys,
                )
                cfg_out_obj["quantization"] = qcfg
                cfg_out_obj.pop("quantization_config", None)
            else:
                cfg_out_obj.pop("quantization", None)
                cfg_out_obj.pop("quantization_config", None)
            if ple_sidecar_meta_name is not None:
                cfg_out_obj["ple_sidecar"] = ple_sidecar_meta_name
            if materialize_pruned_lm_head and not has_explicit_lm_head and remap_ids is not None:
                cfg_out_obj["tie_word_embeddings"] = False
                if isinstance(cfg_out_obj.get("text_config"), dict):
                    cfg_out_obj["text_config"]["tie_word_embeddings"] = False
            (staging_dir / "config.json").write_text(
                json.dumps(cfg_out_obj, indent=4, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        if remap_ids is not None:
            np.savez(staging_dir / "vocab_remap.npz", kept_old_ids=remap_ids, old_ids=remap_ids)

        base_extras = (
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja",
        )
        for extra_name in dict.fromkeys((*base_extras, *extra_files)):
            src_extra = sdir / extra_name
            if src_extra.exists():
                shutil.copy2(src_extra, staging_dir / Path(extra_name).name)

        provenance = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(sdir),
            "source_shards": shard_names,
            "source_config_sha256": cfg_sha256,
            "quantization_policy": {
                "enabled": bool(enable_quant),
                "quant_mode": eff_mode,
                "bits": eff_bits if enable_quant else None,
                "group_size": int(eff_gs) if enable_quant else None,
                "mode": "affine" if enable_quant else None,
                "keep_mlp_bf16": bool(eff_keep_mlp_bf16),
                "ple_sidecar": ple_sidecar_meta_name,
                "quantized_module_count": len(quantized_modules),
                "bf16_mlp_count": len(bf16_mlp_keys),
                "unquantized_other_count": len(unquantized_other_keys),
            },
            "vocab_policy": {
                "input_embeddings_full_rows_preserved": True,
                "full_vocab_size": inferred_vocab,
                "lm_head_vocab_out": report.vocab_out_size,
                "remapped_lm_head_keys": remapped_vocab_keys,
                "materialize_pruned_lm_head": bool(materialize_pruned_lm_head),
            },
            "strip_summary": {
                "strip_multimodal": bool(strip_multimodal),
                "bytes_before": strip_rep.bytes_before,
                "bytes_removed": strip_rep.bytes_removed,
                "bytes_after": strip_rep.bytes_after,
                "bytes_by_prefix": strip_rep.bytes_by_prefix,
                "removed_key_count": len(strip_rep.removed_keys),
            },
            "output_shards": output_shards,
            "output_total_bytes": output_total_bytes,
            "shard_sha256": shard_sha256s,
        }
        (staging_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        _atomic_publish_dir(staging_dir, ddir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    return report


def reload_and_dequantize_dir_cpu(ckpt_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, RawTensor], dict[str, Any]]:
    """Pure-CPU reload helper unpacking `U32` `affine` (`2`/`4`/`8`-bit) -> `float32` alongside raw tensors."""
    cdir = Path(ckpt_dir)
    raw, _, _ = load_safetensors_dir(cdir)
    cfg = json.loads((cdir / "config.json").read_text(encoding="utf-8")) if (cdir / "config.json").exists() else {}
    qcfg = cfg.get("quantization", {}) if isinstance(cfg.get("quantization"), dict) else {}
    default_gs = int(qcfg.get("group_size", DEFAULT_GROUP_SIZE))
    default_bits = int(qcfg.get("bits", DEFAULT_BITS))

    deq: dict[str, np.ndarray] = {}
    for k, t in raw.items():
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        if k.endswith(".weight"):
            prefix = k[: -len(".weight")]
            s_key = f"{prefix}.scales"
            b_key = f"{prefix}.biases"
            if s_key in raw:
                raw_meta = qcfg.get(prefix, {})
                meta = raw_meta if isinstance(raw_meta, dict) else {}
                gs = int(meta.get("group_size", default_gs))
                bits = int(meta.get("bits", default_bits))
                deq[k] = dequantize_affine_u32(t, raw[s_key], raw.get(b_key), bits=bits, group_size=gs)
                continue
        deq[k] = t.to_float32()
    return deq, raw, cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pure-CPU Gemma-4 Checkpoint Packager & Multimodal Stripper")
    parser.add_argument("src_dir", type=Path, help="Input safetensors checkpoint directory")
    parser.add_argument("dst_dir", nargs="?", type=Path, default=None, help="Output packed directory (omit for --inspect-only)")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect headers & multimodal bytes only")
    parser.add_argument("--vocab-remap-npz", type=Path, default=None)
    parser.add_argument("--vocab-out", type=int, default=DEFAULT_KEPT_VOCAB_SIZE)
    parser.add_argument("--no-vocab-remap", action="store_true")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument(
        "--quant-mode",
        choices=("auto", "8bit_bf16_mlp", "qat_q4_0"),
        default="auto",
        help="Quantization policy ('auto' selects qat_q4_0 for *-qat-q4_0-* checkpoints)",
    )
    parser.add_argument("--qat-q4-0", action="store_true", help="Alias for --quant-mode qat_q4_0")
    parser.add_argument("--ple-sidecar-int8", action="store_true", help="Emit clean-page mmap INT8 embed_tokens_per_layer sidecar")
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Optional sidecar file inside src_dir to copy into dst_dir (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.inspect_only or args.dst_dir is None:
        hdrs = inspect_safetensors_headers(args.src_dir)
        tot = sum(v["nbytes"] for v in hdrs.values())
        mm = sum(v["nbytes"] for k, v in hdrs.items() if is_multimodal_key(k) is not None)
        print(f"Total tensors={len(hdrs)} bytes={tot:,} ({tot / 1024**3:.3f} GiB) | multimodal={mm:,} ({mm / 1024**3:.3f} GiB)")
        return 0

    eff_qmode = "qat_q4_0" if args.qat_q4_0 else args.quant_mode
    rep = pack_checkpoint_dir(
        args.src_dir,
        args.dst_dir,
        quantize_non_mlp_8bit=not args.no_quantize,
        quant_mode=eff_qmode,
        ple_sidecar_int8=args.ple_sidecar_int8,
        vocab_remap_npz=args.vocab_remap_npz,
        vocab_out=None if args.no_vocab_remap else args.vocab_out,
        extra_files=args.extra_file,
    )
    print(
        f"Packed -> {args.dst_dir}: mode={rep.quant_mode}, shards={rep.output_shards}, bytes={rep.output_total_bytes:,}, "
        f"stripped={rep.strip_report.bytes_removed:,} ({rep.strip_report.gib_removed:.3f} GiB), "
        f"quant_modules={len(rep.quantized_modules)}, bf16_mlps={len(rep.bf16_mlp_keys)}, vocab_out={rep.vocab_out_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
