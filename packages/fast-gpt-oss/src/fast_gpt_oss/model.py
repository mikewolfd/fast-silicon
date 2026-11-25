"""Bare-Metal Metal 4 + `MXFP4` + Hydragen v2 + SWA-128 `FastGptOssModel`.

Provides:
  - `GptOssConfig`: parsed model geometry cleanly supporting both full `openai/gpt-oss-20b`
    (`H=2880, D_ff=2880, n_q=64, n_kv=8, hd=64, swa=128, E=32`) and compact synthetic
    test shapes.
  - Pure CPU/NumPy helpers (`make_yarn_rope_np`, `infer_affine_quant_spec`,
    `pack_gpt_oss_layer_tensors`, `prune_lm_head_tensors`) testable without GPU contention.
  - `FastGptOssModel`: streamed per-shard safetensors loading (`mx.clear_cache()` per shard),
    retained cold `lm_head` fallback (`_cold_lm_head` + `compute_full_logits_1d`),
    Apple10 (`M5 NAX`) `n_tok >= 2` `mx.gather_qmm(..., lhs_indices=..., sorted_indices=True)`
    and `mx.fast.scaled_dot_product_attention(..., sinks=...)` routing, Apple9 Metal 4
    `Hot-MPP / Cold-QMV` + `swa128_prefill_mma` fast paths, and `append_batch_suffixes`
    for zero-recompute two-phase batched budgeted reasoning.
"""

from __future__ import annotations

import gc
import glob
import json
import math
import os
import platform
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np

from .kernels import KernelSpec, build_kernels
from .tokenizer import FULL_VOCAB_SIZE, O200kVocabRemapper


def _parse_ver_tuple(ver_str: str) -> tuple[int, ...]:
    out: list[int] = []
    for tok in str(ver_str).split("."):
        m = re.match(r"(\d+)", tok)
        if not m:
            break
        out.append(int(m.group(1)))
    return tuple(out)


def _detect_m5_nax_moe(mx: Any) -> bool:
    """Return True on Apple10 (M5+) + macOS >= 26.2 + MLX >= 0.31.2 with NAX MoE/segmented_mm support."""
    if os.environ.get("MLX_METAL_NO_NAX") == "1":
        return False
    if not hasattr(mx, "segmented_mm"):
        return False
    try:
        dev = mx.device_info()
    except Exception:
        return False
    dev_name = str(dev.get("device_name", "")).lower()
    arch = str(dev.get("architecture", "")).lower()
    is_m5 = ("m5" in dev_name) or ("apple10" in arch) or ("g17" in arch)
    if not is_m5:
        return False
    mac_ver = _parse_ver_tuple(platform.mac_ver()[0])
    if mac_ver and mac_ver < (26, 2):
        return False
    mlx_ver = _parse_ver_tuple(getattr(mx, "__version__", "0.0.0"))
    return mlx_ver >= (0, 31, 2)


def _f32_to_u32(val: float) -> int:
    return int(np.array([val], dtype=np.float32).view(np.uint32)[0])


def _peek_all_checkpoint_keys(model_dir: Path, shards: Sequence[str]) -> set[str] | None:
    """Inspect expected tensor keys from `model.safetensors.index.json` or binary headers (0 tensor bytes read)."""
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.is_file():
        try:
            data = json.loads(idx_file.read_text(encoding="utf-8"))
            wm = data.get("weight_map")
            if isinstance(wm, dict) and wm:
                return set(str(k) for k in wm.keys())
        except Exception:
            pass

    keys: set[str] = set()
    try:
        for sp in shards:
            with open(sp, "rb") as fp:
                hdr_raw = fp.read(8)
                if len(hdr_raw) != 8:
                    return None
                hdr_len = struct.unpack("<Q", hdr_raw)[0]
                if hdr_len <= 0 or hdr_len > (64 << 20):
                    return None
                hdr = json.loads(fp.read(hdr_len).decode("utf-8"))
                if not isinstance(hdr, dict):
                    return None
                for k in hdr.keys():
                    if k != "__metadata__":
                        keys.add(str(k))
        return keys if keys else None
    except Exception:
        return None


def make_yarn_rope_np(
    dims: int = 64,
    base: float = 150000.0,
    factor: float = 32.0,
    orig_ctx: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Compute YaRN inverse frequencies (`float32[dims//2]`) and attention mscale on CPU."""
    if factor <= 1.0:
        freqs = (base ** (np.arange(0, dims, 2, dtype=np.float64) / float(dims))).astype(np.float32)
        return freqs, 1.0

    def _find_corr(num_rot: float) -> float:
        return (dims * math.log(orig_ctx / (num_rot * 2.0 * math.pi))) / (2.0 * math.log(base))

    low = max(math.floor(_find_corr(beta_fast)), 0)
    high = min(math.ceil(_find_corr(beta_slow)), dims - 1)
    mscale = 0.1 * 1.0 * math.log(factor) + 1.0

    half = dims // 2
    denom = float(high - low) if high != low else 0.001
    lin = (np.arange(half, dtype=np.float64) - float(low)) / denom
    ramp = np.clip(lin, 0.0, 1.0)
    freq_mask = 1.0 - ramp
    freq_extra = base ** (np.arange(0, dims, 2, dtype=np.float64) / float(dims))
    freq_inter = factor * freq_extra
    freqs = (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1.0 - freq_mask))
    return freqs.astype(np.float32), float(mscale)


def make_yarn_rope(
    dims: int = 64,
    base: float = 150000.0,
    factor: float = 32.0,
    orig_ctx: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[Any, float]:
    """Return `(mx.array[dims//2] float32, mscale)` YaRN RoPE schedule."""
    import mlx.core as mx

    freqs_np, mscale = make_yarn_rope_np(
        dims=dims,
        base=base,
        factor=factor,
        orig_ctx=orig_ctx,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    return mx.array(freqs_np, dtype=mx.float32), mscale


@dataclass(frozen=True)
class GptOssConfig:
    """Parsed `openai/gpt-oss-20b` configuration with derived projection sizes."""

    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_hidden_layers: int = 24
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    sliding_window: int = 128
    rms_norm_eps: float = 1e-5
    vocab_size: int = FULL_VOCAB_SIZE
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702
    rope_theta: float = 150000.0
    rope_factor: float = 32.0
    rope_orig_ctx: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    layer_types: tuple[str, ...] = ()

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def qkv_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GptOssConfig":
        n_layers = int(raw.get("num_hidden_layers", 24))
        default_types = tuple(
            "sliding_attention" if (i % 2 == 0) else "full_attention"
            for i in range(n_layers)
        )
        rs = raw.get("rope_scaling") or {}
        return cls(
            hidden_size=int(raw.get("hidden_size", 2880)),
            intermediate_size=int(raw.get("intermediate_size", raw.get("hidden_size", 2880))),
            num_hidden_layers=n_layers,
            num_attention_heads=int(raw.get("num_attention_heads", 64)),
            num_key_value_heads=int(raw.get("num_key_value_heads", 8)),
            head_dim=int(raw.get("head_dim", 64)),
            sliding_window=int(raw.get("sliding_window", 128)),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            vocab_size=int(raw.get("vocab_size", FULL_VOCAB_SIZE)),
            num_local_experts=int(raw.get("num_local_experts", 32)),
            num_experts_per_tok=int(raw.get("num_experts_per_tok", 4)),
            swiglu_limit=float(raw.get("swiglu_limit", 7.0)),
            swiglu_alpha=float(raw.get("swiglu_alpha", 1.702)),
            rope_theta=float(raw.get("rope_theta", 150000.0)),
            rope_factor=float(rs.get("factor", 32.0)),
            rope_orig_ctx=int(rs.get("original_max_position_embeddings", 4096)),
            rope_beta_fast=float(rs.get("beta_fast", 32.0)),
            rope_beta_slow=float(rs.get("beta_slow", 1.0)),
            layer_types=tuple(raw.get("layer_types") or default_types),
        )

    def to_kernel_spec(self, *, moe_hot_threshold: int = 8) -> KernelSpec:
        return KernelSpec(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_local_experts=self.num_local_experts,
            sliding_window=self.sliding_window,
            swiglu_limit=self.swiglu_limit,
            swiglu_alpha=self.swiglu_alpha,
            moe_hot_threshold=int(moe_hot_threshold),
        )


def infer_affine_quant_spec(
    weight_shape: tuple[int, ...],
    scales_shape: tuple[int, ...] | None,
    in_features: int,
) -> tuple[bool, int, int]:
    """Infer `(is_quantized, bits, group_size)` purely from tensor shapes."""
    if scales_shape is None:
        return False, 16, 0
    w_cols = int(weight_shape[-1])
    s_cols = int(scales_shape[-1])
    if w_cols <= 0 or s_cols <= 0:
        raise ValueError(f"Invalid quantized tensor shapes: weight={weight_shape}, scales={scales_shape}")
    if in_features % w_cols != 0:
        raise ValueError(f"in_features={in_features} not divisible by weight_cols={w_cols}")
    elems_per_u32 = in_features // w_cols
    if 32 % elems_per_u32 != 0:
        raise ValueError(f"Invalid packed elements per uint32: {elems_per_u32}")
    bits = 32 // elems_per_u32
    if in_features % s_cols != 0:
        raise ValueError(f"in_features={in_features} not divisible by scales_cols={s_cols}")
    group_size = in_features // s_cols
    return True, bits, group_size


def _concat_arrays(arrays: list[Any], axis: int, dtype: Any | None = None) -> Any:
    if isinstance(arrays[0], np.ndarray):
        out = np.concatenate(arrays, axis=axis)
        return out.astype(dtype) if dtype is not None else out
    import mlx.core as mx

    out = mx.concatenate(arrays, axis=axis)
    return out.astype(dtype) if dtype is not None else out


def _as_dtype(arr: Any, dtype: Any) -> Any:
    if isinstance(arr, np.ndarray):
        return arr.astype(dtype)
    return arr.astype(dtype)


def _gather_rows(arr: Any, idx: np.ndarray) -> Any:
    if isinstance(arr, np.ndarray):
        return arr[idx]
    import mlx.core as mx

    return arr[mx.array(idx, dtype=mx.int32)]


def pack_gpt_oss_layer_tensors(
    weights: dict[str, Any],
    layer_idx: int,
    layer_type: str,
    hidden_size: int,
    q_dim: int,
    *,
    f16_dtype: Any = np.float16,
    f32_dtype: Any = np.float32,
    bf16_dtype: Any | None = None,
) -> dict[str, Any]:
    """Pack QKV projections and Expert Gate+Up projections for a single layer.

    Works identically on both NumPy arrays (for zero-GPU CPU tests) and MLX arrays.
    """
    p = f"model.layers.{layer_idx}."
    q_w = weights[p + "self_attn.q_proj.weight"]
    k_w = weights[p + "self_attn.k_proj.weight"]
    v_w = weights[p + "self_attn.v_proj.weight"]
    q_s = weights.get(p + "self_attn.q_proj.scales")
    qkv_quant, qkv_bits, qkv_gs = infer_affine_quant_spec(
        tuple(q_w.shape),
        tuple(q_s.shape) if q_s is not None else None,
        hidden_size,
    )
    qkv_w = _concat_arrays([q_w, k_w, v_w], axis=0)
    if qkv_quant:
        qkv_s = _concat_arrays(
            [
                weights[p + "self_attn.q_proj.scales"],
                weights[p + "self_attn.k_proj.scales"],
                weights[p + "self_attn.v_proj.scales"],
            ],
            axis=0,
            dtype=f16_dtype,
        )
        qkv_b = _concat_arrays(
            [
                weights[p + "self_attn.q_proj.biases"],
                weights[p + "self_attn.k_proj.biases"],
                weights[p + "self_attn.v_proj.biases"],
            ],
            axis=0,
            dtype=f16_dtype,
        )
    else:
        qkv_w = _as_dtype(qkv_w, f16_dtype)
        qkv_s = None
        qkv_b = None
    qkv_bias = _concat_arrays(
        [
            weights[p + "self_attn.q_proj.bias"],
            weights[p + "self_attn.k_proj.bias"],
            weights[p + "self_attn.v_proj.bias"],
        ],
        axis=0,
        dtype=f16_dtype,
    )

    o_w = weights[p + "self_attn.o_proj.weight"]
    o_s_raw = weights.get(p + "self_attn.o_proj.scales")
    o_quant, o_bits, o_gs = infer_affine_quant_spec(
        tuple(o_w.shape),
        tuple(o_s_raw.shape) if o_s_raw is not None else None,
        q_dim,
    )

    r_w = weights[p + "mlp.router.weight"]
    r_s_raw = weights.get(p + "mlp.router.scales")
    r_quant, r_bits, r_gs = infer_affine_quant_spec(
        tuple(r_w.shape),
        tuple(r_s_raw.shape) if r_s_raw is not None else None,
        hidden_size,
    )

    gu_w = _concat_arrays(
        [
            weights[p + "mlp.experts.gate_proj.weight"],
            weights[p + "mlp.experts.up_proj.weight"],
        ],
        axis=1,
    )
    gu_s = _concat_arrays(
        [
            weights[p + "mlp.experts.gate_proj.scales"],
            weights[p + "mlp.experts.up_proj.scales"],
        ],
        axis=1,
    )
    gu_bias = _concat_arrays(
        [
            weights[p + "mlp.experts.gate_proj.bias"],
            weights[p + "mlp.experts.up_proj.bias"],
        ],
        axis=1,
        dtype=f16_dtype,
    )

    in_ln_w = weights[p + "input_layernorm.weight"]
    pa_ln_w = weights[p + "post_attention_layernorm.weight"]
    if bf16_dtype is not None:
        in_ln_w = _as_dtype(in_ln_w, bf16_dtype)
        pa_ln_w = _as_dtype(pa_ln_w, bf16_dtype)

    return {
        "is_full": (layer_type == "full_attention"),
        "in_ln": in_ln_w,
        "pa_ln": pa_ln_w,
        "sinks_f32": _as_dtype(weights[p + "self_attn.sinks"], f32_dtype),
        "sinks_f16": _as_dtype(weights[p + "self_attn.sinks"], f16_dtype),
        "qkv_w": qkv_w,
        "qkv_s": qkv_s,
        "qkv_b": qkv_b,
        "qkv_bias": qkv_bias,
        "qkv_quant": qkv_quant,
        "qkv_bits": qkv_bits,
        "qkv_gs": qkv_gs,
        "o_w": o_w if o_quant else _as_dtype(o_w, f16_dtype),
        "o_s": _as_dtype(o_s_raw, f16_dtype) if o_quant else None,
        "o_b": _as_dtype(weights[p + "self_attn.o_proj.biases"], f16_dtype) if o_quant else None,
        "o_bias": _as_dtype(weights[p + "self_attn.o_proj.bias"], f16_dtype),
        "o_quant": o_quant,
        "o_bits": o_bits,
        "o_gs": o_gs,
        "r_w": r_w if r_quant else _as_dtype(r_w, f16_dtype),
        "r_w_f32": None if r_quant else _as_dtype(r_w, f32_dtype),
        "r_s": _as_dtype(r_s_raw, f16_dtype) if r_quant else None,
        "r_b": _as_dtype(weights[p + "mlp.router.biases"], f16_dtype) if r_quant else None,
        "r_bias": _as_dtype(weights[p + "mlp.router.bias"], f16_dtype),
        "r_bias_f32": _as_dtype(weights[p + "mlp.router.bias"], f32_dtype),
        "r_quant": r_quant,
        "r_bits": r_bits,
        "r_gs": r_gs,
        "gu_w": gu_w,
        "gu_s": gu_s,
        "gu_bias": gu_bias,
        "dn_w": weights[p + "mlp.experts.down_proj.weight"],
        "dn_s": weights[p + "mlp.experts.down_proj.scales"],
        "dn_bias": _as_dtype(weights[p + "mlp.experts.down_proj.bias"], f16_dtype),
    }


def prune_lm_head_tensors(
    lm_w: Any,
    lm_s: Any | None,
    lm_b: Any | None,
    kept_ids: np.ndarray | None,
) -> tuple[Any, Any | None, Any | None, np.ndarray, np.ndarray]:
    """Slice `lm_head` weights/scales/biases to `kept_ids` (no-op if `kept_ids is None`)."""
    full_v = int(lm_w.shape[0])
    if kept_ids is None or len(kept_ids) >= full_v:
        full_map = np.arange(full_v, dtype=np.int32)
        return lm_w, lm_s, lm_b, full_map, full_map

    kept = np.asarray(kept_ids, dtype=np.int32)
    pw = _gather_rows(lm_w, kept)
    ps = _gather_rows(lm_s, kept) if lm_s is not None else None
    pb = _gather_rows(lm_b, kept) if lm_b is not None else None
    o2n = np.full(full_v, -1, dtype=np.int32)
    o2n[kept] = np.arange(len(kept), dtype=np.int32)
    return pw, ps, pb, kept, o2n


_COMPILED_SWIGLU_CACHE: dict[tuple[float, float], Any] = {}
_COMPILED_ROUTE_CACHE: dict[int, Any] = {}
_COMPILED_ADD_RESID_BF16: Any = None
_COMPILED_B1_MOE_COMBINE: Any = None
_COMPILED_NAX_MOE_COMBINE: Any = None


def _get_compiled_swiglu(limit: float = 7.0, alpha: float = 1.702):
    key = (float(limit), float(alpha))
    fn = _COMPILED_SWIGLU_CACHE.get(key)
    if fn is None:
        import mlx.core as mx

        def _fn(gu: Any, bias: Any) -> Any:
            gu = gu + bias
            g, u = mx.split(gu, 2, axis=-1)
            g = mx.minimum(g, key[0])
            u = mx.clip(u, -key[0], key[0])
            return (u + 1.0) * (g * mx.sigmoid(g * key[1]))

        fn = mx.compile(_fn)
        _COMPILED_SWIGLU_CACHE[key] = fn
    return fn


def _get_compiled_route(k: int = 4):
    ik = int(k)
    fn = _COMPILED_ROUTE_CACHE.get(ik)
    if fn is None:
        import mlx.core as mx

        def _fn(logits: Any) -> tuple[Any, Any]:
            lg_f32 = logits.astype(mx.float32)
            idx = mx.argpartition(lg_f32, kth=-ik, axis=-1)[..., -ik:]
            vals = mx.take_along_axis(lg_f32, idx, axis=-1)
            scores = mx.softmax(vals, axis=-1, precise=True)
            return idx, scores

        fn = mx.compile(_fn)
        _COMPILED_ROUTE_CACHE[ik] = fn
    return fn


def _get_compiled_add_resid_bf16():
    global _COMPILED_ADD_RESID_BF16
    if _COMPILED_ADD_RESID_BF16 is None:
        import mlx.core as mx

        def _fn(h: Any, delta: Any) -> Any:
            return h + delta.astype(mx.bfloat16)

        _COMPILED_ADD_RESID_BF16 = mx.compile(_fn)
    return _COMPILED_ADD_RESID_BF16


def _get_compiled_b1_moe_combine():
    global _COMPILED_B1_MOE_COMBINE
    if _COMPILED_B1_MOE_COMBINE is None:
        import mlx.core as mx

        def _fn(h: Any, dn: Any, dn_bias_e: Any, scores: Any) -> Any:
            summed = dn[:, 0, :].astype(mx.float32) + dn_bias_e.astype(mx.float32)
            weighted = summed * scores[0, :, None].astype(mx.float32)
            return h + weighted.sum(axis=0).reshape(1, 1, -1).astype(mx.bfloat16)

        _COMPILED_B1_MOE_COMBINE = mx.compile(_fn)
    return _COMPILED_B1_MOE_COMBINE


def _get_compiled_nax_moe_combine():
    global _COMPILED_NAX_MOE_COMBINE
    if _COMPILED_NAX_MOE_COMBINE is None:
        import mlx.core as mx

        def _fn(h_2d: Any, y_2d: Any, dn_bias_e: Any, inv_i: Any, scores_2d: Any) -> Any:
            yw = (y_2d + dn_bias_e)[inv_i].reshape(h_2d.shape[0], 4, h_2d.shape[1])
            acc = mx.sum(yw.astype(mx.float32) * scores_2d[..., None].astype(mx.float32), axis=-2)
            return h_2d + acc.astype(mx.bfloat16)

        _COMPILED_NAX_MOE_COMBINE = mx.compile(_fn)
    return _COMPILED_NAX_MOE_COMBINE


class FastGptOssModel:
    """Bare-metal MLX + Metal 4 MPP / Apple10 NAX model implementation for `openai/gpt-oss-20b`."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        vocab_out: int | None = None,
        vocab_remap_npz: str | Path | None = None,
        tokenizer_json_path: str | Path | None = None,
        compile_kernels: bool = True,
    ) -> None:
        import mlx.core as mx

        self.model_dir = str(Path(model_dir).expanduser().resolve())
        with open(Path(self.model_dir) / "config.json", encoding="utf-8") as f:
            self.raw_config = json.load(f)
        self.config = GptOssConfig.from_dict(self.raw_config)

        self.num_layers = self.config.num_hidden_layers
        self.H = self.config.hidden_size
        self.d_ff = self.config.intermediate_size
        self.n_q = self.config.num_attention_heads
        self.n_kv = self.config.num_key_value_heads
        self.hd = self.config.head_dim
        self.q_dim = self.config.q_dim
        self.kv_dim = self.config.kv_dim
        self.qkv_dim = self.config.qkv_dim
        self.qkv_split_idx = [self.q_dim, self.q_dim + self.kv_dim]
        self.swa = self.config.sliding_window
        self.eps = self.config.rms_norm_eps
        self.num_local_experts = self.config.num_local_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok
        self.layer_types = list(self.config.layer_types)
        self.scale = float(self.hd**-0.5)
        self.scale_u32 = _f32_to_u32(self.scale)
        self.eps_arr = mx.array([self.eps], dtype=mx.float32)

        self.rope_freqs, self.rope_mscale = make_yarn_rope(
            dims=self.hd,
            base=self.config.rope_theta,
            factor=self.config.rope_factor,
            orig_ctx=self.config.rope_orig_ctx,
            beta_fast=self.config.rope_beta_fast,
            beta_slow=self.config.rope_beta_slow,
        )
        self.mscale_arr = mx.array([self.rope_mscale], dtype=mx.float32)

        self._cold_lm_head: dict[str, Any] | None = None
        self._last_hf_2d: Any | None = None
        self._load(
            vocab_out=vocab_out,
            vocab_remap_npz=vocab_remap_npz,
            tokenizer_json_path=tokenizer_json_path,
        )

        # Infer actual MoE intermediate_size from packed down_proj if config omitted it.
        dn_s0 = self.layers[0]["dn_s"]
        if dn_s0 is not None and int(dn_s0.shape[-1]) * 32 != self.d_ff:
            self.d_ff = int(dn_s0.shape[-1]) * 32
            self.config = GptOssConfig.from_dict({**self.raw_config, "intermediate_size": self.d_ff})

        self._lidx4 = mx.arange(4, dtype=mx.uint32)
        self._empty_o_sys = mx.zeros((1, self.n_q, self.hd), dtype=mx.float16)
        self._empty_lse_sys = mx.zeros((1, self.n_q), dtype=mx.float32)
        mx.eval(self._lidx4, self._empty_o_sys, self._empty_lse_sys)

        self._swiglu_c = _get_compiled_swiglu(self.config.swiglu_limit, self.config.swiglu_alpha)
        self._route_c = _get_compiled_route(self.num_experts_per_tok)
        self._add_resid_bf16_c = _get_compiled_add_resid_bf16()
        self._b1_moe_combine_c = _get_compiled_b1_moe_combine()
        self._nax_moe_combine_c = _get_compiled_nax_moe_combine()
        self._is_m5_nax_moe = _detect_m5_nax_moe(mx)
        self._moe_hot_threshold = 8
        self._moe_tile_chunk_tokens = 256
        self._fast_all_quant = bool(
            self.emb_quant
            and self.lm_quant
            and all(ld["qkv_quant"] and ld["o_quant"] and ld["r_quant"] for ld in self.layers)
        )

        self._kernels: dict[str, Any] | None = None
        if compile_kernels:
            self._ensure_kernels()

    def _resolve_kept_vocab_ids(
        self,
        *,
        vocab_out: int | None,
        vocab_remap_npz: str | Path | None,
        tokenizer_json_path: str | Path | None,
    ) -> np.ndarray | None:
        if not (vocab_out and 0 < int(vocab_out) < self.full_vocab_size):
            return None
        target_v = int(vocab_out)
        kept_ids: np.ndarray | None = None
        npz_p = Path(vocab_remap_npz) if vocab_remap_npz is not None else None
        tok_json = Path(tokenizer_json_path) if tokenizer_json_path else Path(self.model_dir) / "tokenizer.json"
        if npz_p is not None and npz_p.is_file():
            cached_kept = np.asarray(np.load(npz_p)["kept_old_ids"], dtype=np.int32)
            if len(cached_kept) == target_v:
                kept_ids = cached_kept
        if kept_ids is None and tok_json.exists():
            remap = O200kVocabRemapper.load_or_build(
                tok_json,
                cache_npz_path=vocab_remap_npz,
                target_size=target_v,
                full_vocab_size=self.full_vocab_size,
            )
            kept_ids = remap.kept_old_ids
        elif kept_ids is None:
            warnings.warn(
                f"Skipping lm_head vocab pruning to {target_v} (missing tokenizer.json={tok_json} "
                f"and matching vocab_remap_npz={vocab_remap_npz}); retaining full vocab={self.full_vocab_size}.",
                UserWarning,
                stacklevel=3,
            )
        if kept_ids is not None:
            valid_mask = (kept_ids >= 0) & (kept_ids < self.full_vocab_size)
            if not bool(np.all(valid_mask)):
                kept_ids = kept_ids[valid_mask]
            if len(kept_ids) == 0:
                kept_ids = None
        return kept_ids

    def _load(
        self,
        *,
        vocab_out: int | None,
        vocab_remap_npz: str | Path | None,
        tokenizer_json_path: str | Path | None,
    ) -> None:
        """Stream safetensors shards one file at a time, releasing raw tensors + `mx.clear_cache()` per shard."""
        import mlx.core as mx

        mdir = Path(self.model_dir)
        shards = sorted(glob.glob(str(mdir / "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files found in {self.model_dir}")

        all_keys = _peek_all_checkpoint_keys(mdir, shards)
        embed_keys = {k for k in all_keys if k.startswith("model.embed_tokens.")} if all_keys else set()
        lm_keys = {k for k in all_keys if k.startswith("lm_head.")} if all_keys else set()
        layer_keys = [
            ({k for k in all_keys if k.startswith(f"model.layers.{i}.")} if all_keys else set())
            for i in range(self.num_layers)
        ]

        pending: dict[str, Any] = {}
        emb_loaded = False
        norm_loaded = False
        lm_loaded = False
        layers_by_idx: dict[int, dict[str, Any]] = {}

        for s_idx, shard_path in enumerate(shards):
            is_last = s_idx == (len(shards) - 1)
            shard_tensors = mx.load(shard_path)
            pending.update(shard_tensors)
            del shard_tensors
            shard_eval: list[Any] = []

            if not emb_loaded and "model.embed_tokens.weight" in pending:
                if (embed_keys and embed_keys.issubset(pending)) or is_last:
                    self.emb_w = pending.pop("model.embed_tokens.weight")
                    emb_s_raw = pending.pop("model.embed_tokens.scales", None)
                    emb_b_raw = pending.pop("model.embed_tokens.biases", None)
                    self.emb_quant, self.emb_bits, self.emb_gs = infer_affine_quant_spec(
                        tuple(self.emb_w.shape),
                        tuple(emb_s_raw.shape) if emb_s_raw is not None else None,
                        self.H,
                    )
                    self.emb_s = emb_s_raw.astype(mx.float16) if self.emb_quant and emb_s_raw is not None else None
                    self.emb_b = emb_b_raw.astype(mx.float16) if self.emb_quant and emb_b_raw is not None else None
                    shard_eval.append(self.emb_w)
                    if self.emb_s is not None:
                        shard_eval.append(self.emb_s)
                    if self.emb_b is not None:
                        shard_eval.append(self.emb_b)
                    emb_loaded = True

            if not norm_loaded and "model.norm.weight" in pending:
                self.final_norm = pending.pop("model.norm.weight").astype(mx.bfloat16)
                shard_eval.append(self.final_norm)
                norm_loaded = True

            if not lm_loaded and "lm_head.weight" in pending:
                if (lm_keys and lm_keys.issubset(pending)) or is_last:
                    lm_w = pending.pop("lm_head.weight")
                    lm_s_raw = pending.pop("lm_head.scales", None)
                    lm_b_raw = pending.pop("lm_head.biases", None)
                    self.lm_quant, self.lm_bits, self.lm_gs = infer_affine_quant_spec(
                        tuple(lm_w.shape),
                        tuple(lm_s_raw.shape) if lm_s_raw is not None else None,
                        self.H,
                    )
                    lm_s = lm_s_raw.astype(mx.float16) if self.lm_quant and lm_s_raw is not None else None
                    lm_b = lm_b_raw.astype(mx.float16) if self.lm_quant and lm_b_raw is not None else None
                    del lm_s_raw, lm_b_raw

                    self.full_vocab_size = int(lm_w.shape[0])
                    kept_ids = self._resolve_kept_vocab_ids(
                        vocab_out=vocab_out,
                        vocab_remap_npz=vocab_remap_npz,
                        tokenizer_json_path=tokenizer_json_path,
                    )
                    self.lm_w, self.lm_s, self.lm_b, self.new_to_old, self.old_to_new = prune_lm_head_tensors(
                        lm_w,
                        lm_s,
                        lm_b,
                        kept_ids,
                    )
                    if not self.lm_quant:
                        self.lm_w = self.lm_w.astype(mx.float16)
                    self.vocab_n = int(self.lm_w.shape[0])

                    if self.vocab_n < self.full_vocab_size:
                        cold_ids = np.flatnonzero(self.old_to_new < 0).astype(np.int32)
                        cw = _gather_rows(lm_w, cold_ids)
                        if not self.lm_quant:
                            cw = cw.astype(mx.float16)
                        cs = _gather_rows(lm_s, cold_ids) if lm_s is not None else None
                        cb = _gather_rows(lm_b, cold_ids) if lm_b is not None else None
                        self._cold_lm_head = {"w": cw, "s": cs, "b": cb, "ids": cold_ids}
                        shard_eval.append(cw)
                        if cs is not None:
                            shard_eval.append(cs)
                        if cb is not None:
                            shard_eval.append(cb)
                    else:
                        self._cold_lm_head = None

                    del lm_w, lm_s, lm_b
                    shard_eval.append(self.lm_w)
                    if self.lm_s is not None:
                        shard_eval.append(self.lm_s)
                    if self.lm_b is not None:
                        shard_eval.append(self.lm_b)
                    lm_loaded = True

            for li in range(self.num_layers):
                if li in layers_by_idx:
                    continue
                pref = f"model.layers.{li}."
                if layer_keys[li]:
                    ready = layer_keys[li].issubset(pending)
                else:
                    ready = is_last and (pref + "self_attn.q_proj.weight") in pending
                if not ready:
                    continue
                ld = pack_gpt_oss_layer_tensors(
                    pending,
                    li,
                    self.layer_types[li],
                    self.H,
                    self.q_dim,
                    f16_dtype=mx.float16,
                    f32_dtype=mx.float32,
                    bf16_dtype=mx.bfloat16,
                )
                for k_drop in [k for k in pending.keys() if k.startswith(pref)]:
                    del pending[k_drop]
                layers_by_idx[li] = ld
                shard_eval.extend(v for v in ld.values() if isinstance(v, mx.array))

            if shard_eval:
                mx.eval(*shard_eval)  # guard: allow-eval
            del shard_eval
            gc.collect()
            mx.clear_cache()

        pending.clear()
        self.layers = [layers_by_idx[i] for i in range(self.num_layers)]
        mx.eval(self.rope_freqs, self.eps_arr, self.mscale_arr)

    def _ensure_kernels(self) -> dict[str, Any]:
        if self._kernels is None:
            self._kernels = build_kernels(
                self.config.to_kernel_spec(moe_hot_threshold=self._moe_hot_threshold)
            )
        return self._kernels

    @property
    def kernels(self) -> dict[str, Any]:
        return self._ensure_kernels()

    def _linear(
        self,
        x: Any,
        w: Any,
        s: Any | None,
        b: Any | None,
        *,
        is_quant: bool,
        bits: int,
        group_size: int,
        bias: Any | None = None,
    ) -> Any:
        import mlx.core as mx

        if is_quant:
            out = mx.quantized_matmul(
                x, w, s, b, transpose=True, group_size=group_size, bits=bits, mode="affine"
            )
        else:
            out = x @ w.T
        if bias is not None:
            out = out + bias
        return out

    def compute_full_logits_1d(
        self,
        pruned_logits_1d: np.ndarray,
        *,
        row_idx: int = 0,
        hf_2d: Any | None = None,
    ) -> np.ndarray | None:
        """Reconstruct full `[full_vocab_size]` logits lazily using `_cold_lm_head` on cold grammar misses."""
        import mlx.core as mx

        if len(pruned_logits_1d) >= self.full_vocab_size:
            return np.asarray(pruned_logits_1d, dtype=np.float32)
        cold = self._cold_lm_head
        if cold is None:
            return None
        src_hf = hf_2d if hf_2d is not None else self._last_hf_2d
        if src_hf is None:
            return None
        hf_row = src_hf.reshape(-1, self.H)[row_idx : row_idx + 1]
        cold_lg = self._linear(
            hf_row,
            cold["w"],
            cold["s"],
            cold["b"],
            is_quant=self.lm_quant,
            bits=self.lm_bits,
            group_size=self.lm_gs,
        )
        mx.eval(cold_lg)
        full = np.full(self.full_vocab_size, -1e9, dtype=np.float32)
        full[self.new_to_old] = np.asarray(pruned_logits_1d, dtype=np.float32)
        full[cold["ids"]] = np.asarray(cold_lg.reshape(-1).astype(mx.float32))
        return full

    def embed(self, tokens: Any) -> Any:
        import mlx.core as mx

        if self.emb_quant:
            return mx.dequantize(
                self.emb_w[tokens],
                self.emb_s[tokens],
                self.emb_b[tokens],
                group_size=self.emb_gs,
                bits=self.emb_bits,
                mode="affine",
                dtype=mx.bfloat16,
            )
        return self.emb_w[tokens].astype(mx.bfloat16)

    def rms_f16(self, h: Any, w: Any) -> Any:
        import mlx.core as mx

        if self._kernels is None:
            return mx.fast.rms_norm(h, w, self.eps).astype(mx.float16)
        shape = h.shape
        n_tok = int(math.prod(shape[:-1]))
        out = self._kernels["rmsnorm_bf16_to_f16"](
            inputs=[h.reshape(n_tok, self.H), w, self.eps_arr],
            grid=(n_tok * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_tok, self.H)],
            output_dtypes=[mx.float16],
        )[0]
        return out.reshape(shape)

    def rms_f16_and_f32(self, h: Any, w: Any) -> tuple[Any, Any]:
        """Compute `fp16` expert input alongside `fp32` router input from one `bfloat16` RMSNorm pass."""
        import mlx.core as mx

        if (
            self._kernels is None
            or "rmsnorm_bf16_to_f16_f32" not in self._kernels
            or h.dtype != mx.bfloat16
            or w.dtype != mx.bfloat16
        ):
            out_f32 = mx.fast.rms_norm(h.astype(mx.float32), w.astype(mx.float32), self.eps)
            return out_f32.astype(mx.float16), out_f32
        shape = h.shape
        n_tok = int(math.prod(shape[:-1]))
        out16, out32 = self._kernels["rmsnorm_bf16_to_f16_f32"](  # guard: allow-fp32
            inputs=[h.reshape(n_tok, self.H), w, self.eps_arr],
            grid=(n_tok * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_tok, self.H), (n_tok, self.H)],
            output_dtypes=[mx.float16, mx.float32],
        )
        return out16.reshape(shape), out32.reshape(shape)

    def _route_f32(self, h: Any, ld: dict[str, Any]) -> tuple[Any, Any, Any]:
        """Compute `(pn_f16, topk_inds, topk_scores_f32)` with exact `float32` router projection + softmax."""
        import mlx.core as mx

        pn, pn_f32 = self.rms_f16_and_f32(h, ld["pa_ln"])
        bias_f32 = ld["r_bias_f32"] if ld.get("r_bias_f32") is not None else ld["r_bias"].astype(mx.float32)
        if ld["r_quant"]:
            rl = (
                mx.quantized_matmul(
                    pn_f32,
                    ld["r_w"],
                    ld["r_s"],
                    ld["r_b"],
                    transpose=True,
                    group_size=ld["r_gs"],
                    bits=ld["r_bits"],
                    mode="affine",
                )
                + bias_f32
            )
        else:
            r_w_f32 = ld["r_w_f32"] if ld.get("r_w_f32") is not None else ld["r_w"].astype(mx.float32)
            rl = mx.matmul(pn_f32, r_w_f32.T) + bias_f32
        inds, scores = self._route_c(rl.reshape(-1, self.num_local_experts))
        return pn, inds, scores

    def _make_swa_additive_mask(self, l_q: int, kv_len: int, dtype: Any) -> Any:
        """Build `Apple10` SWA-128 additive mask in `dtype` (or `'causal'` when `kv_len <= swa`)."""
        import mlx.core as mx

        if kv_len <= self.swa:
            return "causal"
        swa_offset = kv_len - l_q
        q_pos = mx.arange(swa_offset, swa_offset + l_q, dtype=mx.int32)[:, None]
        k_pos = mx.arange(kv_len, dtype=mx.int32)[None, :]
        diff = q_pos - k_pos
        valid = (diff >= 0) & (diff < self.swa)
        return mx.where(valid, mx.array(0.0, dtype=dtype), mx.array(-40000.0, dtype=dtype))

    def _swa_prefill_attn(self, q: Any, k_att: Any, v_att: Any, ld: dict[str, Any]) -> Any:
        """Route `Apple10` (`NAX`) SWA prefill to `mx.fast.scaled_dot_product_attention(..., sinks=...)`
        while keeping `Apple9` M4 `swa128_prefill_mma` intact."""
        import mlx.core as mx

        l_q = int(q.shape[2])
        kv_len = int(k_att.shape[2])
        if self._is_m5_nax_moe or self._kernels is None:
            swa_mask = self._make_swa_additive_mask(l_q, kv_len, q.dtype)
            return mx.fast.scaled_dot_product_attention(
                q, k_att, v_att, scale=self.scale, mask=swa_mask, sinks=ld["sinks_f16"]
            )
        swa_offset = kv_len - l_q
        p_arr = mx.array([l_q, kv_len, swa_offset, self.swa, self.scale_u32], dtype=mx.uint32)
        q_c = mx.contiguous(q)
        k_c = mx.contiguous(k_att)
        v_c = mx.contiguous(v_att)
        return self.kernels["swa128_prefill_mma"](
            inputs=[q_c, k_c, v_c, ld["sinks_f32"], p_arr],
            grid=(((l_q + 7) // 8) * 256, self.n_kv, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, self.n_q, l_q, self.hd)],
            output_dtypes=[mx.float16],
        )[0]

    def _moe_tile_mpp_chunk(
        self,
        pn_2d: Any,
        h_2d: Any,
        inds_2d: Any,
        scores_2d: Any,
        ld: dict[str, Any],
    ) -> Any:
        """Single-launch Hybrid Hot-MPP + Cold-QMV MoE residual update with tight cold grid (`Apple9` fast path)."""
        import mlx.core as mx

        n_tok = int(pn_2d.shape[0])
        n_assign = n_tok * 4
        hot_thresh = int(self._moe_hot_threshold)
        max_hot = n_assign // (hot_thresh + 1)
        max_cold = min(self.num_local_experts, n_assign)
        grid_y = max(1, max_hot + max_cold)
        hot_meta_len = max(1, max_hot) * 3
        cold_meta_len = max(1, max_cold) * 3
        ntok_arr = mx.array([n_tok], dtype=mx.uint32)

        tok_idx, slot_idx, rweights, hot_meta, cold_meta, meta_counts = self.kernels["moe_build_tiles"](  # guard: allow-fp32
            inputs=[inds_2d.astype(mx.uint32), scores_2d.astype(mx.float32), ntok_arr],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(n_assign,), (n_assign,), (n_assign,), (hot_meta_len,), (cold_meta_len,), (2,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.uint32, mx.uint32, mx.uint32],
        )
        mid = self.kernels["moe_batch_gu_unified"](
            inputs=[pn_2d, tok_idx, ld["gu_w"], ld["gu_s"], ld["gu_bias"], hot_meta, cold_meta, meta_counts],
            grid=((self.d_ff // 8) * 128, grid_y, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_assign, self.d_ff)],
            output_dtypes=[mx.float16],
        )[0]
        y_slots = self.kernels["moe_batch_dn_unified"](  # guard: allow-fp32
            inputs=[mid, slot_idx, rweights, ld["dn_w"], ld["dn_s"], ld["dn_bias"], hot_meta, cold_meta, meta_counts],
            grid=((self.H // 16) * 128, grid_y, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_tok, 4, self.H)],
            output_dtypes=[mx.float32],
        )[0]
        return self.kernels["moe_reduce4_resid_bf16"](
            inputs=[y_slots, h_2d],
            grid=(self.H // 4, n_tok, 1),
            threadgroup=(16, 1, 1),
            output_shapes=[(n_tok, self.H)],
            output_dtypes=[mx.bfloat16],
        )[0]

    def _moe_gather_qmm_nax(
        self,
        pn_2d: Any,
        h_2d: Any,
        inds_2d: Any,
        scores_2d: Any,
        ld: dict[str, Any],
    ) -> Any:
        """Apple10 (`M5 NAX`) sorted `gather_qmm` MoE path (`n_tok >= 2`, `lhs_indices = ord_i // 4`, `-75%` gather writes)."""
        import mlx.core as mx

        n_tok = int(pn_2d.shape[0])
        fl = inds_2d.reshape(-1)
        ord_i = mx.argsort(fl)
        inv_i = mx.argsort(ord_i)
        sorted_exp = fl[ord_i].astype(mx.uint32)
        lhs_idx = (ord_i // 4).astype(mx.uint32)
        gu = mx.gather_qmm(
            pn_2d.reshape(n_tok, 1, self.H),
            ld["gu_w"],
            ld["gu_s"],
            lhs_indices=lhs_idx,
            rhs_indices=sorted_exp,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
            sorted_indices=True,
        )
        gu_bias = ld["gu_bias"][sorted_exp]
        if gu_bias.dtype != gu.dtype:
            gu_bias = gu_bias.astype(gu.dtype)
        act = self._swiglu_c(gu.reshape(-1, 2 * self.d_ff), gu_bias).reshape(
            4 * n_tok, 1, self.d_ff
        )
        y = mx.gather_qmm(  # guard: allow-lhs-indices
            act,
            ld["dn_w"],
            ld["dn_s"],
            rhs_indices=sorted_exp,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
            sorted_indices=True,
        )
        dn_bias = ld["dn_bias"][sorted_exp]
        if dn_bias.dtype != y.dtype:
            dn_bias = dn_bias.astype(y.dtype)
        return self._nax_moe_combine_c(h_2d, y.reshape(-1, self.H), dn_bias, inv_i, scores_2d)

    def _moe_forward(
        self,
        pn_2d: Any,
        h_2d: Any,
        inds_2d: Any,
        scores_2d: Any,
        ld: dict[str, Any],
    ) -> Any:
        """Unified multi-token MoE residual update: Hot-MPP + Cold-QMV (`hot_threshold=8`) with NAX fallback."""
        import mlx.core as mx

        n_tok = int(pn_2d.shape[0])
        if self._kernels is None:
            return self._moe_gather_qmm_nax(pn_2d, h_2d, inds_2d, scores_2d, ld)

        chunk = int(self._moe_tile_chunk_tokens)
        if n_tok <= chunk:
            return self._moe_tile_mpp_chunk(pn_2d, h_2d, inds_2d, scores_2d, ld)

        outs = []
        for s0 in range(0, n_tok, chunk):
            s1 = min(n_tok, s0 + chunk)
            outs.append(
                self._moe_tile_mpp_chunk(
                    pn_2d[s0:s1],
                    h_2d[s0:s1],
                    inds_2d[s0:s1],
                    scores_2d[s0:s1],
                    ld,
                )
            )
        return mx.concatenate(outs, axis=0)

    def _apply_rope(self, q: Any, k: Any, offset: int) -> tuple[Any, Any]:
        import mlx.core as mx

        if self.rope_mscale != 1.0:
            q = q * mx.array(self.rope_mscale, dtype=q.dtype)
            k = k * mx.array(self.rope_mscale, dtype=k.dtype)
        q = mx.fast.rope(
            q, self.hd, traditional=False, base=None, scale=1.0, offset=offset, freqs=self.rope_freqs
        )
        k = mx.fast.rope(
            k, self.hd, traditional=False, base=None, scale=1.0, offset=offset, freqs=self.rope_freqs
        )
        return q, k

    def make_b1_cache(self, max_full_len: int) -> list[dict[str, Any]]:
        import mlx.core as mx

        c: list[dict[str, Any]] = []
        eff_full = max(1, int(max_full_len))
        to_eval = []
        for lt in self.layer_types:
            is_full = lt == "full_attention"
            cap = eff_full if is_full else self.swa
            k = mx.zeros((1, self.n_kv, cap, self.hd), dtype=mx.float16)
            v = mx.zeros((1, self.n_kv, cap, self.hd), dtype=mx.float16)
            to_eval.extend((k, v))
            c.append({"k": k, "v": v, "pos": 0, "idx": 0, "cap": cap, "is_full": is_full})
        mx.eval(*to_eval)
        return c

    def forward_b1(self, tokens: Any, cache: list[dict[str, Any]], last_only: bool = True) -> Any:
        """Exact `B=1` forward for both prefill (`L>1`) and single-token decode (`L=1`)."""
        import mlx.core as mx

        b_sz, l_seq = tokens.shape
        if b_sz != 1:
            raise ValueError(f"forward_b1 requires B=1, got {b_sz}")
        h = self.embed(tokens)

        if l_seq == 1:
            if self._fast_all_quant:
                qmm = mx.quantized_matmul
                gmm = mx.gather_qmm
                sdpa = mx.fast.scaled_dot_product_attention
                mscale_f = self.rope_mscale
                mscale_arr = mx.array(mscale_f, dtype=mx.float16) if mscale_f != 1.0 else None
                for li, ld in enumerate(self.layers):
                    xn = self.rms_f16(h, ld["in_ln"])
                    qkv = qmm(xn, ld["qkv_w"], ld["qkv_s"], ld["qkv_b"], transpose=True, group_size=ld["qkv_gs"], bits=ld["qkv_bits"], mode="affine") + ld["qkv_bias"]
                    q, k, v = mx.split(qkv, self.qkv_split_idx, axis=-1)
                    q = q.reshape(1, 1, self.n_q, self.hd).transpose(0, 2, 1, 3)
                    k = k.reshape(1, 1, self.n_kv, self.hd).transpose(0, 2, 1, 3)
                    v = v.reshape(1, 1, self.n_kv, self.hd).transpose(0, 2, 1, 3)

                    lc = cache[li]
                    offset = lc["pos"]
                    if mscale_arr is not None:
                        q = q * mscale_arr
                        k = k * mscale_arr
                    q = mx.fast.rope(q, self.hd, traditional=False, base=None, scale=1.0, offset=offset, freqs=self.rope_freqs)
                    k = mx.fast.rope(k, self.hd, traditional=False, base=None, scale=1.0, offset=offset, freqs=self.rope_freqs)

                    if lc["is_full"]:
                        lc["k"][:, :, offset : offset + 1, :] = k
                        lc["v"][:, :, offset : offset + 1, :] = v
                        lc["pos"] = offset + 1
                        k_att = lc["k"][:, :, : offset + 1, :]
                        v_att = lc["v"][:, :, : offset + 1, :]
                    else:
                        cap = lc["cap"]
                        slot = lc["idx"] % cap
                        lc["k"][:, :, slot : slot + 1, :] = k
                        lc["v"][:, :, slot : slot + 1, :] = v
                        lc["pos"] = offset + 1
                        lc["idx"] += 1
                        valid = min(lc["pos"], cap)
                        k_att = lc["k"][:, :, :valid, :]
                        v_att = lc["v"][:, :, :valid, :]

                    att = sdpa(q, k_att, v_att, scale=self.scale, mask=None, sinks=ld["sinks_f16"])
                    att_1d = att.transpose(0, 2, 1, 3).reshape(1, 1, self.q_dim)
                    o_out = qmm(att_1d, ld["o_w"], ld["o_s"], ld["o_b"], transpose=True, group_size=ld["o_gs"], bits=ld["o_bits"], mode="affine") + ld["o_bias"]
                    h = self._add_resid_bf16_c(h, o_out)

                    pn, inds, scores = self._route_f32(h, ld)
                    exp_ids = inds[0].astype(mx.uint32)
                    gu = gmm(pn.reshape(1, 1, 1, self.H), ld["gu_w"], ld["gu_s"], None, rhs_indices=exp_ids, transpose=True, group_size=32, bits=4, mode="mxfp4")
                    act = self._swiglu_c(gu.reshape(4, 2 * self.d_ff), ld["gu_bias"][exp_ids]).reshape(4, 1, 1, self.d_ff)
                    dn = gmm(act, ld["dn_w"], ld["dn_s"], None, lhs_indices=self._lidx4, rhs_indices=exp_ids, transpose=True, group_size=32, bits=4, mode="mxfp4")
                    h = self._b1_moe_combine_c(h, dn, ld["dn_bias"][exp_ids], scores)

                hf = self.rms_f16(h, self.final_norm)
                self._last_hf_2d = hf.reshape(-1, self.H)
                return qmm(hf, self.lm_w, self.lm_s, self.lm_b, transpose=True, group_size=self.lm_gs, bits=self.lm_bits, mode="affine")

            for li, ld in enumerate(self.layers):
                xn = self.rms_f16(h, ld["in_ln"])
                qkv = self._linear(
                    xn,
                    ld["qkv_w"],
                    ld["qkv_s"],
                    ld["qkv_b"],
                    is_quant=ld["qkv_quant"],
                    bits=ld["qkv_bits"],
                    group_size=ld["qkv_gs"],
                    bias=ld["qkv_bias"],
                )
                q, k, v = mx.split(qkv, self.qkv_split_idx, axis=-1)
                q = q.reshape(1, 1, self.n_q, self.hd).transpose(0, 2, 1, 3)
                k = k.reshape(1, 1, self.n_kv, self.hd).transpose(0, 2, 1, 3)
                v = v.reshape(1, 1, self.n_kv, self.hd).transpose(0, 2, 1, 3)

                lc = cache[li]
                offset = int(lc["pos"])
                q, k = self._apply_rope(q, k, offset=offset)

                if lc["is_full"]:
                    lc["k"][:, :, offset : offset + 1, :] = k
                    lc["v"][:, :, offset : offset + 1, :] = v
                    lc["pos"] = offset + 1
                    k_att = lc["k"][:, :, : offset + 1, :]
                    v_att = lc["v"][:, :, : offset + 1, :]
                else:
                    cap = int(lc["cap"])
                    slot = int(lc["idx"]) % cap
                    lc["k"][:, :, slot : slot + 1, :] = k
                    lc["v"][:, :, slot : slot + 1, :] = v
                    lc["pos"] = offset + 1
                    lc["idx"] = int(lc["idx"]) + 1
                    valid = min(int(lc["pos"]), cap)
                    k_att = lc["k"][:, :, :valid, :]
                    v_att = lc["v"][:, :, :valid, :]

                att = mx.fast.scaled_dot_product_attention(
                    q, k_att, v_att, scale=self.scale, mask=None, sinks=ld["sinks_f16"]
                )
                att_1d = att.transpose(0, 2, 1, 3).reshape(1, 1, self.q_dim)
                o_out = self._linear(
                    att_1d,
                    ld["o_w"],
                    ld["o_s"],
                    ld["o_b"],
                    is_quant=ld["o_quant"],
                    bits=ld["o_bits"],
                    group_size=ld["o_gs"],
                    bias=ld["o_bias"],
                )
                h = self._add_resid_bf16_c(h, o_out)

                pn, inds, scores = self._route_f32(h, ld)
                exp_ids = inds[0].astype(mx.uint32)
                gu = mx.gather_qmm(  # guard: allow-lhs-indices guard: allow-unsorted-qmm
                    pn.reshape(1, 1, 1, self.H),
                    ld["gu_w"],
                    ld["gu_s"],
                    None,
                    rhs_indices=exp_ids,
                    transpose=True,
                    group_size=32,
                    bits=4,
                    mode="mxfp4",
                )
                act = self._swiglu_c(gu.reshape(4, 2 * self.d_ff), ld["gu_bias"][exp_ids]).reshape(
                    4, 1, 1, self.d_ff
                )
                dn = mx.gather_qmm(  # guard: allow-unsorted-qmm
                    act,
                    ld["dn_w"],
                    ld["dn_s"],
                    None,
                    lhs_indices=self._lidx4,
                    rhs_indices=exp_ids,
                    transpose=True,
                    group_size=32,
                    bits=4,
                    mode="mxfp4",
                )
                h = self._b1_moe_combine_c(h, dn, ld["dn_bias"][exp_ids], scores)

            hf = self.rms_f16(h, self.final_norm)
            self._last_hf_2d = hf.reshape(-1, self.H)
            return self._linear(
                hf,
                self.lm_w,
                self.lm_s,
                self.lm_b,
                is_quant=self.lm_quant,
                bits=self.lm_bits,
                group_size=self.lm_gs,
            )

        for li, ld in enumerate(self.layers):
            xn = self.rms_f16(h, ld["in_ln"])
            qkv = self._linear(
                xn,
                ld["qkv_w"],
                ld["qkv_s"],
                ld["qkv_b"],
                is_quant=ld["qkv_quant"],
                bits=ld["qkv_bits"],
                group_size=ld["qkv_gs"],
                bias=ld["qkv_bias"],
            )
            q, k, v = mx.split(qkv, self.qkv_split_idx, axis=-1)
            q = q.reshape(1, l_seq, self.n_q, self.hd).transpose(0, 2, 1, 3)
            k = k.reshape(1, l_seq, self.n_kv, self.hd).transpose(0, 2, 1, 3)
            v = v.reshape(1, l_seq, self.n_kv, self.hd).transpose(0, 2, 1, 3)

            lc = cache[li]
            offset = int(lc["pos"])
            q, k = self._apply_rope(q, k, offset=offset)

            if lc["is_full"]:
                lc["k"][:, :, offset : offset + l_seq, :] = k
                lc["v"][:, :, offset : offset + l_seq, :] = v
                lc["pos"] = offset + l_seq
                k_att = lc["k"][:, :, : offset + l_seq, :]
                v_att = lc["v"][:, :, : offset + l_seq, :]
                att = mx.fast.scaled_dot_product_attention(
                    q, k_att, v_att, scale=self.scale, mask="causal", sinks=ld["sinks_f16"]
                )
            else:
                cap = int(lc["cap"])
                end_pos = offset + l_seq
                if offset == 0:
                    k_att, v_att = k, v
                else:
                    valid = min(offset, cap)
                    ring_idx = mx.arange(lc["idx"] - valid, lc["idx"], dtype=mx.int32) % cap
                    k_att = mx.concatenate([lc["k"][:, :, ring_idx, :], k], axis=2)
                    v_att = mx.concatenate([lc["v"][:, :, ring_idx, :], v], axis=2)

                n_tail = min(int(k_att.shape[2]), cap)
                lc["k"][:, :, :n_tail, :] = k_att[:, :, -n_tail:, :]
                lc["v"][:, :, :n_tail, :] = v_att[:, :, -n_tail:, :]
                lc["pos"] = end_pos
                lc["idx"] = n_tail
                att = self._swa_prefill_attn(q, k_att, v_att, ld)

            att_flat = att.transpose(0, 2, 1, 3).reshape(1, l_seq, self.q_dim)
            o_out = self._linear(
                att_flat,
                ld["o_w"],
                ld["o_s"],
                ld["o_b"],
                is_quant=ld["o_quant"],
                bits=ld["o_bits"],
                group_size=ld["o_gs"],
                bias=ld["o_bias"],
            )
            h = self._add_resid_bf16_c(h, o_out)

            pn, inds, scores = self._route_f32(h, ld)
            h = self._moe_forward(
                pn.reshape(l_seq, self.H),
                h.reshape(l_seq, self.H),
                inds.reshape(l_seq, 4),
                scores.reshape(l_seq, 4),
                ld,
            ).reshape(1, l_seq, self.H)

        hf = self.rms_f16(h[:, -1:, :] if last_only else h, self.final_norm)
        self._last_hf_2d = hf.reshape(-1, self.H)[-1:]
        return self._linear(
            hf,
            self.lm_w,
            self.lm_s,
            self.lm_b,
            is_quant=self.lm_quant,
            bits=self.lm_bits,
            group_size=self.lm_gs,
        )

    # ------------------------------------------------------------------
    # Shared Prefix + Hydragen v2 + Zero-Sync Hybrid Metal 4 MoE Batch
    # ------------------------------------------------------------------
    def compute_shared_prefix(self, prefix_ids: list[int] | np.ndarray) -> dict[str, Any]:
        """Precompute prefix KV once (safe for `L_sys == 0`, `< swa`, or `>= swa`)."""
        import mlx.core as mx

        ids = [int(t) for t in prefix_ids]
        l_sys = len(ids)
        if l_sys == 0:
            layers = [
                {"is_full": (lt == "full_attention"), "k": None, "v": None, "swa_len": 0}
                for lt in self.layer_types
            ]
            return {"L_sys": 0, "layers": layers}

        c = self.make_b1_cache(l_sys)
        mx.eval(self.forward_b1(mx.array(ids, dtype=mx.int32)[None, :], c, last_only=True))
        sys_kv: list[dict[str, Any]] = []
        to_eval = []
        for lc in c:
            if lc["is_full"]:
                k_out = lc["k"][0, :, :l_sys, :]
                v_out = lc["v"][0, :, :l_sys, :]
                sys_kv.append({"is_full": True, "k": k_out, "v": v_out, "swa_len": 0})
                to_eval.extend((k_out, v_out))
            else:
                valid = min(int(lc["pos"]), self.swa)
                ring_idx = mx.arange(lc["idx"] - valid, lc["idx"], dtype=mx.int32) % lc["cap"]
                tail_k = mx.take(lc["k"][0], ring_idx, axis=1)
                tail_v = mx.take(lc["v"][0], ring_idx, axis=1)
                if valid == self.swa:
                    k_buf, v_buf = tail_k, tail_v
                else:
                    # Right-align partial SWA window at [swa - valid : swa] so ring write col=0 wraps cleanly.
                    k_buf = mx.zeros((self.n_kv, self.swa, self.hd), dtype=mx.float16)
                    v_buf = mx.zeros((self.n_kv, self.swa, self.hd), dtype=mx.float16)
                    k_buf[:, self.swa - valid : self.swa, :] = tail_k
                    v_buf[:, self.swa - valid : self.swa, :] = tail_v
                sys_kv.append({"is_full": False, "k": k_buf, "v": v_buf, "swa_len": valid})
                to_eval.extend((k_buf, v_buf))
        mx.eval(*to_eval)
        return {"L_sys": l_sys, "layers": sys_kv}

    def clone_b1_cache_from_prefix(self, sp: dict[str, Any], max_full_len: int) -> list[dict[str, Any]]:
        """Create a mutable `B=1` KV cache initialized from `compute_shared_prefix`."""
        import mlx.core as mx

        l_sys = int(sp["L_sys"])
        c1 = self.make_b1_cache(max(1, max_full_len))
        if l_sys == 0:
            return c1

        to_eval = []
        for li in range(self.num_layers):
            spl = sp["layers"][li]
            lc = c1[li]
            if lc["is_full"]:
                lc["k"][0, :, :l_sys, :] = spl["k"]
                lc["v"][0, :, :l_sys, :] = spl["v"]
                lc["pos"] = l_sys
                to_eval.extend((lc["k"], lc["v"]))
            else:
                valid = int(spl["swa_len"])
                lc["k"][0, :, :valid, :] = spl["k"][:, self.swa - valid : self.swa, :]
                lc["v"][0, :, :valid, :] = spl["v"][:, self.swa - valid : self.swa, :]
                lc["pos"] = l_sys
                lc["idx"] = valid
                to_eval.extend((lc["k"], lc["v"]))
        mx.eval(*to_eval)
        return c1

    def alloc_batch_cache(
        self,
        shared_prefix: dict[str, Any],
        B: int,
        suf_max: int,
        max_dec: int,
    ) -> dict[str, Any]:
        """Allocate zero-prefix-clone batch KV cache for Hydragen v2 + SWA-128 decode."""
        import mlx.core as mx

        cap_full = max(1, int(suf_max) + int(max_dec))
        l_sys = int(shared_prefix["L_sys"])
        layers: list[dict[str, Any]] = []
        to_eval = []
        for sp in shared_prefix["layers"]:
            if sp["is_full"]:
                ks = mx.zeros((B, self.n_kv, cap_full, self.hd), dtype=mx.float16)
                vs = mx.zeros((B, self.n_kv, cap_full, self.hd), dtype=mx.float16)
                layers.append(
                    {
                        "is_full": True,
                        "k_sys": sp["k"],
                        "v_sys": sp["v"],
                        "k_suf": ks,
                        "v_suf": vs,
                    }
                )
                to_eval.extend((ks, vs))
            else:
                if l_sys > 0 and sp["k"] is not None:
                    ks = mx.repeat(sp["k"][None, ...], B, axis=0)
                    vs = mx.repeat(sp["v"][None, ...], B, axis=0)
                else:
                    ks = mx.zeros((B, self.n_kv, self.swa, self.hd), dtype=mx.float16)
                    vs = mx.zeros((B, self.n_kv, self.swa, self.hd), dtype=mx.float16)
                layers.append({"is_full": False, "k_swa": ks, "v_swa": vs, "swa_sys_len": int(sp["swa_len"])})
                to_eval.extend((ks, vs))
        mx.eval(*to_eval)
        return {
            "B": B,
            "L_sys": l_sys,
            "suf_max": suf_max,
            "cap_full": cap_full,
            "dec_cnt": 0,
            "swa_write": 0,
            "layers": layers,
        }

    def prefill_batch_suffixes(
        self,
        suffix_lists: list[list[int]],
        batch_cache: dict[str, Any],
        chunk_docs: int = 8,
    ) -> Any:
        """Packed varlen QKV+O+MoE suffix prefill across `B` requests (no prefix recompute)."""
        import mlx.core as mx

        b_tot = len(suffix_lists)
        l_sys = int(batch_cache["L_sys"])
        suf_lens = [len(s) for s in suffix_lists]
        batch_cache["suf_max"] = max(suf_lens) if suf_lens else 0
        eff_cd = int(chunk_docs)
        if b_tot > 1 and sum(suf_lens) > 1536 and suf_lens:
            eff_cd = max(1, min(eff_cd, 1536 // max(1, batch_cache["suf_max"])))
        last_logits = []
        last_hfs = []

        for b0 in range(0, b_tot, eff_cd):
            b1 = min(b_tot, b0 + eff_cd)
            sub = suffix_lists[b0:b1]
            sub_lens = [len(x) for x in sub]
            flat_ids: list[int] = []
            for x in sub:
                flat_ids.extend(x)
            h_all = self.embed(mx.array(flat_ids, dtype=mx.int32))

            offs = [0]
            for ln in sub_lens:
                offs.append(offs[-1] + ln)

            for li, ld in enumerate(self.layers):
                bc_l = batch_cache["layers"][li]
                xn = self.rms_f16(h_all, ld["in_ln"])
                qkv = self._linear(
                    xn,
                    ld["qkv_w"],
                    ld["qkv_s"],
                    ld["qkv_b"],
                    is_quant=ld["qkv_quant"],
                    bits=ld["qkv_bits"],
                    group_size=ld["qkv_gs"],
                    bias=ld["qkv_bias"],
                )

                att_chunks = []
                for idx_b, ln in enumerate(sub_lens):
                    b_glob = b0 + idx_b
                    s_tok, e_tok = offs[idx_b], offs[idx_b + 1]
                    qkv_b = qkv[s_tok:e_tok]
                    q, k, v = mx.split(qkv_b, self.qkv_split_idx, axis=-1)
                    q = q.reshape(1, ln, self.n_q, self.hd).transpose(0, 2, 1, 3)
                    k = k.reshape(1, ln, self.n_kv, self.hd).transpose(0, 2, 1, 3)
                    v = v.reshape(1, ln, self.n_kv, self.hd).transpose(0, 2, 1, 3)
                    q, k = self._apply_rope(q, k, offset=l_sys)

                    if bc_l["is_full"]:
                        bc_l["k_suf"][b_glob : b_glob + 1, :, :ln, :] = k
                        bc_l["v_suf"][b_glob : b_glob + 1, :, :ln, :] = v
                        if l_sys > 0:
                            k_att = mx.concatenate([bc_l["k_sys"][None, ...], k], axis=2)
                            v_att = mx.concatenate([bc_l["v_sys"][None, ...], v], axis=2)
                        else:
                            k_att, v_att = k, v
                        att_b = mx.fast.scaled_dot_product_attention(
                            q, k_att, v_att, scale=self.scale, mask="causal", sinks=ld["sinks_f16"]
                        )
                    else:
                        swa_sys_len = int(bc_l["swa_sys_len"])
                        if swa_sys_len > 0:
                            k_pre = bc_l["k_swa"][b_glob : b_glob + 1, :, self.swa - swa_sys_len : self.swa, :]
                            v_pre = bc_l["v_swa"][b_glob : b_glob + 1, :, self.swa - swa_sys_len : self.swa, :]
                            k_att = mx.concatenate([k_pre, k], axis=2)
                            v_att = mx.concatenate([v_pre, v], axis=2)
                        else:
                            k_att, v_att = k, v
                        n_tail = min(int(k_att.shape[2]), self.swa)
                        bc_l["k_swa"][b_glob : b_glob + 1, :, self.swa - n_tail : self.swa, :] = k_att[
                            :, :, -n_tail:, :
                        ]
                        bc_l["v_swa"][b_glob : b_glob + 1, :, self.swa - n_tail : self.swa, :] = v_att[
                            :, :, -n_tail:, :
                        ]
                        att_b = self._swa_prefill_attn(q, k_att, v_att, ld)
                    att_chunks.append(att_b.transpose(0, 2, 1, 3).reshape(ln, self.q_dim))

                att_all = mx.concatenate(att_chunks, axis=0)
                o_out = self._linear(
                    att_all,
                    ld["o_w"],
                    ld["o_s"],
                    ld["o_b"],
                    is_quant=ld["o_quant"],
                    bits=ld["o_bits"],
                    group_size=ld["o_gs"],
                    bias=ld["o_bias"],
                )
                h_all = self._add_resid_bf16_c(h_all, o_out)

                pn, inds, scores = self._route_f32(h_all, ld)
                h_all = self._moe_forward(pn, h_all, inds, scores, ld)

            end_indices = mx.array([offs[i + 1] - 1 for i in range(len(sub_lens))], dtype=mx.int32)
            hf = self.rms_f16(h_all[end_indices], self.final_norm)
            lg = self._linear(
                hf,
                self.lm_w,
                self.lm_s,
                self.lm_b,
                is_quant=self.lm_quant,
                bits=self.lm_bits,
                group_size=self.lm_gs,
            )
            eval_bufs = [lg, hf]
            for layer_c in batch_cache["layers"]:
                eval_bufs.append(layer_c.get("k_suf", layer_c.get("k_swa")))
                eval_bufs.append(layer_c.get("v_suf", layer_c.get("v_swa")))
            mx.eval(*eval_bufs)  # guard: allow-eval
            last_logits.append(lg)
            last_hfs.append(hf)

        self._last_hf_2d = mx.concatenate(last_hfs, axis=0)
        batch_cache["suf_lens_arr"] = mx.array(suf_lens, dtype=mx.uint32)
        batch_cache["swa_init_lens_arr"] = mx.array(
            [min(self.swa, l_sys + ln) for ln in suf_lens], dtype=mx.uint32
        )
        batch_cache["rope_pos_arr"] = mx.array([l_sys + ln for ln in suf_lens], dtype=mx.int32)
        mx.eval(
            batch_cache["suf_lens_arr"],
            batch_cache["swa_init_lens_arr"],
            batch_cache["rope_pos_arr"],
        )
        return mx.concatenate(last_logits, axis=0)

    def append_batch_suffixes(
        self,
        extra_lists: list[list[int]],
        batch_cache: dict[str, Any],
        *,
        p1_done_steps: Sequence[int] | None = None,
        saved_swa: Sequence[list[tuple[Any, Any]] | None] | None = None,
        chunk_docs: int = 8,
    ) -> Any:
        """Append Phase-1->Phase-2 transition + upfront-FF tokens onto `batch_cache` without prefix/suffix recompute."""
        import mlx.core as mx

        b_tot = len(extra_lists)
        l_sys = int(batch_cache["L_sys"])
        suf_max_prev = int(batch_cache["suf_max"])
        dec_cnt_prev = int(batch_cache["dec_cnt"])
        prev_suf_lens = [int(x) for x in batch_cache["suf_lens_arr"].tolist()]
        prev_swa_init = [int(x) for x in batch_cache["swa_init_lens_arr"].tolist()]
        done_steps = [
            int(p1_done_steps[b]) if p1_done_steps is not None else dec_cnt_prev
            for b in range(b_tot)
        ]
        base_suf_lens = [prev_suf_lens[b] + done_steps[b] for b in range(b_tot)]

        # Order each slot's chronological SWA window & compact full-layer Phase-1 decode KV onto suffix tail.
        swa_ordered: dict[tuple[int, int], tuple[Any | None, Any | None]] = {}
        swa_ord = 0
        compact_eval: list[Any] = []
        for li, bc_l in enumerate(batch_cache["layers"]):
            if bc_l["is_full"]:
                for b in range(b_tot):
                    d_b = done_steps[b]
                    s_b = prev_suf_lens[b]
                    if d_b > 0 and s_b != suf_max_prev:
                        k_seg = mx.array(bc_l["k_suf"][b : b + 1, :, suf_max_prev : suf_max_prev + d_b, :])
                        v_seg = mx.array(bc_l["v_suf"][b : b + 1, :, suf_max_prev : suf_max_prev + d_b, :])
                        mx.eval(k_seg, v_seg)  # guard: allow-eval
                        bc_l["k_suf"][b : b + 1, :, s_b : s_b + d_b, :] = k_seg
                        bc_l["v_suf"][b : b + 1, :, s_b : s_b + d_b, :] = v_seg
                compact_eval.extend((bc_l["k_suf"], bc_l["v_suf"]))
            else:
                for b in range(b_tot):
                    if saved_swa is not None and saved_swa[b] is not None:
                        swa_ordered[(li, b)] = saved_swa[b][swa_ord]  # type: ignore[index]
                    else:
                        d_b = done_steps[b]
                        valid_b = min(self.swa, prev_swa_init[b] + d_b)
                        if valid_b > 0:
                            ring_idx = mx.arange(d_b - valid_b, d_b, dtype=mx.int32) % self.swa
                            kp = mx.take(bc_l["k_swa"][b : b + 1], ring_idx, axis=2)
                            vp = mx.take(bc_l["v_swa"][b : b + 1], ring_idx, axis=2)
                            swa_ordered[(li, b)] = (kp, vp)
                            compact_eval.extend((kp, vp))
                        else:
                            swa_ordered[(li, b)] = (None, None)
                swa_ord += 1
        if compact_eval:
            mx.eval(*compact_eval)

        last_logits = []
        last_hfs = []
        for b0 in range(0, b_tot, chunk_docs):
            b1 = min(b_tot, b0 + chunk_docs)
            sub = extra_lists[b0:b1]
            sub_lens = [len(x) for x in sub]
            flat_ids: list[int] = []
            for x in sub:
                flat_ids.extend(x)
            h_all = self.embed(mx.array(flat_ids, dtype=mx.int32))

            offs = [0]
            for ln in sub_lens:
                offs.append(offs[-1] + ln)

            for li, ld in enumerate(self.layers):
                bc_l = batch_cache["layers"][li]
                xn = self.rms_f16(h_all, ld["in_ln"])
                qkv = self._linear(
                    xn,
                    ld["qkv_w"],
                    ld["qkv_s"],
                    ld["qkv_b"],
                    is_quant=ld["qkv_quant"],
                    bits=ld["qkv_bits"],
                    group_size=ld["qkv_gs"],
                    bias=ld["qkv_bias"],
                )

                att_chunks = []
                for idx_b, ln in enumerate(sub_lens):
                    b_glob = b0 + idx_b
                    base_ln = base_suf_lens[b_glob]
                    s_tok, e_tok = offs[idx_b], offs[idx_b + 1]
                    qkv_b = qkv[s_tok:e_tok]
                    q, k, v = mx.split(qkv_b, self.qkv_split_idx, axis=-1)
                    q = q.reshape(1, ln, self.n_q, self.hd).transpose(0, 2, 1, 3)
                    k = k.reshape(1, ln, self.n_kv, self.hd).transpose(0, 2, 1, 3)
                    v = v.reshape(1, ln, self.n_kv, self.hd).transpose(0, 2, 1, 3)
                    q, k = self._apply_rope(q, k, offset=l_sys + base_ln)

                    if bc_l["is_full"]:
                        end_ln = base_ln + ln
                        bc_l["k_suf"][b_glob : b_glob + 1, :, base_ln:end_ln, :] = k
                        bc_l["v_suf"][b_glob : b_glob + 1, :, base_ln:end_ln, :] = v
                        k_suf_b = bc_l["k_suf"][b_glob : b_glob + 1, :, :end_ln, :]
                        v_suf_b = bc_l["v_suf"][b_glob : b_glob + 1, :, :end_ln, :]
                        if l_sys > 0 and bc_l["k_sys"] is not None:
                            k_att = mx.concatenate([bc_l["k_sys"][None, ...], k_suf_b], axis=2)
                            v_att = mx.concatenate([bc_l["v_sys"][None, ...], v_suf_b], axis=2)
                        else:
                            k_att, v_att = k_suf_b, v_suf_b
                        att_b = mx.fast.scaled_dot_product_attention(
                            q, k_att, v_att, scale=self.scale, mask="causal", sinks=ld["sinks_f16"]
                        )
                    else:
                        k_pre, v_pre = swa_ordered[(li, b_glob)]
                        if k_pre is not None and v_pre is not None:
                            k_att = mx.concatenate([k_pre, k], axis=2)
                            v_att = mx.concatenate([v_pre, v], axis=2)
                        else:
                            k_att, v_att = k, v
                        n_tail = min(int(k_att.shape[2]), self.swa)
                        bc_l["k_swa"][b_glob : b_glob + 1, :, self.swa - n_tail : self.swa, :] = k_att[
                            :, :, -n_tail:, :
                        ]
                        bc_l["v_swa"][b_glob : b_glob + 1, :, self.swa - n_tail : self.swa, :] = v_att[
                            :, :, -n_tail:, :
                        ]
                        att_b = self._swa_prefill_attn(q, k_att, v_att, ld)
                    att_chunks.append(att_b.transpose(0, 2, 1, 3).reshape(ln, self.q_dim))

                att_all = mx.concatenate(att_chunks, axis=0)
                o_out = self._linear(
                    att_all,
                    ld["o_w"],
                    ld["o_s"],
                    ld["o_b"],
                    is_quant=ld["o_quant"],
                    bits=ld["o_bits"],
                    group_size=ld["o_gs"],
                    bias=ld["o_bias"],
                )
                h_all = self._add_resid_bf16_c(h_all, o_out)

                pn, inds, scores = self._route_f32(h_all, ld)
                h_all = self._moe_forward(pn, h_all, inds, scores, ld)

            end_indices = mx.array([offs[i + 1] - 1 for i in range(len(sub_lens))], dtype=mx.int32)
            hf = self.rms_f16(h_all[end_indices], self.final_norm)
            lg = self._linear(
                hf,
                self.lm_w,
                self.lm_s,
                self.lm_b,
                is_quant=self.lm_quant,
                bits=self.lm_bits,
                group_size=self.lm_gs,
            )
            eval_bufs = [lg, hf]
            for layer_c in batch_cache["layers"]:
                eval_bufs.append(layer_c.get("k_suf", layer_c.get("k_swa")))
                eval_bufs.append(layer_c.get("v_suf", layer_c.get("v_swa")))
            mx.eval(*eval_bufs)  # guard: allow-eval
            last_logits.append(lg)
            last_hfs.append(hf)

        self._last_hf_2d = mx.concatenate(last_hfs, axis=0)
        new_suf_lens = [base_suf_lens[b] + len(extra_lists[b]) for b in range(b_tot)]
        batch_cache["suf_max"] = max(new_suf_lens) if new_suf_lens else 0
        batch_cache["dec_cnt"] = 0
        batch_cache["swa_write"] = 0
        batch_cache["suf_lens_arr"] = mx.array(new_suf_lens, dtype=mx.uint32)
        batch_cache["swa_init_lens_arr"] = mx.array(
            [min(self.swa, l_sys + ln) for ln in new_suf_lens], dtype=mx.uint32
        )
        batch_cache["rope_pos_arr"] = mx.array([l_sys + ln for ln in new_suf_lens], dtype=mx.int32)
        mx.eval(
            batch_cache["suf_lens_arr"],
            batch_cache["swa_init_lens_arr"],
            batch_cache["rope_pos_arr"],
        )
        return mx.concatenate(last_logits, axis=0)

    def decode_step_batch(
        self,
        tokens_1d: Any,
        batch_cache: dict[str, Any],
        samp_indices: Any | None = None,
        skip_lm_head: bool = False,
    ) -> Any:
        """True batched `B=1..256+` decode step using Hydragen v2 + SWA-128 + Hybrid Metal 4 / NAX MoE."""
        import mlx.core as mx

        b_sz = int(batch_cache["B"])
        l_sys = int(batch_cache["L_sys"])
        suf_max = int(batch_cache["suf_max"])
        cap_full = int(batch_cache["cap_full"])
        dec_cnt = int(batch_cache["dec_cnt"])
        col_full = suf_max + dec_cnt
        swa_write = int(batch_cache["swa_write"])
        col_swa = swa_write % self.swa
        end_mod = (swa_write + 1) % self.swa
        has_sys = 1 if l_sys > 0 else 0

        suf_lens_arr = batch_cache["suf_lens_arr"]
        swa_init_lens_arr = batch_cache["swa_init_lens_arr"]
        rope_pos_arr = batch_cache["rope_pos_arr"]
        p_hyd_p1 = mx.array([b_sz, l_sys, self.scale_u32], dtype=mx.uint32)
        p_hyd_p2 = mx.array([cap_full, suf_max, dec_cnt + 1, self.scale_u32, has_sys], dtype=mx.uint32)
        p_swa = mx.array([self.swa, end_mod, dec_cnt, self.scale_u32], dtype=mx.uint32)

        h = self.embed(tokens_1d)
        n_heads_tot = self.n_q + self.n_kv
        last_layer_idx = self.num_layers - 1

        for li, ld in enumerate(self.layers):
            bc_l = batch_cache["layers"][li]
            xn = self.rms_f16(h, ld["in_ln"])
            qkv = self._linear(
                xn,
                ld["qkv_w"],
                ld["qkv_s"],
                ld["qkv_b"],
                is_quant=ld["qkv_quant"],
                bits=ld["qkv_bits"],
                group_size=ld["qkv_gs"],
                bias=ld["qkv_bias"],
            )

            q, k, v = self.kernels["yarn_rope_split_b"](
                inputs=[qkv, rope_pos_arr, self.rope_freqs, self.mscale_arr],
                grid=(n_heads_tot * 32, b_sz, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(b_sz, self.n_q, self.hd), (b_sz, self.n_kv, 1, self.hd), (b_sz, self.n_kv, 1, self.hd)],
                output_dtypes=[mx.float16, mx.float16, mx.float16],
            )

            if bc_l["is_full"]:
                bc_l["k_suf"][:, :, col_full : col_full + 1, :] = k
                bc_l["v_suf"][:, :, col_full : col_full + 1, :] = v
                if skip_lm_head and li == last_layer_idx:
                    break
                if has_sys:
                    o_sys, lse_sys = self.kernels["hydragen_p1_mma"](  # guard: allow-fp32
                        inputs=[q, bc_l["k_sys"], bc_l["v_sys"], p_hyd_p1],
                        grid=(((b_sz + 3) // 4) * 128, self.n_kv, 1),
                        threadgroup=(128, 1, 1),
                        output_shapes=[(b_sz, self.n_q, self.hd), (b_sz, self.n_q)],
                        output_dtypes=[mx.float16, mx.float32],
                    )
                else:
                    o_sys, lse_sys = self._empty_o_sys, self._empty_lse_sys
                att = self.kernels["hydragen_p2_varlen"](
                    inputs=[q, bc_l["k_suf"], bc_l["v_suf"], o_sys, lse_sys, ld["sinks_f32"], suf_lens_arr, p_hyd_p2],
                    grid=(b_sz * 128, self.n_kv, 1),
                    threadgroup=(128, 1, 1),
                    output_shapes=[(b_sz, self.n_q, self.hd)],
                    output_dtypes=[mx.float16],
                )[0]
            else:
                bc_l["k_swa"][:, :, col_swa : col_swa + 1, :] = k
                bc_l["v_swa"][:, :, col_swa : col_swa + 1, :] = v
                if skip_lm_head and li == last_layer_idx:
                    break
                att = self.kernels["swa128_decode"](
                    inputs=[q, bc_l["k_swa"], bc_l["v_swa"], ld["sinks_f32"], swa_init_lens_arr, p_swa],
                    grid=(b_sz * 128, self.n_kv, 1),
                    threadgroup=(128, 1, 1),
                    output_shapes=[(b_sz, self.n_q, self.hd)],
                    output_dtypes=[mx.float16],
                )[0]

            att_flat = att.reshape(b_sz, self.q_dim)
            o_out = self._linear(
                att_flat,
                ld["o_w"],
                ld["o_s"],
                ld["o_b"],
                is_quant=ld["o_quant"],
                bits=ld["o_bits"],
                group_size=ld["o_gs"],
                bias=ld["o_bias"],
            )
            h = self._add_resid_bf16_c(h, o_out)

            pn, inds, scores = self._route_f32(h, ld)
            h = self._moe_forward(pn, h, inds, scores, ld)

        batch_cache["dec_cnt"] = dec_cnt + 1
        batch_cache["swa_write"] = swa_write + 1
        batch_cache["rope_pos_arr"] = rope_pos_arr + 1

        if skip_lm_head:
            self._last_hf_2d = None
            return None

        h_head = h[samp_indices] if samp_indices is not None else h
        hf = self.rms_f16(h_head, self.final_norm)
        self._last_hf_2d = hf
        return self._linear(
            hf,
            self.lm_w,
            self.lm_s,
            self.lm_b,
            is_quant=self.lm_quant,
            bits=self.lm_bits,
            group_size=self.lm_gs,
        )

    def compact_batch_cache(self, batch_cache: dict[str, Any], keep_local: list[int]) -> None:
        """Compact active slots down to `keep_local` mid-wave to drop finished requests."""
        import mlx.core as mx

        idx_mx = mx.array(keep_local, dtype=mx.int32)
        batch_cache["B"] = len(keep_local)
        batch_cache["suf_lens_arr"] = batch_cache["suf_lens_arr"][idx_mx]
        batch_cache["swa_init_lens_arr"] = batch_cache["swa_init_lens_arr"][idx_mx]
        batch_cache["rope_pos_arr"] = batch_cache["rope_pos_arr"][idx_mx]
        eval_list = [
            batch_cache["suf_lens_arr"],
            batch_cache["swa_init_lens_arr"],
            batch_cache["rope_pos_arr"],
        ]
        for layer_c in batch_cache["layers"]:
            if layer_c["is_full"]:
                layer_c["k_suf"] = layer_c["k_suf"][idx_mx]
                layer_c["v_suf"] = layer_c["v_suf"][idx_mx]
                eval_list.extend((layer_c["k_suf"], layer_c["v_suf"]))
            else:
                layer_c["k_swa"] = layer_c["k_swa"][idx_mx]
                layer_c["v_swa"] = layer_c["v_swa"][idx_mx]
                eval_list.extend((layer_c["k_swa"], layer_c["v_swa"]))
        mx.eval(*eval_list)

    def extract_b1_cache(
        self,
        batch_cache: dict[str, Any],
        local_idx: int,
        max_rem: int,
    ) -> list[dict[str, Any]]:
        """Hand off the final remaining batched slot cleanly to `forward_b1`."""
        import mlx.core as mx

        l_sys = int(batch_cache["L_sys"])
        suf_len = int(batch_cache["suf_lens_arr"][local_idx].item())
        suf_max = int(batch_cache["suf_max"])
        dec_cnt = int(batch_cache["dec_cnt"])
        cur_len = l_sys + suf_len + dec_cnt
        cap_alloc = cur_len + max_rem + 16
        swa_write = int(batch_cache["swa_write"])

        c1: list[dict[str, Any]] = []
        to_eval = []
        for layer_c in batch_cache["layers"]:
            if layer_c["is_full"]:
                parts_k = []
                parts_v = []
                if l_sys > 0 and layer_c["k_sys"] is not None:
                    parts_k.append(layer_c["k_sys"][None, ...])
                    parts_v.append(layer_c["v_sys"][None, ...])
                if suf_len > 0:
                    parts_k.append(layer_c["k_suf"][local_idx : local_idx + 1, :, :suf_len, :])
                    parts_v.append(layer_c["v_suf"][local_idx : local_idx + 1, :, :suf_len, :])
                if dec_cnt > 0:
                    parts_k.append(layer_c["k_suf"][local_idx : local_idx + 1, :, suf_max : suf_max + dec_cnt, :])
                    parts_v.append(layer_c["v_suf"][local_idx : local_idx + 1, :, suf_max : suf_max + dec_cnt, :])
                k_full = mx.zeros((1, self.n_kv, cap_alloc, self.hd), dtype=mx.float16)
                v_full = mx.zeros((1, self.n_kv, cap_alloc, self.hd), dtype=mx.float16)
                if parts_k:
                    k_full[:, :, :cur_len, :] = mx.concatenate(parts_k, axis=2)
                    v_full[:, :, :cur_len, :] = mx.concatenate(parts_v, axis=2)
                c1.append({"is_full": True, "k": k_full, "v": v_full, "pos": cur_len, "idx": 0, "cap": cap_alloc})
                to_eval.extend((k_full, v_full))
            else:
                if cur_len >= self.swa:
                    k_swa = mx.array(layer_c["k_swa"][local_idx : local_idx + 1])
                    v_swa = mx.array(layer_c["v_swa"][local_idx : local_idx + 1])
                    swa_idx = swa_write
                else:
                    ring_idx = mx.arange(swa_write - cur_len, swa_write, dtype=mx.int32) % self.swa
                    k_swa = mx.zeros((1, self.n_kv, self.swa, self.hd), dtype=mx.float16)
                    v_swa = mx.zeros((1, self.n_kv, self.swa, self.hd), dtype=mx.float16)
                    if cur_len > 0:
                        k_swa[:, :, :cur_len, :] = mx.take(layer_c["k_swa"][local_idx : local_idx + 1], ring_idx, axis=2)
                        v_swa[:, :, :cur_len, :] = mx.take(layer_c["v_swa"][local_idx : local_idx + 1], ring_idx, axis=2)
                    swa_idx = cur_len
                c1.append({"is_full": False, "k": k_swa, "v": v_swa, "pos": cur_len, "idx": swa_idx, "cap": self.swa})
                to_eval.extend((k_swa, v_swa))
        mx.eval(*to_eval)
        return c1


__all__ = [
    "FastGptOssModel",
    "GptOssConfig",
    "infer_affine_quant_spec",
    "make_yarn_rope",
    "make_yarn_rope_np",
    "pack_gpt_oss_layer_tensors",
    "prune_lm_head_tensors",
]
