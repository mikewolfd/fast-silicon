"""Hybrid `GatedDeltaNet` + Full-Attention + `SparseMoeBlock` `FastQwen3Model` for Qwen3.5 & Qwen3.6."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import glob
import json
import math
from pathlib import Path
import struct
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .kernels import (
    RMSNormGated,
    SpecBucketedLinearGroup,
    apply_partial_rope_trig,
    build_additive_causal_mask,
    compute_gdn_gates,
    dequantize_gdn_state_int8,
    detect_apple10_d256_force_fused,
    detect_rmsnorm_add_one,
    fused_sigmoid_attn_gate,
    fused_swiglu,
    normalize_gdn_qk,
    permute_q_proj_rows_to_q_gate,
    quantize_gdn_state_int8,
    safe_training_sdpa,
    split_linear_q_proj_to_q_gate,
)
from .tokenizer import DEFAULT_QWEN_VOCAB_SIZE, QwenVocabRemapper


def inspect_safetensor_header_keys(shard_path: str | Path) -> set[str]:
    """Read only the 8-byte + JSON safetensors header (`0` tensor bytes read from disk)."""
    with open(shard_path, "rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            return set()
        hdr_len = struct.unpack("<Q", raw_len)[0]
        if hdr_len <= 0 or hdr_len > (64 << 20):
            return set()
        hdr = json.loads(fp.read(hdr_len).decode("utf-8"))
    if not isinstance(hdr, dict):
        return set()
    return {str(k) for k in hdr.keys() if k != "__metadata__"}


def normalize_qwen3_weight_key(raw_key: str) -> str | None:
    """Map VLM-wrapped or standalone HF/MLX Qwen3.5/3.6 keys into canonical namespace, dropping vision keys."""
    if raw_key.startswith(("vision_tower.", "model.visual.", "visual.")):
        return None
    if raw_key.startswith("language_model.model."):
        return "model." + raw_key[len("language_model.model.") :]
    if raw_key.startswith("model.language_model."):
        return "model." + raw_key[len("model.language_model.") :]
    if raw_key.startswith("language_model.lm_head."):
        return "lm_head." + raw_key[len("language_model.lm_head.") :]
    if raw_key.startswith("language_model.mtp."):
        return "mtp." + raw_key[len("language_model.mtp.") :]
    if raw_key.startswith("model.mtp."):
        return "mtp." + raw_key[len("model.mtp.") :]
    if raw_key.startswith(("model.", "lm_head.", "mtp.")):
        return raw_key
    if raw_key.startswith(("layers.", "embed_tokens.", "norm.")):
        return "model." + raw_key
    return None


def normalize_lora_stem(raw_key: str) -> tuple[str, str] | None:
    """Normalize PEFT `base_model.model...lora_A.weight` keys into `(canonical_stem, 'lora_A'|'lora_B')`."""
    k = raw_key
    for pfx in ("base_model.model.", "base_model."):
        if k.startswith(pfx):
            k = k[len(pfx) :]
    norm = normalize_qwen3_weight_key(k)
    if norm is None:
        return None
    for tag in ("lora_A", "lora_B", "lora_a", "lora_b"):
        suf_w = f".{tag}.weight"
        suf_d = f".{tag}.default.weight"
        suf_bare = f".{tag}"
        canon_tag = "lora_A" if tag.lower() == "lora_a" else "lora_B"
        if norm.endswith(suf_w):
            return norm[: -len(suf_w)], canon_tag
        if norm.endswith(suf_d):
            return norm[: -len(suf_d)], canon_tag
        if norm.endswith(suf_bare):
            return norm[: -len(suf_bare)], canon_tag
    return None


@dataclass(frozen=True)
class Qwen3Config:
    """Parsed `Qwen3.5` / `Qwen3.6` configuration supporting dense and sparse MoE hybrids."""

    model_type: str = "qwen3_5_text"
    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = DEFAULT_QWEN_VOCAB_SIZE
    tie_word_embeddings: bool = True
    full_attention_interval: int = 4
    layer_types: tuple[str, ...] = ()
    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    attention_bias: bool = False
    # Sparse MoE (`Qwen3.5/3.6-35B-A3B`)
    num_experts: int = 0
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    # Native 1-layer Multi-Token Prediction (`MTP`)
    mtp_num_hidden_layers: int = 0
    mtp_file: str | None = None

    def __post_init__(self) -> None:
        if not self.layer_types:
            types: list[str] = []
            interval = max(1, int(self.full_attention_interval))
            for i in range(int(self.num_hidden_layers)):
                is_full = ((i + 1) % interval == 0)
                types.append("full_attention" if is_full else "linear_attention")
            object.__setattr__(self, "layer_types", tuple(types))

    @property
    def is_moe(self) -> bool:
        return int(self.num_experts) > 0

    @property
    def rot_dims(self) -> int:
        return int(round(self.head_dim * self.partial_rotary_factor))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Qwen3Config:
        tcfg = dict(raw.get("text_config") or raw)
        rope_p = dict(tcfg.get("rope_parameters") or tcfg.get("rope_scaling") or {})
        num_layers = int(tcfg.get("num_hidden_layers", 32))
        num_q = int(tcfg.get("num_attention_heads", 16))
        hidden = int(tcfg.get("hidden_size", 2560))
        head_dim = int(tcfg.get("head_dim") or (hidden // max(1, num_q)))
        tie_val = tcfg.get("tie_word_embeddings", raw.get("tie_word_embeddings", True))
        raw_ltypes = tcfg.get("layer_types")
        ltypes = tuple(str(x) for x in raw_ltypes[:num_layers]) if isinstance(raw_ltypes, Sequence) and not isinstance(raw_ltypes, str) else ()
        extra_t = raw.get("mlx_lm_extra_tensors")
        mtp_file = raw.get("mtp_file") or (extra_t.get("mtp_file") if isinstance(extra_t, Mapping) else None)
        mtp_layers = int(tcfg.get("mtp_num_hidden_layers", raw.get("mtp_num_hidden_layers", 0)) or 0)
        return cls(
            model_type=str(tcfg.get("model_type", raw.get("model_type", "qwen3_5_text"))),
            hidden_size=hidden,
            intermediate_size=int(tcfg.get("intermediate_size", hidden * 3)),
            num_hidden_layers=num_layers,
            num_attention_heads=num_q,
            num_key_value_heads=int(tcfg.get("num_key_value_heads", num_q)),
            head_dim=head_dim,
            rms_norm_eps=float(tcfg.get("rms_norm_eps", 1e-6)),
            vocab_size=int(tcfg.get("vocab_size", DEFAULT_QWEN_VOCAB_SIZE)),
            tie_word_embeddings=bool(tie_val),
            full_attention_interval=int(tcfg.get("full_attention_interval", 4)),
            layer_types=ltypes,
            linear_num_value_heads=int(tcfg.get("linear_num_value_heads", num_q * 2)),
            linear_num_key_heads=int(tcfg.get("linear_num_key_heads", num_q)),
            linear_key_head_dim=int(tcfg.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(tcfg.get("linear_value_head_dim", 128)),
            linear_conv_kernel_dim=int(tcfg.get("linear_conv_kernel_dim", 4)),
            rope_theta=float(rope_p.get("rope_theta", tcfg.get("rope_theta", 10000000.0))),
            partial_rotary_factor=float(rope_p.get("partial_rotary_factor", tcfg.get("partial_rotary_factor", 0.25))),
            attention_bias=bool(tcfg.get("attention_bias", False)),
            num_experts=int(tcfg.get("num_experts", tcfg.get("num_local_experts", 0)) or 0),
            num_experts_per_tok=int(tcfg.get("num_experts_per_tok", 8)),
            moe_intermediate_size=int(tcfg.get("moe_intermediate_size", 0) or 0),
            shared_expert_intermediate_size=int(tcfg.get("shared_expert_intermediate_size", 0) or 0),
            mtp_num_hidden_layers=mtp_layers,
            mtp_file=str(mtp_file) if mtp_file else None,
        )


class DeltaSlotCache:
    """Fixed-size `O(1)` depthwise-conv + recurrent state for a `GatedDeltaNet` layer, with zero-GEMM rollback."""

    def __init__(self, conv_state: mx.array | None = None, rec_state: mx.array | None = None) -> None:
        self.conv_state = conv_state
        self.rec_state = rec_state
        self.record_verify_trace: bool = False
        self._verify_trace: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array | None] | None = None

    def clear_verify_trace(self) -> None:
        self._verify_trace = None
        self.record_verify_trace = False

    def rollback_to_accepted(self, accepted_tokens: int, conv_k: int) -> None:
        """Advance pre-verify state across `accepted_tokens` using already-computed `(q, k, v, g, beta)` (`0` GEMMs)."""
        if self._verify_trace is None:
            raise RuntimeError("No verify trace recorded on DeltaSlotCache")
        conv_in, q, k, v, g, beta, prev_rec = self._verify_trace
        n_acc = int(accepted_tokens)
        if n_acc <= 0:
            self.conv_state = mx.contiguous(conv_in[:, : conv_k - 1, :])
            self.rec_state = prev_rec
        else:
            self.conv_state = mx.contiguous(conv_in[:, n_acc : n_acc + conv_k - 1, :])
            _, self.rec_state = mx.fast.gated_delta_update(
                q[:, :n_acc],
                k[:, :n_acc],
                v[:, :n_acc],
                g[:, :n_acc],
                beta[:, :n_acc],
                prev_rec,
            )
        self.clear_verify_trace()


class GrowingKVCache:
    """Chunk-allocated `O(1)` decode KV cache with optional per-position `INT8` compression and `rewind_to`."""

    def __init__(self, step: int = 256, *, kv_i8: bool = False) -> None:
        self.step = int(step)
        self.kv_i8 = bool(kv_i8)
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.k_scales: mx.array | None = None
        self.v_scales: mx.array | None = None
        self.offset: int = 0

    def update_and_fetch(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        b, h_kv, s, d = k.shape
        prev = self.offset
        needed = prev + s
        if not self.kv_i8:
            if self.keys is None or needed > int(self.keys.shape[2]) or int(self.keys.shape[0]) != b:
                alloc = ((needed + self.step - 1) // self.step) * self.step
                new_k = mx.zeros((b, h_kv, alloc, d), dtype=k.dtype)
                new_v = mx.zeros((b, h_kv, alloc, d), dtype=v.dtype)
                if self.keys is not None and prev > 0 and int(self.keys.shape[0]) == b:
                    new_k[:, :, :prev, :] = self.keys[:, :, :prev, :]
                    new_v[:, :, :prev, :] = self.values[:, :, :prev, :]  # type: ignore[index]
                self.keys = new_k
                self.values = new_v
            assert self.keys is not None and self.values is not None
            self.keys[:, :, prev:needed, :] = k
            self.values[:, :, prev:needed, :] = v
            self.offset = needed
            return self.keys[:, :, :needed, :], self.values[:, :, :needed, :]

        # Symmetric per-(batch, head, pos) INT8 KV storage (halves KV bandwidth & RAM on large waves / long ctx)
        if self.keys is None or needed > int(self.keys.shape[2]) or int(self.keys.shape[0]) != b:
            alloc = ((needed + self.step - 1) // self.step) * self.step
            new_kq = mx.zeros((b, h_kv, alloc, d), dtype=mx.int8)
            new_vq = mx.zeros((b, h_kv, alloc, d), dtype=mx.int8)
            new_ks = mx.zeros((b, h_kv, alloc, 1), dtype=mx.bfloat16)
            new_vs = mx.zeros((b, h_kv, alloc, 1), dtype=mx.bfloat16)
            if self.keys is not None and prev > 0 and int(self.keys.shape[0]) == b:
                new_kq[:, :, :prev, :] = self.keys[:, :, :prev, :]
                new_vq[:, :, :prev, :] = self.values[:, :, :prev, :]  # type: ignore[index]
                new_ks[:, :, :prev, :] = self.k_scales[:, :, :prev, :]  # type: ignore[index]
                new_vs[:, :, :prev, :] = self.v_scales[:, :, :prev, :]  # type: ignore[index]
            self.keys, self.values = new_kq, new_vq
            self.k_scales, self.v_scales = new_ks, new_vs
        assert self.keys is not None and self.values is not None and self.k_scales is not None and self.v_scales is not None
        kf, vf = k.astype(mx.float32), v.astype(mx.float32)
        ks_s = mx.maximum(mx.max(mx.abs(kf), axis=-1, keepdims=True) * (1.0 / 127.0), 1e-8)
        vs_s = mx.maximum(mx.max(mx.abs(vf), axis=-1, keepdims=True) * (1.0 / 127.0), 1e-8)
        self.keys[:, :, prev:needed, :] = mx.clip(mx.round(kf / ks_s), -127.0, 127.0).astype(mx.int8)
        self.values[:, :, prev:needed, :] = mx.clip(mx.round(vf / vs_s), -127.0, 127.0).astype(mx.int8)
        self.k_scales[:, :, prev:needed, :] = ks_s.astype(mx.bfloat16)
        self.v_scales[:, :, prev:needed, :] = vs_s.astype(mx.bfloat16)
        self.offset = needed
        k_out = (self.keys[:, :, :needed, :].astype(k.dtype) * self.k_scales[:, :, :needed, :].astype(k.dtype))
        v_out = (self.values[:, :, :needed, :].astype(v.dtype) * self.v_scales[:, :, :needed, :].astype(v.dtype))
        return k_out, v_out

    def rewind_to(self, new_offset: int) -> None:
        if new_offset < 0 or new_offset > self.offset:
            raise ValueError(f"Invalid rewind offset {new_offset} for current offset {self.offset}")
        self.offset = int(new_offset)


@dataclass(frozen=True)
class HybridPrefixSnapshot:
    """Compressed LCP prefix snapshot (`1/4` KV slices + `3/4` `BF16` conv + block-32 `INT8` GDN state)."""

    prefix_tokens: tuple[int, ...]
    layer_states: tuple[Any, ...]


def snapshot_hybrid_cache(
    prefix_tokens: Sequence[int],
    cache: Sequence[DeltaSlotCache | GrowingKVCache],
    *,
    compress_gdn_int8: bool = True,
) -> HybridPrefixSnapshot:
    """Snapshot a `B=1` hybrid cache with `4x` block-32 `INT8` GDN recurrent-state compression."""
    saved: list[Any] = []
    for entry in cache:
        if isinstance(entry, DeltaSlotCache):
            conv = mx.contiguous(entry.conv_state) if entry.conv_state is not None else None
            if entry.rec_state is None:
                saved.append(("gdn_none", conv))
            elif compress_gdn_int8 and int(entry.rec_state.shape[-1]) % 32 == 0:
                codes, scales = quantize_gdn_state_int8(entry.rec_state)
                mx.eval(codes, scales)
                saved.append(("gdn_i8", conv, codes, scales, entry.rec_state.dtype))
            else:
                rec = mx.contiguous(entry.rec_state)
                mx.eval(rec)
                saved.append(("gdn_raw", conv, rec))
        elif isinstance(entry, GrowingKVCache):
            off = entry.offset
            if entry.keys is not None and off > 0:
                if entry.kv_i8 and entry.k_scales is not None and entry.v_scales is not None:
                    ks = mx.contiguous(entry.keys[:, :, :off, :].astype(mx.bfloat16) * entry.k_scales[:, :, :off, :])
                    vs = mx.contiguous(entry.values[:, :, :off, :].astype(mx.bfloat16) * entry.v_scales[:, :, :off, :])
                else:
                    ks = mx.contiguous(entry.keys[:, :, :off, :])
                    vs = mx.contiguous(entry.values[:, :, :off, :])  # type: ignore[index]
                mx.eval(ks, vs)
            else:
                ks, vs = None, None
            saved.append(("kv", off, ks, vs))
        else:
            raise TypeError(f"Unexpected cache entry {type(entry)}")
    return HybridPrefixSnapshot(prefix_tokens=tuple(int(t) for t in prefix_tokens), layer_states=tuple(saved))


def restore_hybrid_cache(
    snapshot: HybridPrefixSnapshot,
    *,
    batch_size: int = 1,
    kv_step: int = 256,
    kv_i8: bool = False,
) -> list[DeltaSlotCache | GrowingKVCache]:
    """Materialize an isolated `B=batch_size` hybrid cache from a `HybridPrefixSnapshot`."""
    b = max(1, int(batch_size))
    restored: list[DeltaSlotCache | GrowingKVCache] = []
    for item in snapshot.layer_states:
        kind = item[0]
        if kind == "gdn_none":
            conv = item[1]
            c_b = mx.repeat(conv, b, axis=0) if (conv is not None and b > 1) else conv
            restored.append(DeltaSlotCache(conv_state=c_b, rec_state=None))
        elif kind == "gdn_i8":
            _, conv, codes, scales, orig_dtype = item
            rec1 = dequantize_gdn_state_int8(codes, scales, dtype=orig_dtype)
            c_b = mx.repeat(conv, b, axis=0) if (conv is not None and b > 1) else conv
            r_b = mx.repeat(rec1, b, axis=0) if b > 1 else rec1
            restored.append(DeltaSlotCache(conv_state=c_b, rec_state=r_b))
        elif kind == "gdn_raw":
            _, conv, rec1 = item
            c_b = mx.repeat(conv, b, axis=0) if (conv is not None and b > 1) else conv
            r_b = mx.repeat(rec1, b, axis=0) if b > 1 else rec1
            restored.append(DeltaSlotCache(conv_state=c_b, rec_state=r_b))
        elif kind == "kv":
            _, off, ks, vs = item
            kv = GrowingKVCache(step=kv_step, kv_i8=kv_i8)
            if ks is not None and vs is not None and off > 0:
                kb = mx.repeat(ks, b, axis=0) if b > 1 else ks
                vb = mx.repeat(vs, b, axis=0) if b > 1 else vs
                kv.update_and_fetch(kb, vb)
            restored.append(kv)
        else:
            raise ValueError(f"Unknown snapshot kind {kind!r}")
    return restored


def build_linear_from_weights(
    weights: Mapping[str, mx.array],
    stem: str,
    in_dims: int,
    out_dims: int,
    *,
    has_bias: bool = False,
    quantize_bf16_bits: int | None = None,
    quantize_group_size: int = 64,
) -> nn.Module:
    """Build `QuantizedLinear` (when `.scales` present or runtime quant requested) or `Linear`."""
    sc = weights.get(stem + ".scales")
    if sc is not None:
        wq = weights[stem + ".weight"]
        gs = in_dims // int(sc.shape[-1])
        bits = (int(wq.shape[-1]) * 32) // in_dims
        has_qb = (stem + ".biases") in weights
        has_lb = has_bias and ((stem + ".bias") in weights)
        lin = nn.QuantizedLinear(in_dims, out_dims, bias=has_lb, group_size=gs, bits=bits)
        lin.weight = wq
        lin.scales = sc.astype(mx.bfloat16)
        if has_qb:
            lin.biases = weights[stem + ".biases"].astype(mx.bfloat16)
        if has_lb:
            lin.bias = weights[stem + ".bias"].astype(mx.bfloat16)
        return lin

    w_raw = weights[stem + ".weight"].astype(mx.bfloat16)
    has_lb = has_bias and ((stem + ".bias") in weights)
    if quantize_bf16_bits is not None and in_dims % quantize_group_size == 0:
        wq, sc_new, bi_new = mx.quantize(w_raw, group_size=quantize_group_size, bits=quantize_bf16_bits)
        lin_q = nn.QuantizedLinear(
            in_dims, out_dims, bias=has_lb, group_size=quantize_group_size, bits=quantize_bf16_bits
        )
        lin_q.weight = wq
        lin_q.scales = sc_new
        lin_q.biases = bi_new
        if has_lb:
            lin_q.bias = weights[stem + ".bias"].astype(mx.bfloat16)
        return lin_q

    lin_d = nn.Linear(in_dims, out_dims, bias=has_lb)
    lin_d.weight = w_raw
    if has_lb:
        lin_d.bias = weights[stem + ".bias"].astype(mx.bfloat16)
    return lin_d


class SwiGLUMLP(nn.Module):
    """Spec-bucketed `SwiGLU` MLP (`1` GEMM when `gate/up` match bit-width, `2` GEMMs on mixed `OptiQ`)."""

    def __init__(
        self,
        in_dims: int,
        intermediate_dims: int,
        weights: Mapping[str, mx.array],
        prefix: str,
        *,
        enable_fusion: bool = True,
        quantize_bf16_bits: int | None = None,
        quantize_group_size: int = 64,
    ) -> None:
        super().__init__()
        l_gate = build_linear_from_weights(
            weights,
            f"{prefix}.gate_proj",
            in_dims,
            intermediate_dims,
            quantize_bf16_bits=quantize_bf16_bits,
            quantize_group_size=quantize_group_size,
        )
        l_up = build_linear_from_weights(
            weights,
            f"{prefix}.up_proj",
            in_dims,
            intermediate_dims,
            quantize_bf16_bits=quantize_bf16_bits,
            quantize_group_size=quantize_group_size,
        )
        self.gate_up = SpecBucketedLinearGroup(
            [("gate", l_gate), ("up", l_up)],
            enable_fusion=enable_fusion,
        )
        self.down_proj = build_linear_from_weights(
            weights,
            f"{prefix}.down_proj",
            intermediate_dims,
            in_dims,
            quantize_bf16_bits=quantize_bf16_bits,
            quantize_group_size=quantize_group_size,
        )
        self._down_lora: tuple[mx.array, mx.array, float] | None = None

    def clear_lora(self) -> None:
        self.gate_up.clear_lora()
        self._down_lora = None

    def set_lora(
        self,
        *,
        gate: tuple[mx.array, mx.array, float] | None = None,
        up: tuple[mx.array, mx.array, float] | None = None,
        down: tuple[mx.array, mx.array, float] | None = None,
    ) -> None:
        slices: list[tuple[str, mx.array, mx.array, float]] = []
        if gate is not None:
            slices.append(("gate", gate[0], gate[1], gate[2]))
        if up is not None:
            slices.append(("up", up[0], up[1], up[2]))
        self.gate_up.set_lora_slices(slices)
        if down is not None:
            self._down_lora = (
                mx.contiguous(down[0].astype(mx.bfloat16).T),
                mx.contiguous(down[1].astype(mx.bfloat16).T),
                float(down[2]),
            )
        else:
            self._down_lora = None

    def __call__(self, x: mx.array) -> mx.array:
        gu = self.gate_up(x)
        mid = fused_swiglu(gu["gate"], gu["up"])
        out = self.down_proj(mid)
        if self._down_lora is not None:
            la_t, lb_t, sc = self._down_lora
            delta = (mx.matmul(mx.matmul(mid.astype(mx.bfloat16), la_t), lb_t) * sc).astype(out.dtype)
            out = out + delta
        return out


class SparseMoeBlock(nn.Module):
    """Qwen3.5 / Qwen3.6 Sparse MoE (`FP32` top-k softmax over unquantized `gate` + `SwitchGLU` + `shared_expert`)."""

    def __init__(
        self,
        cfg: Qwen3Config,
        weights: Mapping[str, mx.array],
        prefix: str,
        *,
        enable_fusion: bool = True,
        quantize_bf16_bits: int | None = None,
        quantize_group_size: int = 64,
    ) -> None:
        super().__init__()
        self.num_experts = int(cfg.num_experts)
        self.top_k = int(cfg.num_experts_per_tok)
        self.hidden_size = int(cfg.hidden_size)
        self.moe_inter = int(cfg.moe_intermediate_size)

        # Router gates ALWAYS remain unquantized BF16 (never Q4) per CoreAI & Splash router invariants
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.gate.weight = weights[f"{prefix}.gate.weight"].astype(mx.bfloat16)

        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        self.shared_expert_gate.weight = weights[f"{prefix}.shared_expert_gate.weight"].astype(mx.bfloat16)

        self.shared_expert = SwiGLUMLP(
            self.hidden_size,
            int(cfg.shared_expert_intermediate_size),
            weights,
            f"{prefix}.shared_expert",
            enable_fusion=enable_fusion,
            quantize_bf16_bits=quantize_bf16_bits,
            quantize_group_size=quantize_group_size,
        )

        # Support both `mlx-lm` `switch_mlp.{gate,up,down}_proj` and HF stacked `experts.{gate_up_proj,down_proj}`
        sw_pfx = f"{prefix}.switch_mlp"
        if f"{sw_pfx}.gate_proj.scales" in weights:
            wq_g = weights[f"{sw_pfx}.gate_proj.weight"]
            sc_g = weights[f"{sw_pfx}.gate_proj.scales"].astype(mx.bfloat16)
            self.exp_bits = (int(wq_g.shape[-1]) * 32) // self.hidden_size
            self.exp_gs = self.hidden_size // int(sc_g.shape[-1])
            self.exp_quant = True
            self.gate_w = wq_g
            self.gate_s = sc_g
            self.gate_b = weights.get(f"{sw_pfx}.gate_proj.biases")
            self.up_w = weights[f"{sw_pfx}.up_proj.weight"]
            self.up_s = weights[f"{sw_pfx}.up_proj.scales"].astype(mx.bfloat16)
            self.up_b = weights.get(f"{sw_pfx}.up_proj.biases")
            self.down_w = weights[f"{sw_pfx}.down_proj.weight"]
            self.down_s = weights[f"{sw_pfx}.down_proj.scales"].astype(mx.bfloat16)
            self.down_b = weights.get(f"{sw_pfx}.down_proj.biases")
        elif f"{sw_pfx}.gate_proj.weight" in weights:
            self.exp_quant = False
            self.gate_w = weights[f"{sw_pfx}.gate_proj.weight"].astype(mx.bfloat16)
            self.up_w = weights[f"{sw_pfx}.up_proj.weight"].astype(mx.bfloat16)
            self.down_w = weights[f"{sw_pfx}.down_proj.weight"].astype(mx.bfloat16)
        else:
            gu = weights[f"{prefix}.experts.gate_up_proj"].astype(mx.bfloat16)
            gw, uw = mx.split(gu, 2, axis=1)
            dw = weights[f"{prefix}.experts.down_proj"].astype(mx.bfloat16)
            if quantize_bf16_bits is not None and self.hidden_size % quantize_group_size == 0 and self.moe_inter % quantize_group_size == 0:
                self.exp_quant = True
                self.exp_bits = int(quantize_bf16_bits)
                self.exp_gs = int(quantize_group_size)
                self.gate_w, self.gate_s, self.gate_b = mx.quantize(gw, group_size=self.exp_gs, bits=self.exp_bits)
                self.up_w, self.up_s, self.up_b = mx.quantize(uw, group_size=self.exp_gs, bits=self.exp_bits)
                self.down_w, self.down_s, self.down_b = mx.quantize(dw, group_size=self.exp_gs, bits=self.exp_bits)
            else:
                self.exp_quant = False
                self.gate_w, self.up_w, self.down_w = gw, uw, dw

    def _run_experts(self, x_flat: mx.array, indices: mx.array) -> mx.array:
        # Sort expert assignments so `gather_qmm` / `gather_mm` coalesces expert weight tiles (`sorted_indices=True`).
        n_tok = int(x_flat.shape[0])
        fl = indices.reshape(-1)
        ord_i = mx.argsort(fl)
        inv_i = mx.argsort(ord_i)
        sorted_exp = fl[ord_i].astype(mx.uint32)
        lhs_idx = (ord_i // self.top_k).astype(mx.uint32)
        x_3d = x_flat.reshape(n_tok, 1, self.hidden_size)

        if self.exp_quant:
            g = mx.gather_qmm(
                x_3d,
                self.gate_w,
                self.gate_s,
                self.gate_b,
                lhs_indices=lhs_idx,
                rhs_indices=sorted_exp,
                transpose=True,
                group_size=self.exp_gs,
                bits=self.exp_bits,
                sorted_indices=True,
            )
            u = mx.gather_qmm(
                x_3d,
                self.up_w,
                self.up_s,
                self.up_b,
                lhs_indices=lhs_idx,
                rhs_indices=sorted_exp,
                transpose=True,
                group_size=self.exp_gs,
                bits=self.exp_bits,
                sorted_indices=True,
            )
            act = fused_swiglu(g, u)
            d = mx.gather_qmm(  # guard: allow-lhs-indices
                act,
                self.down_w,
                self.down_s,
                self.down_b,
                rhs_indices=sorted_exp,
                transpose=True,
                group_size=self.exp_gs,
                bits=self.exp_bits,
                sorted_indices=True,
            )
            return d[inv_i].reshape(n_tok, self.top_k, self.hidden_size)

        g_d = mx.gather_mm(
            x_3d,
            mx.swapaxes(self.gate_w, -1, -2),
            lhs_indices=lhs_idx,
            rhs_indices=sorted_exp,
            sorted_indices=True,
        )
        u_d = mx.gather_mm(
            x_3d,
            mx.swapaxes(self.up_w, -1, -2),
            lhs_indices=lhs_idx,
            rhs_indices=sorted_exp,
            sorted_indices=True,
        )
        act_d = fused_swiglu(g_d, u_d)
        d_d = mx.gather_mm(
            act_d,
            mx.swapaxes(self.down_w, -1, -2),
            rhs_indices=sorted_exp,
            sorted_indices=True,
        )
        return d_d[inv_i].reshape(n_tok, self.top_k, self.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        n_tok = int(math.prod(orig_shape[:-1]))
        x_2d = x.reshape(n_tok, self.hidden_size)

        router_logits = self.gate(x_2d).astype(mx.float32)
        neg_idx = mx.argpartition(-router_logits, kth=self.top_k - 1, axis=-1)[:, : self.top_k]
        top_logits = mx.take_along_axis(router_logits, neg_idx, axis=-1)
        scores_f32 = mx.softmax(top_logits, axis=-1, precise=True)

        expert_y = self._run_experts(x_2d, neg_idx.astype(mx.uint32))
        routed = mx.sum(expert_y.astype(mx.float32) * scores_f32[:, :, None], axis=1).astype(x.dtype)

        sh_gate = mx.sigmoid(self.shared_expert_gate(x_2d).astype(mx.float32)).astype(x.dtype)
        shared = sh_gate * self.shared_expert(x_2d)
        return (routed + shared).reshape(orig_shape)


class GatedDeltaLayer(nn.Module):
    """`GatedDeltaNet` (`3/4` of layers) with general spec-bucketing, left-pad invariance, and rollback trace."""

    def __init__(
        self,
        layer_idx: int,
        cfg: Qwen3Config,
        weights: Mapping[str, mx.array],
        *,
        prefix: str | None = None,
        enable_fusion: bool = True,
        quantize_bf16_bits: int | None = None,
        quantize_group_size: int = 64,
    ) -> None:
        super().__init__()
        px = prefix or f"model.layers.{layer_idx}.linear_attn"
        self.num_k_heads = int(cfg.linear_num_key_heads)
        self.num_v_heads = int(cfg.linear_num_value_heads)
        self.head_k_dim = int(cfg.linear_key_head_dim)
        self.head_v_dim = int(cfg.linear_value_head_dim)
        self.conv_k = int(cfg.linear_conv_kernel_dim)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self._inv_scale = self.head_k_dim**-0.5
        self._rms_eps = 1e-6 * (self._inv_scale * self._inv_scale)

        hidden = int(cfg.hidden_size)
        l_qkv = build_linear_from_weights(
            weights, f"{px}.in_proj_qkv", hidden, self.conv_dim,
            quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
        )
        l_z = build_linear_from_weights(
            weights, f"{px}.in_proj_z", hidden, self.value_dim,
            quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
        )
        l_b = build_linear_from_weights(
            weights, f"{px}.in_proj_b", hidden, self.num_v_heads,
            quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
        )
        l_a = build_linear_from_weights(
            weights, f"{px}.in_proj_a", hidden, self.num_v_heads,
            quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
        )
        self.in_group = SpecBucketedLinearGroup(
            [("qkv", l_qkv), ("z", l_z), ("b", l_b), ("a", l_a)],
            enable_fusion=enable_fusion,
        )
        self.out_proj = build_linear_from_weights(
            weights, f"{px}.out_proj", self.value_dim, hidden,
            quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
        )
        self._out_lora: tuple[mx.array, mx.array, float] | None = None

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_k,
            groups=self.conv_dim,
            padding=0,
        )
        cw = weights[f"{px}.conv1d.weight"].astype(mx.bfloat16)
        # MLX Conv1d expects (out_channels, kernel_size, in_channels // groups) = (conv_dim, conv_k, 1)
        if cw.ndim == 3 and int(cw.shape[1]) == 1 and int(cw.shape[2]) == self.conv_k:
            cw = mx.swapaxes(cw, 1, 2)
        self.conv1d.weight = mx.contiguous(cw)
        self.dt_bias = weights[f"{px}.dt_bias"].astype(mx.float32)
        self.A_log = weights[f"{px}.A_log"].astype(mx.float32)
        self.norm = RMSNormGated(weights[f"{px}.norm.weight"], eps=cfg.rms_norm_eps)

    def clear_lora(self) -> None:
        self.in_group.clear_lora()
        self._out_lora = None

    def set_lora(
        self,
        *,
        qkv: tuple[mx.array, mx.array, float] | None = None,
        z: tuple[mx.array, mx.array, float] | None = None,
        b: tuple[mx.array, mx.array, float] | None = None,
        a: tuple[mx.array, mx.array, float] | None = None,
        out: tuple[mx.array, mx.array, float] | None = None,
    ) -> None:
        slices: list[tuple[str, mx.array, mx.array, float]] = []
        for nm, entry in (("qkv", qkv), ("z", z), ("b", b), ("a", a)):
            if entry is not None:
                slices.append((nm, entry[0], entry[1], entry[2]))
        self.in_group.set_lora_slices(slices)
        if out is not None:
            self._out_lora = (
                mx.contiguous(out[0].astype(mx.bfloat16).T),
                mx.contiguous(out[1].astype(mx.bfloat16).T),
                float(out[2]),
            )
        else:
            self._out_lora = None

    def __call__(
        self,
        x: mx.array,
        cache: DeltaSlotCache | None = None,
        *,
        valid_token_scale: mx.array | None = None,
        valid_lengths: mx.array | None = None,
    ) -> mx.array:
        b_sz, s_len, _ = x.shape
        ins = self.in_group(x)
        qkv = ins["qkv"]
        z = ins["z"].reshape(b_sz, s_len, self.num_v_heads, self.head_v_dim)
        b_gate = ins["b"]
        a_gate = ins["a"]

        if valid_token_scale is not None:
            qkv = qkv * valid_token_scale.astype(qkv.dtype)

        prev_rec = None if (cache is None or cache.rec_state is None) else cache.rec_state
        if cache is None or cache.conv_state is None:
            conv_in = mx.pad(qkv, [(0, 0), (self.conv_k - 1, 0), (0, 0)])
        else:
            conv_in = mx.concatenate([cache.conv_state, qkv], axis=1)

        if cache is not None:
            if valid_lengths is not None:
                # Gather exact last (conv_k - 1) valid tokens per slot (works for len=0, 1..s_len without branching!)
                tap_offsets = mx.arange(self.conv_k - 1, dtype=mx.int32)[None, :, None]
                gather_idx = valid_lengths.astype(mx.int32)[:, None, None] + tap_offsets
                cache.conv_state = mx.contiguous(mx.take_along_axis(conv_in, gather_idx, axis=1))
            else:
                cache.conv_state = mx.contiguous(conv_in[:, -(self.conv_k - 1) :, :])

        conv_out = nn.silu(self.conv1d(conv_in))
        if valid_token_scale is not None:
            conv_out = conv_out * valid_token_scale.astype(conv_out.dtype)

        q_raw, k_raw, v_raw = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q_raw.reshape(b_sz, s_len, self.num_k_heads, self.head_k_dim)
        k = k_raw.reshape(b_sz, s_len, self.num_k_heads, self.head_k_dim)
        v = v_raw.reshape(b_sz, s_len, self.num_v_heads, self.head_v_dim)

        q, k = normalize_gdn_qk(q, k, self._inv_scale, self._rms_eps)
        g, beta = compute_gdn_gates(self.A_log, a_gate, self.dt_bias, b_gate)

        if valid_token_scale is not None:
            vmask = valid_token_scale
            g = mx.where(vmask > 0, g, mx.ones_like(g))
            beta = mx.where(vmask > 0, beta, mx.zeros_like(beta))

        if cache is not None and cache.record_verify_trace:
            cache._verify_trace = (conv_in, q, k, v, g, beta, prev_rec)

        y, new_rec = mx.fast.gated_delta_update(q, k, v, g, beta, prev_rec)
        if cache is not None:
            cache.rec_state = new_rec

        normed = self.norm(y, z).reshape(b_sz, s_len, self.value_dim)
        res = self.out_proj(normed)
        if self._out_lora is not None:
            la_t, lb_t, sc = self._out_lora
            delta = (mx.matmul(mx.matmul(normed.astype(mx.bfloat16), la_t), lb_t) * sc).astype(res.dtype)
            res = res + delta
        return res


class FullAttentionLayer(nn.Module):
    """Gated `GQA` (`1/4` of layers) with `0.0`-ULP `[Q|Gate]` split, spec-bucketing, and autograd VJP guards."""

    def __init__(
        self,
        layer_idx: int,
        cfg: Qwen3Config,
        weights: Mapping[str, mx.array],
        *,
        norm_add_one: bool,
        use_apple10_force_fused: bool = False,
        prefix: str | None = None,
        enable_fusion: bool = True,
        quantize_bf16_bits: int | None = None,
        quantize_group_size: int = 64,
    ) -> None:
        super().__init__()
        px = prefix or f"model.layers.{layer_idx}.self_attn"
        self.num_q_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.q_dim = self.num_q_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.rot_dims = int(cfg.rot_dims)
        self._use_apple10_force_fused = bool(use_apple10_force_fused)

        hidden = int(cfg.hidden_size)
        has_bias = bool(cfg.attention_bias)
        l_q_interleaved = build_linear_from_weights(
            weights, f"{px}.q_proj", hidden, 2 * self.q_dim, has_bias=has_bias,
            quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
        )
        l_q_only, l_gate_only = split_linear_q_proj_to_q_gate(
            l_q_interleaved, self.num_q_heads, self.head_dim
        )
        l_k = build_linear_from_weights(
            weights, f"{px}.k_proj", hidden, self.kv_dim, has_bias=has_bias,
            quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
        )
        l_v = build_linear_from_weights(
            weights, f"{px}.v_proj", hidden, self.kv_dim, has_bias=has_bias,
            quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
        )
        self.qkvg_group = SpecBucketedLinearGroup(
            [("q", l_q_only), ("k", l_k), ("v", l_v), ("gate", l_gate_only)],
            enable_fusion=enable_fusion,
        )
        self.o_proj = build_linear_from_weights(
            weights, f"{px}.o_proj", self.q_dim, hidden, has_bias=has_bias,
            quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
        )
        self._o_lora: tuple[mx.array, mx.array, float] | None = None

        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        qw = weights[f"{px}.q_norm.weight"].astype(mx.float32)
        self.q_norm.weight = (qw + 1.0 if norm_add_one else qw).astype(mx.bfloat16)

        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        kw = weights[f"{px}.k_norm.weight"].astype(mx.float32)
        self.k_norm.weight = (kw + 1.0 if norm_add_one else kw).astype(mx.bfloat16)

        self.rope = nn.RoPE(self.rot_dims, traditional=False, base=cfg.rope_theta)

    def clear_lora(self) -> None:
        self.qkvg_group.clear_lora()
        self._o_lora = None

    def set_lora(
        self,
        *,
        q: tuple[mx.array, mx.array, float] | None = None,
        k: tuple[mx.array, mx.array, float] | None = None,
        v: tuple[mx.array, mx.array, float] | None = None,
        o: tuple[mx.array, mx.array, float] | None = None,
    ) -> None:
        slices: list[tuple[str, mx.array, mx.array, float]] = []
        if q is not None:
            la, lb, sc = q
            bq, bg = permute_q_proj_rows_to_q_gate(lb, self.num_q_heads, self.head_dim)
            slices.append(("q", la, bq, sc))
            slices.append(("gate", la, bg, sc))
        if k is not None:
            slices.append(("k", k[0], k[1], k[2]))
        if v is not None:
            slices.append(("v", v[0], v[1], v[2]))
        self.qkvg_group.set_lora_slices(slices)
        if o is not None:
            self._o_lora = (
                mx.contiguous(o[0].astype(mx.bfloat16).T),
                mx.contiguous(o[1].astype(mx.bfloat16).T),
                float(o[2]),
            )
        else:
            self._o_lora = None

    def __call__(
        self,
        x: mx.array,
        cache: GrowingKVCache | None = None,
        *,
        attn_mask: str | mx.array = "causal",
        rope_trig: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        b_sz, s_len, _ = x.shape
        parts = self.qkvg_group(x)
        q = self.q_norm(parts["q"].reshape(b_sz, s_len, self.num_q_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(parts["k"].reshape(b_sz, s_len, self.num_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = parts["v"].reshape(b_sz, s_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        gate = parts["gate"]

        if rope_trig is not None:
            cos_a, sin_a = rope_trig
            q = apply_partial_rope_trig(q, cos_a, sin_a, self.rot_dims)
            k = apply_partial_rope_trig(k, cos_a, sin_a, self.rot_dims)
        elif cache is not None:
            # Cached inference: fuse [Q, K] into 1 RoPE kernel launch
            qk = mx.concatenate([q, k], axis=1)
            qk = self.rope(qk, offset=cache.offset)
            q, k = mx.split(qk, [self.num_q_heads], axis=1)
        else:
            # Training / uncached (`cache is None`): keep Q and K RoPE separate to avoid MLX 0.32.4 B=1 split VJP bug
            q = self.rope(q)
            k = self.rope(k)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
            eff_mask: str | mx.array | None = None if (s_len == 1 and attn_mask == "causal") else attn_mask
            if self._use_apple10_force_fused and self.head_dim == 256:
                attn_out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=eff_mask, force_fused=True
                )
            else:
                attn_out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=eff_mask
                )
        else:
            # Training / uncached path: bypass Apple10 D=256 SDPA VJP NaNs when head_dim >= 256
            if self.head_dim >= 256:
                attn_out = safe_training_sdpa(q, k, v, scale=self.scale, mask=attn_mask)
            else:
                attn_out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=attn_mask
                )

        attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(b_sz, s_len, self.q_dim)
        gated = fused_sigmoid_attn_gate(attn_flat, gate)
        res = self.o_proj(gated)
        if self._o_lora is not None:
            la_t, lb_t, sc = self._o_lora
            delta = (mx.matmul(mx.matmul(gated.astype(mx.bfloat16), la_t), lb_t) * sc).astype(res.dtype)
            res = res + delta
        return res


class Qwen3DecoderBlock(nn.Module):
    """Single hybrid decoder block (`linear_attention` or `full_attention` + `SwiGLUMLP` or `SparseMoeBlock`)."""

    def __init__(
        self,
        layer_idx: int,
        cfg: Qwen3Config,
        weights: Mapping[str, mx.array],
        *,
        norm_add_one: bool,
        use_apple10_force_fused: bool = False,
        layer_prefix: str | None = None,
        force_full_attention: bool = False,
        enable_fusion: bool = True,
        quantize_bf16_bits: int | None = None,
        quantize_group_size: int = 64,
    ) -> None:
        super().__init__()
        px = layer_prefix or f"model.layers.{layer_idx}"
        self.layer_type = "full_attention" if force_full_attention else cfg.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"

        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        iw = weights[f"{px}.input_layernorm.weight"].astype(mx.float32)
        self.input_layernorm.weight = (iw + 1.0 if norm_add_one else iw).astype(mx.bfloat16)

        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        pw = weights[f"{px}.post_attention_layernorm.weight"].astype(mx.float32)
        self.post_attention_layernorm.weight = (pw + 1.0 if norm_add_one else pw).astype(mx.bfloat16)

        if self.is_linear:
            self.linear_attn = GatedDeltaLayer(
                layer_idx,
                cfg,
                weights,
                prefix=f"{px}.linear_attn",
                enable_fusion=enable_fusion,
                quantize_bf16_bits=quantize_bf16_bits,
                quantize_group_size=quantize_group_size,
            )
        else:
            self.self_attn = FullAttentionLayer(
                layer_idx,
                cfg,
                weights,
                norm_add_one=norm_add_one,
                use_apple10_force_fused=use_apple10_force_fused,
                prefix=f"{px}.self_attn",
                enable_fusion=enable_fusion,
                quantize_bf16_bits=quantize_bf16_bits,
                quantize_group_size=quantize_group_size,
            )

        if cfg.is_moe:
            self.mlp: nn.Module = SparseMoeBlock(
                cfg,
                weights,
                f"{px}.mlp",
                enable_fusion=enable_fusion,
                quantize_bf16_bits=quantize_bf16_bits,
                quantize_group_size=quantize_group_size,
            )
        else:
            self.mlp = SwiGLUMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                weights,
                f"{px}.mlp",
                enable_fusion=enable_fusion,
                quantize_bf16_bits=quantize_bf16_bits,
                quantize_group_size=quantize_group_size,
            )

    def __call__(
        self,
        x: mx.array,
        cache: DeltaSlotCache | GrowingKVCache | None = None,
        *,
        valid_token_scale: mx.array | None = None,
        valid_lengths: mx.array | None = None,
        attn_mask: str | mx.array = "causal",
        rope_trig: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        if self.is_linear:
            h = x + self.linear_attn(
                self.input_layernorm(x),
                cache,  # type: ignore[arg-type]
                valid_token_scale=valid_token_scale,
                valid_lengths=valid_lengths,
            )
        else:
            h = x + self.self_attn(
                self.input_layernorm(x),
                cache,  # type: ignore[arg-type]
                attn_mask=attn_mask,
                rope_trig=rope_trig,
            )
        return h + self.mlp(self.post_attention_layernorm(h))


class FastQwen3Model(nn.Module):
    """Self-contained MLX `Qwen3.5` / `Qwen3.6` causal LM with header-filtered shard loading & `lora_accel`."""

    def __init__(
        self,
        cfg: Qwen3Config,
        weights: Mapping[str, mx.array],
        *,
        norm_add_one: bool = False,
        vocab_remap: QwenVocabRemapper | None = None,
        enable_fusion: bool = True,
        quantize_bf16_bits: int | None = None,
        quantize_group_size: int = 64,
        model_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_dir = model_dir
        self.norm_add_one = bool(norm_add_one)
        self.vocab_remap = vocab_remap
        self.use_apple10_force_fused = detect_apple10_d256_force_fused()

        hidden = int(cfg.hidden_size)
        vocab_sz = int(cfg.vocab_size)
        emb_sc = weights.get("model.embed_tokens.scales")
        if emb_sc is not None:
            wq = weights["model.embed_tokens.weight"]
            gs = hidden // int(emb_sc.shape[-1])
            bits = (int(wq.shape[-1]) * 32) // hidden
            self.embed_tokens: Any = nn.QuantizedEmbedding(vocab_sz, hidden, group_size=gs, bits=bits)
            self.embed_tokens.weight = wq
            self.embed_tokens.scales = emb_sc.astype(mx.bfloat16)
            if "model.embed_tokens.biases" in weights:
                self.embed_tokens.biases = weights["model.embed_tokens.biases"].astype(mx.bfloat16)
        elif quantize_bf16_bits is not None and hidden % quantize_group_size == 0:
            eb_bits = max(8, int(quantize_bf16_bits))
            wq, sc_e, bi_e = mx.quantize(
                weights["model.embed_tokens.weight"].astype(mx.bfloat16),
                group_size=quantize_group_size,
                bits=eb_bits,
            )
            self.embed_tokens = nn.QuantizedEmbedding(vocab_sz, hidden, group_size=quantize_group_size, bits=eb_bits)
            self.embed_tokens.weight = wq
            self.embed_tokens.scales = sc_e
            self.embed_tokens.biases = bi_e
        else:
            self.embed_tokens = nn.Embedding(vocab_sz, hidden)
            self.embed_tokens.weight = weights["model.embed_tokens.weight"].astype(mx.bfloat16)

        self.layers = [
            Qwen3DecoderBlock(
                i,
                cfg,
                weights,
                norm_add_one=norm_add_one,
                use_apple10_force_fused=self.use_apple10_force_fused,
                enable_fusion=enable_fusion,
                quantize_bf16_bits=quantize_bf16_bits,
                quantize_group_size=quantize_group_size,
            )
            for i in range(cfg.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(hidden, eps=cfg.rms_norm_eps)
        nw = weights["model.norm.weight"].astype(mx.float32)
        self.norm.weight = (nw + 1.0 if norm_add_one else nw).astype(mx.bfloat16)

        self._cold_lm_head: nn.Module | None = None
        if not cfg.tie_word_embeddings and "lm_head.weight" in weights:
            full_head = build_linear_from_weights(
                weights, "lm_head", hidden, vocab_sz,
                quantize_bf16_bits=quantize_bf16_bits, quantize_group_size=quantize_group_size,
            )
            if vocab_remap is not None and vocab_remap.vocab_size < vocab_sz:
                self._cold_lm_head = full_head
                idx_arr = mx.array(vocab_remap.new_to_old, dtype=mx.int32)
                if isinstance(full_head, nn.QuantizedLinear):
                    sub = nn.QuantizedLinear(
                        hidden, vocab_remap.vocab_size, bias=False,
                        group_size=full_head.group_size, bits=full_head.bits,
                    )
                    sub.weight = mx.take(full_head.weight, idx_arr, axis=0)
                    sub.scales = mx.take(full_head.scales, idx_arr, axis=0)
                    if getattr(full_head, "biases", None) is not None:
                        sub.biases = mx.take(full_head.biases, idx_arr, axis=0)
                    self.lm_head: Any = sub
                else:
                    sub_d = nn.Linear(hidden, vocab_remap.vocab_size, bias=False)
                    sub_d.weight = mx.take(full_head.weight, idx_arr, axis=0)
                    self.lm_head = sub_d
            else:
                self.lm_head = full_head
        else:
            self.lm_head = None

        rot = cfg.rot_dims
        self._inv_freq = mx.array(
            [1.0 / (cfg.rope_theta ** (float(i) / float(rot))) for i in range(0, rot, 2)],
            dtype=mx.float32,
        )

    @property
    def vocab_n(self) -> int:
        if self.lm_head is not None:
            return int(self.lm_head.weight.shape[0])
        return int(self.cfg.vocab_size)

    def make_cache(self, *, kv_step: int = 256, kv_i8: bool = False) -> list[DeltaSlotCache | GrowingKVCache]:
        return [
            DeltaSlotCache() if layer.is_linear else GrowingKVCache(step=kv_step, kv_i8=kv_i8)
            for layer in self.layers
        ]

    def precompute_rope_trig(self, pos_2d: mx.array) -> tuple[mx.array, mx.array]:
        angles = (pos_2d.astype(mx.float32)[:, None, :, None]) * self._inv_freq[None, None, None, :]
        return mx.cos(angles).astype(mx.bfloat16), mx.sin(angles).astype(mx.bfloat16)

    def forward_hidden(
        self,
        input_ids: mx.array,
        cache: Sequence[DeltaSlotCache | GrowingKVCache] | None = None,
        *,
        valid_token_scale: mx.array | None = None,
        valid_lengths: mx.array | None = None,
        attn_mask: str | mx.array = "causal",
        rope_trig: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(input_ids)
        if cache is None:
            for layer in self.layers:
                h = layer(
                    h,
                    None,
                    valid_token_scale=valid_token_scale,
                    valid_lengths=valid_lengths,
                    attn_mask=attn_mask,
                    rope_trig=rope_trig,
                )
        else:
            for layer, c in zip(self.layers, cache):
                h = layer(
                    h,
                    c,
                    valid_token_scale=valid_token_scale,
                    valid_lengths=valid_lengths,
                    attn_mask=attn_mask,
                    rope_trig=rope_trig,
                )
        return self.norm(h)

    def project_logits(self, h: mx.array, *, full_vocab: bool = False) -> mx.array:
        if full_vocab and self._cold_lm_head is not None:
            return self._cold_lm_head(h)
        if self.lm_head is not None:
            return self.lm_head(h)
        if isinstance(self.embed_tokens, nn.QuantizedEmbedding):
            return self.embed_tokens.as_linear(h)
        return mx.matmul(h, self.embed_tokens.weight.T)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Sequence[DeltaSlotCache | GrowingKVCache] | None = None,
        *,
        valid_token_scale: mx.array | None = None,
        valid_lengths: mx.array | None = None,
        attn_mask: str | mx.array = "causal",
        rope_trig: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        h = self.forward_hidden(
            input_ids,
            cache,
            valid_token_scale=valid_token_scale,
            valid_lengths=valid_lengths,
            attn_mask=attn_mask,
            rope_trig=rope_trig,
        )
        return self.project_logits(h)

    def clear_lora_adapters(self) -> None:
        for layer in self.layers:
            if layer.is_linear:
                layer.linear_attn.clear_lora()
            else:
                layer.self_attn.clear_lora()
            if isinstance(layer.mlp, SwiGLUMLP):
                layer.mlp.clear_lora()

    def apply_lora_tensors(
        self,
        lora_map: Mapping[str, tuple[mx.array, mx.array, float]],
    ) -> int:
        """Hot-attach `R1`, `R2`, or `R3` PEFT LoRA tensors (`stem -> (lora_A[r, in], lora_B[out, r], scale)`)."""
        self.clear_lora_adapters()
        applied = 0
        for i, layer in enumerate(self.layers):
            lpx = f"model.layers.{i}"
            if layer.is_linear:
                gpx = f"{lpx}.linear_attn"
                qkv = lora_map.get(f"{gpx}.in_proj_qkv")
                z = lora_map.get(f"{gpx}.in_proj_z")
                b = lora_map.get(f"{gpx}.in_proj_b")
                a = lora_map.get(f"{gpx}.in_proj_a")
                out = lora_map.get(f"{gpx}.out_proj")
                if any(x is not None for x in (qkv, z, b, a, out)):
                    layer.linear_attn.set_lora(qkv=qkv, z=z, b=b, a=a, out=out)
                    applied += sum(1 for x in (qkv, z, b, a, out) if x is not None)
            else:
                apx = f"{lpx}.self_attn"
                q = lora_map.get(f"{apx}.q_proj")
                k = lora_map.get(f"{apx}.k_proj")
                v = lora_map.get(f"{apx}.v_proj")
                o = lora_map.get(f"{apx}.o_proj")
                if any(x is not None for x in (q, k, v, o)):
                    layer.self_attn.set_lora(q=q, k=k, v=v, o=o)
                    applied += sum(1 for x in (q, k, v, o) if x is not None)

            if isinstance(layer.mlp, SwiGLUMLP):
                mpx = f"{lpx}.mlp"
                g = lora_map.get(f"{mpx}.gate_proj")
                u = lora_map.get(f"{mpx}.up_proj")
                d = lora_map.get(f"{mpx}.down_proj")
                if any(x is not None for x in (g, u, d)):
                    layer.mlp.set_lora(gate=g, up=u, down=d)
                    applied += sum(1 for x in (g, u, d) if x is not None)
        return applied

    def load_lora_adapter(self, adapter_dir_or_file: str | Path) -> int:
        """Load a PEFT or MLX LoRA `.safetensors` directory/file onto the frozen fused model."""
        p = Path(adapter_dir_or_file).expanduser()
        scale_default = 2.0
        if p.is_dir():
            cfg_p = p / "adapter_config.json"
            if cfg_p.is_file():
                acfg = json.loads(cfg_p.read_text(encoding="utf-8"))
                r = float(acfg.get("r", 16) or 16)
                alpha = float(acfg.get("lora_alpha", r * 2.0) or (r * 2.0))
                scale_default = alpha / r
            st_files = sorted(p.glob("*.safetensors"))
            if not st_files:
                raise FileNotFoundError(f"No .safetensors found in adapter dir {p}")
            raw_tensors: dict[str, mx.array] = {}
            for sf in st_files:
                raw_tensors.update(mx.load(str(sf)))  # type: ignore[arg-type]
        else:
            raw_tensors = dict(mx.load(str(p)))  # type: ignore[arg-type]

        by_stem: dict[str, dict[str, Any]] = {}
        for rk, val in raw_tensors.items():
            if rk.endswith(".scale"):
                norm_s = normalize_qwen3_weight_key(rk[: -len(".scale")])
                if norm_s:
                    by_stem.setdefault(norm_s, {})["scale"] = float(np.asarray(val.astype(mx.float32)).item())
                continue
            parsed = normalize_lora_stem(rk)
            if parsed is None:
                continue
            stem, tag = parsed
            # MLX LoRALinear stores lora_a as (in, r) and lora_b as (r, out); PEFT stores (r, in) and (out, r)
            by_stem.setdefault(stem, {})[tag] = val

        packed: dict[str, tuple[mx.array, mx.array, float]] = {}
        for stem, d in by_stem.items():
            if "lora_A" not in d or "lora_B" not in d:
                continue
            la, lb = d["lora_A"], d["lora_B"]
            # Normalize to PEFT convention: la=(r, in), lb=(out, r)
            if int(la.shape[0]) > int(la.shape[1]):
                la = mx.contiguous(la.T)
            if int(lb.shape[0]) < int(lb.shape[1]):
                lb = mx.contiguous(lb.T)
            sc = float(d.get("scale", scale_default))
            packed[stem] = (la, lb, sc)
        return self.apply_lora_tensors(packed)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        vocab_remap: QwenVocabRemapper | None = None,
        enable_fusion: bool = True,
        quantize_bf16_bits: int | None = None,
        quantize_group_size: int = 64,
        load_mtp_tensors: bool = True,
        allow_oversubscribe: bool = False,
    ) -> tuple[FastQwen3Model, dict[str, mx.array]]:
        """Stream-load Qwen3.5/3.6 shards skipping `vision_tower`/`visual` shards via header inspection."""
        mdir = Path(model_dir).expanduser().resolve()
        raw_cfg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
        cfg = Qwen3Config.from_dict(raw_cfg)

        # Host RAM guard for 30B+ MoE / 27B dense on <= 24 GB machines
        if not allow_oversubscribe and (cfg.num_experts >= 64 or cfg.hidden_size >= 5120):
            try:
                from fast_shared.hw_check import probe_host_hardware

                hw = probe_host_hardware()
                mem_gb = float(hw.get("memory_gb") or hw.get("mem_gb") or 0.0)
                if 0.0 < mem_gb <= 24.5:
                    raise MemoryError(
                        f"Model at {mdir.name!r} exceeds safe 80% working-set ceiling on a {mem_gb:.0f} GB host. "
                        f"Use Qwen3.5-4B / 9B on 24 GB hosts or pass allow_oversubscribe=True."
                    )
            except ImportError:
                pass

        shards = sorted(glob.glob(str(mdir / "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No .safetensors shards found in {mdir}")

        norm_weights: dict[str, mx.array] = {}
        mtp_weights: dict[str, mx.array] = {}
        raw_keys_seen: list[str] = []

        for sp in shards:
            hdr_keys = inspect_safetensor_header_keys(sp)
            useful = [k for k in hdr_keys if normalize_qwen3_weight_key(k) is not None]
            if not useful:
                continue
            raw_keys_seen.extend(useful)
            loaded: dict[str, mx.array] = mx.load(sp)  # type: ignore[assignment]
            for rk, tensor in loaded.items():
                nk = normalize_qwen3_weight_key(rk)
                if nk is None:
                    continue
                if nk.startswith("mtp."):
                    if load_mtp_tensors:
                        mtp_weights[nk] = tensor
                else:
                    norm_weights[nk] = tensor
            del loaded
            mx.clear_cache()

        if load_mtp_tensors and cfg.mtp_file:
            mtp_p = mdir / cfg.mtp_file
            if mtp_p.is_file():
                extra_mtp: dict[str, mx.array] = mx.load(str(mtp_p))  # type: ignore[assignment]
                for rk, tensor in extra_mtp.items():
                    nk = normalize_qwen3_weight_key(rk)
                    if nk is not None and nk.startswith("mtp."):
                        mtp_weights[nk] = tensor
                    elif rk.startswith("layers.") or rk in ("fc.weight", "fc.scales", "fc.biases", "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
                        mtp_weights["mtp." + rk] = tensor
                del extra_mtp
                mx.clear_cache()

        sample_nw = norm_weights.get("model.layers.0.post_attention_layernorm.weight")
        if sample_nw is None:
            sample_nw = norm_weights.get("model.layers.0.input_layernorm.weight")
        if sample_nw is None:
            sample_nw = norm_weights["model.norm.weight"]
        norm_add_one = detect_rmsnorm_add_one(raw_keys_seen, sample_nw)
        model = cls(
            cfg,
            norm_weights,
            norm_add_one=norm_add_one,
            vocab_remap=vocab_remap,
            enable_fusion=enable_fusion,
            quantize_bf16_bits=quantize_bf16_bits,
            quantize_group_size=quantize_group_size,
            model_dir=mdir,
        )
        del norm_weights
        gc.collect()
        mx.clear_cache()
        return model, mtp_weights
