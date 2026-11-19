"""Metal/MLX GPU engine for pplx-embed-v2-context-0.6b.

Combines:
- Pure MLX/Metal 12-layer hybrid Qwen3.5 backbone (alternating GatedDeltaNet +
  bidirectional Full Attention, overriding config ``full_attention_interval=4`` to ``2``)
- Zero-centered ``+1.0`` weight shifts computed in ``float32`` on ``RMSNorm`` layers
  (leaving ``GatedRMSNorm`` unshifted and multiplied in ``float32``)
- Fused ``qkv_proj`` (``q+gate, k, v``) and ``gate_up_proj`` projections
- Fast ``mx.float16`` GatedDeltaNet path (``S <= 2048``) with zero per-layer ``mx.eval``
- Long-context streamed path (``S > 2048``, up to ``32K``) carrying recurrent GDN + Conv1d
  state across ``1024``-token slices and tile-aligned ``512``-token ``SDPA`` slices
- ``MappedEmbedding`` safetensors memmap keeping ``embed_tokens`` out of Metal GPU RAM
- MIT-adapted ``gated_delta_update`` kernels from ``mlx-lm`` 0.32.2 (Copyright © 2025 Apple Inc.)
"""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from fast_shared.embeddings import (
    MappedEmbedding,
    resolve_embedding_dim,
    set_mlx_cache_limit,
    tile_aligned_slices,
    truncate_and_normalize,
)
from fast_shared.hf import resolve_hf_model_dir

from .config import DEFAULT_MODEL_NAME, DEFAULT_PPLX_CONFIG, PplxConfig
from .paths import RepoPaths, resolve_paths
from .tokenizer import ContextualTokenizer

_STREAMED_SEQ_THRESHOLD = 2048
_GDN_SLICE_LEN = 1024
_SDPA_SLICE_LEN = 512
_SDPA_MIN_TAIL = _SDPA_SLICE_LEN

_ZERO_CENTERED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

_EMBED_TENSOR_CANDIDATES = (
    "language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
)

_ENABLE_GDN_PACKED = os.environ.get("MLX_GDN_PACKED", "1") != "0"


def _open_mapped_embedding(path_or_dir: Path) -> MappedEmbedding | None:
    if not path_or_dir.exists():
        return None
    for cand in _EMBED_TENSOR_CANDIDATES:
        try:
            return MappedEmbedding.from_checkpoint(path_or_dir, tensor_name=cand)
        except (KeyError, FileNotFoundError):
            continue
    return None


# ---------------------------------------------------------------------------
# MIT-adapted GatedDeltaNet kernels (from mlx-lm 0.32.2 gated_delta.py)
# ---------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    a_f32 = a.astype(mx.float32)
    dt_f32 = dt_bias.astype(mx.float32)
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a_f32 + dt_f32))


def normalize_qk(
    q: mx.array,
    k: mx.array,
    *,
    inv_scale: float,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """L2-normalize q/k in ``q.dtype`` while folding ``head_dim**-0.5`` into ``q``."""
    rms_eps = eps * (inv_scale * inv_scale)
    q_scale = mx.array(inv_scale * inv_scale, dtype=q.dtype)
    k_scale = mx.array(inv_scale, dtype=k.dtype)
    q_norm = q_scale * mx.fast.rms_norm(q, None, rms_eps)
    k_norm = k_scale * mx.fast.rms_norm(k, None, rms_eps)
    return q_norm, k_norm


def _make_gated_delta_kernel(has_mask: bool = False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * static_cast<float>(g_[hv_idx]);
              kv_mem += state[i] * static_cast<float>(k_[s_idx]);
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
              out += state[i] * static_cast<float>(q_[s_idx]);
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")
    suffix = "_mask" if has_mask else ""
    return mx.fast.metal_kernel(
        name=f"pplx_gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


def _make_gated_delta_packed_kernel():
    if not mx.metal.is_available():
        return None
    source = r"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="pplx_gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


_GATED_DELTA_KERNEL = _make_gated_delta_kernel(has_mask=False)
_GATED_DELTA_KERNEL_MASKED = _make_gated_delta_kernel(has_mask=True)
_GATED_DELTA_KERNEL_PACKED = _make_gated_delta_packed_kernel()


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    old_state = state
    q_f32 = q.astype(mx.float32)
    k_f32 = k.astype(mx.float32)
    v_f32 = v.astype(mx.float32)
    g_f32 = g.astype(mx.float32)
    beta_f32 = beta.astype(mx.float32)

    state = state * g_f32[..., None, None]
    kv_mem = (state * k_f32[..., None, :]).sum(axis=-1)
    delta = (v_f32 - kv_mem) * beta_f32[..., None]
    state = state + k_f32[..., None, :] * delta[..., None]
    y = (state * q_f32[..., None, :]).sum(axis=-1)

    if mask is not None:
        m = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(m, state, old_state)
    return y.astype(q.dtype), state


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)
    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    return mx.stack(ys, axis=1), state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    *,
    use_kernel: bool = True,
) -> tuple[mx.array, mx.array]:
    """Run GatedDeltaNet recurrence keeping ``q, k, v`` in their activation dtype."""
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)

    if (
        not use_kernel
        or _GATED_DELTA_KERNEL is None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(q, k, v, g, beta, state.astype(mx.float32), mask)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    packed_eligible = (
        mask is None
        and g.ndim == 3
        and Dk == 128
        and Dv % 8 == 0
        and _GATED_DELTA_KERNEL_PACKED is not None
        and _ENABLE_GDN_PACKED
    )
    if packed_eligible:
        kernel = _GATED_DELTA_KERNEL_PACKED
        inputs = [q, k, v, g, beta, state, T]
        grid = (32, Dv // 8, B * Hv)
        threadgroup = (32, 2, 1)
    elif mask is not None and _GATED_DELTA_KERNEL_MASKED is not None:
        kernel = _GATED_DELTA_KERNEL_MASKED
        inputs = [q, k, v, g, beta, state, T, mask]
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)
    else:
        kernel = _GATED_DELTA_KERNEL
        inputs = [q, k, v, g, beta, state, T]
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)

    out_y, out_state = kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )
    return out_y, out_state


# ---------------------------------------------------------------------------
# Hybrid Qwen3.5 text backbone + contextual head
# ---------------------------------------------------------------------------


class PplxRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class PplxGatedRMSNorm(nn.Module):
    """Unshifted RMSNormGated computed in float32 before weight multiply."""

    def __init__(self, dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dims,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        g_f32 = gate.astype(mx.float32)
        w_f32 = self.weight.astype(mx.float32)
        normed_f32 = mx.fast.rms_norm(x_f32, w_f32, self.eps)
        out_f32 = normed_f32 * nn.silu(g_f32)
        return out_f32.astype(x.dtype)


class PplxAttention(nn.Module):
    """Bidirectional grouped-query attention with output gating and partial RoPE."""

    def __init__(self, cfg: PplxConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.rope_dim
        self.rope_theta = float(cfg.rope_theta)
        self.scale = float(cfg.head_dim) ** -0.5

        self.q_proj_dim = 2 * self.num_heads * self.head_dim
        self.kv_proj_dim = self.num_kv_heads * self.head_dim
        total_qkv_dim = self.q_proj_dim + 2 * self.kv_proj_dim

        self.qkv_proj = nn.Linear(cfg.hidden_size, total_qkv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = PplxRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = PplxRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        sdpa_mask: mx.array | None = None,
    ) -> mx.array:
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q_and_gate, keys, values = mx.split(
            qkv,
            [self.q_proj_dim, self.q_proj_dim + self.kv_proj_dim],
            axis=-1,
        )
        q_and_gate = q_and_gate.reshape(B, S, self.num_heads, 2 * self.head_dim)
        queries, gate = mx.split(q_and_gate, 2, axis=-1)
        gate = gate.reshape(B, S, self.num_heads * self.head_dim)

        queries = self.q_norm(queries).swapaxes(1, 2)
        keys = self.k_norm(keys.reshape(B, S, self.num_kv_heads, self.head_dim)).swapaxes(1, 2)
        values = values.reshape(B, S, self.num_kv_heads, self.head_dim).swapaxes(1, 2)

        queries = mx.fast.rope(
            queries,
            self.rope_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=0,
        )
        keys = mx.fast.rope(
            keys,
            self.rope_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=0,
        )

        if S <= _STREAMED_SEQ_THRESHOLD:
            attn_out = mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=self.scale,
                mask=sdpa_mask,
            )
        else:
            slices = tile_aligned_slices(S, slice_len=_SDPA_SLICE_LEN, min_tail=_SDPA_MIN_TAIL)
            parts: list[mx.array] = []
            for s0, s1 in slices:
                part = mx.fast.scaled_dot_product_attention(
                    queries[:, :, s0:s1, :],
                    keys,
                    values,
                    scale=self.scale,
                    mask=sdpa_mask,
                )
                mx.eval(part)  # guard: allow-eval
                parts.append(part)
            attn_out = mx.concatenate(parts, axis=2)

        attn_flat = attn_out.swapaxes(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        gated = attn_flat * mx.sigmoid(gate)
        return self.o_proj(gated)


class PplxGatedDeltaNet(nn.Module):
    """Linear-attention GatedDeltaNet layer with causal Conv1d and state-carried streaming."""

    def __init__(self, cfg: PplxConfig) -> None:
        super().__init__()
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.inv_scale = float(self.head_k_dim) ** -0.5

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones((self.num_v_heads,), dtype=mx.float32)
        self.A_log = mx.zeros((self.num_v_heads,), dtype=mx.float32)
        self.norm = PplxGatedRMSNorm(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, cfg.hidden_size, bias=False)
        self.use_kernel = True

    def _run_step(
        self,
        x_slice: mx.array,
        a_slice: mx.array,
        b_slice: mx.array,
        conv_state: mx.array,
        rec_state: mx.array,
        mask_slice: mx.array | None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        B, S_sub, _ = x_slice.shape

        mixed_qkv = self.in_proj_qkv(x_slice)
        z = self.in_proj_z(x_slice).reshape(B, S_sub, self.num_v_heads, self.head_v_dim)

        conv_in = mx.concatenate([conv_state, mixed_qkv], axis=1)
        new_conv_state = conv_in[:, -(self.conv_kernel_size - 1) :, :]
        conv_out = nn.silu(self.conv1d(conv_in))

        q_flat, k_flat, v_flat = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q_flat.reshape(B, S_sub, self.num_k_heads, self.head_k_dim)
        k = k_flat.reshape(B, S_sub, self.num_k_heads, self.head_k_dim)
        v = v_flat.reshape(B, S_sub, self.num_v_heads, self.head_v_dim)

        q, k = normalize_qk(q, k, inv_scale=self.inv_scale, eps=1e-6)
        out, new_rec_state = gated_delta_update(
            q,
            k,
            v,
            a_slice,
            b_slice,
            self.A_log,
            self.dt_bias,
            state=rec_state,
            mask=mask_slice,
            use_kernel=self.use_kernel,
        )
        normed = self.norm(out, z).reshape(B, S_sub, self.value_dim)
        return self.out_proj(normed), new_conv_state, new_rec_state

    def __call__(
        self,
        x: mx.array,
        ssm_mask: mx.array | None = None,
    ) -> mx.array:
        B, S, _ = x.shape
        if ssm_mask is not None:
            zero_val = mx.array(0.0, dtype=x.dtype)
            x = mx.where(ssm_mask[..., None], x, zero_val)

        # Narrow projections (N=16) depend on M-tile rounding, so evaluate once whole before slicing.
        a_all = self.in_proj_a(x)
        b_all = self.in_proj_b(x)
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        rec_state = mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=x.dtype)
        if S <= _STREAMED_SEQ_THRESHOLD:
            out, _, _ = self._run_step(x, a_all, b_all, conv_state, rec_state, ssm_mask)
            return out

        mx.eval(a_all, b_all)  # guard: allow-eval
        outs: list[mx.array] = []
        for s0, s1 in tile_aligned_slices(S, slice_len=_GDN_SLICE_LEN, min_tail=_GDN_SLICE_LEN):
            mask_slice = None if ssm_mask is None else ssm_mask[:, s0:s1]
            y_slice, conv_state, rec_state = self._run_step(
                x[:, s0:s1, :],
                a_all[:, s0:s1, :],
                b_all[:, s0:s1, :],
                conv_state,
                rec_state,
                mask_slice,
            )
            mx.eval(y_slice, conv_state, rec_state)  # guard: allow-eval
            outs.append(y_slice)
        return mx.concatenate(outs, axis=1)


class PplxMLP(nn.Module):
    """Fused Gate+Up SwiGLU MLP with sequence chunking above 2048 tokens."""

    def __init__(self, cfg: PplxConfig) -> None:
        super().__init__()
        self.intermediate_size = cfg.intermediate_size
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def _run_slice(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(x)
        gate, up = mx.split(gate_up, [self.intermediate_size], axis=-1)
        return self.down_proj(nn.silu(gate) * up)

    def __call__(self, x: mx.array) -> mx.array:
        S = x.shape[1]
        if S <= _STREAMED_SEQ_THRESHOLD:
            return self._run_slice(x)
        outs: list[mx.array] = []
        for s0, s1 in tile_aligned_slices(S, slice_len=_GDN_SLICE_LEN, min_tail=_GDN_SLICE_LEN):
            part = self._run_slice(x[:, s0:s1, :])
            mx.eval(part)  # guard: allow-eval
            outs.append(part)
        return mx.concatenate(outs, axis=1)


class PplxDecoderLayer(nn.Module):
    def __init__(self, cfg: PplxConfig, is_linear: bool) -> None:
        super().__init__()
        self.is_linear = is_linear
        self.input_layernorm = PplxRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = PplxRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = PplxMLP(cfg)
        if is_linear:
            self.linear_attn = PplxGatedDeltaNet(cfg)
        else:
            self.self_attn = PplxAttention(cfg)

    def __call__(
        self,
        h: mx.array,
        *,
        ssm_mask: mx.array | None = None,
        sdpa_mask: mx.array | None = None,
    ) -> mx.array:
        x_in = self.input_layernorm(h)
        if self.is_linear:
            r = self.linear_attn(x_in, ssm_mask=ssm_mask)
        else:
            r = self.self_attn(x_in, sdpa_mask=sdpa_mask)
        h = h + r
        return h + self.mlp(self.post_attention_layernorm(h))


class PplxModel(nn.Module):
    def __init__(self, cfg: PplxConfig, *, has_embed_module: bool = True) -> None:
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) if has_embed_module else None
        self.layers = [
            PplxDecoderLayer(cfg, is_linear=(cfg.layer_types[i] == "linear_attention"))
            for i in range(cfg.num_hidden_layers)
        ]
        self.norm = PplxRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.contextual_projection = nn.Linear(cfg.hidden_size, cfg.embedding_dim, bias=False)

    def forward_hidden(
        self,
        hidden: mx.array,
        *,
        ssm_mask: mx.array | None = None,
        sdpa_mask: mx.array | None = None,
        return_trace: bool = False,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        trace: list[mx.array] = []
        if return_trace:
            trace.append(hidden)
        h = hidden
        is_streamed = hidden.shape[1] > _STREAMED_SEQ_THRESHOLD
        for layer in self.layers:
            h = layer(h, ssm_mask=ssm_mask, sdpa_mask=sdpa_mask)
            if is_streamed:
                mx.eval(h)  # guard: allow-eval
            if return_trace:
                trace.append(h)
        h = self.norm(h)
        if return_trace:
            trace.append(h)
            return h, trace
        return h


def _is_zero_centered_norm_key(key: str) -> bool:
    if key == "norm.weight":
        return True
    if key.endswith("linear_attn.norm.weight"):
        return False
    return any(key.endswith(sfx) for sfx in _ZERO_CENTERED_NORM_SUFFIXES)


def _strip_language_prefix(key: str) -> str | None:
    if key.startswith("model.language_model."):
        return key[len("model.language_model.") :]
    if key.startswith("language_model."):
        return key[len("language_model.") :]
    if key.startswith("visual.") or key.startswith("model.visual."):
        return None
    return key


def _canonicalize_weights(
    raw_weights: Mapping[str, mx.array],
    *,
    num_layers: int,
    act_dtype: Any,
    include_embed: bool,
    norms_already_shifted: bool = False,
) -> tuple[list[tuple[str, mx.array]], mx.array]:
    normalized: dict[str, mx.array] = {}
    head_w: mx.array | None = None

    for orig_key, val in raw_weights.items():
        if orig_key == "contextual_projection.weight" or orig_key.endswith(".contextual_projection.weight"):
            head_w = val.astype(mx.float32)
            continue
        if orig_key.startswith("__"):
            continue
        key = _strip_language_prefix(orig_key)
        if key is None:
            continue
        if key == "embed_tokens.weight" and not include_embed:
            continue

        if "conv1d.weight" in key and val.ndim == 3 and val.shape[-1] != 1:
            val = val.moveaxis(2, 1)

        if not norms_already_shifted and _is_zero_centered_norm_key(key):
            val = val.astype(mx.float32) + 1.0

        if key.endswith(".A_log") or key.endswith(".dt_bias") or key.endswith("linear_attn.norm.weight"):
            val = val.astype(mx.float32)
        else:
            val = val.astype(act_dtype)

        normalized[key] = val

    # Fuse q_proj + k_proj + v_proj -> qkv_proj and gate_proj + up_proj -> gate_up_proj idempotently.
    for i in range(num_layers):
        attn_prefix = f"layers.{i}.self_attn"
        qkvs = (f"{attn_prefix}.q_proj.weight", f"{attn_prefix}.k_proj.weight", f"{attn_prefix}.v_proj.weight")
        if all(k in normalized for k in qkvs):
            normalized[f"{attn_prefix}.qkv_proj.weight"] = mx.concatenate([normalized.pop(k) for k in qkvs], axis=0)

        mlp_prefix = f"layers.{i}.mlp"
        gu = (f"{mlp_prefix}.gate_proj.weight", f"{mlp_prefix}.up_proj.weight")
        if all(k in normalized for k in gu):
            normalized[f"{mlp_prefix}.gate_up_proj.weight"] = mx.concatenate([normalized.pop(k) for k in gu], axis=0)

    if head_w is None:
        raise FileNotFoundError("Missing contextual_projection.weight in loaded weights")

    return sorted(normalized.items()), head_w


def _load_safetensors_dir(model_dir: Path) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    for st_file in st_files:
        weights.update(mx.load(str(st_file)))
    return weights


def _find_local_hf_snapshot(repo_id: str) -> Path | None:
    hf_home = os.environ.get("HF_HOME")
    default_cache = Path.home() / ".cache" / "huggingface"
    hub_dir = Path(os.environ.get("HF_HUB_CACHE") or (Path(hf_home) if hf_home else default_cache) / "hub")
    repo_dir = hub_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        cand = snapshots_dir / ref_main.read_text(encoding="utf-8").strip()
        if (cand / "config.json").exists():
            return cand
    for cand in sorted(snapshots_dir.iterdir()):
        if cand.is_dir() and (cand / "config.json").exists() and (cand / "tokenizer.json").exists():
            return cand
    return None


def _find_checkpoint_source(
    paths: RepoPaths,
    *,
    model_dir: str | Path | None,
    local_files_only: bool,
) -> tuple[str, Path, Path | None]:
    """Return ``(kind, main_path, optional_embed_safetensors)``."""
    if model_dir is not None:
        md = Path(model_dir).expanduser().resolve()
        return "dir", md, None
    if paths.weights_npz.exists():
        embed_st = paths.embed_tokens_safetensors if paths.embed_tokens_safetensors.exists() else None
        return "npz", paths.weights_npz, embed_st
    if (paths.model_dir / "config.json").exists():
        return "dir", paths.model_dir, None
    local_snap = _find_local_hf_snapshot(DEFAULT_MODEL_NAME)
    if local_snap is not None:
        return "dir", local_snap, None
    hf_dir = resolve_hf_model_dir(None, DEFAULT_MODEL_NAME, local_files_only=local_files_only)
    return "dir", hf_dir, None


def _build_sdpa_mask(attention_mask: np.ndarray, *, dtype_name: str) -> mx.array:
    neg = -65504.0 if dtype_name == "float16" else -1e9
    valid_4d = attention_mask.astype(np.bool_)[:, None, None, :]
    if dtype_name == "float16":
        arr = np.where(valid_4d, np.float16(0.0), np.float16(neg))
        return mx.array(arr, dtype=mx.float16)
    arr = np.where(valid_4d, np.float32(0.0), np.float32(neg))
    return mx.array(arr, dtype=mx.bfloat16)


def _quantize_int8_np(projected: np.ndarray) -> np.ndarray:
    scaled = np.tanh(projected) * 127.0
    return np.clip(np.rint(scaled), -128, 127).astype(np.int8)


class FastPplxGPU:
    """High-throughput MLX Metal GPU backend for `pplx-embed-v2-context-0.6b`."""

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        model_dir: str | Path | None = None,
        dtype: str = "float16",
        use_mapped_embedding: bool = True,
        cache_limit_mb: int = 512,
        local_files_only: bool = True,
        max_batch_size: int = 32,
        max_seq_len: int = 32768,
    ) -> None:
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(f"Unsupported dtype {dtype!r}; expected 'float16' or 'bfloat16'")

        self.paths = resolve_paths(home)
        self.dtype_name = dtype
        self.act_dtype = mx.float16 if dtype == "float16" else mx.bfloat16
        self.use_mapped_embedding = bool(use_mapped_embedding)
        self.cache_limit_mb = int(cache_limit_mb)
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        set_mlx_cache_limit(self.cache_limit_mb)

        kind, source_path, embed_st = _find_checkpoint_source(
            self.paths,
            model_dir=model_dir,
            local_files_only=local_files_only,
        )

        # Resolve config & tokenizer
        cfg_candidates = [
            self.paths.model_dir / "config.json",
            self.paths.tokenizer_dir / "config.json",
        ]
        if kind == "dir":
            cfg_candidates.insert(0, source_path / "config.json")
        cfg_file = next((c for c in cfg_candidates if c.exists()), None)
        self.config = PplxConfig.from_json(cfg_file) if cfg_file is not None else DEFAULT_PPLX_CONFIG

        tok_candidates = [
            source_path / "tokenizer.json" if kind == "dir" else self.paths.tokenizer_json,
            self.paths.tokenizer_json,
            self.paths.model_dir / "tokenizer.json",
        ]
        tok_file = next((c for c in tok_candidates if c.exists()), None)
        if tok_file is None:
            raise FileNotFoundError(f"Could not find tokenizer.json in {tok_candidates}")
        self.tokenizer = ContextualTokenizer(tok_file, self.config, default_max_length=self.max_seq_len)

        self.dim = self.config.embedding_dim
        self.max_dim = self.config.embedding_dim
        self.supports_dimensions = True
        self.model_name = DEFAULT_MODEL_NAME

        self._mapped_embed: MappedEmbedding | None = None
        if self.use_mapped_embedding:
            embed_candidates = [
                p
                for p in (
                    embed_st,
                    source_path if kind == "dir" else None,
                    self.paths.embed_tokens_safetensors,
                    self.paths.model_dir,
                )
                if p is not None
            ]
            for cand_path in embed_candidates:
                self._mapped_embed = _open_mapped_embedding(cand_path)
                if self._mapped_embed is not None:
                    break

        using_mapped = self._mapped_embed is not None

        self.model = PplxModel(self.config, has_embed_module=(not using_mapped))

        if kind == "dir":
            raw_weights = _load_safetensors_dir(source_path)
            norms_shifted = False
        else:
            raw_weights = dict(mx.load(str(source_path)))
            flag = raw_weights.pop("__norms_shifted__", None)
            norms_shifted = bool(int(flag.item())) if flag is not None else False

        backbone_items, head_w = _canonicalize_weights(
            raw_weights,
            num_layers=self.config.num_hidden_layers,
            act_dtype=self.act_dtype,
            include_embed=(not using_mapped),
            norms_already_shifted=norms_shifted,
        )
        all_items = [*backbone_items, ("contextual_projection.weight", head_w)]
        self.model.load_weights(all_items, strict=True)
        mx.eval(self.model.parameters())

    def _embed_inputs(self, input_ids: np.ndarray) -> mx.array:
        if self._mapped_embed is not None:
            gather_dtype = "float16" if self.dtype_name == "float16" else "float32"
            gathered = self._mapped_embed.gather(input_ids, out_dtype=gather_dtype)
            return mx.array(gathered, dtype=self.act_dtype)
        assert self.model.embed_tokens is not None
        return self.model.embed_tokens(mx.array(input_ids, dtype=mx.int32))

    def forward_hidden(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray | None = None,
        *,
        return_trace: bool = False,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        ids_np = np.asarray(input_ids, dtype=np.int32)
        if ids_np.ndim != 2:
            raise ValueError(f"Expected 2D input_ids, got shape {ids_np.shape}")
        if attention_mask is None:
            has_padding = False
            ssm_mask = None
            sdpa_mask = None
        else:
            mask_np = np.asarray(attention_mask, dtype=np.bool_)
            has_padding = not bool(np.all(mask_np))
            ssm_mask = mx.array(mask_np, dtype=mx.bool_) if has_padding else None
            sdpa_mask = _build_sdpa_mask(mask_np, dtype_name=self.dtype_name) if has_padding else None

        hidden = self._embed_inputs(ids_np)
        return self.model.forward_hidden(
            hidden,
            ssm_mask=ssm_mask,
            sdpa_mask=sdpa_mask,
            return_trace=return_trace,
        )

    def _project_mean_pool(self, hidden: mx.array, attention_mask: np.ndarray) -> np.ndarray:
        hidden_f32 = hidden.astype(mx.float32)
        mask_f32 = mx.array(attention_mask.astype(np.float32))[..., None]
        denom = mx.maximum(mx.sum(mask_f32, axis=1), mx.array(1.0, dtype=mx.float32))
        pooled = mx.sum(hidden_f32 * mask_f32, axis=1) / denom
        projected = self.model.contextual_projection(pooled)
        mx.eval(projected)
        return np.array(projected, dtype=np.float32, copy=False)

    def _project_span_pool(self, hidden: mx.array, chunk_token_spans: np.ndarray) -> np.ndarray:
        hidden_f32 = hidden.astype(mx.float32)
        B, _, H = hidden_f32.shape
        prefix = mx.concatenate([mx.zeros((B, 1, H), dtype=mx.float32), mx.cumsum(hidden_f32, axis=1)], axis=1)
        spans_mx = mx.array(chunk_token_spans, dtype=mx.int32)
        starts = spans_mx[..., 0]
        ends = spans_mx[..., 1]
        rows = mx.arange(B, dtype=mx.int32)[:, None]
        diff = ends - starts
        valid = diff > 0
        counts = mx.maximum(diff, 1)[..., None].astype(mx.float32)
        pooled = (prefix[rows, ends] - prefix[rows, starts]) / counts
        pooled = mx.where(valid[..., None], pooled, mx.array(0.0, dtype=mx.float32))
        projected = self.model.contextual_projection(pooled)
        projected = mx.where(valid[..., None], projected, mx.array(0.0, dtype=mx.float32))
        mx.eval(projected)
        return np.array(projected, dtype=np.float32, copy=False)

    def _format_output(
        self,
        projected: np.ndarray,
        *,
        dimensions: int | None,
        output_dtype: str,
        normalize: bool,
    ) -> np.ndarray:
        if output_dtype not in ("float32", "float16", "int8"):
            raise ValueError(f"Unsupported output_dtype {output_dtype!r}")
        target_dim = resolve_embedding_dim(dimensions, max_dim=self.dim)
        if projected.shape[0] == 0:
            np_dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8}[output_dtype]
            return np.zeros((0, target_dim), dtype=np_dt)
        sliced = projected[:, :target_dim]
        if output_dtype == "int8":
            return _quantize_int8_np(sliced)
        if normalize:
            vecs = truncate_and_normalize(projected, target_dim)
            # truncate_and_normalize returns unnormalized vectors when target_dim == width.
            if target_dim == projected.shape[-1]:
                norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
                vecs = np.ascontiguousarray(vecs / np.maximum(norms, 1e-12), dtype=np.float32)
        else:
            vecs = np.asarray(sliced, dtype=np.float32)
        if output_dtype == "float16":
            return vecs.astype(np.float16)
        return vecs.astype(np.float32, copy=False)

    def encode_tokens(
        self,
        token_lists: Sequence[Sequence[int]],
        *,
        batch_size: int = 16,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> np.ndarray:
        n = len(token_lists)
        target_dim = resolve_embedding_dim(dimensions, max_dim=self.dim)
        if n == 0:
            np_dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8}[output_dtype]
            return np.zeros((0, target_dim), dtype=np_dt)

        lengths = np.array([len(row) for row in token_lists], dtype=np.int64)
        order = np.argsort(lengths, kind="stable")
        bs = max(1, min(int(batch_size), self.max_batch_size))
        proj_all = np.zeros((n, self.dim), dtype=np.float32)

        for start in range(0, n, bs):
            idxs = order[start : start + bs]
            rows = [token_lists[int(i)] for i in idxs]
            input_ids, attention_mask = self.tokenizer.pad(rows)
            hidden = self.forward_hidden(input_ids, attention_mask)
            proj_batch = self._project_mean_pool(hidden, attention_mask)
            proj_all[idxs] = proj_batch

        return self._format_output(
            proj_all,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
        )

    def encode(
        self,
        texts: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        eff_max = min(int(max_length), self.max_seq_len) if max_length else self.max_seq_len
        if is_query:
            token_lists = [self.tokenizer.encode_query(t, max_length=eff_max) for t in items]
        else:
            token_lists = [self.tokenizer.encode_document(t, max_length=eff_max) for t in items]
        out = self.encode_tokens(
            token_lists,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
        )
        return out

    def encode_queries(
        self,
        queries: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> np.ndarray:
        return self.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            is_query=True,
        )

    def encode_documents(
        self,
        texts: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> np.ndarray:
        return self.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            is_query=False,
        )

    def encode_contextual_chunks(
        self,
        documents: Sequence[Sequence[str]],
        *,
        batch_size: int = 8,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> list[np.ndarray]:
        if not documents:
            return []
        eff_max = min(int(max_length), self.max_seq_len) if max_length else self.max_seq_len
        encoded_docs = [
            self.tokenizer.encode_contextual_document(chunks, return_spans=True, max_length=eff_max)
            for chunks in documents
        ]
        lengths = np.array([len(item[0]) for item in encoded_docs], dtype=np.int64)
        order = np.argsort(lengths, kind="stable")
        bs = max(1, min(int(batch_size), self.max_batch_size))
        results: list[np.ndarray | None] = [None] * len(documents)

        for start in range(0, len(documents), bs):
            idxs = order[start : start + bs]
            sub_docs = [documents[int(i)] for i in idxs]
            rows = [encoded_docs[int(i)][0] for i in idxs]
            spans_list = [encoded_docs[int(i)][1] for i in idxs]
            input_ids, attention_mask = self.tokenizer.pad(rows)
            max_c = max(1, *(len(s) for s in spans_list))
            span_arr = np.zeros((len(idxs), max_c, 2), dtype=np.int32)
            for row_i, sp in enumerate(spans_list):
                for col_j, (t0, t1) in enumerate(sp):
                    span_arr[row_i, col_j] = (t0, t1)

            hidden = self.forward_hidden(input_ids, attention_mask)
            proj = self._project_span_pool(hidden, span_arr)
            for row_i, orig_idx in enumerate(idxs):
                n_chunks = len(sub_docs[row_i])
                doc_proj = proj[row_i, :n_chunks]
                results[int(orig_idx)] = self._format_output(
                    doc_proj,
                    dimensions=dimensions,
                    output_dtype=output_dtype,
                    normalize=normalize,
                )

        return [r for r in results if r is not None]

    def close(self) -> None:
        if self._mapped_embed is not None:
            self._mapped_embed.close()
            self._mapped_embed = None

    def __enter__(self) -> FastPplxGPU:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "FastPplxGPU",
    "MappedEmbedding",
    "PplxAttention",
    "PplxDecoderLayer",
    "PplxGatedDeltaNet",
    "PplxGatedRMSNorm",
    "PplxMLP",
    "PplxModel",
    "PplxRMSNorm",
    "compute_g",
    "gated_delta_ops",
    "gated_delta_update",
    "normalize_qk",
    "set_mlx_cache_limit",
    "tile_aligned_slices",
]
