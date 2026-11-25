"""Continuous-batching rolling slot caches and decode loop for `FastQwen3Model` (`M4` / `M5+`)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import time
from typing import Any

import mlx.core as mx
import numpy as np

from fast_shared.grammar import GrammarHelper, detect_catastrophic_token_loop

from .gdn_kernels import (
    fused_gdn_pre_recurrence,
    gdn_delta_update_slot_indexed,
    quantize_gdn_state_blk32,
    resolve_gdn_state_mode,
)
from .kernels import apply_partial_rope_trig, fused_sigmoid_attn_gate
from .model import (
    DeltaSlotCache,
    FastQwen3Model,
    GrowingKVCache,
    _use_gdn_metal_override,
    dequantize_gdn_state_int8,
)


_FLASH_DECODE_1PASS_SRC = """
    constexpr int kGqa = N_Q / N_KV;
    constexpr int kElemsPerLane = HD / 32;
    constexpr int kTileT = 8;
    constexpr int kThreads = kGqa * 32;

    uint tid = thread_index_in_threadgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    uint blk = threadgroup_position_in_grid.z;
    uint s_q = params[1];
    uint q_i = blk % s_q;
    uint tmp = blk / s_q;
    uint h_kv = tmp % N_KV;
    uint b_idx = tmp / N_KV;
    uint h_q = h_kv * kGqa + sg;

    int vlen = valid_lens[b_idx];
    int k_limit = (s_q == 1) ? vlen : min(vlen, vlen - int(s_q) + int(q_i) + 1);
    size_t q_off = (((size_t(b_idx) * N_Q + h_q) * s_q + q_i) * HD) + lane * kElemsPerLane;

    if (k_limit <= 0) {
      #pragma unroll
      for (int j = 0; j < kElemsPerLane; ++j) {
        out[q_off + j] = T(0.0f);
      }
      return;
    }

    float scale = scale_arr[0];
    float q_reg[kElemsPerLane];
    #pragma unroll
    for (int j = 0; j < kElemsPerLane; ++j) {
      q_reg[j] = float(q[q_off + j]) * scale;
    }

    threadgroup T sh_k[kTileT * HD];
    threadgroup T sh_v[kTileT * HD];

    uint slot_b = uint(slot_ids[b_idx]);
    uint cap = params[0];
    size_t kv_slot_base = size_t(slot_b) * size_t(cap) * N_KV * HD + size_t(h_kv) * HD;
    size_t kv_t_stride = size_t(N_KV) * HD;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_acc[kElemsPerLane];
    #pragma unroll
    for (int j = 0; j < kElemsPerLane; ++j) {
      o_acc[j] = 0.0f;
    }

    for (int t_base = 0; t_base < k_limit; t_base += kTileT) {
      int n_rows = min(kTileT, k_limit - t_base);
      int total_elems = n_rows * HD;
      for (int idx = int(tid) * 2; idx < total_elems; idx += kThreads * 2) {
        int r = idx / HD;
        int d = idx - r * HD;
        size_t g_off = kv_slot_base + size_t(t_base + r) * kv_t_stride + d;
        sh_k[idx + 0] = k_buf[g_off + 0];
        sh_k[idx + 1] = k_buf[g_off + 1];
        sh_v[idx + 0] = v_buf[g_off + 0];
        sh_v[idx + 1] = v_buf[g_off + 1];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      int d0 = int(lane) * kElemsPerLane;
      for (int r = 0; r < n_rows; ++r) {
        const threadgroup T* kr = sh_k + r * HD + d0;
        const threadgroup T* vr = sh_v + r * HD + d0;
        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < kElemsPerLane; ++j) {
          dot += q_reg[j] * float(kr[j]);
        }
        dot += simd_shuffle_xor(dot, 1);
        dot += simd_shuffle_xor(dot, 2);
        dot += simd_shuffle_xor(dot, 4);
        dot += simd_shuffle_xor(dot, 8);
        dot += simd_shuffle_xor(dot, 16);

        float m_new = max(m_i, dot);
        float alpha = (m_i == -INFINITY) ? 0.0f : metal::exp(m_i - m_new);
        float p = metal::exp(dot - m_new);
        l_i = l_i * alpha + p;
        #pragma unroll
        for (int j = 0; j < kElemsPerLane; ++j) {
          o_acc[j] = o_acc[j] * alpha + p * float(vr[j]);
        }
        m_i = m_new;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    #pragma unroll
    for (int j = 0; j < kElemsPerLane; ++j) {
      out[q_off + j] = T(o_acc[j] * inv_l);
    }
"""

_FLASH_DECODE_SPLIT_P1_SRC = """
    constexpr int kGqa = N_Q / N_KV;
    constexpr int kElemsPerLane = HD / 32;
    constexpr int kTileT = 8;
    constexpr int kThreads = kGqa * 32;

    uint tid = thread_index_in_threadgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    uint split_id = threadgroup_position_in_grid.y;
    uint blk = threadgroup_position_in_grid.z;
    uint cap = params[0];
    uint s_q = params[1];
    uint split_len = params[2];
    uint n_splits = params[3];

    uint q_i = blk % s_q;
    uint tmp = blk / s_q;
    uint h_kv = tmp % N_KV;
    uint b_idx = tmp / N_KV;
    uint h_q = h_kv * kGqa + sg;

    int vlen = valid_lens[b_idx];
    int k_limit = (s_q == 1) ? vlen : min(vlen, vlen - int(s_q) + int(q_i) + 1);
    int t_start = int(split_id * split_len);
    int t_end = min(k_limit, t_start + int(split_len));

    size_t ml_base = ((((size_t(b_idx) * N_Q + h_q) * s_q + q_i) * n_splits + split_id) * 2);
    size_t po_base = ((((size_t(b_idx) * N_Q + h_q) * s_q + q_i) * n_splits + split_id) * HD) + lane * kElemsPerLane;

    if (t_start >= t_end) {
      if (lane == 0) {
        ml_part[ml_base + 0] = -INFINITY;
        ml_part[ml_base + 1] = 0.0f;
      }
      #pragma unroll
      for (int j = 0; j < kElemsPerLane; ++j) {
        o_part[po_base + j] = T(0.0f);
      }
      return;
    }

    size_t q_off = (((size_t(b_idx) * N_Q + h_q) * s_q + q_i) * HD) + lane * kElemsPerLane;
    float scale = scale_arr[0];
    float q_reg[kElemsPerLane];
    #pragma unroll
    for (int j = 0; j < kElemsPerLane; ++j) {
      q_reg[j] = float(q[q_off + j]) * scale;
    }

    threadgroup T sh_k[kTileT * HD];
    threadgroup T sh_v[kTileT * HD];

    uint slot_b = uint(slot_ids[b_idx]);
    size_t kv_slot_base = size_t(slot_b) * size_t(cap) * N_KV * HD + size_t(h_kv) * HD;
    size_t kv_t_stride = size_t(N_KV) * HD;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_acc[kElemsPerLane];
    #pragma unroll
    for (int j = 0; j < kElemsPerLane; ++j) {
      o_acc[j] = 0.0f;
    }

    for (int t_base = t_start; t_base < t_end; t_base += kTileT) {
      int n_rows = min(kTileT, t_end - t_base);
      int total_elems = n_rows * HD;
      for (int idx = int(tid) * 2; idx < total_elems; idx += kThreads * 2) {
        int r = idx / HD;
        int d = idx - r * HD;
        size_t g_off = kv_slot_base + size_t(t_base + r) * kv_t_stride + d;
        sh_k[idx + 0] = k_buf[g_off + 0];
        sh_k[idx + 1] = k_buf[g_off + 1];
        sh_v[idx + 0] = v_buf[g_off + 0];
        sh_v[idx + 1] = v_buf[g_off + 1];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      int d0 = int(lane) * kElemsPerLane;
      for (int r = 0; r < n_rows; ++r) {
        const threadgroup T* kr = sh_k + r * HD + d0;
        const threadgroup T* vr = sh_v + r * HD + d0;
        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < kElemsPerLane; ++j) {
          dot += q_reg[j] * float(kr[j]);
        }
        dot += simd_shuffle_xor(dot, 1);
        dot += simd_shuffle_xor(dot, 2);
        dot += simd_shuffle_xor(dot, 4);
        dot += simd_shuffle_xor(dot, 8);
        dot += simd_shuffle_xor(dot, 16);

        float m_new = max(m_i, dot);
        float alpha = (m_i == -INFINITY) ? 0.0f : metal::exp(m_i - m_new);
        float p = metal::exp(dot - m_new);
        l_i = l_i * alpha + p;
        #pragma unroll
        for (int j = 0; j < kElemsPerLane; ++j) {
          o_acc[j] = o_acc[j] * alpha + p * float(vr[j]);
        }
        m_i = m_new;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
      ml_part[ml_base + 0] = m_i;
      ml_part[ml_base + 1] = l_i;
    }
    float inv_ls = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    #pragma unroll
    for (int j = 0; j < kElemsPerLane; ++j) {
      o_part[po_base + j] = T(o_acc[j] * inv_ls);
    }
"""

_FLASH_DECODE_SPLIT_P2_SRC = """
    constexpr int kElemsPerLane = HD / 32;
    uint lane = thread_index_in_simdgroup;
    uint blk = threadgroup_position_in_grid.z;
    uint s_q = params[1];
    uint n_splits = params[3];

    uint q_i = blk % s_q;
    uint tmp = blk / s_q;
    uint h_q = tmp % N_Q;
    uint b_idx = tmp / N_Q;

    size_t head_base = ((size_t(b_idx) * N_Q + h_q) * s_q + q_i);
    size_t ml_head = head_base * n_splits * 2;
    size_t po_head = head_base * n_splits * HD + lane * kElemsPerLane;

    float m_max = -INFINITY;
    for (uint s = 0; s < n_splits; ++s) {
      float ls = ml_part[ml_head + s * 2 + 1];
      if (ls > 0.0f) {
        m_max = max(m_max, ml_part[ml_head + s * 2 + 0]);
      }
    }

    size_t out_off = head_base * HD + lane * kElemsPerLane;
    if (m_max == -INFINITY) {
      #pragma unroll
      for (int j = 0; j < kElemsPerLane; ++j) {
        out[out_off + j] = T(0.0f);
      }
      return;
    }

    float l_tot = 0.0f;
    float o_acc[kElemsPerLane];
    #pragma unroll
    for (int j = 0; j < kElemsPerLane; ++j) {
      o_acc[j] = 0.0f;
    }

    for (uint s = 0; s < n_splits; ++s) {
      float ls = ml_part[ml_head + s * 2 + 1];
      if (ls > 0.0f) {
        float ms = ml_part[ml_head + s * 2 + 0];
        float sw = ls * metal::exp(ms - m_max);
        l_tot += sw;
        size_t s_off = po_head + size_t(s) * HD;
        #pragma unroll
        for (int j = 0; j < kElemsPerLane; ++j) {
          o_acc[j] += float(o_part[s_off + j]) * sw;
        }
      }
    }

    float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    #pragma unroll
    for (int j = 0; j < kElemsPerLane; ++j) {
      out[out_off + j] = T(o_acc[j] * inv_l);
    }
"""

_FLASH_DECODE_KERNELS: dict[str, Any] = {}
_FD_SCALE_CACHE: dict[float, mx.array] = {}
_FD_PARAMS_CACHE: dict[tuple[int, int, int, int], mx.array] = {}


def _get_flash_decode_kernels() -> dict[str, Any]:
    """Lazily compile Metal Flash-Decode kernels (1-pass + split-K pass1/pass2)."""
    if not _FLASH_DECODE_KERNELS:
        _FLASH_DECODE_KERNELS["1pass"] = mx.fast.metal_kernel(
            name="fast_qwen3_flash_decode_1pass",
            input_names=["q", "k_buf", "v_buf", "slot_ids", "valid_lens", "params", "scale_arr"],
            output_names=["out"],
            source=_FLASH_DECODE_1PASS_SRC,
        )
        _FLASH_DECODE_KERNELS["split_p1"] = mx.fast.metal_kernel(
            name="fast_qwen3_flash_decode_split_p1",
            input_names=["q", "k_buf", "v_buf", "slot_ids", "valid_lens", "params", "scale_arr"],
            output_names=["o_part", "ml_part"],
            source=_FLASH_DECODE_SPLIT_P1_SRC,
        )
        _FLASH_DECODE_KERNELS["split_p2"] = mx.fast.metal_kernel(
            name="fast_qwen3_flash_decode_split_p2",
            input_names=["o_part", "ml_part", "params"],
            output_names=["out"],
            source=_FLASH_DECODE_SPLIT_P2_SRC,
        )
    return _FLASH_DECODE_KERNELS


def _get_fd_scale_arr(scale: float) -> mx.array:
    arr = _FD_SCALE_CACHE.get(scale)
    if arr is None:
        arr = mx.array([scale], dtype=mx.float32)  # guard: allow-fp32
        if len(_FD_SCALE_CACHE) < 32:
            _FD_SCALE_CACHE[scale] = arr
    return arr


def _get_fd_params_arr(cap: int, s_q: int, split_len: int, n_splits: int) -> mx.array:
    key = (cap, s_q, split_len, n_splits)
    arr = _FD_PARAMS_CACHE.get(key)
    if arr is None:
        arr = mx.array([cap, s_q, split_len, n_splits], dtype=mx.uint32)
        if len(_FD_PARAMS_CACHE) < 256:
            _FD_PARAMS_CACHE[key] = arr
    return arr


def _can_use_flash_decode_metal(dt: mx.Dtype, n_q: int, n_kv: int, hd: int, s_q: int = 1) -> bool:
    if s_q <= 0 or s_q > 8:
        return False
    if not (mx.metal.is_available() and mx.default_device().type == mx.DeviceType.gpu):
        return False
    if dt not in (mx.bfloat16, mx.float16):
        return False
    if hd < 32 or (hd % 32) != 0 or (hd % 2) != 0 or (2 * 8 * hd * 2) > 28672:
        return False
    if n_kv <= 0 or (n_q % n_kv) != 0 or (n_q // n_kv) > 32:
        return False
    return True


def attend_rolling_flash_decode(
    q: mx.array,
    k_buf: mx.array,
    v_buf: mx.array,
    slot_ids: mx.array,
    valid_lens: mx.array,
    *,
    scale: float,
    max_valid: int,
    force_splits: int | None = None,
    use_kernel: bool = True,
) -> mx.array:
    """Zero-copy slot-indexed Flash-Decode SDPA with dynamic Split-K and CPU/MLX fallback."""
    b_act, n_q, s_q, hd = (int(d) for d in q.shape)
    cap = int(k_buf.shape[1])
    n_kv = int(k_buf.shape[2])
    if max_valid <= 0 or b_act == 0:
        return mx.zeros_like(q)

    dt = q.dtype
    slot_idx = slot_ids.astype(mx.int32) if slot_ids.dtype != mx.int32 else slot_ids
    valid_arr = valid_lens.astype(mx.int32) if valid_lens.dtype != mx.int32 else valid_lens
    if not (
        use_kernel
        and _can_use_flash_decode_metal(dt, n_q, n_kv, hd, s_q=s_q)
        and k_buf.dtype == dt
        and v_buf.dtype == dt
    ):
        eff_max = min(cap, int(max_valid))
        k_b = mx.take(k_buf[:, :eff_max, :, :], slot_idx, axis=0).transpose(0, 2, 1, 3)
        v_b = mx.take(v_buf[:, :eff_max, :, :], slot_idx, axis=0).transpose(0, 2, 1, 3)
        k_pos = mx.arange(eff_max, dtype=mx.int32)[None, None, None, :]
        if s_q == 1:
            allow = k_pos < valid_arr[:, None, None, None]
        else:
            q_pos = (valid_arr - s_q)[:, None, None, None] + mx.arange(s_q, dtype=mx.int32)[None, None, :, None]
            allow = (k_pos < valid_arr[:, None, None, None]) & (k_pos <= q_pos)
        nz = valid_arr > 0
        # For valid_len == 0 rows, expose position 0 in mask (safe finite softmax) then zero output row.
        safe_allow = mx.where(nz[:, None, None, None], allow, k_pos == 0)
        mask_bf16 = mx.where(
            safe_allow,
            mx.array(0.0, dtype=mx.bfloat16),
            mx.array(-1e9, dtype=mx.bfloat16),
        )
        raw_out = mx.fast.scaled_dot_product_attention(q, k_b, v_b, scale=float(scale), mask=mask_bf16)
        return mx.where(nz[:, None, None, None], raw_out, mx.zeros_like(raw_out))

    qc = mx.contiguous(q)
    gqa = n_q // n_kv
    tg_threads = gqa * 32
    total_blks = b_act * n_kv * s_q
    scale_arr = _get_fd_scale_arr(float(scale))
    kerns = _get_flash_decode_kernels()

    if force_splits is not None:
        n_splits = max(1, int(force_splits))
    elif max_valid < 256 or total_blks >= 64:
        n_splits = 1
    else:
        target_tgs = 80
        want = max(1, (target_tgs + total_blks - 1) // total_blks)
        max_by_len = max(1, max_valid // 64)
        n_splits = min(16, want, max_by_len)

    if n_splits <= 1:
        params = _get_fd_params_arr(cap, s_q, max_valid, 1)
        (out,) = kerns["1pass"](
            inputs=[qc, k_buf, v_buf, slot_idx, valid_arr, params, scale_arr],
            template=[("T", dt), ("N_Q", n_q), ("N_KV", n_kv), ("HD", hd)],
            grid=(tg_threads, 1, total_blks),
            threadgroup=(tg_threads, 1, 1),
            output_shapes=[qc.shape],
            output_dtypes=[dt],
        )
        return out

    split_len = max(64, ((max_valid + n_splits - 1) // n_splits + 7) // 8 * 8)
    n_splits_eff = max(1, (max_valid + split_len - 1) // split_len)
    params = _get_fd_params_arr(cap, s_q, split_len, n_splits_eff)
    o_part, ml_part = kerns["split_p1"](  # guard: allow-fp32
        inputs=[qc, k_buf, v_buf, slot_idx, valid_arr, params, scale_arr],
        template=[("T", dt), ("N_Q", n_q), ("N_KV", n_kv), ("HD", hd)],
        grid=(tg_threads, n_splits_eff, total_blks),
        threadgroup=(tg_threads, 1, 1),
        output_shapes=[(b_act, n_q, s_q, n_splits_eff, hd), (b_act, n_q, s_q, n_splits_eff, 2)],
        output_dtypes=[dt, mx.float32],
    )
    (out,) = kerns["split_p2"](
        inputs=[o_part, ml_part, params],
        template=[("T", dt), ("N_Q", n_q), ("N_KV", n_kv), ("HD", hd)],
        grid=(32, 1, b_act * n_q * s_q),
        threadgroup=(32, 1, 1),
        output_shapes=[qc.shape],
        output_dtypes=[dt],
    )
    return out


class RollingKVCache:
    """Preallocated `(B_slots, Cap, n_kv, hd)` BF16 KV slot buffer with in-place MLX scatter."""

    def __init__(self, b_slots: int, cap: int, n_kv: int, head_dim: int) -> None:
        self.b_slots = int(b_slots)
        self.cap = int(cap)
        self.n_kv = int(n_kv)
        self.head_dim = int(head_dim)
        self.k_buf: mx.array = mx.zeros(
            (self.b_slots, self.cap, self.n_kv, self.head_dim), dtype=mx.bfloat16
        )
        self.v_buf: mx.array = mx.zeros(
            (self.b_slots, self.cap, self.n_kv, self.head_dim), dtype=mx.bfloat16
        )

    def reset_slots(self, slot_ids: mx.array | Sequence[int]) -> None:
        """Zero KV buffers for the given `slot_ids`."""
        idx = slot_ids if isinstance(slot_ids, mx.array) else mx.array(slot_ids, dtype=mx.int32)
        n = int(idx.size)
        if n == 0:
            return
        z = mx.zeros((n, self.cap, self.n_kv, self.head_dim), dtype=mx.bfloat16)
        self.k_buf[idx] = z
        self.v_buf[idx] = z

    def scatter_prefill(
        self,
        slot_id: int,
        k_1_h_s_d: mx.array,
        v_1_h_s_d: mx.array,
        start_pos: int = 0,
    ) -> None:
        """Scatter single-slot prefill `(1, n_kv, s, hd)` into `(B_slots, Cap, n_kv, hd)` in-place."""
        sid = int(slot_id)
        s0 = int(start_pos)
        s_len = int(k_1_h_s_d.shape[2])
        end_pos = s0 + s_len
        if s0 < 0 or end_pos > self.cap:
            raise ValueError(f"KV prefill range [{s0}, {end_pos}) out of bounds for Cap={self.cap}")
        if s_len == 0:
            return
        self.k_buf[sid, s0:end_pos] = k_1_h_s_d[0].transpose(1, 0, 2).astype(mx.bfloat16)
        self.v_buf[sid, s0:end_pos] = v_1_h_s_d[0].transpose(1, 0, 2).astype(mx.bfloat16)

    def scatter_prefill_batch(
        self,
        slot_ids: mx.array | Sequence[int],
        k_b_h_s_d: mx.array,
        v_b_h_s_d: mx.array,
        start_pos: int = 0,
    ) -> None:
        """Scatter batched prefill `(b_act, n_kv, s, hd)` starting at `start_pos` across `slot_ids`."""
        s0 = int(start_pos)
        s_len = int(k_b_h_s_d.shape[2])
        end_pos = s0 + s_len
        if s0 < 0 or end_pos > self.cap:
            raise ValueError(f"KV batch prefill range [{s0}, {end_pos}) out of bounds for Cap={self.cap}")
        if s_len == 0:
            return
        slot_idx = slot_ids if isinstance(slot_ids, mx.array) else mx.array(slot_ids, dtype=mx.int32)
        pos_idx = s0 + mx.arange(s_len, dtype=mx.int32)
        self.k_buf[slot_idx[:, None], pos_idx[None, :]] = k_b_h_s_d.transpose(0, 2, 1, 3).astype(mx.bfloat16)
        self.v_buf[slot_idx[:, None], pos_idx[None, :]] = v_b_h_s_d.transpose(0, 2, 1, 3).astype(mx.bfloat16)

    def scatter_decode_step(
        self,
        slot_ids: mx.array | Sequence[int],
        positions: mx.array | Sequence[int],
        k_b_h_1_d: mx.array,
        v_b_h_1_d: mx.array,
    ) -> None:
        """In-place scatter of `(b_act, n_kv, 1, hd)` (or `s>=1`) at per-slot `positions`."""
        slot_idx = slot_ids if isinstance(slot_ids, mx.array) else mx.array(slot_ids, dtype=mx.int32)
        pos_idx = positions if isinstance(positions, mx.array) else mx.array(positions, dtype=mx.int32)
        s_len = int(k_b_h_1_d.shape[2])
        if s_len == 1:
            self.k_buf[slot_idx, pos_idx] = k_b_h_1_d[:, :, 0, :].astype(mx.bfloat16)
            self.v_buf[slot_idx, pos_idx] = v_b_h_1_d[:, :, 0, :].astype(mx.bfloat16)
            return
        pos_2d = pos_idx[:, None] + mx.arange(s_len, dtype=mx.int32)[None, :]
        self.k_buf[slot_idx[:, None], pos_2d] = k_b_h_1_d.transpose(0, 2, 1, 3).astype(mx.bfloat16)
        self.v_buf[slot_idx[:, None], pos_2d] = v_b_h_1_d.transpose(0, 2, 1, 3).astype(mx.bfloat16)

    def _run_sdpa(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        scale: float,
        mask: str | mx.array | None,
        use_apple10_force_fused: bool,
    ) -> mx.array:
        if use_apple10_force_fused and int(q.shape[-1]) == 256:
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask, force_fused=True
            )
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    def attend_slot_single(
        self,
        slot_id: int,
        q_1_h_s_d: mx.array,
        cur_len: int,
        scale: float,
        use_apple10_force_fused: bool = False,
    ) -> mx.array:
        """Single-doc SDPA `(1, n_q, s, hd)` against `(1, n_kv, cur_len, hd)`."""
        sid = int(slot_id)
        clen = int(cur_len)
        k_1 = self.k_buf[sid : sid + 1, :clen, :, :].transpose(0, 2, 1, 3)
        v_1 = self.v_buf[sid : sid + 1, :clen, :, :].transpose(0, 2, 1, 3)
        eff_mask: str | None = None if int(q_1_h_s_d.shape[2]) == 1 else "causal"
        return self._run_sdpa(
            q_1_h_s_d,
            k_1,
            v_1,
            scale=scale,
            mask=eff_mask,
            use_apple10_force_fused=use_apple10_force_fused,
        )

    def attend_prefill_batch(
        self,
        slot_ids: mx.array | Sequence[int],
        q_b_h_s_d: mx.array,
        cur_len: int,
        scale: float,
        use_apple10_force_fused: bool = False,
    ) -> mx.array:
        """Batched causal prefill SDPA `(b_act, n_q, s, hd)` against `(b_act, n_kv, cur_len, hd)`."""
        slot_idx = slot_ids if isinstance(slot_ids, mx.array) else mx.array(slot_ids, dtype=mx.int32)
        clen = int(cur_len)
        k_b = mx.take(self.k_buf[:, :clen, :, :], slot_idx, axis=0).transpose(0, 2, 1, 3)
        v_b = mx.take(self.v_buf[:, :clen, :, :], slot_idx, axis=0).transpose(0, 2, 1, 3)
        eff_mask: str | None = None if int(q_b_h_s_d.shape[2]) == 1 else "causal"
        return self._run_sdpa(
            q_b_h_s_d,
            k_b,
            v_b,
            scale=scale,
            mask=eff_mask,
            use_apple10_force_fused=use_apple10_force_fused,
        )

    def attend_rolling_decode(
        self,
        q_b_h_1_d: mx.array,
        slot_ids: mx.array | Sequence[int],
        valid_lens: mx.array | Sequence[int],
        scale: float,
        use_apple10_force_fused: bool = False,
        *,
        max_valid: int | None = None,
    ) -> mx.array:
        """Gather active KV slices up to `max_valid` and attend with a strict-BF16 valid/causal mask."""
        slot_idx = slot_ids if isinstance(slot_ids, mx.array) else mx.array(slot_ids, dtype=mx.int32)
        if max_valid is not None:
            eff_max = int(max_valid)
            valid_arr = valid_lens if isinstance(valid_lens, mx.array) else mx.array(valid_lens, dtype=mx.int32)
        elif isinstance(valid_lens, mx.array):
            valid_arr = valid_lens.astype(mx.int32)
            eff_max = int(mx.max(valid_arr).item()) if int(valid_arr.size) > 0 else 0
        else:
            eff_max = max((int(v) for v in valid_lens), default=0)
            valid_arr = mx.array(valid_lens, dtype=mx.int32)

        if eff_max <= 0:
            return mx.zeros_like(q_b_h_1_d)
        if eff_max > self.cap:
            raise ValueError(f"max_valid={eff_max} exceeds Cap={self.cap}")

        n_q = int(q_b_h_1_d.shape[1])
        s_q_cur = int(q_b_h_1_d.shape[2])
        if (
            _can_use_flash_decode_metal(q_b_h_1_d.dtype, n_q, self.n_kv, self.head_dim, s_q=s_q_cur)
            and self.k_buf.dtype == q_b_h_1_d.dtype
            and self.v_buf.dtype == q_b_h_1_d.dtype
        ):
            return attend_rolling_flash_decode(
                q_b_h_1_d,
                self.k_buf,
                self.v_buf,
                slot_idx,
                valid_arr,
                scale=float(scale),
                max_valid=eff_max,
            )

        k_b = mx.take(self.k_buf[:, :eff_max, :, :], slot_idx, axis=0).transpose(0, 2, 1, 3)
        v_b = mx.take(self.v_buf[:, :eff_max, :, :], slot_idx, axis=0).transpose(0, 2, 1, 3)

        s_q = int(q_b_h_1_d.shape[2])
        k_pos = mx.arange(eff_max, dtype=mx.int32)[None, None, None, :]
        if s_q == 1:
            allow = k_pos < valid_arr[:, None, None, None]
        else:
            q_pos = (valid_arr - s_q)[:, None, None, None] + mx.arange(s_q, dtype=mx.int32)[None, None, :, None]
            allow = (k_pos < valid_arr[:, None, None, None]) & (k_pos <= q_pos)

        # Keep SDPA mask strictly in BF16 (never bool/int) to satisfy fast-ir-guard --strict-bool-mask.
        mask_bf16 = mx.where(
            allow,
            mx.array(0.0, dtype=mx.bfloat16),
            mx.array(-1e9, dtype=mx.bfloat16),
        )
        return self._run_sdpa(
            q_b_h_1_d,
            k_b,
            v_b,
            scale=scale,
            mask=mask_bf16,
            use_apple10_force_fused=use_apple10_force_fused,
        )


class RollingDeltaSlotCache:
    """Preallocated persistent GDN `conv_state` + `rec_state` (`int8_blk32` / `fp32` / `bf16`) slot buffers."""

    def __init__(
        self,
        b_slots: int,
        conv_k: int,
        d_conv: int,
        num_v_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        *,
        gdn_state_mode: str = "auto",
    ) -> None:
        self.b_slots = int(b_slots)
        self.conv_depth = max(0, int(conv_k) - 1)
        self.d_conv = int(d_conv)
        self.num_v_heads = int(num_v_heads)
        self.head_v_dim = int(head_v_dim)
        self.head_k_dim = int(head_k_dim)
        self.state_mode: str = resolve_gdn_state_mode(
            gdn_state_mode,
            batch_size=self.b_slots,
            head_k_dim=self.head_k_dim,
        )
        self.conv_state: mx.array = mx.zeros(
            (self.b_slots, self.conv_depth, self.d_conv), dtype=mx.bfloat16
        )
        self.rec_codes: mx.array | None = None
        self.rec_scales: mx.array | None = None
        self.rec_state: mx.array | None = None
        if self.state_mode == "int8_blk32":
            self.rec_codes = mx.zeros(
                (self.b_slots, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                dtype=mx.int8,
            )
            self.rec_scales = mx.zeros(
                (self.b_slots, self.num_v_heads, self.head_v_dim, self.head_k_dim // 32),
                dtype=mx.bfloat16,
            )
        else:
            st_dt = mx.bfloat16 if self.state_mode == "bf16" else mx.float32
            self.rec_state = mx.zeros(
                (self.b_slots, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                dtype=st_dt,
            )

    def slice_for_slots(self, slot_ids: mx.array | Sequence[int]) -> DeltaSlotCache:
        """Materialize a `DeltaSlotCache` view for `slot_ids` to feed `GatedDeltaLayer` directly."""
        idx = slot_ids if isinstance(slot_ids, mx.array) else mx.array(slot_ids, dtype=mx.int32)
        conv = mx.take(self.conv_state, idx, axis=0)
        if self.state_mode == "int8_blk32":
            assert self.rec_codes is not None and self.rec_scales is not None
            return DeltaSlotCache(
                conv_state=conv,
                state_mode="int8_blk32",
                rec_codes=mx.take(self.rec_codes, idx, axis=0),
                rec_scales=mx.take(self.rec_scales, idx, axis=0),
            )
        assert self.rec_state is not None
        return DeltaSlotCache(
            conv_state=conv,
            rec_state=mx.take(self.rec_state, idx, axis=0),
            state_mode=self.state_mode,
        )

    def scatter_from_subcache(
        self,
        slot_ids: mx.array | Sequence[int],
        sub_cache: DeltaSlotCache,
    ) -> None:
        """Scatter updated `sub_cache` state back into persistent `slot_ids` in-place."""
        idx = slot_ids if isinstance(slot_ids, mx.array) else mx.array(slot_ids, dtype=mx.int32)
        if sub_cache.conv_state is not None:
            self.conv_state[idx] = sub_cache.conv_state.astype(self.conv_state.dtype)

        if self.state_mode == "int8_blk32":
            assert self.rec_codes is not None and self.rec_scales is not None
            if sub_cache.rec_codes is not None and sub_cache.rec_scales is not None:
                self.rec_codes[idx] = sub_cache.rec_codes
                self.rec_scales[idx] = sub_cache.rec_scales
            elif sub_cache._rec_state is not None:
                codes, scales = quantize_gdn_state_blk32(sub_cache._rec_state)
                self.rec_codes[idx] = codes
                self.rec_scales[idx] = scales
            return

        assert self.rec_state is not None
        if sub_cache._rec_state is not None:
            self.rec_state[idx] = sub_cache._rec_state.astype(self.rec_state.dtype)
        elif sub_cache.rec_codes is not None and sub_cache.rec_scales is not None:
            deq = dequantize_gdn_state_int8(
                sub_cache.rec_codes, sub_cache.rec_scales, dtype=self.rec_state.dtype
            )
            self.rec_state[idx] = deq

    def reset_slots(self, slot_ids: mx.array | Sequence[int]) -> None:
        """Zero `conv_state` and recurrent state for freed/newly admitted `slot_ids`."""
        idx = slot_ids if isinstance(slot_ids, mx.array) else mx.array(slot_ids, dtype=mx.int32)
        n = int(idx.size)
        if n == 0:
            return
        self.conv_state[idx] = mx.zeros(
            (n, self.conv_depth, self.d_conv), dtype=self.conv_state.dtype
        )
        if self.state_mode == "int8_blk32":
            assert self.rec_codes is not None and self.rec_scales is not None
            self.rec_codes[idx] = mx.zeros(
                (n, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.int8
            )
            self.rec_scales[idx] = mx.zeros(
                (n, self.num_v_heads, self.head_v_dim, self.head_k_dim // 32), dtype=mx.bfloat16
            )
        else:
            assert self.rec_state is not None
            self.rec_state[idx] = mx.zeros(
                (n, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                dtype=self.rec_state.dtype,
            )


def _full_attn_step(
    layer: Any,
    kv_cache: RollingKVCache,
    x_in: mx.array,
    *,
    cos_a: mx.array | None,
    sin_a: mx.array | None,
    rope_offset: int,
    slot_ids: mx.array | Sequence[int],
    positions: mx.array | Sequence[int] | None,
    valid_lens: mx.array | Sequence[int] | None,
    is_prefill: bool,
    prefill_start: int = 0,
    prefill_cur_len: int = 0,
    max_valid: int | None = None,
) -> mx.array:
    """Shared `FullAttentionLayer` forward reusing existing `model` weights with zero duplication."""
    sa = layer.self_attn
    b_act, s_len, _ = (int(d) for d in x_in.shape)
    parts = sa.qkvg_group(x_in)
    q = sa.q_norm(parts["q"].reshape(b_act, s_len, sa.num_q_heads, sa.head_dim)).transpose(0, 2, 1, 3)
    k = sa.k_norm(parts["k"].reshape(b_act, s_len, sa.num_kv_heads, sa.head_dim)).transpose(0, 2, 1, 3)
    v = parts["v"].reshape(b_act, s_len, sa.num_kv_heads, sa.head_dim).transpose(0, 2, 1, 3)
    gate = parts["gate"]

    if cos_a is not None and sin_a is not None:
        q = apply_partial_rope_trig(q, cos_a, sin_a, sa.rot_dims)
        k = apply_partial_rope_trig(k, cos_a, sin_a, sa.rot_dims)
    else:
        qk = sa.rope(mx.concatenate([q, k], axis=1), offset=int(rope_offset))
        q, k = mx.split(qk, [sa.num_q_heads], axis=1)

    if is_prefill:
        if b_act == 1:
            sid = int(slot_ids[0]) if not isinstance(slot_ids, mx.array) else int(slot_ids.item())
            kv_cache.scatter_prefill(sid, k, v, start_pos=prefill_start)
            attn_out = kv_cache.attend_slot_single(
                sid,
                q,
                cur_len=prefill_cur_len,
                scale=sa.scale,
                use_apple10_force_fused=sa._use_apple10_force_fused,
            )
        else:
            kv_cache.scatter_prefill_batch(slot_ids, k, v, start_pos=prefill_start)
            attn_out = kv_cache.attend_prefill_batch(
                slot_ids,
                q,
                cur_len=prefill_cur_len,
                scale=sa.scale,
                use_apple10_force_fused=sa._use_apple10_force_fused,
            )
    else:
        assert positions is not None and valid_lens is not None
        kv_cache.scatter_decode_step(slot_ids, positions, k, v)
        attn_out = kv_cache.attend_rolling_decode(
            q,
            slot_ids,
            valid_lens,
            scale=sa.scale,
            use_apple10_force_fused=sa._use_apple10_force_fused,
            max_valid=max_valid,
        )

    attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(b_act, s_len, sa.q_dim)
    gated = fused_sigmoid_attn_gate(attn_flat, gate)
    res = sa.o_proj(gated)
    if sa._o_lora is not None:
        la_t, lb_t, sc = sa._o_lora
        res = res + (mx.matmul(mx.matmul(gated.astype(mx.bfloat16), la_t), lb_t) * sc).astype(res.dtype)
    return res


class RollingHybridCache:
    """Hybrid persistent slot cache (`3/4` `RollingDeltaSlotCache` + `1/4` `RollingKVCache`) for `FastQwen3Model`."""

    def __init__(
        self,
        model: FastQwen3Model,
        *,
        b_slots: int,
        cap: int,
        gdn_state_mode: str = "auto",
    ) -> None:
        self.model = model
        self.b_slots = int(b_slots)
        self.cap = int(cap)
        self.gdn_state_mode: str = resolve_gdn_state_mode(
            gdn_state_mode,
            batch_size=self.b_slots,
            head_k_dim=int(model.cfg.linear_key_head_dim),
        )
        self.layer_caches: list[RollingDeltaSlotCache | RollingKVCache] = []
        for layer in model.layers:
            if layer.is_linear:
                la = layer.linear_attn
                self.layer_caches.append(
                    RollingDeltaSlotCache(
                        self.b_slots,
                        la.conv_k,
                        la.conv_dim,
                        la.num_v_heads,
                        la.head_v_dim,
                        la.head_k_dim,
                        gdn_state_mode=self.gdn_state_mode,
                    )
                )
            else:
                sa = layer.self_attn
                self.layer_caches.append(
                    RollingKVCache(self.b_slots, self.cap, sa.num_kv_heads, sa.head_dim)
                )

        self.slot_pos: list[int] = [0] * self.b_slots
        self.slot_valid_len: list[int] = [0] * self.b_slots
        self.cur_h: mx.array = mx.zeros((self.b_slots, int(model.cfg.hidden_size)), dtype=mx.bfloat16)

    def state_arrays(self) -> list[mx.array]:
        """Return all persistent mutable slot-state arrays for O(1) graph-depth evaluation."""
        arrs: list[mx.array] = []
        for entry in self.layer_caches:
            if isinstance(entry, RollingDeltaSlotCache):
                arrs.append(entry.conv_state)
                if entry.rec_codes is not None and entry.rec_scales is not None:
                    arrs.extend((entry.rec_codes, entry.rec_scales))
                elif entry.rec_state is not None:
                    arrs.append(entry.rec_state)
            else:
                arrs.extend((entry.k_buf, entry.v_buf))
        return arrs

    def reset_slots(self, slot_ids: Sequence[int]) -> None:
        """Reset linear + KV + metadata state for `slot_ids`."""
        if not slot_ids:
            return
        idx = mx.array(slot_ids, dtype=mx.int32)
        for entry in self.layer_caches:
            entry.reset_slots(idx)
        for s in slot_ids:
            si = int(s)
            self.slot_pos[si] = 0
            self.slot_valid_len[si] = 0
        self.cur_h[idx] = mx.zeros((len(slot_ids), int(self.cur_h.shape[1])), dtype=self.cur_h.dtype)

    def restore_slot_from_hybrid_cache(
        self,
        slot_id: int,
        cache_list: Sequence[DeltaSlotCache | GrowingKVCache],
    ) -> int:
        """Copy a `B=1` `[DeltaSlotCache | GrowingKVCache]` into `slot_id` and return restored token count."""
        sid = int(slot_id)
        restored_len = 0
        idx = mx.array([sid], dtype=mx.int32)
        for dst, src in zip(self.layer_caches, cache_list):
            if isinstance(dst, RollingDeltaSlotCache):
                if not isinstance(src, DeltaSlotCache):
                    raise TypeError(f"Expected DeltaSlotCache, got {type(src)}")
                dst.scatter_from_subcache(idx, src)
            elif isinstance(dst, RollingKVCache):
                if not isinstance(src, GrowingKVCache):
                    raise TypeError(f"Expected GrowingKVCache, got {type(src)}")
                off = int(src.offset)
                restored_len = max(restored_len, off)
                if off > 0:
                    if src._qcache is not None:
                        k_dq, v_dq = src._qcache.fetch_dequantized(out_dtype=mx.bfloat16)
                        dst.scatter_prefill(sid, k_dq, v_dq, start_pos=0)
                    elif src.keys is not None and src.values is not None:
                        dst.scatter_prefill(sid, src.keys[:, :, :off, :], src.values[:, :, :off, :], start_pos=0)
        self.slot_pos[sid] = restored_len
        self.slot_valid_len[sid] = restored_len
        return restored_len

    def _prefill_single_chunked(
        self,
        slot_id: int,
        prompt_ids: Sequence[int],
        *,
        start_pos: int,
        prefill_chunk_size: int,
    ) -> mx.array:
        sid = int(slot_id)
        total_len = len(prompt_ids)
        if start_pos >= total_len:
            raise ValueError(f"start_pos={start_pos} must be < len(prompt_ids)={total_len}")
        if total_len > self.cap:
            raise ValueError(f"Prompt length {total_len} exceeds Cap={self.cap}")

        last_h_1d: mx.array | None = None
        slot_idx = mx.array([sid], dtype=mx.int32)
        for c0 in range(int(start_pos), total_len, max(1, int(prefill_chunk_size))):
            c1 = min(total_len, c0 + max(1, int(prefill_chunk_size)))
            chunk_ids = [int(t) for t in prompt_ids[c0:c1]]
            h = self.model.embed_tokens(mx.array([chunk_ids], dtype=mx.int32))
            for layer, entry in zip(self.model.layers, self.layer_caches):
                if layer.is_linear:
                    assert isinstance(entry, RollingDeltaSlotCache)
                    sub_c = entry.slice_for_slots(slot_idx)
                    h = h + layer.linear_attn(layer.input_layernorm(h), sub_c)
                    entry.scatter_from_subcache(slot_idx, sub_c)
                else:
                    assert isinstance(entry, RollingKVCache)
                    h = h + _full_attn_step(
                        layer,
                        entry,
                        layer.input_layernorm(h),
                        cos_a=None,
                        sin_a=None,
                        rope_offset=c0,
                        slot_ids=[sid],
                        positions=None,
                        valid_lens=None,
                        is_prefill=True,
                        prefill_start=c0,
                        prefill_cur_len=c1,
                    )
                h = h + layer.mlp(layer.post_attention_layernorm(h))

            if c1 == total_len:
                last_h_1d = self.model.norm(h[:, -1:, :])[0, 0]
                mx.eval(last_h_1d, *self.state_arrays())  # guard: allow-eval
            else:
                mx.eval(*self.state_arrays())  # guard: allow-eval

        assert last_h_1d is not None
        self.slot_pos[sid] = total_len
        self.slot_valid_len[sid] = total_len
        self.cur_h[slot_idx] = last_h_1d[None, :].astype(self.cur_h.dtype)
        return last_h_1d

    def _prefill_cohort_same_start(
        self,
        slot_ids: Sequence[int],
        prompts: Sequence[Sequence[int]],
        *,
        start_pos: int,
        prefill_chunk_size: int,
    ) -> mx.array:
        k_slots = len(slot_ids)
        if k_slots == 1:
            h1 = self._prefill_single_chunked(
                slot_ids[0],
                prompts[0],
                start_pos=start_pos,
                prefill_chunk_size=prefill_chunk_size,
            )
            return h1[None, :]

        lens = [len(p) for p in prompts]
        max_len = max(lens)
        if max_len > self.cap:
            raise ValueError(f"Prompt length {max_len} exceeds Cap={self.cap}")
        chunk_sz = max(1, int(prefill_chunk_size))
        captured_h: list[mx.array | None] = [None] * k_slots
        pad_id = int(prompts[0][-1]) if prompts[0] else 0

        for c0 in range(int(start_pos), max_len, chunk_sz):
            c1 = min(max_len, c0 + chunk_sz)
            act_rows = [r for r in range(k_slots) if lens[r] > c0]
            if not act_rows:
                break
            if len(act_rows) == 1:
                r0 = act_rows[0]
                captured_h[r0] = self._prefill_single_chunked(
                    slot_ids[r0],
                    prompts[r0],
                    start_pos=c0,
                    prefill_chunk_size=chunk_sz,
                )
                continue

            act_slots = [int(slot_ids[r]) for r in act_rows]
            act_lens = [lens[r] for r in act_rows]
            s_c = c1 - c0
            min_rem = min(ln - c0 for ln in act_lens)
            chunk_has_pad = c1 > (c0 + min_rem)

            padded_chunk = [
                [int(t) for t in prompts[r][c0 : min(lens[r], c1)]]
                + [pad_id] * max(0, s_c - max(0, min(lens[r], c1) - c0))
                for r in act_rows
            ]
            if chunk_has_pad:
                v_np = np.zeros((len(act_rows), s_c, 1), dtype=np.float32)
                rem_lens: list[int] = []
                for idx_a, ln in enumerate(act_lens):
                    rl = max(0, min(ln - c0, s_c))
                    v_np[idx_a, :rl, 0] = 1.0
                    rem_lens.append(rl)
                v_scale: mx.array | None = mx.array(v_np, dtype=mx.bfloat16)
                v_lens: mx.array | None = mx.array(rem_lens, dtype=mx.int32)
            else:
                v_scale = None
                v_lens = None

            slot_idx = mx.array(act_slots, dtype=mx.int32)
            h = self.model.embed_tokens(mx.array(padded_chunk, dtype=mx.int32))
            for layer, entry in zip(self.model.layers, self.layer_caches):
                if layer.is_linear:
                    assert isinstance(entry, RollingDeltaSlotCache)
                    sub_c = entry.slice_for_slots(slot_idx)
                    h = h + layer.linear_attn(
                        layer.input_layernorm(h),
                        sub_c,
                        valid_token_scale=v_scale,
                        valid_lengths=v_lens,
                    )
                    entry.scatter_from_subcache(slot_idx, sub_c)
                else:
                    assert isinstance(entry, RollingKVCache)
                    h = h + _full_attn_step(
                        layer,
                        entry,
                        layer.input_layernorm(h),
                        cos_a=None,
                        sin_a=None,
                        rope_offset=c0,
                        slot_ids=slot_idx,
                        positions=None,
                        valid_lens=None,
                        is_prefill=True,
                        prefill_start=c0,
                        prefill_cur_len=c1,
                    )
                h = h + layer.mlp(layer.post_attention_layernorm(h))

            h_normed = self.model.norm(h)
            to_eval: list[mx.array] = list(self.state_arrays())
            for r in act_rows:
                last_p = lens[r] - 1
                if c0 <= last_p < c1:
                    row_a = act_rows.index(r)
                    h_r = h_normed[row_a, last_p - c0, :]
                    captured_h[r] = h_r
                    to_eval.append(h_r)
            mx.eval(*to_eval)  # guard: allow-eval

        out_list: list[mx.array] = []
        for r, s in enumerate(slot_ids):
            si = int(s)
            hr = captured_h[r]
            assert hr is not None
            self.slot_pos[si] = lens[r]
            self.slot_valid_len[si] = lens[r]
            out_list.append(hr)
        out_h = mx.stack(out_list, axis=0)
        self.cur_h[mx.array(slot_ids, dtype=mx.int32)] = out_h.astype(self.cur_h.dtype)
        return out_h

    def prefill_slots(
        self,
        slot_ids: Sequence[int],
        prompts: Sequence[Sequence[int]],
        *,
        start_positions: Sequence[int] | None = None,
        prefill_chunk_size: int = 512,
    ) -> mx.array:
        """Prefill one or more newly admitted prompts into `slot_ids` (chunked at `prefill_chunk_size`)."""
        if not slot_ids:
            return mx.zeros((0, int(self.model.cfg.hidden_size)), dtype=mx.bfloat16)
        if len(slot_ids) != len(prompts):
            raise ValueError(f"slot_ids len {len(slot_ids)} != prompts len {len(prompts)}")

        starts = [0] * len(slot_ids) if start_positions is None else [int(x) for x in start_positions]
        cold_slots = [int(s) for s, st in zip(slot_ids, starts) if st == 0]
        if cold_slots:
            self.reset_slots(cold_slots)

        if len(set(starts)) == 1:
            return self._prefill_cohort_same_start(
                slot_ids,
                prompts,
                start_pos=starts[0],
                prefill_chunk_size=prefill_chunk_size,
            )

        by_start: dict[int, list[int]] = defaultdict(list)
        for idx, st in enumerate(starts):
            by_start[st].append(idx)

        out_rows: list[mx.array | None] = [None] * len(slot_ids)
        for st, grp_indices in by_start.items():
            grp_h = self._prefill_cohort_same_start(
                [slot_ids[i] for i in grp_indices],
                [prompts[i] for i in grp_indices],
                start_pos=st,
                prefill_chunk_size=prefill_chunk_size,
            )
            for pos_g, orig_idx in enumerate(grp_indices):
                out_rows[orig_idx] = grp_h[pos_g]
        return mx.stack([r for r in out_rows if r is not None], axis=0)


def forward_hidden_rolling(
    model: FastQwen3Model,
    cache: RollingHybridCache,
    token_ids: mx.array,
    slot_ids: Sequence[int] | mx.array,
    positions: Sequence[int] | mx.array,
    valid_lens: Sequence[int] | mx.array,
    *,
    max_valid: int | None = None,
) -> mx.array:
    """Run 1 batched rolling decode step across active `slot_ids` reusing all `model` weights."""
    toks_2d = token_ids[:, None] if token_ids.ndim == 1 else token_ids
    slot_idx = slot_ids if isinstance(slot_ids, mx.array) else mx.array(slot_ids, dtype=mx.int32)
    pos_arr = positions if isinstance(positions, mx.array) else mx.array(positions, dtype=mx.int32)
    valid_arr = valid_lens if isinstance(valid_lens, mx.array) else mx.array(valid_lens, dtype=mx.int32)

    if max_valid is None:
        if isinstance(valid_lens, mx.array):
            eff_max_valid = int(mx.max(valid_arr).item()) if int(valid_arr.size) > 0 else 0
        else:
            eff_max_valid = max((int(v) for v in valid_lens), default=0)
    else:
        eff_max_valid = int(max_valid)

    h = model.embed_tokens(toks_2d)
    cos_a, sin_a = model.precompute_rope_trig(pos_arr[:, None])

    b_act = int(toks_2d.shape[0])
    env_metal = _use_gdn_metal_override()
    use_slot_gdn = bool(
        mx.metal.is_available()
        and mx.default_device().type == mx.DeviceType.gpu
        and int(toks_2d.shape[1]) == 1
        and b_act > 0
    )
    active_list = (
        [int(x) for x in slot_ids]
        if not isinstance(slot_ids, mx.array)
        else [int(x) for x in slot_idx.tolist()]
    )

    for layer, entry in zip(model.layers, cache.layer_caches):
        if layer.is_linear:
            assert isinstance(entry, RollingDeltaSlotCache)
            la = layer.linear_attn
            use_gdn_kernel = (
                entry.state_mode == "int8_blk32"
                or (entry.state_mode == "bf16" and env_metal is not False)
                or (env_metal is True)
            )
            can_fused_gdn = bool(
                use_slot_gdn
                and use_gdn_kernel
                and entry.conv_depth == 3
                and la.conv_k == 4
                and la.head_k_dim == la.head_v_dim
                and la.head_k_dim >= 32
                and (la.head_k_dim % 32) == 0
                and h.dtype in (mx.bfloat16, mx.float16)
            )
            if can_fused_gdn:
                is_contig_full = (
                    b_act == entry.b_slots and active_list == list(range(entry.b_slots))
                )
                ins = la.in_group(layer.input_layernorm(h))
                qkv = ins["qkv"]
                z = ins["z"].reshape(b_act, 1, la.num_v_heads, la.head_v_dim)
                b_gate = ins["b"]
                a_gate = ins["a"]
                cw_raw = la.conv1d.weight
                cached_id = getattr(la, "_conv_w_2d_src_id", None)
                cw_2d = getattr(la, "_conv_w_2d_cached", None)
                if cw_2d is None or cached_id != id(cw_raw):
                    cw_2d = mx.contiguous(cw_raw[:, :, 0])
                    la._conv_w_2d_cached = cw_2d
                    la._conv_w_2d_src_id = id(cw_raw)

                q_g, k_g, v_g, g_g, beta_g, cs_new = fused_gdn_pre_recurrence(
                    qkv,
                    entry.conv_state,
                    slot_idx,
                    cw_2d,
                    a_gate,
                    b_gate,
                    la.A_log,
                    la.dt_bias,
                    hk=la.num_k_heads,
                    hv=la.num_v_heads,
                    head_dim=la.head_k_dim,
                    inv_scale=la._inv_scale,
                    rms_eps=la._rms_eps,
                )
                y_g, rec_new = gdn_delta_update_slot_indexed(
                    q_g,
                    k_g,
                    v_g,
                    g_g,
                    beta_g,
                    slot_idx,
                    codes_pool=entry.rec_codes,
                    scales_pool=entry.rec_scales,
                    state_pool=entry.rec_state,
                    state_mode=entry.state_mode,
                )
                if is_contig_full:
                    entry.conv_state = cs_new.astype(entry.conv_state.dtype)
                    if entry.state_mode == "int8_blk32":
                        assert isinstance(rec_new, tuple)
                        entry.rec_codes, entry.rec_scales = rec_new
                    else:
                        assert not isinstance(rec_new, tuple)
                        entry.rec_state = rec_new
                else:
                    entry.conv_state[slot_idx] = cs_new.astype(entry.conv_state.dtype)
                    if entry.state_mode == "int8_blk32":
                        assert isinstance(rec_new, tuple)
                        assert entry.rec_codes is not None and entry.rec_scales is not None
                        entry.rec_codes[slot_idx] = rec_new[0]
                        entry.rec_scales[slot_idx] = rec_new[1]
                    else:
                        assert not isinstance(rec_new, tuple)
                        assert entry.rec_state is not None
                        entry.rec_state[slot_idx] = rec_new.astype(entry.rec_state.dtype)

                normed = la.norm(y_g, z).reshape(b_act, 1, la.value_dim)
                delta = la.out_proj(normed)
                if la._out_lora is not None:
                    la_t, lb_t, sc = la._out_lora
                    delta = delta + (
                        mx.matmul(mx.matmul(normed.astype(mx.bfloat16), la_t), lb_t) * sc
                    ).astype(delta.dtype)
                h = h + delta
            else:
                sub_c = entry.slice_for_slots(slot_idx)
                h = h + layer.linear_attn(layer.input_layernorm(h), sub_c)
                entry.scatter_from_subcache(slot_idx, sub_c)
        else:
            assert isinstance(entry, RollingKVCache)
            h = h + _full_attn_step(
                layer,
                entry,
                layer.input_layernorm(h),
                cos_a=cos_a,
                sin_a=sin_a,
                rope_offset=0,
                slot_ids=slot_idx,
                positions=pos_arr,
                valid_lens=valid_arr,
                is_prefill=False,
                max_valid=eff_max_valid,
            )
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    return model.norm(h)


def _fold_upfront_ff(
    prompt_ids: Sequence[int],
    gram: GrammarHelper | None,
    stop_token_ids: frozenset[int],
    max_tokens: int,
) -> tuple[Any | None, list[int], list[int], bool, bool]:
    """Fold deterministic `upfront_ff` into prefill; return `(matcher, prefill_ids, ans_ids, done, hit_max)`."""
    matcher = gram.new_matcher() if gram is not None else None
    up_clean: list[int] = []
    hit_stop_up = False
    if matcher is not None and gram is not None and gram.upfront_ff:
        for t in gram.upfront_ff:
            ti = int(t)
            if ti in stop_token_ids:
                hit_stop_up = True
                break
            up_clean.append(ti)

    if len(up_clean) >= max_tokens:
        cap_ff = up_clean[:max_tokens]
        is_done_ok = len(up_clean) == max_tokens and (
            hit_stop_up or (matcher is not None and matcher.is_stopped() and matcher.is_accepting())
        )
        return matcher, list(prompt_ids), cap_ff, True, (not is_done_ok)

    if hit_stop_up or (matcher is not None and matcher.is_stopped()):
        return matcher, list(prompt_ids), up_clean, True, False

    return matcher, list(prompt_ids) + up_clean, list(up_clean), False, False


def run_qwen3_decode_rolling(
    model: FastQwen3Model,
    tokenizer: Any,
    encoded_prompts: Sequence[Sequence[int]],
    grammars: Sequence[GrammarHelper | None],
    schema_objs: Sequence[dict[str, Any] | None],
    finalize_item_fn: Callable[..., Any],
    select_tokens_fn: Callable[..., tuple[dict[int, int], dict[int, dict[str, float] | None], int]],
    B_slots: int = 16,
    Cap: int | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    slot_seeds: Sequence[int | None] | None = None,
    gdn_state_mode: str = "auto",
    collect_telemetry: bool = False,
    prefill_chunk_size: int = 512,
    prefix_restore_fn: Callable[..., int] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Continuous-batching rolling slot decoder with length-sorted admission, FF folding, and exact prompt order."""
    n_docs = len(encoded_prompts)
    if n_docs == 0:
        return [], {
            "b_slots": 0,
            "cap": 0,
            "gdn_state_mode": gdn_state_mode,
            "prefill_sec": 0.0,
            "decode_sec": 0.0,
            "llm_steps": 0,
            "refills": 0,
            "ff_tokens": 0,
            "mask_fallbacks": 0,
            "loop_aborts": 0,
        }
    if len(grammars) != n_docs or len(schema_objs) != n_docs:
        raise ValueError(
            f"Mismatched lengths: prompts={n_docs}, grammars={len(grammars)}, schema_objs={len(schema_objs)}"
        )

    stop_token_ids: frozenset[int] = frozenset(int(t) for t in tokenizer.stop_token_ids)
    eff_max_tokens = max(0, int(max_tokens))

    results: list[Any | None] = [None] * n_docs
    matchers: list[Any | None] = [None] * n_docs
    prefill_ids_by_doc: list[list[int]] = [[] for _ in range(n_docs)]
    ans_ids: list[list[int]] = [[] for _ in range(n_docs)]
    tels: list[list[dict[str, float]] | None] = [
        ([] if collect_telemetry else None) for _ in range(n_docs)
    ]
    hit_max: list[bool] = [False] * n_docs
    rngs: list[np.random.Generator | None] = [
        np.random.default_rng(int(slot_seeds[i])) if (slot_seeds is not None and slot_seeds[i] is not None) else None
        for i in range(n_docs)
    ]

    ff_tokens_count = 0
    pending: list[int] = []
    for d in range(n_docs):
        if eff_max_tokens <= 0:
            m0 = grammars[d].new_matcher() if grammars[d] is not None else None
            done_ok = bool(m0 is not None and m0.is_stopped() and m0.is_accepting())
            results[d] = finalize_item_fn(
                prompt_tokens=len(encoded_prompts[d]),
                answer_ids=[],
                reasoning_text=None,
                hit_max_tokens=(not done_ok if grammars[d] is not None else False),
                schema_obj=schema_objs[d],
                telemetry=tels[d],
            )
            continue
        m_d, pf_d, ans_d, done_d, hm_d = _fold_upfront_ff(
            encoded_prompts[d], grammars[d], stop_token_ids, eff_max_tokens
        )
        matchers[d] = m_d
        prefill_ids_by_doc[d] = pf_d
        ans_ids[d] = ans_d
        hit_max[d] = hm_d
        ff_tokens_count += len(ans_d)
        if done_d:
            results[d] = finalize_item_fn(
                prompt_tokens=len(encoded_prompts[d]),
                answer_ids=ans_d,
                reasoning_text=None,
                hit_max_tokens=hm_d,
                schema_obj=schema_objs[d],
                telemetry=tels[d],
            )
        else:
            pending.append(d)

    if not pending:
        return [r for r in results if r is not None], {
            "b_slots": 0,
            "cap": 0,
            "gdn_state_mode": gdn_state_mode,
            "prefill_sec": 0.0,
            "decode_sec": 0.0,
            "llm_steps": 0,
            "refills": 0,
            "ff_tokens": ff_tokens_count,
            "mask_fallbacks": 0,
            "loop_aborts": 0,
        }

    # Sort pending admission queue by input length (longest first for tight cohorts and bounded tail).
    pending.sort(key=lambda d: (-len(prefill_ids_by_doc[d]), d))

    max_pf = max(len(prefill_ids_by_doc[d]) for d in pending)
    eff_cap = max(128, max_pf + eff_max_tokens + 16) if Cap is None else int(Cap)
    if max_pf > eff_cap:
        raise ValueError(f"Max prefill length {max_pf} exceeds Cap={eff_cap}")

    eff_b_slots = max(1, min(int(B_slots), len(pending)))
    rolling_cache = RollingHybridCache(
        model,
        b_slots=eff_b_slots,
        cap=eff_cap,
        gdn_state_mode=gdn_state_mode,
    )

    slot_doc: list[int] = [-1] * eff_b_slots
    slot_queue: list[list[int]] = [[] for _ in range(eff_b_slots)]
    free_slots: list[int] = list(range(eff_b_slots))

    prefill_sec = 0.0
    decode_sec = 0.0
    llm_steps = 0
    refills = 0
    mask_fallbacks = 0
    loop_aborts = 0

    def _decode_fn(ids: Sequence[int]) -> str:
        return str(tokenizer.decode(list(ids), skip_special_tokens=False))

    def _finish_slot(slot_id: int, *, hit_max_flag: bool) -> None:
        d = slot_doc[slot_id]
        results[d] = finalize_item_fn(
            prompt_tokens=len(encoded_prompts[d]),
            answer_ids=ans_ids[d],
            reasoning_text=None,
            hit_max_tokens=hit_max_flag,
            schema_obj=schema_objs[d],
            telemetry=tels[d],
        )
        slot_doc[slot_id] = -1
        slot_queue[slot_id].clear()
        free_slots.append(slot_id)

    def _consume_picked_token(slot_id: int, tid: int, tel: dict[str, float] | None) -> None:
        nonlocal ff_tokens_count, loop_aborts
        d = slot_doc[slot_id]
        m = matchers[d]
        if tel is not None and tels[d] is not None:
            tels[d].append(tel)  # type: ignore[union-attr]
        if tid in stop_token_ids:
            _finish_slot(slot_id, hit_max_flag=False)
            return

        ans_ids[d].append(tid)
        if len(ans_ids[d]) >= 16 and (len(ans_ids[d]) % 8 == 0):
            if detect_catastrophic_token_loop(ans_ids[d], decode_fn=_decode_fn) is not None:
                loop_aborts += 1
                _finish_slot(slot_id, hit_max_flag=False)
                return

        if m is not None:
            m.consume_token(tid)
            if m.is_stopped():
                is_ok = bool(m.is_accepting())
                hm = bool(len(ans_ids[d]) >= eff_max_tokens and not is_ok)
                _finish_slot(slot_id, hit_max_flag=hm)
                return
            if len(ans_ids[d]) >= eff_max_tokens:
                _finish_slot(slot_id, hit_max_flag=True)
                return
            ff_clean: list[int] = []
            stopped_by_ff = False
            for ft in m.compute_ff_tokens():
                fti = int(ft)
                if fti in stop_token_ids:
                    stopped_by_ff = True
                    break
                if len(ans_ids[d]) + len(ff_clean) >= eff_max_tokens:
                    break
                ff_clean.append(fti)
            if ff_clean:
                m.consume_tokens(ff_clean)
                ans_ids[d].extend(ff_clean)
                ff_tokens_count += len(ff_clean)
            if stopped_by_ff or m.is_stopped():
                is_ok = bool(m.is_accepting())
                hm = bool(len(ans_ids[d]) >= eff_max_tokens and not is_ok)
                _finish_slot(slot_id, hit_max_flag=hm)
                return
            if len(ans_ids[d]) >= eff_max_tokens:
                _finish_slot(slot_id, hit_max_flag=True)
                return
            slot_queue[slot_id] = [tid] + ff_clean
            return

        if len(ans_ids[d]) >= eff_max_tokens:
            _finish_slot(slot_id, hit_max_flag=True)
            return
        slot_queue[slot_id] = [tid]

    def _admit_docs(doc_ids: list[int], target_slots: list[int]) -> None:
        nonlocal prefill_sec, decode_sec, llm_steps, mask_fallbacks
        tp0 = time.perf_counter()
        starts: list[int] | None = None
        if prefix_restore_fn is not None:
            starts = []
            for d_i, s_i in zip(doc_ids, target_slots):
                st = int(prefix_restore_fn(prefill_ids_by_doc[d_i], rolling_cache, s_i))
                if st >= len(prefill_ids_by_doc[d_i]):
                    st = max(0, len(prefill_ids_by_doc[d_i]) - 1)
                starts.append(st)

        for d_i, s_i in zip(doc_ids, target_slots):
            slot_doc[s_i] = d_i
            slot_queue[s_i].clear()

        h_admit = rolling_cache.prefill_slots(
            target_slots,
            [prefill_ids_by_doc[d_i] for d_i in doc_ids],
            start_positions=starts,
            prefill_chunk_size=prefill_chunk_size,
        )
        prefill_sec += time.perf_counter() - tp0
        llm_steps += 1

        td0 = time.perf_counter()
        k_adm = len(target_slots)
        tids, tels_adm, fb = select_tokens_fn(
            h_admit,
            list(range(k_adm)),
            k_adm,
            matchers=[matchers[d_i] for d_i in doc_ids],
            grammars=[grammars[d_i] for d_i in doc_ids],
            recent_tokens_by_row=[ans_ids[d_i] for d_i in doc_ids],
            rngs=[rngs[d_i] for d_i in doc_ids],
            temperature=float(temperature),
            top_p=float(top_p),
            repeat_penalty=True,
            collect_telemetry=collect_telemetry,
        )
        mask_fallbacks += int(fb)
        for r, s_i in enumerate(target_slots):
            _consume_picked_token(s_i, int(tids[r]), tels_adm[r])
        decode_sec += time.perf_counter() - td0

    # Initial wave fill
    init_k = min(len(free_slots), len(pending))
    init_docs = pending[:init_k]
    del pending[:init_k]
    init_slots = free_slots[:init_k]
    del free_slots[:init_k]
    _admit_docs(init_docs, init_slots)

    has_async_eval = callable(getattr(mx, "async_eval", None))

    while True:
        active_slots = [s for s in range(eff_b_slots) if slot_doc[s] >= 0]
        if pending:
            refill_min = 1
            if len(free_slots) >= refill_min or not active_slots:
                k_refill = min(len(free_slots), len(pending))
                r_docs = pending[:k_refill]
                del pending[:k_refill]
                r_slots = free_slots[:k_refill]
                del free_slots[:k_refill]
                _admit_docs(r_docs, r_slots)
                refills += 1
                active_slots = [s for s in range(eff_b_slots) if slot_doc[s] >= 0]

        if not active_slots:
            if not pending:
                break
            continue

        td0 = time.perf_counter()
        step_toks: list[int] = []
        step_pos: list[int] = []
        step_valid: list[int] = []
        need_rows: list[int] = []
        max_v = 0
        for r, s in enumerate(active_slots):
            tok = slot_queue[s].pop(0)
            pos_s = rolling_cache.slot_pos[s]
            vlen_s = rolling_cache.slot_valid_len[s] + 1
            if vlen_s > eff_cap:
                raise ValueError(f"Slot {s} decode length {vlen_s} exceeds Cap={eff_cap}")
            step_toks.append(tok)
            step_pos.append(pos_s)
            step_valid.append(vlen_s)
            rolling_cache.slot_pos[s] = pos_s + 1
            rolling_cache.slot_valid_len[s] = vlen_s
            if vlen_s > max_v:
                max_v = vlen_s
            if not slot_queue[s]:
                need_rows.append(r)

        sh = forward_hidden_rolling(
            model,
            rolling_cache,
            mx.array(step_toks, dtype=mx.int32)[:, None],
            active_slots,
            step_pos,
            step_valid,
            max_valid=max_v,
        )
        llm_steps += 1
        h_2d = sh[:, 0, :]
        rolling_cache.cur_h[mx.array(active_slots, dtype=mx.int32)] = h_2d.astype(rolling_cache.cur_h.dtype)

        if has_async_eval:
            mx.async_eval(*rolling_cache.state_arrays())  # guard: allow-eval

        tids_step, tels_step, step_fb = select_tokens_fn(
            h_2d,
            need_rows,
            len(active_slots),
            matchers=[matchers[slot_doc[s]] for s in active_slots],
            grammars=[grammars[slot_doc[s]] for s in active_slots],
            recent_tokens_by_row=[ans_ids[slot_doc[s]] for s in active_slots],
            rngs=[rngs[slot_doc[s]] for s in active_slots],
            temperature=float(temperature),
            top_p=float(top_p),
            repeat_penalty=True,
            collect_telemetry=collect_telemetry,
        )
        if not has_async_eval:
            mx.eval(*rolling_cache.state_arrays())  # guard: allow-eval

        mask_fallbacks += int(step_fb)
        for r in need_rows:
            _consume_picked_token(active_slots[r], int(tids_step[r]), tels_step[r])
        decode_sec += time.perf_counter() - td0

    final_items = [r for r in results if r is not None]
    stats: dict[str, Any] = {
        "b_slots": eff_b_slots,
        "cap": eff_cap,
        "gdn_state_mode": rolling_cache.gdn_state_mode,
        "prefill_sec": round(prefill_sec, 4),
        "decode_sec": round(decode_sec, 4),
        "llm_steps": llm_steps,
        "refills": refills,
        "ff_tokens": ff_tokens_count,
        "mask_fallbacks": mask_fallbacks,
        "loop_aborts": loop_aborts,
    }
    return final_items, stats


__all__ = [
    "RollingDeltaSlotCache",
    "RollingHybridCache",
    "RollingKVCache",
    "_can_use_flash_decode_metal",
    "attend_rolling_flash_decode",
    "forward_hidden_rolling",
    "run_qwen3_decode_rolling",
]
