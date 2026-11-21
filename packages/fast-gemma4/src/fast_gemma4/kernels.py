"""Custom Metal & Metal 4 MPP BF16 kernels for `fast-gemma4` (parameterized for E2B & E4B)."""

from __future__ import annotations

import mlx.core as mx

from fast_shared.metal_mpp_bf16 import (
    fused_geglu_split,
    make_mpp_bf16_geglu_kernel,
    mpp_fused_geglu_bf16,
    mpp_linear_bf16,
)

MAX_KV_SPLITS = 32

__all__ = [
    "MAX_KV_SPLITS",
    "dequant_kv_pair_i8_bf16",
    "fused_geglu_split",
    "fused_kv_only_norm_rope_ragged",
    "fused_norm_add_norm",
    "fused_ple_geglu_indexed",
    "fused_q_norm_rope_ragged",
    "fused_qkv_packed_norm_rope",
    "fused_qkv_packed_norm_rope_ragged",
    "fused_res_rms_norm",
    "fused_softcap_argmax_bf16",
    "gather_dequant_kv_pair_i8_bf16",
    "mpp_fused_geglu_bf16",
    "mpp_fused_geglu_bf16_v4",
    "mpp_linear_bf16",
    "mqa8_sdpa_prefix_ragged",
    "mqa8_sdpa_prefix_ragged_slot",
    "quantize_kv_pair_i8",
    "scatter_kv_i8_slot_precomputed",
    "scatter_kv_ragged",
    "scatter_kv_ragged_i8_slot",
    "scatter_kv_ragged_slot",
    "scatter_kv_slot_precomputed",
]

_EPS_F32_CACHE: dict[float, mx.array] = {}
_POST_PRE_KERNELS: dict[tuple[mx.Dtype, int, bool, float], object] = {}
_RES_RMS_KERNELS: dict[tuple[mx.Dtype, int, float], object] = {}
_QKV_NORM_ROPE_KERNELS: dict[tuple[mx.Dtype, int, int, int, bool, float], object] = {}
_RAGGED_QKV_KERNELS: dict[tuple[int, int, int, bool], object] = {}
_RAGGED_Q_KERNELS: dict[tuple[int, int], object] = {}
_RAGGED_KV_ONLY_KERNELS: dict[tuple[int, int, bool], object] = {}
_PLE_GEGLU_IDX_KERNELS: dict[int, tuple[object, int]] = {}
_SOFTCAP_ARGMAX_KERNELS: dict[tuple[int, int, float], object] = {}
_SOFTCAP_ARGMAX_TELEM_KERNELS: dict[tuple[int, int, float, bool], object] = {}
_QKV_I8_KERNELS: dict[int, object] = {}
_DEQ_KV_PAIR_KERNELS: dict[tuple[int, int], object] = {}
_GATHER_DEQ_KV_PAIR_KERNELS: dict[tuple[int, int], object] = {}
_PREFIX_RAGGED_SDPA_KERNELS: dict[tuple[int, int, int, bool], tuple[object, int]] = {}
_PREFIX_SLOT_KERNELS: dict[tuple[int, int, int, bool, bool], tuple[object, int]] = {}
_VEC4_REDUCE_KERNELS: dict[tuple[int, int], tuple[object, int]] = {}
_PREFIX_RAGGED_REDUCE_KERNELS: dict[int, object] = {}


def _get_eps_f32(eps: float = 1e-6) -> mx.array:
    arr = _EPS_F32_CACHE.get(eps)
    if arr is None:
        arr = mx.array([eps], dtype=mx.float32)
        mx.eval(arr)
        _EPS_F32_CACHE[eps] = arr
    return arr


def _validate_n_splits(n_splits: int) -> int:
    ns = int(n_splits)
    if ns < 1 or ns > MAX_KV_SPLITS:
        raise ValueError(f"n_splits must be in [1, {MAX_KV_SPLITS}], got {ns}")
    return ns


def fused_norm_add_norm(
    res: mx.array,
    x: mx.array,
    w_post: mx.array,
    w_next: mx.array,
    eps: float = 1e-6,
    layer_scalar: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Single-pass post-RMSNorm + residual add (+ optional layer_scalar) + next pre-RMSNorm."""
    D = int(x.shape[-1])
    if D <= 0 or (x.size > 0 and x.size % D != 0):
        raise ValueError(f"Expected positive last dim, got shape {x.shape}")
    dt = x.dtype
    rows = int(x.size) // D
    if rows == 0:
        return mx.zeros(x.shape, dtype=dt), mx.zeros(x.shape, dtype=dt)
    if D % 256 != 0:
        h_out = res + mx.fast.rms_norm(x, w_post, eps)
        if layer_scalar is not None:
            h_out = h_out * layer_scalar
        return h_out, mx.fast.rms_norm(h_out, w_next, eps)
    has_scalar = layer_scalar is not None
    key = (dt, D, has_scalar, eps)
    kern = _POST_PRE_KERNELS.get(key)
    if kern is None:
        scalar_mul = "h_i *= float(scalar[0]);" if has_scalar else ""
        kern = mx.fast.metal_kernel(
            name=f"g4_fused_norm_add_norm_{D}_{'sc' if has_scalar else 'ns'}",
            input_names=["res", "x", "w_post", "w_next"] + (["scalar"] if has_scalar else []),
            output_names=["h_out", "pre_out"],
            source=f"""
                constexpr uint D = {D};
                constexpr uint EPT = D / 256;
                uint row = threadgroup_position_in_grid.x;
                uint tid = thread_index_in_threadgroup;
                uint sid = tid >> 5;
                uint lid = tid & 31;
                size_t base = (size_t)row * D;

                thread float xv[EPT];
                thread float rv[EPT];
                float sq1 = 0.0f;
                #pragma unroll
                for (uint i = 0; i < EPT; ++i) {{
                    uint col = tid + i * 256;
                    float xi = float(x[base + col]);
                    xv[i] = xi;
                    rv[i] = float(res[base + col]);
                    sq1 += xi * xi;
                }}
                sq1 = simd_sum(sq1);
                threadgroup float sg_sum1[8];
                threadgroup float sg_sum2[8];
                threadgroup float sg_inv1;
                if (lid == 0) sg_sum1[sid] = sq1;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float tot1 = (sid == 0 && lid < 8) ? sg_sum1[lid] : 0.0f;
                tot1 = simd_sum(tot1);
                if (tid == 0) sg_inv1 = rsqrt(tot1 / float(D) + {eps:.8e}f);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float inv_rms1 = sg_inv1;

                thread float hv[EPT];
                float sq2 = 0.0f;
                #pragma unroll
                for (uint i = 0; i < EPT; ++i) {{
                    uint col = tid + i * 256;
                    float h_i = rv[i] + xv[i] * inv_rms1 * float(w_post[col]);
                    {scalar_mul}
                    hv[i] = h_i;
                    h_out[base + col] = static_cast<T>(h_i);
                    sq2 += h_i * h_i;
                }}
                sq2 = simd_sum(sq2);
                if (lid == 0) sg_sum2[sid] = sq2;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float tot2 = (sid == 0 && lid < 8) ? sg_sum2[lid] : 0.0f;
                tot2 = simd_sum(tot2);
                if (tid == 0) sg_sum2[0] = rsqrt(tot2 / float(D) + {eps:.8e}f);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float inv_rms2 = sg_sum2[0];

                #pragma unroll
                for (uint i = 0; i < EPT; ++i) {{
                    uint col = tid + i * 256;
                    pre_out[base + col] = static_cast<T>(hv[i] * inv_rms2 * float(w_next[col]));
                }}
            """,
        )
        _POST_PRE_KERNELS[key] = kern
    sc_1d = layer_scalar.reshape((1,)) if (has_scalar and layer_scalar.ndim == 0) else layer_scalar
    inputs = [res, x, w_post, w_next] + ([sc_1d] if has_scalar else [])
    return kern(
        inputs=inputs,
        template=[("T", dt)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[dt, dt],
    )


def fused_qkv_packed_norm_rope(
    qkv_raw: mx.array,
    w_qn: mx.array,
    w_kn: mx.array,
    inv_freq: mx.array,
    offset: mx.array | int,
    eps: float = 1e-6,
    *,
    n_q: int = 8,
    n_kv: int | None = None,
    k_eq_v: bool | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused Q/K/V RMSNorm + Proportional RoPE on packed `[B, L_q, (n_q + {1,2}*n_kv)*D]` buffer."""
    B, L_q, total_d = qkv_raw.shape
    D = int(w_qn.shape[0])
    n_q_i = int(n_q)
    if n_kv is None:
        rem = int(total_d) - n_q_i * D
        if k_eq_v is True:
            if rem <= 0 or rem % D != 0:
                raise ValueError(f"Cannot infer k_eq_v n_kv from total_d={total_d}, n_q={n_q_i}, D={D}")
            n_kv_i = rem // D
            is_keqv = True
        else:
            if rem <= 0 or rem % (2 * D) != 0:
                raise ValueError(f"Cannot infer n_kv from total_d={total_d}, n_q={n_q_i}, D={D}")
            n_kv_i = rem // (2 * D)
            is_keqv = False
    else:
        n_kv_i = int(n_kv)
        if k_eq_v is None:
            if int(total_d) == (n_q_i + 2 * n_kv_i) * D:
                is_keqv = False
            elif int(total_d) == (n_q_i + n_kv_i) * D:
                is_keqv = True
            else:
                raise ValueError(
                    f"Expected (n_q + 2*n_kv)*D={(n_q_i + 2 * n_kv_i) * D} or (n_q + n_kv)*D={(n_q_i + n_kv_i) * D}, got {total_d}"
                )
        else:
            is_keqv = bool(k_eq_v)
            exp_d = (n_q_i + (n_kv_i if is_keqv else 2 * n_kv_i)) * D
            if int(total_d) != exp_d:
                raise ValueError(f"Expected total_d={exp_d}, got {total_d}")
    n_tot = n_q_i + 2 * n_kv_i
    n_in = n_q_i + (n_kv_i if is_keqv else 2 * n_kv_i)
    dt = qkv_raw.dtype
    key = (dt, D, n_q_i, n_kv_i, is_keqv, eps)
    tpt = D // 2
    nsg = max(1, tpt // 32)
    kern = _QKV_NORM_ROPE_KERNELS.get(key)
    if kern is None:
        keqv_flag = 1 if is_keqv else 0
        kern = mx.fast.metal_kernel(
            name=f"g4_fused_qkv_packed_norm_rope_D{D}_q{n_q_i}_kv{n_kv_i}_keqv{keqv_flag}",
            input_names=["qkv_raw", "w_qn", "w_kn", "inv_freq", "off_lq"],
            output_names=["q_out", "k_out", "v_out"],
            source=f"""
                constexpr uint D = {D};
                constexpr uint HD = D >> 1;
                constexpr uint NSG = {nsg};
                constexpr uint N_Q = {n_q_i};
                constexpr uint N_KV = {n_kv_i};
                constexpr uint N_IN = {n_in};
                constexpr uint K_EQ_V = {keqv_flag};
                uint slot = threadgroup_position_in_grid.x;
                uint b_l = threadgroup_position_in_grid.y;
                uint tid = thread_index_in_threadgroup;
                uint sid = tid >> 5;
                uint lid = tid & 31;

                uint src_slot = (K_EQ_V != 0 && slot >= N_Q + N_KV) ? (slot - N_KV) : slot;
                const device T* src = qkv_raw + ((size_t)b_l * N_IN + src_slot) * D;
                device T* dst;
                const device T* wn = nullptr;
                bool do_rope = true;
                if (slot < N_Q) {{
                    dst = q_out + ((size_t)b_l * N_Q + slot) * D;
                    wn = w_qn;
                }} else if (slot < N_Q + N_KV) {{
                    dst = k_out + ((size_t)b_l * N_KV + (slot - N_Q)) * D;
                    wn = w_kn;
                }} else {{
                    dst = v_out + ((size_t)b_l * N_KV + (slot - N_Q - N_KV)) * D;
                    do_rope = false;
                }}

                float x0 = float(src[tid]);
                float x1 = float(src[tid + HD]);
                float sq = simd_sum(x0 * x0 + x1 * x1);
                threadgroup float sg[NSG];
                if (lid == 0) sg[sid] = sq;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float tot = (sid == 0 && lid < NSG) ? sg[lid] : 0.0f;
                tot = simd_sum(tot);
                if (tid == 0) sg[0] = rsqrt(tot / float(D) + {eps:.8e}f);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float inv_rms = sg[0];

                if (wn != nullptr) {{
                    x0 = x0 * inv_rms * float(wn[tid]);
                    x1 = x1 * inv_rms * float(wn[tid + HD]);
                }} else {{
                    x0 = x0 * inv_rms;
                    x1 = x1 * inv_rms;
                }}

                if (do_rope) {{
                    float pos = float(off_lq[0] + int(b_l % off_lq[1]));
                    float f = float(inv_freq[tid]);
                    if (f != 0.0f) {{
                        float ang = pos * f;
                        float c = metal::cos(ang);
                        float s = metal::sin(ang);
                        float y0 = x0 * c - x1 * s;
                        float y1 = x1 * c + x0 * s;
                        x0 = y0; x1 = y1;
                    }}
                }}
                dst[tid] = static_cast<T>(x0);
                dst[tid + HD] = static_cast<T>(x1);
            """,
        )
        _QKV_NORM_ROPE_KERNELS[key] = kern

    if isinstance(offset, int):
        off_arr = mx.array([offset, L_q], dtype=mx.int32)
    elif offset.size == 2:
        off_arr = offset.astype(mx.int32)
    else:
        off_arr = mx.concatenate([offset.reshape((1,)).astype(mx.int32), mx.array([L_q], dtype=mx.int32)])
    return kern(
        inputs=[qkv_raw, w_qn, w_kn, inv_freq, off_arr],
        template=[("T", dt)],
        grid=(n_tot * tpt, B * L_q, 1),
        threadgroup=(tpt, 1, 1),
        output_shapes=[(B, L_q, n_q_i, D), (B, L_q, n_kv_i, D), (B, L_q, n_kv_i, D)],
        output_dtypes=[dt, dt, dt],
    )


def _get_ragged_qkv_kernel(head_dim: int, n_q: int, n_kv: int, k_eq_v: bool = False):
    key = (head_dim, n_q, n_kv, bool(k_eq_v))
    kern = _RAGGED_QKV_KERNELS.get(key)
    if kern is None:
        half_d = head_dim // 2
        n_in = n_q + (n_kv if k_eq_v else 2 * n_kv)
        keqv_flag = 1 if k_eq_v else 0
        tpt = min(256, head_dim)
        nsg = max(1, tpt // 32)
        src = f"""
    constexpr int D = {head_dim};
    constexpr int HALF_D = {half_d};
    constexpr int TPT = {tpt};
    constexpr int NSG = {nsg};
    constexpr int N_Q = {n_q};
    constexpr int N_KV = {n_kv};
    constexpr int N_IN = {n_in};
    constexpr int K_EQ_V = {keqv_flag};
    int L_q = (int)qkv_shape[1];
    int row = (int)threadgroup_position_in_grid.y;
    int h = (int)threadgroup_position_in_grid.x;
    int tid = (int)thread_position_in_threadgroup.x;
    int b = row / L_q;
    int l = row % L_q;
    int simd_lane = tid & 31;
    int simd_id = tid >> 5;

    threadgroup float tg_w[NSG];
    threadgroup float tg_y[D];

    int src_h = (K_EQ_V != 0 && h >= N_Q + N_KV) ? (h - N_KV) : h;
    const device bfloat* in_ptr = qkv + ((size_t)row * N_IN + src_h) * D;
    float x0 = (float)in_ptr[tid];
    float x1 = (D > TPT) ? (float)in_ptr[tid + TPT] : 0.0f;
    float ss = x0 * x0 + ((D > TPT) ? (x1 * x1) : 0.0f);
    float ws = simd_sum(ss);
    if (simd_lane == 0) tg_w[simd_id] = ws;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        float t = (tid < NSG) ? tg_w[tid] : 0.0f;
        float tot = simd_sum(t);
        if (tid == 0) tg_w[0] = rsqrt(tot / (float)D + eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_w[0];

    if (h >= N_Q + N_KV) {{
        int vh = h - (N_Q + N_KV);
        device bfloat* v_dst = v_out + ((size_t)row * N_KV + vh) * D;
        v_dst[tid] = (bfloat)(x0 * inv_rms);
        if (D > TPT) v_dst[tid + TPT] = (bfloat)(x1 * inv_rms);
        return;
    }}

    const device bfloat* nw = (h < N_Q) ? wq : wk;
    tg_y[tid] = x0 * inv_rms * (float)nw[tid];
    if (D > TPT) tg_y[tid + TPT] = x1 * inv_rms * (float)nw[tid + TPT];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HALF_D) {{
        float y0 = tg_y[tid];
        float y1 = tg_y[tid + HALF_D];
        float freq = inv_freq[tid];
        device bfloat* dst = (h < N_Q)
            ? (q_out + ((size_t)row * N_Q + h) * D)
            : (k_out + ((size_t)row * N_KV + (h - N_Q)) * D);
        if (freq != 0.0f) {{
            float pos = (float)(off_b[b] + l);
            float a = pos * freq;
            float c = metal::cos(a);
            float s = metal::sin(a);
            dst[tid] = (bfloat)(y0 * c - y1 * s);
            dst[tid + HALF_D] = (bfloat)(y1 * c + y0 * s);
        }} else {{
            dst[tid] = (bfloat)y0;
            dst[tid + HALF_D] = (bfloat)y1;
        }}
    }}
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_fused_qkv_ragged_d{head_dim}_q{n_q}_kv{n_kv}_keqv{keqv_flag}",
            input_names=["qkv", "wq", "wk", "inv_freq", "off_b", "eps"],
            output_names=["q_out", "k_out", "v_out"],
            source=src,
        )
        _RAGGED_QKV_KERNELS[key] = (kern, tpt)
    return _RAGGED_QKV_KERNELS[key]


def _get_ragged_q_kernel(head_dim: int, n_q: int):
    key = (head_dim, n_q)
    entry = _RAGGED_Q_KERNELS.get(key)
    if entry is None:
        half_d = head_dim // 2
        tpt = min(256, head_dim)
        nsg = max(1, tpt // 32)
        src = f"""
    constexpr int D = {head_dim};
    constexpr int HALF_D = {half_d};
    constexpr int TPT = {tpt};
    constexpr int NSG = {nsg};
    constexpr int N_Q = {n_q};
    int L_q = (int)q_in_shape[1];
    int row = (int)threadgroup_position_in_grid.y;
    int h = (int)threadgroup_position_in_grid.x;
    int tid = (int)thread_position_in_threadgroup.x;
    int b = row / L_q;
    int l = row % L_q;
    int simd_lane = tid & 31;
    int simd_id = tid >> 5;

    threadgroup float tg_w[NSG];
    threadgroup float tg_y[D];

    const device bfloat* in_ptr = q_in + ((size_t)row * N_Q + h) * D;
    float x0 = (float)in_ptr[tid];
    float x1 = (D > TPT) ? (float)in_ptr[tid + TPT] : 0.0f;
    float ss = x0 * x0 + ((D > TPT) ? (x1 * x1) : 0.0f);
    float ws = simd_sum(ss);
    if (simd_lane == 0) tg_w[simd_id] = ws;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        float t = (tid < NSG) ? tg_w[tid] : 0.0f;
        float tot = simd_sum(t);
        if (tid == 0) tg_w[0] = rsqrt(tot / (float)D + eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_w[0];

    tg_y[tid] = x0 * inv_rms * (float)wq[tid];
    if (D > TPT) tg_y[tid + TPT] = x1 * inv_rms * (float)wq[tid + TPT];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HALF_D) {{
        float y0 = tg_y[tid];
        float y1 = tg_y[tid + HALF_D];
        float freq = inv_freq[tid];
        device bfloat* dst = q_out + ((size_t)row * N_Q + h) * D;
        if (freq != 0.0f) {{
            float pos = (float)(off_b[b] + l);
            float a = pos * freq;
            float c = metal::cos(a);
            float s = metal::sin(a);
            dst[tid] = (bfloat)(y0 * c - y1 * s);
            dst[tid + HALF_D] = (bfloat)(y1 * c + y0 * s);
        }} else {{
            dst[tid] = (bfloat)y0;
            dst[tid + HALF_D] = (bfloat)y1;
        }}
    }}
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_fused_q_ragged_d{head_dim}_q{n_q}",
            input_names=["q_in", "wq", "inv_freq", "off_b", "eps"],
            output_names=["q_out"],
            source=src,
        )
        entry = (kern, tpt)
        _RAGGED_Q_KERNELS[key] = entry
    return entry


def fused_qkv_packed_norm_rope_ragged(
    qkv: mx.array,
    wq: mx.array,
    wk: mx.array,
    inv_freq: mx.array,
    off_b: mx.array,
    eps: float = 1e-6,
    *,
    n_q: int = 8,
    n_kv: int | None = None,
    k_eq_v: bool | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    B, L_q, total_d = qkv.shape
    D = int(wq.shape[0])
    n_q_i = int(n_q)
    if n_kv is None:
        rem = int(total_d) - n_q_i * D
        if k_eq_v is True:
            if rem <= 0 or rem % D != 0:
                raise ValueError(f"Cannot infer k_eq_v n_kv from total_d={total_d}, n_q={n_q_i}, D={D}")
            n_kv_i = rem // D
            is_keqv = True
        else:
            if rem <= 0 or rem % (2 * D) != 0:
                raise ValueError(f"Cannot infer n_kv from total_d={total_d}, n_q={n_q_i}, D={D}")
            n_kv_i = rem // (2 * D)
            is_keqv = False
    else:
        n_kv_i = int(n_kv)
        if k_eq_v is None:
            if int(total_d) == (n_q_i + 2 * n_kv_i) * D:
                is_keqv = False
            elif int(total_d) == (n_q_i + n_kv_i) * D:
                is_keqv = True
            else:
                raise ValueError(
                    f"Expected (n_q + 2*n_kv)*D={(n_q_i + 2 * n_kv_i) * D} or (n_q + n_kv)*D={(n_q_i + n_kv_i) * D}, got {total_d}"
                )
        else:
            is_keqv = bool(k_eq_v)
            exp_d = (n_q_i + (n_kv_i if is_keqv else 2 * n_kv_i)) * D
            if int(total_d) != exp_d:
                raise ValueError(f"Expected total_d={exp_d}, got {total_d}")
    n_tot = n_q_i + 2 * n_kv_i
    kern, tpt = _get_ragged_qkv_kernel(D, n_q_i, n_kv_i, k_eq_v=is_keqv)
    eps_arr = _get_eps_f32(eps)
    q, k, v = kern(
        inputs=[qkv, wq, wk, inv_freq, off_b, eps_arr],
        output_shapes=[(B, L_q, n_q_i, D), (B, L_q, n_kv_i, D), (B, L_q, n_kv_i, D)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
        grid=(n_tot * tpt, B * L_q, 1),
        threadgroup=(tpt, 1, 1),
    )
    return q, k, v


def fused_q_norm_rope_ragged(
    q_in: mx.array,
    wq: mx.array,
    inv_freq: mx.array,
    off_b: mx.array,
    eps: float = 1e-6,
    *,
    n_q: int | None = None,
) -> mx.array:
    B, L_q, total_d = q_in.shape
    D = int(wq.shape[0])
    n_q_i = int(n_q) if n_q is not None else int(total_d) // D
    if n_q_i * D != int(total_d):
        raise ValueError(f"Expected n_q*D={n_q_i * D}, got {total_d}")
    kern, tpt = _get_ragged_q_kernel(D, n_q_i)
    eps_arr = _get_eps_f32(eps)
    return kern(
        inputs=[q_in, wq, inv_freq, off_b, eps_arr],
        output_shapes=[(B, L_q, n_q_i, D)],
        output_dtypes=[mx.bfloat16],
        grid=(n_q_i * tpt, B * L_q, 1),
        threadgroup=(tpt, 1, 1),
    )[0]


def _get_ragged_kv_only_kernel(head_dim: int, n_kv: int, k_eq_v: bool = False):
    key = (head_dim, n_kv, bool(k_eq_v))
    entry = _RAGGED_KV_ONLY_KERNELS.get(key)
    if entry is None:
        half_d = head_dim // 2
        tpt = min(256, head_dim)
        nsg = max(1, tpt // 32)
        n_in = n_kv if k_eq_v else (2 * n_kv)
        keqv_flag = 1 if k_eq_v else 0
        src = f"""
    constexpr int D = {head_dim};
    constexpr int HALF_D = {half_d};
    constexpr int TPT = {tpt};
    constexpr int NSG = {nsg};
    constexpr int N_KV = {n_kv};
    constexpr int N_IN = {n_in};
    constexpr int K_EQ_V = {keqv_flag};
    int L_q = (int)kv_shape[1];
    int row = (int)threadgroup_position_in_grid.y;
    int h = (int)threadgroup_position_in_grid.x;
    int tid = (int)thread_position_in_threadgroup.x;
    int b = row / L_q;
    int l = row % L_q;
    int simd_lane = tid & 31;
    int simd_id = tid >> 5;

    threadgroup float tg_w[NSG];
    threadgroup float tg_y[D];

    int src_h = (K_EQ_V != 0 && h >= N_KV) ? (h - N_KV) : h;
    const device bfloat* in_ptr = kv + ((size_t)row * N_IN + src_h) * D;
    float x0 = (float)in_ptr[tid];
    float x1 = (D > TPT) ? (float)in_ptr[tid + TPT] : 0.0f;
    float ss = x0 * x0 + ((D > TPT) ? (x1 * x1) : 0.0f);
    float ws = simd_sum(ss);
    if (simd_lane == 0) tg_w[simd_id] = ws;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        float t = (tid < NSG) ? tg_w[tid] : 0.0f;
        float tot = simd_sum(t);
        if (tid == 0) tg_w[0] = rsqrt(tot / (float)D + eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_w[0];

    if (h >= N_KV) {{
        int vh = h - N_KV;
        device bfloat* v_dst = v_out + ((size_t)row * N_KV + vh) * D;
        v_dst[tid] = (bfloat)(x0 * inv_rms);
        if (D > TPT) v_dst[tid + TPT] = (bfloat)(x1 * inv_rms);
        return;
    }}

    tg_y[tid] = x0 * inv_rms * (float)wk[tid];
    if (D > TPT) tg_y[tid + TPT] = x1 * inv_rms * (float)wk[tid + TPT];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HALF_D) {{
        float y0 = tg_y[tid];
        float y1 = tg_y[tid + HALF_D];
        float freq = inv_freq[tid];
        device bfloat* dst = k_out + ((size_t)row * N_KV + h) * D;
        if (freq != 0.0f) {{
            float pos = (float)(off_b[b] + l);
            float a = pos * freq;
            float c = metal::cos(a);
            float s = metal::sin(a);
            dst[tid] = (bfloat)(y0 * c - y1 * s);
            dst[tid + HALF_D] = (bfloat)(y1 * c + y0 * s);
        }} else {{
            dst[tid] = (bfloat)y0;
            dst[tid + HALF_D] = (bfloat)y1;
        }}
    }}
        """
        k = mx.fast.metal_kernel(
            name=f"g4_fused_kv_only_ragged_d{head_dim}_kv{n_kv}_keqv{keqv_flag}",
            input_names=["kv", "wk", "inv_freq", "off_b", "eps"],
            output_names=["k_out", "v_out"],
            source=src,
        )
        entry = (k, tpt)
        _RAGGED_KV_ONLY_KERNELS[key] = entry
    return entry


def fused_kv_only_norm_rope_ragged(
    kv: mx.array,
    wk: mx.array,
    inv_freq: mx.array,
    off_b: mx.array,
    eps: float = 1e-6,
    *,
    n_kv: int | None = None,
    k_eq_v: bool | None = None,
) -> tuple[mx.array, mx.array]:
    """Fused K RMSNorm+RoPE and V RMSNorm on `[B, L_q, {1,2}*n_kv*D]` -> `(k_rot, v_norm)` `[B, L_q, n_kv, D]`."""
    B, L_q, total_d = kv.shape
    D = int(wk.shape[0])
    if n_kv is None:
        if k_eq_v is True:
            n_kv_i = int(total_d) // D
            is_keqv = True
        else:
            n_kv_i = int(total_d) // (2 * D)
            is_keqv = False
    else:
        n_kv_i = int(n_kv)
        if k_eq_v is None:
            if int(total_d) == 2 * n_kv_i * D:
                is_keqv = False
            elif int(total_d) == n_kv_i * D:
                is_keqv = True
            else:
                raise ValueError(f"Expected 2*n_kv*D={2 * n_kv_i * D} or n_kv*D={n_kv_i * D}, got {total_d}")
        else:
            is_keqv = bool(k_eq_v)
    exp_d = (n_kv_i if is_keqv else 2 * n_kv_i) * D
    if int(total_d) != exp_d:
        raise ValueError(f"Expected {exp_d}, got {total_d}")
    kern, tpt = _get_ragged_kv_only_kernel(D, n_kv_i, k_eq_v=is_keqv)
    eps_arr = _get_eps_f32(eps)
    k, v = kern(
        inputs=[kv, wk, inv_freq, off_b, eps_arr],
        output_shapes=[(B, L_q, n_kv_i, D), (B, L_q, n_kv_i, D)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
        grid=(2 * n_kv_i * tpt, B * L_q, 1),
        threadgroup=(tpt, 1, 1),
    )
    return k, v


def fused_res_rms_norm(res: mx.array, x: mx.array, w: mx.array, eps: float = 1e-6) -> mx.array:
    """Zero-recompile Metal kernel for `res + rms_norm(x, w, eps)` across dynamic M."""
    D = int(x.shape[-1])
    if D <= 0 or (x.size > 0 and x.size % D != 0):
        raise ValueError(f"Expected positive last dim, got {D}")
    dt = x.dtype
    rows = int(x.size) // D
    if rows == 0:
        return mx.zeros(x.shape, dtype=dt)
    if D % 256 != 0:
        return res + mx.fast.rms_norm(x, w, eps)
    key = (dt, D, eps)
    kern = _RES_RMS_KERNELS.get(key)
    if kern is None:
        ept = D // 256
        src = f"""
    constexpr uint D = {D};
    constexpr uint EPT = {ept};
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint sid = tid >> 5;
    uint lid = tid & 31;
    size_t base = (size_t)row * D;

    thread float xv[EPT];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < EPT; ++i) {{
        uint col = tid + i * 256;
        float xi = float(x[base + col]);
        xv[i] = xi;
        sq += xi * xi;
    }}
    sq = simd_sum(sq);
    threadgroup float sg_sum[8];
    if (lid == 0) sg_sum[sid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = (sid == 0 && lid < 8) ? sg_sum[lid] : 0.0f;
    tot = simd_sum(tot);
    if (tid == 0) sg_sum[0] = rsqrt(tot / float(D) + {eps:.8e}f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = sg_sum[0];

    #pragma unroll
    for (uint i = 0; i < EPT; ++i) {{
        uint col = tid + i * 256;
        float h_i = float(res[base + col]) + xv[i] * inv_rms * float(w[col]);
        out[base + col] = static_cast<T>(h_i);
    }}
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_fused_res_rms_norm_d{D}",
            input_names=["res", "x", "w"],
            output_names=["out"],
            source=src,
        )
        _RES_RMS_KERNELS[key] = kern
    return kern(
        inputs=[res, x, w],
        template=[("T", dt)],
        output_shapes=[x.shape],
        output_dtypes=[dt],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]


def fused_ple_geglu_indexed(gate: mx.array, pli: mx.array, layer_idx_mx: mx.array) -> mx.array:
    """Zero-slice Metal kernel for `gelu_approx(gate) * pli[:, :, layer_idx, :]` with contiguity guards."""
    ple_d = int(gate.shape[-1])
    if ple_d % 4 != 0:
        raise ValueError(f"Expected ple_dim divisible by 4, got {ple_d}")
    entry = _PLE_GEGLU_IDX_KERNELS.get(ple_d)
    if entry is None:
        tg = ple_d // 4
        src = f"""
    constexpr int PD = {ple_d};
    int N_L = (int)pli_shape[2];
    int l_idx = layer_idx[0];
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    const device bfloat4* gp = (const device bfloat4*)(gate + (size_t)row * PD + tid * 4);
    const device bfloat4* pp = (const device bfloat4*)(pli + ((size_t)row * N_L + l_idx) * PD + tid * 4);
    device bfloat4* op = (device bfloat4*)(out + (size_t)row * PD + tid * 4);
    bfloat4 gv = *gp;
    bfloat4 pv = *pp;
    float4 g = float4((float)gv[0], (float)gv[1], (float)gv[2], (float)gv[3]);
    float4 p = float4((float)pv[0], (float)pv[1], (float)pv[2], (float)pv[3]);
    float4 t = metal::precise::tanh(0.7978845608f * g * (1.0f + 0.044715f * g * g));
    float4 r = 0.5f * g * (1.0f + t) * p;
    *op = bfloat4((bfloat)r.x, (bfloat)r.y, (bfloat)r.z, (bfloat)r.w);
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_fused_ple_geglu_idx_d{ple_d}",
            input_names=["gate", "pli", "layer_idx"],
            output_names=["out"],
            source=src,
        )
        entry = (kern, tg)
        _PLE_GEGLU_IDX_KERNELS[ple_d] = entry
    kern, tg = entry
    rows = int(gate.size) // ple_d
    if rows == 0:
        return mx.zeros(gate.shape, dtype=gate.dtype)
    gate_c = mx.contiguous(gate)
    pli_c = mx.contiguous(pli)
    lidx_c = mx.contiguous(layer_idx_mx)
    return kern(
        inputs=[gate_c, pli_c, lidx_c],
        output_shapes=[gate_c.shape],
        output_dtypes=[gate_c.dtype],
        grid=(tg, rows, 1),
        threadgroup=(tg, 1, 1),
    )[0]


mpp_fused_geglu_bf16_v4 = mpp_fused_geglu_bf16


def _get_softcap_argmax_telem_kernel(stride: int, v_lim: int, softcap: float, has_mask: bool):
    key = (stride, v_lim, float(softcap), bool(has_mask))
    kern = _SOFTCAP_ARGMAX_TELEM_KERNELS.get(key)
    if kern is not None:
        return kern
    out_cols = 4 if has_mask else 3
    mask_flag = 1 if has_mask else 0
    mask_ptr_decl = (
        "int mask_words = (int)valid_mask_shape[1];\n    const device uint* mp = valid_mask + (size_t)row * mask_words;"
        if has_mask
        else ""
    )
    mask_loop_code = (
        """
        uint mword = mp[j >> 5];
        if (((mword >> (j & 31)) & 1u) != 0u) {
            if (s_val == 0.0f) {
                m_val = v;
                s_val = 1.0f;
            } else if (v > m_val) {
                s_val = s_val * metal::precise::exp(m_val - v) + 1.0f;
                m_val = v;
            } else {
                s_val += metal::precise::exp(v - m_val);
            }
        }
        """
        if has_mask
        else ""
    )
    mask_simd_code = (
        """
        float omv = simd_shuffle_down(m_val, offset);
        float osv = simd_shuffle_down(s_val, offset);
        if (osv > 0.0f) {
            if (s_val == 0.0f) {
                m_val = omv;
                s_val = osv;
            } else {
                float nmv = max(m_val, omv);
                s_val = s_val * metal::precise::exp(m_val - nmv) + osv * metal::precise::exp(omv - nmv);
                m_val = nmv;
            }
        }
        """
        if has_mask
        else ""
    )
    mask_tg_decl = "threadgroup float tg_mv[8];\n    threadgroup float tg_sv[8];" if has_mask else ""
    mask_tg_store = "tg_mv[sid] = m_val;\n        tg_sv[sid] = s_val;" if has_mask else ""
    mask_final_load = (
        "float fmv = (lid < 8) ? tg_mv[lid] : -INFINITY;\n        float fsv = (lid < 8) ? tg_sv[lid] : 0.0f;"
        if has_mask
        else ""
    )
    mask_final_simd = (
        """
            float omv = simd_shuffle_down(fmv, offset);
            float osv = simd_shuffle_down(fsv, offset);
            if (osv > 0.0f) {
                if (fsv == 0.0f) {
                    fmv = omv;
                    fsv = osv;
                } else {
                    float nmv = max(fmv, omv);
                    fsv = fsv * metal::precise::exp(fmv - nmv) + osv * metal::precise::exp(omv - nmv);
                    fmv = nmv;
                }
            }
        """
        if has_mask
        else ""
    )
    mask_epilogue = (
        "stats[tbase + 3] = (fsv > 0.0f) ? (fmv + metal::precise::log(fsv)) : -INFINITY;"
        if has_mask
        else ""
    )
    src = f"""
    constexpr int STRIDE = {stride};
    constexpr int V = {v_lim};
    constexpr int OUT_COLS = {out_cols};
    constexpr bfloat SC_BF = (bfloat){softcap:.8e}f;

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint sid = tid >> 5;
    uint lid = tid & 31;

    const device bfloat* rp = raw_lg + (size_t)row * STRIDE;
    {mask_ptr_decl}
    float best_v = -INFINITY;
    int best_i = -1;
    float second_v = -INFINITY;
    float m_all = -INFINITY;
    float s_all = 0.0f;
    float w_all = 0.0f;
    float m_val = -INFINITY;
    float s_val = 0.0f;
    bool saw_nan = false;

    for (int j = (int)tid; j < V; j += 256) {{
        float raw_f = (float)rp[j];
        if (isnan(raw_f)) {{
            saw_nan = true;
            continue;
        }}
        bfloat x_div = rp[j] / SC_BF;
        bfloat y_bf = (bfloat)(metal::precise::tanh((float)x_div)) * SC_BF;
        float v = (float)y_bf;

        if (best_i < 0 || v > best_v) {{
            second_v = best_v;
            best_v = v;
            best_i = j;
        }} else if (v > second_v) {{
            second_v = v;
        }}

        if (s_all == 0.0f) {{
            m_all = v;
            s_all = 1.0f;
            w_all = 0.0f;
        }} else if (v > m_all) {{
            float alpha = metal::precise::exp(m_all - v);
            w_all = (w_all + (v - m_all) * s_all) * alpha;
            s_all = s_all * alpha + 1.0f;
            m_all = v;
        }} else {{
            float d = m_all - v;
            float ev = metal::precise::exp(-d);
            s_all += ev;
            w_all += d * ev;
        }}
        {mask_loop_code}
    }}

    saw_nan = simd_any(saw_nan);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {{
        float ov = simd_shuffle_down(best_v, offset);
        int oi = simd_shuffle_down(best_i, offset);
        float os2 = simd_shuffle_down(second_v, offset);
        if (oi >= 0) {{
            if (best_i < 0) {{
                best_v = ov;
                best_i = oi;
                second_v = os2;
            }} else if (ov > best_v || (ov == best_v && oi < best_i)) {{
                second_v = max(best_v, os2);
                best_v = ov;
                best_i = oi;
            }} else {{
                second_v = max(second_v, ov);
            }}
        }}

        float om = simd_shuffle_down(m_all, offset);
        float os = simd_shuffle_down(s_all, offset);
        float ow = simd_shuffle_down(w_all, offset);
        if (os > 0.0f) {{
            if (s_all == 0.0f) {{
                m_all = om;
                s_all = os;
                w_all = ow;
            }} else {{
                float nm = max(m_all, om);
                float aa = metal::precise::exp(m_all - nm);
                float ab = metal::precise::exp(om - nm);
                w_all = (w_all + (nm - m_all) * s_all) * aa + (ow + (nm - om) * os) * ab;
                s_all = s_all * aa + os * ab;
                m_all = nm;
            }}
        }}
        {mask_simd_code}
    }}

    threadgroup float tg_v[8];
    threadgroup int tg_i[8];
    threadgroup float tg_s2[8];
    threadgroup uint tg_nan[8];
    threadgroup float tg_m[8];
    threadgroup float tg_s[8];
    threadgroup float tg_w[8];
    {mask_tg_decl}
    if (lid == 0) {{
        tg_v[sid] = best_v;
        tg_i[sid] = best_i;
        tg_s2[sid] = second_v;
        tg_nan[sid] = saw_nan ? 1u : 0u;
        tg_m[sid] = m_all;
        tg_s[sid] = s_all;
        tg_w[sid] = w_all;
        {mask_tg_store}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sid == 0) {{
        float fv = (lid < 8) ? tg_v[lid] : -INFINITY;
        int fi = (lid < 8) ? tg_i[lid] : -1;
        float fs2 = (lid < 8) ? tg_s2[lid] : -INFINITY;
        bool fnan = (lid < 8) ? (tg_nan[lid] != 0u) : false;
        float fm = (lid < 8) ? tg_m[lid] : -INFINITY;
        float fs = (lid < 8) ? tg_s[lid] : 0.0f;
        float fw = (lid < 8) ? tg_w[lid] : 0.0f;
        {mask_final_load}
        fnan = simd_any(fnan);
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {{
            float ov = simd_shuffle_down(fv, offset);
            int oi = simd_shuffle_down(fi, offset);
            float os2 = simd_shuffle_down(fs2, offset);
            if (oi >= 0) {{
                if (fi < 0) {{
                    fv = ov;
                    fi = oi;
                    fs2 = os2;
                }} else if (ov > fv || (ov == fv && oi < fi)) {{
                    fs2 = max(fv, os2);
                    fv = ov;
                    fi = oi;
                }} else {{
                    fs2 = max(fs2, ov);
                }}
            }}

            float om = simd_shuffle_down(fm, offset);
            float os = simd_shuffle_down(fs, offset);
            float ow = simd_shuffle_down(fw, offset);
            if (os > 0.0f) {{
                if (fs == 0.0f) {{
                    fm = om;
                    fs = os;
                    fw = ow;
                }} else {{
                    float nm = max(fm, om);
                    float aa = metal::precise::exp(fm - nm);
                    float ab = metal::precise::exp(om - nm);
                    fw = (fw + (nm - fm) * fs) * aa + (ow + (nm - om) * os) * ab;
                    fs = fs * aa + os * ab;
                    fm = nm;
                }}
            }}
            {mask_final_simd}
        }}
        if (lid == 0) {{
            size_t tbase = (size_t)row * OUT_COLS;
            if (fnan || fi < 0) {{
                out_idx[row] = -1;
                #pragma unroll
                for (int c = 0; c < OUT_COLS; ++c) {{
                    stats[tbase + c] = NAN;
                }}
            }} else {{
                out_idx[row] = fi;
                float log_s = (fs > 0.0f) ? metal::precise::log(fs) : 0.0f;
                float lse_all = fm + log_s;
                float h_pre = (fs > 0.0f) ? max(0.0f, (fw / fs) + log_s) : 0.0f;
                float margin = (fs2 > -INFINITY) ? max(0.0f, fv - fs2) : 0.0f;
                stats[tbase + 0] = lse_all;
                stats[tbase + 1] = h_pre;
                stats[tbase + 2] = margin;
                {mask_epilogue}
            }}
        }}
    }}
    """
    in_names = ["raw_lg", "valid_mask"] if has_mask else ["raw_lg"]
    kern = mx.fast.metal_kernel(
        name=f"g4_fused_softcap_argmax_telem_s{stride}_v{v_lim}_m{mask_flag}",
        input_names=in_names,
        output_names=["out_idx", "stats"],
        source=src,
    )
    _SOFTCAP_ARGMAX_TELEM_KERNELS[key] = kern
    return kern


def fused_softcap_argmax_bf16(
    raw_lg: mx.array,
    softcap: float = 30.0,
    vocab_limit: int | None = None,
    *,
    collect_telemetry: bool = False,
    valid_mask_u32: mx.array | None = None,
) -> mx.array | tuple[mx.array, mx.array]:
    """Single-pass BF16 softcap + argmax (+ optional zero-copy GPU telemetry) without extra `[B, V]` reads.

    Returns `-1` on any row containing `NaN` (including all-NaN rows) instead of silently emitting `0`.
    When `collect_telemetry=True` (or `valid_mask_u32` is provided), returns `(out_idx, telemetry)` where
    `telemetry` is `float32 [B, 3]` (`[lse_all, H_pre, top2_margin]`) or `[B, 4]` (`[..., lse_valid]`).
    """
    stride = int(raw_lg.shape[-1])
    v_lim = int(vocab_limit) if vocab_limit is not None else stride
    if stride <= 0 or v_lim <= 0 or v_lim > stride:
        raise ValueError(f"Invalid vocab_limit={v_lim} for stride={stride}")
    B = int(raw_lg.size) // stride
    want_telem = bool(collect_telemetry or valid_mask_u32 is not None)
    has_mask = valid_mask_u32 is not None
    out_cols = 4 if has_mask else 3
    if B == 0:
        empty_idx = mx.zeros((0,), dtype=mx.int32)
        if want_telem:
            return empty_idx, mx.zeros((0, out_cols), dtype=mx.float32)
        return empty_idx
    raw_lg = mx.contiguous(raw_lg)
    if want_telem:
        inputs = [raw_lg]
        if valid_mask_u32 is not None:
            if valid_mask_u32.dtype != mx.uint32:
                valid_mask_u32 = valid_mask_u32.astype(mx.uint32)
            mask_2d = mx.contiguous(valid_mask_u32.reshape(B, -1))
            if int(mask_2d.shape[1]) * 32 < v_lim:
                raise ValueError(
                    f"valid_mask_u32 has {int(mask_2d.shape[1]) * 32} bits per row < vocab_limit={v_lim}"
                )
            inputs.append(mask_2d)
        tkern = _get_softcap_argmax_telem_kernel(stride, v_lim, float(softcap), has_mask=has_mask)
        out_idx, stats = tkern(
            inputs=inputs,
            output_shapes=[(B,), (B, out_cols)],
            output_dtypes=[mx.int32, mx.float32],
            grid=(B * 256, 1, 1),
            threadgroup=(256, 1, 1),
        )
        return out_idx, stats

    key = (stride, v_lim, float(softcap))
    kern = _SOFTCAP_ARGMAX_KERNELS.get(key)
    if kern is None:
        src = f"""
    constexpr int STRIDE = {stride};
    constexpr int V = {v_lim};
    constexpr bfloat SC_BF = (bfloat){softcap:.8e}f;

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint sid = tid >> 5;
    uint lid = tid & 31;

    const device bfloat* rp = raw_lg + (size_t)row * STRIDE;
    float best_v = -INFINITY;
    int best_i = -1;
    bool saw_nan = false;

    for (int j = (int)tid; j < V; j += 256) {{
        float raw_f = (float)rp[j];
        if (isnan(raw_f)) {{
            saw_nan = true;
            continue;
        }}
        bfloat x_div = rp[j] / SC_BF;
        bfloat y_bf = (bfloat)(metal::precise::tanh((float)x_div)) * SC_BF;
        float v = (float)y_bf;
        if (v > best_v) {{
            best_v = v;
            best_i = j;
        }}
    }}

    saw_nan = simd_any(saw_nan);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {{
        float ov = simd_shuffle_down(best_v, offset);
        int oi = simd_shuffle_down(best_i, offset);
        if (oi >= 0 && (best_i < 0 || ov > best_v || (ov == best_v && oi < best_i))) {{
            best_v = ov;
            best_i = oi;
        }}
    }}

    threadgroup float tg_v[8];
    threadgroup int tg_i[8];
    threadgroup uint tg_nan[8];
    if (lid == 0) {{
        tg_v[sid] = best_v;
        tg_i[sid] = best_i;
        tg_nan[sid] = saw_nan ? 1u : 0u;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sid == 0) {{
        float fv = (lid < 8) ? tg_v[lid] : -INFINITY;
        int fi = (lid < 8) ? tg_i[lid] : -1;
        bool fnan = (lid < 8) ? (tg_nan[lid] != 0u) : false;
        fnan = simd_any(fnan);
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {{
            float ov = simd_shuffle_down(fv, offset);
            int oi = simd_shuffle_down(fi, offset);
            if (oi >= 0 && (fi < 0 || ov > fv || (ov == fv && oi < fi))) {{
                fv = ov;
                fi = oi;
            }}
        }}
        if (lid == 0) {{
            out_idx[row] = (fnan || fi < 0) ? -1 : fi;
        }}
    }}
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_fused_softcap_argmax_s{stride}_v{v_lim}",
            input_names=["raw_lg"],
            output_names=["out_idx"],
            source=src,
        )
        _SOFTCAP_ARGMAX_KERNELS[key] = kern
    return kern(
        inputs=[raw_lg],
        output_shapes=[(B,)],
        output_dtypes=[mx.int32],
        grid=(B * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]


def scatter_kv_ragged(
    k_doc: mx.array,
    v_doc: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    off_b: mx.array,
    l_sys: int,
    doc_ring_cap: int = 0,
) -> tuple[mx.array, mx.array]:
    """Write newly computed K,V `[B, L_q, n_kv, D]` (or `[B, L_q, D]`) into `[B, T_cap, n_kv, D]` (or `[B, T_cap, D]`)."""
    B, L_q = int(k_new.shape[0]), int(k_new.shape[1])
    T_cap = int(k_doc.shape[1])
    b_idx = mx.arange(B, dtype=mx.int32)
    is_pow2 = (T_cap & (T_cap - 1)) == 0
    if k_doc.ndim == 4 and k_new.ndim == 3:
        k_new = k_new.reshape(B, L_q, 1, int(k_new.shape[2]))
        v_new = v_new.reshape(B, L_q, 1, int(v_new.shape[2]))
    elif k_doc.ndim == 3 and k_new.ndim == 4:
        k_new = k_new.reshape(B, L_q, int(k_new.shape[3]))
        v_new = v_new.reshape(B, L_q, int(v_new.shape[3]))
    if L_q == 1:
        priv = off_b - l_sys
        slot = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
        k_doc[b_idx, slot] = k_new[:, 0]
        v_doc[b_idx, slot] = v_new[:, 0]
    else:
        priv = off_b[:, None] + mx.arange(L_q, dtype=mx.int32)[None, :] - l_sys
        slot = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
        k_doc[b_idx[:, None], slot] = k_new
        v_doc[b_idx[:, None], slot] = v_new
    return k_doc, v_doc


def _build_gqa_sdpa_source(
    *,
    head_dim: int,
    n_q: int,
    n_kv: int,
    use_slot_idx: bool,
    kv_i8: bool,
    single_pass: bool,
) -> tuple[str, int]:
    """Generate parameterized GQA/MQA Metal SDPA source for `D in {256, 512}`, arbitrary `n_q % n_kv == 0`."""
    if n_q <= 0 or n_kv <= 0 or (n_q % n_kv) != 0:
        raise ValueError(f"Expected n_q ({n_q}) divisible by n_kv ({n_kv})")
    if head_dim % 128 != 0:
        raise ValueError(f"Expected head_dim divisible by 128, got {head_dim}")
    dv = head_dim // 128
    gqa = n_q // n_kv
    if head_dim == 256 and (gqa % 4 == 0):
        qh_per_sg = 4
    elif gqa % 2 == 0:
        qh_per_sg = 2
    else:
        qh_per_sg = 1
    tg_y = n_q // qh_per_sg

    slot_expr = "int slot_row = slot_idx[b];" if use_slot_idx else "int slot_row = b;"
    if single_pass:
        split_bounds = """
    int s0 = t_lo;
    int s1 = t_hi;
        """
    else:
        split_bounds = """
    int split_id = (int)threadgroup_position_in_grid.z;
    int tot = t_hi - t_lo;
    int chunk = (tot + n_splits - 1) / n_splits;
    int s0 = t_lo + split_id * chunk;
    int s1 = min(t_hi, s0 + chunk);
        """

    q_decls = ", ".join(f"q{k}[DV]" for k in range(qh_per_sg))
    acc_decls = ", ".join(f"acc{k}[DV]" for k in range(qh_per_sg))
    m_decls = ", ".join(f"m{k} = -1e38f" for k in range(qh_per_sg))
    l_decls = ", ".join(f"l{k} = 0.0f" for k in range(qh_per_sg))

    load_q_lines = []
    for k in range(qh_per_sg):
        load_q_lines.append(
            f"        bfloat4 v{k} = *(const device bfloat4*)(qb + (h0 + {k}) * D + 128 * r);\n"
            f"        q{k}[r] = float4((float)v{k}[0], (float)v{k}[1], (float)v{k}[2], (float)v{k}[3]);\n"
            f"        acc{k}[r] = float4(0.0f);"
        )
    load_q_code = "\n".join(load_q_lines)

    p_init = ", ".join(f"p{k} = 0" for k in range(qh_per_sg))
    p_acc = " ".join(f"p{k} += q{k}[r] * kr;" for k in range(qh_per_sg))

    def _softmax_update(scale_k: bool, scale_v: bool) -> str:
        k_mul = " * ksc" if scale_k else ""
        lines = [f"            float d{k} = simd_sum(p{k}.x + p{k}.y + p{k}.z + p{k}.w){k_mul};" for k in range(qh_per_sg)]
        lines.append("            " + " ".join(f"float nm{k} = max(d{k}, m{k});" for k in range(qh_per_sg)))
        lines.append(
            "            "
            + " ".join(
                f"float a{k} = metal::fast::exp(m{k} - nm{k}), b{k} = metal::fast::exp(d{k} - nm{k});"
                for k in range(qh_per_sg)
            )
        )
        lines.append("            " + " ".join(f"l{k} = l{k} * a{k} + b{k};" for k in range(qh_per_sg)))
        if scale_v:
            lines.append("            " + " ".join(f"float bv{k} = b{k} * vsc;" for k in range(qh_per_sg)))
            val_coeff = "bv"
        else:
            val_coeff = "b"
        lines.append("            #pragma unroll")
        lines.append("            for (int r = 0; r < DV; ++r) {")
        lines.append(
            "                "
            + " ".join(f"acc{k}[r] = acc{k}[r] * a{k} + {val_coeff}{k} * vv_r[r];" for k in range(qh_per_sg))
        )
        lines.append("            }")
        lines.append("            " + " ".join(f"m{k} = nm{k};" for k in range(qh_per_sg)))
        return "\n".join(lines)

    sys_update = _softmax_update(scale_k=False, scale_v=False)
    if kv_i8:
        doc_update = _softmax_update(scale_k=True, scale_v=True)
        doc_block = f"""
    int doc_lo = max(s0, L_sys);
    if (doc_lo < s1) {{
        const device int8_t* kd_b = k_doc + ((size_t)slot_row * T_cap * N_KV + kv_h) * D + lane * 4;
        const device int8_t* vd_b = v_doc + ((size_t)slot_row * T_cap * N_KV + kv_h) * D + lane * 4;
        const device half* ks_d = k_s_doc + (size_t)slot_row * T_cap * N_KV + kv_h;
        const device half* vs_d = v_s_doc + (size_t)slot_row * T_cap * N_KV + kv_h;
        for (int t = doc_lo; t < s1; ++t) {{
            int priv = t - L_sys;
            int slot = (ring_mask >= 0) ? (priv & ring_mask) : (ring ? (priv % T_cap) : priv);
            size_t off = (size_t)slot * (N_KV * D);
            float ksc = (float)ks_d[(size_t)slot * N_KV];
            float vsc = (float)vs_d[(size_t)slot * N_KV];
            float4 {p_init};
            float4 vv_r[DV];
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                vec<int8_t, 4> kv = *(const device vec<int8_t, 4>*)(kd_b + off + 128 * r);
                vec<int8_t, 4> vv = *(const device vec<int8_t, 4>*)(vd_b + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                {p_acc}
            }}
{doc_update}
        }}
    }}
        """
    else:
        doc_update = _softmax_update(scale_k=False, scale_v=False)
        doc_block = f"""
    int doc_lo = max(s0, L_sys);
    if (doc_lo < s1) {{
        const device bfloat* kd_b = k_doc + ((size_t)slot_row * T_cap * N_KV + kv_h) * D + lane * 4;
        const device bfloat* vd_b = v_doc + ((size_t)slot_row * T_cap * N_KV + kv_h) * D + lane * 4;
        for (int t = doc_lo; t < s1; ++t) {{
            int priv = t - L_sys;
            int slot = (ring_mask >= 0) ? (priv & ring_mask) : (ring ? (priv % T_cap) : priv);
            size_t off = (size_t)slot * (N_KV * D);
            float4 {p_init};
            float4 vv_r[DV];
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(kd_b + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vd_b + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                {p_acc}
            }}
{doc_update}
        }}
    }}
        """

    il_lines = " ".join(f"float il{k} = (l{k} > 0.0f) ? (1.0f / l{k}) : 0.0f;" for k in range(qh_per_sg))
    store_o_lines = "\n".join(
        f"        float4 vo{k} = acc{k}[r] * il{k};\n"
        f"        *(device bfloat4*)(od + {k} * D + 128 * r) = bfloat4((bfloat)vo{k}.x, (bfloat)vo{k}.y, (bfloat)vo{k}.z, (bfloat)vo{k}.w);"
        for k in range(qh_per_sg)
    )

    if single_pass:
        epilogue = f"""
    size_t pb = (size_t)row * N_Q + h0;
    {il_lines}
    device bfloat* od = out + pb * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
{store_o_lines}
    }}
        """
    else:
        ml_lines = "\n".join(
            f"        ml_part[(pb + {k}) * 2] = m{k}; ml_part[(pb + {k}) * 2 + 1] = l{k};" for k in range(qh_per_sg)
        )
        epilogue = f"""
    size_t pb = ((size_t)row * n_splits + split_id) * N_Q + h0;
    {il_lines}
    device bfloat* od = o_part + pb * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
{store_o_lines}
    }}
    if (lane == 0) {{
{ml_lines}
    }}
        """

    src = f"""
    constexpr int D = {head_dim};
    constexpr int DV = {dv};
    constexpr int N_Q = {n_q};
    constexpr int N_KV = {n_kv};
    constexpr int GQA = {gqa};
    constexpr int QH_PER_SG = {qh_per_sg};
    int L_q = (int)q_shape[1];
    int L_sys = meta[0];
    int T_cap = meta[1];
    int win = meta[2];
    int ring = meta[3];
    int n_splits = meta[4];
    int ring_mask = meta[5];

    int row = (int)threadgroup_position_in_grid.y;
    int b = row / L_q;
    int l_q = row % L_q;
    {slot_expr}
    int lane = (int)thread_position_in_threadgroup.x;
    int sg = (int)thread_position_in_threadgroup.y;

    int q_pos = off_b[b] + l_q;
    int t_hi = q_pos + 1;
    int t_lo = (win > 0 && q_pos - win + 1 > 0) ? (q_pos - win + 1) : 0;
{split_bounds}

    int h0 = sg * QH_PER_SG;
    int kv_h = h0 / GQA;
    float4 {q_decls};
    float4 {acc_decls};
    const device bfloat* qb = q + (size_t)row * N_Q * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
{load_q_code}
    }}
    float {m_decls};
    float {l_decls};

    int sys_end = min(s1, L_sys);
    if (s0 < sys_end) {{
        const device bfloat* ks_base = k_sys + (size_t)kv_h * D + lane * 4;
        const device bfloat* vs_base = v_sys + (size_t)kv_h * D + lane * 4;
        for (int t = s0; t < sys_end; ++t) {{
            size_t off = (size_t)t * (N_KV * D);
            float4 {p_init};
            float4 vv_r[DV];
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(ks_base + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vs_base + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                {p_acc}
            }}
{sys_update}
        }}
    }}

{doc_block}

{epilogue}
    """
    return src, tg_y


def _get_prefix_ragged_part_kernel(head_dim: int, n_q: int, n_kv: int, single_pass: bool):
    key = (head_dim, n_q, n_kv, single_pass)
    k = _PREFIX_RAGGED_SDPA_KERNELS.get(key)
    if k is None:
        src, tg_y = _build_gqa_sdpa_source(
            head_dim=head_dim,
            n_q=n_q,
            n_kv=n_kv,
            use_slot_idx=False,
            kv_i8=False,
            single_pass=single_pass,
        )
        out_names = ["out"] if single_pass else ["o_part", "ml_part"]
        mode_tag = "sp" if single_pass else "part"
        kern = mx.fast.metal_kernel(
            name=f"g4_gqa_prefix_{mode_tag}_d{head_dim}_q{n_q}_kv{n_kv}",
            input_names=["q", "k_sys", "v_sys", "k_doc", "v_doc", "off_b", "meta"],
            output_names=out_names,
            source=src,
        )
        k = (kern, tg_y)
        _PREFIX_RAGGED_SDPA_KERNELS[key] = k
    return k


def _get_slot_reduce_kernel_vec4(head_dim: int, n_q: int):
    key = (head_dim, n_q)
    k = _VEC4_REDUCE_KERNELS.get(key)
    if k is None:
        tg = head_dim // 4
        src = f"""
    constexpr int D = {head_dim};
    constexpr int N_Q = {n_q};
    int n_splits = (int)o_part_shape[1];
    int row_h = (int)threadgroup_position_in_grid.y;
    int row = row_h / N_Q;
    int h = row_h % N_Q;
    int tid = (int)thread_position_in_threadgroup.x;

    float g_m = -1e38f;
    for (int s = 0; s < n_splits; ++s) {{
        size_t idx = ((size_t)row * n_splits + s) * N_Q + h;
        float m_s = ml_part[idx * 2 + 0];
        if (ml_part[idx * 2 + 1] > 0.0f && m_s > g_m) g_m = m_s;
    }}
    float g_l = 0.0f;
    float weights[{MAX_KV_SPLITS}];
    for (int s = 0; s < n_splits; ++s) {{
        size_t idx = ((size_t)row * n_splits + s) * N_Q + h;
        float l_s = ml_part[idx * 2 + 1];
        float w = (l_s > 0.0f) ? (l_s * metal::fast::exp(ml_part[idx * 2 + 0] - g_m)) : 0.0f;
        weights[s] = w;
        g_l += w;
    }}
    float inv_gl = (g_l > 0.0f) ? (1.0f / g_l) : 0.0f;
    int d4 = tid * 4;
    float4 acc = float4(0.0f);
    for (int s = 0; s < n_splits; ++s) {{
        size_t idx = ((size_t)row * n_splits + s) * N_Q + h;
        bfloat4 pv = *(const device bfloat4*)(o_part + idx * D + d4);
        acc += weights[s] * float4((float)pv[0], (float)pv[1], (float)pv[2], (float)pv[3]);
    }}
    float4 res = acc * inv_gl;
    *(device bfloat4*)(out + (size_t)row_h * D + d4) = bfloat4((bfloat)res.x, (bfloat)res.y, (bfloat)res.z, (bfloat)res.w);
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_gqa_prefix_reduce_vec4_d{head_dim}_q{n_q}",
            input_names=["o_part", "ml_part"],
            output_names=["out"],
            source=src,
        )
        k = (kern, tg)
        _VEC4_REDUCE_KERNELS[key] = k
    return k


def _get_prefix_ragged_reduce_kernel(n_q: int):
    kern = _PREFIX_RAGGED_REDUCE_KERNELS.get(n_q)
    if kern is None:
        src = f"""
    constexpr int N_Q = {n_q};
    int D = (int)o_part_shape[3];
    int n_splits = (int)o_part_shape[1];
    int row_h = (int)threadgroup_position_in_grid.y;
    int row = row_h / N_Q;
    int h = row_h % N_Q;
    int tid = (int)thread_position_in_threadgroup.x;

    float g_m = -1e38f;
    for (int s = 0; s < n_splits; ++s) {{
        size_t idx = ((size_t)row * n_splits + s) * N_Q + h;
        float m_s = ml_part[idx * 2 + 0];
        if (ml_part[idx * 2 + 1] > 0.0f && m_s > g_m) g_m = m_s;
    }}
    float g_l = 0.0f;
    float weights[{MAX_KV_SPLITS}];
    for (int s = 0; s < n_splits; ++s) {{
        size_t idx = ((size_t)row * n_splits + s) * N_Q + h;
        float l_s = ml_part[idx * 2 + 1];
        float w = (l_s > 0.0f) ? (l_s * metal::fast::exp(ml_part[idx * 2 + 0] - g_m)) : 0.0f;
        weights[s] = w;
        g_l += w;
    }}
    float inv_gl = (g_l > 0.0f) ? (1.0f / g_l) : 0.0f;
    device bfloat* dst = out + (size_t)row_h * D;
    for (int d = tid; d < D; d += 256) {{
        float acc = 0.0f;
        for (int s = 0; s < n_splits; ++s) {{
            size_t idx = ((size_t)row * n_splits + s) * N_Q + h;
            acc += weights[s] * (float)o_part[idx * D + d];
        }}
        dst[d] = (bfloat)(acc * inv_gl);
    }}
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_gqa_prefix_reduce_bf16_q{n_q}",
            input_names=["o_part", "ml_part"],
            output_names=["out"],
            source=src,
        )
        _PREFIX_RAGGED_REDUCE_KERNELS[n_q] = kern
    return kern


def _reduce_split_kv(
    o_part: mx.array,
    ml_part: mx.array,
    B: int,
    L_q: int,
    D: int,
    vec4_reduce: bool,
    n_q: int = 8,
) -> mx.array:
    n_splits = _validate_n_splits(int(o_part.shape[1]))
    if n_splits == 1:
        return o_part.reshape(B, L_q, n_q * D)
    rows = B * L_q
    if vec4_reduce:
        red_k, red_tg = _get_slot_reduce_kernel_vec4(D, n_q)
        return red_k(
            inputs=[o_part, ml_part],
            output_shapes=[(B, L_q, n_q * D)],
            output_dtypes=[mx.bfloat16],
            grid=(red_tg, rows * n_q, 1),
            threadgroup=(red_tg, 1, 1),
        )[0]
    red_k = _get_prefix_ragged_reduce_kernel(n_q)
    return red_k(
        inputs=[o_part, ml_part],
        output_shapes=[(B, L_q, n_q * D)],
        output_dtypes=[mx.bfloat16],
        grid=(256, rows * n_q, 1),
        threadgroup=(256, 1, 1),
    )[0]


def mqa8_sdpa_prefix_ragged(
    q: mx.array,
    k_sys: mx.array,
    v_sys: mx.array,
    k_doc: mx.array,
    v_doc: mx.array,
    off_b: mx.array,
    window_size: int = 0,
    doc_ring_cap: int = 0,
    n_splits: int = 0,
    meta: mx.array | None = None,
    vec4_reduce: bool = True,
) -> mx.array:
    """GQA/MQA SDPA over shared `KV_sys [L_sys, n_kv, D]` + ragged per-doc `KV_doc [B, T_cap, n_kv, D]`."""
    B, L_q, n_q, D = (int(x) for x in q.shape)
    L_sys = int(k_sys.shape[0])
    T_cap = int(k_doc.shape[1])
    n_kv = int(k_doc.shape[2]) if k_doc.ndim == 4 else 1
    ring = 1 if doc_ring_cap > 0 else 0
    ring_mask = (T_cap - 1) if (ring and (T_cap & (T_cap - 1)) == 0) else -1
    if n_splits <= 0:
        rows = B * L_q
        if rows >= 128:
            n_splits = 1
        elif window_size > 0 and window_size <= 512:
            n_splits = 2 if rows >= 96 else 4
        else:
            n_splits = 4 if rows >= 96 else 8
    n_splits = _validate_n_splits(n_splits)
    if meta is None:
        meta = mx.array([L_sys, T_cap, window_size, ring, n_splits, ring_mask], dtype=mx.int32)
    single_pass = n_splits == 1
    part_k, tg_y = _get_prefix_ragged_part_kernel(D, n_q, n_kv, single_pass=single_pass)
    if single_pass:
        return part_k(
            inputs=[q, k_sys, v_sys, k_doc, v_doc, off_b, meta],
            output_shapes=[(B, L_q, n_q * D)],
            output_dtypes=[mx.bfloat16],
            grid=(32, B * L_q * tg_y, 1),
            threadgroup=(32, tg_y, 1),
        )[0]
    o_part, ml_part = part_k(
        inputs=[q, k_sys, v_sys, k_doc, v_doc, off_b, meta],
        output_shapes=[(B * L_q, n_splits, n_q, D), (B * L_q, n_splits, n_q, 2)],
        output_dtypes=[mx.bfloat16, mx.float32],
        grid=(32, B * L_q * tg_y, n_splits),
        threadgroup=(32, tg_y, 1),
    )
    return _reduce_split_kv(o_part, ml_part, B, L_q, D, vec4_reduce=vec4_reduce, n_q=n_q)


def _get_quantize_kv_pair_kernel(head_dim: int):
    k = _QKV_I8_KERNELS.get(head_dim)
    if k is None:
        ept = head_dim // 32
        assert head_dim % 32 == 0 and ept % 4 == 0
        nv = ept // 4
        src = f"""
    constexpr int D = {head_dim};
    constexpr int NV = {nv};
    int row = (int)threadgroup_position_in_grid.x;
    int lane = (int)thread_position_in_threadgroup.x;
    const device bfloat* kp = k_in + (size_t)row * D + lane * 4;
    const device bfloat* vp = v_in + (size_t)row * D + lane * 4;
    float4 kv[NV], vv[NV];
    float kamax = 0.0f, vamax = 0.0f;
    #pragma unroll
    for (int r = 0; r < NV; ++r) {{
        bfloat4 kb = *(const device bfloat4*)(kp + 128 * r);
        bfloat4 vb = *(const device bfloat4*)(vp + 128 * r);
        kv[r] = float4((float)kb[0], (float)kb[1], (float)kb[2], (float)kb[3]);
        vv[r] = float4((float)vb[0], (float)vb[1], (float)vb[2], (float)vb[3]);
        float4 ka = abs(kv[r]);
        float4 va = abs(vv[r]);
        kamax = max(kamax, max(max(ka.x, ka.y), max(ka.z, ka.w)));
        vamax = max(vamax, max(max(va.x, va.y), max(va.z, va.w)));
    }}
    kamax = simd_max(kamax);
    vamax = simd_max(vamax);
    float ks = max(kamax * (1.0f / 127.0f), 1e-6f);
    float vs = max(vamax * (1.0f / 127.0f), 1e-6f);
    float iks = 1.0f / ks;
    float ivs = 1.0f / vs;
    if (lane == 0) {{
        k_s[row] = (half)ks;
        v_s[row] = (half)vs;
    }}
    device int8_t* kqo = k_q + (size_t)row * D + lane * 4;
    device int8_t* vqo = v_q + (size_t)row * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < NV; ++r) {{
        float4 kqf = clamp(rint(kv[r] * iks), -127.0f, 127.0f);
        float4 vqf = clamp(rint(vv[r] * ivs), -127.0f, 127.0f);
        *(device vec<int8_t, 4>*)(kqo + 128 * r) = vec<int8_t, 4>((int8_t)kqf.x, (int8_t)kqf.y, (int8_t)kqf.z, (int8_t)kqf.w);
        *(device vec<int8_t, 4>*)(vqo + 128 * r) = vec<int8_t, 4>((int8_t)vqf.x, (int8_t)vqf.y, (int8_t)vqf.z, (int8_t)vqf.w);
    }}
        """
        k = mx.fast.metal_kernel(
            name=f"g4_quantize_kv_pair_i8_d{head_dim}",
            input_names=["k_in", "v_in"],
            output_names=["k_q", "k_s", "v_q", "v_s"],
            source=src,
        )
        _QKV_I8_KERNELS[head_dim] = k
    return k


def quantize_kv_pair_i8(k_new: mx.array, v_new: mx.array):
    """Quantize K, V `[..., D]` bfloat16 -> `(k_q int8, k_s fp16, v_q int8, v_s fp16)`."""
    D = int(k_new.shape[-1])
    rows = int(k_new.size) // D
    kern = _get_quantize_kv_pair_kernel(D)
    s_shape = k_new.shape[:-1]
    return kern(
        inputs=[k_new, v_new],
        output_shapes=[k_new.shape, s_shape, v_new.shape, s_shape],
        output_dtypes=[mx.int8, mx.float16, mx.int8, mx.float16],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
    )


def dequant_kv_pair_i8_bf16(kq: mx.array, ks: mx.array, vq: mx.array, vs: mx.array):
    """Fused zero-temp dequantize of contiguous `(kq, ks, vq, vs)` -> `(kd, vd)` shaped `(Bg, n_kv, Lc, D)` bfloat16."""
    if kq.ndim == 4:
        Bg, Lc, n_kv, D = (int(x) for x in kq.shape)
    else:
        Bg, Lc, D = (int(x) for x in kq.shape)
        n_kv = 1
    tg = D // 4
    key = (D, n_kv)
    kern = _DEQ_KV_PAIR_KERNELS.get(key)
    if kern is None:
        src = f"""
    constexpr uint D = {D};
    constexpr uint N_KV = {n_kv};
    uint Lc = (uint)kq_shape[1];
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint b = row / (N_KV * Lc);
    uint rem = row - b * (N_KV * Lc);
    uint h = rem / Lc;
    uint t = rem - h * Lc;
    size_t src_idx = ((size_t)b * Lc + t) * N_KV + h;
    size_t src_base = src_idx * D + tid * 4;
    size_t dst_base = (size_t)row * D + tid * 4;
    float sk = (float)ks[src_idx];
    float sv = (float)vs[src_idx];
    char4 kv = *((const device char4*)(kq + src_base));
    char4 vv = *((const device char4*)(vq + src_base));
    *((device bfloat4*)(kd + dst_base)) = bfloat4(
        (bfloat)((float)kv[0] * sk), (bfloat)((float)kv[1] * sk),
        (bfloat)((float)kv[2] * sk), (bfloat)((float)kv[3] * sk)
    );
    *((device bfloat4*)(vd + dst_base)) = bfloat4(
        (bfloat)((float)vv[0] * sv), (bfloat)((float)vv[1] * sv),
        (bfloat)((float)vv[2] * sv), (bfloat)((float)vv[3] * sv)
    );
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_dequant_kv_pair_i8_d{D}_kv{n_kv}",
            input_names=["kq", "ks", "vq", "vs"],
            output_names=["kd", "vd"],
            source=src,
        )
        _DEQ_KV_PAIR_KERNELS[key] = kern
    rows = Bg * n_kv * Lc
    return kern(
        inputs=[kq, ks, vq, vs],
        output_shapes=[(Bg, n_kv, Lc, D), (Bg, n_kv, Lc, D)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
        grid=(tg, rows, 1),
        threadgroup=(tg, 1, 1),
    )


def gather_dequant_kv_pair_i8_bf16(
    k_c: mx.array,
    ks_c: mx.array,
    v_c: mx.array,
    vs_c: mx.array,
    b_idx: mx.array,
    p_slots: mx.array,
):
    """Fused 2D slot+pos gather + int8->bfloat16 dequantize -> `(kd, vd)` `(Bg, n_kv, L_kv, D)`."""
    if k_c.ndim == 4:
        n_kv, D = int(k_c.shape[2]), int(k_c.shape[3])
    else:
        n_kv, D = 1, int(k_c.shape[2])
    Bg = int(b_idx.shape[0])
    L_kv = int(p_slots.shape[0])
    tg = D // 4
    key = (D, n_kv)
    kern = _GATHER_DEQ_KV_PAIR_KERNELS.get(key)
    if kern is None:
        src = f"""
    constexpr uint D = {D};
    constexpr uint N_KV = {n_kv};
    int T_cap = (int)k_c_shape[1];
    int L_kv = (int)p_slots_shape[0];
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint b = row / (N_KV * (uint)L_kv);
    uint rem = row - b * (N_KV * (uint)L_kv);
    uint h = rem / (uint)L_kv;
    uint t = rem - h * (uint)L_kv;
    int slot = b_idx[b];
    int pos = p_slots[t];
    size_t st_idx = ((size_t)slot * (size_t)T_cap + (size_t)pos) * N_KV + (size_t)h;
    size_t src_base = st_idx * D + tid * 4;
    size_t dst_base = (size_t)row * D + tid * 4;
    float sk = (float)ks_c[st_idx];
    float sv = (float)vs_c[st_idx];
    char4 kv = *((const device char4*)(k_c + src_base));
    char4 vv = *((const device char4*)(v_c + src_base));
    *((device bfloat4*)(kd + dst_base)) = bfloat4(
        (bfloat)((float)kv[0] * sk), (bfloat)((float)kv[1] * sk),
        (bfloat)((float)kv[2] * sk), (bfloat)((float)kv[3] * sk)
    );
    *((device bfloat4*)(vd + dst_base)) = bfloat4(
        (bfloat)((float)vv[0] * sv), (bfloat)((float)vv[1] * sv),
        (bfloat)((float)vv[2] * sv), (bfloat)((float)vv[3] * sv)
    );
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_gather_dequant_kv_pair_i8_d{D}_kv{n_kv}",
            input_names=["k_c", "ks_c", "v_c", "vs_c", "b_idx", "p_slots"],
            output_names=["kd", "vd"],
            source=src,
        )
        _GATHER_DEQ_KV_PAIR_KERNELS[key] = kern
    rows = Bg * n_kv * L_kv
    return kern(
        inputs=[k_c, ks_c, v_c, vs_c, b_idx, p_slots],
        output_shapes=[(Bg, n_kv, L_kv, D), (Bg, n_kv, L_kv, D)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
        grid=(tg, rows, 1),
        threadgroup=(tg, 1, 1),
    )


def scatter_kv_slot_precomputed(
    k_doc: mx.array,
    v_doc: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    slot_idx: mx.array,
    pos_slot: mx.array,
) -> None:
    """Zero-overhead BF16 KV scatter using precomputed `slot_idx` and `pos_slot` (`[B, Cap, n_kv, D]` or 3D)."""
    B, L_q = int(k_new.shape[0]), int(k_new.shape[1])
    if k_doc.ndim == 4:
        n_kv, D = int(k_doc.shape[2]), int(k_doc.shape[3])
        if k_new.ndim == 3:
            k_new = k_new.reshape(B, L_q, n_kv, D)
            v_new = v_new.reshape(B, L_q, n_kv, D)
        if L_q == 1:
            k_doc[slot_idx, pos_slot] = k_new.reshape(B, n_kv, D)
            v_doc[slot_idx, pos_slot] = v_new.reshape(B, n_kv, D)
        else:
            s_idx = slot_idx[:, None] if slot_idx.ndim == 1 else slot_idx
            k_doc[s_idx, pos_slot] = k_new
            v_doc[s_idx, pos_slot] = v_new
        return
    D = int(k_doc.shape[2])
    if k_new.ndim == 4:
        k_new = k_new.reshape(B, L_q, D)
        v_new = v_new.reshape(B, L_q, D)
    if L_q == 1:
        k_doc[slot_idx, pos_slot] = k_new.reshape(B, D)
        v_doc[slot_idx, pos_slot] = v_new.reshape(B, D)
    else:
        s_idx = slot_idx[:, None] if slot_idx.ndim == 1 else slot_idx
        k_doc[s_idx, pos_slot] = k_new
        v_doc[s_idx, pos_slot] = v_new


def scatter_kv_i8_slot_precomputed(
    k_q_doc: mx.array,
    k_s_doc: mx.array,
    v_q_doc: mx.array,
    v_s_doc: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    slot_idx: mx.array,
    pos_slot: mx.array,
) -> None:
    """Zero-overhead INT8 KV quantize + scatter using precomputed `slot_idx` and `pos_slot` (`[B, Cap, n_kv, D]` or 3D)."""
    B, L_q = int(k_new.shape[0]), int(k_new.shape[1])
    if k_q_doc.ndim == 4:
        n_kv, D = int(k_q_doc.shape[2]), int(k_q_doc.shape[3])
        if k_new.ndim == 3:
            k_new = k_new.reshape(B, L_q, n_kv, D)
            v_new = v_new.reshape(B, L_q, n_kv, D)
        kq, ks, vq, vs = quantize_kv_pair_i8(k_new, v_new)
        if L_q == 1:
            k_q_doc[slot_idx, pos_slot] = kq.reshape(B, n_kv, D)
            k_s_doc[slot_idx, pos_slot] = ks.reshape(B, n_kv)
            v_q_doc[slot_idx, pos_slot] = vq.reshape(B, n_kv, D)
            v_s_doc[slot_idx, pos_slot] = vs.reshape(B, n_kv)
        else:
            s_idx = slot_idx[:, None] if slot_idx.ndim == 1 else slot_idx
            k_q_doc[s_idx, pos_slot] = kq
            k_s_doc[s_idx, pos_slot] = ks
            v_q_doc[s_idx, pos_slot] = vq
            v_s_doc[s_idx, pos_slot] = vs
        return
    D = int(k_q_doc.shape[2])
    if k_new.ndim == 4:
        k_new = k_new.reshape(B, L_q, D)
        v_new = v_new.reshape(B, L_q, D)
    kq, ks, vq, vs = quantize_kv_pair_i8(k_new, v_new)
    if L_q == 1:
        k_q_doc[slot_idx, pos_slot] = kq.reshape(B, D)
        k_s_doc[slot_idx, pos_slot] = ks.reshape(B)
        v_q_doc[slot_idx, pos_slot] = vq.reshape(B, D)
        v_s_doc[slot_idx, pos_slot] = vs.reshape(B)
    else:
        s_idx = slot_idx[:, None] if slot_idx.ndim == 1 else slot_idx
        k_q_doc[s_idx, pos_slot] = kq
        k_s_doc[s_idx, pos_slot] = ks
        v_q_doc[s_idx, pos_slot] = vq
        v_s_doc[s_idx, pos_slot] = vs


def scatter_kv_ragged_slot(
    k_doc: mx.array,
    v_doc: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    off_b: mx.array,
    l_sys: int,
    doc_ring_cap: int = 0,
    slot_idx: mx.array | None = None,
) -> None:
    B, L_q = int(k_new.shape[0]), int(k_new.shape[1])
    T_cap = int(k_doc.shape[1])
    b_idx = mx.arange(B, dtype=mx.int32) if slot_idx is None else slot_idx
    is_pow2 = (T_cap & (T_cap - 1)) == 0
    if L_q == 1:
        priv = off_b - l_sys
        pos = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
    else:
        priv = off_b[:, None] + mx.arange(L_q, dtype=mx.int32)[None, :] - l_sys
        pos = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
    scatter_kv_slot_precomputed(k_doc, v_doc, k_new, v_new, b_idx, pos)


def scatter_kv_ragged_i8_slot(
    k_q_doc: mx.array,
    k_s_doc: mx.array,
    v_q_doc: mx.array,
    v_s_doc: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    off_b: mx.array,
    l_sys: int,
    doc_ring_cap: int = 0,
    slot_idx: mx.array | None = None,
) -> None:
    B, L_q = int(k_new.shape[0]), int(k_new.shape[1])
    T_cap = int(k_q_doc.shape[1])
    b_idx = mx.arange(B, dtype=mx.int32) if slot_idx is None else slot_idx
    is_pow2 = (T_cap & (T_cap - 1)) == 0
    if L_q == 1:
        priv = off_b - l_sys
        pos = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
    else:
        priv = off_b[:, None] + mx.arange(L_q, dtype=mx.int32)[None, :] - l_sys
        pos = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
    scatter_kv_i8_slot_precomputed(k_q_doc, k_s_doc, v_q_doc, v_s_doc, k_new, v_new, b_idx, pos)


def _get_prefix_slot_part_kernel(head_dim: int, n_q: int, n_kv: int, kv_i8: bool, single_pass: bool):
    key = (head_dim, n_q, n_kv, kv_i8, single_pass)
    k = _PREFIX_SLOT_KERNELS.get(key)
    if k is not None:
        return k

    src, tg_y = _build_gqa_sdpa_source(
        head_dim=head_dim,
        n_q=n_q,
        n_kv=n_kv,
        use_slot_idx=True,
        kv_i8=kv_i8,
        single_pass=single_pass,
    )
    in_names = ["q", "k_sys", "v_sys", "k_doc", "v_doc"]
    if kv_i8:
        in_names += ["k_s_doc", "v_s_doc"]
    in_names += ["off_b", "slot_idx", "meta"]
    out_names = ["out"] if single_pass else ["o_part", "ml_part"]
    mode_tag = "sp" if single_pass else "part"

    kern = mx.fast.metal_kernel(
        name=f"g4_gqa_prefix_slot_{'i8' if kv_i8 else 'bf16'}_{mode_tag}_d{head_dim}_q{n_q}_kv{n_kv}",
        input_names=in_names,
        output_names=out_names,
        source=src,
    )
    k = (kern, tg_y)
    _PREFIX_SLOT_KERNELS[key] = k
    return k


def mqa8_sdpa_prefix_ragged_slot(
    q: mx.array,
    k_sys: mx.array,
    v_sys: mx.array,
    k_doc: mx.array,
    v_doc: mx.array,
    off_b: mx.array,
    slot_idx: mx.array,
    n_splits: int,
    meta: mx.array,
    k_s_doc: mx.array | None = None,
    v_s_doc: mx.array | None = None,
    vec4_reduce: bool = True,
) -> mx.array:
    """Slot-indexed GQA/MQA SDPA over `KV_sys` + `KV_doc` (BF16 or INT8) with `n_splits==1` single-pass fast path."""
    B, L_q, n_q, D = (int(x) for x in q.shape)
    n_splits = _validate_n_splits(n_splits)
    n_kv = int(k_doc.shape[2]) if k_doc.ndim == 4 else 1
    rows = B * L_q
    kv_i8 = k_s_doc is not None
    single_pass = n_splits == 1
    part_k, tg_y = _get_prefix_slot_part_kernel(D, n_q, n_kv, kv_i8=kv_i8, single_pass=single_pass)
    inputs = [q, k_sys, v_sys, k_doc, v_doc]
    if kv_i8:
        inputs += [k_s_doc, v_s_doc]
    inputs += [off_b, slot_idx, meta]

    if single_pass:
        return part_k(
            inputs=inputs,
            output_shapes=[(B, L_q, n_q * D)],
            output_dtypes=[mx.bfloat16],
            grid=(32, rows * tg_y, 1),
            threadgroup=(32, tg_y, 1),
        )[0]

    o_part, ml_part = part_k(
        inputs=inputs,
        output_shapes=[(rows, n_splits, n_q, D), (rows, n_splits, n_q, 2)],
        output_dtypes=[mx.bfloat16, mx.float32],
        grid=(32, rows * tg_y, n_splits),
        threadgroup=(32, tg_y, 1),
    )
    return _reduce_split_kv(o_part, ml_part, B, L_q, D, vec4_reduce=vec4_reduce, n_q=n_q)


# ---------------------------------------------------------------------------
# Weight-Reusing Exact Multi-Row `qmv` & `gemv` Kernels (`0.0` ULP vs `L_q=1`)
# ---------------------------------------------------------------------------
_MULTI_ROW_QMV_KERNELS: dict[tuple[int, int, int, int, int, str], Any] = {}
_MULTI_ROW_GEMV_KERNELS: dict[tuple[int, int, int], Any] = {}


def _get_multi_row_qmv_kernel(
    n_rows: int, in_d: int, out_d: int, group_size: int, bits: int, variant: str
) -> Any:
    key = (n_rows, in_d, out_d, group_size, bits, variant)
    kern = _MULTI_ROW_QMV_KERNELS.get(key)
    if kern is not None:
        return kern

    if variant == "quad":
        src = f"""
    constexpr int R = {n_rows};
    constexpr int IN_D = {in_d};
    constexpr int OUT_D = {out_d};
    constexpr int GS = {group_size};
    constexpr int BITS = {bits};
    constexpr int PF = 32 / BITS;
    constexpr int VPT = IN_D / 4;
    constexpr int PPT = VPT / PF;
    constexpr int STEP = GS / VPT;

    uint tid_y = threadgroup_position_in_grid.y;
    uint q_gid = quadgroup_index_in_threadgroup;
    uint q_lid = thread_index_in_quadgroup;

    int in_w = IN_D / PF;
    int in_g = IN_D / GS;
    int out_row = (int)tid_y * 64 + (int)q_gid;

    const device uint32_t* wp = w + (size_t)out_row * in_w + q_lid * PPT;
    const device bfloat* sp = scales + (size_t)out_row * in_g + q_lid / STEP;
    const device bfloat* bp = biases + (size_t)out_row * in_g + q_lid / STEP;

    float x_th[R][VPT];
    float sum_x[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {{
        const device bfloat* xr = x + (size_t)r * IN_D + q_lid * VPT;
        float s = 0.0f;
        if (BITS == 2) {{
            #pragma unroll
            for (int i = 0; i < VPT; i += 4) {{
                s += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
                x_th[r][i] = float(xr[i]);
                x_th[r][i + 1] = float(xr[i + 1]) / 4.0f;
                x_th[r][i + 2] = float(xr[i + 2]) / 16.0f;
                x_th[r][i + 3] = float(xr[i + 3]) / 64.0f;
            }}
        }} else if (BITS == 4) {{
            #pragma unroll
            for (int i = 0; i < VPT; i += 4) {{
                s += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
                x_th[r][i] = float(xr[i]);
                x_th[r][i + 1] = float(xr[i + 1]) / 16.0f;
                x_th[r][i + 2] = float(xr[i + 2]) / 256.0f;
                x_th[r][i + 3] = float(xr[i + 3]) / 4096.0f;
            }}
        }} else {{
            #pragma unroll
            for (int i = 0; i < VPT; ++i) {{
                s += float(xr[i]);
                x_th[r][i] = float(xr[i]);
            }}
        }}
        sum_x[r] = s;
    }}

    float res[R][8];
    #pragma unroll
    for (int r = 0; r < R; ++r) {{
        #pragma unroll
        for (int row = 0; row < 8; ++row) {{
            res[r][row] = 0.0f;
        }}
    }}

    #pragma unroll
    for (int row = 0; row < 8; ++row) {{
        if (row * 8 + out_row < OUT_D) {{
            const device uint8_t* wl = (const device uint8_t*)(wp + (size_t)row * in_w * 8);
            float sv = float(*(sp + (size_t)row * in_g * 8));
            float bv = float(*(bp + (size_t)row * in_g * 8));
            if (BITS == 2) {{
                uint8_t w_reg[VPT / 4];
                #pragma unroll
                for (int i = 0; i < VPT / 4; ++i) w_reg[i] = wl[i];
                #pragma unroll
                for (int r = 0; r < R; ++r) {{
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VPT / 4; ++i) {{
                        uint8_t wb = w_reg[i];
                        acc += (x_th[r][4 * i] * (wb & 0x03) +
                                x_th[r][4 * i + 1] * (wb & 0x0c) +
                                x_th[r][4 * i + 2] * (wb & 0x30) +
                                x_th[r][4 * i + 3] * (wb & 0xc0));
                    }}
                    res[r][row] += sv * acc + sum_x[r] * bv;
                }}
            }} else if (BITS == 4) {{
                const device uint16_t* ws16 = (const device uint16_t*)wl;
                uint16_t w_reg[VPT / 4];
                #pragma unroll
                for (int i = 0; i < VPT / 4; ++i) w_reg[i] = ws16[i];
                #pragma unroll
                for (int r = 0; r < R; ++r) {{
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VPT / 4; ++i) {{
                        uint16_t ww = w_reg[i];
                        acc += (x_th[r][4 * i] * (ww & 0x000f) +
                                x_th[r][4 * i + 1] * (ww & 0x00f0) +
                                x_th[r][4 * i + 2] * (ww & 0x0f00) +
                                x_th[r][4 * i + 3] * (ww & 0xf000));
                    }}
                    res[r][row] += sv * acc + sum_x[r] * bv;
                }}
            }} else {{
                uint8_t w_reg[VPT];
                #pragma unroll
                for (int i = 0; i < VPT; ++i) w_reg[i] = wl[i];
                #pragma unroll
                for (int r = 0; r < R; ++r) {{
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VPT; ++i) {{
                        acc += x_th[r][i] * w_reg[i];
                    }}
                    res[r][row] += sv * acc + sum_x[r] * bv;
                }}
            }}
        }}
    }}

    #pragma unroll
    for (int row = 0; row < 8; ++row) {{
        int r_idx = row * 8 + out_row;
        #pragma unroll
        for (int r = 0; r < R; ++r) {{
            float v = quad_sum(res[r][row]);
            if (q_lid == 0 && r_idx < OUT_D) {{
                out[(size_t)r * OUT_D + r_idx] = bfloat(v);
            }}
        }}
    }}
        """
    elif variant == "fast":
        ppt = 1 if bits <= 2 else 2
        src = f"""
    constexpr int R = {n_rows};
    constexpr int IN_D = {in_d};
    constexpr int OUT_D = {out_d};
    constexpr int GS = {group_size};
    constexpr int BITS = {bits};
    constexpr int PF = 32 / BITS;
    constexpr int PPT = {ppt};
    constexpr int VPT = PF * PPT;
    constexpr int BLK = VPT * 32;
    constexpr int STEP = GS / VPT;

    uint tid_y = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint sl = thread_index_in_simdgroup;

    int in_w = IN_D * 4 / PF;
    int in_g = IN_D / GS;
    int out_row = (int)tid_y * 8 + (int)sg * 4;
    if (out_row >= OUT_D) return;
    int last_row = min(3, OUT_D - 1 - out_row);

    const device uint8_t* ws = (const device uint8_t*)w + (size_t)out_row * in_w + sl * PPT * 4;
    const device bfloat* sp = scales + (size_t)out_row * in_g + sl / STEP;
    const device bfloat* bp = biases + (size_t)out_row * in_g + sl / STEP;

    float res[R][4];
    #pragma unroll
    for (int r = 0; r < R; ++r) {{
        #pragma unroll
        for (int row = 0; row < 4; ++row) {{
            res[r][row] = 0.0f;
        }}
    }}

    float x_th[R][VPT];
    float sum_x[R];
    for (int k = 0; k < IN_D; k += BLK) {{
        #pragma unroll
        for (int r = 0; r < R; ++r) {{
            const device bfloat* xr = x + (size_t)r * IN_D + k + sl * VPT;
            float s = 0.0f;
            if (BITS == 2) {{
                #pragma unroll
                for (int i = 0; i < VPT; i += 4) {{
                    s += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
                    x_th[r][i] = float(xr[i]);
                    x_th[r][i + 1] = float(xr[i + 1]) / 4.0f;
                    x_th[r][i + 2] = float(xr[i + 2]) / 16.0f;
                    x_th[r][i + 3] = float(xr[i + 3]) / 64.0f;
                }}
            }} else if (BITS == 4) {{
                #pragma unroll
                for (int i = 0; i < VPT; i += 4) {{
                    s += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
                    x_th[r][i] = float(xr[i]);
                    x_th[r][i + 1] = float(xr[i + 1]) / 16.0f;
                    x_th[r][i + 2] = float(xr[i + 2]) / 256.0f;
                    x_th[r][i + 3] = float(xr[i + 3]) / 4096.0f;
                }}
            }} else {{
                #pragma unroll
                for (int i = 0; i < VPT; ++i) {{
                    s += float(xr[i]);
                    x_th[r][i] = float(xr[i]);
                }}
            }}
            sum_x[r] = s;
        }}

        #pragma unroll
        for (int row = 0; row < 4; ++row) {{
            int src_r = min(row, last_row);
            const device uint8_t* wl = ws + (size_t)src_r * in_w;
            float sv = float(*(sp + (size_t)src_r * in_g));
            float bv = float(*(bp + (size_t)src_r * in_g));
            if (BITS == 2) {{
                uint8_t w_reg[VPT / 4];
                #pragma unroll
                for (int i = 0; i < VPT / 4; ++i) w_reg[i] = wl[i];
                #pragma unroll
                for (int r = 0; r < R; ++r) {{
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VPT / 4; ++i) {{
                        uint8_t wb = w_reg[i];
                        acc += (x_th[r][4 * i] * (wb & 0x03) +
                                x_th[r][4 * i + 1] * (wb & 0x0c) +
                                x_th[r][4 * i + 2] * (wb & 0x30) +
                                x_th[r][4 * i + 3] * (wb & 0xc0));
                    }}
                    res[r][row] += sv * acc + sum_x[r] * bv;
                }}
            }} else if (BITS == 4) {{
                const device uint16_t* ws16 = (const device uint16_t*)wl;
                uint16_t w_reg[VPT / 4];
                #pragma unroll
                for (int i = 0; i < VPT / 4; ++i) w_reg[i] = ws16[i];
                #pragma unroll
                for (int r = 0; r < R; ++r) {{
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VPT / 4; ++i) {{
                        uint16_t ww = w_reg[i];
                        acc += (x_th[r][4 * i] * (ww & 0x000f) +
                                x_th[r][4 * i + 1] * (ww & 0x00f0) +
                                x_th[r][4 * i + 2] * (ww & 0x0f00) +
                                x_th[r][4 * i + 3] * (ww & 0xf000));
                    }}
                    res[r][row] += sv * acc + sum_x[r] * bv;
                }}
            }} else {{
                uint8_t w_reg[VPT];
                #pragma unroll
                for (int i = 0; i < VPT; ++i) w_reg[i] = wl[i];
                #pragma unroll
                for (int r = 0; r < R; ++r) {{
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VPT; ++i) {{
                        acc += x_th[r][i] * w_reg[i];
                    }}
                    res[r][row] += sv * acc + sum_x[r] * bv;
                }}
            }}
        }}
        ws += BLK * 4 / PF;
        sp += BLK / GS;
        bp += BLK / GS;
    }}

    #pragma unroll
    for (int row = 0; row < 4; ++row) {{
        #pragma unroll
        for (int r = 0; r < R; ++r) {{
            float v = simd_sum(res[r][row]);
            if (sl == 0 && row <= last_row) {{
                out[(size_t)r * OUT_D + out_row + row] = bfloat(v);
            }}
        }}
    }}
        """
    else:
        src = f"""
    constexpr int R = {n_rows};
    constexpr int IN_D = {in_d};
    constexpr int OUT_D = {out_d};
    constexpr int GS = {group_size};
    constexpr int BITS = {bits};
    constexpr int PF = 32 / BITS;
    constexpr int VPT = PF;
    constexpr int BLK = VPT * 32;
    constexpr int STEP = GS / VPT;

    uint tid_y = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint sl = thread_index_in_simdgroup;

    int in_w = IN_D * 4 / PF;
    int in_g = IN_D / GS;
    int out_row = (int)tid_y * 8 + (int)sg * 4;
    if (out_row >= OUT_D) return;
    int used_out_row = (OUT_D < 8) ? out_row : min(OUT_D - 4, out_row);
    int n_rows_tile = (OUT_D < 8) ? min(4, OUT_D - out_row) : 4;

    const device uint8_t* ws = (const device uint8_t*)w + (size_t)used_out_row * in_w + sl * 4;
    const device bfloat* sp = scales + (size_t)used_out_row * in_g + sl / STEP;
    const device bfloat* bp = biases + (size_t)used_out_row * in_g + sl / STEP;

    float res[R][4];
    #pragma unroll
    for (int r = 0; r < R; ++r) {{
        #pragma unroll
        for (int row = 0; row < 4; ++row) {{
            res[r][row] = 0.0f;
        }}
    }}

    float x_th[R][VPT];
    float sum_x[R];
    int k = 0;
    for (; k < IN_D - BLK; k += BLK) {{
        #pragma unroll
        for (int r = 0; r < R; ++r) {{
            const device bfloat* xr = x + (size_t)r * IN_D + k + sl * VPT;
            float s = 0.0f;
            if (BITS == 2) {{
                #pragma unroll
                for (int i = 0; i < VPT; i += 4) {{
                    s += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
                    x_th[r][i] = float(xr[i]);
                    x_th[r][i + 1] = float(xr[i + 1]) / 4.0f;
                    x_th[r][i + 2] = float(xr[i + 2]) / 16.0f;
                    x_th[r][i + 3] = float(xr[i + 3]) / 64.0f;
                }}
            }} else if (BITS == 4) {{
                #pragma unroll
                for (int i = 0; i < VPT; i += 4) {{
                    s += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
                    x_th[r][i] = float(xr[i]);
                    x_th[r][i + 1] = float(xr[i + 1]) / 16.0f;
                    x_th[r][i + 2] = float(xr[i + 2]) / 256.0f;
                    x_th[r][i + 3] = float(xr[i + 3]) / 4096.0f;
                }}
            }} else {{
                #pragma unroll
                for (int i = 0; i < VPT; ++i) {{
                    s += float(xr[i]);
                    x_th[r][i] = float(xr[i]);
                }}
            }}
            sum_x[r] = s;
        }}
        for (int row = 0; row < n_rows_tile; ++row) {{
            const device uint8_t* wl = ws + (size_t)row * in_w;
            float sv = float(*(sp + (size_t)row * in_g));
            float bv = float(*(bp + (size_t)row * in_g));
            if (BITS == 2) {{
                uint8_t w_reg[VPT / 4];
                #pragma unroll
                for (int i = 0; i < VPT / 4; ++i) w_reg[i] = wl[i];
                #pragma unroll
                for (int r = 0; r < R; ++r) {{
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VPT / 4; ++i) {{
                        uint8_t wb = w_reg[i];
                        acc += (x_th[r][4 * i] * (wb & 0x03) +
                                x_th[r][4 * i + 1] * (wb & 0x0c) +
                                x_th[r][4 * i + 2] * (wb & 0x30) +
                                x_th[r][4 * i + 3] * (wb & 0xc0));
                    }}
                    res[r][row] += sv * acc + sum_x[r] * bv;
                }}
            }} else if (BITS == 4) {{
                const device uint16_t* ws16 = (const device uint16_t*)wl;
                uint16_t w_reg[VPT / 4];
                #pragma unroll
                for (int i = 0; i < VPT / 4; ++i) w_reg[i] = ws16[i];
                #pragma unroll
                for (int r = 0; r < R; ++r) {{
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VPT / 4; ++i) {{
                        uint16_t ww = w_reg[i];
                        acc += (x_th[r][4 * i] * (ww & 0x000f) +
                                x_th[r][4 * i + 1] * (ww & 0x00f0) +
                                x_th[r][4 * i + 2] * (ww & 0x0f00) +
                                x_th[r][4 * i + 3] * (ww & 0xf000));
                    }}
                    res[r][row] += sv * acc + sum_x[r] * bv;
                }}
            }} else {{
                uint8_t w_reg[VPT];
                #pragma unroll
                for (int i = 0; i < VPT; ++i) w_reg[i] = wl[i];
                #pragma unroll
                for (int r = 0; r < R; ++r) {{
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VPT; ++i) {{
                        acc += x_th[r][i] * w_reg[i];
                    }}
                    res[r][row] += sv * acc + sum_x[r] * bv;
                }}
            }}
        }}
        ws += BLK * 4 / PF;
        sp += BLK / GS;
        bp += BLK / GS;
    }}

    int remaining = clamp((int)(IN_D - k - (int)sl * VPT), 0, VPT);
    if (remaining > 0) {{
        for (int r = 0; r < R; ++r) {{
            const device bfloat* xr = x + (size_t)r * IN_D + k + sl * VPT;
            float s = 0.0f;
            if (BITS == 2) {{
                for (int i = 0; i < remaining; i += 4) {{
                    s += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
                    x_th[r][i] = float(xr[i]);
                    x_th[r][i + 1] = float(xr[i + 1]) / 4.0f;
                    x_th[r][i + 2] = float(xr[i + 2]) / 16.0f;
                    x_th[r][i + 3] = float(xr[i + 3]) / 64.0f;
                }}
            }} else if (BITS == 4) {{
                for (int i = 0; i < remaining; i += 4) {{
                    s += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
                    x_th[r][i] = float(xr[i]);
                    x_th[r][i + 1] = float(xr[i + 1]) / 16.0f;
                    x_th[r][i + 2] = float(xr[i + 2]) / 256.0f;
                    x_th[r][i + 3] = float(xr[i + 3]) / 4096.0f;
                }}
            }} else {{
                for (int i = 0; i < remaining; ++i) {{
                    s += float(xr[i]);
                    x_th[r][i] = float(xr[i]);
                }}
            }}
            for (int i = remaining; i < VPT; ++i) x_th[r][i] = 0.0f;
            sum_x[r] = s;
        }}
        for (int row = 0; row < n_rows_tile; ++row) {{
            const device uint8_t* wl = ws + (size_t)row * in_w;
            float sv = float(*(sp + (size_t)row * in_g));
            float bv = float(*(bp + (size_t)row * in_g));
            for (int r = 0; r < R; ++r) {{
                float acc = 0.0f;
                if (BITS == 2) {{
                    for (int i = 0; i < remaining / 4; ++i) {{
                        uint8_t wb = wl[i];
                        acc += (x_th[r][4 * i] * (wb & 0x03) +
                                x_th[r][4 * i + 1] * (wb & 0x0c) +
                                x_th[r][4 * i + 2] * (wb & 0x30) +
                                x_th[r][4 * i + 3] * (wb & 0xc0));
                    }}
                }} else if (BITS == 4) {{
                    const device uint16_t* ws16 = (const device uint16_t*)wl;
                    for (int i = 0; i < remaining / 4; ++i) {{
                        uint16_t ww = ws16[i];
                        acc += (x_th[r][4 * i] * (ww & 0x000f) +
                                x_th[r][4 * i + 1] * (ww & 0x00f0) +
                                x_th[r][4 * i + 2] * (ww & 0x0f00) +
                                x_th[r][4 * i + 3] * (ww & 0xf000));
                    }}
                }} else {{
                    for (int i = 0; i < remaining; ++i) acc += x_th[r][i] * wl[i];
                }}
                res[r][row] += sv * acc + sum_x[r] * bv;
            }}
        }}
    }}

    for (int row = 0; row < n_rows_tile; ++row) {{
        #pragma unroll
        for (int r = 0; r < R; ++r) {{
            float v = simd_sum(res[r][row]);
            if (sl == 0) {{
                out[(size_t)r * OUT_D + used_out_row + row] = bfloat(v);
            }}
        }}
    }}
        """

    kern = mx.fast.metal_kernel(
        name=f"g4_multi_qmv_{variant}_r{n_rows}_i{in_d}_o{out_d}_g{group_size}_b{bits}",
        input_names=["w", "scales", "biases", "x"],
        output_names=["out"],
        source=src,
    )
    _MULTI_ROW_QMV_KERNELS[key] = kern
    return kern


def multi_row_qmv_exact_bf16(
    x_3d: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int,
    bits: int,
) -> mx.array | None:
    """Single-pass weight-reusing multi-row `qmv` (`2 <= R <= 16`) matching MLX `L_q=1` `qmv` `0.0` ULP."""
    if x_3d.ndim != 3 or x_3d.dtype != mx.bfloat16 or scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        return None
    if bits not in (2, 4, 8) or group_size not in (32, 64, 128):
        return None
    bsz, seq_len, in_d = (int(d) for d in x_3d.shape)
    n_rows = bsz * seq_len
    if not (2 <= n_rows <= 16):
        return None
    out_d = int(weight.shape[0])
    pf = 32 // bits
    if in_d < group_size or in_d % group_size != 0 or int(weight.shape[1]) * pf != in_d:
        return None

    ppt_fast = 1 if bits <= 2 else 2
    blk_fast = pf * ppt_fast * 32
    if in_d in (64, 128) and out_d % 8 == 0 and (in_d // 4) <= group_size and group_size % (in_d // 4) == 0 and (in_d // 4) >= pf:
        variant = "quad"
    elif in_d % blk_fast == 0 and out_d >= 8 and group_size % (pf * ppt_fast) == 0:
        variant = "fast"
    elif group_size % pf == 0:
        variant = "plain"
    else:
        return None

    kern = _get_multi_row_qmv_kernel(n_rows, in_d, out_d, group_size, bits, variant)
    x_contig = mx.contiguous(x_3d).reshape(n_rows, in_d)
    if variant == "quad":
        ny = (out_d + 63) // 64
        grid = (32, ny, 1)
        tg = (32, 1, 1)
    else:
        ny = (out_d + 7) // 8
        grid = (32, ny * 2, 1)
        tg = (32, 2, 1)
    return kern(
        inputs=[weight, scales, biases, x_contig],
        output_shapes=[(bsz, seq_len, out_d)],
        output_dtypes=[mx.bfloat16],
        grid=grid,
        threadgroup=tg,
    )[0]


def multi_row_gemv_exact_bf16(x_3d: mx.array, weight: mx.array) -> mx.array | None:
    """Single-pass weight-reusing BF16 multi-row `gemv` (`2 <= R <= 16`) matching MLX `L_q=1` `gemv` `0.0` ULP."""
    if x_3d.ndim != 3 or x_3d.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16:
        return None
    bsz, seq_len, in_d = (int(d) for d in x_3d.shape)
    n_rows = bsz * seq_len
    out_d = int(weight.shape[0])
    if not (2 <= n_rows <= 16) or in_d < 128 or in_d % 128 != 0 or out_d < 16 or int(weight.shape[1]) != in_d:
        return None

    key = (n_rows, in_d, out_d)
    kern = _MULTI_ROW_GEMV_KERNELS.get(key)
    if kern is None:
        src = f"""
    constexpr int R = {n_rows};
    constexpr int IN_D = {in_d};
    constexpr int OUT_D = {out_d};
    constexpr int BM = 4, TM = 4, TN = 4;
    constexpr int blockM = BM * TM;
    constexpr int blockN = 32 * TN;

    uint tid_x = threadgroup_position_in_grid.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint sl = thread_index_in_simdgroup;

    int bm = (int)sg * TM;
    int bn = (int)sl * TN;
    int out_row = (int)tid_x * blockM + bm;
    if (out_row >= OUT_D) return;
    out_row = (out_row + TM <= OUT_D) ? out_row : (OUT_D - TM);

    const device bfloat* mat = w + (size_t)out_row * IN_D;
    float res[R][TM];
    #pragma unroll
    for (int r = 0; r < R; ++r) {{
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {{
            res[r][tm] = 0.0f;
        }}
    }}

    constexpr int n_iter = IN_D / blockN;
    for (int it = 0; it < n_iter; ++it) {{
        float vc[R][TN];
        #pragma unroll
        for (int r = 0; r < R; ++r) {{
            const device bfloat* xv = x + (size_t)r * IN_D + bn;
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) vc[r][tn] = float(xv[tn]);
        }}
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {{
            bfloat inter[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) inter[tn] = mat[(size_t)tm * IN_D + bn + tn];
            #pragma unroll
            for (int r = 0; r < R; ++r) {{
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {{
                    res[r][tm] += float(inter[tn]) * vc[r][tn];
                }}
            }}
        }}
        bn += blockN;
    }}

    #pragma unroll
    for (int r = 0; r < R; ++r) {{
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {{
            #pragma unroll
            for (ushort sn = 16; sn >= 1; sn >>= 1) {{
                res[r][tm] += simd_shuffle_down(res[r][tm], sn);
            }}
            if (sl == 0) {{
                out[(size_t)r * OUT_D + out_row + tm] = bfloat(res[r][tm]);
            }}
        }}
    }}
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_multi_gemv_r{n_rows}_i{in_d}_o{out_d}",
            input_names=["w", "x"],
            output_names=["out"],
            source=src,
        )
        _MULTI_ROW_GEMV_KERNELS[key] = kern

    n_tgs = (out_d + 15) // 16
    x_contig = mx.contiguous(x_3d).reshape(n_rows, in_d)
    return kern(
        inputs=[weight, x_contig],
        output_shapes=[(bsz, seq_len, out_d)],
        output_dtypes=[mx.bfloat16],
        grid=(n_tgs * 128, 1, 1),
        threadgroup=(128, 1, 1),
    )[0]
