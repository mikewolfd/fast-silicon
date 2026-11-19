"""Custom Metal & Metal 4 MPP BF16 kernels for `fast-gemma4`."""

from __future__ import annotations

import mlx.core as mx

from fast_shared.metal_mpp_bf16 import (
    fused_geglu_split,
    make_mpp_bf16_geglu_kernel,
    mpp_fused_geglu_bf16,
    mpp_linear_bf16,
)

__all__ = [
    "dequant_kv_pair_i8_bf16",
    "fused_geglu_split",
    "fused_gemma4_res_rms_bf16",
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
_VEC4_RES_RMS_KERNELS: dict[tuple[int, int, bool, bool, bool, float], object] = {}
_QKV_NORM_ROPE_KERNELS: dict[tuple[mx.Dtype, int, float], object] = {}
_RAGGED_QKV_KERNELS: dict[int, object] = {}
_RAGGED_Q_KERNELS: dict[int, object] = {}
_RAGGED_KV_ONLY_KERNELS: dict[int, object] = {}
_PLE_GEGLU_IDX_KERNELS: dict[int, tuple[object, int]] = {}
_SOFTCAP_ARGMAX_KERNELS: dict[tuple[int, int, float], object] = {}
_QKV_I8_KERNELS: dict[int, object] = {}
_DEQ_KV_PAIR_KERNELS: dict[int, object] = {}
_GATHER_DEQ_KV_PAIR_KERNELS: dict[int, object] = {}
_PREFIX_RAGGED_SDPA_KERNELS: dict[int, tuple[object, int]] = {}
_PREFIX_SLOT_KERNELS: dict[tuple[int, bool], tuple[object, int]] = {}
_VEC4_REDUCE_KERNELS: dict[int, tuple[object, int]] = {}
_PREFIX_RAGGED_REDUCE_KERNEL = None


def _get_eps_f32(eps: float = 1e-6) -> mx.array:
    arr = _EPS_F32_CACHE.get(eps)
    if arr is None:
        arr = mx.array([eps], dtype=mx.float32)
        mx.eval(arr)
        _EPS_F32_CACHE[eps] = arr
    return arr


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
    if D <= 0 or D % 256 != 0 or (x.size > 0 and x.size % D != 0):
        raise ValueError(f"Expected last dim divisible by 256, got shape {x.shape}")
    dt = x.dtype
    rows = int(x.size) // D
    if rows == 0:
        return mx.zeros(x.shape, dtype=dt), mx.zeros(x.shape, dtype=dt)
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


def fused_gemma4_res_rms_bf16(
    h: mx.array,
    sub: mx.array,
    w_post: mx.array,
    w_next: mx.array | None = None,
    *,
    eps: float = 1e-6,
    scalar: mx.array | None = None,
    emit_res: bool = True,
) -> mx.array | tuple[mx.array, mx.array]:
    """Zero-barrier SIMD-warp-per-row vectorized `bfloat4` fused Residual + RMSNorm (`m_rows == 0` safe).

    Computes `Y_res = (H + rms_norm(Sub, W_post, eps)) [* Scalar]` and optional
    `Y_norm = rms_norm(Y_res, W_next, eps)` entirely in registers using 1 SIMD group (32 lanes)
    per row (`0` `threadgroup_barrier`s, `bfloat4` coalesced loads/stores).
    """
    if h.ndim < 1 or int(h.shape[-1]) % 128 != 0:
        raise ValueError(f"Expected last dim divisible by 128, got shape {h.shape}")
    D = int(h.shape[-1])
    if sub.shape != h.shape:
        raise ValueError(f"Expected sub shape {h.shape}, got {sub.shape}")
    if w_post.shape != (D,):
        raise ValueError(f"Expected w_post shape ({D},), got {w_post.shape}")
    if w_next is not None and w_next.shape != (D,):
        raise ValueError(f"Expected w_next shape ({D},), got {w_next.shape}")
    if w_next is None and not emit_res:
        raise ValueError("emit_res=False requires w_next to be provided")
    if h.dtype != mx.bfloat16 or sub.dtype != mx.bfloat16 or w_post.dtype != mx.bfloat16:
        raise TypeError(f"Expected bfloat16 inputs, got h={h.dtype}, sub={sub.dtype}, w_post={w_post.dtype}")
    if w_next is not None and w_next.dtype != mx.bfloat16:
        raise TypeError(f"Expected bfloat16 w_next, got {w_next.dtype}")

    orig_shape = tuple(int(d) for d in h.shape)
    m_rows = int(h.size) // D
    if m_rows == 0:
        empty = mx.zeros(orig_shape, dtype=mx.bfloat16)
        if w_next is None or not emit_res:
            return empty
        return empty, mx.zeros(orig_shape, dtype=mx.bfloat16)

    sc_1d: mx.array | None = None
    if scalar is not None:
        if scalar.dtype != mx.bfloat16:
            raise TypeError(f"Expected bfloat16 scalar, got {scalar.dtype}")
        sc_1d = scalar.reshape((1,)) if scalar.ndim == 0 else scalar
        if sc_1d.ndim != 1 or int(sc_1d.size) != 1:
            raise ValueError(f"Expected 1D scalar shape (1,), got {sc_1d.shape}")

    warps = 1 if m_rows <= 4 else 8
    has_sc = sc_1d is not None
    has_next = w_next is not None
    key = (D, warps, has_sc, has_next, bool(emit_res), float(eps))
    kern = _VEC4_RES_RMS_KERNELS.get(key)
    if kern is None:
        nv = D // 128
        inv_d = 1.0 / float(D)
        in_names = ["H", "Sub", "W_post"]
        if has_sc:
            in_names.append("Scalar")
        if has_next:
            in_names.append("W_next")
        out_names: list[str] = []
        if emit_res:
            out_names.append("Y_res")
        if has_next:
            out_names.append("Y_norm")

        sc_decl = "float sc = (float)Scalar[0];" if has_sc else ""
        sc_mul = " * sc" if has_sc else ""
        write_res = (
            """
            device bfloat* yrp = Y_res + base;
            #pragma unroll
            for (uint k = 0; k < NV; ++k) {
                float4 rv = r_vec[k];
                *(device bfloat4*)(yrp + 128u * k) = bfloat4((bfloat)rv.x, (bfloat)rv.y, (bfloat)rv.z, (bfloat)rv.w);
            }
            """
            if emit_res
            else ""
        )
        next_block = (
            f"""
            float sq2 = 0.0f;
            #pragma unroll
            for (uint k = 0; k < NV; ++k) {{
                sq2 += dot(r_vec[k], r_vec[k]);
            }}
            float inv2 = rsqrt(simd_sum(sq2) * {inv_d:.9e}f + {eps:.8e}f);
            const device bfloat* wnp = W_next + lane * 4u;
            device bfloat* ynp = Y_norm + base;
            #pragma unroll
            for (uint k = 0; k < NV; ++k) {{
                bfloat4 wn4 = *(const device bfloat4*)(wnp + 128u * k);
                float4 wf = float4((float)wn4[0], (float)wn4[1], (float)wn4[2], (float)wn4[3]);
                float4 yn = r_vec[k] * inv2 * wf;
                *(device bfloat4*)(ynp + 128u * k) = bfloat4((bfloat)yn.x, (bfloat)yn.y, (bfloat)yn.z, (bfloat)yn.w);
            }}
            """
            if has_next
            else ""
        )
        row_expr = (
            "uint row = threadgroup_position_in_grid.x;"
            if warps == 1
            else f"uint row = threadgroup_position_in_grid.x * {warps}u + (thread_position_in_threadgroup.x >> 5);"
        )
        src = f"""
            constexpr uint D = {D};
            constexpr uint NV = {nv};
            uint lane = thread_position_in_threadgroup.x & 31u;
            {row_expr}
            if (row >= (uint)H_shape[0]) return;
            size_t base = (size_t)row * D + lane * 4u;
            const device bfloat* sp = Sub + base;
            float4 s_vec[NV];
            float sq1 = 0.0f;
            #pragma unroll
            for (uint k = 0; k < NV; ++k) {{
                bfloat4 sv = *(const device bfloat4*)(sp + 128u * k);
                float4 sf = float4((float)sv[0], (float)sv[1], (float)sv[2], (float)sv[3]);
                s_vec[k] = sf;
                sq1 += dot(sf, sf);
            }}
            float inv1 = rsqrt(simd_sum(sq1) * {inv_d:.9e}f + {eps:.8e}f);
            {sc_decl}
            const device bfloat* hp = H + base;
            const device bfloat* wpp = W_post + lane * 4u;
            float4 r_vec[NV];
            #pragma unroll
            for (uint k = 0; k < NV; ++k) {{
                bfloat4 hv = *(const device bfloat4*)(hp + 128u * k);
                bfloat4 wp4 = *(const device bfloat4*)(wpp + 128u * k);
                float4 hf = float4((float)hv[0], (float)hv[1], (float)hv[2], (float)hv[3]);
                float4 wf = float4((float)wp4[0], (float)wp4[1], (float)wp4[2], (float)wp4[3]);
                r_vec[k] = (hf + s_vec[k] * inv1 * wf){sc_mul};
            }}
            {write_res}
            {next_block}
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_vec4_res_rms_d{D}_w{warps}_{'sc' if has_sc else 'ns'}_{'nx' if has_next else 'add'}_{'er' if emit_res else 'nr'}",
            input_names=in_names,
            output_names=out_names,
            source=src,
        )
        _VEC4_RES_RMS_KERNELS[key] = kern

    h2d = mx.contiguous(h)
    sub2d = mx.contiguous(sub)
    if h2d.ndim != 2:
        h2d = h2d.reshape(m_rows, D)
        sub2d = sub2d.reshape(m_rows, D)
    inputs: list[mx.array] = [h2d, sub2d, w_post]
    if sc_1d is not None:
        inputs.append(sc_1d)
    if w_next is not None:
        inputs.append(w_next)

    tg_x = warps * 32
    grid_x = ((m_rows + warps - 1) // warps) * tg_x
    if emit_res and has_next:
        y_res, y_norm = kern(
            inputs=inputs,
            grid=(grid_x, 1, 1),
            threadgroup=(tg_x, 1, 1),
            output_shapes=[(m_rows, D), (m_rows, D)],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )
        if len(orig_shape) != 2:
            y_res = y_res.reshape(orig_shape)
            y_norm = y_norm.reshape(orig_shape)
        return y_res, y_norm

    y_single = kern(
        inputs=inputs,
        grid=(grid_x, 1, 1),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(m_rows, D)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return y_single if len(orig_shape) == 2 else y_single.reshape(orig_shape)


def fused_qkv_packed_norm_rope(
    qkv_raw: mx.array,
    w_qn: mx.array,
    w_kn: mx.array,
    inv_freq: mx.array,
    offset: mx.array | int,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused Q/K/V RMSNorm + Proportional RoPE on packed [B, L_q, 10*D] prefill buffer."""
    B, L_q, total_d = qkv_raw.shape
    D = w_qn.shape[0]
    if total_d != 10 * D:
        raise ValueError(f"Expected 10*D={10*D}, got {total_d}")
    dt = qkv_raw.dtype
    key = (dt, D, eps)
    tpt = D // 2
    nsg = tpt // 32
    kern = _QKV_NORM_ROPE_KERNELS.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"g4_fused_qkv_packed_norm_rope_D{D}",
            input_names=["qkv_raw", "w_qn", "w_kn", "inv_freq", "off_lq"],
            output_names=["q_out", "k_out", "v_out"],
            source=f"""
                constexpr uint D = {D};
                constexpr uint HD = D >> 1;
                constexpr uint NSG = {nsg};
                uint slot = threadgroup_position_in_grid.x;
                uint b_l = threadgroup_position_in_grid.y;
                uint tid = thread_index_in_threadgroup;
                uint sid = tid >> 5;
                uint lid = tid & 31;

                const device T* src = qkv_raw + (b_l * 10 + slot) * D;
                device T* dst;
                const device T* wn = nullptr;
                bool do_rope = true;
                if (slot < 8) {{
                    dst = q_out + (b_l * 8 + slot) * D;
                    wn = w_qn;
                }} else if (slot == 8) {{
                    dst = k_out + b_l * D;
                    wn = w_kn;
                }} else {{
                    dst = v_out + b_l * D;
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
        grid=(10 * tpt, B * L_q, 1),
        threadgroup=(tpt, 1, 1),
        output_shapes=[(B, L_q, 8, D), (B, L_q, D), (B, L_q, D)],
        output_dtypes=[dt, dt, dt],
    )


def _get_ragged_qkv_kernel(head_dim: int):
    kern = _RAGGED_QKV_KERNELS.get(head_dim)
    if kern is None:
        half_d = head_dim // 2
        src = f"""
    constexpr int D = {head_dim};
    constexpr int HALF_D = {half_d};
    int L_q = (int)qkv_shape[1];
    int row = (int)threadgroup_position_in_grid.y;
    int h = (int)threadgroup_position_in_grid.x;
    int tid = (int)thread_position_in_threadgroup.x;
    int b = row / L_q;
    int l = row % L_q;
    int simd_lane = tid & 31;
    int simd_id = tid >> 5;

    threadgroup float tg_w[8];
    threadgroup float tg_y[D];

    const device bfloat* in_ptr = qkv + (size_t)row * (10 * D) + h * D;
    float x0 = (float)in_ptr[tid];
    float x1 = (D > 256) ? (float)in_ptr[tid + 256] : 0.0f;
    float ss = x0 * x0 + ((D > 256) ? (x1 * x1) : 0.0f);
    float ws = simd_sum(ss);
    if (simd_lane == 0) tg_w[simd_id] = ws;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        float t = (tid < 8) ? tg_w[tid] : 0.0f;
        float tot = simd_sum(t);
        if (tid == 0) tg_w[0] = rsqrt(tot / (float)D + eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_w[0];

    if (h == 9) {{
        device bfloat* v_dst = v_out + (size_t)row * D;
        v_dst[tid] = (bfloat)(x0 * inv_rms);
        if (D > 256) v_dst[tid + 256] = (bfloat)(x1 * inv_rms);
        return;
    }}

    const device bfloat* nw = (h < 8) ? wq : wk;
    tg_y[tid] = x0 * inv_rms * (float)nw[tid];
    if (D > 256) tg_y[tid + 256] = x1 * inv_rms * (float)nw[tid + 256];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HALF_D) {{
        float y0 = tg_y[tid];
        float y1 = tg_y[tid + HALF_D];
        float freq = inv_freq[tid];
        device bfloat* dst = (h < 8)
            ? (q_out + ((size_t)row * 8 + h) * D)
            : (k_out + (size_t)row * D);
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
            name=f"g4_fused_qkv_ragged_d{head_dim}",
            input_names=["qkv", "wq", "wk", "inv_freq", "off_b", "eps"],
            output_names=["q_out", "k_out", "v_out"],
            source=src,
        )
        _RAGGED_QKV_KERNELS[head_dim] = kern
    return kern


def _get_ragged_q_kernel(head_dim: int):
    kern = _RAGGED_Q_KERNELS.get(head_dim)
    if kern is None:
        half_d = head_dim // 2
        src = f"""
    constexpr int D = {head_dim};
    constexpr int HALF_D = {half_d};
    int L_q = (int)q_in_shape[1];
    int row = (int)threadgroup_position_in_grid.y;
    int h = (int)threadgroup_position_in_grid.x;
    int tid = (int)thread_position_in_threadgroup.x;
    int b = row / L_q;
    int l = row % L_q;
    int simd_lane = tid & 31;
    int simd_id = tid >> 5;

    threadgroup float tg_w[8];
    threadgroup float tg_y[D];

    const device bfloat* in_ptr = q_in + ((size_t)row * 8 + h) * D;
    float x0 = (float)in_ptr[tid];
    float x1 = (D > 256) ? (float)in_ptr[tid + 256] : 0.0f;
    float ss = x0 * x0 + ((D > 256) ? (x1 * x1) : 0.0f);
    float ws = simd_sum(ss);
    if (simd_lane == 0) tg_w[simd_id] = ws;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        float t = (tid < 8) ? tg_w[tid] : 0.0f;
        float tot = simd_sum(t);
        if (tid == 0) tg_w[0] = rsqrt(tot / (float)D + eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_w[0];

    tg_y[tid] = x0 * inv_rms * (float)wq[tid];
    if (D > 256) tg_y[tid + 256] = x1 * inv_rms * (float)wq[tid + 256];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HALF_D) {{
        float y0 = tg_y[tid];
        float y1 = tg_y[tid + HALF_D];
        float freq = inv_freq[tid];
        device bfloat* dst = q_out + ((size_t)row * 8 + h) * D;
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
            name=f"g4_fused_q_ragged_d{head_dim}",
            input_names=["q_in", "wq", "inv_freq", "off_b", "eps"],
            output_names=["q_out"],
            source=src,
        )
        _RAGGED_Q_KERNELS[head_dim] = kern
    return kern


def fused_qkv_packed_norm_rope_ragged(
    qkv: mx.array,
    wq: mx.array,
    wk: mx.array,
    inv_freq: mx.array,
    off_b: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array]:
    B, L_q, ten_d = qkv.shape
    D = ten_d // 10
    kern = _get_ragged_qkv_kernel(D)
    eps_arr = _get_eps_f32(eps)
    q, k, v = kern(
        inputs=[qkv, wq, wk, inv_freq, off_b, eps_arr],
        output_shapes=[(B, L_q, 8, D), (B, L_q, D), (B, L_q, D)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
        grid=(10 * 256, B * L_q, 1),
        threadgroup=(256, 1, 1),
    )
    return q, k, v


def fused_q_norm_rope_ragged(
    q_in: mx.array,
    wq: mx.array,
    inv_freq: mx.array,
    off_b: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    B, L_q, eight_d = q_in.shape
    D = eight_d // 8
    kern = _get_ragged_q_kernel(D)
    eps_arr = _get_eps_f32(eps)
    return kern(
        inputs=[q_in, wq, inv_freq, off_b, eps_arr],
        output_shapes=[(B, L_q, 8, D)],
        output_dtypes=[mx.bfloat16],
        grid=(8 * 256, B * L_q, 1),
        threadgroup=(256, 1, 1),
    )[0]


def _get_ragged_kv_only_kernel(head_dim: int):
    k = _RAGGED_KV_ONLY_KERNELS.get(head_dim)
    if k is None:
        half_d = head_dim // 2
        src = f"""
    constexpr int D = {head_dim};
    constexpr int HALF_D = {half_d};
    int L_q = (int)kv_shape[1];
    int row = (int)threadgroup_position_in_grid.y;
    int h = (int)threadgroup_position_in_grid.x;
    int tid = (int)thread_position_in_threadgroup.x;
    int b = row / L_q;
    int l = row % L_q;
    int simd_lane = tid & 31;
    int simd_id = tid >> 5;

    threadgroup float tg_w[8];
    threadgroup float tg_y[D];

    const device bfloat* in_ptr = kv + (size_t)row * (2 * D) + h * D;
    float x0 = (float)in_ptr[tid];
    float x1 = (D > 256) ? (float)in_ptr[tid + 256] : 0.0f;
    float ss = x0 * x0 + ((D > 256) ? (x1 * x1) : 0.0f);
    float ws = simd_sum(ss);
    if (simd_lane == 0) tg_w[simd_id] = ws;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        float t = (tid < 8) ? tg_w[tid] : 0.0f;
        float tot = simd_sum(t);
        if (tid == 0) tg_w[0] = rsqrt(tot / (float)D + eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_w[0];

    if (h == 1) {{
        device bfloat* v_dst = v_out + (size_t)row * D;
        v_dst[tid] = (bfloat)(x0 * inv_rms);
        if (D > 256) v_dst[tid + 256] = (bfloat)(x1 * inv_rms);
        return;
    }}

    tg_y[tid] = x0 * inv_rms * (float)wk[tid];
    if (D > 256) tg_y[tid + 256] = x1 * inv_rms * (float)wk[tid + 256];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HALF_D) {{
        float y0 = tg_y[tid];
        float y1 = tg_y[tid + HALF_D];
        float freq = inv_freq[tid];
        device bfloat* dst = k_out + (size_t)row * D;
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
            name=f"g4_fused_kv_only_ragged_d{head_dim}",
            input_names=["kv", "wk", "inv_freq", "off_b", "eps"],
            output_names=["k_out", "v_out"],
            source=src,
        )
        _RAGGED_KV_ONLY_KERNELS[head_dim] = k
    return k


def fused_kv_only_norm_rope_ragged(
    kv: mx.array,
    wk: mx.array,
    inv_freq: mx.array,
    off_b: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """Fused K RMSNorm+RoPE and V RMSNorm on [B, L_q, 2*D] -> (k_rot, v_norm) [B, L_q, D]."""
    B, L_q, two_d = kv.shape
    D = two_d // 2
    kern = _get_ragged_kv_only_kernel(D)
    eps_arr = _get_eps_f32(eps)
    k, v = kern(
        inputs=[kv, wk, inv_freq, off_b, eps_arr],
        output_shapes=[(B, L_q, D), (B, L_q, D)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
        grid=(2 * 256, B * L_q, 1),
        threadgroup=(256, 1, 1),
    )
    return k, v


def fused_res_rms_norm(res: mx.array, x: mx.array, w: mx.array, eps: float = 1e-6) -> mx.array:
    """Zero-recompile Metal kernel for `res + rms_norm(x, w, eps)` across dynamic M."""
    D = int(x.shape[-1])
    if D <= 0 or D % 256 != 0:
        raise ValueError(f"Expected last dim divisible by 256, got {D}")
    dt = x.dtype
    rows = int(x.size) // D
    if rows == 0:
        return mx.zeros(x.shape, dtype=dt)
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
    """Zero-slice Metal kernel for `gelu_approx(gate) * pli[:, :, layer_idx, :]`."""
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
    return kern(
        inputs=[gate, pli, layer_idx_mx],
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
        grid=(tg, rows, 1),
        threadgroup=(tg, 1, 1),
    )[0]


mpp_fused_geglu_bf16_v4 = mpp_fused_geglu_bf16


def fused_softcap_argmax_bf16(
    raw_lg: mx.array,
    softcap: float = 30.0,
    vocab_limit: int | None = None,
) -> mx.array:
    """Single-pass BF16 softcap + argmax over vocab axis without materializing full logit tensor.

    Returns `-1` on any row containing `NaN` (including all-NaN rows) instead of silently emitting `0`.
    """
    stride = int(raw_lg.shape[-1])
    v_lim = int(vocab_limit) if vocab_limit is not None else stride
    if stride <= 0 or v_lim <= 0 or v_lim > stride:
        raise ValueError(f"Invalid vocab_limit={v_lim} for stride={stride}")
    B = int(raw_lg.size) // stride
    if B == 0:
        return mx.zeros((0,), dtype=mx.int32)
    raw_lg = mx.contiguous(raw_lg)
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
    """Write newly computed K,V [B, L_q, D] into per-doc buffer [B, T_cap, D]."""
    B, L_q, _ = k_new.shape
    T_cap = k_doc.shape[1]
    b_idx = mx.arange(B, dtype=mx.int32)
    is_pow2 = (T_cap & (T_cap - 1)) == 0
    if L_q == 1:
        priv = off_b - l_sys
        slot = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
        k_doc[b_idx, slot] = k_new[:, 0, :]
        v_doc[b_idx, slot] = v_new[:, 0, :]
    else:
        priv = off_b[:, None] + mx.arange(L_q, dtype=mx.int32)[None, :] - l_sys
        slot = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
        k_doc[b_idx[:, None], slot] = k_new
        v_doc[b_idx[:, None], slot] = v_new
    return k_doc, v_doc


def _get_prefix_ragged_part_kernel(head_dim: int):
    k = _PREFIX_RAGGED_SDPA_KERNELS.get(head_dim)
    if k is None:
        dv = head_dim // 128
        if head_dim == 256:
            src = f"""
    constexpr int D = {head_dim};
    constexpr int DV = {dv};
    int L_q = (int)q_shape[1];
    int L_sys = meta[0];
    int T_cap = meta[1];
    int win = meta[2];
    int ring = meta[3];
    int n_splits = meta[4];
    int ring_mask = meta[5];

    int row = (int)threadgroup_position_in_grid.y;
    int split_id = (int)threadgroup_position_in_grid.z;
    int b = row / L_q;
    int l_q = row % L_q;
    int lane = (int)thread_position_in_threadgroup.x;
    int sg = (int)thread_position_in_threadgroup.y;

    int q_pos = off_b[b] + l_q;
    int t_hi = q_pos + 1;
    int t_lo = (win > 0 && q_pos - win + 1 > 0) ? (q_pos - win + 1) : 0;
    int tot = t_hi - t_lo;
    int chunk = (tot + n_splits - 1) / n_splits;
    int s0 = t_lo + split_id * chunk;
    int s1 = min(t_hi, s0 + chunk);

    int h0 = sg * 4;
    float4 q0[DV], q1[DV], q2[DV], q3[DV];
    float4 acc0[DV], acc1[DV], acc2[DV], acc3[DV];
    const device bfloat* qb = q + (size_t)row * 8 * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
        bfloat4 v0 = *(const device bfloat4*)(qb + (h0 + 0) * D + 128 * r);
        bfloat4 v1 = *(const device bfloat4*)(qb + (h0 + 1) * D + 128 * r);
        bfloat4 v2 = *(const device bfloat4*)(qb + (h0 + 2) * D + 128 * r);
        bfloat4 v3 = *(const device bfloat4*)(qb + (h0 + 3) * D + 128 * r);
        q0[r] = float4((float)v0[0], (float)v0[1], (float)v0[2], (float)v0[3]);
        q1[r] = float4((float)v1[0], (float)v1[1], (float)v1[2], (float)v1[3]);
        q2[r] = float4((float)v2[0], (float)v2[1], (float)v2[2], (float)v2[3]);
        q3[r] = float4((float)v3[0], (float)v3[1], (float)v3[2], (float)v3[3]);
        acc0[r] = float4(0.0f); acc1[r] = float4(0.0f); acc2[r] = float4(0.0f); acc3[r] = float4(0.0f);
    }}
    float m0 = -1e38f, m1 = -1e38f, m2 = -1e38f, m3 = -1e38f;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;

    int sys_end = min(s1, L_sys);
    if (s0 < sys_end) {{
        const device bfloat* ks_base = k_sys + lane * 4;
        const device bfloat* vs_base = v_sys + lane * 4;
        for (int t = s0; t < sys_end; ++t) {{
            size_t off = (size_t)t * D;
            float4 p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            float4 vv_r[DV];
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(ks_base + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vs_base + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr; p2 += q2[r] * kr; p3 += q3[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w);
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w);
            float d2 = simd_sum(p2.x + p2.y + p2.z + p2.w);
            float d3 = simd_sum(p3.x + p3.y + p3.z + p3.w);
            float nm0 = max(d0, m0), nm1 = max(d1, m1), nm2 = max(d2, m2), nm3 = max(d3, m3);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            float a2 = metal::fast::exp(m2 - nm2), b2 = metal::fast::exp(d2 - nm2);
            float a3 = metal::fast::exp(m3 - nm3), b3 = metal::fast::exp(d3 - nm3);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1; l2 = l2 * a2 + b2; l3 = l3 * a3 + b3;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + b0 * vv_r[r]; acc1[r] = acc1[r] * a1 + b1 * vv_r[r];
                acc2[r] = acc2[r] * a2 + b2 * vv_r[r]; acc3[r] = acc3[r] * a3 + b3 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1; m2 = nm2; m3 = nm3;
        }}
    }}

    int doc_lo = max(s0, L_sys);
    if (doc_lo < s1) {{
        const device bfloat* kd_b = k_doc + (size_t)b * T_cap * D + lane * 4;
        const device bfloat* vd_b = v_doc + (size_t)b * T_cap * D + lane * 4;
        for (int t = doc_lo; t < s1; ++t) {{
            int priv = t - L_sys;
            int slot = (ring_mask >= 0) ? (priv & ring_mask) : (ring ? (priv % T_cap) : priv);
            size_t off = (size_t)slot * D;
            float4 p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            float4 vv_r[DV];
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(kd_b + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vd_b + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr; p2 += q2[r] * kr; p3 += q3[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w);
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w);
            float d2 = simd_sum(p2.x + p2.y + p2.z + p2.w);
            float d3 = simd_sum(p3.x + p3.y + p3.z + p3.w);
            float nm0 = max(d0, m0), nm1 = max(d1, m1), nm2 = max(d2, m2), nm3 = max(d3, m3);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            float a2 = metal::fast::exp(m2 - nm2), b2 = metal::fast::exp(d2 - nm2);
            float a3 = metal::fast::exp(m3 - nm3), b3 = metal::fast::exp(d3 - nm3);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1; l2 = l2 * a2 + b2; l3 = l3 * a3 + b3;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + b0 * vv_r[r]; acc1[r] = acc1[r] * a1 + b1 * vv_r[r];
                acc2[r] = acc2[r] * a2 + b2 * vv_r[r]; acc3[r] = acc3[r] * a3 + b3 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1; m2 = nm2; m3 = nm3;
        }}
    }}

    size_t pb = ((size_t)row * n_splits + split_id) * 8 + h0;
    float il0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f, il1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    float il2 = (l2 > 0.0f) ? (1.0f / l2) : 0.0f, il3 = (l3 > 0.0f) ? (1.0f / l3) : 0.0f;
    device bfloat* od = o_part + pb * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
        float4 vo0 = acc0[r] * il0, vo1 = acc1[r] * il1, vo2 = acc2[r] * il2, vo3 = acc3[r] * il3;
        *(device bfloat4*)(od + 0 * D + 128 * r) = bfloat4((bfloat)vo0.x, (bfloat)vo0.y, (bfloat)vo0.z, (bfloat)vo0.w);
        *(device bfloat4*)(od + 1 * D + 128 * r) = bfloat4((bfloat)vo1.x, (bfloat)vo1.y, (bfloat)vo1.z, (bfloat)vo1.w);
        *(device bfloat4*)(od + 2 * D + 128 * r) = bfloat4((bfloat)vo2.x, (bfloat)vo2.y, (bfloat)vo2.z, (bfloat)vo2.w);
        *(device bfloat4*)(od + 3 * D + 128 * r) = bfloat4((bfloat)vo3.x, (bfloat)vo3.y, (bfloat)vo3.z, (bfloat)vo3.w);
    }}
    if (lane == 0) {{
        ml_part[(pb + 0) * 2] = m0; ml_part[(pb + 0) * 2 + 1] = l0;
        ml_part[(pb + 1) * 2] = m1; ml_part[(pb + 1) * 2 + 1] = l1;
        ml_part[(pb + 2) * 2] = m2; ml_part[(pb + 2) * 2 + 1] = l2;
        ml_part[(pb + 3) * 2] = m3; ml_part[(pb + 3) * 2 + 1] = l3;
    }}
            """
            tg_y = 2
        else:
            src = f"""
    constexpr int D = {head_dim};
    constexpr int DV = {dv};
    int L_q = (int)q_shape[1];
    int L_sys = meta[0];
    int T_cap = meta[1];
    int win = meta[2];
    int ring = meta[3];
    int n_splits = meta[4];
    int ring_mask = meta[5];

    int row = (int)threadgroup_position_in_grid.y;
    int split_id = (int)threadgroup_position_in_grid.z;
    int b = row / L_q;
    int l_q = row % L_q;
    int lane = (int)thread_position_in_threadgroup.x;
    int sg = (int)thread_position_in_threadgroup.y;

    int q_pos = off_b[b] + l_q;
    int t_hi = q_pos + 1;
    int t_lo = (win > 0 && q_pos - win + 1 > 0) ? (q_pos - win + 1) : 0;
    int tot = t_hi - t_lo;
    int chunk = (tot + n_splits - 1) / n_splits;
    int s0 = t_lo + split_id * chunk;
    int s1 = min(t_hi, s0 + chunk);

    int h0 = sg * 2;
    int h1 = h0 + 1;
    float4 q0[DV], q1[DV], acc0[DV], acc1[DV];
    const device bfloat* qp0 = q + ((size_t)row * 8 + h0) * D + lane * 4;
    const device bfloat* qp1 = q + ((size_t)row * 8 + h1) * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
        bfloat4 v0 = *(const device bfloat4*)(qp0 + 128 * r);
        bfloat4 v1 = *(const device bfloat4*)(qp1 + 128 * r);
        q0[r] = float4((float)v0[0], (float)v0[1], (float)v0[2], (float)v0[3]);
        q1[r] = float4((float)v1[0], (float)v1[1], (float)v1[2], (float)v1[3]);
        acc0[r] = float4(0.0f);
        acc1[r] = float4(0.0f);
    }}
    float m0 = -1e38f, l0 = 0.0f;
    float m1 = -1e38f, l1 = 0.0f;

    int sys_end = min(s1, L_sys);
    if (s0 < sys_end) {{
        const device bfloat* ks_base = k_sys + lane * 4;
        const device bfloat* vs_base = v_sys + lane * 4;
        for (int t = s0; t < sys_end; ++t) {{
            size_t off = (size_t)t * D;
            float4 vv_r[DV];
            float4 p0 = 0, p1 = 0;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(ks_base + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vs_base + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w);
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w);
            float nm0 = max(d0, m0), nm1 = max(d1, m1);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + b0 * vv_r[r];
                acc1[r] = acc1[r] * a1 + b1 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1;
        }}
    }}

    int doc_lo = max(s0, L_sys);
    if (doc_lo < s1) {{
        const device bfloat* kd_b = k_doc + (size_t)b * T_cap * D + lane * 4;
        const device bfloat* vd_b = v_doc + (size_t)b * T_cap * D + lane * 4;
        for (int t = doc_lo; t < s1; ++t) {{
            int priv = t - L_sys;
            int slot = (ring_mask >= 0) ? (priv & ring_mask) : (ring ? (priv % T_cap) : priv);
            size_t off = (size_t)slot * D;
            float4 vv_r[DV];
            float4 p0 = 0, p1 = 0;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(kd_b + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vd_b + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w);
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w);
            float nm0 = max(d0, m0), nm1 = max(d1, m1);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + b0 * vv_r[r];
                acc1[r] = acc1[r] * a1 + b1 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1;
        }}
    }}

    size_t ph0 = ((size_t)row * n_splits + split_id) * 8 + h0;
    size_t ph1 = ph0 + 1;
    float il0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f, il1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    device bfloat* od0 = o_part + ph0 * D + lane * 4;
    device bfloat* od1 = o_part + ph1 * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
        float4 vo0 = acc0[r] * il0, vo1 = acc1[r] * il1;
        *(device bfloat4*)(od0 + 128 * r) = bfloat4((bfloat)vo0.x, (bfloat)vo0.y, (bfloat)vo0.z, (bfloat)vo0.w);
        *(device bfloat4*)(od1 + 128 * r) = bfloat4((bfloat)vo1.x, (bfloat)vo1.y, (bfloat)vo1.z, (bfloat)vo1.w);
    }}
    if (lane == 0) {{
        ml_part[ph0 * 2] = m0; ml_part[ph0 * 2 + 1] = l0;
        ml_part[ph1 * 2] = m1; ml_part[ph1 * 2 + 1] = l1;
    }}
            """
            tg_y = 4
        kern = mx.fast.metal_kernel(
            name=f"g4_mqa8_prefix_part_mh_d{head_dim}",
            input_names=["q", "k_sys", "v_sys", "k_doc", "v_doc", "off_b", "meta"],
            output_names=["o_part", "ml_part"],
            source=src,
        )
        k = (kern, tg_y)
        _PREFIX_RAGGED_SDPA_KERNELS[head_dim] = k
    return k


def _get_slot_reduce_kernel_vec4(head_dim: int):
    k = _VEC4_REDUCE_KERNELS.get(head_dim)
    if k is None:
        tg = head_dim // 4
        src = f"""
    constexpr int D = {head_dim};
    int n_splits = (int)o_part_shape[1];
    int row_h = (int)threadgroup_position_in_grid.y;
    int row = row_h >> 3;
    int h = row_h & 7;
    int tid = (int)thread_position_in_threadgroup.x;

    float g_m = -1e38f;
    for (int s = 0; s < n_splits; ++s) {{
        size_t idx = ((size_t)row * n_splits + s) * 8 + h;
        float m_s = ml_part[idx * 2 + 0];
        if (ml_part[idx * 2 + 1] > 0.0f && m_s > g_m) g_m = m_s;
    }}
    float g_l = 0.0f;
    float weights[32];
    for (int s = 0; s < n_splits; ++s) {{
        size_t idx = ((size_t)row * n_splits + s) * 8 + h;
        float l_s = ml_part[idx * 2 + 1];
        float w = (l_s > 0.0f) ? (l_s * metal::fast::exp(ml_part[idx * 2 + 0] - g_m)) : 0.0f;
        weights[s] = w;
        g_l += w;
    }}
    float inv_gl = (g_l > 0.0f) ? (1.0f / g_l) : 0.0f;
    int d4 = tid * 4;
    float4 acc = float4(0.0f);
    for (int s = 0; s < n_splits; ++s) {{
        size_t idx = ((size_t)row * n_splits + s) * 8 + h;
        bfloat4 pv = *(const device bfloat4*)(o_part + idx * D + d4);
        acc += weights[s] * float4((float)pv[0], (float)pv[1], (float)pv[2], (float)pv[3]);
    }}
    float4 res = acc * inv_gl;
    *(device bfloat4*)(out + (size_t)row_h * D + d4) = bfloat4((bfloat)res.x, (bfloat)res.y, (bfloat)res.z, (bfloat)res.w);
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_mqa8_prefix_reduce_vec4_d{head_dim}",
            input_names=["o_part", "ml_part"],
            output_names=["out"],
            source=src,
        )
        k = (kern, tg)
        _VEC4_REDUCE_KERNELS[head_dim] = k
    return k


def _get_prefix_ragged_reduce_kernel():
    global _PREFIX_RAGGED_REDUCE_KERNEL
    if _PREFIX_RAGGED_REDUCE_KERNEL is None:
        src = """
    int D = (int)o_part_shape[3];
    int n_splits = (int)o_part_shape[1];
    int row_h = (int)threadgroup_position_in_grid.y;
    int row = row_h / 8;
    int h = row_h % 8;
    int tid = (int)thread_position_in_threadgroup.x;

    float g_m = -1e38f;
    for (int s = 0; s < n_splits; ++s) {
        size_t idx = ((size_t)row * n_splits + s) * 8 + h;
        float m_s = ml_part[idx * 2 + 0];
        if (ml_part[idx * 2 + 1] > 0.0f && m_s > g_m) g_m = m_s;
    }
    float g_l = 0.0f;
    float weights[32];
    for (int s = 0; s < n_splits; ++s) {
        size_t idx = ((size_t)row * n_splits + s) * 8 + h;
        float l_s = ml_part[idx * 2 + 1];
        float w = (l_s > 0.0f) ? (l_s * metal::fast::exp(ml_part[idx * 2 + 0] - g_m)) : 0.0f;
        weights[s] = w;
        g_l += w;
    }
    float inv_gl = (g_l > 0.0f) ? (1.0f / g_l) : 0.0f;
    device bfloat* dst = out + (size_t)row_h * D;
    for (int d = tid; d < D; d += 256) {
        float acc = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            size_t idx = ((size_t)row * n_splits + s) * 8 + h;
            acc += weights[s] * (float)o_part[idx * D + d];
        }
        dst[d] = (bfloat)(acc * inv_gl);
    }
        """
        _PREFIX_RAGGED_REDUCE_KERNEL = mx.fast.metal_kernel(
            name="g4_mqa8_prefix_reduce_bf16",
            input_names=["o_part", "ml_part"],
            output_names=["out"],
            source=src,
        )
    return _PREFIX_RAGGED_REDUCE_KERNEL


def _reduce_split_kv(o_part: mx.array, ml_part: mx.array, B: int, L_q: int, D: int, vec4_reduce: bool) -> mx.array:
    rows = B * L_q
    if vec4_reduce:
        red_k, red_tg = _get_slot_reduce_kernel_vec4(D)
        return red_k(
            inputs=[o_part, ml_part],
            output_shapes=[(B, L_q, 8 * D)],
            output_dtypes=[mx.bfloat16],
            grid=(red_tg, rows * 8, 1),
            threadgroup=(red_tg, 1, 1),
        )[0]
    red_k = _get_prefix_ragged_reduce_kernel()
    return red_k(
        inputs=[o_part, ml_part],
        output_shapes=[(B, L_q, 8 * D)],
        output_dtypes=[mx.bfloat16],
        grid=(256, rows * 8, 1),
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
    """MQA-8 SDPA over shared `KV_sys [L_sys, D]` + ragged per-doc `KV_doc [B, T_cap, D]`."""
    B, L_q, _, D = q.shape
    L_sys = k_sys.shape[0]
    T_cap = k_doc.shape[1]
    ring = 1 if doc_ring_cap > 0 else 0
    ring_mask = (T_cap - 1) if (ring and (T_cap & (T_cap - 1)) == 0) else -1
    if n_splits <= 0:
        if window_size > 0 and window_size <= 512:
            n_splits = 2 if (B * L_q >= 96) else 4
        else:
            n_splits = 4 if (B * L_q >= 96) else 8
    if meta is None:
        meta = mx.array([L_sys, T_cap, window_size, ring, n_splits, ring_mask], dtype=mx.int32)
    part_k, tg_y = _get_prefix_ragged_part_kernel(D)
    o_part, ml_part = part_k(
        inputs=[q, k_sys, v_sys, k_doc, v_doc, off_b, meta],
        output_shapes=[(B * L_q, n_splits, 8, D), (B * L_q, n_splits, 8, 2)],
        output_dtypes=[mx.bfloat16, mx.float32],
        grid=(32, B * L_q * tg_y, n_splits),
        threadgroup=(32, tg_y, 1),
    )
    return _reduce_split_kv(o_part, ml_part, B, L_q, D, vec4_reduce=vec4_reduce)


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
    D = k_new.shape[-1]
    rows = k_new.size // D
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
    """Fused zero-temp dequantize of contiguous `(kq, ks, vq, vs)` -> `(kd, vd)` shaped `(Bg, 1, Lc, D)` bfloat16."""
    Bg, Lc, D = kq.shape
    tg = D // 4
    kern = _DEQ_KV_PAIR_KERNELS.get(D)
    if kern is None:
        src = f"""
    constexpr uint D = {D};
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    size_t base = (size_t)row * D + tid * 4;
    float sk = (float)ks[row];
    float sv = (float)vs[row];
    char4 kv = *((const device char4*)(kq + base));
    char4 vv = *((const device char4*)(vq + base));
    *((device bfloat4*)(kd + base)) = bfloat4(
        (bfloat)((float)kv[0] * sk), (bfloat)((float)kv[1] * sk),
        (bfloat)((float)kv[2] * sk), (bfloat)((float)kv[3] * sk)
    );
    *((device bfloat4*)(vd + base)) = bfloat4(
        (bfloat)((float)vv[0] * sv), (bfloat)((float)vv[1] * sv),
        (bfloat)((float)vv[2] * sv), (bfloat)((float)vv[3] * sv)
    );
        """
        kern = mx.fast.metal_kernel(
            name=f"g4_dequant_kv_pair_i8_d{D}",
            input_names=["kq", "ks", "vq", "vs"],
            output_names=["kd", "vd"],
            source=src,
        )
        _DEQ_KV_PAIR_KERNELS[D] = kern
    rows = Bg * Lc
    return kern(
        inputs=[kq, ks, vq, vs],
        output_shapes=[(Bg, 1, Lc, D), (Bg, 1, Lc, D)],
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
    """Fused 2D slot+pos gather + int8->bfloat16 dequantize -> `(kd, vd)` `(Bg, 1, L_kv, D)`."""
    D = int(k_c.shape[2])
    Bg = int(b_idx.shape[0])
    L_kv = int(p_slots.shape[0])
    tg = D // 4
    kern = _GATHER_DEQ_KV_PAIR_KERNELS.get(D)
    if kern is None:
        src = f"""
    constexpr uint D = {D};
    int T_cap = (int)k_c_shape[1];
    int L_kv = (int)p_slots_shape[0];
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    int b = (int)(row / (uint)L_kv);
    int t = (int)(row - (uint)(b * L_kv));
    int slot = b_idx[b];
    int pos = p_slots[t];
    size_t st_idx = (size_t)slot * T_cap + (size_t)pos;
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
            name=f"g4_gather_dequant_kv_pair_i8_d{D}",
            input_names=["k_c", "ks_c", "v_c", "vs_c", "b_idx", "p_slots"],
            output_names=["kd", "vd"],
            source=src,
        )
        _GATHER_DEQ_KV_PAIR_KERNELS[D] = kern
    rows = Bg * L_kv
    return kern(
        inputs=[k_c, ks_c, v_c, vs_c, b_idx, p_slots],
        output_shapes=[(Bg, 1, L_kv, D), (Bg, 1, L_kv, D)],
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
    """Zero-overhead BF16 KV scatter using precomputed `slot_idx` and `pos_slot`."""
    B, L_q, D = k_new.shape
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
    """Zero-overhead INT8 KV quantize + scatter using precomputed `slot_idx` and `pos_slot`."""
    B, L_q, D = k_new.shape
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
    B, L_q, D = k_new.shape
    T_cap = k_doc.shape[1]
    b_idx = mx.arange(B, dtype=mx.int32) if slot_idx is None else slot_idx
    is_pow2 = (T_cap & (T_cap - 1)) == 0
    if L_q == 1:
        priv = off_b - l_sys
        slot = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
        k_doc[b_idx, slot] = k_new.reshape(B, D)
        v_doc[b_idx, slot] = v_new.reshape(B, D)
    else:
        priv = off_b[:, None] + mx.arange(L_q, dtype=mx.int32)[None, :] - l_sys
        slot = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
        k_doc[b_idx[:, None], slot] = k_new
        v_doc[b_idx[:, None], slot] = v_new


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
    B, L_q, D = k_new.shape
    T_cap = k_q_doc.shape[1]
    kq, ks, vq, vs = quantize_kv_pair_i8(k_new, v_new)
    b_idx = mx.arange(B, dtype=mx.int32) if slot_idx is None else slot_idx
    is_pow2 = (T_cap & (T_cap - 1)) == 0
    if L_q == 1:
        priv = off_b - l_sys
        slot = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
        k_q_doc[b_idx, slot] = kq.reshape(B, D)
        k_s_doc[b_idx, slot] = ks.reshape(B)
        v_q_doc[b_idx, slot] = vq.reshape(B, D)
        v_s_doc[b_idx, slot] = vs.reshape(B)
    else:
        priv = off_b[:, None] + mx.arange(L_q, dtype=mx.int32)[None, :] - l_sys
        slot = (priv & (T_cap - 1)) if (doc_ring_cap > 0 and is_pow2) else (priv % T_cap if doc_ring_cap > 0 else priv)
        k_q_doc[b_idx[:, None], slot] = kq
        k_s_doc[b_idx[:, None], slot] = ks
        v_q_doc[b_idx[:, None], slot] = vq
        v_s_doc[b_idx[:, None], slot] = vs


def _get_prefix_slot_part_kernel(head_dim: int, kv_i8: bool):
    key = (head_dim, kv_i8)
    k = _PREFIX_SLOT_KERNELS.get(key)
    if k is not None:
        return k

    dv = head_dim // 128
    if head_dim == 256:
        if kv_i8:
            doc_block = f"""
    int doc_lo = max(s0, L_sys);
    if (doc_lo < s1) {{
        const device int8_t* kd_b = k_doc + (size_t)slot_row * T_cap * D + lane * 4;
        const device int8_t* vd_b = v_doc + (size_t)slot_row * T_cap * D + lane * 4;
        const device half* ks_d = k_s_doc + (size_t)slot_row * T_cap;
        const device half* vs_d = v_s_doc + (size_t)slot_row * T_cap;
        for (int t = doc_lo; t < s1; ++t) {{
            int priv = t - L_sys;
            int slot = (ring_mask >= 0) ? (priv & ring_mask) : (ring ? (priv % T_cap) : priv);
            size_t off = (size_t)slot * D;
            float ksc = (float)ks_d[slot];
            float vsc = (float)vs_d[slot];
            float4 p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            float4 vv_r[DV];
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                vec<int8_t, 4> kv = *(const device vec<int8_t, 4>*)(kd_b + off + 128 * r);
                vec<int8_t, 4> vv = *(const device vec<int8_t, 4>*)(vd_b + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr; p2 += q2[r] * kr; p3 += q3[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w) * ksc;
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w) * ksc;
            float d2 = simd_sum(p2.x + p2.y + p2.z + p2.w) * ksc;
            float d3 = simd_sum(p3.x + p3.y + p3.z + p3.w) * ksc;
            float nm0 = max(d0, m0), nm1 = max(d1, m1), nm2 = max(d2, m2), nm3 = max(d3, m3);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            float a2 = metal::fast::exp(m2 - nm2), b2 = metal::fast::exp(d2 - nm2);
            float a3 = metal::fast::exp(m3 - nm3), b3 = metal::fast::exp(d3 - nm3);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1; l2 = l2 * a2 + b2; l3 = l3 * a3 + b3;
            float bv0 = b0 * vsc, bv1 = b1 * vsc, bv2 = b2 * vsc, bv3 = b3 * vsc;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + bv0 * vv_r[r]; acc1[r] = acc1[r] * a1 + bv1 * vv_r[r];
                acc2[r] = acc2[r] * a2 + bv2 * vv_r[r]; acc3[r] = acc3[r] * a3 + bv3 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1; m2 = nm2; m3 = nm3;
        }}
    }}
            """
        else:
            doc_block = f"""
    int doc_lo = max(s0, L_sys);
    if (doc_lo < s1) {{
        const device bfloat* kd_b = k_doc + (size_t)slot_row * T_cap * D + lane * 4;
        const device bfloat* vd_b = v_doc + (size_t)slot_row * T_cap * D + lane * 4;
        for (int t = doc_lo; t < s1; ++t) {{
            int priv = t - L_sys;
            int slot = (ring_mask >= 0) ? (priv & ring_mask) : (ring ? (priv % T_cap) : priv);
            size_t off = (size_t)slot * D;
            float4 p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            float4 vv_r[DV];
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(kd_b + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vd_b + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr; p2 += q2[r] * kr; p3 += q3[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w);
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w);
            float d2 = simd_sum(p2.x + p2.y + p2.z + p2.w);
            float d3 = simd_sum(p3.x + p3.y + p3.z + p3.w);
            float nm0 = max(d0, m0), nm1 = max(d1, m1), nm2 = max(d2, m2), nm3 = max(d3, m3);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            float a2 = metal::fast::exp(m2 - nm2), b2 = metal::fast::exp(d2 - nm2);
            float a3 = metal::fast::exp(m3 - nm3), b3 = metal::fast::exp(d3 - nm3);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1; l2 = l2 * a2 + b2; l3 = l3 * a3 + b3;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + b0 * vv_r[r]; acc1[r] = acc1[r] * a1 + b1 * vv_r[r];
                acc2[r] = acc2[r] * a2 + b2 * vv_r[r]; acc3[r] = acc3[r] * a3 + b3 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1; m2 = nm2; m3 = nm3;
        }}
    }}
            """
        src = f"""
    constexpr int D = {head_dim};
    constexpr int DV = {dv};
    int L_q = (int)q_shape[1];
    int L_sys = meta[0];
    int T_cap = meta[1];
    int win = meta[2];
    int ring = meta[3];
    int n_splits = meta[4];
    int ring_mask = meta[5];

    int row = (int)threadgroup_position_in_grid.y;
    int split_id = (int)threadgroup_position_in_grid.z;
    int b = row / L_q;
    int l_q = row % L_q;
    int slot_row = slot_idx[b];
    int lane = (int)thread_position_in_threadgroup.x;
    int sg = (int)thread_position_in_threadgroup.y;

    int q_pos = off_b[b] + l_q;
    int t_hi = q_pos + 1;
    int t_lo = (win > 0 && q_pos - win + 1 > 0) ? (q_pos - win + 1) : 0;
    int tot = t_hi - t_lo;
    int chunk = (tot + n_splits - 1) / n_splits;
    int s0 = t_lo + split_id * chunk;
    int s1 = min(t_hi, s0 + chunk);

    int h0 = sg * 4;
    float4 q0[DV], q1[DV], q2[DV], q3[DV];
    float4 acc0[DV], acc1[DV], acc2[DV], acc3[DV];
    const device bfloat* qb = q + (size_t)row * 8 * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
        bfloat4 v0 = *(const device bfloat4*)(qb + (h0 + 0) * D + 128 * r);
        bfloat4 v1 = *(const device bfloat4*)(qb + (h0 + 1) * D + 128 * r);
        bfloat4 v2 = *(const device bfloat4*)(qb + (h0 + 2) * D + 128 * r);
        bfloat4 v3 = *(const device bfloat4*)(qb + (h0 + 3) * D + 128 * r);
        q0[r] = float4((float)v0[0], (float)v0[1], (float)v0[2], (float)v0[3]);
        q1[r] = float4((float)v1[0], (float)v1[1], (float)v1[2], (float)v1[3]);
        q2[r] = float4((float)v2[0], (float)v2[1], (float)v2[2], (float)v2[3]);
        q3[r] = float4((float)v3[0], (float)v3[1], (float)v3[2], (float)v3[3]);
        acc0[r] = float4(0.0f); acc1[r] = float4(0.0f); acc2[r] = float4(0.0f); acc3[r] = float4(0.0f);
    }}
    float m0 = -1e38f, m1 = -1e38f, m2 = -1e38f, m3 = -1e38f;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;

    int sys_end = min(s1, L_sys);
    if (s0 < sys_end) {{
        const device bfloat* ks_base = k_sys + lane * 4;
        const device bfloat* vs_base = v_sys + lane * 4;
        for (int t = s0; t < sys_end; ++t) {{
            size_t off = (size_t)t * D;
            float4 p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            float4 vv_r[DV];
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(ks_base + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vs_base + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr; p2 += q2[r] * kr; p3 += q3[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w);
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w);
            float d2 = simd_sum(p2.x + p2.y + p2.z + p2.w);
            float d3 = simd_sum(p3.x + p3.y + p3.z + p3.w);
            float nm0 = max(d0, m0), nm1 = max(d1, m1), nm2 = max(d2, m2), nm3 = max(d3, m3);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            float a2 = metal::fast::exp(m2 - nm2), b2 = metal::fast::exp(d2 - nm2);
            float a3 = metal::fast::exp(m3 - nm3), b3 = metal::fast::exp(d3 - nm3);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1; l2 = l2 * a2 + b2; l3 = l3 * a3 + b3;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + b0 * vv_r[r]; acc1[r] = acc1[r] * a1 + b1 * vv_r[r];
                acc2[r] = acc2[r] * a2 + b2 * vv_r[r]; acc3[r] = acc3[r] * a3 + b3 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1; m2 = nm2; m3 = nm3;
        }}
    }}

    {doc_block}

    size_t pb = ((size_t)row * n_splits + split_id) * 8 + h0;
    float il0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f, il1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    float il2 = (l2 > 0.0f) ? (1.0f / l2) : 0.0f, il3 = (l3 > 0.0f) ? (1.0f / l3) : 0.0f;
    device bfloat* od = o_part + pb * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
        float4 vo0 = acc0[r] * il0, vo1 = acc1[r] * il1, vo2 = acc2[r] * il2, vo3 = acc3[r] * il3;
        *(device bfloat4*)(od + 0 * D + 128 * r) = bfloat4((bfloat)vo0.x, (bfloat)vo0.y, (bfloat)vo0.z, (bfloat)vo0.w);
        *(device bfloat4*)(od + 1 * D + 128 * r) = bfloat4((bfloat)vo1.x, (bfloat)vo1.y, (bfloat)vo1.z, (bfloat)vo1.w);
        *(device bfloat4*)(od + 2 * D + 128 * r) = bfloat4((bfloat)vo2.x, (bfloat)vo2.y, (bfloat)vo2.z, (bfloat)vo2.w);
        *(device bfloat4*)(od + 3 * D + 128 * r) = bfloat4((bfloat)vo3.x, (bfloat)vo3.y, (bfloat)vo3.z, (bfloat)vo3.w);
    }}
    if (lane == 0) {{
        ml_part[(pb + 0) * 2] = m0; ml_part[(pb + 0) * 2 + 1] = l0;
        ml_part[(pb + 1) * 2] = m1; ml_part[(pb + 1) * 2 + 1] = l1;
        ml_part[(pb + 2) * 2] = m2; ml_part[(pb + 2) * 2 + 1] = l2;
        ml_part[(pb + 3) * 2] = m3; ml_part[(pb + 3) * 2 + 1] = l3;
    }}
        """
        tg_y = 2
    else:
        if kv_i8:
            doc_block = f"""
    int doc_lo = max(s0, L_sys);
    if (doc_lo < s1) {{
        const device int8_t* kd_b = k_doc + (size_t)slot_row * T_cap * D + lane * 4;
        const device int8_t* vd_b = v_doc + (size_t)slot_row * T_cap * D + lane * 4;
        const device half* ks_d = k_s_doc + (size_t)slot_row * T_cap;
        const device half* vs_d = v_s_doc + (size_t)slot_row * T_cap;
        for (int t = doc_lo; t < s1; ++t) {{
            int priv = t - L_sys;
            int slot = (ring_mask >= 0) ? (priv & ring_mask) : (ring ? (priv % T_cap) : priv);
            size_t off = (size_t)slot * D;
            float ksc = (float)ks_d[slot];
            float vsc = (float)vs_d[slot];
            float4 vv_r[DV];
            float4 p0 = 0, p1 = 0;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                vec<int8_t, 4> kv = *(const device vec<int8_t, 4>*)(kd_b + off + 128 * r);
                vec<int8_t, 4> vv = *(const device vec<int8_t, 4>*)(vd_b + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w) * ksc;
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w) * ksc;
            float nm0 = max(d0, m0), nm1 = max(d1, m1);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1;
            float bv0 = b0 * vsc, bv1 = b1 * vsc;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + bv0 * vv_r[r];
                acc1[r] = acc1[r] * a1 + bv1 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1;
        }}
    }}
            """
        else:
            doc_block = f"""
    int doc_lo = max(s0, L_sys);
    if (doc_lo < s1) {{
        const device bfloat* kd_b = k_doc + (size_t)slot_row * T_cap * D + lane * 4;
        const device bfloat* vd_b = v_doc + (size_t)slot_row * T_cap * D + lane * 4;
        for (int t = doc_lo; t < s1; ++t) {{
            int priv = t - L_sys;
            int slot = (ring_mask >= 0) ? (priv & ring_mask) : (ring ? (priv % T_cap) : priv);
            size_t off = (size_t)slot * D;
            float4 vv_r[DV];
            float4 p0 = 0, p1 = 0;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(kd_b + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vd_b + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w);
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w);
            float nm0 = max(d0, m0), nm1 = max(d1, m1);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + b0 * vv_r[r];
                acc1[r] = acc1[r] * a1 + b1 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1;
        }}
    }}
            """
        src = f"""
    constexpr int D = {head_dim};
    constexpr int DV = {dv};
    int L_q = (int)q_shape[1];
    int L_sys = meta[0];
    int T_cap = meta[1];
    int win = meta[2];
    int ring = meta[3];
    int n_splits = meta[4];
    int ring_mask = meta[5];

    int row = (int)threadgroup_position_in_grid.y;
    int split_id = (int)threadgroup_position_in_grid.z;
    int b = row / L_q;
    int l_q = row % L_q;
    int slot_row = slot_idx[b];
    int lane = (int)thread_position_in_threadgroup.x;
    int sg = (int)thread_position_in_threadgroup.y;

    int q_pos = off_b[b] + l_q;
    int t_hi = q_pos + 1;
    int t_lo = (win > 0 && q_pos - win + 1 > 0) ? (q_pos - win + 1) : 0;
    int tot = t_hi - t_lo;
    int chunk = (tot + n_splits - 1) / n_splits;
    int s0 = t_lo + split_id * chunk;
    int s1 = min(t_hi, s0 + chunk);

    int h0 = sg * 2;
    int h1 = h0 + 1;
    float4 q0[DV], q1[DV], acc0[DV], acc1[DV];
    const device bfloat* qp0 = q + ((size_t)row * 8 + h0) * D + lane * 4;
    const device bfloat* qp1 = q + ((size_t)row * 8 + h1) * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
        bfloat4 v0 = *(const device bfloat4*)(qp0 + 128 * r);
        bfloat4 v1 = *(const device bfloat4*)(qp1 + 128 * r);
        q0[r] = float4((float)v0[0], (float)v0[1], (float)v0[2], (float)v0[3]);
        q1[r] = float4((float)v1[0], (float)v1[1], (float)v1[2], (float)v1[3]);
        acc0[r] = float4(0.0f);
        acc1[r] = float4(0.0f);
    }}
    float m0 = -1e38f, l0 = 0.0f;
    float m1 = -1e38f, l1 = 0.0f;

    int sys_end = min(s1, L_sys);
    if (s0 < sys_end) {{
        const device bfloat* ks_base = k_sys + lane * 4;
        const device bfloat* vs_base = v_sys + lane * 4;
        for (int t = s0; t < sys_end; ++t) {{
            size_t off = (size_t)t * D;
            float4 vv_r[DV];
            float4 p0 = 0, p1 = 0;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                bfloat4 kv = *(const device bfloat4*)(ks_base + off + 128 * r);
                bfloat4 vv = *(const device bfloat4*)(vs_base + off + 128 * r);
                float4 kr = float4((float)kv[0], (float)kv[1], (float)kv[2], (float)kv[3]);
                vv_r[r] = float4((float)vv[0], (float)vv[1], (float)vv[2], (float)vv[3]);
                p0 += q0[r] * kr; p1 += q1[r] * kr;
            }}
            float d0 = simd_sum(p0.x + p0.y + p0.z + p0.w);
            float d1 = simd_sum(p1.x + p1.y + p1.z + p1.w);
            float nm0 = max(d0, m0), nm1 = max(d1, m1);
            float a0 = metal::fast::exp(m0 - nm0), b0 = metal::fast::exp(d0 - nm0);
            float a1 = metal::fast::exp(m1 - nm1), b1 = metal::fast::exp(d1 - nm1);
            l0 = l0 * a0 + b0; l1 = l1 * a1 + b1;
            #pragma unroll
            for (int r = 0; r < DV; ++r) {{
                acc0[r] = acc0[r] * a0 + b0 * vv_r[r];
                acc1[r] = acc1[r] * a1 + b1 * vv_r[r];
            }}
            m0 = nm0; m1 = nm1;
        }}
    }}

    {doc_block}

    size_t ph0 = ((size_t)row * n_splits + split_id) * 8 + h0;
    size_t ph1 = ph0 + 1;
    float il0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f, il1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    device bfloat* od0 = o_part + ph0 * D + lane * 4;
    device bfloat* od1 = o_part + ph1 * D + lane * 4;
    #pragma unroll
    for (int r = 0; r < DV; ++r) {{
        float4 vo0 = acc0[r] * il0, vo1 = acc1[r] * il1;
        *(device bfloat4*)(od0 + 128 * r) = bfloat4((bfloat)vo0.x, (bfloat)vo0.y, (bfloat)vo0.z, (bfloat)vo0.w);
        *(device bfloat4*)(od1 + 128 * r) = bfloat4((bfloat)vo1.x, (bfloat)vo1.y, (bfloat)vo1.z, (bfloat)vo1.w);
    }}
    if (lane == 0) {{
        ml_part[ph0 * 2] = m0; ml_part[ph0 * 2 + 1] = l0;
        ml_part[ph1 * 2] = m1; ml_part[ph1 * 2 + 1] = l1;
    }}
        """
        tg_y = 4

    in_names = ["q", "k_sys", "v_sys", "k_doc", "v_doc"]
    if kv_i8:
        in_names += ["k_s_doc", "v_s_doc"]
    in_names += ["off_b", "slot_idx", "meta"]

    kern = mx.fast.metal_kernel(
        name=f"g4_mqa8_prefix_slot_{'i8' if kv_i8 else 'bf16'}_d{head_dim}",
        input_names=in_names,
        output_names=["o_part", "ml_part"],
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
    """Slot-indexed MQA-8 SDPA over `KV_sys` + `KV_doc` (BF16 or INT8) with optional `bfloat4` combine."""
    B, L_q, _, D = q.shape
    rows = B * L_q
    kv_i8 = k_s_doc is not None
    part_k, tg_y = _get_prefix_slot_part_kernel(D, kv_i8=kv_i8)
    inputs = [q, k_sys, v_sys, k_doc, v_doc]
    if kv_i8:
        inputs += [k_s_doc, v_s_doc]
    inputs += [off_b, slot_idx, meta]

    o_part, ml_part = part_k(
        inputs=inputs,
        output_shapes=[(rows, n_splits, 8, D), (rows, n_splits, 8, 2)],
        output_dtypes=[mx.bfloat16, mx.float32],
        grid=(32, rows * tg_y, n_splits),
        threadgroup=(32, tg_y, 1),
    )
    return _reduce_split_kv(o_part, ml_part, B, L_q, D, vec4_reduce=vec4_reduce)
