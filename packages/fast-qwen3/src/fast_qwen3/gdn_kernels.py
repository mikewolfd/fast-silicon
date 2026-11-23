"""Threadgroup-staged `GdnDecodeShared` Metal kernels (`FP32`/`BF16`/`INT8 block-32`) and pure-MLX fallback."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

from .kernels import dequantize_gdn_state_int8

VALID_GDN_STATE_MODES: tuple[str, ...] = ("fp32", "bf16", "int8_blk32")

_STATE_MODE_ALIASES: dict[str, str] = {
    "fp32": "fp32",
    "float32": "fp32",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "int8_blk32": "int8_blk32",
    "int8": "int8_blk32",
    "i8": "int8_blk32",
}


def _host_phys_memory_gb() -> float:
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024.0**3)
    except Exception:
        return 24.0


def normalize_gdn_state_mode(mode: str) -> str:
    """Normalize a canonical GDN live-state storage mode (`'fp32'`, `'bf16'`, or `'int8_blk32'`)."""
    key = str(mode).strip().lower()
    norm = _STATE_MODE_ALIASES.get(key)
    if norm is None:
        raise ValueError(
            f"Unsupported GDN state mode {mode!r}; expected one of {VALID_GDN_STATE_MODES} or 'auto'"
        )
    return norm


def resolve_gdn_state_mode(
    mode: str | None = "auto",
    *,
    batch_size: int = 1,
    head_k_dim: int = 128,
    host_gb: float | None = None,
) -> str:
    """Resolve GDN live-state storage mode from explicit mode, `FAST_QWEN3_GDN_STATE_MODE`, batch size, and host RAM."""
    metal_ok = bool(mx.metal.is_available())
    dk = int(head_k_dim)
    blk32_ok = dk >= 32 and (dk % 32 == 0)

    raw = "" if mode is None else str(mode).strip().lower()
    if raw in ("", "auto"):
        env_raw = os.environ.get("FAST_QWEN3_GDN_STATE_MODE", "").strip()
        if env_raw and env_raw.lower() != "auto":
            chosen = normalize_gdn_state_mode(env_raw)
            if chosen == "int8_blk32" and not blk32_ok:
                return "bf16" if metal_ok else "fp32"
            return chosen
        eff_host_gb = _host_phys_memory_gb() if host_gb is None else float(host_gb)
        b = int(batch_size)
        if metal_ok and blk32_ok and (b >= 8 or (eff_host_gb <= 28.0 and b >= 4)):
            return "int8_blk32"
        # At B < 8 (< 4 on 24 GB), GDN state fits comfortably in SLC and native
        # `mx.fast.gated_delta_update` avoids Metal kernel dispatch overhead on PLD T<=5 steps.
        return "fp32"

    chosen = normalize_gdn_state_mode(raw)
    if chosen == "int8_blk32" and not blk32_ok:
        return "bf16" if metal_ok else "fp32"
    return chosen


def quantize_gdn_state_blk32(state: mx.array) -> tuple[mx.array, mx.array]:
    """Compress `[B, Hv, Dv, Dk]` GDN state into block-32 symmetric `INT8` + `BF16` scales with `NaN`/`Inf` guard."""
    st = state.astype(mx.float32)
    b_sz, hv, dv, dk = (int(d) for d in st.shape)
    if dk < 32 or (dk % 32) != 0:
        raise ValueError(f"GDN Dk must be >= 32 and divisible by 32 for int8_blk32, got {dk}")
    if st.size == 0:
        return (
            mx.zeros((b_sz, hv, dv, dk), dtype=mx.int8),
            mx.zeros((b_sz, hv, dv, dk // 32), dtype=mx.bfloat16),
        )
    blocks = st.reshape(b_sz, hv, dv, dk // 32, 32)
    finite_blk = mx.all(mx.isfinite(blocks), axis=-1)
    amax = mx.max(mx.abs(mx.where(finite_blk[..., None], blocks, 0.0)), axis=-1)
    sc_bf = (amax * (1.0 / 127.0)).astype(mx.bfloat16)
    nan_bf = mx.array(float("nan"), dtype=mx.bfloat16)
    scales = mx.where(finite_blk, sc_bf, nan_bf)
    sc_f32 = scales.astype(mx.float32)[..., None]
    ok = finite_blk[..., None] & (sc_f32 > 0.0)
    denom = mx.where(ok, sc_f32, 1.0)
    scaled = mx.where(ok, blocks / denom, 0.0)
    codes = mx.clip(mx.round(scaled), -127.0, 127.0).astype(mx.int8).reshape(b_sz, hv, dv, dk)
    return codes, scales


def _make_gdn_decode_shared_kernel(*, segmented: bool = False, state_i8: bool = False):
    """Build threadgroup-staged `GdnDecodeShared` Metal kernel (`128` threads, `TG_DV=16`, `FP32` `g`)."""
    if not mx.metal.is_available():
        return None

    seg_decl = "threadgroup uint8_t sh_f[TILE_ROWS];" if segmented else ""
    flg_ptr = "auto flg_base = flags + size_t(b_idx) * size_t(T);" if segmented else ""
    seg_load = (
        "if (int(tid) < tile_rows) { sh_f[tid] = flg_base[t_base + int(tid)]; }"
        if segmented
        else ""
    )
    seg_open = (
        """
            uint8_t flg = sh_f[r];
            if (flg == 0) {
              if (lane_in_row == 0 && rdv < Dv) {
                y_row[rdv] = static_cast<InT>(0);
              }
            } else {
              if (flg == 2) {
                #pragma unroll
                for (int pb = 0; pb < kQuads; ++pb) {
                  st[pb] = float4(0.0f);
                }
              }
        """
        if segmented
        else "{"
    )

    if state_i8:
        state_load_block = """
        if (rdv < Dv) {
          auto rc = state_codes_in + (size_t(n) * Dv + rdv) * Dk;
          auto rs = state_scales_in + (size_t(n) * Dv + rdv) * kQuads;
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            bfloat sc_bf = rs[pb];
            ushort sc_bits = as_type<ushort>(sc_bf);
            bool sc_finite = (sc_bits & 0x7f80u) < 0x7f80u;
            int d0 = pb * 32 + int(lane_in_row) * 4;
            if (sc_finite) {
              float sc = float(sc_bf);
              st[pb] = float4(
                  static_cast<float>(rc[d0 + 0]) * sc,
                  static_cast<float>(rc[d0 + 1]) * sc,
                  static_cast<float>(rc[d0 + 2]) * sc,
                  static_cast<float>(rc[d0 + 3]) * sc);
            } else {
              st[pb] = float4(as_type<float>(0x7fc00000u));
            }
          }
        } else {
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            st[pb] = float4(0.0f);
          }
        }
        """
        state_store_block = """
        if (rdv < Dv) {
          auto wc = state_codes_out + (size_t(n) * Dv + rdv) * Dk;
          auto ws = state_scales_out + (size_t(n) * Dv + rdv) * kQuads;
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            float4 v4 = st[pb];
            uint m0 = as_type<uint>(v4.x) & 0x7fffffffu;
            uint m1 = as_type<uint>(v4.y) & 0x7fffffffu;
            uint m2 = as_type<uint>(v4.z) & 0x7fffffffu;
            uint m3 = as_type<uint>(v4.w) & 0x7fffffffu;
            uint mag = max(max(m0, m1), max(m2, m3));
            mag = max(mag, simd_shuffle_xor(mag, 1));
            mag = max(mag, simd_shuffle_xor(mag, 2));
            mag = max(mag, simd_shuffle_xor(mag, 4));
            bfloat sc_bf = (mag < 0x7f800000u)
                ? bfloat(as_type<float>(mag) * (1.0f / 127.0f))
                : as_type<bfloat>(ushort(0x7fc0));
            if (lane_in_row == 0) {
              ws[pb] = sc_bf;
            }
            float sc_f = float(sc_bf);
            bool ok = (mag < 0x7f800000u) && (sc_f > 0.0f);
            int d0 = pb * 32 + int(lane_in_row) * 4;
            if (ok) {
              wc[d0 + 0] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v4.x, sc_f)), -127.0f, 127.0f));
              wc[d0 + 1] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v4.y, sc_f)), -127.0f, 127.0f));
              wc[d0 + 2] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v4.z, sc_f)), -127.0f, 127.0f));
              wc[d0 + 3] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v4.w, sc_f)), -127.0f, 127.0f));
            } else {
              wc[d0 + 0] = static_cast<int8_t>(0);
              wc[d0 + 1] = static_cast<int8_t>(0);
              wc[d0 + 2] = static_cast<int8_t>(0);
              wc[d0 + 3] = static_cast<int8_t>(0);
            }
          }
        }
        """
        input_names = ["q", "k", "v", "g", "beta", "state_codes_in", "state_scales_in", "T_arr"]
        output_names = ["y", "state_codes_out", "state_scales_out"]
        kname = f"fast_qwen3_gdn_decode_shared{'_seg' if segmented else ''}_i8"
    else:
        state_load_block = """
        if (rdv < Dv) {
          auto rs = state_in + (size_t(n) * Dv + rdv) * Dk;
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            int d0 = pb * 32 + int(lane_in_row) * 4;
            st[pb] = float4(
                static_cast<float>(rs[d0 + 0]),
                static_cast<float>(rs[d0 + 1]),
                static_cast<float>(rs[d0 + 2]),
                static_cast<float>(rs[d0 + 3]));
          }
        } else {
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            st[pb] = float4(0.0f);
          }
        }
        """
        state_store_block = """
        if (rdv < Dv) {
          auto ws = state_out + (size_t(n) * Dv + rdv) * Dk;
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            int d0 = pb * 32 + int(lane_in_row) * 4;
            float4 v4 = st[pb];
            ws[d0 + 0] = static_cast<StT>(v4.x);
            ws[d0 + 1] = static_cast<StT>(v4.y);
            ws[d0 + 2] = static_cast<StT>(v4.z);
            ws[d0 + 3] = static_cast<StT>(v4.w);
          }
        }
        """
        input_names = ["q", "k", "v", "g", "beta", "state_in", "T_arr"]
        output_names = ["y", "state_out"]
        kname = f"fast_qwen3_gdn_decode_shared{'_seg' if segmented else ''}"

    if segmented:
        input_names.append("flags")

    source = f"""
        #pragma clang fp reassociate(off)
        constexpr int kThreads = 128;
        constexpr int kLanesPerRow = 8;
        constexpr int kSgs = kThreads / 32;
        constexpr int kRowsPerSg = 32 / kLanesPerRow;
        static_assert(Dk >= 32 && (Dk % 32) == 0, "Dk must be a positive multiple of 32");
        constexpr int kQuads = Dk / 32;
        static_assert(TG_DV == kSgs * kRowsPerSg, "TG_DV must equal 16");

        int T = int(T_arr[0]);
        uint tid = thread_position_in_threadgroup.x;
        uint row_tile = thread_position_in_grid.y;
        uint n = thread_position_in_grid.z;
        uint b_idx = n / Hv;
        uint hv_idx = n % Hv;
        uint hk_idx = hv_idx / (Hv / Hk);

        uint sg = tid >> 5;
        uint lane = tid & 31u;
        uint row_in_sg = lane / kLanesPerRow;
        uint lane_in_row = lane & (kLanesPerRow - 1u);

        threadgroup InT sh_q[TILE_ROWS * Dk];
        threadgroup InT sh_k[TILE_ROWS * Dk];
        threadgroup InT sh_v[TILE_ROWS * TG_DV];
        threadgroup float sh_g[TILE_ROWS];
        threadgroup float sh_b[TILE_ROWS];
        {seg_decl}

        auto q_base = q + (size_t(b_idx) * size_t(T) * Hk + hk_idx) * Dk;
        auto k_base = k + (size_t(b_idx) * size_t(T) * Hk + hk_idx) * Dk;
        auto v_base = v + (size_t(b_idx) * size_t(T) * Hv + hv_idx) * Dv;
        auto y_base = y + (size_t(b_idx) * size_t(T) * Hv + hv_idx) * Dv;
        auto g_base = g + size_t(b_idx) * size_t(T) * Hv + hv_idx;
        static_assert(sizeof(*g_base) == sizeof(float), "g must stay FP32 float");
        auto b_base = beta + size_t(b_idx) * size_t(T) * Hv + hv_idx;
        {flg_ptr}

        int ldv = int(sg) * kRowsPerSg + int(row_in_sg);
        int rdv = int(row_tile) * TG_DV + ldv;
        float4 st[kQuads];
        {state_load_block}

        int total_qk_stride = Hk * Dk;
        int total_vy_stride = Hv * Dv;
        int dv_tile_base = int(row_tile) * TG_DV;
        constexpr int kQkVecs = Dk / 4;
        constexpr int kVVecs = TG_DV / 4;

        for (int t_base = 0; t_base < T; t_base += TILE_ROWS) {{
          int tile_rows = min(TILE_ROWS, T - t_base);
          int total_qk4 = tile_rows * kQkVecs;
          for (int idx4 = int(tid); idx4 < total_qk4; idx4 += kThreads) {{
            int r = idx4 / kQkVecs;
            int d = (idx4 - r * kQkVecs) * 4;
            int src_off = (t_base + r) * total_qk_stride + d;
            int dst_off = r * Dk + d;
            sh_q[dst_off + 0] = q_base[src_off + 0];
            sh_q[dst_off + 1] = q_base[src_off + 1];
            sh_q[dst_off + 2] = q_base[src_off + 2];
            sh_q[dst_off + 3] = q_base[src_off + 3];
            sh_k[dst_off + 0] = k_base[src_off + 0];
            sh_k[dst_off + 1] = k_base[src_off + 1];
            sh_k[dst_off + 2] = k_base[src_off + 2];
            sh_k[dst_off + 3] = k_base[src_off + 3];
          }}
          int total_v4 = tile_rows * kVVecs;
          for (int idx4 = int(tid); idx4 < total_v4; idx4 += kThreads) {{
            int r = idx4 / kVVecs;
            int dv_off = (idx4 - r * kVVecs) * 4;
            int grdv = dv_tile_base + dv_off;
            int dst_off = r * TG_DV + dv_off;
            if (grdv + 3 < Dv) {{
              int src_off = (t_base + r) * total_vy_stride + grdv;
              sh_v[dst_off + 0] = v_base[src_off + 0];
              sh_v[dst_off + 1] = v_base[src_off + 1];
              sh_v[dst_off + 2] = v_base[src_off + 2];
              sh_v[dst_off + 3] = v_base[src_off + 3];
            }} else {{
              #pragma unroll
              for (int j = 0; j < 4; ++j) {{
                sh_v[dst_off + j] = (grdv + j < Dv)
                    ? v_base[(t_base + r) * total_vy_stride + grdv + j]
                    : static_cast<InT>(0);
              }}
            }}
          }}
          if (int(tid) < tile_rows) {{
            sh_g[tid] = g_base[(t_base + int(tid)) * Hv];
            sh_b[tid] = static_cast<float>(b_base[(t_base + int(tid)) * Hv]);
          }}
          {seg_load}
          threadgroup_barrier(mem_flags::mem_threadgroup);

          for (int r = 0; r < tile_rows; ++r) {{
            auto y_row = y_base + (t_base + r) * total_vy_stride;
            {seg_open}
            float gt = sh_g[r];
            float bt = sh_b[r];
            const threadgroup InT* kr_ptr = sh_k + r * Dk;
            const threadgroup InT* qr_ptr = sh_q + r * Dk;
            float v_val = static_cast<float>(sh_v[r * TG_DV + ldv]);

            float4 k_reg[kQuads];
            float4 q_reg[kQuads];
            float4 kv_sum4 = float4(0.0f);
            #pragma unroll
            for (int pb = 0; pb < kQuads; ++pb) {{
              int d0 = pb * 32 + int(lane_in_row) * 4;
              k_reg[pb] = float4(
                  static_cast<float>(kr_ptr[d0 + 0]),
                  static_cast<float>(kr_ptr[d0 + 1]),
                  static_cast<float>(kr_ptr[d0 + 2]),
                  static_cast<float>(kr_ptr[d0 + 3]));
              q_reg[pb] = float4(
                  static_cast<float>(qr_ptr[d0 + 0]),
                  static_cast<float>(qr_ptr[d0 + 1]),
                  static_cast<float>(qr_ptr[d0 + 2]),
                  static_cast<float>(qr_ptr[d0 + 3]));
              st[pb] *= gt;
              kv_sum4 += st[pb] * k_reg[pb];
            }}
            float kv_mem = (kv_sum4.x + kv_sum4.y) + (kv_sum4.z + kv_sum4.w);
            kv_mem += simd_shuffle_xor(kv_mem, 1);
            kv_mem += simd_shuffle_xor(kv_mem, 2);
            kv_mem += simd_shuffle_xor(kv_mem, 4);

            float delta = (v_val - kv_mem) * bt;

            float4 out_sum4 = float4(0.0f);
            #pragma unroll
            for (int pb = 0; pb < kQuads; ++pb) {{
              st[pb] += k_reg[pb] * delta;
              out_sum4 += st[pb] * q_reg[pb];
            }}
            float out_val = (out_sum4.x + out_sum4.y) + (out_sum4.z + out_sum4.w);
            out_val += simd_shuffle_xor(out_val, 1);
            out_val += simd_shuffle_xor(out_val, 2);
            out_val += simd_shuffle_xor(out_val, 4);

            if (lane_in_row == 0 && rdv < Dv) {{
              y_row[rdv] = static_cast<InT>(out_val);
            }}
            }}
          }}
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        {state_store_block}
    """
    return mx.fast.metal_kernel(
        name=kname,
        input_names=input_names,
        output_names=output_names,
        source=source,
    )


_GDN_DECODE_SHARED = _make_gdn_decode_shared_kernel(segmented=False, state_i8=False)
_GDN_DECODE_SHARED_SEG = _make_gdn_decode_shared_kernel(segmented=True, state_i8=False)
_GDN_DECODE_SHARED_I8 = _make_gdn_decode_shared_kernel(segmented=False, state_i8=True)
_GDN_DECODE_SHARED_SEG_I8 = _make_gdn_decode_shared_kernel(segmented=True, state_i8=True)


@mx.compile
def _gdn_step_ops(
    q_t: mx.array,
    k_t: mx.array,
    v_t: mx.array,
    g_t: mx.array,
    beta_t: mx.array,
    state: mx.array,
    mask_t: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    old_state = state
    qf = q_t.astype(mx.float32)
    kf = k_t.astype(mx.float32)
    vf = v_t.astype(mx.float32)
    gf = g_t.astype(mx.float32)
    bf = beta_t.astype(mx.float32)

    state = state * gf[..., None, None]
    kv_mem = (state * kf[..., None, :]).sum(axis=-1)
    delta = (vf - kv_mem) * bf[..., None]
    state = state + kf[..., None, :] * delta[..., None]
    y_t = (state * qf[..., None, :]).sum(axis=-1)

    if mask_t is not None:
        m_bool = mask_t.astype(mx.bool_)
        state = mx.where(mx.expand_dims(m_bool, axis=(1, 2, 3)), state, old_state)
        y_t = mx.where(m_bool[:, None, None], y_t, mx.zeros_like(y_t))
    return y_t.astype(q_t.dtype), state


def _format_empty_state(
    b: int,
    hv: int,
    dv: int,
    dk: int,
    state: mx.array | tuple[mx.array, mx.array] | None,
    mode: str,
) -> mx.array | tuple[mx.array, mx.array]:
    if mode == "int8_blk32" and dk >= 32 and (dk % 32 == 0):
        if state is None:
            return (
                mx.zeros((b, hv, dv, dk), dtype=mx.int8),
                mx.zeros((b, hv, dv, dk // 32), dtype=mx.bfloat16),
            )
        if isinstance(state, tuple):
            c, s = state
            return (
                c if c.dtype == mx.int8 else c.astype(mx.int8),
                s if s.dtype == mx.bfloat16 else s.astype(mx.bfloat16),
            )
        return quantize_gdn_state_blk32(state)
    target_dt = mx.bfloat16 if mode == "bf16" else mx.float32
    if state is None:
        return mx.zeros((b, hv, dv, dk), dtype=target_dt)
    if isinstance(state, tuple):
        return dequantize_gdn_state_int8(state[0], state[1], dtype=target_dt)
    return state if state.dtype == target_dt else state.astype(target_dt)


def gated_delta_update_fallback(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | tuple[mx.array, mx.array] | None = None,
    mask: mx.array | None = None,
    *,
    state_mode: str = "fp32",
) -> tuple[mx.array, mx.array | tuple[mx.array, mx.array]]:
    """Pure-MLX FP32-recurrence fallback for GatedDeltaNet when Metal is unavailable or `Dk % 32 != 0`."""
    mode = normalize_gdn_state_mode(state_mode)
    if g.ndim == 4 and int(g.shape[-1]) == 1:
        g = mx.squeeze(g, axis=-1)
    if beta.ndim == 4 and int(beta.shape[-1]) == 1:
        beta = mx.squeeze(beta, axis=-1)

    b_sz, t_len, hk, dk = (int(d) for d in q.shape)
    hv, dv = int(v.shape[2]), int(v.shape[3])
    in_dtype = q.dtype

    if t_len == 0 or b_sz == 0:
        return mx.zeros((b_sz, t_len, hv, dv), dtype=in_dtype), _format_empty_state(
            b_sz, hv, dv, dk, state, mode
        )

    if state is None:
        st_f32 = mx.zeros((b_sz, hv, dv, dk), dtype=mx.float32)
    elif isinstance(state, tuple):
        st_f32 = dequantize_gdn_state_int8(state[0], state[1], dtype=mx.float32)
    else:
        st_f32 = state.astype(mx.float32)

    if hv != hk:
        rep = hv // hk
        q_run = mx.repeat(q, rep, axis=-2)
        k_run = mx.repeat(k, rep, axis=-2)
    else:
        q_run, k_run = q, k

    ys: list[mx.array] = []
    for t in range(t_len):
        mask_t = None if mask is None else mask[:, t]
        y_t, st_f32 = _gdn_step_ops(
            q_run[:, t],
            k_run[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            st_f32,
            mask_t,
        )
        ys.append(y_t)
    y_out = mx.stack(ys, axis=1)

    if mode == "int8_blk32" and dk >= 32 and (dk % 32 == 0):
        return y_out, quantize_gdn_state_blk32(st_f32)
    if mode == "bf16":
        return y_out, st_f32.astype(mx.bfloat16)
    return y_out, st_f32


_T_LEN_ARR_CACHE: dict[int, mx.array] = {}


def _get_t_len_arr(t_len: int) -> mx.array:
    key = int(t_len)
    arr = _T_LEN_ARR_CACHE.get(key)
    if arr is None:
        arr = mx.array([key], dtype=mx.int32)
        if len(_T_LEN_ARR_CACHE) < 128:
            _T_LEN_ARR_CACHE[key] = arr
    return arr


def _select_tile_rows(t_len: int, dk: int, in_dtype: mx.Dtype) -> int:
    if t_len <= 1:
        return 1
    elem_bytes = 4 if in_dtype == mx.float32 else 2
    per_row_bytes = (2 * dk + 16) * elem_bytes + 9
    max_rows = max(4, (28672 // per_row_bytes // 4) * 4)
    return min(16, max_rows)


def gdn_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | tuple[mx.array, mx.array] | None = None,
    mask: mx.array | None = None,
    *,
    state_mode: str = "fp32",
    use_kernel: bool = True,
) -> tuple[mx.array, mx.array | tuple[mx.array, mx.array]]:
    """Run GatedDeltaNet recurrence via `GdnDecodeShared` (`FP32` `g`, `FP32`/`BF16`/`INT8-blk32` state) or fallback."""
    mode = normalize_gdn_state_mode(state_mode)
    if g.ndim == 4 and int(g.shape[-1]) == 1:
        g = mx.squeeze(g, axis=-1)
    if beta.ndim == 4 and int(beta.shape[-1]) == 1:
        beta = mx.squeeze(beta, axis=-1)

    b_sz, t_len, hk, dk = (int(d) for d in q.shape)
    hv, dv = int(v.shape[2]), int(v.shape[3])
    in_dtype = q.dtype
    k_in = k if k.dtype == in_dtype else k.astype(in_dtype)
    v_in = v if v.dtype == in_dtype else v.astype(in_dtype)
    g_f32 = g if g.dtype == mx.float32 else g.astype(mx.float32)

    if t_len == 0 or b_sz == 0:
        return mx.zeros((b_sz, t_len, hv, dv), dtype=in_dtype), _format_empty_state(
            b_sz, hv, dv, dk, state, mode
        )

    metal_ok = (
        use_kernel
        and mx.metal.is_available()
        and mx.default_device().type == mx.DeviceType.gpu
        and dk >= 32
        and (dk % 32 == 0)
        and g_f32.ndim == 3
    )
    if not metal_ok:
        return gated_delta_update_fallback(
            q,
            k_in,
            v_in,
            g_f32,
            beta,
            state,
            mask,
            state_mode=mode,
        )

    tile_rows = _select_tile_rows(t_len, dk, in_dtype)
    t_arr = _get_t_len_arr(t_len)
    is_seg = mask is not None

    if mode == "int8_blk32":
        kern = _GDN_DECODE_SHARED_SEG_I8 if is_seg else _GDN_DECODE_SHARED_I8
        if kern is None:
            return gated_delta_update_fallback(
                q, k_in, v_in, g_f32, beta, state, mask, state_mode=mode
            )
        if state is None:
            codes_in = mx.zeros((b_sz, hv, dv, dk), dtype=mx.int8)
            scales_in = mx.zeros((b_sz, hv, dv, dk // 32), dtype=mx.bfloat16)
        elif isinstance(state, tuple):
            codes_in, scales_in = state
            if codes_in.dtype != mx.int8:
                codes_in = codes_in.astype(mx.int8)
            if scales_in.dtype != mx.bfloat16:
                scales_in = scales_in.astype(mx.bfloat16)
        else:
            codes_in, scales_in = quantize_gdn_state_blk32(state)

        inputs: list[Any] = [q, k_in, v_in, g_f32, beta, codes_in, scales_in, t_arr]
        if is_seg and mask is not None:
            inputs.append(mask.astype(mx.uint8))
        y_out, codes_out, scales_out = kern(
            inputs=inputs,
            template=[
                ("InT", in_dtype),
                ("Dk", dk),
                ("Dv", dv),
                ("Hk", hk),
                ("Hv", hv),
                ("TILE_ROWS", tile_rows),
                ("TG_DV", 16),
            ],
            grid=(128, (dv + 15) // 16, b_sz * hv),
            threadgroup=(128, 1, 1),
            output_shapes=[(b_sz, t_len, hv, dv), codes_in.shape, scales_in.shape],
            output_dtypes=[in_dtype, mx.int8, mx.bfloat16],
        )
        return y_out, (codes_out, scales_out)

    kern = _GDN_DECODE_SHARED_SEG if is_seg else _GDN_DECODE_SHARED
    if kern is None:
        return gated_delta_update_fallback(
            q, k_in, v_in, g_f32, beta, state, mask, state_mode=mode
        )
    st_dt = mx.bfloat16 if mode == "bf16" else mx.float32
    if state is None:
        state_in = mx.zeros((b_sz, hv, dv, dk), dtype=st_dt)
    elif isinstance(state, tuple):
        state_in = dequantize_gdn_state_int8(state[0], state[1], dtype=st_dt)
    else:
        state_in = state if state.dtype == st_dt else state.astype(st_dt)

    dense_inputs: list[Any] = [q, k_in, v_in, g_f32, beta, state_in, t_arr]
    if is_seg and mask is not None:
        dense_inputs.append(mask.astype(mx.uint8))
    y_out, state_out = kern(  # guard: allow-fp32
        inputs=dense_inputs,
        template=[
            ("InT", in_dtype),
            ("StT", st_dt),
            ("Dk", dk),
            ("Dv", dv),
            ("Hk", hk),
            ("Hv", hv),
            ("TILE_ROWS", tile_rows),
            ("TG_DV", 16),
        ],
        grid=(128, (dv + 15) // 16, b_sz * hv),
        threadgroup=(128, 1, 1),
        output_shapes=[(b_sz, t_len, hv, dv), state_in.shape],
        output_dtypes=[in_dtype, st_dt],
    )
    return y_out, state_out


__all__ = [
    "VALID_GDN_STATE_MODES",
    "gated_delta_update_fallback",
    "gdn_delta_update",
    "normalize_gdn_state_mode",
    "quantize_gdn_state_blk32",
    "resolve_gdn_state_mode",
]
