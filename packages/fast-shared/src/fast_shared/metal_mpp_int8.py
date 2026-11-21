"""Apple10+ NAX ``MPP::matmul2d`` INT8-activation (`W8A8` and `W4A8`) quantized prefill kernels.

Implements group-64 affine quantized matrix multiplication (`Y[M, N] = X[M, K] @ W[N, K]^T + bias`)
for Apple10+ GPUs with Metal 4 Hardware TensorOps (`MPP::matmul2d` `16x32x16`/`TMxTNx64` int8->int32):

- Dynamic per-(row, group64) symmetric INT8 activation quantization (`quantize_activations_i8_k64`):
  - `#pragma clang fp reassociate(off)` and `metal::precise::divide` for exact rounding parity.
  - Bitwise `bfloat16` non-finite exponent guard (`(bits & 0x7f80u) == 0x7f80u`) propagating quiet NaN
    bits (`0x7fc00000u`) into `x_meta` even when Metal fast-math is active.
  - Outputs `[M, K] int8` codes and `[K//64, M, 2] float32` metadata (`sc`, `sc * sum(q)`).
- Offline/lazy weight repacking (`pack_affine_weights_for_mpp_i8`):
  - `bits == 8` (`W8A8`): `w_i8 = (wq ^ 0x80808080).view(int8)` + `b_eff = bq + 128 * sq`.
  - `bits == 4` (`W4A8`): `w_i4 = (wq ^ 0x88888888)` (consumed as `metal::int4b_format`) + `b_eff = bq + 8 * sq`.
  - Compact `uint32` metadata `[K//64, N]` packing `(bfloat(sq), bfloat(b_eff))` per `(group, col)`.
- Single-barrier double-buffered 64-tile (`TM=32` for `M < 64`, `TM=64` for `M >= 64`, `TN=64`) kernels
  with 16-column-band tile walk (`prefill_int8_walk`) and optional fused epilogue vector bias.

Strictly gated OFF on Apple9 (`supports_apple10_nax() == False`) and gated on Apple10+ only for
`(bits, M, K, N)` regimes empirically validated to beat both `MLX qmm` and `BF16 MPP` by >= 1.03x.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

HEADER_MPP_INT8 = """#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#pragma clang fp reassociate(off)
using namespace metal;
using namespace mpp::tensor_ops;
"""

_APPLE10_NAX_CACHE: bool | None = None
_QX_KERNELS: dict[tuple[int, str], tuple[Any, int]] = {}
_GEMM_KERNELS: dict[tuple[int, int, int, int, int, int, bool, str], Any] = {}


def _apple10_nax_supported(apple10_nax: bool | None = None) -> bool:
    """Return cached Apple10+ NAX availability, unless overridden explicitly for tests."""
    if apple10_nax is not None:
        return bool(apple10_nax)
    global _APPLE10_NAX_CACHE
    if _APPLE10_NAX_CACHE is not None:
        return _APPLE10_NAX_CACHE
    detected = False
    try:
        from .hw_check import supports_apple10_nax

        detected = bool(supports_apple10_nax())
    except Exception:
        detected = False
    _APPLE10_NAX_CACHE = detected
    return detected


def should_use_mpp_w8a8(
    m: int,
    k: int,
    n: int,
    *,
    group_size: int = 64,
    mode: str = "affine",
    apple10_nax: bool | None = None,
) -> bool:
    """Gate ``W8A8`` (`int8 * int8`) strictly to Apple10+ and winning `>= 8M`-element prefill shapes.

    Empirical Base M5 E2E + microbenchmarks vs both `mx.quantized_matmul(bits=8)` and `BF16 MPP`:
    - Wide `>= 8M` element matrices (`1536x12288`, `6144x1536` in E2B QAT-4bit): win `1.16x..1.57x` TTFT
      with `cos=0.999993`, `KL=0.0`, and zero downstream F1/schema regression.
    - Medium attention/narrow matrices (`< 8M`, e.g. `1536x2560`, `2048x1536`, `256`-wide PLE) score `< 1.03x`
      E2E/micro and fail Two-Gate (`valid_schema` drop `0.875 -> 0.75` on `e2b_w8`), so stay on `MLX qmm`.
    """
    if not _apple10_nax_supported(apple10_nax):
        return False
    m_i, k_i, n_i = int(m), int(k), int(n)
    if mode != "affine" or int(group_size) != 64 or m_i < 32:
        return False
    if k_i < 1536 or n_i < 1536 or (k_i % 64) != 0 or (n_i % 64) != 0:
        return False
    return (k_i * n_i) >= 8_000_000


def should_use_mpp_w4a8(
    m: int,
    k: int,
    n: int,
    *,
    group_size: int = 64,
    mode: str = "affine",
    apple10_nax: bool | None = None,
) -> bool:
    """Gate ``W4A8`` (`int8 * int4b_format`) strictly to Apple10+ and winning large prefill shapes.

    Medium shapes (`1536x2560`, `2048x1536`) and narrow `256`-wide projections score `< 1.0x` vs
    native `qmm(bits=4)` and are excluded cleanly.
    """
    if not _apple10_nax_supported(apple10_nax):
        return False
    m_i, k_i, n_i = int(m), int(k), int(n)
    if mode != "affine" or int(group_size) != 64 or m_i < 32:
        return False
    if k_i < 1536 or n_i < 1536 or (k_i % 64) != 0 or (n_i % 64) != 0:
        return False
    elems = k_i * n_i
    if elems < 8_000_000:
        return False
    if n_i >= 12288:
        return m_i >= 32
    if k_i >= 6144:
        return (32 <= m_i <= 128) or (m_i >= 512)
    if n_i >= 2880 and k_i >= 2880:
        if n_i >= 5120 or k_i >= 4096:
            return m_i >= 32
        return m_i >= 256
    return m_i >= 512


def should_use_mpp_int8_linear(
    m: int,
    k: int,
    n: int,
    *,
    bits: int,
    group_size: int = 64,
    mode: str = "affine",
    apple10_nax: bool | None = None,
) -> bool:
    """Dispatch `bits in {4, 8}` prefill gate check cleanly."""
    if int(bits) == 8:
        return should_use_mpp_w8a8(m, k, n, group_size=group_size, mode=mode, apple10_nax=apple10_nax)
    if int(bits) == 4:
        return should_use_mpp_w4a8(m, k, n, group_size=group_size, mode=mode, apple10_nax=apple10_nax)
    return False


def _get_qx_kernel(k: int, in_dtype_name: str = "bf16") -> tuple[Any, int]:
    """Return cached dynamic INT8 activation quantizer `(kernel, groups_per_tg)` for inner dim `k`."""
    key = (k, in_dtype_name)
    cached = _QX_KERNELS.get(key)
    if cached is not None:
        return cached
    ng = k // 64
    gptg = 4 if ng >= 4 else 1
    if in_dtype_name == "f16":
        load_stmt = """
    const device half2* xr2 = reinterpret_cast<const device half2*>(x + (size_t)m * {k}u + g * 64u);
    half2 hv = xr2[lane];
    bool nf = ((as_type<ushort>(hv[0]) & 0x7c00u) == 0x7c00u) || ((as_type<ushort>(hv[1]) & 0x7c00u) == 0x7c00u);
    float f0 = float(hv[0]), f1 = float(hv[1]);""".format(k=k)
    else:
        load_stmt = """
    const device ushort2* xr2 = reinterpret_cast<const device ushort2*>(x + (size_t)m * {k}u + g * 64u);
    bfloat2 bf = as_type<bfloat2>(xr2[lane]);
    bool nf = ((as_type<ushort>(bf[0]) & 0x7f80u) == 0x7f80u) || ((as_type<ushort>(bf[1]) & 0x7f80u) == 0x7f80u);
    float f0 = float(bf[0]), f1 = float(bf[1]);""".format(k=k)
    src = f"""
    #pragma clang fp reassociate(off)
    int M = (int)x_shape[0];
    uint m = threadgroup_position_in_grid.y;
    if ((int)m >= M) return;
    uint tid = thread_index_in_threadgroup;
    uint g = threadgroup_position_in_grid.x * {gptg}u + (tid >> 5);
    if (g >= {ng}u) return;
    uint lane = tid & 31u;{load_stmt}
    bool any_nf = simd_any(nf);
    float amax = simd_max(max(abs(f0), abs(f1)));
    float sc = (amax > 0x1p-100f) ? metal::precise::divide(amax, 127.0f) : 0.0f;
    float inv = (sc != 0.0f) ? metal::precise::divide(1.0f, sc) : 0.0f;
    int q0 = int(clamp(rint(f0 * inv), -127.0f, 127.0f));
    int q1 = int(clamp(rint(f1 * inv), -127.0f, 127.0f));
    int qsum = simd_sum(q0 + q1);
    uint pair = (uint(uint8_t(int8_t(q0)))) | (uint(uint8_t(int8_t(q1))) << 8u);
    uint other = simd_shuffle_xor(pair, 1u);
    if ((lane & 1u) == 0u) {{
        uint packed = pair | (other << 16u);
        device uint* xq4 = reinterpret_cast<device uint*>(x_q + (size_t)m * {k}u + g * 64u);
        xq4[lane >> 1] = packed;
    }}
    if (lane == 0u) {{
        size_t si = (size_t)g * (size_t)M + m;
        device uint2* xm = reinterpret_cast<device uint2*>(x_meta);
        xm[si] = any_nf
            ? uint2(0x7fc00000u, 0x7fc00000u)
            : uint2(as_type<uint>(sc), as_type<uint>(sc * float(qsum)));
    }}
    """
    kern = mx.fast.metal_kernel(
        f"mpp_qx_i8_k64_{in_dtype_name}_{k}",
        ["x"],
        ["x_q", "x_meta"],
        source=src,
    )
    _QX_KERNELS[key] = (kern, gptg)
    return kern, gptg


def quantize_activations_i8_k64(x: mx.array) -> tuple[mx.array, mx.array]:
    """Quantize `[M, K]` `bfloat16`/`float16` activations per 64-element group into `(x_q [M, K] int8, x_meta [K//64, M, 2] uint32)`."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D [M, K] activation matrix, got shape={x.shape}")
    m, k = int(x.shape[0]), int(x.shape[1])
    if (k % 64) != 0:
        raise ValueError(f"Inner dim K={k} must be a multiple of 64")
    if x.dtype == mx.float16:
        in_dtype_name = "f16"
        x_in = mx.contiguous(x)
    elif x.dtype == mx.bfloat16:
        in_dtype_name = "bf16"
        x_in = mx.contiguous(x)
    else:
        in_dtype_name = "bf16"
        x_in = mx.contiguous(x.astype(mx.bfloat16))
    ng = k // 64
    kern, gptg = _get_qx_kernel(k, in_dtype_name)
    n_tg_x = (ng + gptg - 1) // gptg
    x_q, x_meta = kern(
        inputs=[x_in],
        grid=(n_tg_x * gptg * 32, m, 1),
        threadgroup=(gptg * 32, 1, 1),
        output_shapes=[(m, k), (ng, m, 2)],
        output_dtypes=[mx.int8, mx.uint32],
    )
    return x_q, x_meta


def pack_affine_weights_for_mpp_i8(
    wq: mx.array,
    sq: mx.array,
    bq: mx.array,
    *,
    bits: int,
    group_size: int = 64,
) -> tuple[mx.array, mx.array]:
    """Repack MLX unsigned affine `(wq, sq, bq)` into signed `(w_packed, w_meta [ng, N] uint32)` on GPU."""
    b_i = int(bits)
    if int(group_size) != 64 or b_i not in (4, 8):
        raise ValueError(f"Expected group_size=64 and bits in (4, 8), got group_size={group_size}, bits={bits}")
    sq_f = sq.astype(mx.float32)
    bq_f = bq.astype(mx.float32)
    if b_i == 8:
        w_packed = mx.contiguous(mx.bitwise_xor(wq, mx.array(0x80808080, dtype=mx.uint32)).view(mx.int8))
        b_eff = (bq_f + 128.0 * sq_f).astype(mx.bfloat16)
    else:
        w_packed = mx.contiguous(mx.bitwise_xor(wq, mx.array(0x88888888, dtype=mx.uint32)))
        b_eff = (bq_f + 8.0 * sq_f).astype(mx.bfloat16)
    sq_bf = sq.astype(mx.bfloat16)
    sq_u32 = sq_bf.view(mx.uint16).astype(mx.uint32)
    be_u32 = b_eff.view(mx.uint16).astype(mx.uint32)
    w_meta = mx.contiguous((sq_u32 | (be_u32 << mx.array(16, dtype=mx.uint32))).T)
    return w_packed, w_meta


def _get_gemm_kernel(
    *,
    bits: int,
    k: int,
    n: int,
    tm: int,
    tn: int,
    sgs: int,
    has_bias: bool,
    out_dtype_name: str = "bf16",
) -> Any:
    key = (bits, k, n, tm, tn, sgs, has_bias, out_dtype_name)
    cached = _GEMM_KERNELS.get(key)
    if cached is not None:
        return cached
    ng = k // 64
    if bits == 8:
        w_arg = "w_i8"
        right_decl = "device int8_t* wp = const_cast<device int8_t*>(w_i8);"
        right_tensor = "auto W_t = tensor(wp, dextents<int, 2>{K, N}, array<int, 2>{1, K});"
    else:
        w_arg = "w_i4"
        right_decl = "device uchar* wp = const_cast<device uchar*>(reinterpret_cast<const device uchar*>(w_i4));"
        right_tensor = (
            "auto W_t = tensor<device metal::int4b_format, dextents<int, 2>, tensor_inline>"
            "(wp, dextents<int, 2>{K, N}, array<int, 2>{1, K});"
        )
    bias_tg = "threadgroup float tg_bias[TN];" if has_bias else ""
    bias_load = (
        f"""
    if (tid < (uint)TN) {{
        uint gn = n0 + tid;
        tg_bias[tid] = ((int)gn < N) ? float(bias[gn]) : 0.0f;
    }}"""
        if has_bias
        else ""
    )
    bias_add = " + tg_bias[nl[i]]" if has_bias else ""
    out_metal_t = "half" if out_dtype_name == "f16" else "bfloat"
    out_nan_bits = "0x7e00u" if out_dtype_name == "f16" else "0x7fc0u"
    input_names = ["x_q", "x_meta", w_arg, "w_meta"] + (["bias"] if has_bias else [])
    src = f"""
    constexpr int K = {k}, N = {n}, NG = {ng}, TM = {tm}, TN = {tn};
    int M = (int)x_q_shape[0];
    uint n_tiles = (uint)((N + TN - 1) / TN);
    uint m_tiles = (uint)((M + TM - 1) / TM);
    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint band = tg_x / 16u;
    uint band_w = min(16u, n_tiles - band * 16u);
    uint lin_in_band = (tg_x % 16u) * m_tiles + tg_y;
    uint n0 = (band * 16u + (lin_in_band % band_w)) * TN;
    uint m0 = (lin_in_band / band_w) * TM;
    if ((int)m0 >= M || (int)n0 >= N) return;

    threadgroup float2 tg_xm[TM];
    threadgroup uint tg_wm[TN];
    threadgroup uint tg_nf[TM];
    {bias_tg}
    uint tid = thread_index_in_threadgroup;
    if (tid < (uint)TM) {{
        tg_nf[tid] = 0u;
    }}
    {bias_load}
    device int8_t* xp = const_cast<device int8_t*>(x_q);
    {right_decl}
    auto X_t = tensor(xp, dextents<int, 2>{{K, M}}, array<int, 2>{{1, K}});
    {right_tensor}
    auto Y_t = tensor(out, dextents<int, 2>{{N, M}}, array<int, 2>{{1, N}});
    constexpr auto d0 = matmul2d_descriptor(TM, TN, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d0, execution_simdgroups<{sgs}>> op0;
    auto xs0 = X_t.slice<64, TM>(0, (int)m0);
    auto ws0 = W_t.slice<64, TN>(0, (int)n0);
    auto c = op0.get_destination_cooperative_tensor<decltype(xs0), decltype(ws0), int>();
    uint16_t cap = c.get_capacity();
    short ml[32], nl[32];
    float acc[32];
    for (uint16_t i = 0; i < cap; ++i) {{
        acc[i] = 0.0f;
        if (c.is_valid_element(i)) {{
            auto coord = c.get_multidimensional_index(i);
            ml[i] = (short)coord[1];
            nl[i] = (short)coord[0];
        }} else {{
            ml[i] = -1;
            nl[i] = -1;
        }}
    }}
    const device float2* x_meta2 = reinterpret_cast<const device float2*>(x_meta);
    for (int g = 0; g < NG; ++g) {{
        if (tid < (uint)TM) {{
            int gm = (int)m0 + (int)tid;
            float2 xm_t = (gm < M) ? x_meta2[(size_t)g * (size_t)M + gm] : float2(0.0f, 0.0f);
            tg_xm[tid] = xm_t;
            if ((as_type<uint>(xm_t.x) & 0x7f800000u) == 0x7f800000u) {{
                tg_nf[tid] = 1u;
            }}
        }}
        if (tid >= (uint)TM && tid < (uint)(TM + TN)) {{
            uint r = tid - (uint)TM;
            uint gn = n0 + r;
            tg_wm[r] = ((int)gn < N) ? w_meta[(size_t)g * (size_t)N + gn] : 0u;
        }}
        auto xs = X_t.slice<64, TM>(g * 64, (int)m0);
        auto ws = W_t.slice<64, TN>(g * 64, (int)n0);
        op0.run(xs, ws, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < cap; ++i) {{
            short rm = ml[i];
            if (rm >= 0) {{
                float2 xm = tg_xm[rm];
                uint wu = tg_wm[nl[i]];
                float ws_v = float(as_type<bfloat>(ushort(wu & 0xFFFFu)));
                float wb_v = float(as_type<bfloat>(ushort(wu >> 16)));
                acc[i] = fma(float(c[i]), xm.x * ws_v, fma(xm.y, wb_v, acc[i]));
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if ((int)m0 + TM <= M && (int)n0 + TN <= N) {{
        auto yo = op0.get_destination_cooperative_tensor<decltype(xs0), decltype(ws0), {out_metal_t}>();
        for (uint16_t i = 0; i < cap; ++i) {{
            short rm = ml[i];
            if (rm >= 0) {{
                yo[i] = (tg_nf[rm] != 0u)
                    ? as_type<{out_metal_t}>(ushort({out_nan_bits}))
                    : {out_metal_t}(acc[i]{bias_add});
            }} else {{
                yo[i] = {out_metal_t}(0.0f);
            }}
        }}
        auto ys = Y_t.slice<TN, TM>((int)n0, (int)m0);
        yo.store(ys);
    }} else {{
        for (uint16_t i = 0; i < cap; ++i) {{
            short rm = ml[i];
            if (rm >= 0 && ((int)m0 + (int)rm) < M && nl[i] >= 0 && ((int)n0 + (int)nl[i]) < N) {{
                size_t gm = (size_t)m0 + (size_t)rm;
                size_t gn = (size_t)n0 + (size_t)nl[i];
                out[gm * (size_t)N + gn] = (tg_nf[rm] != 0u)
                    ? as_type<{out_metal_t}>(ushort({out_nan_bits}))
                    : {out_metal_t}(acc[i]{bias_add});
            }}
        }}
    }}
    """
    kern = mx.fast.metal_kernel(
        f"mpp_w{bits}a8_gemm_{out_dtype_name}_{k}_{n}_{tm}x{tn}_{sgs}{'_b' if has_bias else ''}",
        input_names,
        ["out"],
        header=HEADER_MPP_INT8,
        source=src,
    )
    _GEMM_KERNELS[key] = kern
    return kern


def mpp_quantized_linear(
    x: mx.array,
    wq: mx.array,
    sq: mx.array,
    bq: mx.array,
    *,
    bits: int,
    group_size: int = 64,
    mode: str = "affine",
    bias: mx.array | None = None,
    out_dtype: mx.Dtype | None = None,
    packed_weights: tuple[mx.array, mx.array] | None = None,
    prequantized_x: tuple[mx.array, mx.array] | None = None,
) -> mx.array:
    """Execute Apple10+ `W8A8`/`W4A8` `MPP::matmul2d` affine quantized linear projection."""
    orig_shape = tuple(int(s) for s in x.shape)
    k = orig_shape[-1]
    n = int(wq.shape[0])
    m = x.size // k if x.size > 0 else 0
    eff_out_dtype = out_dtype if out_dtype is not None else (x.dtype if x.dtype in (mx.bfloat16, mx.float16) else mx.bfloat16)
    if eff_out_dtype not in (mx.bfloat16, mx.float16):
        eff_out_dtype = mx.bfloat16
    if m == 0:
        return mx.zeros(orig_shape[:-1] + (n,), dtype=eff_out_dtype)
    x_2d = x.reshape(m, k)
    if packed_weights is None:
        w_packed, w_meta = pack_affine_weights_for_mpp_i8(wq, sq, bq, bits=int(bits), group_size=int(group_size))
    else:
        w_packed, w_meta = packed_weights
    if prequantized_x is None:
        x_q, x_meta = quantize_activations_i8_k64(x_2d)
    else:
        x_q, x_meta = prequantized_x

    tm = 32 if m < 64 else 64
    tn = 64
    sgs = 4
    has_bias = bias is not None
    out_dtype_name = "f16" if eff_out_dtype == mx.float16 else "bf16"
    kern = _get_gemm_kernel(
        bits=int(bits),
        k=k,
        n=n,
        tm=tm,
        tn=tn,
        sgs=sgs,
        has_bias=has_bias,
        out_dtype_name=out_dtype_name,
    )
    nt = (n + tn - 1) // tn
    mt = (m + tm - 1) // tm
    inputs: list[mx.array] = [x_q, x_meta, w_packed, w_meta]
    if has_bias:
        bias_in = mx.contiguous(bias if bias.dtype == eff_out_dtype else bias.astype(eff_out_dtype))
        inputs.append(bias_in)
    out = kern(
        inputs=inputs,
        grid=(nt * sgs * 32, mt, 1),
        threadgroup=(sgs * 32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[eff_out_dtype],
    )[0]
    if len(orig_shape) == 2:
        return out
    return out.reshape(orig_shape[:-1] + (n,))


def get_or_pack_affine_module_weights(
    layer_dict: dict[str, Any],
    cache_key: str,
    mod: Any,
    *,
    bits: int,
    group_size: int = 64,
) -> tuple[mx.array, mx.array]:
    """Lazily pack and cache `(w_packed, w_meta)` on `layer_dict` for a quantized module."""
    cached = layer_dict.get(cache_key)
    if cached is not None:
        return cached
    packed = pack_affine_weights_for_mpp_i8(
        mod.weight,
        mod.scales,
        mod.biases,
        bits=bits,
        group_size=group_size,
    )
    layer_dict[cache_key] = packed
    return packed


__all__ = [
    "get_or_pack_affine_module_weights",
    "mpp_quantized_linear",
    "pack_affine_weights_for_mpp_i8",
    "quantize_activations_i8_k64",
    "should_use_mpp_int8_linear",
    "should_use_mpp_w4a8",
    "should_use_mpp_w8a8",
]
