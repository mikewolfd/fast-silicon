"""Shared Metal 4 MPP BF16 linear and fused GeGLU kernels for Apple Silicon."""

from __future__ import annotations

import mlx.core as mx

from .hw_check import supports_apple10_nax, supports_metal4_bf16
from .metal_mpp import HEADER_MPP

_GEGLU_SPLIT_CACHE: dict[tuple[mx.Dtype, int], object] = {}
_MPP_BF16_LINEAR_CACHE: dict[tuple[int, int, bool], object] = {}
_MPP_BF16_GEGLU_CACHE: dict[tuple[int, int, bool], object] = {}

_MPP_BF16_SUPPORTED: bool | None = None
_APPLE10_NAX: bool | None = None


def _mpp_bf16_supported() -> bool:
    """Cached runtime check for macOS >= 26.1 Metal 4 MPP BF16 tensor operand support."""
    global _MPP_BF16_SUPPORTED
    if _MPP_BF16_SUPPORTED is None:
        _MPP_BF16_SUPPORTED = supports_metal4_bf16()
    return _MPP_BF16_SUPPORTED


def _is_apple10_nax() -> bool:
    """Cached runtime check for Apple10 (M5+) NAX GPU family."""
    global _APPLE10_NAX
    if _APPLE10_NAX is None:
        _APPLE10_NAX = supports_apple10_nax()
    return _APPLE10_NAX


def _ensure_contiguous(arr: mx.array) -> mx.array:
    contig = getattr(mx, "contiguous", None)
    if callable(contig):
        try:
            return contig(arr)
        except TypeError:
            return arr
    return arr


def _default_linear_min_m() -> int:
    # Plain dense linear MPP only wins at large saturated M on M4 (Apple9, M >= 2048);
    # on M5 (Apple10/NAX), native MLX matmul already dispatches to NAX.
    return 4096 if _is_apple10_nax() else 2048


def _default_geglu_min_m() -> int:
    # Fused Gate+Up+GeGLU eliminates intermediate DRAM write/read: wins from M >= 16 on Apple9 (M4),
    # while Apple10 (M5/NAX) prefers saturated >= 64-row tiles before paying pad overhead.
    return 64 if _is_apple10_nax() else 16


def fused_geglu_split(gate_up: mx.array) -> mx.array:
    """Compute `gelu_approx(gate_up[..., :H]) * gate_up[..., H:]` in a single Metal pass."""
    half = gate_up.shape[-1] // 2
    if half % 4 != 0:
        raise ValueError(f"Expected half width divisible by 4, got {half}")
    dt = gate_up.dtype
    if dt not in (mx.bfloat16, mx.float16):
        g = gate_up[..., :half].astype(mx.float32)
        u = gate_up[..., half:].astype(mx.float32)
        t = mx.tanh(0.7978845608 * g * (1.0 + 0.044715 * g * g))
        return (0.5 * g * (1.0 + t) * u).astype(dt)
    gate_up_c = _ensure_contiguous(gate_up)
    rows = gate_up_c.size // (2 * half)
    key = (dt, half)
    kern = _GEGLU_SPLIT_CACHE.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"fs_fused_geglu_H{half}_{'bf16' if dt == mx.bfloat16 else 'f16'}",
            input_names=["gate_up"],
            output_names=["out"],
            source=f"""
                constexpr uint H = {half};
                uint row = thread_position_in_grid.y;
                uint col = thread_position_in_grid.x << 2;
                if (col >= H) return;
                uint base_in = row * (2 * H) + col;
                uint base_out = row * H + col;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {{
                    float g = float(gate_up[base_in + i]);
                    float u = float(gate_up[base_in + H + i]);
                    float t = metal::precise::tanh(0.7978845608f * g * (1.0f + 0.044715f * g * g));
                    out[base_out + i] = static_cast<T>(0.5f * g * (1.0f + t) * u);
                }}
            """,
        )
        _GEGLU_SPLIT_CACHE[key] = kern
    out_shape = gate_up_c.shape[:-1] + (half,)
    gx = half // 4
    tgx = min(256, gx)
    gx_padded = ((gx + tgx - 1) // tgx) * tgx
    return kern(
        inputs=[gate_up_c],
        template=[("T", dt)],
        grid=(gx_padded, rows, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[dt],
    )[0]


def make_mpp_bf16_linear_kernel(tm: int = 32, tn: int = 32, *, relaxed: bool = False):
    """Build or fetch cached Metal 4 MPP::matmul2d BF16 kernel computing `A @ B^T`.

    Uses `relaxed=false` by default; `relaxed=True` is opt-in only.
    """
    key = (tm, tn, bool(relaxed))
    kern = _MPP_BF16_LINEAR_CACHE.get(key)
    if kern is None:
        relaxed_str = "true" if relaxed else "false"
        src = f"""
    int M = (int)A_shape[0];
    int K = (int)A_shape[1];
    int N = (int)B_shape[0];
    int tg_y = (int)(threadgroup_position_in_grid.y * {tm});
    int tg_x = (int)(threadgroup_position_in_grid.x * {tn});
    if (tg_y >= M || tg_x >= N) return;
    auto tA = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(A), dextents<int32_t, 2>(K, M));
    auto tB = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(B), dextents<int32_t, 2>(K, N));
    auto tC = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor({tm}, {tn}, static_cast<int>(dynamic_extent), false, true, {relaxed_str});
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = tA.slice(0, tg_y);
    auto mB = tB.slice(0, tg_x);
    auto mC = tC.slice(tg_x, tg_y);
    op.run(mA, mB, mC);
        """
        suffix = "_rel" if relaxed else ""
        kern = mx.fast.metal_kernel(
            name=f"fs_mpp_mm_bf16_{tm}x{tn}{suffix}",
            input_names=["A", "B"],
            output_names=["C"],
            header=HEADER_MPP,
            source=src,
        )
        _MPP_BF16_LINEAR_CACHE[key] = kern
    return kern


def make_mpp_bf16_geglu_kernel(tm: int = 64, tn: int = 32, *, relaxed: bool = False):
    """Build or fetch cached Metal 4 MPP::matmul2d fused Gate+Up+GeGLU BF16 kernel.

    Uses `relaxed=false` by default; `relaxed=True` is opt-in only.
    """
    key = (tm, tn, bool(relaxed))
    kern = _MPP_BF16_GEGLU_CACHE.get(key)
    if kern is None:
        relaxed_str = "true" if relaxed else "false"
        src = f"""
    int M = (int)A_shape[0];
    int K = (int)A_shape[1];
    int H = (int)Wg_shape[0];
    int tg_y = (int)(threadgroup_position_in_grid.y * {tm});
    int tg_x = (int)(threadgroup_position_in_grid.x * {tn});
    if (tg_y >= M || tg_x >= H) return;
    auto tA  = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(A),  dextents<int32_t, 2>(K, M));
    auto tWg = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(Wg), dextents<int32_t, 2>(K, H));
    auto tWu = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(Wu), dextents<int32_t, 2>(K, H));
    auto tO  = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(Out, dextents<int32_t, 2>(H, M));
    constexpr auto desc = matmul2d_descriptor({tm}, {tn}, static_cast<int>(dynamic_extent), false, true, {relaxed_str});
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = tA.slice(0, tg_y);
    auto mG = tWg.slice(0, tg_x);
    auto mU = tWu.slice(0, tg_x);
    auto mO = tO.slice(tg_x, tg_y);
    auto cG = op.get_destination_cooperative_tensor<decltype(mA), decltype(mG), float>();
    auto cU = op.get_destination_cooperative_tensor<decltype(mA), decltype(mU), float>();
    auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mG), bfloat>();
    #pragma unroll
    for (uint16_t i = 0; i < cG.get_capacity(); ++i) {{
        if (cG.is_valid_element(i)) {{ cG[i] = 0.0f; cU[i] = 0.0f; }}
    }}
    op.run(mA, mG, cG);
    op.run(mA, mU, cU);
    #pragma unroll
    for (uint16_t i = 0; i < cG.get_capacity(); ++i) {{
        if (cG.is_valid_element(i)) {{
            float g = cG[i];
            float u = cU[i];
            float t = metal::precise::tanh(0.7978845608f * g * (1.0f + 0.044715f * g * g));
            cO[i] = static_cast<bfloat>(0.5f * g * (1.0f + t) * u);
        }}
    }}
    cO.store(mO);
        """
        suffix = "_rel" if relaxed else ""
        kern = mx.fast.metal_kernel(
            name=f"fs_mpp_fused_geglu_bf16_{tm}x{tn}{suffix}",
            input_names=["A", "Wg", "Wu"],
            output_names=["Out"],
            header=HEADER_MPP,
            source=src,
        )
        _MPP_BF16_GEGLU_CACHE[key] = kern
    return kern


def mpp_linear_bf16(
    x: mx.array,
    w_nk: mx.array,
    tm: int = 32,
    tn: int = 32,
    *,
    min_m: int | None = None,
) -> mx.array:
    """Compute `x @ w_nk.T` in bfloat16 using Apple Metal 4 `MPP::matmul2d` when eligible.

    Validates macOS >= 26.1 BF16 MPP support, matching `bfloat16` dtypes, `K % 32 == 0`,
    `N % tn == 0`, contiguous storage, and Apple9 vs Apple10 M-threshold gates before
    entering the custom kernel; otherwise falls back cleanly to native `x @ w_nk.T`.
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    M = x.size // K if K > 0 else 0
    N = w_nk.shape[0]
    if M == 0:
        return mx.zeros((*orig_shape[:-1], N), dtype=x.dtype)
    eff_min_m = _default_linear_min_m() if min_m is None else int(min_m)
    if (
        M < eff_min_m
        or not _mpp_bf16_supported()
        or x.dtype != mx.bfloat16
        or w_nk.dtype != mx.bfloat16
        or K < 32
        or (K % 32 != 0)
        or N < tn
        or (N % tn != 0)
    ):
        return x @ w_nk.T
    a2 = _ensure_contiguous(x.reshape(M, K))
    w_c = _ensure_contiguous(w_nk)
    pad_m = (tm - (M % tm)) % tm
    if pad_m:
        a2 = _ensure_contiguous(mx.pad(a2, [(0, pad_m), (0, 0)]))
    Mp = M + pad_m
    kern = make_mpp_bf16_linear_kernel(tm, tn)
    out = kern(
        inputs=[a2, w_c],
        output_shapes=[(Mp, N)],
        output_dtypes=[mx.bfloat16],
        grid=(((N + tn - 1) // tn) * 128, (Mp + tm - 1) // tm, 1),
        threadgroup=(128, 1, 1),
    )[0]
    if pad_m:
        out = out[:M]
    return out.reshape(*orig_shape[:-1], N)


def mpp_fused_geglu_bf16(
    x: mx.array,
    w_g: mx.array,
    w_u: mx.array,
    w_gu: mx.array | None = None,
    m32_cutoff: int = 96,
    *,
    min_m: int | None = None,
) -> mx.array:
    """Adaptive Metal 4 MPP BF16 fused Gate+Up+GeGLU projection (`32x32` up to `m32_cutoff`, `64x32` above).

    Validates macOS >= 26.1 BF16 MPP support, matching `bfloat16` dtypes, `K % 32 == 0`,
    `H % 32 == 0`, contiguous storage, and Apple9 vs Apple10 M-threshold gates before
    entering the custom kernel; otherwise falls back cleanly to `fused_geglu_split` or MLX GeGLU.
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    M = x.size // K if K > 0 else 0
    H = w_g.shape[0]
    if M == 0:
        return mx.zeros((*orig_shape[:-1], H), dtype=x.dtype)
    eff_min_m = _default_geglu_min_m() if min_m is None else int(min_m)
    eligible = (
        M >= eff_min_m
        and _mpp_bf16_supported()
        and x.dtype == mx.bfloat16
        and w_g.dtype == mx.bfloat16
        and w_u.dtype == mx.bfloat16
        and K >= 32
        and (K % 32 == 0)
        and H >= 32
        and (H % 32 == 0)
    )
    if not eligible:
        if H % 4 == 0 and x.dtype in (mx.bfloat16, mx.float16) and w_g.dtype == x.dtype and w_u.dtype == x.dtype:
            if w_gu is not None and w_gu.dtype == x.dtype:
                return fused_geglu_split(x @ w_gu.T)
            return fused_geglu_split(mx.concatenate([x @ w_g.T, x @ w_u.T], axis=-1))
        g = (x @ w_g.T).astype(mx.float32)
        u = (x @ w_u.T).astype(mx.float32)
        t = mx.tanh(0.7978845608 * g * (1.0 + 0.044715 * g * g))
        return (0.5 * g * (1.0 + t) * u).astype(x.dtype)
    tm = 32 if M <= m32_cutoff else 64
    tn = 32
    a2 = _ensure_contiguous(x.reshape(M, K))
    wg_c = _ensure_contiguous(w_g)
    wu_c = _ensure_contiguous(w_u)
    pad_m = (tm - (M % tm)) % tm
    if pad_m:
        a2 = _ensure_contiguous(mx.pad(a2, [(0, pad_m), (0, 0)]))
    Mp = M + pad_m
    kern = make_mpp_bf16_geglu_kernel(tm, tn)
    out = kern(
        inputs=[a2, wg_c, wu_c],
        output_shapes=[(Mp, H)],
        output_dtypes=[mx.bfloat16],
        grid=(((H + tn - 1) // tn) * 128, (Mp + tm - 1) // tm, 1),
        threadgroup=(128, 1, 1),
    )[0]
    if pad_m:
        out = out[:M]
    return out.reshape(*orig_shape[:-1], H)
