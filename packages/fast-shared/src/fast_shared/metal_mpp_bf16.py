"""Shared Metal 4 MPP BF16 linear and fused GeGLU kernels for Apple Silicon."""

from __future__ import annotations

import mlx.core as mx

from .metal_mpp import HEADER_MPP

_GEGLU_SPLIT_CACHE: dict[tuple[mx.Dtype, int], object] = {}
_MPP_BF16_LINEAR_CACHE: dict[tuple[int, int], object] = {}
_MPP_BF16_GEGLU_CACHE: dict[tuple[int, int], object] = {}


def fused_geglu_split(gate_up: mx.array) -> mx.array:
    """Compute `gelu_approx(gate_up[..., :H]) * gate_up[..., H:]` in a single Metal pass."""
    half = gate_up.shape[-1] // 2
    if half % 4 != 0:
        raise ValueError(f"Expected half width divisible by 4, got {half}")
    rows = gate_up.size // (2 * half)
    dt = gate_up.dtype
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
    out_shape = gate_up.shape[:-1] + (half,)
    gx = half // 4
    tgx = min(256, gx)
    gx_padded = ((gx + tgx - 1) // tgx) * tgx
    return kern(
        inputs=[gate_up],
        template=[("T", dt)],
        grid=(gx_padded, rows, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[dt],
    )[0]


def make_mpp_bf16_linear_kernel(tm: int = 32, tn: int = 32):
    """Build or fetch cached Metal 4 MPP::matmul2d BF16 kernel computing `A @ B^T`."""
    key = (tm, tn)
    kern = _MPP_BF16_LINEAR_CACHE.get(key)
    if kern is None:
        src = f"""
    int M = (int)A_shape[0];
    int K = (int)A_shape[1];
    int N = (int)B_shape[0];
    auto tA = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(A), dextents<int32_t, 2>(K, M));
    auto tB = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(B), dextents<int32_t, 2>(K, N));
    auto tC = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor({tm}, {tn}, static_cast<int>(dynamic_extent), false, true, true);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = tA.slice(0, (int)(threadgroup_position_in_grid.y * {tm}));
    auto mB = tB.slice(0, (int)(threadgroup_position_in_grid.x * {tn}));
    auto mC = tC.slice((int)(threadgroup_position_in_grid.x * {tn}), (int)(threadgroup_position_in_grid.y * {tm}));
    op.run(mA, mB, mC);
        """
        kern = mx.fast.metal_kernel(
            name=f"fs_mpp_mm_bf16_{tm}x{tn}",
            input_names=["A", "B"],
            output_names=["C"],
            header=HEADER_MPP,
            source=src,
        )
        _MPP_BF16_LINEAR_CACHE[key] = kern
    return kern


def make_mpp_bf16_geglu_kernel(tm: int = 64, tn: int = 32):
    """Build or fetch cached Metal 4 MPP::matmul2d fused Gate+Up+GeGLU BF16 kernel."""
    key = (tm, tn)
    kern = _MPP_BF16_GEGLU_CACHE.get(key)
    if kern is None:
        src = f"""
    int M = (int)A_shape[0];
    int K = (int)A_shape[1];
    int H = (int)Wg_shape[0];
    auto tA  = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(A),  dextents<int32_t, 2>(K, M));
    auto tWg = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(Wg), dextents<int32_t, 2>(K, H));
    auto tWu = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(const_cast<device bfloat*>(Wu), dextents<int32_t, 2>(K, H));
    auto tO  = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(Out, dextents<int32_t, 2>(H, M));
    constexpr auto desc = matmul2d_descriptor({tm}, {tn}, static_cast<int>(dynamic_extent), false, true, true);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = tA.slice(0, (int)(threadgroup_position_in_grid.y * {tm}));
    auto mG = tWg.slice(0, (int)(threadgroup_position_in_grid.x * {tn}));
    auto mU = tWu.slice(0, (int)(threadgroup_position_in_grid.x * {tn}));
    auto mO = tO.slice((int)(threadgroup_position_in_grid.x * {tn}), (int)(threadgroup_position_in_grid.y * {tm}));
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
        kern = mx.fast.metal_kernel(
            name=f"fs_mpp_fused_geglu_bf16_{tm}x{tn}",
            input_names=["A", "Wg", "Wu"],
            output_names=["Out"],
            header=HEADER_MPP,
            source=src,
        )
        _MPP_BF16_GEGLU_CACHE[key] = kern
    return kern


def mpp_linear_bf16(x: mx.array, w_nk: mx.array, tm: int = 32, tn: int = 32) -> mx.array:
    """Compute `x @ w_nk.T` in bfloat16 using Apple Metal 4 `MPP::matmul2d`."""
    orig_shape = x.shape
    K = orig_shape[-1]
    M = x.size // K
    N = w_nk.shape[0]
    if M < 16 or (N % tn != 0):
        return x @ w_nk.T
    a2 = x.reshape(M, K)
    pad_m = (tm - (M % tm)) % tm
    if pad_m:
        a2 = mx.pad(a2, [(0, pad_m), (0, 0)])
    Mp = M + pad_m
    kern = make_mpp_bf16_linear_kernel(tm, tn)
    out = kern(
        inputs=[a2, w_nk],
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
) -> mx.array:
    """Adaptive Metal 4 MPP BF16 fused Gate+Up+GeGLU projection (`32x32` up to `m32_cutoff`, `64x32` above)."""
    orig_shape = x.shape
    K = orig_shape[-1]
    M = x.size // K
    H = w_g.shape[0]
    if M == 0:
        return mx.zeros((*orig_shape[:-1], H), dtype=x.dtype)
    if M < 16 or (H % 32 != 0):
        if H % 4 == 0:
            if w_gu is not None:
                return fused_geglu_split(x @ w_gu.T)
            return fused_geglu_split(mx.concatenate([x @ w_g.T, x @ w_u.T], axis=-1))
        g = (x @ w_g.T).astype(mx.float32)
        u = (x @ w_u.T).astype(mx.float32)
        t = mx.tanh(0.7978845608 * g * (1.0 + 0.044715 * g * g))
        return (0.5 * g * (1.0 + t) * u).astype(x.dtype)
    tm = 32 if M <= m32_cutoff else 64
    tn = 32
    a2 = x.reshape(M, K)
    pad_m = (tm - (M % tm)) % tm
    if pad_m:
        a2 = mx.pad(a2, [(0, pad_m), (0, 0)])
    Mp = M + pad_m
    kern = make_mpp_bf16_geglu_kernel(tm, tn)
    out = kern(
        inputs=[a2, w_g, w_u],
        output_shapes=[(Mp, H)],
        output_dtypes=[mx.bfloat16],
        grid=(((H + tn - 1) // tn) * 128, (Mp + tm - 1) // tm, 1),
        threadgroup=(128, 1, 1),
    )[0]
    if pad_m:
        out = out[:M]
    return out.reshape(*orig_shape[:-1], H)
