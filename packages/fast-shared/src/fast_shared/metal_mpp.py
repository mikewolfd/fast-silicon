"""Shared Metal standard headers and Apple Metal 4 MPP::matmul2d kernel generator."""

from __future__ import annotations

import mlx.core as mx

HEADER_STD = """#include <metal_stdlib>
using namespace metal;
"""

HEADER_SIMDGROUP = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

HEADER_MPP = """#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

inline float exact_erf(float x) {
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return x >= 0.0f ? y : -y;
}
"""


def make_mpp_matmul_kernel(
    name: str,
    *,
    epilogue: str = "none",
    tm: int = 64,
    tn: int = 64,
    sgs: int = 4,
    relaxed: bool = False,
):
    """Build an `mx.fast.metal_kernel` backed by Metal 4 `MPP::matmul2d` (`A[M,K] @ B[N,K]^T -> C[M,N]`).

    Uses `relaxed=false` by default per the Metal 4 MPP precision playbook; callers may opt into
    `relaxed=True` only when explicitly benchmarked and parity-verified on the target chip.
    """
    relaxed_str = "true" if relaxed else "false"
    if epilogue == "none":
        inputs = ["A", "B"]
        n_expr = "N"
        b_cols_expr = "N"
        ep_body = "op.run(mA, mB, mC);"
    elif epilogue == "bias":
        inputs = ["A", "B", "Bias"]
        n_expr = "N"
        b_cols_expr = "N"
        ep_body = """
        auto cF = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
        #pragma unroll
        for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
            if (cF.is_valid_element(i)) cF[i] = 0.0f;
        }
        op.run(mA, mB, cF);
        #pragma unroll
        for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
            if (cF.is_valid_element(i)) {
                auto idx = cF.get_multidimensional_index(i);
                int r = tg_y + (int)idx[1];
                int c = tg_x + (int)idx[0];
                if (r < M && c < N) {
                    cO[i] = static_cast<half>(cF[i] + (float)Bias[c]);
                }
            }
        }
        cO.store(mC);
        """
    elif epilogue == "bias_res":
        inputs = ["A", "B", "Bias", "Res"]
        n_expr = "N"
        b_cols_expr = "N"
        ep_body = """
        auto cF = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
        #pragma unroll
        for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
            if (cF.is_valid_element(i)) cF[i] = 0.0f;
        }
        op.run(mA, mB, cF);
        #pragma unroll
        for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
            if (cF.is_valid_element(i)) {
                auto idx = cF.get_multidimensional_index(i);
                int r = tg_y + (int)idx[1];
                int c = tg_x + (int)idx[0];
                if (r < M && c < N) {
                    cO[i] = static_cast<half>(cF[i] + (float)Bias[c] + (float)Res[r * N + c]);
                }
            }
        }
        cO.store(mC);
        """
    elif epilogue == "bias_gelu":
        inputs = ["A", "B", "Bias"]
        n_expr = "N"
        b_cols_expr = "N"
        ep_body = """
        auto cF = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
        #pragma unroll
        for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
            if (cF.is_valid_element(i)) cF[i] = 0.0f;
        }
        op.run(mA, mB, cF);
        const float inv_s2 = 0.7071067811865475f;
        #pragma unroll
        for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
            if (cF.is_valid_element(i)) {
                auto idx = cF.get_multidimensional_index(i);
                int r = tg_y + (int)idx[1];
                int c = tg_x + (int)idx[0];
                if (r < M && c < N) {
                    float v = cF[i] + (float)Bias[c];
                    cO[i] = static_cast<half>(v * 0.5f * (1.0f + exact_erf(v * inv_s2)));
                }
            }
        }
        cO.store(mC);
        """
    elif epilogue == "geglu":
        inputs = ["A", "B"]
        src = f"""
        int M = (int)A_shape[0], K = (int)A_shape[1], N_tot = (int)B_shape[0];
        int N_half = N_tot >> 1;
        int tg_y = (int)(threadgroup_position_in_grid.y * {tm});
        int tg_x = (int)(threadgroup_position_in_grid.x * {tn});
        if (tg_y >= M || tg_x >= N_half) return;
        int tile_m = min({tm}, M - tg_y);
        int tile_n = min({tn}, N_half - tg_x);
        auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(const_cast<device half*>(A), dextents<int32_t, 2>(K, M));
        auto tG = tensor<device half, dextents<int32_t, 2>, tensor_inline>(const_cast<device half*>(B), dextents<int32_t, 2>(K, N_half));
        auto tU = tensor<device half, dextents<int32_t, 2>, tensor_inline>(const_cast<device half*>(B) + (size_t)N_half * (size_t)K, dextents<int32_t, 2>(K, N_half));
        auto tC = tensor<device half, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(N_half, M));
        constexpr auto desc = matmul2d_descriptor({tm}, {tn}, static_cast<int>(dynamic_extent), false, true, {relaxed_str});
        matmul2d<desc, execution_simdgroups<{sgs}>> op;
        auto mA = tA.slice(0, tg_y);
        auto mG = tG.slice(0, tg_x);
        auto mU = tU.slice(0, tg_x);
        auto mC = tC.slice(tg_x, tg_y);
        auto cG = op.get_destination_cooperative_tensor<decltype(mA), decltype(mG), float>();
        auto cU = op.get_destination_cooperative_tensor<decltype(mA), decltype(mU), float>();
        auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mG), half>();
        #pragma unroll
        for (uint16_t i = 0; i < cG.get_capacity(); ++i) {{
            if (cG.is_valid_element(i)) {{ cG[i] = 0.0f; cU[i] = 0.0f; }}
        }}
        op.run(mA, mG, cG);
        op.run(mA, mU, cU);
        const float inv_s2 = 0.7071067811865475f;
        #pragma unroll
        for (uint16_t i = 0; i < cG.get_capacity(); ++i) {{
            if (cG.is_valid_element(i)) {{
                auto idx = cG.get_multidimensional_index(i);
                if ((int)idx[1] < tile_m && (int)idx[0] < tile_n) {{
                    float g = cG[i];
                    float u = cU[i];
                    cO[i] = static_cast<half>(0.5f * g * (1.0f + exact_erf(g * inv_s2)) * u);
                }}
            }}
        }}
        cO.store(mC);
        """
        return mx.fast.metal_kernel(name=name, input_names=inputs, output_names=["C"], header=HEADER_MPP, source=src)
    elif epilogue == "gelu_mul_pli":
        inputs = ["A", "B", "PLI"]
        n_expr = "N"
        b_cols_expr = "N"
        ep_body = """
        auto cF = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
        #pragma unroll
        for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
            if (cF.is_valid_element(i)) cF[i] = 0.0f;
        }
        op.run(mA, mB, cF);
        const float inv_s2 = 0.7071067811865475f;
        #pragma unroll
        for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
            if (cF.is_valid_element(i)) {
                auto idx = cF.get_multidimensional_index(i);
                int r = tg_y + (int)idx[1];
                int c = tg_x + (int)idx[0];
                if (r < M && c < N) {
                    float g = cF[i];
                    float p = (float)PLI[r * N + c];
                    cO[i] = static_cast<half>(0.5f * g * (1.0f + exact_erf(g * inv_s2)) * p);
                }
            }
        }
        cO.store(mC);
        """
    else:
        raise ValueError(f"Unknown MPP epilogue {epilogue!r}")

    src = f"""
    int M = (int)A_shape[0], K = (int)A_shape[1], N = (int)B_shape[0];
    int tg_y = (int)(threadgroup_position_in_grid.y * {tm});
    int tg_x = (int)(threadgroup_position_in_grid.x * {tn});
    if (tg_y >= M || tg_x >= {n_expr}) return;
    auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(const_cast<device half*>(A), dextents<int32_t, 2>(K, M));
    auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(const_cast<device half*>(B), dextents<int32_t, 2>(K, {b_cols_expr}));
    auto tC = tensor<device half, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>({n_expr}, M));
    constexpr auto desc = matmul2d_descriptor({tm}, {tn}, static_cast<int>(dynamic_extent), false, true, {relaxed_str});
    matmul2d<desc, execution_simdgroups<{sgs}>> op;
    auto mA = tA.slice(0, tg_y);
    auto mB = tB.slice(0, tg_x);
    auto mC = tC.slice(tg_x, tg_y);
    {ep_body}
    """
    return mx.fast.metal_kernel(name=name, input_names=inputs, output_names=["C"], header=HEADER_MPP, source=src)
