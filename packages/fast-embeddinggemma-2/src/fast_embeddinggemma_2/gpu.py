"""Metal 4 MPP + custom fused QKNorm/RoPE + packed SDPA GPU backend for EmbeddingGemma-2."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from fast_shared.metal_mpp import HEADER_STD, make_mpp_matmul_kernel

NUM_LAYERS = 24
HIDDEN_SIZE = 512
EMBED_DIM = 768
MAX_SEQ_LEN = 2048
FULL_LAYERS = frozenset({5, 11, 17, 23})

# Measured ms cost per ANE lane capacity for cost-aware multi-bucket packing (~96.3-96.5% fill).
ANE_MS_COST: dict[int, float] = {
    128: 4.503,
    192: 7.731,
    256: 9.855,
    320: 13.231,
    384: 16.604,
    448: 20.151,
    512: 24.785,
}

GPU_B_OPTS: dict[int, tuple[int, ...]] = {
    128: (32, 16, 4, 1),
    192: (16, 4, 1),
    256: (16, 8, 2, 1),
    320: (12, 4, 1),
    384: (10, 6, 2, 1),
    448: (8, 4, 1),
    512: (8, 4, 2, 1),
}

OVER_BUCKETS: tuple[int, ...] = (640, 768, 1024, 1536, 2048)
OVER_B_OPTS: dict[int, tuple[int, ...]] = {
    640: (4, 2, 1),
    768: (4, 2, 1),
    1024: (2, 1),
    1536: (1,),
    2048: (1,),
}

_MPP_64 = make_mpp_matmul_kernel("emb2_mpp64", tm=64, tn=64, sgs=4, epilogue="none")
_MPP_128 = make_mpp_matmul_kernel("emb2_mpp128", tm=128, tn=64, sgs=8, epilogue="none")
_MPP_GEGLU_64 = make_mpp_matmul_kernel("emb2_geglu64", tm=64, tn=64, sgs=4, epilogue="geglu")
_MPP_PG_64 = make_mpp_matmul_kernel("emb2_pg64", tm=64, tn=64, sgs=4, epilogue="gelu_mul_pli")

# Fused QKNorm + VNorm + RoPE emitting directly in [B, H, L, D] layout for SDPA.
_QKNORM_ROPE_256_BHLD = mx.fast.metal_kernel(
    name="emb2_qknr_256_bhld",
    input_names=["QKV", "Wq", "Wk", "C", "S"],
    output_names=["Q", "K", "V"],
    header=HEADER_STD,
    source="""
    uint row = threadgroup_position_in_grid.x;
    uint sg = thread_position_in_threadgroup.x >> 5;
    uint lane = thread_position_in_threadgroup.x & 31;
    uint B = (uint)C_shape[0];
    uint L = (uint)C_shape[1];
    uint M = B * L;
    if (row >= M) return;
    uint b = row / L;
    uint l = row - b * L;
    const device half* sp = QKV + row * 2048 + sg * 256;
    float4 x_lo = float4(*(const device half4*)(sp + lane * 4));
    float4 x_hi = float4(*(const device half4*)(sp + 128 + lane * 4));
    float inv = rsqrt(simd_sum(dot(x_lo, x_lo) + dot(x_hi, x_hi)) * (1.0f / 256.0f) + 1e-6f);
    if (sg >= 6) {
        uint kh = sg - 6;
        device half* vp = V + ((b * 2 + kh) * L + l) * 256;
        *(device half4*)(vp + lane * 4)       = half4(x_lo * inv);
        *(device half4*)(vp + 128 + lane * 4) = half4(x_hi * inv);
        return;
    }
    const device half* wp = (sg < 4) ? Wq : Wk;
    float4 n_lo = x_lo * inv * float4(*(const device half4*)(wp + lane * 4));
    float4 n_hi = x_hi * inv * float4(*(const device half4*)(wp + 128 + lane * 4));
    float4 c4 = float4(*(const device half4*)(C + row * 128 + lane * 4));
    float4 s4 = float4(*(const device half4*)(S + row * 128 + lane * 4));
    device half* dst = (sg < 4)
        ? (Q + ((b * 4 + sg) * L + l) * 256)
        : (K + ((b * 2 + (sg - 4)) * L + l) * 256);
    *(device half4*)(dst + lane * 4)       = half4(n_lo * c4 - n_hi * s4);
    *(device half4*)(dst + 128 + lane * 4) = half4(n_hi * c4 + n_lo * s4);
""",
)

_QKNORM_ROPE_512_BHLD = mx.fast.metal_kernel(
    name="emb2_qknr_512_bhld",
    input_names=["QKV", "Wq", "Wk", "C", "S"],
    output_names=["Q", "K", "V"],
    header=HEADER_STD,
    source="""
    uint row = threadgroup_position_in_grid.x;
    uint sg = thread_position_in_threadgroup.x >> 5;
    uint lane = thread_position_in_threadgroup.x & 31;
    uint B = (uint)C_shape[0];
    uint L = (uint)C_shape[1];
    uint M = B * L;
    if (row >= M) return;
    uint b = row / L;
    uint l = row - b * L;
    const device half* sp = QKV + row * 3072 + sg * 512;
    float4 xl0 = float4(*(const device half4*)(sp + lane * 8));
    float4 xl1 = float4(*(const device half4*)(sp + lane * 8 + 4));
    float4 xh0 = float4(*(const device half4*)(sp + 256 + lane * 8));
    float4 xh1 = float4(*(const device half4*)(sp + 256 + lane * 8 + 4));
    float inv = rsqrt(simd_sum(dot(xl0,xl0)+dot(xl1,xl1)+dot(xh0,xh0)+dot(xh1,xh1)) * (1.0f / 512.0f) + 1e-6f);
    if (sg == 5) {
        device half* vp = V + (b * L + l) * 512;
        *(device half4*)(vp + lane * 8)           = half4(xl0 * inv);
        *(device half4*)(vp + lane * 8 + 4)       = half4(xl1 * inv);
        *(device half4*)(vp + 256 + lane * 8)     = half4(xh0 * inv);
        *(device half4*)(vp + 256 + lane * 8 + 4) = half4(xh1 * inv);
        return;
    }
    const device half* wp = (sg < 4) ? Wq : Wk;
    float4 nl0 = xl0 * inv * float4(*(const device half4*)(wp + lane * 8));
    float4 nl1 = xl1 * inv * float4(*(const device half4*)(wp + lane * 8 + 4));
    float4 nh0 = xh0 * inv * float4(*(const device half4*)(wp + 256 + lane * 8));
    float4 nh1 = xh1 * inv * float4(*(const device half4*)(wp + 256 + lane * 8 + 4));
    float4 c0 = float4(*(const device half4*)(C + row * 256 + lane * 8));
    float4 c1 = float4(*(const device half4*)(C + row * 256 + lane * 8 + 4));
    float4 s0 = float4(*(const device half4*)(S + row * 256 + lane * 8));
    float4 s1 = float4(*(const device half4*)(S + row * 256 + lane * 8 + 4));
    device half* dst = (sg < 4) ? (Q + ((b * 4 + sg) * L + l) * 512) : (K + (b * L + l) * 512);
    *(device half4*)(dst + lane * 8)           = half4(nl0 * c0 - nh0 * s0);
    *(device half4*)(dst + lane * 8 + 4)       = half4(nl1 * c1 - nh1 * s1);
    *(device half4*)(dst + 256 + lane * 8)     = half4(nh0 * c0 + nl0 * s0);
    *(device half4*)(dst + 256 + lane * 8 + 4) = half4(nh1 * c1 + nl1 * s1);
""",
)


_FUSED_GEMMA_RES_RMS_SRC = """
    uint sg = thread_position_in_threadgroup.x >> 5;
    uint lane = thread_position_in_threadgroup.x & 31;
    uint row = threadgroup_position_in_grid.x * 8 + sg;
    if (row >= (uint)H_shape[0]) return;
    const device half* sp = Sub + row * 512;
    float4 f0 = float4(*(const device half4*)(sp + lane * 4));
    float4 f1 = float4(*(const device half4*)(sp + 128 + lane * 4));
    float4 f2 = float4(*(const device half4*)(sp + 256 + lane * 4));
    float4 f3 = float4(*(const device half4*)(sp + 384 + lane * 4));
    float inv1 = rsqrt(simd_sum(dot(f0, f0) + dot(f1, f1) + dot(f2, f2) + dot(f3, f3)) * (1.0f / 512.0f) + 1e-6f);
    const device half* hp = H + row * 512;
    float4 r0 = float4(*(const device half4*)(hp + lane * 4))       + f0 * inv1 * float4(*(const device half4*)(W_post + lane * 4));
    float4 r1 = float4(*(const device half4*)(hp + 128 + lane * 4)) + f1 * inv1 * float4(*(const device half4*)(W_post + 128 + lane * 4));
    float4 r2 = float4(*(const device half4*)(hp + 256 + lane * 4)) + f2 * inv1 * float4(*(const device half4*)(W_post + 256 + lane * 4));
    float4 r3 = float4(*(const device half4*)(hp + 384 + lane * 4)) + f3 * inv1 * float4(*(const device half4*)(W_post + 384 + lane * 4));
    device half* yrp = Y_res + row * 512;
    *(device half4*)(yrp + lane * 4)       = half4(r0);
    *(device half4*)(yrp + 128 + lane * 4) = half4(r1);
    *(device half4*)(yrp + 256 + lane * 4) = half4(r2);
    *(device half4*)(yrp + 384 + lane * 4) = half4(r3);
    float inv2 = rsqrt(simd_sum(dot(r0, r0) + dot(r1, r1) + dot(r2, r2) + dot(r3, r3)) * (1.0f / 512.0f) + 1e-6f);
    device half* ynp = Y_norm + row * 512;
    *(device half4*)(ynp + lane * 4)       = half4(r0 * inv2 * float4(*(const device half4*)(W_next + lane * 4)));
    *(device half4*)(ynp + 128 + lane * 4) = half4(r1 * inv2 * float4(*(const device half4*)(W_next + 128 + lane * 4)));
    *(device half4*)(ynp + 256 + lane * 4) = half4(r2 * inv2 * float4(*(const device half4*)(W_next + 256 + lane * 4)));
    *(device half4*)(ynp + 384 + lane * 4) = half4(r3 * inv2 * float4(*(const device half4*)(W_next + 384 + lane * 4)));
"""
_FUSED_RES_RMS_KERN = mx.fast.metal_kernel(
    name="emb2_fused_res_rms_512_f16",
    input_names=["H", "Sub", "W_post", "W_next"],
    output_names=["Y_res", "Y_norm"],
    header=HEADER_STD,
    source=_FUSED_GEMMA_RES_RMS_SRC,
)

_FUSED_RES_ADD_SRC = """
    uint sg = thread_position_in_threadgroup.x >> 5;
    uint lane = thread_position_in_threadgroup.x & 31;
    uint row = threadgroup_position_in_grid.x * 8 + sg;
    if (row >= (uint)H_shape[0]) return;
    const device half* sp = Sub + row * 512;
    float4 f0 = float4(*(const device half4*)(sp + lane * 4));
    float4 f1 = float4(*(const device half4*)(sp + 128 + lane * 4));
    float4 f2 = float4(*(const device half4*)(sp + 256 + lane * 4));
    float4 f3 = float4(*(const device half4*)(sp + 384 + lane * 4));
    float inv1 = rsqrt(simd_sum(dot(f0, f0) + dot(f1, f1) + dot(f2, f2) + dot(f3, f3)) * (1.0f / 512.0f) + 1e-6f);
    const device half* hp = H + row * 512;
    device half* yrp = Y_res + row * 512;
    *(device half4*)(yrp + lane * 4)       = half4(float4(*(const device half4*)(hp + lane * 4))       + f0 * inv1 * float4(*(const device half4*)(W_post + lane * 4)));
    *(device half4*)(yrp + 128 + lane * 4) = half4(float4(*(const device half4*)(hp + 128 + lane * 4)) + f1 * inv1 * float4(*(const device half4*)(W_post + 128 + lane * 4)));
    *(device half4*)(yrp + 256 + lane * 4) = half4(float4(*(const device half4*)(hp + 256 + lane * 4)) + f2 * inv1 * float4(*(const device half4*)(W_post + 256 + lane * 4)));
    *(device half4*)(yrp + 384 + lane * 4) = half4(float4(*(const device half4*)(hp + 384 + lane * 4)) + f3 * inv1 * float4(*(const device half4*)(W_post + 384 + lane * 4)));
"""
_FUSED_RES_ADD_KERN = mx.fast.metal_kernel(
    name="emb2_fused_res_add_512_f16",
    input_names=["H", "Sub", "W_post"],
    output_names=["Y_res"],
    header=HEADER_STD,
    source=_FUSED_RES_ADD_SRC,
)

_FUSED_RES_RMS_SCALAR_RMS_SRC = """
    uint sg = thread_position_in_threadgroup.x >> 5;
    uint lane = thread_position_in_threadgroup.x & 31;
    uint row = threadgroup_position_in_grid.x * 8 + sg;
    if (row >= (uint)H_shape[0]) return;
    const device half* sp = Sub + row * 512;
    float4 f0 = float4(*(const device half4*)(sp + lane * 4));
    float4 f1 = float4(*(const device half4*)(sp + 128 + lane * 4));
    float4 f2 = float4(*(const device half4*)(sp + 256 + lane * 4));
    float4 f3 = float4(*(const device half4*)(sp + 384 + lane * 4));
    float inv1 = rsqrt(simd_sum(dot(f0, f0) + dot(f1, f1) + dot(f2, f2) + dot(f3, f3)) * (1.0f / 512.0f) + 1e-6f);
    float sc = float(Scalar[0]);
    const device half* hp = H + row * 512;
    float4 r0 = (float4(*(const device half4*)(hp + lane * 4))       + f0 * inv1 * float4(*(const device half4*)(W_post + lane * 4))) * sc;
    float4 r1 = (float4(*(const device half4*)(hp + 128 + lane * 4)) + f1 * inv1 * float4(*(const device half4*)(W_post + 128 + lane * 4))) * sc;
    float4 r2 = (float4(*(const device half4*)(hp + 256 + lane * 4)) + f2 * inv1 * float4(*(const device half4*)(W_post + 256 + lane * 4))) * sc;
    float4 r3 = (float4(*(const device half4*)(hp + 384 + lane * 4)) + f3 * inv1 * float4(*(const device half4*)(W_post + 384 + lane * 4))) * sc;
    device half* yrp = Y_res + row * 512;
    *(device half4*)(yrp + lane * 4)       = half4(r0);
    *(device half4*)(yrp + 128 + lane * 4) = half4(r1);
    *(device half4*)(yrp + 256 + lane * 4) = half4(r2);
    *(device half4*)(yrp + 384 + lane * 4) = half4(r3);
    float inv2 = rsqrt(simd_sum(dot(r0, r0) + dot(r1, r1) + dot(r2, r2) + dot(r3, r3)) * (1.0f / 512.0f) + 1e-6f);
    device half* ynp = Y_norm + row * 512;
    *(device half4*)(ynp + lane * 4)       = half4(r0 * inv2 * float4(*(const device half4*)(W_next + lane * 4)));
    *(device half4*)(ynp + 128 + lane * 4) = half4(r1 * inv2 * float4(*(const device half4*)(W_next + 128 + lane * 4)));
    *(device half4*)(ynp + 256 + lane * 4) = half4(r2 * inv2 * float4(*(const device half4*)(W_next + 256 + lane * 4)));
    *(device half4*)(ynp + 384 + lane * 4) = half4(r3 * inv2 * float4(*(const device half4*)(W_next + 384 + lane * 4)));
"""
_FUSED_RES_RMS_SC_KERN = mx.fast.metal_kernel(
    name="emb2_fused_res_rms_sc_512_f16",
    input_names=["H", "Sub", "W_post", "Scalar", "W_next"],
    output_names=["Y_res", "Y_norm"],
    header=HEADER_STD,
    source=_FUSED_RES_RMS_SCALAR_RMS_SRC,
)

_FUSED_RES_ADD_SCALAR_SRC = """
    uint sg = thread_position_in_threadgroup.x >> 5;
    uint lane = thread_position_in_threadgroup.x & 31;
    uint row = threadgroup_position_in_grid.x * 8 + sg;
    if (row >= (uint)H_shape[0]) return;
    const device half* sp = Sub + row * 512;
    float4 f0 = float4(*(const device half4*)(sp + lane * 4));
    float4 f1 = float4(*(const device half4*)(sp + 128 + lane * 4));
    float4 f2 = float4(*(const device half4*)(sp + 256 + lane * 4));
    float4 f3 = float4(*(const device half4*)(sp + 384 + lane * 4));
    float inv1 = rsqrt(simd_sum(dot(f0, f0) + dot(f1, f1) + dot(f2, f2) + dot(f3, f3)) * (1.0f / 512.0f) + 1e-6f);
    float sc = float(Scalar[0]);
    const device half* hp = H + row * 512;
    device half* yrp = Y_res + row * 512;
    *(device half4*)(yrp + lane * 4)       = half4((float4(*(const device half4*)(hp + lane * 4))       + f0 * inv1 * float4(*(const device half4*)(W_post + lane * 4))) * sc);
    *(device half4*)(yrp + 128 + lane * 4) = half4((float4(*(const device half4*)(hp + 128 + lane * 4)) + f1 * inv1 * float4(*(const device half4*)(W_post + 128 + lane * 4))) * sc);
    *(device half4*)(yrp + 256 + lane * 4) = half4((float4(*(const device half4*)(hp + 256 + lane * 4)) + f2 * inv1 * float4(*(const device half4*)(W_post + 256 + lane * 4))) * sc);
    *(device half4*)(yrp + 384 + lane * 4) = half4((float4(*(const device half4*)(hp + 384 + lane * 4)) + f3 * inv1 * float4(*(const device half4*)(W_post + 384 + lane * 4))) * sc);
"""
_FUSED_RES_ADD_SC_KERN = mx.fast.metal_kernel(
    name="emb2_fused_res_add_sc_512_f16",
    input_names=["H", "Sub", "W_post", "Scalar"],
    output_names=["Y_res"],
    header=HEADER_STD,
    source=_FUSED_RES_ADD_SCALAR_SRC,
)

_SEG_MEAN_POOL_512_SRC = """
    uint doc = threadgroup_position_in_grid.x;
    if (doc >= (uint)SegStarts_shape[0]) return;
    uint tid = thread_position_in_threadgroup.x;
    int start = SegStarts[doc];
    int len = SegLens[doc];
    int n_pos = len > 0 ? len : 1;
    float4 acc = float4(0.0f);
    const device half* base = Ho + (size_t)start * 512 + tid * 4;
    for (int t = 0; t < len; ++t) {
        acc += float4(*(const device half4*)(base + (size_t)t * 512));
    }
    device half* outp = Out + (size_t)doc * 512 + tid * 4;
    *(device half4*)outp = half4(acc / float(n_pos));
"""
_SEG_MEAN_POOL_512_KERN = mx.fast.metal_kernel(
    name="emb2_seg_mean_pool_512_f16",
    input_names=["Ho", "SegStarts", "SegLens"],
    output_names=["Out"],
    header=HEADER_STD,
    source=_SEG_MEAN_POOL_512_SRC,
)


def mean_pool_segments(
    ho: mx.array,
    seg_starts: Sequence[int] | np.ndarray | mx.array,
    seg_lens: Sequence[int] | np.ndarray | mx.array,
) -> mx.array:
    """Mean-pool variable-length token spans from `[M, 512]` FP16 hidden states into `[n_docs, 512]` FP32."""
    if ho.ndim != 2 or int(ho.shape[1]) != HIDDEN_SIZE:
        raise ValueError(f"Expected ho shape (M, {HIDDEN_SIZE}), got {ho.shape}")
    if ho.dtype != mx.float16:
        raise TypeError(f"Expected float16 ho for mean_pool_segments, got {ho.dtype}")
    starts_mx = seg_starts if isinstance(seg_starts, mx.array) else mx.array(seg_starts, dtype=mx.int32)
    lens_mx = seg_lens if isinstance(seg_lens, mx.array) else mx.array(seg_lens, dtype=mx.int32)
    if starts_mx.dtype != mx.int32:
        starts_mx = starts_mx.astype(mx.int32)
    if lens_mx.dtype != mx.int32:
        lens_mx = lens_mx.astype(mx.int32)
    n_docs = int(starts_mx.shape[0])
    if n_docs == 0:
        return mx.zeros((0, HIDDEN_SIZE), dtype=mx.float32)
    if int(lens_mx.shape[0]) != n_docs:
        raise ValueError(f"Mismatched seg_starts ({n_docs}) vs seg_lens ({int(lens_mx.shape[0])})")
    out_f16 = _SEG_MEAN_POOL_512_KERN(
        inputs=[ho, starts_mx, lens_mx],
        grid=(n_docs * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_docs, HIDDEN_SIZE)],
        output_dtypes=[mx.float16],
    )[0]
    return out_f16.astype(mx.float32)


def fused_gemma_res_rms(
    h: mx.array,
    sub: mx.array,
    w_post: mx.array,
    w_next: mx.array | None = None,
    *,
    scalar: mx.array | None = None,
) -> mx.array | tuple[mx.array, mx.array]:
    """Fused FP16 `Y_res = (H + rms_norm(Sub, W_post)) [* Scalar]` and optional `Y_norm = rms_norm(Y_res, W_next)`."""
    if HIDDEN_SIZE != 512:
        raise ValueError(f"Fused residual RMSNorm Metal kernels require HIDDEN_SIZE=512, got {HIDDEN_SIZE}")
    if h.ndim < 2 or int(h.shape[-1]) != HIDDEN_SIZE:
        raise ValueError(f"Expected last dim D={HIDDEN_SIZE}, got shape {h.shape}")
    if sub.shape != h.shape:
        raise ValueError(f"Expected sub shape {h.shape}, got {sub.shape}")
    if w_post.shape != (HIDDEN_SIZE,):
        raise ValueError(f"Expected w_post shape ({HIDDEN_SIZE},), got {w_post.shape}")
    if w_next is not None and w_next.shape != (HIDDEN_SIZE,):
        raise ValueError(f"Expected w_next shape ({HIDDEN_SIZE},), got {w_next.shape}")
    if scalar is not None and (scalar.ndim != 1 or int(scalar.size) != 1):
        raise ValueError(f"Expected 1D scalar shape (1,) for Metal pointer binding, got {scalar.shape}")
    if h.dtype != mx.float16 or sub.dtype != mx.float16 or w_post.dtype != mx.float16:
        raise TypeError(
            f"Expected float16 inputs for fused_gemma_res_rms, got h={h.dtype}, sub={sub.dtype}, w_post={w_post.dtype}"
        )
    if w_next is not None and w_next.dtype != mx.float16:
        raise TypeError(f"Expected float16 w_next, got {w_next.dtype}")
    if scalar is not None and scalar.dtype != mx.float16:
        raise TypeError(f"Expected float16 scalar, got {scalar.dtype}")

    orig_shape = tuple(int(d) for d in h.shape)
    m_rows = int(h.size) // HIDDEN_SIZE
    if m_rows == 0:
        empty = mx.zeros(orig_shape, dtype=mx.float16)
        return empty if w_next is None else (empty, mx.zeros(orig_shape, dtype=mx.float16))

    h2d = h if h.ndim == 2 else h.reshape(m_rows, HIDDEN_SIZE)
    sub2d = sub if sub.ndim == 2 else sub.reshape(m_rows, HIDDEN_SIZE)
    grid = (((m_rows + 7) // 8) * 256, 1, 1)
    tg = (256, 1, 1)

    if scalar is not None and w_next is not None:
        y_res, y_norm = _FUSED_RES_RMS_SC_KERN(
            inputs=[h2d, sub2d, w_post, scalar, w_next],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(m_rows, HIDDEN_SIZE), (m_rows, HIDDEN_SIZE)],
            output_dtypes=[mx.float16, mx.float16],
        )
        if len(orig_shape) != 2:
            y_res = y_res.reshape(orig_shape)
            y_norm = y_norm.reshape(orig_shape)
        return y_res, y_norm

    if scalar is not None:
        y_res = _FUSED_RES_ADD_SC_KERN(
            inputs=[h2d, sub2d, w_post, scalar],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(m_rows, HIDDEN_SIZE)],
            output_dtypes=[mx.float16],
        )[0]
        return y_res if len(orig_shape) == 2 else y_res.reshape(orig_shape)

    if w_next is not None:
        y_res, y_norm = _FUSED_RES_RMS_KERN(
            inputs=[h2d, sub2d, w_post, w_next],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(m_rows, HIDDEN_SIZE), (m_rows, HIDDEN_SIZE)],
            output_dtypes=[mx.float16, mx.float16],
        )
        if len(orig_shape) != 2:
            y_res = y_res.reshape(orig_shape)
            y_norm = y_norm.reshape(orig_shape)
        return y_res, y_norm

    y_res = _FUSED_RES_ADD_KERN(
        inputs=[h2d, sub2d, w_post],
        grid=grid,
        threadgroup=tg,
        output_shapes=[(m_rows, HIDDEN_SIZE)],
        output_dtypes=[mx.float16],
    )[0]
    return y_res if len(orig_shape) == 2 else y_res.reshape(orig_shape)


def make_rope_tables(max_l: int = MAX_SEQ_LEN) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    pos = mx.arange(max_l, dtype=mx.float32)[:, None]
    f256 = 1.0 / (10000.0 ** (mx.arange(0, 256, 2, dtype=mx.float32) / 256.0))
    f512 = 1.0 / (1000000.0 ** (mx.arange(0, 512, 2, dtype=mx.float32) / 512.0))
    c256, s256 = mx.cos(pos * f256).astype(mx.float16), mx.sin(pos * f256).astype(mx.float16)
    c512, s512 = mx.cos(pos * f512).astype(mx.float16), mx.sin(pos * f512).astype(mx.float16)
    mx.eval(c256, s256, c512, s512)
    return c256, s256, c512, s512


def _mpp_matmul(x2d: mx.array, w_nk: mx.array) -> mx.array:
    m_rows = int(x2d.shape[0])
    n_cols = int(w_nk.shape[0])
    if m_rows % 128 == 0:
        return _MPP_128(
            inputs=[x2d, w_nk],
            grid=((n_cols // 64) * 256, m_rows // 128, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(m_rows, n_cols)],
            output_dtypes=[mx.float16],
        )[0]
    return _MPP_64(
        inputs=[x2d, w_nk],
        grid=((n_cols // 64) * 128, m_rows // 64, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m_rows, n_cols)],
        output_dtypes=[mx.float16],
    )[0]


def fast_pack_cost_aware(
    seqs_with_idx: Sequence[tuple[int, Sequence[int]]],
    cost_map: dict[int, float] = ANE_MS_COST,
) -> tuple[dict[int, deque], list[tuple[int, Sequence[int]]]]:
    """Cost-aware multi-bucket packing for <=512 token sequences plus >512 overflow list."""
    bkts = sorted(cost_map.keys())
    max_cap = max(bkts)
    bkt_lut = [max_cap] * (max_cap + 1)
    b_idx = 0
    for s in range(max_cap + 1):
        while b_idx < len(bkts) - 1 and s > bkts[b_idx]:
            b_idx += 1
        bkt_lut[s] = bkts[b_idx]

    over = [(i, t) for i, t in seqs_with_idx if len(t) > max_cap]
    fit = sorted([(i, t) for i, t in seqs_with_idx if len(t) <= max_cap], key=lambda x: len(x[1]), reverse=True)

    open_by_sum: list[list] = [[] for _ in range(max_cap + 1)]
    active_sums: set[int] = set()
    all_lanes: list[tuple[int, list]] = []

    for idx, tok in fit:
        sl = len(tok)
        nb0 = bkt_lut[sl]
        best_score = (cost_map[nb0], -sl, nb0)
        best_s = -1
        limit = max_cap - sl
        for cs in active_sums:
            if cs > limit:
                continue
            ns = cs + sl
            nb = bkt_lut[ns]
            dc = cost_map[nb] - cost_map[bkt_lut[cs]]
            sc = (dc, nb - ns, nb)
            if sc < best_score:
                best_score = sc
                best_s = cs
                if dc <= 0.0 and nb == ns:
                    break
        if best_s >= 0:
            q = open_by_sum[best_s]
            ln = q.pop()
            if not q:
                active_sums.discard(best_s)
            ln.append((idx, tok))
            ns = best_s + sl
            if ns == max_cap:
                all_lanes.append((ns, ln))
            else:
                if not open_by_sum[ns]:
                    active_sums.add(ns)
                open_by_sum[ns].append(ln)
        else:
            ln = [(idx, tok)]
            if sl == max_cap:
                all_lanes.append((sl, ln))
            else:
                if not open_by_sum[sl]:
                    active_sums.add(sl)
                open_by_sum[sl].append(ln)

    for cs in active_sums:
        for ln in open_by_sum[cs]:
            all_lanes.append((cs, ln))

    by_b: dict[int, deque] = {b: deque() for b in bkts}
    for cs, ln in all_lanes:
        by_b[bkt_lut[cs]].append(ln)
    return by_b, over


def build_over_chunks(over: Sequence[tuple[int, Sequence[int]]], max_toks: int = 4096) -> deque:
    """Group >512 token documents into discrete `OVER_BUCKETS` and greedy `OVER_B_OPTS` batch sizes."""
    if not over:
        return deque()
    groups: dict[int, list] = defaultdict(list)
    for idx, tok in sorted(over, key=lambda x: len(x[1])):
        sl = min(MAX_SEQ_LEN, len(tok))
        ml = OVER_BUCKETS[-1]
        for bkt in OVER_BUCKETS:
            if sl <= bkt:
                ml = bkt
                break
        groups[ml].append((idx, tok))
    chunks: deque = deque()
    for ml in sorted(groups):
        docs = groups[ml]
        opts = OVER_B_OPTS[ml]
        pos = 0
        rem = len(docs)
        while rem > 0:
            chosen = opts[-1]
            for opt in opts:
                if opt <= rem:
                    chosen = opt
                    break
            chunks.append((ml, docs[pos : pos + chosen]))
            pos += chosen
            rem -= chosen
    return chunks


class FastEmbeddingGemma2GPU:
    """Compiled Metal MPP + custom-kernel EmbeddingGemma-2 forward pass on Apple GPU."""

    def __init__(self, weights_npz: str | Path) -> None:
        raw = mx.load(str(weights_npz))
        scale_emb = mx.array(HIDDEN_SIZE ** 0.5, dtype=mx.float16)
        self.scale_ple = mx.array(HIDDEN_SIZE ** -0.5, dtype=mx.float16)

        self.embed_scaled = mx.contiguous(raw["embed_tokens"].astype(mx.float16) * scale_emb)
        self.ple_proj = mx.contiguous(raw["ple_proj"][:, :, 0, 0].astype(mx.float16))
        self.ple_norm = mx.contiguous(raw["ple_norm"].reshape(HIDDEN_SIZE).astype(mx.float16))
        self.final_norm = mx.contiguous(raw["final_norm"].reshape(HIDDEN_SIZE).astype(mx.float16))
        self.emb_proj_wT32 = mx.contiguous(raw["emb_proj"][:, :, 0, 0].astype(mx.float32).T)
        self._emb_proj_wT32_slices: dict[int, mx.array] = {EMBED_DIM: self.emb_proj_wT32}

        layers: list[dict] = []
        eval_list: list[mx.array] = [
            self.embed_scaled,
            self.ple_proj,
            self.ple_norm,
            self.final_norm,
            self.emb_proj_wT32,
        ]
        for i in range(NUM_LAYERS):
            q = f"l{i}_"
            is_swa = i not in FULL_LAYERS
            hd = 256 if is_swa else 512
            ld = {
                "is_swa": is_swa,
                "hd": hd,
                "in_ln": mx.contiguous(raw[q + "in_ln"].reshape(HIDDEN_SIZE).astype(mx.float16)),
                "qkv": mx.contiguous(raw[q + "qkv"][:, :, 0, 0].astype(mx.float16)),
                "qn": mx.contiguous(raw[q + "qn"].reshape(hd).astype(mx.float16)),
                "kn": mx.contiguous(raw[q + "kn"].reshape(hd).astype(mx.float16)),
                "o": mx.contiguous(raw[q + "o"][:, :, 0, 0].astype(mx.float16)),
                "pa_ln": mx.contiguous(raw[q + "pa_ln"].reshape(HIDDEN_SIZE).astype(mx.float16)),
                "pff_ln": mx.contiguous(raw[q + "pff_ln"].reshape(HIDDEN_SIZE).astype(mx.float16)),
                "gu": mx.contiguous(raw[q + "gu"][:, :, 0, 0].astype(mx.float16)),
                "dn": mx.contiguous(raw[q + "dn"][:, :, 0, 0].astype(mx.float16)),
                "poff_ln": mx.contiguous(raw[q + "poff_ln"].reshape(HIDDEN_SIZE).astype(mx.float16)),
                "pg": mx.contiguous(raw[q + "pg"][:, :, 0, 0].astype(mx.float16)),
                "pp": mx.contiguous(raw[q + "pp"][:, :, 0, 0].astype(mx.float16)),
                "pple_ln": mx.contiguous(raw[q + "pple_ln"].reshape(HIDDEN_SIZE).astype(mx.float16)),
                "scalar": mx.contiguous(raw[q + "scalar"].reshape((1,)).astype(mx.float16)),
            }
            layers.append(ld)
            eval_list.extend(v for k, v in ld.items() if isinstance(v, mx.array))
        del raw
        mx.eval(*eval_list)
        self.layers = layers

        self.c256, self.s256, self.c512, self.s512 = make_rope_tables(MAX_SEQ_LEN)
        self._arange_max = np.arange(MAX_SEQ_LEN, dtype=np.int32)

        def _forward_trunk(ids_mx: mx.array, pos_mx: mx.array, mask_swa: mx.array, mask_full: mx.array) -> mx.array:
            b_sz, seq_l = ids_mx.shape
            m_tot = b_sz * seq_l
            cs = self.c256[pos_mx]
            ss = self.s256[pos_mx]
            cf = self.c512[pos_mx]
            sf = self.s512[pos_mx]

            h = self.embed_scaled[ids_mx.reshape(m_tot)]
            pl_raw = _mpp_matmul(h, self.ple_proj) * self.scale_ple
            pl_layers = mx.contiguous(
                mx.fast.rms_norm(pl_raw.reshape(m_tot, NUM_LAYERS, HIDDEN_SIZE), self.ple_norm, 1e-6).transpose(1, 0, 2)
            )
            grid_geglu = ((2048 // 64) * 128, m_tot // 64, 1)
            grid_pg = ((512 // 64) * 128, m_tot // 64, 1)

            xn = mx.fast.rms_norm(h, self.layers[0]["in_ln"], 1e-6)
            for li, ld in enumerate(self.layers):
                qkv = _mpp_matmul(xn, ld["qkv"])
                if ld["is_swa"]:
                    q, k, v = _QKNORM_ROPE_256_BHLD(
                        inputs=[qkv, ld["qn"], ld["kn"], cs, ss],
                        output_shapes=[(b_sz, 4, seq_l, 256), (b_sz, 2, seq_l, 256), (b_sz, 2, seq_l, 256)],
                        output_dtypes=[mx.float16, mx.float16, mx.float16],
                        grid=(m_tot * 256, 1, 1),
                        threadgroup=(256, 1, 1),
                    )
                    att_mask = mask_swa
                else:
                    q, k, v = _QKNORM_ROPE_512_BHLD(
                        inputs=[qkv, ld["qn"], ld["kn"], cf, sf],
                        output_shapes=[(b_sz, 4, seq_l, 512), (b_sz, 1, seq_l, 512), (b_sz, 1, seq_l, 512)],
                        output_dtypes=[mx.float16, mx.float16, mx.float16],
                        grid=(m_tot * 192, 1, 1),
                        threadgroup=(192, 1, 1),
                    )
                    att_mask = mask_full
                att = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=att_mask)
                ao = _mpp_matmul(att.transpose(0, 2, 1, 3).reshape(m_tot, 4 * ld["hd"]), ld["o"])
                h1, m_in = fused_gemma_res_rms(h, ao, ld["pa_ln"], ld["pff_ln"])

                act = _MPP_GEGLU_64(
                    inputs=[m_in, ld["gu"]],
                    grid=grid_geglu,
                    threadgroup=(128, 1, 1),
                    output_shapes=[(m_tot, 2048)],
                    output_dtypes=[mx.float16],
                )[0]
                dn = _mpp_matmul(act, ld["dn"])
                h2 = fused_gemma_res_rms(h1, dn, ld["poff_ln"])

                pm = _MPP_PG_64(
                    inputs=[h2, ld["pg"], pl_layers[li]],
                    grid=grid_pg,
                    threadgroup=(128, 1, 1),
                    output_shapes=[(m_tot, HIDDEN_SIZE)],
                    output_dtypes=[mx.float16],
                )[0]
                pp = _mpp_matmul(pm, ld["pp"])
                next_ln = self.layers[li + 1]["in_ln"] if li + 1 < NUM_LAYERS else self.final_norm
                h, xn = fused_gemma_res_rms(h2, pp, ld["pple_ln"], next_ln, scalar=ld["scalar"])

            return xn

        @mx.compile
        def _packed_trunk_compiled(ids_mx: mx.array, pos_mx: mx.array, blk_mx: mx.array) -> mx.array:
            msk = blk_mx[:, None, :, None] == blk_mx[:, None, None, :]
            return _forward_trunk(ids_mx, pos_mx, msk, msk)

        @mx.compile
        def _over_compiled(ids_mx: mx.array, msk_mx: mx.array) -> mx.array:
            b_sz, seq_l = ids_mx.shape
            pos_1d = mx.arange(seq_l)
            pos_mx = mx.broadcast_to(pos_1d[None], (b_sz, seq_l))
            fm = msk_mx[:, None, None, :]
            sm = fm & (mx.abs(pos_1d[:, None] - pos_1d[None, :]) <= 512)
            ho = _forward_trunk(ids_mx, pos_mx, sm, fm).reshape(b_sz, seq_l, HIDDEN_SIZE)
            mf = msk_mx.astype(mx.float16)[:, :, None]
            return mx.sum(ho * mf, axis=1).astype(mx.float32) / mx.sum(msk_mx.astype(mx.float32), axis=1, keepdims=True)

        self._packed_trunk_compiled = _packed_trunk_compiled
        self._over_compiled = _over_compiled

    def _proj_wT(self, dimensions: int | None = None) -> mx.array:
        eff_dim = EMBED_DIM if dimensions is None or int(dimensions) >= EMBED_DIM else int(dimensions)
        w = self._emb_proj_wT32_slices.get(eff_dim)
        if w is None:
            w = mx.contiguous(self.emb_proj_wT32[:, :eff_dim])
            mx.eval(w)
            self._emb_proj_wT32_slices[eff_dim] = w
        return w

    def exec_packed(
        self,
        L_lane: int,
        chunk: list[list[tuple[int, Sequence[int]]]],
        *,
        dimensions: int | None = None,
    ) -> tuple[int, int, list[int], mx.array]:
        B = len(chunk)
        M = B * L_lane
        ids_np = np.zeros((B, L_lane), dtype=np.int32)
        pos_np = np.zeros((B, L_lane), dtype=np.int32)
        blk_np = np.full((B, L_lane), 9999, dtype=np.int32)
        n_docs = sum(len(ln) for ln in chunk)
        starts_np = np.empty(n_docs, dtype=np.int32)
        lens_np = np.empty(n_docs, dtype=np.int32)
        doc_idxs: list[int] = []
        d_i = 0
        real_in_b = 0
        for b, ln in enumerate(chunk):
            cur = 0
            base_b = b * L_lane
            for seq_i, (idx, tok) in enumerate(ln):
                ll = len(tok)
                real_in_b += ll
                ids_np[b, cur : cur + ll] = tok
                pos_np[b, cur : cur + ll] = self._arange_max[:ll]
                blk_np[b, cur : cur + ll] = seq_i
                starts_np[d_i] = base_b + cur
                lens_np[d_i] = ll
                doc_idxs.append(idx)
                d_i += 1
                cur += ll

        ho = self._packed_trunk_compiled(mx.array(ids_np), mx.array(pos_np), mx.array(blk_np))
        pooled = mean_pool_segments(ho, mx.array(starts_np), mx.array(lens_np))
        p = pooled @ self._proj_wT(dimensions)
        out = p * mx.rsqrt(mx.maximum((p * p).sum(axis=-1, keepdims=True), 1e-12))
        return real_in_b, M, doc_idxs, out

    def exec_over(
        self,
        ml: int,
        chunk: list[tuple[int, Sequence[int]]],
        *,
        dimensions: int | None = None,
    ) -> tuple[int, int, list[int], mx.array]:
        B = len(chunk)
        ids_np = np.zeros((B, ml), dtype=np.int32)
        msk_np = np.zeros((B, ml), dtype=np.bool_)
        idxs: list[int] = []
        rt = 0
        for bi, (idx, t) in enumerate(chunk):
            ll = min(len(t), ml)
            ids_np[bi, :ll] = t[:ll]
            msk_np[bi, :ll] = True
            idxs.append(idx)
            rt += ll
        pooled = self._over_compiled(mx.array(ids_np), mx.array(msk_np))
        p = pooled @ self._proj_wT(dimensions)
        out = p * mx.rsqrt(mx.maximum((p * p).sum(axis=-1, keepdims=True), 1e-12))
        return rt, B * ml, idxs, out

    def encode_single(
        self,
        token_ids: Sequence[int] | np.ndarray,
        *,
        dimensions: int | None = None,
    ) -> np.ndarray:
        """Single-sequence `(dim,)` L2-normalized float32 embedding on GPU (`1..2048` tokens)."""
        sl = min(len(token_ids), MAX_SEQ_LEN)
        if sl <= 0:
            arr = [2, 1]
            sl = 2
        else:
            arr = list(token_ids[:sl]) if not isinstance(token_ids, np.ndarray) else token_ids[:sl]
        ml = max(64, ((sl + 63) // 64) * 64)
        ids_np = np.zeros((1, ml), dtype=np.int32)
        msk_np = np.zeros((1, ml), dtype=np.bool_)
        ids_np[0, :sl] = arr
        msk_np[0, :sl] = True
        pooled = self._over_compiled(mx.array(ids_np), mx.array(msk_np))
        p = pooled @ self._proj_wT(dimensions)
        out = p * mx.rsqrt(mx.maximum((p * p).sum(axis=-1, keepdims=True), 1e-12))
        mx.eval(out)
        return np.asarray(out[0], dtype=np.float32).copy()
