#pragma once
#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace emb2 {

// Transpose an 8x8 tile of 16-bit elements into 8 strided rows at `dst`.
inline void transpose_u16_8x8_store(
    uint16x8_t r0,
    uint16x8_t r1,
    uint16x8_t r2,
    uint16x8_t r3,
    uint16x8_t r4,
    uint16x8_t r5,
    uint16x8_t r6,
    uint16x8_t r7,
    uint16_t* dst,
    size_t dst_stride
) {
    const uint16x8x2_t a0 = vtrnq_u16(r0, r1);
    const uint16x8x2_t a1 = vtrnq_u16(r2, r3);
    const uint16x8x2_t a2 = vtrnq_u16(r4, r5);
    const uint16x8x2_t a3 = vtrnq_u16(r6, r7);

    const uint32x4x2_t b0 = vtrnq_u32(vreinterpretq_u32_u16(a0.val[0]), vreinterpretq_u32_u16(a1.val[0]));
    const uint32x4x2_t b1 = vtrnq_u32(vreinterpretq_u32_u16(a0.val[1]), vreinterpretq_u32_u16(a1.val[1]));
    const uint32x4x2_t b2 = vtrnq_u32(vreinterpretq_u32_u16(a2.val[0]), vreinterpretq_u32_u16(a3.val[0]));
    const uint32x4x2_t b3 = vtrnq_u32(vreinterpretq_u32_u16(a2.val[1]), vreinterpretq_u32_u16(a3.val[1]));

    const uint64x2_t col0 = vtrn1q_u64(vreinterpretq_u64_u32(b0.val[0]), vreinterpretq_u64_u32(b2.val[0]));
    const uint64x2_t col4 = vtrn2q_u64(vreinterpretq_u64_u32(b0.val[0]), vreinterpretq_u64_u32(b2.val[0]));
    const uint64x2_t col1 = vtrn1q_u64(vreinterpretq_u64_u32(b1.val[0]), vreinterpretq_u64_u32(b3.val[0]));
    const uint64x2_t col5 = vtrn2q_u64(vreinterpretq_u64_u32(b1.val[0]), vreinterpretq_u64_u32(b3.val[0]));
    const uint64x2_t col2 = vtrn1q_u64(vreinterpretq_u64_u32(b0.val[1]), vreinterpretq_u64_u32(b2.val[1]));
    const uint64x2_t col6 = vtrn2q_u64(vreinterpretq_u64_u32(b0.val[1]), vreinterpretq_u64_u32(b2.val[1]));
    const uint64x2_t col3 = vtrn1q_u64(vreinterpretq_u64_u32(b1.val[1]), vreinterpretq_u64_u32(b3.val[1]));
    const uint64x2_t col7 = vtrn2q_u64(vreinterpretq_u64_u32(b1.val[1]), vreinterpretq_u64_u32(b3.val[1]));

    vst1q_u16(dst + 0 * dst_stride, vreinterpretq_u16_u64(col0));
    vst1q_u16(dst + 1 * dst_stride, vreinterpretq_u16_u64(col1));
    vst1q_u16(dst + 2 * dst_stride, vreinterpretq_u16_u64(col2));
    vst1q_u16(dst + 3 * dst_stride, vreinterpretq_u16_u64(col3));
    vst1q_u16(dst + 4 * dst_stride, vreinterpretq_u16_u64(col4));
    vst1q_u16(dst + 5 * dst_stride, vreinterpretq_u16_u64(col5));
    vst1q_u16(dst + 6 * dst_stride, vreinterpretq_u16_u64(col6));
    vst1q_u16(dst + 7 * dst_stride, vreinterpretq_u16_u64(col7));
}

// Untile CoreML NCHW [1, 512, 1, S] output into row-major [S, 512] FP16 buffer.
inline void transpose_512xS_to_Sx512_u16(const uint16_t* src, uint16_t* dst, int S) {
    if (S <= 0 || !src || !dst) return;
    const size_t s_stride = static_cast<size_t>(S);
    int l = 0;
    for (; l + 8 <= S; l += 8) {
        for (int d = 0; d < 512; d += 8) {
            const uint16_t* sp = src + static_cast<size_t>(d) * s_stride + l;
            transpose_u16_8x8_store(
                vld1q_u16(sp + 0 * s_stride),
                vld1q_u16(sp + 1 * s_stride),
                vld1q_u16(sp + 2 * s_stride),
                vld1q_u16(sp + 3 * s_stride),
                vld1q_u16(sp + 4 * s_stride),
                vld1q_u16(sp + 5 * s_stride),
                vld1q_u16(sp + 6 * s_stride),
                vld1q_u16(sp + 7 * s_stride),
                dst + static_cast<size_t>(l) * 512 + d,
                512
            );
        }
    }
    for (; l < S; ++l) {
        uint16_t* dp = dst + static_cast<size_t>(l) * 512;
        for (int d = 0; d < 512; ++d) {
            dp[d] = src[static_cast<size_t>(d) * s_stride + l];
        }
    }
}

}  // namespace emb2
