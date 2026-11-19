#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "emb2_fast_io.h"
#include "fast_shared_apple.hpp"

extern "C" {

void emb2_prep_batch(
    int N,
    int S,
    const uint16_t* emb_scaled,
    const uint16_t* c256,
    const uint16_t* s256,
    const uint16_t* c512,
    const uint16_t* s512,
    const int32_t* ids,
    const int32_t* pos,
    const int32_t* span_offsets,
    const int32_t* spans_se,
    uint16_t neg_inf_h,
    uint16_t* h_all,
    uint16_t* b_all,
    uint16_t* cs_all,
    uint16_t* ss_all,
    uint16_t* cf_all,
    uint16_t* sf_all
) {
    alignas(64) uint16_t tmp[16][512];
    const size_t ss_stride = static_cast<size_t>(S) * S;
    const size_t h_stride = static_cast<size_t>(512) * S;

    for (int n = 0; n < N; ++n) {
        const int32_t* ids_n = ids + static_cast<size_t>(n) * S;
        const int32_t* pos_n = pos + static_cast<size_t>(n) * S;
        uint16_t* h_n = h_all + static_cast<size_t>(n) * h_stride;
        uint16_t* b_n = b_all + static_cast<size_t>(n) * ss_stride;
        uint16_t* cs_n = cs_all + static_cast<size_t>(n) * S * 128;
        uint16_t* ss_n = ss_all + static_cast<size_t>(n) * S * 128;
        uint16_t* cf_n = cf_all + static_cast<size_t>(n) * S * 256;
        uint16_t* sf_n = sf_all + static_cast<size_t>(n) * S * 256;

        // 1. Gather + transpose 8 tokens in registers via NEON ([S, 512] -> [512, S])
        static const uint16_t zero_row[512] = {0};
        for (int l0 = 0; l0 < S; l0 += 8) {
            const int blk = (l0 + 8 <= S) ? 8 : (S - l0);
            if (blk == 8) {
                const uint16_t* rp[8];
                for (int k = 0; k < 8; ++k) {
                    const int tid = ids_n[l0 + k];
                    rp[k] = (tid >= 0) ? (emb_scaled + static_cast<size_t>(tid) * 512) : zero_row;
                }
                for (int d = 0; d < 512; d += 8) {
                    const uint16x8_t r0 = vld1q_u16(rp[0] + d);
                    const uint16x8_t r1 = vld1q_u16(rp[1] + d);
                    const uint16x8_t r2 = vld1q_u16(rp[2] + d);
                    const uint16x8_t r3 = vld1q_u16(rp[3] + d);
                    const uint16x8_t r4 = vld1q_u16(rp[4] + d);
                    const uint16x8_t r5 = vld1q_u16(rp[5] + d);
                    const uint16x8_t r6 = vld1q_u16(rp[6] + d);
                    const uint16x8_t r7 = vld1q_u16(rp[7] + d);

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

                    uint16_t* dst = h_n + static_cast<size_t>(d) * S + l0;
                    vst1q_u16(dst + 0 * static_cast<size_t>(S), vreinterpretq_u16_u64(col0));
                    vst1q_u16(dst + 1 * static_cast<size_t>(S), vreinterpretq_u16_u64(col1));
                    vst1q_u16(dst + 2 * static_cast<size_t>(S), vreinterpretq_u16_u64(col2));
                    vst1q_u16(dst + 3 * static_cast<size_t>(S), vreinterpretq_u16_u64(col3));
                    vst1q_u16(dst + 4 * static_cast<size_t>(S), vreinterpretq_u16_u64(col4));
                    vst1q_u16(dst + 5 * static_cast<size_t>(S), vreinterpretq_u16_u64(col5));
                    vst1q_u16(dst + 6 * static_cast<size_t>(S), vreinterpretq_u16_u64(col6));
                    vst1q_u16(dst + 7 * static_cast<size_t>(S), vreinterpretq_u16_u64(col7));
                }
            } else {
                for (int k = 0; k < blk; ++k) {
                    const int tid = ids_n[l0 + k];
                    if (tid >= 0) {
                        std::memcpy(tmp[k], emb_scaled + static_cast<size_t>(tid) * 512, 512 * sizeof(uint16_t));
                    } else {
                        std::memset(tmp[k], 0, 512 * sizeof(uint16_t));
                    }
                }
                for (int d = 0; d < 512; ++d) {
                    uint16_t* dst = h_n + static_cast<size_t>(d) * S + l0;
                    for (int k = 0; k < blk; ++k) {
                        dst[k] = tmp[k][d];
                    }
                }
            }
        }

        // 2. Block-diagonal packed attention bias [S, S]
        std::fill_n(b_n, ss_stride, neg_inf_h);
        const int sp0 = span_offsets[n];
        const int sp1 = span_offsets[n + 1];
        int last_end = 0;
        for (int s = sp0; s < sp1; ++s) {
            const int st = spans_se[s * 2];
            const int en = spans_se[s * 2 + 1];
            const int len = en - st;
            for (int q = st; q < en; ++q) {
                std::memset(b_n + static_cast<size_t>(q) * S + st, 0, static_cast<size_t>(len) * sizeof(uint16_t));
            }
            if (en > last_end) last_end = en;
        }
        if (last_end < S) {
            const int tail = S - last_end;
            for (int q = last_end; q < S; ++q) {
                std::memset(b_n + static_cast<size_t>(q) * S + last_end, 0, static_cast<size_t>(tail) * sizeof(uint16_t));
            }
        }

        // 3. Per-position RoPE cos/sin slices ([S, 128] SWA / [S, 256] full)
        for (int l = 0; l < S; ++l) {
            const int p = pos_n[l];
            std::memcpy(cs_n + static_cast<size_t>(l) * 128, c256 + static_cast<size_t>(p) * 128, 128 * sizeof(uint16_t));
            std::memcpy(ss_n + static_cast<size_t>(l) * 128, s256 + static_cast<size_t>(p) * 128, 128 * sizeof(uint16_t));
            std::memcpy(cf_n + static_cast<size_t>(l) * 256, c512 + static_cast<size_t>(p) * 256, 256 * sizeof(uint16_t));
            std::memcpy(sf_n + static_cast<size_t>(l) * 256, s512 + static_cast<size_t>(p) * 256, 256 * sizeof(uint16_t));
        }
    }
}

void emb2_pool_project_batch_dim(
    int S,
    const uint16_t* out_all_u16,
    int n_docs,
    const int32_t* doc_lane_se,
    const float* emb_proj_T,
    float* scratch_pooled,
    float* out_embs,
    int proj_dim
) {
    if (n_docs <= 0) return;
    const int eff_dim = (proj_dim > 0 && proj_dim <= 768) ? proj_dim : 768;
    const auto* out_all = reinterpret_cast<const float16_t*>(out_all_u16);

    for (int d = 0; d < n_docs; ++d) {
        const int ln = doc_lane_se[d * 3];
        const int st = doc_lane_se[d * 3 + 1];
        const int en = doc_lane_se[d * 3 + 2];
        const int len = std::max(1, en - st);
        float* dst = scratch_pooled + static_cast<size_t>(d) * 512;
        std::memset(dst, 0, 512 * sizeof(float));

        const float16_t* base = out_all + (static_cast<size_t>(ln) * S + st) * 512;
        for (int l = 0; l < len; ++l) {
            const float16_t* row = base + static_cast<size_t>(l) * 512;
            for (int c = 0; c < 512; c += 8) {
                const float16x8_t h8 = vld1q_f16(row + c);
                const float32x4_t f_lo = vcvt_f32_f16(vget_low_f16(h8));
                const float32x4_t f_hi = vcvt_f32_f16(vget_high_f16(h8));
                vst1q_f32(dst + c, vaddq_f32(vld1q_f32(dst + c), f_lo));
                vst1q_f32(dst + c + 4, vaddq_f32(vld1q_f32(dst + c + 4), f_hi));
            }
        }
        const float inv = 1.0f / static_cast<float>(len);
        vDSP_vsmul(dst, 1, &inv, dst, 1, 512);
    }

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n_docs, eff_dim, 512,
        1.0f, scratch_pooled, 512,
        emb_proj_T, 768,
        0.0f, out_embs, eff_dim
    );

    for (int d = 0; d < n_docs; ++d) {
        float* row = out_embs + static_cast<size_t>(d) * eff_dim;
        fast_shared::l2_normalize_f32(row, row, eff_dim);
    }
}

void emb2_pool_project_batch(
    int S,
    const uint16_t* out_all_u16,
    int n_docs,
    const int32_t* doc_lane_se,
    const float* emb_proj_T,
    float* scratch_pooled,
    float* out_embs
) {
    emb2_pool_project_batch_dim(S, out_all_u16, n_docs, doc_lane_se, emb_proj_T, scratch_pooled, out_embs, 768);
}

}  // extern "C"
