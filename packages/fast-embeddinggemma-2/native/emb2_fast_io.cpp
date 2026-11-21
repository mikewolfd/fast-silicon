#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "emb2_fast_io.h"
#include "emb2_neon_transpose.hpp"
#include "fast_shared_apple.hpp"

namespace {

static constexpr int kSwaWindow = 512;
static constexpr int kDefaultVocabSize = 262144;
static constexpr int kDefaultMaxPos = 2048;

static inline void dequant_row_512_neon(
    const int8_t* __restrict__ src_i8,
    float row_scale,
    uint16_t* __restrict__ dst_u16
) {
    const float32x4_t v_sc = vdupq_n_f32(row_scale);
    auto* dst = reinterpret_cast<float16_t*>(dst_u16);
    for (int d = 0; d < 512; d += 16) {
        const int8x16_t q16 = vld1q_s8(src_i8 + d);
        const int16x8_t q_lo = vmovl_s8(vget_low_s8(q16));
        const int16x8_t q_hi = vmovl_s8(vget_high_s8(q16));

        const float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_lo))), v_sc);
        const float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_lo))), v_sc);
        const float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_hi))), v_sc);
        const float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_hi))), v_sc);

        vst1q_f16(dst + d, vcombine_f16(vcvt_f16_f32(f0), vcvt_f16_f32(f1)));
        vst1q_f16(dst + d + 8, vcombine_f16(vcvt_f16_f32(f2), vcvt_f16_f32(f3)));
    }
}

static inline void fill_swa_band_mask(
    int S,
    const int32_t* span_offsets,
    const int32_t* spans_se,
    int n,
    uint16_t neg_inf_h,
    uint16_t* __restrict__ b_n
) {
    const size_t ss_stride = static_cast<size_t>(S) * S;
    std::fill_n(b_n, ss_stride, neg_inf_h);
    const int sp0 = span_offsets[n];
    const int sp1 = span_offsets[n + 1];
    int last_end = 0;
    for (int s = sp0; s < sp1; ++s) {
        const int st = spans_se[s * 2];
        const int en = spans_se[s * 2 + 1];
        for (int q = st; q < en; ++q) {
            const int k0 = std::max(st, q - kSwaWindow);
            const int k1 = std::min(en, q + kSwaWindow + 1);
            if (k1 > k0) {
                std::memset(b_n + static_cast<size_t>(q) * S + k0, 0, static_cast<size_t>(k1 - k0) * sizeof(uint16_t));
            }
        }
        if (en > last_end) last_end = en;
    }
    if (last_end < S) {
        for (int q = last_end; q < S; ++q) {
            const int k0 = std::max(last_end, q - kSwaWindow);
            const int k1 = std::min(S, q + kSwaWindow + 1);
            if (k1 > k0) {
                std::memset(b_n + static_cast<size_t>(q) * S + k0, 0, static_cast<size_t>(k1 - k0) * sizeof(uint16_t));
            }
        }
    }
}

}  // namespace

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
                    rp[k] = (tid >= 0 && tid < kDefaultVocabSize)
                        ? (emb_scaled + static_cast<size_t>(tid) * 512)
                        : zero_row;
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

                    emb2::transpose_u16_8x8_store(
                        r0, r1, r2, r3, r4, r5, r6, r7,
                        h_n + static_cast<size_t>(d) * S + l0,
                        static_cast<size_t>(S)
                    );
                }
            } else {
                for (int k = 0; k < blk; ++k) {
                    const int tid = ids_n[l0 + k];
                    if (tid >= 0 && tid < kDefaultVocabSize) {
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

        // 2. Block-diagonal packed attention bias [S, S] with explicit |q - k| <= 512 SWA band
        fill_swa_band_mask(S, span_offsets, spans_se, n, neg_inf_h, b_n);

        // 3. Per-position RoPE cos/sin slices ([S, 128] SWA / [S, 256] full)
        for (int l = 0; l < S; ++l) {
            const int raw_p = pos_n[l];
            const int p = (raw_p >= 0 && raw_p < kDefaultMaxPos) ? raw_p : 0;
            std::memcpy(cs_n + static_cast<size_t>(l) * 128, c256 + static_cast<size_t>(p) * 128, 128 * sizeof(uint16_t));
            std::memcpy(ss_n + static_cast<size_t>(l) * 128, s256 + static_cast<size_t>(p) * 128, 128 * sizeof(uint16_t));
            std::memcpy(cf_n + static_cast<size_t>(l) * 256, c512 + static_cast<size_t>(p) * 256, 256 * sizeof(uint16_t));
            std::memcpy(sf_n + static_cast<size_t>(l) * 256, s512 + static_cast<size_t>(p) * 256, 256 * sizeof(uint16_t));
        }
    }
}

void emb2_prep_batch_i8(
    int N,
    int S,
    const int8_t* emb_i8,
    const uint16_t* emb_scale_f16,
    int vocab_size,
    int max_pos,
    float extra_scale,
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
    alignas(64) uint16_t tmp[8][512];
    const size_t ss_stride = static_cast<size_t>(S) * S;
    const size_t h_stride = static_cast<size_t>(512) * S;
    const auto* sc_h = reinterpret_cast<const float16_t*>(emb_scale_f16);
    const int eff_vocab = (vocab_size > 0) ? vocab_size : kDefaultVocabSize;
    const int eff_max_pos = (max_pos > 0) ? max_pos : kDefaultMaxPos;

    for (int n = 0; n < N; ++n) {
        const int32_t* ids_n = ids + static_cast<size_t>(n) * S;
        const int32_t* pos_n = pos + static_cast<size_t>(n) * S;
        uint16_t* h_n = h_all + static_cast<size_t>(n) * h_stride;
        uint16_t* b_n = b_all + static_cast<size_t>(n) * ss_stride;
        uint16_t* cs_n = cs_all + static_cast<size_t>(n) * S * 128;
        uint16_t* ss_n = ss_all + static_cast<size_t>(n) * S * 128;
        uint16_t* cf_n = cf_all + static_cast<size_t>(n) * S * 256;
        uint16_t* sf_n = sf_all + static_cast<size_t>(n) * S * 256;

        for (int l0 = 0; l0 < S; l0 += 8) {
            const int blk = (l0 + 8 <= S) ? 8 : (S - l0);
            for (int k = 0; k < blk; ++k) {
                const int tid = ids_n[l0 + k];
                if (tid >= 0 && tid < eff_vocab) {
                    const float row_sc = static_cast<float>(sc_h[tid]) * extra_scale;
                    dequant_row_512_neon(emb_i8 + static_cast<size_t>(tid) * 512, row_sc, tmp[k]);
                } else {
                    std::memset(tmp[k], 0, 512 * sizeof(uint16_t));
                }
            }
            if (blk == 8) {
                for (int d = 0; d < 512; d += 8) {
                    emb2::transpose_u16_8x8_store(
                        vld1q_u16(tmp[0] + d),
                        vld1q_u16(tmp[1] + d),
                        vld1q_u16(tmp[2] + d),
                        vld1q_u16(tmp[3] + d),
                        vld1q_u16(tmp[4] + d),
                        vld1q_u16(tmp[5] + d),
                        vld1q_u16(tmp[6] + d),
                        vld1q_u16(tmp[7] + d),
                        h_n + static_cast<size_t>(d) * S + l0,
                        static_cast<size_t>(S)
                    );
                }
            } else {
                for (int d = 0; d < 512; ++d) {
                    uint16_t* dst = h_n + static_cast<size_t>(d) * S + l0;
                    for (int k = 0; k < blk; ++k) {
                        dst[k] = tmp[k][d];
                    }
                }
            }
        }

        fill_swa_band_mask(S, span_offsets, spans_se, n, neg_inf_h, b_n);

        for (int l = 0; l < S; ++l) {
            const int raw_p = pos_n[l];
            const int p = (raw_p >= 0 && raw_p < eff_max_pos) ? raw_p : 0;
            std::memcpy(cs_n + static_cast<size_t>(l) * 128, c256 + static_cast<size_t>(p) * 128, 128 * sizeof(uint16_t));
            std::memcpy(ss_n + static_cast<size_t>(l) * 128, s256 + static_cast<size_t>(p) * 128, 128 * sizeof(uint16_t));
            std::memcpy(cf_n + static_cast<size_t>(l) * 256, c512 + static_cast<size_t>(p) * 256, 256 * sizeof(uint16_t));
            std::memcpy(sf_n + static_cast<size_t>(l) * 256, s512 + static_cast<size_t>(p) * 256, 256 * sizeof(uint16_t));
        }
    }
}

void emb2_gather_scaled_i8(
    const int8_t* emb_i8,
    const uint16_t* emb_scale_f16,
    float extra_scale,
    const int32_t* ids,
    int vocab_size,
    int M,
    uint16_t* out_f16
) {
    const auto* sc_h = reinterpret_cast<const float16_t*>(emb_scale_f16);
    for (int i = 0; i < M; ++i) {
        const int tid = ids[i];
        uint16_t* dst = out_f16 + static_cast<size_t>(i) * 512;
        if (tid >= 0 && tid < vocab_size) {
            const float row_sc = static_cast<float>(sc_h[tid]) * extra_scale;
            dequant_row_512_neon(emb_i8 + static_cast<size_t>(tid) * 512, row_sc, dst);
        } else {
            std::memset(dst, 0, 512 * sizeof(uint16_t));
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
