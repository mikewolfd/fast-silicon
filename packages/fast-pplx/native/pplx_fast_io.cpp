#include "pplx_fast_io.h"
#include "fast_shared_apple.hpp"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

inline uint16_t f32_to_u16(float v) {
    _Float16 h = static_cast<_Float16>(v);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

inline float u16_to_f32(uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

// Reusable per-thread scratch buffers avoid heap allocations on every bucket.
thread_local std::vector<uint16_t> tl_gathered_f16;
thread_local std::vector<float> tl_pooled_f32;
thread_local std::vector<float> tl_tok_norm_f32;

// Fast NEON 8x8 FP16 matrix transpose from row-major src[seq_len, hidden_dim]
// into channel-major dst[hidden_dim, seq_len]. Mirrors emb2_fast_io NEON kernel.
void neon_transpose_seq_hidden_f16(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int seq_len,
    int hidden_dim) {
    if ((seq_len & 7) == 0 && (hidden_dim & 7) == 0) {
        for (int s = 0; s < seq_len; s += 8) {
            const uint16_t* s0 = src + static_cast<size_t>(s + 0) * hidden_dim;
            const uint16_t* s1 = src + static_cast<size_t>(s + 1) * hidden_dim;
            const uint16_t* s2 = src + static_cast<size_t>(s + 2) * hidden_dim;
            const uint16_t* s3 = src + static_cast<size_t>(s + 3) * hidden_dim;
            const uint16_t* s4 = src + static_cast<size_t>(s + 4) * hidden_dim;
            const uint16_t* s5 = src + static_cast<size_t>(s + 5) * hidden_dim;
            const uint16_t* s6 = src + static_cast<size_t>(s + 6) * hidden_dim;
            const uint16_t* s7 = src + static_cast<size_t>(s + 7) * hidden_dim;

            for (int c = 0; c < hidden_dim; c += 8) {
                uint16x8_t r0 = vld1q_u16(s0 + c);
                uint16x8_t r1 = vld1q_u16(s1 + c);
                uint16x8_t r2 = vld1q_u16(s2 + c);
                uint16x8_t r3 = vld1q_u16(s3 + c);
                uint16x8_t r4 = vld1q_u16(s4 + c);
                uint16x8_t r5 = vld1q_u16(s5 + c);
                uint16x8_t r6 = vld1q_u16(s6 + c);
                uint16x8_t r7 = vld1q_u16(s7 + c);

                uint16x8_t a0 = vtrn1q_u16(r0, r1);
                uint16x8_t a1 = vtrn2q_u16(r0, r1);
                uint16x8_t a2 = vtrn1q_u16(r2, r3);
                uint16x8_t a3 = vtrn2q_u16(r2, r3);
                uint16x8_t a4 = vtrn1q_u16(r4, r5);
                uint16x8_t a5 = vtrn2q_u16(r4, r5);
                uint16x8_t a6 = vtrn1q_u16(r6, r7);
                uint16x8_t a7 = vtrn2q_u16(r6, r7);

                uint32x4_t b0 = vtrn1q_u32(vreinterpretq_u32_u16(a0), vreinterpretq_u32_u16(a2));
                uint32x4_t b1 = vtrn1q_u32(vreinterpretq_u32_u16(a1), vreinterpretq_u32_u16(a3));
                uint32x4_t b2 = vtrn2q_u32(vreinterpretq_u32_u16(a0), vreinterpretq_u32_u16(a2));
                uint32x4_t b3 = vtrn2q_u32(vreinterpretq_u32_u16(a1), vreinterpretq_u32_u16(a3));
                uint32x4_t b4 = vtrn1q_u32(vreinterpretq_u32_u16(a4), vreinterpretq_u32_u16(a6));
                uint32x4_t b5 = vtrn1q_u32(vreinterpretq_u32_u16(a5), vreinterpretq_u32_u16(a7));
                uint32x4_t b6 = vtrn2q_u32(vreinterpretq_u32_u16(a4), vreinterpretq_u32_u16(a6));
                uint32x4_t b7 = vtrn2q_u32(vreinterpretq_u32_u16(a5), vreinterpretq_u32_u16(a7));

                uint64x2_t c0 = vtrn1q_u64(vreinterpretq_u64_u32(b0), vreinterpretq_u64_u32(b4));
                uint64x2_t c4 = vtrn2q_u64(vreinterpretq_u64_u32(b0), vreinterpretq_u64_u32(b4));
                uint64x2_t c1 = vtrn1q_u64(vreinterpretq_u64_u32(b1), vreinterpretq_u64_u32(b5));
                uint64x2_t c5 = vtrn2q_u64(vreinterpretq_u64_u32(b1), vreinterpretq_u64_u32(b5));
                uint64x2_t c2 = vtrn1q_u64(vreinterpretq_u64_u32(b2), vreinterpretq_u64_u32(b6));
                uint64x2_t c6 = vtrn2q_u64(vreinterpretq_u64_u32(b2), vreinterpretq_u64_u32(b6));
                uint64x2_t c3 = vtrn1q_u64(vreinterpretq_u64_u32(b3), vreinterpretq_u64_u32(b7));
                uint64x2_t c7 = vtrn2q_u64(vreinterpretq_u64_u32(b3), vreinterpretq_u64_u32(b7));

                uint16_t* dc = dst + static_cast<size_t>(c) * seq_len + s;
                vst1q_u16(dc + 0 * seq_len, vreinterpretq_u16_u64(c0));
                vst1q_u16(dc + 1 * seq_len, vreinterpretq_u16_u64(c1));
                vst1q_u16(dc + 2 * seq_len, vreinterpretq_u16_u64(c2));
                vst1q_u16(dc + 3 * seq_len, vreinterpretq_u16_u64(c3));
                vst1q_u16(dc + 4 * seq_len, vreinterpretq_u16_u64(c4));
                vst1q_u16(dc + 5 * seq_len, vreinterpretq_u16_u64(c5));
                vst1q_u16(dc + 6 * seq_len, vreinterpretq_u16_u64(c6));
                vst1q_u16(dc + 7 * seq_len, vreinterpretq_u16_u64(c7));
            }
        }
        return;
    }

    for (int c = 0; c < hidden_dim; ++c) {
        uint16_t* dc = dst + static_cast<size_t>(c) * seq_len;
        for (int s = 0; s < seq_len; ++s) {
            dc[s] = src[static_cast<size_t>(s) * hidden_dim + c];
        }
    }
}

}  // namespace

extern "C" {

void pplx_build_rope_table_f16(
    uint16_t* out_cos_u16,
    uint16_t* out_sin_u16,
    int max_pos,
    int rope_half,
    double rope_theta) {
    if (!out_cos_u16 || !out_sin_u16 || max_pos <= 0 || rope_half <= 0) {
        return;
    }
    const double rot_dim = static_cast<double>(rope_half * 2);
    std::vector<double> inv_freq(rope_half);
    for (int k = 0; k < rope_half; ++k) {
        inv_freq[k] = 1.0 / std::pow(rope_theta, static_cast<double>(2 * k) / rot_dim);
    }
    for (int p = 0; p < max_pos; ++p) {
        uint16_t* row_c = out_cos_u16 + static_cast<size_t>(p) * rope_half;
        uint16_t* row_s = out_sin_u16 + static_cast<size_t>(p) * rope_half;
        const double pd = static_cast<double>(p);
        for (int k = 0; k < rope_half; ++k) {
            const double ang = pd * inv_freq[k];
            row_c[k] = f32_to_u16(static_cast<float>(std::cos(ang)));
            row_s[k] = f32_to_u16(static_cast<float>(std::sin(ang)));
        }
    }
}

void pplx_pack_bucket_f16(
    const uint16_t* embed_table_u16,
    int vocab_size,
    int hidden_dim,
    int seq_len,
    int chunk_size,
    int rope_half,
    const int32_t* token_ids,
    const int32_t* doc_ids,
    const int32_t* pos_ids,
    const uint16_t* rope_cos_u16,
    const uint16_t* rope_sin_u16,
    int max_pos,
    uint16_t* out_h_u16,
    uint16_t* out_attn_bias_u16,
    uint16_t* out_cos32_u16,
    uint16_t* out_sin32_u16,
    uint16_t* out_tok_valid_u16,
    uint16_t* out_gdn_carry_u16) {
    if (!embed_table_u16 || !token_ids || !doc_ids || !pos_ids ||
        !rope_cos_u16 || !rope_sin_u16 || !out_h_u16 ||
        !out_attn_bias_u16 || !out_cos32_u16 || !out_sin32_u16 ||
        !out_tok_valid_u16 || !out_gdn_carry_u16) {
        return;
    }
    if (vocab_size <= 0 || hidden_dim <= 0 || seq_len <= 0 || chunk_size <= 0 ||
        rope_half <= 0 || max_pos <= 0 || (seq_len % chunk_size) != 0) {
        return;
    }

    const uint16_t one_u16 = f32_to_u16(1.0f);
    const uint16_t zero_u16 = 0;
    const uint16_t mask_neg_u16 = f32_to_u16(-10000.0f);

    // 1. Gather row-major [S, H] embeddings into reusable thread-local scratch then transpose to [H, S].
    const size_t total_elems = static_cast<size_t>(seq_len) * static_cast<size_t>(hidden_dim);
    if (tl_gathered_f16.size() < total_elems) {
        tl_gathered_f16.resize(total_elems);
    }
    uint16_t* gathered = tl_gathered_f16.data();
    const size_t row_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
    const size_t rope_bytes = static_cast<size_t>(rope_half) * sizeof(uint16_t);

    for (int s = 0; s < seq_len; ++s) {
        const int32_t did = doc_ids[s];
        const bool is_valid = (did >= 0);
        out_tok_valid_u16[s] = is_valid ? one_u16 : zero_u16;
        uint16_t* dst_row = gathered + static_cast<size_t>(s) * hidden_dim;

        if (is_valid) {
            const int32_t tid = token_ids[s];
            if (tid >= 0 && tid < vocab_size) {
                std::memcpy(
                    dst_row,
                    embed_table_u16 + static_cast<size_t>(tid) * hidden_dim,
                    row_bytes);
            } else {
                std::memset(dst_row, 0, row_bytes);
            }
            const int32_t pid = std::max(0, std::min(pos_ids[s], max_pos - 1));
            std::memcpy(
                out_cos32_u16 + static_cast<size_t>(s) * rope_half,
                rope_cos_u16 + static_cast<size_t>(pid) * rope_half,
                rope_bytes);
            std::memcpy(
                out_sin32_u16 + static_cast<size_t>(s) * rope_half,
                rope_sin_u16 + static_cast<size_t>(pid) * rope_half,
                rope_bytes);
        } else {
            std::memset(dst_row, 0, row_bytes);
            // Identity RoPE at pad positions (cos=1, sin=0) avoids garbage values.
            uint16_t* cos_row = out_cos32_u16 + static_cast<size_t>(s) * rope_half;
            uint16_t* sin_row = out_sin32_u16 + static_cast<size_t>(s) * rope_half;
            for (int k = 0; k < rope_half; ++k) {
                cos_row[k] = one_u16;
                sin_row[k] = zero_u16;
            }
        }
    }

    neon_transpose_seq_hidden_f16(gathered, out_h_u16, seq_len, hidden_dim);

    // 2. Block-diagonal bidirectional attention bias [S, S]:
    //    same non-negative doc_id -> 0.0; pad query -> pad keys are 0.0 (avoids NaN softmax).
    for (int r = 0; r < seq_len; ++r) {
        const int32_t dr = doc_ids[r];
        uint16_t* row = out_attn_bias_u16 + static_cast<size_t>(r) * seq_len;
        if (dr >= 0) {
            for (int c = 0; c < seq_len; ++c) {
                row[c] = (doc_ids[c] == dr) ? zero_u16 : mask_neg_u16;
            }
        } else {
            for (int c = 0; c < seq_len; ++c) {
                row[c] = (doc_ids[c] < 0) ? zero_u16 : mask_neg_u16;
            }
        }
    }

    // 3. Inter-chunk carry flags for GatedDeltaNet:
    //    Carry state into chunk c only if c>0 continues the exact same document across the boundary.
    const int n_chunks = seq_len / chunk_size;
    for (int c = 0; c < n_chunks; ++c) {
        if (c == 0) {
            out_gdn_carry_u16[c] = zero_u16;
        } else {
            const int boundary = c * chunk_size;
            const bool carry = (doc_ids[boundary] >= 0) && (doc_ids[boundary] == doc_ids[boundary - 1]);
            out_gdn_carry_u16[c] = carry ? one_u16 : zero_u16;
        }
    }
}

void pplx_pool_and_project_f32(
    const uint16_t* h_out_u16,
    int hidden_dim,
    int seq_len,
    int n_spans,
    const int32_t* span_starts,
    const int32_t* span_ends,
    const float* final_norm_f32,
    const float* contextual_proj_f32,
    int proj_rows,
    int out_dim,
    int apply_tanh_int8,
    int apply_l2_norm,
    float* out_f32) {
    if (!h_out_u16 || !span_starts || !span_ends || !contextual_proj_f32 || !out_f32 ||
        n_spans <= 0 || hidden_dim <= 0 || seq_len <= 0 || proj_rows <= 0 || out_dim <= 0) {
        return;
    }
    const int eff_dim = std::min(out_dim, proj_rows);
    const size_t pooled_elems = static_cast<size_t>(n_spans) * static_cast<size_t>(hidden_dim);
    if (tl_pooled_f32.size() < pooled_elems) {
        tl_pooled_f32.resize(pooled_elems);
    }
    float* pooled = tl_pooled_f32.data();

    if (final_norm_f32 == nullptr) {
        // Fast path: ANE stage already ran final RMSNorm; sum contiguous slice [start, end) per channel.
        for (int k = 0; k < n_spans; ++k) {
            float* dst = pooled + static_cast<size_t>(k) * hidden_dim;
            const int s0 = std::max(0, span_starts[k]);
            const int s1 = std::min(seq_len, span_ends[k]);
            if (s1 <= s0) {
                std::memset(dst, 0, static_cast<size_t>(hidden_dim) * sizeof(float));
                continue;
            }
            const float inv_len = 1.0f / static_cast<float>(s1 - s0);
            for (int c = 0; c < hidden_dim; ++c) {
                const uint16_t* ch = h_out_u16 + static_cast<size_t>(c) * seq_len;
                float32x4_t acc = vdupq_n_f32(0.0f);
                int s = s0;
                for (; s + 4 <= s1; s += 4) {
                    float16x4_t hv;
                    std::memcpy(&hv, ch + s, sizeof(hv));
                    acc = vaddq_f32(acc, vcvt_f32_f16(hv));
                }
                float sum = vaddvq_f32(acc);
                for (; s < s1; ++s) {
                    sum += u16_to_f32(ch[s]);
                }
                dst[c] = sum * inv_len;
            }
        }
    } else {
        // Optional path: apply final RMSNorm across channels per token before span pooling.
        if (tl_tok_norm_f32.size() < static_cast<size_t>(hidden_dim)) {
            tl_tok_norm_f32.resize(static_cast<size_t>(hidden_dim));
        }
        float* tok_buf = tl_tok_norm_f32.data();
        for (int k = 0; k < n_spans; ++k) {
            float* dst = pooled + static_cast<size_t>(k) * hidden_dim;
            std::memset(dst, 0, static_cast<size_t>(hidden_dim) * sizeof(float));
            const int s0 = std::max(0, span_starts[k]);
            const int s1 = std::min(seq_len, span_ends[k]);
            if (s1 <= s0) {
                continue;
            }
            const float inv_len = 1.0f / static_cast<float>(s1 - s0);
            for (int s = s0; s < s1; ++s) {
                double sq = 0.0;
                for (int c = 0; c < hidden_dim; ++c) {
                    float v = u16_to_f32(h_out_u16[static_cast<size_t>(c) * seq_len + s]);
                    tok_buf[c] = v;
                    sq += static_cast<double>(v) * static_cast<double>(v);
                }
                const float rrms = 1.0f / std::sqrt(static_cast<float>(sq / hidden_dim) + 1e-6f);
                for (int c = 0; c < hidden_dim; ++c) {
                    dst[c] += tok_buf[c] * rrms * final_norm_f32[c];
                }
            }
            for (int c = 0; c < hidden_dim; ++c) {
                dst[c] *= inv_len;
            }
        }
    }

    // contextual_proj_f32 is [proj_rows, hidden_dim] row-major; slice first eff_dim rows cleanly for MRL.
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        n_spans,
        eff_dim,
        hidden_dim,
        1.0f,
        pooled,
        hidden_dim,
        contextual_proj_f32,
        hidden_dim,
        0.0f,
        out_f32,
        eff_dim);

    if (apply_tanh_int8) {
        const size_t n_elem = static_cast<size_t>(n_spans) * eff_dim;
        for (size_t i = 0; i < n_elem; ++i) {
            float q = std::nearbyint(std::tanh(out_f32[i]) * 127.0f);
            if (q > 127.0f) q = 127.0f;
            if (q < -128.0f) q = -128.0f;
            out_f32[i] = q;
        }
    }

    if (apply_l2_norm) {
        pplx_l2_normalize_f32(out_f32, n_spans, eff_dim);
    }
}

void pplx_l2_normalize_f32(float* vecs, int rows, int dim) {
    if (!vecs || rows <= 0 || dim <= 0) {
        return;
    }
    for (int r = 0; r < rows; ++r) {
        float* row = vecs + static_cast<size_t>(r) * dim;
        fast_shared::l2_normalize_f32(row, row, dim);
    }
}

}  // extern "C"
