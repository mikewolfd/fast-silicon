#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "bge_cpu_query.h"

static inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct CPULayer {
    std::vector<float> w_qkv;
    std::vector<float> b_qkv;
    std::vector<float> w_o;
    std::vector<float> b_o;
    std::vector<float> ln1_w;
    std::vector<float> ln1_b;
    std::vector<float> w_up;
    std::vector<float> b_up;
    std::vector<float> w_dn;
    std::vector<float> b_dn;
    std::vector<float> ln2_w;
    std::vector<float> ln2_b;
};

static int g_vocab_size = 0;
static int g_max_pos = 0;
static std::vector<float> g_wt0_f32;
static std::vector<float> g_pos_f32;
static std::vector<float> g_emb_ln_w;
static std::vector<float> g_emb_ln_b;
static std::vector<CPULayer> g_layers;

static std::vector<float> g_h;
static std::vector<float> g_qkv;
static std::vector<float> g_attn_out;
static std::vector<float> g_ao;
static std::vector<float> g_up;
static std::vector<float> g_dn;
static std::vector<float> g_scores;

static inline void layernorm_res_f32(
    const float* __restrict__ x,
    const float* __restrict__ res,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int S
) {
    for (int s = 0; s < S; ++s) {
        const float* xp = x + s * 384;
        float* yp = y + s * 384;
        float sum = 0.0f;
        if (res) {
            const float* rp = res + s * 384;
            for (int c = 0; c < 384; ++c) {
                float v = xp[c] + rp[c];
                yp[c] = v;
                sum += v;
            }
        } else {
            for (int c = 0; c < 384; ++c) {
                float v = xp[c];
                yp[c] = v;
                sum += v;
            }
        }
        float mean = sum * (1.0f / 384.0f);
        float var = 0.0f;
        for (int c = 0; c < 384; ++c) {
            float d = yp[c] - mean;
            yp[c] = d;
            var += d * d;
        }
        float inv_std = 1.0f / std::sqrt(var * (1.0f / 384.0f) + 1e-12f);
        for (int c = 0; c < 384; ++c) {
            yp[c] = yp[c] * inv_std * w[c] + b[c];
        }
    }
}

static inline void bias_add_rows(float* __restrict__ m, const float* __restrict__ bias, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* rp = m + r * cols;
        for (int c = 0; c < cols; ++c) rp[c] += bias[c];
    }
}

extern "C" void bge_cpu_init(
    const float* wt0, int vocab_size,
    const float* pos, int max_pos,
    const float* emb_ln_w, const float* emb_ln_b,
    const float* w_qkv_all, const float* b_qkv_all,
    const float* w_o_all, const float* b_o_all,
    const float* ln1_w_all, const float* ln1_b_all,
    const float* w_up_all, const float* b_up_all,
    const float* w_dn_all, const float* b_dn_all,
    const float* ln2_w_all, const float* ln2_b_all
) {
    g_vocab_size = vocab_size;
    g_max_pos = max_pos;
    g_wt0_f32.assign(wt0, wt0 + static_cast<size_t>(vocab_size) * 384);
    g_pos_f32.assign(pos, pos + static_cast<size_t>(max_pos) * 384);
    g_emb_ln_w.assign(emb_ln_w, emb_ln_w + 384);
    g_emb_ln_b.assign(emb_ln_b, emb_ln_b + 384);
    g_layers.resize(12);
    for (int l = 0; l < 12; ++l) {
        CPULayer& L = g_layers[l];
        L.w_qkv.assign(w_qkv_all + l * 1152 * 384, w_qkv_all + (l + 1) * 1152 * 384);
        L.b_qkv.assign(b_qkv_all + l * 1152, b_qkv_all + (l + 1) * 1152);
        L.w_o.assign(w_o_all + l * 384 * 384, w_o_all + (l + 1) * 384 * 384);
        L.b_o.assign(b_o_all + l * 384, b_o_all + (l + 1) * 384);
        L.ln1_w.assign(ln1_w_all + l * 384, ln1_w_all + (l + 1) * 384);
        L.ln1_b.assign(ln1_b_all + l * 384, ln1_b_all + (l + 1) * 384);
        L.w_up.assign(w_up_all + l * 1536 * 384, w_up_all + (l + 1) * 1536 * 384);
        L.b_up.assign(b_up_all + l * 1536, b_up_all + (l + 1) * 1536);
        L.w_dn.assign(w_dn_all + l * 384 * 1536, w_dn_all + (l + 1) * 384 * 1536);
        L.b_dn.assign(b_dn_all + l * 384, b_dn_all + (l + 1) * 384);
        L.ln2_w.assign(ln2_w_all + l * 384, ln2_w_all + (l + 1) * 384);
        L.ln2_b.assign(ln2_b_all + l * 384, ln2_b_all + (l + 1) * 384);
    }
    g_h.resize(512 * 384);
    g_qkv.resize(512 * 1152);
    g_attn_out.resize(512 * 384);
    g_ao.resize(512 * 384);
    g_up.resize(512 * 1536);
    g_dn.resize(512 * 384);
    g_scores.resize(512 * 1536);
}

extern "C" double bge_cpu_query(const int32_t* tok_ids, int S, float* out_vec384, int iters) {
    if (g_layers.empty() || !tok_ids || !out_vec384 || S <= 0 || iters <= 0) return -1.0;
    int sl = std::clamp(S, 1, std::min(512, g_max_pos));
    const float inv_s32 = 1.0f / std::sqrt(32.0f);
    const float inv_s2 = 0.7071067811865475f;
    double t0 = now_sec();
    for (int it = 0; it < iters; ++it) {
        for (int s = 0; s < sl; ++s) {
            int tid = tok_ids[s];
            if (static_cast<unsigned>(tid) >= static_cast<unsigned>(g_vocab_size)) tid = 100;
            const float* wt = g_wt0_f32.data() + static_cast<size_t>(tid) * 384;
            const float* wp = g_pos_f32.data() + static_cast<size_t>(s) * 384;
            float* hp = g_ao.data() + s * 384;
            vDSP_vadd(wt, 1, wp, 1, hp, 1, 384);
        }
        layernorm_res_f32(g_ao.data(), nullptr, g_emb_ln_w.data(), g_emb_ln_b.data(), g_h.data(), sl);

        for (int l = 0; l < 12; ++l) {
            const CPULayer& L = g_layers[l];
            int q_rows = (l == 11) ? 1 : sl;

            if (l < 11) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, sl, 1152, 384, 1.0f, g_h.data(), 384, L.w_qkv.data(), 384, 0.0f, g_qkv.data(), 1152);
                bias_add_rows(g_qkv.data(), L.b_qkv.data(), sl, 1152);
            } else {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, 384, 384, 1.0f, g_h.data(), 384, L.w_qkv.data(), 384, 0.0f, g_qkv.data(), 1152);
                for (int c = 0; c < 384; ++c) g_qkv[c] += L.b_qkv[c];
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, sl, 768, 384, 1.0f, g_h.data(), 384, L.w_qkv.data() + 384 * 384, 384, 0.0f, g_qkv.data() + 384, 1152);
                for (int s = 0; s < sl; ++s) {
                    float* kp = g_qkv.data() + s * 1152 + 384;
                    for (int c = 0; c < 768; ++c) kp[c] += L.b_qkv[384 + c];
                }
            }

            for (int h = 0; h < 12; ++h) {
                int h_off = h * 32;
                const float* q_ptr = g_qkv.data() + h_off;
                const float* k_ptr = g_qkv.data() + 384 + h_off;
                const float* v_ptr = g_qkv.data() + 768 + h_off;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q_rows, sl, 32, inv_s32, q_ptr, 1152, k_ptr, 1152, 0.0f, g_scores.data(), sl);
                for (int r = 0; r < q_rows; ++r) {
                    float* row = g_scores.data() + r * sl;
                    float mxv = row[0];
                    for (int c = 1; c < sl; ++c) if (row[c] > mxv) mxv = row[c];
                    float sum = 0.0f;
                    for (int c = 0; c < sl; ++c) {
                        float e = std::exp(row[c] - mxv);
                        row[c] = e;
                        sum += e;
                    }
                    float inv = 1.0f / sum;
                    for (int c = 0; c < sl; ++c) row[c] *= inv;
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, q_rows, 32, sl, 1.0f, g_scores.data(), sl, v_ptr, 1152, 0.0f, g_attn_out.data() + h_off, 384);
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q_rows, 384, 384, 1.0f, g_attn_out.data(), 384, L.w_o.data(), 384, 0.0f, g_ao.data(), 384);
            bias_add_rows(g_ao.data(), L.b_o.data(), q_rows, 384);
            layernorm_res_f32(g_ao.data(), g_h.data(), L.ln1_w.data(), L.ln1_b.data(), g_h.data(), q_rows);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q_rows, 1536, 384, 1.0f, g_h.data(), 384, L.w_up.data(), 384, 0.0f, g_up.data(), 1536);
            bias_add_rows(g_up.data(), L.b_up.data(), q_rows, 1536);
            int total_up = q_rows * 1536;
            for (int i = 0; i < total_up; ++i) {
                float x = g_up[i];
                g_up[i] = x * (0.5f * (1.0f + std::erff(x * inv_s2)));
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q_rows, 384, 1536, 1.0f, g_up.data(), 1536, L.w_dn.data(), 1536, 0.0f, g_dn.data(), 384);
            bias_add_rows(g_dn.data(), L.b_dn.data(), q_rows, 384);
            layernorm_res_f32(g_dn.data(), g_h.data(), L.ln2_w.data(), L.ln2_b.data(), g_h.data(), q_rows);
        }

        float n2 = 0.0f;
        vDSP_svesq(g_h.data(), 1, &n2, 384);
        float inv = 1.0f / std::sqrt(n2 + 1e-24f);
        vDSP_vsmul(g_h.data(), 1, &inv, out_vec384, 1, 384);
    }
    return now_sec() - t0;
}
