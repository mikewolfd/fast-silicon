#ifndef PPLX_FAST_IO_H
#define PPLX_FAST_IO_H

#include <cstdint>

extern "C" {

// Precompute [max_pos, rope_half] cosine and sine tables in IEEE FP16.
// For Qwen3.5 0.6B: head_dim=256, partial_rotary_factor=0.25 -> rope_half=32.
void pplx_build_rope_table_f16(
    uint16_t* out_cos_u16,
    uint16_t* out_sin_u16,
    int max_pos,
    int rope_half,
    double rope_theta);

// Pack a single bucket of S tokens (S % chunk_size == 0, H == 1024):
// - Gathers embeddings from F16 [vocab_size, H] into scratch and transposes
//   via NEON 8x8 register transpose to [H, S] channel-major FP16.
// - Writes block-diagonal FP16 attn_bias [S, S] (0.0 for same doc_id or pad-to-pad,
//   -10000.0 elsewhere so softmax never hits all-(-10000) NaN).
// - Copies per-position RoPE cos32/sin32 [S, rope_half].
// - Writes tok_valid [S] (1.0f16 for doc_id >= 0, 0.0f16 for pad) and
//   gdn_carry [S / chunk_size] (1.0f16 iff chunk c>0 starts in the same doc as c*C-1).
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
    uint16_t* out_gdn_carry_u16);

// Directly pool ANE output [hidden_dim, seq_len] FP16 over span ranges [start, end),
// multiply by FP32 contextual_projection [proj_rows, hidden_dim] (row-major, optionally
// truncated to out_dim <= proj_rows for MRL), optionally apply final RMSNorm (if
// final_norm_f32 != nullptr), optionally apply round(tanh(x)*127) and/or L2 norm.
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
    float* out_f32);

// Convenience C wrapper matching standard row normalization.
void pplx_l2_normalize_f32(float* vecs, int rows, int dim);

}  // extern "C"

#endif  // PPLX_FAST_IO_H
