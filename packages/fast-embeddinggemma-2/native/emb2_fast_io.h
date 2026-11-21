#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Gather token embeddings [V, 512] -> [N, 512, S], build packed [N, S, S] attention bias,
// and gather per-position SWA [N, S, 128] and full [N, S, 256] cos/sin RoPE tables.
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
);

// Same as emb2_prep_batch, reading directly from clean-page int8 [V, 512] + fp16 scale [V] mmap tables.
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
);

// Vectorized NEON row gather [M, 512] float16 from int8 [V, 512] + fp16 scale [V] * extra_scale.
void emb2_gather_scaled_i8(
    const int8_t* emb_i8,
    const uint16_t* emb_scale_f16,
    float extra_scale,
    const int32_t* ids,
    int vocab_size,
    int M,
    uint16_t* out_f16
);

// Mean-pool per-doc spans from [N, S, 512] FP16 output, project [n_docs, 512] @ emb_proj_T[512, :proj_dim]
// via Accelerate cblas_sgemm, and L2-normalize into out_embs[n_docs, proj_dim].
void emb2_pool_project_batch_dim(
    int S,
    const uint16_t* out_all,
    int n_docs,
    const int32_t* doc_lane_se,
    const float* emb_proj_T,
    float* scratch_pooled,
    float* out_embs,
    int proj_dim
);

void emb2_pool_project_batch(
    int S,
    const uint16_t* out_all,
    int n_docs,
    const int32_t* doc_lane_se,
    const float* emb_proj_T,
    float* scratch_pooled,
    float* out_embs
);

#ifdef __cplusplus
}
#endif
