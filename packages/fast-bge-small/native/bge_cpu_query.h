#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void bge_cpu_init(
    const float* wt0, int vocab_size,
    const float* pos, int max_pos,
    const float* emb_ln_w, const float* emb_ln_b,
    const float* w_qkv_all, const float* b_qkv_all,
    const float* w_o_all, const float* b_o_all,
    const float* ln1_w_all, const float* ln1_b_all,
    const float* w_up_all, const float* b_up_all,
    const float* w_dn_all, const float* b_dn_all,
    const float* ln2_w_all, const float* ln2_b_all
);

double bge_cpu_query(const int32_t* tok_ids, int S, float* out_vec384, int iters);

#ifdef __cplusplus
}
#endif
