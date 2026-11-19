#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

int bge_ane_init_tables(const uint16_t* wt0_fp16, int vocab_size, const uint16_t* pos_fp16, int max_pos);
void bge_ane_inspect_plan(const char* mlmodelc_path, int compute_mode);
int bge_ane_load_bucket(const char* mlmodelc_path, int B, int S, int n_slots, int compute_mode);
double bge_ane_query(int bucket_id, const int32_t* tok_ids, int len, float* out_vec384, int iters);
double bge_ane_encode_bucket(
    int bucket_id,
    int n_docs,
    const int32_t* flat_toks,
    const int32_t* offsets,
    const int32_t* doc_orig_idx,
    float* out_vecs,
    int pipe_depth
);
void bge_ane_free_all(void);

#ifdef __cplusplus
}
#endif
