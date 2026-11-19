#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Inspect MLComputePlan placement (ANE / GPU / CPU op counts) via stdout.
void emb2_inspect_plan(const char* mlmodelc_path, int compute_mode);

// Load a packed CoreML bundle for sequence length S with pipe_depth async prediction slots.
// compute_mode: 0 = CPUAndNeuralEngine (ANE), 1 = CPUOnly (AMX/SME), 2 = All.
void* emb2_pk_load_units(const char* mlmodelc_path, int S, int pipe_depth, int compute_mode);

// Submit N packed sequences asynchronously and return an inflight handle (must be awaited by emb2_pk_wait_queue).
void* emb2_pk_submit_queue(
    void* handle,
    int N,
    const uint16_t* h_all,
    const uint16_t* b_all,
    const uint16_t* cs_all,
    const uint16_t* ss_all,
    const uint16_t* cf_all,
    const uint16_t* sf_all,
    uint16_t* out_all
);

// Wait for a submitted inflight queue to complete and return elapsed milliseconds.
double emb2_pk_wait_queue(void* flight);

// Execute N packed sequences of length S asynchronously across allocated slots.
// Returns elapsed milliseconds.
double emb2_pk_exec_queue(
    void* handle,
    int N,
    const uint16_t* h_all,
    const uint16_t* b_all,
    const uint16_t* cs_all,
    const uint16_t* ss_all,
    const uint16_t* cf_all,
    const uint16_t* sf_all,
    uint16_t* out_all
);

void emb2_pk_free(void* handle);

#ifdef __cplusplus
}
#endif
