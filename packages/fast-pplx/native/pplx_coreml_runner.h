#ifndef PPLX_COREML_RUNNER_H
#define PPLX_COREML_RUNNER_H

#include <cstdint>

extern "C" {

// Load one or more staged CoreML .mlmodelc bundles forming a full seq_len bucket
// pipeline on `compute_units` (0=CPU_AND_NE, 1=CPU_ONLY, 2=ALL, 3=CPU_AND_GPU).
void* pplx_ane_load_stages(
    const char* const* model_paths,
    int num_stages,
    int seq_len,
    int hidden_dim,
    int chunk_size,
    int rope_half,
    int pipe_depth,
    int compute_units);

// Submit one async pipelined staged execution. Copies the 6 FP16 input arrays
// into an available ring slot immediately so caller staging arrays can be reused
// upon return; writes final [hidden_dim, seq_len] FP16 output to h_out_u16
// before pplx_ane_wait completes.
void* pplx_ane_submit(
    void* runner,
    const uint16_t* h_in_u16,
    const uint16_t* attn_bias_u16,
    const uint16_t* cos32_u16,
    const uint16_t* sin32_u16,
    const uint16_t* tok_valid_u16,
    const uint16_t* gdn_carry_u16,
    uint16_t* h_out_u16);

// Wait for an async job handle, return elapsed wall seconds (-1.0 on error), and free job.
double pplx_ane_wait(void* job);

// Synchronous staged execution helper (submit + wait).
double pplx_ane_exec(
    void* runner,
    const uint16_t* h_in_u16,
    const uint16_t* attn_bias_u16,
    const uint16_t* cos32_u16,
    const uint16_t* sin32_u16,
    const uint16_t* tok_valid_u16,
    const uint16_t* gdn_carry_u16,
    uint16_t* h_out_u16);

// Inspect MLComputePlan for a single .mlmodelc stage and write summary JSON into out_json_buf.
// Returns bytes written (or <0 on failure).
int pplx_ane_inspect_plan(
    const char* model_path,
    int compute_units,
    char* out_json_buf,
    int buf_len);

void pplx_ane_free(void* runner);

}  // extern "C"

#endif  // PPLX_COREML_RUNNER_H
