#ifndef PPLX_COREML_RUNNER_H
#define PPLX_COREML_RUNNER_H

#include <cstdint>

extern "C" {

// Direct writable pointers into an acquired CoreML MLMultiArray ring slot
// (valid only when all input MLMultiArrays are dense 4D layouts).
struct PplxSlotBuffers {
    int slot_idx;
    uint16_t* in_h;
    uint16_t* in_bias;
    uint16_t* in_cos32;
    uint16_t* in_sin32;
    uint16_t* in_tok_valid;
    uint16_t* in_gdn_carry;
};

// Return thread-local diagnostic message from the most recent error (empty string if none).
const char* pplx_ane_last_error(void);

// Load one or more staged CoreML .mlmodelc bundles forming a full seq_len bucket
// pipeline on `compute_units` (0=CPU_AND_NE, 1=CPU_ONLY, 2=ALL, 3=CPU_AND_GPU).
// When compute_units == 0 (CPU_AND_NE), verifies ANE residency via MLComputePlan
// for each stage (rejects stages with ane_ops == 0 or CPU/GPU fallback ops unless
// FAST_PPLX_ALLOW_ANE_FALLBACK=1).
void* pplx_ane_load_stages(
    const char* const* model_paths,
    int num_stages,
    int seq_len,
    int hidden_dim,
    int chunk_size,
    int rope_half,
    int pipe_depth,
    int compute_units);

// Acquire an available ring slot for zero-copy input packing directly into
// MLMultiArray backing storage. Returns slot_idx >= 0 on success, or -1 if the
// slot's MLMultiArrays are non-dense (caller should fall back to pplx_ane_submit).
int pplx_ane_acquire_slot(void* runner, PplxSlotBuffers* out_bufs);

// Submit a previously acquired ring slot without copying input arrays. The slot
// remains reserved until pplx_ane_release_slot is called after pplx_ane_wait_slot.
void* pplx_ane_submit_slot(void* runner, int slot_idx);

// Wait for a job submitted via pplx_ane_submit_slot, returning elapsed seconds
// (-1.0 on error) and setting *out_h_ptr to dense [hidden_dim, seq_len] FP16 output
// storage valid until pplx_ane_release_slot(runner, *out_slot_idx).
double pplx_ane_wait_slot(
    void* job,
    const uint16_t** out_h_ptr,
    int* out_slot_idx);

// Release a ring slot acquired via pplx_ane_acquire_slot.
void pplx_ane_release_slot(void* runner, int slot_idx);

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
// Returns bytes written (>0) on success, or <0 on failure (still populating out_json_buf with
// an error JSON object when buf_len allows).
int pplx_ane_inspect_plan(
    const char* model_path,
    int compute_units,
    char* out_json_buf,
    int buf_len);

void pplx_ane_free(void* runner);

}  // extern "C"

#endif  // PPLX_COREML_RUNNER_H
