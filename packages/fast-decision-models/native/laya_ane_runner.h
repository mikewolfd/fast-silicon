#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute unit mode constants aligned 1:1 with `fast_shared::ComputeMode` (`fast_shared_apple.hpp`).
typedef enum LayaComputeMode {
    LAYA_COMPUTE_CPU_AND_ANE = 0,  // MLComputeUnitsCPUAndNeuralEngine
    LAYA_COMPUTE_CPU_ONLY = 1,     // MLComputeUnitsCPUOnly
    LAYA_COMPUTE_ALL = 2,          // MLComputeUnitsAll
} LayaComputeMode;

// Inspect CoreML MLComputePlan device assignment (ANE / GPU / CPU op counts).
// compute_mode: LAYA_COMPUTE_CPU_AND_ANE (0), LAYA_COMPUTE_CPU_ONLY (1), or LAYA_COMPUTE_ALL (2).
// Writes [ane_ops, gpu_ops, cpu_ops] into out_counts3 (length 3).
void laya_ane_inspect_plan(const char* mlmodelc_path, int compute_mode, int* out_counts3);

// Load a compiled Laya-Typed-421M CoreML ANE bundle (.mlmodelc) and allocate `pipe_depth` async slots.
// Copies FP16 token (`[vocab_size, H]`) and question-type (`[3, H]`) embedding tables for LUT gather + NEON 8x8 transpose.
// Returns opaque handle on success, NULL on failure.
void* laya_ane_load(
    const char* mlmodelc_path,
    int H,
    int S,
    int pipe_depth,
    int compute_mode,
    const uint16_t* word_fp16,
    int vocab_size,
    const uint16_t* type_fp16
);

// Execute batched single-pass prefill scoring across `N` sequences of padded length `S`.
// Writes raw per-position float32 scorer logits `[N, S]` into `out_logits_f32` and returns wall-clock seconds
// (`>= 0.0` on success, `< 0.0` on CoreML prediction error; query `laya_ane_last_error(handle)` for details).
double laya_ane_exec_batch(
    void* handle,
    int N,
    const int32_t* tok_ids_all,   // [N, S]
    const int32_t* valid_lens,    // [N]
    const int32_t* qtypes,        // [N] (0=choice, 1=score, 2=noul)
    float* out_logits_f32,        // [N, S]
    int use_async_pipe
);

// Return the last CoreML error description captured by `handle` (empty string if none).
const char* laya_ane_last_error(void* handle);

// Free handle and associated CoreML / IOSurface buffers.
void laya_ane_free(void* handle);

#ifdef __cplusplus
}
#endif
