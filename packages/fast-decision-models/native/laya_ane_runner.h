#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Inspect CoreML MLComputePlan device assignment (ANE / GPU / CPU op counts).
// compute_mode: 0=MLComputeUnitsCPUOnly, 1=MLComputeUnitsCPUAndNeuralEngine, 2=MLComputeUnitsAll.
// Writes [ane_ops, gpu_ops, cpu_ops] into out_counts3 (length 3).
void laya_ane_inspect_plan(const char* mlmodelc_path, int compute_mode, int* out_counts3);

// Load a compiled Laya-Typed-421M CoreML ANE bundle (.mlmodelc) and allocate `pipe_depth` async slots.
// Copies FP16 token (`[vocab_size, H]`) and question-type (`[3, H]`) embedding tables for NEON gather+transpose.
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
// Writes raw per-position float32 scorer logits `[N, S]` into `out_logits_f32` and returns wall-clock seconds.
double laya_ane_exec_batch(
    void* handle,
    int N,
    const int32_t* tok_ids_all,   // [N, S]
    const int32_t* valid_lens,    // [N]
    const int32_t* qtypes,        // [N] (0=choice, 1=score, 2=noul)
    float* out_logits_f32,        // [N, S]
    int use_async_pipe
);

// Free handle and associated CoreML buffers.
void laya_ane_free(void* handle);

#ifdef __cplusplus
}
#endif
