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

// Compile a CoreML `.mlpackage` bundle to `.mlmodelc` using macOS `CoreML.framework` (zero `coremltools` needed).
// Returns 0 on success, -1 on failure.
int laya_ane_compile_mlpackage(const char* mlpackage_path, const char* dst_mlmodelc_path);

// Inspect CoreML MLComputePlan device assignment (ANE / GPU / CPU op counts) for function `"main"` or `fn_name`.
void laya_ane_inspect_plan(const char* mlmodelc_path, int compute_mode, int* out_counts3);
void laya_ane_inspect_plan_fn(const char* mlmodelc_path, const char* fn_name, int compute_mode, int* out_counts3);

// Load a compiled Laya-Typed-421M CoreML ANE bundle (.mlmodelc), supporting single-function, multifunction (`fn_name`),
// legacy `[1, 1, 1, S]` scorer, upgraded `[1, 1, 1, M]` `marker_map` scorer, and prebuilt `NMTZ` `.mlmodelc` graphs.
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

void* laya_ane_load_fn(
    const char* mlmodelc_path,
    const char* fn_name,
    int H,
    int S,
    int local_half_win,
    int pipe_depth,
    int compute_mode,
    const uint16_t* word_fp16,
    int vocab_size,
    const uint16_t* type_fp16,
    int pad_id
);

// Return `> 0` (`M`, typically 32) when the loaded model uses pre-scorer `marker_map` gather, or `0` for legacy `[1,1,1,S]`.
int laya_ane_max_markers(void* handle);

// Execute batched single-pass prefill scoring across `N` sequences of padded length `S` (legacy `[N, S]` interface).
double laya_ane_exec_batch(
    void* handle,
    int N,
    const int32_t* tok_ids_all,   // [N, S]
    const int32_t* valid_lens,    // [N]
    const int32_t* qtypes,        // [N] (0=choice, 1=score, 2=noul)
    float* out_logits_f32,        // [N, S]
    int use_async_pipe
);

// Execute batched marker-gathered ANE scoring across `N` items, writing `[N, marker_stride]` float32 option logits.
// Works uniformly across multifunction `marker_map`, prebuilt `NMTZ`, and legacy `[1, 1, 1, S]` models.
double laya_ane_exec_markers_batch(
    void* handle,
    int N,
    const int32_t* tok_ids_all,      // [N, S]
    const int32_t* valid_lens,       // [N]
    const int32_t* qtypes,           // [N]
    const int32_t* marker_pos_all,   // [N, marker_stride]
    const int32_t* marker_counts,    // [N]
    int marker_stride,
    float* out_marker_logits_f32,    // [N, marker_stride]
    int use_async_pipe
);

// Return the last CoreML error description captured by `handle` (empty string if none).
const char* laya_ane_last_error(void* handle);

// Free handle and associated CoreML / IOSurface buffers.
void laya_ane_free(void* handle);

#ifdef __cplusplus
}
#endif
