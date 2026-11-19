#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Inspect CoreML MLComputePlan device assignment (ANE / GPU / CPU op counts) with an 8s timeout guard.
// compute_mode matches fast_shared::ComputeMode (0=kCpuAndNeuralEngine, 1=kCpuOnly, 2=kAll).
// Returns ANE op count on success, or -1 on error / timeout.
int gemma4_px_inspect_plan(const char* mlmodelc_path, int compute_mode);

// Extended CoreML MLComputePlan inspection separating runtime CPU ops from compile-time constexpr weight ops.
// Returns ANE op count on success, or -1 on error / timeout.
int gemma4_px_inspect_plan_ex(
    const char* mlmodelc_path,
    int compute_mode,
    int* out_ane,
    int* out_cpu_runtime,
    int* out_cpu_constexpr,
    int* out_gpu
);

// Load an Early14/Early23 ANE prefix chunk CoreML bundle (.mlmodelc) with unlocked IOSurface I/O slots
// (default 8-slot output ring) and compile the 2-dispatch Metal KV_i8 scatter shaders.
// Returns opaque handle on success, NULL on failure.
void* gemma4_px_load(const char* mlmodelc_path, int S, int P_swa, int P_full, int max_pipe);

// Extended loader allowing a custom-sized unlocked IOSurface output ring (e.g. 64 slots for async rolling).
void* gemma4_px_load_ex(const char* mlmodelc_path, int S, int P_swa, int P_full, int max_pipe, int out_ring_slots);

// Returns 1 if CoreML outputBackings CVPixelBuffer/IOSurface pointer identity + Metal IOSurface binding succeeded.
int gemma4_px_is_zero_copy(void* handle);

// Returns the number of allocated output IOSurface ring slots on handle.
int gemma4_px_out_ring_size(void* handle);

// Query runtime tensor dimensions inferred from the loaded CoreML .mlmodelc bundle (supports E2B and E4B).
// Returns 0 on success, -1 on NULL handle. Any out_* pointer may be NULL.
int gemma4_px_get_dims(
    void* handle,
    int* out_H,
    int* out_C_pli,
    int* out_n_swa,
    int* out_n_full_past,
    int* out_n_full_out,
    int* out_n_kv
);

// Free handle and associated CoreML / IOSurface / Metal resources.
void gemma4_px_free(void* handle);

// Execute batched Early14 chunk inference across B sequences using up to pipe_depth async pipeline slots.
// Output pointers may be NULL if the caller only warms up or inspects ring slots.
// Returns elapsed wall-clock seconds, or < 0.0 on failure.
double gemma4_px_run_batch(
    void* handle,
    int B,
    const uint16_t* h_all,      // [B, H, S] float16 bits
    const uint16_t* pli_all,    // [B, C_pli, S] float16 bits
    const uint16_t* cos_s_all,  // [1 or B, S, 256]
    const uint16_t* sin_s_all,
    const uint16_t* cos_f_all,  // [1 or B, S, 512]
    const uint16_t* sin_f_all,
    const uint16_t* kp_s_all,   // [1 or B, n_swa, P_swa, 256]
    const uint16_t* vp_s_all,
    const uint16_t* kp_f_all,   // [1 or B, n_full_past, P_full, 512]
    const uint16_t* vp_f_all,
    const uint16_t* ms_all,     // [1 or B, S, P_swa + S]
    const uint16_t* mf_all,     // [1 or B, S, P_full + S]
    uint16_t* k_swa_out,        // [B, n_swa, S, 256] (optional, NULL skips CPU readback)
    uint16_t* v_swa_out,        // [B, n_swa, S, 256]
    uint16_t* k_full_out,       // [B, n_full, S, 512]
    uint16_t* v_full_out,       // [B, n_full, S, 512]
    int shared_past_kv,
    int shared_geom,
    int pipe_depth
);

// Zero-CPU-output-copy ANE batch execution directly into specified `out_ring_ids[0..B-1]` IOSurface slots.
// Chunk `chunk_idx` covers absolute positions `lcp + chunk_idx * S ...`; it needs `lcp + chunk_idx * S <= P_full`.
// Full past  = `sys_full_{k,v}` followed by every earlier chunk of the same doc, read from `prev_ring_ids`.
// SWA past   = chunk 0: shared `sys_swa_{k,v}` [n_swa, 512, 256]; chunk 1: `sys_swa_{k,v}` [n_swa, 256, 256] + chunk 0;
//              chunk >= 2: the two chunks before it (`sys_swa_*` unused, may be NULL).
// Replaces `gemma4_px_run_into_ring`, which took at most two earlier chunks; the new name keeps a stale dylib from
// being called with the new argument list.
// Returns ANE wall-clock seconds, or < 0.0 on failure.
double gemma4_px_run_chunk_into_ring(
    void* handle,
    int B,
    int chunk_idx,
    int lcp,
    const uint16_t* h_all,
    const uint16_t* pli_all,
    const uint16_t* cos_s,
    const uint16_t* sin_s,
    const uint16_t* cos_f,
    const uint16_t* sin_f,
    const uint16_t* sys_swa_k,
    const uint16_t* sys_swa_v,
    const uint16_t* sys_full_k,   // [n_full_past, lcp, 512] (or NULL if lcp == 0)
    const uint16_t* sys_full_v,
    const uint16_t* ms,
    const uint16_t* mf,
    const int32_t* prev_ring_ids, // [chunk_idx, B] row-major: ring slot of doc b's chunk j at [j * B + b]; NULL for chunk 0
    const int32_t* out_ring_ids,  // [B] destination ring indices in [0, out_ring_slots), distinct from the doc's earlier slots
    int pipe_depth
);

// Bind MLX persistent slot-cache KV_i8 UMA buffers `(k, k_s, v, v_s)` via 16 KiB page-aligned zero-copy MTLBuffers.
// `swa_ptrs` / `swa_nbytes` have length `n_swa * 4` (`k, k_s, v, v_s` per SWA layer).
// `full_ptrs` / `full_nbytes` have length `n_full * 4` (`k, k_s, v, v_s` per Full layer).
void* gemma4_px_bind_slot_cache_i8(
    void* handle,
    int B_slots,
    int swa_cap,
    int full_cap,
    int n_swa,
    const uint64_t* swa_ptrs,
    const uint64_t* swa_nbytes,
    int n_full,
    const uint64_t* full_ptrs,
    const uint64_t* full_nbytes
);

void gemma4_px_unbind_slot_cache_i8(void* bind_handle);

// Execute fused 2-dispatch Metal `bridge_scatter_swa12` + `bridge_scatter_full3` directly from unlocked
// IOSurface output ring slots into bound MLX `KV_i8` slot-cache buffers (`0` CPU copies).
// Returns GPU execution time in seconds (`GPUEndTime - GPUStartTime`), or < 0.0 on error.
double gemma4_px_scatter_ring_kv_i8(
    void* handle,
    void* bind_handle,
    int n_items,
    const int32_t* ring_slot_ids,
    const int32_t* doc_slots,
    const int32_t* priv_offs,
    const int32_t* valid_lens
);

#ifdef __cplusplus
}
#endif
