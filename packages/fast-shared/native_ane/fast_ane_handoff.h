#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FastAneStatus {
    FAST_ANE_OK = 0,
    FAST_ANE_UNAVAILABLE = 1,
    FAST_ANE_INVALID_ARG = 2,
    FAST_ANE_COMPILE_FAILED = 3,
    FAST_ANE_LOAD_FAILED = 4,
    FAST_ANE_EVAL_FAILED = 5,
    FAST_ANE_METAL_FAILED = 6,
    FAST_ANE_TIMEOUT = 7,
} FastAneStatus;

typedef struct FastAneContext FastAneContext;
typedef struct FastAneProgram FastAneProgram;
typedef struct FastAneSurface FastAneSurface;
typedef struct FastAneWave FastAneWave;

/* Allocates and returns a NUL-terminated JSON string describing runtime
 * private _ANEClient / MTLSharedEvent / Metal / IOSurface compatibility.
 * Caller must release with fast_ane_free_string(). */
char* fast_ane_probe_compat_json(void);
void fast_ane_free_string(char* str);

/* Context lifecycle (holds MTLDevice, MTLCommandQueue, MTLSharedEvent,
 * runtime-compiled Metal library, and verified private ANE class/client state).
 * Returns FAST_ANE_OK even when ANE is unavailable if allow_unavailable != 0;
 * query fast_ane_context_ane_available() to inspect ANE availability. */
FastAneStatus fast_ane_context_create(int32_t allow_unavailable,
                                      FastAneContext** out_ctx,
                                      char* err_buf,
                                      size_t err_cap);
void fast_ane_context_destroy(FastAneContext* ctx);
int32_t fast_ane_context_ane_available(const FastAneContext* ctx);
int32_t fast_ane_context_shared_event_available(const FastAneContext* ctx);
int32_t fast_ane_context_iosurface_metal_available(const FastAneContext* ctx);

/* Surface allocation & host transfer.
 * pixel_format: e.g. 'L00h' (0x4C303068) for fp16, 'L008' (0x4C303038) for int8. */
FastAneStatus fast_ane_surface_create(uint32_t width,
                                      uint32_t height,
                                      uint32_t bytes_per_elem,
                                      uint32_t pixel_format,
                                      FastAneSurface** out_surf,
                                      char* err_buf,
                                      size_t err_cap);
void fast_ane_surface_destroy(FastAneSurface* surf);
uint32_t fast_ane_surface_width(const FastAneSurface* surf);
uint32_t fast_ane_surface_height(const FastAneSurface* surf);
uint32_t fast_ane_surface_row_bytes(const FastAneSurface* surf);
size_t fast_ane_surface_alloc_bytes(const FastAneSurface* surf);
FastAneStatus fast_ane_surface_write(FastAneSurface* surf,
                                     const void* src,
                                     uint32_t src_row_bytes,
                                     uint32_t rows);
FastAneStatus fast_ane_surface_read(const FastAneSurface* surf,
                                    void* dst,
                                    uint32_t dst_row_bytes,
                                    uint32_t rows);

/* Compiles and loads a MIL text program + optional constant weight blob
 * (raw constant data; this library prepends ANE's 128-byte chunked blob header
 * unless data_is_preformatted != 0). */
FastAneStatus fast_ane_program_create(FastAneContext* ctx,
                                      const char* mil_text,
                                      size_t mil_len,
                                      const void* weight_data,
                                      size_t weight_bytes,
                                      uint32_t weight_dtype_code,
                                      int32_t data_is_preformatted,
                                      uint32_t qos,
                                      FastAneProgram** out_prog,
                                      char* err_buf,
                                      size_t err_cap);

/* Compiles and loads an existing directory containing `model.mil` and optional `weights/weight.bin`
 * without copying or deleting the directory on destroy. */
FastAneStatus fast_ane_program_create_from_dir(FastAneContext* ctx,
                                               const char* bundle_dir,
                                               uint32_t qos,
                                               FastAneProgram** out_prog,
                                               char* err_buf,
                                               size_t err_cap);
void fast_ane_program_destroy(FastAneProgram* prog);

/* Synchronous ANE evaluation via _ANEClient (no shared event). */
FastAneStatus fast_ane_program_eval(FastAneContext* ctx,
                                    FastAneProgram* prog,
                                    FastAneSurface* const* inputs,
                                    uint32_t n_inputs,
                                    FastAneSurface* const* outputs,
                                    uint32_t n_outputs,
                                    double* out_ms,
                                    char* err_buf,
                                    size_t err_cap);

/* GPU (Metal compute) -> ANE -> GPU (Metal compute) zero-CPU handoff test / eval.
 * mode == 0: shared event GPU->ANE->GPU async pipelined handoff (requires MTLSharedEvent).
 * mode == 1: CPU-synchronized fallback (waitUntilCompleted between stages) for comparison.
 * Runs a Metal pre-op on host_in -> surf_in, triggers prog(surf_in -> surf_out),
 * then runs Metal post-op surf_out -> host_out. Populates timing telemetry. */
typedef struct FastAneHandoffMetrics {
    double gpu_pre_ms;
    double ane_submit_ms;
    double gpu_post_and_total_ms;
    double total_ms;
    int32_t used_shared_event;
    int32_t used_iosurface_metal_buffer;
    uint64_t final_shared_event_value;
} FastAneHandoffMetrics;

FastAneStatus fast_ane_handoff_roundtrip(FastAneContext* ctx,
                                         FastAneProgram* prog,
                                         FastAneSurface* surf_in,
                                         FastAneSurface* surf_out,
                                         const uint16_t* host_in_fp16,
                                         uint16_t* host_out_fp16,
                                         uint32_t count_elems,
                                         int32_t mode,
                                         uint32_t timeout_ms,
                                         FastAneHandoffMetrics* out_metrics,
                                         char* err_buf,
                                         size_t err_cap);

/* Pipelined FFN split wave:
 *   1. GPU `ane_ffn_rotate_pack_fp16` applies 128-wide post-norm Hadamard (+ optional A8 quantize)
 *      and packs transposed [H, M] directly into `surf_in` IOSurface Metal buffer, signaling MTLSharedEvent.
 *   2. `_ANEClient` submits `n_segs` ANE FFN segments (each waiting on `surf_in` event and signaling completion).
 *   3. GPU `ane_ffn_join_fp16` waits on ANE completion event(s) in hardware and sums segment outputs (+ optional tail).
 * Split into `begin` / `finish` so the caller can run concurrent GPU tail FFN work while ANE executes. */
typedef struct FastAneWaveMetrics {
    double gpu_submit_ms;
    double ane_submit_ms;
    double finish_wait_ms;
    double total_wall_ms;
    int32_t used_shared_event;
} FastAneWaveMetrics;

FastAneStatus fast_ane_ffn_wave_begin(FastAneContext* ctx,
                                      FastAneProgram* const* seg_progs,
                                      FastAneSurface* const* seg_out_surfs,
                                      uint32_t n_segs,
                                      FastAneSurface* surf_in,
                                      const uint16_t* host_in_fp16,
                                      const uint16_t* host_signs_fp16,
                                      uint32_t tokens,
                                      uint32_t hidden_size,
                                      int32_t quantize_a8,
                                      int32_t use_shared_event,
                                      uint32_t timeout_ms,
                                      FastAneWave** out_wave,
                                      char* err_buf,
                                      size_t err_cap);

FastAneStatus fast_ane_ffn_wave_finish(FastAneWave* wave,
                                       const uint16_t* opt_gpu_tail_fp16,
                                       uint16_t* host_out_fp16,
                                       FastAneWaveMetrics* out_metrics,
                                       char* err_buf,
                                       size_t err_cap);

void fast_ane_ffn_wave_destroy(FastAneWave* wave);

#ifdef __cplusplus
}
#endif
