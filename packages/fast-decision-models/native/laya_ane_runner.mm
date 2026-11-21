// Native Objective-C++ CoreML ANE Runner for Laya-Typed-Decisions-421M (`laya_ane_runner.mm`).
// Supports:
//   1. Multi-function (`s64..s512`) & single-function `[1, S, 1, 32]` `marker_map` W8A16 graphs
//   2. Prebuilt `NMTZ/laya-typed-decisions-421m-coreml-ane` (`L192..L640`) graphs + native `.mlpackage` compile
//   3. Backward-compatible legacy 3-input `[1, 1, 1, S]` W8A16 graphs
// Scalar FP16 LUT gather + NEON 8x8 transpose into IOSurface-backed [1, 1024, 1, S], pipe_depth=4 async queue.

#include "laya_ane_runner.h"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <arm_neon.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if __has_include("fast_shared_apple.hpp")
#include "fast_shared_apple.hpp"
#define FAST_DECISION_HAS_SHARED_APPLE 1
#else
#define FAST_DECISION_HAS_SHARED_APPLE 0
#endif

namespace {

inline double now_sec() {
#if FAST_DECISION_HAS_SHARED_APPLE
    return fast_shared::now_sec();
#else
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
#endif
}

enum LayaGraphKind {
    kGraphLegacyPos = 0,   // tok_emb_in, ext_mask, type_vec -> pos_logits [1,1,1,S]
    kGraphMarkerW8 = 1,    // tok_emb_in, ext_mask, type_vec, marker_map -> pos_logits [1,1,1,M]
    kGraphNmtz = 2,        // embeddings, full_mask, local_mask, type_vectors, marker_map -> logits [1,1,1,M]
};

struct LayaAneHandle {
    MLModel* model = nil;
    LayaGraphKind graph_kind = kGraphLegacyPos;
    int H = 1024;
    int S = 512;
    int M = 0;                // >0 when marker_map input is present (typically 32)
    int local_half_win = 64;  // ModernBERT local_attention // 2
    int pad_id = 50283;
    bool mask_on_axis1 = false;
    int pipe_depth = 4;
    int vocab_size = 0;
    NSString* out_logit_name = @"pos_logits";

    std::vector<uint16_t> word_emb;  // [vocab_size, H] fp16
    std::vector<uint16_t> type_emb;  // [3, H] fp16

    // Preallocated per-slot [S, H] gather scratch buffers (zero per-call heap allocation).
    std::vector<std::vector<uint16_t>> slot_scratch;
    // Owned IOSurface-backed CVPixelBuffers (released cleanly in ~LayaAneHandle).
    std::vector<CVPixelBufferRef> owned_pbs;

    std::vector<MLMultiArray*> in_emb;
    std::vector<MLMultiArray*> in_mask;
    std::vector<MLMultiArray*> in_local_mask;
    std::vector<MLMultiArray*> in_type;
    std::vector<MLMultiArray*> in_marker_map;
    std::vector<MLMultiArray*> out_logits;
    std::vector<MLDictionaryFeatureProvider*> feats;
    std::vector<MLPredictionOptions*> opts;
    dispatch_queue_t queue = nil;

    std::mutex err_mu;
    std::string last_error;

    ~LayaAneHandle() {
        for (CVPixelBufferRef pb : owned_pbs) {
            if (pb) CVPixelBufferRelease(pb);
        }
        owned_pbs.clear();
    }
};

MLMultiArray* make_slot_f16_array(
    int height,
    int width,
    NSArray<NSNumber*>* shape,
    std::vector<CVPixelBufferRef>& owned_pbs,
    NSError** err_out
) {
    if (height > 0 && width > 0) {
        NSDictionary* attrs = @{
            (NSString*)kCVPixelBufferIOSurfacePropertiesKey: @{},
            (NSString*)kCVPixelBufferMetalCompatibilityKey: @YES,
        };
        CVPixelBufferRef pb = nullptr;
        CVReturn cv_rc = CVPixelBufferCreate(
            kCFAllocatorDefault,
            static_cast<size_t>(width),
            static_cast<size_t>(height),
            kCVPixelFormatType_OneComponent16Half,
            (__bridge CFDictionaryRef)attrs,
            &pb
        );
        if (cv_rc == kCVReturnSuccess && pb != nullptr) {
            MLMultiArray* arr = [[MLMultiArray alloc] initWithPixelBuffer:pb shape:shape];
            if (arr != nil) {
                owned_pbs.push_back(pb);
                if (err_out) *err_out = nil;
                return arr;
            }
            CVPixelBufferRelease(pb);
        }
    }
    return [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:err_out];
}

inline void zero_mla_f16(MLMultiArray* arr, int rows, int cols) {
    if (!arr) return;
    [arr getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
        (void)strides;
        if (mutableBytes && size > 0) {
            std::memset(mutableBytes, 0, static_cast<size_t>(size));
        } else if (mutableBytes && rows > 0 && cols > 0) {
            std::memset(mutableBytes, 0, static_cast<size_t>(rows) * cols * sizeof(uint16_t));
        }
    }];
}

inline void transpose_emb_to_mla(
    const uint16_t* scratch,
    int H,
    int S,
    MLMultiArray* emb_arr
) {
    [emb_arr getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
        (void)size;
        uint16_t* dst_emb = static_cast<uint16_t*>(mutableBytes);
        const size_t row_stride = static_cast<size_t>((strides.count >= 2) ? strides[1].integerValue : S);
        const size_t col_stride = static_cast<size_t>((strides.count >= 4) ? strides[3].integerValue : 1);

        if (col_stride == 1 && (S % 8 == 0) && (H % 8 == 0)) {
            for (int s = 0; s < S; s += 8) {
                for (int c = 0; c < H; c += 8) {
                    const uint16_t* sp = scratch + static_cast<size_t>(s) * H + c;
                    uint16x8_t r0 = vld1q_u16(sp + 0 * H);
                    uint16x8_t r1 = vld1q_u16(sp + 1 * H);
                    uint16x8_t r2 = vld1q_u16(sp + 2 * H);
                    uint16x8_t r3 = vld1q_u16(sp + 3 * H);
                    uint16x8_t r4 = vld1q_u16(sp + 4 * H);
                    uint16x8_t r5 = vld1q_u16(sp + 5 * H);
                    uint16x8_t r6 = vld1q_u16(sp + 6 * H);
                    uint16x8_t r7 = vld1q_u16(sp + 7 * H);

                    uint16x8_t t0 = vtrn1q_u16(r0, r1);
                    uint16x8_t t1 = vtrn2q_u16(r0, r1);
                    uint16x8_t t2 = vtrn1q_u16(r2, r3);
                    uint16x8_t t3 = vtrn2q_u16(r2, r3);
                    uint16x8_t t4 = vtrn1q_u16(r4, r5);
                    uint16x8_t t5 = vtrn2q_u16(r4, r5);
                    uint16x8_t t6 = vtrn1q_u16(r6, r7);
                    uint16x8_t t7 = vtrn2q_u16(r6, r7);

                    uint32x4_t u0 = vtrn1q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2));
                    uint32x4_t u2 = vtrn2q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2));
                    uint32x4_t u1 = vtrn1q_u32(vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3));
                    uint32x4_t u3 = vtrn2q_u32(vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3));
                    uint32x4_t u4 = vtrn1q_u32(vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6));
                    uint32x4_t u6 = vtrn2q_u32(vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6));
                    uint32x4_t u5 = vtrn1q_u32(vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7));
                    uint32x4_t u7 = vtrn2q_u32(vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7));

                    uint16_t* dp = dst_emb + static_cast<size_t>(c) * row_stride + s;
                    vst1q_u16(dp + 0 * row_stride, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4))));
                    vst1q_u16(dp + 4 * row_stride, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4))));
                    vst1q_u16(dp + 1 * row_stride, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5))));
                    vst1q_u16(dp + 5 * row_stride, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5))));
                    vst1q_u16(dp + 2 * row_stride, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6))));
                    vst1q_u16(dp + 6 * row_stride, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6))));
                    vst1q_u16(dp + 3 * row_stride, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7))));
                    vst1q_u16(dp + 7 * row_stride, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7))));
                }
            }
        } else {
            for (int c = 0; c < H; ++c) {
                uint16_t* dp = dst_emb + static_cast<size_t>(c) * row_stride;
                for (int s = 0; s < S; ++s) {
                    dp[static_cast<size_t>(s) * col_stride] = scratch[static_cast<size_t>(s) * H + c];
                }
            }
        }
    }];
}

inline void prep_marker_map_f16(
    MLMultiArray* mm_arr,
    int S,
    int M,
    const int32_t* marker_pos,
    int k_cnt
) {
    if (!mm_arr || M <= 0) return;
    const float16_t one16 = static_cast<float16_t>(1.0f);
    uint16_t u_one = 0;
    std::memcpy(&u_one, &one16, 2);

    [mm_arr getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
        uint16_t* dst = static_cast<uint16_t*>(mutableBytes);
        const size_t st_s = static_cast<size_t>((strides.count >= 2) ? strides[1].integerValue : M);
        const size_t st_m = static_cast<size_t>((strides.count >= 4) ? strides[3].integerValue : 1);
        if (size > 0) {
            std::memset(dst, 0, static_cast<size_t>(size));
        } else {
            for (int s = 0; s < S; ++s) {
                std::memset(dst + static_cast<size_t>(s) * st_s, 0, static_cast<size_t>(M) * sizeof(uint16_t));
            }
        }
        const int eff_k = std::max(0, std::min(M, k_cnt));
        if (marker_pos) {
            for (int j = 0; j < eff_k; ++j) {
                int mp = std::max(0, std::min(S - 1, static_cast<int>(marker_pos[j])));
                dst[static_cast<size_t>(mp) * st_s + static_cast<size_t>(j) * st_m] = u_one;
            }
        }
    }];
}

inline void prep_laya_slot_fp16(
    const LayaAneHandle* h,
    int slot_idx,
    const int32_t* tok_ids,
    int valid_len,
    int qtype,
    const int32_t* marker_pos,
    int marker_cnt
) {
    const int H = h->H;
    const int S = h->S;
    const int vl = std::max(0, std::min(S, valid_len));
    uint16_t* scratch = const_cast<LayaAneHandle*>(h)->slot_scratch[static_cast<size_t>(slot_idx)].data();

    const float16_t zero16 = static_cast<float16_t>(0.0f);
    const float16_t neg16 = static_cast<float16_t>(-10000.0f);
    uint16_t u_zero = 0, u_neg = 0;
    std::memcpy(&u_zero, &zero16, 2);
    std::memcpy(&u_neg, &neg16, 2);

    const int qt = std::max(0, std::min(2, qtype));
    const uint16_t* src_type = h->type_emb.data() + static_cast<size_t>(qt) * H;
    [h->in_type[static_cast<size_t>(slot_idx)] getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
        (void)size;
        uint16_t* dst_type = static_cast<uint16_t*>(mutableBytes);
        const NSInteger st_c = (strides.count >= 2) ? strides[1].integerValue : 1;
        if (st_c == 1) {
            std::memcpy(dst_type, src_type, static_cast<size_t>(H) * sizeof(uint16_t));
        } else {
            for (int c = 0; c < H; ++c) {
                dst_type[static_cast<size_t>(c) * static_cast<size_t>(st_c)] = src_type[c];
            }
        }
    }];

    const int eff_pad = (h->pad_id >= 0 && h->pad_id < h->vocab_size) ? h->pad_id : 0;
    for (int s = 0; s < S; ++s) {
        if (s >= vl) {
            if (h->graph_kind == kGraphNmtz) {
                std::memcpy(
                    scratch + static_cast<size_t>(s) * H,
                    h->word_emb.data() + static_cast<size_t>(eff_pad) * H,
                    static_cast<size_t>(H) * sizeof(uint16_t)
                );
            } else {
                std::memset(scratch + static_cast<size_t>(s) * H, 0, static_cast<size_t>(H) * sizeof(uint16_t));
            }
            continue;
        }
        int tid = tok_ids[s];
        if (tid < 0 || tid >= h->vocab_size) tid = 0;
        std::memcpy(
            scratch + static_cast<size_t>(s) * H,
            h->word_emb.data() + static_cast<size_t>(tid) * H,
            static_cast<size_t>(H) * sizeof(uint16_t)
        );
    }

    transpose_emb_to_mla(scratch, H, S, h->in_emb[static_cast<size_t>(slot_idx)]);

    if (h->graph_kind == kGraphNmtz) {
        // full_mask [1, S, 1, S]: row k is 0.0 for k < vl else -10000.0
        [h->in_mask[static_cast<size_t>(slot_idx)] getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
            (void)size;
            uint16_t* dst = static_cast<uint16_t*>(mutableBytes);
            const size_t row_st = static_cast<size_t>((strides.count >= 2) ? strides[1].integerValue : S);
            const size_t col_st = static_cast<size_t>((strides.count >= 4) ? strides[3].integerValue : 1);
            for (int k = 0; k < S; ++k) {
                const uint16_t val = (k < vl) ? u_zero : u_neg;
                uint16_t* row = dst + static_cast<size_t>(k) * row_st;
                if (col_st == 1) {
                    std::fill_n(row, S, val);
                } else {
                    for (int q = 0; q < S; ++q) row[static_cast<size_t>(q) * col_st] = val;
                }
            }
        }];
        // local_mask [1, S, 1, S]: (k < vl) && ((|q - k| <= half_win) || (q >= vl)) -> 0.0 else -10000.0
        const int hw = h->local_half_win;
        [h->in_local_mask[static_cast<size_t>(slot_idx)] getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
            (void)size;
            uint16_t* dst = static_cast<uint16_t*>(mutableBytes);
            const size_t row_st = static_cast<size_t>((strides.count >= 2) ? strides[1].integerValue : S);
            const size_t col_st = static_cast<size_t>((strides.count >= 4) ? strides[3].integerValue : 1);
            for (int k = 0; k < S; ++k) {
                uint16_t* row = dst + static_cast<size_t>(k) * row_st;
                if (k >= vl) {
                    if (col_st == 1) {
                        std::fill_n(row, S, u_neg);
                    } else {
                        for (int q = 0; q < S; ++q) row[static_cast<size_t>(q) * col_st] = u_neg;
                    }
                    continue;
                }
                const int q_lo = std::max(0, k - hw);
                const int q_hi = std::min(vl - 1, k + hw);
                if (col_st == 1) {
                    if (q_lo > 0) std::fill_n(row, q_lo, u_neg);
                    if (q_hi >= q_lo) std::fill_n(row + q_lo, q_hi - q_lo + 1, u_zero);
                    if (vl > q_hi + 1) std::fill_n(row + q_hi + 1, vl - (q_hi + 1), u_neg);
                    if (S > vl) std::fill_n(row + vl, S - vl, u_zero);
                } else {
                    for (int q = 0; q < S; ++q) {
                        const bool ok = (q >= vl) || (std::abs(q - k) <= hw);
                        row[static_cast<size_t>(q) * col_st] = ok ? u_zero : u_neg;
                    }
                }
            }
        }];
    } else {
        const bool axis1 = h->mask_on_axis1;
        [h->in_mask[static_cast<size_t>(slot_idx)] getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
            (void)size;
            uint16_t* dst_mask = static_cast<uint16_t*>(mutableBytes);
            const NSInteger st_s = axis1
                ? ((strides.count >= 2) ? strides[1].integerValue : 1)
                : ((strides.count >= 4) ? strides[3].integerValue : 1);
            for (int s = 0; s < S; ++s) {
                dst_mask[static_cast<size_t>(s) * static_cast<size_t>(st_s)] = (s < vl) ? u_zero : u_neg;
            }
        }];
    }

    if (h->M > 0) {
        prep_marker_map_f16(h->in_marker_map[static_cast<size_t>(slot_idx)], S, h->M, marker_pos, marker_cnt);
    }
}

inline void copy_out_to_f32(MLMultiArray* arr, float* dst, int count) {
    [arr getBytesWithHandler:^(const void* bytes, NSInteger size) {
        (void)size;
        NSArray<NSNumber*>* strides = arr.strides;
        const NSInteger st_s = (strides.count >= 4) ? strides[3].integerValue : 1;
        if (arr.dataType == MLMultiArrayDataTypeFloat32) {
            const float* pf = static_cast<const float*>(bytes);
            if (st_s == 1) {
                std::memcpy(dst, pf, static_cast<size_t>(count) * sizeof(float));
            } else {
                for (int i = 0; i < count; ++i) dst[i] = pf[static_cast<size_t>(i) * static_cast<size_t>(st_s)];
            }
        } else {
            const float16_t* p16 = static_cast<const float16_t*>(bytes);
            if (st_s == 1) {
                for (int i = 0; i < count; ++i) dst[i] = static_cast<float>(p16[i]);
            } else {
                for (int i = 0; i < count; ++i) dst[i] = static_cast<float>(p16[static_cast<size_t>(i) * static_cast<size_t>(st_s)]);
            }
        }
    }];
}

MLComputeUnits resolve_units(int mode) {
#if FAST_DECISION_HAS_SHARED_APPLE
    return fast_shared::resolve_compute_units(static_cast<fast_shared::ComputeMode>(mode));
#else
    switch (mode) {
        case LAYA_COMPUTE_CPU_AND_ANE:
            return MLComputeUnitsCPUAndNeuralEngine;
        case LAYA_COMPUTE_CPU_ONLY:
            return MLComputeUnitsCPUOnly;
        case LAYA_COMPUTE_ALL:
        default:
            return MLComputeUnitsAll;
    }
#endif
}

inline void record_error(LayaAneHandle* h, const char* ctx, NSError* err) {
    std::string msg = ctx;
    msg += ": ";
    msg += (err && err.localizedDescription) ? err.localizedDescription.UTF8String : "unknown CoreML error";
    fprintf(stderr, "%s\n", msg.c_str());
    std::lock_guard<std::mutex> lk(h->err_mu);
    if (h->last_error.empty()) {
        h->last_error = std::move(msg);
    }
}

}  // namespace

extern "C" {

int laya_ane_compile_mlpackage(const char* mlpackage_path, const char* dst_mlmodelc_path) {
    @autoreleasepool {
        if (!mlpackage_path || !dst_mlmodelc_path) return -1;
        NSURL* src_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlpackage_path]];
        NSURL* dst_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:dst_mlmodelc_path]];
        NSError* err = nil;
        NSURL* tmp_url = [MLModel compileModelAtURL:src_url error:&err];
        if (!tmp_url || err) {
            fprintf(stderr, "laya_ane_compile_mlpackage failed (%s): %s\n",
                    mlpackage_path, err ? err.localizedDescription.UTF8String : "null");
            return -1;
        }
        NSFileManager* fm = [NSFileManager defaultManager];
        NSURL* parent_url = [dst_url URLByDeletingLastPathComponent];
        [fm createDirectoryAtURL:parent_url withIntermediateDirectories:YES attributes:nil error:nil];
        [fm removeItemAtURL:dst_url error:nil];
        if (![fm moveItemAtURL:tmp_url toURL:dst_url error:&err]) {
            if (![fm copyItemAtURL:tmp_url toURL:dst_url error:&err]) {
                fprintf(stderr, "laya_ane_compile_mlpackage move/copy failed (%s): %s\n",
                        dst_mlmodelc_path, err ? err.localizedDescription.UTF8String : "unknown");
                [fm removeItemAtURL:tmp_url error:nil];
                return -1;
            }
            [fm removeItemAtURL:tmp_url error:nil];
        }
        return 0;
    }
}

void laya_ane_inspect_plan_fn(const char* mlmodelc_path, const char* fn_name, int compute_mode, int* out_counts3) {
    @autoreleasepool {
        if (!mlmodelc_path) {
            if (out_counts3) out_counts3[0] = out_counts3[1] = out_counts3[2] = 0;
            return;
        }
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = resolve_units(compute_mode);
        NSString* target_fn = (fn_name && fn_name[0] != '\0') ? [NSString stringWithUTF8String:fn_name] : @"main";
        if (![target_fn isEqualToString:@"main"]) {
            if (@available(macOS 15.0, *)) {
                cfg.functionName = target_fn;
            }
        }
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block int ane_ops = 0, gpu_ops = 0, cpu_ops = 0;
        [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
            if (err) {
                fprintf(stderr, "laya_ane_inspect_plan failed (%s, fn=%s): %s\n",
                        mlmodelc_path, target_fn.UTF8String, err.localizedDescription.UTF8String);
            } else if (plan && plan.modelStructure.program) {
                auto* fn = plan.modelStructure.program.functions[target_fn];
                if (!fn) fn = plan.modelStructure.program.functions[@"main"];
                for (MLModelStructureProgramOperation* op in fn.block.operations) {
                    id pref = [plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
                    if ([pref isKindOfClass:[MLNeuralEngineComputeDevice class]]) ane_ops++;
                    else if ([pref isKindOfClass:[MLGPUComputeDevice class]]) gpu_ops++;
                    else if ([pref isKindOfClass:[MLCPUComputeDevice class]]) cpu_ops++;
                }
            }
            dispatch_semaphore_signal(sem);
        }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        if (out_counts3) {
            out_counts3[0] = ane_ops;
            out_counts3[1] = gpu_ops;
            out_counts3[2] = cpu_ops;
        }
    }
}

void laya_ane_inspect_plan(const char* mlmodelc_path, int compute_mode, int* out_counts3) {
    laya_ane_inspect_plan_fn(mlmodelc_path, "main", compute_mode, out_counts3);
}

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
) {
    @autoreleasepool {
        if (!mlmodelc_path || !word_fp16 || !type_fp16 || H <= 0 || S <= 0 || vocab_size <= 0) {
            return nullptr;
        }
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = resolve_units(compute_mode);
        bool explicit_fn = (fn_name && fn_name[0] != '\0' && std::strcmp(fn_name, "main") != 0);
        if (explicit_fn) {
            if (@available(macOS 15.0, *)) {
                cfg.functionName = [NSString stringWithUTF8String:fn_name];
            }
        }
        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if ((!model || err) && !explicit_fn) {
            if (@available(macOS 15.0, *)) {
                err = nil;
                cfg.functionName = [NSString stringWithFormat:@"s%d", S];
                model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
            }
        }
        if (!model || err) {
            fprintf(stderr, "laya_ane_load_fn failed (%s, fn=%s): %s\n",
                    mlmodelc_path, fn_name ? fn_name : "main", err ? err.localizedDescription.UTF8String : "null");
            return nullptr;
        }

        auto* h = new LayaAneHandle();
        h->model = model;
        h->H = H;
        h->S = S;
        h->local_half_win = (local_half_win > 0) ? local_half_win : 64;
        h->pad_id = pad_id;
        h->pipe_depth = std::max(1, pipe_depth);
        h->vocab_size = vocab_size;
        h->word_emb.assign(word_fp16, word_fp16 + static_cast<size_t>(vocab_size) * H);
        h->type_emb.assign(type_fp16, type_fp16 + static_cast<size_t>(3) * H);
        h->slot_scratch.resize(static_cast<size_t>(h->pipe_depth));

        NSDictionary<NSString*, MLFeatureDescription*>* inDescs = model.modelDescription.inputDescriptionsByName;
        NSDictionary<NSString*, MLFeatureDescription*>* outDescs = model.modelDescription.outputDescriptionsByName;

        const bool is_nmtz = (inDescs[@"embeddings"] != nil && inDescs[@"full_mask"] != nil && inDescs[@"local_mask"] != nil);
        const bool has_marker_map = (inDescs[@"marker_map"] != nil);
        if (is_nmtz) {
            h->graph_kind = kGraphNmtz;
        } else if (has_marker_map) {
            h->graph_kind = kGraphMarkerW8;
        } else {
            h->graph_kind = kGraphLegacyPos;
        }

        if (has_marker_map) {
            NSArray<NSNumber*>* mmShape = inDescs[@"marker_map"].multiArrayConstraint.shape;
            h->M = (mmShape.count >= 4) ? mmShape[3].intValue : 32;
        } else {
            h->M = 0;
        }

        if (!is_nmtz && inDescs[@"ext_mask"] != nil) {
            NSArray<NSNumber*>* mShape = inDescs[@"ext_mask"].multiArrayConstraint.shape;
            h->mask_on_axis1 = (mShape.count >= 4 && mShape[1].intValue == S && mShape[3].intValue == 1);
        }

        // Resolve primary logit output name (`pos_logits` or NMTZ `[1, 1, 1, M]` output)
        MLFeatureDescription* outDesc = outDescs[@"pos_logits"];
        if (outDesc != nil) {
            h->out_logit_name = @"pos_logits";
        } else {
            for (NSString* key in outDescs) {
                MLMultiArrayConstraint* mc = outDescs[key].multiArrayConstraint;
                if (mc && mc.shape.count >= 4 && mc.shape[1].intValue == 1) {
                    outDesc = outDescs[key];
                    h->out_logit_name = key;
                    break;
                }
            }
        }
        const int out_w = (h->M > 0) ? h->M : S;
        MLMultiArrayDataType outType = outDesc ? outDesc.multiArrayConstraint.dataType : MLMultiArrayDataTypeFloat16;

        for (int i = 0; i < h->pipe_depth; ++i) {
            h->slot_scratch[static_cast<size_t>(i)].assign(static_cast<size_t>(S) * H, 0u);
            NSError* e = nil;
            MLMultiArray* emb = make_slot_f16_array(H, S, @[@1, @(H), @1, @(S)], h->owned_pbs, &e);
            MLMultiArray* msk = nil;
            MLMultiArray* lmsk = nil;
            if (!e) {
                if (h->graph_kind == kGraphNmtz) {
                    msk = make_slot_f16_array(S, S, @[@1, @(S), @1, @(S)], h->owned_pbs, &e);
                    if (!e) lmsk = make_slot_f16_array(S, S, @[@1, @(S), @1, @(S)], h->owned_pbs, &e);
                } else if (h->mask_on_axis1) {
                    msk = make_slot_f16_array(S, 1, @[@1, @(S), @1, @1], h->owned_pbs, &e);
                } else {
                    msk = make_slot_f16_array(1, S, @[@1, @1, @1, @(S)], h->owned_pbs, &e);
                }
            }
            MLMultiArray* typ = !e ? make_slot_f16_array(H, 1, @[@1, @(H), @1, @1], h->owned_pbs, &e) : nil;
            MLMultiArray* mmap = (!e && h->M > 0) ? make_slot_f16_array(S, h->M, @[@1, @(S), @1, @(h->M)], h->owned_pbs, &e) : nil;
            MLMultiArray* out = nil;
            if (!e) {
                if (outType == MLMultiArrayDataTypeFloat16) {
                    out = make_slot_f16_array(1, out_w, @[@1, @1, @1, @(out_w)], h->owned_pbs, &e);
                } else {
                    out = [[MLMultiArray alloc] initWithShape:@[@1, @1, @1, @(out_w)] dataType:outType error:&e];
                }
            }
            if (!emb || !msk || !typ || !out || (h->graph_kind == kGraphNmtz && !lmsk) || (h->M > 0 && !mmap) || e) {
                fprintf(stderr, "laya_ane_load slot allocation failed: %s\n",
                        e ? e.localizedDescription.UTF8String : "null");
                delete h;
                return nullptr;
            }
            zero_mla_f16(emb, H, S);
            zero_mla_f16(msk, (h->graph_kind == kGraphNmtz || h->mask_on_axis1) ? S : 1, (h->mask_on_axis1 ? 1 : S));
            if (lmsk) zero_mla_f16(lmsk, S, S);
            zero_mla_f16(typ, H, 1);
            if (mmap) zero_mla_f16(mmap, S, h->M);

            NSDictionary* dict = nil;
            if (h->graph_kind == kGraphNmtz) {
                dict = @{
                    @"embeddings": emb,
                    @"full_mask": msk,
                    @"local_mask": lmsk,
                    @"type_vectors": typ,
                    @"marker_map": mmap,
                };
            } else if (h->graph_kind == kGraphMarkerW8) {
                dict = @{
                    @"tok_emb_in": emb,
                    @"ext_mask": msk,
                    @"type_vec": typ,
                    @"marker_map": mmap,
                };
            } else {
                dict = @{@"tok_emb_in": emb, @"ext_mask": msk, @"type_vec": typ};
            }

            MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:&e];
            if (!fp || e) {
                fprintf(stderr, "laya_ane_load feature provider failed: %s\n",
                        e ? e.localizedDescription.UTF8String : "null");
                delete h;
                return nullptr;
            }
            MLPredictionOptions* opt = [MLPredictionOptions new];
            if (outDescs.count == 1) {
                opt.outputBackings = @{h->out_logit_name: out};
            }
            h->in_emb.push_back(emb);
            h->in_mask.push_back(msk);
            h->in_local_mask.push_back(lmsk);
            h->in_type.push_back(typ);
            h->in_marker_map.push_back(mmap);
            h->out_logits.push_back(out);
            h->feats.push_back(fp);
            h->opts.push_back(opt);
        }
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0
        );
        h->queue = dispatch_queue_create("io.github.fast_decision_models.laya_ane", attr);
        id<MLFeatureProvider> warm_out = [model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
        if (!warm_out || err) {
            fprintf(stderr, "laya_ane_load warmup failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "null output");
            delete h;
            return nullptr;
        }
        return static_cast<void*>(h);
    }
}

void* laya_ane_load(
    const char* mlmodelc_path,
    int H,
    int S,
    int pipe_depth,
    int compute_mode,
    const uint16_t* word_fp16,
    int vocab_size,
    const uint16_t* type_fp16
) {
    return laya_ane_load_fn(
        mlmodelc_path,
        "main",
        H,
        S,
        64,
        pipe_depth,
        compute_mode,
        word_fp16,
        vocab_size,
        type_fp16,
        50283
    );
}

int laya_ane_max_markers(void* handle) {
    if (!handle) return 0;
    return static_cast<LayaAneHandle*>(handle)->M;
}

double laya_ane_exec_markers_batch(
    void* handle,
    int N,
    const int32_t* tok_ids_all,
    const int32_t* valid_lens,
    const int32_t* qtypes,
    const int32_t* marker_pos_all,
    const int32_t* marker_counts,
    int marker_stride,
    float* out_marker_logits_f32,
    int use_async_pipe
) {
    if (!handle || N <= 0 || marker_stride <= 0) return 0.0;
    auto* h = static_cast<LayaAneHandle*>(handle);
    {
        std::lock_guard<std::mutex> lk(h->err_mu);
        h->last_error.clear();
    }
    const int S = h->S;
    const int M = h->M;
    double t0 = now_sec();
    std::atomic<bool> had_error{false};

    auto extract_item_markers = [&](int slot, int item_i, id<MLFeatureProvider> pred) {
        MLMultiArray* out_arr = pred ? [pred featureValueForName:h->out_logit_name].multiArrayValue : nil;
        if (!out_arr) out_arr = h->out_logits[static_cast<size_t>(slot)];
        float* dst = out_marker_logits_f32 + static_cast<size_t>(item_i) * static_cast<size_t>(marker_stride);
        const int k_cnt = std::max(0, std::min(marker_stride, marker_counts[item_i]));
        if (M > 0) {
            const int n_read = std::min(M, k_cnt);
            if (n_read > 0) {
                copy_out_to_f32(out_arr, dst, n_read);
            }
        } else {
            std::vector<float> tmp_s(static_cast<size_t>(S), 0.0f);
            copy_out_to_f32(out_arr, tmp_s.data(), S);
            const int32_t* mp = marker_pos_all + static_cast<size_t>(item_i) * static_cast<size_t>(marker_stride);
            for (int j = 0; j < k_cnt; ++j) {
                int p_idx = std::max(0, std::min(S - 1, static_cast<int>(mp[j])));
                dst[j] = tmp_s[static_cast<size_t>(p_idx)];
            }
        }
    };

    @autoreleasepool {
        if (!use_async_pipe || N == 1 || h->pipe_depth == 1) {
            for (int i = 0; i < N; ++i) {
                prep_laya_slot_fp16(
                    h,
                    0,
                    tok_ids_all + static_cast<size_t>(i) * S,
                    valid_lens[i],
                    qtypes[i],
                    marker_pos_all + static_cast<size_t>(i) * static_cast<size_t>(marker_stride),
                    marker_counts[i]
                );
                NSError* err = nil;
                id<MLFeatureProvider> pred = [h->model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
                if (!pred || err) {
                    record_error(h, "laya_ane_exec_markers_batch (sync)", err);
                    had_error.store(true, std::memory_order_release);
                    break;
                }
                extract_item_markers(0, i, pred);
            }
        } else {
            const int P = std::min(h->pipe_depth, N);
            dispatch_group_t grp = dispatch_group_create();
            std::atomic<bool>* had_error_ptr = &had_error;
            for (int p = 0; p < P; ++p) {
                dispatch_group_async(grp, h->queue, ^{
                    @autoreleasepool {
                        for (int i = p; i < N; i += P) {
                            if (had_error_ptr->load(std::memory_order_acquire)) break;
                            prep_laya_slot_fp16(
                                h,
                                p,
                                tok_ids_all + static_cast<size_t>(i) * S,
                                valid_lens[i],
                                qtypes[i],
                                marker_pos_all + static_cast<size_t>(i) * static_cast<size_t>(marker_stride),
                                marker_counts[i]
                            );
                            NSError* err = nil;
                            id<MLFeatureProvider> pred = [h->model predictionFromFeatures:h->feats[p] options:h->opts[p] error:&err];
                            if (!pred || err) {
                                record_error(h, "laya_ane_exec_markers_batch (async)", err);
                                had_error_ptr->store(true, std::memory_order_release);
                                break;
                            }
                            extract_item_markers(p, i, pred);
                        }
                    }
                });
            }
            dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
        }
    }
    if (had_error.load(std::memory_order_acquire)) return -1.0;
    return now_sec() - t0;
}

double laya_ane_exec_batch(
    void* handle,
    int N,
    const int32_t* tok_ids_all,
    const int32_t* valid_lens,
    const int32_t* qtypes,
    float* out_logits_f32,
    int use_async_pipe
) {
    if (!handle || N <= 0) return 0.0;
    auto* h = static_cast<LayaAneHandle*>(handle);
    {
        std::lock_guard<std::mutex> lk(h->err_mu);
        h->last_error.clear();
    }
    const int S = h->S;
    const int out_w = (h->M > 0) ? h->M : S;
    double t0 = now_sec();
    std::atomic<bool> had_error{false};

    @autoreleasepool {
        if (!use_async_pipe || N == 1 || h->pipe_depth == 1) {
            for (int i = 0; i < N; ++i) {
                prep_laya_slot_fp16(
                    h,
                    0,
                    tok_ids_all + static_cast<size_t>(i) * S,
                    valid_lens[i],
                    qtypes[i],
                    nullptr,
                    0
                );
                NSError* err = nil;
                id<MLFeatureProvider> pred = [h->model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
                if (!pred || err) {
                    record_error(h, "laya_ane_exec_batch (sync)", err);
                    had_error.store(true, std::memory_order_release);
                    break;
                }
                MLMultiArray* out_arr = pred ? [pred featureValueForName:h->out_logit_name].multiArrayValue : nil;
                float* dst = out_logits_f32 + static_cast<size_t>(i) * S;
                if (out_w < S) std::memset(dst, 0, static_cast<size_t>(S) * sizeof(float));
                copy_out_to_f32(out_arr ? out_arr : h->out_logits[0], dst, std::min(out_w, S));
            }
        } else {
            const int P = std::min(h->pipe_depth, N);
            dispatch_group_t grp = dispatch_group_create();
            std::atomic<bool>* had_error_ptr = &had_error;
            for (int p = 0; p < P; ++p) {
                dispatch_group_async(grp, h->queue, ^{
                    @autoreleasepool {
                        for (int i = p; i < N; i += P) {
                            if (had_error_ptr->load(std::memory_order_acquire)) break;
                            prep_laya_slot_fp16(
                                h,
                                p,
                                tok_ids_all + static_cast<size_t>(i) * S,
                                valid_lens[i],
                                qtypes[i],
                                nullptr,
                                0
                            );
                            NSError* err = nil;
                            id<MLFeatureProvider> pred = [h->model predictionFromFeatures:h->feats[p] options:h->opts[p] error:&err];
                            if (!pred || err) {
                                record_error(h, "laya_ane_exec_batch (async)", err);
                                had_error_ptr->store(true, std::memory_order_release);
                                break;
                            }
                            MLMultiArray* out_arr = pred ? [pred featureValueForName:h->out_logit_name].multiArrayValue : nil;
                            float* dst = out_logits_f32 + static_cast<size_t>(i) * S;
                            if (out_w < S) std::memset(dst, 0, static_cast<size_t>(S) * sizeof(float));
                            copy_out_to_f32(out_arr ? out_arr : h->out_logits[p], dst, std::min(out_w, S));
                        }
                    }
                });
            }
            dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
        }
    }
    if (had_error.load(std::memory_order_acquire)) return -1.0;
    return now_sec() - t0;
}

const char* laya_ane_last_error(void* handle) {
    if (!handle) return "";
    auto* h = static_cast<LayaAneHandle*>(handle);
    std::lock_guard<std::mutex> lk(h->err_mu);
    return h->last_error.c_str();
}

void laya_ane_free(void* handle) {
    if (!handle) return;
    auto* h = static_cast<LayaAneHandle*>(handle);
    delete h;
}

}  // extern "C"
