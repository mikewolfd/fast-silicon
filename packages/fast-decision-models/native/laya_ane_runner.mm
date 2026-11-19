// Native Objective-C++ CoreML ANE Runner for Laya-Typed-Decisions-421M (`laya_ane_runner.mm`).
// Scalar FP16 LUT gather + NEON 8x8 transpose into IOSurface-backed [1, 1024, 1, S], dynamic qtype [1, 1024, 1, 1],
// MLComputePlan ANE-op verification, and pipe_depth=4 async batch queue with full NSError propagation.

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

struct LayaAneHandle {
    MLModel* model = nil;
    int H = 1024;
    int S = 512;
    int pipe_depth = 4;
    int vocab_size = 0;
    std::vector<uint16_t> word_emb;  // [vocab_size, H] fp16
    std::vector<uint16_t> type_emb;  // [3, H] fp16

    // Preallocated per-slot [S, H] gather scratch buffers (zero per-call heap allocation).
    std::vector<std::vector<uint16_t>> slot_scratch;
    // Owned IOSurface-backed CVPixelBuffers (released cleanly in ~LayaAneHandle).
    std::vector<CVPixelBufferRef> owned_pbs;

    std::vector<MLMultiArray*> in_emb;
    std::vector<MLMultiArray*> in_mask;
    std::vector<MLMultiArray*> in_type;
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

// Allocate a page-aligned IOSurface-backed FP16 MLMultiArray via CVPixelBuffer,
// falling back cleanly to a standard heap MLMultiArray if IOSurface creation is unavailable.
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
    [arr getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
        (void)strides;
        if (mutableBytes && size > 0) {
            std::memset(mutableBytes, 0, static_cast<size_t>(size));
        } else if (mutableBytes && rows > 0 && cols > 0) {
            std::memset(mutableBytes, 0, static_cast<size_t>(rows) * cols * sizeof(uint16_t));
        }
    }];
}

inline void prep_laya_slot_fp16(
    const LayaAneHandle* h,
    uint16_t* scratch,          // preallocated [S, H] fp16 buffer
    const int32_t* tok_ids,     // [S]
    int valid_len,              // [0..S]
    int qtype,                  // 0=choice, 1=score, 2=noul
    MLMultiArray* emb_arr,      // [1, H, 1, S] fp16 (IOSurface or heap)
    MLMultiArray* mask_arr,     // [1, 1, 1, S] fp16 (IOSurface or heap)
    MLMultiArray* type_arr      // [1, H, 1, 1] fp16 (IOSurface or heap)
) {
    const int H = h->H;
    const int S = h->S;

    const float16_t zero16 = static_cast<float16_t>(0.0f);
    const float16_t neg16 = static_cast<float16_t>(-10000.0f);
    uint16_t u_zero = 0, u_neg = 0;
    std::memcpy(&u_zero, &zero16, 2);
    std::memcpy(&u_neg, &neg16, 2);

    const int qt = std::max(0, std::min(2, qtype));
    const uint16_t* src_type = h->type_emb.data() + static_cast<size_t>(qt) * H;
    [type_arr getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
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

    [mask_arr getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
        (void)size;
        uint16_t* dst_mask = static_cast<uint16_t*>(mutableBytes);
        const NSInteger st_s = (strides.count >= 4) ? strides[3].integerValue : 1;
        for (int s = 0; s < S; ++s) {
            if (s >= valid_len) {
                dst_mask[static_cast<size_t>(s) * static_cast<size_t>(st_s)] = u_neg;
                std::memset(scratch + static_cast<size_t>(s) * H, 0, static_cast<size_t>(H) * sizeof(uint16_t));
                continue;
            }
            dst_mask[static_cast<size_t>(s) * static_cast<size_t>(st_s)] = u_zero;
            int tid = tok_ids[s];
            if (tid < 0 || tid >= h->vocab_size) tid = 0;
            std::memcpy(
                scratch + static_cast<size_t>(s) * H,
                h->word_emb.data() + static_cast<size_t>(tid) * H,
                static_cast<size_t>(H) * sizeof(uint16_t)
            );
        }
    }];

    [emb_arr getMutableBytesWithHandler:^(void* mutableBytes, NSInteger size, NSArray<NSNumber*>* strides) {
        (void)size;
        uint16_t* dst_emb = static_cast<uint16_t*>(mutableBytes);
        const size_t row_stride = static_cast<size_t>((strides.count >= 2) ? strides[1].integerValue : S);
        const size_t col_stride = static_cast<size_t>((strides.count >= 4) ? strides[3].integerValue : 1);

        if (col_stride == 1 && (S % 8 == 0) && (H % 8 == 0)) {
            // 8x8 NEON transpose [S, H] -> [H, row_stride]
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

void laya_ane_inspect_plan(const char* mlmodelc_path, int compute_mode, int* out_counts3) {
    @autoreleasepool {
        if (!mlmodelc_path) {
            if (out_counts3) out_counts3[0] = out_counts3[1] = out_counts3[2] = 0;
            return;
        }
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = resolve_units(compute_mode);
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block int ane_ops = 0, gpu_ops = 0, cpu_ops = 0;
        [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
            if (err) {
                fprintf(stderr, "laya_ane_inspect_plan failed (%s): %s\n",
                        mlmodelc_path, err.localizedDescription.UTF8String);
            } else if (plan && plan.modelStructure.program) {
                auto* fn = plan.modelStructure.program.functions[@"main"];
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
    @autoreleasepool {
        if (!mlmodelc_path || !word_fp16 || !type_fp16 || H <= 0 || S <= 0 || vocab_size <= 0) {
            return nullptr;
        }
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = resolve_units(compute_mode);
        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model || err) {
            fprintf(stderr, "laya_ane_load failed (%s): %s\n",
                    mlmodelc_path, err ? err.localizedDescription.UTF8String : "null");
            return nullptr;
        }
        auto* h = new LayaAneHandle();
        h->model = model;
        h->H = H;
        h->S = S;
        h->pipe_depth = std::max(1, pipe_depth);
        h->vocab_size = vocab_size;
        h->word_emb.assign(word_fp16, word_fp16 + static_cast<size_t>(vocab_size) * H);
        h->type_emb.assign(type_fp16, type_fp16 + static_cast<size_t>(3) * H);
        h->slot_scratch.resize(static_cast<size_t>(h->pipe_depth));

        MLFeatureDescription* outDesc = model.modelDescription.outputDescriptionsByName[@"pos_logits"];
        MLMultiArrayDataType outType = outDesc ? outDesc.multiArrayConstraint.dataType : MLMultiArrayDataTypeFloat16;

        for (int i = 0; i < h->pipe_depth; ++i) {
            h->slot_scratch[static_cast<size_t>(i)].assign(static_cast<size_t>(S) * H, 0u);
            NSError* e = nil;
            MLMultiArray* emb = make_slot_f16_array(H, S, @[@1, @(H), @1, @(S)], h->owned_pbs, &e);
            MLMultiArray* msk = !e ? make_slot_f16_array(1, S, @[@1, @1, @1, @(S)], h->owned_pbs, &e) : nil;
            MLMultiArray* typ = !e ? make_slot_f16_array(H, 1, @[@1, @(H), @1, @1], h->owned_pbs, &e) : nil;
            MLMultiArray* out = nil;
            if (!e) {
                if (outType == MLMultiArrayDataTypeFloat16) {
                    out = make_slot_f16_array(1, S, @[@1, @1, @1, @(S)], h->owned_pbs, &e);
                } else {
                    out = [[MLMultiArray alloc] initWithShape:@[@1, @1, @1, @(S)] dataType:outType error:&e];
                }
            }
            if (!emb || !msk || !typ || !out || e) {
                fprintf(stderr, "laya_ane_load slot allocation failed: %s\n",
                        e ? e.localizedDescription.UTF8String : "null");
                delete h;
                return nullptr;
            }
            zero_mla_f16(emb, H, S);
            zero_mla_f16(msk, 1, S);
            zero_mla_f16(typ, H, 1);
            MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"tok_emb_in": emb, @"ext_mask": msk, @"type_vec": typ} error:&e];
            if (!fp || e) {
                fprintf(stderr, "laya_ane_load feature provider failed: %s\n",
                        e ? e.localizedDescription.UTF8String : "null");
                delete h;
                return nullptr;
            }
            MLPredictionOptions* opt = [MLPredictionOptions new];
            opt.outputBackings = @{@"pos_logits": out};
            h->in_emb.push_back(emb);
            h->in_mask.push_back(msk);
            h->in_type.push_back(typ);
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

double laya_ane_exec_batch(
    void* handle,
    int N,
    const int32_t* tok_ids_all,   // [N, S]
    const int32_t* valid_lens,    // [N]
    const int32_t* qtypes,        // [N]
    float* out_logits_f32,        // [N, S]
    int use_async_pipe
) {
    if (!handle || N <= 0) return 0.0;
    auto* h = static_cast<LayaAneHandle*>(handle);
    {
        std::lock_guard<std::mutex> lk(h->err_mu);
        h->last_error.clear();
    }
    const int S = h->S;
    double t0 = now_sec();
    std::atomic<bool> had_error{false};

    @autoreleasepool {
        if (!use_async_pipe || N == 1 || h->pipe_depth == 1) {
            for (int i = 0; i < N; ++i) {
                prep_laya_slot_fp16(
                    h,
                    h->slot_scratch[0].data(),
                    tok_ids_all + static_cast<size_t>(i) * S,
                    valid_lens[i],
                    qtypes[i],
                    h->in_emb[0],
                    h->in_mask[0],
                    h->in_type[0]
                );
                NSError* err = nil;
                id<MLFeatureProvider> pred = [h->model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
                if (!pred || err) {
                    record_error(h, "laya_ane_exec_batch (sync)", err);
                    had_error.store(true, std::memory_order_release);
                    break;
                }
                MLMultiArray* out_arr = pred ? [pred featureValueForName:@"pos_logits"].multiArrayValue : nil;
                copy_out_to_f32(out_arr ? out_arr : h->out_logits[0], out_logits_f32 + static_cast<size_t>(i) * S, S);
            }
        } else {
            const int P = std::min(h->pipe_depth, N);
            dispatch_group_t grp = dispatch_group_create();
            std::atomic<bool>* had_error_ptr = &had_error;
            for (int p = 0; p < P; ++p) {
                dispatch_group_async(grp, h->queue, ^{
                    @autoreleasepool {
                        uint16_t* scratch = h->slot_scratch[static_cast<size_t>(p)].data();
                        for (int i = p; i < N; i += P) {
                            if (had_error_ptr->load(std::memory_order_acquire)) {
                                break;
                            }
                            prep_laya_slot_fp16(
                                h,
                                scratch,
                                tok_ids_all + static_cast<size_t>(i) * S,
                                valid_lens[i],
                                qtypes[i],
                                h->in_emb[p],
                                h->in_mask[p],
                                h->in_type[p]
                            );
                            NSError* err = nil;
                            id<MLFeatureProvider> pred = [h->model predictionFromFeatures:h->feats[p] options:h->opts[p] error:&err];
                            if (!pred || err) {
                                record_error(h, "laya_ane_exec_batch (async)", err);
                                had_error_ptr->store(true, std::memory_order_release);
                                break;
                            }
                            MLMultiArray* out_arr = pred ? [pred featureValueForName:@"pos_logits"].multiArrayValue : nil;
                            copy_out_to_f32(out_arr ? out_arr : h->out_logits[p], out_logits_f32 + static_cast<size_t>(i) * S, S);
                        }
                    }
                });
            }
            dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
        }
    }
    if (had_error.load(std::memory_order_acquire)) {
        return -1.0;
    }
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
