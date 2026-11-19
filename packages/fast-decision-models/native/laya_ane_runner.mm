// Native Objective-C++ CoreML ANE Runner for Laya-Typed-Decisions-421M (`laya_ane_runner.mm`).
// NEON FP16 token gather + 8x8 transpose into [1, 1024, 1, S], dynamic qtype [1, 1024, 1, 1],
// MLComputePlan ANE-op verification, and pipe_depth=4 async batch queue.

#include "laya_ane_runner.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

    std::vector<MLMultiArray*> in_emb;
    std::vector<MLMultiArray*> in_mask;
    std::vector<MLMultiArray*> in_type;
    std::vector<MLMultiArray*> out_logits;
    std::vector<MLDictionaryFeatureProvider*> feats;
    std::vector<MLPredictionOptions*> opts;
    dispatch_queue_t queue = nil;
};

inline void prep_laya_slot_fp16(
    const LayaAneHandle* h,
    const int32_t* tok_ids,  // [S]
    int valid_len,           // [0..S]
    int qtype,               // 0=choice, 1=score, 2=noul
    uint16_t* dst_emb,       // [H, S]
    uint16_t* dst_mask,      // [S]
    uint16_t* dst_type       // [H]
) {
    const int H = h->H;
    const int S = h->S;
    std::vector<uint16_t> tmp(static_cast<size_t>(S) * H);

    const float16_t zero16 = static_cast<float16_t>(0.0f);
    const float16_t neg16 = static_cast<float16_t>(-10000.0f);
    uint16_t u_zero, u_neg;
    memcpy(&u_zero, &zero16, 2);
    memcpy(&u_neg, &neg16, 2);

    int qt = std::max(0, std::min(2, qtype));
    memcpy(dst_type, h->type_emb.data() + static_cast<size_t>(qt) * H, static_cast<size_t>(H) * sizeof(uint16_t));

    for (int s = 0; s < S; ++s) {
        if (s >= valid_len) {
            dst_mask[s] = u_neg;
            memset(tmp.data() + static_cast<size_t>(s) * H, 0, static_cast<size_t>(H) * sizeof(uint16_t));
            continue;
        }
        dst_mask[s] = u_zero;
        int tid = tok_ids[s];
        if (tid < 0 || tid >= h->vocab_size) tid = 0;
        memcpy(
            tmp.data() + static_cast<size_t>(s) * H,
            h->word_emb.data() + static_cast<size_t>(tid) * H,
            static_cast<size_t>(H) * sizeof(uint16_t)
        );
    }

    // 8x8 NEON transpose [S, H] -> [H, S]
    for (int s = 0; s < S; s += 8) {
        for (int c = 0; c < H; c += 8) {
            const uint16_t* sp = tmp.data() + static_cast<size_t>(s) * H + c;
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

            uint16_t* dp = dst_emb + static_cast<size_t>(c) * S + s;
            vst1q_u16(dp + 0 * S, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4))));
            vst1q_u16(dp + 4 * S, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4))));
            vst1q_u16(dp + 1 * S, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5))));
            vst1q_u16(dp + 5 * S, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5))));
            vst1q_u16(dp + 2 * S, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6))));
            vst1q_u16(dp + 6 * S, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6))));
            vst1q_u16(dp + 3 * S, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7))));
            vst1q_u16(dp + 7 * S, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7))));
        }
    }
}

inline void copy_out_to_f32(MLMultiArray* arr, float* dst, int count) {
    if (arr.dataType == MLMultiArrayDataTypeFloat32) {
        memcpy(dst, arr.dataPointer, static_cast<size_t>(count) * sizeof(float));
    } else {
        const float16_t* p16 = static_cast<const float16_t*>(arr.dataPointer);
        for (int i = 0; i < count; ++i) dst[i] = static_cast<float>(p16[i]);
    }
}

MLComputeUnits resolve_units(int mode) {
    if (mode == 0) return MLComputeUnitsCPUOnly;
    if (mode == 1) return MLComputeUnitsCPUAndNeuralEngine;
    return MLComputeUnitsAll;
}

}  // namespace

extern "C" {

void laya_ane_inspect_plan(const char* mlmodelc_path, int compute_mode, int* out_counts3) {
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = resolve_units(compute_mode);
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block int ane_ops = 0, gpu_ops = 0, cpu_ops = 0;
        [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
            (void)err;
            if (plan && plan.modelStructure.program) {
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

        MLFeatureDescription* outDesc = model.modelDescription.outputDescriptionsByName[@"pos_logits"];
        MLMultiArrayDataType outType = outDesc ? outDesc.multiArrayConstraint.dataType : MLMultiArrayDataTypeFloat16;

        for (int i = 0; i < h->pipe_depth; ++i) {
            NSError* e = nil;
            MLMultiArray* emb = [[MLMultiArray alloc] initWithShape:@[@1, @(H), @1, @(S)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            MLMultiArray* msk = [[MLMultiArray alloc] initWithShape:@[@1, @1, @1, @(S)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            MLMultiArray* typ = [[MLMultiArray alloc] initWithShape:@[@1, @(H), @1, @1] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            MLMultiArray* out = [[MLMultiArray alloc] initWithShape:@[@1, @1, @1, @(S)] dataType:outType error:&e];
            if (!emb || !msk || !typ || !out || e) {
                delete h;
                return nullptr;
            }
            memset(emb.dataPointer, 0, static_cast<size_t>(H) * S * 2);
            memset(msk.dataPointer, 0, static_cast<size_t>(S) * 2);
            memset(typ.dataPointer, 0, static_cast<size_t>(H) * 2);
            MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"tok_emb_in": emb, @"ext_mask": msk, @"type_vec": typ} error:&e];
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
        [model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
        if (err) {
            fprintf(stderr, "laya_ane_load warmup failed: %s\n", err.localizedDescription.UTF8String);
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
    const int S = h->S;
    double t0 = now_sec();

    @autoreleasepool {
        if (!use_async_pipe || N == 1 || h->pipe_depth == 1) {
            for (int i = 0; i < N; ++i) {
                prep_laya_slot_fp16(
                    h,
                    tok_ids_all + static_cast<size_t>(i) * S,
                    valid_lens[i],
                    qtypes[i],
                    static_cast<uint16_t*>(h->in_emb[0].dataPointer),
                    static_cast<uint16_t*>(h->in_mask[0].dataPointer),
                    static_cast<uint16_t*>(h->in_type[0].dataPointer)
                );
                NSError* err = nil;
                [h->model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
                copy_out_to_f32(h->out_logits[0], out_logits_f32 + static_cast<size_t>(i) * S, S);
            }
        } else {
            const int P = std::min(h->pipe_depth, N);
            dispatch_group_t grp = dispatch_group_create();
            for (int p = 0; p < P; ++p) {
                dispatch_group_async(grp, h->queue, ^{
                    @autoreleasepool {
                        for (int i = p; i < N; i += P) {
                            prep_laya_slot_fp16(
                                h,
                                tok_ids_all + static_cast<size_t>(i) * S,
                                valid_lens[i],
                                qtypes[i],
                                static_cast<uint16_t*>(h->in_emb[p].dataPointer),
                                static_cast<uint16_t*>(h->in_mask[p].dataPointer),
                                static_cast<uint16_t*>(h->in_type[p].dataPointer)
                            );
                            NSError* err = nil;
                            [h->model predictionFromFeatures:h->feats[p] options:h->opts[p] error:&err];
                            copy_out_to_f32(h->out_logits[p], out_logits_f32 + static_cast<size_t>(i) * S, S);
                        }
                    }
                });
            }
            dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
        }
    }
    return now_sec() - t0;
}

void laya_ane_free(void* handle) {
    if (!handle) return;
    auto* h = static_cast<LayaAneHandle*>(handle);
    delete h;
}

}  // extern "C"
