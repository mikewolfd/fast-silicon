#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bge_ane_runner.h"
#include "fast_shared_apple.hpp"

using fast_shared::now_sec;

static int g_vocab_size = 0;
static int g_max_pos = 0;
static std::vector<uint16_t> g_wt0;
static std::vector<uint16_t> g_pos;

struct ANEBucket {
    MLModel* model;
    int B;
    int S;
    int n_slots;
    std::vector<MLMultiArray*> in_emb;
    std::vector<MLMultiArray*> in_mask;
    std::vector<MLMultiArray*> out_cls;
    std::vector<MLDictionaryFeatureProvider*> feats;
    std::vector<MLPredictionOptions*> opts;
};

static std::vector<ANEBucket*> g_buckets;

// Fast NEON FP16 word + pos lookup into [seqlen, 384] then 8x8 transpose into [384, S].
// S must be a multiple of 8 and <= 512.
static inline void fill_emb_doc_fp16(
    const uint16_t* __restrict__ w_wt0,
    int vocab_size,
    const uint16_t* __restrict__ w_pos,
    int max_pos,
    const int32_t* __restrict__ tok_ids,
    int seqlen,
    int S,
    uint16_t* __restrict__ dst_emb,  // [384, S]
    uint16_t* __restrict__ dst_mask  // [S]
) {
    alignas(64) uint16_t tmp[512 * 384];
    int sl = std::clamp(seqlen, 0, std::min(S, max_pos));
    for (int s = 0; s < sl; ++s) {
        int tid = tok_ids[s];
        if (static_cast<unsigned>(tid) >= static_cast<unsigned>(vocab_size)) {
            tid = 100;  // [UNK] fallback
        }
        const float16_t* wt = reinterpret_cast<const float16_t*>(w_wt0 + static_cast<size_t>(tid) * 384);
        const float16_t* wp = reinterpret_cast<const float16_t*>(w_pos + static_cast<size_t>(s) * 384);
        float16_t* d = reinterpret_cast<float16_t*>(tmp + static_cast<size_t>(s) * 384);
        for (int c = 0; c < 384; c += 32) {
            float16x8_t a0 = vld1q_f16(wt + c);
            float16x8_t a1 = vld1q_f16(wt + c + 8);
            float16x8_t a2 = vld1q_f16(wt + c + 16);
            float16x8_t a3 = vld1q_f16(wt + c + 24);
            float16x8_t b0 = vld1q_f16(wp + c);
            float16x8_t b1 = vld1q_f16(wp + c + 8);
            float16x8_t b2 = vld1q_f16(wp + c + 16);
            float16x8_t b3 = vld1q_f16(wp + c + 24);
            vst1q_f16(d + c,      vaddq_f16(a0, b0));
            vst1q_f16(d + c + 8,  vaddq_f16(a1, b1));
            vst1q_f16(d + c + 16, vaddq_f16(a2, b2));
            vst1q_f16(d + c + 24, vaddq_f16(a3, b3));
        }
    }
    if (sl < S) {
        memset(tmp + static_cast<size_t>(sl) * 384, 0, static_cast<size_t>(S - sl) * 384 * sizeof(uint16_t));
    }

    // 8x8 NEON uint16 transpose [S, 384] -> [384, S]
    for (int s = 0; s < S; s += 8) {
        for (int c = 0; c < 384; c += 8) {
            const uint16_t* sp = tmp + static_cast<size_t>(s) * 384 + c;
            uint16x8_t r0 = vld1q_u16(sp + 0 * 384);
            uint16x8_t r1 = vld1q_u16(sp + 1 * 384);
            uint16x8_t r2 = vld1q_u16(sp + 2 * 384);
            uint16x8_t r3 = vld1q_u16(sp + 3 * 384);
            uint16x8_t r4 = vld1q_u16(sp + 4 * 384);
            uint16x8_t r5 = vld1q_u16(sp + 5 * 384);
            uint16x8_t r6 = vld1q_u16(sp + 6 * 384);
            uint16x8_t r7 = vld1q_u16(sp + 7 * 384);

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

    const uint16_t mask_zero = 0x0000;
    float16_t m_neg = static_cast<float16_t>(-10000.0f);
    uint16_t mask_pad;
    memcpy(&mask_pad, &m_neg, 2);
    for (int s = 0; s < sl; ++s) dst_mask[s] = mask_zero;
    for (int s = sl; s < S; ++s) dst_mask[s] = mask_pad;
}

static inline void l2_normalize_row(const float* src, float* dst) {
    fast_shared::l2_normalize_f32(src, dst, 384);
}

static inline void write_cls_outputs(
    MLMultiArray* out_arr,
    int d_start,
    int count,
    const int32_t* doc_orig_idx,
    float* out_vecs
) {
    const float* o_f32 = (out_arr.dataType == MLMultiArrayDataTypeFloat32)
        ? static_cast<const float*>(out_arr.dataPointer)
        : nullptr;
    const float16_t* o_f16 = (o_f32 == nullptr)
        ? static_cast<const float16_t*>(out_arr.dataPointer)
        : nullptr;
    for (int j = 0; j < count; ++j) {
        int oi = doc_orig_idx[d_start + j];
        if (o_f32) {
            l2_normalize_row(o_f32 + j * 384, out_vecs + static_cast<size_t>(oi) * 384);
        } else {
            float tmp[384];
            for (int c = 0; c < 384; ++c) tmp[c] = static_cast<float>(o_f16[j * 384 + c]);
            l2_normalize_row(tmp, out_vecs + static_cast<size_t>(oi) * 384);
        }
    }
}

static MLComputeUnits resolve_compute_units(int compute_mode) {
    if (compute_mode == 0) return MLComputeUnitsCPUOnly;
    if (compute_mode == 1) return MLComputeUnitsCPUAndNeuralEngine;
    return MLComputeUnitsAll;
}

extern "C" int bge_ane_init_tables(const uint16_t* wt0_fp16, int vocab_size, const uint16_t* pos_fp16, int max_pos) {
    if (!wt0_fp16 || !pos_fp16 || vocab_size <= 0 || max_pos <= 0) return -1;
    g_vocab_size = vocab_size;
    g_max_pos = max_pos;
    g_wt0.assign(wt0_fp16, wt0_fp16 + static_cast<size_t>(vocab_size) * 384);
    g_pos.assign(pos_fp16, pos_fp16 + static_cast<size_t>(max_pos) * 384);
    return 0;
}

extern "C" void bge_ane_inspect_plan(const char* mlmodelc_path, int compute_mode) {
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = resolve_compute_units(compute_mode);
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block int ane_ops = 0, gpu_ops = 0, cpu_ops = 0;
        [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
            (void)err;
            if (plan && plan.modelStructure.program) {
                auto* fn = plan.modelStructure.program.functions[@"main"];
                for (MLModelStructureProgramOperation* op in fn.block.operations) {
                    MLComputePlanDeviceUsage* u = [plan computeDeviceUsageForMLProgramOperation:op];
                    id pref = u.preferredComputeDevice;
                    if ([pref isKindOfClass:[MLNeuralEngineComputeDevice class]]) ane_ops++;
                    else if ([pref isKindOfClass:[MLGPUComputeDevice class]]) gpu_ops++;
                    else if ([pref isKindOfClass:[MLCPUComputeDevice class]]) cpu_ops++;
                }
            }
            dispatch_semaphore_signal(sem);
        }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        printf("[ANE Plan mode=%d] %s -> ANE=%d GPU=%d CPU=%d\n", compute_mode, mlmodelc_path, ane_ops, gpu_ops, cpu_ops);
    }
}

extern "C" int bge_ane_load_bucket(const char* mlmodelc_path, int B, int S, int n_slots, int compute_mode) {
    @autoreleasepool {
        if (!mlmodelc_path || B <= 0 || S <= 0 || (S % 8) != 0 || S > 512 || n_slots <= 0) {
            return -3;
        }
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = resolve_compute_units(compute_mode);
        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model || err) {
            fprintf(stderr, "Failed to load %s: %s\n", mlmodelc_path, err ? err.localizedDescription.UTF8String : "null");
            return -1;
        }
        ANEBucket* bk = new ANEBucket();
        bk->model = model;
        bk->B = B;
        bk->S = S;
        bk->n_slots = n_slots;
        for (int i = 0; i < n_slots; ++i) {
            NSError* e = nil;
            MLMultiArray* emb = [[MLMultiArray alloc] initWithShape:@[@(B), @384, @1, @(S)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            MLMultiArray* msk = [[MLMultiArray alloc] initWithShape:@[@(B), @1, @1, @(S)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            MLFeatureDescription* outDesc = model.modelDescription.outputDescriptionsByName[@"cls_emb"];
            MLMultiArray* outCls = [[MLMultiArray alloc] initWithShape:@[@(B), @384] dataType:outDesc.multiArrayConstraint.dataType error:&e];
            MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"emb_sum": emb, @"ext_mask": msk} error:&e];
            MLPredictionOptions* opt = [MLPredictionOptions new];
            opt.outputBackings = @{@"cls_emb": outCls};
            bk->in_emb.push_back(emb);
            bk->in_mask.push_back(msk);
            bk->out_cls.push_back(outCls);
            bk->feats.push_back(fp);
            bk->opts.push_back(opt);
        }
        // Zero-initialize warmup inputs
        memset(bk->in_emb[0].dataPointer, 0, static_cast<size_t>(B) * 384 * S * sizeof(uint16_t));
        memset(bk->in_mask[0].dataPointer, 0, static_cast<size_t>(B) * S * sizeof(uint16_t));
        [model predictionFromFeatures:bk->feats[0] options:bk->opts[0] error:&err];
        if (err) {
            fprintf(stderr, "Warmup error %s: %s\n", mlmodelc_path, err.localizedDescription.UTF8String);
            delete bk;
            return -2;
        }
        int bid = static_cast<int>(g_buckets.size());
        g_buckets.push_back(bk);
        return bid;
    }
}

extern "C" double bge_ane_query(int bucket_id, const int32_t* tok_ids, int len, float* out_vec384, int iters) {
    @autoreleasepool {
        if (bucket_id < 0 || bucket_id >= static_cast<int>(g_buckets.size()) || !out_vec384 || iters <= 0) {
            return -1.0;
        }
        if (g_wt0.empty() || g_pos.empty()) return -2.0;
        ANEBucket* bk = g_buckets[bucket_id];
        int S = bk->S;
        int sl = std::clamp(len, 0, S);
        double t0 = now_sec();
        for (int it = 0; it < iters; ++it) {
            uint16_t* emb_ptr = static_cast<uint16_t*>(bk->in_emb[0].dataPointer);
            uint16_t* msk_ptr = static_cast<uint16_t*>(bk->in_mask[0].dataPointer);
            fill_emb_doc_fp16(g_wt0.data(), g_vocab_size, g_pos.data(), g_max_pos, tok_ids, sl, S, emb_ptr, msk_ptr);
            NSError* err = nil;
            [bk->model predictionFromFeatures:bk->feats[0] options:bk->opts[0] error:&err];
            if (bk->out_cls[0].dataType == MLMultiArrayDataTypeFloat32) {
                l2_normalize_row(static_cast<const float*>(bk->out_cls[0].dataPointer), out_vec384);
            } else {
                const float16_t* hp = static_cast<const float16_t*>(bk->out_cls[0].dataPointer);
                float tmp32[384];
                for (int c = 0; c < 384; ++c) tmp32[c] = static_cast<float>(hp[c]);
                l2_normalize_row(tmp32, out_vec384);
            }
        }
        return now_sec() - t0;
    }
}

extern "C" double bge_ane_encode_bucket(
    int bucket_id,
    int n_docs,
    const int32_t* flat_toks,
    const int32_t* offsets,
    const int32_t* doc_orig_idx,
    float* out_vecs,
    int pipe_depth
) {
    @autoreleasepool {
        if (n_docs <= 0) return 0.0;
        if (bucket_id < 0 || bucket_id >= static_cast<int>(g_buckets.size()) || !offsets || !doc_orig_idx || !out_vecs) {
            return -1.0;
        }
        if (g_wt0.empty() || g_pos.empty()) return -2.0;

        ANEBucket* bk = g_buckets[bucket_id];
        int B = bk->B;
        int S = bk->S;
        int n_batches = (n_docs + B - 1) / B;
        int depth = std::min(pipe_depth, bk->n_slots);
        size_t doc_emb_elems = static_cast<size_t>(384) * S;

        double t0 = now_sec();
        auto fill_slot = [&](int slot, int d_start, int d_end) {
            uint16_t* emb_ptr = static_cast<uint16_t*>(bk->in_emb[slot].dataPointer);
            uint16_t* msk_ptr = static_cast<uint16_t*>(bk->in_mask[slot].dataPointer);
            for (int j = 0; j < B; ++j) {
                int d = d_start + j;
                if (d < d_end) {
                    int s0 = offsets[d], s1 = offsets[d + 1];
                    fill_emb_doc_fp16(
                        g_wt0.data(), g_vocab_size, g_pos.data(), g_max_pos,
                        flat_toks + s0, std::min(s1 - s0, S), S,
                        emb_ptr + static_cast<size_t>(j) * doc_emb_elems,
                        msk_ptr + static_cast<size_t>(j) * S
                    );
                } else {
                    memset(emb_ptr + static_cast<size_t>(j) * doc_emb_elems, 0, doc_emb_elems * sizeof(uint16_t));
                    memset(msk_ptr + static_cast<size_t>(j) * S, 0, S * sizeof(uint16_t));
                }
            }
        };

        if (depth <= 1) {
            for (int bi = 0; bi < n_batches; ++bi) {
                int d_start = bi * B;
                int d_end = std::min(n_docs, d_start + B);
                fill_slot(0, d_start, d_end);
                NSError* err = nil;
                [bk->model predictionFromFeatures:bk->feats[0] options:bk->opts[0] error:&err];
                write_cls_outputs(bk->out_cls[0], d_start, d_end - d_start, doc_orig_idx, out_vecs);
            }
        } else {
            std::vector<dispatch_semaphore_t> sems(depth);
            for (int i = 0; i < depth; ++i) sems[i] = dispatch_semaphore_create(1);
            dispatch_group_t grp = dispatch_group_create();

            for (int bi = 0; bi < n_batches; ++bi) {
                int s = bi % depth;
                dispatch_semaphore_wait(sems[s], DISPATCH_TIME_FOREVER);

                int d_start = bi * B;
                int d_end = std::min(n_docs, d_start + B);
                fill_slot(s, d_start, d_end);

                dispatch_group_enter(grp);
                dispatch_semaphore_t sem = sems[s];
                MLMultiArray* out_arr = bk->out_cls[s];
                [bk->model predictionFromFeatures:bk->feats[s]
                                           options:bk->opts[s]
                                 completionHandler:^(id<MLFeatureProvider> out, NSError* err) {
                    (void)out;
                    (void)err;
                    write_cls_outputs(out_arr, d_start, d_end - d_start, doc_orig_idx, out_vecs);
                    dispatch_semaphore_signal(sem);
                    dispatch_group_leave(grp);
                }];
            }
            dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
        }
        return now_sec() - t0;
    }
}

extern "C" void bge_ane_free_all(void) {
    @autoreleasepool {
        for (ANEBucket* bk : g_buckets) delete bk;
        g_buckets.clear();
        g_wt0.clear();
        g_pos.clear();
        g_vocab_size = 0;
        g_max_pos = 0;
    }
}
