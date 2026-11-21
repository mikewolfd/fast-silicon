#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "bge_ane_runner.h"
#include "fast_shared_apple.hpp"

using fast_shared::now_sec;

static std::shared_mutex g_state_mu;
static int g_vocab_size = 0;
static int g_max_pos = 0;
static std::vector<uint16_t> g_wt0;
static std::vector<uint16_t> g_pos;
static const uint16_t* g_wt0_ptr = nullptr;
static const uint16_t* g_pos_ptr = nullptr;
static constexpr float kAneMaskNeg = -40000.0f;
static std::mutex g_err_mu;
static std::string g_last_err;

namespace {

struct SemaphoreGuard {
    dispatch_semaphore_t sem;
    explicit SemaphoreGuard(dispatch_semaphore_t s) : sem(s) {
        if (sem != nil) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    }
    ~SemaphoreGuard() {
        if (sem != nil) dispatch_semaphore_signal(sem);
    }
    SemaphoreGuard(const SemaphoreGuard&) = delete;
    SemaphoreGuard& operator=(const SemaphoreGuard&) = delete;
};

static NSString* infer_bge_symlink_fn(NSURL* url) {
    if (!url) return nil;
    NSString* leaf = url.lastPathComponent;
    NSURL* resolved = [url URLByResolvingSymlinksInPath];
    if (!resolved || [resolved.lastPathComponent isEqualToString:leaf]) {
        return nil;
    }
    int b_val = 0, s_val = 0;
    if (leaf.UTF8String && std::sscanf(leaf.UTF8String, "bge_b%d_s%d_", &b_val, &s_val) == 2 &&
        b_val > 0 && s_val > 0) {
        return [NSString stringWithFormat:@"bge_b%d_s%d", b_val, s_val];
    }
    return nil;
}

}  // namespace

static void record_ane_error(const char* ctx, NSError* err, const char* fallback_msg = nullptr) {
    std::string msg = ctx ? std::string(ctx) : std::string("bge_ane");
    msg += ": ";
    if (err != nil) {
        const char* desc = err.localizedDescription ? err.localizedDescription.UTF8String : "unknown NSError";
        msg += desc ? desc : "unknown NSError";
    } else if (fallback_msg) {
        msg += fallback_msg;
    } else {
        msg += "CoreML prediction returned nil";
    }
    fprintf(stderr, "[bge_ane_runner] %s\n", msg.c_str());
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_last_err = std::move(msg);
}

static void clear_ane_error() {
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_last_err.clear();
}

static inline uint16_t fp16_bits(float v) {
    float16_t h = static_cast<float16_t>(v);
    uint16_t bits = 0;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Fill a non-constant finite embedding + valid first-position attention mask (0.0h at pos 0,
// -40000.0h elsewhere) so LayerNorm never sees var==0 and softmax never sees an all-masked row.
static void fill_valid_dummy_doc_fp16(int S, uint16_t* __restrict__ dst_emb, uint16_t* __restrict__ dst_mask) {
    const uint16_t pos_half = fp16_bits(0.5f);
    const uint16_t neg_half = fp16_bits(-0.5f);
    for (int c = 0; c < 384; ++c) {
        const uint16_t val = (c & 1) ? neg_half : pos_half;
        uint16_t* row = dst_emb + static_cast<size_t>(c) * S;
        std::fill_n(row, S, val);
    }
    const uint16_t mask_zero = 0x0000;
    const uint16_t mask_pad = fp16_bits(kAneMaskNeg);
    if (S > 0) {
        dst_mask[0] = mask_zero;
        for (int s = 1; s < S; ++s) dst_mask[s] = mask_pad;
    }
}

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
    std::vector<dispatch_semaphore_t> sems;
};

static std::vector<ANEBucket*> g_buckets;

static void drain_and_delete_bucket(ANEBucket* bk) {
    if (!bk) return;
    for (dispatch_semaphore_t sem : bk->sems) {
        if (sem != nil) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    }
    for (dispatch_semaphore_t sem : bk->sems) {
        if (sem != nil) dispatch_semaphore_signal(sem);
    }
    delete bk;
}

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
    const uint16_t mask_pad = fp16_bits(kAneMaskNeg);
    int active = (sl > 0) ? sl : 1;
    for (int s = 0; s < active && s < S; ++s) dst_mask[s] = mask_zero;
    for (int s = active; s < S; ++s) dst_mask[s] = mask_pad;
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

extern "C" const char* bge_ane_last_error(void) {
    std::lock_guard<std::mutex> lk(g_err_mu);
    return g_last_err.c_str();
}

extern "C" int bge_ane_init_tables(const uint16_t* wt0_fp16, int vocab_size, const uint16_t* pos_fp16, int max_pos) {
    if (!wt0_fp16 || !pos_fp16 || vocab_size <= 0 || max_pos <= 0) return -1;
    std::unique_lock<std::shared_mutex> lk(g_state_mu);
    g_vocab_size = vocab_size;
    g_max_pos = max_pos;
    g_wt0.assign(wt0_fp16, wt0_fp16 + static_cast<size_t>(vocab_size) * 384);
    g_pos.assign(pos_fp16, pos_fp16 + static_cast<size_t>(max_pos) * 384);
    g_wt0_ptr = g_wt0.data();
    g_pos_ptr = g_pos.data();
    return 0;
}

extern "C" int bge_ane_init_tables_borrowed(const uint16_t* wt0_fp16, int vocab_size, const uint16_t* pos_fp16, int max_pos) {
    if (!wt0_fp16 || !pos_fp16 || vocab_size <= 0 || max_pos <= 0) return -1;
    std::unique_lock<std::shared_mutex> lk(g_state_mu);
    g_vocab_size = vocab_size;
    g_max_pos = max_pos;
    g_wt0.clear();
    g_wt0.shrink_to_fit();
    g_pos.clear();
    g_pos.shrink_to_fit();
    g_wt0_ptr = wt0_fp16;
    g_pos_ptr = pos_fp16;
    return 0;
}

extern "C" void bge_ane_clear_borrowed_tables(const uint16_t* wt0_fp16) {
    std::unique_lock<std::shared_mutex> lk(g_state_mu);
    if (wt0_fp16 == nullptr || g_wt0_ptr == wt0_fp16) {
        if (g_wt0.empty()) {
            g_wt0_ptr = nullptr;
            g_pos_ptr = nullptr;
            g_vocab_size = 0;
            g_max_pos = 0;
        }
    }
}

extern "C" int bge_ane_get_plan_counts_fn(
    const char* mlmodelc_path,
    const char* function_name,
    int compute_mode,
    int* out_ane_ops,
    int* out_gpu_ops,
    int* out_cpu_ops
) {
    @autoreleasepool {
        if (!mlmodelc_path) {
            record_ane_error("bge_ane_get_plan_counts", nil, "null mlmodelc_path");
            return -1;
        }
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = resolve_compute_units(compute_mode);
        NSString* fn_key = @"main";
        if (function_name && function_name[0] != '\0') {
            fn_key = [NSString stringWithUTF8String:function_name];
            if (std::strcmp(function_name, "main") != 0) {
                if (@available(macOS 15.0, *)) {
                    cfg.functionName = fn_key;
                }
            }
        } else if (NSString* sym_fn = infer_bge_symlink_fn(url)) {
            fn_key = sym_fn;
            if (@available(macOS 15.0, *)) {
                cfg.functionName = sym_fn;
            }
        }
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block int ane_ops = 0, gpu_ops = 0, cpu_ops = 0;
        __block int status = 0;
        [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
            if (err != nil || !plan) {
                record_ane_error("MLComputePlan", err, "plan load returned nil");
                status = -2;
            } else if (plan.modelStructure.program) {
                auto* fn = plan.modelStructure.program.functions[fn_key];
                if (!fn && plan.modelStructure.program.functions.count > 0) {
                    fn = plan.modelStructure.program.functions.allValues.firstObject;
                }
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
        if (out_ane_ops) *out_ane_ops = ane_ops;
        if (out_gpu_ops) *out_gpu_ops = gpu_ops;
        if (out_cpu_ops) *out_cpu_ops = cpu_ops;
        return status;
    }
}

extern "C" int bge_ane_get_plan_counts(
    const char* mlmodelc_path,
    int compute_mode,
    int* out_ane_ops,
    int* out_gpu_ops,
    int* out_cpu_ops
) {
    return bge_ane_get_plan_counts_fn(mlmodelc_path, nullptr, compute_mode, out_ane_ops, out_gpu_ops, out_cpu_ops);
}

extern "C" void bge_ane_inspect_plan(const char* mlmodelc_path, int compute_mode) {
    int ane_ops = 0, gpu_ops = 0, cpu_ops = 0;
    int rc = bge_ane_get_plan_counts(mlmodelc_path, compute_mode, &ane_ops, &gpu_ops, &cpu_ops);
    if (rc == 0) {
        printf("[ANE Plan mode=%d] %s -> ANE=%d GPU=%d CPU=%d\n", compute_mode, mlmodelc_path, ane_ops, gpu_ops, cpu_ops);
    } else {
        fprintf(stderr, "[ANE Plan mode=%d] %s -> FAILED (rc=%d): %s\n", compute_mode, mlmodelc_path, rc, bge_ane_last_error());
    }
}

extern "C" int bge_ane_load_bucket_fn(
    const char* mlmodelc_path,
    const char* function_name,
    int B,
    int S,
    int n_slots,
    int compute_mode
) {
    @autoreleasepool {
        if (!mlmodelc_path || B <= 0 || S <= 0 || (S % 8) != 0 || S > 512 || n_slots <= 0) {
            record_ane_error("bge_ane_load_bucket", nil, "invalid bucket parameters");
            return -3;
        }
        clear_ane_error();
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = resolve_compute_units(compute_mode);
        const bool explicit_fn = (function_name && function_name[0] != '\0');
        if (explicit_fn) {
            if (std::strcmp(function_name, "main") != 0) {
                if (@available(macOS 15.0, *)) {
                    cfg.functionName = [NSString stringWithUTF8String:function_name];
                }
            }
        } else if (infer_bge_symlink_fn(url) != nil) {
            if (@available(macOS 15.0, *)) {
                cfg.functionName = [NSString stringWithFormat:@"bge_b%d_s%d", B, S];
            }
        }
        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!explicit_fn) {
            bool shape_ok = false;
            if (model && !err) {
                MLFeatureDescription* in_desc = model.modelDescription.inputDescriptionsByName[@"emb_sum"];
                NSArray<NSNumber*>* in_shape = in_desc.multiArrayConstraint.shape;
                shape_ok = (in_shape.count == 0) ||
                           (in_shape.count == 4 && in_shape[0].intValue == B && in_shape[3].intValue == S);
            }
            if (!model || err || !shape_ok) {
                if (@available(macOS 15.0, *)) {
                    NSString* inferred_fn = [NSString stringWithFormat:@"bge_b%d_s%d", B, S];
                    if (cfg.functionName == nil || ![cfg.functionName isEqualToString:inferred_fn]) {
                        cfg.functionName = inferred_fn;
                        err = nil;
                        model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
                    } else if (!model || err) {
                        cfg.functionName = nil;
                        err = nil;
                        model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
                    }
                }
            }
        }
        if (!model || err) {
            record_ane_error("bge_ane_load_bucket(load)", err, mlmodelc_path);
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
            if (!emb || e) {
                record_ane_error("bge_ane_load_bucket(alloc emb)", e);
                delete bk;
                return -4;
            }
            MLMultiArray* msk = [[MLMultiArray alloc] initWithShape:@[@(B), @1, @1, @(S)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            if (!msk || e) {
                record_ane_error("bge_ane_load_bucket(alloc mask)", e);
                delete bk;
                return -4;
            }
            MLFeatureDescription* outDesc = model.modelDescription.outputDescriptionsByName[@"cls_emb"];
            MLMultiArray* outCls = [[MLMultiArray alloc] initWithShape:@[@(B), @384] dataType:outDesc.multiArrayConstraint.dataType error:&e];
            if (!outCls || e) {
                record_ane_error("bge_ane_load_bucket(alloc out)", e);
                delete bk;
                return -4;
            }
            MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"emb_sum": emb, @"ext_mask": msk} error:&e];
            if (!fp || e) {
                record_ane_error("bge_ane_load_bucket(feature provider)", e);
                delete bk;
                return -4;
            }
            MLPredictionOptions* opt = [MLPredictionOptions new];
            opt.outputBackings = @{@"cls_emb": outCls};
            bk->in_emb.push_back(emb);
            bk->in_mask.push_back(msk);
            bk->out_cls.push_back(outCls);
            bk->feats.push_back(fp);
            bk->opts.push_back(opt);
            bk->sems.push_back(dispatch_semaphore_create(1));
        }
        // Initialize warmup inputs with valid finite embeddings and a valid attention mask
        // (never all-zero emb/mask which hits LayerNorm variance==0 on FP16 ANE graphs).
        uint16_t* w_emb = static_cast<uint16_t*>(bk->in_emb[0].dataPointer);
        uint16_t* w_msk = static_cast<uint16_t*>(bk->in_mask[0].dataPointer);
        const size_t doc_emb_elems = static_cast<size_t>(384) * S;
        const int32_t kWarmToks[2] = {101, 102};
        {
            std::shared_lock<std::shared_mutex> lk(g_state_mu);
            for (int b = 0; b < B; ++b) {
                if (g_wt0_ptr && g_pos_ptr) {
                    fill_emb_doc_fp16(
                        g_wt0_ptr, g_vocab_size, g_pos_ptr, g_max_pos,
                        kWarmToks, std::min(2, S), S,
                        w_emb + static_cast<size_t>(b) * doc_emb_elems,
                        w_msk + static_cast<size_t>(b) * S
                    );
                } else {
                    fill_valid_dummy_doc_fp16(
                        S,
                        w_emb + static_cast<size_t>(b) * doc_emb_elems,
                        w_msk + static_cast<size_t>(b) * S
                    );
                }
            }
        }
        id<MLFeatureProvider> warm_out = [model predictionFromFeatures:bk->feats[0] options:bk->opts[0] error:&err];
        if (!warm_out || err) {
            record_ane_error("bge_ane_load_bucket(warmup)", err, mlmodelc_path);
            delete bk;
            return -2;
        }
        std::unique_lock<std::shared_mutex> lk(g_state_mu);
        int bid = static_cast<int>(g_buckets.size());
        g_buckets.push_back(bk);
        return bid;
    }
}

extern "C" int bge_ane_load_bucket(const char* mlmodelc_path, int B, int S, int n_slots, int compute_mode) {
    return bge_ane_load_bucket_fn(mlmodelc_path, nullptr, B, S, n_slots, compute_mode);
}

extern "C" double bge_ane_query(int bucket_id, const int32_t* tok_ids, int len, float* out_vec384, int iters) {
    @autoreleasepool {
        if (!out_vec384 || iters <= 0) {
            return -1.0;
        }
        ANEBucket* bk = nullptr;
        {
            std::shared_lock<std::shared_mutex> lk(g_state_mu);
            if (bucket_id < 0 || bucket_id >= static_cast<int>(g_buckets.size())) {
                return -1.0;
            }
            bk = g_buckets[bucket_id];
            if (!bk) return -1.0;
            if (!g_wt0_ptr || !g_pos_ptr) return -2.0;
        }
        if (bk->B != 1) {
            record_ane_error("bge_ane_query", nil, "requires B=1 bucket");
            return -4.0;
        }
        int S = bk->S;
        int sl = std::clamp(len, 0, S);
        double t0 = now_sec();
        for (int it = 0; it < iters; ++it) {
            SemaphoreGuard slot_guard(bk->sems[0]);
            {
                std::shared_lock<std::shared_mutex> lk(g_state_mu);
                if (!g_wt0_ptr || !g_pos_ptr) return -2.0;
                uint16_t* emb_ptr = static_cast<uint16_t*>(bk->in_emb[0].dataPointer);
                uint16_t* msk_ptr = static_cast<uint16_t*>(bk->in_mask[0].dataPointer);
                fill_emb_doc_fp16(g_wt0_ptr, g_vocab_size, g_pos_ptr, g_max_pos, tok_ids, sl, S, emb_ptr, msk_ptr);
            }
            NSError* err = nil;
            id<MLFeatureProvider> res = [bk->model predictionFromFeatures:bk->feats[0] options:bk->opts[0] error:&err];
            if (!res || err) {
                record_ane_error("bge_ane_query", err);
                return -3.0;
            }
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
        if (!offsets || !doc_orig_idx || !out_vecs) {
            return -1.0;
        }
        ANEBucket* bk = nullptr;
        {
            std::shared_lock<std::shared_mutex> lk(g_state_mu);
            if (bucket_id < 0 || bucket_id >= static_cast<int>(g_buckets.size())) {
                return -1.0;
            }
            bk = g_buckets[bucket_id];
            if (!bk) return -1.0;
            if (!g_wt0_ptr || !g_pos_ptr) return -2.0;
        }

        int B = bk->B;
        int S = bk->S;
        int n_batches = (n_docs + B - 1) / B;
        int depth = std::max(1, std::min(pipe_depth, bk->n_slots));
        size_t doc_emb_elems = static_cast<size_t>(384) * S;
        static const int32_t kDummyToks[2] = {101, 102};

        double t0 = now_sec();
        auto fill_slot = [&](int slot, int d_start, int d_end) -> bool {
            std::shared_lock<std::shared_mutex> lk(g_state_mu);
            if (!g_wt0_ptr || !g_pos_ptr) return false;
            uint16_t* emb_ptr = static_cast<uint16_t*>(bk->in_emb[slot].dataPointer);
            uint16_t* msk_ptr = static_cast<uint16_t*>(bk->in_mask[slot].dataPointer);
            const bool has_valid_row0 = (d_start < d_end) && (offsets[d_start + 1] > offsets[d_start]);
            for (int j = 0; j < B; ++j) {
                int d = d_start + j;
                if (d < d_end) {
                    int s0 = offsets[d], s1 = offsets[d + 1];
                    int sl = std::min(s1 - s0, S);
                    if (sl > 0) {
                        fill_emb_doc_fp16(
                            g_wt0_ptr, g_vocab_size, g_pos_ptr, g_max_pos,
                            flat_toks + s0, sl, S,
                            emb_ptr + static_cast<size_t>(j) * doc_emb_elems,
                            msk_ptr + static_cast<size_t>(j) * S
                        );
                    } else {
                        fill_emb_doc_fp16(
                            g_wt0_ptr, g_vocab_size, g_pos_ptr, g_max_pos,
                            kDummyToks, std::min(2, S), S,
                            emb_ptr + static_cast<size_t>(j) * doc_emb_elems,
                            msk_ptr + static_cast<size_t>(j) * S
                        );
                    }
                } else if (has_valid_row0) {
                    memcpy(emb_ptr + static_cast<size_t>(j) * doc_emb_elems, emb_ptr, doc_emb_elems * sizeof(uint16_t));
                    memcpy(msk_ptr + static_cast<size_t>(j) * S, msk_ptr, static_cast<size_t>(S) * sizeof(uint16_t));
                } else {
                    fill_emb_doc_fp16(
                        g_wt0_ptr, g_vocab_size, g_pos_ptr, g_max_pos,
                        kDummyToks, std::min(2, S), S,
                        emb_ptr + static_cast<size_t>(j) * doc_emb_elems,
                        msk_ptr + static_cast<size_t>(j) * S
                    );
                }
            }
            return true;
        };

        if (depth <= 1) {
            for (int bi = 0; bi < n_batches; ++bi) {
                SemaphoreGuard slot_guard(bk->sems[0]);
                int d_start = bi * B;
                int d_end = std::min(n_docs, d_start + B);
                if (!fill_slot(0, d_start, d_end)) return -2.0;
                NSError* err = nil;
                id<MLFeatureProvider> res = [bk->model predictionFromFeatures:bk->feats[0] options:bk->opts[0] error:&err];
                if (!res || err) {
                    record_ane_error("bge_ane_encode_bucket(sync)", err);
                    return -3.0;
                }
                write_cls_outputs(bk->out_cls[0], d_start, d_end - d_start, doc_orig_idx, out_vecs);
            }
        } else {
            dispatch_group_t grp = dispatch_group_create();
            std::atomic<int> err_flag{0};
            std::atomic<int>* err_flag_ptr = &err_flag;

            for (int bi = 0; bi < n_batches; ++bi) {
                if (err_flag.load(std::memory_order_acquire) != 0) break;
                int s = bi % depth;
                dispatch_semaphore_t sem = bk->sems[s];
                dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
                if (err_flag.load(std::memory_order_acquire) != 0) {
                    dispatch_semaphore_signal(sem);
                    break;
                }

                int d_start = bi * B;
                int d_end = std::min(n_docs, d_start + B);
                if (!fill_slot(s, d_start, d_end)) {
                    err_flag.store(2, std::memory_order_release);
                    dispatch_semaphore_signal(sem);
                    break;
                }

                dispatch_group_enter(grp);
                MLMultiArray* out_arr = bk->out_cls[s];
                [bk->model predictionFromFeatures:bk->feats[s]
                                           options:bk->opts[s]
                                 completionHandler:^(id<MLFeatureProvider> out, NSError* err) {
                    if (!out || err != nil) {
                        int expected = 0;
                        if (err_flag_ptr->compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                            record_ane_error("bge_ane_encode_bucket(async)", err);
                        }
                    } else if (err_flag_ptr->load(std::memory_order_acquire) == 0) {
                        write_cls_outputs(out_arr, d_start, d_end - d_start, doc_orig_idx, out_vecs);
                    }
                    dispatch_semaphore_signal(sem);
                    dispatch_group_leave(grp);
                }];
            }
            dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
            int ef = err_flag.load(std::memory_order_acquire);
            if (ef == 2) return -2.0;
            if (ef != 0) return -3.0;
        }
        return now_sec() - t0;
    }
}

extern "C" void bge_ane_free_bucket(int bucket_id) {
    @autoreleasepool {
        ANEBucket* bk = nullptr;
        {
            std::unique_lock<std::shared_mutex> lk(g_state_mu);
            if (bucket_id >= 0 && bucket_id < static_cast<int>(g_buckets.size())) {
                bk = g_buckets[bucket_id];
                g_buckets[bucket_id] = nullptr;
            }
        }
        drain_and_delete_bucket(bk);
    }
}

extern "C" void bge_ane_free_all(void) {
    @autoreleasepool {
        std::vector<ANEBucket*> to_free;
        {
            std::unique_lock<std::shared_mutex> lk(g_state_mu);
            to_free.swap(g_buckets);
            g_wt0.clear();
            g_pos.clear();
            g_wt0_ptr = nullptr;
            g_pos_ptr = nullptr;
            g_vocab_size = 0;
            g_max_pos = 0;
        }
        for (ANEBucket* bk : to_free) {
            drain_and_delete_bucket(bk);
        }
    }
}
