#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "emb2_coreml_runner.h"
#include "emb2_neon_transpose.hpp"
#include "fast_shared_apple.hpp"

namespace {

static constexpr int kMaxCpuSmeSlots = 2;

static std::mutex g_err_mu;
static std::string g_last_err;

static void record_emb2_error(const char* ctx, NSError* err, const char* fallback_msg = nullptr) {
    std::string msg = ctx ? std::string(ctx) : std::string("emb2_coreml");
    msg += ": ";
    if (err != nil) {
        const char* desc = err.localizedDescription ? err.localizedDescription.UTF8String : "unknown NSError";
        msg += desc ? desc : "unknown NSError";
    } else if (fallback_msg) {
        msg += fallback_msg;
    } else {
        msg += "CoreML prediction returned nil";
    }
    fprintf(stderr, "[emb2_coreml_runner] %s\n", msg.c_str());
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_last_err = std::move(msg);
}

static void clear_emb2_error() {
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_last_err.clear();
}

static NSString* infer_emb2_symlink_fn(NSURL* url) {
    if (!url) return nil;
    NSString* leaf = url.lastPathComponent;
    NSURL* resolved = [url URLByResolvingSymlinksInPath];
    if (!resolved || [resolved.lastPathComponent isEqualToString:leaf]) {
        return nil;
    }
    int s_val = 0;
    if (leaf.UTF8String && std::sscanf(leaf.UTF8String, "emb2_pk_s%d_", &s_val) == 1 && s_val > 0) {
        return [NSString stringWithFormat:@"emb2_pk_s%d", s_val];
    }
    return nil;
}

struct Emb2PkHandle {
    MLModel* model = nil;
    int S = 0;
    int slots = 1;
    int compute_mode = 0;
    bool output_is_nchw = false;
    std::vector<MLMultiArray*> ah, ab, acs, ass, acf, asf, ao;
    std::vector<MLDictionaryFeatureProvider*> feats;
    std::vector<MLPredictionOptions*> opts;
    std::vector<dispatch_semaphore_t> sems;
};

static inline void copy_out_buffer(
    uint16_t* dst,
    const void* src,
    int S,
    size_t o_bytes,
    bool output_is_nchw
) {
    if (output_is_nchw) {
        emb2::transpose_512xS_to_Sx512_u16(static_cast<const uint16_t*>(src), dst, S);
    } else {
        memcpy(dst, src, o_bytes);
    }
}

}  // namespace

extern "C" {

const char* emb2_last_error(void) {
    std::lock_guard<std::mutex> lk(g_err_mu);
    return g_last_err.c_str();
}

int emb2_get_plan_counts_fn(
    const char* mlmodelc_path,
    const char* function_name,
    int compute_mode,
    int* out_ane_ops,
    int* out_gpu_ops,
    int* out_cpu_ops
) {
    @autoreleasepool {
        if (!mlmodelc_path) {
            record_emb2_error("emb2_get_plan_counts", nil, "null mlmodelc_path");
            return -1;
        }
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = fast_shared::resolve_compute_units(static_cast<fast_shared::ComputeMode>(compute_mode));
        NSString* fn_key = @"main";
        if (function_name && function_name[0] != '\0') {
            fn_key = [NSString stringWithUTF8String:function_name];
            if (std::strcmp(function_name, "main") != 0) {
                if (@available(macOS 15.0, *)) {
                    cfg.functionName = fn_key;
                }
            }
        } else if (NSString* sym_fn = infer_emb2_symlink_fn(url)) {
            fn_key = sym_fn;
            if (@available(macOS 15.0, *)) {
                cfg.functionName = sym_fn;
            }
        }

        __block int ane_ops = 0;
        __block int gpu_ops = 0;
        __block int cpu_ops = 0;
        __block int status = 0;
        dispatch_semaphore_t psem = dispatch_semaphore_create(0);
        [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
            if (err != nil || !plan) {
                record_emb2_error("MLComputePlan", err, "plan load returned nil");
                status = -2;
            } else if (plan.modelStructure.program) {
                MLModelStructureProgramFunction* fn = plan.modelStructure.program.functions[fn_key];
                if (!fn && plan.modelStructure.program.functions.count > 0) {
                    fn = plan.modelStructure.program.functions.allValues.firstObject;
                }
                for (MLModelStructureProgramOperation* op in fn.block.operations) {
                    id pref = [plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
                    if ([pref isKindOfClass:[MLNeuralEngineComputeDevice class]]) ane_ops++;
                    else if ([pref isKindOfClass:[MLGPUComputeDevice class]]) gpu_ops++;
                    else if ([pref isKindOfClass:[MLCPUComputeDevice class]]) cpu_ops++;
                }
            }
            dispatch_semaphore_signal(psem);
        }];
        dispatch_semaphore_wait(psem, DISPATCH_TIME_FOREVER);
        if (out_ane_ops) *out_ane_ops = ane_ops;
        if (out_gpu_ops) *out_gpu_ops = gpu_ops;
        if (out_cpu_ops) *out_cpu_ops = cpu_ops;
        return status;
    }
}

int emb2_get_plan_counts(
    const char* mlmodelc_path,
    int compute_mode,
    int* out_ane_ops,
    int* out_gpu_ops,
    int* out_cpu_ops
) {
    return emb2_get_plan_counts_fn(mlmodelc_path, nullptr, compute_mode, out_ane_ops, out_gpu_ops, out_cpu_ops);
}

void emb2_inspect_plan(const char* mlmodelc_path, int compute_mode) {
    int ane_ops = 0, gpu_ops = 0, cpu_ops = 0;
    int rc = emb2_get_plan_counts(mlmodelc_path, compute_mode, &ane_ops, &gpu_ops, &cpu_ops);
    if (rc == 0) {
        printf("[MLComputePlan] %s (mode=%d) -> ANE=%d GPU=%d CPU=%d\n",
               mlmodelc_path, compute_mode, ane_ops, gpu_ops, cpu_ops);
    } else {
        fprintf(stderr, "[MLComputePlan] %s (mode=%d) -> FAILED (rc=%d): %s\n",
                mlmodelc_path, compute_mode, rc, emb2_last_error());
    }
}

void* emb2_pk_load_units_fn(
    const char* mlmodelc_path,
    const char* function_name,
    int S,
    int pipe_depth,
    int compute_mode
) {
    @autoreleasepool {
        if (!mlmodelc_path || S <= 0) {
            record_emb2_error("emb2_pk_load_units", nil, "invalid path or S");
            return nullptr;
        }
        clear_emb2_error();
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = fast_shared::resolve_compute_units(static_cast<fast_shared::ComputeMode>(compute_mode));
        const bool explicit_fn = (function_name && function_name[0] != '\0');
        if (explicit_fn) {
            if (std::strcmp(function_name, "main") != 0) {
                if (@available(macOS 15.0, *)) {
                    cfg.functionName = [NSString stringWithUTF8String:function_name];
                }
            }
        } else if (infer_emb2_symlink_fn(url) != nil) {
            if (@available(macOS 15.0, *)) {
                cfg.functionName = [NSString stringWithFormat:@"emb2_pk_s%d", S];
            }
        }

        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!explicit_fn) {
            bool shape_ok = false;
            if (model && !err) {
                MLFeatureDescription* in_desc = model.modelDescription.inputDescriptionsByName[@"h_in"];
                NSArray<NSNumber*>* in_shape = in_desc.multiArrayConstraint.shape;
                shape_ok = (in_shape.count == 0) || (in_shape.lastObject.intValue == S);
            }
            if (!model || err || !shape_ok) {
                if (@available(macOS 15.0, *)) {
                    NSString* inferred_fn = [NSString stringWithFormat:@"emb2_pk_s%d", S];
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
            record_emb2_error("emb2_pk_load_units(load)", err, mlmodelc_path);
            return nullptr;
        }

        bool output_is_nchw = false;
        MLFeatureDescription* out_desc = model.modelDescription.outputDescriptionsByName[@"out_h"];
        NSArray<NSNumber*>* out_shape = out_desc.multiArrayConstraint.shape;
        if (out_shape.count == 4) {
            if (out_shape[0].intValue != 1 || out_shape[1].intValue != 512 ||
                out_shape[2].intValue != 1 || out_shape[3].intValue != S) {
                record_emb2_error("emb2_pk_load_units(shape)", nil, "unexpected 4D out_h shape");
                return nullptr;
            }
            output_is_nchw = true;
        } else if (out_shape.count == 2) {
            if (out_shape[0].intValue != S || out_shape[1].intValue != 512) {
                record_emb2_error("emb2_pk_load_units(shape)", nil, "unexpected 2D out_h shape");
                return nullptr;
            }
        } else if (out_shape.count != 0) {
            record_emb2_error("emb2_pk_load_units(shape)", nil, "unsupported out_h rank");
            return nullptr;
        }

        auto* h = new Emb2PkHandle();
        h->model = model;
        h->S = S;
        h->compute_mode = compute_mode;
        h->output_is_nchw = output_is_nchw;
        int eff_slots = std::max(1, pipe_depth);
        // Cap CPU-SME concurrent slots to <= 2 (2 physical shared SME2 units on M4/M5).
        if (compute_mode == static_cast<int>(fast_shared::ComputeMode::kCpuOnly)) {
            eff_slots = std::min(eff_slots, kMaxCpuSmeSlots);
        }
        h->slots = eff_slots;
        h->ah.resize(h->slots);
        h->ab.resize(h->slots);
        h->acs.resize(h->slots);
        h->ass.resize(h->slots);
        h->acf.resize(h->slots);
        h->asf.resize(h->slots);
        h->ao.resize(h->slots);
        h->feats.resize(h->slots);
        h->opts.resize(h->slots);
        h->sems.resize(h->slots);

        for (int i = 0; i < h->slots; ++i) {
            NSError* e = nil;
            h->ah[i]  = [[MLMultiArray alloc] initWithShape:@[@1, @512, @1, @(S)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            h->ab[i]  = [[MLMultiArray alloc] initWithShape:@[@1, @1, @(S), @(S)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            h->acs[i] = [[MLMultiArray alloc] initWithShape:@[@1, @1, @(S), @128] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            h->ass[i] = [[MLMultiArray alloc] initWithShape:@[@1, @1, @(S), @128] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            h->acf[i] = [[MLMultiArray alloc] initWithShape:@[@1, @1, @(S), @256] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            h->asf[i] = [[MLMultiArray alloc] initWithShape:@[@1, @1, @(S), @256] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            NSArray<NSNumber*>* ao_shape = h->output_is_nchw ? @[@1, @512, @1, @(S)] : @[@(S), @512];
            h->ao[i]  = [[MLMultiArray alloc] initWithShape:ao_shape dataType:MLMultiArrayDataTypeFloat16 error:&e];
            if (!h->ah[i] || !h->ab[i] || !h->acs[i] || !h->ass[i] || !h->acf[i] || !h->asf[i] || !h->ao[i] || e) {
                record_emb2_error("emb2_pk_load_units(alloc)", e, mlmodelc_path);
                delete h;
                return nullptr;
            }
            NSArray<NSNumber*>* ao_strides = h->ao[i].strides;
            const bool ao_contig =
                (h->ao[i].count == static_cast<NSInteger>(512) * S) &&
                (ao_strides.lastObject.integerValue == 1) &&
                (h->output_is_nchw
                     ? (ao_strides.count == 4 && ao_strides[1].integerValue == S)
                     : (ao_strides.count == 2 && ao_strides[0].integerValue == 512));
            if (!ao_contig) {
                record_emb2_error("emb2_pk_load_units(strides)", nil, mlmodelc_path);
                delete h;
                return nullptr;
            }
            memset(h->ah[i].dataPointer,  0, static_cast<size_t>(512) * S * 2);
            memset(h->ab[i].dataPointer,  0, static_cast<size_t>(S) * S * 2);
            memset(h->acs[i].dataPointer, 0, static_cast<size_t>(S) * 128 * 2);
            memset(h->ass[i].dataPointer, 0, static_cast<size_t>(S) * 128 * 2);
            memset(h->acf[i].dataPointer, 0, static_cast<size_t>(S) * 256 * 2);
            memset(h->asf[i].dataPointer, 0, static_cast<size_t>(S) * 256 * 2);

            h->feats[i] = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"h_in": h->ah[i],
                @"attn_bias": h->ab[i],
                @"cos_swa": h->acs[i],
                @"sin_swa": h->ass[i],
                @"cos_full": h->acf[i],
                @"sin_full": h->asf[i],
            } error:&e];
            if (!h->feats[i] || e) {
                record_emb2_error("emb2_pk_load_units(feature provider)", e, mlmodelc_path);
                delete h;
                return nullptr;
            }
            h->opts[i] = [MLPredictionOptions new];
            h->opts[i].outputBackings = @{@"out_h": h->ao[i]};
            h->sems[i] = dispatch_semaphore_create(1);
        }

        // Single warmup prediction compiles ANE/CPU plan eagerly at load time.
        id<MLFeatureProvider> warm_res = [model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
        if (!warm_res || err) {
            record_emb2_error("emb2_pk_load_units(warmup)", err, mlmodelc_path);
            delete h;
            return nullptr;
        }
        return static_cast<void*>(h);
    }
}

void* emb2_pk_load_units(const char* mlmodelc_path, int S, int pipe_depth, int compute_mode) {
    return emb2_pk_load_units_fn(mlmodelc_path, nullptr, S, pipe_depth, compute_mode);
}

struct Emb2Inflight {
    dispatch_group_t grp;
    double t0;
    std::atomic<int> err_flag{0};
};

void* emb2_pk_submit_queue(
    void* handle,
    int N,
    const uint16_t* h_all,
    const uint16_t* b_all,
    const uint16_t* cs_all,
    const uint16_t* ss_all,
    const uint16_t* cf_all,
    const uint16_t* sf_all,
    uint16_t* out_all
) {
    auto* h = static_cast<Emb2PkHandle*>(handle);
    if (!h || N <= 0) return nullptr;
    auto* fl = new Emb2Inflight{dispatch_group_create(), fast_shared::now_sec()};
    dispatch_group_t grp = fl->grp;
    // Use USER_INITIATED QoS for CPU-SME bundles to avoid preempting interactive/GPU threads;
    // keep USER_INITIATED for ANE dispatch queues as well (ANE hardware scheduling is independent).
    const dispatch_qos_class_t qos = QOS_CLASS_USER_INITIATED;
    dispatch_group_async(grp, dispatch_get_global_queue(qos, 0), ^{
        @autoreleasepool {
            const int S = h->S;
            const int slots = h->slots;
            const bool out_nchw = h->output_is_nchw;
            const size_t h_bytes  = static_cast<size_t>(512) * S * 2;
            const size_t b_bytes  = static_cast<size_t>(S) * S * 2;
            const size_t cs_bytes = static_cast<size_t>(S) * 128 * 2;
            const size_t cf_bytes = static_cast<size_t>(S) * 256 * 2;
            const size_t o_bytes  = static_cast<size_t>(S) * 512 * 2;
            if (slots == 1) {
                for (int idx = 0; idx < N; ++idx) {
                    memcpy(h->ah[0].dataPointer,  h_all  + static_cast<size_t>(idx) * 512 * S, h_bytes);
                    memcpy(h->ab[0].dataPointer,  b_all  + static_cast<size_t>(idx) * S * S,   b_bytes);
                    memcpy(h->acs[0].dataPointer, cs_all + static_cast<size_t>(idx) * S * 128, cs_bytes);
                    memcpy(h->ass[0].dataPointer, ss_all + static_cast<size_t>(idx) * S * 128, cs_bytes);
                    memcpy(h->acf[0].dataPointer, cf_all + static_cast<size_t>(idx) * S * 256, cf_bytes);
                    memcpy(h->asf[0].dataPointer, sf_all + static_cast<size_t>(idx) * S * 256, cf_bytes);
                    NSError* err = nil;
                    id<MLFeatureProvider> res = [h->model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
                    if (!res || err != nil) {
                        int expected = 0;
                        if (fl->err_flag.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                            record_emb2_error("emb2_pk_submit_queue(sync)", err);
                        }
                        break;
                    }
                    if (out_all) {
                        copy_out_buffer(
                            out_all + static_cast<size_t>(idx) * S * 512,
                            h->ao[0].dataPointer,
                            S,
                            o_bytes,
                            out_nchw
                        );
                    }
                }
            } else {
                for (int idx = 0; idx < N; ++idx) {
                    if (fl->err_flag.load(std::memory_order_acquire) != 0) break;
                    const int s = idx % slots;
                    dispatch_semaphore_wait(h->sems[s], DISPATCH_TIME_FOREVER);
                    if (fl->err_flag.load(std::memory_order_acquire) != 0) {
                        dispatch_semaphore_signal(h->sems[s]);
                        break;
                    }
                    memcpy(h->ah[s].dataPointer,  h_all  + static_cast<size_t>(idx) * 512 * S, h_bytes);
                    memcpy(h->ab[s].dataPointer,  b_all  + static_cast<size_t>(idx) * S * S,   b_bytes);
                    memcpy(h->acs[s].dataPointer, cs_all + static_cast<size_t>(idx) * S * 128, cs_bytes);
                    memcpy(h->ass[s].dataPointer, ss_all + static_cast<size_t>(idx) * S * 128, cs_bytes);
                    memcpy(h->acf[s].dataPointer, cf_all + static_cast<size_t>(idx) * S * 256, cf_bytes);
                    memcpy(h->asf[s].dataPointer, sf_all + static_cast<size_t>(idx) * S * 256, cf_bytes);
                    dispatch_group_enter(grp);
                    dispatch_semaphore_t sem = h->sems[s];
                    uint16_t* dst = out_all ? (out_all + static_cast<size_t>(idx) * S * 512) : nullptr;
                    void* src = h->ao[s].dataPointer;
                    [h->model predictionFromFeatures:h->feats[s] options:h->opts[s]
                                    completionHandler:^(id<MLFeatureProvider> out, NSError* e) {
                        if (!out || e != nil) {
                            int expected = 0;
                            if (fl->err_flag.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                                record_emb2_error("emb2_pk_submit_queue(async)", e);
                            }
                        } else if (fl->err_flag.load(std::memory_order_acquire) == 0 && dst) {
                            copy_out_buffer(dst, src, S, o_bytes, out_nchw);
                        }
                        dispatch_semaphore_signal(sem);
                        dispatch_group_leave(grp);
                    }];
                }
            }
        }
    });
    return static_cast<void*>(fl);
}

double emb2_pk_wait_queue(void* flight) {
    if (!flight) return -1.0;
    auto* fl = static_cast<Emb2Inflight*>(flight);
    dispatch_group_wait(fl->grp, DISPATCH_TIME_FOREVER);
    const bool failed = (fl->err_flag.load(std::memory_order_acquire) != 0);
    const double ms = (fast_shared::now_sec() - fl->t0) * 1e3;
    delete fl;
    return failed ? -1.0 : ms;
}

double emb2_pk_exec_queue(
    void* handle,
    int N,
    const uint16_t* h_all,
    const uint16_t* b_all,
    const uint16_t* cs_all,
    const uint16_t* ss_all,
    const uint16_t* cf_all,
    const uint16_t* sf_all,
    uint16_t* out_all
) {
    if (N == 0 && handle) return 0.0;
    void* fl = emb2_pk_submit_queue(handle, N, h_all, b_all, cs_all, ss_all, cf_all, sf_all, out_all);
    if (!fl) return -1.0;
    return emb2_pk_wait_queue(fl);
}

void emb2_pk_free(void* handle) {
    if (handle) delete static_cast<Emb2PkHandle*>(handle);
}

void emb2_transpose_512xS_to_Sx512(int S, const uint16_t* src, uint16_t* dst) {
    emb2::transpose_512xS_to_Sx512_u16(src, dst, S);
}

int emb2_pk_output_is_nchw(void* handle) {
    if (!handle) return -1;
    return static_cast<const Emb2PkHandle*>(handle)->output_is_nchw ? 1 : 0;
}

struct Emb2OiHandle {
    MLModel* model = nil;
    int S = 0;
    int slots = 1;
    MLMultiArrayDataType mask_dtype = MLMultiArrayDataTypeFloat16;
    MLMultiArrayDataType out_dtype = MLMultiArrayDataTypeFloat16;
    std::vector<MLMultiArray*> in_emb;
    std::vector<MLMultiArray*> in_mask;
    std::vector<MLMultiArray*> out_emb;
    std::vector<MLDictionaryFeatureProvider*> feats;
    std::vector<MLPredictionOptions*> opts;
    std::vector<dispatch_semaphore_t> sems;
};

static inline void copy_oi_out_768(float* dst, MLMultiArray* arr) {
    if (arr.dataType == MLMultiArrayDataTypeFloat32) {
        const auto* sp = static_cast<const float*>(arr.dataPointer);
        fast_shared::l2_normalize_f32(sp, dst, 768);
    } else {
        const auto* hp = static_cast<const float16_t*>(arr.dataPointer);
        float tmp[768];
        for (int c = 0; c < 768; ++c) tmp[c] = static_cast<float>(hp[c]);
        fast_shared::l2_normalize_f32(tmp, dst, 768);
    }
}

void* emb2_oi_load_embed(const char* mlmodelc_path, const char* function_name, int S, int pipe_depth, int compute_mode) {
    @autoreleasepool {
        if (!mlmodelc_path || S <= 0) {
            record_emb2_error("emb2_oi_load_embed", nil, "invalid path or S");
            return nullptr;
        }
        clear_emb2_error();
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = fast_shared::resolve_compute_units(static_cast<fast_shared::ComputeMode>(compute_mode));
        if (function_name && function_name[0] != '\0' && std::strcmp(function_name, "main") != 0) {
            if (@available(macOS 15.0, *)) {
                cfg.functionName = [NSString stringWithUTF8String:function_name];
            }
        }
        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model || err) {
            record_emb2_error("emb2_oi_load_embed(load)", err, mlmodelc_path);
            return nullptr;
        }

        MLFeatureDescription* msk_desc = model.modelDescription.inputDescriptionsByName[@"attention_mask"];
        MLFeatureDescription* out_desc = model.modelDescription.outputDescriptionsByName[@"embedding"];
        if (!msk_desc || !out_desc) {
            record_emb2_error("emb2_oi_load_embed(desc)", nil, "missing attention_mask or embedding feature");
            return nullptr;
        }

        auto* h = new Emb2OiHandle();
        h->model = model;
        h->S = S;
        h->slots = std::max(1, pipe_depth);
        h->mask_dtype = msk_desc.multiArrayConstraint.dataType;
        h->out_dtype = out_desc.multiArrayConstraint.dataType;
        h->in_emb.resize(h->slots);
        h->in_mask.resize(h->slots);
        h->out_emb.resize(h->slots);
        h->feats.resize(h->slots);
        h->opts.resize(h->slots);
        h->sems.resize(h->slots);

        for (int i = 0; i < h->slots; ++i) {
            NSError* e = nil;
            h->in_emb[i] = [[MLMultiArray alloc] initWithShape:@[@1, @(S), @512] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            h->in_mask[i] = [[MLMultiArray alloc] initWithShape:@[@1, @(S)] dataType:h->mask_dtype error:&e];
            h->out_emb[i] = [[MLMultiArray alloc] initWithShape:@[@1, @768] dataType:h->out_dtype error:&e];
            if (!h->in_emb[i] || !h->in_mask[i] || !h->out_emb[i] || e) {
                record_emb2_error("emb2_oi_load_embed(alloc)", e, mlmodelc_path);
                delete h;
                return nullptr;
            }
            memset(h->in_emb[i].dataPointer, 0, static_cast<size_t>(S) * 512 * sizeof(uint16_t));
            if (h->mask_dtype == MLMultiArrayDataTypeFloat32) {
                auto* mp = static_cast<float*>(h->in_mask[i].dataPointer);
                mp[0] = 1.0f;
                for (int k = 1; k < S; ++k) mp[k] = 0.0f;
            } else if (h->mask_dtype == MLMultiArrayDataTypeInt32) {
                auto* mp = static_cast<int32_t*>(h->in_mask[i].dataPointer);
                mp[0] = 1;
                for (int k = 1; k < S; ++k) mp[k] = 0;
            } else {
                auto* mp = static_cast<float16_t*>(h->in_mask[i].dataPointer);
                mp[0] = static_cast<float16_t>(1.0f);
                for (int k = 1; k < S; ++k) mp[k] = static_cast<float16_t>(0.0f);
            }
            h->feats[i] = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"inputs_embeds": h->in_emb[i],
                @"attention_mask": h->in_mask[i],
            } error:&e];
            if (!h->feats[i] || e) {
                record_emb2_error("emb2_oi_load_embed(feats)", e, mlmodelc_path);
                delete h;
                return nullptr;
            }
            h->opts[i] = [MLPredictionOptions new];
            h->opts[i].outputBackings = @{@"embedding": h->out_emb[i]};
            h->sems[i] = dispatch_semaphore_create(1);
        }

        // Fill non-zero first row so max-abs RMSNorm warmup never divides 0/0.
        auto* warm_p = static_cast<float16_t*>(h->in_emb[0].dataPointer);
        for (int c = 0; c < 512; ++c) warm_p[c] = static_cast<float16_t>((c & 1) ? -0.25f : 0.25f);
        id<MLFeatureProvider> warm = [model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
        if (!warm || err) {
            record_emb2_error("emb2_oi_load_embed(warmup)", err, mlmodelc_path);
            delete h;
            return nullptr;
        }
        return static_cast<void*>(h);
    }
}

double emb2_oi_exec_embed(void* handle, int N, const uint16_t* embeds_f16, const uint16_t* mask_f16, float* out_f32_768) {
    @autoreleasepool {
        auto* h = static_cast<Emb2OiHandle*>(handle);
        if (!h || N < 0 || !out_f32_768) return -1.0;
        if (N == 0) return 0.0;
        const int S = h->S;
        const int slots = h->slots;
        const size_t emb_bytes = static_cast<size_t>(S) * 512 * sizeof(uint16_t);
        const double t0 = fast_shared::now_sec();

        auto fill_mask = [&](int slot, const uint16_t* m_src) {
            if (h->mask_dtype == MLMultiArrayDataTypeFloat16) {
                memcpy(h->in_mask[slot].dataPointer, m_src, static_cast<size_t>(S) * sizeof(uint16_t));
            } else if (h->mask_dtype == MLMultiArrayDataTypeFloat32) {
                const auto* mh = reinterpret_cast<const float16_t*>(m_src);
                auto* mf = static_cast<float*>(h->in_mask[slot].dataPointer);
                for (int k = 0; k < S; ++k) mf[k] = static_cast<float>(mh[k]);
            } else {
                const auto* mh = reinterpret_cast<const float16_t*>(m_src);
                auto* mi = static_cast<int32_t*>(h->in_mask[slot].dataPointer);
                for (int k = 0; k < S; ++k) mi[k] = (static_cast<float>(mh[k]) > 0.5f) ? 1 : 0;
            }
        };

        if (slots == 1 || N == 1) {
            for (int i = 0; i < N; ++i) {
                memcpy(h->in_emb[0].dataPointer, embeds_f16 + static_cast<size_t>(i) * S * 512, emb_bytes);
                fill_mask(0, mask_f16 + static_cast<size_t>(i) * S);
                NSError* err = nil;
                id<MLFeatureProvider> res = [h->model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
                if (!res || err) {
                    record_emb2_error("emb2_oi_exec_embed(sync)", err);
                    return -1.0;
                }
                copy_oi_out_768(out_f32_768 + static_cast<size_t>(i) * 768, h->out_emb[0]);
            }
        } else {
            dispatch_group_t grp = dispatch_group_create();
            std::atomic<int> err_flag{0};
            std::atomic<int>* err_ptr = &err_flag;
            for (int i = 0; i < N; ++i) {
                if (err_flag.load(std::memory_order_acquire) != 0) break;
                const int s = i % slots;
                dispatch_semaphore_wait(h->sems[s], DISPATCH_TIME_FOREVER);
                if (err_flag.load(std::memory_order_acquire) != 0) {
                    dispatch_semaphore_signal(h->sems[s]);
                    break;
                }
                memcpy(h->in_emb[s].dataPointer, embeds_f16 + static_cast<size_t>(i) * S * 512, emb_bytes);
                fill_mask(s, mask_f16 + static_cast<size_t>(i) * S);
                dispatch_group_enter(grp);
                dispatch_semaphore_t sem = h->sems[s];
                MLMultiArray* o_arr = h->out_emb[s];
                float* dst = out_f32_768 + static_cast<size_t>(i) * 768;
                [h->model predictionFromFeatures:h->feats[s] options:h->opts[s]
                                completionHandler:^(id<MLFeatureProvider> out, NSError* e) {
                    if (!out || e != nil) {
                        int exp = 0;
                        if (err_ptr->compare_exchange_strong(exp, 1, std::memory_order_acq_rel)) {
                            record_emb2_error("emb2_oi_exec_embed(async)", e);
                        }
                    } else if (err_ptr->load(std::memory_order_acquire) == 0) {
                        copy_oi_out_768(dst, o_arr);
                    }
                    dispatch_semaphore_signal(sem);
                    dispatch_group_leave(grp);
                }];
            }
            dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
            if (err_flag.load(std::memory_order_acquire) != 0) return -1.0;
        }
        return (fast_shared::now_sec() - t0) * 1e3;
    }
}

void emb2_oi_free(void* handle) {
    if (handle) delete static_cast<Emb2OiHandle*>(handle);
}

}  // extern "C"
