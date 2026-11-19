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

struct Emb2PkHandle {
    MLModel* model = nil;
    int S = 0;
    int slots = 1;
    int compute_mode = 0;
    std::vector<MLMultiArray*> ah, ab, acs, ass, acf, asf, ao;
    std::vector<MLDictionaryFeatureProvider*> feats;
    std::vector<MLPredictionOptions*> opts;
    std::vector<dispatch_semaphore_t> sems;
};

}  // namespace

extern "C" {

const char* emb2_last_error(void) {
    std::lock_guard<std::mutex> lk(g_err_mu);
    return g_last_err.c_str();
}

int emb2_get_plan_counts(
    const char* mlmodelc_path,
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
                MLModelStructureProgramFunction* fn = plan.modelStructure.program.functions[@"main"];
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

void* emb2_pk_load_units(const char* mlmodelc_path, int S, int pipe_depth, int compute_mode) {
    @autoreleasepool {
        if (!mlmodelc_path || S <= 0) {
            record_emb2_error("emb2_pk_load_units", nil, "invalid path or S");
            return nullptr;
        }
        clear_emb2_error();
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = fast_shared::resolve_compute_units(static_cast<fast_shared::ComputeMode>(compute_mode));

        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model || err) {
            record_emb2_error("emb2_pk_load_units(load)", err, mlmodelc_path);
            return nullptr;
        }

        auto* h = new Emb2PkHandle();
        h->model = model;
        h->S = S;
        h->compute_mode = compute_mode;
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
            h->ao[i]  = [[MLMultiArray alloc] initWithShape:@[@(S), @512] dataType:MLMultiArrayDataTypeFloat16 error:&e];
            if (!h->ah[i] || !h->ab[i] || !h->acs[i] || !h->ass[i] || !h->acf[i] || !h->asf[i] || !h->ao[i] || e) {
                record_emb2_error("emb2_pk_load_units(alloc)", e, mlmodelc_path);
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
                        memcpy(out_all + static_cast<size_t>(idx) * S * 512, h->ao[0].dataPointer, o_bytes);
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
                            memcpy(dst, src, o_bytes);
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

}  // extern "C"
