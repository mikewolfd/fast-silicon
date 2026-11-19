#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "emb2_coreml_runner.h"
#include "fast_shared_apple.hpp"

namespace {

struct Emb2PkHandle {
    MLModel* model = nil;
    int S = 0;
    int slots = 1;
    std::vector<MLMultiArray*> ah, ab, acs, ass, acf, asf, ao;
    std::vector<MLDictionaryFeatureProvider*> feats;
    std::vector<MLPredictionOptions*> opts;
    std::vector<dispatch_semaphore_t> sems;
};

}  // namespace

extern "C" {

void emb2_inspect_plan(const char* mlmodelc_path, int compute_mode) {
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = fast_shared::resolve_compute_units(static_cast<fast_shared::ComputeMode>(compute_mode));

        __block int ane_ops = 0;
        __block int gpu_ops = 0;
        __block int cpu_ops = 0;
        dispatch_semaphore_t psem = dispatch_semaphore_create(0);
        [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
            if (!err && plan && plan.modelStructure.program) {
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
        printf("[MLComputePlan] %s (mode=%d) -> ANE=%d GPU=%d CPU=%d\n",
               url.lastPathComponent.UTF8String, compute_mode, ane_ops, gpu_ops, cpu_ops);
    }
}

void* emb2_pk_load_units(const char* mlmodelc_path, int S, int pipe_depth, int compute_mode) {
    @autoreleasepool {
        if (!mlmodelc_path || S <= 0) return nullptr;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]];
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = fast_shared::resolve_compute_units(static_cast<fast_shared::ComputeMode>(compute_mode));

        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model || err) {
            if (err) {
                fprintf(stderr, "emb2_pk_load_units(%s) failed: %s\n",
                        mlmodelc_path, err.localizedDescription.UTF8String);
            }
            return nullptr;
        }

        auto* h = new Emb2PkHandle();
        h->model = model;
        h->S = S;
        h->slots = std::max(1, pipe_depth);
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
            h->opts[i] = [MLPredictionOptions new];
            h->opts[i].outputBackings = @{@"out_h": h->ao[i]};
            h->sems[i] = dispatch_semaphore_create(1);
        }

        // Single warmup prediction compiles ANE/CPU plan eagerly at load time.
        [model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
        if (err) {
            fprintf(stderr, "emb2_pk_load_units warmup(%s) failed: %s\n",
                    mlmodelc_path, err.localizedDescription.UTF8String);
            delete h;
            return nullptr;
        }
        return static_cast<void*>(h);
    }
}

struct Emb2Inflight {
    dispatch_group_t grp;
    double t0;
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
    dispatch_group_async(grp, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
        @autoreleasepool {
            const int S = h->S;
            const int slots = h->slots;
            const size_t h_bytes  = static_cast<size_t>(512) * S * 2;
            const size_t b_bytes  = static_cast<size_t>(S) * S * 2;
            const size_t cs_bytes = static_cast<size_t>(S) * 128 * 2;
            const size_t cf_bytes = static_cast<size_t>(S) * 256 * 2;
            const size_t o_bytes  = static_cast<size_t>(S) * 512 * 2;
            if (slots == 1) {
                NSError* err = nil;
                for (int idx = 0; idx < N; ++idx) {
                    memcpy(h->ah[0].dataPointer,  h_all  + static_cast<size_t>(idx) * 512 * S, h_bytes);
                    memcpy(h->ab[0].dataPointer,  b_all  + static_cast<size_t>(idx) * S * S,   b_bytes);
                    memcpy(h->acs[0].dataPointer, cs_all + static_cast<size_t>(idx) * S * 128, cs_bytes);
                    memcpy(h->ass[0].dataPointer, ss_all + static_cast<size_t>(idx) * S * 128, cs_bytes);
                    memcpy(h->acf[0].dataPointer, cf_all + static_cast<size_t>(idx) * S * 256, cf_bytes);
                    memcpy(h->asf[0].dataPointer, sf_all + static_cast<size_t>(idx) * S * 256, cf_bytes);
                    [h->model predictionFromFeatures:h->feats[0] options:h->opts[0] error:&err];
                    if (out_all) {
                        memcpy(out_all + static_cast<size_t>(idx) * S * 512, h->ao[0].dataPointer, o_bytes);
                    }
                }
            } else {
                for (int idx = 0; idx < N; ++idx) {
                    const int s = idx % slots;
                    dispatch_semaphore_wait(h->sems[s], DISPATCH_TIME_FOREVER);
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
                        (void)out;
                        (void)e;
                        if (dst) memcpy(dst, src, o_bytes);
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
    if (!flight) return 0.0;
    auto* fl = static_cast<Emb2Inflight*>(flight);
    dispatch_group_wait(fl->grp, DISPATCH_TIME_FOREVER);
    const double ms = (fast_shared::now_sec() - fl->t0) * 1e3;
    delete fl;
    return ms;
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
    void* fl = emb2_pk_submit_queue(handle, N, h_all, b_all, cs_all, ss_all, cf_all, sf_all, out_all);
    return emb2_pk_wait_queue(fl);
}

void emb2_pk_free(void* handle) {
    if (handle) delete static_cast<Emb2PkHandle*>(handle);
}

}  // extern "C"
