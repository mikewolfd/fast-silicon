#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include "fast_shared_apple.hpp"
#include "fast_shared_coreml.hpp"
#include "pplx_coreml_runner.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kMaxPipeDepth = 8;

inline MLComputeUnits map_compute_units(int compute_units) {
    if (compute_units == 1) return fast_shared::resolve_compute_units(fast_shared::ComputeMode::kCpuOnly);
    if (compute_units == 2) return fast_shared::resolve_compute_units(fast_shared::ComputeMode::kAll);
    if (compute_units == 3) return MLComputeUnitsCPUAndGPU;
    return fast_shared::resolve_compute_units(fast_shared::ComputeMode::kCpuAndNeuralEngine);
}

struct PplxSlot {
    MLMultiArray* in_h = nil;
    MLMultiArray* in_bias = nil;
    MLMultiArray* in_cos32 = nil;
    MLMultiArray* in_sin32 = nil;
    MLMultiArray* in_tok_valid = nil;
    MLMultiArray* in_gdn_carry = nil;
    MLMultiArray* ping = nil;
    MLMultiArray* pong = nil;
};

struct PplxRunner {
    std::vector<MLModel*> stages;
    std::vector<PplxSlot> slots;
    std::vector<int> free_slots;
    std::mutex slot_mu;

    dispatch_semaphore_t sem = nil;
    dispatch_queue_t q = nil;

    int seq_len = 0;
    int hidden_dim = 1024;
    int chunk_size = 32;
    int rope_half = 32;
    int pipe_depth = 4;

    size_t n_h = 0;
    size_t n_bias = 0;
    size_t n_rope = 0;
    size_t n_valid = 0;
    size_t n_carry = 0;

    ~PplxRunner() {
        if (sem != nil) {
            for (int i = 0; i < pipe_depth; ++i) {
                dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            }
            for (int i = 0; i < pipe_depth; ++i) {
                dispatch_semaphore_signal(sem);
            }
        }
    }
};

struct PplxJob {
    PplxRunner* runner = nullptr;
    dispatch_group_t grp = nil;
    int slot_idx = 0;
    double submit_t = 0.0;
    double elapsed_s = -1.0;
    int ok = 1;
};

bool init_slot(PplxSlot& slot, int seq_len, int hidden_dim, int chunk_size, int rope_half) {
    const int n_chunks = std::max(1, seq_len / std::max(1, chunk_size));
    slot.in_h = fast_shared::make_f16_array(@[@1, @(hidden_dim), @1, @(seq_len)]);
    slot.in_bias = fast_shared::make_f16_array(@[@1, @1, @(seq_len), @(seq_len)]);
    slot.in_cos32 = fast_shared::make_f16_array(@[@1, @1, @(seq_len), @(rope_half)]);
    slot.in_sin32 = fast_shared::make_f16_array(@[@1, @1, @(seq_len), @(rope_half)]);
    slot.in_tok_valid = fast_shared::make_f16_array(@[@1, @1, @1, @(seq_len)]);
    slot.in_gdn_carry = fast_shared::make_f16_array(@[@1, @(n_chunks), @1, @1]);
    slot.ping = fast_shared::make_f16_array(@[@1, @(hidden_dim), @1, @(seq_len)]);
    slot.pong = fast_shared::make_f16_array(@[@1, @(hidden_dim), @1, @(seq_len)]);

    return slot.in_h && slot.in_bias && slot.in_cos32 && slot.in_sin32 &&
           slot.in_tok_valid && slot.in_gdn_carry && slot.ping && slot.pong;
}

}  // namespace

extern "C" {

void* pplx_ane_load_stages(
    const char* const* model_paths,
    int num_stages,
    int seq_len,
    int hidden_dim,
    int chunk_size,
    int rope_half,
    int pipe_depth,
    int compute_units) {
    @autoreleasepool {
        if (!model_paths || num_stages <= 0 || seq_len <= 0 || hidden_dim <= 0) {
            return nullptr;
        }
        const int eff_depth = std::max(1, std::min(pipe_depth, kMaxPipeDepth));
        auto* runner = new PplxRunner();
        runner->seq_len = seq_len;
        runner->hidden_dim = hidden_dim;
        runner->chunk_size = std::max(1, chunk_size);
        runner->rope_half = std::max(1, rope_half);
        runner->pipe_depth = eff_depth;

        const int n_chunks = std::max(1, seq_len / runner->chunk_size);
        runner->n_h = static_cast<size_t>(hidden_dim) * seq_len;
        runner->n_bias = static_cast<size_t>(seq_len) * seq_len;
        runner->n_rope = static_cast<size_t>(seq_len) * runner->rope_half;
        runner->n_valid = static_cast<size_t>(seq_len);
        runner->n_carry = static_cast<size_t>(n_chunks);

        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = map_compute_units(compute_units);

        runner->stages.reserve(static_cast<size_t>(num_stages));
        for (int i = 0; i < num_stages; ++i) {
            if (!model_paths[i]) {
                delete runner;
                return nullptr;
            }
            NSString* path = [NSString stringWithUTF8String:model_paths[i]];
            NSURL* url = [NSURL fileURLWithPath:path];
            NSError* err = nil;
            MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
            if (!model || err != nil) {
                NSLog(@"pplx_ane_load_stages failed on stage %d (%@): %@", i, path, err);
                delete runner;
                return nullptr;
            }
            runner->stages.push_back(model);
        }

        runner->slots.resize(static_cast<size_t>(eff_depth));
        runner->free_slots.reserve(static_cast<size_t>(eff_depth));
        for (int s = 0; s < eff_depth; ++s) {
            if (!init_slot(runner->slots[s], seq_len, hidden_dim, runner->chunk_size, runner->rope_half)) {
                delete runner;
                return nullptr;
            }
            runner->free_slots.push_back(eff_depth - 1 - s);
        }

        runner->sem = dispatch_semaphore_create(eff_depth);
        runner->q = dispatch_queue_create("fast_pplx.ane.runner", DISPATCH_QUEUE_CONCURRENT);
        return runner;
    }
}

void* pplx_ane_submit(
    void* runner_ptr,
    const uint16_t* h_in_u16,
    const uint16_t* attn_bias_u16,
    const uint16_t* cos32_u16,
    const uint16_t* sin32_u16,
    const uint16_t* tok_valid_u16,
    const uint16_t* gdn_carry_u16,
    uint16_t* h_out_u16) {
    if (!runner_ptr || !h_in_u16 || !attn_bias_u16 || !cos32_u16 || !sin32_u16 ||
        !tok_valid_u16 || !gdn_carry_u16 || !h_out_u16) {
        return nullptr;
    }

    auto* runner = static_cast<PplxRunner*>(runner_ptr);
    dispatch_semaphore_wait(runner->sem, DISPATCH_TIME_FOREVER);

    int slot_idx = 0;
    {
        std::lock_guard<std::mutex> lock(runner->slot_mu);
        slot_idx = runner->free_slots.back();
        runner->free_slots.pop_back();
    }

    PplxSlot& slot = runner->slots[slot_idx];
    fast_shared::copy_to_mla_u16(slot.in_h, h_in_u16);
    fast_shared::copy_to_mla_u16(slot.in_bias, attn_bias_u16);
    fast_shared::copy_to_mla_u16(slot.in_cos32, cos32_u16);
    fast_shared::copy_to_mla_u16(slot.in_sin32, sin32_u16);
    fast_shared::copy_to_mla_u16(slot.in_tok_valid, tok_valid_u16);
    fast_shared::copy_to_mla_u16(slot.in_gdn_carry, gdn_carry_u16);

    auto* job = new PplxJob();
    job->runner = runner;
    job->grp = dispatch_group_create();
    job->slot_idx = slot_idx;
    job->submit_t = fast_shared::now_sec();
    job->ok = 1;

    dispatch_group_async(job->grp, runner->q, ^{
      @autoreleasepool {
          PplxSlot& s = runner->slots[job->slot_idx];
          MLMultiArray* cur_h = s.in_h;
          const int num_stages = static_cast<int>(runner->stages.size());

          for (int st = 0; st < num_stages; ++st) {
              MLMultiArray* target_out = (cur_h == s.ping) ? s.pong : s.ping;
              NSError* err = nil;
              NSDictionary* feed = @{
                  @"h_in" : cur_h,
                  @"attn_bias" : s.in_bias,
                  @"cos32" : s.in_cos32,
                  @"sin32" : s.in_sin32,
                  @"tok_valid" : s.in_tok_valid,
                  @"gdn_carry" : s.in_gdn_carry,
              };
              MLDictionaryFeatureProvider* provider =
                  [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:&err];
              if (!provider || err != nil) {
                  job->ok = 0;
                  break;
              }
              MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
              opts.outputBackings = @{@"h_out" : target_out};

              id<MLFeatureProvider> out =
                  [runner->stages[st] predictionFromFeatures:provider options:opts error:&err];
              if (!out || err != nil) {
                  NSLog(@"pplx_ane_submit stage %d prediction error: %@", st, err);
                  job->ok = 0;
                  break;
              }
              MLMultiArray* res = [out featureValueForName:@"h_out"].multiArrayValue;
              cur_h = res ? res : target_out;
          }

          if (job->ok && cur_h != nil) {
              fast_shared::copy_from_mla_u16(h_out_u16, cur_h);
          }
          job->elapsed_s = fast_shared::now_sec() - job->submit_t;

          {
              std::lock_guard<std::mutex> lock(runner->slot_mu);
              runner->free_slots.push_back(job->slot_idx);
          }
          dispatch_semaphore_signal(runner->sem);
      }
    });

    return job;
}

double pplx_ane_wait(void* job_ptr) {
    if (!job_ptr) {
        return -1.0;
    }
    auto* job = static_cast<PplxJob*>(job_ptr);
    dispatch_group_wait(job->grp, DISPATCH_TIME_FOREVER);
    double dt = job->ok ? job->elapsed_s : -1.0;
    delete job;
    return dt;
}

double pplx_ane_exec(
    void* runner_ptr,
    const uint16_t* h_in_u16,
    const uint16_t* attn_bias_u16,
    const uint16_t* cos32_u16,
    const uint16_t* sin32_u16,
    const uint16_t* tok_valid_u16,
    const uint16_t* gdn_carry_u16,
    uint16_t* h_out_u16) {
    void* job = pplx_ane_submit(
        runner_ptr, h_in_u16, attn_bias_u16, cos32_u16, sin32_u16, tok_valid_u16, gdn_carry_u16, h_out_u16);
    if (!job) {
        return -1.0;
    }
    return pplx_ane_wait(job);
}

int pplx_ane_inspect_plan(
    const char* model_path,
    int compute_units,
    char* out_json_buf,
    int buf_len) {
    @autoreleasepool {
        if (!model_path || !out_json_buf || buf_len <= 0) {
            return -1;
        }
        NSString* path = [NSString stringWithUTF8String:model_path];
        NSURL* url = [NSURL fileURLWithPath:path];
        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = map_compute_units(compute_units);

        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block MLComputePlan* captured = nil;
        __block NSError* captured_err = nil;

        [MLComputePlan loadContentsOfURL:url
                           configuration:cfg
                       completionHandler:^(MLComputePlan* _Nullable plan, NSError* _Nullable error) {
                         captured = plan;
                         captured_err = error;
                         dispatch_semaphore_signal(done);
                       }];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);

        if (!captured || captured_err != nil) {
            int n = std::snprintf(
                out_json_buf,
                static_cast<size_t>(buf_len),
                "{\"error\":\"%s\"}",
                captured_err ? [[captured_err localizedDescription] UTF8String] : "load_failed");
            return n;
        }

        __block int ane_ops = 0;
        __block int gpu_ops = 0;
        __block int cpu_ops = 0;
        __block int other_ops = 0;
        __block int total_ops = 0;

        MLModelStructureProgram* prog = captured.modelStructure.program;
        if (prog != nil) {
            MLModelStructureProgramFunction* main_fn = prog.functions[@"main"];
            if (main_fn != nil) {
                for (MLModelStructureProgramOperation* op in main_fn.block.operations) {
                    ++total_ops;
                    MLComputePlanDeviceUsage* usage = [captured computeDeviceUsageForMLProgramOperation:op];
                    id<MLComputeDeviceProtocol> dev = usage.preferredComputeDevice;
                    if ([dev isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
                        ++ane_ops;
                    } else if ([dev isKindOfClass:[MLGPUComputeDevice class]]) {
                        ++gpu_ops;
                    } else if ([dev isKindOfClass:[MLCPUComputeDevice class]]) {
                        ++cpu_ops;
                    } else {
                        ++other_ops;
                    }
                }
            }
        }

        int written = std::snprintf(
            out_json_buf,
            static_cast<size_t>(buf_len),
            "{\"total_ops\":%d,\"ane_ops\":%d,\"gpu_ops\":%d,\"cpu_ops\":%d,\"other_ops\":%d}",
            total_ops,
            ane_ops,
            gpu_ops,
            cpu_ops,
            other_ops);
        return (written >= 0 && written < buf_len) ? written : -1;
    }
}

void pplx_ane_free(void* runner_ptr) {
    if (!runner_ptr) {
        return;
    }
    delete static_cast<PplxRunner*>(runner_ptr);
}

}  // extern "C"
