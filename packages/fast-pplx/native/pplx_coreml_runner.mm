#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include "fast_shared_apple.hpp"
#include "fast_shared_coreml.hpp"
#include "pplx_coreml_runner.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kMaxPipeDepth = 8;

thread_local std::string tl_last_error;

void clear_last_error() {
    tl_last_error.clear();
}

void set_last_error(const std::string& msg) {
    tl_last_error = msg;
}

std::string format_nserror(NSError* err, const char* fallback) {
    if (err == nil) {
        return fallback ? std::string(fallback) : std::string("unknown CoreML error");
    }
    NSString* desc = [err localizedDescription];
    NSString* domain = [err domain];
    std::string out;
    if (domain != nil) {
        out += [domain UTF8String];
        out += "(" + std::to_string(static_cast<long long>([err code])) + "): ";
    }
    if (desc != nil) {
        out += [desc UTF8String];
    } else if (fallback) {
        out += fallback;
    }
    return out;
}

std::string json_escape(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() + 8);
    for (unsigned char ch : raw) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(ch));
                }
        }
    }
    return out;
}

inline MLComputeUnits map_compute_units(int compute_units) {
    if (compute_units == 1) return fast_shared::resolve_compute_units(fast_shared::ComputeMode::kCpuOnly);
    if (compute_units == 2) return fast_shared::resolve_compute_units(fast_shared::ComputeMode::kAll);
    if (compute_units == 3) return MLComputeUnitsCPUAndGPU;
    return fast_shared::resolve_compute_units(fast_shared::ComputeMode::kCpuAndNeuralEngine);
}

struct PlanCounts {
    int total_ops = 0;
    int ane_ops = 0;
    int gpu_ops = 0;
    int cpu_ops = 0;
    int other_ops = 0;
};

bool inspect_stage_plan_counts_once(
    NSURL* url,
    MLModelConfiguration* cfg,
    PlanCounts& out_counts,
    std::string& err_out) {
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
        err_out = format_nserror(captured_err, "MLComputePlan load failed");
        return false;
    }

    PlanCounts counts{};
    MLModelStructureProgram* prog = captured.modelStructure.program;
    if (prog != nil) {
        MLModelStructureProgramFunction* main_fn = prog.functions[@"main"];
        if (main_fn != nil) {
            for (MLModelStructureProgramOperation* op in main_fn.block.operations) {
                ++counts.total_ops;
                MLComputePlanDeviceUsage* usage = [captured computeDeviceUsageForMLProgramOperation:op];
                id<MLComputeDeviceProtocol> dev = usage.preferredComputeDevice;
                if ([dev isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
                    ++counts.ane_ops;
                } else if ([dev isKindOfClass:[MLGPUComputeDevice class]]) {
                    ++counts.gpu_ops;
                } else if ([dev isKindOfClass:[MLCPUComputeDevice class]]) {
                    ++counts.cpu_ops;
                } else {
                    ++counts.other_ops;
                }
            }
        }
    }
    out_counts = counts;
    return true;
}

bool inspect_stage_plan_counts_via_clone(
    NSURL* src_url,
    MLModelConfiguration* cfg,
    NSString* subdir_name,
    PlanCounts& out_counts,
    std::string& err_out) {
    NSFileManager* fm = [NSFileManager defaultManager];
    NSString* base_dir = [NSTemporaryDirectory() stringByAppendingPathComponent:subdir_name];
    NSError* fs_err = nil;
    if (![fm createDirectoryAtPath:base_dir withIntermediateDirectories:YES attributes:nil error:&fs_err]) {
        err_out = "temp plan dir creation failed: " + format_nserror(fs_err, "mkdir failed");
        return false;
    }
    NSString* model_name = [src_url lastPathComponent];
    NSURL* dst_url = [NSURL fileURLWithPath:[base_dir stringByAppendingPathComponent:model_name]];
    [fm removeItemAtURL:dst_url error:nil];
    if (![fm copyItemAtURL:src_url toURL:dst_url error:&fs_err]) {
        err_out = "temp plan copy failed: " + format_nserror(fs_err, "copyItemAtURL failed");
        return false;
    }
    PlanCounts retry_counts{};
    std::string retry_err;
    bool ok = inspect_stage_plan_counts_once(dst_url, cfg, retry_counts, retry_err);
    [fm removeItemAtURL:dst_url error:nil];
    if (!ok) {
        err_out = retry_err;
        return false;
    }
    out_counts = retry_counts;
    return true;
}

bool inspect_stage_plan_counts(
    NSURL* url,
    MLModelConfiguration* cfg,
    PlanCounts& out_counts,
    std::string& err_out) {
    PlanCounts counts{};
    std::string first_err;
    bool ok = inspect_stage_plan_counts_once(url, cfg, counts, first_err);
    auto is_degenerate = [](const PlanCounts& c) -> bool {
        return c.total_ops > 0 && c.ane_ops == 0 && c.gpu_ops == 0 && c.cpu_ops == 0;
    };
    if (ok && !is_degenerate(counts)) {
        out_counts = counts;
        return true;
    }

    // macOS E5RT can retain partially evicted e5bundlecache entries (missing weights.bin
    // or empty analytics device usages). Retry against a clean APFS CoW clone URL if the
    // source .mlmodelc exists on disk.
    NSFileManager* fm = [NSFileManager defaultManager];
    BOOL is_dir = NO;
    if (![fm fileExistsAtPath:[url path] isDirectory:&is_dir] || !is_dir) {
        if (ok) {
            out_counts = counts;
            return true;
        }
        err_out = first_err;
        return false;
    }

    PlanCounts retry_counts{};
    std::string retry_err;
    if (inspect_stage_plan_counts_via_clone(url, cfg, @"fast_pplx_e5plan_retry", retry_counts, retry_err) &&
        !is_degenerate(retry_counts)) {
        out_counts = retry_counts;
        return true;
    }

    NSString* unique_dir = [NSString stringWithFormat:@"fast_pplx_e5plan_%@", [[NSUUID UUID] UUIDString]];
    if (inspect_stage_plan_counts_via_clone(url, cfg, unique_dir, retry_counts, retry_err)) {
        [fm removeItemAtPath:[NSTemporaryDirectory() stringByAppendingPathComponent:unique_dir] error:nil];
        out_counts = retry_counts;
        return true;
    }
    [fm removeItemAtPath:[NSTemporaryDirectory() stringByAppendingPathComponent:unique_dir] error:nil];

    if (ok) {
        out_counts = counts;
        return true;
    }
    err_out = first_err.empty() ? retry_err : first_err;
    return false;
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
    bool all_inputs_dense = false;
    std::vector<uint16_t> fallback_out;
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
    bool owns_slot_release = true;
    const uint16_t* out_h_ptr = nullptr;
    std::string error_msg;
};

bool init_slot(
    PplxSlot& slot,
    int seq_len,
    int hidden_dim,
    int chunk_size,
    int rope_half,
    std::string& err_out) {
    const int n_chunks = std::max(1, seq_len / std::max(1, chunk_size));
    NSError* err = nil;

    auto make_arr = [&](NSArray<NSNumber*>* shape, const char* name) -> MLMultiArray* {
        err = nil;
        MLMultiArray* a = fast_shared::make_f16_array(shape, &err);
        if (!a || err != nil) {
            err_out = std::string("MLMultiArray alloc failed for ") + name + ": " +
                      format_nserror(err, "nil array");
            return nil;
        }
        return a;
    };

    slot.in_h = make_arr(@[@1, @(hidden_dim), @1, @(seq_len)], "in_h");
    if (!slot.in_h) return false;
    slot.in_bias = make_arr(@[@1, @1, @(seq_len), @(seq_len)], "in_bias");
    if (!slot.in_bias) return false;
    slot.in_cos32 = make_arr(@[@1, @1, @(seq_len), @(rope_half)], "in_cos32");
    if (!slot.in_cos32) return false;
    slot.in_sin32 = make_arr(@[@1, @1, @(seq_len), @(rope_half)], "in_sin32");
    if (!slot.in_sin32) return false;
    slot.in_tok_valid = make_arr(@[@1, @1, @1, @(seq_len)], "in_tok_valid");
    if (!slot.in_tok_valid) return false;
    slot.in_gdn_carry = make_arr(@[@1, @(n_chunks), @1, @1], "in_gdn_carry");
    if (!slot.in_gdn_carry) return false;
    slot.ping = make_arr(@[@1, @(hidden_dim), @1, @(seq_len)], "ping");
    if (!slot.ping) return false;
    slot.pong = make_arr(@[@1, @(hidden_dim), @1, @(seq_len)], "pong");
    if (!slot.pong) return false;

    slot.all_inputs_dense =
        fast_shared::is_dense_4d(slot.in_h) &&
        fast_shared::is_dense_4d(slot.in_bias) &&
        fast_shared::is_dense_4d(slot.in_cos32) &&
        fast_shared::is_dense_4d(slot.in_sin32) &&
        fast_shared::is_dense_4d(slot.in_tok_valid) &&
        fast_shared::is_dense_4d(slot.in_gdn_carry);

    return true;
}

void run_stage_chain_async(PplxJob* job, uint16_t* copy_dst_h_out) {
    PplxRunner* runner = job->runner;
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
                  job->error_msg = "stage " + std::to_string(st) + " feature provider error: " +
                                   format_nserror(err, "nil provider");
                  NSLog(@"pplx_ane_submit %s", job->error_msg.c_str());
                  job->ok = 0;
                  break;
              }
              MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
              opts.outputBackings = @{@"h_out" : target_out};

              id<MLFeatureProvider> out =
                  [runner->stages[st] predictionFromFeatures:provider options:opts error:&err];
              if (!out || err != nil) {
                  job->error_msg = "stage " + std::to_string(st) + " prediction error: " +
                                   format_nserror(err, "nil output");
                  NSLog(@"pplx_ane_submit %s", job->error_msg.c_str());
                  job->ok = 0;
                  break;
              }
              MLMultiArray* res = [out featureValueForName:@"h_out"].multiArrayValue;
              if (res == nil) {
                  job->error_msg = "stage " + std::to_string(st) + " missing h_out multiArrayValue";
                  NSLog(@"pplx_ane_submit %s", job->error_msg.c_str());
                  job->ok = 0;
                  break;
              }
              cur_h = res;
          }

          if (job->ok && cur_h != nil) {
              if (copy_dst_h_out != nullptr) {
                  fast_shared::copy_from_mla_u16(copy_dst_h_out, cur_h);
                  job->out_h_ptr = copy_dst_h_out;
              } else if (fast_shared::is_dense_4d(cur_h)) {
                  job->out_h_ptr = reinterpret_cast<const uint16_t*>(cur_h.dataPointer);
              } else {
                  if (s.fallback_out.size() < runner->n_h) {
                      s.fallback_out.resize(runner->n_h);
                  }
                  fast_shared::copy_from_mla_u16(s.fallback_out.data(), cur_h);
                  job->out_h_ptr = s.fallback_out.data();
              }
          }
          job->elapsed_s = fast_shared::now_sec() - job->submit_t;

          if (job->owns_slot_release) {
              {
                  std::lock_guard<std::mutex> lock(runner->slot_mu);
                  runner->free_slots.push_back(job->slot_idx);
              }
              dispatch_semaphore_signal(runner->sem);
          }
      }
    });
}

}  // namespace

extern "C" {

const char* pplx_ane_last_error(void) {
    return tl_last_error.c_str();
}

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
        clear_last_error();
        if (!model_paths || num_stages <= 0 || seq_len <= 0 || hidden_dim <= 0 ||
            chunk_size <= 0 || rope_half <= 0) {
            set_last_error("Invalid arguments to pplx_ane_load_stages");
            return nullptr;
        }
        if ((seq_len % chunk_size) != 0) {
            set_last_error(
                "seq_len (" + std::to_string(seq_len) + ") must be divisible by chunk_size (" +
                std::to_string(chunk_size) + ")");
            std::fprintf(stderr, "pplx_ane_load_stages: %s\n", tl_last_error.c_str());
            return nullptr;
        }

        const int eff_depth = std::max(1, std::min(pipe_depth, kMaxPipeDepth));
        auto* runner = new PplxRunner();
        runner->seq_len = seq_len;
        runner->hidden_dim = hidden_dim;
        runner->chunk_size = chunk_size;
        runner->rope_half = rope_half;
        runner->pipe_depth = eff_depth;

        const int n_chunks = seq_len / runner->chunk_size;
        runner->n_h = static_cast<size_t>(hidden_dim) * seq_len;
        runner->n_bias = static_cast<size_t>(seq_len) * seq_len;
        runner->n_rope = static_cast<size_t>(seq_len) * runner->rope_half;
        runner->n_valid = static_cast<size_t>(seq_len);
        runner->n_carry = static_cast<size_t>(n_chunks);

        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = map_compute_units(compute_units);

        const char* allow_fb_env = std::getenv("FAST_PPLX_ALLOW_ANE_FALLBACK");
        const bool require_pure_ane =
            (compute_units == 0) && !(allow_fb_env && std::strcmp(allow_fb_env, "1") == 0);

        runner->stages.reserve(static_cast<size_t>(num_stages));
        for (int i = 0; i < num_stages; ++i) {
            if (!model_paths[i]) {
                set_last_error("NULL model path at stage " + std::to_string(i));
                delete runner;
                return nullptr;
            }
            NSString* path = [NSString stringWithUTF8String:model_paths[i]];
            NSURL* url = [NSURL fileURLWithPath:path];

            if (require_pure_ane) {
                PlanCounts plan{};
                std::string plan_err;
                if (!inspect_stage_plan_counts(url, cfg, plan, plan_err)) {
                    set_last_error(
                        "MLComputePlan failed for stage " + std::to_string(i) + " (" +
                        std::string(model_paths[i]) + "): " + plan_err);
                    NSLog(@"pplx_ane_load_stages %s", tl_last_error.c_str());
                    delete runner;
                    return nullptr;
                }
                if (plan.ane_ops <= 0 || plan.cpu_ops > 0 || plan.gpu_ops > 0) {
                    set_last_error(
                        "ANE residency check failed for stage " + std::to_string(i) + " (" +
                        std::string(model_paths[i]) + "): ane_ops=" + std::to_string(plan.ane_ops) +
                        " cpu_ops=" + std::to_string(plan.cpu_ops) +
                        " gpu_ops=" + std::to_string(plan.gpu_ops) +
                        " total_ops=" + std::to_string(plan.total_ops));
                    NSLog(@"pplx_ane_load_stages %s", tl_last_error.c_str());
                    delete runner;
                    return nullptr;
                }
            }

            NSError* err = nil;
            MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
            if (!model || err != nil) {
                set_last_error(
                    "MLModel load failed on stage " + std::to_string(i) + " (" +
                    std::string(model_paths[i]) + "): " + format_nserror(err, "nil model"));
                NSLog(@"pplx_ane_load_stages %s", tl_last_error.c_str());
                delete runner;
                return nullptr;
            }
            runner->stages.push_back(model);
        }

        runner->slots.resize(static_cast<size_t>(eff_depth));
        runner->free_slots.reserve(static_cast<size_t>(eff_depth));
        for (int s = 0; s < eff_depth; ++s) {
            std::string slot_err;
            if (!init_slot(
                    runner->slots[s], seq_len, hidden_dim, runner->chunk_size, runner->rope_half, slot_err)) {
                set_last_error("Slot " + std::to_string(s) + " init failed: " + slot_err);
                NSLog(@"pplx_ane_load_stages %s", tl_last_error.c_str());
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

int pplx_ane_acquire_slot(void* runner_ptr, PplxSlotBuffers* out_bufs) {
    clear_last_error();
    if (!runner_ptr || !out_bufs) {
        set_last_error("Invalid arguments to pplx_ane_acquire_slot");
        return -1;
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
    if (!slot.all_inputs_dense) {
        {
            std::lock_guard<std::mutex> lock(runner->slot_mu);
            runner->free_slots.push_back(slot_idx);
        }
        dispatch_semaphore_signal(runner->sem);
        set_last_error("MLMultiArray slot inputs are non-dense; fallback required");
        return -1;
    }

    out_bufs->slot_idx = slot_idx;
    out_bufs->in_h = reinterpret_cast<uint16_t*>(slot.in_h.dataPointer);
    out_bufs->in_bias = reinterpret_cast<uint16_t*>(slot.in_bias.dataPointer);
    out_bufs->in_cos32 = reinterpret_cast<uint16_t*>(slot.in_cos32.dataPointer);
    out_bufs->in_sin32 = reinterpret_cast<uint16_t*>(slot.in_sin32.dataPointer);
    out_bufs->in_tok_valid = reinterpret_cast<uint16_t*>(slot.in_tok_valid.dataPointer);
    out_bufs->in_gdn_carry = reinterpret_cast<uint16_t*>(slot.in_gdn_carry.dataPointer);
    return slot_idx;
}

void* pplx_ane_submit_slot(void* runner_ptr, int slot_idx) {
    clear_last_error();
    if (!runner_ptr) {
        set_last_error("NULL runner in pplx_ane_submit_slot");
        return nullptr;
    }
    auto* runner = static_cast<PplxRunner*>(runner_ptr);
    if (slot_idx < 0 || slot_idx >= runner->pipe_depth) {
        set_last_error("Invalid slot_idx in pplx_ane_submit_slot");
        return nullptr;
    }

    auto* job = new PplxJob();
    job->runner = runner;
    job->grp = dispatch_group_create();
    job->slot_idx = slot_idx;
    job->submit_t = fast_shared::now_sec();
    job->ok = 1;
    job->owns_slot_release = false;

    run_stage_chain_async(job, nullptr);
    return job;
}

double pplx_ane_wait_slot(
    void* job_ptr,
    const uint16_t** out_h_ptr,
    int* out_slot_idx) {
    clear_last_error();
    if (!job_ptr) {
        set_last_error("NULL job in pplx_ane_wait_slot");
        return -1.0;
    }
    auto* job = static_cast<PplxJob*>(job_ptr);
    dispatch_group_wait(job->grp, DISPATCH_TIME_FOREVER);
    if (out_slot_idx) {
        *out_slot_idx = job->slot_idx;
    }
    if (out_h_ptr) {
        *out_h_ptr = job->ok ? job->out_h_ptr : nullptr;
    }
    if (!job->ok && !job->error_msg.empty()) {
        set_last_error(job->error_msg);
    }
    double dt = job->ok ? job->elapsed_s : -1.0;
    delete job;
    return dt;
}

void pplx_ane_release_slot(void* runner_ptr, int slot_idx) {
    if (!runner_ptr) {
        return;
    }
    auto* runner = static_cast<PplxRunner*>(runner_ptr);
    if (slot_idx < 0 || slot_idx >= runner->pipe_depth) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(runner->slot_mu);
        runner->free_slots.push_back(slot_idx);
    }
    dispatch_semaphore_signal(runner->sem);
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
    clear_last_error();
    if (!runner_ptr || !h_in_u16 || !attn_bias_u16 || !cos32_u16 || !sin32_u16 ||
        !tok_valid_u16 || !gdn_carry_u16 || !h_out_u16) {
        set_last_error("NULL argument passed to pplx_ane_submit");
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
    job->owns_slot_release = true;

    run_stage_chain_async(job, h_out_u16);
    return job;
}

double pplx_ane_wait(void* job_ptr) {
    clear_last_error();
    if (!job_ptr) {
        set_last_error("NULL job passed to pplx_ane_wait");
        return -1.0;
    }
    auto* job = static_cast<PplxJob*>(job_ptr);
    dispatch_group_wait(job->grp, DISPATCH_TIME_FOREVER);
    if (!job->ok && !job->error_msg.empty()) {
        set_last_error(job->error_msg);
    }
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
        clear_last_error();
        if (!model_path || !out_json_buf || buf_len <= 0) {
            set_last_error("Invalid arguments to pplx_ane_inspect_plan");
            return -1;
        }
        NSString* path = [NSString stringWithUTF8String:model_path];
        NSURL* url = [NSURL fileURLWithPath:path];
        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = map_compute_units(compute_units);

        PlanCounts plan{};
        std::string plan_err;
        if (!inspect_stage_plan_counts(url, cfg, plan, plan_err)) {
            set_last_error("pplx_ane_inspect_plan failed: " + plan_err);
            NSLog(@"%s", tl_last_error.c_str());
            std::string escaped = json_escape(plan_err);
            std::snprintf(
                out_json_buf,
                static_cast<size_t>(buf_len),
                "{\"error\":\"%s\"}",
                escaped.c_str());
            return -1;
        }

        int written = std::snprintf(
            out_json_buf,
            static_cast<size_t>(buf_len),
            "{\"total_ops\":%d,\"ane_ops\":%d,\"gpu_ops\":%d,\"cpu_ops\":%d,\"other_ops\":%d}",
            plan.total_ops,
            plan.ane_ops,
            plan.gpu_ops,
            plan.cpu_ops,
            plan.other_ops);
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
