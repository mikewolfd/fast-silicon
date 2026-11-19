// Portable Apple Silicon Multi-Engine Concurrency & CoreML Placement Probe (GPU + ANE + CPU SME2).
// Compile:
//   xcrun clang++ -O3 -std=c++20 -fobjc-arc \
//     -framework Foundation -framework CoreML -framework Metal \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate \
//     probe_ane_and_hetero.mm -o /tmp/probe_ane_hetero
// Run:
//   /tmp/probe_ane_hetero [--mlmodelc /path/to/model.mlmodelc] [--m 1024] [--k 2048] [--n 2048] [--secs 1.5]

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Accelerate/Accelerate.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <mutex>
#include <thread>
#include <vector>

static double now_sec() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

static void init_f16(__fp16* p, size_t n, float scale = 0.01f) {
    for (size_t i = 0; i < n; i++) p[i] = (__fp16)(scale * float((int(i * 1315423911u) % 200) - 100));
}

static void init_bf16(uint16_t* p, size_t n, float scale = 0.01f) {
    for (size_t i = 0; i < n; i++) {
        float v = scale * float((int(i * 1315423911u) % 200) - 100);
        uint32_t u;
        memcpy(&u, &v, 4);
        p[i] = uint16_t(u >> 16);
    }
}

static void print_runtime_diagnostics(id<MTLDevice> dev) {
    NSString* osStr = NSProcessInfo.processInfo.operatingSystemVersionString;
    BOOL a9 = [dev supportsFamily:MTLGPUFamilyApple9];
    BOOL a10 = [dev supportsFamily:MTLGPUFamilyApple10];
    BOOL m4 = NO;
    if (@available(macOS 26.0, *)) {
        m4 = [dev supportsFamily:MTLGPUFamilyMetal4];
    }
    const char* sparseStr = "unavailable(<26.4)";
    if ([dev respondsToSelector:@selector(supportsPlacementSparse)]) {
        if (@available(macOS 26.4, *)) {
            sparseStr = dev.supportsPlacementSparse ? "1" : "0";
        }
    }
    printf("Host OS: %s\n", osStr.UTF8String);
    printf("Metal  : %s | Apple9(M4)=%d Apple10(M5/NAX)=%d Metal4=%d supportsPlacementSparse=%s\n",
           dev.name.UTF8String, (int)a9, (int)a10, (int)m4, sparseStr);

    // CoreML compute devices & MLNeuralEngineComputeDevice.totalCoreCount diagnostic
    if (@available(macOS 14.0, *)) {
        NSArray<id<MLComputeDeviceProtocol>>* devices = [MLModel availableComputeDevices];
        int cpuDevs = 0, gpuDevs = 0, aneDevs = 0;
        NSInteger aneCoreCount = -1;
        for (id<MLComputeDeviceProtocol> d in devices) {
            if ([d isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
                aneDevs++;
                aneCoreCount = ((MLNeuralEngineComputeDevice*)d).totalCoreCount;
            } else if ([d isKindOfClass:[MLGPUComputeDevice class]]) {
                gpuDevs++;
            } else if ([d isKindOfClass:[MLCPUComputeDevice class]]) {
                cpuDevs++;
            }
        }
        printf("CoreML : devices(CPU=%d GPU=%d ANE=%d) | MLNeuralEngineComputeDevice.totalCoreCount=%ld%s\n",
               cpuDevs, gpuDevs, aneDevs, (long)aneCoreCount,
               (aneCoreCount == 2 ? " [WARNING: macOS 26.2 CoreML reporting bug; physical ANE is 16 cores]" : ""));
    }

    // Core AI (macOS 27.0+) shared-MTLBuffer / async InferenceFunction availability check
    BOOL coreAIOs = NO;
    if (@available(macOS 27.0, *)) {
        coreAIOs = YES;
    }
    NSBundle* coreAIBundle = [NSBundle bundleWithPath:@"/System/Library/Frameworks/CoreAI.framework"];
    void* coreAIHandle = dlopen("/System/Library/Frameworks/CoreAI.framework/CoreAI", RTLD_LAZY | RTLD_LOCAL);
    BOOL coreAILoaded = (coreAIHandle != nullptr);
    if (coreAIHandle) dlclose(coreAIHandle);
    printf("CoreAI : macOS>=27.0=%d framework_present=%d runtime_loaded=%d (%s)\n",
           (int)coreAIOs, (int)(coreAIBundle != nil), (int)coreAILoaded,
           (coreAIOs && coreAILoaded)
               ? "shared-MTLBuffer InferenceFunction.encode eligible"
               : "falling back to MPSGraph Metal queue + CoreML");
}

static void inspect_compute_plan(NSURL* url, MLModelConfiguration* cfg, const char* label) {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    __block int ane_ops = 0, gpu_ops = 0, cpu_ops = 0;
    __block NSError* planErr = nil;
    [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
        if (err) {
            planErr = err;
        } else if (plan && plan.modelStructure.program) {
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
    if (planErr) {
        fprintf(stderr, "  [%s] MLComputePlan error: %s\n", label, planErr.localizedDescription.UTF8String);
    } else {
        printf("  [%s] MLComputePlan ops: ANE=%d GPU=%d CPU=%d\n", label, ane_ops, gpu_ops, cpu_ops);
    }
}

struct CoreMLRunner {
    MLModel* model = nil;
    NSMutableArray<MLDictionaryFeatureProvider*>* inFeats = nil;
    NSMutableArray<MLPredictionOptions*>* opts = nil;
    int slots = 0;
};

static bool fill_multiarray_input(MLMultiArray* arr, MLMultiArrayDataType dt) {
    if (!arr || !arr.dataPointer) return false;
    NSInteger count = arr.count;
    switch (dt) {
        case MLMultiArrayDataTypeFloat16:
            init_f16((__fp16*)arr.dataPointer, (size_t)count);
            return true;
        case MLMultiArrayDataTypeFloat32: {
            float* p = (float*)arr.dataPointer;
            for (NSInteger k = 0; k < count; k++) p[k] = 0.01f * float(k % 10);
            return true;
        }
        case MLMultiArrayDataTypeDouble: {
            double* p = (double*)arr.dataPointer;
            for (NSInteger k = 0; k < count; k++) p[k] = 0.01 * double(k % 10);
            return true;
        }
        case MLMultiArrayDataTypeInt32: {
            int32_t* p = (int32_t*)arr.dataPointer;
            for (NSInteger k = 0; k < count; k++) p[k] = (int32_t)(k % 32);
            return true;
        }
        default: {
            // Safely zero-fill any other integer/packed format (e.g., Int8) without float* buffer overrun
            memset(arr.dataPointer, 0, (size_t)count);
            return true;
        }
    }
}

static CoreMLRunner make_coreml_runner(NSString* path, MLComputeUnits units, int slots = 4) {
    CoreMLRunner r;
    r.slots = slots;
    NSURL* url = [NSURL fileURLWithPath:path];
    MLModelConfiguration* cfg = [MLModelConfiguration new];
    cfg.computeUnits = units;
    NSError* err = nil;
    inspect_compute_plan(url, cfg, "CoreML-ANE");
    r.model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (!r.model || err) {
        fprintf(stderr, "Failed to load %s: %s\n", path.UTF8String,
                err ? err.localizedDescription.UTF8String : "null model");
        r.model = nil;
        return r;
    }

    r.inFeats = [NSMutableArray arrayWithCapacity:slots];
    r.opts = [NSMutableArray arrayWithCapacity:slots];
    auto* inDescs = r.model.modelDescription.inputDescriptionsByName;
    auto* outDescs = r.model.modelDescription.outputDescriptionsByName;
    for (int s = 0; s < slots; s++) {
        NSMutableDictionary* inDict = [NSMutableDictionary new];
        for (NSString* name in inDescs) {
            MLFeatureDescription* d = inDescs[name];
            if (d.type != MLFeatureTypeMultiArray) continue;
            err = nil;
            MLMultiArray* arr = [[MLMultiArray alloc] initWithShape:d.multiArrayConstraint.shape
                                                           dataType:d.multiArrayConstraint.dataType
                                                              error:&err];
            if (!arr || err) {
                fprintf(stderr, "Input MLMultiArray alloc failed for %s: %s\n",
                        name.UTF8String, err ? err.localizedDescription.UTF8String : "null");
                r.model = nil;
                return r;
            }
            fill_multiarray_input(arr, d.multiArrayConstraint.dataType);
            inDict[name] = arr;
        }
        NSMutableDictionary* outDict = [NSMutableDictionary new];
        for (NSString* name in outDescs) {
            MLFeatureDescription* d = outDescs[name];
            if (d.type != MLFeatureTypeMultiArray) continue;
            err = nil;
            MLMultiArray* arr = [[MLMultiArray alloc] initWithShape:d.multiArrayConstraint.shape
                                                           dataType:d.multiArrayConstraint.dataType
                                                              error:&err];
            if (!arr || err) {
                fprintf(stderr, "Output MLMultiArray alloc failed for %s: %s\n",
                        name.UTF8String, err ? err.localizedDescription.UTF8String : "null");
                r.model = nil;
                return r;
            }
            outDict[name] = arr;
        }
        err = nil;
        MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inDict error:&err];
        if (!fp || err) {
            fprintf(stderr, "MLDictionaryFeatureProvider err: %s\n",
                    err ? err.localizedDescription.UTF8String : "null");
            r.model = nil;
            return r;
        }
        MLPredictionOptions* opt = [MLPredictionOptions new];
        opt.outputBackings = outDict;
        [r.inFeats addObject:fp];
        [r.opts addObject:opt];
    }
    return r;
}

int main(int argc, char** argv) {
    NSString* mlmodelcPath = nil;
    int M = 1024, K = 2048, N = 4096;
    double secs = 1.2;
    int smeWorkers = 2;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--mlmodelc") && i + 1 < argc) mlmodelcPath = [NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i], "--m") && i + 1 < argc) M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--secs") && i + 1 < argc) secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sme-workers") && i + 1 < argc) smeWorkers = std::max(1, atoi(argv[++i]));
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "MTLCreateSystemDefaultDevice returned nil\n");
            return 1;
        }
        print_runtime_diagnostics(dev);

        id<MTLCommandQueue> queue = [dev newCommandQueue];
        MPSGraph* graph = [MPSGraph new];
        MPSGraphTensor* tA = [graph placeholderWithShape:@[@(M), @(K)] dataType:MPSDataTypeBFloat16 name:@"A"];
        MPSGraphTensor* tB = [graph placeholderWithShape:@[@(K), @(N)] dataType:MPSDataTypeBFloat16 name:@"B"];
        MPSGraphTensor* tOut = [graph matrixMultiplicationWithPrimaryTensor:tA secondaryTensor:tB name:@"mm"];
        id<MTLBuffer> bufA = [dev newBufferWithLength:size_t(M) * K * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [dev newBufferWithLength:size_t(K) * N * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [dev newBufferWithLength:size_t(M) * N * 2 options:MTLResourceStorageModeShared];
        init_bf16((uint16_t*)bufA.contents, size_t(M) * K);
        init_bf16((uint16_t*)bufB.contents, size_t(K) * N);
        MPSGraphExecutable* exec = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:dev]
                                                      feeds:@{
                                                          tA: [[MPSGraphShapedType alloc] initWithShape:@[@(M), @(K)] dataType:MPSDataTypeBFloat16],
                                                          tB: [[MPSGraphShapedType alloc] initWithShape:@[@(K), @(N)] dataType:MPSDataTypeBFloat16]
                                                      }
                                              targetTensors:@[tOut]
                                           targetOperations:nil
                                      compilationDescriptor:nil];
        MPSGraphTensorData* dA = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA shape:@[@(M), @(K)] dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData* dB = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:@[@(K), @(N)] dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData* dC = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufC shape:@[@(M), @(N)] dataType:MPSDataTypeBFloat16];

        // Warmup GPU
        for (int i = 0; i < 4; i++) {
            MPSCommandBuffer* cb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            [exec encodeToCommandBuffer:cb inputsArray:@[dA, dB] resultsArray:@[dC] executionDescriptor:nil];
            [cb commit];
            if (i == 3) [cb waitUntilCompleted];
        }

        // CPU SME2 setup: 2 independent worker buffers/workspaces (saturates both physical SME2 units)
        std::vector<uint16_t> cA(size_t(M) * K), cB(size_t(N) * K);
        init_bf16(cA.data(), cA.size());
        init_bf16(cB.data(), cB.size());

        std::vector<std::vector<uint16_t>> cCs((size_t)smeWorkers, std::vector<uint16_t>(size_t(M) * N));
        std::vector<BNNSNDArrayDescriptor> bAs((size_t)smeWorkers), bBs((size_t)smeWorkers), bCs((size_t)smeWorkers);
        std::vector<void*> b_ws((size_t)smeWorkers, nullptr);
        bool sme_ready = true;
        for (int w = 0; w < smeWorkers; ++w) {
            bAs[w] = {}; bBs[w] = {}; bCs[w] = {};
            bAs[w].layout = BNNSDataLayoutRowMajorMatrix; bAs[w].size[0] = K; bAs[w].size[1] = M; bAs[w].data = cA.data(); bAs[w].data_type = BNNSDataTypeBFloat16;
            bBs[w].layout = BNNSDataLayoutRowMajorMatrix; bBs[w].size[0] = K; bBs[w].size[1] = N; bBs[w].data = cB.data(); bBs[w].data_type = BNNSDataTypeBFloat16;
            bCs[w].layout = BNNSDataLayoutRowMajorMatrix; bCs[w].size[0] = N; bCs[w].size[1] = M; bCs[w].data = cCs[w].data(); bCs[w].data_type = BNNSDataTypeBFloat16;
            ssize_t ws_sz = BNNSMatMulWorkspaceSize(false, true, 1.0f, &bAs[w], &bBs[w], &bCs[w], nullptr);
            if (ws_sz < 0) {
                fprintf(stderr, "BNNSMatMulWorkspaceSize failed (rc=%zd) for worker %d\n", ws_sz, w);
                sme_ready = false;
                break;
            }
            if (ws_sz > 0) {
                b_ws[w] = malloc((size_t)ws_sz);
                if (!b_ws[w]) {
                    fprintf(stderr, "Failed to allocate %zd-byte BNNS workspace for worker %d\n", ws_sz, w);
                    sme_ready = false;
                    break;
                }
            }
        }

        auto gpu_loop = [&](double dur) -> double {
            double t0 = now_sec();
            int n = 0;
            while (now_sec() - t0 < dur) {
                MPSCommandBuffer* last = nil;
                for (int j = 0; j < 8; j++) {
                    MPSCommandBuffer* cb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
                    [exec encodeToCommandBuffer:cb inputsArray:@[dA, dB] resultsArray:@[dC] executionDescriptor:nil];
                    [cb commit];
                    last = cb;
                    n++;
                }
                [last waitUntilCompleted];
            }
            return (2.0 * M * K * N * n) / (now_sec() - t0) / 1e12;
        };

        // Runs `smeWorkers` (default 2) concurrent SME2 threads to measure true peak SME2 throughput.
        auto cpu_loop = [&](std::atomic<bool>& stop, double& out_tf) {
            if (!sme_ready) {
                out_tf = 0.0;
                return;
            }
            std::atomic<int> total_calls{0};
            double t0 = now_sec();
            std::vector<std::thread> workers;
            workers.reserve((size_t)smeWorkers);
            for (int w = 0; w < smeWorkers; ++w) {
                workers.emplace_back([&, w]() {
                    int local_n = 0;
                    while (!stop.load(std::memory_order_relaxed)) {
                        int rc = BNNSMatMul(false, true, 1.0f, &bAs[w], &bBs[w], &bCs[w], b_ws[w], nullptr);
                        if (rc != 0) {
                            fprintf(stderr, "BNNSMatMul failed (rc=%d) on worker %d\n", rc, w);
                            break;
                        }
                        local_n++;
                    }
                    total_calls.fetch_add(local_n, std::memory_order_relaxed);
                });
            }
            for (auto& th : workers) th.join();
            out_tf = (2.0 * M * K * N * total_calls.load()) / std::max(now_sec() - t0, 1e-9) / 1e12;
        };

        double solo_g = gpu_loop(secs);
        std::atomic<bool> stop{false};
        double solo_c = 0.0;
        {
            std::thread w(cpu_loop, std::ref(stop), std::ref(solo_c));
            usleep((useconds_t)(secs * 1e6));
            stop = true;
            w.join();
        }
        printf("Solo (M=%d,K=%d,N=%d, SME_workers=%d): GPU=%.2f TF | CPU-SME2=%.2f TF\n",
               M, K, N, smeWorkers, solo_g, solo_c);

        stop = false;
        double gc_c = 0.0;
        std::thread w_c(cpu_loop, std::ref(stop), std::ref(gc_c));
        double gc_g = gpu_loop(secs);
        stop = true;
        w_c.join();
        printf("GPU+CPU Concurrent:    GPU=%.2f TF + CPU=%.2f TF (total=%.2f TF vs soloGPU %+.2f TF)\n",
               gc_g, gc_c, gc_g + gc_c, (gc_g + gc_c) - solo_g);

        if (mlmodelcPath) {
            constexpr int kPipeDepth = 4;
            CoreMLRunner ane = make_coreml_runner(mlmodelcPath, MLComputeUnitsCPUAndNeuralEngine, kPipeDepth);
            if (ane.model) {
                auto ane_loop = [&](std::atomic<bool>& st, double& calls_per_s) {
                    // Explicit synchronized free-slot pool: prevents out-of-order completion slot races
                    // that occur when indexing `inFeats[n % 4]` without tracking which slot finished.
                    dispatch_semaphore_t inflight = dispatch_semaphore_create(kPipeDepth);
                    dispatch_group_t grp = dispatch_group_create();
                    std::mutex slot_mu;
                    std::vector<int> free_slots;
                    free_slots.reserve(kPipeDepth);
                    for (int s = 0; s < kPipeDepth; ++s) free_slots.push_back(s);

                    std::atomic<int> err_count{0};
                    std::atomic<int> ok_count{0};
                    auto* slot_mu_p = &slot_mu;
                    auto* free_slots_p = &free_slots;
                    auto* err_count_p = &err_count;
                    auto* ok_count_p = &ok_count;
                    double t0 = now_sec();
                    while (!st.load(std::memory_order_relaxed)) {
                        dispatch_semaphore_wait(inflight, DISPATCH_TIME_FOREVER);
                        int slot_idx = -1;
                        {
                            std::lock_guard<std::mutex> lk(slot_mu);
                            slot_idx = free_slots.back();
                            free_slots.pop_back();
                        }
                        dispatch_group_enter(grp);
                        [ane.model predictionFromFeatures:ane.inFeats[(NSUInteger)slot_idx]
                                                  options:ane.opts[(NSUInteger)slot_idx]
                                        completionHandler:^(id<MLFeatureProvider> o, NSError* e) {
                            if (e || !o) {
                                int prev = err_count_p->fetch_add(1, std::memory_order_relaxed);
                                if (prev == 0) {
                                    fprintf(stderr, "ANE async prediction error: %s\n",
                                            e ? e.localizedDescription.UTF8String : "nil output");
                                }
                            } else {
                                ok_count_p->fetch_add(1, std::memory_order_relaxed);
                            }
                            {
                                std::lock_guard<std::mutex> lk(*slot_mu_p);
                                free_slots_p->push_back(slot_idx);
                            }
                            dispatch_semaphore_signal(inflight);
                            dispatch_group_leave(grp);
                        }];
                    }
                    dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
                    if (err_count.load() > 0) {
                        fprintf(stderr, "ANE loop encountered %d prediction error(s)\n", err_count.load());
                    }
                    calls_per_s = ok_count.load() / std::max(now_sec() - t0, 1e-9);
                };

                stop = false;
                double solo_a_cps = 0.0;
                {
                    std::thread w(ane_loop, std::ref(stop), std::ref(solo_a_cps));
                    usleep((useconds_t)(secs * 1e6));
                    stop = true;
                    w.join();
                }
                stop = false;
                double ga_a = 0.0;
                std::thread w_a(ane_loop, std::ref(stop), std::ref(ga_a));
                double ga_g = gpu_loop(secs);
                stop = true; w_a.join();

                stop = false;
                double all_a = 0.0, all_c = 0.0;
                std::thread w_a2(ane_loop, std::ref(stop), std::ref(all_a));
                std::thread w_c2(cpu_loop, std::ref(stop), std::ref(all_c));
                double all_g = gpu_loop(secs);
                stop = true; w_a2.join(); w_c2.join();

                double solo_a_denom = std::max(solo_a_cps, 1e-9);
                printf("ANE Solo pipe4:        %.1f calls/s (%.2f ms/call)\n", solo_a_cps, 1000.0 / solo_a_denom);
                printf("GPU+ANE Concurrent:    GPU=%.2f TF (%+.1f%%) | ANE=%.1f calls/s (%+.1f%%)\n",
                       ga_g, 100.0 * (ga_g - solo_g) / solo_g, ga_a, 100.0 * (ga_a - solo_a_cps) / solo_a_denom);
                printf("GPU+ANE+CPU 3-Way:     GPU=%.2f TF (%+.1f%%) | ANE=%.1f calls/s (%+.1f%%) | CPU=%.2f TF\n",
                       all_g, 100.0 * (all_g - solo_g) / solo_g, all_a, 100.0 * (all_a - solo_a_cps) / solo_a_denom, all_c);
            }
        }
        for (void* ptr : b_ws) {
            if (ptr) free(ptr);
        }
    }
    return 0;
}
