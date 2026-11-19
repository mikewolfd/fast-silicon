// Portable Apple Silicon Multi-Engine Concurrency & CoreML Placement Probe (GPU + ANE + CPU SME).
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
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
#include <dispatch/dispatch.h>

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

static void inspect_compute_plan(NSURL* url, MLModelConfiguration* cfg, const char* label) {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    __block int ane_ops = 0, gpu_ops = 0, cpu_ops = 0;
    [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
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
    printf("  [%s] MLComputePlan ops: ANE=%d GPU=%d CPU=%d\n", label, ane_ops, gpu_ops, cpu_ops);
}

struct CoreMLRunner {
    MLModel* model = nil;
    NSMutableArray<MLDictionaryFeatureProvider*>* inFeats = nil;
    NSMutableArray<MLPredictionOptions*>* opts = nil;
};

static CoreMLRunner make_coreml_runner(NSString* path, MLComputeUnits units, int slots = 4) {
    CoreMLRunner r;
    NSURL* url = [NSURL fileURLWithPath:path];
    MLModelConfiguration* cfg = [MLModelConfiguration new];
    cfg.computeUnits = units;
    NSError* err = nil;
    inspect_compute_plan(url, cfg, "CoreML-ANE");
    r.model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (!r.model) {
        fprintf(stderr, "Failed to load %s: %s\n", path.UTF8String, err.localizedDescription.UTF8String);
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
            MLMultiArray* arr = [[MLMultiArray alloc] initWithShape:d.multiArrayConstraint.shape dataType:d.multiArrayConstraint.dataType error:&err];
            if (d.multiArrayConstraint.dataType == MLMultiArrayDataTypeFloat16) {
                init_f16((__fp16*)arr.dataPointer, arr.count);
            } else if (d.multiArrayConstraint.dataType == MLMultiArrayDataTypeInt32) {
                int32_t* p = (int32_t*)arr.dataPointer;
                for (NSInteger k = 0; k < arr.count; k++) p[k] = (int32_t)(k % 32);
            } else {
                float* p = (float*)arr.dataPointer;
                for (NSInteger k = 0; k < arr.count; k++) p[k] = 0.01f * (k % 10);
            }
            inDict[name] = arr;
        }
        NSMutableDictionary* outDict = [NSMutableDictionary new];
        for (NSString* name in outDescs) {
            MLFeatureDescription* d = outDescs[name];
            if (d.type != MLFeatureTypeMultiArray) continue;
            MLMultiArray* arr = [[MLMultiArray alloc] initWithShape:d.multiArrayConstraint.shape dataType:d.multiArrayConstraint.dataType error:&err];
            outDict[name] = arr;
        }
        MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inDict error:&err];
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
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--mlmodelc") && i + 1 < argc) mlmodelcPath = [NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i], "--m") && i + 1 < argc) M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--secs") && i + 1 < argc) secs = atof(argv[++i]);
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
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

        // CPU SME setup (BNNSMatMul BF16)
        std::vector<uint16_t> cA(size_t(M) * K), cB(size_t(N) * K), cC(size_t(M) * N);
        init_bf16(cA.data(), cA.size());
        init_bf16(cB.data(), cB.size());
        BNNSNDArrayDescriptor bA = {}, bB = {}, bC = {};
        bA.layout = BNNSDataLayoutRowMajorMatrix; bA.size[0] = K; bA.size[1] = M; bA.data = cA.data(); bA.data_type = BNNSDataTypeBFloat16;
        bB.layout = BNNSDataLayoutRowMajorMatrix; bB.size[0] = K; bB.size[1] = N; bB.data = cB.data(); bB.data_type = BNNSDataTypeBFloat16;
        bC.layout = BNNSDataLayoutRowMajorMatrix; bC.size[0] = N; bC.size[1] = M; bC.data = cC.data(); bC.data_type = BNNSDataTypeBFloat16;
        ssize_t ws_sz = BNNSMatMulWorkspaceSize(false, true, 1.0f, &bA, &bB, &bC, nullptr);
        void* b_ws = malloc(ws_sz);

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

        auto cpu_loop = [&](std::atomic<bool>& stop, double& out_tf) {
            double t0 = now_sec();
            int n = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                BNNSMatMul(false, true, 1.0f, &bA, &bB, &bC, b_ws, nullptr);
                n++;
            }
            out_tf = (2.0 * M * K * N * n) / (now_sec() - t0) / 1e12;
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
        printf("Solo (M=%d,K=%d,N=%d): GPU=%.2f TF | CPU-SME=%.2f TF\n", M, K, N, solo_g, solo_c);

        stop = false;
        double gc_c = 0.0;
        std::thread w_c(cpu_loop, std::ref(stop), std::ref(gc_c));
        double gc_g = gpu_loop(secs);
        stop = true;
        w_c.join();
        printf("GPU+CPU Concurrent:    GPU=%.2f TF + CPU=%.2f TF (total=%.2f TF vs soloGPU %+.2f TF)\n",
               gc_g, gc_c, gc_g + gc_c, (gc_g + gc_c) - solo_g);

        if (mlmodelcPath) {
            CoreMLRunner ane = make_coreml_runner(mlmodelcPath, MLComputeUnitsCPUAndNeuralEngine, 4);
            if (ane.model) {
                auto ane_loop = [&](std::atomic<bool>& st, double& calls_per_s) {
                    dispatch_semaphore_t inflight = dispatch_semaphore_create(4);
                    dispatch_group_t grp = dispatch_group_create();
                    double t0 = now_sec();
                    int n = 0;
                    while (!st.load(std::memory_order_relaxed)) {
                        dispatch_semaphore_wait(inflight, DISPATCH_TIME_FOREVER);
                        dispatch_group_enter(grp);
                        int idx = n % 4;
                        [ane.model predictionFromFeatures:ane.inFeats[idx] options:ane.opts[idx] completionHandler:^(id<MLFeatureProvider> o, NSError* e) {
                            dispatch_semaphore_signal(inflight);
                            dispatch_group_leave(grp);
                        }];
                        n++;
                    }
                    dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
                    calls_per_s = n / (now_sec() - t0);
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

                printf("ANE Solo pipe4:        %.1f calls/s (%.2f ms/call)\n", solo_a_cps, 1000.0 / solo_a_cps);
                printf("GPU+ANE Concurrent:    GPU=%.2f TF (%+.1f%%) | ANE=%.1f calls/s (%+.1f%%)\n",
                       ga_g, 100.0 * (ga_g - solo_g) / solo_g, ga_a, 100.0 * (ga_a - solo_a_cps) / solo_a_cps);
                printf("GPU+ANE+CPU 3-Way:     GPU=%.2f TF (%+.1f%%) | ANE=%.1f calls/s (%+.1f%%) | CPU=%.2f TF\n",
                       all_g, 100.0 * (all_g - solo_g) / solo_g, all_a, 100.0 * (all_a - solo_a_cps) / solo_a_cps, all_c);
            }
        }
        free(b_ws);
    }
    return 0;
}
