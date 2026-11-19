// Portable Apple Silicon Metal 4 MPP::matmul2d & NAX/Apple10 Hardware Probe.
// Compile:
//   xcrun clang++ -O3 -std=c++20 -fobjc-arc \
//     -framework Foundation -framework Metal -framework MetalPerformancePrimitives \
//     probe_mpp_metal4.mm -o /tmp/probe_mpp_metal4
// Run:
//   /tmp/probe_mpp_metal4 [--k 2048] [--n 8192] [--quick]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static double now_sec() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

static NSString* kMetalSrc = @R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename LA, typename RB, typename DC, int TM, int TN, bool RELAXED, bool TRANS_R>
inline void run_matmul_tile(
    tensor<device LA, dextents<int32_t, 2>> A,
    tensor<device RB, dextents<int32_t, 2>> B,
    tensor<device DC, dextents<int32_t, 2>> C,
    uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent), false, TRANS_R, RELAXED);
    matmul2d<desc, execution_simdgroups<4>> op;
    int m_off = (int)(tgid.y * TM);
    int n_off = (int)(tgid.x * TN);
    auto sA = A.slice(0, m_off);
    auto sC = C.slice(n_off, m_off);
    if constexpr (TRANS_R) {
        auto sB = B.slice(0, n_off);
        op.run(sA, sB, sC);
    } else {
        auto sB = B.slice(n_off, 0);
        op.run(sA, sB, sC);
    }
}

#define DEF_KERNEL(NAME, LA, RB, DC, TM, TN, RELAXED, TRANS_R) \
kernel void NAME( \
    tensor<device LA, dextents<int32_t, 2>> A [[buffer(0)]], \
    tensor<device RB, dextents<int32_t, 2>> B [[buffer(1)]], \
    tensor<device DC, dextents<int32_t, 2>> C [[buffer(2)]], \
    uint2 tgid [[threadgroup_position_in_grid]]) \
{ \
    run_matmul_tile<LA, RB, DC, TM, TN, RELAXED, TRANS_R>(A, B, C, tgid); \
}

DEF_KERNEL(mm_f16_64x32_nt,  half,   half,         half,   64, 32, false, true)
DEF_KERNEL(mm_f16_64x64_nt,  half,   half,         half,   64, 64, false, true)
DEF_KERNEL(mm_f16_64x64_rel, half,   half,         half,   64, 64, true,  true)
DEF_KERNEL(mm_bf16_64x32_nt, bfloat, bfloat,       bfloat, 64, 32, false, true)
DEF_KERNEL(mm_bf16_64x64_nt, bfloat, bfloat,       bfloat, 64, 64, false, true)
DEF_KERNEL(mm_bf16_64x64_rel,bfloat, bfloat,       bfloat, 64, 64, true,  true)
DEF_KERNEL(mm_f16_i8_64x64,  half,   int8_t,       half,   64, 64, false, true)
DEF_KERNEL(mm_bf_i8_64x64,   bfloat, int8_t,       bfloat, 64, 64, false, true)
DEF_KERNEL(mm_i8_i8_64x64,   int8_t, int8_t,       int32_t,64, 64, false, true)
DEF_KERNEL(mm_f16_i4_64x64,  half,   int4b_format, half,   64, 64, false, true)
DEF_KERNEL(mm_bf_i4_64x64,   bfloat, int4b_format, bfloat, 64, 64, false, true)
)";

static size_t elem_size_bits(MTLTensorDataType dt) {
    switch (dt) {
        case MTLTensorDataTypeFloat16:
        case MTLTensorDataTypeBFloat16: return 16;
        case MTLTensorDataTypeInt8:     return 8;
        case MTLTensorDataTypeInt4:     return 4;
        case MTLTensorDataTypeInt32:    return 32;
        case MTLTensorDataTypeFloat32:  return 32;
        default: return 16;
    }
}

static id<MTLTensor> make_tensor(id<MTLDevice> dev, NSInteger rows, NSInteger cols, MTLTensorDataType dt, id<MTLBuffer>* outBuf = nullptr) {
    NSInteger dims[2] = {cols, rows};
    size_t bits = elem_size_bits(dt);
    NSInteger rowBytes = ((cols * bits) + 7) / 8;
    rowBytes = (rowBytes + 127) & ~NSInteger(127);
    NSInteger stride1 = (rowBytes * 8) / (NSInteger)bits;
    NSInteger strides[2] = {1, stride1};

    size_t totalBytes = (size_t)rowBytes * (size_t)rows;
    id<MTLBuffer> buf = [dev newBufferWithLength:totalBytes options:MTLResourceStorageModeShared];
    if (!buf) return nil;
    if (outBuf) *outBuf = buf;

    MTLTensorDescriptor* desc = [MTLTensorDescriptor new];
    desc.dimensions = [[MTLTensorExtents alloc] initWithRank:2 values:dims];
    desc.strides = [[MTLTensorExtents alloc] initWithRank:2 values:strides];
    desc.dataType = dt;
    desc.usage = MTLTensorUsageCompute;

    NSError* err = nil;
    id<MTLTensor> t = [buf newTensorWithDescriptor:desc offset:0 error:&err];
    if (err || !t) {
        fprintf(stderr, "Tensor alloc err: %s\n", err ? err.localizedDescription.UTF8String : "null");
        return nil;
    }
    return t;
}

static void fill_pattern(id<MTLBuffer> buf, MTLTensorDataType dt, NSInteger rows, NSInteger cols, NSInteger stride1) {
    size_t total = (size_t)rows * (size_t)stride1;
    if (dt == MTLTensorDataTypeFloat16) {
        __fp16* p = (__fp16*)buf.contents;
        for (size_t i = 0; i < total; ++i) p[i] = (__fp16)(0.01f * ((i % 7) - 3));
    } else if (dt == MTLTensorDataTypeBFloat16) {
        uint16_t* p = (uint16_t*)buf.contents;
        for (size_t i = 0; i < total; ++i) {
            float v = 0.01f * ((i % 7) - 3);
            uint32_t u;
            memcpy(&u, &v, 4);
            p[i] = (uint16_t)(u >> 16);
        }
    } else if (dt == MTLTensorDataTypeInt8) {
        int8_t* p = (int8_t*)buf.contents;
        for (size_t i = 0; i < total; ++i) p[i] = (int8_t)((i % 11) - 5);
    } else if (dt == MTLTensorDataTypeInt4) {
        uint8_t* p = (uint8_t*)buf.contents;
        for (size_t i = 0; i < total / 2; ++i) p[i] = 0x21;
    }
}

struct BenchSpec {
    const char* label;
    NSString* fn;
    MTLTensorDataType dtA, dtB, dtC;
    int TM, TN;
};

int main(int argc, char** argv) {
    int K = 2048, N = 8192;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--k") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quick")) quick = true;
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        BOOL a9 = [dev supportsFamily:MTLGPUFamilyApple9];
        BOOL a10 = [dev supportsFamily:MTLGPUFamilyApple10];
        BOOL m4 = [dev supportsFamily:MTLGPUFamilyMetal4];
        printf("Device: %s | apple9(M4)=%d apple10(M5/NAX)=%d metal4=%d\n",
               dev.name.UTF8String, a9, a10, m4);

        NSError* err = nil;
        id<MTL4CommandQueue> queue = [dev newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
        id<MTL4CommandBuffer> cmdBuf = [dev newCommandBuffer];
        id<MTL4Compiler> compiler = [dev newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&err];
        if (err || !compiler) { fprintf(stderr, "Compiler err: %s\n", err ? err.localizedDescription.UTF8String : "null"); return 1; }

        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        MTL4LibraryDescriptor* libDesc = [MTL4LibraryDescriptor new];
        libDesc.source = kMetalSrc;
        libDesc.options = opts;
        id<MTLLibrary> lib = [compiler newLibraryWithDescriptor:libDesc error:&err];
        if (err || !lib) { fprintf(stderr, "Lib compile err:\n%s\n", err ? err.localizedDescription.UTF8String : "null"); return 1; }

        MTL4ArgumentTableDescriptor* atDesc = [MTL4ArgumentTableDescriptor new];
        atDesc.maxBufferBindCount = 4;
        id<MTL4ArgumentTable> argTable = [dev newArgumentTableWithDescriptor:atDesc error:&err];

        BenchSpec specs[] = {
            {"MPP f16*f16->f16 64x32",       @"mm_f16_64x32_nt",  MTLTensorDataTypeFloat16,  MTLTensorDataTypeFloat16,  MTLTensorDataTypeFloat16,  64, 32},
            {"MPP f16*f16->f16 64x64",       @"mm_f16_64x64_nt",  MTLTensorDataTypeFloat16,  MTLTensorDataTypeFloat16,  MTLTensorDataTypeFloat16,  64, 64},
            {"MPP f16*f16->f16 64x64 relax", @"mm_f16_64x64_rel", MTLTensorDataTypeFloat16,  MTLTensorDataTypeFloat16,  MTLTensorDataTypeFloat16,  64, 64},
            {"MPP bf16*bf16->bf16 64x32",    @"mm_bf16_64x32_nt", MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, 64, 32},
            {"MPP bf16*bf16->bf16 64x64",    @"mm_bf16_64x64_nt", MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, 64, 64},
            {"MPP bf16*bf16->bf16 64x64 rel",@"mm_bf16_64x64_rel",MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, 64, 64},
            {"MPP f16*int8->f16 64x64",      @"mm_f16_i8_64x64",  MTLTensorDataTypeFloat16,  MTLTensorDataTypeInt8,     MTLTensorDataTypeFloat16,  64, 64},
            {"MPP bf16*int8->bf16 64x64",    @"mm_bf_i8_64x64",   MTLTensorDataTypeBFloat16, MTLTensorDataTypeInt8,     MTLTensorDataTypeBFloat16, 64, 64},
            {"MPP int8*int8->int32 64x64",   @"mm_i8_i8_64x64",   MTLTensorDataTypeInt8,     MTLTensorDataTypeInt8,     MTLTensorDataTypeInt32,    64, 64},
            {"MPP f16*int4->f16 64x64",      @"mm_f16_i4_64x64",  MTLTensorDataTypeFloat16,  MTLTensorDataTypeInt4,     MTLTensorDataTypeFloat16,  64, 64},
            {"MPP bf16*int4->bf16 64x64",    @"mm_bf_i4_64x64",   MTLTensorDataTypeBFloat16, MTLTensorDataTypeInt4,     MTLTensorDataTypeBFloat16, 64, 64},
        };

        NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* psos = [NSMutableDictionary new];
        for (auto& sp : specs) {
            MTL4LibraryFunctionDescriptor* fnDesc = [MTL4LibraryFunctionDescriptor new];
            fnDesc.library = lib;
            fnDesc.name = sp.fn;
            MTL4ComputePipelineDescriptor* pd = [MTL4ComputePipelineDescriptor new];
            pd.computeFunctionDescriptor = fnDesc;
            pd.requiredThreadsPerThreadgroup = MTLSizeMake(128, 1, 1);
            id<MTLComputePipelineState> pso = [compiler newComputePipelineStateWithDescriptor:pd compilerTaskOptions:nil error:&err];
            if (!pso || err) {
                err = nil;
                continue;
            }
            psos[sp.fn] = pso;
        }

        std::vector<int> Ms = quick ? std::vector<int>{64, 512} : std::vector<int>{64, 128, 512, 2048};
        for (int M : Ms) {
            printf("\n=== Shape: M=%d, K=%d, N=%d (%.2f GFLOPs) ===\n", M, K, N, 2.0 * M * K * N / 1e9);
            for (auto& sp : specs) {
                id<MTLComputePipelineState> pso = psos[sp.fn];
                if (!pso) continue;

                id<MTLBuffer> bA = nil, bB = nil, bC = nil;
                id<MTLTensor> tA = make_tensor(dev, M, K, sp.dtA, &bA);
                id<MTLTensor> tB = make_tensor(dev, N, K, sp.dtB, &bB);
                id<MTLTensor> tC = make_tensor(dev, M, N, sp.dtC, &bC);
                if (!tA || !tB || !tC) continue;
                fill_pattern(bA, sp.dtA, M, K, [tA.strides extentAtDimensionIndex:1]);
                fill_pattern(bB, sp.dtB, N, K, [tB.strides extentAtDimensionIndex:1]);

                MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
                rsd.initialCapacity = 8;
                id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rsd error:&err];
                [rs addAllocation:bA]; [rs addAllocation:bB]; [rs addAllocation:bC];
                [rs addAllocation:tA]; [rs addAllocation:tB]; [rs addAllocation:tC];
                [rs commit];
                [rs requestResidency];

                [argTable setResource:tA.gpuResourceID atBufferIndex:0];
                [argTable setResource:tB.gpuResourceID atBufferIndex:1];
                [argTable setResource:tC.gpuResourceID atBufferIndex:2];

                MTLSize tgs = MTLSizeMake((N + sp.TN - 1) / sp.TN, (M + sp.TM - 1) / sp.TM, 1);
                MTLSize tpt = MTLSizeMake(128, 1, 1);

                auto run_n = [&](int reps, double* out_gpu_ms, double* out_wall_ms) {
                    __block CFTimeInterval gpu_dt = 0.0;
                    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
                    MTL4CommitOptions* co = [MTL4CommitOptions new];
                    [co addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
                        gpu_dt = fb.GPUEndTime - fb.GPUStartTime;
                        dispatch_semaphore_signal(sem);
                    }];
                    [alloc reset];
                    [cmdBuf beginCommandBufferWithAllocator:alloc];
                    [cmdBuf useResidencySet:rs];
                    id<MTL4ComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setArgumentTable:argTable];
                    for (int i = 0; i < reps; ++i) {
                        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpt];
                        if (i + 1 < reps) {
                            [enc barrierAfterEncoderStages:MTLStageDispatch
                                        beforeEncoderStages:MTLStageDispatch
                                          visibilityOptions:MTL4VisibilityOptionNone];
                        }
                    }
                    [enc endEncoding];
                    [cmdBuf endCommandBuffer];
                    double w0 = now_sec();
                    [queue commit:&cmdBuf count:1 options:co];
                    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
                    double w1 = now_sec();
                    *out_gpu_ms = (gpu_dt / reps) * 1e3;
                    *out_wall_ms = ((w1 - w0) / reps) * 1e3;
                };

                double g0, w0, g_ms, w_ms;
                run_n(2, &g0, &w0);
                int reps = (M <= 128) ? 12 : (M <= 512 ? 8 : 4);
                run_n(reps, &g_ms, &w_ms);
                double flops = 2.0 * M * K * N;
                double gpu_tf = (g_ms > 0) ? (flops / (g_ms * 1e-3) / 1e12) : 0.0;
                double wall_tf = flops / (w_ms * 1e-3) / 1e12;
                printf("  %-30s | GPU: %6.3f ms (%5.2f TF) | Wall: %6.3f ms (%5.2f TF)\n",
                       sp.label, g_ms, gpu_tf, w_ms, wall_tf);
                [rs endResidency];
            }
        }
    }
    return 0;
}
