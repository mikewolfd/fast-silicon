// Portable Apple Silicon Metal 4 MPP::matmul2d, Cooperative Tensor & NAX/Apple10 Hardware Probe.
// Compile:
//   xcrun clang++ -O3 -std=c++20 -fobjc-arc \
//     -framework Foundation -framework Metal -framework MetalPerformancePrimitives \
//     probe_mpp_metal4.mm -o /tmp/probe_mpp_metal4
// Run:
//   /tmp/probe_mpp_metal4 [--k 2048] [--n 8192] [--quick]

#import <AvailabilityMacros.h>
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

// Common Metal 4 MPP matmul2d tile helper (used by every OS-tiered shader library).
static NSString* kMetalHeader = @R"(
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
)";

// Tier 1 (macOS 26.0+): FP16 MPP tensor operands
static NSString* kMetalSrcF16 = @R"(
DEF_KERNEL(mm_f16_64x32_nt,  half, half, half, 64, 32, false, true)
DEF_KERNEL(mm_f16_64x64_nt,  half, half, half, 64, 64, false, true)
DEF_KERNEL(mm_f16_64x64_rel, half, half, half, 64, 64, true,  true)
)";

// Tier 2 (macOS 26.1+): BF16 MPP tensor operands
static NSString* kMetalSrcBF16 = @R"(
DEF_KERNEL(mm_bf16_64x32_nt,  bfloat, bfloat, bfloat, 64, 32, false, true)
DEF_KERNEL(mm_bf16_64x64_nt,  bfloat, bfloat, bfloat, 64, 64, false, true)
DEF_KERNEL(mm_bf16_64x64_rel, bfloat, bfloat, bfloat, 64, 64, true,  true)
)";

// Tier 3 (macOS 26.3+): 1-SIMD-group (32 threads) compile-time-K cooperative tensor chaining probe
static NSString* kMetalSrcCoop263 = @R"(
kernel void mm_coop_chain_f16_32x32_k32(
    tensor<device half, dextents<int32_t, 2>> A  [[buffer(0)]],
    tensor<device half, dextents<int32_t, 2>> B1 [[buffer(1)]],
    tensor<device half, dextents<int32_t, 2>> B2 [[buffer(2)]],
    tensor<device half, dextents<int32_t, 2>> C  [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc1 = matmul2d_descriptor(32, 32, 32, false, true, false);
    constexpr auto desc2 = matmul2d_descriptor(32, 32, 32, false, true, false);
    matmul2d<desc1, execution_simdgroup> op1;
    matmul2d<desc2, execution_simdgroup> op2;

    int m_off = (int)(tgid.y * 32);
    int n_off = (int)(tgid.x * 32);
    auto sA  = A.slice(0, m_off);
    auto sB1 = B1.slice(0, 0);
    auto sB2 = B2.slice(0, n_off);
    auto sC  = C.slice(n_off, m_off);

    auto cMid = op1.get_destination_cooperative_tensor<decltype(sA), decltype(sB1), half>();
    #pragma unroll
    for (uint16_t i = 0; i < cMid.get_capacity(); ++i) {
        if (cMid.is_valid_element(i)) cMid[i] = half(0.0h);
    }
    op1.run(sA, sB1, cMid);
    if (op2.is_compatible_as_left_input<half, half, half>(cMid)) {
        auto leftIn = op2.get_left_input_cooperative_tensor<half, half, half>(cMid);
        op2.run(leftIn, sB2, sC);
    }
}
)";

// Tier 4 (macOS 26.4+): Direct INT8 & INT4 MPP tensor operands
static NSString* kMetalSrcInt264 = @R"(
DEF_KERNEL(mm_f16_i8_64x64,  half,   int8_t,       half,    64, 64, false, true)
DEF_KERNEL(mm_bf_i8_64x64,   bfloat, int8_t,       bfloat,  64, 64, false, true)
DEF_KERNEL(mm_i8_i8_64x64,   int8_t, int8_t,       int32_t, 64, 64, false, true)
DEF_KERNEL(mm_f16_i4_64x64,  half,   int4b_format, half,    64, 64, false, true)
DEF_KERNEL(mm_bf_i4_64x64,   bfloat, int4b_format, bfloat,  64, 64, false, true)
)";

// Tier 5 (macOS 27.0+ / MSL 4.1): FP8, FP4 (E2M1), and INT2 tensor operands
static NSString* kMetalSrcQuant270 = @R"(
DEF_KERNEL(mm_f16_fp8_64x64,  half,   metal_fp8_e4m3_format, half,   64, 64, false, true)
DEF_KERNEL(mm_bf_fp8_64x64,   bfloat, metal_fp8_e4m3_format, bfloat, 64, 64, false, true)
DEF_KERNEL(mm_f16_fp4_64x64,  half,   metal_fp4_e2m1_format, half,   64, 64, false, true)
DEF_KERNEL(mm_bf_fp4_64x64,   bfloat, metal_fp4_e2m1_format, bfloat, 64, 64, false, true)
DEF_KERNEL(mm_f16_i2_64x64,   half,   int2b_format,          half,   64, 64, false, true)
)";

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && (__MAC_OS_X_VERSION_MAX_ALLOWED >= 260400)
#define HAS_SDK_26_4 1
#else
#define HAS_SDK_26_4 0
#endif

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && (__MAC_OS_X_VERSION_MAX_ALLOWED >= 270000)
#define HAS_SDK_27_0 1
#else
#define HAS_SDK_27_0 0
#endif

static size_t elem_size_bits(MTLTensorDataType dt) {
    if (dt == MTLTensorDataTypeFloat16 || dt == MTLTensorDataTypeBFloat16) return 16;
    if (dt == MTLTensorDataTypeInt8 || dt == MTLTensorDataTypeUInt8) return 8;
    if (dt == MTLTensorDataTypeInt32 || dt == MTLTensorDataTypeFloat32) return 32;
#if HAS_SDK_26_4
    if (@available(macOS 26.4, *)) {
        if (dt == MTLTensorDataTypeInt4 || dt == MTLTensorDataTypeUInt4) return 4;
    }
#endif
#if HAS_SDK_27_0
    if (@available(macOS 27.0, *)) {
        if (dt == MTLTensorDataTypeMetalFloat8E4M3 || dt == MTLTensorDataTypeMetalFloat8E5M2) return 8;
        if (dt == MTLTensorDataTypeMetalFloat4E2M1) return 4;
        if (dt == MTLTensorDataTypeInt2 || dt == MTLTensorDataTypeUInt2) return 2;
    }
#endif
    return 16;
}

static id<MTLTensor> make_tensor(id<MTLDevice> dev, NSInteger rows, NSInteger cols, MTLTensorDataType dt, id<MTLBuffer>* outBuf = nullptr) {
    NSInteger dims[2] = {cols, rows};
    size_t bits = elem_size_bits(dt);
    NSInteger rowBytes = (NSInteger)(((size_t)cols * bits + 7) / 8);
    // Align row pitch to 128 bytes (covers 128-el FP8, 256-el FP4, 512-el Int2 stride requirements).
    rowBytes = (rowBytes + 127) & ~NSInteger(127);
    NSInteger stride1 = (rowBytes * 8) / (NSInteger)bits;
    NSInteger strides[2] = {1, stride1};

    size_t totalBytes = (size_t)rowBytes * (size_t)rows;
    id<MTLBuffer> buf = [dev newBufferWithLength:totalBytes options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "Buffer alloc failed (%zu bytes)\n", totalBytes);
        return nil;
    }
    if (outBuf) *outBuf = buf;

    MTLTensorDescriptor* desc = [MTLTensorDescriptor new];
    desc.dimensions = [[MTLTensorExtents alloc] initWithRank:2 values:dims];
    desc.strides = [[MTLTensorExtents alloc] initWithRank:2 values:strides];
    desc.dataType = dt;
    desc.usage = MTLTensorUsageCompute;

    NSError* err = nil;
    id<MTLTensor> t = [buf newTensorWithDescriptor:desc offset:0 error:&err];
    if (err || !t) {
        fprintf(stderr, "Tensor alloc err (dt=%ld, %ldx%ld): %s\n",
                (long)dt, (long)rows, (long)cols, err ? err.localizedDescription.UTF8String : "null");
        return nil;
    }
    return t;
}

static void fill_pattern(id<MTLBuffer> buf, MTLTensorDataType dt, NSInteger rows, NSInteger stride1) {
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
    } else {
        size_t bits = elem_size_bits(dt);
        size_t bytes = (total * bits + 7) / 8;
        memset(buf.contents, 0x21, bytes);
    }
}

struct BenchSpec {
    const char* label;
    NSString* fn;
    id<MTLLibrary> lib;
    MTLTensorDataType dtA, dtB, dtC;
    int TM, TN;
};

static id<MTLLibrary> compile_tier_lib(
    id<MTL4Compiler> compiler,
    MTLCompileOptions* opts,
    NSString* tierName,
    NSString* tierBody)
{
    NSError* err = nil;
    MTL4LibraryDescriptor* libDesc = [MTL4LibraryDescriptor new];
    libDesc.source = [kMetalHeader stringByAppendingString:tierBody];
    libDesc.options = opts;
    id<MTLLibrary> lib = [compiler newLibraryWithDescriptor:libDesc error:&err];
    if (err || !lib) {
        fprintf(stderr, "[tier:%s] Metal library compile skipped/failed: %s\n",
                tierName.UTF8String, err ? err.localizedDescription.UTF8String : "null");
        return nil;
    }
    return lib;
}

static void run_coop_263_probe(
    id<MTLDevice> dev,
    id<MTL4CommandQueue> queue,
    id<MTL4CommandAllocator> alloc,
    id<MTL4CommandBuffer> cmdBuf,
    id<MTL4Compiler> compiler,
    id<MTL4ArgumentTable> argTable,
    id<MTLLibrary> coopLib)
{
    if (!coopLib) return;
    @autoreleasepool {
        NSError* err = nil;
        MTL4LibraryFunctionDescriptor* fnDesc = [MTL4LibraryFunctionDescriptor new];
        fnDesc.library = coopLib;
        fnDesc.name = @"mm_coop_chain_f16_32x32_k32";
        MTL4ComputePipelineDescriptor* pd = [MTL4ComputePipelineDescriptor new];
        pd.computeFunctionDescriptor = fnDesc;
        pd.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);  // 1 SIMD group required on 26.3+
        id<MTLComputePipelineState> pso = [compiler newComputePipelineStateWithDescriptor:pd compilerTaskOptions:nil error:&err];
        if (!pso || err) {
            fprintf(stderr, "[tier:26.3-coop] PSO build failed: %s\n", err ? err.localizedDescription.UTF8String : "null");
            return;
        }

        id<MTLBuffer> bA = nil, bB1 = nil, bB2 = nil, bC = nil;
        id<MTLTensor> tA  = make_tensor(dev, 32, 32, MTLTensorDataTypeFloat16, &bA);
        id<MTLTensor> tB1 = make_tensor(dev, 32, 32, MTLTensorDataTypeFloat16, &bB1);
        id<MTLTensor> tB2 = make_tensor(dev, 32, 32, MTLTensorDataTypeFloat16, &bB2);
        id<MTLTensor> tC  = make_tensor(dev, 32, 32, MTLTensorDataTypeFloat16, &bC);
        if (!tA || !tB1 || !tB2 || !tC) return;

        // Fill A = 0.5, B1 = 0.25, B2 = 0.125 -> (A @ B1^T) = 32*0.125 = 4.0 -> (Mid @ B2^T) = 32*4.0*0.125 = 16.0
        __fp16* pA  = (__fp16*)bA.contents;
        __fp16* pB1 = (__fp16*)bB1.contents;
        __fp16* pB2 = (__fp16*)bB2.contents;
        NSInteger sA = [tA.strides extentAtDimensionIndex:1];
        NSInteger sB1 = [tB1.strides extentAtDimensionIndex:1];
        NSInteger sB2 = [tB2.strides extentAtDimensionIndex:1];
        NSInteger sC = [tC.strides extentAtDimensionIndex:1];
        for (NSInteger r = 0; r < 32; ++r) {
            for (NSInteger c = 0; c < 32; ++c) {
                pA[r * sA + c] = (__fp16)0.5f;
                pB1[r * sB1 + c] = (__fp16)0.25f;
                pB2[r * sB2 + c] = (__fp16)0.125f;
            }
        }
        memset(bC.contents, 0, bC.length);

        MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
        rsd.initialCapacity = 10;
        id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rsd error:&err];
        if (!rs || err) {
            fprintf(stderr, "[tier:26.3-coop] ResidencySet err: %s\n", err ? err.localizedDescription.UTF8String : "null");
            return;
        }
        [rs addAllocation:bA]; [rs addAllocation:bB1]; [rs addAllocation:bB2]; [rs addAllocation:bC];
        [rs addAllocation:tA]; [rs addAllocation:tB1]; [rs addAllocation:tB2]; [rs addAllocation:tC];
        [rs commit];
        [rs requestResidency];

        [argTable setResource:tA.gpuResourceID atBufferIndex:0];
        [argTable setResource:tB1.gpuResourceID atBufferIndex:1];
        [argTable setResource:tB2.gpuResourceID atBufferIndex:2];
        [argTable setResource:tC.gpuResourceID atBufferIndex:3];

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        MTL4CommitOptions* co = [MTL4CommitOptions new];
        [co addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
            (void)fb;
            dispatch_semaphore_signal(sem);
        }];
        [alloc reset];
        [cmdBuf beginCommandBufferWithAllocator:alloc];
        [cmdBuf useResidencySet:rs];
        id<MTL4ComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setArgumentTable:argTable];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cmdBuf endCommandBuffer];
        [queue commit:&cmdBuf count:1 options:co];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

        __fp16* pC = (__fp16*)bC.contents;
        float got0 = (float)pC[0];
        float gotLast = (float)pC[31 * sC + 31];
        bool ok = (fabsf(got0 - 16.0f) < 1e-2f) && (fabsf(gotLast - 16.0f) < 1e-2f);
        printf("Coop26.3 1-SG const-K chain (32x32x32): C[0]=%.3f C[31,31]=%.3f expected=16.000 => %s\n",
               got0, gotLast, ok ? "PASS" : "FAIL");
        [rs endResidency];
    }
}

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
        if (!dev) {
            fprintf(stderr, "MTLCreateSystemDefaultDevice returned nil\n");
            return 1;
        }
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
        printf("Host OS: %s\n", NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
        printf("Device : %s | apple9(M4)=%d apple10(M5/NAX)=%d metal4=%d supportsPlacementSparse=%s\n",
               dev.name.UTF8String, (int)a9, (int)a10, (int)m4, sparseStr);

        if (!m4) {
            fprintf(stderr, "Metal 4 is not supported on this device/OS (requires macOS 26.0+).\n");
            return 1;
        }

        NSError* err = nil;
        id<MTL4CommandQueue> queue = [dev newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
        id<MTL4CommandBuffer> cmdBuf = [dev newCommandBuffer];
        id<MTL4Compiler> compiler = [dev newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&err];
        if (err || !compiler) {
            fprintf(stderr, "Compiler err: %s\n", err ? err.localizedDescription.UTF8String : "null");
            return 1;
        }

        MTLCompileOptions* opts = [MTLCompileOptions new];
        const char* mslLabel = "4.0";
#if HAS_SDK_27_0
        if (@available(macOS 27.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_1;
            mslLabel = "4.1";
        } else
#endif
        {
            opts.languageVersion = MTLLanguageVersion4_0;
        }
        printf("Metal4 : MSL %s\n", mslLabel);

        MTL4ArgumentTableDescriptor* atDesc = [MTL4ArgumentTableDescriptor new];
        atDesc.maxBufferBindCount = 4;
        id<MTL4ArgumentTable> argTable = [dev newArgumentTableWithDescriptor:atDesc error:&err];
        if (err || !argTable) {
            fprintf(stderr, "ArgumentTable err: %s\n", err ? err.localizedDescription.UTF8String : "null");
            return 1;
        }

        std::vector<BenchSpec> specs;

        // Tier 1 (macOS 26.0+): FP16
        id<MTLLibrary> libF16 = compile_tier_lib(compiler, opts, @"26.0-fp16", kMetalSrcF16);
        if (libF16) {
            specs.push_back({"MPP f16*f16->f16 64x32",       @"mm_f16_64x32_nt",  libF16, MTLTensorDataTypeFloat16, MTLTensorDataTypeFloat16, MTLTensorDataTypeFloat16, 64, 32});
            specs.push_back({"MPP f16*f16->f16 64x64",       @"mm_f16_64x64_nt",  libF16, MTLTensorDataTypeFloat16, MTLTensorDataTypeFloat16, MTLTensorDataTypeFloat16, 64, 64});
            specs.push_back({"MPP f16*f16->f16 64x64 relax", @"mm_f16_64x64_rel", libF16, MTLTensorDataTypeFloat16, MTLTensorDataTypeFloat16, MTLTensorDataTypeFloat16, 64, 64});
        }

        // Tier 2 (macOS 26.1+): BF16
        if (@available(macOS 26.1, *)) {
            id<MTLLibrary> libBF16 = compile_tier_lib(compiler, opts, @"26.1-bf16", kMetalSrcBF16);
            if (libBF16) {
                specs.push_back({"MPP bf16*bf16->bf16 64x32",     @"mm_bf16_64x32_nt",  libBF16, MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, 64, 32});
                specs.push_back({"MPP bf16*bf16->bf16 64x64",     @"mm_bf16_64x64_nt",  libBF16, MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, 64, 64});
                specs.push_back({"MPP bf16*bf16->bf16 64x64 rel", @"mm_bf16_64x64_rel", libBF16, MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, MTLTensorDataTypeBFloat16, 64, 64});
            }
        } else {
            printf("[tier:26.1-bf16] skipped (requires macOS >= 26.1)\n");
        }

        // Tier 3 (macOS 26.3+): 1-SG const-K cooperative tensor chaining verification
        if (@available(macOS 26.3, *)) {
            id<MTLLibrary> libCoop = compile_tier_lib(compiler, opts, @"26.3-coop", kMetalSrcCoop263);
            run_coop_263_probe(dev, queue, alloc, cmdBuf, compiler, argTable, libCoop);
        } else {
            printf("[tier:26.3-coop] skipped (requires macOS >= 26.3)\n");
        }

        // Tier 4 (macOS 26.4+): INT8 & INT4 operands
#if HAS_SDK_26_4
        if (@available(macOS 26.4, *)) {
            id<MTLLibrary> libInt = compile_tier_lib(compiler, opts, @"26.4-int8-int4", kMetalSrcInt264);
            if (libInt) {
                specs.push_back({"MPP f16*int8->f16 64x64",    @"mm_f16_i8_64x64", libInt, MTLTensorDataTypeFloat16,  MTLTensorDataTypeInt8, MTLTensorDataTypeFloat16,  64, 64});
                specs.push_back({"MPP bf16*int8->bf16 64x64",  @"mm_bf_i8_64x64",  libInt, MTLTensorDataTypeBFloat16, MTLTensorDataTypeInt8, MTLTensorDataTypeBFloat16, 64, 64});
                specs.push_back({"MPP int8*int8->int32 64x64", @"mm_i8_i8_64x64",  libInt, MTLTensorDataTypeInt8,     MTLTensorDataTypeInt8, MTLTensorDataTypeInt32,    64, 64});
                specs.push_back({"MPP f16*int4->f16 64x64",    @"mm_f16_i4_64x64", libInt, MTLTensorDataTypeFloat16,  MTLTensorDataTypeInt4, MTLTensorDataTypeFloat16,  64, 64});
                specs.push_back({"MPP bf16*int4->bf16 64x64",  @"mm_bf_i4_64x64",  libInt, MTLTensorDataTypeBFloat16, MTLTensorDataTypeInt4, MTLTensorDataTypeBFloat16, 64, 64});
            }
        } else {
            printf("[tier:26.4-int8-int4] skipped (requires macOS >= 26.4)\n");
        }
#else
        printf("[tier:26.4-int8-int4] skipped (compiled with SDK < 26.4)\n");
#endif

        // Tier 5 (macOS 27.0+): MSL 4.1 FP8 / FP4 (E2M1) / INT2 operands
#if HAS_SDK_27_0
        if (@available(macOS 27.0, *)) {
            id<MTLLibrary> lib27 = compile_tier_lib(compiler, opts, @"27.0-fp8-fp4-int2", kMetalSrcQuant270);
            if (lib27) {
                specs.push_back({"MPP f16*fp8->f16 64x64",   @"mm_f16_fp8_64x64", lib27, MTLTensorDataTypeFloat16,  MTLTensorDataTypeMetalFloat8E4M3, MTLTensorDataTypeFloat16,  64, 64});
                specs.push_back({"MPP bf16*fp8->bf16 64x64", @"mm_bf_fp8_64x64",  lib27, MTLTensorDataTypeBFloat16, MTLTensorDataTypeMetalFloat8E4M3, MTLTensorDataTypeBFloat16, 64, 64});
                specs.push_back({"MPP f16*fp4->f16 64x64",   @"mm_f16_fp4_64x64", lib27, MTLTensorDataTypeFloat16,  MTLTensorDataTypeMetalFloat4E2M1, MTLTensorDataTypeFloat16,  64, 64});
                specs.push_back({"MPP bf16*fp4->bf16 64x64", @"mm_bf_fp4_64x64",  lib27, MTLTensorDataTypeBFloat16, MTLTensorDataTypeMetalFloat4E2M1, MTLTensorDataTypeBFloat16, 64, 64});
                specs.push_back({"MPP f16*int2->f16 64x64",  @"mm_f16_i2_64x64",  lib27, MTLTensorDataTypeFloat16,  MTLTensorDataTypeInt2,            MTLTensorDataTypeFloat16,  64, 64});
            }
        } else {
            printf("[tier:27.0-fp8-fp4-int2] skipped (requires macOS >= 27.0 / MSL 4.1)\n");
        }
#else
        printf("[tier:27.0-fp8-fp4-int2] skipped (compiled with SDK < 27.0)\n");
#endif

        NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* psos = [NSMutableDictionary new];
        for (auto& sp : specs) {
            err = nil;
            MTL4LibraryFunctionDescriptor* fnDesc = [MTL4LibraryFunctionDescriptor new];
            fnDesc.library = sp.lib;
            fnDesc.name = sp.fn;
            MTL4ComputePipelineDescriptor* pd = [MTL4ComputePipelineDescriptor new];
            pd.computeFunctionDescriptor = fnDesc;
            pd.requiredThreadsPerThreadgroup = MTLSizeMake(128, 1, 1);
            id<MTLComputePipelineState> pso = [compiler newComputePipelineStateWithDescriptor:pd compilerTaskOptions:nil error:&err];
            if (!pso || err) {
                fprintf(stderr, "[pso:%s] skipped: %s\n",
                        sp.fn.UTF8String, err ? err.localizedDescription.UTF8String : "null");
                continue;
            }
            psos[sp.fn] = pso;
        }

        std::vector<int> Ms = quick ? std::vector<int>{64, 512} : std::vector<int>{64, 128, 512, 2048};
        for (int M : Ms) {
            printf("\n=== Shape: M=%d, K=%d, N=%d (%.2f GFLOPs) ===\n", M, K, N, 2.0 * M * K * N / 1e9);
            for (auto& sp : specs) {
                @autoreleasepool {
                    id<MTLComputePipelineState> pso = psos[sp.fn];
                    if (!pso) continue;

                    id<MTLBuffer> bA = nil, bB = nil, bC = nil;
                    id<MTLTensor> tA = make_tensor(dev, M, K, sp.dtA, &bA);
                    id<MTLTensor> tB = make_tensor(dev, N, K, sp.dtB, &bB);
                    id<MTLTensor> tC = make_tensor(dev, M, N, sp.dtC, &bC);
                    if (!tA || !tB || !tC) continue;
                    fill_pattern(bA, sp.dtA, M, [tA.strides extentAtDimensionIndex:1]);
                    fill_pattern(bB, sp.dtB, N, [tB.strides extentAtDimensionIndex:1]);

                    NSError* rsErr = nil;
                    MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
                    rsd.initialCapacity = 8;
                    id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rsd error:&rsErr];
                    if (!rs || rsErr) {
                        fprintf(stderr, "[residency:%s] err: %s\n",
                                sp.fn.UTF8String, rsErr ? rsErr.localizedDescription.UTF8String : "null");
                        continue;
                    }
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
    }
    return 0;
}
