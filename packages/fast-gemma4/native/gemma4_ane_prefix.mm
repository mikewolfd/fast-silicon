#import <CoreFoundation/CoreFoundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <pthread.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fast_shared_apple.hpp"
#include "fast_shared_coreml.hpp"
#include "gemma4_ane_prefix.h"

@protocol MTLDeviceSPI_Gemma4IOSurface <NSObject>
- (nullable id<MTLBuffer>)newBufferWithIOSurface:(IOSurfaceRef)iosurface;
@end

// Fused 2-dispatch Metal shaders: read FP16 KV directly from unlocked CoreML IOSurface outputBackings,
// compute per-token absmax INT8 + FP16 scale across 32 SIMD lanes, and scatter into MLX KV_i8 slot cache.
static NSString* kGemma4BridgeScatterSrc = @R"(
#include <metal_stdlib>
using namespace metal;

struct LayerDestPtrs {
    device int8_t* k;
    device half*   ks;
    device int8_t* v;
    device half*   vs;
};

struct BridgeScatterParams {
    uint doc_slot;
    uint priv_off;
    uint valid_len;
    uint cap;
    uint is_ring;
    uint src_row_f16;
    uint chunk_s;
    uint n_kv;
};

template <uint D>
inline void quant_scatter_layer_tok(
    const device half* k_iso,
    const device half* v_iso,
    constant LayerDestPtrs* dst_tbl,
    constant BridgeScatterParams& p,
    uint lane,
    uint tok,
    uint lh_idx)
{
    if (tok >= p.valid_len) return;
    constexpr uint NV = D / 128;
    const uint n_kv = max(1u, p.n_kv);
    const uint l_idx = lh_idx / n_kv;
    const uint kv_h = lh_idx % n_kv;
    size_t src_row = (size_t)lh_idx * (size_t)p.chunk_s + (size_t)tok;
    const device half* kp = k_iso + src_row * (size_t)p.src_row_f16 + lane * 4u;
    const device half* vp = v_iso + src_row * (size_t)p.src_row_f16 + lane * 4u;

    float4 kv[NV], vv[NV];
    float kamax = 0.0f, vamax = 0.0f;
    bool bad = false;
    #pragma unroll
    for (uint r = 0; r < NV; ++r) {
        kv[r] = float4(*(const device half4*)(kp + 128u * r));
        vv[r] = float4(*(const device half4*)(vp + 128u * r));
        if (any(isnan(kv[r])) || any(isinf(kv[r])) || any(isnan(vv[r])) || any(isinf(vv[r]))) {
            bad = true;
        }
        float4 ka = abs(kv[r]);
        float4 va = abs(vv[r]);
        kamax = max(kamax, max(max(ka.x, ka.y), max(ka.z, ka.w)));
        vamax = max(vamax, max(max(va.x, va.y), max(va.z, va.w)));
    }
    bad = simd_any(bad);
    kamax = simd_max(kamax);
    vamax = simd_max(vamax);
    float ks = max(kamax * (1.0f / 127.0f), 1e-6f);
    float vs = max(vamax * (1.0f / 127.0f), 1e-6f);
    if (!isfinite(ks) || !isfinite(vs) || ks > 65504.0f || vs > 65504.0f) {
        bad = true;
    }

    uint pos = p.priv_off + tok;
    if (p.is_ring != 0u) {
        pos &= (p.cap - 1u);
    }

    size_t st_h = ((size_t)p.doc_slot * (size_t)p.cap + (size_t)pos) * (size_t)n_kv + (size_t)kv_h;
    LayerDestPtrs dst = dst_tbl[l_idx];
    if (lane == 0u) {
        dst.ks[st_h] = bad ? (half)NAN : (half)ks;
        dst.vs[st_h] = bad ? (half)NAN : (half)vs;
    }
    device int8_t* kd = dst.k + st_h * (size_t)D + lane * 4u;
    device int8_t* vd = dst.v + st_h * (size_t)D + lane * 4u;
    #pragma unroll
    for (uint r = 0; r < NV; ++r) {
        if (bad) {
            *(device char4*)(kd + 128u * r) = char4(0);
            *(device char4*)(vd + 128u * r) = char4(0);
        } else {
            float4 kq = clamp(rint(metal::precise::divide(kv[r], float4(ks))), -127.0f, 127.0f);
            float4 vq = clamp(rint(metal::precise::divide(vv[r], float4(vs))), -127.0f, 127.0f);
            *(device char4*)(kd + 128u * r) = char4((char)kq.x, (char)kq.y, (char)kq.z, (char)kq.w);
            *(device char4*)(vd + 128u * r) = char4((char)vq.x, (char)vq.y, (char)vq.z, (char)vq.w);
        }
    }
}

kernel void bridge_scatter_swa12(
    const device half* k_iso          [[buffer(0)]],
    const device half* v_iso          [[buffer(1)]],
    constant LayerDestPtrs* dst_tbl   [[buffer(2)]],
    constant BridgeScatterParams& p   [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    quant_scatter_layer_tok<256>(k_iso, v_iso, dst_tbl, p, lane, tgid.y, tgid.z);
}

kernel void bridge_scatter_full3(
    const device half* k_iso          [[buffer(0)]],
    const device half* v_iso          [[buffer(1)]],
    constant LayerDestPtrs* dst_tbl   [[buffer(2)]],
    constant BridgeScatterParams& p   [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    quant_scatter_layer_tok<512>(k_iso, v_iso, dst_tbl, p, lane, tgid.y, tgid.z);
}
)";

namespace {

struct HostLayerDestPtrs {
    uint64_t k;
    uint64_t ks;
    uint64_t v;
    uint64_t vs;
};

struct HostBridgeScatterParams {
    uint32_t doc_slot;
    uint32_t priv_off;
    uint32_t valid_len;
    uint32_t cap;
    uint32_t is_ring;
    uint32_t src_row_f16;
    uint32_t chunk_s;
    uint32_t n_kv;
};

// Unlocked CVPixelBuffer + IOSurface + MLMultiArray + zero-copy MTLBuffer wrapper.
// CRITICAL SILICON INVARIANT: NEVER access `mla.dataPointer`! Calling `.dataPointer` on a pixel-buffer-backed
// MLMultiArray permanently locks the CVPixelBuffer, breaking CoreML ANE outputBackings and cutting Metal BW 2.85x.
struct SurfBuf {
    CVPixelBufferRef pb = nullptr;
    IOSurfaceRef surf = nullptr;
    MLMultiArray* mla = nil;
    id<MTLBuffer> mtl = nil;
    size_t w = 0;
    size_t h = 0;
    size_t row_bytes = 0;
    size_t alloc_size = 0;
    void* base = nullptr;

    bool init(id<MTLDevice> dev, NSArray<NSNumber*>* shape) {
        if (!shape || shape.count == 0) return false;
        w = static_cast<size_t>(shape.lastObject.integerValue);
        h = 1;
        for (NSUInteger i = 0; i + 1 < shape.count; ++i) {
            h *= static_cast<size_t>(shape[i].integerValue);
        }
        NSDictionary* attrs = @{
            (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
            (id)kCVPixelBufferMetalCompatibilityKey : @YES,
        };
        if (CVPixelBufferCreate(
                kCFAllocatorDefault,
                w,
                h,
                kCVPixelFormatType_OneComponent16Half,
                (__bridge CFDictionaryRef)attrs,
                &pb) != kCVReturnSuccess || !pb) {
            return false;
        }
        surf = CVPixelBufferGetIOSurface(pb);
        if (!surf) {
            destroy();
            return false;
        }
        row_bytes = IOSurfaceGetBytesPerRow(surf);
        alloc_size = IOSurfaceGetAllocSize(surf);
        base = IOSurfaceGetBaseAddress(surf);
        mla = [[MLMultiArray alloc] initWithPixelBuffer:pb shape:shape];
        if (!mla) {
            destroy();
            return false;
        }
        if (dev) {
            if ([dev respondsToSelector:@selector(newBufferWithIOSurface:)]) {
                mtl = [(id<MTLDeviceSPI_Gemma4IOSurface>)dev newBufferWithIOSurface:surf];
            }
            if (!mtl && (reinterpret_cast<uintptr_t>(base) % 16384u) == 0) {
                size_t page_len = (alloc_size + 16383u) & ~size_t(16383u);
                mtl = [dev newBufferWithBytesNoCopy:base
                                             length:page_len
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
            }
        }
        return true;
    }

    void write_dense_u16(const uint16_t* src) {
        if (!src || !surf) return;
        IOSurfaceLock(surf, 0, nullptr);
        if (row_bytes == w * 2) {
            std::memcpy(base, src, w * h * 2);
        } else {
            uint8_t* dst = static_cast<uint8_t*>(base);
            for (size_t r = 0; r < h; ++r) {
                std::memcpy(dst + r * row_bytes, src + r * w, w * 2);
            }
        }
        IOSurfaceUnlock(surf, 0, nullptr);
    }

    void read_dense_u16(uint16_t* dst) const {
        if (!dst || !surf) return;
        IOSurfaceLock(surf, kIOSurfaceLockReadOnly, nullptr);
        if (row_bytes == w * 2) {
            std::memcpy(dst, base, w * h * 2);
        } else {
            const uint8_t* src = static_cast<const uint8_t*>(base);
            for (size_t r = 0; r < h; ++r) {
                std::memcpy(dst + r * w, src + r * row_bytes, w * 2);
            }
        }
        IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, nullptr);
    }

    void zero_fill() {
        if (!surf) return;
        IOSurfaceLock(surf, 0, nullptr);
        std::memset(base, 0, row_bytes * h);
        IOSurfaceUnlock(surf, 0, nullptr);
    }

    void destroy() {
        mla = nil;
        mtl = nil;
        if (pb) {
            CVPixelBufferRelease(pb);
            pb = nullptr;
            surf = nullptr;
        }
    }
};

struct PxInSlot {
    SurfBuf h_in;
    SurfBuf pli_raw;
    SurfBuf cos_swa;
    SurfBuf sin_swa;
    SurfBuf cos_full;
    SurfBuf sin_full;
    SurfBuf k_past_swa;
    SurfBuf v_past_swa;
    SurfBuf k_past_full;
    SurfBuf v_past_full;
    SurfBuf mask_swa;
    SurfBuf mask_full;
    MLDictionaryFeatureProvider* provider = nil;

    void destroy() {
        provider = nil;
        h_in.destroy();
        pli_raw.destroy();
        cos_swa.destroy();
        sin_swa.destroy();
        cos_full.destroy();
        sin_full.destroy();
        k_past_swa.destroy();
        v_past_swa.destroy();
        k_past_full.destroy();
        v_past_full.destroy();
        mask_swa.destroy();
        mask_full.destroy();
    }
};

struct PxOutSlot {
    SurfBuf ks;
    SurfBuf vs;
    SurfBuf kf;
    SurfBuf vf;
    MLPredictionOptions* opts = nil;

    void destroy() {
        opts = nil;
        ks.destroy();
        vs.destroy();
        kf.destroy();
        vf.destroy();
    }
};

struct PxSlotCacheBind {
    int B_slots = 0;
    int swa_cap = 0;
    int full_cap = 0;
    int n_swa = 0;
    int n_full = 0;
    std::vector<id<MTLBuffer>> wrapped_bufs;
    std::vector<id<MTLResource>> resources;
    id<MTLBuffer> swa_tbl_buf = nil;
    id<MTLBuffer> full_tbl_buf = nil;
};

struct PxHandle {
    MLModel* model = nil;
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> mtl_queue = nil;
    id<MTLComputePipelineState> pso_swa = nil;
    id<MTLComputePipelineState> pso_full = nil;
    dispatch_queue_t queue = nil;
    std::vector<PxInSlot> in_slots;
    std::vector<PxOutSlot> out_slots;
    int S = 256;
    int P_swa = 512;
    int P_full = 1920;
    int H_dim = 1536;
    int C_pli = 3584;
    int n_swa = 12;
    int n_full_past = 2;
    int n_full_out = 3;
    int n_kv = 1;
    bool zero_copy_ok = false;
};

static NSArray<NSNumber*>* desc_shape_or(NSDictionary<NSString*, MLFeatureDescription*>* descs, NSString* key, NSArray<NSNumber*>* fallback) {
    MLFeatureDescription* d = descs[key];
    if (d && d.multiArrayConstraint && d.multiArrayConstraint.shape.count > 0) {
        return d.multiArrayConstraint.shape;
    }
    return fallback;
}

static bool wrap_uma_ptr(id<MTLDevice> dev, uint64_t raw_addr, uint64_t nbytes, id<MTLBuffer>* out_buf, uint64_t* out_gpu_va) {
    if (!dev || raw_addr == 0 || nbytes == 0) return false;
    const uintptr_t addr = static_cast<uintptr_t>(raw_addr);
    const uintptr_t base_page = addr & ~uintptr_t(0x3FFFull);
    const size_t byte_off = static_cast<size_t>(addr - base_page);
    const size_t page_len = (byte_off + static_cast<size_t>(nbytes) + 0x3FFFull) & ~size_t(0x3FFFull);
    id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:reinterpret_cast<void*>(base_page)
                                               length:page_len
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    if (!buf) return false;
    *out_buf = buf;
    *out_gpu_va = buf.gpuAddress + static_cast<uint64_t>(byte_off);
    return true;
}

// Stitch SWA past `[n_swa * n_kv, P_swa=512, 256]` from two `[n_swa * n_kv, 256, 256]` halves directly into `dst_surf`.
static void stitch_swa_2x256(
    SurfBuf& dst_surf,
    const uint16_t* left_u16,
    size_t left_row_u16,
    const uint16_t* right_u16,
    size_t right_row_u16,
    int outer_heads,
    int S_half
) {
    IOSurfaceLock(dst_surf.surf, 0, nullptr);
    uint8_t* dst_base = static_cast<uint8_t*>(dst_surf.base);
    const size_t dst_rb = dst_surf.row_bytes;
    const size_t row_bytes_256 = 256 * sizeof(uint16_t);
    for (int l = 0; l < outer_heads; ++l) {
        const size_t dst_l_row0 = static_cast<size_t>(l) * static_cast<size_t>(2 * S_half);
        const uint16_t* l_src = left_u16 + static_cast<size_t>(l) * static_cast<size_t>(S_half) * left_row_u16;
        const uint16_t* r_src = right_u16 + static_cast<size_t>(l) * static_cast<size_t>(S_half) * right_row_u16;
        for (int t = 0; t < S_half; ++t) {
            std::memcpy(dst_base + (dst_l_row0 + t) * dst_rb, l_src + t * left_row_u16, row_bytes_256);
        }
        for (int t = 0; t < S_half; ++t) {
            std::memcpy(dst_base + (dst_l_row0 + S_half + t) * dst_rb, r_src + t * right_row_u16, row_bytes_256);
        }
    }
    IOSurfaceUnlock(dst_surf.surf, 0, nullptr);
}

// Stitch Full past `[n_full_past * n_kv, P_full, 512]` from `sys [lcp, 512]` followed by `n_chunks` earlier chunks
// `[S, 512]` in order, so buffer row == absolute position. Rows past the last chunk are zeroed (masked by `mf`).
static void stitch_full_past(
    SurfBuf& dst_surf,
    const uint16_t* sys_u16,
    int lcp,
    const uint16_t* const* chunk_u16,
    const size_t* chunk_row_u16,
    int n_chunks,
    int outer_heads,
    int P_full,
    int S_chunk
) {
    IOSurfaceLock(dst_surf.surf, 0, nullptr);
    uint8_t* dst_base = static_cast<uint8_t*>(dst_surf.base);
    const size_t dst_rb = dst_surf.row_bytes;
    const size_t row_bytes_512 = 512 * sizeof(uint16_t);
    for (int l = 0; l < outer_heads; ++l) {
        const size_t row0 = static_cast<size_t>(l) * static_cast<size_t>(P_full);
        int cur = 0;
        if (lcp > 0 && sys_u16) {
            const uint16_t* s_ptr = sys_u16 + static_cast<size_t>(l) * static_cast<size_t>(lcp) * 512;
            for (int t = 0; t < lcp && cur < P_full; ++t, ++cur) {
                std::memcpy(dst_base + (row0 + cur) * dst_rb, s_ptr + static_cast<size_t>(t) * 512, row_bytes_512);
            }
        }
        for (int c = 0; c < n_chunks; ++c) {
            const size_t row_u16 = chunk_row_u16[c];
            const uint16_t* c_ptr = chunk_u16[c] + static_cast<size_t>(l) * static_cast<size_t>(S_chunk) * row_u16;
            for (int t = 0; t < S_chunk && cur < P_full; ++t, ++cur) {
                std::memcpy(dst_base + (row0 + cur) * dst_rb, c_ptr + static_cast<size_t>(t) * row_u16, row_bytes_512);
            }
        }
        if (cur < P_full) {
            std::memset(dst_base + (row0 + cur) * dst_rb, 0, static_cast<size_t>(P_full - cur) * dst_rb);
        }
    }
    IOSurfaceUnlock(dst_surf.surf, 0, nullptr);
}

}  // namespace

extern "C" {

int gemma4_px_inspect_plan_ex(
    const char* mlmodelc_path,
    int compute_mode,
    int* out_ane,
    int* out_cpu_runtime,
    int* out_cpu_constexpr,
    int* out_gpu
) {
    @autoreleasepool {
        if (!mlmodelc_path) return -1;
        NSURL* url = [[NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]] URLByResolvingSymlinksInPath];
        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = fast_shared::resolve_compute_units(static_cast<fast_shared::ComputeMode>(compute_mode));

        __block int ane_ops = 0;
        __block int cpu_rt_ops = 0;
        __block int cpu_ce_ops = 0;
        __block int gpu_ops = 0;
        __block int failed = 0;
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* plan, NSError* err) {
            if (err || !plan || !plan.modelStructure.program) {
                failed = 1;
            } else {
                MLModelStructureProgramFunction* fn = plan.modelStructure.program.functions[@"main"];
                for (MLModelStructureProgramOperation* op in fn.block.operations) {
                    NSString* opName = op.operatorName;
                    if ([opName isEqualToString:@"const"]) continue;
                    id dev = [plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
                    if ([dev isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
                        ane_ops++;
                    } else if ([dev isKindOfClass:[MLGPUComputeDevice class]]) {
                        gpu_ops++;
                    } else {
                        if ([opName containsString:@"constexpr_"]) cpu_ce_ops++;
                        else cpu_rt_ops++;
                    }
                }
            }
            dispatch_semaphore_signal(sem);
        }];
        // 30-second timeout guard prevents unhandled E5RT _ANECompiler C++ STL exceptions from hanging MLComputePlan under contention.
        long timed_out = dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 30LL * NSEC_PER_SEC));
        if (timed_out != 0) {
            fprintf(stderr, "[MLComputePlan] TIMEOUT (>30s) inspecting %s\n", mlmodelc_path);
            failed = 1;
        }
        if (out_ane) *out_ane = ane_ops;
        if (out_cpu_runtime) *out_cpu_runtime = cpu_rt_ops;
        if (out_cpu_constexpr) *out_cpu_constexpr = cpu_ce_ops;
        if (out_gpu) *out_gpu = gpu_ops;
        if (failed) return -1;
        return ane_ops;
    }
}

int gemma4_px_inspect_plan(const char* mlmodelc_path, int compute_mode) {
    int ane_ops = 0, cpu_rt = 0, cpu_ce = 0, gpu_ops = 0;
    int rc = gemma4_px_inspect_plan_ex(mlmodelc_path, compute_mode, &ane_ops, &cpu_rt, &cpu_ce, &gpu_ops);
    if (rc < 0) return -1;
    @autoreleasepool {
        NSString* p = [NSString stringWithUTF8String:mlmodelc_path];
        printf("[MLComputePlan] %s (mode=%d) -> ANE=%d GPU=%d CPU_runtime=%d CPU_constexpr=%d\n",
               p.lastPathComponent.UTF8String, compute_mode, ane_ops, gpu_ops, cpu_rt, cpu_ce);
    }
    return ane_ops;
}

void* gemma4_px_load_ex(const char* mlmodelc_path, int S, int P_swa, int P_full, int max_pipe, int out_ring_slots) {
    @autoreleasepool {
        if (!mlmodelc_path || S <= 0 || P_swa < 0 || P_full < 0 || max_pipe <= 0) return nullptr;
        int n_ring = std::max(4, out_ring_slots);
        NSURL* url = [[NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelc_path]] URLByResolvingSymlinksInPath];
        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        NSError* err = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model || err) {
            fprintf(stderr, "gemma4_px_load(%s) error: %s\n",
                    mlmodelc_path, err ? err.localizedDescription.UTF8String : "unknown");
            return nullptr;
        }

        auto* h = new PxHandle();
        h->model = model;
        h->dev = MTLCreateSystemDefaultDevice();
        if (h->dev) {
            h->mtl_queue = [h->dev newCommandQueue];
            MTLCompileOptions* copts = [[MTLCompileOptions alloc] init];
            id<MTLLibrary> lib = [h->dev newLibraryWithSource:kGemma4BridgeScatterSrc options:copts error:&err];
            if (lib && !err) {
                h->pso_swa = [h->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bridge_scatter_swa12"] error:&err];
                h->pso_full = [h->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bridge_scatter_full3"] error:&err];
            }
            err = nil;
        }

        NSDictionary<NSString*, MLFeatureDescription*>* in_d = model.modelDescription.inputDescriptionsByName;
        NSDictionary<NSString*, MLFeatureDescription*>* out_d = model.modelDescription.outputDescriptionsByName;
        NSArray<NSNumber*>* sh_h    = desc_shape_or(in_d,  @"h_in",        @[@1, @1536, @1, @(S)]);
        NSArray<NSNumber*>* sh_pli  = desc_shape_or(in_d,  @"pli_raw",     @[@1, @3584, @1, @(S)]);
        NSArray<NSNumber*>* sh_cs   = desc_shape_or(in_d,  @"cos_swa",     @[@1, @1, @(S), @256]);
        NSArray<NSNumber*>* sh_ss   = desc_shape_or(in_d,  @"sin_swa",     @[@1, @1, @(S), @256]);
        NSArray<NSNumber*>* sh_cf   = desc_shape_or(in_d,  @"cos_full",    @[@1, @1, @(S), @512]);
        NSArray<NSNumber*>* sh_sf   = desc_shape_or(in_d,  @"sin_full",    @[@1, @1, @(S), @512]);
        NSArray<NSNumber*>* sh_kps  = desc_shape_or(in_d,  @"k_past_swa",  @[@12, @1, @(P_swa), @256]);
        NSArray<NSNumber*>* sh_vps  = desc_shape_or(in_d,  @"v_past_swa",  @[@12, @1, @(P_swa), @256]);
        NSArray<NSNumber*>* sh_kpf  = desc_shape_or(in_d,  @"k_past_full", @[@2, @1, @(P_full), @512]);
        NSArray<NSNumber*>* sh_vpf  = desc_shape_or(in_d,  @"v_past_full", @[@2, @1, @(P_full), @512]);
        NSArray<NSNumber*>* sh_ms   = desc_shape_or(in_d,  @"mask_swa",    @[@1, @1, @(S), @(P_swa + S)]);
        NSArray<NSNumber*>* sh_mf   = desc_shape_or(in_d,  @"mask_full",   @[@1, @1, @(S), @(P_full + S)]);
        NSArray<NSNumber*>* sh_oks  = desc_shape_or(out_d, @"k_swa",       @[@12, @1, @(S), @256]);
        NSArray<NSNumber*>* sh_ovs  = desc_shape_or(out_d, @"v_swa",       @[@12, @1, @(S), @256]);
        NSArray<NSNumber*>* sh_okf  = desc_shape_or(out_d, @"k_full",      @[@3, @1, @(S), @512]);
        NSArray<NSNumber*>* sh_ovf  = desc_shape_or(out_d, @"v_full",      @[@3, @1, @(S), @512]);

        h->S = S;
        h->P_swa = P_swa;
        h->P_full = P_full;
        h->H_dim = static_cast<int>(sh_h[1].integerValue);
        h->C_pli = static_cast<int>(sh_pli[1].integerValue);
        h->n_swa = static_cast<int>(sh_oks[0].integerValue);
        h->n_kv = static_cast<int>(sh_oks[1].integerValue);
        h->n_full_past = static_cast<int>(sh_kpf[0].integerValue);
        h->n_full_out = static_cast<int>(sh_okf[0].integerValue);
        h->queue = dispatch_queue_create("gemma4.ane.px.queue", DISPATCH_QUEUE_CONCURRENT);

        h->in_slots.resize(max_pipe);
        for (int p = 0; p < max_pipe; ++p) {
            PxInSlot& sl = h->in_slots[p];
            if (!sl.h_in.init(h->dev, sh_h) ||
                !sl.pli_raw.init(h->dev, sh_pli) ||
                !sl.cos_swa.init(h->dev, sh_cs) ||
                !sl.sin_swa.init(h->dev, sh_ss) ||
                !sl.cos_full.init(h->dev, sh_cf) ||
                !sl.sin_full.init(h->dev, sh_sf) ||
                !sl.k_past_swa.init(h->dev, sh_kps) ||
                !sl.v_past_swa.init(h->dev, sh_vps) ||
                !sl.k_past_full.init(h->dev, sh_kpf) ||
                !sl.v_past_full.init(h->dev, sh_vpf) ||
                !sl.mask_swa.init(h->dev, sh_ms) ||
                !sl.mask_full.init(h->dev, sh_mf)) {
                gemma4_px_free(h);
                return nullptr;
            }
            sl.h_in.zero_fill();
            sl.pli_raw.zero_fill();
            sl.cos_swa.zero_fill();
            sl.sin_swa.zero_fill();
            sl.cos_full.zero_fill();
            sl.sin_full.zero_fill();
            sl.k_past_swa.zero_fill();
            sl.v_past_swa.zero_fill();
            sl.k_past_full.zero_fill();
            sl.v_past_full.zero_fill();
            sl.mask_swa.zero_fill();
            sl.mask_full.zero_fill();

            NSDictionary* dict = @{
                @"h_in" : sl.h_in.mla,
                @"pli_raw" : sl.pli_raw.mla,
                @"cos_swa" : sl.cos_swa.mla,
                @"sin_swa" : sl.sin_swa.mla,
                @"cos_full" : sl.cos_full.mla,
                @"sin_full" : sl.sin_full.mla,
                @"k_past_swa" : sl.k_past_swa.mla,
                @"v_past_swa" : sl.v_past_swa.mla,
                @"k_past_full" : sl.k_past_full.mla,
                @"v_past_full" : sl.v_past_full.mla,
                @"mask_swa" : sl.mask_swa.mla,
                @"mask_full" : sl.mask_full.mla,
            };
            sl.provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:&err];
            if (!sl.provider || err) {
                gemma4_px_free(h);
                return nullptr;
            }
        }

        h->out_slots.resize(n_ring);
        for (int r = 0; r < n_ring; ++r) {
            PxOutSlot& osl = h->out_slots[r];
            if (!osl.ks.init(h->dev, sh_oks) ||
                !osl.vs.init(h->dev, sh_ovs) ||
                !osl.kf.init(h->dev, sh_okf) ||
                !osl.vf.init(h->dev, sh_ovf)) {
                gemma4_px_free(h);
                return nullptr;
            }
            osl.ks.zero_fill();
            osl.vs.zero_fill();
            osl.kf.zero_fill();
            osl.vf.zero_fill();
            osl.opts = [[MLPredictionOptions alloc] init];
            osl.opts.outputBackings = @{
                @"k_swa" : osl.ks.mla,
                @"v_swa" : osl.vs.mla,
                @"k_full" : osl.kf.mla,
                @"v_full" : osl.vf.mla,
            };
        }

        // Warmup 1 call with unlocked IOSurface outputBackings and verify zero-copy pointer identity!
        id<MLFeatureProvider> out = [h->model predictionFromFeatures:h->in_slots[0].provider
                                                             options:h->out_slots[0].opts
                                                               error:&err];
        if (!out || err) {
            fprintf(stderr, "gemma4_px_load warmup(%s) error: %s\n",
                    mlmodelc_path, err ? err.localizedDescription.UTF8String : "unknown");
            gemma4_px_free(h);
            return nullptr;
        }
        MLMultiArray* out_ks = [out featureValueForName:@"k_swa"].multiArrayValue;
        MLMultiArray* out_kf = [out featureValueForName:@"k_full"].multiArrayValue;
        h->zero_copy_ok = (out_ks.pixelBuffer == h->out_slots[0].ks.pb) &&
                          (out_kf.pixelBuffer == h->out_slots[0].kf.pb) &&
                          (h->out_slots[0].ks.mtl != nil) &&
                          (h->out_slots[0].kf.mtl != nil) &&
                          (h->pso_swa != nil) &&
                          (h->pso_full != nil);
        return static_cast<void*>(h);
    }
}

void* gemma4_px_load(const char* mlmodelc_path, int S, int P_swa, int P_full, int max_pipe) {
    return gemma4_px_load_ex(mlmodelc_path, S, P_swa, P_full, max_pipe, 8);
}

int gemma4_px_is_zero_copy(void* handle) {
    if (!handle) return 0;
    return static_cast<PxHandle*>(handle)->zero_copy_ok ? 1 : 0;
}

int gemma4_px_out_ring_size(void* handle) {
    if (!handle) return 0;
    return static_cast<int>(static_cast<PxHandle*>(handle)->out_slots.size());
}

int gemma4_px_get_dims(
    void* handle,
    int* out_H,
    int* out_C_pli,
    int* out_n_swa,
    int* out_n_full_past,
    int* out_n_full_out,
    int* out_n_kv
) {
    if (!handle) return -1;
    const auto* h = static_cast<const PxHandle*>(handle);
    if (out_H) *out_H = h->H_dim;
    if (out_C_pli) *out_C_pli = h->C_pli;
    if (out_n_swa) *out_n_swa = h->n_swa;
    if (out_n_full_past) *out_n_full_past = h->n_full_past;
    if (out_n_full_out) *out_n_full_out = h->n_full_out;
    if (out_n_kv) *out_n_kv = h->n_kv;
    return 0;
}

void gemma4_px_free(void* handle) {
    if (!handle) return;
    @autoreleasepool {
        auto* h = static_cast<PxHandle*>(handle);
        for (auto& sl : h->in_slots) sl.destroy();
        for (auto& osl : h->out_slots) osl.destroy();
        delete h;
    }
}

double gemma4_px_run_batch(
    void* handle,
    int B,
    const uint16_t* h_all,
    const uint16_t* pli_all,
    const uint16_t* cos_s_all,
    const uint16_t* sin_s_all,
    const uint16_t* cos_f_all,
    const uint16_t* sin_f_all,
    const uint16_t* kp_s_all,
    const uint16_t* vp_s_all,
    const uint16_t* kp_f_all,
    const uint16_t* vp_f_all,
    const uint16_t* ms_all,
    const uint16_t* mf_all,
    uint16_t* k_swa_out,
    uint16_t* v_swa_out,
    uint16_t* k_full_out,
    uint16_t* v_full_out,
    int shared_past_kv,
    int shared_geom,
    int pipe_depth
) {
    @autoreleasepool {
        auto* h = static_cast<PxHandle*>(handle);
        if (!h || B <= 0) return -1.0;

        const int P = std::max(1, std::min(pipe_depth, static_cast<int>(h->in_slots.size())));
        const int R = static_cast<int>(h->out_slots.size());
        const int S = h->S;
        const int P_swa = h->P_swa;
        const int P_full = h->P_full;

        const size_t sz_h   = static_cast<size_t>(h->H_dim) * S;
        const size_t sz_pli = static_cast<size_t>(h->C_pli) * S;
        const size_t sz_cs  = static_cast<size_t>(S) * 256;
        const size_t sz_cf  = static_cast<size_t>(S) * 512;
        const size_t sz_kps = static_cast<size_t>(h->n_swa) * h->n_kv * P_swa * 256;
        const size_t sz_kpf = static_cast<size_t>(h->n_full_past) * h->n_kv * P_full * 512;
        const size_t sz_ms  = static_cast<size_t>(S) * (P_swa + S);
        const size_t sz_mf  = static_cast<size_t>(S) * (P_full + S);
        const size_t sz_oks = static_cast<size_t>(h->n_swa) * h->n_kv * S * 256;
        const size_t sz_okf = static_cast<size_t>(h->n_full_out) * h->n_kv * S * 512;

        if (shared_past_kv) {
            for (int p = 0; p < P; ++p) {
                h->in_slots[p].k_past_swa.write_dense_u16(kp_s_all);
                h->in_slots[p].v_past_swa.write_dense_u16(vp_s_all);
                h->in_slots[p].k_past_full.write_dense_u16(kp_f_all);
                h->in_slots[p].v_past_full.write_dense_u16(vp_f_all);
            }
        }
        if (shared_geom) {
            for (int p = 0; p < P; ++p) {
                h->in_slots[p].cos_swa.write_dense_u16(cos_s_all);
                h->in_slots[p].sin_swa.write_dense_u16(sin_s_all);
                h->in_slots[p].cos_full.write_dense_u16(cos_f_all);
                h->in_slots[p].sin_full.write_dense_u16(sin_f_all);
                h->in_slots[p].mask_swa.write_dense_u16(ms_all);
                h->in_slots[p].mask_full.write_dense_u16(mf_all);
            }
        }

        const double t0 = fast_shared::now_sec();
        dispatch_group_t grp = dispatch_group_create();
        __block int failed = 0;
        for (int p = 0; p < P; ++p) {
            dispatch_group_async(grp, h->queue, ^{
                @autoreleasepool {
                    PxInSlot& sl = h->in_slots[p];
                    PxOutSlot& osl = h->out_slots[p % R];
                    for (int b = p; b < B; b += P) {
                        sl.h_in.write_dense_u16(h_all + static_cast<size_t>(b) * sz_h);
                        sl.pli_raw.write_dense_u16(pli_all + static_cast<size_t>(b) * sz_pli);
                        if (!shared_geom) {
                            sl.cos_swa.write_dense_u16(cos_s_all + static_cast<size_t>(b) * sz_cs);
                            sl.sin_swa.write_dense_u16(sin_s_all + static_cast<size_t>(b) * sz_cs);
                            sl.cos_full.write_dense_u16(cos_f_all + static_cast<size_t>(b) * sz_cf);
                            sl.sin_full.write_dense_u16(sin_f_all + static_cast<size_t>(b) * sz_cf);
                            sl.mask_swa.write_dense_u16(ms_all + static_cast<size_t>(b) * sz_ms);
                            sl.mask_full.write_dense_u16(mf_all + static_cast<size_t>(b) * sz_mf);
                        }
                        if (!shared_past_kv) {
                            sl.k_past_swa.write_dense_u16(kp_s_all + static_cast<size_t>(b) * sz_kps);
                            sl.v_past_swa.write_dense_u16(vp_s_all + static_cast<size_t>(b) * sz_kps);
                            sl.k_past_full.write_dense_u16(kp_f_all + static_cast<size_t>(b) * sz_kpf);
                            sl.v_past_full.write_dense_u16(vp_f_all + static_cast<size_t>(b) * sz_kpf);
                        }
                        NSError* err = nil;
                        id<MLFeatureProvider> out = [h->model predictionFromFeatures:sl.provider
                                                                             options:osl.opts
                                                                               error:&err];
                        if (!out || err) {
                            failed = 1;
                            return;
                        }
                        if (k_swa_out) osl.ks.read_dense_u16(k_swa_out + static_cast<size_t>(b) * sz_oks);
                        if (v_swa_out) osl.vs.read_dense_u16(v_swa_out + static_cast<size_t>(b) * sz_oks);
                        if (k_full_out) osl.kf.read_dense_u16(k_full_out + static_cast<size_t>(b) * sz_okf);
                        if (v_full_out) osl.vf.read_dense_u16(v_full_out + static_cast<size_t>(b) * sz_okf);
                    }
                }
            });
        }
        dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
        return failed ? -1.0 : (fast_shared::now_sec() - t0);
    }
}

double gemma4_px_run_chunk_into_ring(
    void* handle,
    int B,
    int chunk_idx,
    int lcp,
    const uint16_t* h_all,
    const uint16_t* pli_all,
    const uint16_t* cos_s,
    const uint16_t* sin_s,
    const uint16_t* cos_f,
    const uint16_t* sin_f,
    const uint16_t* sys_swa_k,
    const uint16_t* sys_swa_v,
    const uint16_t* sys_full_k,
    const uint16_t* sys_full_v,
    const uint16_t* ms,
    const uint16_t* mf,
    const int32_t* prev_ring_ids,
    const int32_t* out_ring_ids,
    int pipe_depth
) {
    @autoreleasepool {
        auto* h = static_cast<PxHandle*>(handle);
        if (!h || B <= 0 || !out_ring_ids || chunk_idx < 0 || lcp < 0) return -1.0;
        // The chunk starts at absolute position `lcp + chunk_idx * S`; everything before it must fit the Full past.
        if (static_cast<long>(lcp) + static_cast<long>(chunk_idx) * h->S > h->P_full) return -1.0;
        if (chunk_idx >= 1 && !prev_ring_ids) return -1.0;
        if (chunk_idx <= 1 && (!sys_swa_k || !sys_swa_v)) return -1.0;

        const int R = static_cast<int>(h->out_slots.size());
        for (int b = 0; b < B; ++b) {
            if (out_ring_ids[b] < 0 || out_ring_ids[b] >= R) return -1.0;
            for (int j = 0; j < chunk_idx; ++j) {
                const int32_t r = prev_ring_ids[static_cast<size_t>(j) * B + b];
                if (r < 0 || r >= R || r == out_ring_ids[b]) return -1.0;
            }
        }

        const int P = std::max(1, std::min(pipe_depth, static_cast<int>(h->in_slots.size())));
        const int S = h->S;
        const size_t sz_h   = static_cast<size_t>(h->H_dim) * S;
        const size_t sz_pli = static_cast<size_t>(h->C_pli) * S;
        const int outer_swa = h->n_swa * h->n_kv;
        const int outer_fp  = h->n_full_past * h->n_kv;

        for (int p = 0; p < P; ++p) {
            PxInSlot& sl = h->in_slots[p];
            sl.cos_swa.write_dense_u16(cos_s);
            sl.sin_swa.write_dense_u16(sin_s);
            sl.cos_full.write_dense_u16(cos_f);
            sl.sin_full.write_dense_u16(sin_f);
            sl.mask_swa.write_dense_u16(ms);
            sl.mask_full.write_dense_u16(mf);
            if (chunk_idx == 0) {
                sl.k_past_swa.write_dense_u16(sys_swa_k);
                sl.v_past_swa.write_dense_u16(sys_swa_v);
                stitch_full_past(sl.k_past_full, sys_full_k, lcp, nullptr, nullptr, 0, outer_fp, h->P_full, S);
                stitch_full_past(sl.v_past_full, sys_full_v, lcp, nullptr, nullptr, 0, outer_fp, h->P_full, S);
            }
        }

        const double t0 = fast_shared::now_sec();
        dispatch_group_t grp = dispatch_group_create();
        __block int failed = 0;
        for (int p = 0; p < P; ++p) {
            dispatch_group_async(grp, h->queue, ^{
                @autoreleasepool {
                    PxInSlot& sl = h->in_slots[p];
                    std::vector<const PxOutSlot*> prev(static_cast<size_t>(chunk_idx));
                    std::vector<const uint16_t*> kf_src(prev.size()), vf_src(prev.size());
                    std::vector<size_t> kf_row(prev.size()), vf_row(prev.size());
                    for (int b = p; b < B; b += P) {
                        sl.h_in.write_dense_u16(h_all + static_cast<size_t>(b) * sz_h);
                        sl.pli_raw.write_dense_u16(pli_all + static_cast<size_t>(b) * sz_pli);

                        if (chunk_idx >= 1) {
                            for (int j = 0; j < chunk_idx; ++j) {
                                prev[j] = &h->out_slots[prev_ring_ids[static_cast<size_t>(j) * B + b]];
                            }
                            // Sliding window (512) = the two chunks before this one; Full = every earlier chunk.
                            const int swa_first = std::max(0, chunk_idx - 2);
                            for (int j = 0; j < chunk_idx; ++j) {
                                if (j >= swa_first) {
                                    IOSurfaceLock(prev[j]->ks.surf, kIOSurfaceLockReadOnly, nullptr);
                                    IOSurfaceLock(prev[j]->vs.surf, kIOSurfaceLockReadOnly, nullptr);
                                }
                                IOSurfaceLock(prev[j]->kf.surf, kIOSurfaceLockReadOnly, nullptr);
                                IOSurfaceLock(prev[j]->vf.surf, kIOSurfaceLockReadOnly, nullptr);
                                kf_src[j] = static_cast<const uint16_t*>(prev[j]->kf.base);
                                vf_src[j] = static_cast<const uint16_t*>(prev[j]->vf.base);
                                kf_row[j] = prev[j]->kf.row_bytes / 2;
                                vf_row[j] = prev[j]->vf.row_bytes / 2;
                            }
                            const PxOutSlot& last = *prev[chunk_idx - 1];
                            if (chunk_idx == 1) {
                                // Left half is the last 256 positions of the shared prefix (zero-padded when shorter).
                                stitch_swa_2x256(
                                    sl.k_past_swa, sys_swa_k, 256,
                                    static_cast<const uint16_t*>(last.ks.base), last.ks.row_bytes / 2, outer_swa, S);
                                stitch_swa_2x256(
                                    sl.v_past_swa, sys_swa_v, 256,
                                    static_cast<const uint16_t*>(last.vs.base), last.vs.row_bytes / 2, outer_swa, S);
                            } else {
                                const PxOutSlot& older = *prev[chunk_idx - 2];
                                stitch_swa_2x256(
                                    sl.k_past_swa,
                                    static_cast<const uint16_t*>(older.ks.base), older.ks.row_bytes / 2,
                                    static_cast<const uint16_t*>(last.ks.base), last.ks.row_bytes / 2, outer_swa, S);
                                stitch_swa_2x256(
                                    sl.v_past_swa,
                                    static_cast<const uint16_t*>(older.vs.base), older.vs.row_bytes / 2,
                                    static_cast<const uint16_t*>(last.vs.base), last.vs.row_bytes / 2, outer_swa, S);
                            }
                            stitch_full_past(
                                sl.k_past_full, sys_full_k, lcp, kf_src.data(), kf_row.data(), chunk_idx,
                                outer_fp, h->P_full, S);
                            stitch_full_past(
                                sl.v_past_full, sys_full_v, lcp, vf_src.data(), vf_row.data(), chunk_idx,
                                outer_fp, h->P_full, S);
                            for (int j = chunk_idx - 1; j >= 0; --j) {
                                IOSurfaceUnlock(prev[j]->vf.surf, kIOSurfaceLockReadOnly, nullptr);
                                IOSurfaceUnlock(prev[j]->kf.surf, kIOSurfaceLockReadOnly, nullptr);
                                if (j >= swa_first) {
                                    IOSurfaceUnlock(prev[j]->vs.surf, kIOSurfaceLockReadOnly, nullptr);
                                    IOSurfaceUnlock(prev[j]->ks.surf, kIOSurfaceLockReadOnly, nullptr);
                                }
                            }
                        }

                        PxOutSlot& osl = h->out_slots[out_ring_ids[b]];
                        NSError* err = nil;
                        id<MLFeatureProvider> out = [h->model predictionFromFeatures:sl.provider
                                                                             options:osl.opts
                                                                               error:&err];
                        if (!out || err) {
                            failed = 1;
                            return;
                        }
                    }
                }
            });
        }
        dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
        return failed ? -1.0 : (fast_shared::now_sec() - t0);
    }
}

void* gemma4_px_bind_slot_cache_i8(
    void* handle,
    int B_slots,
    int swa_cap,
    int full_cap,
    int n_swa,
    const uint64_t* swa_ptrs,
    const uint64_t* swa_nbytes,
    int n_full,
    const uint64_t* full_ptrs,
    const uint64_t* full_nbytes
) {
    @autoreleasepool {
        if (B_slots <= 0 || swa_cap <= 0 || full_cap <= 0 || n_swa <= 0 || n_full <= 0) {
            return nullptr;
        }
        // `quant_scatter_layer_tok` wraps ring positions with `pos &= (p.cap - 1u)`.
        if ((swa_cap & (swa_cap - 1)) != 0) {
            fprintf(stderr, "gemma4_px_bind_slot_cache_i8 error: swa_cap=%d must be a power of 2\n", swa_cap);
            return nullptr;
        }
        auto* h = static_cast<PxHandle*>(handle);
        if (!h || !h->dev || !swa_ptrs || !swa_nbytes || !full_ptrs || !full_nbytes) {
            return nullptr;
        }
        auto* bind = new PxSlotCacheBind();
        bind->B_slots = B_slots;
        bind->swa_cap = swa_cap;
        bind->full_cap = full_cap;
        bind->n_swa = n_swa;
        bind->n_full = n_full;

        std::vector<HostLayerDestPtrs> swa_tbl(n_swa);
        for (int l = 0; l < n_swa; ++l) {
            id<MTLBuffer> bk = nil, bks = nil, bv = nil, bvs = nil;
            if (!wrap_uma_ptr(h->dev, swa_ptrs[l * 4 + 0], swa_nbytes[l * 4 + 0], &bk,  &swa_tbl[l].k)  ||
                !wrap_uma_ptr(h->dev, swa_ptrs[l * 4 + 1], swa_nbytes[l * 4 + 1], &bks, &swa_tbl[l].ks) ||
                !wrap_uma_ptr(h->dev, swa_ptrs[l * 4 + 2], swa_nbytes[l * 4 + 2], &bv,  &swa_tbl[l].v)  ||
                !wrap_uma_ptr(h->dev, swa_ptrs[l * 4 + 3], swa_nbytes[l * 4 + 3], &bvs, &swa_tbl[l].vs)) {
                delete bind;
                return nullptr;
            }
            bind->wrapped_bufs.insert(bind->wrapped_bufs.end(), {bk, bks, bv, bvs});
            bind->resources.insert(bind->resources.end(), {bk, bks, bv, bvs});
        }

        std::vector<HostLayerDestPtrs> full_tbl(n_full);
        for (int l = 0; l < n_full; ++l) {
            id<MTLBuffer> bk = nil, bks = nil, bv = nil, bvs = nil;
            if (!wrap_uma_ptr(h->dev, full_ptrs[l * 4 + 0], full_nbytes[l * 4 + 0], &bk,  &full_tbl[l].k)  ||
                !wrap_uma_ptr(h->dev, full_ptrs[l * 4 + 1], full_nbytes[l * 4 + 1], &bks, &full_tbl[l].ks) ||
                !wrap_uma_ptr(h->dev, full_ptrs[l * 4 + 2], full_nbytes[l * 4 + 2], &bv,  &full_tbl[l].v)  ||
                !wrap_uma_ptr(h->dev, full_ptrs[l * 4 + 3], full_nbytes[l * 4 + 3], &bvs, &full_tbl[l].vs)) {
                delete bind;
                return nullptr;
            }
            bind->wrapped_bufs.insert(bind->wrapped_bufs.end(), {bk, bks, bv, bvs});
            bind->resources.insert(bind->resources.end(), {bk, bks, bv, bvs});
        }

        bind->swa_tbl_buf = [h->dev newBufferWithBytes:swa_tbl.data()
                                                length:swa_tbl.size() * sizeof(HostLayerDestPtrs)
                                               options:MTLResourceStorageModeShared];
        bind->full_tbl_buf = [h->dev newBufferWithBytes:full_tbl.data()
                                                 length:full_tbl.size() * sizeof(HostLayerDestPtrs)
                                                options:MTLResourceStorageModeShared];
        if (!bind->swa_tbl_buf || !bind->full_tbl_buf) {
            delete bind;
            return nullptr;
        }
        return static_cast<void*>(bind);
    }
}

void gemma4_px_unbind_slot_cache_i8(void* bind_handle) {
    if (!bind_handle) return;
    @autoreleasepool {
        auto* bind = static_cast<PxSlotCacheBind*>(bind_handle);
        bind->resources.clear();
        bind->wrapped_bufs.clear();
        bind->swa_tbl_buf = nil;
        bind->full_tbl_buf = nil;
        delete bind;
    }
}

double gemma4_px_scatter_ring_kv_i8(
    void* handle,
    void* bind_handle,
    int n_items,
    const int32_t* ring_slot_ids,
    const int32_t* doc_slots,
    const int32_t* priv_offs,
    const int32_t* valid_lens
) {
    @autoreleasepool {
        auto* h = static_cast<PxHandle*>(handle);
        auto* bind = static_cast<PxSlotCacheBind*>(bind_handle);
        if (!h || !bind || !h->mtl_queue || !h->pso_swa || !h->pso_full || n_items <= 0) return -1.0;

        const int R = static_cast<int>(h->out_slots.size());
        id<MTLCommandBuffer> cb = [h->mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return -1.0;

        [enc useResources:bind->resources.data()
                    count:bind->resources.size()
                    usage:MTLResourceUsageWrite];

        for (int i = 0; i < n_items; ++i) {
            const int r_id = ring_slot_ids[i];
            const int d_slot = doc_slots[i];
            const int p_off = priv_offs[i];
            const int v_len = valid_lens[i];
            if (r_id < 0 || r_id >= R || d_slot < 0 || d_slot >= bind->B_slots || v_len <= 0) continue;
            const uint32_t eff_vl = static_cast<uint32_t>(std::min(v_len, h->S));
            const PxOutSlot& osl = h->out_slots[r_id];
            if (!osl.ks.mtl || !osl.vs.mtl || !osl.kf.mtl || !osl.vf.mtl) {
                [enc endEncoding];
                return -1.0;
            }

            const uint32_t eff_n_kv = static_cast<uint32_t>(std::max(1, h->n_kv));
            HostBridgeScatterParams ps = {
                static_cast<uint32_t>(d_slot),
                static_cast<uint32_t>(p_off),
                eff_vl,
                static_cast<uint32_t>(bind->swa_cap),
                1u,
                static_cast<uint32_t>(osl.ks.row_bytes / 2),
                static_cast<uint32_t>(h->S),
                eff_n_kv,
            };
            [enc setComputePipelineState:h->pso_swa];
            [enc setBuffer:osl.ks.mtl offset:0 atIndex:0];
            [enc setBuffer:osl.vs.mtl offset:0 atIndex:1];
            [enc setBuffer:bind->swa_tbl_buf offset:0 atIndex:2];
            [enc setBytes:&ps length:sizeof(ps) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, eff_vl, static_cast<NSUInteger>(bind->n_swa) * eff_n_kv)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            HostBridgeScatterParams pf = {
                static_cast<uint32_t>(d_slot),
                static_cast<uint32_t>(p_off),
                eff_vl,
                static_cast<uint32_t>(bind->full_cap),
                0u,
                static_cast<uint32_t>(osl.kf.row_bytes / 2),
                static_cast<uint32_t>(h->S),
                eff_n_kv,
            };
            [enc setComputePipelineState:h->pso_full];
            [enc setBuffer:osl.kf.mtl offset:0 atIndex:0];
            [enc setBuffer:osl.vf.mtl offset:0 atIndex:1];
            [enc setBuffer:bind->full_tbl_buf offset:0 atIndex:2];
            [enc setBytes:&pf length:sizeof(pf) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, eff_vl, static_cast<NSUInteger>(bind->n_full) * eff_n_kv)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "gemma4_px_scatter_ring_kv_i8 Metal error: %s\n",
                    cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
            return -1.0;
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

}  // extern "C"
