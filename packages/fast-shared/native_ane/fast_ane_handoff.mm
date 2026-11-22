#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include "fast_ane_handoff.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <sstream>
#include <string>
#include <vector>

@protocol FastAneMTLDeviceSPI <NSObject>
- (nullable id<MTLBuffer>)newBufferWithIOSurface:(IOSurfaceRef)iosurface;
@end

@protocol FastAneModelSPI <NSObject>
+ (nullable id)modelAtURL:(NSURL*)url key:(NSString*)key;
- (nullable NSDictionary*)modelAttributes;
@end

@protocol FastAneClientSPI <NSObject>
+ (nullable id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary*)options qos:(unsigned int)qos error:(NSError**)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary*)options qos:(unsigned int)qos error:(NSError**)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary*)options qos:(unsigned int)qos error:(NSError**)error;
- (BOOL)evaluateWithModel:(id)model
                  options:(NSDictionary*)options
                  request:(id)request
                      qos:(unsigned int)qos
                    error:(NSError**)error;
@end

@protocol FastAneIOSurfaceObjectSPI <NSObject>
+ (nullable id)objectWithIOSurface:(IOSurfaceRef)surface;
@end

@protocol FastAneRequestSPI <NSObject>
+ (nullable id)requestWithInputs:(NSArray*)inputs
                    inputIndices:(NSArray*)inputIndices
                         outputs:(NSArray*)outputs
                   outputIndices:(NSArray*)outputIndices
                   weightsBuffer:(nullable id)weights
                       perfStats:(nullable id)stats
                  procedureIndex:(nullable NSNumber*)procedure
                    sharedEvents:(nullable id)events
               transactionHandle:(nullable NSNumber*)transaction;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError* _Nullable error))handler;
@end

@protocol FastAneSharedWaitEventSPI <NSObject>
+ (nullable id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event eventType:(uint64_t)type;
@end

@protocol FastAneSharedSignalEventSPI <NSObject>
+ (nullable id)signalEventWithValue:(uint64_t)value
                        symbolIndex:(unsigned int)symbol
                          eventType:(int64_t)type
                        sharedEvent:(id)event;
@end

@protocol FastAneSharedEventsSPI <NSObject>
+ (nullable id)sharedEventsWithSignalEvents:(NSArray*)signals waitEvents:(NSArray*)waits;
@end

namespace {

constexpr uint32_t kAneWeightHeaderSize = 128u;
constexpr uint32_t kAneWeightOffset = 64u;
constexpr uint32_t kAneWeightMagic = 0xDEADBEEFu;
constexpr uint32_t kAneQoSDefault = 21u;

struct SignatureSpec {
    const char* class_name;
    bool is_class_method;
    const char* selector_name;
    const char* expected_encoding;
};

constexpr SignatureSpec kSignatures[] = {
    {"_ANEModel", true, "modelAtURL:key:", "@@:@@"},
    {"_ANEModel", false, "modelAttributes", "@@:"},
    {"_ANEClient", true, "sharedConnection", "@@:"},
    {"_ANEClient", false, "compileModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", false, "compiledModelExistsFor:", "B@:@"},
    {"_ANEClient", false, "purgeCompiledModel:", "v@:@"},
    {"_ANEClient", false, "loadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", false, "unloadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", false, "evaluateWithModel:options:request:qos:error:", "B@:@@@I^@"},
    {"_ANEIOSurfaceObject", true, "objectWithIOSurface:", "@@:^{__IOSurface=}"},
    {"_ANERequest",
     true,
     "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"
     "sharedEvents:transactionHandle:",
     "@@:@@@@@@@@@"},
    {"_ANERequest", false, "setCompletionHandler:", "v@:@?"},
    {"_ANESharedWaitEvent", true, "waitEventWithValue:sharedEvent:eventType:", "@@:Q@Q"},
    {"_ANESharedSignalEvent", true, "signalEventWithValue:symbolIndex:eventType:sharedEvent:", "@@:QIq@"},
    {"_ANESharedEvents", true, "sharedEventsWithSignalEvents:waitEvents:", "@@:@@"},
};

struct RuntimeAneClasses {
    void* framework_handle = nullptr;
    Class cls_model = Nil;
    Class cls_client = Nil;
    Class cls_surface = Nil;
    Class cls_request = Nil;
    Class cls_wait_event = Nil;
    Class cls_signal_event = Nil;
    Class cls_shared_events = Nil;
    bool all_selectors_present = false;
    bool all_encodings_exact = false;
    std::string reason;
};

static const char* kEmbeddedMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct SmokeParams {
    uint channels;
    uint width;
    uint row_stride_elems;
};

struct RotatePackParams {
    uint tokens;
    uint hidden_size;
    uint surf_stride_elems;
    uint quantize_a8;
    uint write_unit_scale_row;
    float peak_floor;
};

struct JoinParams {
    uint tokens;
    uint hidden_size;
    uint n_segs;
    uint has_tail;
    uint has_unit_scale;
    uint surf_in_stride_elems;
    uint strides[8];
};

constant float kInvSqrt128 = 0.08838834764831845f;

kernel void ane_smoke_pre(device const half* src [[buffer(0)]],
                          device half* dst_surface [[buffer(1)]],
                          constant SmokeParams& p [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.channels) return;
    const uint src_idx = gid.y * p.width + gid.x;
    const uint dst_idx = gid.y * p.row_stride_elems + gid.x;
    const float v = float(src[src_idx]);
    dst_surface[dst_idx] = half(v * 2.0f + 1.0f);
}

kernel void ane_smoke_post(device const half* src_surface [[buffer(0)]],
                           device half* dst [[buffer(1)]],
                           constant SmokeParams& p [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.channels) return;
    const uint src_idx = gid.y * p.row_stride_elems + gid.x;
    const uint dst_idx = gid.y * p.width + gid.x;
    const float v = float(src_surface[src_idx]);
    dst[dst_idx] = half(v + 5.0f);
}

inline float4 wht128_block(device const half* x_in,
                           device const half* signs,
                           uint row_off,
                           uint base,
                           uint simd_lane) {
    float v0 = float(x_in[row_off + base + 0u]) * float(signs[base + 0u]);
    float v1 = float(x_in[row_off + base + 1u]) * float(signs[base + 1u]);
    float v2 = float(x_in[row_off + base + 2u]) * float(signs[base + 2u]);
    float v3 = float(x_in[row_off + base + 3u]) * float(signs[base + 3u]);

    // Stride 1 & 2 in registers
    float a0 = v0 + v1;
    float a1 = v0 - v1;
    float a2 = v2 + v3;
    float a3 = v2 - v3;
    float b0 = a0 + a2;
    float b1 = a1 + a3;
    float b2 = a0 - a2;
    float b3 = a1 - a3;

    // Strides 4, 8, 16, 32, 64 across 32 lanes via simd_shuffle_xor
    for (uint mask = 1u; mask <= 16u; mask <<= 1u) {
        float p0 = simd_shuffle_xor(b0, mask);
        float p1 = simd_shuffle_xor(b1, mask);
        float p2 = simd_shuffle_xor(b2, mask);
        float p3 = simd_shuffle_xor(b3, mask);
        const bool odd = ((simd_lane & mask) != 0u);
        b0 = odd ? (p0 - b0) : (b0 + p0);
        b1 = odd ? (p1 - b1) : (b1 + p1);
        b2 = odd ? (p2 - b2) : (b2 + p2);
        b3 = odd ? (p3 - b3) : (b3 + p3);
    }

    return float4(b0, b1, b2, b3) * kInvSqrt128;
}

inline half pack_scaled_elem(float v,
                            float row_max,
                            float inv_scale,
                            float out_scale,
                            uint quantize_a8,
                            uint write_unit_scale_row) {
    float val = (quantize_a8 != 0u)
                    ? clamp(rint(v * inv_scale), -127.0f, 127.0f) * out_scale
                    : ((write_unit_scale_row != 0u) ? (v / row_max) : v);
    return half(val);
}

// One threadgroup (e.g. 192 threads = 6 simdgroups) per token row m.
// Supports arbitrary hidden_size % 128 == 0 (including H=1536, E4B H=2560, H=4096).
kernel void ane_ffn_rotate_pack_fp16(device const half* x_in [[buffer(0)]],
                                     device const half* signs [[buffer(1)]],
                                     device half* dst_surface [[buffer(2)]],
                                     constant RotatePackParams& p [[buffer(3)]],
                                     uint tg_id [[threadgroup_position_in_grid]],
                                     uint lid [[thread_position_in_threadgroup]],
                                     uint tg_threads [[threads_per_threadgroup]],
                                     uint simd_lane [[thread_index_in_simdgroup]],
                                     uint simd_id [[simdgroup_index_in_threadgroup]]) {
    const uint m = tg_id;
    if (m >= p.tokens) return;

    const uint num_blocks = p.hidden_size >> 7;
    const uint num_sgs = max(1u, tg_threads >> 5);
    const uint row_off = m * p.hidden_size;

    threadgroup float tg_rot[2048];
    threadgroup float tg_sg_max[16];

    float local_max = 0.0f;
    const bool need_scale = (p.quantize_a8 != 0u || p.write_unit_scale_row != 0u);
    const bool cache_rot = need_scale && (p.hidden_size <= 2048u);

    for (uint blk = simd_id; blk < num_blocks; blk += num_sgs) {
        const uint base = blk * 128u + simd_lane * 4u;
        const float4 b = wht128_block(x_in, signs, row_off, base, simd_lane);

        if (need_scale) {
            if (cache_rot) {
                tg_rot[base + 0u] = b.x;
                tg_rot[base + 1u] = b.y;
                tg_rot[base + 2u] = b.z;
                tg_rot[base + 3u] = b.w;
            }
            local_max = max(local_max, max(max(fabs(b.x), fabs(b.y)), max(fabs(b.z), fabs(b.w))));
        } else {
            dst_surface[(base + 0u) * p.surf_stride_elems + m] = half(b.x);
            dst_surface[(base + 1u) * p.surf_stride_elems + m] = half(b.y);
            dst_surface[(base + 2u) * p.surf_stride_elems + m] = half(b.z);
            dst_surface[(base + 3u) * p.surf_stride_elems + m] = half(b.w);
        }
    }

    if (need_scale) {
        float sg_max = simd_max(local_max);
        if (simd_lane == 0u && simd_id < 16u) {
            tg_sg_max[simd_id] = sg_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float row_max = p.peak_floor;
        for (uint s = 0u; s < min(num_sgs, 16u); ++s) {
            row_max = max(row_max, tg_sg_max[s]);
        }
        const float scale = row_max / 127.0f;
        const float inv_scale = 127.0f / row_max;
        const float out_scale = (p.write_unit_scale_row != 0u) ? (1.0f / 127.0f) : scale;

        if (cache_rot) {
            for (uint c = lid; c < p.hidden_size; c += tg_threads) {
                dst_surface[c * p.surf_stride_elems + m] = pack_scaled_elem(
                    tg_rot[c], row_max, inv_scale, out_scale, p.quantize_a8, p.write_unit_scale_row);
            }
        } else {
            // For H > 2048 (e.g. E4B H=2560, H=4096), recompute WHT-128 in pass 2 instead
            // of skipping row-max scaling, keeping threadgroup memory bounded.
            for (uint blk = simd_id; blk < num_blocks; blk += num_sgs) {
                const uint base = blk * 128u + simd_lane * 4u;
                const float4 b = wht128_block(x_in, signs, row_off, base, simd_lane);
                dst_surface[(base + 0u) * p.surf_stride_elems + m] = pack_scaled_elem(
                    b.x, row_max, inv_scale, out_scale, p.quantize_a8, p.write_unit_scale_row);
                dst_surface[(base + 1u) * p.surf_stride_elems + m] = pack_scaled_elem(
                    b.y, row_max, inv_scale, out_scale, p.quantize_a8, p.write_unit_scale_row);
                dst_surface[(base + 2u) * p.surf_stride_elems + m] = pack_scaled_elem(
                    b.z, row_max, inv_scale, out_scale, p.quantize_a8, p.write_unit_scale_row);
                dst_surface[(base + 3u) * p.surf_stride_elems + m] = pack_scaled_elem(
                    b.w, row_max, inv_scale, out_scale, p.quantize_a8, p.write_unit_scale_row);
            }
        }
        if (p.write_unit_scale_row != 0u && lid == 0u) {
            dst_surface[p.hidden_size * p.surf_stride_elems + m] = half(row_max);
        }
    }
}

kernel void ane_ffn_join_fp16(device const half* s0 [[buffer(0)]],
                              device const half* s1 [[buffer(1)]],
                              device const half* s2 [[buffer(2)]],
                              device const half* s3 [[buffer(3)]],
                              device const half* s4 [[buffer(4)]],
                              device const half* tail [[buffer(5)]],
                              device half* dst [[buffer(6)]],
                              constant JoinParams& p [[buffer(7)]],
                              device const half* src_in [[buffer(8)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint m = gid.x;
    const uint c = gid.y;
    if (m >= p.tokens || c >= p.hidden_size) return;

    float acc = float(s0[c * p.strides[0] + m]);
    if (p.n_segs > 1u) acc += float(s1[c * p.strides[1] + m]);
    if (p.n_segs > 2u) acc += float(s2[c * p.strides[2] + m]);
    if (p.n_segs > 3u) acc += float(s3[c * p.strides[3] + m]);
    if (p.n_segs > 4u) acc += float(s4[c * p.strides[4] + m]);
    if (p.has_unit_scale != 0u) {
        acc *= float(src_in[p.hidden_size * p.surf_in_stride_elems + m]);
    }
    if (p.has_tail != 0u) acc += float(tail[m * p.hidden_size + c]);
    dst[m * p.hidden_size + c] = half(acc);
}
)METAL";

void set_error(char* buf, size_t cap, const std::string& msg) {
    if (!buf || cap == 0) return;
    std::snprintf(buf, cap, "%s", msg.c_str());
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char tmp[8];
                    std::snprintf(tmp, sizeof(tmp), "\\u%04x", static_cast<unsigned>(c));
                    out += tmp;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

double now_ms() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

RuntimeAneClasses inspect_ane_runtime(std::ostringstream* json_sigs) {
    RuntimeAneClasses rt;
    rt.framework_handle =
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!rt.framework_handle) {
        const char* dl_err = dlerror();
        rt.reason = std::string("dlopen failed: ") + (dl_err ? dl_err : "unknown");
        return rt;
    }

    rt.cls_model = NSClassFromString(@"_ANEModel");
    rt.cls_client = NSClassFromString(@"_ANEClient");
    rt.cls_surface = NSClassFromString(@"_ANEIOSurfaceObject");
    rt.cls_request = NSClassFromString(@"_ANERequest");
    rt.cls_wait_event = NSClassFromString(@"_ANESharedWaitEvent");
    rt.cls_signal_event = NSClassFromString(@"_ANESharedSignalEvent");
    rt.cls_shared_events = NSClassFromString(@"_ANESharedEvents");

    bool classes_ok = rt.cls_model && rt.cls_client && rt.cls_surface && rt.cls_request && rt.cls_wait_event &&
                      rt.cls_signal_event && rt.cls_shared_events;
    if (!classes_ok) {
        rt.reason = "one or more private _ANE* classes missing";
    }

    rt.all_selectors_present = classes_ok;
    rt.all_encodings_exact = classes_ok;

    if (json_sigs) {
        *json_sigs << "[";
    }
    const size_t n_sigs = sizeof(kSignatures) / sizeof(kSignatures[0]);
    for (size_t i = 0; i < n_sigs; ++i) {
        const SignatureSpec& spec = kSignatures[i];
        Class cls = NSClassFromString([NSString stringWithUTF8String:spec.class_name]);
        SEL sel = sel_registerName(spec.selector_name);
        Method m = nullptr;
        if (cls) {
            m = spec.is_class_method ? class_getClassMethod(cls, sel) : class_getInstanceMethod(cls, sel);
        }
        const char* raw_enc = m ? method_getTypeEncoding(m) : nullptr;
        std::string canonical_enc;
        if (m) {
            char* ret_t = method_copyReturnType(m);
            if (ret_t) {
                canonical_enc += ret_t;
                std::free(ret_t);
            }
            const unsigned int nargs = method_getNumberOfArguments(m);
            for (unsigned int a = 0; a < nargs; ++a) {
                char* arg_t = method_copyArgumentType(m, a);
                if (arg_t) {
                    canonical_enc += arg_t;
                    std::free(arg_t);
                }
            }
        }
        const bool present = (m != nullptr);
        const bool exact = (!canonical_enc.empty() && canonical_enc == spec.expected_encoding);
        if (!present) {
            rt.all_selectors_present = false;
            if (rt.reason.empty()) {
                rt.reason = std::string("missing selector ") + spec.class_name + " " + spec.selector_name;
            }
        }
        if (!exact) {
            rt.all_encodings_exact = false;
            if (rt.reason.empty()) {
                rt.reason = std::string("encoding mismatch for ") + spec.class_name + " " + spec.selector_name +
                            ": expected " + spec.expected_encoding + ", got " +
                            (canonical_enc.empty() ? (raw_enc ? raw_enc : "null") : canonical_enc);
            }
        }
        if (json_sigs) {
            if (i > 0) *json_sigs << ",";
            *json_sigs << "{"
                       << "\"class\":\"" << spec.class_name << "\","
                       << "\"kind\":\"" << (spec.is_class_method ? "+" : "-") << "\","
                       << "\"selector\":\"" << spec.selector_name << "\","
                       << "\"expected\":\"" << json_escape(spec.expected_encoding) << "\","
                       << "\"actual\":\"" << json_escape(canonical_enc) << "\","
                       << "\"raw\":\"" << json_escape(raw_enc ? raw_enc : "") << "\","
                       << "\"present\":" << (present ? "true" : "false") << ","
                       << "\"exact\":" << (exact ? "true" : "false") << "}";
        }
    }
    if (json_sigs) {
        *json_sigs << "]";
    }

    if (rt.all_selectors_present && rt.all_encodings_exact) {
        rt.reason = "ok";
    }
    return rt;
}

id<MTLBuffer> wrap_iosurface_metal_buffer(id<MTLDevice> device,
                                          IOSurfaceRef iosurface,
                                          size_t alloc_bytes,
                                          bool* out_used_spi) {
    if (out_used_spi) *out_used_spi = false;
    if (!device || !iosurface) return nil;
    if ([device respondsToSelector:@selector(newBufferWithIOSurface:)]) {
        id<MTLBuffer> buf = [static_cast<id<FastAneMTLDeviceSPI>>(device) newBufferWithIOSurface:iosurface];
        if (buf) {
            if (out_used_spi) *out_used_spi = true;
            return buf;
        }
    }
    void* base = IOSurfaceGetBaseAddress(iosurface);
    if (!base || alloc_bytes == 0) return nil;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
    if ((addr & 0x3FFFull) == 0) {
        const size_t page_len = (alloc_bytes + 0x3FFFull) & ~size_t(0x3FFFull);
        id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:base
                                                      length:page_len
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
        if (buf) return buf;
    }
    return nil;
}

}  // namespace

struct FastAneContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLSharedEvent> shared_event = nil;
    id<MTLLibrary> metal_lib = nil;
    id<MTLComputePipelineState> pso_smoke_pre = nil;
    id<MTLComputePipelineState> pso_smoke_post = nil;
    id<MTLComputePipelineState> pso_rotate_pack = nil;
    id<MTLComputePipelineState> pso_join = nil;
    RuntimeAneClasses ane;
    id client = nil;
    bool shared_event_has_port = false;
    bool iosurface_metal_supported = false;
    std::atomic<uint64_t> event_counter{0};
};

struct FastAneSurface {
    IOSurfaceRef iosurface = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bytes_per_elem = 0;
    uint32_t bytes_per_row = 0;
    size_t alloc_bytes = 0;

    ~FastAneSurface() {
        if (iosurface) {
            CFRelease(iosurface);
            iosurface = nullptr;
        }
    }
};

struct FastAneProgram {
    id model = nil;
    NSString* bundle_dir = nil;
    uint32_t qos = kAneQoSDefault;
    bool loaded = false;
    bool owns_dir = true;
    id client = nil;

    ~FastAneProgram() {
        if (loaded && client && model) {
            NSError* err = nil;
            [static_cast<id<FastAneClientSPI>>(client) unloadModel:model options:@{} qos:qos error:&err];
            loaded = false;
        }
        if (client && model) {
            [static_cast<id<FastAneClientSPI>>(client) purgeCompiledModel:model];
        }
        if (owns_dir && bundle_dir) {
            [[NSFileManager defaultManager] removeItemAtPath:bundle_dir error:nil];
        }
        bundle_dir = nil;
    }
};

struct FastAneWave {
    FastAneContext* ctx = nullptr;
    uint32_t tokens = 0;
    uint32_t hidden_size = 0;
    uint32_t n_segs = 0;
    uint32_t timeout_ms = 5000u;
    bool use_shared_event = true;
    uint64_t final_sig_val = 0;
    double gpu_submit_ms = 0.0;
    double ane_submit_ms = 0.0;
    double t_start_ms = 0.0;
    id<MTLBuffer> buf_host_out = nil;
    id<MTLBuffer> buf_dummy = nil;
    dispatch_semaphore_t gpu_b_sem = nullptr;
    std::vector<dispatch_semaphore_t> ane_sems;
    std::vector<BOOL> ane_ok;
    NSError* __strong ane_err = nil;
    NSError* __strong gpu_b_err = nil;
    FastAneStatus early_status = FAST_ANE_OK;
    std::string early_error;
};

extern "C" {

char* fast_ane_probe_compat_json(void) {
    @autoreleasepool {
        std::ostringstream sigs_ss;
        RuntimeAneClasses rt = inspect_ane_runtime(&sigs_ss);

        NSProcessInfo* pi = [NSProcessInfo processInfo];
        NSOperatingSystemVersion os_ver = [pi operatingSystemVersion];
        std::string os_ver_str =
            std::to_string(os_ver.majorVersion) + "." + std::to_string(os_ver.minorVersion) + "." +
            std::to_string(os_ver.patchVersion);
        NSString* os_build = [pi operatingSystemVersionString];

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLSharedEvent> shared_ev = dev ? [dev newSharedEvent] : nil;
        const bool ev_has_port = shared_ev ? ([shared_ev respondsToSelector:NSSelectorFromString(@"eventPort")] == YES)
                                           : false;
        const bool iosurf_spi = dev ? ([dev respondsToSelector:@selector(newBufferWithIOSurface:)] == YES) : false;

        bool metal_runtime_compile_ok = false;
        std::string metal_compile_err;
        if (dev) {
            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:kEmbeddedMetalSource];
            id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
            metal_runtime_compile_ok = (lib != nil);
            if (!lib && err) {
                metal_compile_err = [[err localizedDescription] UTF8String];
            }
        }

        id client = nil;
        if (rt.cls_client && [rt.cls_client respondsToSelector:@selector(sharedConnection)]) {
            client = [static_cast<id<FastAneClientSPI>>(rt.cls_client) sharedConnection];
        }

        const bool ane_available =
            (rt.framework_handle != nullptr) && rt.all_selectors_present && rt.all_encodings_exact && (client != nil);

        std::ostringstream out;
        out << "{"
            << "\"macos_version\":\"" << json_escape(os_ver_str) << "\","
            << "\"macos_build_string\":\"" << json_escape(os_build ? [os_build UTF8String] : "") << "\","
            << "\"metal_device_name\":\"" << json_escape(dev ? [[dev name] UTF8String] : "") << "\","
            << "\"has_unified_memory\":" << ((dev && [dev hasUnifiedMemory]) ? "true" : "false") << ","
            << "\"metal_runtime_compile_ok\":" << (metal_runtime_compile_ok ? "true" : "false") << ","
            << "\"metal_compile_error\":\"" << json_escape(metal_compile_err) << "\","
            << "\"iosurface_metal_spi\":" << (iosurf_spi ? "true" : "false") << ","
            << "\"shared_event_created\":" << (shared_ev ? "true" : "false") << ","
            << "\"shared_event_has_event_port\":" << (ev_has_port ? "true" : "false") << ","
            << "\"framework_loaded\":" << (rt.framework_handle ? "true" : "false") << ","
            << "\"client_connected\":" << (client ? "true" : "false") << ","
            << "\"all_selectors_present\":" << (rt.all_selectors_present ? "true" : "false") << ","
            << "\"all_encodings_exact\":" << (rt.all_encodings_exact ? "true" : "false") << ","
            << "\"ane_available\":" << (ane_available ? "true" : "false") << ","
            << "\"reason\":\"" << json_escape(rt.reason) << "\","
            << "\"signatures\":" << sigs_ss.str()
            << "}";

        const std::string json = out.str();
        char* result = static_cast<char*>(std::malloc(json.size() + 1));
        if (result) {
            std::memcpy(result, json.c_str(), json.size() + 1);
        }
        return result;
    }
}

void fast_ane_free_string(char* str) {
    std::free(str);
}

FastAneStatus fast_ane_context_create(int32_t allow_unavailable,
                                      FastAneContext** out_ctx,
                                      char* err_buf,
                                      size_t err_cap) {
    if (!out_ctx) {
        set_error(err_buf, err_cap, "out_ctx is null");
        return FAST_ANE_INVALID_ARG;
    }
    *out_ctx = nullptr;

    @autoreleasepool {
        auto* ctx = new FastAneContext();
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            set_error(err_buf, err_cap, "MTLCreateSystemDefaultDevice returned nil");
            delete ctx;
            return FAST_ANE_METAL_FAILED;
        }
        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            set_error(err_buf, err_cap, "newCommandQueue returned nil");
            delete ctx;
            return FAST_ANE_METAL_FAILED;
        }

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kEmbeddedMetalSource];
        ctx->metal_lib = [ctx->device newLibraryWithSource:src options:nil error:&err];
        if (!ctx->metal_lib) {
            set_error(err_buf,
                      err_cap,
                      std::string("runtime Metal compile failed: ") + (err ? [[err localizedDescription] UTF8String] : "nil"));
            delete ctx;
            return FAST_ANE_METAL_FAILED;
        }

        id<MTLFunction> fn_pre = [ctx->metal_lib newFunctionWithName:@"ane_smoke_pre"];
        id<MTLFunction> fn_post = [ctx->metal_lib newFunctionWithName:@"ane_smoke_post"];
        id<MTLFunction> fn_rot = [ctx->metal_lib newFunctionWithName:@"ane_ffn_rotate_pack_fp16"];
        id<MTLFunction> fn_join = [ctx->metal_lib newFunctionWithName:@"ane_ffn_join_fp16"];
        if (!fn_pre || !fn_post || !fn_rot || !fn_join) {
            set_error(err_buf, err_cap, "missing Metal kernel functions");
            delete ctx;
            return FAST_ANE_METAL_FAILED;
        }
        ctx->pso_smoke_pre = [ctx->device newComputePipelineStateWithFunction:fn_pre error:&err];
        if (!ctx->pso_smoke_pre) {
            set_error(err_buf, err_cap, "failed creating ane_smoke_pre PSO");
            delete ctx;
            return FAST_ANE_METAL_FAILED;
        }
        ctx->pso_smoke_post = [ctx->device newComputePipelineStateWithFunction:fn_post error:&err];
        if (!ctx->pso_smoke_post) {
            set_error(err_buf, err_cap, "failed creating ane_smoke_post PSO");
            delete ctx;
            return FAST_ANE_METAL_FAILED;
        }
        ctx->pso_rotate_pack = [ctx->device newComputePipelineStateWithFunction:fn_rot error:&err];
        if (!ctx->pso_rotate_pack) {
            set_error(err_buf, err_cap, "failed creating ane_ffn_rotate_pack_fp16 PSO");
            delete ctx;
            return FAST_ANE_METAL_FAILED;
        }
        ctx->pso_join = [ctx->device newComputePipelineStateWithFunction:fn_join error:&err];
        if (!ctx->pso_join) {
            set_error(err_buf, err_cap, "failed creating ane_ffn_join_fp16 PSO");
            delete ctx;
            return FAST_ANE_METAL_FAILED;
        }

        ctx->shared_event = [ctx->device newSharedEvent];
        ctx->shared_event_has_port =
            ctx->shared_event && ([ctx->shared_event respondsToSelector:NSSelectorFromString(@"eventPort")] == YES);
        ctx->iosurface_metal_supported =
            [ctx->device respondsToSelector:@selector(newBufferWithIOSurface:)] == YES;

        ctx->ane = inspect_ane_runtime(nullptr);
        if (ctx->ane.all_selectors_present && ctx->ane.all_encodings_exact && ctx->ane.cls_client) {
            ctx->client = [static_cast<id<FastAneClientSPI>>(ctx->ane.cls_client) sharedConnection];
        }

        const bool ane_ok = (ctx->client != nil) && ctx->ane.all_selectors_present && ctx->ane.all_encodings_exact;
        if (!ane_ok && !allow_unavailable) {
            set_error(err_buf, err_cap, ctx->ane.reason.empty() ? "ANE private classes unavailable" : ctx->ane.reason);
            delete ctx;
            return FAST_ANE_UNAVAILABLE;
        }

        *out_ctx = ctx;
        return FAST_ANE_OK;
    }
}

void fast_ane_context_destroy(FastAneContext* ctx) {
    delete ctx;
}

int32_t fast_ane_context_ane_available(const FastAneContext* ctx) {
    if (!ctx) return 0;
    return (ctx->client != nil && ctx->ane.all_selectors_present && ctx->ane.all_encodings_exact) ? 1 : 0;
}

int32_t fast_ane_context_shared_event_available(const FastAneContext* ctx) {
    if (!ctx) return 0;
    return (ctx->shared_event != nil && ctx->shared_event_has_port) ? 1 : 0;
}

int32_t fast_ane_context_iosurface_metal_available(const FastAneContext* ctx) {
    if (!ctx) return 0;
    return ctx->iosurface_metal_supported ? 1 : 0;
}

FastAneStatus fast_ane_surface_create(uint32_t width,
                                      uint32_t height,
                                      uint32_t bytes_per_elem,
                                      uint32_t pixel_format,
                                      FastAneSurface** out_surf,
                                      char* err_buf,
                                      size_t err_cap) {
    if (!out_surf || width == 0 || height == 0 || bytes_per_elem == 0) {
        set_error(err_buf, err_cap, "invalid surface geometry");
        return FAST_ANE_INVALID_ARG;
    }
    *out_surf = nullptr;

    @autoreleasepool {
        const uint64_t raw_row = static_cast<uint64_t>(width) * bytes_per_elem;
        const uint32_t bytes_per_row = static_cast<uint32_t>((raw_row + 63ull) & ~63ull);
        const size_t alloc_bytes = static_cast<size_t>(bytes_per_row) * height;

        NSDictionary* props = @{
            (NSString*)kIOSurfaceWidth : @(width),
            (NSString*)kIOSurfaceHeight : @(height),
            (NSString*)kIOSurfaceBytesPerElement : @(bytes_per_elem),
            (NSString*)kIOSurfaceBytesPerRow : @(bytes_per_row),
            (NSString*)kIOSurfaceAllocSize : @(alloc_bytes),
            (NSString*)kIOSurfacePixelFormat : @(pixel_format),
        };
        IOSurfaceRef ref = IOSurfaceCreate((__bridge CFDictionaryRef)props);
        if (!ref) {
            set_error(err_buf, err_cap, "IOSurfaceCreate failed");
            return FAST_ANE_INVALID_ARG;
        }
        auto* surf = new FastAneSurface();
        surf->iosurface = ref;
        surf->width = width;
        surf->height = height;
        surf->bytes_per_elem = bytes_per_elem;
        surf->bytes_per_row = bytes_per_row;
        surf->alloc_bytes = alloc_bytes;
        *out_surf = surf;
        return FAST_ANE_OK;
    }
}

void fast_ane_surface_destroy(FastAneSurface* surf) {
    delete surf;
}

uint32_t fast_ane_surface_width(const FastAneSurface* surf) {
    return surf ? surf->width : 0u;
}
uint32_t fast_ane_surface_height(const FastAneSurface* surf) {
    return surf ? surf->height : 0u;
}
uint32_t fast_ane_surface_row_bytes(const FastAneSurface* surf) {
    return surf ? surf->bytes_per_row : 0u;
}
size_t fast_ane_surface_alloc_bytes(const FastAneSurface* surf) {
    return surf ? surf->alloc_bytes : 0u;
}

FastAneStatus fast_ane_surface_write(FastAneSurface* surf,
                                     const void* src,
                                     uint32_t src_row_bytes,
                                     uint32_t rows) {
    if (!surf || !surf->iosurface || !src || rows > surf->height || src_row_bytes > surf->bytes_per_row) {
        return FAST_ANE_INVALID_ARG;
    }
    IOSurfaceLock(surf->iosurface, 0, nullptr);
    auto* dst_base = static_cast<uint8_t*>(IOSurfaceGetBaseAddress(surf->iosurface));
    const auto* src_base = static_cast<const uint8_t*>(src);
    if (src_row_bytes == surf->bytes_per_row) {
        std::memcpy(dst_base, src_base, static_cast<size_t>(src_row_bytes) * rows);
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            std::memcpy(dst_base + static_cast<size_t>(r) * surf->bytes_per_row,
                        src_base + static_cast<size_t>(r) * src_row_bytes,
                        src_row_bytes);
        }
    }
    IOSurfaceUnlock(surf->iosurface, 0, nullptr);
    return FAST_ANE_OK;
}

FastAneStatus fast_ane_surface_read(const FastAneSurface* surf,
                                    void* dst,
                                    uint32_t dst_row_bytes,
                                    uint32_t rows) {
    if (!surf || !surf->iosurface || !dst || rows > surf->height || dst_row_bytes > surf->bytes_per_row) {
        return FAST_ANE_INVALID_ARG;
    }
    IOSurfaceLock(surf->iosurface, kIOSurfaceLockReadOnly, nullptr);
    const auto* src_base = static_cast<const uint8_t*>(IOSurfaceGetBaseAddress(surf->iosurface));
    auto* dst_base = static_cast<uint8_t*>(dst);
    if (dst_row_bytes == surf->bytes_per_row) {
        std::memcpy(dst_base, src_base, static_cast<size_t>(dst_row_bytes) * rows);
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            std::memcpy(dst_base + static_cast<size_t>(r) * dst_row_bytes,
                        src_base + static_cast<size_t>(r) * surf->bytes_per_row,
                        dst_row_bytes);
        }
    }
    IOSurfaceUnlock(surf->iosurface, kIOSurfaceLockReadOnly, nullptr);
    return FAST_ANE_OK;
}

FastAneStatus fast_ane_program_create(FastAneContext* ctx,
                                      const char* mil_text,
                                      size_t mil_len,
                                      const void* weight_data,
                                      size_t weight_bytes,
                                      uint32_t weight_dtype_code,
                                      int32_t data_is_preformatted,
                                      uint32_t qos,
                                      FastAneProgram** out_prog,
                                      char* err_buf,
                                      size_t err_cap) {
    if (!ctx || !mil_text || mil_len == 0 || !out_prog) {
        set_error(err_buf, err_cap, "invalid arguments to fast_ane_program_create");
        return FAST_ANE_INVALID_ARG;
    }
    *out_prog = nullptr;
    if (!fast_ane_context_ane_available(ctx)) {
        set_error(err_buf, err_cap, ctx->ane.reason.empty() ? "ANE unavailable" : ctx->ane.reason);
        return FAST_ANE_UNAVAILABLE;
    }

    @autoreleasepool {
        NSFileManager* fm = [NSFileManager defaultManager];
        NSString* dir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:@"fast_ane_%@", [[NSUUID UUID] UUIDString]]];
        NSError* err = nil;
        if (![fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:&err]) {
            set_error(err_buf, err_cap, err ? [[err localizedDescription] UTF8String] : "mkdir failed");
            return FAST_ANE_COMPILE_FAILED;
        }

        NSData* mil_data = [NSData dataWithBytes:mil_text length:mil_len];
        NSString* mil_path = [dir stringByAppendingPathComponent:@"model.mil"];
        if (![mil_data writeToFile:mil_path options:NSDataWritingAtomic error:&err]) {
            set_error(err_buf, err_cap, err ? [[err localizedDescription] UTF8String] : "write model.mil failed");
            [fm removeItemAtPath:dir error:nil];
            return FAST_ANE_COMPILE_FAILED;
        }

        if (weight_data && weight_bytes > 0) {
            NSString* wdir = [dir stringByAppendingPathComponent:@"weights"];
            if (![fm createDirectoryAtPath:wdir withIntermediateDirectories:YES attributes:nil error:&err]) {
                set_error(err_buf, err_cap, "create weights dir failed");
                [fm removeItemAtPath:dir error:nil];
                return FAST_ANE_COMPILE_FAILED;
            }
            NSString* wpath = [wdir stringByAppendingPathComponent:@"weight.bin"];
            NSMutableData* blob = nil;
            if (data_is_preformatted) {
                blob = [NSMutableData dataWithBytes:weight_data length:weight_bytes];
            } else {
                blob = [NSMutableData dataWithLength:kAneWeightHeaderSize + weight_bytes];
                auto* bytes = static_cast<uint8_t*>([blob mutableBytes]);
                std::memset(bytes, 0, kAneWeightHeaderSize);
                const uint32_t u1 = 1u;
                const uint32_t u2 = 2u;
                const uint32_t magic = kAneWeightMagic;
                const uint32_t dtype = weight_dtype_code ? weight_dtype_code : 1u;
                const uint64_t len64 = static_cast<uint64_t>(weight_bytes);
                const uint64_t off64 = static_cast<uint64_t>(kAneWeightHeaderSize);
                std::memcpy(bytes + 0, &u1, 4);
                std::memcpy(bytes + 4, &u2, 4);
                std::memcpy(bytes + kAneWeightOffset + 0, &magic, 4);
                std::memcpy(bytes + kAneWeightOffset + 4, &dtype, 4);
                std::memcpy(bytes + kAneWeightOffset + 8, &len64, 8);
                std::memcpy(bytes + kAneWeightOffset + 16, &off64, 8);
                std::memcpy(bytes + kAneWeightHeaderSize, weight_data, weight_bytes);
            }
            if (![blob writeToFile:wpath options:NSDataWritingAtomic error:&err]) {
                set_error(err_buf, err_cap, err ? [[err localizedDescription] UTF8String] : "write weight.bin failed");
                [fm removeItemAtPath:dir error:nil];
                return FAST_ANE_COMPILE_FAILED;
            }
        }

        NSURL* model_url = [NSURL fileURLWithPath:dir isDirectory:YES];
        id model = [static_cast<id<FastAneModelSPI>>(ctx->ane.cls_model) modelAtURL:model_url key:@"main"];
        if (!model) {
            set_error(err_buf, err_cap, "_ANEModel modelAtURL:key: returned nil");
            [fm removeItemAtPath:dir error:nil];
            return FAST_ANE_COMPILE_FAILED;
        }

        NSDictionary* compile_opts = @{
            @"kANEFModelType" : @"kANEFModelMIL",
            @"kANEFNetPlistFilenameKey" : @"model.mil",
        };
        const unsigned int use_qos = qos ? qos : kAneQoSDefault;
        auto client = static_cast<id<FastAneClientSPI>>(ctx->client);

        if (![client compileModel:model options:compile_opts qos:use_qos error:&err]) {
            set_error(err_buf,
                      err_cap,
                      std::string("compileModel failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
            [fm removeItemAtPath:dir error:nil];
            return FAST_ANE_COMPILE_FAILED;
        }
        if (![client loadModel:model options:@{} qos:use_qos error:&err]) {
            set_error(err_buf,
                      err_cap,
                      std::string("loadModel failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
            [client purgeCompiledModel:model];
            [fm removeItemAtPath:dir error:nil];
            return FAST_ANE_LOAD_FAILED;
        }

        auto* prog = new FastAneProgram();
        prog->model = model;
        prog->bundle_dir = dir;
        prog->qos = use_qos;
        prog->loaded = true;
        prog->owns_dir = true;
        prog->client = ctx->client;
        *out_prog = prog;
        return FAST_ANE_OK;
    }
}

FastAneStatus fast_ane_program_create_from_dir(FastAneContext* ctx,
                                               const char* bundle_dir,
                                               uint32_t qos,
                                               FastAneProgram** out_prog,
                                               char* err_buf,
                                               size_t err_cap) {
    if (!ctx || !bundle_dir || !out_prog) {
        set_error(err_buf, err_cap, "invalid arguments to fast_ane_program_create_from_dir");
        return FAST_ANE_INVALID_ARG;
    }
    *out_prog = nullptr;
    if (!fast_ane_context_ane_available(ctx)) {
        set_error(err_buf, err_cap, ctx->ane.reason.empty() ? "ANE unavailable" : ctx->ane.reason);
        return FAST_ANE_UNAVAILABLE;
    }

    @autoreleasepool {
        NSString* dir = [NSString stringWithUTF8String:bundle_dir];
        NSURL* model_url = [NSURL fileURLWithPath:dir isDirectory:YES];
        id model = [static_cast<id<FastAneModelSPI>>(ctx->ane.cls_model) modelAtURL:model_url key:@"main"];
        if (!model) {
            set_error(err_buf, err_cap, "_ANEModel modelAtURL:key: returned nil");
            return FAST_ANE_COMPILE_FAILED;
        }
        NSDictionary* compile_opts = @{
            @"kANEFModelType" : @"kANEFModelMIL",
            @"kANEFNetPlistFilenameKey" : @"model.mil",
        };
        const unsigned int use_qos = qos ? qos : kAneQoSDefault;
        auto client = static_cast<id<FastAneClientSPI>>(ctx->client);
        NSError* err = nil;
        if (![client compileModel:model options:compile_opts qos:use_qos error:&err]) {
            set_error(err_buf,
                      err_cap,
                      std::string("compileModel failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
            return FAST_ANE_COMPILE_FAILED;
        }
        if (![client loadModel:model options:@{} qos:use_qos error:&err]) {
            set_error(err_buf,
                      err_cap,
                      std::string("loadModel failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
            [client purgeCompiledModel:model];
            return FAST_ANE_LOAD_FAILED;
        }

        auto* prog = new FastAneProgram();
        prog->model = model;
        prog->bundle_dir = dir;
        prog->qos = use_qos;
        prog->loaded = true;
        prog->owns_dir = false;
        prog->client = ctx->client;
        *out_prog = prog;
        return FAST_ANE_OK;
    }
}

void fast_ane_program_destroy(FastAneProgram* prog) {
    delete prog;
}

static id build_ane_request(const RuntimeAneClasses& ane,
                            FastAneSurface* const* inputs,
                            uint32_t n_inputs,
                            FastAneSurface* const* outputs,
                            uint32_t n_outputs,
                            id shared_events,
                            uint64_t tx_id) {
    NSMutableArray* in_objs = [NSMutableArray arrayWithCapacity:n_inputs];
    NSMutableArray* in_idx = [NSMutableArray arrayWithCapacity:n_inputs];
    for (uint32_t i = 0; i < n_inputs; ++i) {
        id surf_obj = [static_cast<id<FastAneIOSurfaceObjectSPI>>(ane.cls_surface) objectWithIOSurface:inputs[i]->iosurface];
        if (!surf_obj) return nil;
        [in_objs addObject:surf_obj];
        [in_idx addObject:@(i)];
    }

    NSMutableArray* out_objs = [NSMutableArray arrayWithCapacity:n_outputs];
    NSMutableArray* out_idx = [NSMutableArray arrayWithCapacity:n_outputs];
    for (uint32_t i = 0; i < n_outputs; ++i) {
        id surf_obj = [static_cast<id<FastAneIOSurfaceObjectSPI>>(ane.cls_surface) objectWithIOSurface:outputs[i]->iosurface];
        if (!surf_obj) return nil;
        [out_objs addObject:surf_obj];
        [out_idx addObject:@(i)];
    }

    return [static_cast<id<FastAneRequestSPI>>(ane.cls_request) requestWithInputs:in_objs
                                                                      inputIndices:in_idx
                                                                           outputs:out_objs
                                                                     outputIndices:out_idx
                                                                     weightsBuffer:nil
                                                                         perfStats:nil
                                                                    procedureIndex:@0
                                                                      sharedEvents:shared_events
                                                                 transactionHandle:@(tx_id)];
}

FastAneStatus fast_ane_program_eval(FastAneContext* ctx,
                                    FastAneProgram* prog,
                                    FastAneSurface* const* inputs,
                                    uint32_t n_inputs,
                                    FastAneSurface* const* outputs,
                                    uint32_t n_outputs,
                                    double* out_ms,
                                    char* err_buf,
                                    size_t err_cap) {
    if (!ctx || !prog || !prog->loaded || !inputs || !outputs || n_inputs == 0 || n_outputs == 0) {
        set_error(err_buf, err_cap, "invalid arguments to fast_ane_program_eval");
        return FAST_ANE_INVALID_ARG;
    }

    @autoreleasepool {
        const uint64_t tx_id = ctx->event_counter.fetch_add(1) + 1;
        id req = build_ane_request(ctx->ane, inputs, n_inputs, outputs, n_outputs, nil, tx_id);
        if (!req) {
            set_error(err_buf, err_cap, "failed to create _ANERequest");
            return FAST_ANE_EVAL_FAILED;
        }

        NSError* err = nil;
        auto client = static_cast<id<FastAneClientSPI>>(ctx->client);
        const double t0 = now_ms();
        const BOOL ok = [client evaluateWithModel:prog->model options:@{} request:req qos:prog->qos error:&err];
        const double t1 = now_ms();
        if (out_ms) *out_ms = (t1 - t0);
        if (!ok) {
            set_error(err_buf,
                      err_cap,
                      std::string("evaluateWithModel failed: ") + (err ? [[err localizedDescription] UTF8String] : "unknown"));
            return FAST_ANE_EVAL_FAILED;
        }
        return FAST_ANE_OK;
    }
}

FastAneStatus fast_ane_handoff_roundtrip(FastAneContext* ctx,
                                         FastAneProgram* prog,
                                         FastAneSurface* surf_in,
                                         FastAneSurface* surf_out,
                                         const uint16_t* host_in_fp16,
                                         uint16_t* host_out_fp16,
                                         uint32_t count_elems,
                                         int32_t mode,
                                         uint32_t timeout_ms,
                                         FastAneHandoffMetrics* out_metrics,
                                         char* err_buf,
                                         size_t err_cap) {
    if (!ctx || !prog || !prog->loaded || !surf_in || !surf_out || !host_in_fp16 || !host_out_fp16 || count_elems == 0) {
        set_error(err_buf, err_cap, "invalid arguments to fast_ane_handoff_roundtrip");
        return FAST_ANE_INVALID_ARG;
    }
    if (surf_in->width * surf_in->height != count_elems || surf_out->width * surf_out->height != count_elems) {
        set_error(err_buf, err_cap, "count_elems does not match surface width * height");
        return FAST_ANE_INVALID_ARG;
    }

    @autoreleasepool {
        bool used_spi_in = false;
        bool used_spi_out = false;
        id<MTLBuffer> buf_surf_in =
            wrap_iosurface_metal_buffer(ctx->device, surf_in->iosurface, surf_in->alloc_bytes, &used_spi_in);
        id<MTLBuffer> buf_surf_out =
            wrap_iosurface_metal_buffer(ctx->device, surf_out->iosurface, surf_out->alloc_bytes, &used_spi_out);
        if (!buf_surf_in || !buf_surf_out) {
            set_error(err_buf, err_cap, "failed wrapping IOSurface into MTLBuffer");
            return FAST_ANE_METAL_FAILED;
        }

        const size_t flat_bytes = static_cast<size_t>(count_elems) * sizeof(uint16_t);
        id<MTLBuffer> buf_host_in = [ctx->device newBufferWithBytes:host_in_fp16
                                                             length:flat_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_host_out = [ctx->device newBufferWithLength:flat_bytes
                                                              options:MTLResourceStorageModeShared];
        if (!buf_host_in || !buf_host_out) {
            set_error(err_buf, err_cap, "failed allocating Metal staging buffers");
            return FAST_ANE_METAL_FAILED;
        }

        struct SmokeParams {
            uint32_t channels;
            uint32_t width;
            uint32_t row_stride_elems;
        };
        SmokeParams p_in{surf_in->height, surf_in->width, surf_in->bytes_per_row / surf_in->bytes_per_elem};
        SmokeParams p_out{surf_out->height, surf_out->width, surf_out->bytes_per_row / surf_out->bytes_per_elem};

        const MTLSize grid = MTLSizeMake(surf_in->width, surf_in->height, 1);
        const NSUInteger tg_x = std::min<NSUInteger>(32, surf_in->width);
        const NSUInteger tg_y = std::max<NSUInteger>(1, std::min<NSUInteger>(8, surf_in->height));
        const MTLSize tg = MTLSizeMake(tg_x, tg_y, 1);

        const uint32_t eff_timeout_ms = timeout_ms ? timeout_ms : 5000u;
        const bool use_shared = (mode == 0);
        if (use_shared && !fast_ane_context_shared_event_available(ctx)) {
            set_error(err_buf, err_cap, "MTLSharedEvent eventPort unavailable");
            return FAST_ANE_UNAVAILABLE;
        }

        const double t_total_0 = now_ms();
        double gpu_pre_ms = 0.0;
        double ane_submit_ms = 0.0;
        double gpu_post_ms = 0.0;
        uint64_t final_ev_val = ctx->shared_event ? [ctx->shared_event signaledValue] : 0;

        if (use_shared) {
            const uint64_t v1 = ctx->event_counter.fetch_add(2) + 1;
            const uint64_t v_cur = [ctx->shared_event signaledValue];
            const uint64_t wait_val = (v1 > v_cur) ? v1 : (v_cur + 1);
            const uint64_t sig_val = wait_val + 1;
            ctx->event_counter.store(sig_val);

            id wait_ev = [static_cast<id<FastAneSharedWaitEventSPI>>(ctx->ane.cls_wait_event)
                waitEventWithValue:wait_val
                       sharedEvent:ctx->shared_event
                         eventType:0];
            id sig_ev = [static_cast<id<FastAneSharedSignalEventSPI>>(ctx->ane.cls_signal_event)
                signalEventWithValue:sig_val
                         symbolIndex:0
                           eventType:0
                         sharedEvent:ctx->shared_event];
            id shared_evs = [static_cast<id<FastAneSharedEventsSPI>>(ctx->ane.cls_shared_events)
                sharedEventsWithSignalEvents:@[sig_ev]
                                  waitEvents:@[wait_ev]];
            FastAneSurface* ins[] = {surf_in};
            FastAneSurface* outs[] = {surf_out};
            id req = build_ane_request(ctx->ane, ins, 1, outs, 1, shared_evs, sig_val);
            if (!wait_ev || !sig_ev || !shared_evs || !req) {
                set_error(err_buf, err_cap, "failed creating _ANESharedEvents request");
                return FAST_ANE_EVAL_FAILED;
            }

            dispatch_semaphore_t ane_sem = dispatch_semaphore_create(0);
            __block BOOL ane_cb_ok = NO;
            __block NSError* ane_cb_err = nil;
            [static_cast<id<FastAneRequestSPI>>(req) setCompletionHandler:^(BOOL success, NSError* error) {
                ane_cb_ok = success;
                ane_cb_err = error;
                dispatch_semaphore_signal(ane_sem);
            }];

            id<MTLCommandBuffer> cmdA = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> encA = [cmdA computeCommandEncoder];
            [encA setComputePipelineState:ctx->pso_smoke_pre];
            [encA setBuffer:buf_host_in offset:0 atIndex:0];
            [encA setBuffer:buf_surf_in offset:0 atIndex:1];
            [encA setBytes:&p_in length:sizeof(p_in) atIndex:2];
            [encA dispatchThreads:grid threadsPerThreadgroup:tg];
            [encA endEncoding];
            [cmdA encodeSignalEvent:ctx->shared_event value:wait_val];

            id<MTLCommandBuffer> cmdB = [ctx->queue commandBuffer];
            [cmdB encodeWaitForEvent:ctx->shared_event value:sig_val];
            id<MTLComputeCommandEncoder> encB = [cmdB computeCommandEncoder];
            [encB setComputePipelineState:ctx->pso_smoke_post];
            [encB setBuffer:buf_surf_out offset:0 atIndex:0];
            [encB setBuffer:buf_host_out offset:0 atIndex:1];
            [encB setBytes:&p_out length:sizeof(p_out) atIndex:2];
            [encB dispatchThreads:grid threadsPerThreadgroup:tg];
            [encB endEncoding];

            dispatch_semaphore_t gpu_b_sem = dispatch_semaphore_create(0);
            __block NSError* gpu_b_err = nil;
            [cmdB addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                gpu_b_err = [cb error];
                dispatch_semaphore_signal(gpu_b_sem);
            }];

            // Commit cmdA and cmdB before submitting ANE evaluateWithModel so neither GPU pre
            // nor GPU post awaits CPU intervention regardless of sync vs async evaluate behavior.
            const double t_pre_0 = now_ms();
            [cmdA commit];
            [cmdB commit];
            gpu_pre_ms = now_ms() - t_pre_0;

            NSError* eval_err = nil;
            const double t_ane_0 = now_ms();
            const BOOL submit_ok = [static_cast<id<FastAneClientSPI>>(ctx->client)
                evaluateWithModel:prog->model
                          options:@{}
                          request:req
                              qos:prog->qos
                            error:&eval_err];
            ane_submit_ms = now_ms() - t_ane_0;

            if (!submit_ok) {
                if ([ctx->shared_event signaledValue] < sig_val) {
                    [ctx->shared_event setSignaledValue:sig_val];
                }
                dispatch_semaphore_wait(
                    gpu_b_sem, dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(eff_timeout_ms) * NSEC_PER_MSEC));
                set_error(err_buf,
                          err_cap,
                          std::string("shared evaluateWithModel failed: ") +
                              (eval_err ? [[eval_err localizedDescription] UTF8String] : "unknown"));
                return FAST_ANE_EVAL_FAILED;
            }

            const dispatch_time_t deadline =
                dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(eff_timeout_ms) * NSEC_PER_MSEC);
            if (dispatch_semaphore_wait(ane_sem, deadline) != 0) {
                if ([ctx->shared_event signaledValue] < sig_val) {
                    [ctx->shared_event setSignaledValue:sig_val];
                }
                dispatch_semaphore_wait(gpu_b_sem,
                                        dispatch_time(DISPATCH_TIME_NOW, 1000LL * NSEC_PER_MSEC));
                set_error(err_buf, err_cap, "ANE shared-event completion timed out");
                return FAST_ANE_TIMEOUT;
            }
            if (!ane_cb_ok) {
                if ([ctx->shared_event signaledValue] < sig_val) {
                    [ctx->shared_event setSignaledValue:sig_val];
                }
                dispatch_semaphore_wait(gpu_b_sem,
                                        dispatch_time(DISPATCH_TIME_NOW, 1000LL * NSEC_PER_MSEC));
                set_error(err_buf,
                          err_cap,
                          std::string("ANE completion handler error: ") +
                              (ane_cb_err ? [[ane_cb_err localizedDescription] UTF8String] : "unknown"));
                return FAST_ANE_EVAL_FAILED;
            }

            const double t_wait_b0 = now_ms();
            if (dispatch_semaphore_wait(gpu_b_sem, deadline) != 0) {
                set_error(err_buf, err_cap, "Metal post command buffer timed out");
                return FAST_ANE_TIMEOUT;
            }
            gpu_post_ms = now_ms() - t_wait_b0;
            if (gpu_b_err) {
                set_error(err_buf, err_cap, [[gpu_b_err localizedDescription] UTF8String]);
                return FAST_ANE_METAL_FAILED;
            }
            final_ev_val = [ctx->shared_event signaledValue];
        } else {
            // CPU-synchronized baseline: cmdA waitUntilCompleted -> ANE eval -> cmdB waitUntilCompleted.
            id<MTLCommandBuffer> cmdA = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> encA = [cmdA computeCommandEncoder];
            [encA setComputePipelineState:ctx->pso_smoke_pre];
            [encA setBuffer:buf_host_in offset:0 atIndex:0];
            [encA setBuffer:buf_surf_in offset:0 atIndex:1];
            [encA setBytes:&p_in length:sizeof(p_in) atIndex:2];
            [encA dispatchThreads:grid threadsPerThreadgroup:tg];
            [encA endEncoding];
            const double tA0 = now_ms();
            [cmdA commit];
            [cmdA waitUntilCompleted];
            gpu_pre_ms = now_ms() - tA0;
            if ([cmdA error]) {
                set_error(err_buf, err_cap, [[[cmdA error] localizedDescription] UTF8String]);
                return FAST_ANE_METAL_FAILED;
            }

            FastAneSurface* ins[] = {surf_in};
            FastAneSurface* outs[] = {surf_out};
            FastAneStatus st = fast_ane_program_eval(ctx, prog, ins, 1, outs, 1, &ane_submit_ms, err_buf, err_cap);
            if (st != FAST_ANE_OK) return st;

            id<MTLCommandBuffer> cmdB = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> encB = [cmdB computeCommandEncoder];
            [encB setComputePipelineState:ctx->pso_smoke_post];
            [encB setBuffer:buf_surf_out offset:0 atIndex:0];
            [encB setBuffer:buf_host_out offset:0 atIndex:1];
            [encB setBytes:&p_out length:sizeof(p_out) atIndex:2];
            [encB dispatchThreads:grid threadsPerThreadgroup:tg];
            [encB endEncoding];
            const double tB0 = now_ms();
            [cmdB commit];
            [cmdB waitUntilCompleted];
            gpu_post_ms = now_ms() - tB0;
            if ([cmdB error]) {
                set_error(err_buf, err_cap, [[[cmdB error] localizedDescription] UTF8String]);
                return FAST_ANE_METAL_FAILED;
            }
        }

        const double total_ms = now_ms() - t_total_0;
        std::memcpy(host_out_fp16, [buf_host_out contents], flat_bytes);
        if (out_metrics) {
            out_metrics->gpu_pre_ms = gpu_pre_ms;
            out_metrics->ane_submit_ms = ane_submit_ms;
            out_metrics->gpu_post_and_total_ms = gpu_post_ms;
            out_metrics->total_ms = total_ms;
            out_metrics->used_shared_event = use_shared ? 1 : 0;
            out_metrics->used_iosurface_metal_buffer = (used_spi_in && used_spi_out) ? 1 : 0;
            out_metrics->final_shared_event_value = final_ev_val;
        }
        return FAST_ANE_OK;
    }
}

FastAneStatus fast_ane_ffn_wave_begin(FastAneContext* ctx,
                                      FastAneProgram* const* seg_progs,
                                      FastAneSurface* const* seg_out_surfs,
                                      uint32_t n_segs,
                                      FastAneSurface* surf_in,
                                      const uint16_t* host_in_fp16,
                                      const uint16_t* host_signs_fp16,
                                      uint32_t tokens,
                                      uint32_t hidden_size,
                                      int32_t quantize_a8,
                                      int32_t use_shared_event,
                                      uint32_t timeout_ms,
                                      FastAneWave** out_wave,
                                      char* err_buf,
                                      size_t err_cap) {
    if (!ctx || !seg_progs || !seg_out_surfs || !surf_in || !host_in_fp16 || !host_signs_fp16 || !out_wave ||
        n_segs == 0 || n_segs > 5 || tokens == 0 || hidden_size == 0 || (hidden_size % 128u) != 0) {
        set_error(err_buf, err_cap, "invalid arguments to fast_ane_ffn_wave_begin");
        return FAST_ANE_INVALID_ARG;
    }
    *out_wave = nullptr;
    const bool unit_scaled = (surf_in->height == hidden_size + 1u);
    if (surf_in->width != tokens || (surf_in->height != hidden_size && !unit_scaled)) {
        set_error(err_buf,
                  err_cap,
                  "surf_in dimensions mismatch (expected width=tokens, height=hidden_size or hidden_size+1)");
        return FAST_ANE_INVALID_ARG;
    }
    for (uint32_t i = 0; i < n_segs; ++i) {
        if (!seg_progs[i] || !seg_progs[i]->loaded || !seg_out_surfs[i] ||
            seg_out_surfs[i]->width != tokens || seg_out_surfs[i]->height != hidden_size) {
            set_error(err_buf, err_cap, "segment program or output surface invalid/mismatched");
            return FAST_ANE_INVALID_ARG;
        }
    }
    if (use_shared_event && !fast_ane_context_shared_event_available(ctx)) {
        set_error(err_buf, err_cap, "MTLSharedEvent eventPort unavailable");
        return FAST_ANE_UNAVAILABLE;
    }

    @autoreleasepool {
        bool spi_ok = false;
        id<MTLBuffer> buf_surf_in =
            wrap_iosurface_metal_buffer(ctx->device, surf_in->iosurface, surf_in->alloc_bytes, &spi_ok);
        if (!buf_surf_in) {
            set_error(err_buf, err_cap, "failed wrapping surf_in IOSurface into MTLBuffer");
            return FAST_ANE_METAL_FAILED;
        }
        std::vector<id<MTLBuffer>> seg_out_bufs(n_segs);
        for (uint32_t i = 0; i < n_segs; ++i) {
            seg_out_bufs[i] = wrap_iosurface_metal_buffer(
                ctx->device, seg_out_surfs[i]->iosurface, seg_out_surfs[i]->alloc_bytes, &spi_ok);
            if (!seg_out_bufs[i]) {
                set_error(err_buf, err_cap, "failed wrapping segment output IOSurface into MTLBuffer");
                return FAST_ANE_METAL_FAILED;
            }
        }

        const size_t io_bytes = static_cast<size_t>(tokens) * hidden_size * sizeof(uint16_t);
        const size_t sign_bytes = static_cast<size_t>(hidden_size) * sizeof(uint16_t);
        id<MTLBuffer> buf_host_in = [ctx->device newBufferWithBytes:host_in_fp16
                                                             length:io_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_signs = [ctx->device newBufferWithBytes:host_signs_fp16
                                                           length:sign_bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_host_out = [ctx->device newBufferWithLength:io_bytes
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_dummy = [ctx->device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        if (!buf_host_in || !buf_signs || !buf_host_out || !buf_dummy) {
            set_error(err_buf, err_cap, "failed allocating Metal buffers for wave");
            return FAST_ANE_METAL_FAILED;
        }

        struct RotatePackParams {
            uint32_t tokens;
            uint32_t hidden_size;
            uint32_t surf_stride_elems;
            uint32_t quantize_a8;
            uint32_t write_unit_scale_row;
            float peak_floor;
        };
        const uint32_t in_stride = surf_in->bytes_per_row / surf_in->bytes_per_elem;
        RotatePackParams rot_p{
            tokens,
            hidden_size,
            in_stride,
            quantize_a8 ? 1u : 0u,
            unit_scaled ? 1u : 0u,
            1.0f / 16384.0f,
        };

        struct JoinParams {
            uint32_t tokens;
            uint32_t hidden_size;
            uint32_t n_segs;
            uint32_t has_tail;
            uint32_t has_unit_scale;
            uint32_t surf_in_stride_elems;
            uint32_t strides[8];
        };
        JoinParams join_p{};
        join_p.tokens = tokens;
        join_p.hidden_size = hidden_size;
        join_p.n_segs = n_segs;
        join_p.has_tail = 0u;
        join_p.has_unit_scale = unit_scaled ? 1u : 0u;
        join_p.surf_in_stride_elems = in_stride;
        for (uint32_t i = 0; i < n_segs; ++i) {
            join_p.strides[i] = seg_out_surfs[i]->bytes_per_row / seg_out_surfs[i]->bytes_per_elem;
        }

        auto* wave = new FastAneWave();
        wave->ctx = ctx;
        wave->tokens = tokens;
        wave->hidden_size = hidden_size;
        wave->n_segs = n_segs;
        wave->timeout_ms = timeout_ms ? timeout_ms : 5000u;
        wave->use_shared_event = (use_shared_event != 0);
        wave->buf_host_out = buf_host_out;
        wave->buf_dummy = buf_dummy;
        wave->t_start_ms = now_ms();

        const MTLSize rot_grid = MTLSizeMake(tokens, 1, 1);
        const MTLSize rot_tg = MTLSizeMake(192, 1, 1);
        const MTLSize join_grid = MTLSizeMake(tokens, hidden_size, 1);
        const MTLSize join_tg = MTLSizeMake(std::min<NSUInteger>(32, tokens), std::min<NSUInteger>(8, hidden_size), 1);

        if (wave->use_shared_event) {
            const uint64_t v_cur = [ctx->shared_event signaledValue];
            const uint64_t v_next = ctx->event_counter.load() + 1;
            const uint64_t wait_val = (v_next > v_cur) ? v_next : (v_cur + 1);
            wave->final_sig_val = wait_val + n_segs;
            ctx->event_counter.store(wave->final_sig_val);

            id<MTLCommandBuffer> cmdA = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> encA = [cmdA computeCommandEncoder];
            [encA setComputePipelineState:ctx->pso_rotate_pack];
            [encA setBuffer:buf_host_in offset:0 atIndex:0];
            [encA setBuffer:buf_signs offset:0 atIndex:1];
            [encA setBuffer:buf_surf_in offset:0 atIndex:2];
            [encA setBytes:&rot_p length:sizeof(rot_p) atIndex:3];
            [encA dispatchThreadgroups:rot_grid threadsPerThreadgroup:rot_tg];
            [encA endEncoding];
            [cmdA encodeSignalEvent:ctx->shared_event value:wait_val];

            id<MTLCommandBuffer> cmdB = [ctx->queue commandBuffer];
            [cmdB encodeWaitForEvent:ctx->shared_event value:wave->final_sig_val];
            id<MTLComputeCommandEncoder> encB = [cmdB computeCommandEncoder];
            [encB setComputePipelineState:ctx->pso_join];
            for (uint32_t i = 0; i < 5; ++i) {
                [encB setBuffer:(i < n_segs ? seg_out_bufs[i] : buf_dummy) offset:0 atIndex:i];
            }
            [encB setBuffer:buf_dummy offset:0 atIndex:5];
            [encB setBuffer:buf_host_out offset:0 atIndex:6];
            [encB setBytes:&join_p length:sizeof(join_p) atIndex:7];
            [encB setBuffer:buf_surf_in offset:0 atIndex:8];
            [encB dispatchThreads:join_grid threadsPerThreadgroup:join_tg];
            [encB endEncoding];

            wave->gpu_b_sem = dispatch_semaphore_create(0);
            FastAneWave* raw_wave = wave;
            [cmdB addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                raw_wave->gpu_b_err = [cb error];
                dispatch_semaphore_signal(raw_wave->gpu_b_sem);
            }];

            const double t_g0 = now_ms();
            [cmdA commit];
            [cmdB commit];
            wave->gpu_submit_ms = now_ms() - t_g0;

            wave->ane_sems.resize(n_segs);
            wave->ane_ok.assign(n_segs, NO);
            const double t_a0 = now_ms();
            auto client = static_cast<id<FastAneClientSPI>>(ctx->client);
            for (uint32_t i = 0; i < n_segs; ++i) {
                const uint64_t seg_wait = wait_val + i;
                const uint64_t seg_sig = seg_wait + 1;
                id wait_ev = [static_cast<id<FastAneSharedWaitEventSPI>>(ctx->ane.cls_wait_event)
                    waitEventWithValue:seg_wait
                           sharedEvent:ctx->shared_event
                             eventType:0];
                id sig_ev = [static_cast<id<FastAneSharedSignalEventSPI>>(ctx->ane.cls_signal_event)
                    signalEventWithValue:seg_sig
                             symbolIndex:0
                               eventType:0
                             sharedEvent:ctx->shared_event];
                id shared_evs = [static_cast<id<FastAneSharedEventsSPI>>(ctx->ane.cls_shared_events)
                    sharedEventsWithSignalEvents:@[sig_ev]
                                      waitEvents:@[wait_ev]];
                FastAneSurface* ins[] = {surf_in};
                FastAneSurface* outs[] = {seg_out_surfs[i]};
                id req = build_ane_request(ctx->ane, ins, 1, outs, 1, shared_evs, seg_sig);
                if (!wait_ev || !sig_ev || !shared_evs || !req) {
                    if ([ctx->shared_event signaledValue] < wave->final_sig_val) {
                        [ctx->shared_event setSignaledValue:wave->final_sig_val];
                    }
                    dispatch_semaphore_wait(wave->gpu_b_sem,
                                            dispatch_time(DISPATCH_TIME_NOW, 1000LL * NSEC_PER_MSEC));
                    set_error(err_buf, err_cap, "failed creating segment _ANERequest");
                    delete wave;
                    return FAST_ANE_EVAL_FAILED;
                }
                dispatch_semaphore_t seg_sem = dispatch_semaphore_create(0);
                wave->ane_sems[i] = seg_sem;
                [static_cast<id<FastAneRequestSPI>>(req) setCompletionHandler:^(BOOL success, NSError* error) {
                    raw_wave->ane_ok[i] = success;
                    if (!success && !raw_wave->ane_err) {
                        raw_wave->ane_err = error;
                    }
                    dispatch_semaphore_signal(seg_sem);
                }];
                NSError* eval_err = nil;
                if (![client evaluateWithModel:seg_progs[i]->model
                                       options:@{}
                                       request:req
                                           qos:seg_progs[i]->qos
                                         error:&eval_err]) {
                    if ([ctx->shared_event signaledValue] < wave->final_sig_val) {
                        [ctx->shared_event setSignaledValue:wave->final_sig_val];
                    }
                    dispatch_semaphore_wait(wave->gpu_b_sem,
                                            dispatch_time(DISPATCH_TIME_NOW, 1000LL * NSEC_PER_MSEC));
                    set_error(err_buf,
                              err_cap,
                              std::string("segment evaluateWithModel failed: ") +
                                  (eval_err ? [[eval_err localizedDescription] UTF8String] : "unknown"));
                    delete wave;
                    return FAST_ANE_EVAL_FAILED;
                }
            }
            wave->ane_submit_ms = now_ms() - t_a0;
        } else {
            // CPU-synchronized path: rotate_pack -> wait -> ANE evals -> join -> wait
            const double t_g0 = now_ms();
            id<MTLCommandBuffer> cmdA = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> encA = [cmdA computeCommandEncoder];
            [encA setComputePipelineState:ctx->pso_rotate_pack];
            [encA setBuffer:buf_host_in offset:0 atIndex:0];
            [encA setBuffer:buf_signs offset:0 atIndex:1];
            [encA setBuffer:buf_surf_in offset:0 atIndex:2];
            [encA setBytes:&rot_p length:sizeof(rot_p) atIndex:3];
            [encA dispatchThreadgroups:rot_grid threadsPerThreadgroup:rot_tg];
            [encA endEncoding];
            [cmdA commit];
            [cmdA waitUntilCompleted];
            wave->gpu_submit_ms = now_ms() - t_g0;
            if ([cmdA error]) {
                set_error(err_buf, err_cap, [[[cmdA error] localizedDescription] UTF8String]);
                delete wave;
                return FAST_ANE_METAL_FAILED;
            }

            const double t_a0 = now_ms();
            for (uint32_t i = 0; i < n_segs; ++i) {
                FastAneSurface* ins[] = {surf_in};
                FastAneSurface* outs[] = {seg_out_surfs[i]};
                double seg_ms = 0.0;
                FastAneStatus st = fast_ane_program_eval(
                    ctx, seg_progs[i], ins, 1, outs, 1, &seg_ms, err_buf, err_cap);
                if (st != FAST_ANE_OK) {
                    delete wave;
                    return st;
                }
            }
            wave->ane_submit_ms = now_ms() - t_a0;

            id<MTLCommandBuffer> cmdB = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> encB = [cmdB computeCommandEncoder];
            [encB setComputePipelineState:ctx->pso_join];
            for (uint32_t i = 0; i < 5; ++i) {
                [encB setBuffer:(i < n_segs ? seg_out_bufs[i] : buf_dummy) offset:0 atIndex:i];
            }
            [encB setBuffer:buf_dummy offset:0 atIndex:5];
            [encB setBuffer:buf_host_out offset:0 atIndex:6];
            [encB setBytes:&join_p length:sizeof(join_p) atIndex:7];
            [encB setBuffer:buf_surf_in offset:0 atIndex:8];
            [encB dispatchThreads:join_grid threadsPerThreadgroup:join_tg];
            [encB endEncoding];
            [cmdB commit];
            [cmdB waitUntilCompleted];
            if ([cmdB error]) {
                set_error(err_buf, err_cap, [[[cmdB error] localizedDescription] UTF8String]);
                delete wave;
                return FAST_ANE_METAL_FAILED;
            }
        }

        *out_wave = wave;
        return FAST_ANE_OK;
    }
}

FastAneStatus fast_ane_ffn_wave_finish(FastAneWave* wave,
                                       const uint16_t* opt_gpu_tail_fp16,
                                       uint16_t* host_out_fp16,
                                       FastAneWaveMetrics* out_metrics,
                                       char* err_buf,
                                       size_t err_cap) {
    if (!wave || !host_out_fp16) {
        set_error(err_buf, err_cap, "invalid arguments to fast_ane_ffn_wave_finish");
        return FAST_ANE_INVALID_ARG;
    }

    @autoreleasepool {
        const double t_wait0 = now_ms();
        if (wave->use_shared_event) {
            const dispatch_time_t deadline =
                dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(wave->timeout_ms) * NSEC_PER_MSEC);
            for (uint32_t i = 0; i < wave->n_segs; ++i) {
                if (dispatch_semaphore_wait(wave->ane_sems[i], deadline) != 0) {
                    if ([wave->ctx->shared_event signaledValue] < wave->final_sig_val) {
                        [wave->ctx->shared_event setSignaledValue:wave->final_sig_val];
                    }
                    dispatch_semaphore_wait(wave->gpu_b_sem,
                                            dispatch_time(DISPATCH_TIME_NOW, 1000LL * NSEC_PER_MSEC));
                    set_error(err_buf, err_cap, "ANE wave segment timed out");
                    return FAST_ANE_TIMEOUT;
                }
                if (!wave->ane_ok[i]) {
                    if ([wave->ctx->shared_event signaledValue] < wave->final_sig_val) {
                        [wave->ctx->shared_event setSignaledValue:wave->final_sig_val];
                    }
                    dispatch_semaphore_wait(wave->gpu_b_sem,
                                            dispatch_time(DISPATCH_TIME_NOW, 1000LL * NSEC_PER_MSEC));
                    set_error(err_buf,
                              err_cap,
                              std::string("ANE wave segment error: ") +
                                  (wave->ane_err ? [[wave->ane_err localizedDescription] UTF8String] : "unknown"));
                    return FAST_ANE_EVAL_FAILED;
                }
            }
            if (dispatch_semaphore_wait(wave->gpu_b_sem, deadline) != 0) {
                set_error(err_buf, err_cap, "Metal join command buffer timed out");
                return FAST_ANE_TIMEOUT;
            }
            if (wave->gpu_b_err) {
                set_error(err_buf, err_cap, [[wave->gpu_b_err localizedDescription] UTF8String]);
                return FAST_ANE_METAL_FAILED;
            }
        }
        const double finish_wait_ms = now_ms() - t_wait0;

        const size_t n_elems = static_cast<size_t>(wave->tokens) * wave->hidden_size;
        const auto* ane_ptr = static_cast<const __fp16*>([wave->buf_host_out contents]);
        if (opt_gpu_tail_fp16) {
            const auto* tail_ptr = reinterpret_cast<const __fp16*>(opt_gpu_tail_fp16);
            auto* dst_ptr = reinterpret_cast<__fp16*>(host_out_fp16);
            for (size_t i = 0; i < n_elems; ++i) {
                dst_ptr[i] = static_cast<__fp16>(static_cast<float>(ane_ptr[i]) + static_cast<float>(tail_ptr[i]));
            }
        } else {
            std::memcpy(host_out_fp16, ane_ptr, n_elems * sizeof(uint16_t));
        }

        const double total_wall_ms = now_ms() - wave->t_start_ms;
        if (out_metrics) {
            out_metrics->gpu_submit_ms = wave->gpu_submit_ms;
            out_metrics->ane_submit_ms = wave->ane_submit_ms;
            out_metrics->finish_wait_ms = finish_wait_ms;
            out_metrics->total_wall_ms = total_wall_ms;
            out_metrics->used_shared_event = wave->use_shared_event ? 1 : 0;
        }
        return FAST_ANE_OK;
    }
}

void fast_ane_ffn_wave_destroy(FastAneWave* wave) {
    delete wave;
}

}  // extern "C"
