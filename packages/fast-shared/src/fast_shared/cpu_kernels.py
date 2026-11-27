"""Cross-platform C++17 retrieval & BM25 kernels for Apple Silicon and non-Apple CPUs.

Provides ISA capability detection (`detect_cpu_isa`), backend resolution (`resolve_cpu_backend`),
and cached JIT-compiled C++17 shared libraries spanning:
  - ``neon`` (ARMv8.2/8.4-A DotProd ``vdotq_s32`` + 64B NEON Hamming; default on M-series & Graviton)
  - ``neon_i8mm`` (ARMv8.6-A ``vmmlaq_s32`` opt-in; disabled in auto due to permute overhead on M4 Pro)
  - ``avx512`` (x86_64 AVX-512 BW + VNNI ``vpdpbusd`` + ``vpopcntq``)
  - ``avx2`` (x86_64 AVX2 non-saturating ``cvtepi8_epi16`` + ``madd_epi16`` + ``popcntll``)
  - ``portable_cpp`` (pure multi-core C++17 scalar + ``popcnt64``, works on any CPU architecture)
  - ``numpy`` (pure Python/NumPy fallback)
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Literal, Mapping

CpuBackendMode = Literal["auto", "neon_i8mm", "neon", "avx512", "avx2", "portable_cpp", "numpy"]
VALID_CPU_BACKEND_MODES: frozenset[str] = frozenset(
    {"auto", "neon_i8mm", "neon", "avx512", "avx2", "portable_cpp", "numpy"}
)

_ISA_CACHE: dict[str, Any] | None = None
_ISA_LOCK = threading.Lock()
_LIB_CACHE: dict[tuple[str, str], ctypes.CDLL | None] = {}
_LIB_LOCK = threading.Lock()


def _sysctl_str(key: str) -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
        ).strip()
    except Exception:
        return ""


def _sysctl_int(key: str, default: int = 0) -> int:
    raw = _sysctl_str(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _parse_proc_cpuinfo_tokens(text: str, field_names: tuple[str, ...]) -> set[str]:
    tokens: set[str] = set()
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip().lower() in field_names:
            tokens.update(tok.strip().lower() for tok in v.split() if tok.strip())
    return tokens


def detect_cpu_isa(*, refresh: bool = False) -> dict[str, Any]:
    """Detect host CPU ISA capabilities across macOS, Linux, and generic platforms."""
    global _ISA_CACHE
    if _ISA_CACHE is not None and not refresh:
        return dict(_ISA_CACHE)
    with _ISA_LOCK:
        if _ISA_CACHE is not None and not refresh:
            return dict(_ISA_CACHE)

        sys_name = platform.system()
        mach = platform.machine().lower()
        is_arm64 = mach in ("arm64", "aarch64")
        is_x86_64 = mach in ("x86_64", "amd64")

        cpu_count = int(os.cpu_count() or 1)
        p_cores = cpu_count
        arm_dotprod = False
        arm_i8mm = False
        x86_avx2 = False
        x86_popcnt = False
        x86_avx512_vnni = False
        x86_avx512_vpopcntdq = False

        if sys_name == "Darwin":
            if is_arm64:
                p_val = _sysctl_int("hw.perflevel0.physicalcpu", 0)
                if p_val > 0:
                    p_cores = p_val
                arm_dotprod = _sysctl_int("hw.optional.arm.FEAT_DotProd", 0) == 1
                arm_i8mm = _sysctl_int("hw.optional.arm.FEAT_I8MM", 0) == 1
            elif is_x86_64:
                p_val = _sysctl_int("hw.physicalcpu", 0)
                if p_val > 0:
                    p_cores = p_val
                feats = {
                    t.lower()
                    for t in (
                        _sysctl_str("machdep.cpu.features")
                        + " "
                        + _sysctl_str("machdep.cpu.leaf7_features")
                    ).split()
                    if t
                }
                x86_avx2 = "avx2" in feats
                x86_popcnt = "popcnt" in feats
                x86_avx512_vnni = ("avx512f" in feats and "avx512bw" in feats and "avx512vnni" in feats)
                x86_avx512_vpopcntdq = "avx512vpopcntdq" in feats
        elif sys_name == "Linux":
            cpuinfo_path = Path("/proc/cpuinfo")
            if cpuinfo_path.is_file():
                info_txt = cpuinfo_path.read_text(errors="replace")
                if is_arm64:
                    feats = _parse_proc_cpuinfo_tokens(info_txt, ("features",))
                    arm_dotprod = "asimddp" in feats
                    arm_i8mm = "i8mm" in feats
                elif is_x86_64:
                    flags = _parse_proc_cpuinfo_tokens(info_txt, ("flags",))
                    x86_avx2 = "avx2" in flags
                    x86_popcnt = "popcnt" in flags
                    x86_avx512_vnni = (
                        "avx512f" in flags and "avx512bw" in flags and "avx512_vnni" in flags
                    )
                    x86_avx512_vpopcntdq = "avx512_vpopcntdq" in flags

        has_cxx = bool(_find_cxx_compiler())
        if not has_cxx:
            recommended = "numpy"
        elif is_arm64 and arm_dotprod:
            recommended = "neon"
        elif is_x86_64 and x86_avx512_vnni and x86_avx512_vpopcntdq:
            recommended = "avx512"
        elif is_x86_64 and x86_avx2 and x86_popcnt:
            recommended = "avx2"
        else:
            recommended = "portable_cpp"

        info: dict[str, Any] = {
            "system": sys_name,
            "machine": mach,
            "cpu_count": cpu_count,
            "p_cores": max(1, p_cores),
            "has_cxx": has_cxx,
            "arm_dotprod": arm_dotprod,
            "arm_i8mm": arm_i8mm,
            "x86_avx2": x86_avx2,
            "x86_popcnt": x86_popcnt,
            "x86_avx512_vnni": x86_avx512_vnni,
            "x86_avx512_vpopcntdq": x86_avx512_vpopcntdq,
            "recommended_backend": recommended,
        }
        _ISA_CACHE = info
        return dict(info)


def _find_cxx_compiler() -> str | None:
    for candidate in ("clang++", "g++", "c++"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def resolve_cpu_backend(
    cpu_backend: CpuBackendMode = "auto",
    *,
    use_neon: bool = True,
    isa_info: Mapping[str, Any] | None = None,
) -> str:
    """Resolve effective CPU kernel backend (`neon`, `neon_i8mm`, `avx512`, `avx2`, `portable_cpp`, or `numpy`)."""
    req = str(cpu_backend).strip().lower()
    if req not in VALID_CPU_BACKEND_MODES:
        raise ValueError(
            f"cpu_backend must be one of {sorted(VALID_CPU_BACKEND_MODES)}, got {cpu_backend!r}"
        )
    if not use_neon or os.environ.get("FAST_SILICON_DISABLE_NEON_INDEX") == "1":
        return "numpy"

    if req == "auto":
        env_req = os.environ.get("FAST_SILICON_CPU_BACKEND", "").strip().lower()
        if env_req:
            if env_req not in VALID_CPU_BACKEND_MODES:
                raise ValueError(
                    f"FAST_SILICON_CPU_BACKEND must be one of {sorted(VALID_CPU_BACKEND_MODES)}, got {env_req!r}"
                )
            req = env_req

    if req == "numpy":
        return "numpy"

    isa = dict(isa_info) if isa_info is not None else detect_cpu_isa()
    if not isa.get("has_cxx", True):
        return "numpy"

    if req == "auto":
        return str(isa.get("recommended_backend") or "portable_cpp")

    mach = str(isa.get("machine") or "").lower()
    is_arm64 = mach in ("arm64", "aarch64")
    is_x86_64 = mach in ("x86_64", "amd64")

    if req == "neon_i8mm":
        if is_arm64 and isa.get("arm_i8mm"):
            return "neon_i8mm"
        return str(isa.get("recommended_backend") or "portable_cpp")
    if req == "neon":
        if is_arm64 and isa.get("arm_dotprod"):
            return "neon"
        return str(isa.get("recommended_backend") or "portable_cpp")
    if req == "avx512":
        if is_x86_64 and isa.get("x86_avx512_vnni") and isa.get("x86_avx512_vpopcntdq"):
            return "avx512"
        return str(isa.get("recommended_backend") or "portable_cpp")
    if req == "avx2":
        if is_x86_64 and isa.get("x86_avx2") and isa.get("x86_popcnt"):
            return "avx2"
        return str(isa.get("recommended_backend") or "portable_cpp")
    return "portable_cpp"


def cpu_compile_flags(
    mode: str,
    *,
    system: str | None = None,
    arch: str | None = None,
    shared: bool = True,
) -> list[str]:
    """Return deterministic C++17 compiler flags for a target CPU backend and architecture."""
    eff_sys = system or platform.system()
    eff_arch = (arch or platform.machine()).lower()
    flags: list[str] = ["-O3", "-std=c++17", "-Wall", "-Wextra"]
    if shared:
        if eff_sys == "Darwin":
            flags.append("-dynamiclib")
        else:
            flags.extend(["-shared", "-fPIC"])
        if eff_sys == "Linux":
            flags.append("-pthread")
    if eff_sys == "Darwin":
        darwin_arch = "arm64" if eff_arch in ("arm64", "aarch64") else ("x86_64" if eff_arch in ("x86_64", "amd64") else eff_arch)
        flags.extend(["-arch", darwin_arch])

    if eff_arch in ("arm64", "aarch64"):
        if mode == "neon_i8mm":
            if eff_sys == "Darwin":
                flags.extend(["-march=armv8.6-a+dotprod+i8mm+fp16", "-DFS_ENABLE_I8MM=1", "-DFS_ENABLE_DOTPROD=1"])
            else:
                flags.extend(["-march=armv8.6-a+dotprod+i8mm", "-DFS_ENABLE_I8MM=1", "-DFS_ENABLE_DOTPROD=1"])
        elif mode == "neon":
            if eff_sys == "Darwin":
                flags.extend(["-march=armv8.4-a+dotprod+fp16", "-DFS_ENABLE_I8MM=0", "-DFS_ENABLE_DOTPROD=1"])
            else:
                flags.extend(["-march=armv8.2-a+dotprod", "-DFS_ENABLE_I8MM=0", "-DFS_ENABLE_DOTPROD=1"])
        else:
            flags.extend(["-DFS_ENABLE_I8MM=0", "-DFS_ENABLE_DOTPROD=0"])
    elif eff_arch in ("x86_64", "amd64"):
        if mode == "avx512":
            flags.extend(
                [
                    "-mavx512f",
                    "-mavx512bw",
                    "-mavx512vnni",
                    "-mavx512vpopcntdq",
                    "-DFS_ENABLE_AVX512=1",
                ]
            )
        elif mode == "avx2":
            flags.extend(["-mavx2", "-mpopcnt", "-DFS_ENABLE_AVX2=1"])
        else:
            flags.extend(["-DFS_ENABLE_AVX512=0", "-DFS_ENABLE_AVX2=0"])
    return flags


_RETRIEVAL_CPP_SOURCE = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define FS_ARM64 1
#else
#define FS_ARM64 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define FS_X86_64 1
#else
#define FS_X86_64 0
#endif

#ifndef FS_ENABLE_I8MM
#define FS_ENABLE_I8MM 0
#endif

#ifndef FS_ENABLE_DOTPROD
#define FS_ENABLE_DOTPROD 1
#endif

#ifndef FS_ENABLE_AVX512
#define FS_ENABLE_AVX512 0
#endif

#ifndef FS_ENABLE_AVX2
#define FS_ENABLE_AVX2 0
#endif

namespace {

static inline uint32_t popcnt64(uint64_t x) {
#if defined(_MSC_VER)
    return static_cast<uint32_t>(__popcnt64(x));
#else
    return static_cast<uint32_t>(__builtin_popcountll(x));
#endif
}

static inline float fp16_bits_to_f32(uint16_t h) {
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__FLT16_MANT_DIG__)
    _Float16 f;
    std::memcpy(&f, &h, sizeof(f));
    return static_cast<float>(f);
#else
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t out = 0;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign;
        } else {
            exp = 1u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            out = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float res = 0.0f;
    std::memcpy(&res, &out, sizeof(res));
    return res;
#endif
}

static int detect_worker_threads(int max_hint) {
    if (max_hint == 1) return 1;
    int threads = 0;
#if defined(__APPLE__)
    int pcores = 0;
    size_t sz = sizeof(pcores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &sz, nullptr, 0) == 0 && pcores > 0) {
        threads = pcores;
    }
#endif
    if (threads <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        threads = hw > 0u ? static_cast<int>(hw) : 4;
    }
    if (max_hint > 0 && max_hint < threads) {
        threads = max_hint;
    }
    return std::max(1, std::min(threads, 64));
}

template <typename Fn>
static void parallel_for_chunks(int n_tasks, Fn&& fn) {
    if (n_tasks <= 1) {
        if (n_tasks == 1) fn(0);
        return;
    }
#if defined(__APPLE__)
    struct Ctx {
        Fn* fp;
    } ctx{&fn};
    dispatch_apply_f(
        static_cast<size_t>(n_tasks),
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        &ctx,
        [](void* raw, size_t idx) {
            (*static_cast<Ctx*>(raw)->fp)(static_cast<int>(idx));
        }
    );
#else
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_tasks - 1));
    for (int t = 1; t < n_tasks; ++t) {
        workers.emplace_back([&, t]() { fn(t); });
    }
    fn(0);
    for (auto& w : workers) {
        w.join();
    }
#endif
}

struct ScoreId {
    float score;
    int32_t id;
};

static inline bool is_better_hit(float sc_a, int32_t id_a, float sc_b, int32_t id_b) {
    if (sc_a != sc_b) return sc_a > sc_b;
    return id_a < id_b;
}

struct HitWorseFirst {
    bool operator()(const ScoreId& lhs, const ScoreId& rhs) const {
        return is_better_hit(lhs.score, lhs.id, rhs.score, rhs.id);
    }
};

static inline void topk_push(std::vector<ScoreId>& heap, int eff_k, float score, int32_t id) {
    if (eff_k <= 0) return;
    if (static_cast<int>(heap.size()) < eff_k) {
        heap.push_back(ScoreId{score, id});
        std::push_heap(heap.begin(), heap.end(), HitWorseFirst{});
    } else if (is_better_hit(score, id, heap[0].score, heap[0].id)) {
        std::pop_heap(heap.begin(), heap.end(), HitWorseFirst{});
        heap.back() = ScoreId{score, id};
        std::push_heap(heap.begin(), heap.end(), HitWorseFirst{});
    }
}

static inline void topk_finalize(
    std::vector<ScoreId>& heap,
    int eff_k,
    int top_k,
    int32_t* out_idx,
    float* out_sc
) {
    std::sort(heap.begin(), heap.end(), [](const ScoreId& a, const ScoreId& b) {
        return is_better_hit(a.score, a.id, b.score, b.id);
    });
    const int n_fill = std::min(eff_k, static_cast<int>(heap.size()));
    for (int k = 0; k < n_fill; ++k) {
        out_idx[k] = heap[k].id;
        out_sc[k] = heap[k].score;
    }
    for (int k = n_fill; k < top_k; ++k) {
        out_idx[k] = -1;
        out_sc[k] = -1e30f;
    }
}

static inline int32_t dot_i8_1(const int8_t* __restrict__ qr, const int8_t* __restrict__ dr, int D) {
    int k = 0;
#if FS_ARM64 && FS_ENABLE_DOTPROD
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);
    for (; k + 64 <= D; k += 64) {
        acc0 = vdotq_s32(acc0, vld1q_s8(qr + k + 0), vld1q_s8(dr + k + 0));
        acc1 = vdotq_s32(acc1, vld1q_s8(qr + k + 16), vld1q_s8(dr + k + 16));
        acc2 = vdotq_s32(acc2, vld1q_s8(qr + k + 32), vld1q_s8(dr + k + 32));
        acc3 = vdotq_s32(acc3, vld1q_s8(qr + k + 48), vld1q_s8(dr + k + 48));
    }
    for (; k + 16 <= D; k += 16) {
        acc0 = vdotq_s32(acc0, vld1q_s8(qr + k), vld1q_s8(dr + k));
    }
    int32_t dot = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
#elif FS_X86_64 && FS_ENABLE_AVX512
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    const __m512i sign_flip = _mm512_set1_epi8(static_cast<char>(0x80));
    __m512i q_sum_v = _mm512_setzero_si512();
    const __m512i ones_i8 = _mm512_set1_epi8(1);
    for (; k + 128 <= D; k += 128) {
        __m512i q0 = _mm512_loadu_si512(qr + k);
        __m512i d0 = _mm512_xor_si512(_mm512_loadu_si512(dr + k), sign_flip);
        __m512i q1 = _mm512_loadu_si512(qr + k + 64);
        __m512i d1 = _mm512_xor_si512(_mm512_loadu_si512(dr + k + 64), sign_flip);
        acc0 = _mm512_dpbusd_epi32(acc0, d0, q0);
        acc1 = _mm512_dpbusd_epi32(acc1, d1, q1);
        q_sum_v = _mm512_dpbusd_epi32(q_sum_v, ones_i8, q0);
        q_sum_v = _mm512_dpbusd_epi32(q_sum_v, ones_i8, q1);
    }
    for (; k + 64 <= D; k += 64) {
        __m512i q0 = _mm512_loadu_si512(qr + k);
        __m512i d0 = _mm512_xor_si512(_mm512_loadu_si512(dr + k), sign_flip);
        acc0 = _mm512_dpbusd_epi32(acc0, d0, q0);
        q_sum_v = _mm512_dpbusd_epi32(q_sum_v, ones_i8, q0);
    }
    int32_t biased_dot = _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
    int32_t q_sum = _mm512_reduce_add_epi32(q_sum_v);
    int32_t dot = biased_dot - 128 * q_sum;
#elif FS_X86_64 && FS_ENABLE_AVX2
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    for (; k + 32 <= D; k += 32) {
        __m256i q0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(qr + k)));
        __m256i d0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dr + k)));
        __m256i q1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(qr + k + 16)));
        __m256i d1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dr + k + 16)));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q0, d0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(q1, d1));
    }
    for (; k + 16 <= D; k += 16) {
        __m256i q0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(qr + k)));
        __m256i d0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dr + k)));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q0, d0));
    }
    __m256i sum256 = _mm256_add_epi32(acc0, acc1);
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0x4E));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0xB1));
    int32_t dot = _mm_cvtsi128_si32(sum128);
#else
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (; k + 4 <= D; k += 4) {
        s0 += static_cast<int32_t>(qr[k + 0]) * static_cast<int32_t>(dr[k + 0]);
        s1 += static_cast<int32_t>(qr[k + 1]) * static_cast<int32_t>(dr[k + 1]);
        s2 += static_cast<int32_t>(qr[k + 2]) * static_cast<int32_t>(dr[k + 2]);
        s3 += static_cast<int32_t>(qr[k + 3]) * static_cast<int32_t>(dr[k + 3]);
    }
    int32_t dot = (s0 + s1) + (s2 + s3);
#endif
    for (; k < D; ++k) {
        dot += static_cast<int32_t>(qr[k]) * static_cast<int32_t>(dr[k]);
    }
    return dot;
}

static inline void dot_i8_4q_1d(
    const int8_t* __restrict__ q0,
    const int8_t* __restrict__ q1,
    const int8_t* __restrict__ q2,
    const int8_t* __restrict__ q3,
    const int8_t* __restrict__ dr,
    int D,
    int32_t* __restrict__ out4
) {
    int k = 0;
#if FS_ARM64 && FS_ENABLE_DOTPROD
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int32x4_t b0 = vdupq_n_s32(0), b1 = vdupq_n_s32(0), b2 = vdupq_n_s32(0), b3 = vdupq_n_s32(0);
    for (; k + 32 <= D; k += 32) {
        int8x16_t d_lo = vld1q_s8(dr + k);
        int8x16_t d_hi = vld1q_s8(dr + k + 16);
        a0 = vdotq_s32(a0, vld1q_s8(q0 + k), d_lo);
        b0 = vdotq_s32(b0, vld1q_s8(q0 + k + 16), d_hi);
        a1 = vdotq_s32(a1, vld1q_s8(q1 + k), d_lo);
        b1 = vdotq_s32(b1, vld1q_s8(q1 + k + 16), d_hi);
        a2 = vdotq_s32(a2, vld1q_s8(q2 + k), d_lo);
        b2 = vdotq_s32(b2, vld1q_s8(q2 + k + 16), d_hi);
        a3 = vdotq_s32(a3, vld1q_s8(q3 + k), d_lo);
        b3 = vdotq_s32(b3, vld1q_s8(q3 + k + 16), d_hi);
    }
    for (; k + 16 <= D; k += 16) {
        int8x16_t d_lo = vld1q_s8(dr + k);
        a0 = vdotq_s32(a0, vld1q_s8(q0 + k), d_lo);
        a1 = vdotq_s32(a1, vld1q_s8(q1 + k), d_lo);
        a2 = vdotq_s32(a2, vld1q_s8(q2 + k), d_lo);
        a3 = vdotq_s32(a3, vld1q_s8(q3 + k), d_lo);
    }
    int32_t s0 = vaddvq_s32(vaddq_s32(a0, b0));
    int32_t s1 = vaddvq_s32(vaddq_s32(a1, b1));
    int32_t s2 = vaddvq_s32(vaddq_s32(a2, b2));
    int32_t s3 = vaddvq_s32(vaddq_s32(a3, b3));
#elif FS_X86_64 && FS_ENABLE_AVX512
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    const __m512i sign_flip = _mm512_set1_epi8(static_cast<char>(0x80));
    const __m512i ones_i8 = _mm512_set1_epi8(1);
    __m512i qs0 = _mm512_setzero_si512(), qs1 = _mm512_setzero_si512();
    __m512i qs2 = _mm512_setzero_si512(), qs3 = _mm512_setzero_si512();
    for (; k + 64 <= D; k += 64) {
        __m512i d = _mm512_xor_si512(_mm512_loadu_si512(dr + k), sign_flip);
        __m512i vq0 = _mm512_loadu_si512(q0 + k);
        __m512i vq1 = _mm512_loadu_si512(q1 + k);
        __m512i vq2 = _mm512_loadu_si512(q2 + k);
        __m512i vq3 = _mm512_loadu_si512(q3 + k);
        a0 = _mm512_dpbusd_epi32(a0, d, vq0); qs0 = _mm512_dpbusd_epi32(qs0, ones_i8, vq0);
        a1 = _mm512_dpbusd_epi32(a1, d, vq1); qs1 = _mm512_dpbusd_epi32(qs1, ones_i8, vq1);
        a2 = _mm512_dpbusd_epi32(a2, d, vq2); qs2 = _mm512_dpbusd_epi32(qs2, ones_i8, vq2);
        a3 = _mm512_dpbusd_epi32(a3, d, vq3); qs3 = _mm512_dpbusd_epi32(qs3, ones_i8, vq3);
    }
    int32_t s0 = _mm512_reduce_add_epi32(a0) - 128 * _mm512_reduce_add_epi32(qs0);
    int32_t s1 = _mm512_reduce_add_epi32(a1) - 128 * _mm512_reduce_add_epi32(qs1);
    int32_t s2 = _mm512_reduce_add_epi32(a2) - 128 * _mm512_reduce_add_epi32(qs2);
    int32_t s3 = _mm512_reduce_add_epi32(a3) - 128 * _mm512_reduce_add_epi32(qs3);
#elif FS_X86_64 && FS_ENABLE_AVX2
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    for (; k + 16 <= D; k += 16) {
        __m256i d = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(dr + k)));
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(d, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q0 + k)))));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(d, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q1 + k)))));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(d, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q2 + k)))));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(d, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q3 + k)))));
    }
    auto hsum = [](__m256i v) -> int32_t {
        __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
        return _mm_cvtsi128_si32(lo);
    };
    int32_t s0 = hsum(a0), s1 = hsum(a1), s2 = hsum(a2), s3 = hsum(a3);
#else
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#endif
    for (; k < D; ++k) {
        int32_t dv = static_cast<int32_t>(dr[k]);
        s0 += dv * static_cast<int32_t>(q0[k]);
        s1 += dv * static_cast<int32_t>(q1[k]);
        s2 += dv * static_cast<int32_t>(q2[k]);
        s3 += dv * static_cast<int32_t>(q3[k]);
    }
    out4[0] = s0; out4[1] = s1; out4[2] = s2; out4[3] = s3;
}

#if FS_ARM64 && FS_ENABLE_I8MM
static inline void dot_i8_4q_2d_i8mm(
    const int8_t* __restrict__ q0,
    const int8_t* __restrict__ q1,
    const int8_t* __restrict__ q2,
    const int8_t* __restrict__ q3,
    const int8_t* __restrict__ d0,
    const int8_t* __restrict__ d1,
    int D,
    int32_t* __restrict__ out_d0_4,
    int32_t* __restrict__ out_d1_4
) {
    int32x4_t c_0_01 = vdupq_n_s32(0);
    int32x4_t c_0_23 = vdupq_n_s32(0);
    int k = 0;
    for (; k + 16 <= D; k += 16) {
        int8x16_t vd0 = vld1q_s8(d0 + k);
        int8x16_t vd1 = vld1q_s8(d1 + k);
        int8x16_t vq0 = vld1q_s8(q0 + k);
        int8x16_t vq1 = vld1q_s8(q1 + k);
        int8x16_t vq2 = vld1q_s8(q2 + k);
        int8x16_t vq3 = vld1q_s8(q3 + k);

        int8x16_t d_lo = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s8(vd0), vreinterpretq_s64_s8(vd1)));
        int8x16_t d_hi = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s8(vd0), vreinterpretq_s64_s8(vd1)));
        int8x16_t q01_lo = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s8(vq0), vreinterpretq_s64_s8(vq1)));
        int8x16_t q01_hi = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s8(vq0), vreinterpretq_s64_s8(vq1)));
        int8x16_t q23_lo = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s8(vq2), vreinterpretq_s64_s8(vq3)));
        int8x16_t q23_hi = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s8(vq2), vreinterpretq_s64_s8(vq3)));

        c_0_01 = vmmlaq_s32(c_0_01, d_lo, q01_lo);
        c_0_01 = vmmlaq_s32(c_0_01, d_hi, q01_hi);
        c_0_23 = vmmlaq_s32(c_0_23, d_lo, q23_lo);
        c_0_23 = vmmlaq_s32(c_0_23, d_hi, q23_hi);
    }
    int32_t d0_0 = vgetq_lane_s32(c_0_01, 0), d0_1 = vgetq_lane_s32(c_0_01, 1);
    int32_t d1_0 = vgetq_lane_s32(c_0_01, 2), d1_1 = vgetq_lane_s32(c_0_01, 3);
    int32_t d0_2 = vgetq_lane_s32(c_0_23, 0), d0_3 = vgetq_lane_s32(c_0_23, 1);
    int32_t d1_2 = vgetq_lane_s32(c_0_23, 2), d1_3 = vgetq_lane_s32(c_0_23, 3);
    for (; k < D; ++k) {
        int32_t a0 = static_cast<int32_t>(d0[k]), a1 = static_cast<int32_t>(d1[k]);
        int32_t w0 = static_cast<int32_t>(q0[k]), w1 = static_cast<int32_t>(q1[k]);
        int32_t w2 = static_cast<int32_t>(q2[k]), w3 = static_cast<int32_t>(q3[k]);
        d0_0 += a0 * w0; d0_1 += a0 * w1; d0_2 += a0 * w2; d0_3 += a0 * w3;
        d1_0 += a1 * w0; d1_1 += a1 * w1; d1_2 += a1 * w2; d1_3 += a1 * w3;
    }
    out_d0_4[0] = d0_0; out_d0_4[1] = d0_1; out_d0_4[2] = d0_2; out_d0_4[3] = d0_3;
    out_d1_4[0] = d1_0; out_d1_4[1] = d1_1; out_d1_4[2] = d1_2; out_d1_4[3] = d1_3;
}

static inline void dot_i8_2q_2d_i8mm(
    const int8_t* __restrict__ q0,
    const int8_t* __restrict__ q1,
    const int8_t* __restrict__ d0,
    const int8_t* __restrict__ d1,
    int D,
    int32_t* __restrict__ out_d0_2,
    int32_t* __restrict__ out_d1_2
) {
    int32x4_t c = vdupq_n_s32(0);
    int k = 0;
    for (; k + 16 <= D; k += 16) {
        int8x16_t vd0 = vld1q_s8(d0 + k);
        int8x16_t vd1 = vld1q_s8(d1 + k);
        int8x16_t vq0 = vld1q_s8(q0 + k);
        int8x16_t vq1 = vld1q_s8(q1 + k);
        int8x16_t d_lo = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s8(vd0), vreinterpretq_s64_s8(vd1)));
        int8x16_t d_hi = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s8(vd0), vreinterpretq_s64_s8(vd1)));
        int8x16_t q_lo = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s8(vq0), vreinterpretq_s64_s8(vq1)));
        int8x16_t q_hi = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s8(vq0), vreinterpretq_s64_s8(vq1)));
        c = vmmlaq_s32(c, d_lo, q_lo);
        c = vmmlaq_s32(c, d_hi, q_hi);
    }
    int32_t s00 = vgetq_lane_s32(c, 0), s01 = vgetq_lane_s32(c, 1);
    int32_t s10 = vgetq_lane_s32(c, 2), s11 = vgetq_lane_s32(c, 3);
    for (; k < D; ++k) {
        int32_t a0 = static_cast<int32_t>(d0[k]), a1 = static_cast<int32_t>(d1[k]);
        int32_t b0 = static_cast<int32_t>(q0[k]), b1 = static_cast<int32_t>(q1[k]);
        s00 += a0 * b0; s01 += a0 * b1;
        s10 += a1 * b0; s11 += a1 * b1;
    }
    out_d0_2[0] = s00; out_d0_2[1] = s01;
    out_d1_2[0] = s10; out_d1_2[1] = s11;
}
#endif

static inline uint32_t hamming_distance(const uint8_t* __restrict__ qb, const uint8_t* __restrict__ db, int n_bytes) {
    uint32_t dist = 0;
    int b = 0;
#if FS_ARM64
    uint32x4_t acc = vdupq_n_u32(0);
    for (; b + 64 <= n_bytes; b += 64) {
        uint8x16_t p0 = vcntq_u8(veorq_u8(vld1q_u8(qb + b + 0), vld1q_u8(db + b + 0)));
        uint8x16_t p1 = vcntq_u8(veorq_u8(vld1q_u8(qb + b + 16), vld1q_u8(db + b + 16)));
        uint8x16_t p2 = vcntq_u8(veorq_u8(vld1q_u8(qb + b + 32), vld1q_u8(db + b + 32)));
        uint8x16_t p3 = vcntq_u8(veorq_u8(vld1q_u8(qb + b + 48), vld1q_u8(db + b + 48)));
        uint8x16_t sum8 = vaddq_u8(vaddq_u8(p0, p1), vaddq_u8(p2, p3));
        acc = vpadalq_u16(acc, vpaddlq_u8(sum8));
    }
    for (; b + 16 <= n_bytes; b += 16) {
        uint8x16_t p = vcntq_u8(veorq_u8(vld1q_u8(qb + b), vld1q_u8(db + b)));
        acc = vpadalq_u16(acc, vpaddlq_u8(p));
    }
    dist += vaddvq_u32(acc);
#elif FS_X86_64 && FS_ENABLE_AVX512
    __m512i acc = _mm512_setzero_si512();
    for (; b + 64 <= n_bytes; b += 64) {
        __m512i xr = _mm512_xor_si512(_mm512_loadu_si512(qb + b), _mm512_loadu_si512(db + b));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(xr));
    }
    dist += static_cast<uint32_t>(_mm512_reduce_add_epi64(acc));
#endif
    for (; b + 32 <= n_bytes; b += 32) {
        uint64_t q0, q1, q2, q3, d0, d1, d2, d3;
        std::memcpy(&q0, qb + b + 0, 8); std::memcpy(&d0, db + b + 0, 8);
        std::memcpy(&q1, qb + b + 8, 8); std::memcpy(&d1, db + b + 8, 8);
        std::memcpy(&q2, qb + b + 16, 8); std::memcpy(&d2, db + b + 16, 8);
        std::memcpy(&q3, qb + b + 24, 8); std::memcpy(&d3, db + b + 24, 8);
        dist += popcnt64(q0 ^ d0) + popcnt64(q1 ^ d1) + popcnt64(q2 ^ d2) + popcnt64(q3 ^ d3);
    }
    for (; b + 8 <= n_bytes; b += 8) {
        uint64_t q0, d0;
        std::memcpy(&q0, qb + b, 8);
        std::memcpy(&d0, db + b, 8);
        dist += popcnt64(q0 ^ d0);
    }
    for (; b < n_bytes; ++b) {
        dist += popcnt64(static_cast<uint64_t>(qb[b] ^ db[b]));
    }
    return dist;
}

static inline float reduce_window_scores(
    const float* __restrict__ w_scores,
    int n_win,
    int agg_mode,
    float hybrid_alpha
) {
    if (n_win <= 0) return -1e30f;
    if (agg_mode == 0) {
        float mx = w_scores[0];
        for (int i = 1; i < n_win; ++i) {
            if (w_scores[i] > mx) mx = w_scores[i];
        }
        return mx;
    }
    if (agg_mode == 3) {
        double sum = 0.0;
        for (int i = 0; i < n_win; ++i) sum += static_cast<double>(w_scores[i]);
        return static_cast<float>(sum / static_cast<double>(n_win));
    }
    if (agg_mode == 1) {
        float mx = w_scores[0];
        double sum = static_cast<double>(w_scores[0]);
        for (int i = 1; i < n_win; ++i) {
            float v = w_scores[i];
            if (v > mx) mx = v;
            sum += static_cast<double>(v);
        }
        float mean_v = static_cast<float>(sum / static_cast<double>(n_win));
        return hybrid_alpha * mx + (1.0f - hybrid_alpha) * mean_v;
    }
    if (n_win == 1) return w_scores[0];
    float m1 = w_scores[0], m2 = w_scores[1];
    if (m2 > m1) std::swap(m1, m2);
    for (int i = 2; i < n_win; ++i) {
        float v = w_scores[i];
        if (v > m1) { m2 = m1; m1 = v; }
        else if (v > m2) { m2 = v; }
    }
    return 0.5f * (m1 + m2);
}

static void score_int8_q_tile_slab(
    const int8_t* __restrict__ q_i8,
    const float* __restrict__ q_sc_f32,
    const int8_t* __restrict__ d_i8,
    const float* __restrict__ d_sc_f32,
    int Q,
    int q0,
    int q1,
    int d0,
    int d1,
    int N,
    int D,
    float* __restrict__ out_scores
) {
    (void)Q;
    int qi = q0;
    for (; qi + 4 <= q1; qi += 4) {
        const int8_t* qr0 = q_i8 + static_cast<size_t>(qi + 0) * D;
        const int8_t* qr1 = q_i8 + static_cast<size_t>(qi + 1) * D;
        const int8_t* qr2 = q_i8 + static_cast<size_t>(qi + 2) * D;
        const int8_t* qr3 = q_i8 + static_cast<size_t>(qi + 3) * D;
        const float qs0 = q_sc_f32[qi + 0], qs1 = q_sc_f32[qi + 1];
        const float qs2 = q_sc_f32[qi + 2], qs3 = q_sc_f32[qi + 3];
        float* o0 = out_scores + static_cast<size_t>(qi + 0) * N;
        float* o1 = out_scores + static_cast<size_t>(qi + 1) * N;
        float* o2 = out_scores + static_cast<size_t>(qi + 2) * N;
        float* o3 = out_scores + static_cast<size_t>(qi + 3) * N;
        int di = d0;
#if FS_ARM64 && FS_ENABLE_I8MM
        for (; di + 2 <= d1; di += 2) {
            int32_t d0_4[4], d1_4[4];
            dot_i8_4q_2d_i8mm(qr0, qr1, qr2, qr3, d_i8 + static_cast<size_t>(di) * D, d_i8 + static_cast<size_t>(di + 1) * D, D, d0_4, d1_4);
            const float ds0 = d_sc_f32[di], ds1 = d_sc_f32[di + 1];
            o0[di] = static_cast<float>(d0_4[0]) * (qs0 * ds0);
            o1[di] = static_cast<float>(d0_4[1]) * (qs1 * ds0);
            o2[di] = static_cast<float>(d0_4[2]) * (qs2 * ds0);
            o3[di] = static_cast<float>(d0_4[3]) * (qs3 * ds0);
            o0[di + 1] = static_cast<float>(d1_4[0]) * (qs0 * ds1);
            o1[di + 1] = static_cast<float>(d1_4[1]) * (qs1 * ds1);
            o2[di + 1] = static_cast<float>(d1_4[2]) * (qs2 * ds1);
            o3[di + 1] = static_cast<float>(d1_4[3]) * (qs3 * ds1);
        }
#endif
        for (; di < d1; ++di) {
            int32_t d4[4];
            dot_i8_4q_1d(qr0, qr1, qr2, qr3, d_i8 + static_cast<size_t>(di) * D, D, d4);
            const float ds = d_sc_f32[di];
            o0[di] = static_cast<float>(d4[0]) * (qs0 * ds);
            o1[di] = static_cast<float>(d4[1]) * (qs1 * ds);
            o2[di] = static_cast<float>(d4[2]) * (qs2 * ds);
            o3[di] = static_cast<float>(d4[3]) * (qs3 * ds);
        }
    }
#if FS_ARM64 && FS_ENABLE_I8MM
    for (; qi + 2 <= q1; qi += 2) {
        const int8_t* qr0 = q_i8 + static_cast<size_t>(qi + 0) * D;
        const int8_t* qr1 = q_i8 + static_cast<size_t>(qi + 1) * D;
        const float qs0 = q_sc_f32[qi + 0], qs1 = q_sc_f32[qi + 1];
        float* o0 = out_scores + static_cast<size_t>(qi + 0) * N;
        float* o1 = out_scores + static_cast<size_t>(qi + 1) * N;
        int di = d0;
        for (; di + 2 <= d1; di += 2) {
            int32_t d0_2[2], d1_2[2];
            dot_i8_2q_2d_i8mm(qr0, qr1, d_i8 + static_cast<size_t>(di) * D, d_i8 + static_cast<size_t>(di + 1) * D, D, d0_2, d1_2);
            const float ds0 = d_sc_f32[di], ds1 = d_sc_f32[di + 1];
            o0[di] = static_cast<float>(d0_2[0]) * (qs0 * ds0);
            o1[di] = static_cast<float>(d0_2[1]) * (qs1 * ds0);
            o0[di + 1] = static_cast<float>(d1_2[0]) * (qs0 * ds1);
            o1[di + 1] = static_cast<float>(d1_2[1]) * (qs1 * ds1);
        }
        for (; di < d1; ++di) {
            const int8_t* dr = d_i8 + static_cast<size_t>(di) * D;
            const float ds = d_sc_f32[di];
            o0[di] = static_cast<float>(dot_i8_1(qr0, dr, D)) * (qs0 * ds);
            o1[di] = static_cast<float>(dot_i8_1(qr1, dr, D)) * (qs1 * ds);
        }
    }
#endif
    for (; qi < q1; ++qi) {
        const int8_t* qr = q_i8 + static_cast<size_t>(qi) * D;
        const float qs = q_sc_f32[qi];
        float* o = out_scores + static_cast<size_t>(qi) * N;
        for (int di = d0; di < d1; ++di) {
            o[di] = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(di) * D, D)) * (qs * d_sc_f32[di]);
        }
    }
}

static inline int find_hist_cutoff(const int* hist_counts, int hist_len, int target_r, int& rem_eq) {
    int acc = 0;
    for (int b = 0; b < hist_len; ++b) {
        int c = hist_counts[b];
        if (acc + c >= target_r) {
            rem_eq = target_r - acc;
            return b;
        }
        acc += c;
    }
    rem_eq = 0;
    return hist_len - 1;
}

} // namespace

extern "C" {

void fast_cpu_int8_scores(
    const int8_t* __restrict__ q_i8,
    const uint16_t* __restrict__ q_sc_u16,
    const int8_t* __restrict__ d_i8,
    const uint16_t* __restrict__ d_sc_u16,
    int Q,
    int N,
    int D,
    int max_threads,
    float* __restrict__ out_scores
) {
    if (Q <= 0 || N <= 0) return;
    std::vector<float> q_sc(static_cast<size_t>(Q));
    std::vector<float> d_sc(static_cast<size_t>(N));
    for (int qi = 0; qi < Q; ++qi) q_sc[qi] = fp16_bits_to_f32(q_sc_u16[qi]);
    for (int di = 0; di < N; ++di) d_sc[di] = fp16_bits_to_f32(d_sc_u16[di]);

    const int workers = detect_worker_threads(max_threads);
    const uint64_t work = static_cast<uint64_t>(Q) * static_cast<uint64_t>(N) * static_cast<uint64_t>(std::max(1, D));
    if (workers <= 1 || work < (16ULL << 20)) {
        score_int8_q_tile_slab(q_i8, q_sc.data(), d_i8, d_sc.data(), Q, 0, Q, 0, N, N, D, out_scores);
        return;
    }

    if (Q >= workers * 2) {
        const int n_tasks = std::min(workers, (Q + 3) / 4);
        parallel_for_chunks(n_tasks, [&](int tid) {
            const int q_blocks = (Q + 3) / 4;
            const int b0 = (tid * q_blocks) / n_tasks;
            const int b1 = ((tid + 1) * q_blocks) / n_tasks;
            const int q0 = b0 * 4;
            const int q1 = std::min(Q, b1 * 4);
            if (q0 < q1) {
                score_int8_q_tile_slab(q_i8, q_sc.data(), d_i8, d_sc.data(), Q, q0, q1, 0, N, N, D, out_scores);
            }
        });
    } else {
        const int n_tasks = std::min(workers, std::max(1, N / 1024));
        parallel_for_chunks(n_tasks, [&](int tid) {
            const int d0 = static_cast<int>((static_cast<int64_t>(tid) * N) / n_tasks);
            const int d1 = static_cast<int>((static_cast<int64_t>(tid + 1) * N) / n_tasks);
            if (d0 < d1) {
                score_int8_q_tile_slab(q_i8, q_sc.data(), d_i8, d_sc.data(), Q, 0, Q, d0, d1, N, D, out_scores);
            }
        });
    }
}

void fast_cpu_int8_topk(
    const int8_t* __restrict__ q_i8,
    const uint16_t* __restrict__ q_sc_u16,
    const int8_t* __restrict__ d_i8,
    const uint16_t* __restrict__ d_sc_u16,
    int Q,
    int N,
    int D,
    int top_k,
    int max_threads,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_scores
) {
    if (Q <= 0 || top_k <= 0) return;
    if (N <= 0) {
        for (int qi = 0; qi < Q; ++qi) {
            for (int k = 0; k < top_k; ++k) {
                out_indices[static_cast<size_t>(qi) * top_k + k] = -1;
                out_scores[static_cast<size_t>(qi) * top_k + k] = -1e30f;
            }
        }
        return;
    }

    const int eff_k = std::min(top_k, N);
    std::vector<float> q_sc(static_cast<size_t>(Q));
    std::vector<float> d_sc(static_cast<size_t>(N));
    for (int qi = 0; qi < Q; ++qi) q_sc[qi] = fp16_bits_to_f32(q_sc_u16[qi]);
    for (int di = 0; di < N; ++di) d_sc[di] = fp16_bits_to_f32(d_sc_u16[di]);

    const int workers = detect_worker_threads(max_threads);
    const uint64_t work = static_cast<uint64_t>(Q) * static_cast<uint64_t>(N) * static_cast<uint64_t>(std::max(1, D));

    auto run_range = [&](int q0, int q1, int d0, int d1, std::vector<ScoreId>* dest_heaps, int heap_stride, int do_finalize) {
        std::vector<ScoreId> local_heaps[4];
        for (int h = 0; h < 4; ++h) local_heaps[h].reserve(static_cast<size_t>(eff_k));

        int qi = q0;
        for (; qi + 4 <= q1; qi += 4) {
            auto* h0 = dest_heaps ? &dest_heaps[static_cast<size_t>(qi + 0) * heap_stride] : &local_heaps[0];
            auto* h1 = dest_heaps ? &dest_heaps[static_cast<size_t>(qi + 1) * heap_stride] : &local_heaps[1];
            auto* h2 = dest_heaps ? &dest_heaps[static_cast<size_t>(qi + 2) * heap_stride] : &local_heaps[2];
            auto* h3 = dest_heaps ? &dest_heaps[static_cast<size_t>(qi + 3) * heap_stride] : &local_heaps[3];
            if (!dest_heaps) {
                h0->clear(); h1->clear(); h2->clear(); h3->clear();
            }
            const int8_t* qr0 = q_i8 + static_cast<size_t>(qi + 0) * D;
            const int8_t* qr1 = q_i8 + static_cast<size_t>(qi + 1) * D;
            const int8_t* qr2 = q_i8 + static_cast<size_t>(qi + 2) * D;
            const int8_t* qr3 = q_i8 + static_cast<size_t>(qi + 3) * D;
            const float qs0 = q_sc[qi + 0], qs1 = q_sc[qi + 1];
            const float qs2 = q_sc[qi + 2], qs3 = q_sc[qi + 3];
            int di = d0;
#if FS_ARM64 && FS_ENABLE_I8MM
            for (; di + 2 <= d1; di += 2) {
                int32_t d0_4[4], d1_4[4];
                dot_i8_4q_2d_i8mm(qr0, qr1, qr2, qr3, d_i8 + static_cast<size_t>(di) * D, d_i8 + static_cast<size_t>(di + 1) * D, D, d0_4, d1_4);
                const float ds0 = d_sc[di], ds1 = d_sc[di + 1];
                topk_push(*h0, eff_k, static_cast<float>(d0_4[0]) * (qs0 * ds0), di);
                topk_push(*h1, eff_k, static_cast<float>(d0_4[1]) * (qs1 * ds0), di);
                topk_push(*h2, eff_k, static_cast<float>(d0_4[2]) * (qs2 * ds0), di);
                topk_push(*h3, eff_k, static_cast<float>(d0_4[3]) * (qs3 * ds0), di);
                topk_push(*h0, eff_k, static_cast<float>(d1_4[0]) * (qs0 * ds1), di + 1);
                topk_push(*h1, eff_k, static_cast<float>(d1_4[1]) * (qs1 * ds1), di + 1);
                topk_push(*h2, eff_k, static_cast<float>(d1_4[2]) * (qs2 * ds1), di + 1);
                topk_push(*h3, eff_k, static_cast<float>(d1_4[3]) * (qs3 * ds1), di + 1);
            }
#endif
            for (; di < d1; ++di) {
                int32_t d4[4];
                dot_i8_4q_1d(qr0, qr1, qr2, qr3, d_i8 + static_cast<size_t>(di) * D, D, d4);
                const float ds = d_sc[di];
                topk_push(*h0, eff_k, static_cast<float>(d4[0]) * (qs0 * ds), di);
                topk_push(*h1, eff_k, static_cast<float>(d4[1]) * (qs1 * ds), di);
                topk_push(*h2, eff_k, static_cast<float>(d4[2]) * (qs2 * ds), di);
                topk_push(*h3, eff_k, static_cast<float>(d4[3]) * (qs3 * ds), di);
            }
            if (do_finalize) {
                topk_finalize(*h0, eff_k, top_k, out_indices + static_cast<size_t>(qi + 0) * top_k, out_scores + static_cast<size_t>(qi + 0) * top_k);
                topk_finalize(*h1, eff_k, top_k, out_indices + static_cast<size_t>(qi + 1) * top_k, out_scores + static_cast<size_t>(qi + 1) * top_k);
                topk_finalize(*h2, eff_k, top_k, out_indices + static_cast<size_t>(qi + 2) * top_k, out_scores + static_cast<size_t>(qi + 2) * top_k);
                topk_finalize(*h3, eff_k, top_k, out_indices + static_cast<size_t>(qi + 3) * top_k, out_scores + static_cast<size_t>(qi + 3) * top_k);
            }
        }
        for (; qi < q1; ++qi) {
            auto* h = dest_heaps ? &dest_heaps[static_cast<size_t>(qi) * heap_stride] : &local_heaps[0];
            if (!dest_heaps) h->clear();
            const int8_t* qr = q_i8 + static_cast<size_t>(qi) * D;
            const float qs = q_sc[qi];
            for (int di = d0; di < d1; ++di) {
                float sc = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(di) * D, D)) * (qs * d_sc[di]);
                topk_push(*h, eff_k, sc, di);
            }
            if (do_finalize) {
                topk_finalize(*h, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
            }
        }
    };

    if (workers <= 1 || work < (8ULL << 20)) {
        run_range(0, Q, 0, N, nullptr, 1, 1);
        return;
    }

    if (Q >= workers * 2) {
        const int n_tasks = std::min(workers, (Q + 3) / 4);
        parallel_for_chunks(n_tasks, [&](int tid) {
            const int q_blocks = (Q + 3) / 4;
            const int b0 = (tid * q_blocks) / n_tasks;
            const int b1 = ((tid + 1) * q_blocks) / n_tasks;
            const int q0 = b0 * 4;
            const int q1 = std::min(Q, b1 * 4);
            if (q0 < q1) run_range(q0, q1, 0, N, nullptr, 1, 1);
        });
        return;
    }

    const int n_tasks = std::min(workers, std::max(1, N / 2048));
    std::vector<ScoreId> partial;
    std::vector<std::vector<ScoreId>> per_shard(static_cast<size_t>(n_tasks * Q));
    for (auto& vh : per_shard) vh.reserve(static_cast<size_t>(eff_k));
    (void)partial;
    parallel_for_chunks(n_tasks, [&](int tid) {
        const int d0 = static_cast<int>((static_cast<int64_t>(tid) * N) / n_tasks);
        const int d1 = static_cast<int>((static_cast<int64_t>(tid + 1) * N) / n_tasks);
        run_range(0, Q, d0, d1, per_shard.data() + static_cast<size_t>(tid) * Q, 1, 0);
    });

    std::vector<ScoreId> merged;
    merged.reserve(static_cast<size_t>(n_tasks * eff_k));
    for (int qi = 0; qi < Q; ++qi) {
        merged.clear();
        for (int tid = 0; tid < n_tasks; ++tid) {
            const auto& h = per_shard[static_cast<size_t>(tid * Q + qi)];
            merged.insert(merged.end(), h.begin(), h.end());
        }
        topk_finalize(merged, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
    }
}

void fast_cpu_binary_topr_rerank_int8(
    const uint8_t* __restrict__ q_bits,
    const int8_t* __restrict__ q_i8,
    const uint16_t* __restrict__ q_sc_u16,
    const uint8_t* __restrict__ d_bits,
    const int8_t* __restrict__ d_i8,
    const uint16_t* __restrict__ d_sc_u16,
    int Q,
    int N,
    int n_bytes,
    int D_rerank,
    int top_r,
    int top_k,
    int max_threads,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_scores
) {
    if (Q <= 0 || top_k <= 0) return;
    if (N <= 0) {
        for (int qi = 0; qi < Q; ++qi) {
            for (int k = 0; k < top_k; ++k) {
                out_indices[static_cast<size_t>(qi) * top_k + k] = -1;
                out_scores[static_cast<size_t>(qi) * top_k + k] = -1e30f;
            }
        }
        return;
    }

    const int eff_r = std::min(std::max(top_k, top_r), N);
    const int eff_k = std::min(top_k, eff_r);
    const int hist_len = std::max(1, n_bytes * 8) + 1;

    std::vector<float> q_sc(static_cast<size_t>(Q));
    std::vector<float> d_sc(static_cast<size_t>(N));
    for (int qi = 0; qi < Q; ++qi) q_sc[qi] = fp16_bits_to_f32(q_sc_u16[qi]);
    for (int di = 0; di < N; ++di) d_sc[di] = fp16_bits_to_f32(d_sc_u16[di]);

    const int workers = detect_worker_threads(max_threads);

    auto run_single_query = [&](int qi, std::vector<uint32_t>& dists, std::vector<int>& hist, std::vector<ScoreId>& cand_heap) {
        const uint8_t* qb = q_bits + static_cast<size_t>(qi) * n_bytes;
        const int8_t* qr = q_i8 + static_cast<size_t>(qi) * D_rerank;
        const float qs = q_sc[qi];

        cand_heap.clear();
        if (eff_r >= N) {
            for (int di = 0; di < N; ++di) {
                float sc = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(di) * D_rerank, D_rerank)) * (qs * d_sc[di]);
                topk_push(cand_heap, eff_k, sc, di);
            }
            topk_finalize(cand_heap, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
            return;
        }

        std::fill(hist.begin(), hist.end(), 0);
        for (int di = 0; di < N; ++di) {
            uint32_t d = hamming_distance(qb, d_bits + static_cast<size_t>(di) * n_bytes, n_bytes);
            dists[di] = d;
            hist[d] += 1;
        }
        int rem_eq = 0;
        const uint32_t cut_u = static_cast<uint32_t>(find_hist_cutoff(hist.data(), hist_len, eff_r, rem_eq));
        for (int di = 0; di < N; ++di) {
            uint32_t d = dists[di];
            if (d < cut_u) {
                float sc = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(di) * D_rerank, D_rerank)) * (qs * d_sc[di]);
                topk_push(cand_heap, eff_k, sc, di);
            } else if (d == cut_u && rem_eq > 0) {
                --rem_eq;
                float sc = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(di) * D_rerank, D_rerank)) * (qs * d_sc[di]);
                topk_push(cand_heap, eff_k, sc, di);
            }
        }
        topk_finalize(cand_heap, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
    };

    if (workers <= 1 || (static_cast<uint64_t>(Q) * static_cast<uint64_t>(N) * static_cast<uint64_t>(std::max(1, n_bytes))) < (2ULL << 20)) {
        std::vector<uint32_t> dists(static_cast<size_t>(N));
        std::vector<int> hist(static_cast<size_t>(hist_len));
        std::vector<ScoreId> cand_heap;
        cand_heap.reserve(static_cast<size_t>(eff_k));
        for (int qi = 0; qi < Q; ++qi) run_single_query(qi, dists, hist, cand_heap);
        return;
    }

    if (Q >= workers) {
        const int n_tasks = std::min(workers, Q);
        parallel_for_chunks(n_tasks, [&](int tid) {
            const int q0 = (tid * Q) / n_tasks;
            const int q1 = ((tid + 1) * Q) / n_tasks;
            std::vector<uint32_t> dists(static_cast<size_t>(N));
            std::vector<int> hist(static_cast<size_t>(hist_len));
            std::vector<ScoreId> cand_heap;
            cand_heap.reserve(static_cast<size_t>(eff_k));
            for (int qi = q0; qi < q1; ++qi) run_single_query(qi, dists, hist, cand_heap);
        });
        return;
    }

    const int n_tasks = std::min(workers, std::max(1, N / 4096));
    std::vector<uint32_t> dists(static_cast<size_t>(N));
    std::vector<int> shard_hists(static_cast<size_t>(n_tasks * hist_len));
    std::vector<int> merged_hist(static_cast<size_t>(hist_len));
    std::vector<int32_t> short_ids(static_cast<size_t>(eff_r));
    std::vector<ScoreId> cand_heap;
    cand_heap.reserve(static_cast<size_t>(eff_k));

    for (int qi = 0; qi < Q; ++qi) {
        const uint8_t* qb = q_bits + static_cast<size_t>(qi) * n_bytes;
        const int8_t* qr = q_i8 + static_cast<size_t>(qi) * D_rerank;
        const float qs = q_sc[qi];

        if (eff_r >= N) {
            cand_heap.clear();
            for (int di = 0; di < N; ++di) {
                float sc = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(di) * D_rerank, D_rerank)) * (qs * d_sc[di]);
                topk_push(cand_heap, eff_k, sc, di);
            }
            topk_finalize(cand_heap, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
            continue;
        }

        std::fill(shard_hists.begin(), shard_hists.end(), 0);
        parallel_for_chunks(n_tasks, [&](int tid) {
            const int d0 = static_cast<int>((static_cast<int64_t>(tid) * N) / n_tasks);
            const int d1 = static_cast<int>((static_cast<int64_t>(tid + 1) * N) / n_tasks);
            int* h = shard_hists.data() + static_cast<size_t>(tid) * hist_len;
            for (int di = d0; di < d1; ++di) {
                uint32_t dist = hamming_distance(qb, d_bits + static_cast<size_t>(di) * n_bytes, n_bytes);
                dists[di] = dist;
                h[dist] += 1;
            }
        });
        for (int b = 0; b < hist_len; ++b) {
            int s = 0;
            for (int tid = 0; tid < n_tasks; ++tid) s += shard_hists[static_cast<size_t>(tid * hist_len + b)];
            merged_hist[b] = s;
        }
        int rem_eq = 0;
        const uint32_t cut_u = static_cast<uint32_t>(find_hist_cutoff(merged_hist.data(), hist_len, eff_r, rem_eq));
        int gathered = 0;
        for (int di = 0; di < N; ++di) {
            uint32_t d = dists[di];
            if (d < cut_u) {
                short_ids[gathered++] = di;
            } else if (d == cut_u && rem_eq > 0) {
                --rem_eq;
                short_ids[gathered++] = di;
            }
        }
        cand_heap.clear();
        for (int r = 0; r < gathered; ++r) {
            int32_t di = short_ids[r];
            float sc = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(di) * D_rerank, D_rerank)) * (qs * d_sc[di]);
            topk_push(cand_heap, eff_k, sc, di);
        }
        topk_finalize(cand_heap, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
    }
}

void fast_cpu_int8_doc_topk(
    const int8_t* __restrict__ q_i8,
    const uint16_t* __restrict__ q_sc_u16,
    const int8_t* __restrict__ d_i8,
    const uint16_t* __restrict__ d_sc_u16,
    const int32_t* __restrict__ doc_offsets,
    int Q,
    int num_docs,
    int D,
    int agg_mode,
    float hybrid_alpha,
    int top_k,
    int max_threads,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_scores
) {
    if (Q <= 0 || top_k <= 0) return;
    if (num_docs <= 0) {
        for (int qi = 0; qi < Q; ++qi) {
            for (int k = 0; k < top_k; ++k) {
                out_indices[static_cast<size_t>(qi) * top_k + k] = -1;
                out_scores[static_cast<size_t>(qi) * top_k + k] = -1e30f;
            }
        }
        return;
    }

    const int eff_k = std::min(top_k, num_docs);
    const int W = doc_offsets[num_docs];
    std::vector<float> q_sc(static_cast<size_t>(Q));
    std::vector<float> d_sc(static_cast<size_t>(W));
    for (int qi = 0; qi < Q; ++qi) q_sc[qi] = fp16_bits_to_f32(q_sc_u16[qi]);
    for (int wi = 0; wi < W; ++wi) d_sc[wi] = fp16_bits_to_f32(d_sc_u16[wi]);

    int max_win_doc = 1;
    for (int doc = 0; doc < num_docs; ++doc) {
        int nw = doc_offsets[doc + 1] - doc_offsets[doc];
        if (nw > max_win_doc) max_win_doc = nw;
    }

    const int workers = detect_worker_threads(max_threads);
    const uint64_t work = static_cast<uint64_t>(Q) * static_cast<uint64_t>(std::max(1, W)) * static_cast<uint64_t>(std::max(1, D));

    auto eval_docs_q_range = [&](int doc_start, int doc_end, int q0, int q1, std::vector<ScoreId>* out_heaps, int heap_stride) {
        std::vector<float> win_buf[4];
        for (int b = 0; b < 4; ++b) win_buf[b].resize(static_cast<size_t>(max_win_doc));

        int qi = q0;
        for (; qi + 4 <= q1; qi += 4) {
            const int8_t* qr0 = q_i8 + static_cast<size_t>(qi + 0) * D;
            const int8_t* qr1 = q_i8 + static_cast<size_t>(qi + 1) * D;
            const int8_t* qr2 = q_i8 + static_cast<size_t>(qi + 2) * D;
            const int8_t* qr3 = q_i8 + static_cast<size_t>(qi + 3) * D;
            const float qs0 = q_sc[qi + 0], qs1 = q_sc[qi + 1];
            const float qs2 = q_sc[qi + 2], qs3 = q_sc[qi + 3];
            auto& h0 = out_heaps[static_cast<size_t>((qi + 0) * heap_stride)];
            auto& h1 = out_heaps[static_cast<size_t>((qi + 1) * heap_stride)];
            auto& h2 = out_heaps[static_cast<size_t>((qi + 2) * heap_stride)];
            auto& h3 = out_heaps[static_cast<size_t>((qi + 3) * heap_stride)];

            for (int doc = doc_start; doc < doc_end; ++doc) {
                const int w0 = doc_offsets[doc], w1 = doc_offsets[doc + 1], nw = w1 - w0;
                if (nw <= 0) {
                    topk_push(h0, eff_k, -1e30f, doc); topk_push(h1, eff_k, -1e30f, doc);
                    topk_push(h2, eff_k, -1e30f, doc); topk_push(h3, eff_k, -1e30f, doc);
                    continue;
                }
                if (agg_mode == 0) {
                    float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f, m3 = -1e30f;
                    int wi = w0;
#if FS_ARM64 && FS_ENABLE_I8MM
                    for (; wi + 2 <= w1; wi += 2) {
                        int32_t d0_4[4], d1_4[4];
                        dot_i8_4q_2d_i8mm(qr0, qr1, qr2, qr3, d_i8 + static_cast<size_t>(wi) * D, d_i8 + static_cast<size_t>(wi + 1) * D, D, d0_4, d1_4);
                        const float ds0 = d_sc[wi], ds1 = d_sc[wi + 1];
                        m0 = std::max(m0, std::max(static_cast<float>(d0_4[0]) * (qs0 * ds0), static_cast<float>(d1_4[0]) * (qs0 * ds1)));
                        m1 = std::max(m1, std::max(static_cast<float>(d0_4[1]) * (qs1 * ds0), static_cast<float>(d1_4[1]) * (qs1 * ds1)));
                        m2 = std::max(m2, std::max(static_cast<float>(d0_4[2]) * (qs2 * ds0), static_cast<float>(d1_4[2]) * (qs2 * ds1)));
                        m3 = std::max(m3, std::max(static_cast<float>(d0_4[3]) * (qs3 * ds0), static_cast<float>(d1_4[3]) * (qs3 * ds1)));
                    }
#endif
                    for (; wi < w1; ++wi) {
                        int32_t d4[4];
                        dot_i8_4q_1d(qr0, qr1, qr2, qr3, d_i8 + static_cast<size_t>(wi) * D, D, d4);
                        const float ds = d_sc[wi];
                        m0 = std::max(m0, static_cast<float>(d4[0]) * (qs0 * ds));
                        m1 = std::max(m1, static_cast<float>(d4[1]) * (qs1 * ds));
                        m2 = std::max(m2, static_cast<float>(d4[2]) * (qs2 * ds));
                        m3 = std::max(m3, static_cast<float>(d4[3]) * (qs3 * ds));
                    }
                    topk_push(h0, eff_k, m0, doc); topk_push(h1, eff_k, m1, doc);
                    topk_push(h2, eff_k, m2, doc); topk_push(h3, eff_k, m3, doc);
                } else {
                    for (int wi = w0; wi < w1; ++wi) {
                        int32_t d4[4];
                        dot_i8_4q_1d(qr0, qr1, qr2, qr3, d_i8 + static_cast<size_t>(wi) * D, D, d4);
                        const float ds = d_sc[wi];
                        const int loc = wi - w0;
                        win_buf[0][loc] = static_cast<float>(d4[0]) * (qs0 * ds);
                        win_buf[1][loc] = static_cast<float>(d4[1]) * (qs1 * ds);
                        win_buf[2][loc] = static_cast<float>(d4[2]) * (qs2 * ds);
                        win_buf[3][loc] = static_cast<float>(d4[3]) * (qs3 * ds);
                    }
                    topk_push(h0, eff_k, reduce_window_scores(win_buf[0].data(), nw, agg_mode, hybrid_alpha), doc);
                    topk_push(h1, eff_k, reduce_window_scores(win_buf[1].data(), nw, agg_mode, hybrid_alpha), doc);
                    topk_push(h2, eff_k, reduce_window_scores(win_buf[2].data(), nw, agg_mode, hybrid_alpha), doc);
                    topk_push(h3, eff_k, reduce_window_scores(win_buf[3].data(), nw, agg_mode, hybrid_alpha), doc);
                }
            }
        }
        for (; qi < q1; ++qi) {
            const int8_t* qr = q_i8 + static_cast<size_t>(qi) * D;
            const float qs = q_sc[qi];
            auto& h = out_heaps[static_cast<size_t>(qi * heap_stride)];
            for (int doc = doc_start; doc < doc_end; ++doc) {
                const int w0 = doc_offsets[doc], w1 = doc_offsets[doc + 1], nw = w1 - w0;
                if (nw <= 0) {
                    topk_push(h, eff_k, -1e30f, doc);
                    continue;
                }
                if (agg_mode == 0) {
                    float mx = -1e30f;
                    for (int wi = w0; wi < w1; ++wi) {
                        float sc = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(wi) * D, D)) * (qs * d_sc[wi]);
                        if (sc > mx) mx = sc;
                    }
                    topk_push(h, eff_k, mx, doc);
                } else {
                    for (int wi = w0; wi < w1; ++wi) {
                        win_buf[0][wi - w0] = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(wi) * D, D)) * (qs * d_sc[wi]);
                    }
                    topk_push(h, eff_k, reduce_window_scores(win_buf[0].data(), nw, agg_mode, hybrid_alpha), doc);
                }
            }
        }
    };

    if (workers <= 1 || work < (8ULL << 20)) {
        std::vector<std::vector<ScoreId>> q_heaps(static_cast<size_t>(Q));
        for (auto& vh : q_heaps) vh.reserve(static_cast<size_t>(eff_k));
        eval_docs_q_range(0, num_docs, 0, Q, q_heaps.data(), 1);
        for (int qi = 0; qi < Q; ++qi) {
            topk_finalize(q_heaps[qi], eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
        }
        return;
    }

    if (Q >= workers * 2) {
        const int n_tasks = std::min(workers, (Q + 3) / 4);
        std::vector<std::vector<ScoreId>> q_heaps(static_cast<size_t>(Q));
        for (auto& vh : q_heaps) vh.reserve(static_cast<size_t>(eff_k));
        parallel_for_chunks(n_tasks, [&](int tid) {
            const int q_blocks = (Q + 3) / 4;
            const int b0 = (tid * q_blocks) / n_tasks;
            const int b1 = ((tid + 1) * q_blocks) / n_tasks;
            const int q0 = b0 * 4;
            const int q1 = std::min(Q, b1 * 4);
            if (q0 < q1) eval_docs_q_range(0, num_docs, q0, q1, q_heaps.data(), 1);
        });
        for (int qi = 0; qi < Q; ++qi) {
            topk_finalize(q_heaps[qi], eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
        }
        return;
    }

    const int n_tasks = std::min(workers, std::max(1, num_docs / 256));
    std::vector<int> doc_splits(static_cast<size_t>(n_tasks + 1), 0);
    doc_splits[n_tasks] = num_docs;
    for (int t = 1; t < n_tasks; ++t) {
        const int target_w = static_cast<int>((static_cast<int64_t>(t) * W) / n_tasks);
        const int32_t* it = std::lower_bound(doc_offsets, doc_offsets + num_docs + 1, target_w);
        int pos = static_cast<int>(it - doc_offsets);
        doc_splits[t] = std::max(doc_splits[t - 1], std::min(pos, num_docs));
    }

    std::vector<std::vector<ScoreId>> partial(static_cast<size_t>(n_tasks * Q));
    for (auto& vh : partial) vh.reserve(static_cast<size_t>(eff_k));
    parallel_for_chunks(n_tasks, [&](int tid) {
        const int d0 = doc_splits[tid], d1 = doc_splits[tid + 1];
        if (d0 < d1) eval_docs_q_range(d0, d1, 0, Q, partial.data() + static_cast<size_t>(tid) * Q, 1);
    });

    std::vector<ScoreId> merged;
    merged.reserve(static_cast<size_t>(n_tasks * eff_k));
    for (int qi = 0; qi < Q; ++qi) {
        merged.clear();
        for (int tid = 0; tid < n_tasks; ++tid) {
            const auto& h = partial[static_cast<size_t>(tid * Q + qi)];
            merged.insert(merged.end(), h.begin(), h.end());
        }
        topk_finalize(merged, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
    }
}

void fast_cpu_binary_doc_rerank_int8(
    const uint8_t* __restrict__ q_bits,
    const int8_t* __restrict__ q_i8,
    const uint16_t* __restrict__ q_sc_u16,
    const uint8_t* __restrict__ d_bits,
    const int8_t* __restrict__ d_i8,
    const uint16_t* __restrict__ d_sc_u16,
    const int32_t* __restrict__ doc_offsets,
    const int32_t* __restrict__ window_to_doc,
    int Q,
    int num_docs,
    int num_windows,
    int n_bytes,
    int D_rerank,
    int win_top_r,
    int agg_mode,
    float hybrid_alpha,
    int top_k,
    int max_threads,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_scores
) {
    if (Q <= 0 || top_k <= 0) return;
    if (num_docs <= 0 || num_windows <= 0) {
        for (int qi = 0; qi < Q; ++qi) {
            for (int k = 0; k < top_k; ++k) {
                out_indices[static_cast<size_t>(qi) * top_k + k] = -1;
                out_scores[static_cast<size_t>(qi) * top_k + k] = -1e30f;
            }
        }
        return;
    }

    const int eff_wr = std::min(std::max(1, win_top_r), num_windows);
    const int eff_k = std::min(top_k, num_docs);
    const int hist_len = std::max(1, n_bytes * 8) + 1;

    std::vector<float> q_sc(static_cast<size_t>(Q));
    std::vector<float> d_sc(static_cast<size_t>(num_windows));
    for (int qi = 0; qi < Q; ++qi) q_sc[qi] = fp16_bits_to_f32(q_sc_u16[qi]);
    for (int wi = 0; wi < num_windows; ++wi) d_sc[wi] = fp16_bits_to_f32(d_sc_u16[wi]);

    int max_win_doc = 1;
    for (int doc = 0; doc < num_docs; ++doc) {
        int nw = doc_offsets[doc + 1] - doc_offsets[doc];
        if (nw > max_win_doc) max_win_doc = nw;
    }
    const int workers = detect_worker_threads(max_threads);

    auto score_shortlisted_docs = [&](int qi, const std::vector<uint32_t>& dists, int cutoff_dist, int rem_eq, std::vector<uint8_t>& seen_doc, std::vector<int32_t>& cand_docs, std::vector<float>& win_buf, std::vector<ScoreId>& cand_heap) {
        const uint32_t cut_u = static_cast<uint32_t>(cutoff_dist);
        cand_docs.clear();
        for (int wi = 0; wi < num_windows; ++wi) {
            uint32_t d = dists[wi];
            bool take = (d < cut_u);
            if (!take && d == cut_u && rem_eq > 0) { --rem_eq; take = true; }
            if (take) {
                int32_t doc = window_to_doc[wi];
                if (doc >= 0 && doc < num_docs && !seen_doc[doc]) {
                    seen_doc[doc] = 1u;
                    cand_docs.push_back(doc);
                }
            }
        }
        const int8_t* qr = q_i8 + static_cast<size_t>(qi) * D_rerank;
        const float qs = q_sc[qi];
        cand_heap.clear();
        for (int32_t doc : cand_docs) {
            seen_doc[doc] = 0u;
            const int w0 = doc_offsets[doc], w1 = doc_offsets[doc + 1], nw = w1 - w0;
            if (nw <= 0) {
                topk_push(cand_heap, eff_k, -1e30f, doc);
                continue;
            }
            if (agg_mode == 0) {
                float mx = -1e30f;
                for (int wi = w0; wi < w1; ++wi) {
                    float sc = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(wi) * D_rerank, D_rerank)) * (qs * d_sc[wi]);
                    if (sc > mx) mx = sc;
                }
                topk_push(cand_heap, eff_k, mx, doc);
            } else {
                for (int wi = w0; wi < w1; ++wi) {
                    win_buf[wi - w0] = static_cast<float>(dot_i8_1(qr, d_i8 + static_cast<size_t>(wi) * D_rerank, D_rerank)) * (qs * d_sc[wi]);
                }
                topk_push(cand_heap, eff_k, reduce_window_scores(win_buf.data(), nw, agg_mode, hybrid_alpha), doc);
            }
        }
        topk_finalize(cand_heap, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
    };

    if (workers <= 1 || (static_cast<uint64_t>(Q) * static_cast<uint64_t>(num_windows) * static_cast<uint64_t>(std::max(1, n_bytes))) < (2ULL << 20)) {
        std::vector<uint32_t> dists(static_cast<size_t>(num_windows));
        std::vector<int> hist(static_cast<size_t>(hist_len));
        std::vector<uint8_t> seen_doc(static_cast<size_t>(num_docs), 0u);
        std::vector<int32_t> cand_docs;
        std::vector<float> win_buf(static_cast<size_t>(max_win_doc));
        std::vector<ScoreId> cand_heap;
        cand_docs.reserve(static_cast<size_t>(eff_wr));
        cand_heap.reserve(static_cast<size_t>(eff_k));
        for (int qi = 0; qi < Q; ++qi) {
            const uint8_t* qb = q_bits + static_cast<size_t>(qi) * n_bytes;
            std::fill(hist.begin(), hist.end(), 0);
            for (int wi = 0; wi < num_windows; ++wi) {
                uint32_t d = hamming_distance(qb, d_bits + static_cast<size_t>(wi) * n_bytes, n_bytes);
                dists[wi] = d;
                hist[d] += 1;
            }
            int rem_eq = 0;
            int cut = find_hist_cutoff(hist.data(), hist_len, eff_wr, rem_eq);
            score_shortlisted_docs(qi, dists, cut, rem_eq, seen_doc, cand_docs, win_buf, cand_heap);
        }
        return;
    }

    if (Q >= workers) {
        const int n_tasks = std::min(workers, Q);
        parallel_for_chunks(n_tasks, [&](int tid) {
            const int q0 = (tid * Q) / n_tasks;
            const int q1 = ((tid + 1) * Q) / n_tasks;
            std::vector<uint32_t> dists(static_cast<size_t>(num_windows));
            std::vector<int> hist(static_cast<size_t>(hist_len));
            std::vector<uint8_t> seen_doc(static_cast<size_t>(num_docs), 0u);
            std::vector<int32_t> cand_docs;
            std::vector<float> win_buf(static_cast<size_t>(max_win_doc));
            std::vector<ScoreId> cand_heap;
            cand_docs.reserve(static_cast<size_t>(eff_wr));
            cand_heap.reserve(static_cast<size_t>(eff_k));
            for (int qi = q0; qi < q1; ++qi) {
                const uint8_t* qb = q_bits + static_cast<size_t>(qi) * n_bytes;
                std::fill(hist.begin(), hist.end(), 0);
                for (int wi = 0; wi < num_windows; ++wi) {
                    uint32_t d = hamming_distance(qb, d_bits + static_cast<size_t>(wi) * n_bytes, n_bytes);
                    dists[wi] = d;
                    hist[d] += 1;
                }
                int rem_eq = 0;
                int cut = find_hist_cutoff(hist.data(), hist_len, eff_wr, rem_eq);
                score_shortlisted_docs(qi, dists, cut, rem_eq, seen_doc, cand_docs, win_buf, cand_heap);
            }
        });
        return;
    }

    const int n_tasks = std::min(workers, std::max(1, num_windows / 4096));
    std::vector<uint32_t> dists(static_cast<size_t>(num_windows));
    std::vector<int> shard_hists(static_cast<size_t>(n_tasks * hist_len));
    std::vector<int> merged_hist(static_cast<size_t>(hist_len));
    std::vector<uint8_t> seen_doc(static_cast<size_t>(num_docs), 0u);
    std::vector<int32_t> cand_docs;
    std::vector<float> win_buf(static_cast<size_t>(max_win_doc));
    std::vector<ScoreId> cand_heap;
    cand_docs.reserve(static_cast<size_t>(eff_wr));
    cand_heap.reserve(static_cast<size_t>(eff_k));
    for (int qi = 0; qi < Q; ++qi) {
        const uint8_t* qb = q_bits + static_cast<size_t>(qi) * n_bytes;
        std::fill(shard_hists.begin(), shard_hists.end(), 0);
        parallel_for_chunks(n_tasks, [&](int tid) {
            const int w0 = static_cast<int>((static_cast<int64_t>(tid) * num_windows) / n_tasks);
            const int w1 = static_cast<int>((static_cast<int64_t>(tid + 1) * num_windows) / n_tasks);
            int* h = shard_hists.data() + static_cast<size_t>(tid) * hist_len;
            for (int wi = w0; wi < w1; ++wi) {
                uint32_t d = hamming_distance(qb, d_bits + static_cast<size_t>(wi) * n_bytes, n_bytes);
                dists[wi] = d;
                h[d] += 1;
            }
        });
        for (int b = 0; b < hist_len; ++b) {
            int s = 0;
            for (int tid = 0; tid < n_tasks; ++tid) s += shard_hists[static_cast<size_t>(tid * hist_len + b)];
            merged_hist[b] = s;
        }
        int rem_eq = 0;
        int cut = find_hist_cutoff(merged_hist.data(), hist_len, eff_wr, rem_eq);
        score_shortlisted_docs(qi, dists, cut, rem_eq, seen_doc, cand_docs, win_buf, cand_heap);
    }
}

} // extern "C"
"""

_BM25_CPP_SOURCE = r"""
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif

namespace {

static int detect_worker_threads(int max_hint) {
    if (max_hint == 1) return 1;
    int threads = 0;
#if defined(__APPLE__)
    int pcores = 0;
    size_t sz = sizeof(pcores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &sz, nullptr, 0) == 0 && pcores > 0) {
        threads = pcores;
    }
#endif
    if (threads <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        threads = hw > 0u ? static_cast<int>(hw) : 4;
    }
    if (max_hint > 0 && max_hint < threads) threads = max_hint;
    return std::max(1, std::min(threads, 64));
}

template <typename Fn>
static void parallel_for_chunks(int n_tasks, Fn&& fn) {
    if (n_tasks <= 1) {
        if (n_tasks == 1) fn(0);
        return;
    }
#if defined(__APPLE__)
    struct Ctx { Fn* fp; } ctx{&fn};
    dispatch_apply_f(
        static_cast<size_t>(n_tasks),
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        &ctx,
        [](void* raw, size_t idx) { (*static_cast<Ctx*>(raw)->fp)(static_cast<int>(idx)); }
    );
#else
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_tasks - 1));
    for (int t = 1; t < n_tasks; ++t) workers.emplace_back([&, t]() { fn(t); });
    fn(0);
    for (auto& w : workers) w.join();
#endif
}

struct Hit { float score; int32_t id; };

static inline bool is_better(float sa, int32_t ia, float sb, int32_t ib) {
    if (sa != sb) return sa > sb;
    return ia < ib;
}

struct HitWorse {
    bool operator()(const Hit& a, const Hit& b) const {
        return is_better(a.score, a.id, b.score, b.id);
    }
};

static inline void topk_push(std::vector<Hit>& heap, int eff_k, float score, int32_t id) {
    if (eff_k <= 0) return;
    if (static_cast<int>(heap.size()) < eff_k) {
        heap.push_back(Hit{score, id});
        std::push_heap(heap.begin(), heap.end(), HitWorse{});
    } else if (is_better(score, id, heap[0].score, heap[0].id)) {
        std::pop_heap(heap.begin(), heap.end(), HitWorse{});
        heap.back() = Hit{score, id};
        std::push_heap(heap.begin(), heap.end(), HitWorse{});
    }
}

static inline float reduce_windows(
    const float* __restrict__ acc,
    int w0,
    int w1,
    int agg_mode,
    float hybrid_alpha
) {
    const int nw = w1 - w0;
    if (nw <= 0) return -1e30f;
    if (agg_mode == 0) {
        float mx = acc[w0];
        for (int wi = w0 + 1; wi < w1; ++wi) {
            if (acc[wi] > mx) mx = acc[wi];
        }
        return mx;
    }
    if (agg_mode == 3) {
        float sum = 0.0f;
        for (int wi = w0; wi < w1; ++wi) sum += acc[wi];
        return sum / static_cast<float>(nw);
    }
    if (agg_mode == 1) {
        float mx = acc[w0], sum = acc[w0];
        for (int wi = w0 + 1; wi < w1; ++wi) {
            float v = acc[wi];
            if (v > mx) mx = v;
            sum += v;
        }
        return hybrid_alpha * mx + (1.0f - hybrid_alpha) * (sum / static_cast<float>(nw));
    }
    if (nw == 1) return acc[w0];
    float m1 = acc[w0], m2 = acc[w0 + 1];
    if (m2 > m1) std::swap(m1, m2);
    for (int wi = w0 + 2; wi < w1; ++wi) {
        float v = acc[wi];
        if (v > m1) { m2 = m1; m1 = v; }
        else if (v > m2) { m2 = v; }
    }
    return 0.5f * (m1 + m2);
}

static inline void finalize_topk(
    std::vector<Hit>& heap,
    int eff_k,
    int top_k,
    int32_t* out_idx,
    float* out_sc
) {
    std::sort(heap.begin(), heap.end(), [](const Hit& a, const Hit& b) {
        return is_better(a.score, a.id, b.score, b.id);
    });
    const int n_fill = std::min(eff_k, static_cast<int>(heap.size()));
    for (int i = 0; i < n_fill; ++i) {
        out_idx[i] = heap[i].id;
        out_sc[i] = heap[i].score;
    }
    for (int i = n_fill; i < top_k; ++i) {
        out_idx[i] = -1;
        out_sc[i] = -1e30f;
    }
}

struct PostingCursor {
    const int32_t* docs;
    const float* scores;
    int64_t pos;
    int64_t end;
    int64_t blk_pos;
    int64_t blk_end;
    float tf_weight;
    float max_term_score;
    int32_t cur_doc;
};

static inline void cursor_advance_to(PostingCursor& c, int32_t target, int32_t sentinel_doc) {
    const int32_t* p = c.docs + c.pos;
    const int32_t* e = c.docs + c.end;
    if (c.end - c.pos > 16) p = std::lower_bound(p, e, target);
    else while (p < e && *p < target) ++p;
    c.pos = static_cast<int64_t>(p - c.docs);
    c.cur_doc = (c.pos < c.end) ? c.docs[c.pos] : sentinel_doc;
}

} // namespace

extern "C" {

void fast_cpu_bm25_build_blocks(
    const int64_t* __restrict__ term_offsets,
    const int32_t* __restrict__ posting_doc_ids,
    const float* __restrict__ posting_scores,
    int vocab_size,
    int block_size,
    int64_t* __restrict__ term_blk_offsets,
    int32_t* __restrict__ blk_max_doc,
    float* __restrict__ blk_max_score,
    float* __restrict__ term_max_score
) {
    term_blk_offsets[0] = 0;
    int64_t b_acc = 0;
    for (int tid = 0; tid < vocab_size; ++tid) {
        const int64_t s0 = term_offsets[tid], s1 = term_offsets[tid + 1];
        float t_mx = 0.0f;
        for (int64_t p = s0; p < s1; p += block_size) {
            const int64_t pe = std::min(s1, p + block_size);
            float b_mx = 0.0f;
            for (int64_t x = p; x < pe; ++x) {
                if (posting_scores[x] > b_mx) b_mx = posting_scores[x];
            }
            blk_max_doc[b_acc] = posting_doc_ids[pe - 1];
            blk_max_score[b_acc] = b_mx;
            if (b_mx > t_mx) t_mx = b_mx;
            ++b_acc;
        }
        term_max_score[tid] = t_mx;
        term_blk_offsets[tid + 1] = b_acc;
    }
}

void fast_cpu_bm25_topk(
    const int64_t* __restrict__ term_offsets,
    const int32_t* __restrict__ posting_doc_ids,
    const float* __restrict__ posting_scores,
    const int64_t* __restrict__ term_blk_offsets,
    const int32_t* __restrict__ blk_max_doc,
    const float* __restrict__ blk_max_score,
    const float* __restrict__ term_max_score,
    const int32_t* __restrict__ doc_offsets,
    const int32_t* __restrict__ window_to_doc,
    int num_windows,
    int num_docs,
    int block_size,
    int agg_mode,
    float hybrid_alpha,
    int positive_only,
    int Q,
    const int64_t* __restrict__ q_offsets,
    const int32_t* __restrict__ q_tids,
    const float* __restrict__ q_tfs,
    int top_k,
    int max_threads,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_scores
) {
    (void)block_size;
    if (Q <= 0 || top_k <= 0) return;
    if (num_docs <= 0 || num_windows <= 0) {
        for (int qi = 0; qi < Q; ++qi) {
            for (int k = 0; k < top_k; ++k) {
                out_indices[static_cast<size_t>(qi) * top_k + k] = -1;
                out_scores[static_cast<size_t>(qi) * top_k + k] = -1e30f;
            }
        }
        return;
    }

    const int eff_k = std::min(top_k, num_docs);
    const bool has_blocks = (term_blk_offsets != nullptr && blk_max_doc != nullptr && blk_max_score != nullptr && term_max_score != nullptr && doc_offsets == nullptr);
    const int workers = detect_worker_threads(max_threads);

    auto solve_one_query = [&](
        int qi,
        std::vector<float>& acc,
        std::vector<uint8_t>& seen,
        std::vector<int32_t>& touched,
        std::vector<Hit>& heap,
        std::vector<PostingCursor>& cursors
    ) {
        const int64_t qt0 = q_offsets[qi], qt1 = q_offsets[qi + 1];
        const int n_terms = static_cast<int>(qt1 - qt0);
        heap.clear();

        if (n_terms == 0) {
            int filled = 0;
            if (!positive_only) {
                if (doc_offsets == nullptr) {
                    const int take = std::min(eff_k, num_docs);
                    for (int d = 0; d < take; ++d) {
                        out_indices[static_cast<size_t>(qi) * top_k + filled] = d;
                        out_scores[static_cast<size_t>(qi) * top_k + filled] = 0.0f;
                        ++filled;
                    }
                } else {
                    for (int d = 0; d < num_docs && filled < eff_k; ++d) {
                        if (doc_offsets[d + 1] > doc_offsets[d]) {
                            out_indices[static_cast<size_t>(qi) * top_k + filled] = d;
                            out_scores[static_cast<size_t>(qi) * top_k + filled] = 0.0f;
                            ++filled;
                        }
                    }
                    for (int d = 0; d < num_docs && filled < eff_k; ++d) {
                        if (doc_offsets[d + 1] <= doc_offsets[d]) {
                            out_indices[static_cast<size_t>(qi) * top_k + filled] = d;
                            out_scores[static_cast<size_t>(qi) * top_k + filled] = -1e30f;
                            ++filled;
                        }
                    }
                }
            }
            for (int k = filled; k < top_k; ++k) {
                out_indices[static_cast<size_t>(qi) * top_k + k] = -1;
                out_scores[static_cast<size_t>(qi) * top_k + k] = -1e30f;
            }
            return;
        }

        int64_t total_postings = 0;
        int64_t min_term_postings = static_cast<int64_t>(num_windows) + 1;
        for (int64_t p = qt0; p < qt1; ++p) {
            int32_t tid = q_tids[p];
            int64_t tp = (term_offsets[tid + 1] - term_offsets[tid]);
            total_postings += tp;
            if (tp > 0 && tp < min_term_postings) min_term_postings = tp;
        }

        if (
            has_blocks
            && n_terms >= 2
            && eff_k <= 256
            && total_postings > static_cast<int64_t>(eff_k) * 8
            && min_term_postings <= std::max<int64_t>(1024, num_docs / 16)
            && total_postings <= (static_cast<int64_t>(num_docs) * 3) / 4
        ) {
            cursors.clear();
            const int32_t sentinel = num_windows;
            for (int64_t p = qt0; p < qt1; ++p) {
                int32_t tid = q_tids[p];
                int64_t s0 = term_offsets[tid], s1 = term_offsets[tid + 1];
                if (s1 <= s0) continue;
                float qtf = q_tfs[p];
                PostingCursor c{posting_doc_ids, posting_scores, s0, s1, term_blk_offsets[tid], term_blk_offsets[tid + 1], qtf, qtf * term_max_score[tid], posting_doc_ids[s0]};
                cursors.push_back(c);
            }
            if (cursors.empty()) {
                finalize_topk(heap, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
                return;
            }
            const int m = static_cast<int>(cursors.size());
            std::vector<int> order(static_cast<size_t>(m));
            for (int i = 0; i < m; ++i) order[i] = i;

            while (true) {
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    return cursors[a].cur_doc < cursors[b].cur_doc;
                });
                if (cursors[order[0]].cur_doc >= sentinel) break;

                const float threshold = (static_cast<int>(heap.size()) == eff_k && heap[0].score > 0.0f) ? heap[0].score : 0.0f;
                double upper = 0.0;
                int pivot_idx = -1;
                int32_t pivot_doc = sentinel;
                for (int i = 0; i < m; ++i) {
                    const auto& c = cursors[order[i]];
                    if (c.cur_doc >= sentinel) break;
                    upper += static_cast<double>(c.max_term_score);
                    if (upper + 1e-5 >= static_cast<double>(threshold) && threshold > 0.0f) {
                        pivot_idx = i;
                        pivot_doc = c.cur_doc;
                        break;
                    }
                }
                if (threshold == 0.0f) {
                    pivot_idx = 0;
                    pivot_doc = cursors[order[0]].cur_doc;
                }
                if (pivot_idx < 0 || pivot_doc >= sentinel) break;
                while (pivot_idx + 1 < m && cursors[order[pivot_idx + 1]].cur_doc == pivot_doc) ++pivot_idx;

                if (cursors[order[0]].cur_doc == pivot_doc) {
                    if (threshold > 0.0f) {
                        double bm_upper = 0.0;
                        for (int i = 0; i <= pivot_idx; ++i) {
                            PostingCursor& c = cursors[order[i]];
                            while (c.blk_pos < c.blk_end && blk_max_doc[c.blk_pos] < pivot_doc) ++c.blk_pos;
                            if (c.blk_pos < c.blk_end) bm_upper += static_cast<double>(c.tf_weight * blk_max_score[c.blk_pos]);
                        }
                        if (bm_upper + 1e-5 < static_cast<double>(threshold)) {
                            int32_t next_blk_jump = sentinel;
                            for (int i = 0; i <= pivot_idx; ++i) {
                                const auto& c = cursors[order[i]];
                                if (c.blk_pos < c.blk_end && blk_max_doc[c.blk_pos] + 1 < next_blk_jump) next_blk_jump = blk_max_doc[c.blk_pos] + 1;
                            }
                            if (pivot_idx + 1 < m && cursors[order[pivot_idx + 1]].cur_doc < next_blk_jump) next_blk_jump = cursors[order[pivot_idx + 1]].cur_doc;
                            if (next_blk_jump <= pivot_doc) next_blk_jump = pivot_doc + 1;
                            cursor_advance_to(cursors[order[0]], next_blk_jump, sentinel);
                            continue;
                        }
                    }
                    float exact_sc = 0.0f;
                    for (int i = 0; i < m; ++i) {
                        PostingCursor& c = cursors[i];
                        if (c.cur_doc == pivot_doc) {
                            exact_sc += c.tf_weight * c.scores[c.pos];
                            ++c.pos;
                            c.cur_doc = (c.pos < c.end) ? c.docs[c.pos] : sentinel;
                        }
                    }
                    if (!positive_only || exact_sc > 0.0f) topk_push(heap, eff_k, exact_sc, pivot_doc);
                } else {
                    cursor_advance_to(cursors[order[0]], pivot_doc, sentinel);
                }
            }

            if (!positive_only && static_cast<int>(heap.size()) < eff_k) {
                std::vector<uint8_t> in_heap(static_cast<size_t>(std::min(num_docs, eff_k + static_cast<int>(heap.size()) + 16)), 0u);
                std::vector<int32_t> pos_seen;
                for (const auto& h : heap) {
                    if (h.id < static_cast<int32_t>(in_heap.size())) in_heap[h.id] = 1u;
                    else pos_seen.push_back(h.id);
                }
                for (int32_t d = 0; d < num_docs && static_cast<int>(heap.size()) < eff_k; ++d) {
                    if (d < static_cast<int32_t>(in_heap.size())) {
                        if (!in_heap[d]) topk_push(heap, eff_k, 0.0f, d);
                    } else if (std::find(pos_seen.begin(), pos_seen.end(), d) == pos_seen.end()) {
                        topk_push(heap, eff_k, 0.0f, d);
                    }
                }
            }
            finalize_topk(heap, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
            return;
        }

        touched.clear();
        for (int64_t p = qt0; p < qt1; ++p) {
            const int32_t tid = q_tids[p];
            const float w = q_tfs[p];
            const int64_t s0 = term_offsets[tid], s1 = term_offsets[tid + 1];
            for (int64_t idx = s0; idx < s1; ++idx) {
                const int32_t wid = posting_doc_ids[idx];
                if (!seen[wid]) {
                    seen[wid] = 1u;
                    touched.push_back(wid);
                    acc[wid] = w * posting_scores[idx];
                } else {
                    acc[wid] += w * posting_scores[idx];
                }
            }
        }

        if (doc_offsets == nullptr) {
            if (positive_only || static_cast<int>(touched.size()) >= eff_k) {
                for (int32_t doc : touched) {
                    float sc = acc[doc];
                    if (!positive_only || sc > 0.0f) topk_push(heap, eff_k, sc, doc);
                }
            }
            if (!positive_only && static_cast<int>(heap.size()) < eff_k) {
                for (int32_t doc : touched) topk_push(heap, eff_k, acc[doc], doc);
                for (int32_t doc = 0; doc < num_docs && static_cast<int>(heap.size()) < eff_k; ++doc) {
                    if (!seen[doc]) topk_push(heap, eff_k, 0.0f, doc);
                }
            }
            for (int32_t wid : touched) { seen[wid] = 0u; acc[wid] = 0.0f; }
            finalize_topk(heap, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
            return;
        }

        std::vector<int32_t> cand_docs;
        std::vector<uint8_t> doc_seen(static_cast<size_t>(num_docs), 0u);
        cand_docs.reserve(touched.size());
        for (int32_t wid : touched) {
            int32_t doc = window_to_doc[wid];
            if (doc >= 0 && doc < num_docs && !doc_seen[doc]) {
                doc_seen[doc] = 1u;
                cand_docs.push_back(doc);
            }
        }
        for (int32_t doc : cand_docs) {
            float sc = reduce_windows(acc.data(), doc_offsets[doc], doc_offsets[doc + 1], agg_mode, hybrid_alpha);
            if (!positive_only || sc > 0.0f) topk_push(heap, eff_k, sc, doc);
        }
        if (!positive_only && static_cast<int>(heap.size()) < eff_k) {
            for (int32_t doc = 0; doc < num_docs && static_cast<int>(heap.size()) < eff_k; ++doc) {
                if (!doc_seen[doc] && (doc_offsets[doc + 1] > doc_offsets[doc])) topk_push(heap, eff_k, 0.0f, doc);
            }
            for (int32_t doc = 0; doc < num_docs && static_cast<int>(heap.size()) < eff_k; ++doc) {
                if (!doc_seen[doc] && (doc_offsets[doc + 1] <= doc_offsets[doc])) topk_push(heap, eff_k, -1e30f, doc);
            }
        }
        for (int32_t wid : touched) { seen[wid] = 0u; acc[wid] = 0.0f; }
        finalize_topk(heap, eff_k, top_k, out_indices + static_cast<size_t>(qi) * top_k, out_scores + static_cast<size_t>(qi) * top_k);
    };

    if (workers <= 1 || Q < 2 || num_windows < 2048) {
        std::vector<float> acc(static_cast<size_t>(num_windows), 0.0f);
        std::vector<uint8_t> seen(static_cast<size_t>(num_windows), 0u);
        std::vector<int32_t> touched;
        std::vector<Hit> heap;
        std::vector<PostingCursor> cursors;
        touched.reserve(static_cast<size_t>(std::min(num_windows, 4096)));
        heap.reserve(static_cast<size_t>(eff_k));
        for (int qi = 0; qi < Q; ++qi) solve_one_query(qi, acc, seen, touched, heap, cursors);
        return;
    }

    const int n_tasks = std::min(workers, Q);
    parallel_for_chunks(n_tasks, [&](int tid) {
        const int q0 = (tid * Q) / n_tasks;
        const int q1 = ((tid + 1) * Q) / n_tasks;
        std::vector<float> acc(static_cast<size_t>(num_windows), 0.0f);
        std::vector<uint8_t> seen(static_cast<size_t>(num_windows), 0u);
        std::vector<int32_t> touched;
        std::vector<Hit> heap;
        std::vector<PostingCursor> cursors;
        touched.reserve(static_cast<size_t>(std::min(num_windows, 4096)));
        heap.reserve(static_cast<size_t>(eff_k));
        for (int qi = q0; qi < q1; ++qi) solve_one_query(qi, acc, seen, touched, heap, cursors);
    });
}

} // extern "C"
"""


def _compile_shared_kernel(kind: str, source: str, mode: str) -> ctypes.CDLL | None:
    cxx = _find_cxx_compiler()
    if not cxx:
        return None
    flags = cpu_compile_flags(mode, shared=True)
    digest = hashlib.sha256((kind + "|" + source + "|" + " ".join(flags)).encode("utf-8")).hexdigest()[:12]
    cache_dir = Path(os.environ.get("FAST_SILICON_CACHE_DIR") or (Path.home() / ".cache" / "fast_silicon"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    lib_path = cache_dir / f"libfast_{kind}_{mode}_{platform.machine().lower()}_{digest}{ext}"
    if not lib_path.is_file():
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", dir=str(cache_dir), delete=False, encoding="utf-8"
        ) as tf:
            tf.write(source)
            src_path = Path(tf.name)
        tmp_lib = cache_dir / f".tmp_{kind}_{os.getpid()}_{digest}{ext}"
        try:
            subprocess.run([cxx, *flags, str(src_path), "-o", str(tmp_lib)], check=True, capture_output=True, timeout=30)
            os.replace(tmp_lib, lib_path)
        finally:
            src_path.unlink(missing_ok=True)
            tmp_lib.unlink(missing_ok=True)
    return ctypes.CDLL(str(lib_path))


def _bind_retrieval_argtypes(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.fast_cpu_int8_scores.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.fast_cpu_int8_scores.restype = None
    lib.fast_cpu_int8_topk.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.fast_cpu_int8_topk.restype = None
    lib.fast_cpu_binary_topr_rerank_int8.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.fast_cpu_binary_topr_rerank_int8.restype = None
    lib.fast_cpu_int8_doc_topk.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.fast_cpu_int8_doc_topk.restype = None
    lib.fast_cpu_binary_doc_rerank_int8.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.fast_cpu_binary_doc_rerank_int8.restype = None
    return lib


def _bind_bm25_argtypes(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.fast_cpu_bm25_build_blocks.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.fast_cpu_bm25_build_blocks.restype = None
    lib.fast_cpu_bm25_topk.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.fast_cpu_bm25_topk.restype = None
    return lib


def _get_kernel(
    kind: Literal["retrieval", "bm25"],
    cpu_backend: CpuBackendMode = "auto",
    *,
    use_neon: bool = True,
) -> tuple[str, ctypes.CDLL | None]:
    resolved = resolve_cpu_backend(cpu_backend, use_neon=use_neon)
    if resolved == "numpy":
        return "numpy", None
    candidates = [resolved] if resolved == "portable_cpp" else [resolved, "portable_cpp"]
    source = _RETRIEVAL_CPP_SOURCE if kind == "retrieval" else _BM25_CPP_SOURCE
    binder = _bind_retrieval_argtypes if kind == "retrieval" else _bind_bm25_argtypes
    for cand in candidates:
        key = (kind, cand)
        with _LIB_LOCK:
            if key in _LIB_CACHE:
                cached = _LIB_CACHE[key]
                if cached is not None:
                    return cand, cached
                continue
        try:
            lib = _compile_shared_kernel(kind, source, cand)
            bound = binder(lib) if lib is not None else None
        except Exception:
            bound = None
        with _LIB_LOCK:
            _LIB_CACHE[key] = bound
        if bound is not None:
            return cand, bound
    return "numpy", None


def get_cpu_retrieval_lib(
    cpu_backend: CpuBackendMode = "auto",
    *,
    use_neon: bool = True,
) -> tuple[str, ctypes.CDLL | None]:
    """Return `(resolved_backend, loaded_cdll_or_none)` for vector index operations."""
    return _get_kernel("retrieval", cpu_backend=cpu_backend, use_neon=use_neon)


def get_cpu_bm25_lib(
    cpu_backend: CpuBackendMode = "auto",
    *,
    use_neon: bool = True,
) -> tuple[str, ctypes.CDLL | None]:
    """Return `(resolved_backend, loaded_cdll_or_none)` for native CSR / Block-Max WAND BM25."""
    return _get_kernel("bm25", cpu_backend=cpu_backend, use_neon=use_neon)


__all__ = [
    "CpuBackendMode",
    "VALID_CPU_BACKEND_MODES",
    "cpu_compile_flags",
    "detect_cpu_isa",
    "get_cpu_bm25_lib",
    "get_cpu_retrieval_lib",
    "resolve_cpu_backend",
]
