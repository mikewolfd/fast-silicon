#pragma once

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

#ifdef __OBJC__
#import <CoreML/CoreML.h>
#endif

namespace fast_shared {

enum class ComputeMode : int {
    kCpuAndNeuralEngine = 0,
    kCpuOnly = 1,
    kAll = 2,
};

inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline void l2_normalize_f32(const float* src, float* dst, int dim) {
    float n2 = 0.0f;
    vDSP_svesq(src, 1, &n2, static_cast<vDSP_Length>(dim));
    float inv = 1.0f / std::sqrt(std::max(n2, 1e-24f));
    vDSP_vsmul(src, 1, &inv, dst, 1, static_cast<vDSP_Length>(dim));
}

#ifdef __OBJC__
inline MLComputeUnits resolve_compute_units(ComputeMode mode) {
    switch (mode) {
        case ComputeMode::kCpuOnly:
            return MLComputeUnitsCPUOnly;
        case ComputeMode::kCpuAndNeuralEngine:
            return MLComputeUnitsCPUAndNeuralEngine;
        case ComputeMode::kAll:
        default:
            return MLComputeUnitsAll;
    }
}
#endif

}  // namespace fast_shared
