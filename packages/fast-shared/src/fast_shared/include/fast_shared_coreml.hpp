#pragma once

#ifdef __OBJC__
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <cstdint>
#include <cstring>

namespace fast_shared {

inline bool is_dense_4d(MLMultiArray* arr) {
    NSArray<NSNumber*>* sh = arr.shape;
    NSArray<NSNumber*>* st = arr.strides;
    if (sh.count < 4 || st.count < 4) return false;
    const NSInteger s3 = sh[3].integerValue;
    const NSInteger s2 = sh[2].integerValue;
    const NSInteger s1 = sh[1].integerValue;
    return st[3].integerValue == 1 &&
           st[2].integerValue == s3 &&
           st[1].integerValue == s2 * s3 &&
           st[0].integerValue == s1 * s2 * s3;
}

inline MLMultiArray* make_f16_array(NSArray<NSNumber*>* shape, NSError** err_out = nil) {
    NSError* err = nil;
    MLMultiArray* a = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:&err];
    if (err_out) *err_out = err;
    if (!a || err) return nil;
    return a;
}

inline void copy_to_mla_u16(MLMultiArray* dst, const uint16_t* src) {
    NSArray<NSNumber*>* sh = dst.shape;
    NSArray<NSNumber*>* st = dst.strides;
    uint16_t* dptr = reinterpret_cast<uint16_t*>(dst.dataPointer);
    const NSInteger d0 = sh[0].integerValue;
    const NSInteger d1 = sh[1].integerValue;
    const NSInteger d2 = sh[2].integerValue;
    const NSInteger d3 = sh[3].integerValue;
    if (is_dense_4d(dst)) {
        std::memcpy(dptr, src, static_cast<size_t>(d0 * d1 * d2 * d3) * sizeof(uint16_t));
        return;
    }
    const NSInteger st0 = st[0].integerValue;
    const NSInteger st1 = st[1].integerValue;
    const NSInteger st2 = st[2].integerValue;
    const NSInteger st3 = st[3].integerValue;
    for (NSInteger i0 = 0; i0 < d0; ++i0) {
        for (NSInteger i1 = 0; i1 < d1; ++i1) {
            for (NSInteger i2 = 0; i2 < d2; ++i2) {
                const uint16_t* srow = src + ((i0 * d1 + i1) * d2 + i2) * d3;
                uint16_t* drow = dptr + i0 * st0 + i1 * st1 + i2 * st2;
                if (st3 == 1) {
                    std::memcpy(drow, srow, static_cast<size_t>(d3) * sizeof(uint16_t));
                } else {
                    for (NSInteger i3 = 0; i3 < d3; ++i3) {
                        drow[i3 * st3] = srow[i3];
                    }
                }
            }
        }
    }
}

inline void copy_from_mla_u16(uint16_t* dst, MLMultiArray* src) {
    NSArray<NSNumber*>* sh = src.shape;
    NSArray<NSNumber*>* st = src.strides;
    const uint16_t* sptr = reinterpret_cast<const uint16_t*>(src.dataPointer);
    const NSInteger d0 = sh[0].integerValue;
    const NSInteger d1 = sh[1].integerValue;
    const NSInteger d2 = sh[2].integerValue;
    const NSInteger d3 = sh[3].integerValue;
    if (is_dense_4d(src)) {
        std::memcpy(dst, sptr, static_cast<size_t>(d0 * d1 * d2 * d3) * sizeof(uint16_t));
        return;
    }
    const NSInteger st0 = st[0].integerValue;
    const NSInteger st1 = st[1].integerValue;
    const NSInteger st2 = st[2].integerValue;
    const NSInteger st3 = st[3].integerValue;
    for (NSInteger i0 = 0; i0 < d0; ++i0) {
        for (NSInteger i1 = 0; i1 < d1; ++i1) {
            for (NSInteger i2 = 0; i2 < d2; ++i2) {
                uint16_t* drow = dst + ((i0 * d1 + i1) * d2 + i2) * d3;
                const uint16_t* srow = sptr + i0 * st0 + i1 * st1 + i2 * st2;
                if (st3 == 1) {
                    std::memcpy(drow, srow, static_cast<size_t>(d3) * sizeof(uint16_t));
                } else {
                    for (NSInteger i3 = 0; i3 < d3; ++i3) {
                        drow[i3] = srow[i3 * st3];
                    }
                }
            }
        }
    }
}

}  // namespace fast_shared
#endif
