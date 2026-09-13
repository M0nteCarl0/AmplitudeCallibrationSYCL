#pragma once

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include "FLR_IO.hpp"
#include <cstdint>

namespace sycl_ip {

class SYCL_IP_API SYCL_Transforms {
public:
    // Invert: out[i] = 65535 - in[i]
    static bool invert(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, size_t count);

    // Subtract: out[i] = clamp(inA[i] - inB[i] + offset)
    static bool subtract(SYCLContext& ctx, const uint16_t* d_inA, const uint16_t* d_inB, uint16_t* d_out, size_t count, int offset = 0);

    // Subtract Absolute: out[i] = |inA[i] - inB[i]|
    static bool subtractAbs(SYCLContext& ctx, const uint16_t* d_inA, const uint16_t* d_inB, uint16_t* d_out, size_t count);

    // Flip horizontal or vertical
    static bool flip(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int width, int height, bool horizontal = true);

    // Rotate 90, 180, 270 degrees
    static bool rotate(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int inW, int inH, int angleDeg, int& outW, int& outH);

    // Crop ROI
    static bool crop(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int inW, int inH, const RectROI& roi);
};

} // namespace sycl_ip
