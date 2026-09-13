#pragma once

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include "FLR_IO.hpp"
#include <cstdint>

namespace sycl_ip {

class SYCL_IP_API SYCL_Filters {
public:
    // 2D Separable Box Blur (Smooth)
    static bool smooth(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int width, int height, int radiusK = 2);

    // Adaptive noise averaging filter (SYCL equivalent of legacy AdaptAverage)
    static bool adaptAverage(SYCLContext& ctx, 
                            const uint16_t* d_in, 
                            const uint16_t* d_aver, 
                            const uint16_t* d_rms, 
                            uint16_t* d_out, 
                            int width, 
                            int height,
                            double p0Noise = 10.0, double p1Noise = 0.05,
                            double p0Quench = 0.0, double p1Quench = 1.0,
                            int smoothRadius = 2,
                            int kSize = 1);

    // Complete Adaptive Noise Filter Pipeline (smooth -> diff -> smooth diff -> adaptAverage)
    static bool applyAdaptiveFilter(SYCLContext& ctx,
                                    const uint16_t* d_in,
                                    uint16_t* d_out,
                                    int width,
                                    int height,
                                    double p0Noise = 10.0, double p1Noise = 0.05,
                                    double z0Quench = 1.0, double y0Quench = 1.0,
                                    double z1Quench = 4.0, double y1Quench = 0.0,
                                    int smoothK = 2);

    // Zoom out (subsampling)
    static bool copyZoom(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int inW, int inH, int factor = 2);

    // Host CPU reference implementations
    static void hostSmooth(const uint16_t* in, uint16_t* out, int width, int height, int radiusK = 2);
};

} // namespace sycl_ip
