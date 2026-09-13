#pragma once

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include "FLR_IO.hpp"
#include <cstdint>

namespace sycl_ip {

class SYCL_IP_API SYCL_Suppress {
public:
    // Parallel annular concentric spike suppression filter (3x3, 5x5, 7x7 windows)
    static bool suppressSpikes(SYCLContext& ctx, 
                               const uint16_t* d_in, 
                               uint16_t* d_out, 
                               int width, 
                               int height,
                               double p0Noise = 10.0, 
                               double p1Noise = 0.05,
                               double coeff1 = 2.0, 
                               double coeff2 = 3.0, 
                               int coeff3 = 2);

    // Host CPU reference implementation
    static void hostSuppressSpikes(const uint16_t* in, 
                                  uint16_t* out, 
                                  int width, 
                                  int height,
                                  double p0Noise = 10.0, 
                                  double p1Noise = 0.05,
                                  double coeff1 = 2.0, 
                                  double coeff2 = 3.0, 
                                  int coeff3 = 2);
};

} // namespace sycl_ip
