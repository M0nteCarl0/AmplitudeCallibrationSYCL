#pragma once

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include "FLR_IO.hpp"
#include "SYCL_Calibrate.hpp"

namespace sycl_ip {

class SYCL_IP_API SYCL_Repair {
public:
    // Device-side single image calibration repair kernel
    static bool repairImage(SYCLContext& ctx, 
                           const uint16_t* d_in, 
                           uint16_t* d_out, 
                           size_t count, 
                           const FitParam* d_params, 
                           float bkAverage,
                           const int32_t* d_inlTable = nullptr,
                           bool enableINL = true);

    // High-level image repair from host FLRImage using calibration model
    static bool repairImage(SYCLContext& ctx,
                           const FLRImage& rawImage,
                           FLRImage& outCalibratedImage,
                           const SYCL_Calibrate& calib,
                           bool enableINL = true);

    // Multi-voltage linear interpolation (between hv0 and hv1 to targetHV)
    static bool interpolateVoltages(SYCLContext& ctx,
                                   const uint16_t* d_out0, float hv0,
                                   const uint16_t* d_out1, float hv1,
                                   uint16_t* d_outTarget, float targetHV,
                                   size_t count);

    // Host CPU reference implementation
    static void hostRepairImage(const uint16_t* in, uint16_t* out, size_t count, const FitParam* params, float bkAverage, const int32_t* inlTable = nullptr, bool enableINL = true);
};

} // namespace sycl_ip
