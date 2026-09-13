#pragma once

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include "FLR_IO.hpp"
#include <cstdint>

namespace sycl_ip {

class SYCL_IP_API SYCL_Statistics {
public:
    // Parallel reduction of Mean over whole image or ROI on GPU/CPU
    static double computeMean(SYCLContext& ctx, const uint16_t* d_image, int width, int height, const RectROI* roi = nullptr);

    // Parallel reduction of Mean and RMS over whole image or ROI on GPU/CPU
    static void computeMeanRMS(SYCLContext& ctx, const uint16_t* d_image, int width, int height, double& outMean, double& outRMS, const RectROI* roi = nullptr);

    // Sliding grid mean and RMS (SYCL equivalent of legacy FulfilMeanRms)
    static void computeSlidingMeanRMS(SYCLContext& ctx, const uint16_t* d_image, int width, int height, const RectROI& roi, double& outMean, double& outRMS, int rmsSize = 30, int rmsStep = 10);

    // Parallel Min and Max reduction
    static void computeMinMax(SYCLContext& ctx, const uint16_t* d_image, int width, int height, uint16_t& outMin, uint16_t& outMax);

    // Host CPU reference implementations for validation
    static double hostComputeMean(const uint16_t* image, int width, int height, const RectROI* roi = nullptr);
    static void hostComputeMeanRMS(const uint16_t* image, int width, int height, double& outMean, double& outRMS, const RectROI* roi = nullptr);
    static void hostComputeSlidingMeanRMS(const uint16_t* image, int width, int height, const RectROI& roi, double& outMean, double& outRMS, int rmsSize = 30, int rmsStep = 10);
};

} // namespace sycl_ip
