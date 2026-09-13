#include "sycl_ip/SYCL_Repair.hpp"
#include <cmath>
#include <algorithm>

namespace sycl_ip {

bool SYCL_Repair::repairImage(SYCLContext& ctx,
                              const uint16_t* d_in,
                              uint16_t* d_out,
                              size_t count,
                              const FitParam* d_params,
                              float bkAverage,
                              const int32_t* d_inlTable,
                              bool enableINL) {
    if (!d_in || !d_out || !d_params || count == 0) return false;

    auto& q = ctx.getQueue();
    bool useINL = enableINL && (d_inlTable != nullptr);

    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        uint16_t rawVal = d_in[i];
        int A = static_cast<int>(rawVal);

        if (useINL) {
            int inlCorrection = d_inlTable[rawVal];
            A = A - inlCorrection;
            A = sycl::clamp(A, 0, 65535);
        }

        FitParam p = d_params[i];
        float V = static_cast<float>(A) - p.p0;
        float ampl = p.p1 * V + bkAverage;

        ampl = sycl::clamp(ampl + 0.5f, 0.0f, 65535.0f);
        d_out[i] = static_cast<uint16_t>(ampl);
    }).wait();

    return true;
}

bool SYCL_Repair::repairImage(SYCLContext& ctx,
                              const FLRImage& rawImage,
                              FLRImage& outCalibratedImage,
                              const SYCL_Calibrate& calib,
                              bool enableINL) {
    size_t count = rawImage.getPixelCount();
    if (count == 0 || calib.getFitParams().size() != count) {
        std::cerr << "[SYCL_Repair] Error: Image size does not match calibration parameters ("
                  << count << " vs " << calib.getFitParams().size() << ")\n";
        return false;
    }

    uint16_t* d_in = ctx.allocateDevice<uint16_t>(count);
    uint16_t* d_out = ctx.allocateDevice<uint16_t>(count);
    FitParam* d_params = calib.getDeviceFitParams(ctx);
    int32_t* d_inl = enableINL ? calib.getDeviceINLTable(ctx) : nullptr;

    rawImage.copyToDevice(ctx, d_in);

    float bkAvg = calib.getHeader().bkAverage;
    repairImage(ctx, d_in, d_out, count, d_params, bkAvg, d_inl, enableINL);

    outCalibratedImage.resize(rawImage.getWidth(), rawImage.getHeight());
    outCalibratedImage.copyFromDevice(ctx, d_out);

    ctx.free<uint16_t>(d_in);
    ctx.free<uint16_t>(d_out);
    ctx.free<FitParam>(d_params);
    if (d_inl) ctx.free<int32_t>(d_inl);

    return true;
}

bool SYCL_Repair::interpolateVoltages(SYCLContext& ctx,
                                     const uint16_t* d_out0, float hv0,
                                     const uint16_t* d_out1, float hv1,
                                     uint16_t* d_outTarget, float targetHV,
                                     size_t count) {
    if (!d_out0 || !d_out1 || !d_outTarget || count == 0) return false;
    float dHV10 = hv1 - hv0;
    if (std::abs(dHV10) < 1e-6f) return false;

    float invDHV = 1.0f / dHV10;
    float dHV1 = hv1 - targetHV;
    float dHV0 = targetHV - hv0;

    auto& q = ctx.getQueue();
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        float v = invDHV * (static_cast<float>(d_out0[i]) * dHV1 + static_cast<float>(d_out1[i]) * dHV0);
        v = sycl::clamp(v + 0.5f, 0.0f, 65535.0f);
        d_outTarget[i] = static_cast<uint16_t>(v);
    }).wait();

    return true;
}

void SYCL_Repair::hostRepairImage(const uint16_t* in, uint16_t* out, size_t count, const FitParam* params, float bkAverage, const int32_t* inlTable, bool enableINL) {
    bool useINL = enableINL && (inlTable != nullptr);
    for (size_t i = 0; i < count; ++i) {
        uint16_t rawVal = in[i];
        int A = rawVal;
        if (useINL) {
            A = A - inlTable[rawVal];
            if (A < 0) A = 0;
            if (A > 65535) A = 65535;
        }
        FitParam p = params[i];
        double V = static_cast<double>(A) - static_cast<double>(p.p0);
        double ampl = static_cast<double>(p.p1) * V + static_cast<double>(bkAverage);
        if (ampl < 0.0) out[i] = 0;
        else if (ampl >= 65535.0) out[i] = 65535;
        else out[i] = static_cast<uint16_t>(ampl + 0.5);
    }
}

} // namespace sycl_ip
