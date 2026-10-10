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

    constexpr size_t VEC = 4;
    size_t vecCount = count / VEC;
    size_t remainder = count % VEC;
    constexpr size_t WG = 256;
    size_t numWG = (vecCount + WG - 1) / WG;

    if (vecCount > 0) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(numWG * WG, WG), [=](sycl::nd_item<1> it) {
                size_t idx = it.get_global_id(0);
                if (idx < vecCount) {
                    auto inV = reinterpret_cast<const sycl::vec<uint16_t, VEC>*>(d_in)[idx];
                    const FitParam* pBase = d_params + (idx * VEC);
                    sycl::vec<uint16_t, VEC> outV;

                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        uint16_t rawVal = inV[k];
                        int A = static_cast<int>(rawVal);
                        if (useINL) {
                            int inlCorrection = d_inlTable[rawVal];
                            A = A - inlCorrection;
                            A = sycl::clamp(A, 0, 65535);
                        }
                        FitParam p = pBase[k];
                        float V = static_cast<float>(A) - p.p0;
                        float ampl = p.p1 * V + bkAverage;
                        outV[k] = static_cast<uint16_t>(sycl::clamp(ampl + 0.5f, 0.0f, 65535.0f));
                    }
                    reinterpret_cast<sycl::vec<uint16_t, VEC>*>(d_out)[idx] = outV;
                }
            });
        });
    }

    if (remainder > 0) {
        size_t off = vecCount * VEC;
        q.parallel_for(sycl::range<1>(remainder), [=](sycl::id<1> id) {
            size_t i = off + id;
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
            d_out[i] = static_cast<uint16_t>(sycl::clamp(ampl + 0.5f, 0.0f, 65535.0f));
        });
    }

    q.wait();
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

    constexpr size_t VEC = 8;
    size_t vecCount = count / VEC;
    size_t remainder = count % VEC;
    constexpr size_t WG = 256;
    size_t numWG = (vecCount + WG - 1) / WG;

    if (vecCount > 0) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(numWG * WG, WG), [=](sycl::nd_item<1> it) {
                size_t idx = it.get_global_id(0);
                if (idx < vecCount) {
                    auto in0 = reinterpret_cast<const sycl::vec<uint16_t, VEC>*>(d_out0)[idx];
                    auto in1 = reinterpret_cast<const sycl::vec<uint16_t, VEC>*>(d_out1)[idx];
                    sycl::vec<uint16_t, VEC> outV;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        float v = invDHV * (static_cast<float>(in0[k]) * dHV1 + static_cast<float>(in1[k]) * dHV0);
                        outV[k] = static_cast<uint16_t>(sycl::clamp(v + 0.5f, 0.0f, 65535.0f));
                    }
                    reinterpret_cast<sycl::vec<uint16_t, VEC>*>(d_outTarget)[idx] = outV;
                }
            });
        });
    }

    if (remainder > 0) {
        size_t off = vecCount * VEC;
        q.parallel_for(sycl::range<1>(remainder), [=](sycl::id<1> id) {
            size_t i = off + id;
            float v = invDHV * (static_cast<float>(d_out0[i]) * dHV1 + static_cast<float>(d_out1[i]) * dHV0);
            d_outTarget[i] = static_cast<uint16_t>(sycl::clamp(v + 0.5f, 0.0f, 65535.0f));
        });
    }

    q.wait();

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
