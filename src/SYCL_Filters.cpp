#include "sycl_ip/SYCL_Filters.hpp"
#include <cmath>
#include <algorithm>

namespace sycl_ip {

bool SYCL_Filters::smooth(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int width, int height, int radiusK) {
    if (!d_in || !d_out || width <= 0 || height <= 0 || radiusK <= 0) return false;

    size_t total = static_cast<size_t>(width) * height;
    auto& q = ctx.getQueue();
    uint16_t* d_temp = ctx.allocateDevice<uint16_t>(total);

    int winSize = 2 * radiusK + 1;
    float invWin = 1.0f / static_cast<float>(winSize);

    // Pass 1: Horizontal 1D Blur
    q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        int y = static_cast<int>(id[0]);
        int x = static_cast<int>(id[1]);

        int sum = 0;
        for (int k = -radiusK; k <= radiusK; ++k) {
            int cx = x + k;
            if (cx < 0) cx = -cx;
            else if (cx >= width) cx = 2 * width - cx - 1;
            if (cx < 0) cx = 0;
            if (cx >= width) cx = width - 1;

            sum += static_cast<int>(d_in[y * width + cx]);
        }
        float val = static_cast<float>(sum) * invWin;
        d_temp[y * width + x] = static_cast<uint16_t>(sycl::clamp(val + 0.5f, 0.0f, 65535.0f));
    }).wait();

    // Pass 2: Vertical 1D Blur
    q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        int y = static_cast<int>(id[0]);
        int x = static_cast<int>(id[1]);

        int sum = 0;
        for (int k = -radiusK; k <= radiusK; ++k) {
            int cy = y + k;
            if (cy < 0) cy = -cy;
            else if (cy >= height) cy = 2 * height - cy - 1;
            if (cy < 0) cy = 0;
            if (cy >= height) cy = height - 1;

            sum += static_cast<int>(d_temp[cy * width + x]);
        }
        float val = static_cast<float>(sum) * invWin;
        d_out[y * width + x] = static_cast<uint16_t>(sycl::clamp(val + 0.5f, 0.0f, 65535.0f));
    }).wait();

    ctx.free<uint16_t>(d_temp);
    return true;
}

bool SYCL_Filters::adaptAverage(SYCLContext& ctx,
                               const uint16_t* d_in,
                               const uint16_t* d_aver,
                               const uint16_t* d_rms,
                               uint16_t* d_out,
                               int width,
                               int height,
                               double p0Noise, double p1Noise,
                               double p0Quench, double p1Quench,
                               int smoothRadius,
                               int kSize) {
    if (!d_in || !d_aver || !d_rms || !d_out || width <= 0 || height <= 0) return false;

    auto& q = ctx.getQueue();
    int N = (2 * kSize + 1) * (2 * kSize + 1);
    float p0N = static_cast<float>(p0Noise);
    float p1N = static_cast<float>(p1Noise);
    float p0Q = static_cast<float>(p0Quench);
    float p1Q = static_cast<float>(p1Quench);

    q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        int y = static_cast<int>(id[0]);
        int x = static_cast<int>(id[1]);

        float RmsF = p1N * static_cast<float>(d_aver[y * width + x]) + p0N;
        float Rms0 = static_cast<float>(d_rms[y * width + x]);
        float Z = (RmsF > 1e-5f) ? (Rms0 / RmsF) : 1.0f;
        if (Z < 1.0f) Z = 1.0f;

        float W = p1Q * Z + p0Q;
        if (W < 0.0f) W = 0.0f;
        if (W > 1000.0f) W = 1000.0f;

        float Norma = static_cast<float>(N - 1) + W;
        if (Norma < 1.0f) Norma = 1.0f;

        int sum = 0;
        for (int dy = -kSize; dy <= kSize; ++dy) {
            int cy = y + dy;
            if (cy < 0) cy = -cy;
            else if (cy >= height) cy = 2 * height - cy - 1;
            if (cy < 0) cy = 0;
            if (cy >= height) cy = height - 1;

            for (int dx = -kSize; dx <= kSize; ++dx) {
                int cx = x + dx;
                if (cx < 0) cx = -cx;
                else if (cx >= width) cx = 2 * width - cx - 1;
                if (cx < 0) cx = 0;
                if (cx >= width) cx = width - 1;

                sum += static_cast<int>(d_in[cy * width + cx]);
            }
        }

        uint16_t centerVal = d_in[y * width + x];
        float totalSum = static_cast<float>(sum) + static_cast<float>(centerVal) * (W - 1.0f);
        float outVal = totalSum / Norma;

        d_out[y * width + x] = static_cast<uint16_t>(sycl::clamp(outVal + 0.5f, 0.0f, 65535.0f));
    }).wait();

    return true;
}

bool SYCL_Filters::applyAdaptiveFilter(SYCLContext& ctx,
                                      const uint16_t* d_in,
                                      uint16_t* d_out,
                                      int width,
                                      int height,
                                      double p0Noise, double p1Noise,
                                      double z0Quench, double y0Quench,
                                      double z1Quench, double y1Quench,
                                      int smoothK) {
    if (!d_in || !d_out || width <= 0 || height <= 0) return false;

    // Calculate quench line parameters
    double p1Quench = 0.0, p0Quench = 0.0;
    if (std::abs(z1Quench - z0Quench) > 1e-6) {
        p1Quench = (y1Quench - y0Quench) / (z1Quench - z0Quench);
        p0Quench = y0Quench - p1Quench * z0Quench;
    }

    size_t total = static_cast<size_t>(width) * height;
    auto& q = ctx.getQueue();

    uint16_t* d_aver = ctx.allocateDevice<uint16_t>(total);
    uint16_t* d_diff = ctx.allocateDevice<uint16_t>(total);
    uint16_t* d_rms  = ctx.allocateDevice<uint16_t>(total);

    // Step 1: Smooth input
    smooth(ctx, d_in, d_aver, width, height, smoothK);

    // Step 2: Subtract absolute difference |in - aver|
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
        int diff = std::abs(static_cast<int>(d_in[idx]) - static_cast<int>(d_aver[idx]));
        d_diff[idx] = static_cast<uint16_t>(diff);
    }).wait();

    // Step 3: Smooth the difference to get local RMS
    smooth(ctx, d_diff, d_rms, width, height, 2);

    // Step 4: Run adaptive filter
    adaptAverage(ctx, d_in, d_aver, d_rms, d_out, width, height, p0Noise, p1Noise, p0Quench, p1Quench, smoothK, 1);

    ctx.free<uint16_t>(d_aver);
    ctx.free<uint16_t>(d_diff);
    ctx.free<uint16_t>(d_rms);

    return true;
}

bool SYCL_Filters::copyZoom(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int inW, int inH, int factor) {
    if (!d_in || !d_out || inW <= 0 || inH <= 0 || factor <= 0) return false;
    int outW = inW / factor;
    int outH = inH / factor;

    auto& q = ctx.getQueue();
    q.parallel_for(sycl::range<2>(outH, outW), [=](sycl::id<2> id) {
        int outY = static_cast<int>(id[0]);
        int outX = static_cast<int>(id[1]);
        int inY = outY * factor;
        int inX = outX * factor;
        d_out[outY * outW + outX] = d_in[inY * inW + inX];
    }).wait();

    return true;
}

void SYCL_Filters::hostSmooth(const uint16_t* in, uint16_t* out, int width, int height, int radiusK) {
    std::vector<uint16_t> temp(width * height);
    int winSize = 2 * radiusK + 1;
    double invWin = 1.0 / winSize;

    // Horizontal
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int k = -radiusK; k <= radiusK; ++k) {
                int cx = x + k;
                if (cx < 0) cx = -cx;
                else if (cx >= width) cx = 2 * width - cx - 1;
                if (cx < 0) cx = 0;
                if (cx >= width) cx = width - 1;
                sum += in[y * width + cx];
            }
            temp[y * width + x] = static_cast<uint16_t>(std::clamp(sum * invWin + 0.5, 0.0, 65535.0));
        }
    }

    // Vertical
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int k = -radiusK; k <= radiusK; ++k) {
                int cy = y + k;
                if (cy < 0) cy = -cy;
                else if (cy >= height) cy = 2 * height - cy - 1;
                if (cy < 0) cy = 0;
                if (cy >= height) cy = height - 1;
                sum += temp[cy * width + x];
            }
            out[y * width + x] = static_cast<uint16_t>(std::clamp(sum * invWin + 0.5, 0.0, 65535.0));
        }
    }
}

} // namespace sycl_ip
