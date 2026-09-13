#include "sycl_ip/SYCL_Transforms.hpp"
#include <cmath>
#include <algorithm>

namespace sycl_ip {

bool SYCL_Transforms::invert(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, size_t count) {
    if (!d_in || !d_out || count == 0) return false;
    auto& q = ctx.getQueue();
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
        d_out[idx] = static_cast<uint16_t>(65535 - d_in[idx]);
    }).wait();
    return true;
}

bool SYCL_Transforms::subtract(SYCLContext& ctx, const uint16_t* d_inA, const uint16_t* d_inB, uint16_t* d_out, size_t count, int offset) {
    if (!d_inA || !d_inB || !d_out || count == 0) return false;
    auto& q = ctx.getQueue();
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
        int diff = static_cast<int>(d_inA[idx]) - static_cast<int>(d_inB[idx]) + offset;
        d_out[idx] = static_cast<uint16_t>(sycl::clamp(diff, 0, 65535));
    }).wait();
    return true;
}

bool SYCL_Transforms::subtractAbs(SYCLContext& ctx, const uint16_t* d_inA, const uint16_t* d_inB, uint16_t* d_out, size_t count) {
    if (!d_inA || !d_inB || !d_out || count == 0) return false;
    auto& q = ctx.getQueue();
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
        int diff = std::abs(static_cast<int>(d_inA[idx]) - static_cast<int>(d_inB[idx]));
        d_out[idx] = static_cast<uint16_t>(sycl::clamp(diff, 0, 65535));
    }).wait();
    return true;
}

bool SYCL_Transforms::flip(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int width, int height, bool horizontal) {
    if (!d_in || !d_out || width <= 0 || height <= 0) return false;
    auto& q = ctx.getQueue();
    q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        int y = static_cast<int>(id[0]);
        int x = static_cast<int>(id[1]);
        int srcY = horizontal ? y : (height - 1 - y);
        int srcX = horizontal ? (width - 1 - x) : x;
        d_out[y * width + x] = d_in[srcY * width + srcX];
    }).wait();
    return true;
}

bool SYCL_Transforms::rotate(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int inW, int inH, int angleDeg, int& outW, int& outH) {
    if (!d_in || !d_out || inW <= 0 || inH <= 0) return false;

    auto& q = ctx.getQueue();
    int angle = ((angleDeg % 360) + 360) % 360;

    if (angle == 90) {
        outW = inH;
        outH = inW;
        q.parallel_for(sycl::range<2>(outH, outW), [=](sycl::id<2> id) {
            int outY = static_cast<int>(id[0]);
            int outX = static_cast<int>(id[1]);
            int srcY = inH - 1 - outX;
            int srcX = outY;
            d_out[outY * outW + outX] = d_in[srcY * inW + srcX];
        }).wait();
    } else if (angle == 180) {
        outW = inW;
        outH = inH;
        q.parallel_for(sycl::range<2>(outH, outW), [=](sycl::id<2> id) {
            int outY = static_cast<int>(id[0]);
            int outX = static_cast<int>(id[1]);
            int srcY = inH - 1 - outY;
            int srcX = inW - 1 - outX;
            d_out[outY * outW + outX] = d_in[srcY * inW + srcX];
        }).wait();
    } else if (angle == 270) {
        outW = inH;
        outH = inW;
        q.parallel_for(sycl::range<2>(outH, outW), [=](sycl::id<2> id) {
            int outY = static_cast<int>(id[0]);
            int outX = static_cast<int>(id[1]);
            int srcY = outX;
            int srcX = inW - 1 - outY;
            d_out[outY * outW + outX] = d_in[srcY * inW + srcX];
        }).wait();
    } else {
        outW = inW;
        outH = inH;
        q.memcpy(d_out, d_in, static_cast<size_t>(inW) * inH * sizeof(uint16_t)).wait();
    }

    return true;
}

bool SYCL_Transforms::crop(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int inW, int inH, const RectROI& roi) {
    if (!d_in || !d_out || !roi.isValid(inW, inH)) return false;
    int outW = roi.width();
    int outH = roi.height();

    auto& q = ctx.getQueue();
    q.parallel_for(sycl::range<2>(outH, outW), [=](sycl::id<2> id) {
        int y = static_cast<int>(id[0]);
        int x = static_cast<int>(id[1]);
        int srcY = roi.top + y;
        int srcX = roi.left + x;
        d_out[y * outW + x] = d_in[srcY * inW + srcX];
    }).wait();

    return true;
}

} // namespace sycl_ip
