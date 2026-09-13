#include "sycl_ip/SYCL_Statistics.hpp"
#include <cmath>
#include <algorithm>

namespace sycl_ip {

double SYCL_Statistics::computeMean(SYCLContext& ctx, const uint16_t* d_image, int width, int height, const RectROI* roi) {
    if (!d_image || width <= 0 || height <= 0) return 0.0;

    int roiL = 0, roiT = 0, roiW = width, roiH = height;
    if (roi) {
        roiL = std::max(0, roi->left);
        roiT = std::max(0, roi->top);
        roiW = std::max(0, std::min(width, roi->right) - roiL);
        roiH = std::max(0, std::min(height, roi->bottom) - roiT);
    }
    size_t totalPixels = static_cast<size_t>(roiW) * roiH;
    if (totalPixels == 0) return 0.0;

    auto& q = ctx.getQueue();
    uint64_t* d_sum = ctx.allocateShared<uint64_t>(1);
    *d_sum = 0;

    q.submit([&](sycl::handler& h) {
        auto sum_reduction = sycl::reduction(d_sum, sycl::plus<uint64_t>());
        h.parallel_for(sycl::range<2>(roiH, roiW), sum_reduction, [=](sycl::id<2> id, auto& sum_acc) {
            int y = roiT + static_cast<int>(id[0]);
            int x = roiL + static_cast<int>(id[1]);
            sum_acc += static_cast<uint64_t>(d_image[y * width + x]);
        });
    }).wait();

    double mean = static_cast<double>(*d_sum) / static_cast<double>(totalPixels);
    ctx.free<uint64_t>(d_sum);
    return mean;
}

void SYCL_Statistics::computeMeanRMS(SYCLContext& ctx, const uint16_t* d_image, int width, int height, double& outMean, double& outRMS, const RectROI* roi) {
    if (!d_image || width <= 0 || height <= 0) {
        outMean = 0.0;
        outRMS = 0.0;
        return;
    }

    int roiL = 0, roiT = 0, roiW = width, roiH = height;
    if (roi) {
        roiL = std::max(0, roi->left);
        roiT = std::max(0, roi->top);
        roiW = std::max(0, std::min(width, roi->right) - roiL);
        roiH = std::max(0, std::min(height, roi->bottom) - roiT);
    }
    size_t totalPixels = static_cast<size_t>(roiW) * roiH;
    if (totalPixels == 0) {
        outMean = 0.0;
        outRMS = 0.0;
        return;
    }

    outMean = computeMean(ctx, d_image, width, height, roi);

    auto& q = ctx.getQueue();
    float* d_sqDiffSum = ctx.allocateShared<float>(1);
    *d_sqDiffSum = 0.0f;
    float meanF = static_cast<float>(outMean);

    q.submit([&](sycl::handler& h) {
        auto sq_reduction = sycl::reduction(d_sqDiffSum, sycl::plus<float>());
        h.parallel_for(sycl::range<2>(roiH, roiW), sq_reduction, [=](sycl::id<2> id, auto& sq_acc) {
            int y = roiT + static_cast<int>(id[0]);
            int x = roiL + static_cast<int>(id[1]);
            float diff = static_cast<float>(d_image[y * width + x]) - meanF;
            sq_acc += diff * diff;
        });
    }).wait();

    outRMS = std::sqrt(static_cast<double>(*d_sqDiffSum) / static_cast<double>(totalPixels));
    ctx.free<float>(d_sqDiffSum);
}

void SYCL_Statistics::computeSlidingMeanRMS(SYCLContext& ctx, const uint16_t* d_image, int width, int height, const RectROI& roi, double& outMean, double& outRMS, int rmsSize, int rmsStep) {
    int roiW = roi.width();
    int roiH = roi.height();

    if (rmsStep <= 0 || rmsSize > roiW || rmsSize > roiH) {
        outMean = 0.0;
        outRMS = 0.0;
        return;
    }

    int nx = (roiW - rmsSize) / rmsStep;
    int ny = (roiH - rmsSize) / rmsStep;
    size_t totalTiles = static_cast<size_t>(nx) * ny;
    if (totalTiles == 0) {
        outMean = 0.0;
        outRMS = 0.0;
        return;
    }

    auto& q = ctx.getQueue();
    float* d_sums = ctx.allocateShared<float>(2); // [0] = sumMean, [1] = sumRMS
    d_sums[0] = 0.0f;
    d_sums[1] = 0.0f;

    int x0 = roi.left;
    int y0 = roi.top;
    float tileSizeArea = static_cast<float>(rmsSize * rmsSize);

    q.submit([&](sycl::handler& h) {
        auto redMean = sycl::reduction(&d_sums[0], sycl::plus<float>());
        auto redRMS  = sycl::reduction(&d_sums[1], sycl::plus<float>());

        h.parallel_for(sycl::range<2>(ny, nx), redMean, redRMS, [=](sycl::id<2> id, auto& accMean, auto& accRMS) {
            int tileY = static_cast<int>(id[0]);
            int tileX = static_cast<int>(id[1]);

            int X = x0 + tileX * rmsStep;
            int Y = y0 + tileY * rmsStep;

            float tileSum = 0.0f;
            for (int dy = 0; dy < rmsSize; ++dy) {
                for (int dx = 0; dx < rmsSize; ++dx) {
                    tileSum += static_cast<float>(d_image[(Y + dy) * width + (X + dx)]);
                }
            }
            float tileMean = tileSum / tileSizeArea;

            float tileDiffSq = 0.0f;
            for (int dy = 0; dy < rmsSize; ++dy) {
                for (int dx = 0; dx < rmsSize; ++dx) {
                    float diff = static_cast<float>(d_image[(Y + dy) * width + (X + dx)]) - tileMean;
                    tileDiffSq += diff * diff;
                }
            }
            float tileRms = sycl::sqrt(tileDiffSq / tileSizeArea);

            accMean += tileMean;
            accRMS += tileRms;
        });
    }).wait();

    outMean = static_cast<double>(d_sums[0]) / static_cast<double>(totalTiles);
    outRMS  = static_cast<double>(d_sums[1]) / static_cast<double>(totalTiles);

    ctx.free<float>(d_sums);
}

void SYCL_Statistics::computeMinMax(SYCLContext& ctx, const uint16_t* d_image, int width, int height, uint16_t& outMin, uint16_t& outMax) {
    if (!d_image || width <= 0 || height <= 0) {
        outMin = 0;
        outMax = 0;
        return;
    }

    size_t total = static_cast<size_t>(width) * height;
    auto& q = ctx.getQueue();
    uint16_t* d_results = ctx.allocateShared<uint16_t>(2); // [0] = min, [1] = max
    d_results[0] = 0xFFFF;
    d_results[1] = 0x0000;

    q.submit([&](sycl::handler& h) {
        auto redMin = sycl::reduction(&d_results[0], sycl::minimum<uint16_t>());
        auto redMax = sycl::reduction(&d_results[1], sycl::maximum<uint16_t>());

        h.parallel_for(sycl::range<1>(total), redMin, redMax, [=](sycl::id<1> idx, auto& accMin, auto& accMax) {
            uint16_t val = d_image[idx];
            accMin.combine(val);
            accMax.combine(val);
        });
    }).wait();

    outMin = d_results[0];
    outMax = d_results[1];
    ctx.free<uint16_t>(d_results);
}

// Host reference implementations
double SYCL_Statistics::hostComputeMean(const uint16_t* image, int width, int height, const RectROI* roi) {
    if (!image || width <= 0 || height <= 0) return 0.0;
    int roiL = roi ? roi->left : 0;
    int roiT = roi ? roi->top : 0;
    int roiW = roi ? roi->width() : width;
    int roiH = roi ? roi->height() : height;

    double sum = 0.0;
    size_t count = 0;
    for (int y = roiT; y < roiT + roiH; ++y) {
        for (int x = roiL; x < roiL + roiW; ++x) {
            sum += image[y * width + x];
            count++;
        }
    }
    return count > 0 ? sum / count : 0.0;
}

void SYCL_Statistics::hostComputeMeanRMS(const uint16_t* image, int width, int height, double& outMean, double& outRMS, const RectROI* roi) {
    outMean = hostComputeMean(image, width, height, roi);
    int roiL = roi ? roi->left : 0;
    int roiT = roi ? roi->top : 0;
    int roiW = roi ? roi->width() : width;
    int roiH = roi ? roi->height() : height;

    double sumSq = 0.0;
    size_t count = 0;
    for (int y = roiT; y < roiT + roiH; ++y) {
        for (int x = roiL; x < roiL + roiW; ++x) {
            double diff = image[y * width + x] - outMean;
            sumSq += diff * diff;
            count++;
        }
    }
    outRMS = count > 0 ? std::sqrt(sumSq / count) : 0.0;
}

void SYCL_Statistics::hostComputeSlidingMeanRMS(const uint16_t* image, int width, int height, const RectROI& roi, double& outMean, double& outRMS, int rmsSize, int rmsStep) {
    int roiW = roi.width();
    int roiH = roi.height();

    if (rmsStep <= 0 || rmsSize > roiW || rmsSize > roiH) {
        outMean = 0.0;
        outRMS = 0.0;
        return;
    }

    int nx = (roiW - rmsSize) / rmsStep;
    int ny = (roiH - rmsSize) / rmsStep;

    double summRms = 0.0;
    double summMean = 0.0;

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int X = roi.left + x * rmsStep;
            int Y = roi.top + y * rmsStep;
            RectROI tileRoi(X, Y, X + rmsSize, Y + rmsSize);
            double tileMean = 0.0, tileRms = 0.0;
            hostComputeMeanRMS(image, width, height, tileMean, tileRms, &tileRoi);
            summMean += tileMean;
            summRms += tileRms;
        }
    }

    double total = static_cast<double>(nx) * ny;
    outMean = total > 0 ? summMean / total : 0.0;
    outRMS = total > 0 ? summRms / total : 0.0;
}

} // namespace sycl_ip
