#include "sycl_ip/SYCL_Calibrate.hpp"
#include "sycl_ip/SYCL_Statistics.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

namespace sycl_ip {

SYCL_Calibrate::SYCL_Calibrate(int width, int height)
    : m_width(width), m_height(height) {
    m_header.w = width;
    m_header.h = height;
    m_inlAmpl.resize(INL_TABLE_SIZE, 0);
    m_inlSumm.resize(INL_TABLE_SIZE, 0.0);
    m_inlNum.resize(INL_TABLE_SIZE, 0);
    m_fitParams.resize(static_cast<size_t>(width) * height);
}

bool SYCL_Calibrate::calibrate(SYCLContext& ctx,
                               const std::vector<FLRImage>& exposureImages,
                               const std::vector<double>& exposureMeans,
                               const std::vector<double>& exposureRMSs,
                               float hv, int binning, int fokus, bool computeINL) {
    if (exposureImages.size() < 2) {
        std::cerr << "[SYCL_Calibrate] Error: Need at least 2 exposure points (Background + signal) for calibration.\n";
        return false;
    }

    size_t numPoints = exposureImages.size();
    size_t totalPixels = static_cast<size_t>(m_width) * m_height;

    m_header.key = CALIBRATION_KEY;
    m_header.x0 = 0;
    m_header.y0 = 0;
    m_header.w = m_width;
    m_header.h = m_height;
    m_header.fokus = fokus;
    m_header.binning = binning;
    m_header.hv = hv;
    m_header.bkAverage = static_cast<float>(exposureMeans[0]);
    m_header.time = std::time(nullptr);

    m_fitParams.resize(totalPixels);
    std::fill(m_inlAmpl.begin(), m_inlAmpl.end(), 0);
    std::fill(m_inlSumm.begin(), m_inlSumm.end(), 0.0);
    std::fill(m_inlNum.begin(), m_inlNum.end(), 0);

    // Allocate USM device buffers for each exposure image
    std::vector<uint16_t*> d_images(numPoints, nullptr);
    for (size_t i = 0; i < numPoints; ++i) {
        d_images[i] = ctx.allocateDevice<uint16_t>(totalPixels);
        exposureImages[i].copyToDevice(ctx, d_images[i]);
    }

    // Allocate USM shared array for device pointers
    uint16_t** d_imagePtrs = ctx.allocateShared<uint16_t*>(numPoints);
    for (size_t i = 0; i < numPoints; ++i) {
        d_imagePtrs[i] = d_images[i];
    }

    // Means and RMSs in USM shared memory (using float for fp64-free portability)
    float* d_means = ctx.allocateShared<float>(numPoints);
    float* d_rms   = ctx.allocateShared<float>(numPoints);
    for (size_t i = 0; i < numPoints; ++i) {
        d_means[i] = static_cast<float>(exposureMeans[i]);
        d_rms[i]   = static_cast<float>(exposureRMSs[i]);
    }

    // Output fit parameters buffer
    FitParam* d_params = ctx.allocateDevice<FitParam>(totalPixels);

    // INL accumulation buffers (shared, float and int32)
    float* d_inlSumm = ctx.allocateShared<float>(INL_TABLE_SIZE);
    int32_t* d_inlNum = ctx.allocateShared<int32_t>(INL_TABLE_SIZE);
    std::fill_n(d_inlSumm, INL_TABLE_SIZE, 0.0f);
    std::fill_n(d_inlNum, INL_TABLE_SIZE, 0);

    auto& q = ctx.getQueue();

    std::cout << "[SYCL_Calibrate] Launching parallel pixel regression on "
              << ctx.getDevice().get_info<sycl::info::device::name>()
              << " (" << totalPixels << " pixels, " << numPoints << " exposure points)...\n";

    int width = m_width;
    int height = m_height;
    int margin = 100; // exclude border for INL accumulation

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(totalPixels), [=](sycl::id<1> idx) {
            size_t p = idx[0];
            uint16_t bk = d_imagePtrs[0][p];
            float p0 = static_cast<float>(bk);

            float num = 0.0f;
            float den = 0.0f;
            float bkMean = d_means[0];

            for (size_t k = 1; k < numPoints; ++k) {
                float X = static_cast<float>(d_imagePtrs[k][p]) - static_cast<float>(bk);
                float Y = d_means[k] - bkMean;
                float Er = d_rms[k];
                if (Er <= 1e-5f) Er = 1.0f;

                num += (Y * X) / Er;
                den += (X * X) / Er;
            }

            float p1 = (den > 1e-10f) ? (num / den) : 1.0f;
            d_params[p].p0 = p0;
            d_params[p].p1 = p1;

            if (computeINL && p1 > 0.001f) {
                int y = static_cast<int>(p / width);
                int x = static_cast<int>(p % width);

                if (x >= margin && x < (width - margin) && y >= margin && y < (height - margin)) {
                    for (size_t k = 1; k < numPoints; ++k) {
                        float X = static_cast<float>(d_imagePtrs[k][p]) - static_cast<float>(bk);
                        float Y = d_means[k] - bkMean;
                        float X0 = Y / p1;
                        float V = X - X0;
                        int I = static_cast<int>(X0 + static_cast<float>(bk) + 0.5f);

                        if (I >= 0 && I < INL_TABLE_SIZE) {
                            auto atomic_sum = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::generic_space>(d_inlSumm[I]);
                            auto atomic_num = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::generic_space>(d_inlNum[I]);
                            atomic_sum.fetch_add(V);
                            atomic_num.fetch_add(1);
                        }
                    }
                }
            }
        });
    }).wait();

    // Copy results back to host
    q.memcpy(m_fitParams.data(), d_params, totalPixels * sizeof(FitParam)).wait();

    if (computeINL) {
        for (int i = 0; i < INL_TABLE_SIZE; ++i) {
            m_inlSumm[i] = static_cast<double>(d_inlSumm[i]);
            m_inlNum[i] = d_inlNum[i];
        }
        computeINLAveraging();
    }

    // Cleanup device memory
    for (size_t i = 0; i < numPoints; ++i) {
        ctx.free<uint16_t>(d_images[i]);
    }
    ctx.free<uint16_t*>(d_imagePtrs);
    ctx.free<float>(d_means);
    ctx.free<float>(d_rms);
    ctx.free<FitParam>(d_params);
    ctx.free<float>(d_inlSumm);
    ctx.free<int32_t>(d_inlNum);

    std::cout << "[SYCL_Calibrate] Calibration completed successfully.\n";
    return true;
}

void SYCL_Calibrate::computeINLAveraging() {
    std::vector<double> smoothed(INL_TABLE_SIZE, 0.0);
    const int K0 = 5;
    const double size = 2 * K0 + 1;

    for (int i = K0; i < INL_TABLE_SIZE - K0 - 1; ++i) {
        double sum = 0.0;
        for (int k = -K0; k <= K0; ++k) {
            sum += m_inlSumm[i + k];
        }
        smoothed[i] = sum / size;
    }

    for (int i = 0; i < INL_TABLE_SIZE; ++i) {
        if (m_inlNum[i] >= 100) {
            double v = smoothed[i] / static_cast<double>(m_inlNum[i]);
            m_inlAmpl[i] = (v > 0) ? static_cast<int32_t>(v + 0.5) : static_cast<int32_t>(v - 0.5);
        } else {
            m_inlAmpl[i] = 0;
        }
    }
}

bool SYCL_Calibrate::calibrateFromDirectory(SYCLContext& ctx, const fs::path& dirPath, float hv, int binning, int fokus) {
    fs::path armFile;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ARM") {
            armFile = entry.path();
            break;
        }
    }

    if (armFile.empty()) {
        std::cerr << "[SYCL_Calibrate] Error: No .ARM file found in " << dirPath.u8string() << "\n";
        return false;
    }

    ARMHeader armHeader;
    std::map<double, ARMRecord> armRecords;
    if (!SYCL_Average::loadARM(armFile, armHeader, armRecords) || armRecords.empty()) {
        std::cerr << "[SYCL_Calibrate] Error: Failed to load ARM metadata from " << armFile.u8string() << "\n";
        return false;
    }

    std::vector<FLRImage> images;
    std::vector<double> means;
    std::vector<double> rmss;

    for (const auto& [ampl, rec] : armRecords) {
        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".flr") {
                FLRImage testImg;
                if (testImg.load(entry.path())) {
                    double testMean = 0.0, testRms = 0.0;
                    RectROI roi(testImg.getWidth() / 4, testImg.getHeight() / 4, testImg.getWidth() * 3 / 4, testImg.getHeight() * 3 / 4);
                    SYCL_Statistics::hostComputeSlidingMeanRMS(testImg.getHostData(), testImg.getWidth(), testImg.getHeight(), roi, testMean, testRms);
                    if (std::abs(testMean - rec.amplAver) < 2.0) {
                        images.push_back(std::move(testImg));
                        means.push_back(rec.amplAver);
                        rmss.push_back(rec.rmsSingle);
                        break;
                    }
                }
            }
        }
    }

    if (images.size() < 2) {
        std::vector<std::string> subDirs;
        fs::path listFilePath = dirPath / "ListOfDIRs.txt";
        if (fs::exists(listFilePath)) {
            std::ifstream listFile(listFilePath);
            std::string line;
            while (std::getline(listFile, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) line.pop_back();
                if (!line.empty()) subDirs.push_back(line);
            }
        }

        images.clear();
        means.clear();
        rmss.clear();

        for (const auto& dir : subDirs) {
            fs::path flrPath = dirPath / (dir + ".flr");
            if (fs::exists(flrPath)) {
                FLRImage img;
                if (img.load(flrPath)) {
                    double mean = 0.0, rms = 0.0;
                    RectROI roi(img.getWidth() / 4, img.getHeight() / 4, img.getWidth() * 3 / 4, img.getHeight() * 3 / 4);
                    SYCL_Statistics::hostComputeSlidingMeanRMS(img.getHostData(), img.getWidth(), img.getHeight(), roi, mean, rms);
                    images.push_back(std::move(img));
                    means.push_back(mean);
                    rmss.push_back(rms);
                }
            }
        }
    }

    if (images.size() < 2) {
        std::cerr << "[SYCL_Calibrate] Error: Could not find at least 2 calibration FLR images in " << dirPath.u8string() << "\n";
        return false;
    }

    m_width = images[0].getWidth();
    m_height = images[0].getHeight();

    return calibrate(ctx, images, means, rmss, hv, binning, fokus, true);
}

bool SYCL_Calibrate::saveFit(const fs::path& fitFilePath, const fs::path& inlFilePath) const {
    std::ofstream file(fitFilePath, std::ios::binary);
    if (!file.is_open()) return false;

    file.write(reinterpret_cast<const char*>(&m_header), sizeof(FitHeader));
    size_t paramBytes = m_fitParams.size() * sizeof(FitParam);
    file.write(reinterpret_cast<const char*>(m_fitParams.data()), paramBytes);

    fs::path inlPath = inlFilePath;
    if (inlPath.empty()) {
        inlPath = fitFilePath;
        inlPath.replace_extension(".INL");
    }

    std::ofstream inlFile(inlPath);
    if (inlFile.is_open()) {
        for (size_t i = 0; i < m_inlAmpl.size(); ++i) {
            inlFile << i << "\t" << m_inlAmpl[i] << "\t" << m_inlNum[i] << "\n";
        }
    }

    std::cout << "[SYCL_Calibrate] Saved calibration model: " << fitFilePath.u8string() << " and INL: " << inlPath.u8string() << "\n";
    return true;
}

bool SYCL_Calibrate::loadFit(const fs::path& fitFilePath, const fs::path& inlFilePath) {
    std::ifstream file(fitFilePath, std::ios::binary);
    if (!file.is_open()) return false;

    file.read(reinterpret_cast<char*>(&m_header), sizeof(FitHeader));
    m_width = m_header.w;
    m_height = m_header.h;

    size_t totalPixels = static_cast<size_t>(m_width) * m_height;
    m_fitParams.resize(totalPixels);
    file.read(reinterpret_cast<char*>(m_fitParams.data()), totalPixels * sizeof(FitParam));

    m_inlAmpl.assign(INL_TABLE_SIZE, 0);
    fs::path inlPath = inlFilePath;
    if (inlPath.empty()) {
        inlPath = fitFilePath;
        inlPath.replace_extension(".INL");
    }

    std::ifstream inlFile(inlPath);
    if (inlFile.is_open()) {
        int idx, amp, num;
        while (inlFile >> idx >> amp >> num) {
            if (idx >= 0 && idx < INL_TABLE_SIZE) {
                m_inlAmpl[idx] = amp;
            }
        }
    }

    return true;
}

FitParam* SYCL_Calibrate::getDeviceFitParams(SYCLContext& ctx) const {
    size_t count = m_fitParams.size();
    if (count == 0) return nullptr;
    FitParam* d_ptr = ctx.allocateDevice<FitParam>(count);
    ctx.getQueue().memcpy(d_ptr, m_fitParams.data(), count * sizeof(FitParam)).wait();
    return d_ptr;
}

int32_t* SYCL_Calibrate::getDeviceINLTable(SYCLContext& ctx) const {
    int32_t* d_ptr = ctx.allocateDevice<int32_t>(INL_TABLE_SIZE);
    ctx.getQueue().memcpy(d_ptr, m_inlAmpl.data(), INL_TABLE_SIZE * sizeof(int32_t)).wait();
    return d_ptr;
}

} // namespace sycl_ip
