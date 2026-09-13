#pragma once

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include "FLR_IO.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

namespace sycl_ip {

#pragma pack(push, 1)
struct ARMHeader {
    int32_t suppressLevel = 0;
    int32_t pad0 = 0;
    int64_t time = 0;
    int32_t rectRmsLeft = 0;
    int32_t rectRmsTop = 0;
    int32_t rectRmsRight = 0;
    int32_t rectRmsBottom = 0;
    int32_t rectAmplLeft = 0;
    int32_t rectAmplTop = 0;
    int32_t rectAmplRight = 0;
    int32_t rectAmplBottom = 0;
    int32_t points = 0;
    int32_t pad1 = 0;
};

struct ARMRecord {
    double rmsSingle = 0.0;
    double amplAver = 0.0;
    double rmsAver = 0.0;
    int32_t statistic = 0;
    int32_t pad = 0;
};
#pragma pack(pop)

struct ExposureAverageResult {
    std::string exposureName;
    std::filesystem::path outputFlrPath;
    double amplAver = 0.0;
    double rmsAver = 0.0;
    double rmsSingle = 0.0;
    int numImages = 0;
    FLRImage averagedImage;
};

class SYCL_IP_API SYCL_Average {
public:
    SYCL_Average(int width = 3328, int height = 3328);
    ~SYCL_Average() = default;

    void setImageDimensions(int width, int height) { m_width = width; m_height = height; }
    void setROIs(const RectROI& roiMean, const RectROI& roiRms) { m_roiMean = roiMean; m_roiRms = roiRms; }

    // Accumulate a single image on device
    static void accumulate(SYCLContext& ctx, uint32_t* d_accum, const uint16_t* d_in, size_t count);

    // Divide accumulator by count and clamp to uint16
    static void normalize(SYCLContext& ctx, const uint32_t* d_accum, uint16_t* d_out, size_t count, int numImages);

    // Average a list of FLR image files on GPU/CPU
    bool averageFileList(SYCLContext& ctx, const std::vector<std::filesystem::path>& fileList, const std::filesystem::path& outFlrPath, ExposureAverageResult& outResult);

    // Process a directory containing FLR subdirectories (e.g. 0.00_Mass, 1.00_Mass, etc.)
    bool processDirectory(SYCLContext& ctx, const std::filesystem::path& baseDir, const std::filesystem::path& outArmPath = "");

    // Save and load .ARM metadata
    static bool saveARM(const std::filesystem::path& filePath, const ARMHeader& header, const std::map<double, ARMRecord>& records);
    static bool loadARM(const std::filesystem::path& filePath, ARMHeader& header, std::map<double, ARMRecord>& records);

    const std::map<double, ExposureAverageResult>& getResults() const { return m_results; }

private:
    int m_width = 3328;
    int m_height = 3328;
    RectROI m_roiMean;
    RectROI m_roiRms;
    std::map<double, ExposureAverageResult> m_results;
};

} // namespace sycl_ip
