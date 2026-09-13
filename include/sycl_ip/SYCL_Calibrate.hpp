#pragma once

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include "FLR_IO.hpp"
#include "SYCL_Average.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace sycl_ip {

constexpr int CALIBRATION_KEY = 0x5C4E0011;
constexpr int INL_TABLE_SIZE = 65536;

#pragma pack(push, 1)
struct FitHeader {
    int32_t key = CALIBRATION_KEY;
    int32_t x0 = 0;
    int32_t y0 = 0;
    int32_t w = 0;
    int32_t h = 0;
    int32_t fokus = 0;   // 0 - Small, 1 - Big
    int32_t binning = 1;
    float hv = 0.0f;
    float bkAverage = 0.0f;
    int64_t time = 0;
};

struct FitParam {
    float p0 = 0.0f; // Background offset
    float p1 = 1.0f; // Gain slope
};
#pragma pack(pop)

class SYCL_IP_API SYCL_Calibrate {
public:
    SYCL_Calibrate(int width = 3328, int height = 3328);
    ~SYCL_Calibrate() = default;

    void setDimensions(int width, int height) { m_width = width; m_height = height; }

    // Execute parallel amplitude calibration fitting on GPU or CPU
    bool calibrate(SYCLContext& ctx, 
                   const std::vector<FLRImage>& exposureImages, 
                   const std::vector<double>& exposureMeans, 
                   const std::vector<double>& exposureRMSs,
                   float hv = 490.0f, int binning = 1, int fokus = 0, bool computeINL = true);

    // Run calibration from a directory with averaged FLR files and .ARM file
    bool calibrateFromDirectory(SYCLContext& ctx, const std::filesystem::path& dirPath, float hv = 490.0f, int binning = 1, int fokus = 0);

    // Save and load .fit and .INL files
    bool saveFit(const std::filesystem::path& fitFilePath, const std::filesystem::path& inlFilePath = "") const;
    bool loadFit(const std::filesystem::path& fitFilePath, const std::filesystem::path& inlFilePath = "");

    const FitHeader& getHeader() const { return m_header; }
    const std::vector<FitParam>& getFitParams() const { return m_fitParams; }
    const std::vector<int32_t>& getINLTable() const { return m_inlAmpl; }

    FitParam* getDeviceFitParams(SYCLContext& ctx) const;
    int32_t* getDeviceINLTable(SYCLContext& ctx) const;

private:
    void computeINLAveraging();

    int m_width = 3328;
    int m_height = 3328;
    FitHeader m_header;
    std::vector<FitParam> m_fitParams;
    std::vector<int32_t> m_inlAmpl; // [65536]
    std::vector<double> m_inlSumm;  // [65536]
    std::vector<int32_t> m_inlNum;  // [65536]
};

} // namespace sycl_ip
