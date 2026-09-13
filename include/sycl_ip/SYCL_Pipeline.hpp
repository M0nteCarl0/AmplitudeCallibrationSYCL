#pragma once

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include "FLR_IO.hpp"
#include "SYCL_Statistics.hpp"
#include "SYCL_Average.hpp"
#include "SYCL_Calibrate.hpp"
#include "SYCL_Repair.hpp"
#include "SYCL_Filters.hpp"
#include "SYCL_Suppress.hpp"
#include "SYCL_Transforms.hpp"
#include <filesystem>

namespace sycl_ip {

class SYCL_IP_API SYCL_Pipeline {
public:
    SYCL_Pipeline() = default;
    ~SYCL_Pipeline() = default;

    // Run comprehensive test suite validating SYCL GPU/CPU against Host CPU reference
    static bool runValidationSuite(DeviceType deviceType);

    // Run performance benchmarks comparing GPU and CPU throughput
    static void runBenchmarkSuite();

    // End-to-end pipeline: Average directory -> Calibrate model -> Repair Raw image -> Filter
    static bool runFullPipeline(const std::filesystem::path& calibDir,
                               const std::filesystem::path& rawImagePath,
                               const std::filesystem::path& outCalibratedPath,
                               DeviceType deviceType = DeviceType::GPU);
};

} // namespace sycl_ip
