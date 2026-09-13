#include "sycl_ip/SYCL_Pipeline.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace sycl_ip {

static bool compareBuffers(const uint16_t* a, const uint16_t* b, size_t count, int maxAllowedDiff = 1, const std::string& testName = "") {
    size_t diffCount = 0;
    int maxDiffFound = 0;
    for (size_t i = 0; i < count; ++i) {
        int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        if (diff > maxDiffFound) maxDiffFound = diff;
        if (diff > maxAllowedDiff) {
            diffCount++;
            if (diffCount <= 5) {
                std::cerr << "  [" << testName << "] Mismatch at index " << i << ": a=" << a[i] << ", b=" << b[i] << ", diff=" << diff << "\n";
            }
        }
    }
    if (diffCount > 0) {
        std::cerr << "  [" << testName << "] FAILED: " << diffCount << " / " << count << " pixels differed (Max diff: " << maxDiffFound << ")\n";
        return false;
    }
    std::cout << "  [" << testName << "] PASSED (Max diff: " << maxDiffFound << ")\n";
    return true;
}

bool SYCL_Pipeline::runValidationSuite(DeviceType deviceType) {
    std::cout << "\n=================================================================\n";
    std::cout << "          STARTING SYCL VALIDATION & SELF-TEST SUITE             \n";
    std::cout << "=================================================================\n";

    SYCLContext ctx(deviceType);
    ctx.printDeviceInfo();

    bool allPassed = true;
    const int testW = 512;
    const int testH = 512;
    const size_t testCount = testW * testH;

    // Test 1: Synthetic Data Creation and Reductions
    std::cout << "\n[Test 1] Parallel Reduction (Mean, RMS, MinMax)...\n";
    std::vector<uint16_t> hostTestImg(testCount);
    for (size_t y = 0; y < testH; ++y) {
        for (size_t x = 0; x < testW; ++x) {
            hostTestImg[y * testW + x] = static_cast<uint16_t>(1000 + (x * 10) + (y * 5) + ((x * y) % 37));
        }
    }

    uint16_t* d_testImg = ctx.allocateDevice<uint16_t>(testCount);
    ctx.getQueue().memcpy(d_testImg, hostTestImg.data(), testCount * sizeof(uint16_t)).wait();

    double hostMean = 0.0, hostRMS = 0.0;
    SYCL_Statistics::hostComputeMeanRMS(hostTestImg.data(), testW, testH, hostMean, hostRMS);

    double syclMean = 0.0, syclRMS = 0.0;
    SYCL_Statistics::computeMeanRMS(ctx, d_testImg, testW, testH, syclMean, syclRMS);

    uint16_t hostMin = *std::min_element(hostTestImg.begin(), hostTestImg.end());
    uint16_t hostMax = *std::max_element(hostTestImg.begin(), hostTestImg.end());
    uint16_t syclMin = 0, syclMax = 0;
    SYCL_Statistics::computeMinMax(ctx, d_testImg, testW, testH, syclMin, syclMax);

    std::cout << "  Host Mean: " << hostMean << ", SYCL Mean: " << syclMean << " (diff: " << std::abs(hostMean - syclMean) << ")\n";
    std::cout << "  Host RMS:  " << hostRMS  << ", SYCL RMS:  " << syclRMS  << " (diff: " << std::abs(hostRMS - syclRMS)   << ")\n";
    std::cout << "  Host Min/Max: [" << hostMin << ", " << hostMax << "], SYCL Min/Max: [" << syclMin << ", " << syclMax << "]\n";

    if (std::abs(hostMean - syclMean) < 1e-3 && std::abs(hostRMS - syclRMS) < 1e-3 && hostMin == syclMin && hostMax == syclMax) {
        std::cout << "  [Reductions] PASSED\n";
    } else {
        std::cerr << "  [Reductions] FAILED\n";
        allPassed = false;
    }

    // Test 2: Sliding Grid Mean and RMS (FulfilMeanRms)
    std::cout << "\n[Test 2] Sliding Grid RMS (FulfilMeanRms)...\n";
    RectROI testRoi(50, 50, 450, 450);
    double hostGridMean = 0.0, hostGridRms = 0.0;
    SYCL_Statistics::hostComputeSlidingMeanRMS(hostTestImg.data(), testW, testH, testRoi, hostGridMean, hostGridRms, 30, 10);

    double syclGridMean = 0.0, syclGridRms = 0.0;
    SYCL_Statistics::computeSlidingMeanRMS(ctx, d_testImg, testW, testH, testRoi, syclGridMean, syclGridRms, 30, 10);

    std::cout << "  Host Grid Mean: " << hostGridMean << ", SYCL Grid Mean: " << syclGridMean
              << " (diff: " << std::abs(hostGridMean - syclGridMean) << ")\n";
    std::cout << "  Host Grid RMS:  " << hostGridRms  << ", SYCL Grid RMS:  " << syclGridRms
              << " (diff: " << std::abs(hostGridRms - syclGridRms) << ")\n";

    if (std::abs(hostGridMean - syclGridMean) < 1e-4 && std::abs(hostGridRms - syclGridRms) < 1e-4) {
        std::cout << "  [Sliding Grid] PASSED\n";
    } else {
        std::cerr << "  [Sliding Grid] FAILED\n";
        allPassed = false;
    }

    // Test 3: 2D Box Blur Smoothing
    std::cout << "\n[Test 3] 2D Box Blur (Smooth)...\n";
    std::vector<uint16_t> hostSmoothOut(testCount);
    SYCL_Filters::hostSmooth(hostTestImg.data(), hostSmoothOut.data(), testW, testH, 2);

    uint16_t* d_smoothOut = ctx.allocateDevice<uint16_t>(testCount);
    SYCL_Filters::smooth(ctx, d_testImg, d_smoothOut, testW, testH, 2);

    std::vector<uint16_t> syclSmoothOut(testCount);
    ctx.getQueue().memcpy(syclSmoothOut.data(), d_smoothOut, testCount * sizeof(uint16_t)).wait();

    if (!compareBuffers(hostSmoothOut.data(), syclSmoothOut.data(), testCount, 1, "2D Smooth")) {
        allPassed = false;
    }

    // Test 4: Defect / Spike Suppression
    std::cout << "\n[Test 4] Spike / Defect Noise Suppression...\n";
    std::vector<uint16_t> noisyImg = hostTestImg;
    // Inject isolated spikes
    noisyImg[100 * testW + 100] = 50000;
    noisyImg[200 * testW + 200] = 60000;
    noisyImg[300 * testW + 300] = 45000;

    std::vector<uint16_t> hostSuppressOut(testCount);
    SYCL_Suppress::hostSuppressSpikes(noisyImg.data(), hostSuppressOut.data(), testW, testH, 10.0, 0.05, 2.0, 3.0, 2);

    ctx.getQueue().memcpy(d_testImg, noisyImg.data(), testCount * sizeof(uint16_t)).wait();
    uint16_t* d_suppressOut = ctx.allocateDevice<uint16_t>(testCount);
    SYCL_Suppress::suppressSpikes(ctx, d_testImg, d_suppressOut, testW, testH, 10.0, 0.05, 2.0, 3.0, 2);

    std::vector<uint16_t> syclSuppressOut(testCount);
    ctx.getQueue().memcpy(syclSuppressOut.data(), d_suppressOut, testCount * sizeof(uint16_t)).wait();

    if (!compareBuffers(hostSuppressOut.data(), syclSuppressOut.data(), testCount, 0, "Spike Suppression")) {
        allPassed = false;
    }

    // Test 5: Image Transforms (Invert, Subtract, Flip, Rotate, Crop)
    std::cout << "\n[Test 5] Image Transforms (Invert, Subtract, Flip, Rotate, Crop)...\n";
    uint16_t* d_transOut = ctx.allocateDevice<uint16_t>(testCount);
    SYCL_Transforms::invert(ctx, d_testImg, d_transOut, testCount);
    std::vector<uint16_t> syclInvertOut(testCount);
    ctx.getQueue().memcpy(syclInvertOut.data(), d_transOut, testCount * sizeof(uint16_t)).wait();

    bool invOk = true;
    for (size_t i = 0; i < testCount; ++i) {
        if (syclInvertOut[i] != (65535 - noisyImg[i])) { invOk = false; break; }
    }
    if (invOk) std::cout << "  [Invert] PASSED\n";
    else { std::cerr << "  [Invert] FAILED\n"; allPassed = false; }

    int rotW = 0, rotH = 0;
    SYCL_Transforms::rotate(ctx, d_testImg, d_transOut, testW, testH, 90, rotW, rotH);
    std::vector<uint16_t> syclRotOut(testCount);
    ctx.getQueue().memcpy(syclRotOut.data(), d_transOut, testCount * sizeof(uint16_t)).wait();
    if (rotW == testH && rotH == testW && syclRotOut[0] == noisyImg[(testH - 1) * testW]) {
        std::cout << "  [Rotate 90] PASSED\n";
    } else {
        std::cerr << "  [Rotate 90] FAILED\n";
        allPassed = false;
    }

    // Test 6: Real .FLR File Loading & Amplitude Calibration on Dataset
    std::cout << "\n[Test 6] Real .FLR Dataset Processing (490_kV)...\n";
    fs::path testDatasetDir = "490_kV";
    if (fs::exists(testDatasetDir)) {
        SYCL_Average averager(3328, 3328);
        bool avgSuccess = averager.processDirectory(ctx, testDatasetDir);
        if (avgSuccess) {
            std::cout << "  [Dataset Multi-Image Averaging] PASSED\n";

            SYCL_Calibrate calibrator(3328, 3328);
            bool calibSuccess = calibrator.calibrateFromDirectory(ctx, testDatasetDir);
            if (calibSuccess) {
                std::cout << "  [Dataset Calibration Regression] PASSED\n";
                calibrator.saveFit("Calibration_Test.fit", "Calibration_Test.INL");

                // Test 7: Calibrating / Repairing RawImage.flr
                if (fs::exists("RawImage.flr")) {
                    std::cout << "\n[Test 7] Calibrating RawImage.flr -> Output_Calibrated.flr...\n";
                    FLRImage rawImg("RawImage.flr");
                    FLRImage calibImg;
                    bool repSuccess = SYCL_Repair::repairImage(ctx, rawImg, calibImg, calibrator, true);
                    if (repSuccess) {
                        calibImg.save("Output_Calibrated.flr");
                        std::cout << "  [Raw Image Calibration] PASSED -> Saved Output_Calibrated.flr\n";

                        // Verify calibrated image statistics
                        double rawMean = 0.0, rawRms = 0.0;
                        double calMean = 0.0, calRms = 0.0;
                        uint16_t* d_rawBuf = rawImg.allocateDeviceBuffer(ctx);
                        rawImg.copyToDevice(ctx, d_rawBuf);
                        uint16_t* d_calBuf = calibImg.allocateDeviceBuffer(ctx);
                        calibImg.copyToDevice(ctx, d_calBuf);

                        SYCL_Statistics::computeMeanRMS(ctx, d_rawBuf, rawImg.getWidth(), rawImg.getHeight(), rawMean, rawRms);
                        SYCL_Statistics::computeMeanRMS(ctx, d_calBuf, calibImg.getWidth(), calibImg.getHeight(), calMean, calRms);

                        std::cout << "  Raw Image:        Mean = " << rawMean << ", RMS = " << rawRms << "\n";
                        std::cout << "  Calibrated Image: Mean = " << calMean << ", RMS = " << calRms << "\n";

                        ctx.free<uint16_t>(d_rawBuf);
                        ctx.free<uint16_t>(d_calBuf);
                    } else {
                        std::cerr << "  [Raw Image Calibration] FAILED\n";
                        allPassed = false;
                    }
                }
            } else {
                std::cerr << "  [Dataset Calibration Regression] FAILED\n";
                allPassed = false;
            }
        } else {
            std::cerr << "  [Dataset Multi-Image Averaging] FAILED\n";
            allPassed = false;
        }
    } else {
        std::cout << "  [Dataset] 490_kV folder not found in working directory, skipping dataset test.\n";
    }

    // Cleanup
    ctx.free<uint16_t>(d_testImg);
    ctx.free<uint16_t>(d_smoothOut);
    ctx.free<uint16_t>(d_suppressOut);
    ctx.free<uint16_t>(d_transOut);

    std::cout << "\n=================================================================\n";
    std::cout << "  VALIDATION SUITE RESULT: " << (allPassed ? "ALL TESTS PASSED [OK]" : "SOME TESTS FAILED [FAIL]") << "\n";
    std::cout << "=================================================================\n\n";

    return allPassed;
}

void SYCL_Pipeline::runBenchmarkSuite() {
    std::cout << "\n=================================================================\n";
    std::cout << "           RUNNING SYCL GPU vs CPU BENCHMARK SUITE               \n";
    std::cout << "=================================================================\n";

    const int W = 3328;
    const int H = 3328;
    const size_t count = static_cast<size_t>(W) * H;
    const double imgSizeMB = (count * sizeof(uint16_t)) / (1024.0 * 1024.0);
    const int iterations = 10;

    std::cout << "Image Dimensions: " << W << " x " << H << " (" << count << " pixels, " << std::fixed << std::setprecision(2) << imgSizeMB << " MB per frame)\n";
    std::cout << "Iterations per kernel: " << iterations << "\n\n";

    auto benchmarkDevice = [&](DeviceType devType, const std::string& devLabel) {
        SYCLContext ctx(devType);
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "Testing on " << devLabel << ": " << ctx.getDevice().get_info<sycl::info::device::name>() << "\n";
        std::cout << "-----------------------------------------------------------------\n";

        uint16_t* d_inA = ctx.allocateDevice<uint16_t>(count);
        uint16_t* d_inB = ctx.allocateDevice<uint16_t>(count);
        uint16_t* d_out = ctx.allocateDevice<uint16_t>(count);
        uint32_t* d_accum = ctx.allocateDevice<uint32_t>(count);

        ctx.getQueue().memset(d_inA, 0x12, count * sizeof(uint16_t)).wait();
        ctx.getQueue().memset(d_inB, 0x34, count * sizeof(uint16_t)).wait();

        // 1. Mean & RMS Reduction Benchmark
        {
            double mean = 0, rms = 0;
            // Warmup
            SYCL_Statistics::computeMeanRMS(ctx, d_inA, W, H, mean, rms);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                SYCL_Statistics::computeMeanRMS(ctx, d_inA, W, H, mean, rms);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
            double gbps = (imgSizeMB / 1024.0) / (ms / 1000.0);
            std::cout << "  [Mean & RMS Reduction]: " << std::setw(8) << ms << " ms (" << std::setw(6) << (1000.0 / ms) << " FPS, " << std::setw(6) << gbps << " GB/s)\n";
        }

        // 2. Multi-Image Accumulation Benchmark
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                SYCL_Average::accumulate(ctx, d_accum, d_inA, count);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
            double gbps = (imgSizeMB / 1024.0) / (ms / 1000.0);
            std::cout << "  [Image Accumulate]:     " << std::setw(8) << ms << " ms (" << std::setw(6) << (1000.0 / ms) << " FPS, " << std::setw(6) << gbps << " GB/s)\n";
        }

        // 3. 2D Box Blur (Smooth) Benchmark
        {
            // Warmup
            SYCL_Filters::smooth(ctx, d_inA, d_out, W, H, 2);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                SYCL_Filters::smooth(ctx, d_inA, d_out, W, H, 2);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
            double gbps = (2 * imgSizeMB / 1024.0) / (ms / 1000.0);
            std::cout << "  [2D Box Blur (Smooth)]: " << std::setw(8) << ms << " ms (" << std::setw(6) << (1000.0 / ms) << " FPS, " << std::setw(6) << gbps << " GB/s)\n";
        }

        // 4. Defect / Spike Suppression Benchmark
        {
            // Warmup
            SYCL_Suppress::suppressSpikes(ctx, d_inA, d_out, W, H, 10.0, 0.05);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                SYCL_Suppress::suppressSpikes(ctx, d_inA, d_out, W, H, 10.0, 0.05);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
            std::cout << "  [Spike Suppression]:    " << std::setw(8) << ms << " ms (" << std::setw(6) << (1000.0 / ms) << " FPS)\n";
        }

        // 5. Image Calibration Repair Benchmark
        {
            FitParam* d_params = ctx.allocateDevice<FitParam>(count);
            int32_t* d_inl = ctx.allocateDevice<int32_t>(INL_TABLE_SIZE);

            // Warmup
            SYCL_Repair::repairImage(ctx, d_inA, d_out, count, d_params, 500.0f, d_inl, true);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                SYCL_Repair::repairImage(ctx, d_inA, d_out, count, d_params, 500.0f, d_inl, true);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
            double gbps = (imgSizeMB / 1024.0) / (ms / 1000.0);
            std::cout << "  [Calibration Repair]:   " << std::setw(8) << ms << " ms (" << std::setw(6) << (1000.0 / ms) << " FPS, " << std::setw(6) << gbps << " GB/s)\n";

            ctx.free<FitParam>(d_params);
            ctx.free<int32_t>(d_inl);
        }

        // 6. Image Subtract Benchmark
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                SYCL_Transforms::subtract(ctx, d_inA, d_inB, d_out, count);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
            double gbps = (2 * imgSizeMB / 1024.0) / (ms / 1000.0);
            std::cout << "  [Image Subtract]:       " << std::setw(8) << ms << " ms (" << std::setw(6) << (1000.0 / ms) << " FPS, " << std::setw(6) << gbps << " GB/s)\n";
        }

        ctx.free<uint16_t>(d_inA);
        ctx.free<uint16_t>(d_inB);
        ctx.free<uint16_t>(d_out);
        ctx.free<uint32_t>(d_accum);
        std::cout << "\n";
    };

    benchmarkDevice(DeviceType::GPU, "GPU");
    benchmarkDevice(DeviceType::CPU, "CPU");

    std::cout << "=================================================================\n\n";
}

bool SYCL_Pipeline::runFullPipeline(const fs::path& calibDir,
                                   const fs::path& rawImagePath,
                                   const fs::path& outCalibratedPath,
                                   DeviceType deviceType) {
    std::cout << "\n=================================================================\n";
    std::cout << "           RUNNING END-TO-END CALIBRATION PIPELINE               \n";
    std::cout << "=================================================================\n";

    SYCLContext ctx(deviceType);
    ctx.printDeviceInfo();

    // Step 1: Multi-image averaging
    std::cout << "[Step 1] Processing exposure calibration dataset from: " << calibDir.u8string() << "\n";
    SYCL_Average averager(3328, 3328);
    if (!averager.processDirectory(ctx, calibDir)) {
        std::cerr << "[Pipeline] Error in Step 1: Multi-image averaging failed.\n";
        return false;
    }

    // Step 2: Fit model
    std::cout << "\n[Step 2] Performing parallel amplitude calibration fitting on " << ctx.getDevice().get_info<sycl::info::device::name>() << "...\n";
    SYCL_Calibrate calibrator(3328, 3328);
    if (!calibrator.calibrateFromDirectory(ctx, calibDir)) {
        std::cerr << "[Pipeline] Error in Step 2: Calibration fitting failed.\n";
        return false;
    }
    calibrator.saveFit("Output_CalibModel.fit", "Output_CalibModel.INL");

    // Step 3: Repair raw image
    std::cout << "\n[Step 3] Applying calibration to raw image: " << rawImagePath.u8string() << "...\n";
    FLRImage rawImg;
    if (!rawImg.load(rawImagePath)) {
        std::cerr << "[Pipeline] Error in Step 3: Cannot load raw image " << rawImagePath.u8string() << "\n";
        return false;
    }

    FLRImage calibImg;
    if (!SYCL_Repair::repairImage(ctx, rawImg, calibImg, calibrator, true)) {
        std::cerr << "[Pipeline] Error in Step 3: Image repair failed.\n";
        return false;
    }

    // Step 4: Optional defect suppression & filtering
    std::cout << "\n[Step 4] Applying spike defect noise suppression...\n";
    uint16_t* d_calib = calibImg.allocateDeviceBuffer(ctx);
    calibImg.copyToDevice(ctx, d_calib);
    uint16_t* d_filtered = ctx.allocateDevice<uint16_t>(calibImg.getPixelCount());

    SYCL_Suppress::suppressSpikes(ctx, d_calib, d_filtered, calibImg.getWidth(), calibImg.getHeight(), 10.0, 0.05);

    FLRImage finalImg(calibImg.getWidth(), calibImg.getHeight());
    finalImg.copyFromDevice(ctx, d_filtered);
    finalImg.save(outCalibratedPath);

    ctx.free<uint16_t>(d_calib);
    ctx.free<uint16_t>(d_filtered);

    std::cout << "\n[Pipeline] Successfully completed end-to-end processing!\n";
    std::cout << "[Pipeline] Final calibrated & filtered image saved to: " << outCalibratedPath.u8string() << "\n";
    std::cout << "=================================================================\n\n";

    return true;
}

} // namespace sycl_ip
