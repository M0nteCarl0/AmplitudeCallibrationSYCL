#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include "sycl_ip/SYCL_ImageProcess.hpp"

using namespace sycl_ip;

void printUsage(const char* progName) {
    std::cout << "=================================================================\n";
    std::cout << "      SYCL Amplitude Calibration & Image Processing Engine       \n";
    std::cout << "=================================================================\n";
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --device <gpu|cpu|auto>    Select SYCL compute target (default: gpu)\n";
    std::cout << "  --test                     Run full automated validation test suite\n";
    std::cout << "  --bench                    Run GPU vs CPU performance benchmarks\n";
    std::cout << "  --average <dir>            Perform multi-image averaging on exposure directory\n";
    std::cout << "  --calibrate <dir>          Fit amplitude calibration model from averaged dataset\n";
    std::cout << "  --repair <in.flr> <out.flr> [calib.fit]\n";
    std::cout << "                             Calibrate / repair raw FLR image using calibration model\n";
    std::cout << "  --all                      Run tests on GPU & CPU, benchmarks, and dataset pipeline\n";
    std::cout << "  --help                     Show this help message\n\n";
}

int main(int argc, char** argv) {
    try {
        std::cout << "Initializing SYCL Image Processing & Amplitude Calibration System...\n" << std::flush;

    DeviceType selectedDevice = DeviceType::GPU;
    bool runTest = false;
    bool runBench = false;
    bool runAll = false;
    std::string averageDir = "";
    std::string calibrateDir = "";
    std::string repairIn = "", repairOut = "", repairModel = "";

    if (argc <= 1) {
        // Default mode: run complete self-test, benchmark, and dataset processing
        runAll = true;
    } else {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--device" && i + 1 < argc) {
                std::string dev = argv[++i];
                if (dev == "gpu" || dev == "GPU") selectedDevice = DeviceType::GPU;
                else if (dev == "cpu" || dev == "CPU") selectedDevice = DeviceType::CPU;
                else selectedDevice = DeviceType::AUTO;
            } else if (arg == "--test") {
                runTest = true;
            } else if (arg == "--bench") {
                runBench = true;
            } else if (arg == "--all") {
                runAll = true;
            } else if (arg == "--average" && i + 1 < argc) {
                averageDir = argv[++i];
            } else if (arg == "--calibrate" && i + 1 < argc) {
                calibrateDir = argv[++i];
            } else if (arg == "--repair" && i + 2 < argc) {
                repairIn = argv[++i];
                repairOut = argv[++i];
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    repairModel = argv[++i];
                }
            } else if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            }
        }
    }

    // List available platforms and devices
    std::cout << "\nEnumerating available SYCL devices in system:\n";
    auto devices = SYCLContext::enumerateDevices();
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "  [" << i << "] " << (devices[i].isGPU ? "[GPU]" : (devices[i].isCPU ? "[CPU]" : "[ACC]"))
                  << " " << devices[i].name << " (" << devices[i].vendor << ", " << devices[i].driverVersion << ")\n";
    }

    if (runAll) {
        std::cout << "\n>>> MODE: FULL VALIDATION, BENCHMARKS, AND PIPELINE EXECUTION <<<\n";

        // 1. Validation on GPU
        std::cout << "\n--- STEP 1: VALIDATING ON GPU ---\n";
        bool gpuOk = SYCL_Pipeline::runValidationSuite(DeviceType::GPU);

        // 2. Validation on CPU
        std::cout << "\n--- STEP 2: VALIDATING ON CPU ---\n";
        bool cpuOk = SYCL_Pipeline::runValidationSuite(DeviceType::CPU);

        // 3. Benchmarks
        std::cout << "\n--- STEP 3: GPU vs CPU PERFORMANCE BENCHMARK ---\n";
        SYCL_Pipeline::runBenchmarkSuite();

        // 4. End-to-end dataset pipeline
        std::cout << "\n--- STEP 4: END-TO-END DATASET PIPELINE ---\n";
        if (std::filesystem::exists("490_kV") && std::filesystem::exists("RawImage.flr")) {
            SYCL_Pipeline::runFullPipeline("490_kV", "RawImage.flr", "ProcessedImage_SYCL.flr", DeviceType::GPU);
        }

        std::cout << "\n=================================================================\n";
        std::cout << "                 ALL TASKS COMPLETED SUCCESSFULLY                \n";
        std::cout << "  GPU Validation: " << (gpuOk ? "PASSED" : "FAILED") << "\n";
        std::cout << "  CPU Validation: " << (cpuOk ? "PASSED" : "FAILED") << "\n";
        std::cout << "=================================================================\n";
        return (gpuOk && cpuOk) ? 0 : 1;
    }

    if (runTest) {
        bool ok = SYCL_Pipeline::runValidationSuite(selectedDevice);
        return ok ? 0 : 1;
    }

    if (runBench) {
        SYCL_Pipeline::runBenchmarkSuite();
        return 0;
    }

    if (!averageDir.empty()) {
        SYCLContext ctx(selectedDevice);
        ctx.printDeviceInfo();
        SYCL_Average averager;
        bool ok = averager.processDirectory(ctx, averageDir);
        return ok ? 0 : 1;
    }

    if (!calibrateDir.empty()) {
        SYCLContext ctx(selectedDevice);
        ctx.printDeviceInfo();
        SYCL_Calibrate calibrator;
        bool ok = calibrator.calibrateFromDirectory(ctx, calibrateDir);
        if (ok) calibrator.saveFit("Calibration.fit", "Calibration.INL");
        return ok ? 0 : 1;
    }

    if (!repairIn.empty()) {
        SYCLContext ctx(selectedDevice);
        ctx.printDeviceInfo();
        FLRImage rawImg;
        if (!rawImg.load(repairIn)) return 1;

        SYCL_Calibrate calib;
        if (!repairModel.empty()) {
            if (!calib.loadFit(repairModel)) return 1;
        } else if (std::filesystem::exists("Calibration_Test.fit")) {
            if (!calib.loadFit("Calibration_Test.fit")) return 1;
        } else if (std::filesystem::exists("490_kV")) {
            calib.calibrateFromDirectory(ctx, "490_kV");
        }

        FLRImage outImg;
        bool ok = SYCL_Repair::repairImage(ctx, rawImg, outImg, calib, true);
        if (ok) outImg.save(repairOut);
        return ok ? 0 : 1;
    }

    return 0;
} catch (const sycl::exception& e) {
    std::cerr << "\n[CRITICAL SYCL EXCEPTION] " << e.what() << " (code: " << e.code() << ")\n" << std::flush;
    return 1;
} catch (const std::exception& e) {
    std::cerr << "\n[CRITICAL STD EXCEPTION] " << e.what() << "\n" << std::flush;
    return 1;
} catch (...) {
    std::cerr << "\n[UNKNOWN CRITICAL EXCEPTION CAUGHT]\n" << std::flush;
    return 1;
}
}