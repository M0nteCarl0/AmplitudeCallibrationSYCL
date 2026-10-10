# SYCL Amplitude Calibration & Image Processing

[![Build and Package Artifacts](https://github.com/M0nteCarl0/AmplitudeCallibrationSYCL/actions/workflows/build-artifacts.yml/badge.svg)](https://github.com/M0nteCarl0/AmplitudeCallibrationSYCL/actions/workflows/build-artifacts.yml)
[![GitHub Release](https://img.shields.io/github/v/release/M0nteCarl0/AmplitudeCallibrationSYCL)](https://github.com/M0nteCarl0/AmplitudeCallibrationSYCL/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/M0nteCarl0/AmplitudeCallibrationSYCL/total?color=blue)](https://github.com/M0nteCarl0/AmplitudeCallibrationSYCL/releases)
[![Clones](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FM0nteCarl0%2FAmplitudeCallibrationSYCL%2Ftraffic%2Fclones.json)](docs/clone_statistics.md)
[![License: Custom Attribution](https://img.shields.io/badge/License-Attribution%20Required-blue.svg)](LICENSE)
[![SYCL](https://img.shields.io/badge/SYCL-2020%20%2F%20DPC%2B%2B-orange.svg)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)](docs/environment_setup.md)

High-performance, modular C++ library powered by **SYCL / DPC++ (Intel oneAPI)** for parallel 16-bit X-ray and optical detector image processing (`.flr`, `.dcm` DICOM Part 10) and amplitude calibration across **GPUs** (Intel Iris Xe, Arc, Data Center GPU) and multi-core **CPUs** (Intel Core / Xeon).

---

## ⚡ Highlights

- **Heterogeneous Acceleration:** Unified single-source SYCL kernels executing on Intel GPUs (Level-Zero / OpenCL) and multi-core CPUs.
- **Detector Amplitude Calibration:** Fast parallel linear regression across 11+ million detector pixels with Integral Non-Linearity (INL) table generation.
- **16-bit Image Processing & Formats:** Fused Mean/RMS reductions, separable 2D Box Blur, ring-aperture spike & cosmic ray suppression, geometric transforms, native FLR and DICOM Part 10 IO / batch conversion.
- **High Throughput:** Up to **1,180+ FPS** (24.5 GB/s) for statistical reductions, **800+ FPS** (49.7 GB/s Mem BW) for frame subtraction, and **430+ FPS** (53.8 GB/s Mem BW) for calibration repair on integrated Intel Iris Xe.


---

## 🚀 Quick Start

```cpp
#include <sycl_ip/SYCL_ImageProcess.hpp>
#include <iostream>

int main() {
    // 1. Initialize GPU context (with automatic CPU fallback)
    sycl_ip::SYCLContext ctx(sycl_ip::DeviceType::GPU);

    // 2. Load 16-bit FLR image
    sycl_ip::FLRImage img("RawImage.flr");
    uint16_t* d_raw = img.allocateDeviceBuffer(ctx);
    uint16_t* d_out = ctx.allocateDevice<uint16_t>(img.getPixelCount());
    img.copyToDevice(ctx, d_raw);

    // 3. Parallel Mean & RMS reduction on GPU
    double mean = 0.0, rms = 0.0;
    sycl_ip::SYCL_Statistics::computeMeanRMS(ctx, d_raw, img.getWidth(), img.getHeight(), mean, rms);

    // 4. Suppress defect spikes and apply 2D Box Blur
    sycl_ip::SYCL_Suppress::suppressSpikes(ctx, d_raw, d_out, img.getWidth(), img.getHeight());
    sycl_ip::SYCL_Filters::smooth(ctx, d_out, d_raw, img.getWidth(), img.getHeight(), 2);

    // 5. Retrieve result and clean up
    img.copyFromDevice(ctx, d_raw);
    img.save("Processed.flr");
    ctx.free(d_raw);
    ctx.free(d_out);

    std::cout << "Done. Mean: " << mean << ", RMS: " << rms << "\n";
    return 0;
}
```

---

## 📊 Performance (3328 × 3328, 11M Pixels)

Benchmarked on **Intel Iris Xe Graphics (96 EU)** vs **13th Gen Intel Core i9-13900H (20 threads)**:

| Operation | GPU Time | GPU Throughput | CPU Time | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Mean & RMS Reduction** | **0.84 ms** | **1,187 FPS** (24.50 GB/s Mem BW) | 8.16 ms | **~9.7x** |
| **Image Subtract** | **1.25 ms** | **803 FPS** (49.70 GB/s Mem BW) | 1.63 ms | **~1.3x** |
| **Image Accumulation** | **2.09 ms** | **477 FPS** (49.27 GB/s Mem BW) | 2.60 ms | **~1.2x** |
| **Calibration Model Repair** | **2.30 ms** | **434 FPS** (53.81 GB/s Mem BW) | 3.10 ms | **~1.3x** |
| **2D Box Blur Filter (5x5)** | **2.84 ms** | **352 FPS** (29.11 GB/s Mem BW) | 11.62 ms | **~4.1x** |
| **Spike Noise Suppression** | **3.02 ms** | **331 FPS** | 71.93 ms | **~23.8x** |

---

## 🔨 Building

### Windows (Visual Studio / MSBuild)
```powershell
& "MSBuild.exe" AmplitudeCallibrationSYCL.sln /p:Configuration=Release /p:Platform=x64
```
*Or run [`build_and_run.bat`](build_and_run.bat) to build and launch automated validation suites.*

### CMake (Windows & Linux)
```bash
source /opt/intel/oneapi/setvars.sh   # On Windows: call setvars.bat intel64
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## 📚 Documentation

- 📖 **[Class & API Reference](docs/api_reference.md)** — Detailed descriptions of all classes, methods, and structures.
- ⚙️ **[Environment & Prerequisites Guide](docs/environment_setup.md)** — Hardware drivers, Intel oneAPI, and toolchain installation for Windows and Linux.
- 🧪 **[Validation & Benchmark Suite](run_all.bat)** — Pre-configured test scripts ([`run_gpu_test.bat`](run_gpu_test.bat), [`run_cpu_test.bat`](run_cpu_test.bat), [`run_benchmarks.bat`](run_benchmarks.bat)).
- 📈 **[Clone Statistics Badge](docs/clone_statistics.md)** — How the lifetime clones counter is collected and published to the `traffic` branch.

---

## 📄 License

This library is licensed under a permissive license with a **mandatory product attribution clause**:
You may freely use, modify, and distribute this software, provided that any product, application, or device incorporating it **visibly credits the author ("Alex" / "A.")** in the user interface, CLI `--help` banner, or user manual.

See the full terms in [`LICENSE`](LICENSE).
