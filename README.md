# SYCL Amplitude Calibration & Image Processing

[![Build and Package Artifacts](https://github.com/M0nteCarl0/AmplitudeCallibrationSYCL/actions/workflows/build-artifacts.yml/badge.svg)](https://github.com/M0nteCarl0/AmplitudeCallibrationSYCL/actions/workflows/build-artifacts.yml)
[![License: Custom Attribution](https://img.shields.io/badge/License-Attribution%20Required-blue.svg)](LICENSE)
[![SYCL](https://img.shields.io/badge/SYCL-2020%20%2F%20DPC%2B%2B-orange.svg)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)](docs/environment_setup.md)

High-performance, modular C++ library powered by **SYCL / DPC++ (Intel oneAPI)** for parallel 16-bit X-ray and optical detector image processing (`.flr`) and amplitude calibration across **GPUs** (Intel Iris Xe, Arc, Data Center GPU) and multi-core **CPUs** (Intel Core / Xeon).

---

## ⚡ Highlights

- **Heterogeneous Acceleration:** Unified single-source SYCL kernels executing on Intel GPUs (Level-Zero / OpenCL) and multi-core CPUs.
- **Detector Amplitude Calibration:** Fast parallel linear regression across 11+ million detector pixels with Integral Non-Linearity (INL) table generation.
- **16-bit Image Processing:** Fused Mean/RMS reductions, separable 2D Box Blur, ring-aperture spike & cosmic ray suppression, and geometric transforms.
- **High Throughput:** Up to **480+ FPS** (9.9 GB/s) for statistical reductions and **330+ FPS** for spike defect suppression on integrated Intel Iris Xe.

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
| **Mean & RMS Reduction** | **2.07 ms** | **482 FPS** (9.96 GB/s) | 14.34 ms | **~6.9x** |
| **Spike Noise Suppression** | **3.02 ms** | **330 FPS** | 70.94 ms | **~23.4x** |
| **Calibration Model Repair** | **2.64 ms** | **378 FPS** (7.80 GB/s) | 3.34 ms | **~1.3x** |
| **2D Box Blur Filter (5x5)** | **4.66 ms** | **214 FPS** (8.85 GB/s) | 12.79 ms | **~2.7x** |
| **Image Accumulation** | **11.22 ms** | **89 FPS** (1.84 GB/s) | 20.80 ms | **~1.9x** |

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

---

## 📄 License

This library is licensed under a permissive license with a **mandatory product attribution clause**:
You may freely use, modify, and distribute this software, provided that any product, application, or device incorporating it **visibly credits the author ("Alex" / "A.")** in the user interface, CLI `--help` banner, or user manual.

See the full terms in [`LICENSE`](LICENSE).
