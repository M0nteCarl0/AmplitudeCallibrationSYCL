# SYCL_ImageProcess Class & API Reference

All classes, structs, and functions belong to the `sycl_ip` C++ namespace. To use the library, include the umbrella header:

```cpp
#include <sycl_ip/SYCL_ImageProcess.hpp>
```

---

## 📋 Table of Contents
- [1. SYCLContext](#1-syclcontext)
- [2. FLRImage & FLRHeader](#2-flrimage--flrheader)
- [3. SYCL_Statistics](#3-sycl_statistics)
- [4. SYCL_Average](#4-sycl_average)
- [5. SYCL_Calibrate](#5-sycl_calibrate)
- [6. SYCL_Repair](#6-sycl_repair)
- [7. SYCL_Filters](#7-sycl_filters)
- [8. SYCL_Suppress](#8-sycl_suppress)
- [9. SYCL_Transforms](#9-sycl_transforms)
- [10. SYCL_Pipeline](#10-sycl_pipeline)
- [11. Helper Structures & Utilities](#11-helper-structures--utilities)

---

## 1. `SYCLContext`

**Header:** `sycl_ip/SYCLContext.hpp`

Manages SYCL device selection, command queues, Unified Shared Memory (USM) device allocations, and execution synchronization.

```cpp
namespace sycl_ip {

enum class DeviceType {
    GPU,    // Target GPU (Level-Zero or OpenCL)
    CPU,    // Target Multi-core CPU (OpenCL)
    AUTO    // Prefer GPU; automatically fallback to CPU if unavailable
};

class SYCLContext {
public:
    explicit SYCLContext(DeviceType type = DeviceType::GPU);
    explicit SYCLContext(const sycl::device& dev);
    ~SYCLContext() = default;

    sycl::queue& getQueue();
    const sycl::queue& getQueue() const;
    sycl::device getDevice() const;
    sycl::context getContext() const;

    DeviceInfo getDeviceInfo() const;
    void printDeviceInfo() const;

    // USM Memory Allocation helpers
    template <typename T>
    T* allocateDevice(size_t count);  // Device-only USM allocation

    template <typename T>
    T* allocateShared(size_t count);  // Host-device shared USM allocation

    template <typename T>
    T* allocateHost(size_t count);    // Pinned host USM allocation

    template <typename T>
    void free(T* ptr);                // Free allocated USM memory

    void wait();                      // Wait for all queued commands to finish

    static std::vector<DeviceInfo> enumerateDevices();
};

}
```

### Key Methods:
- `allocateDevice<T>(count)`: Allocates `count * sizeof(T)` bytes on the active device using `sycl::malloc_device`. Fastest access path for device kernels.
- `free<T>(ptr)`: Safely frees memory allocated via any of the USM allocation methods.
- `enumerateDevices()`: Discovers and queries all SYCL-compatible devices in the host system.

---

## 2. `FLRImage` & `FLRHeader`

**Header:** `sycl_ip/FLR_IO.hpp`

Manages 16-bit `.flr` digital detector images, file serialization, host buffers, and device transfers.

```cpp
namespace sycl_ip {

#pragma pack(push, 1)
struct FLRHeader {
    uint16_t type;        // Magic 'RF' (0x5246)
    uint16_t width;       // Width in pixels (e.g., 3328)
    uint16_t height;      // Height in pixels (e.g., 3328)
    uint16_t bpp;         // Bits per pixel (16)
    uint16_t bitsOffset;  // Byte offset to image data (0)
};
#pragma pack(pop)

class FLRImage {
public:
    FLRImage();
    FLRImage(uint16_t width, uint16_t height);
    explicit FLRImage(const std::filesystem::path& filePath);

    bool load(const std::filesystem::path& filePath);
    bool save(const std::filesystem::path& filePath) const;
    bool save(const std::filesystem::path& filePath, const uint16_t* customData, uint16_t width, uint16_t height) const;

    void resize(uint16_t width, uint16_t height);
    void clear();

    uint16_t getWidth() const;
    uint16_t getHeight() const;
    size_t getPixelCount() const;
    size_t getByteSize() const;
    const FLRHeader& getHeader() const;

    uint16_t* getHostData();
    const uint16_t* getHostData() const;
    uint16_t& at(size_t x, size_t y);

    // Device Memory Interop
    uint16_t* allocateDeviceBuffer(SYCLContext& ctx) const;
    bool copyToDevice(SYCLContext& ctx, uint16_t* d_dst) const;
    bool copyFromDevice(SYCLContext& ctx, const uint16_t* d_src);
};

}
```

---

## 3. `SYCL_Statistics`

**Header:** `sycl_ip/SYCL_Statistics.hpp`

High-throughput parallel reductions executing across GPU execution units / work-groups.

```cpp
namespace sycl_ip {

class SYCL_Statistics {
public:
    // Parallel reduction for Mean value over image or ROI
    static double computeMean(SYCLContext& ctx, const uint16_t* d_image, int width, int height, const RectROI* roi = nullptr);

    // Fused parallel reduction for Mean and RMS concurrently
    static void computeMeanRMS(SYCLContext& ctx, const uint16_t* d_image, int width, int height, double& outMean, double& outRMS, const RectROI* roi = nullptr);

    // Sliding RMS window analysis across a grid
    static void computeSlidingMeanRMS(SYCLContext& ctx, const uint16_t* d_image, int width, int height, const RectROI& roi, double& outMean, double& outRMS, int rmsSize = 30, int rmsStep = 10);

    // Parallel Min/Max reduction
    static void computeMinMax(SYCLContext& ctx, const uint16_t* d_image, int width, int height, uint16_t& outMin, uint16_t& outMax);
};

}
```

---

## 4. `SYCL_Average`

**Header:** `sycl_ip/SYCL_Average.hpp`

Multi-frame accumulation, noise reduction through frame averaging, and calibration `.ARM` record generation.

```cpp
namespace sycl_ip {

struct ExposureAverageResult {
    std::string exposureName;
    std::filesystem::path outputFlrPath;
    double amplAver = 0.0;
    double rmsAver = 0.0;
    double rmsSingle = 0.0;
    int numImages = 0;
    FLRImage averagedImage;
};

class SYCL_Average {
public:
    SYCL_Average(int width = 3328, int height = 3328);

    void setImageDimensions(int width, int height);
    void setROIs(const RectROI& roiMean, const RectROI& roiRms);

    // Device kernel: 32-bit integer accumulation without overflow
    static void accumulate(SYCLContext& ctx, uint32_t* d_accum, const uint16_t* d_in, size_t count);

    // Device kernel: Normalizes 32-bit accumulator back to 16-bit clamped values
    static void normalize(SYCLContext& ctx, const uint32_t* d_accum, uint16_t* d_out, size_t count, int numImages);

    // Average a list of FLR image files
    bool averageFileList(SYCLContext& ctx, const std::vector<std::filesystem::path>& fileList, const std::filesystem::path& outFlrPath, ExposureAverageResult& outResult);

    // Process a full dataset folder with exposure series (e.g. 0.00_Mass, 1.00_Mass, ...)
    bool processDirectory(SYCLContext& ctx, const std::filesystem::path& baseDir, const std::filesystem::path& outArmPath = "");

    static bool saveARM(const std::filesystem::path& filePath, const ARMHeader& header, const std::map<double, ARMRecord>& records);
    static bool loadARM(const std::filesystem::path& filePath, ARMHeader& header, std::map<double, ARMRecord>& records);
};

}
```

---

## 5. `SYCL_Calibrate`

**Header:** `sycl_ip/SYCL_Calibrate.hpp`

Executes parallel pixel-by-pixel linear regression across millions of detector elements to model gain and dark current, and computes Integral Non-Linearity (INL) tables.

```cpp
namespace sycl_ip {

struct FitParam {
    float p0 = 0.0f; // Dark current offset
    float p1 = 1.0f; // Gain coefficient slope
};

class SYCL_Calibrate {
public:
    SYCL_Calibrate(int width = 3328, int height = 3328);

    // Executes parallel regression across all pixels concurrently on GPU/CPU
    bool calibrate(SYCLContext& ctx, 
                   const std::vector<FLRImage>& exposureImages, 
                   const std::vector<double>& exposureMeans, 
                   const std::vector<double>& exposureRMSs,
                   float hv = 490.0f, int binning = 1, int fokus = 0, bool computeINL = true);

    // Calibrates directly from a directory of averaged exposure FLRs and .ARM
    bool calibrateFromDirectory(SYCLContext& ctx, const std::filesystem::path& dirPath, float hv = 490.0f, int binning = 1, int fokus = 0);

    bool saveFit(const std::filesystem::path& fitFilePath, const std::filesystem::path& inlFilePath = "") const;
    bool loadFit(const std::filesystem::path& fitFilePath, const std::filesystem::path& inlFilePath = "");

    const FitHeader& getHeader() const;
    const std::vector<FitParam>& getFitParams() const;
    const std::vector<int32_t>& getINLTable() const;
};

}
```

---

## 6. `SYCL_Repair`

**Header:** `sycl_ip/SYCL_Repair.hpp`

Applies the calibration model to raw frames, correcting per-pixel gain and offset, applying INL table corrections, and interpolating multi-voltage calibration models.

```cpp
namespace sycl_ip {

class SYCL_Repair {
public:
    // Core GPU kernel applying calibration model to raw device buffer
    static bool repairImage(SYCLContext& ctx, 
                           const uint16_t* d_in, 
                           uint16_t* d_out, 
                           size_t count, 
                           const FitParam* d_params, 
                           float bkAverage,
                           const int32_t* d_inlTable = nullptr,
                           bool enableINL = true);

    // High-level wrapper operating on FLRImage host objects
    static bool repairImage(SYCLContext& ctx,
                           const FLRImage& rawImage,
                           FLRImage& outCalibratedImage,
                           const SYCL_Calibrate& calib,
                           bool enableINL = true);

    // Multi-voltage interpolation kernel (interpolates between models at hv0 and hv1)
    static bool interpolateVoltages(SYCLContext& ctx,
                                   const uint16_t* d_out0, float hv0,
                                   const uint16_t* d_out1, float hv1,
                                   uint16_t* d_outTarget, float targetHV,
                                   size_t count);
};

}
```

---

## 7. `SYCL_Filters`

**Header:** `sycl_ip/SYCL_Filters.hpp`

High-performance spatial image filters implemented via separable 2D SYCL kernels.

```cpp
namespace sycl_ip {

class SYCL_Filters {
public:
    // Separable 2D Box Blur filter with configurable kernel radius
    static bool smooth(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int width, int height, int radiusK = 2);

    // Adaptive noise averaging filter
    static bool adaptAverage(SYCLContext& ctx, 
                            const uint16_t* d_in, 
                            const uint16_t* d_aver, 
                            const uint16_t* d_rms, 
                            uint16_t* d_out, 
                            int width, 
                            int height,
                            double p0Noise = 10.0, double p1Noise = 0.05,
                            double p0Quench = 0.0, double p1Quench = 1.0,
                            int smoothRadius = 2,
                            int kSize = 1);

    // Subsampling / zoom out kernel
    static bool copyZoom(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int inW, int inH, int factor = 2);
};

}
```

---

## 8. `SYCL_Suppress`

**Header:** `sycl_ip/SYCL_Suppress.hpp`

Ring-aperture concentric filtering to detect and suppress cosmic-ray spikes, sensor defects, and impulse noise.

```cpp
namespace sycl_ip {

class SYCL_Suppress {
public:
    // Annular concentric spike suppression filter (3x3, 5x5, 7x7 windows)
    static bool suppressSpikes(SYCLContext& ctx, 
                               const uint16_t* d_in, 
                               uint16_t* d_out, 
                               int width, 
                               int height,
                               double p0Noise = 10.0, 
                               double p1Noise = 0.05,
                               double coeff1 = 2.0, 
                               double coeff2 = 3.0, 
                               int coeff3 = 2);
};

}
```

---

## 9. `SYCL_Transforms`

**Header:** `sycl_ip/SYCL_Transforms.hpp`

Geometry, cropping, and arithmetic transformations optimized for GPU thread arrays.

```cpp
namespace sycl_ip {

class SYCL_Transforms {
public:
    // Invert: out[i] = 65535 - in[i]
    static bool invert(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, size_t count);

    // Clamped subtraction: out[i] = clamp(inA[i] - inB[i] + offset)
    static bool subtract(SYCLContext& ctx, const uint16_t* d_inA, const uint16_t* d_inB, uint16_t* d_out, size_t count, int offset = 0);

    // Absolute subtraction: out[i] = |inA[i] - inB[i]|
    static bool subtractAbs(SYCLContext& ctx, const uint16_t* d_inA, const uint16_t* d_inB, uint16_t* d_out, size_t count);

    // Flip horizontally or vertically
    static bool flip(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int width, int height, bool horizontal = true);

    // Rotate by 90, 180, or 270 degrees
    static bool rotate(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int inW, int inH, int angleDeg, int& outW, int& outH);

    // Crop rectangular Region of Interest (ROI)
    static bool crop(SYCLContext& ctx, const uint16_t* d_in, uint16_t* d_out, int inW, int inH, const RectROI& roi);
};

}
```

---

## 10. `SYCL_Pipeline`

**Header:** `sycl_ip/SYCL_Pipeline.hpp`

High-level automation pipelines, benchmarks, and validation suites.

```cpp
namespace sycl_ip {

class SYCL_Pipeline {
public:
    // Automated self-test verifying all kernels against CPU reference implementations
    static bool runValidationSuite(DeviceType deviceType);

    // Runs performance benchmarks comparing GPU vs CPU execution times and throughput
    static void runBenchmarkSuite();

    // Complete pipeline: Averaging -> Model Fitting -> Raw Image Repair -> Filtering
    static bool runFullPipeline(const std::filesystem::path& calibDir,
                               const std::filesystem::path& rawImagePath,
                               const std::filesystem::path& outCalibratedPath,
                               DeviceType deviceType = DeviceType::GPU);
};

}
```

---

## 11. Helper Structures & Utilities

### `RectROI`
```cpp
struct RectROI {
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;

    int width() const;
    int height() const;
    size_t area() const;
    bool isValid(int imgW, int imgH) const;
};
```

### `SYCLScopedTimer`
RAII-based GPU/CPU execution timer:
```cpp
{
    sycl_ip::SYCLScopedTimer timer("2D Box Blur", &ctx);
    sycl_ip::SYCL_Filters::smooth(ctx, d_in, d_out, width, height, 2);
} // Automatically synchronizes queue and prints elapsed time in milliseconds
```
