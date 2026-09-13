#include <sycl_ip/SYCL_ImageProcess.hpp>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    std::cout << "=================================================================\n";
    std::cout << "       SYCL_ImageProcess Library - Simple Example Application     \n";
    std::cout << "=================================================================\n\n";

    // 1. Initialize SYCL context on GPU (with fallback to CPU)
    std::cout << "Initializing SYCL context...\n";
    sycl_ip::SYCLContext ctx(sycl_ip::DeviceType::GPU);
    ctx.printDeviceInfo();

    // 2. Create or load a 16-bit FLR image
    const int width = 1024;
    const int height = 1024;
    sycl_ip::FLRImage inputImg;

    if (argc > 1 && inputImg.load(argv[1])) {
        std::cout << "Loaded input image: " << argv[1] << " (" 
                  << inputImg.getWidth() << "x" << inputImg.getHeight() << ")\n";
    } else {
        std::cout << "Generating synthetic 16-bit test pattern (" << width << "x" << height << ")...\n";
        inputImg.resize(width, height);
        uint16_t* hostData = inputImg.getHostData();
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                hostData[y * width + x] = static_cast<uint16_t>(5000 + (x * 10) + (y * 5) + ((x * y) % 100));
            }
        }
    }

    // 3. Allocate device memory and copy input
    size_t pixelCount = inputImg.getPixelCount();
    uint16_t* d_in = inputImg.allocateDeviceBuffer(ctx);
    uint16_t* d_out = ctx.allocateDevice<uint16_t>(pixelCount);
    inputImg.copyToDevice(ctx, d_in);

    // 4. Compute statistics on GPU
    double meanVal = 0.0, rmsVal = 0.0;
    uint16_t minVal = 0, maxVal = 0;
    sycl_ip::SYCL_Statistics::computeMeanRMS(ctx, d_in, inputImg.getWidth(), inputImg.getHeight(), meanVal, rmsVal);
    sycl_ip::SYCL_Statistics::computeMinMax(ctx, d_in, inputImg.getWidth(), inputImg.getHeight(), minVal, maxVal);

    std::cout << "\nInput Image Statistics:\n";
    std::cout << "  Mean:    " << std::fixed << std::setprecision(2) << meanVal << "\n";
    std::cout << "  RMS:     " << rmsVal << "\n";
    std::cout << "  Min/Max: [" << minVal << ", " << maxVal << "]\n\n";

    // 5. Apply 2D Separable Box Blur Filter (Radius = 3 -> 7x7 window) on GPU
    std::cout << "Applying 2D Box Blur filter (Radius=3) on GPU...\n";
    {
        sycl_ip::SYCLScopedTimer timer("2D Smooth Filter", &ctx);
        sycl_ip::SYCL_Filters::smooth(ctx, d_in, d_out, inputImg.getWidth(), inputImg.getHeight(), 3);
    }

    // 6. Copy back and save filtered image
    sycl_ip::FLRImage outputImg(inputImg.getWidth(), inputImg.getHeight());
    outputImg.copyFromDevice(ctx, d_out);
    outputImg.save("SimpleExample_Output.flr");

    std::cout << "Saved processed output to: SimpleExample_Output.flr\n\n";

    // 7. Cleanup
    ctx.free<uint16_t>(d_in);
    ctx.free<uint16_t>(d_out);

    std::cout << "=================================================================\n";
    std::cout << "                Example Completed Successfully                   \n";
    std::cout << "=================================================================\n";
    return 0;
}
