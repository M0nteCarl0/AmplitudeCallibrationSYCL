#include <sycl_ip/SYCL_ImageProcess.hpp>
#include <iostream>

int main() {
    std::cout << "External CMake project linked with SYCL_ImageProcess library!\n";
    
    sycl_ip::SYCLContext ctx(sycl_ip::DeviceType::GPU);
    ctx.printDeviceInfo();

    // Verify fast reduction on GPU
    const int W = 512, H = 512;
    sycl_ip::FLRImage img(W, H);
    uint16_t* ptr = img.getHostData();
    for (int i = 0; i < W * H; ++i) ptr[i] = static_cast<uint16_t>(i % 5000);

    uint16_t* d_buf = img.allocateDeviceBuffer(ctx);
    img.copyToDevice(ctx, d_buf);

    double mean = 0.0, rms = 0.0;
    sycl_ip::SYCL_Statistics::computeMeanRMS(ctx, d_buf, W, H, mean, rms);

    std::cout << "Computed Mean: " << mean << ", RMS: " << rms << "\n";
    ctx.free<uint16_t>(d_buf);

    return 0;
}
