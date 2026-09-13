#pragma once

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace sycl_ip {

#define FLR_MARKER ((uint16_t)(('R' << 8) | 'F'))

#pragma pack(push, 1)
struct FLRHeader {
    uint16_t type = FLR_MARKER;
    uint16_t width = 0;
    uint16_t height = 0;
    uint16_t bpp = 16;
    uint16_t bitsOffset = 0;
};
#pragma pack(pop)

struct RectROI {
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;

    RectROI() = default;
    RectROI(int l, int t, int r, int b) : left(l), top(t), right(r), bottom(b) {}

    int width() const { return right - left; }
    int height() const { return bottom - top; }
    size_t area() const { return static_cast<size_t>(width() > 0 && height() > 0 ? width() * height() : 0); }
    bool isValid(int imgW, int imgH) const {
        return left >= 0 && top >= 0 && right <= imgW && bottom <= imgH && left < right && top < bottom;
    }
};

class SYCL_IP_API FLRImage {
public:
    FLRImage();
    FLRImage(uint16_t width, uint16_t height);
    FLRImage(const std::filesystem::path& filePath);
    ~FLRImage() = default;

    bool load(const std::filesystem::path& filePath);
    bool save(const std::filesystem::path& filePath) const;
    bool save(const std::filesystem::path& filePath, const uint16_t* customData, uint16_t width, uint16_t height) const;

    void resize(uint16_t width, uint16_t height);
    void clear();

    uint16_t getWidth() const { return m_header.width; }
    uint16_t getHeight() const { return m_header.height; }
    size_t getPixelCount() const { return static_cast<size_t>(m_header.width) * m_header.height; }
    size_t getByteSize() const { return getPixelCount() * sizeof(uint16_t); }
    const FLRHeader& getHeader() const { return m_header; }

    uint16_t* getHostData() { return m_pixels.data(); }
    const uint16_t* getHostData() const { return m_pixels.data(); }

    uint16_t& at(size_t x, size_t y) { return m_pixels[y * m_header.width + x]; }
    const uint16_t& at(size_t x, size_t y) const { return m_pixels[y * m_header.width + x]; }

    // Device memory operations
    uint16_t* allocateDeviceBuffer(SYCLContext& ctx) const;
    bool copyToDevice(SYCLContext& ctx, uint16_t* d_dst) const;
    bool copyFromDevice(SYCLContext& ctx, const uint16_t* d_src);

private:
    FLRHeader m_header;
    std::vector<uint16_t> m_pixels;
    std::filesystem::path m_loadedPath;
};

} // namespace sycl_ip
