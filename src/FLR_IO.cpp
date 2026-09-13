#include "sycl_ip/FLR_IO.hpp"
#include <cstring>

namespace sycl_ip {

FLRImage::FLRImage() {
    clear();
}

FLRImage::FLRImage(uint16_t width, uint16_t height) {
    resize(width, height);
}

FLRImage::FLRImage(const std::filesystem::path& filePath) {
    load(filePath);
}

void FLRImage::clear() {
    m_header.type = FLR_MARKER;
    m_header.width = 0;
    m_header.height = 0;
    m_header.bpp = 16;
    m_header.bitsOffset = 0;
    m_pixels.clear();
    m_loadedPath.clear();
}

void FLRImage::resize(uint16_t width, uint16_t height) {
    m_header.type = FLR_MARKER;
    m_header.width = width;
    m_header.height = height;
    m_header.bpp = 16;
    m_header.bitsOffset = 0;
    m_pixels.resize(static_cast<size_t>(width) * height, 0);
}

bool FLRImage::load(const std::filesystem::path& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[FLRImage] Error: Cannot open file: " << filePath.u8string() << "\n";
        return false;
    }

    file.read(reinterpret_cast<char*>(&m_header), sizeof(FLRHeader));
    if (file.gcount() != sizeof(FLRHeader)) {
        std::cerr << "[FLRImage] Error: Failed to read FLR header from " << filePath.u8string() << "\n";
        return false;
    }

    if (m_header.type != FLR_MARKER) {
        std::cerr << "[FLRImage] Warning: Invalid FLR marker (0x" << std::hex << m_header.type << std::dec
                  << ") in " << filePath.u8string() << ", expected 0x" << std::hex << FLR_MARKER << std::dec << "\n";
    }

    size_t pixelCount = static_cast<size_t>(m_header.width) * m_header.height;
    if (pixelCount == 0) {
        std::cerr << "[FLRImage] Error: Image dimensions are zero: " << m_header.width << "x" << m_header.height << "\n";
        return false;
    }

    m_pixels.resize(pixelCount);
    file.read(reinterpret_cast<char*>(m_pixels.data()), pixelCount * sizeof(uint16_t));
    if (file.gcount() != static_cast<std::streamsize>(pixelCount * sizeof(uint16_t))) {
        std::cerr << "[FLRImage] Warning: Incomplete read (" << file.gcount() << " of " << (pixelCount * sizeof(uint16_t))
                  << " bytes) from " << filePath.u8string() << "\n";
    }

    m_loadedPath = filePath;
    return true;
}

bool FLRImage::save(const std::filesystem::path& filePath) const {
    return save(filePath, m_pixels.data(), m_header.width, m_header.height);
}

bool FLRImage::save(const std::filesystem::path& filePath, const uint16_t* customData, uint16_t width, uint16_t height) const {
    if (!customData || width == 0 || height == 0) {
        std::cerr << "[FLRImage] Error: Invalid data or dimensions for saving to " << filePath.u8string() << "\n";
        return false;
    }

    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[FLRImage] Error: Cannot open destination file: " << filePath.u8string() << "\n";
        return false;
    }

    FLRHeader head;
    head.type = FLR_MARKER;
    head.width = width;
    head.height = height;
    head.bpp = 16;
    head.bitsOffset = 0;

    file.write(reinterpret_cast<const char*>(&head), sizeof(FLRHeader));
    size_t totalBytes = static_cast<size_t>(width) * height * sizeof(uint16_t);
    file.write(reinterpret_cast<const char*>(customData), totalBytes);

    return file.good();
}

uint16_t* FLRImage::allocateDeviceBuffer(SYCLContext& ctx) const {
    size_t pixelCount = getPixelCount();
    if (pixelCount == 0) return nullptr;
    return ctx.allocateDevice<uint16_t>(pixelCount);
}

bool FLRImage::copyToDevice(SYCLContext& ctx, uint16_t* d_dst) const {
    if (!d_dst || m_pixels.empty()) return false;
    size_t count = getPixelCount();
    ctx.getQueue().memcpy(d_dst, m_pixels.data(), count * sizeof(uint16_t)).wait();
    return true;
}

bool FLRImage::copyFromDevice(SYCLContext& ctx, const uint16_t* d_src) {
    if (!d_src || m_pixels.empty()) return false;
    size_t count = getPixelCount();
    ctx.getQueue().memcpy(m_pixels.data(), d_src, count * sizeof(uint16_t)).wait();
    return true;
}

} // namespace sycl_ip
