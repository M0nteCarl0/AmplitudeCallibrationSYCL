#pragma once

#include "SYCL_Export.hpp"
#include <sycl/sycl.hpp>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <chrono>

namespace sycl_ip {

enum class DeviceType {
    GPU,
    CPU,
    AUTO
};

struct DeviceInfo {
    std::string name;
    std::string vendor;
    std::string driverVersion;
    std::string openclVersion;
    bool isGPU = false;
    bool isCPU = false;
    uint32_t maxComputeUnits = 0;
    size_t maxWorkGroupSize = 0;
    uint64_t globalMemSizeBytes = 0;
    uint64_t localMemSizeBytes = 0;
    std::vector<size_t> maxWorkItemSizes;
};

class SYCL_IP_API SYCLContext {
public:
    SYCLContext(DeviceType type = DeviceType::GPU);
    SYCLContext(const sycl::device& dev);
    ~SYCLContext() = default;

    sycl::queue& getQueue() { return *m_queue; }
    const sycl::queue& getQueue() const { return *m_queue; }
    sycl::device getDevice() const { return m_queue->get_device(); }
    sycl::context getContext() const { return m_queue->get_context(); }

    DeviceInfo getDeviceInfo() const;
    void printDeviceInfo() const;

    template <typename T>
    T* allocateDevice(size_t count) {
        return sycl::malloc_device<T>(count, *m_queue);
    }

    template <typename T>
    T* allocateShared(size_t count) {
        return sycl::malloc_shared<T>(count, *m_queue);
    }

    template <typename T>
    T* allocateHost(size_t count) {
        return sycl::malloc_host<T>(count, *m_queue);
    }

    template <typename T>
    void free(T* ptr) {
        if (ptr && m_queue) {
            sycl::free(ptr, *m_queue);
        }
    }

    void wait() {
        if (m_queue) m_queue->wait();
    }

    static std::vector<DeviceInfo> enumerateDevices();

private:
    bool initQueue(DeviceType type);

    std::unique_ptr<sycl::queue> m_queue;
    DeviceType m_requestedType;
};

class SYCL_IP_API SYCLScopedTimer {
public:
    SYCLScopedTimer(const std::string& name, SYCLContext* ctx = nullptr)
        : m_name(name), m_ctx(ctx), m_start(std::chrono::high_resolution_clock::now()) {
        if (m_ctx) m_ctx->wait();
    }

    ~SYCLScopedTimer() {
        if (m_ctx) m_ctx->wait();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - m_start).count();
        std::cout << "[Timer] " << m_name << ": " << ms << " ms\n";
    }

    double elapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - m_start).count();
    }

private:
    std::string m_name;
    SYCLContext* m_ctx;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

} // namespace sycl_ip
