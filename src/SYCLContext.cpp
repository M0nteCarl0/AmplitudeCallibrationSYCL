#include "sycl_ip/SYCLContext.hpp"
#include <iostream>

namespace sycl_ip {

SYCLContext::SYCLContext(DeviceType type)
    : m_requestedType(type) {
    if (!initQueue(type)) {
        std::cerr << "[SYCLContext] Warning: Failed to initialize requested device, falling back to default device.\n";
        initQueue(DeviceType::AUTO);
    }
}

SYCLContext::SYCLContext(const sycl::device& dev)
    : m_requestedType(DeviceType::AUTO) {
    try {
        m_queue = std::make_unique<sycl::queue>(dev, sycl::property_list{sycl::property::queue::in_order{}});
    } catch (const std::exception& ex) {
        std::cerr << "[SYCLContext] Error creating queue from device: " << ex.what() << "\n";
    }
}

bool SYCLContext::initQueue(DeviceType type) {
    try {
        sycl::property_list props{sycl::property::queue::in_order{}};
        switch (type) {
            case DeviceType::GPU: {
                m_queue = std::make_unique<sycl::queue>(sycl::gpu_selector_v, props);
                break;
            }
            case DeviceType::CPU: {
                m_queue = std::make_unique<sycl::queue>(sycl::cpu_selector_v, props);
                break;
            }
            case DeviceType::AUTO:
            default: {
                m_queue = std::make_unique<sycl::queue>(sycl::default_selector_v, props);
                break;
            }
        }
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[SYCLContext] Error creating SYCL queue: " << ex.what() << "\n";
        return false;
    }
}

DeviceInfo SYCLContext::getDeviceInfo() const {
    DeviceInfo info;
    if (!m_queue) return info;

    sycl::device d = m_queue->get_device();
    try { info.name = d.get_info<sycl::info::device::name>(); } catch (...) { info.name = "Unknown Device"; }
    try { info.vendor = d.get_info<sycl::info::device::vendor>(); } catch (...) { info.vendor = "Unknown Vendor"; }
    try { info.driverVersion = d.get_info<sycl::info::device::driver_version>(); } catch (...) { info.driverVersion = "N/A"; }
    try { info.openclVersion = d.get_info<sycl::info::device::opencl_c_version>(); } catch (...) { info.openclVersion = "N/A (Level Zero / Native)"; }
    try { info.isGPU = d.is_gpu(); } catch (...) { info.isGPU = false; }
    try { info.isCPU = d.is_cpu(); } catch (...) { info.isCPU = false; }
    try { info.maxComputeUnits = d.get_info<sycl::info::device::max_compute_units>(); } catch (...) { info.maxComputeUnits = 0; }
    try { info.maxWorkGroupSize = d.get_info<sycl::info::device::max_work_group_size>(); } catch (...) { info.maxWorkGroupSize = 0; }
    try { info.globalMemSizeBytes = d.get_info<sycl::info::device::global_mem_size>(); } catch (...) { info.globalMemSizeBytes = 0; }
    try { info.localMemSizeBytes = d.get_info<sycl::info::device::local_mem_size>(); } catch (...) { info.localMemSizeBytes = 0; }
    try {
        auto itemSizes = d.get_info<sycl::info::device::max_work_item_sizes<3>>();
        info.maxWorkItemSizes = {itemSizes[0], itemSizes[1], itemSizes[2]};
    } catch (...) {
        info.maxWorkItemSizes = {0, 0, 0};
    }

    return info;
}

void SYCLContext::printDeviceInfo() const {
    DeviceInfo info = getDeviceInfo();
    std::cout << "\n================ SYCL Device Information ================\n";
    std::cout << "  Device Name:        " << info.name << "\n";
    std::cout << "  Vendor:             " << info.vendor << "\n";
    std::cout << "  Type:               " << (info.isGPU ? "GPU" : (info.isCPU ? "CPU" : "Other / Accelerator")) << "\n";
    std::cout << "  Driver Version:     " << info.driverVersion << "\n";
    std::cout << "  OpenCL/Backend:     " << info.openclVersion << "\n";
    std::cout << "  Max Compute Units:  " << info.maxComputeUnits << "\n";
    std::cout << "  Max Work Group Size:" << info.maxWorkGroupSize << "\n";
    std::cout << "  Global Memory:      " << (info.globalMemSizeBytes / (1024 * 1024)) << " MB\n";
    std::cout << "  Local Memory:       " << (info.localMemSizeBytes / 1024) << " KB\n";
    std::cout << "=========================================================\n\n";
}

std::vector<DeviceInfo> SYCLContext::enumerateDevices() {
    std::vector<DeviceInfo> result;
    auto platforms = sycl::platform::get_platforms();
    for (const auto& p : platforms) {
        auto devices = p.get_devices();
        for (const auto& d : devices) {
            DeviceInfo info;
            try { info.name = d.get_info<sycl::info::device::name>(); } catch (...) { info.name = "Unknown Device"; }
            try { info.vendor = d.get_info<sycl::info::device::vendor>(); } catch (...) { info.vendor = "Unknown Vendor"; }
            try { info.driverVersion = d.get_info<sycl::info::device::driver_version>(); } catch (...) { info.driverVersion = "N/A"; }
            try {
                info.openclVersion = d.get_info<sycl::info::device::opencl_c_version>();
            } catch (...) {
                info.openclVersion = "N/A (Level Zero / Native)";
            }
            try { info.isGPU = d.is_gpu(); } catch (...) { info.isGPU = false; }
            try { info.isCPU = d.is_cpu(); } catch (...) { info.isCPU = false; }
            try { info.maxComputeUnits = d.get_info<sycl::info::device::max_compute_units>(); } catch (...) { info.maxComputeUnits = 0; }
            try { info.maxWorkGroupSize = d.get_info<sycl::info::device::max_work_group_size>(); } catch (...) { info.maxWorkGroupSize = 0; }
            try { info.globalMemSizeBytes = d.get_info<sycl::info::device::global_mem_size>(); } catch (...) { info.globalMemSizeBytes = 0; }
            try { info.localMemSizeBytes = d.get_info<sycl::info::device::local_mem_size>(); } catch (...) { info.localMemSizeBytes = 0; }
            try {
                auto devSizes = d.get_info<sycl::info::device::max_work_item_sizes<3>>();
                info.maxWorkItemSizes = {devSizes[0], devSizes[1], devSizes[2]};
            } catch (...) {
                info.maxWorkItemSizes = {0, 0, 0};
            }
            result.push_back(info);
        }
    }
    return result;
}

} // namespace sycl_ip
