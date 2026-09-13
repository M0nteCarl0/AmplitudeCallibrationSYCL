#include "sycl_ip/SYCL_Average.hpp"
#include "sycl_ip/SYCL_Statistics.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>

namespace fs = std::filesystem;

namespace sycl_ip {

SYCL_Average::SYCL_Average(int width, int height)
    : m_width(width), m_height(height) {
    m_roiMean = RectROI(width / 4, height / 4, width * 3 / 4, height * 3 / 4);
    m_roiRms  = RectROI(width / 4, height / 4, width * 3 / 4, height * 3 / 4);
}

void SYCL_Average::accumulate(SYCLContext& ctx, uint32_t* d_accum, const uint16_t* d_in, size_t count) {
    auto& q = ctx.getQueue();
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
        d_accum[idx] += static_cast<uint32_t>(d_in[idx]);
    }).wait();
}

void SYCL_Average::normalize(SYCLContext& ctx, const uint32_t* d_accum, uint16_t* d_out, size_t count, int numImages) {
    if (numImages <= 0) return;
    float invCount = 1.0f / static_cast<float>(numImages);
    auto& q = ctx.getQueue();

    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
        float val = static_cast<float>(d_accum[idx]) * invCount;
        val = sycl::clamp(val + 0.5f, 0.0f, 65535.0f);
        d_out[idx] = static_cast<uint16_t>(val);
    }).wait();
}

bool SYCL_Average::averageFileList(SYCLContext& ctx, const std::vector<fs::path>& fileList, const fs::path& outFlrPath, ExposureAverageResult& outResult) {
    if (fileList.empty()) {
        std::cerr << "[SYCL_Average] Error: No input files provided for averaging.\n";
        return false;
    }

    size_t pixelCount = static_cast<size_t>(m_width) * m_height;
    uint32_t* d_accum = ctx.allocateDevice<uint32_t>(pixelCount);
    uint16_t* d_in = ctx.allocateDevice<uint16_t>(pixelCount);
    uint16_t* d_out = ctx.allocateDevice<uint16_t>(pixelCount);

    ctx.getQueue().memset(d_accum, 0, pixelCount * sizeof(uint32_t)).wait();

    double totalSingleRms = 0.0;
    int validCount = 0;

    for (const auto& file : fileList) {
        FLRImage img;
        if (!img.load(file)) {
            std::cerr << "[SYCL_Average] Warning: Skipping unreadable file " << file.u8string() << "\n";
            continue;
        }

        if (img.getWidth() != m_width || img.getHeight() != m_height) {
            std::cerr << "[SYCL_Average] Warning: Size mismatch in " << file.u8string() << " ("
                      << img.getWidth() << "x" << img.getHeight() << " vs " << m_width << "x" << m_height << ")\n";
            m_width = img.getWidth();
            m_height = img.getHeight();
            pixelCount = static_cast<size_t>(m_width) * m_height;
        }

        img.copyToDevice(ctx, d_in);

        // Compute single image Mean and RMS with sliding grid
        double singleMean = 0.0, singleRms = 0.0;
        SYCL_Statistics::computeSlidingMeanRMS(ctx, d_in, m_width, m_height, m_roiRms, singleMean, singleRms);
        totalSingleRms += singleRms;

        accumulate(ctx, d_accum, d_in, pixelCount);
        validCount++;
    }

    if (validCount == 0) {
        ctx.free<uint32_t>(d_accum);
        ctx.free<uint16_t>(d_in);
        ctx.free<uint16_t>(d_out);
        return false;
    }

    // Normalize accumulator
    normalize(ctx, d_accum, d_out, pixelCount, validCount);

    // Compute averaged image statistics
    double averMean = 0.0, averRms = 0.0;
    SYCL_Statistics::computeSlidingMeanRMS(ctx, d_out, m_width, m_height, m_roiRms, averMean, averRms);

    outResult.amplAver = averMean;
    outResult.rmsAver = averRms;
    outResult.rmsSingle = totalSingleRms / validCount;
    outResult.numImages = validCount;
    outResult.outputFlrPath = outFlrPath;

    outResult.averagedImage.resize(m_width, m_height);
    outResult.averagedImage.copyFromDevice(ctx, d_out);

    if (!outFlrPath.empty()) {
        outResult.averagedImage.save(outFlrPath);
        std::cout << "[SYCL_Average] Saved averaged image to: " << outFlrPath.u8string()
                  << " (Mean: " << averMean << ", RMS: " << averRms << ", Count: " << validCount << ")\n";
    }

    ctx.free<uint32_t>(d_accum);
    ctx.free<uint16_t>(d_in);
    ctx.free<uint16_t>(d_out);
    return true;
}

bool SYCL_Average::processDirectory(SYCLContext& ctx, const fs::path& baseDir, const fs::path& outArmPath) {
    m_results.clear();

    std::vector<std::string> subDirs;
    fs::path listFilePath = baseDir / "ListOfDIRs.txt";

    if (fs::exists(listFilePath)) {
        std::ifstream listFile(listFilePath);
        std::string line;
        while (std::getline(listFile, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
                line.pop_back();
            }
            if (!line.empty()) {
                subDirs.push_back(line);
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(baseDir)) {
            if (entry.is_directory()) {
                subDirs.push_back(entry.path().filename().u8string());
            }
        }
    }

    if (subDirs.empty()) {
        std::cerr << "[SYCL_Average] Error: No exposure subdirectories found in " << baseDir.u8string() << "\n";
        return false;
    }

    std::cout << "[SYCL_Average] Processing " << subDirs.size() << " exposure directories in " << baseDir.u8string() << "...\n";

    ARMHeader armHeader;
    armHeader.suppressLevel = 0;
    armHeader.time = std::time(nullptr);
    armHeader.rectAmplLeft = m_roiMean.left;
    armHeader.rectAmplTop = m_roiMean.top;
    armHeader.rectAmplRight = m_roiMean.right;
    armHeader.rectAmplBottom = m_roiMean.bottom;
    armHeader.rectRmsLeft = m_roiRms.left;
    armHeader.rectRmsTop = m_roiRms.top;
    armHeader.rectRmsRight = m_roiRms.right;
    armHeader.rectRmsBottom = m_roiRms.bottom;

    std::map<double, ARMRecord> armRecords;

    for (const auto& subDirName : subDirs) {
        fs::path dirPath = baseDir / subDirName;
        if (!fs::is_directory(dirPath)) continue;

        std::vector<fs::path> flrFiles;
        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".flr") {
                flrFiles.push_back(entry.path());
            }
        }

        if (flrFiles.empty()) {
            std::cerr << "[SYCL_Average] Warning: No .flr files in " << dirPath.u8string() << "\n";
            continue;
        }

        fs::path outFlr = baseDir / (subDirName + ".flr");
        ExposureAverageResult res;
        res.exposureName = subDirName;

        if (averageFileList(ctx, flrFiles, outFlr, res)) {
            m_results[res.amplAver] = res;

            ARMRecord rec;
            rec.amplAver = res.amplAver;
            rec.rmsAver = res.rmsAver;
            rec.rmsSingle = res.rmsSingle;
            rec.statistic = res.numImages;
            armRecords[res.amplAver] = rec;
        }
    }

    armHeader.points = static_cast<int32_t>(armRecords.size());

    fs::path armFile = outArmPath;
    if (armFile.empty()) {
        std::string folderName = baseDir.filename().u8string();
        if (folderName.empty()) folderName = "Calibration";
        armFile = baseDir / (folderName + ".ARM");
    }

    saveARM(armFile, armHeader, armRecords);
    std::cout << "[SYCL_Average] Generated calibration ARM file: " << armFile.u8string() << " (" << armRecords.size() << " points)\n";

    return !m_results.empty();
}

bool SYCL_Average::saveARM(const fs::path& filePath, const ARMHeader& header, const std::map<double, ARMRecord>& records) {
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) return false;

    file.write(reinterpret_cast<const char*>(&header), sizeof(ARMHeader));
    for (const auto& [ampl, rec] : records) {
        file.write(reinterpret_cast<const char*>(&rec), sizeof(ARMRecord));
    }
    return file.good();
}

bool SYCL_Average::loadARM(const fs::path& filePath, ARMHeader& header, std::map<double, ARMRecord>& records) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) return false;

    file.read(reinterpret_cast<char*>(&header), sizeof(ARMHeader));
    records.clear();

    for (int i = 0; i < header.points; ++i) {
        ARMRecord rec;
        file.read(reinterpret_cast<char*>(&rec), sizeof(ARMRecord));
        if (!file.good()) break;
        records[rec.amplAver] = rec;
    }
    return true;
}

} // namespace sycl_ip
