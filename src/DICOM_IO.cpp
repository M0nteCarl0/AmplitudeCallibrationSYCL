#include "sycl_ip/DICOM_IO.hpp"
#include "sycl_ip/FLR_IO.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <random>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
static std::string pathToUtf8(const std::filesystem::path& p) {
    const std::wstring& ws = p.native();
    if (ws.empty()) return {};
    int size = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0) return {};
    std::string s(size, 0);
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()), &s[0], size, nullptr, nullptr);
    return s;
}
#else
static std::string pathToUtf8(const std::filesystem::path& p) {
    return p.u8string();
}
#endif

namespace sycl_ip {

// Helper: Generates unique UID adhering to DICOM length limit (<= 64 chars)
static std::string generateUID(const std::string& suffix) {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
    
    static std::mt19937_64 rng(static_cast<uint64_t>(ms));
    std::uniform_int_distribution<uint32_t> dist(100000, 999999);
    uint32_t randVal = dist(rng);

    // Prefix: 1.2.826.0.1.3680043.9.7128 (standard non-commercial root)
    std::ostringstream ss;
    ss << "1.2.826.0.1.3680043.9.7128." << ms << "." << randVal;
    if (!suffix.empty()) {
        ss << "." << suffix;
    }
    std::string uid = ss.str();
    if (uid.length() > 64) {
        uid = uid.substr(0, 64);
    }
    return uid;
}

static void getCurrentDateTime(std::string& outDate, std::string& outTime) {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tmNow{};
#if defined(_WIN32)
    localtime_s(&tmNow, &tt);
#else
    localtime_r(&tt, &tmNow);
#endif
    char bufDate[32];
    char bufTime[32];
    std::strftime(bufDate, sizeof(bufDate), "%Y%m%d", &tmNow);
    std::strftime(bufTime, sizeof(bufTime), "%H%M%S", &tmNow);
    outDate = bufDate;
    outTime = bufTime;
}

// Write Explicit VR Little Endian Tag Element
static void writeTagElement(std::ostream& os, uint16_t group, uint16_t element, const char vr[2], const void* data, size_t length) {
    os.write(reinterpret_cast<const char*>(&group), 2);
    os.write(reinterpret_cast<const char*>(&element), 2);
    os.write(vr, 2);

    bool isLongVR = (std::memcmp(vr, "OB", 2) == 0 ||
                     std::memcmp(vr, "OD", 2) == 0 ||
                     std::memcmp(vr, "OF", 2) == 0 ||
                     std::memcmp(vr, "OL", 2) == 0 ||
                     std::memcmp(vr, "OW", 2) == 0 ||
                     std::memcmp(vr, "SQ", 2) == 0 ||
                     std::memcmp(vr, "UC", 2) == 0 ||
                     std::memcmp(vr, "UR", 2) == 0 ||
                     std::memcmp(vr, "UT", 2) == 0 ||
                     std::memcmp(vr, "UN", 2) == 0);

    bool needPad = (length % 2 != 0);
    uint32_t finalLen = static_cast<uint32_t>(needPad ? length + 1 : length);

    if (isLongVR) {
        uint16_t reserved = 0;
        os.write(reinterpret_cast<const char*>(&reserved), 2);
        os.write(reinterpret_cast<const char*>(&finalLen), 4);
    } else {
        uint16_t len16 = static_cast<uint16_t>(finalLen);
        os.write(reinterpret_cast<const char*>(&len16), 2);
    }

    if (data && length > 0) {
        os.write(reinterpret_cast<const char*>(data), length);
    }

    if (needPad) {
        // String VRs (UI, CS, LO, SH, etc.) pad with space or null
        char padChar = (std::memcmp(vr, "UI", 2) == 0) ? '\0' : ' ';
        os.write(&padChar, 1);
    }
}

static void writeTagString(std::ostream& os, uint16_t group, uint16_t element, const char vr[2], const std::string& str) {
    writeTagElement(os, group, element, vr, str.data(), str.size());
}

static void writeTagUS(std::ostream& os, uint16_t group, uint16_t element, uint16_t val) {
    writeTagElement(os, group, element, "US", &val, 2);
}

static void writeTagUL(std::ostream& os, uint16_t group, uint16_t element, uint32_t val) {
    writeTagElement(os, group, element, "UL", &val, 4);
}

bool DICOM_IO::isDICOM(const std::filesystem::path& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) return false;

    char preamble[128];
    char prefix[4];
    is.read(preamble, 128);
    is.read(prefix, 4);

    return (is.gcount() == 4 && std::memcmp(prefix, "DICM", 4) == 0);
}

bool DICOM_IO::save(const std::filesystem::path& path,
                    const uint16_t* pixels,
                    uint16_t width,
                    uint16_t height,
                    const DICOMMetadata& meta) {
    if (!pixels || width == 0 || height == 0) {
        std::cerr << "[DICOM_IO] Invalid image dimensions or null buffer for saving: " << pathToUtf8(path) << "\n";
        return false;
    }

    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        std::cerr << "[DICOM_IO] Failed to open file for writing: " << pathToUtf8(path) << "\n";
        return false;
    }

    // 1. Write 128-byte Preamble and 'DICM' Prefix
    char preamble[128] = {0};
    os.write(preamble, 128);
    os.write("DICM", 4);

    // Prepare Date & Time and UIDs if not set
    std::string curDate, curTime;
    getCurrentDateTime(curDate, curTime);
    std::string studyDate = meta.studyDate.empty() ? curDate : meta.studyDate;
    std::string studyTime = meta.studyTime.empty() ? curTime : meta.studyTime;
    std::string studyUID = meta.studyInstanceUID.empty() ? generateUID("1") : meta.studyInstanceUID;
    std::string seriesUID = meta.seriesInstanceUID.empty() ? generateUID("2") : meta.seriesInstanceUID;
    std::string sopUID = meta.sopInstanceUID.empty() ? generateUID("3") : meta.sopInstanceUID;
    std::string sopClass = meta.sopClassUID.empty() ? "1.2.840.10008.5.1.4.1.1.1" : meta.sopClassUID;

    // 2. Build Group 0002 (File Meta Information) in memory to compute its group length
    std::ostringstream metaStream(std::ios::binary);
    uint8_t metaVersion[2] = {0x00, 0x01};
    writeTagElement(metaStream, 0x0002, 0x0001, "OB", metaVersion, 2);
    writeTagString(metaStream, 0x0002, 0x0002, "UI", sopClass);
    writeTagString(metaStream, 0x0002, 0x0003, "UI", sopUID);
    writeTagString(metaStream, 0x0002, 0x0010, "UI", "1.2.840.10008.1.2.1"); // Explicit VR Little Endian
    writeTagString(metaStream, 0x0002, 0x0012, "UI", "1.2.826.0.1.3680043.9.7128.1.1");
    writeTagString(metaStream, 0x0002, 0x0013, "SH", "SYCL_IP_1_0");

    std::string metaBytes = metaStream.str();
    writeTagUL(os, 0x0002, 0x0000, static_cast<uint32_t>(metaBytes.size()));
    os.write(metaBytes.data(), metaBytes.size());

    // 3. Write Main Dataset Tags (Group > 0002) in ascending (Group, Element) order
    // Group 0008: Identification
    writeTagString(os, 0x0008, 0x0008, "CS", "ORIGINAL\\PRIMARY");
    writeTagString(os, 0x0008, 0x0016, "UI", sopClass);
    writeTagString(os, 0x0008, 0x0018, "UI", sopUID);
    writeTagString(os, 0x0008, 0x0020, "DA", studyDate);
    writeTagString(os, 0x0008, 0x0030, "TM", studyTime);
    writeTagString(os, 0x0008, 0x0060, "CS", meta.modality.empty() ? "DX" : meta.modality);
    writeTagString(os, 0x0008, 0x0070, "LO", meta.manufacturer.empty() ? "Rayence / SYCL_ImageProcess" : meta.manufacturer);
    if (!meta.institutionName.empty()) {
        writeTagString(os, 0x0008, 0x0080, "LO", meta.institutionName);
    }
    if (!meta.studyDescription.empty()) {
        writeTagString(os, 0x0008, 0x1030, "LO", meta.studyDescription);
    }
    if (!meta.seriesDescription.empty()) {
        writeTagString(os, 0x0008, 0x103E, "LO", meta.seriesDescription);
    }

    // Group 0010: Patient
    writeTagString(os, 0x0010, 0x0010, "PN", meta.patientName.empty() ? "Anonymous^Patient" : meta.patientName);
    writeTagString(os, 0x0010, 0x0020, "LO", meta.patientID.empty() ? "DET-3328" : meta.patientID);
    if (!meta.patientBirthDate.empty()) {
        writeTagString(os, 0x0010, 0x0030, "DA", meta.patientBirthDate);
    }
    writeTagString(os, 0x0010, 0x0040, "CS", meta.patientSex.empty() ? "O" : meta.patientSex);

    // Group 0018: Acquisition
    if (!meta.bodyPartExamined.empty()) {
        writeTagString(os, 0x0018, 0x0015, "CS", meta.bodyPartExamined);
    }
    if (meta.kvp > 0.0f) {
        std::ostringstream ssKvp;
        ssKvp << std::fixed << std::setprecision(1) << meta.kvp;
        writeTagString(os, 0x0018, 0x0060, "DS", ssKvp.str());
    }
    if (meta.exposureTimeMs > 0.0f) {
        std::ostringstream ssExp;
        ssExp << std::fixed << std::setprecision(1) << meta.exposureTimeMs;
        writeTagString(os, 0x0018, 0x1150, "IS", ssExp.str());
    }
    if (meta.xRayTubeCurrentMA > 0.0f) {
        std::ostringstream ssMa;
        ssMa << std::fixed << std::setprecision(1) << meta.xRayTubeCurrentMA;
        writeTagString(os, 0x0018, 0x1151, "IS", ssMa.str());
    }
    if (meta.exposureMAS > 0.0f) {
        std::ostringstream ssMas;
        ssMas << std::fixed << std::setprecision(2) << meta.exposureMAS;
        writeTagString(os, 0x0018, 0x1152, "IS", ssMas.str());
    }

    // Group 0020: Relationship
    writeTagString(os, 0x0020, 0x000D, "UI", studyUID);
    writeTagString(os, 0x0020, 0x000E, "UI", seriesUID);
    writeTagString(os, 0x0020, 0x0010, "SH", "1");
    writeTagString(os, 0x0020, 0x0011, "IS", "1");
    writeTagString(os, 0x0020, 0x0013, "IS", "1");

    // Group 0028: Image Presentation (Monochrome 16-bit)
    writeTagUS(os, 0x0028, 0x0002, 1);                             // Samples per Pixel
    writeTagString(os, 0x0028, 0x0004, "CS", "MONOCHROME2");       // Photometric Interpretation
    writeTagUS(os, 0x0028, 0x0010, height);                        // Rows
    writeTagUS(os, 0x0028, 0x0011, width);                         // Columns
    writeTagUS(os, 0x0028, 0x0100, 16);                            // Bits Allocated
    writeTagUS(os, 0x0028, 0x0101, 16);                            // Bits Stored
    writeTagUS(os, 0x0028, 0x0102, 15);                            // High Bit
    writeTagUS(os, 0x0028, 0x0103, 0);                             // Pixel Representation (0 = unsigned)

    // Calculate VOI Window Center / Width if not supplied
    double wc = meta.windowCenter;
    double ww = meta.windowWidth;
    if (wc <= 0.0 || ww <= 0.0) {
        uint16_t minVal = pixels[0];
        uint16_t maxVal = pixels[0];
        double sum = 0.0;
        size_t count = static_cast<size_t>(width) * height;
        for (size_t i = 0; i < count; ++i) {
            uint16_t v = pixels[i];
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
            sum += v;
        }
        wc = sum / static_cast<double>(count);
        ww = std::max<double>(100.0, static_cast<double>(maxVal - minVal));
    }

    std::ostringstream ssWc, ssWw;
    ssWc << std::fixed << std::setprecision(1) << wc;
    ssWw << std::fixed << std::setprecision(1) << ww;
    writeTagString(os, 0x0028, 0x1050, "DS", ssWc.str());
    writeTagString(os, 0x0028, 0x1051, "DS", ssWw.str());

    std::ostringstream ssRi, ssRs;
    ssRi << std::fixed << std::setprecision(1) << meta.rescaleIntercept;
    ssRs << std::fixed << std::setprecision(1) << meta.rescaleSlope;
    writeTagString(os, 0x0028, 0x1052, "DS", ssRi.str());
    writeTagString(os, 0x0028, 0x1053, "DS", ssRs.str());

    // 4. Pixel Data (7FE0, 0010)
    size_t pixelBytes = static_cast<size_t>(width) * height * sizeof(uint16_t);
    writeTagElement(os, 0x7FE0, 0x0010, "OW", pixels, pixelBytes);

    os.flush();
    return os.good();
}

bool DICOM_IO::load(const std::filesystem::path& path,
                    std::vector<uint16_t>& outPixels,
                    uint16_t& outWidth,
                    uint16_t& outHeight,
                    DICOMMetadata& outMeta) {
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        std::cerr << "[DICOM_IO] Cannot open DICOM file: " << pathToUtf8(path) << "\n";
        return false;
    }

    // Read 128 bytes preamble and 'DICM'
    char preamble[128];
    char prefix[4];
    is.read(preamble, 128);
    is.read(prefix, 4);

    bool hasPrefix = (is.gcount() == 4 && std::memcmp(prefix, "DICM", 4) == 0);
    if (!hasPrefix) {
        // Rewind to beginning in case it is raw DICOM stream without preamble
        is.clear();
        is.seekg(0, std::ios::beg);
    }

    bool isExplicitVR = true; // Default for Part 10 files
    uint16_t rows = 0;
    uint16_t cols = 0;
    uint16_t bitsAllocated = 16;
    bool pixelDataFound = false;

    while (is && !pixelDataFound) {
        uint16_t group = 0, element = 0;
        is.read(reinterpret_cast<char*>(&group), 2);
        is.read(reinterpret_cast<char*>(&element), 2);
        if (!is) break;

        // Determine VR and Length
        uint32_t length = 0;
        char vr[3] = {0, 0, 0};

        if (group == 0x0002 || isExplicitVR) {
            is.read(vr, 2);
            if (!is) break;

            bool isLongVR = (std::memcmp(vr, "OB", 2) == 0 ||
                             std::memcmp(vr, "OD", 2) == 0 ||
                             std::memcmp(vr, "OF", 2) == 0 ||
                             std::memcmp(vr, "OL", 2) == 0 ||
                             std::memcmp(vr, "OW", 2) == 0 ||
                             std::memcmp(vr, "SQ", 2) == 0 ||
                             std::memcmp(vr, "UC", 2) == 0 ||
                             std::memcmp(vr, "UR", 2) == 0 ||
                             std::memcmp(vr, "UT", 2) == 0 ||
                             std::memcmp(vr, "UN", 2) == 0);
            if (isLongVR) {
                uint16_t reserved = 0;
                is.read(reinterpret_cast<char*>(&reserved), 2);
                is.read(reinterpret_cast<char*>(&length), 4);
            } else {
                uint16_t len16 = 0;
                is.read(reinterpret_cast<char*>(&len16), 2);
                length = len16;
            }
        } else {
            // Implicit VR Little Endian
            is.read(reinterpret_cast<char*>(&length), 4);
        }

        // Handle Transfer Syntax tag
        if (group == 0x0002 && element == 0x0010) {
            std::string syntax(length, '\0');
            is.read(&syntax[0], length);
            while (!syntax.empty() && (syntax.back() == '\0' || syntax.back() == ' ')) {
                syntax.pop_back();
            }
            if (syntax == "1.2.840.10008.1.2") {
                isExplicitVR = false; // Implicit VR Little Endian
            }
            continue;
        }

        // Handle Image Dimensions & Metadata
        if (group == 0x0028 && element == 0x0010) { // Rows
            is.read(reinterpret_cast<char*>(&rows), sizeof(uint16_t));
            if (length > 2) is.seekg(length - 2, std::ios::cur);
            continue;
        } else if (group == 0x0028 && element == 0x0011) { // Columns
            is.read(reinterpret_cast<char*>(&cols), sizeof(uint16_t));
            if (length > 2) is.seekg(length - 2, std::ios::cur);
            continue;
        } else if (group == 0x0028 && element == 0x0100) { // Bits Allocated
            is.read(reinterpret_cast<char*>(&bitsAllocated), sizeof(uint16_t));
            if (length > 2) is.seekg(length - 2, std::ios::cur);
            continue;
        } else if (group == 0x0008 && element == 0x0060) { // Modality
            outMeta.modality.resize(length);
            is.read(&outMeta.modality[0], length);
            continue;
        } else if (group == 0x0018 && element == 0x0060) { // KVP
            std::string kvpStr(length, '\0');
            is.read(&kvpStr[0], length);
            try { outMeta.kvp = std::stof(kvpStr); } catch (...) {}
            continue;
        } else if (group == 0x0010 && element == 0x0010) { // PatientName
            outMeta.patientName.resize(length);
            is.read(&outMeta.patientName[0], length);
            continue;
        } else if (group == 0x0010 && element == 0x0020) { // PatientID
            outMeta.patientID.resize(length);
            is.read(&outMeta.patientID[0], length);
            continue;
        } else if (group == 0x0008 && element == 0x1030) { // StudyDescription
            outMeta.studyDescription.resize(length);
            is.read(&outMeta.studyDescription[0], length);
            continue;
        } else if (group == 0x0008 && element == 0x103E) { // SeriesDescription
            outMeta.seriesDescription.resize(length);
            is.read(&outMeta.seriesDescription[0], length);
            continue;
        } else if (group == 0x7FE0 && element == 0x0010) { // Pixel Data
            if (rows == 0 || cols == 0) {
                std::cerr << "[DICOM_IO] Pixel Data tag found before valid Rows/Columns!\n";
                return false;
            }

            size_t pixelCount = static_cast<size_t>(rows) * cols;
            outPixels.resize(pixelCount);
            is.read(reinterpret_cast<char*>(outPixels.data()), pixelCount * sizeof(uint16_t));
            if (is) {
                pixelDataFound = true;
                outWidth = cols;
                outHeight = rows;
            }
            break;
        } else {
            // Skip other tags
            is.seekg(length, std::ios::cur);
        }
    }

    if (!pixelDataFound) {
        std::cerr << "[DICOM_IO] Failed to find or read (7FE0, 0010) Pixel Data in: " << pathToUtf8(path) << "\n";
        return false;
    }

    return true;
}

bool DICOM_IO::convertFLRtoDICOM(const std::filesystem::path& flrPath,
                                const std::filesystem::path& dcmPath,
                                const DICOMMetadata& meta) {
    FLRImage flr;
    if (!flr.load(flrPath)) {
        std::cerr << "[DICOM_IO] Failed to load source FLR image: " << pathToUtf8(flrPath) << "\n";
        return false;
    }

    DICOMMetadata m = meta;
    if (m.seriesDescription.empty()) {
        m.seriesDescription = pathToUtf8(flrPath.stem());
    }

    bool ok = save(dcmPath, flr.getHostData(), flr.getWidth(), flr.getHeight(), m);
    if (ok) {
        std::cout << "[DICOM_IO] Converted " << pathToUtf8(flrPath.filename()) 
                  << " -> " << pathToUtf8(dcmPath) 
                  << " (" << flr.getWidth() << "x" << flr.getHeight() << ")\n";
    }
    return ok;
}

bool DICOM_IO::convertDICOMtoFLR(const std::filesystem::path& dcmPath,
                                const std::filesystem::path& flrPath) {
    std::vector<uint16_t> pixels;
    uint16_t w = 0, h = 0;
    DICOMMetadata meta;

    if (!load(dcmPath, pixels, w, h, meta)) {
        std::cerr << "[DICOM_IO] Failed to load source DICOM image: " << pathToUtf8(dcmPath) << "\n";
        return false;
    }

    FLRImage flr(w, h);
    std::memcpy(flr.getHostData(), pixels.data(), pixels.size() * sizeof(uint16_t));
    bool ok = flr.save(flrPath);
    if (ok) {
        std::cout << "[DICOM_IO] Converted " << pathToUtf8(dcmPath.filename()) 
                  << " -> " << pathToUtf8(flrPath) 
                  << " (" << w << "x" << h << ")\n";
    }
    return ok;
}

int DICOM_IO::convertDirectory(const std::filesystem::path& srcDir,
                              const std::filesystem::path& dstDir,
                              bool flrToDcm,
                              float defaultKVP) {
    if (!std::filesystem::exists(srcDir)) {
        std::cerr << "[DICOM_IO] Source directory does not exist: " << pathToUtf8(srcDir) << "\n";
        return 0;
    }

    std::filesystem::create_directories(dstDir);
    int convertedCount = 0;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(srcDir)) {
        if (!entry.is_regular_file()) continue;

        auto relPath = std::filesystem::relative(entry.path(), srcDir);
        auto targetPath = dstDir / relPath;

        if (flrToDcm && entry.path().extension() == ".flr") {
            targetPath.replace_extension(".dcm");
            std::filesystem::create_directories(targetPath.parent_path());

            DICOMMetadata meta;
            meta.seriesDescription = pathToUtf8(entry.path().stem());
            if (defaultKVP > 0.0f) {
                meta.kvp = defaultKVP;
            }

            if (convertFLRtoDICOM(entry.path(), targetPath, meta)) {
                convertedCount++;
            }
        } else if (!flrToDcm && (entry.path().extension() == ".dcm" || entry.path().extension() == ".dicom")) {
            targetPath.replace_extension(".flr");
            std::filesystem::create_directories(targetPath.parent_path());

            if (convertDICOMtoFLR(entry.path(), targetPath)) {
                convertedCount++;
            }
        }
    }

    std::cout << "[DICOM_IO] Successfully converted " << convertedCount << " files.\n";
    return convertedCount;
}

} // namespace sycl_ip
