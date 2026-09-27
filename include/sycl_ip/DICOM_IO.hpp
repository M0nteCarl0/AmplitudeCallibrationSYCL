#pragma once

#include "SYCL_Export.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>

namespace sycl_ip {

struct SYCL_IP_API DICOMMetadata {
    std::string patientName = "Anonymous^Patient";
    std::string patientID = "DET-3328";
    std::string patientBirthDate = "";
    std::string patientSex = "O";
    std::string studyDate = "";        // YYYYMMDD (auto-populated if empty)
    std::string studyTime = "";        // HHMMSS (auto-populated if empty)
    std::string studyDescription = "Amplitude Calibration Acquisition";
    std::string seriesDescription = "16-bit Detector Exposure";
    std::string modality = "DX";       // Digital Radiography (DX) or Computed Radiography (CR)
    std::string manufacturer = "Rayence / SYCL_ImageProcess";
    std::string institutionName = "Radiography Facility";
    std::string bodyPartExamined = "PHANTOM";
    std::string sopClassUID = "1.2.840.10008.5.1.4.1.1.1"; // Digital X-Ray Image Storage - For Presentation
    std::string studyInstanceUID = "";  // Auto-generated if empty
    std::string seriesInstanceUID = ""; // Auto-generated if empty
    std::string sopInstanceUID = "";    // Auto-generated if empty
    
    // Acquisition parameters
    float kvp = 0.0f;                  // Tube voltage in kV (e.g. 490.0)
    float exposureTimeMs = 0.0f;       // Exposure time in milliseconds
    float xRayTubeCurrentMA = 0.0f;    // Tube current in mA
    float exposureMAS = 0.0f;          // Tube current-time product in mAs

    // Display & VOI LUT parameters
    double windowCenter = 0.0;         // 0.0 = auto-calculate from image mean
    double windowWidth = 0.0;          // 0.0 = auto-calculate from image dynamic range
    double rescaleIntercept = 0.0;
    double rescaleSlope = 1.0;
};

class SYCL_IP_API DICOM_IO {
public:
    // Check if a file is a valid Part-10 DICOM file (examines preamble and 'DICM' prefix)
    static bool isDICOM(const std::filesystem::path& path);

    // Save a 16-bit monochrome image buffer to a DICOM Part-10 file (Explicit VR Little Endian)
    static bool save(const std::filesystem::path& path,
                     const uint16_t* pixels,
                     uint16_t width,
                     uint16_t height,
                     const DICOMMetadata& meta = {});

    // Load a 16-bit monochrome image from a DICOM file
    static bool load(const std::filesystem::path& path,
                     std::vector<uint16_t>& outPixels,
                     uint16_t& outWidth,
                     uint16_t& outHeight,
                     DICOMMetadata& outMeta);

    // Convert .flr file directly to .dcm file
    static bool convertFLRtoDICOM(const std::filesystem::path& flrPath,
                                 const std::filesystem::path& dcmPath,
                                 const DICOMMetadata& meta = {});

    // Convert .dcm file directly to .flr file
    static bool convertDICOMtoFLR(const std::filesystem::path& dcmPath,
                                 const std::filesystem::path& flrPath);

    // Batch convert all .flr files in a source directory to .dcm files in destination directory
    static int convertDirectory(const std::filesystem::path& srcDir,
                                const std::filesystem::path& dstDir,
                                bool flrToDcm = true,
                                float defaultKVP = 0.0f);
};

} // namespace sycl_ip
