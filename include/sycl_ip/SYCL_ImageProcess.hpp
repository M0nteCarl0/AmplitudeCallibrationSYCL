#pragma once

/**
 * @file SYCL_ImageProcess.hpp
 * @brief Master include header for the SYCL Image Processing & Amplitude Calibration Library.
 * 
 * Includes all public headers for device context management, FLR file I/O,
 * statistics/reductions, multi-image averaging, polynomial calibration fitting,
 * image repair/reconstruction, 2D filters, defect spike suppression, and transforms.
 */

#include "SYCL_Export.hpp"
#include "SYCLContext.hpp"
#include "FLR_IO.hpp"
#include "SYCL_Statistics.hpp"
#include "SYCL_Average.hpp"
#include "SYCL_Calibrate.hpp"
#include "SYCL_Repair.hpp"
#include "SYCL_Filters.hpp"
#include "SYCL_Suppress.hpp"
#include "SYCL_Transforms.hpp"
#include "SYCL_Pipeline.hpp"
