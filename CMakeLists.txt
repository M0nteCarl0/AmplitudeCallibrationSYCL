cmake_minimum_required(VERSION 3.20)
project(SYCL_ImageProcess_Suite VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable SYCL offload with Intel oneAPI Compiler
if (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fsycl)
    add_link_options(-fsycl)
    message(STATUS "Configured Intel SYCL offload compiler flags (-fsycl)")
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ==============================================================================
# 1. Core SYCL Image Processing Library Target
# ==============================================================================
set(SYCL_IP_HEADERS
    include/sycl_ip/SYCL_Export.hpp
    include/sycl_ip/SYCL_ImageProcess.hpp
    include/sycl_ip/SYCLContext.hpp
    include/sycl_ip/FLR_IO.hpp
    include/sycl_ip/SYCL_Statistics.hpp
    include/sycl_ip/SYCL_Average.hpp
    include/sycl_ip/SYCL_Calibrate.hpp
    include/sycl_ip/SYCL_Repair.hpp
    include/sycl_ip/SYCL_Filters.hpp
    include/sycl_ip/SYCL_Suppress.hpp
    include/sycl_ip/SYCL_Transforms.hpp
    include/sycl_ip/SYCL_Pipeline.hpp
)

set(SYCL_IP_SOURCES
    src/SYCLContext.cpp
    src/FLR_IO.cpp
    src/SYCL_Statistics.cpp
    src/SYCL_Average.cpp
    src/SYCL_Calibrate.cpp
    src/SYCL_Repair.cpp
    src/SYCL_Filters.cpp
    src/SYCL_Suppress.cpp
    src/SYCL_Transforms.cpp
    src/SYCL_Pipeline.cpp
)

# Option to build as SHARED or STATIC library (default: STATIC for seamless link)
option(BUILD_SHARED_LIBS "Build as dynamic shared library" OFF)

add_library(SYCL_ImageProcess ${SYCL_IP_SOURCES} ${SYCL_IP_HEADERS})
add_library(sycl_image_process::sycl_image_process ALIAS SYCL_ImageProcess)

target_include_directories(SYCL_ImageProcess
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (BUILD_SHARED_LIBS)
    target_compile_definitions(SYCL_ImageProcess PRIVATE SYCL_IP_EXPORTS)
else()
    target_compile_definitions(SYCL_ImageProcess PUBLIC SYCL_IP_STATIC)
endif()

# ==============================================================================
# 2. Example Applications & CLI Tools
# ==============================================================================
option(BUILD_EXAMPLES "Build example applications" ON)

if (BUILD_EXAMPLES)
    # CLI Pipeline & Self-Test Application
    add_executable(AmplitudeCallibrationSYCL
        AmplitudeCallibrationSYCL/AmplitudeCallibrationSYCL.cpp
    )
    target_link_libraries(AmplitudeCallibrationSYCL PRIVATE SYCL_ImageProcess)

    # Simple Consumer Example Application
    add_executable(SimpleExample
        examples/simple_example/main.cpp
    )
    target_link_libraries(SimpleExample PRIVATE SYCL_ImageProcess)
endif()

# ==============================================================================
# 3. Installation & CMake Package Config Export
# ==============================================================================
include(GNUInstallDirs)

install(TARGETS SYCL_ImageProcess
    EXPORT SYCL_ImageProcessTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT SYCL_ImageProcessTargets
    FILE SYCL_ImageProcessTargets.cmake
    NAMESPACE sycl_image_process::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sycl_image_process
)
