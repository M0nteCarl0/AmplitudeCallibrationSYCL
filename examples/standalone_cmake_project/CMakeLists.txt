cmake_minimum_required(VERSION 3.20)
project(ExternalSYCLApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable SYCL offload with Intel oneAPI Compiler
if (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fsycl)
    add_link_options(-fsycl)
endif()

# Option A: When using find_package after installing sycl_image_process
# find_package(sycl_image_process REQUIRED)
# add_executable(ExternalApp main.cpp)
# target_link_libraries(ExternalApp PRIVATE sycl_image_process::sycl_image_process)

# Option B: When including as a subdirectory or referencing headers and lib directly
# Include directories for sycl_ip headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../build/lib ${CMAKE_CURRENT_SOURCE_DIR}/../../x64/Release)

add_executable(ExternalApp main.cpp)

# Link against the SYCL Image Processing library
target_link_libraries(ExternalApp PRIVATE SYCL_ImageProcess)
