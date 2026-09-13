# Environment Requirements & Installation Guide

This guide provides comprehensive instructions for configuring the hardware drivers, toolchains, and environment required to build and run the **SYCL Amplitude Calibration & Image Processing Library** on **Windows** and **Linux**.

---

## 📋 Table of Contents
- [Hardware & Compute Targets](#-hardware--compute-targets)
- [Software Prerequisites](#-software-prerequisites)
- [Windows Installation Guide](#-windows-installation-guide)
  - [1. Visual Studio 2022](#1-install-visual-studio-2022)
  - [2. Intel Graphics Drivers](#2-install-intel-graphics-drivers)
  - [3. Intel® oneAPI Base Toolkit](#3-install-intel-oneapi-base-toolkit)
  - [4. Environment Setup (`setvars.bat`)](#4-initialize-environment-variables)
  - [5. Verifying SYCL Device Detection](#5-verify-device-detection)
- [Linux Installation Guide (Ubuntu)](#-linux-installation-guide-ubuntu)
  - [1. System Packages & Build Tools](#1-install-system-packages)
  - [2. Intel Compute Runtime (Level-Zero / OpenCL)](#2-install-intel-compute-runtime)
  - [3. Intel® oneAPI DPC++ Compiler via APT](#3-install-intel-oneapi-compiler)
  - [4. Environment Setup (`setvars.sh`)](#4-initialize-environment)
- [Verifying the Setup](#-verifying-the-setup)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 💻 Hardware & Compute Targets

The library relies on **SYCL 2020 / DPC++** for heterogeneous acceleration:

| Target | Supported Hardware | Recommended Runtime / Backend |
| :--- | :--- | :--- |
| **GPU (Integrated)** | Intel® Iris® Xe Graphics (Gen12 / 96 EU), UHD Graphics 770+ | **Intel Level-Zero** or OpenCL NEO |
| **GPU (Discrete)** | Intel® Arc™ A-Series (A380, A750, A770), Arc B-Series, Data Center GPU Flex / Max | **Intel Level-Zero** |
| **CPU** | Multi-core Intel® Core™ (11th, 12th, 13th, 14th Gen+), Intel® Xeon® Scalable | **Intel OpenCL CPU Runtime** |
| **RAM** | Minimum 8 GB (16 GB+ strongly recommended for 3328x3328 16-bit multi-frame datasets) | Host RAM |

---

## 🛠 Software Prerequisites

1. **C++17 standard compliance**
2. **Intel® oneAPI Base Toolkit (v2024.1 or newer)**:
   - Intel® oneAPI DPC++/C++ Compiler (`icx` on Windows, `icpx` on Linux)
   - Intel® oneAPI DPC++ Library (oneDPL)
   - Intel® oneAPI Level-Zero Loader and OpenCL ICD loader
3. **Build Systems**:
   - **Windows:** Visual Studio 2022 (v17.5+) with MSVC v143 and C++ Desktop workload, OR CMake 3.20+ with Ninja
   - **Linux:** CMake 3.20+, Ninja 1.10+, GCC/Clang base toolchain

---

## 🪟 Windows Installation Guide

### 1. Install Visual Studio 2022
1. Download [Visual Studio 2022 Installer](https://visualstudio.microsoft.com/downloads/) (Community, Professional, or Enterprise).
2. During installation, select the workload:
   - **Desktop development with C++**
   - Ensure the following components are checked:
     - *MSVC v143 - VS 2022 C++ x64/x86 build tools*
     - *Windows 10 or 11 SDK (10.0.22621.0 or newer)*
     - *C++ CMake tools for Windows*

### 2. Install Intel Graphics Drivers
To enable Level-Zero and OpenCL GPU compute support:
1. Download the latest **Intel Graphics Driver** for Windows:
   - [Intel® Arc™ & Iris® Xe Graphics - Windows](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)
2. Run the installer and perform a clean installation. Reboot your computer if prompted.

### 3. Install Intel® oneAPI Base Toolkit

#### Method A: Official Installer (Recommended)
1. Go to the [Intel® oneAPI Base Toolkit Download Page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
2. Select **Windows**, **Online or Local Installer**, and run the setup.
3. In the component selection screen:
   - Ensure **Intel® oneAPI DPC++/C++ Compiler** is selected.
   - Ensure **Integration with Microsoft Visual Studio** is checked (this registers the `Intel C++ Compiler 2024` platform toolset inside Visual Studio).
4. Complete the installation (default directory: `C:\Program Files (x86)\Intel\oneAPI` or `D:\Program Files (x86)\Intel\oneAPI`).

#### Method B: Via Windows Package Manager (`winget`)
In an administrative PowerShell terminal:
```powershell
winget install --id Intel.OneAPI.BaseToolkit --accept-package-agreements --accept-source-agreements --silent
```

### 4. Initialize Environment Variables
Before running CMake or command-line compilation, initialize the oneAPI environment:

```cmd
:: If installed on C: drive
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

:: If installed on D: drive
call "D:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

> [!TIP]
> The provided scripts (`build_and_run.bat`, `run_all.bat`, `run_gpu_test.bat`, etc.) automatically locate and initialize `setvars.bat` if it is not already in your `%PATH%`.

### 5. Verify Device Detection
In a terminal with the initialized oneAPI environment, run:
```powershell
sycl-ls
```
Expected output example:
```text
[opencl:gpu:0] Intel(R) OpenCL Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO [31.0.101.4502]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900H OpenCL 3.0 [2024.17.3.0.08]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Iris(R) Xe Graphics 1.3 [1.3.26370]
```

---

## 🐧 Linux Installation Guide (Ubuntu)

### 1. Install System Packages
```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    ca-certificates \
    gnupg \
    wget
```

### 2. Install Intel Compute Runtime
To enable Intel GPU acceleration on Linux:
```bash
sudo apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero
```
Ensure your user belongs to the `render` and `video` groups to access GPU devices:
```bash
sudo usermod -a -G render,video $USER
```

### 3. Install Intel® oneAPI Compiler
Configure the official Intel APT repository:
```bash
# Add Intel GPG key
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# Add oneAPI APT repository
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    sudo tee /etc/apt/sources.list.d/oneAPI.list

# Update package lists and install DPC++ compiler
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    intel-oneapi-compiler-dpcpp-cpp \
    intel-oneapi-runtime-libs
```

### 4. Initialize Environment
Add the following line to your `~/.bashrc` or run before building:
```bash
source /opt/intel/oneapi/setvars.sh
```

---

## 🔍 Verifying the Setup

### 1. Check SYCL Device Availability
```bash
sycl-ls
```
Ensure at least one GPU (`[ext_oneapi_level_zero:gpu]` or `[opencl:gpu]`) and/or CPU device is listed.

### 2. Build and Run the Validation Suite

#### On Windows:
Double-click or run from command prompt:
```cmd
build_and_run.bat
```
Or run the pre-built binary:
```cmd
run_gpu_test.bat
```

#### On Linux:
```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/bin/AmplitudeCallibrationSYCL --test --device gpu
```

All self-tests should report `PASSED`.

---

## ❓ Troubleshooting & FAQ

### 1. `sycl-ls` does not list any GPU device
- **Cause:** GPU drivers or Level-Zero runtime are missing or outdated.
- **Solution:** Update your Intel Graphics Driver to the latest release. On Linux, ensure `intel-level-zero-gpu` is installed and your user is added to group `render` (`sudo usermod -a -G render $USER`), then reboot.

### 2. Visual Studio Error: `The build tools for Intel C++ Compiler 2024 cannot be found`
- **Cause:** Visual Studio integration was not selected during Intel oneAPI installation, or Visual Studio was updated after oneAPI was installed.
- **Solution:** Re-run the Intel oneAPI Base Toolkit installer, select **Modify**, and make sure **Integration with Microsoft Visual Studio** is checked. Alternatively, run:
  ```cmd
  "C:\Program Files (x86)\Intel\oneAPI\compiler_ide\latest\ide_integration.bat"
  ```

### 3. Out of Memory on GPU during large calibration
- **Cause:** 11M pixel 16-bit frames require substantial VRAM when allocating multiple working buffers concurrently.
- **Solution:** The library automatically falls back to system host memory or CPU target if device USM allocation fails. You can explicitly force CPU execution by passing `--device cpu`:
  ```cmd
  AmplitudeCallibrationSYCL.exe --device cpu --all
  ```
