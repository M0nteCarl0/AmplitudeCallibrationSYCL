#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SYCL Amplitude Calibration: NVIDIA Jetson Test & Build Runner
# Target: Jetson Orin (sm_87) / Linux aarch64
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEVICE="gpu"
RUN_MODE="test"
CLEAN_BUILD=false
PACKAGE_ARTIFACTS=false

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --device <gpu|cpu|auto>    SYCL target device (default: gpu)"
    echo "  --test                     Run validation test suite (default)"
    echo "  --bench                    Run performance benchmark suite"
    echo "  --all                      Run full validation, benchmarks, and dataset"
    echo "  --build-only               Compile only without executing tests"
    echo "  --package                  Package standalone ARM64 release tarball"
    echo "  --clean                    Clean build directory before compiling"
    echo "  -h, --help                 Show this help message"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --test)
            RUN_MODE="test"
            shift
            ;;
        --bench)
            RUN_MODE="bench"
            shift
            ;;
        --all)
            RUN_MODE="all"
            shift
            ;;
        --build-only)
            RUN_MODE="build"
            shift
            ;;
        --package)
            PACKAGE_ARTIFACTS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

echo "================================================================="
echo "   NVIDIA Jetson SYCL Test & Build Suite                        "
echo "================================================================="
echo "Repository:  ${REPO_DIR}"
echo "Target:      Linux aarch64 (NVIDIA Jetson)"
echo "SYCL Target: ${DEVICE}"
echo "Run Mode:    ${RUN_MODE}"
echo ""

# 1. Detect AdaptiveCpp compiler
ACPP_BIN=""
if command -v acpp &>/dev/null; then
    ACPP_BIN="$(command -v acpp)"
elif [[ -x "/home/montecarlo/.local/share/mamba/envs/sycl/bin/acpp" ]]; then
    ACPP_BIN="/home/montecarlo/.local/share/mamba/envs/sycl/bin/acpp"
elif [[ -x "${HOME}/.local/share/mamba/envs/sycl/bin/acpp" ]]; then
    ACPP_BIN="${HOME}/.local/share/mamba/envs/sycl/bin/acpp"
else
    echo "[ERROR] AdaptiveCpp compiler ('acpp') not found!"
    echo "Please activate your SYCL conda/mamba environment or install AdaptiveCpp."
    exit 1
fi
echo "[OK] Found AdaptiveCpp: ${ACPP_BIN}"

# 2. Detect CUDA toolkit installation
CUDA_PATH=""
if [[ -d "/usr/local/cuda-12.6" ]]; then
    CUDA_PATH="/usr/local/cuda-12.6"
elif [[ -d "/usr/local/cuda" ]]; then
    CUDA_PATH="/usr/local/cuda"
elif [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
    CUDA_PATH="${CUDA_HOME}"
fi

if [[ -n "${CUDA_PATH}" ]]; then
    echo "[OK] Found CUDA Toolkit: ${CUDA_PATH}"
else
    echo "[WARN] CUDA Toolkit directory not found under /usr/local/cuda*. Relying on default clang search paths."
fi

# 3. Detect GPU Architecture (default sm_87 for Jetson Orin)
CUDA_ARCH="sm_87"
if [[ -x "${CUDA_PATH}/bin/nvcc" ]]; then
    echo "[OK] NVCC present: $("${CUDA_PATH}/bin/nvcc" --version | grep 'release' || true)"
fi

# 4. Prepare CMake flags
BUILD_DIR="${REPO_DIR}/build"
if [[ "${CLEAN_BUILD}" == true && -d "${BUILD_DIR}" ]]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

CXX_FLAGS="--acpp-targets=generic;cuda:${CUDA_ARCH}"
if [[ -n "${CUDA_PATH}" ]]; then
    CXX_FLAGS="${CXX_FLAGS} --acpp-cuda-path=${CUDA_PATH}"
fi
CXX_FLAGS="${CXX_FLAGS} -I/usr/include"

LINK_FLAGS="-L/usr/lib/aarch64-linux-gnu"
if [[ -n "${CUDA_PATH}" && -d "${CUDA_PATH}/lib64" ]]; then
    LINK_FLAGS="${LINK_FLAGS} -L${CUDA_PATH}/lib64"
fi

echo ""
echo "=== Step 1: Configuring with CMake & Ninja ==="
cmake -B "${BUILD_DIR}" -S "${REPO_DIR}" -G Ninja \
    -DCMAKE_CXX_COMPILER="${ACPP_BIN}" \
    -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${LINK_FLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${LINK_FLAGS}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_EXAMPLES=ON

echo ""
echo "=== Step 2: Compiling Binaries ==="
cmake --build "${BUILD_DIR}" --config Release

EXE="${BUILD_DIR}/bin/AmplitudeCallibrationSYCL"
if [[ ! -x "${EXE}" ]]; then
    echo "[ERROR] Executable not found at ${EXE}!"
    exit 1
fi
echo "[OK] Executable ready: ${EXE}"

# Suppress harmless OpenCL driver warning on systems without OpenCL ICD
export ACPP_VISIBILITY_MASK="cuda;omp"

# Execute requested run mode
if [[ "${RUN_MODE}" == "build" ]]; then
    echo "[OK] Build completed successfully."
elif [[ "${RUN_MODE}" == "test" ]]; then
    echo ""
    echo "=== Step 3: Executing Validation Suite on ${DEVICE} ==="
    cd "${REPO_DIR}"
    "${EXE}" --test --device "${DEVICE}"
elif [[ "${RUN_MODE}" == "bench" ]]; then
    echo ""
    echo "=== Step 3: Executing GPU vs CPU Benchmark Suite ==="
    cd "${REPO_DIR}"
    "${EXE}" --bench
elif [[ "${RUN_MODE}" == "all" ]]; then
    echo ""
    echo "=== Step 3: Executing Full Suite (--all) ==="
    cd "${REPO_DIR}"
    "${EXE}" --all
fi

# 5. Optional packaging
if [[ "${PACKAGE_ARTIFACTS}" == true ]]; then
    echo ""
    echo "=== Step 4: Packaging Standalone ARM64 Release ==="
    "${SCRIPT_DIR}/package_jetson_arm64.sh"
fi

echo ""
echo "================================================================="
echo "   NVIDIA JETSON SYCL EXECUTION COMPLETED SUCCESSFULLY [OK]      "
echo "================================================================="
