#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SYCL Amplitude Calibration: Linux ARM64 Standalone Packager
# Builds self-contained distribution tarball with all SYCL/CUDA runtimes bundled
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGE_DIR="${REPO_DIR}/artifacts/sycl-imageprocess-linux-arm64"
OUTPUT_TAR="${REPO_DIR}/artifacts/sycl-imageprocess-linux-arm64.tar.gz"

echo "================================================================="
echo "   Staging Linux ARM64 Standalone Distribution Package           "
echo "================================================================="
echo "Staging Path: ${STAGE_DIR}"
echo "Archive Path: ${OUTPUT_TAR}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/bin" "${STAGE_DIR}/lib" "${STAGE_DIR}/lib/hipSYCL" "${STAGE_DIR}/include" "${REPO_DIR}/artifacts"

# 1. Copy built executables and libraries
BUILD_BIN="${REPO_DIR}/build/bin"
BUILD_LIB="${REPO_DIR}/build/lib"

if [[ -d "${BUILD_BIN}" ]]; then
    find "${BUILD_BIN}" -maxdepth 1 -type f -executable -exec cp -v {} "${STAGE_DIR}/bin/" \;
fi

if [[ -d "${BUILD_LIB}" ]]; then
    find "${BUILD_LIB}" -maxdepth 1 -type f \( -name '*.a' -o -name '*.so*' \) -exec cp -v {} "${STAGE_DIR}/lib/" \;
fi

# 2. Copy include headers and documentation
if [[ -d "${REPO_DIR}/include/sycl_ip" ]]; then
    cp -r "${REPO_DIR}/include/sycl_ip" "${STAGE_DIR}/include/"
elif [[ -d "${REPO_DIR}/include" ]]; then
    cp -r "${REPO_DIR}/include" "${STAGE_DIR}/"
fi

[[ -f "${REPO_DIR}/README.md" ]] && cp "${REPO_DIR}/README.md" "${STAGE_DIR}/"
[[ -f "${REPO_DIR}/LICENSE" ]]   && cp "${REPO_DIR}/LICENSE"   "${STAGE_DIR}/"

# 3. Locate and bundle AdaptiveCpp and CUDA runtime shared libraries
SEARCH_LIB_DIRS=(
    "/home/montecarlo/.local/share/mamba/envs/sycl/lib"
    "${HOME}/.local/share/mamba/envs/sycl/lib"
    "${CONDA_PREFIX:-}/lib"
    "/usr/local/cuda-12.6/lib64"
    "/usr/local/cuda/lib64"
    "/usr/lib/aarch64-linux-gnu"
)

RUNTIME_PATTERNS=(
    "libacpp-rt.so*"
    "libacpp-common.so*"
    "libcudart.so*"
    "libboost_context*.so*"
    "libboost_fiber*.so*"
    "libstdc++.so*"
)

for libdir in "${SEARCH_LIB_DIRS[@]}"; do
    if [[ -d "${libdir}" ]]; then
        for pat in "${RUNTIME_PATTERNS[@]}"; do
            find "${libdir}" -maxdepth 1 -name "${pat}" -exec cp -a -u {} "${STAGE_DIR}/lib/" \; 2>/dev/null || true
        done
        # AdaptiveCpp backend plugins directory
        if [[ -d "${libdir}/hipSYCL" ]]; then
            cp -a -u "${libdir}/hipSYCL"/* "${STAGE_DIR}/lib/hipSYCL/" 2>/dev/null || true
        fi
        if [[ -d "${libdir}/AdaptiveCpp" ]]; then
            cp -a -u "${libdir}/AdaptiveCpp"/* "${STAGE_DIR}/lib/hipSYCL/" 2>/dev/null || true
        fi
    fi
done

# 4. Resolve transitive runtime dependencies via ldd
for bin in "${STAGE_DIR}/bin"/* "${STAGE_DIR}/lib"/*.so* "${STAGE_DIR}/lib/hipSYCL"/*.so*; do
    if [[ -f "${bin}" && -r "${bin}" ]]; then
        ldd "${bin}" 2>/dev/null | grep '=>' | awk '{print $3}' | grep -v '^$' | while read -r dep; do
            if [[ -f "${dep}" ]]; then
                base="$(basename "${dep}")"
                if [[ "${base}" == libacpp* || "${base}" == librt-backend* || "${base}" == libcudart* || "${base}" == libboost_* ]]; then
                    cp -a -u "${dep}"* "${STAGE_DIR}/lib/" 2>/dev/null || true
                fi
            fi
        done
    fi
done

# 5. Patch RPATH on binaries and shared libraries using patchelf
if command -v patchelf &>/dev/null; then
    echo "Patching ELF RPATH..."
    for bin in "${STAGE_DIR}/bin"/*; do
        if [[ -f "${bin}" && -x "${bin}" ]]; then
            patchelf --set-rpath '$ORIGIN/../lib:$ORIGIN/../lib/hipSYCL:$ORIGIN' "${bin}" 2>/dev/null || true
        fi
    done

    for lib in "${STAGE_DIR}/lib"/*.so* "${STAGE_DIR}/lib/hipSYCL"/*.so*; do
        if [[ -f "${lib}" ]]; then
            patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib:$ORIGIN/hipSYCL' "${lib}" 2>/dev/null || true
        fi
    done
else
    echo "[WARN] patchelf not found; run.sh launcher will set LD_LIBRARY_PATH and plugin paths."
fi

# 6. Create run.sh standalone launcher
cat << 'EOF' > "${STAGE_DIR}/run.sh"
#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${DIR}/lib:${DIR}/lib/hipSYCL:${LD_LIBRARY_PATH:-}"
export ACPP_RT_PLUGIN_DIR="${DIR}/lib/hipSYCL"
export HIPSYCL_RT_PLUGIN_DIR="${DIR}/lib/hipSYCL"
export ACPP_VISIBILITY_MASK="cuda;omp"

EXE="${DIR}/bin/AmplitudeCallibrationSYCL"
if [[ ! -x "${EXE}" ]]; then
    echo "[ERROR] Executable not found at ${EXE}"
    exit 1
fi

if [[ $# -eq 0 ]]; then
    exec "${EXE}" --help
else
    exec "${EXE}" "$@"
fi
EOF

chmod +x "${STAGE_DIR}/run.sh"
chmod +x "${STAGE_DIR}/bin"/* 2>/dev/null || true

# 7. Verification test in clean subshell
echo ""
echo "=== Verifying standalone execution ==="
"${STAGE_DIR}/run.sh" --help

echo ""
echo "=== Creating tarball archive ==="
tar -czvf "${OUTPUT_TAR}" -C "${STAGE_DIR}" .

echo ""
echo "=== Staged Artifact Contents ==="
ls -lh "${OUTPUT_TAR}"
echo "[OK] Standalone ARM64 package ready: ${OUTPUT_TAR}"
