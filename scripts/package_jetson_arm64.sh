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
mkdir -p "${STAGE_DIR}/bin" "${STAGE_DIR}/lib" "${STAGE_DIR}/include" "${REPO_DIR}/artifacts"

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

# 3. Locate the toolchain prefix providing AdaptiveCpp and the CUDA runtime
TOOLCHAIN_LIB=""
for candidate in "${CONDA_PREFIX:-/nonexistent}/lib" \
                 "${HOME}/.local/share/mamba/envs/sycl/lib" \
                 "/home/montecarlo/.local/share/mamba/envs/sycl/lib"; do
    if [[ -d "${candidate}" ]]; then
        if [[ -e "${candidate}/libacpp-rt.so" || -d "${candidate}/hipSYCL" || -d "${candidate}/AdaptiveCpp" ]]; then
            TOOLCHAIN_LIB="${candidate}"
            break
        fi
    fi
done

CUDA_LIBDIR=""
for candidate in "${CUDA_HOME:-/nonexistent}/lib64" "/usr/local/cuda-12.6/lib64" "/usr/local/cuda/lib64"; do
    if [[ -d "${candidate}" ]]; then
        CUDA_LIBDIR="${candidate}"
        break
    fi
done

APT_LIB="/usr/lib/aarch64-linux-gnu"

echo "AdaptiveCpp toolchain lib: ${TOOLCHAIN_LIB:-<not found>}"
echo "CUDA lib directory:        ${CUDA_LIBDIR:-<not found>}"

if [[ -z "${TOOLCHAIN_LIB}" ]]; then
    echo "[ERROR] Could not find the AdaptiveCpp toolchain (acpp) - activate the SYCL environment first." >&2
    exit 1
fi

# 3a. Copy AdaptiveCpp runtime, backend plugins, JIT bitcode and Boost/libstdc++ runtimes.
#     Symlinks are dereferenced (-L) so the package never points back into the toolchain.
for pattern in 'libacpp*' 'librt-backend*' 'libllvm-to-*' 'libcudart.so*' \
               'libboost_context*' 'libboost_fiber*' 'libstdc++.so*' 'libgcc_s.so*'; do
    find "${TOOLCHAIN_LIB}" -maxdepth 1 -name "${pattern}" -exec cp -aL {} "${STAGE_DIR}/lib/" \; 2>/dev/null || true
done
for plugindir in "${TOOLCHAIN_LIB}/hipSYCL" "${TOOLCHAIN_LIB}/AdaptiveCpp"; do
    if [[ -d "${plugindir}" ]]; then
        mkdir -p "${STAGE_DIR}/lib/hipSYCL"
        cp -aL "${plugindir}"/* "${STAGE_DIR}/lib/hipSYCL/" 2>/dev/null || true
    fi
done

# 3c. LLVM tools for AdaptiveCpp's runtime JIT. The runtime resolves opt/llc relative to
#     libacpp-common.so, i.e. as $BUNDLE/lib/hipSYCL/ext/llvm/bin/<tool>, and otherwise falls back
#     to the (non-relocatable) path of the build machine. Without them a "generic" target binary
#     fails at runtime with "LLVMToPtx: opt invocation failed".
TOOLCHAIN_BIN="$(dirname "${TOOLCHAIN_LIB}")/bin"
LLVM_TOOL_BIN="${STAGE_DIR}/lib/hipSYCL/ext/llvm/bin"
mkdir -p "${LLVM_TOOL_BIN}"
for pattern in 'opt' 'opt-[0-9]*' 'llc' 'llc-[0-9]*' 'lld' 'lld-[0-9]*' 'ld.lld' 'ld.lld-[0-9]*'; do
    find "${TOOLCHAIN_BIN}" -maxdepth 1 -name "${pattern}" -exec cp -aL {} "${LLVM_TOOL_BIN}/" \; 2>/dev/null || true
done
if ls "${LLVM_TOOL_BIN}"/opt* >/dev/null 2>&1; then
    echo "Staged JIT tools: $(ls "${LLVM_TOOL_BIN}" | tr '\n' ' ')"
else
    echo "[WARN] No LLVM 'opt'/'llc' found in ${TOOLCHAIN_BIN}; runtime JIT of kernels will not work." >&2
    echo "       Install the matching 'llvm-tools' package into the SYCL environment." >&2
fi

# 3b. CUDA runtime and Boost from the system (Jetson provides them outside the SYCL env)
if [[ -n "${CUDA_LIBDIR}" ]]; then
    for pattern in 'libcudart.so*' 'libnvrtc.so*'; do
        find "${CUDA_LIBDIR}" -maxdepth 1 -name "${pattern}" -exec cp -aL {} "${STAGE_DIR}/lib/" \; 2>/dev/null || true
    done
fi
if [[ -d "${APT_LIB}" ]]; then
    find "${APT_LIB}" -maxdepth 1 \( -name 'libboost_context*.so*' -o -name 'libboost_fiber*.so*' \) \
        -exec cp -aL {} "${STAGE_DIR}/lib/" \; 2>/dev/null || true
fi

# 4. Stage every runtime library the packaged objects need and the toolchain provides.
#    ldd is deliberately not used: it resolves NEEDED entries through the system paths, so a
#    missing bundled copy silently resolves to /usr/lib and the package then fails on a clean
#    machine with "No CPU backend has been loaded. Terminating." (libLLVM, libnuma, libgomp...).
#    GPU driver libraries of the target machine are never staged.
EXTERNAL_LIBS="libcuda.so.1 libnvidia-ml.so.1 libamdhip64.so.6 libhiprtc.so.6 libOpenCL.so.1"

DEP_SOURCE_DIRS=("${TOOLCHAIN_LIB}")
if [[ -n "${CUDA_LIBDIR}" ]]; then
    DEP_SOURCE_DIRS+=("${CUDA_LIBDIR}")
fi

if ! command -v readelf >/dev/null 2>&1; then
    echo "[ERROR] readelf is required to analyse package dependencies (apt-get install binutils)." >&2
    exit 1
fi

needed_of() {
    readelf -d "$1" 2>/dev/null | sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p'
}

staged_elfs() {
    find "${STAGE_DIR}/bin" "${STAGE_DIR}/lib" -type f \( -name '*.so*' -o -perm -u+x \) 2>/dev/null
}

in_bundle() {
    [[ -e "${STAGE_DIR}/lib/$1" || -e "${STAGE_DIR}/lib/hipSYCL/$1" || -e "${STAGE_DIR}/lib/hipSYCL/llvm-to-backend/$1" ]]
}

is_external() {
    [[ " ${EXTERNAL_LIBS} " == *" $1 "* ]]
}

find_in_toolchain() {
    local soname="$1" dir hit
    for dir in "${DEP_SOURCE_DIRS[@]}"; do
        hit="$(find "${dir}" -maxdepth 1 -name "${soname}" -print -quit 2>/dev/null || true)"
        if [[ -n "${hit}" ]]; then
            echo "${hit}"
            return 0
        fi
    done
    return 0
}

for _pass in 1 2 3 4 5 6 7 8; do
    added=0
    while IFS= read -r elf; do
        while IFS= read -r soname; do
            if [[ -z "${soname}" ]]; then
                continue
            fi
            if in_bundle "${soname}" || is_external "${soname}"; then
                continue
            fi
            src="$(find_in_toolchain "${soname}")"
            if [[ -n "${src}" ]]; then
                cp -aL "${src}" "${STAGE_DIR}/lib/"
                echo "  [deps] staged ${soname} (needed by $(basename "${elf}"))"
                added=$((added + 1))
            fi
        done < <(needed_of "${elf}")
    done < <(staged_elfs)
    if [[ "${added}" -eq 0 ]]; then
        break
    fi
done

# 5. Patch RPATH on binaries and shared libraries using patchelf.
#    The paths cover every lib level: a single directory broke the plugin ->
#    libllvm-to-*.so lookup (they live in lib/hipSYCL/llvm-to-backend).
if command -v patchelf &>/dev/null; then
    echo "Patching ELF RPATH..."
    RPATH_LIB='$ORIGIN:$ORIGIN/..:$ORIGIN/../..:$ORIGIN/../../..:$ORIGIN/../../../..:$ORIGIN/llvm-to-backend:$ORIGIN/../llvm-to-backend:$ORIGIN/../../llvm-to-backend'
    RPATH_BIN='$ORIGIN/../lib:$ORIGIN/../lib/hipSYCL:$ORIGIN/../lib/hipSYCL/llvm-to-backend:$ORIGIN'

    for bin in "${STAGE_DIR}/bin"/*; do
        if [[ -f "${bin}" && -x "${bin}" ]]; then
            patchelf --set-rpath "${RPATH_BIN}" "${bin}" 2>/dev/null || true
        fi
    done

    # Shared objects and the LLVM JIT tools (executables under lib/hipSYCL/ext/llvm/bin)
    while IFS= read -r -d '' elf; do
        # Only real ELF objects: patchelf chokes on linker scripts such as conda's libstdc++.so
        if head -c 4 "${elf}" | grep -q ELF; then
            patchelf --set-rpath "${RPATH_LIB}" "${elf}" 2>/dev/null || true
        fi
    done < <(find "${STAGE_DIR}/lib" -type f \( -name '*.so*' -o -perm -u+x \) -print0)
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

# 7. Verify that every NEEDED library is bundled, provided by the GPU driver, or is a plain
#    system library the toolchain cannot supply (glibc, dynamic loader).
failures=0
while IFS= read -r elf; do
    while IFS= read -r soname; do
        if [[ -z "${soname}" ]]; then
            continue
        fi
        if in_bundle "${soname}" || is_external "${soname}"; then
            continue
        fi
        if [[ -z "$(find_in_toolchain "${soname}")" ]]; then
            continue
        fi
        echo "MISSING: $(basename "${elf}") -> ${soname}"
        failures=$((failures + 1))
    done < <(needed_of "${elf}")
done < <(staged_elfs)

if [[ "${failures}" -ne 0 ]]; then
    echo "[ERROR] Staged package is missing runtime libraries (see MISSING lines above)." >&2
    exit 1
fi

echo ""
echo "Staged backends: $(cd "${STAGE_DIR}/lib/hipSYCL" 2>/dev/null && ls librt-backend-*.so | tr '\n' ' ')"
echo "Driver libraries expected on the target machine: ${EXTERNAL_LIBS}"

# 8. Verification test in clean subshell
echo ""
echo "=== Verifying standalone execution ==="
"${STAGE_DIR}/run.sh" --help

if [[ "${RUN_SMOKE_TEST:-1}" == "1" ]]; then
    echo ""
    echo "=== Running CPU validation suite from the staged package ==="
    "${STAGE_DIR}/run.sh" --test --device cpu
fi

echo ""
echo "=== Creating tarball archive ==="
tar -czvf "${OUTPUT_TAR}" -C "${STAGE_DIR}" .

echo ""
echo "=== Staged Artifact Contents ==="
ls -lh "${OUTPUT_TAR}"
echo "[OK] Standalone ARM64 package ready: ${OUTPUT_TAR}"
