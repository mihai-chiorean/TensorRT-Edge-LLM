#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs inside the `edgellm-src` image (see Dockerfile) under `docker run`
# with no GPU. Configures and compiles the fork for Jetson Orin sm_87, builds
# the pybind module, and lays out two output trees plus a machine-readable
# record. No produced binary is executed: the container has no driver, and
# the outputs are only meaningful on the device.
#
# Mounts: /out (builder/ and runtime/ trees), /record (logs and JSON).
# Env: JOBS (make -j), HOST_UID/HOST_GID (ownership of the outputs),
#      EXPECTED_SOURCE_TREE_SHA256 (optional bundle-contract digest).
set -euo pipefail

SRC=/src/TensorRT-Edge-LLM
BUILD=${SRC}/build-jp721
VENV=/opt/venv
OUT=/out
REC=/record
LOGS=${REC}/logs
JOBS=${JOBS:-8}
CUDA_CTK_VERSION=13.2
CUDA_TARGET_LIB=/usr/local/cuda/targets/sbsa-linux/lib

mkdir -p "${LOGS}" "${REC}/phases" "${OUT}/builder" "${OUT}/runtime"

phase_json() {
    local name=$1 start=$2 end=$3 rc=$4 timelog=$5
    local peak="unknown"
    [ -r /sys/fs/cgroup/memory.peak ] && peak=$(cat /sys/fs/cgroup/memory.peak)
    local maxrss
    maxrss=$(awk -F': ' '/Maximum resident set size/{print $2}' "${timelog}" 2>/dev/null || echo "")
    printf '{"phase":"%s","start_epoch":%s,"end_epoch":%s,"wall_seconds":%s,"exit_code":%s,"max_rss_kib_single_process":"%s","cgroup_memory_peak_bytes_cumulative":"%s"}\n' \
        "${name}" "${start}" "${end}" "$((end - start))" "${rc}" "${maxrss}" "${peak}" \
        > "${REC}/phases/${name}.json"
}

# phase NAME FUNCTION: re-invokes this script to run FUNCTION under
# /usr/bin/time -v (functions cannot be exec'd), logging to $LOGS/NAME.log.
phase() {
    local name=$1 fn=$2
    local start end rc=0
    echo "=== phase ${name}: $(date -u +%FT%TZ) ==="
    start=$(date +%s)
    /usr/bin/time -v -o "${LOGS}/${name}.time" bash "$0" --run "${fn}" \
        > "${LOGS}/${name}.log" 2>&1 || rc=$?
    end=$(date +%s)
    phase_json "${name}" "${start}" "${end}" "${rc}" "${LOGS}/${name}.time"
    echo "=== phase ${name}: rc=${rc} wall=$((end - start))s ==="
    if [ "${rc}" -ne 0 ]; then
        echo "phase ${name} failed; last 60 log lines:" >&2
        tail -n 60 "${LOGS}/${name}.log" >&2
        exit "${rc}"
    fi
}

# ---------------------------------------------------------------------------
preflight() {
    set -x
    : "no GPU device nodes, no driver, no NVIDIA container runtime hooks"
    ! ls /dev/nvidia* 2>/dev/null
    ! command -v nvidia-smi
    ! ldconfig -p | grep -q 'libcuda\.so'
    test -z "${NVIDIA_VISIBLE_DEVICES:-}"
    test -z "${NVIDIA_DRIVER_CAPABILITIES:-}"
    ls -la /dev
    cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null || true
    nproc
    uname -a
    cat /etc/os-release
    gcc --version; g++ --version; cmake --version; git --version
    /usr/bin/aarch64-linux-gnu-gcc --version
    /usr/local/cuda/bin/nvcc --version
    python3 --version; ${VENV}/bin/python --version
    ${VENV}/bin/pip freeze --all
    cat /etc/apt/sources.list.d/nvidia-jetson-*.list
    cat /etc/jetson-toolchain-pins.txt
    dpkg-query -W -f='${binary:Package}\t${Version}\t${Architecture}\n' \
        | grep -E '^(cuda-|libcudnn|libcurand|libcublas|libnv|tensorrt|nvidia-)' | sort
    readlink -f /usr/local/cuda
    cd "${SRC}"
    git rev-parse HEAD
    git submodule status --recursive
    test -f kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_87_cuda13.tar.gz
    sha256sum kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_87_cuda13.tar.gz
    set +x
}

configure() {
    mkdir -p "${BUILD}"
    cd "${BUILD}"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR=/usr \
        -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
        -DEMBEDDED_TARGET=jetson-orin \
        -DCUDA_CTK_VERSION=${CUDA_CTK_VERSION} \
        -DENABLE_CUTE_DSL=ALL \
        -DCMAKE_CUDA_ARCHITECTURES=87
}

check_configure() {
    set -x
    local cache=${BUILD}/CMakeCache.txt
    grep -E '^CMAKE_CUDA_ARCHITECTURES:[A-Z]+=87$' "${cache}"
    grep -E '^EMBEDDED_TARGET:[^=]+=jetson-orin$' "${cache}"
    grep -E '^CUDA_CTK_VERSION:[^=]+=13\.2$' "${cache}"
    grep -E '^CMAKE_BUILD_TYPE:STRING=Release$' "${cache}"
    grep -E "^CMAKE_HOME_DIRECTORY:INTERNAL=${SRC}\$" "${cache}"
    local cudainfo
    cudainfo=$(ls "${BUILD}"/CMakeFiles/*/CMakeCUDACompiler.cmake)
    grep -F 'set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")' "${cudainfo}"
    grep -F 'set(CMAKE_CUDA_COMPILER_VERSION "13.2.86")' "${cudainfo}"
    grep -F 'set(CMAKE_CUDA_ARCHITECTURES_NATIVE "")' "${cudainfo}"
    grep -E '^TensorRT_LIBRARY:[^=]+=/usr/lib/aarch64-linux-gnu/libnvinfer\.so$' "${cache}"
    grep -E "^CUDART_LIB:[^=]+=${CUDA_TARGET_LIB}/libcudart\.so\$" "${cache}"
    grep -E "^NVRTC_DYNAMIC_LIB:[^=]+=${CUDA_TARGET_LIB}/libnvrtc\.so\$" "${cache}"
    : "no native-architecture detection and no device probe ran"
    ! grep -iE 'ARCHITECTURES_NATIVE|=native' "${cache}"
    ! grep -iE 'nvidia-smi|detected GPU|Autodetected CUDA architecture' "${LOGS}/configure.log"
    grep -E 'CuTe DSL: (extracting|arch=)' "${LOGS}/configure.log"
    grep -E 'NVRTC JIT ENABLED' "${LOGS}/configure.log"
    grep -E 'Found TensorRT' "${LOGS}/configure.log"
    set +x
}

build() {
    cd "${BUILD}"
    make -j"${JOBS}"
}

configure_pybind() {
    cd "${BUILD}"
    local pb
    pb=$(${VENV}/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())")
    cmake .. \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DPython_EXECUTABLE=${VENV}/bin/python \
        -DPython3_EXECUTABLE=${VENV}/bin/python \
        -Dpybind11_DIR="${pb}"
}

build_pybind() {
    cd "${BUILD}"
    make -j"${JOBS}" _edgellm_runtime
}

collect() {
    set -x
    local plugin_real
    plugin_real=$(readlink -f "${BUILD}/libNvInfer_edgellm_plugin.so")
    test -f "${plugin_real}"
    local pybind_so=${BUILD}/pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so
    test -f "${pybind_so}"

    # builder/: engine-builder tools plus the plugin with its soname chain.
    rm -rf "${OUT}/builder" "${OUT}/runtime"
    install -d -m 0755 "${OUT}/builder/lib" "${OUT}/builder/bin"
    cp -a "${BUILD}"/libNvInfer_edgellm_plugin.so* "${OUT}/builder/lib/"
    install -m 0755 "${BUILD}/examples/llm/llm_build" "${OUT}/builder/bin/llm_build"
    install -m 0755 "${BUILD}/examples/multimodal/visual_build" "${OUT}/builder/bin/visual_build"

    # runtime/: what the experimental server loads natively, laid out the way
    # edge-conversation's create-bundle.sh stages its `runtime/` tree.
    install -d -m 0755 "${OUT}/runtime/plugin" "${OUT}/runtime/pybind" \
        "${OUT}/runtime/source" "${OUT}/runtime/wheelhouse"
    install -m 0644 "${plugin_real}" "${OUT}/runtime/plugin/libNvInfer_edgellm_plugin.so"
    install -m 0644 "${pybind_so}" "${OUT}/runtime/pybind/$(basename "${pybind_so}")"
    (cd "${SRC}" && git archive --format=tar HEAD experimental tensorrt_edgellm LICENSE) \
        | tar -x -C "${OUT}/runtime/source"
    find "${OUT}/runtime/source" -type d -exec chmod 0755 {} +
    find "${OUT}/runtime/source" -type f -exec chmod 0644 {} +
    (cd "${SRC}" && git rev-parse HEAD) > "${OUT}/runtime/REVISION"
    ${VENV}/bin/python -m pip freeze --all --exclude-editable > "${OUT}/runtime/requirements.lock"
    ${VENV}/bin/python -m pip download --disable-pip-version-check \
        --only-binary=:all: --no-deps \
        --requirement "${OUT}/runtime/requirements.lock" \
        --dest "${OUT}/runtime/wheelhouse"

    for tree in builder runtime; do
        (cd "${OUT}/${tree}" && find . -type f ! -name SHA256SUMS -print0 \
            | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS)
        (cd "${OUT}/${tree}" && find . \( -type f -o -type l \) -printf '%y %10s %p -> %l\n' \
            | LC_ALL=C sort -k3 > "${REC}/${tree}-files.txt")
    done
    set +x
}

verify_outputs() {
    set -x
    # Embedded GPU code: every ELF/PTX must target sm_87/compute_87 only.
    local f
    for f in "${OUT}/builder/lib/libNvInfer_edgellm_plugin.so.1.0" \
             "${OUT}/builder/bin/llm_build" "${OUT}/builder/bin/visual_build" \
             "${OUT}/runtime/pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so"; do
        echo "## cuobjdump ${f}"
        # cuobjdump exits non-zero on host-only executables (llm_build,
        # visual_build carry no device code); only the sm list matters.
        { /usr/local/cuda/bin/cuobjdump --list-elf "${f}" 2>&1 || true; } \
            | { grep -oE 'sm_[0-9]+a?' || true; } | sort | uniq -c
        { /usr/local/cuda/bin/cuobjdump --list-ptx "${f}" 2>&1 || true; } \
            | { grep -oE 'compute_[0-9]+a?' || true; } | sort | uniq -c
        if { /usr/local/cuda/bin/cuobjdump --list-elf "${f}" 2>/dev/null || true; } \
                | grep -oE 'sm_[0-9]+a?' | grep -qv '^sm_87$'; then
            echo "unexpected SM target in ${f}" >&2; return 1
        fi
    done
    # Architecture flags actually passed to nvcc, from the generated makefiles.
    grep -rhoE 'arch=compute_[0-9]+,code=[^ "]+' "${BUILD}" --include=flags.make | sort | uniq -c
    if grep -rhoE 'arch=compute_[0-9]+' "${BUILD}" --include=flags.make | grep -qv 'compute_87'; then
        echo "unexpected compute target in nvcc flags" >&2; return 1
    fi
    # Dynamic dependencies, resolved without executing anything.
    python3 /opt/edgellm/elf-needed.py "${OUT}/builder" "${OUT}/runtime" \
        --cuda-lib-dir "${CUDA_TARGET_LIB}" \
        --json "${REC}/elf-needed.json" --text "${REC}/elf-needed.txt"
    cat "${REC}/elf-needed.txt"
    # The bundle contract's digest of the Python source surface.
    python3 - "${OUT}/runtime/source" "${EXPECTED_SOURCE_TREE_SHA256:-}" > "${REC}/source-tree-digest.txt" <<'PY'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1]); expected = sys.argv[2]
h = hashlib.sha256(); h.update(b"edge-conversation-trt-source-tree-v1\0")
files = []
for d in ("experimental", "tensorrt_edgellm"):
    for p in (root / d).rglob("*"):
        rel = p.relative_to(root)
        if "__pycache__" in rel.parts or p.suffix in {".pyc", ".pyo"}:
            continue
        if p.is_file():
            files.append((rel.as_posix(), p))
for rel, p in sorted(files):
    e = rel.encode(); h.update(len(e).to_bytes(4, "little")); h.update(e)
    h.update(p.stat().st_size.to_bytes(8, "little")); h.update(p.read_bytes())
d = h.hexdigest()
print(f"source_tree_sha256={d}")
print(f"expected={expected or 'n/a'}")
print(f"match={'yes' if expected and d == expected else ('no' if expected else 'n/a')}")
PY
    cat "${REC}/source-tree-digest.txt"
    set +x
}

record() {
    set -x
    cd "${SRC}"
    local cache=${BUILD}/CMakeCache.txt
    grep -E '^(CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CMAKE_CUDA_HOST_COMPILER|CMAKE_CXX_COMPILER|CMAKE_C_COMPILER|CMAKE_BUILD_TYPE|CMAKE_TOOLCHAIN_FILE|CMAKE_HOME_DIRECTORY|EMBEDDED_TARGET|CUDA_CTK_VERSION|CUDA_DIR|CUDART_LIB|CUDA_DRIVER_LIB|CUDA_RUNTIME_API_INCLUDE_DIR|CURAND_KERNEL_INCLUDE_DIR|NVRTC_DYNAMIC_LIB|NVRTC_INCLUDE_DIR|NVRTC_LIB|TensorRT_LIBRARY|TensorRT_INCLUDE_DIR|TensorRT_OnnxParser_LIBRARY|TensorRT_OnnxParser_INCLUDE_DIR|TRT_PACKAGE_DIR|ENABLE_CUTE_DSL|CUTE_DSL_ARTIFACT_TAG|BUILD_PYTHON_BINDINGS|Python_EXECUTABLE|Python3_EXECUTABLE|pybind11_DIR|BUILD_UNIT_TESTS|ENABLE_NVTX_PROFILING):' "${cache}" \
        | sort > "${REC}/cmake-cache-selected.txt"
    cp "${cache}" "${REC}/CMakeCache.txt"
    cp "${BUILD}"/CMakeFiles/*/CMakeCUDACompiler.cmake "${REC}/CMakeCUDACompiler.cmake"
    grep -rhoE 'arch=compute_[0-9]+,code=[^ "]+' "${BUILD}" --include=flags.make | sort | uniq -c \
        > "${REC}/nvcc-gencode-flags.txt"
    cp "${LOGS}/configure.log" "${REC}/cmake-configure.log"
    dpkg-query -W -f='${binary:Package}\t${Version}\t${Architecture}\n' \
        | grep -E '^(cuda-|libcudnn|libcurand|libcublas|libnv|tensorrt|nvidia-)' | sort > "${REC}/dpkg-nvidia.txt"
    dpkg-query -W -f='${binary:Package}\t${Version}\n' gcc g++ cpp binutils cmake make git python3 libc6 libstdc++6 \
        | sort > "${REC}/dpkg-toolchain.txt"
    cp /etc/apt/sources.list.d/nvidia-jetson-*.list "${REC}/apt-source.list"
    cp /etc/jetson-toolchain-pins.txt "${REC}/apt-pins.txt"
    git rev-parse HEAD > "${REC}/fork-commit.txt"
    git submodule status --recursive > "${REC}/submodules.txt"
    ${VENV}/bin/pip freeze --all > "${REC}/venv-freeze.txt"
    [ -r /sys/fs/cgroup/memory.peak ] && cat /sys/fs/cgroup/memory.peak > "${REC}/cgroup-memory-peak-bytes.txt"
    [ -r /sys/fs/cgroup/memory.max ] && cat /sys/fs/cgroup/memory.max > "${REC}/cgroup-memory-max.txt"
    [ -r /sys/fs/cgroup/cpu.max ] && cat /sys/fs/cgroup/cpu.max > "${REC}/cgroup-cpu-max.txt"
    python3 - "${REC}" <<'PY'
import json, sys, subprocess, pathlib
rec = pathlib.Path(sys.argv[1])
def v(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
out = {
    "container": {
        "uname": v("uname -srm"),
        "os_release": v("sed -n 's/^PRETTY_NAME=//p' /etc/os-release").strip('"'),
        "nproc": v("nproc"),
        "cgroup_cpu_max": v("cat /sys/fs/cgroup/cpu.max 2>/dev/null"),
        "cgroup_memory_max": v("cat /sys/fs/cgroup/memory.max 2>/dev/null"),
        "cgroup_memory_peak_bytes": v("cat /sys/fs/cgroup/memory.peak 2>/dev/null"),
        "dev_nvidia_nodes": v("ls /dev/nvidia* 2>/dev/null"),
    },
    "versions": {
        "gcc": v("gcc -dumpfullversion"),
        "gxx": v("g++ -dumpfullversion"),
        "cmake": v("cmake --version | head -1 | awk '{print $3}'"),
        "nvcc": v("/usr/local/cuda/bin/nvcc --version | grep -oE 'V[0-9.]+' | head -1"),
        "python_system": v("python3 --version | awk '{print $2}'"),
        "python_venv": v("/opt/venv/bin/python --version | awk '{print $2}'"),
        "git": v("git --version | awk '{print $3}'"),
        "binutils_ld": v("ld --version | head -1"),
    },
    "phases": [json.load(open(p)) for p in sorted(rec.glob("phases/*.json"))],
    "apt_source_line": (rec / "apt-source.list").read_text().strip(),
    "apt_pins": (rec / "apt-pins.txt").read_text().split(),
    "dpkg_nvidia": [dict(zip(("package", "version", "arch"), l.split("\t")))
                    for l in (rec / "dpkg-nvidia.txt").read_text().splitlines()],
    "dpkg_toolchain": [dict(zip(("package", "version"), l.split("\t")))
                       for l in (rec / "dpkg-toolchain.txt").read_text().splitlines()],
    "fork_commit": (rec / "fork-commit.txt").read_text().strip(),
    "submodules": [l.strip() for l in (rec / "submodules.txt").read_text().splitlines()],
    "cmake_cache_selected": dict(l.split("=", 1) for l in (rec / "cmake-cache-selected.txt").read_text().splitlines()),
    "source_tree_digest": dict(l.split("=", 1) for l in (rec / "source-tree-digest.txt").read_text().splitlines()),
    "venv_freeze": (rec / "venv-freeze.txt").read_text().split(),
}
json.dump(out, open(rec / "container-record.json", "w"), indent=2)
PY
    set +x
}

main() {
    phase preflight preflight
    phase configure configure
    phase check-configure check_configure
    phase build build
    phase configure-pybind configure_pybind
    phase build-pybind build_pybind
    phase collect collect
    phase verify-outputs verify_outputs
    phase record record
    if [ -n "${HOST_UID:-}" ]; then
        chown -R "${HOST_UID}:${HOST_GID:-${HOST_UID}}" "${OUT}" "${REC}"
    fi
    echo "BUILD OK"
}

if [ "${1:-}" = "--run" ]; then
    "$2"
    exit $?
fi
main
