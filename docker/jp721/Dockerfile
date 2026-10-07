# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Off-device compile environment for Jetson Orin (JetPack 7.2.1 / L4T R39.2,
# CUDA 13.2, TensorRT 10.16, cuDNN 9.20, sm_87) on any arm64 Docker host.
#
# Stage `jetson-toolchain` is generic: Ubuntu 24.04 arm64 plus the JetPack
# 7.2.1 developer packages from NVIDIA's Jetson apt repository, pinned to
# exact versions and verified after installation. It carries no GPU driver;
# the CUDA driver is linked through the toolkit's stub, so the image must be
# used for compiling only, never with `--gpus` or the nvidia runtime.
#
# Stage `edgellm-src` is specific to TensorRT Edge-LLM: it clones the fork at
# a pinned commit and prepares the Python 3.12 venv the pybind module links
# against. The compile itself runs in `docker run` (see build.sh), so cgroup
# limits, wall-clock time and peak memory can be recorded per phase.

ARG BASE_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

# JetPack 7.2.1 component pins (nvidia-jetpack 7.2.1-b49 in the r39.2 pocket).
# The same pocket also carries the JetPack 7.2 revisions (CUDA 13.2.75/.78,
# nvidia-jetpack 7.2-b184/b187); the pins below select 7.2.1 explicitly.
ARG JETSON_REPO_URL=https://repo.download.nvidia.com/jetson/common
ARG JETSON_REPO_SUITE=r39.2
ARG JETSON_REPO_KEY_URL=https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
ARG JETSON_REPO_KEY_FINGERPRINT=3C6D1FF3100C8C3ABB0869C0E6543461A9996195
ARG CUDA_VERSION_MAJOR_MINOR=13.2
ARG CUDA_PKG_VERSION=13.2.86-1
ARG CURAND_PKG_VERSION=10.4.2.66-1
ARG TENSORRT_PKG_VERSION=10.16.2.10-1+cuda13.2
ARG CUDNN_PKG_VERSION=9.20.0.46-1

# ---------------------------------------------------------------------------
# Stage 1: generic Jetson JetPack 7.2.1 toolchain
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS jetson-toolchain
ARG JETSON_REPO_URL
ARG JETSON_REPO_SUITE
ARG JETSON_REPO_KEY_URL
ARG JETSON_REPO_KEY_FINGERPRINT
ARG CUDA_VERSION_MAJOR_MINOR
ARG CUDA_PKG_VERSION
ARG CURAND_PKG_VERSION
ARG TENSORRT_PKG_VERSION
ARG CUDNN_PKG_VERSION

ENV DEBIAN_FRONTEND=noninteractive

# Host build tools. `gcc`/`g++` on arm64 also provide the
# /usr/bin/aarch64-linux-gnu-{gcc,g++} names that the project's toolchain file
# hard-codes; `time` provides /usr/bin/time for per-phase RSS accounting.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
        build-essential cmake git pkg-config \
        python3 python3-venv python3-dev python3-pip \
        binutils file time xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && test -x /usr/bin/aarch64-linux-gnu-gcc \
    && test -x /usr/bin/aarch64-linux-gnu-g++

# NVIDIA Jetson apt source. The OTA key is verified by fingerprint before it is
# trusted; it is scoped to this one source with signed-by rather than added to
# the global trust store. A flashed JetPack device gets the same key and the
# same `common` line (plus a `t234` line for the L4T BSP, which is not needed
# to compile) from the nvidia-l4t-apt-source package.
RUN set -eu; \
    curl -fsSL --retry 3 -o /tmp/jetson-ota-public.asc "${JETSON_REPO_KEY_URL}"; \
    actual="$(gpg --batch --show-keys --with-fingerprint /tmp/jetson-ota-public.asc \
        | tr -d ' ' | grep -m1 -oE '^[0-9A-F]{40}$')"; \
    test "${actual}" = "${JETSON_REPO_KEY_FINGERPRINT}" \
        || { echo "Jetson OTA key fingerprint ${actual} != ${JETSON_REPO_KEY_FINGERPRINT}" >&2; exit 1; }; \
    gpg --batch --dearmor -o /usr/share/keyrings/nvidia-jetson-ota.gpg /tmp/jetson-ota-public.asc; \
    rm -f /tmp/jetson-ota-public.asc; \
    printf 'deb [arch=arm64 signed-by=/usr/share/keyrings/nvidia-jetson-ota.gpg] %s %s main\n' \
        "${JETSON_REPO_URL}" "${JETSON_REPO_SUITE}" \
        > /etc/apt/sources.list.d/nvidia-jetson-${JETSON_REPO_SUITE}.list

# JetPack developer packages, every one pinned. Only what a TensorRT plugin /
# engine-builder compile needs: the nvcc front end and its backends, the CUDA
# runtime headers and driver stub, NVRTC (Edge-LLM JIT-compiles its XQA decode
# kernels at engine-build time), cuRAND device headers, cuobjdump (to audit the
# SM targets embedded in the outputs), TensorRT with the plugin header
# (NvInferPlugin.h is in libnvinfer-headers-plugin-dev, not libnvinfer-dev;
# nothing links libnvinfer_plugin), the ONNX parser, and cuDNN.
# After installation every NVIDIA package present must be in this list at this
# version, otherwise the image build fails.
RUN set -eu; \
    V="${CUDA_PKG_VERSION}"; T="${TENSORRT_PKG_VERSION}"; D="${CUDNN_PKG_VERSION}"; \
    C="${CUDA_VERSION_MAJOR_MINOR}"; P="$(echo "${C}" | tr . -)"; \
    pins=" \
        cuda-toolkit-config-common=${V} \
        cuda-toolkit-13-config-common=${V} \
        cuda-toolkit-${P}-config-common=${V} \
        cuda-cccl-${P}=${V} \
        cuda-crt-${P}=${V} \
        cuda-cudart-${P}=${V} \
        cuda-cudart-dev-${P}=${V} \
        cuda-culibos-${P}=${V} \
        cuda-culibos-dev-${P}=${V} \
        cuda-driver-dev-${P}=${V} \
        cuda-nvcc-${P}=${V} \
        cuda-nvrtc-${P}=${V} \
        cuda-nvrtc-dev-${P}=${V} \
        cuda-cuobjdump-${P}=${V} \
        libnvvm-${P}=${V} \
        libnvptxcompiler-${P}=${V} \
        libcurand-${P}=${CURAND_PKG_VERSION} \
        libcurand-dev-${P}=${CURAND_PKG_VERSION} \
        libnvinfer10=${T} \
        libnvinfer-headers-dev=${T} \
        libnvinfer-safe-headers-dev=${T} \
        libnvinfer-headers-plugin-dev=${T} \
        libnvinfer-dev=${T} \
        libnvonnxparsers10=${T} \
        libnvonnxparsers-dev=${T} \
        libcudnn9-cuda-13=${D} \
        libcudnn9-headers-cuda-13=${D} \
        libcudnn9-dev-cuda-13=${D} \
    "; \
    apt-get update; \
    apt-get install -y --no-install-recommends ${pins}; \
    rm -rf /var/lib/apt/lists/*; \
    printf '%s\n' ${pins} | sort > /etc/jetson-toolchain-pins.txt; \
    dpkg-query -W -f='${binary:Package}=${Version}\n' \
        | grep -E '^(cuda-|libcudnn|libcurand|libcublas|libnv|tensorrt|nvidia-)' | sort > /tmp/installed-nvidia.txt; \
    if ! diff -u /etc/jetson-toolchain-pins.txt /tmp/installed-nvidia.txt; then \
        echo "installed NVIDIA packages differ from the pinned set" >&2; exit 1; fi; \
    rm -f /tmp/installed-nvidia.txt

# Layout checks: the project's aarch64 toolchain file hard-codes
# /usr/local/cuda/bin/nvcc and, for CUDA >= 13 on Orin,
# /usr/local/cuda/targets/sbsa-linux as CUDA_DIR. The driver is only a stub.
RUN set -eu; \
    C="${CUDA_VERSION_MAJOR_MINOR}"; \
    test -x /usr/local/cuda-${C}/bin/nvcc; \
    test "$(readlink -f /usr/local/cuda)" = "/usr/local/cuda-${C}"; \
    test -f /usr/local/cuda/targets/sbsa-linux/lib/libcudart.so; \
    test -f /usr/local/cuda/targets/sbsa-linux/lib/libnvrtc.so; \
    test -f /usr/local/cuda/targets/sbsa-linux/lib/stubs/libcuda.so; \
    test -f /usr/local/cuda/targets/sbsa-linux/include/curand_kernel.h; \
    test -f /usr/include/aarch64-linux-gnu/NvInfer.h; \
    test -f /usr/lib/aarch64-linux-gnu/libnvinfer.so; \
    test -f /usr/lib/aarch64-linux-gnu/libnvonnxparser.so; \
    ! ls /usr/lib/aarch64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so.1 2>/dev/null; \
    /usr/local/cuda/bin/nvcc --version | grep -F "release ${C}"

ENV PATH=/usr/local/cuda/bin:${PATH}

# ---------------------------------------------------------------------------
# Stage 2: TensorRT Edge-LLM fork source at a pinned commit + Python venv
# ---------------------------------------------------------------------------
FROM jetson-toolchain AS edgellm-src
ARG EDGELLM_REPO_URL=https://github.com/mihai-chiorean/TensorRT-Edge-LLM.git
ARG EDGELLM_COMMIT=cda4fc1821d1bd0941a755a515269ba928bba7d7
ARG EDGELLM_PYBIND11_VERSION=3.0.4

# Full clone (not shallow) so `git describe` and the upstream base commit are
# resolvable for the build record; submodules are recursive because XGrammar
# needs its nested dlpack checkout (cmake/GuidedDecoding.cmake is fatal
# without it).
RUN set -eu; \
    git clone "${EDGELLM_REPO_URL}" /src/TensorRT-Edge-LLM; \
    cd /src/TensorRT-Edge-LLM; \
    git checkout --detach "${EDGELLM_COMMIT}"; \
    test "$(git rev-parse HEAD)" = "${EDGELLM_COMMIT}"; \
    git submodule update --init --recursive --depth 1

# Runtime-only server dependencies plus pybind11 for the binding. torch and
# the export stack are deliberately absent: the server never imports them and
# their CUDA wheels would shadow the JetPack CUDA install on the device.
RUN set -eu; \
    python3 -m venv /opt/venv; \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel; \
    /opt/venv/bin/pip install --no-cache-dir \
        -r /src/TensorRT-Edge-LLM/requirements-server.txt \
        "pybind11==${EDGELLM_PYBIND11_VERSION}"

COPY build-edgellm.sh /opt/edgellm/build-edgellm.sh
COPY elf-needed.py /opt/edgellm/elf-needed.py
RUN chmod 0755 /opt/edgellm/build-edgellm.sh

WORKDIR /src/TensorRT-Edge-LLM
