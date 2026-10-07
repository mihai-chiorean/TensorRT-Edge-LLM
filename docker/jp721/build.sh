#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Host-side driver: builds the JetPack 7.2.1 toolchain image, then compiles
# the fork for Jetson Orin sm_87 in a CPU-only container with cgroup limits.
# Produces $ROOT/out/{builder,runtime} and $ROOT/record (logs, manifests,
# container-record.json, docker-*.json). Run it detached on an arm64 Docker
# host with no GPU passthrough; it never passes --gpus or an NVIDIA runtime.
#
#   ROOT=~/trt-edge-build CPUS=10 MEMORY=14g JOBS=8 ./build.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-${HOME}/trt-edge-build}
IMAGE=${IMAGE:-trt-edge-llm/jp721-build:cda4fc1}
CONTAINER=${CONTAINER:-trt-edge-build-jp721}
CPUS=${CPUS:-10}
MEMORY=${MEMORY:-14g}
JOBS=${JOBS:-8}
EXPECTED_SOURCE_TREE_SHA256=${EXPECTED_SOURCE_TREE_SHA256:-}

REC=${ROOT}/record
mkdir -p "${ROOT}/out" "${REC}/logs"

[ "$(uname -m)" = aarch64 ] || { echo "build host must be arm64" >&2; exit 1; }

echo "=== docker build: $(date -u +%FT%TZ)"
t0=$(date +%s)
docker build --progress=plain --target edgellm-src -t "${IMAGE}" \
    -f "${HERE}/Dockerfile" "${HERE}" 2>&1 | tee "${REC}/logs/docker-build.log"
t1=$(date +%s)
echo "=== docker build done in $((t1 - t0))s"

sha256sum "${HERE}/Dockerfile" "${HERE}/build-edgellm.sh" "${HERE}/elf-needed.py" \
    "${HERE}/build.sh" > "${REC}/docker-inputs.sha256"
docker image inspect "${IMAGE}" > "${REC}/docker-image.json"
docker image inspect ubuntu:24.04 > "${REC}/docker-base-image.json" 2>/dev/null || true
docker version > "${REC}/docker-version.txt"
printf '{"image_build_wall_seconds":%s,"image":"%s"}\n' "$((t1 - t0))" "${IMAGE}" \
    > "${REC}/docker-build-timing.json"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

# Memory sampler: docker stats every 10 s while the build container runs.
(
    sleep 20
    while [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" = true ]; do
        docker stats --no-stream --format '{{.Name}} {{.MemUsage}} {{.CPUPerc}}' "${CONTAINER}" 2>/dev/null \
            | sed "s/^/$(date -u +%FT%TZ) /" >> "${REC}/logs/docker-stats.log" || true
        sleep 10
    done
) &
SAMPLER=$!

echo "=== docker run: $(date -u +%FT%TZ)"
t2=$(date +%s)
rc=0
docker run --name "${CONTAINER}" \
    --cpus "${CPUS}" --memory "${MEMORY}" --memory-swap "${MEMORY}" \
    --security-opt no-new-privileges \
    -e JOBS="${JOBS}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -e EXPECTED_SOURCE_TREE_SHA256="${EXPECTED_SOURCE_TREE_SHA256}" \
    -v "${ROOT}/out:/out" -v "${REC}:/record" \
    "${IMAGE}" /opt/edgellm/build-edgellm.sh 2>&1 | tee "${REC}/logs/docker-run.log" || rc=${PIPESTATUS[0]}
t3=$(date +%s)
echo "=== docker run done in $((t3 - t2))s rc=${rc}"

docker container inspect "${CONTAINER}" > "${REC}/docker-container.json"
kill "${SAMPLER}" 2>/dev/null || true
printf '{"container_run_wall_seconds":%s,"exit_code":%s,"cpus":"%s","memory":"%s","jobs":%s}\n' \
    "$((t3 - t2))" "${rc}" "${CPUS}" "${MEMORY}" "${JOBS}" > "${REC}/docker-run-timing.json"
docker rm "${CONTAINER}" >/dev/null
exit "${rc}"
