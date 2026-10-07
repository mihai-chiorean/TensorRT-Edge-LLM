# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Assemble build-record.json from a build.sh record directory.

Usage: make-record.py RECORD_DIR OUT_DIR_SHA_ROOT build-record.json

RECORD_DIR is $ROOT/record as written by build.sh and build-edgellm.sh;
OUT_DIR_SHA_ROOT holds builder/SHA256SUMS and runtime/SHA256SUMS (the output
trees themselves, or a copy of just the manifests).
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM_BASE = "95515c2"  # TensorRT Edge-LLM v0.11.0


def read(path: Path) -> str:
    return path.read_text().strip() if path.exists() else ""


def manifest(tree: Path, listing: Path) -> list:
    sizes = {}
    links = {}
    for line in read(listing).splitlines():
        m = re.match(r"(\w) +(\d+) (\S+) -> (.*)", line)
        if m:
            kind, size, path, target = m.groups()
            sizes[path] = int(size)
            if kind == "l":
                links[path] = target
    entries = []
    for line in read(tree / "SHA256SUMS").splitlines():
        digest, path = line.split("  ", 1)
        entries.append({
            "path": path.lstrip("./"),
            "sha256": digest,
            "size": sizes.get(path),
        })
    for path, target in sorted(links.items()):
        entries.append({"path": path.lstrip("./"), "symlink_to": target})
    return entries


def docker_stats_peak(path: Path) -> str:
    peak = 0.0
    unit = ""
    for line in read(path).splitlines():
        m = re.search(r" (\d+(?:\.\d+)?)(GiB|MiB) / ", line)
        if m:
            value = float(m.group(1)) * (1024 if m.group(2) == "GiB" else 1)
            if value > peak:
                peak = value
                unit = "MiB"
    return f"{peak:.0f} {unit}" if unit else ""


def main() -> None:
    rec = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    target = Path(sys.argv[3])

    container = json.loads(read(rec / "container-record.json") or "{}")
    image = json.loads(read(rec / "docker-image.json") or "[]")
    base = json.loads(read(rec / "docker-base-image.json") or "[]")
    cont = json.loads(read(rec / "docker-container.json") or "[]")
    build_timing = json.loads(read(rec / "docker-build-timing.json") or "{}")
    run_timing = json.loads(read(rec / "docker-run-timing.json") or "{}")
    inputs = dict(
        reversed(line.split("  ", 1))
        for line in read(rec / "docker-inputs.sha256").splitlines())
    host_config = cont[0]["HostConfig"] if cont else {}

    record = {
        "generated_at":
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": {
            "device": "Jetson AGX Orin / Orin NX (sm_87)",
            "jetpack": "7.2.1 (nvidia-jetpack 7.2.1-b49)",
            "l4t": "R39.2",
            "cuda": "13.2 (13.2.86)",
            "tensorrt": "10.16.2.10",
            "cudnn": "9.20.0.46",
            "cuda_architectures": "87",
        },
        "build_host": {
            "hostname":
            "spark-094a",
            "arch":
            "aarch64",
            "docker":
            read(rec / "docker-version.txt").splitlines()[1].strip()
            if read(rec / "docker-version.txt") else "",
            "gpu_passthrough":
            False,
        },
        "docker": {
            "dockerfile_sha256":
            next((v for k, v in inputs.items() if k.endswith("/Dockerfile")),
                 ""),
            "build_edgellm_sh_sha256":
            next((v for k, v in inputs.items()
                  if k.endswith("/build-edgellm.sh")), ""),
            "image":
            build_timing.get("image"),
            "image_id":
            image[0]["Id"] if image else "",
            "image_created":
            image[0]["Created"] if image else "",
            "image_size_bytes":
            image[0]["Size"] if image else None,
            "base_image_repo_digest":
            base[0]["RepoDigests"][0] if base else "",
            "base_image_id":
            base[0]["Id"] if base else "",
            "image_build_wall_seconds":
            build_timing.get("image_build_wall_seconds"),
            "container_run_wall_seconds":
            run_timing.get("container_run_wall_seconds"),
            "container_exit_code":
            run_timing.get("exit_code"),
            "container_limits": {
                "cpus": run_timing.get("cpus"),
                "memory": run_timing.get("memory"),
                "make_jobs": run_timing.get("jobs"),
                "NanoCpus": host_config.get("NanoCpus"),
                "Memory": host_config.get("Memory"),
                "MemorySwap": host_config.get("MemorySwap"),
            },
            "container_isolation": {
                "Runtime": host_config.get("Runtime"),
                "DeviceRequests": host_config.get("DeviceRequests"),
                "Devices": host_config.get("Devices"),
                "Privileged": host_config.get("Privileged"),
                "env": cont[0]["Config"]["Env"] if cont else [],
            },
            "docker_stats_peak_mem":
            docker_stats_peak(rec / "logs" / "docker-stats.log"),
        },
        "apt": {
            "source_line": container.get("apt_source_line"),
            "key_url":
            "https://repo.download.nvidia.com/jetson/jetson-ota-public.asc",
            "key_fingerprint": "3C6D1FF3100C8C3ABB0869C0E6543461A9996195",
            "pins": container.get("apt_pins"),
        },
        "packages": {
            "nvidia": container.get("dpkg_nvidia"),
            "toolchain": container.get("dpkg_toolchain"),
        },
        "versions": container.get("versions"),
        "source": {
            "repo": "https://github.com/mihai-chiorean/TensorRT-Edge-LLM.git",
            "branch": "feat/jp72-sm87-0.11.0",
            "fork_commit": container.get("fork_commit"),
            "upstream_base_commit": UPSTREAM_BASE,
            "upstream_base_tag": "v0.11.0",
            "submodules": container.get("submodules"),
            "source_tree_digest": container.get("source_tree_digest"),
        },
        "configure": {
            "flags": [
                "-DCMAKE_BUILD_TYPE=Release",
                "-DTRT_PACKAGE_DIR=/usr",
                "-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake",
                "-DEMBEDDED_TARGET=jetson-orin",
                "-DCUDA_CTK_VERSION=13.2",
                "-DENABLE_CUTE_DSL=ALL",
                "-DCMAKE_CUDA_ARCHITECTURES=87",
            ],
            "pybind_flags": [
                "-DBUILD_PYTHON_BINDINGS=ON",
                "-DPython_EXECUTABLE=/opt/venv/bin/python",
                "-DPython3_EXECUTABLE=/opt/venv/bin/python",
                "-Dpybind11_DIR=<pybind11.get_cmake_dir()>",
            ],
            "cmake_cache_selected":
            container.get("cmake_cache_selected"),
        },
        "phases": container.get("phases"),
        "container": container.get("container"),
        "venv_freeze": container.get("venv_freeze"),
        "outputs": {
            "builder": manifest(out_root / "builder",
                                rec / "builder-files.txt"),
            "runtime": manifest(out_root / "runtime",
                                rec / "runtime-files.txt"),
        },
        "elf_dependencies": json.loads(read(rec / "elf-needed.json") or "[]"),
    }
    target.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
