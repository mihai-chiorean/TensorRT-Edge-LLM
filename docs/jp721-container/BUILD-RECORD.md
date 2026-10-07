<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Off-device container build for Jetson AGX Orin, JetPack 7.2.1 (sm_87)

Record of building this fork at `cda4fc1` for a Jetson AGX Orin / Orin NX
running JetPack 7.2.1 (L4T R39.2, CUDA 13.2, TensorRT 10.16.2, cuDNN 9.20)
inside a CPU-only Docker container on an arm64 host, instead of compiling on
the device. Everything the container used came from NVIDIA's Jetson apt
repository at exact pinned versions; the host's own CUDA was never visible to
the build, and no produced binary was executed. The machine-readable
companion is `build-record.json`; the manifests and logs referenced below sit
next to this file. The Dockerfile and scripts are in `docker/jp721/`.

## Result

| Item | Value |
|---|---|
| Fork commit | `cda4fc1821d1bd0941a755a515269ba928bba7d7` (`feat/jp72-sm87-0.11.0`) |
| Upstream base | `95515c2` = TensorRT Edge-LLM v0.11.0 |
| Build host | `spark-094a`, aarch64, Ubuntu 24.04.5, kernel `7.0.0-1019-nvidia`, Docker 29.6.2 (runtime `runc`, no `--gpus`, `DeviceRequests: null`, `Devices: []`) |
| Image | `trt-edge-llm/jp721-build:cda4fc1`, id `sha256:0572e32c9aaf46df1e940af94bd15646cd9f7199a4328ba4c7ee6271a069ff5f`, 9,168,165,769 B |
| Base image | `ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55` (arm64, image id `sha256:95d16dfcd4ab…`) |
| Dockerfile sha256 | `0d6bca044676342dad60e6953c3add79d6bf93187b7aa12e5d7cc8b8d8816332` |
| `build-edgellm.sh` sha256 | `5b34950a7929dfbb95ce5f54c2570f5c054309add54f1611bf0acee4d5c11c37` |
| Compilers | gcc/g++ 13.3.0 (`/usr/bin/aarch64-linux-gnu-g++` as the CUDA host compiler), nvcc V13.2.86, GNU ld 2.42, CMake 3.28.3, GNU make 4.3, git 2.43.0, Python 3.12.3 (system and venv) |
| CUDA architectures | `87` only, in `CMakeCache.txt`, in every `flags.make` (`arch=compute_87,code=[compute_87,sm_87]`, 4 directories) and in the embedded cubins (`cuobjdump --list-elf`: `sm_87` only in the plugin and the pybind module) |
| GPU probing | none: `CMAKE_CUDA_ARCHITECTURES_NATIVE ""` in `CMakeCUDACompiler.cmake.txt`, no `/dev/nvidia*`, no `nvidia-smi`, no `libcuda.so.1` in the container |
| Bundle source digest | `c1520a35d4e72839ec6a2ecbc2681b9b674c1c28ab98e7fd66700ad2b04522cf`, equal to `EXPECTED_SOURCE_TREE_SHA256` in edge-conversation's `create-bundle.sh` |
| Outputs on the Spark | `~/trt-edge-build/out/builder/` and `~/trt-edge-build/out/runtime/` (binaries stay there; only manifests are committed) |

## Jetson apt repository

Sources line installed at `/etc/apt/sources.list.d/nvidia-jetson-r39.2.list`:

```text
deb [arch=arm64 signed-by=/usr/share/keyrings/nvidia-jetson-ota.gpg] https://repo.download.nvidia.com/jetson/common r39.2 main
```

Signing key: `https://repo.download.nvidia.com/jetson/jetson-ota-public.asc`,
RSA-4096, `NVIDIA Corporation <linux-tegra-bugs@nvidia.com>`, fingerprint
`3C6D 1FF3 100C 8C3A BB08 69C0 E654 3461 A999 6195`. The Dockerfile checks the
fingerprint before dearmoring the key and scopes it to this one source with
`signed-by`. A flashed device gets the same key and the same `common` line
(plus a `t234` line for the L4T BSP, not needed to compile) from
`nvidia-l4t-apt-source`. The `r39.2` pocket (`Release` dated 2026-09-17) is
the only R39.2 suite; JetPack 7.2.1 is expressed by package versions, not a
separate suite. The pocket carries both the JetPack 7.2 GA revisions
(`cuda-*` 13.2.75/13.2.78, `nvidia-jetpack 7.2-b184/b187`) and the 7.2.1
revisions (`cuda-*` 13.2.86, `nvidia-jetpack 7.2.1-b49`, whose `nvidia-cuda`
pins `cuda-cudart-dev-13-2 = 13.2.86-1`); the pins below select 7.2.1.

## Installed NVIDIA packages (dpkg-query)

28 packages, 5,060 MB of archives, 8,085 MB on disk. Installation into a plain
`ubuntu:24.04` arm64 container succeeded on the first try: no package pulled
in an `nvidia-l4t-*` dependency, no postinst touched Tegra hardware,
`/etc/nv_tegra_release` or dpkg diversions; the only postinst output was
`update-alternatives` wiring `/usr/local/cuda` and `/usr/local/cuda-13` to
`/usr/local/cuda-13.2`. The Dockerfile fails if the installed set differs
from the pins (`dpkg-query` diff against `/etc/jetson-toolchain-pins.txt`).

| Package | Version |
|---|---|
| cuda-cccl-13-2, cuda-crt-13-2, cuda-cudart-13-2, cuda-cudart-dev-13-2, cuda-culibos-13-2, cuda-culibos-dev-13-2, cuda-cuobjdump-13-2, cuda-driver-dev-13-2, cuda-nvcc-13-2, cuda-nvrtc-13-2, cuda-nvrtc-dev-13-2, cuda-toolkit-13-2-config-common, cuda-toolkit-13-config-common, cuda-toolkit-config-common, libnvptxcompiler-13-2, libnvvm-13-2 | 13.2.86-1 |
| libcurand-13-2, libcurand-dev-13-2 | 10.4.2.66-1 |
| libnvinfer10, libnvinfer-dev, libnvinfer-headers-dev, libnvinfer-headers-plugin-dev, libnvinfer-safe-headers-dev, libnvonnxparsers10, libnvonnxparsers-dev | 10.16.2.10-1+cuda13.2 |
| libcudnn9-cuda-13, libcudnn9-dev-cuda-13, libcudnn9-headers-cuda-13 | 9.20.0.46-1 |

Full list with architectures: `dpkg-nvidia.txt`; Ubuntu toolchain packages:
`dpkg-toolchain.txt` (gcc 4:13.2.0-7ubuntu1 meta, gcc-13 13.3.0, cmake
3.28.3-1build7, libc6 2.39-0ubuntu8.9, libstdc++6 14.2.0-4ubuntu2~24.04.1).

Two install-time findings, both required to compile and both recorded as
pins rather than worked around:

1. `NvInferPlugin.h` is not in `libnvinfer-dev`; it is in
   `libnvinfer-headers-plugin-dev` (6 KB), which a device gets through the
   `tensorrt-dev` metapackage. Without it the first compile stopped at
   `cpp/plugins/qsaAttentionPlugin/qsaAttentionPlugin.h:20: fatal error:
   NvInferPlugin.h: No such file or directory` (log kept as
   `logs/build-attempt3-missing-plugin-header.log` on the Spark). Nothing
   links `libnvinfer_plugin`; the header alone suffices.
2. `cuda-nvrtc-dev-13-2` is needed at configure time (`cmake/FindNVRTC.cmake`
   is fatal without `nvrtc.h`), as the 0.10.1 addendum of `JP72-BUILD.md`
   already states for the device. The plugin links `libnvrtc.so.13`.

## Configure

Recipe from `docs/JP72-BUILD.md`, run from `build-jp721/` inside the clone,
with the architecture also passed explicitly so it is visible in the cache:

```text
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr
         -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake
         -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=13.2
         -DENABLE_CUTE_DSL=ALL -DCMAKE_CUDA_ARCHITECTURES=87
```

Pybind reconfigure of the same tree: `-DBUILD_PYTHON_BINDINGS=ON
-DPython_EXECUTABLE=/opt/venv/bin/python -DPython3_EXECUTABLE=/opt/venv/bin/python
-Dpybind11_DIR=$(pybind11.get_cmake_dir())`, then `make -j8 _edgellm_runtime`.

Evidence (`cmake-configure.log`, `cmake-cache-selected.txt`,
`CMakeCUDACompiler.cmake.txt`, `nvcc-gencode-flags.txt`):

- `The CUDA compiler identification is NVIDIA 13.2.86`;
  `CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc"` (toolchain file), host
  compiler `/usr/bin/aarch64-linux-gnu-g++`. The toolchain file sets
  `CMAKE_CUDA_COMPILER_FORCED TRUE`, so CMake compiled its compiler-id source
  but ran no test program.
- `CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=87`, `EMBEDDED_TARGET=jetson-orin`,
  `CUDA_CTK_VERSION=13.2`, `CMAKE_BUILD_TYPE=Release`,
  `CMAKE_HOME_DIRECTORY=/src/TensorRT-Edge-LLM`.
- `CMAKE_CUDA_ARCHITECTURES_NATIVE ""`: CMake's native-architecture detection
  (the only path that would run a CUDA program) never ran. A repository-wide
  search found one `execute_process` in the CMake tree (`cmake/CuteDsl.cmake`,
  extracting the prebuilt tarball) and no `try_run`/`nvidia-smi` use, so
  nothing had to be disabled.
- `CUDA_DIR=/usr/local/cuda/targets/sbsa-linux`;
  `CUDART_LIB=…/sbsa-linux/lib/libcudart.so`;
  `CUDA_DRIVER_LIB=…/sbsa-linux/lib/stubs/libcuda.so` (link-time stub,
  linked as `-lcuda`, not baked into RUNPATH);
  `NVRTC_DYNAMIC_LIB=…/sbsa-linux/lib/libnvrtc.so`;
  `TensorRT_LIBRARY=/usr/lib/aarch64-linux-gnu/libnvinfer.so`;
  `TensorRT_INCLUDE_DIR=/usr/include/aarch64-linux-gnu`.
- `FMHA Kernels: Excluding SM architectures: EXCLUDE_SM_80;…;EXCLUDE_SM_121`
  (only 87 kept); `Guided decoding: XGrammar, 20 sources`;
  `CuTe DSL: extracting prebuilt from …/cutedsl_aarch64_sm_87_cuda13.tar.gz`
  (sha256 `22fbc32e92546f98504a10d8adbd3b79809b72f11058521eef5409db2dbc4c72`),
  `arch=aarch64 artifact_tag=sm_87 groups=[f16_moe;fmha;gdn;gemm;int4_fp16_gemm;layernorm;rmsnorm;ssd]`.
- Submodules (`git submodule status --recursive`, all seven initialised,
  listed in `submodules.txt`): NVTX `2fb879e`, googletest `f132c89`,
  nlohmannJson `55f9368`, xgrammar `5b4e9ce` with nested cpptrace `6689d14`,
  dlpack `bbd2f4d`, googletest `df1544b`.

## Time and memory

Container limits: `--cpus 10 --memory 14g --memory-swap 14g` (cgroup
`cpu.max 1000000 100000`, `memory.max 15032385536`), `make -j8`, host with 20
cores and ~17 GB free beside its other services. Clean run, all phases
`rc=0` (`build-record.json` → `phases`):

| Phase | Wall | Largest single-process max RSS |
|---|---|---|
| preflight | 1 s | 31 MiB |
| configure | 1 s | 92 MiB |
| check-configure | 0 s | 3 MiB |
| build (`make -j8`, default target) | 93 s | 767 MiB |
| configure-pybind | 1 s | 25 MiB |
| build-pybind (`_edgellm_runtime`) | 17 s | 1,509 MiB (link) |
| collect | 4 s | 122 MiB |
| verify-outputs | 1 s | 75 MiB |
| record | 0 s | 31 MiB |
| container total (`docker run`) | 118 s | |

Peak memory of the whole container: `memory.peak` = 4,000,264,192 B (3.7 GiB)
at the end of the run; `docker stats` sampled every 10 s saw at most
3,434 MiB. Earlier runs of the same recipe compiled in 153 s (cold page
cache) and 106 s, with a `memory.peak` of 4,288,159,744 B. Image build,
uncached from the key layer down (`record/logs/docker-build.log` on the
Spark holds the full apt transcript): 127 s in total, of which the pinned apt
layer 79 s (5,060 MB fetched at about 110 MB/s), clone with recursive
submodules 24 s, venv 9 s. With all layers cached the image step takes 2 s.

## Output trees

Hashes are in `builder-SHA256SUMS` and `runtime-SHA256SUMS` (relative to
each tree); both trees were produced from the same `make` run.

### `out/builder/` (engine building; 25,105,544 B plus two symlinks)

| Path | Size (B) | sha256 |
|---|---|---|
| `lib/libNvInfer_edgellm_plugin.so.1.0` | 24,236,384 | `8d5166944dff3185fb0a588ad89b67c88948f43112999e32860c18d170b5df71` |
| `lib/libNvInfer_edgellm_plugin.so.1` | symlink → `.so.1.0` (SONAME) | |
| `lib/libNvInfer_edgellm_plugin.so` | symlink → `.so.1` | |
| `bin/llm_build` | 542,904 | `e472944e59cd404828bdf63abb3a5cffd0ceb6b319f6abf7c73589bfa86abf22` |
| `bin/visual_build` | 326,256 | `9a7af2a628f02a42987c49c0e874bf07a19b55a5f6b93be69db13d293703a2c3` |

### `out/runtime/` (what the experimental server loads; 122,013,741 B, 519 files)

Laid out the way edge-conversation's `create-bundle.sh` stages `runtime/`.

| Path | Size (B) | sha256 |
|---|---|---|
| `plugin/libNvInfer_edgellm_plugin.so` | 24,236,384 | `8d5166944dff3185fb0a588ad89b67c88948f43112999e32860c18d170b5df71` |
| `pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so` | 32,564,232 | `b7a73c8714c0dc6f54156643c36db9b7443fe63bcd315231c1b580a239768e21` |
| `source/experimental/`, `source/tensorrt_edgellm/`, `source/LICENSE` | 5,296,821 (485 files, `git archive` of `cda4fc1`) | per file in `runtime-SHA256SUMS` |
| `REVISION` | 41 | `4814afdec793a70a4ba7b9c7399532f6c564b1a4e36358cc5f8f38af8c429359` |
| `requirements.lock` | 508 (`pip freeze --all` of the venv, 30 pins) | `a9e02feec607b841c3bd5956561b62bcde70f98760b05a655a15ec6fc875c665` |
| `wheelhouse/` | 59,850,008 (30 wheels, aarch64 / cp312 / abi3 / pure) | per file in `runtime-SHA256SUMS` |

Shared between the trees: exactly one file, the plugin library.
`builder/lib/libNvInfer_edgellm_plugin.so.1.0` and
`runtime/plugin/libNvInfer_edgellm_plugin.so` are byte-identical
(`8d516694…`); the builder tree keeps the soname chain, the runtime tree
carries the plain copy that `EDGELLM_PLUGIN_PATH` points at. Nothing else is
shared: `llm_build`/`visual_build` exist only in `builder/`, and the pybind
module exists only in `runtime/`. The pybind module links `edgellmCore` and
`edgellmBuilder` statically and does not link the plugin; the C++ runtime
dlopens the plugin from `EDGELLM_PLUGIN_PATH` (`cpp/common/trtUtils.h`).

### Dynamic dependencies (`readelf -d`, resolved without loading anything; `elf-needed.txt`)

All four ELF objects carry `RUNPATH /usr/local/cuda/targets/sbsa-linux/lib`,
the same path the JetPack packages use on the device, and no stub directory.

| Object | NEEDED from Jetson packages | Other NEEDED |
|---|---|---|
| `libNvInfer_edgellm_plugin.so.1.0` (both copies) | `libcudart.so.13` [cuda-cudart-13-2], `libnvinfer.so.10` [libnvinfer10], `libnvrtc.so.13` [cuda-nvrtc-13-2], `libcuda.so.1` (driver, device only) | libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6, ld-linux-aarch64.so.1 |
| `_edgellm_runtime.cpython-312-aarch64-linux-gnu.so` | `libnvinfer.so.10`, `libnvonnxparser.so.10` [libnvonnxparsers10], `libcudart.so.13`, `libcuda.so.1` (driver, device only) | libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6, ld-linux-aarch64.so.1 |
| `llm_build` | `libcudart.so.13`, `libnvonnxparser.so.10`, `libnvinfer.so.10` | libstdc++.so.6, libgcc_s.so.1, libc.so.6, ld-linux-aarch64.so.1 |
| `visual_build` | `libcudart.so.13`, `libnvonnxparser.so.10`, `libnvinfer.so.10` | libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6, ld-linux-aarch64.so.1 |

No object needs `libcudnn*`, `libnvinfer_plugin`, `libcublas` or any
`/usr/local/cuda-13.x` path from the host: the container's only CUDA is the
apt-installed `/usr/local/cuda-13.2` (alternatives target of
`/usr/local/cuda`), and the Spark's `/usr/local/cuda*` directories were not
mounted. `libcuda.so.1` is unresolved in the container by design (no driver);
on the device it comes from the L4T driver stack.

## Discrepancies against the target and the previous AGX build

- The target is JetPack 7.2.1; the AGX build in `PORT-0.11.0.md` used L4T
  R39.2.1 with CUDA 13.2.86 at `/usr/local/cuda-13.2` and
  `libnvinfer 10.16.2.10-1+cuda13.2`. The container matches it on every
  linked component: nvcc V13.2.86 / `cuda-*` 13.2.86-1, TensorRT
  10.16.2.10-1+cuda13.2, cuDNN 9.20.0.46-1, gcc 13.3.0, CMake 3.28.3,
  Python 3.12.3. "R39.2.1" has no apt suite of its own; the toolchain pocket
  is `r39.2` and the 7.2.1 revision is the 13.2.86 package set.
- The Orin NX record in `JP72-BUILD.md` (JetPack 7.2 GA) shows nvcc V13.2.78.
  If the NX has not moved to 7.2.1, its `libcudart.so.13`/`libnvrtc.so.13`
  are 13.2.78 while these binaries were compiled against 13.2.86 headers.
  Same major version, so the loader will accept them, but it was not
  verified here.
- `-DCMAKE_CUDA_ARCHITECTURES=87` was passed in addition to the recipe; the
  toolchain file sets the same value, so the result is identical and the
  cache now shows it. `CMAKE_HOME_DIRECTORY` is `/src/TensorRT-Edge-LLM`,
  not a device checkout path, so `create-bundle.sh`'s `CMakeCache.txt`
  provenance check would need `TRT_SOURCE_DIR` set accordingly or the
  staged-release path (`TRT_RUNTIME_BINARIES_FROM`) used.
- The default `make` target was built (it also produced `llm_inference`,
  `llm_bench`, `qwen3_tts_inference`, `audio_build`, `action_build`, …);
  only the requested outputs were collected.
- The venv holds `requirements-server.txt` plus `pybind11==3.0.4`, as the
  0.11.0 notes prescribe. edge-conversation's `requirements-jp72.txt` still
  lists `jinja2` and `transformers`; they are absent here (templates render
  in C++ since 0.11.0), and `pyyaml` is present only as a dependency of
  `huggingface-hub`. `huggingface-hub` is a range in
  `requirements-server.txt` and resolved to 1.33.0 on build day; the lock
  and wheelhouse freeze what was used.
- The build is not bit-reproducible for the CUDA objects. Two runs of the
  same image and source gave identical `llm_build` and `visual_build` but a
  different plugin (`d55b7a68…` then `8d516694…`) and pybind module
  (`6b538b15…` then `b7a73c87…`), with identical sizes. The tree has no
  `__DATE__`/`__TIME__` use; the variation is in the nvcc-produced objects
  (fatbin packaging), which was not investigated further. The hashes above
  therefore identify this run's artifacts, not "the" build of `cda4fc1`.
- Plugin size is 24,236,384 B; the NX 0.10.0 build documented 33 MB (a
  different release, so not comparable). The pybind module is 32,564,232 B,
  the "32.6 MB" the AGX 0.11.0 build reported; no hash of the AGX artifacts
  was available, so bit-identity with the device build is not established.

## Not verifiable without the device

- Loading: `dlopen` of the plugin and `import _edgellm_runtime` need
  `libcuda.so.1`; `llm_build --help` was deliberately not run.
- XQA decode kernels are JIT-compiled by NVRTC at engine-build time on the
  device (`cpp/kernels/decodeAttentionKernels/decoderXQAJitCompiler.cpp`),
  so the device needs `cuda-nvrtc-13-2` at run time; this build only proves
  the embedding of the kernel sources and the link to `libnvrtc.so.13`.
- The CuTe DSL sm_87 kernels come prebuilt in-tree
  (`kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_87_cuda13.tar.gz`) and
  were linked as-is; their correctness is inherited from upstream 0.11.0.
- Engine building (`llm_build`, `visual_build` against the
  `onnx-int8emb-0.11` export), the TensorRT engine/runtime behaviour and
  `verify-engine.py` all require the device.
- `create-bundle.sh` refuses to run anywhere without
  `/etc/nv_tegra_release` and an Orin device tree, so bundling from these
  trees still happens on a device (or through its staged-release path).

## Generic vs. fork-specific parts

Reusable as a Jetson toolchain image (Dockerfile stage `jetson-toolchain`):
the base image pin, Ubuntu build tools, the NVIDIA key and source layer, the
pinned JetPack package layer with its post-install audit, the layout checks
and `PATH`. Everything in that stage is parameterised by the version `ARG`s
at the top of the Dockerfile and knows nothing about this repository.

Specific to this fork (stage `edgellm-src` and the scripts): the clone at the
pinned commit with recursive submodules, the venv from
`requirements-server.txt` + pybind11, `build-edgellm.sh` (configure flags,
phases, output layout, audits), `elf-needed.py` (generic helper, used by the
audit), `make-record.py`, and `build.sh`'s image name and
`EXPECTED_SOURCE_TREE_SHA256` input.

## Reproducing

On an arm64 Docker host:

```bash
cd docker/jp721
EXPECTED_SOURCE_TREE_SHA256=c1520a35d4e72839ec6a2ecbc2681b9b674c1c28ab98e7fd66700ad2b04522cf \
ROOT=~/trt-edge-build CPUS=10 MEMORY=14g JOBS=8 ./build.sh
python3 make-record.py ~/trt-edge-build/record ~/trt-edge-build/out build-record.json
```

Network access is needed for apt, the git clone and pip; the apt layer is
fully pinned, the clone is pinned by commit, and pip is pinned by
`requirements-server.txt` except for the `huggingface-hub` range.
