<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native build on Jetson Orin NX — JetPack 7.2 / CUDA 13.2 / sm_87

Reproducible record of the native toolchain build of this fork on a Jetson
Orin NX 16 GB running JetPack 7.2. Written against branch
`feat/jp72-cuda13-sm87` (upstream base `bb29145` = v0.10.0).

## Headline: no port was required

Upstream **v0.10.0 (`bb29145`) already lists Jetson Orin on JetPack 7.2 /
CUDA 13.2 as an `Official` platform** — see
`docs/source/user_guide/getting_started/support-matrix.md`:

| Platform | Level | OS / SDK | CUDA Toolkit | TensorRT |
|---|---|---|---|---|
| Jetson Orin | **Official** | **JetPack 7.2** | **13.2** | JetPack package |
| Jetson Orin | Compatible | JetPack 6.2+ | 12.6 | JetPack package |

JetPack 7.2 Orin is *better* supported than the JP6.2 path used previously,
which upstream classifies only as `Compatible`. `installation.md` ships an
exact `JetPack 7.2 Orin` CMake recipe.

**Zero source changes were needed for JP7.2 / CUDA 13.2 / sm_87.** This
document is the only commit produced by the JP7.2 bring-up. The three existing
fork commits (`ec2d6cf`, `66051d8`, `f9368b9`) build unmodified on CUDA 13.2 —
none of them contain CUDA-version-specific code.

### Rebase vs. patch

Not applicable in either direction. As of this build, upstream
`NVIDIA/TensorRT-Edge-LLM` `main`, `release/0.10.0` and tag `v0.10.0` all point
at `bb29145` — the base we are already on. There is no newer upstream base to
rebase onto, and no patching of `bb29145` was necessary. Verified with:

```bash
git ls-remote --heads --tags https://github.com/NVIDIA/TensorRT-Edge-LLM.git
# refs/heads/main            bb291453bc10d9bd1142070c605e090d4e483ee2
# refs/heads/release/0.10.0  bb291453bc10d9bd1142070c605e090d4e483ee2
```

### The three JP6.2 workarounds, and what replaced them

| JP6.2 / CUDA 12.6 workaround | JP7.2 / CUDA 13.2 resolution |
|---|---|
| Hand-built SM87/CUDA-12 CuTe DSL kernel archive (upstream shipped none) | **Not needed.** `kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_87_cuda13.tar.gz` ships in the tree and CMake auto-extracts it. There is no `..._sm_87_cuda12.tar.gz`, which is exactly why JP6.2 needed a hand-built archive. |
| `python3.10-venv` | `python3.12-venv`. `pyproject.toml` declares `requires-python = ">=3.10"` and a `Programming Language :: Python :: 3.12` classifier. Python 3.10 is **not** required. |
| `libcurand-dev-12-6` | `libcurand-dev-13-2` (`10.4.2.66-1`), present in the stock JetPack 7.2 apt sources. |

No side-by-side CUDA toolkit and no sbsa apt repo were required. The
JetPack-owned CUDA 13.2 install and L4T driver were not touched.

## Verified environment

| Component | Version |
|---|---|
| Device | `orin-nx-vqplnc`, Orin NX 16 GB, sm_87 (compute cap 8.7) |
| OS | Ubuntu 24.04.4 LTS |
| L4T | R39 rev 2.0 (JetPack 7.2), GCID 45755727 |
| CUDA Toolkit | 13.2, `nvcc` V13.2.78, at `/usr/local/cuda-13.2` |
| TensorRT | 10.16.2.10-1+cuda13.2 (`libnvinfer-dev`) |
| cuDNN | 9.20.0.46-1 (`libcudnn9-cuda-13`) |
| cuRAND dev | 10.4.2.66-1 (`libcurand-dev-13-2`) |
| CMake | 3.28.3 |
| GCC | 13.3.0 |
| Python | 3.12.3 |
| CuTe DSL artifact | prebuilt `aarch64/sm_87`, cuda13, groups `f16_moe;fmha;gdn;gemm;int4_fp16_gemm;ssd` |

## System-level changes made on the device

Re-do all of these after a reflash.

1. **apt packages** (all from stock JetPack 7.2 / Ubuntu 24.04 sources):

   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake build-essential git \
       libcurand-dev-13-2 python3.12-venv python3-dev
   ```

   `python3.12-venv` pulls `python3.12-full`, `python3.12-dev`,
   `libpython3.12-dev` and friends.

2. **Python virtualenv** at `~/.venvs/tensorrt-edge-llm` (Python 3.12.3).
   Deliberately separate from `/opt/edge-conversation/venv` and
   `~/reachy-venv`, which run the live voice assistant and must not be touched.

3. **Source checkout** at `~/workspace/TensorRT-Edge-LLM-gemma-sidecars`,
   build tree at `build-jp72-sm87/`. Build tree is ~429 MB; venv ~360 MB.

Nothing else on the device was modified. No `edge-*` systemd unit,
`reachy-daemon`, CUDA install, or L4T driver was stopped, started, or
reconfigured.

### Build-host (beelink) note — device internet access

The Orin reaches the network only through the USB gadget link to the build
host, and the host had no NAT rule for that subnet, so `apt` and `pip` on the
device timed out. Add on the **build host** (in-memory, not persistent):

```bash
sudo iptables -t nat -A POSTROUTING \
    -s 192.168.55.0/24 ! -d 192.168.55.0/24 -j MASQUERADE
```

Remove with the same command and `-D` when finished. This is a build-host
change, not a device change, and it does not survive a host reboot.

## Build procedure

### 1. Source and submodules

```bash
git clone git@github.com:mihai-chiorean/TensorRT-Edge-LLM.git \
    ~/workspace/TensorRT-Edge-LLM-gemma-sidecars
cd ~/workspace/TensorRT-Edge-LLM-gemma-sidecars
git checkout feat/jp72-cuda13-sm87
git submodule update --init --recursive --depth 1
```

Submodule pins:

```text
3rdParty/NVTX          2fb879e512ed208f83c6aa4d4c96b958789edf49
3rdParty/googletest    f132c893119698e10daef8525d0ad7a3f05176f2
3rdParty/nlohmannJson  722c03495f9978eb727f480b6ea0742f652e06a9
```

> If you stage the tree by `rsync` from a build host rather than cloning,
> **do not** use an `--exclude='build*/'` pattern. It also matches
> `cpp/builder/`, and the build then fails with
> `fatal error: builder/llmBuilder.h: No such file or directory`.
> Anchor the exclude (`--exclude='/build-jp72-sm87/'`).

### 2. Configure

This is upstream's documented `JetPack 7.2 Orin` recipe, used verbatim:

```bash
mkdir -p build-jp72-sm87 && cd build-jp72-sm87
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCUDA_CTK_VERSION=13.2 \
    -DENABLE_CUTE_DSL=ALL
```

Expected, and observed, configure output:

```text
-- The CUDA compiler identification is NVIDIA 13.2.78
-- Found TensorRT: /usr/lib/aarch64-linux-gnu/libnvinfer.so
-- XQA Kernels: generating cubins for SM architectures: 87
-- CuTe DSL: extracting prebuilt from .../cutedsl_aarch64_sm_87_cuda13.tar.gz
-- CuTe DSL: arch=aarch64  artifact_tag=sm_87
             groups=[f16_moe;fmha;gdn;gemm;int4_fp16_gemm;ssd]
```

`EMBEDDED_TARGET=jetson-orin` makes `cmake/CuteDsl.cmake` infer
`CUTE_DSL_ARTIFACT_TAG=sm_87` automatically; do not pass it by hand.
Do not pass `-DCUDA_VERSION` — CMake reserves that name.

### 3. Build the core toolchain

```bash
make -j6
```

`-j6` rather than `-j$(nproc)` (8) leaves headroom on the shared device. The
build compiles ~200 XQA cubins for sm_87 first, then the core libraries.

Produces:

```text
libNvInfer_edgellm_plugin.so -> .so.1 -> .so.1.0   33 MB
examples/llm/llm_build                            405 KB
examples/llm/llm_inference                         21 MB
examples/llm/llm_bench                             20 MB
examples/multimodal/visual_build                  326 KB
examples/multimodal/audio_build                   327 KB
examples/multimodal/action_build                  322 KB
examples/omni/qwen3_tts_inference                  22 MB
```

### 4. Python environment (device side)

The device only needs the **runtime/server** dependencies. Do **not** run
`pip install -e ".[server]"` here: the base package pins `torch==2.13.0`,
which the OpenAI-compatible server never imports, and which would pull
gigabytes of CUDA wheels that can shadow the JetPack CUDA install.
Quantization and ONNX export run on an x86 host, per upstream's split.

```bash
python3 -m venv ~/.venvs/tensorrt-edge-llm
~/.venvs/tensorrt-edge-llm/bin/pip install --upgrade pip setuptools wheel
cd ~/workspace/TensorRT-Edge-LLM-gemma-sidecars
~/.venvs/tensorrt-edge-llm/bin/pip install -r requirements-server.txt
```

Installs cleanly on Python 3.12 aarch64: `fastapi 0.139.2`, `uvicorn 0.51.0`,
`pybind11 3.0.4`, `transformers 5.14.1`, `tokenizers 0.22.2`, `av 17.1.0`,
`numpy 2.2.6`, `jinja2 3.1.6`, `pyyaml 6.0.3`, `python-multipart 0.0.32`.
No torch, no CUDA wheels.

### 5. Python bindings for the OpenAI-compatible server

`BUILD_PYTHON_BINDINGS` defaults to `OFF`; the experimental server needs the
`_edgellm_runtime` module, so reconfigure the same build tree pointing at the
venv interpreter:

```bash
V=~/.venvs/tensorrt-edge-llm
PB=$($V/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())")
cd ~/workspace/TensorRT-Edge-LLM-gemma-sidecars/build-jp72-sm87
cmake .. \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPython_EXECUTABLE=$V/bin/python \
    -DPython3_EXECUTABLE=$V/bin/python \
    -Dpybind11_DIR=$PB
make -j6 _edgellm_runtime
```

Produces `pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so` (38 MB).

`Python_EXECUTABLE` and `Python3_EXECUTABLE` must both be set: the root
`CMakeLists.txt` uses `find_package(Python3 ...)` while
`experimental/pybind/CMakeLists.txt` uses `find_package(Python 3.10 ...)`.
Without both, the bindings link against the system interpreter and the venv
cannot import them.

## Verification performed

```bash
B=~/workspace/TensorRT-Edge-LLM-gemma-sidecars/build-jp72-sm87

# 1. Builder CLIs run
$B/examples/llm/llm_build --help
$B/examples/multimodal/visual_build --help

# 2. Plugin has no unresolved dependencies and links the JP7.2 stack
ldd $B/libNvInfer_edgellm_plugin.so | grep -i "not found"   # (empty)
ldd $B/libNvInfer_edgellm_plugin.so | grep -E "nvinfer|cudart"
#   libcudart.so.13 => /usr/local/cuda/targets/sbsa-linux/lib/libcudart.so.13
#   libnvinfer.so.10 => /lib/aarch64-linux-gnu/libnvinfer.so.10

# 3. Plugin dlopens against the live driver
python3 -c "import ctypes; ctypes.CDLL('$B/libNvInfer_edgellm_plugin.so', \
    mode=ctypes.RTLD_GLOBAL); print('plugin dlopen OK')"

# 4. Server stack imports, with the bindings resolved
cd ~/workspace/TensorRT-Edge-LLM-gemma-sidecars
EDGELLM_PYBIND_DIR=$B/pybind \
EDGELLM_PLUGIN_PATH=$B/libNvInfer_edgellm_plugin.so \
~/.venvs/tensorrt-edge-llm/bin/python -c "
import sys; sys.path.insert(0, '.')
import experimental.server.engine, experimental.server.api_server
print('server stack OK')"
```

All four pass. `_edgellm_runtime` exposes `LLMBuilder`, `LLMBuilderConfig`,
`LLMRuntime`, `AudioBuilder`, `FormattedRequest`, `ImageData` and the rest of
the runtime surface.

## Runtime environment variables

```bash
export EDGELLM_PLUGIN_PATH=$B/libNvInfer_edgellm_plugin.so
export EDGELLM_PYBIND_DIR=$B/pybind
```

`experimental/server/engine.py` auto-discovers both if the build tree is in a
conventional location, but setting them explicitly is what was validated.

## Known constraints on this device

- Jetson Orin does not support FP8, MXFP8, FP4 or NVFP4 runtime precision.
  Use FP16, INT8 or INT4 checkpoints only. This is an upstream Orin platform
  constraint on both JP6.2 and JP7.2, not a JP7.2 regression.
- Export and quantization are x86-host steps. The device builds engines from
  ONNX; it does not run ModelOpt.
- The device is shared with a live voice assistant. At build time roughly
  11.8 GB of the 15.6 GB unified memory was already resident, leaving about
  3.7 GB available. Compiling is fine at that headroom; **engine building is
  memory-hungry and may need the `edge-*` services stopped first.** That is
  the main session's call, not this build's.
- `-j$(nproc)` saturates all 8 cores and audibly degrades live speech. `-j6`
  is the tested compromise.

## Addendum: upstream 0.10.1 (`e8b2952`)

Branch `feat/jp72-sm87-0.10.1` rebases the fork line onto TensorRT Edge-LLM
0.10.1. The recipe above still applies with these changes, verified on a
Jetson AGX Orin 32 GB (JetPack 7.2 / CUDA 13.2 / TensorRT 10.16.2):

1. **NVRTC is a build requirement.** 0.10.1 JIT-compiles the XQA decode
   kernels with NVRTC at engine-build time and `cmake/FindNVRTC.cmake` is fatal
   without `nvrtc.h` and `libnvrtc.so`, which JetPack does not install:

   ```bash
   sudo apt-get install -y cuda-nvrtc-dev-13-2   # 13.2.86-1, stock apt source
   ```

   `libNvInfer_edgellm_plugin.so` then links `libnvrtc.so.13` at runtime.

2. **Unit tests build into per-area binaries** (`unittests/unitTest*`) when
   configured with `-DBUILD_UNIT_TESTS=ON`; there is no single `unitTest`.

3. **Engines must be rebuilt.** `AttentionPlugin` now serialises its XQA
   kernels into the engine and `Int4GroupwiseGemmPluginV2` gained a
   serialised `max_lock_workspace_bytes`; an engine built by 0.10.0 loads
   (the version mismatch is a warning) and then fails in
   `pluginV3Runner.cpp::onShapeChange` on the first prefill. Re-run
   `scripts/jp72/build_engines.sh` against the existing ONNX export; the
   0.10.0 `--int8-embedding` ONNX builds unchanged (E4B: 269 s for
   `llm_build`, 8192/16384/1).

4. **Server launch changed.** The server no longer takes `--model` and
   `--multimodal-engine-dir`: pass the engine tree (the directory holding
   `llm/` and `visual/`) as the positional argument, enable context reuse with
   `--enable-context-reuse`, and allow `tool_choice=auto/required` with
   `--enable-auto-tool-choice`. Remote `http(s)` media is refused unless
   `--allow-remote-media` is given. The Gemma 4 tool parser is selected
   automatically from the engine's `config.json`. See
   `scripts/jp72/start_server.sh`.

5. **Python runtime dependencies** are `requirements-server.txt` plus the
   `server-tools` extra (`transformers`, `jinja2`) for tool chat templates; the
   venv from the 0.10.0 recipe already satisfies both.
