#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Build the Gemma 4 E4B SM87 engines from the INT8-embedding ONNX directory.
# Run ON the Jetson (orin-nx-vqplnc, JP7.2 / CUDA 13.2 / TRT 10.16.2).
#
# NOTE: use llm_build/visual_build against the ONNX dir, NOT
# `tensorrt-edgellm-build --model-dir <ckpt>` -- the direct checkpoint builder
# still binds FP16 embedding tables and discards the 3.21 GiB INT8 sidecar
# saving. See docs/JP72-ARTIFACTS.md.
set -euo pipefail

# Defaults are the recorded device paths; set B (build tree), O (ONNX export) and E (engine output) to
# the locations used when following docs/JP72-BUILD.md.
#
# LoRA is opt-in: LORA_RANK=16 builds the LLM engine from $O/llm/lora_model.onnx with LoRA inputs up to
# rank 16 and writes to a separate default engine directory. The default 0 builds the production engine
# without the per-GEMM LoRA branches. lora_model.onnx is written on the export host by
# `tensorrt-edgellm-insert-lora --onnx-dir <llm>` (it needs onnx/torch, which the device does not have);
# it is run here only when that command is on PATH.
#
# HW_COMPAT=same_compute_capability builds engines that load on any SM87 device, for building on the
# AGX and deploying to the NX; the default `none` targets the build GPU.
B=${B:-$HOME/workspace/TensorRT-Edge-LLM-port-0.11/build-0.11.0}
O=${O:-$HOME/tensorrt-edgellm-workspace/gemma-4-E4B/onnx-int8emb-0.11}
LORA_RANK=${LORA_RANK:-0}
HW_COMPAT=${HW_COMPAT:-none}
if [ "$LORA_RANK" -gt 0 ]; then
    E=${E:-$HOME/tensorrt-edgellm-workspace/gemma-4-E4B/engines-int8-16384-lora$LORA_RANK}
else
    E=${E:-$HOME/tensorrt-edgellm-workspace/gemma-4-E4B/engines-int8-16384}
fi

if [ "$LORA_RANK" -gt 0 ]; then
    if [ ! -f "$O/llm/lora_model.onnx" ] && command -v tensorrt-edgellm-insert-lora >/dev/null; then
        tensorrt-edgellm-insert-lora --onnx-dir "$O/llm"
    fi
    if [ ! -f "$O/llm/lora_model.onnx" ]; then
        echo "missing $O/llm/lora_model.onnx: run tensorrt-edgellm-insert-lora --onnx-dir <llm> on the export host" >&2
        exit 1
    fi
    if [ "$O/llm/model.onnx" -nt "$O/llm/lora_model.onnx" ]; then
        echo "$O/llm/lora_model.onnx is older than model.onnx: rerun insert-lora" >&2
        exit 1
    fi
fi

export EDGELLM_PLUGIN_PATH=$B/libNvInfer_edgellm_plugin.so
mkdir -p "$E"

# Peak-RSS-sampling wrapper: /usr/bin/time is not installed on this device, and
# a systemd scope cannot carry SupplementaryGroups=, so a scope would lose the
# `video` group and fail on /dev/nvmap with "NvRmMemInitNvmap: Permission denied".
run_measured() {
    local tag=$1; shift
    local t0 hwm=0 v
    t0=$(date +%s)
    "$@" &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
        v=$(awk '/VmHWM/{print $2}' /proc/$pid/status 2>/dev/null || true)
        [ -n "${v:-}" ] && [ "$v" -gt "$hwm" ] && hwm=$v
        sleep 2
    done
    local rc=0
    wait "$pid" || rc=$?
    echo "$tag EXITCODE=$rc ELAPSED_SEC=$(( $(date +%s) - t0 )) PEAK_VmHWM_KB=$hwm"
    return $rc
}

# 16384 KV / 8192 input chosen from the model's real KV geometry; see
# docs/JP72-ARTIFACTS.md "Why 16384, and what it costs".
run_measured llm_build "$B/examples/llm/llm_build" \
    --onnxDir "$O/llm" --engineDir "$E/llm" \
    --maxInputLen 8192 --maxKVCacheCapacity 16384 --maxBatchSize 1 \
    --maxLoraRank "$LORA_RANK" --hardware-compatibility "$HW_COMPAT" 2>&1 | tee "$E/llm_build.log"

# The runtime stops on eos_token_id; Gemma 4 needs <turn|> (106) and <|tool_response> (50) beside <eos>.
python3 - "$E/llm/config.json" "$LORA_RANK" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
eos = config.get("eos_token_id")
if eos != [1, 106, 50]:
    sys.exit(f"{sys.argv[1]}: eos_token_id={eos!r}, expected [1, 106, 50]")
rank = config.get("builder_config", {}).get("max_lora_rank")
if rank != int(sys.argv[2]):
    sys.exit(f"{sys.argv[1]}: builder_config.max_lora_rank={rank!r}, expected {sys.argv[2]}")
PY

# visual_build writes to <engineDir>/visual/, so pass $E, not $E/visual.
# Gemma 4 E4B declares vision_soft_tokens_per_image: 280; 4-560 covers it.
run_measured visual_build "$B/examples/multimodal/visual_build" \
    --onnxDir "$O/visual" --engineDir "$E" \
    --minImageTokens 4 --maxImageTokens 560 --maxImageTokensPerImage 560 \
    --hardware-compatibility "$HW_COMPAT"
