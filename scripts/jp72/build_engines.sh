#!/bin/bash
# Build the Gemma 4 E4B SM87 engines from the INT8-embedding ONNX directory.
# Run ON the Jetson (orin-nx-vqplnc, JP7.2 / CUDA 13.2 / TRT 10.16.2).
#
# NOTE: use llm_build/visual_build against the ONNX dir, NOT
# `tensorrt-edgellm-build --model-dir <ckpt>` -- the direct checkpoint builder
# still binds FP16 embedding tables and discards the 3.21 GiB INT8 sidecar
# saving. See docs/JP72-ARTIFACTS.md.
set -euo pipefail

B=${B:-$HOME/workspace/TensorRT-Edge-LLM-gemma-sidecars/build-jp72-sm87}
O=${O:-$HOME/tensorrt-edgellm-workspace/gemma-4-E4B/onnx-int8emb}
E=${E:-$HOME/tensorrt-edgellm-workspace/gemma-4-E4B/engines-int8-16384}

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
    wait $pid; local rc=$?
    echo "$tag EXITCODE=$rc ELAPSED_SEC=$(( $(date +%s) - t0 )) PEAK_VmHWM_KB=$hwm"
    return $rc
}

# 16384 KV / 8192 input chosen from the model's real KV geometry; see
# docs/JP72-ARTIFACTS.md "Why 16384, and what it costs".
run_measured llm_build "$B/examples/llm/llm_build" \
    --onnxDir "$O/llm" --engineDir "$E/llm" \
    --maxInputLen 8192 --maxKVCacheCapacity 16384 --maxBatchSize 1

# visual_build writes to <engineDir>/visual/, so pass $E, not $E/visual.
# Gemma 4 E4B declares vision_soft_tokens_per_image: 280; 4-560 covers it.
run_measured visual_build "$B/examples/multimodal/visual_build" \
    --onnxDir "$O/visual" --engineDir "$E" \
    --minImageTokens 4 --maxImageTokens 560 --maxImageTokensPerImage 560
