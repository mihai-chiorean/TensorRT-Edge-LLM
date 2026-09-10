#!/bin/bash
# Start the experimental OpenAI-compatible server on the Jetson, loopback-only,
# inside a measurable cgroup (system.slice/trt-llm.service) so the
# edge-conversation golden harness can read it via --cgroup-unit.
#
# A transient *service* unit is used rather than `systemd-run --scope`: scopes
# reject exec properties, so SupplementaryGroups= cannot be set, and the process
# then loses the `video` group and fails on /dev/nvmap.
set -euo pipefail

B=${B:-$HOME/workspace/TensorRT-Edge-LLM-gemma-sidecars/build-0.10.1}
R=${R:-$HOME/workspace/TensorRT-Edge-LLM-gemma-sidecars}
E=${E:-$HOME/tensorrt-edgellm-workspace/gemma-4-E4B/engines-int8-16384}
V=${V:-$HOME/.venvs/tensorrt-edge-llm}
LOG=${LOG:-$HOME/trtbench/server.log}
PORT=${PORT:-8003}

mkdir -p "$(dirname "$LOG")"; : > "$LOG"
sudo systemctl reset-failed trt-llm.service 2>/dev/null || true

sudo systemd-run --unit=trt-llm --service-type=exec \
  -p User="$USER" -p 'SupplementaryGroups=video render' \
  -p WorkingDirectory="$R" \
  -p Environment=EDGELLM_PLUGIN_PATH="$B/libNvInfer_edgellm_plugin.so" \
  -p Environment=EDGELLM_PYBIND_DIR="$B/pybind" \
  -p Environment=PYTHONPATH="$R" \
  -p StandardOutput=append:"$LOG" \
  -p StandardError=append:"$LOG" \
  "$V/bin/python" -m experimental.server "$E" \
    --host 127.0.0.1 --port "$PORT" \
    --max-input-len 8192 --max-kv-cache-capacity 16384 --max-batch-size 1 \
    --enable-context-reuse --context-cache-max-records 1024 \
    --enable-auto-tool-choice --served-model-name gemma

# Stop with: sudo systemctl stop trt-llm.service
