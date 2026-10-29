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

# Start the experimental OpenAI-compatible server on the Jetson, loopback-only,
# inside a measurable cgroup (system.slice/trt-llm.service) so the
# edge-conversation golden harness can read it via --cgroup-unit.
#
# A transient *service* unit is used rather than `systemd-run --scope`: scopes
# reject exec properties, so SupplementaryGroups= cannot be set, and the process
# then loses the `video` group and fails on /dev/nvmap.
set -euo pipefail

# Defaults are the recorded device paths; set B (build tree), R (checkout), E (engines) and V (venv) to
# the locations used when following docs/JP72-BUILD.md.
B=${B:-$HOME/workspace/TensorRT-Edge-LLM-port-0.11/build-0.11.0}
R=${R:-$HOME/workspace/TensorRT-Edge-LLM-port-0.11}
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
