#!/usr/bin/env python3
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
"""Smoke + streaming-evidence probe for the TRT Edge-LLM OpenAI server. Stdlib only."""
import base64
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8003"
IMG = sys.argv[2] if len(sys.argv) > 2 else None


def post(path, body, stream=False, timeout=300):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


print("=== 1. GET /v1/models ===")
with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
    models = json.load(r)
print(json.dumps(models, indent=2)[:800])
MODEL = models["data"][0]["id"]
print("MODEL_ID:", MODEL)

print("\n=== 2. non-streaming text completion ===")
t0 = time.time()
with post(
        "/v1/chat/completions", {
            "model":
            MODEL,
            "messages": [{
                "role":
                "user",
                "content":
                "What sound does a cow make? Answer in one short sentence."
            }],
            "max_tokens":
            64,
            "temperature":
            0.0
        }) as r:
    out = json.load(r)
print("elapsed %.2fs" % (time.time() - t0))
print("TEXT:", repr(out["choices"][0]["message"]["content"]))
print("usage:", out.get("usage"))

print("\n=== 3. STREAMING text completion (inter-chunk timing) ===")
t0 = time.time()
deltas, arrivals = [], []
r = post(
    "/v1/chat/completions", {
        "model":
        MODEL,
        "messages":
        [{
            "role": "user",
            "content": "Count from one to ten in words, separated by commas."
        }],
        "max_tokens":
        96,
        "temperature":
        0.0,
        "stream":
        True,
        "stream_options": {
            "include_usage": True
        }
    })
print("Content-Type:", r.headers.get("Content-Type"))
for raw in r:
    line = raw.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        arrivals.append((time.time() - t0, "[DONE]"))
        break
    obj = json.loads(payload)
    ch = obj.get("choices") or [{}]
    d = (ch[0].get("delta") or {}).get("content")
    if d:
        deltas.append(d)
        arrivals.append((time.time() - t0, d))
print("n_sse_events_with_content:", len(deltas))
print("first 12 arrivals (t_sec, delta):")
for t, d in arrivals[:12]:
    print("   %7.3f  %r" % (t, d))
if len(arrivals) > 12:
    print("   ... last: %7.3f %r" % (arrivals[-1][0], arrivals[-1][1]))
print("TTFT_stream: %.3fs   total: %.3fs" %
      (arrivals[0][0] if arrivals else -1, time.time() - t0))
print("FULL TEXT:", repr("".join(deltas)))
spread = (arrivals[-1][0] - arrivals[0][0]) if len(arrivals) > 1 else 0
print("INCREMENTAL?  spread between first and last chunk = %.3fs" % spread)

if IMG:
    print("\n=== 4. image completion ===")
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    t0 = time.time()
    with post(
            "/v1/chat/completions",
        {
            "model":
            MODEL,
            "messages": [{
                "role":
                "user",
                "content":
                [{
                    "type": "text",
                    "text": "What do you see? Answer in two short sentences."
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + b64
                    }
                }]
            }],
            "max_tokens":
            192,
            "temperature":
            0.0
        }) as r:
        out = json.load(r)
    print("elapsed %.2fs" % (time.time() - t0))
    print("VISION TEXT:", repr(out["choices"][0]["message"]["content"]))
    print("usage:", out.get("usage"))
