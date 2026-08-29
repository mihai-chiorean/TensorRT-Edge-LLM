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
"""Independent prefix-cache probe: send the SAME long prompt 3x and compare TTFT.
If a prefix cache is active, runs 2 and 3 will have dramatically lower TTFT."""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2]
SYS = open(sys.argv[3]).read()
FILLER = (
    "The robot noted the weather, the time, and the colour of the room. " *
    260)


def ttft(msgs, mt=32):
    body = {
        "model": MODEL,
        "messages": msgs,
        "max_tokens": mt,
        "temperature": 0.0,
        "stream": True
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=300)
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:") and line[5:].strip() not in ("[DONE]", ""):
            o = json.loads(line[5:].strip())
            ch = o.get("choices") or [{}]
            if (ch[0].get("delta") or {}).get("content"):
                first = time.time() - t0
                for _ in r:
                    pass
                return first
    return None


msgs = [{
    "role": "system",
    "content": SYS
}, {
    "role": "user",
    "content": FILLER + "\nWhat is the capital of France? One word."
}]
print("approx prompt chars:", len(SYS) + len(msgs[1]["content"]))
for i in range(3):
    t = ttft(msgs)
    print("identical-prompt run %d TTFT: %.3f s" % (i + 1, t))
# now a short prompt for contrast
short = [{
    "role": "system",
    "content": SYS
}, {
    "role": "user",
    "content": "What is the capital of Spain? One word."
}]
for i in range(2):
    print("short-prompt run %d TTFT: %.3f s" % (i + 1, ttft(short)))
