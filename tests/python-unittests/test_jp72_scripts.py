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

import io
import json
import pathlib
import runpy
import sys
import time
import urllib.request

import pytest

G_SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "jp72"


def _content_event(content):
    return {"choices": [{"delta": {"content": content}}]}


def _stream(events):
    payload = "".join("data: " + json.dumps(event) + "\n\n"
                      for event in events)
    response = io.BytesIO((payload + "data: [DONE]\n\n").encode())
    response.headers = {"Content-Type": "text/event-stream"}
    return response


@pytest.mark.parametrize("content_before_error", [False, True])
def test_cacheprobe_reports_sse_error(monkeypatch, tmp_path,
                                      content_before_error):
    system_prompt = tmp_path / "system.txt"
    system_prompt.write_text("Be concise.")
    script = G_SCRIPT_DIR / "cacheprobe.py"
    monkeypatch.setattr(
        sys, "argv",
        [str(script), "http://localhost", "test-model",
         str(system_prompt)])
    events = [_content_event("partial")] if content_before_error else []
    events.append({"error": {"message": "generation failed"}})
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *args, **kwargs: _stream(events))

    with pytest.raises(SystemExit, match="generation failed"):
        runpy.run_path(str(script))


def test_cacheprobe_preserves_first_content_timing(monkeypatch, tmp_path,
                                                   capsys):
    system_prompt = tmp_path / "system.txt"
    system_prompt.write_text("Be concise.")
    script = G_SCRIPT_DIR / "cacheprobe.py"
    monkeypatch.setattr(
        sys, "argv",
        [str(script), "http://localhost", "test-model",
         str(system_prompt)])
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: _stream(
            [_content_event("Paris"),
             _content_event(".")]))
    timestamps = iter([0.0, 0.25] * 5)
    monkeypatch.setattr(time, "time", lambda: next(timestamps))

    runpy.run_path(str(script))

    assert capsys.readouterr().out.count("TTFT: 0.250 s") == 5


@pytest.mark.parametrize("content_count", [1, 2])
def test_smoke_measures_content_spread(monkeypatch, capsys, content_count):
    script = G_SCRIPT_DIR / "smoke.py"
    monkeypatch.setattr(sys, "argv", [str(script)])
    responses = iter([
        io.BytesIO(json.dumps({
            "data": [{
                "id": "test-model"
            }]
        }).encode()),
        io.BytesIO(
            json.dumps({
                "choices": [{
                    "message": {
                        "content": "Moo."
                    }
                }]
            }).encode()),
        _stream([_content_event("word")] * content_count),
    ])
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *args, **kwargs: next(responses))
    timestamps = iter([0.0, 1.0, 2.0, 3.0] +
                      ([5.0] if content_count == 2 else []) + [10.0, 11.0])
    monkeypatch.setattr(time, "time", lambda: next(timestamps))

    runpy.run_path(str(script))

    output = capsys.readouterr().out
    assert "n_sse_events_with_content: %d" % content_count in output
    if content_count == 1:
        assert "no evidence: fewer than two content events" in output
    else:
        assert "spread between first and last chunk = 2.000s" in output
