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
"""
Server request-parsing coverage for every media modality, no engine / C++
runtime: OpenAI chat requests flow through the real HTTP endpoint and engine
conversion with stubs only at the C++-binding and video-decode boundaries,
plus the ``/v1/audio/transcriptions`` endpoint (FastAPI TestClient).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import tempfile
import time
import types

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                         ".."))

# Only genuinely optional external dependencies may turn an ImportError into
# a skip; a project-internal import failure must fail the test, not go green.
_OPTIONAL_TOP_MODULES = {
    "fastapi", "httpx", "av", "torch", "_edgellm_runtime", "python_multipart",
    "multipart", "librosa", "numpy", "soundfile", "uvicorn"
}


def _skip_or_raise(exc: ImportError, what: str):
    top = (getattr(exc, "name", None) or "").split(".")[0]
    if top in _OPTIONAL_TOP_MODULES:
        pytest.skip(f"{what} unavailable: {exc}")
    raise exc


# ---------------------------------------------------------------------------
# Stub runtime module (no C++): only what engine.py's message/buffer helpers use
# ---------------------------------------------------------------------------


class _FakeBuffer(tuple):
    """Tuple-comparable stub buffer that also accepts attribute writes (e.g. do_resize)."""

    def __new__(cls, *items):
        return super().__new__(cls, items)


class _Content:

    def __init__(self, ctype, data=""):
        self.type = ctype
        self.data = data


class _Message:

    def __init__(self):
        self.role = ""
        self.contents = []


class _StubRt:
    """Records which load_* binding each visual buffer used."""

    def MessageContent(self, ctype, data=""):
        return _Content(ctype, data)

    def Message(self):
        return _Message()

    def load_image_from_path(self, path):
        return _FakeBuffer("image", path)

    def load_image_from_bytes(self, data):
        return _FakeBuffer("image_bytes", data)

    def load_video_from_array(self, frames, fps, timestamps=()):
        return _FakeBuffer("video_array", fps)

    def load_video_from_paths(self, paths, fps, timestamps=()):
        return _FakeBuffer("video_paths", list(paths), fps)


def _engine():
    try:
        from experimental.server import engine
    except ImportError as exc:  # skip only for missing external deps
        _skip_or_raise(exc, "engine import")
    return engine


def test_openai_image_url_data_is_converted_and_loaded_from_memory():
    eng = _engine()
    # A measurable container: the data: preflight reads the dimensions out of
    # the header before the bytes reach the decoder.
    payload = _jpeg_bytes(320, 240)
    encoded = base64.b64encode(payload).decode("ascii")
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}"
            },
        }, {
            "type": "text",
            "text": "describe",
        }],
    }]

    cpp_messages = eng._convert_messages_to_cpp(_StubRt(), messages)
    assert [content.type
            for content in cpp_messages[0].contents] == ["image", "text"]

    buffers = eng._load_image_buffers(_StubRt(), messages)
    assert buffers == [("image_bytes", payload)]


def test_openai_image_url_rejects_remote_fetches():
    eng = _engine()
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.jpg"
            },
        }],
    }]

    with pytest.raises(ValueError, match="remote image URLs"):
        eng._load_image_buffers(_StubRt(), messages)


def test_openai_image_url_rejects_non_string_source():
    eng = _engine()
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": 123
            },
        }],
    }]

    with pytest.raises(ValueError, match="image source must be a string"):
        eng._load_image_buffers(_StubRt(), messages)


# ---------------------------------------------------------------------------
# Issue #191: data: image preflight (dimension / size / format bounds)
# ---------------------------------------------------------------------------


def _png_bytes(width, height):
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" +
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))


def _jpeg_bytes(width, height):
    """SOI + a skipped APP0 + SOF0 carrying the dimensions + SOS."""
    app0 = b"\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00"
    sof0 = (b"\xff\xc0" + struct.pack(">H", 11) + b"\x08" +
            struct.pack(">HH", height, width) + b"\x01\x01\x11\x00")
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xda"


def _gif_bytes(width, height):
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00"


def _bmp_bytes(width, height):
    # Negative height marks a top-down bitmap; the probe takes the magnitude.
    return (b"BM" + b"\x00" * 12 + struct.pack("<I", 40) +
            struct.pack("<ii", width, -height) + b"\x00" * 4)


def _bmp_core_bytes(width, height):
    return (b"BM" + b"\x00" * 12 + struct.pack("<I", 12) +
            struct.pack("<HH", width, height) + b"\x00" * 4)


def _image_data_url(payload, media_type="image/png"):
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _image_messages(url):
    return [{
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": url
            },
        }],
    }]


def _preflight():
    from experimental.server import image_preflight
    return image_preflight


@pytest.mark.parametrize("payload_fn,media_type", [
    (_png_bytes, "image/png"),
    (_jpeg_bytes, "image/jpeg"),
    (_gif_bytes, "image/gif"),
    (_bmp_bytes, "image/bmp"),
])
def test_preflight_probes_every_supported_container(payload_fn, media_type):
    pf = _preflight()
    data, size = pf.preflight_image_data_url(
        _image_data_url(payload_fn(640, 480), media_type))
    assert size == (640, 480)
    assert data == payload_fn(640, 480)


def test_preflight_rejects_oversized_dimension_without_decoding():
    """A ~50-byte PNG declaring a 60000x60000 canvas must never reach
    stb_image, which would allocate ~10 GB for it."""
    pf = _preflight()
    bomb = _png_bytes(60000, 60000)
    assert len(bomb) < 100
    with pytest.raises(ValueError, match="per side"):
        pf.preflight_image_data_url(_image_data_url(bomb))


def test_preflight_does_not_wrap_unsigned_bmp_core_dimensions():
    pf = _preflight()
    with pytest.raises(ValueError, match="per side"):
        pf.preflight_image_data_url(
            _image_data_url(_bmp_core_bytes(65535, 1), "image/bmp"))


@pytest.mark.parametrize("payload", [
    b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 12) + b"IHDR" +
    struct.pack(">II", 1, 1),
    b"\xff\xd8\xff\xc0\x00\x02" + b"\x00" * 8,
])
def test_preflight_rejects_malformed_dimension_headers(payload):
    pf = _preflight()
    with pytest.raises(ValueError, match="malformed"):
        pf.preflight_image_data_url(_image_data_url(payload))


def test_preflight_rejects_oversized_pixel_area():
    pf = _preflight()
    # Both sides under MAX_IMAGE_DIMENSION, product over MAX_IMAGE_PIXELS.
    side = pf.MAX_IMAGE_DIMENSION
    assert side * side > pf.MAX_IMAGE_PIXELS
    with pytest.raises(ValueError, match="decoded pixels"):
        pf.preflight_image_data_url(_image_data_url(_png_bytes(side, side)))


def test_preflight_accepts_dimensions_on_the_limit():
    pf = _preflight()
    side = pf.MAX_IMAGE_DIMENSION
    height = pf.MAX_IMAGE_PIXELS // side
    _, size = pf.preflight_image_data_url(
        _image_data_url(_png_bytes(side, height)))
    assert size == (side, height)


def test_preflight_rejects_oversized_payload():
    pf = _preflight()
    # The encoded-length pre-check trips before any base64 decode allocation.
    oversized = "A" * (-(-pf.MAX_IMAGE_SOURCE_BYTES // 3) * 4 + 4)
    with pytest.raises(ValueError, match="exceeds the supported maximum"):
        pf.preflight_image_data_url(f"data:image/png;base64,{oversized}")


@pytest.mark.parametrize("payload", [
    b"RIFF\x00\x00\x00\x00WEBPVP8 ",
    b"\x00\x01\x02\x03",
    b"P6\n60000 60000\n255\n",
])
def test_preflight_rejects_unmeasurable_formats(payload):
    """Anything the probe cannot measure is refused rather than passed to the
    decoder unbounded -- a PNM header in particular is a one-line bomb."""
    pf = _preflight()
    with pytest.raises(ValueError, match="unsupported image format"):
        pf.preflight_image_data_url(_image_data_url(payload))


@pytest.mark.parametrize("payload", [
    b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHD",
    b"GIF89a\x10",
    b"BM" + b"\x00" * 8,
    b"\xff\xd8\xff\xc0\x00\x0b",
])
def test_preflight_rejects_truncated_headers(payload):
    pf = _preflight()
    with pytest.raises(ValueError):
        pf.preflight_image_data_url(_image_data_url(payload))


def test_preflight_rejects_empty_canvas():
    pf = _preflight()
    with pytest.raises(ValueError, match="empty canvas"):
        pf.preflight_image_data_url(_image_data_url(_png_bytes(0, 32)))


def test_preflight_errors_never_echo_the_payload():
    """Error text reaches the client verbatim in a 400 body, so it must carry
    only measured integers -- no media type, no slice of the base64."""
    pf = _preflight()
    secret = "SECRETMARKER"
    urls = [
        _image_data_url(_png_bytes(60000, 60000), f"image/{secret}"),
        f"data:image/{secret};base64," + "A" * 40 + secret,
        _image_data_url(b"\x00\x01" + secret.encode(), f"image/{secret}"),
    ]
    for url in urls:
        with pytest.raises(ValueError) as excinfo:
            pf.preflight_image_data_url(url)
        assert secret not in str(excinfo.value)


def test_data_url_image_still_loads_from_memory_after_preflight():
    eng = _engine()
    payload = _png_bytes(64, 48)
    buffers = eng._load_image_buffers(
        _StubRt(), _image_messages(_image_data_url(payload)))
    assert buffers == [("image_bytes", payload)]


def test_data_url_image_is_charged_to_the_visual_token_budget(monkeypatch):
    """A data: image used to be invisible to the phase-1 reservation, so a
    video in the same request could claim the whole engine budget."""
    eng = _engine()
    import experimental.server.video_sampling as vs_mod

    limits = {
        "model_type": "qwen2_5_vl",
        "min_image_tokens": 4,
        "max_image_tokens": 4096,
        "max_image_tokens_per_image": 4096,
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
    }
    image_est = vs_mod.estimate_image_tokens_for_size(640, 480, "qwen", limits)
    assert image_est > 0

    seen = []

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        seen.append(budget)
        return ("video", 0), 0, 0, 1

    monkeypatch.setattr(vs_mod, "load_video_buffer", fake_load_video_buffer)

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(_png_bytes(640, 480))
                },
            },
            {
                "type": "video",
                "video": "/nonexistent.mp4"
            },
        ],
    }]
    eng._load_image_buffers(_StubRt(),
                            messages,
                            video_family_fn=lambda: "qwen",
                            video_frame_limits_fn=lambda: limits)
    assert seen == [limits["max_image_tokens"] - image_est]


def test_chat_response_strips_model_terminal_token(tmp_path):
    from experimental.server.api_server import _build_message_body

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({
            "eos_token": "<eos>",
            "eot_token": "<turn|>",
        }))
    tool_config = types.SimpleNamespace(parse_output=False)

    body, has_tool_calls = _build_message_body('{"answer":"ok"}<turn|>',
                                               tool_config, str(tmp_path))

    assert body["content"] == '{"answer":"ok"}'
    assert not has_tool_calls


# ---------------------------------------------------------------------------
# engine.py: video content -> MessageContent("video") + ordered image_buffers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OpenAI HTTP layer (FastAPI TestClient + stub LLM)
# ---------------------------------------------------------------------------
def _make_stub_llm():

    class _Resp:
        output_texts = ["a caption"]
        output_ids = [[1, 2, 3]]
        finish_reasons = []  # -> endpoint defaults finish_reason to "stop"

    class _RT:
        _resp_text = None  # tests may override the canned output

        def handle_request(self, request):
            resp = _Resp()
            if self._resp_text is not None:
                resp.output_texts = [self._resp_text]
            return resp

    class _LLM:
        model_dir = "/fake/qwen3_vl"
        _model_id = "qwen3-vl"
        _rt = None
        has_draft_model = False
        _audio_buffers = ()  # tests may inject decoded-audio stubs

        def __init__(self):
            self._runtime = _RT()
            self.captured = None
            self.captured_params = None
            self.captured_tool_config = None
            # Advertise ASR capability (the endpoint probes for an audio/
            # engine subdir with an ASR-typed config).
            self._multimodal_engine_dir = tempfile.mkdtemp()
            audio_dir = os.path.join(self._multimodal_engine_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            with open(os.path.join(audio_dir, "config.json"),
                      "w",
                      encoding="utf-8") as f:
                f.write('{"model_type": "qwen3_asr"}')

        def _handle_request(self, request):
            return self._runtime.handle_request(request)

        def _admission(self):
            import threading
            sem = self.__dict__.get("_admission_sem")
            if sem is None:
                sem = self.__dict__.setdefault("_admission_sem",
                                               threading.Semaphore(1))
            return sem

        def _make_generation_request(self,
                                     messages,
                                     params,
                                     *,
                                     tools=None,
                                     tool_choice=None,
                                     tool_config=None):
            self.captured = messages
            self.captured_params = params
            self.captured_tool_config = tool_config
            req = types.SimpleNamespace(
                audio_buffers=list(self._audio_buffers))
            return types.SimpleNamespace(requests=[req])

        def count_prompt_tokens(self, messages, **kw):
            return 7

        def generate_stream(self, messages, params, **kw):
            from experimental.server.engine import StreamDelta
            yield StreamDelta(text="hi",
                              token_ids=[1, 2],
                              finished=True,
                              finish_reason="stop")

    return _LLM()


@pytest.fixture
def client_and_llm():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")  # fastapi.testclient needs httpx
    try:
        from fastapi.testclient import TestClient

        from experimental.server.api_server import _create_app
    except ImportError as exc:  # skip only for missing external deps
        _skip_or_raise(exc, "api_server / fastapi TestClient")
    llm = _make_stub_llm()
    # Local media is opt-in; these cases exercise the media pipeline itself.
    return TestClient(_create_app(llm, allowed_local_media_path="/")), llm


def test_health_exposes_context_cache_metrics(client_and_llm):
    client, llm = client_and_llm
    expected = {
        "hit_sequences": 3,
        "reused_tokens": 768,
        "media_aware_sequences": 4,
    }
    llm.context_cache_metrics = lambda: expected

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["context_cache"] == expected


def test_health_allows_runtimes_without_context_cache_metrics(client_and_llm):
    client, _ = client_and_llm

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["context_cache"] is None


def test_streaming_forced_tool_preserves_exact_choice(client_and_llm):
    client, llm = client_and_llm
    response = client.post(
        "/v1/chat/completions",
        json={
            "model":
            "test",
            "messages": [{
                "role": "user",
                "content": "set volume"
            }],
            "max_tokens":
            16,
            "stream":
            True,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "set_volume",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    },
                },
            }],
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": "set_volume"
                },
            },
        },
    )

    assert response.status_code == 200, response.text
    assert llm.captured_tool_config is not None
    assert llm.captured_tool_config.tool_choice == "function"
    assert llm.captured_tool_config.forced_name == "set_volume"


@pytest.mark.parametrize("response_format,expect_json", [("json", True),
                                                         ("text", False)])
def test_audio_transcriptions_endpoint(client_and_llm, response_format,
                                       expect_json):
    client, llm = client_and_llm
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "asr",
            "response_format": response_format
        })
    assert resp.status_code == 200, resp.text
    if expect_json:
        assert resp.json()["text"] == "a caption"
    else:
        assert resp.text == "a caption"
    forwarded = llm.captured[0]["content"]
    assert any(c.get("type") == "input_audio" for c in forwarded)


def test_audio_transcriptions_rejects_empty(client_and_llm):
    client, _ = client_and_llm
    resp = client.post("/v1/audio/transcriptions",
                       files={"file": ("e.wav", b"", "audio/wav")},
                       data={"model": "asr"})
    assert resp.status_code == 400


def test_audio_transcriptions_protocol(client_and_llm):
    # Qwen3-ASR protocol: `language` becomes a system turn, the user turn
    # carries the audio, and the "language <LANG><asr_text><text>" output
    # splits on the delimiter.
    client, llm = client_and_llm
    llm._runtime._resp_text = "language English<asr_text>Hello world"
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "asr",
            "language": "en"
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == "Hello world"
    assert body["language"] == "English"
    assert [m["role"] for m in llm.captured] == ["system", "user"]
    # HF empty-audio sentinel: "language None" prefix yields no language key.
    llm._runtime._resp_text = "language None<asr_text>"
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={"model": "asr"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == ""
    assert "language" not in body
    # No <asr_text> tag: the whole string is the transcription (HF semantics).
    llm._runtime._resp_text = "language None"
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={"model": "asr"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "language None"
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "asr",
            "language": "klingon"
        })
    assert resp.status_code == 400
    # Languages beyond the original short list validate too (official set).
    for code in ("tr", "vi", "hi"):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
            data={
                "model": "asr",
                "language": code
            })
        assert resp.status_code == 200, resp.text
        assert [m["role"] for m in llm.captured] == ["system", "user"]


def test_audio_transcriptions_prompt_is_user_context(client_and_llm):
    # Qwen3-ASR semantics: `language` is the system turn, and the user turn
    # carries the audio followed by the context prompt.
    client, llm = client_and_llm
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "asr",
            "prompt": "NVIDIA TensorRT jargon",
            "language": "en"
        })
    assert resp.status_code == 200, resp.text
    roles = [m["role"] for m in llm.captured]
    assert roles == ["system", "user"]
    assert llm.captured[0]["content"] == "English"
    user_types = [c.get("type") for c in llm.captured[1]["content"]]
    assert user_types == ["input_audio", "text"]


def test_audio_transcriptions_duration_limit(client_and_llm):
    # Decoded PCM longer than the engine's audio profile (builder
    # max_time_steps / 100 s; default 30 s without a config) is a 413 before
    # inference, carrying the actual duration and the cap.
    client, llm = client_and_llm
    llm._audio_buffers = [
        types.SimpleNamespace(num_samples=31 * 16000, sample_rate=16000)
    ]
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={"model": "asr"})
    assert resp.status_code == 413, resp.text
    assert "31.0" in resp.json()["error"]
    assert "30.0" in resp.json()["error"]
    llm._audio_buffers = [
        types.SimpleNamespace(num_samples=29 * 16000, sample_rate=16000)
    ]
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={"model": "asr"})
    assert resp.status_code == 200, resp.text
    # A recorded builder profile overrides the default cap (1000 steps = 10s).
    audio_dir = os.path.join(llm._multimodal_engine_dir, "audio")
    with open(os.path.join(audio_dir, "config.json"), "w",
              encoding="utf-8") as f:
        f.write('{"model_type": "qwen3_asr", '
                '"builder_config": {"max_time_steps": 1000}}')
    llm._audio_buffers = [
        types.SimpleNamespace(num_samples=11 * 16000, sample_rate=16000)
    ]
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={"model": "asr"})
    assert resp.status_code == 413, resp.text
    assert "10.0" in resp.json()["error"]


def test_audio_transcriptions_error_stage_mapping(client_and_llm):
    # Prepare failures (incl. C++ decode RuntimeError) are client errors;
    # infer-stage failures are 500 except the input-too-long marker (413).
    client, llm = client_and_llm

    def _post():
        return client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
            data={"model": "asr"})

    def _prepare_boom(*args, **kwargs):
        raise RuntimeError("Audio decode failed (corrupt bytes)")

    orig_prepare = llm._make_generation_request
    llm._make_generation_request = _prepare_boom
    resp = _post()
    assert resp.status_code == 400
    assert "Invalid audio" in resp.json()["error"]
    llm._make_generation_request = orig_prepare

    def _infer_oom(request):
        raise RuntimeError("CUDA error: out of memory")

    llm._handle_request = _infer_oom
    resp = _post()
    assert resp.status_code == 500

    def _infer_too_long(request):
        raise RuntimeError("EDGELLM_INPUT_TOO_LONG: rebuild with larger "
                           "--maxInputLen")

    llm._handle_request = _infer_too_long
    resp = _post()
    assert resp.status_code == 413


def test_media_count_mismatch_is_client_error():
    """A placeholder/media count mismatch is malformed input, so it must stay a
    400 with the runner's diagnostic rather than a generic 500."""
    from experimental.server.api_server import _inference_error_response
    exc = RuntimeError(
        "EDGELLM_BAD_MEDIA_COUNT: QwenViTRunner::textPreprocess()"
        " pad count exceeds this request's media count")
    resp = _inference_error_response(exc)
    assert resp.status_code == 400
    assert b"EDGELLM_BAD_MEDIA_COUNT" in resp.body


def test_media_count_mismatch_returns_400_over_http(client_and_llm):
    """The marker reaches the client as a 400 through the route, not just
    through the mapper. The C++ -> pybind hop is not covered here."""
    client, llm = client_and_llm

    def _boom(request):
        raise RuntimeError(
            "EDGELLM_BAD_MEDIA_COUNT: QwenViTRunner::textPreprocess() pad count"
            " exceeds this request's media count")

    llm._handle_request = _boom
    resp = client.post("/v1/chat/completions",
                       json={"messages": [{
                           "role": "user",
                           "content": "hi"
                       }]})
    assert resp.status_code == 400, resp.text
    assert "EDGELLM_BAD_MEDIA_COUNT" in resp.text


def test_content_length_parsing():
    # Malformed Content-Length (proxy-injected lists, floats) must map to a
    # clean 400, never an unhandled ValueError in the middleware.
    from experimental.server.api_server import _parse_content_length
    assert _parse_content_length(None) is None
    assert _parse_content_length("1024") == 1024
    assert _parse_content_length("abc") == -1
    assert _parse_content_length("1.5") == -1
    assert _parse_content_length("10, 20") == -1


def test_audio_transcriptions_requires_audio_engine(client_and_llm):
    # A model without an audio encoder must reject transcription up front.
    client, llm = client_and_llm
    saved = llm._multimodal_engine_dir
    llm._multimodal_engine_dir = ""
    try:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
            data={"model": "asr"})
        assert resp.status_code == 400
        assert "audio" in resp.json()["error"]
    finally:
        llm._multimodal_engine_dir = saved


def test_audio_transcriptions_limits(client_and_llm, monkeypatch):
    # Uploads are buffered in memory (plus a base64 copy): oversize is 413,
    # and unsupported response_format values are rejected instead of being
    # silently served as JSON.
    client, _ = client_and_llm
    from experimental.server import api_server
    monkeypatch.setattr(api_server, "MAX_AUDIO_UPLOAD_BYTES", 16)
    resp = client.post("/v1/audio/transcriptions",
                       files={"file": ("big.wav", b"x" * 17, "audio/wav")},
                       data={"model": "asr"})
    assert resp.status_code == 413
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "asr",
            "response_format": "srt"
        })
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Model-family detection
# ---------------------------------------------------------------------------


def test_video_model_family_resolution(tmp_path):
    """All model_type -> family branches of LLM._video_model_family:
    accepted families, rejected model types, and the nested-layout priority
    (the C++ runtime loads <root>/visual first)."""
    eng = _engine()

    def family_for(model_type, nested=None, visual=True, sub=None):
        root = tmp_path / (sub or model_type or "empty")
        root.mkdir(exist_ok=True)
        if visual:
            (root / "visual").mkdir(exist_ok=True)
        if model_type is not None:
            (root / "config.json").write_text('{"model_type": "%s"}' %
                                              model_type)
        if nested is not None:
            (root / "visual" / "config.json").write_text(
                '{"model_type": "%s"}' % nested)
        llm = eng.LLM.__new__(eng.LLM)
        llm._multimodal_engine_dir = str(root)
        return llm._video_model_family()

    assert family_for("internvl") == "internvl"
    assert family_for("qwen3_vl") == "qwen"
    # Nested <root>/visual/config.json wins over a legacy flat config.json.
    assert family_for("qwen3_vl", nested="internvl_chat",
                      sub="nested") == "internvl"
    # No visual engine at all: video must be rejected, not defaulted to qwen.
    with pytest.raises(ValueError, match="not supported"):
        family_for(None, visual=False, sub="none")
    # Model families without a video preprocessing path (phi4mm reads only
    # the first frame) must be rejected up front.
    with pytest.raises(ValueError, match="not supported"):
        family_for("phi4mm")
    # Audio-side omni/asr types share the qwen prefix but have no video path.
    for mt in ("qwen3_omni_audio_encoder", "qwen3_omni_code2wav", "qwen3_asr"):
        with pytest.raises(ValueError, match="not supported"):
            family_for(mt)


def test_video_model_family_nemotron(tmp_path):
    # A visual engine whose config.json model_type is the Nemotron-Omni vision
    # encoder resolves to the "nemotron" frame-sampling family.
    eng = _engine()
    root = tmp_path / "nemo"
    root.mkdir()
    (root / "visual").mkdir()
    (root / "visual" / "config.json"
     ).write_text('{"model_type": "nemotron_omni_vision_encoder"}')
    llm = eng.LLM.__new__(eng.LLM)
    llm._multimodal_engine_dir = str(root)
    assert llm._video_model_family() == "nemotron"


class _FakeVideoBuffer:
    """ImageData stand-in exposing the fields the min-profile check reads."""

    def __init__(self, video, frames):
        self.video = video
        self.frames = frames


def test_load_image_buffers_nemotron_minimum():
    # A Nemotron video buffer is built and its EVS token estimate is honored
    # against the request-wide engine minimum (no cu_seqlens binding).
    eng = _engine()
    limits = {
        "model_type": "nemotron_omni_vision_encoder",
        "min_image_tokens": 256,
        "max_image_tokens": 4096,
        "max_image_tokens_per_image": 4096,
        "video_pruning_rate": 0.0,
        "video_temporal_patch_size": 2,
        "video_target_num_patches": 1024,
        "downsample_ratio": 0.5,
    }

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        # 8 frames = 4 tubelets (T=2), 4*256 EVS tokens, no cu_seqlens groups.
        return _FakeVideoBuffer(item["video"], frames=8), 4 * 256, 0, 0

    import experimental.server.video_sampling as vs_mod
    orig = vs_mod.load_video_buffer
    vs_mod.load_video_buffer = fake_load_video_buffer
    try:
        bufs = eng._load_image_buffers(
            None, [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": "a.mp4"
                }]
            }], lambda: "nemotron", lambda: limits)
        assert [b.video for b in bufs] == ["a.mp4"]
    finally:
        vs_mod.load_video_buffer = orig


def test_load_image_buffers_nemotron_minimum_uses_raw_tubelets():
    # The ViT processes every tubelet, so the engine-minimum check must use the
    # pre-EVS tubelet count, not the pruned estimate: a heavily-pruned clip whose
    # raw tubelets clear the minimum must not be rejected.
    eng = _engine()
    limits = {
        "model_type": "nemotron_omni_vision_encoder",
        "min_image_tokens": 1024,
        "max_image_tokens": 8192,
        "max_image_tokens_per_image": 8192,
        "video_pruning_rate": 0.7,
        "video_temporal_patch_size": 2,
        "video_target_num_patches": 1024,
        "downsample_ratio": 0.5,
    }

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        # 8 frames = 4 tubelets: raw 1024 >= min, but EVS(0.7) prunes to ~307.
        return _FakeVideoBuffer(item["video"], frames=8), 307, 0, 0

    import experimental.server.video_sampling as vs_mod
    orig = vs_mod.load_video_buffer
    vs_mod.load_video_buffer = fake_load_video_buffer
    try:
        bufs = eng._load_image_buffers(
            None, [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": "a.mp4"
                }]
            }], lambda: "nemotron", lambda: limits)
        assert [b.video for b in bufs] == ["a.mp4"]
    finally:
        vs_mod.load_video_buffer = orig


def test_load_image_buffers_nemotron_rejects_multiple_and_mixed():
    # The C++ Nemotron video path is single-video, no mixed images, batch 1; the
    # server rejects other layouts up front instead of crashing the runner.
    eng = _engine()
    two_videos = [{
        "role":
        "user",
        "content": [{
            "type": "video",
            "video": "a.mp4"
        }, {
            "type": "video",
            "video": "b.mp4"
        }],
    }]
    with pytest.raises(ValueError, match="exactly one video"):
        eng._load_image_buffers(None, two_videos, lambda: "nemotron",
                                lambda: {})
    mixed = [{
        "role":
        "user",
        "content": [{
            "type": "image",
            "image": "x.jpg"
        }, {
            "type": "video",
            "video": "a.mp4"
        }],
    }]
    with pytest.raises(ValueError, match="exactly one video"):
        eng._load_image_buffers(None, mixed, lambda: "nemotron", lambda: {})


def test_load_image_buffers_request_wide_internvl_minimum():
    # Two 2-frame videos jointly reach the 4-block engine minimum; one alone
    # must be rejected AFTER accumulation (the bound is request-wide).
    eng = _engine()
    limits = {
        "model_type": "internvl",
        "min_image_tokens": 1024,
        "max_image_tokens": 4096,
        "max_image_tokens_per_image": 512,
    }

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        return ("video", item["video"]), 2 * 256, 0, 2  # 2 frames = 2 blocks

    import experimental.server.video_sampling as vs_mod
    orig = vs_mod.load_video_buffer
    vs_mod.load_video_buffer = fake_load_video_buffer
    try:
        two = eng._load_image_buffers(None, [{
            "role":
            "user",
            "content": [{
                "type": "video",
                "video": "a.mp4"
            }, {
                "type": "video",
                "video": "b.mp4"
            }]
        }], lambda: "internvl", lambda: limits)
        assert len(two) == 2
        # Missing image files are skipped and must NOT count toward the
        # minimum (they produce no buffer, so no blocks).
        with pytest.raises(ValueError, match="at least 1024"):
            eng._load_image_buffers(None, [{
                "role":
                "user",
                "content": [{
                    "type": "video",
                    "video": "a.mp4"
                }] + [{
                    "type": "image",
                    "image": f"/no/such/img{i}.jpg"
                } for i in range(3)]
            }], lambda: "internvl", lambda: limits)
        with pytest.raises(ValueError, match="at least 1024"):
            eng._load_image_buffers(
                None, [{
                    "role": "user",
                    "content": [{
                        "type": "video",
                        "video": "a.mp4"
                    }]
                }], lambda: "internvl", lambda: limits)
    finally:
        vs_mod.load_video_buffer = orig


def test_load_image_buffers_request_wide_qwen_minimum(tmp_path, monkeypatch):
    # The engine minimum is request-wide for Qwen too: resized media are
    # floored per item, but do_resize=false media can undershoot and the C++
    # runner checks the accumulated totalSeqLength against the profile MIN.
    eng = _engine()
    limits = {
        "model_type": "qwen3_vl",
        "min_image_tokens": 128,
        "max_image_tokens": 4096,
        "max_image_tokens_per_image": 4096,
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
    }
    est_by_source = {"tiny.mp4": 8, "a.mp4": 64, "b.mp4": 64}

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        est = est_by_source[item["video"]]
        return ("video", item["video"]), est, 0, 1

    import experimental.server.video_sampling as vs_mod
    monkeypatch.setattr(vs_mod, "load_video_buffer", fake_load_video_buffer)

    def _load(content):
        return eng._load_image_buffers(_StubRt(), [{
            "role": "user",
            "content": content
        }], lambda: "qwen", lambda: limits)

    # A single raw video far below the minimum -> 400 up front, not a C++
    # runtime failure.
    with pytest.raises(ValueError, match="at least 128"):
        _load([{"type": "video", "video": "tiny.mp4"}])
    # Two raw media jointly reaching the minimum pass.
    assert len(
        _load([{
            "type": "video",
            "video": "a.mp4"
        }, {
            "type": "video",
            "video": "b.mp4"
        }])) == 2
    # image + video still short of the minimum -> 400.
    img = tmp_path / "small.jpg"
    img.write_bytes(b"x")
    monkeypatch.setattr(vs_mod,
                        "estimate_image_tokens",
                        lambda path, family, lim, do_resize=True: 32)
    with pytest.raises(ValueError, match="at least 128"):
        _load([{
            "type": "image",
            "image": str(img)
        }, {
            "type": "video",
            "video": "a.mp4"
        }])


def test_load_image_buffers_budget_order_independent(tmp_path, monkeypatch):
    # The video sampler must see the same remaining budget whether the image
    # appears before or after the video, so both orders produce identical
    # buffers (images are probed and reserved up front, phase 1).
    eng = _engine()
    limits = {
        "model_type": "qwen2_5_vl",
        "min_image_tokens": 4,
        "max_image_tokens": 4096,
        "max_image_tokens_per_image": 4096,
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
    }
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    img = tmp_path / "a.mp4"
    container = av.open(str(img), mode="w")
    stream = container.add_stream("mpeg4", rate=1)
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8),
                                       format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    # The REAL estimator is used for the phase-1 reservation (it must accept
    # do_resize=...; a dropped parameter once surfaced as TypeError).
    import experimental.server.video_sampling as vs_mod
    image_est = vs_mod.estimate_image_tokens(str(img), "qwen", limits)
    assert image_est > 0
    seen_budgets = []

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        # The sampler clamps to the remaining budget: consume min(cap, need).
        seen_budgets.append(budget)
        need = 4000
        if budget is not None and budget < need:
            return ("video", budget), budget, 0, 1
        return ("video", need), need, 0, 1

    monkeypatch.setattr(vs_mod, "load_video_buffer", fake_load_video_buffer)

    def items(order):
        content = [{
            "type": "image",
            "image": str(img)
        }, {
            "type": "video",
            "video": "v.mp4"
        }]
        return [{
            "role": "user",
            "content": content if order == "iv" else content[::-1]
        }]

    rt = _StubRt()
    results = {}
    for order in ("iv", "vi"):
        buffers = eng._load_image_buffers(rt, items(order), lambda: "qwen",
                                          lambda: limits)
        video_buf = next(b for b in buffers if b[0] == "video")
        results[order] = video_buf
    # Identical remaining budget in both orders -> identical video buffer.
    assert seen_budgets[0] == seen_budgets[1] == 4096 - image_est
    assert results["iv"] == results["vi"]


def test_stream_disconnect_before_first_byte_releases_admission(
        client_and_llm):
    # A disconnect before the first body frame means the SSE generator never
    # starts, so its finally cannot run; the ASGI-call finally must release.
    client, llm = client_and_llm
    app = client.app
    body = json.dumps({
        "messages": [{
            "role": "user",
            "content": "hi"
        }],
        "stream": True
    }).encode()
    scope = {
        "type":
        "http",
        "http_version":
        "1.1",
        "method":
        "POST",
        "path":
        "/v1/chat/completions",
        "raw_path":
        b"/v1/chat/completions",
        "root_path":
        "",
        "scheme":
        "http",
        "query_string":
        b"",
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
        "client": ("test", 1),
        "server": ("test", 80),
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    class _Disconnect(Exception):
        pass

    async def send(message):
        raise _Disconnect()  # abort on the headers frame

    async def run():
        try:
            await app(scope, receive, send)
        except _Disconnect:
            pass

    asyncio.run(run())
    assert llm._admission().acquire(blocking=False), "admission gate leaked"
    llm._admission().release()


def test_admission_busy_returns_503(client_and_llm):
    # A held runtime slot must fail fast (503 overloaded + Retry-After), never
    # park a server pool thread on acquire: a parked thread starves the sync
    # SSE generator that has to release the slot.
    client, llm = client_and_llm
    assert llm._admission().acquire(blocking=False)
    try:
        body = {"messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 503
        assert resp.headers.get("retry-after") == "1"
        resp = client.post("/v1/chat/completions",
                           json={
                               **body, "stream": True
                           })
        assert resp.status_code == 503
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("c.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
            data={"model": "asr"})
        assert resp.status_code == 503
    finally:
        llm._admission().release()
    # Gate released: the same request now succeeds.
    resp = client.post("/v1/chat/completions",
                       json={"messages": [{
                           "role": "user",
                           "content": "hi"
                       }]})
    assert resp.status_code == 200


def test_admission_spans_decode_and_infer():
    # The admission gate must already be held while media decodes, not just
    # around handle_request: concurrent requests may not stack decoded
    # buffers while queueing on the inference lock.
    eng = _engine()
    llm = eng.LLM.__new__(eng.LLM)
    llm._rt = None
    seen = {}

    def fake_build(messages, params, **kw):
        seen["held_during_build"] = llm._admission()._value == 0
        return "req"

    def fake_handle(request):
        seen["held_during_infer"] = llm._admission()._value == 0

        class _R:
            output_texts = ["ok"]
            output_ids = [[1]]
            finish_reasons = []
            logprobs = []

        return _R()

    llm._make_generation_request = fake_build
    llm._handle_request = fake_handle
    llm._parse_generation_output = (
        lambda text, ids, reason, cfg: eng.CompletionOutput(
            text=text, token_ids=ids, finish_reason=reason))
    out = llm.generate(["hi"], eng.SamplingParams(max_tokens=4))
    assert out and seen["held_during_build"] and seen["held_during_infer"]
    assert llm._admission()._value == 1  # released afterwards


def test_stream_disconnect_keeps_gate_until_worker_exits(monkeypatch):
    # A join timeout after disconnect must NOT release admission: the C++
    # worker may still be inside prefill holding the inference lock. The
    # gate transfers to the worker and frees only when it really exits.
    eng = _engine()
    monkeypatch.setattr(eng, "_STREAM_JOIN_TIMEOUT_S", 0.05)
    llm = eng.LLM.__new__(eng.LLM)
    import threading as _t
    worker_may_exit = _t.Event()
    state = {"cancelled": False}

    class _Chunk:
        text = "t"
        token_ids = [1]
        finished = False
        reason = None
        logprobs = []

    class _Channel:

        def set_skip_special_tokens(self, flag):
            pass

        def wait_pop(self, timeout_ms=0):
            return None if state["cancelled"] else _Chunk()

        def is_finished(self):
            return False

        def is_cancelled(self):
            return state["cancelled"]

        def cancel(self):
            state["cancelled"] = True

    class _RT:

        class StreamChannel:

            @staticmethod
            def create():
                return _Channel()

    llm._rt = _RT()
    llm._handle_request = lambda request: worker_may_exit.wait(timeout=10)

    sem = llm._admission()
    assert sem.acquire(blocking=False)
    from experimental.server import api_server
    handoff = api_server._AdmissionHandoff(sem)

    class _Req:
        stream_channels = None

    gen = llm.generate_stream([],
                              eng.SamplingParams(),
                              prebuilt_request=_Req(),
                              admission_handoff=handoff)
    next(gen)  # worker started
    gen.close()  # disconnect: cancel + join times out; worker still runs
    assert not sem.acquire(blocking=False), "gate released while worker alive"
    worker_may_exit.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        if sem.acquire(blocking=False):
            sem.release()
            break
        time.sleep(0.01)
    else:
        raise AssertionError("worker exit did not release the gate")


def test_stream_close_cancels_channel():
    # Closing the generator early (client disconnect) must cancel the
    # channel so the worker releases the inference lock.
    eng = _engine()
    llm = eng.LLM.__new__(eng.LLM)
    state = {"cancelled": False}

    class _Chunk:
        text = "t"
        token_ids = [1]
        finished = False
        reason = None
        logprobs = []

    class _Channel:

        def set_skip_special_tokens(self, flag):
            pass

        def wait_pop(self, timeout_ms=0):
            return None if state["cancelled"] else _Chunk()

        def is_finished(self):
            return False

        def is_cancelled(self):
            return state["cancelled"]

        def cancel(self):
            state["cancelled"] = True

    class _RT:

        class StreamChannel:

            @staticmethod
            def create():
                return _Channel()

    llm._rt = _RT()
    llm._handle_request = lambda request: None

    class _Req:
        stream_channels = None

    gen = llm.generate_stream([],
                              eng.SamplingParams(max_tokens=4),
                              prebuilt_request=_Req())
    first = next(gen)
    assert first.text == "t"
    gen.close()
    assert state["cancelled"]


def test_cli_parser_builds():
    # Guards against duplicate argparse registrations (a rebase once left two
    # --multimodal-engine-dir definitions and the server could not start;
    # the Jedha CI --help smoke failed on exactly this).
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-m", "experimental.server.api_server", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ, "PYTHONPATH": REPO_ROOT
        })
    assert r.returncode == 0, r.stderr[-500:]
    assert "--multimodal-engine-dir" in r.stdout


def test_video_frame_limits_from_engine_config(tmp_path):
    # builder_config token bounds + preprocessor geometry feed the sampler's
    # profile clamping.
    eng = _engine()
    (tmp_path / "visual").mkdir()
    (tmp_path / "visual" / "config.json").write_text(
        '{"model_type": "qwen2_5_vl", "builder_config": '
        '{"min_image_tokens": 128, "max_image_tokens": 4096, '
        '"max_image_tokens_per_image": 4096}}')
    (tmp_path / "visual" / "preprocessor_config.json").write_text(
        '{"patch_size": 14, "merge_size": 2, "temporal_patch_size": 2}')
    llm = eng.LLM.__new__(eng.LLM)
    llm._multimodal_engine_dir = str(tmp_path)
    limits = llm._video_frame_limits()
    assert limits["max_image_tokens"] == 4096
    assert limits["patch_size"] == 14
    assert limits["model_type"] == "qwen2_5_vl"


def test_video_frame_limits_carry_recorded_cu_capacity(tmp_path):
    # config/server wiring: a cu_seqlens capacity recorded by the builder in
    # builder_config must reach the limits dict the sampler and the engine
    # budget consume, taking precedence over the fallback formula.
    eng = _engine()
    (tmp_path / "visual").mkdir()
    (tmp_path / "visual" / "config.json").write_text(
        '{"model_type": "qwen3_vl", "builder_config": '
        '{"min_image_tokens": 4096, "max_image_tokens": 8192, '
        '"max_cu_seqlen_groups": 512}}')
    (tmp_path / "visual" / "preprocessor_config.json").write_text(
        '{"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}')
    llm = eng.LLM.__new__(eng.LLM)
    llm._multimodal_engine_dir = str(tmp_path)
    limits = llm._video_frame_limits()
    assert limits["max_cu_seqlen_groups"] == 512


def test_chat_streaming_invalid_video_returns_400(chain_client):
    # The streaming path must reject invalid video BEFORE the SSE response
    # starts; without prevalidation the same input is a 400 non-streaming but
    # a 200 SSE with an in-band error.
    client, _ = chain_client
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [{
                                   "type": "video_url",
                                   "video_url": {
                                       "url": "file:///no/such/clip.mp4"
                                   }
                               }]
                           }],
                           "stream":
                           True,
                           "max_tokens":
                           8
                       })
    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# Request-parsing chains: OpenAI JSON in -> parsed generation request out,
# asserting on buffer order, placeholder sequence, and decoded bytes.
# ---------------------------------------------------------------------------


class _CapturingRt:
    """Stub of the pybind module at the C++ boundary; records buffer loads."""

    class Message:

        def __init__(self):
            self.role = ""
            self.contents = []

    class LLMGenerationRequest:
        pass

    class Request:

        def __init__(self, messages=None):
            self.messages = messages

    @staticmethod
    def MessageContent(ctype, data=""):
        c = _Content(ctype, data)
        return c

    @staticmethod
    def load_image_from_path(path):
        return _FakeBuffer("image", path)

    @staticmethod
    def load_video_from_paths(paths, fps, timestamps=()):
        return _FakeBuffer("video_paths", list(paths), fps, list(timestamps))

    @staticmethod
    def load_video_from_array(frames, fps, timestamps=()):
        return _FakeBuffer("video_array", frames, fps, list(timestamps))

    @staticmethod
    def load_audio_buffer_from_bytes(raw):
        return ("audio_bytes", raw)


@pytest.fixture
def chain_client(tmp_path):
    """TestClient over a real _create_app + a real-code LLM whose only stubs
    are the pybind module and the engine runtime (captures the request)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    try:
        from fastapi.testclient import TestClient

        from experimental.server.api_server import _create_app
        from experimental.server.engine import LLM
    except ImportError as exc:  # skip only for missing external deps
        _skip_or_raise(exc, "server imports")

    captured = {}

    class _RT:

        @staticmethod
        def has_draft_model():
            return False

        def handle_request(self, request):
            captured["request"] = request

            class _Resp:
                output_texts = ["ok"]
                output_ids = [[1]]
                finish_reasons = []

            return _Resp()

    llm = LLM.__new__(LLM)  # bypass __init__: no engine on a unit host
    llm._rt = _CapturingRt()
    llm._runtime = _RT()
    llm._model_dir = str(tmp_path)
    llm._model_id = "chain-test"
    visual_dir = tmp_path / "mm" / "visual"
    visual_dir.mkdir(parents=True)
    (visual_dir / "config.json").write_text('{"model_type": "qwen3_vl"}')
    llm._multimodal_engine_dir = str(tmp_path / "mm")
    llm._tool_template_formatter = None
    return TestClient(_create_app(llm, allowed_local_media_path="/")), captured


def test_chat_text_request_parses(chain_client):
    client, captured = chain_client
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role": "user",
                               "content": "Hello there"
                           }],
                           "max_tokens": 8,
                       })
    assert resp.status_code == 200, resp.text
    req = captured["request"].requests[0]
    assert [c.type for c in req.messages[0].contents] == ["text"]
    assert req.image_buffers == [] and req.audio_buffers == []


def test_chat_video_frames_request_parses_to_ordered_buffers(
        chain_client, tmp_path):
    client, captured = chain_client
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    f0, f1 = tmp_path / "f0.jpg", tmp_path / "f1.jpg"
    f0.write_bytes(b"x")
    f1.write_bytes(b"x")
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [
                                   {
                                       "type": "image",
                                       "image": str(img)
                                   },
                                   {
                                       "type": "video",
                                       "frames": [str(f0), str(f1)],
                                       "fps": 1.0
                                   },
                                   {
                                       "type": "text",
                                       "text": "describe"
                                   },
                               ],
                           }],
                           "max_tokens":
                           8,
                       })
    assert resp.status_code == 200, resp.text
    req = captured["request"].requests[0]
    # Buffers arrive in message order (the C++ runner pairs them positionally
    # with the visual placeholders).
    assert req.image_buffers == [("image", str(img)),
                                 ("video_paths", [str(f0),
                                                  str(f1)], 1.0, [0.0, 1.0])]
    # The chat-template message carries the placeholder sequence.
    types = [c.type for c in req.messages[0].contents]
    assert types == ["image", "video", "text"]


def test_chat_video_request_end_to_end(chain_client, monkeypatch):
    # One streaming request covers the whole chain: the generation request is
    # built before the SSE response and reused in the generator (exactly one
    # decode); sampler args and buffer + timestamps arrive untouched.
    client, captured = chain_client
    from experimental.server import video_sampling
    sentinel = object()
    calls = {"n": 0}
    seen = {}

    def fake_sample_video(source, **kw):
        calls["n"] += 1
        seen["source"] = source
        seen["kw"] = kw
        return sentinel, 2.0, [0.0, 0.5], 0, 0

    monkeypatch.setattr(video_sampling, "sample_video", fake_sample_video)
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [{
                                   "type": "video",
                                   "video": "/clips/x.mp4",
                                   "nframes": 8
                               }, {
                                   "type": "text",
                                   "text": "describe"
                               }]
                           }],
                           "stream":
                           True,
                           "max_tokens":
                           8
                       })
    assert resp.status_code == 200, resp.text
    assert calls["n"] == 1
    assert seen["source"] == "/clips/x.mp4" and seen["kw"]["nframes"] == 8
    assert seen["kw"]["family"] == "qwen"
    # The streaming stub short-circuits before the C++ request; a non-stream
    # round-trip checks the buffer + timestamps reach it untouched.
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [{
                                   "type": "video",
                                   "video": "/clips/x.mp4",
                                   "nframes": 8
                               }],
                           }],
                           "max_tokens":
                           8,
                       })
    assert resp.status_code == 200, resp.text
    req = captured["request"].requests[0]
    assert req.image_buffers == [("video_array", sentinel, 2.0, [0.0, 0.5])]


@pytest.mark.parametrize("content", [
    {
        "type": "video_url",
        "video_url": {
            "url": "https://evil/clip.mp4"
        }
    },
    {
        "type": "video_url",
        "video_url": {
            "url": "file:///no/such/clip.mp4"
        }
    },
    {
        "type": "input_audio",
        "input_audio": {}
    },
])
def test_chat_rejects_bad_media_via_api(chain_client, content):
    # The security/validation boundary holds through the full request chain:
    # remote video URLs and malformed audio come back as 400, not 500.
    client, _ = chain_client
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role": "user",
                               "content": [content]
                           }],
                           "max_tokens": 8
                       })
    assert resp.status_code == 400, resp.text


def test_chat_audio_request_parses_to_bytes(chain_client):
    import base64
    client, captured = chain_client
    raw = b"RIFF0000WAVEfmt "
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [{
                                   "type": "input_audio",
                                   "input_audio": {
                                       "data": base64.b64encode(raw).decode(),
                                       "format": "wav"
                                   },
                               }],
                           }],
                           "max_tokens":
                           8,
                       })
    assert resp.status_code == 200, resp.text
    req = captured["request"].requests[0]
    assert req.audio_buffers == [("audio_bytes", raw)]


def test_chat_audio_data_url_over_cap_returns_400(chain_client, monkeypatch):
    # Chat audio shares the transcription upload cap: an oversized data: URL
    # is rejected as a 400 before it is buffered/decoded.
    import base64
    client, _ = chain_client
    from experimental.server import audio_preprocess
    monkeypatch.setattr(audio_preprocess, "MAX_AUDIO_UPLOAD_BYTES", 16)
    payload = base64.b64encode(b"x" * 17).decode()
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [{
                                   "type": "audio_url",
                                   "audio_url": {
                                       "url":
                                       "data:audio/wav;base64," + payload
                                   },
                               }],
                           }],
                           "max_tokens":
                           8,
                       })
    assert resp.status_code == 400, resp.text
    assert "exceeds" in resp.json()["error"]


# ---------------------------------------------------------------------------
# /v1/completions: OpenAI legacy text-completions endpoint
# ---------------------------------------------------------------------------


def test_completions_non_stream(client_and_llm):
    client, llm = client_and_llm
    llm._runtime._resp_text = "a completion"
    resp = client.post("/v1/completions",
                       json={
                           "prompt": "Once upon a time",
                           "max_tokens": 8
                       })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    choice = body["choices"][0]
    assert choice["text"] == "a completion"
    assert choice["finish_reason"] == "stop"
    assert choice["logprobs"] is None
    assert body["usage"]["completion_tokens"] == 3
    # The prompt reaches the request builder as a single verbatim user turn.
    assert llm.captured == [{"role": "user", "content": "Once upon a time"}]


def test_completions_raw_prompt_chain(chain_client):
    # Legacy completions must NOT apply the chat template: the prompt text
    # reaches the C++ request verbatim with templating disabled.
    client, captured = chain_client
    resp = client.post("/v1/completions",
                       json={
                           "prompt": "2+2=",
                           "max_tokens": 8
                       })
    assert resp.status_code == 200, resp.text
    request = captured["request"]
    assert request.apply_chat_template is False
    assert request.add_generation_prompt is False
    contents = request.requests[0].messages[0].contents
    assert [c.type for c in contents] == ["text"]
    assert contents[0].data == "2+2="


def test_completions_stream(client_and_llm):
    import json

    client, llm = client_and_llm
    deltas = [
        types.SimpleNamespace(text="Hello", finished=False,
                              finish_reason=None),
        types.SimpleNamespace(text=" world",
                              finished=True,
                              finish_reason="stop"),
    ]

    def fake_stream(messages, params, prebuilt_request=None, **kw):
        assert prebuilt_request is not None
        yield from deltas

    llm.generate_stream = fake_stream
    resp = client.post("/v1/completions",
                       json={
                           "prompt": "Hi",
                           "stream": True,
                           "max_tokens": 8
                       })
    assert resp.status_code == 200, resp.text
    lines = [l for l in resp.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[len("data: "):]) for l in lines[:-1]]
    assert all(c["object"] == "text_completion" for c in chunks)
    assert [c["choices"][0]["text"] for c in chunks] == ["Hello", " world", ""]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # Admission released once the stream drains.
    assert llm._admission().acquire(blocking=False)
    llm._admission().release()


def test_completions_rejects_bad_prompt(client_and_llm):
    client, _ = client_and_llm
    resp = client.post("/v1/completions", json={"max_tokens": 8})
    assert resp.status_code == 400
    assert "error" in resp.json()
    resp = client.post("/v1/completions", json={"prompt": ["a", "b"]})
    assert resp.status_code == 400
    assert "batch" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Local-media policy, sampling validation, usage
# ---------------------------------------------------------------------------


def _msgs(ref):
    return [{"role": "user", "content": [{"type": "audio", "audio": ref}]}]


def test_local_media_rejected_when_unset():
    from experimental.server.api_server import enforce_local_media_policy
    with pytest.raises(PermissionError):
        enforce_local_media_policy(_msgs("/etc/passwd"), None)
    with pytest.raises(PermissionError):
        enforce_local_media_policy(_msgs("file:///etc/passwd"), None)


def test_local_media_every_accepted_spelling_is_policed():
    """The policy reads every media key in both spellings, so no form the
    loaders accept -- notably video's {"url": ...} -- can slip past it."""
    from experimental.server.api_server import enforce_local_media_policy
    items = [
        {
            "type": "video",
            "video": "/etc/passwd.mp4"
        },
        {
            "type": "video",
            "video": {
                "url": "/etc/passwd.mp4"
            }
        },
        {
            "type": "video_url",
            "video_url": {
                "url": "/etc/passwd.mp4"
            }
        },
        {
            "type": "video",
            "frames": ["/etc/passwd.png"]
        },
        {
            "type": "image",
            "image": {
                "url": "/etc/passwd.png"
            }
        },
        {
            "type": "audio",
            "audio": {
                "url": "/etc/passwd.wav"
            }
        },
    ]
    for item in items:
        messages = [{"role": "user", "content": [item]}]
        with pytest.raises(PermissionError):
            enforce_local_media_policy(messages, None)


def test_local_media_allowed_inside_root(tmp_path):
    from experimental.server.api_server import enforce_local_media_policy
    media = tmp_path / "clip.wav"
    media.write_bytes(b"")
    enforce_local_media_policy(_msgs(str(media)), str(tmp_path))


def test_local_media_escape_rejected(tmp_path):
    from experimental.server.api_server import enforce_local_media_policy
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"")
    with pytest.raises(PermissionError):
        enforce_local_media_policy(_msgs(str(root / ".." / "outside.wav")),
                                   str(root))


def test_data_and_http_refs_bypass_local_policy():
    from experimental.server.api_server import enforce_local_media_policy
    enforce_local_media_policy(_msgs("data:audio/wav;base64,AAAA"), None)
    enforce_local_media_policy(_msgs("https://example.com/a.wav"), None)


@pytest.mark.parametrize("body", [
    {
        "temperature": "hot"
    },
    {
        "max_tokens": 0
    },
    {
        "top_p": 2.0
    },
    {
        "top_k": -1
    },
    {
        "temperature": float("inf")
    },
])
def test_sampling_params_rejected(body):
    from experimental.server.api_server import parse_sampling_params
    with pytest.raises(ValueError):
        parse_sampling_params(body, default_max_tokens=16)


def test_sampling_params_defaults():
    from experimental.server.api_server import parse_sampling_params
    out = parse_sampling_params({}, default_max_tokens=16)
    assert out["max_tokens"] == 16 and out["top_k"] == 50


# ---------------------------------------------------------------------------
# Issue #180: temperature=0 normalized to the greedy tuple at ingress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0.0, 0, 1e-9, 9.99e-4])
def test_greedy_temperature_pins_top_p_and_top_k(temperature):
    from experimental.server.engine import normalize_greedy_sampling
    assert normalize_greedy_sampling(temperature, 0.9, 50) == (1.0, 1)


@pytest.mark.parametrize("temperature", [1e-3, 0.7, 1.0, 2.0])
def test_sampling_temperature_leaves_top_p_and_top_k_alone(temperature):
    from experimental.server.engine import normalize_greedy_sampling
    assert normalize_greedy_sampling(temperature, 0.9, 50) == (0.9, 50)


def test_negative_temperature_is_left_for_the_native_range_check():
    from experimental.server.engine import normalize_greedy_sampling
    assert normalize_greedy_sampling(-1.0, 0.9, 50) == (0.9, 50)


def test_sampling_params_normalizes_bare_temperature_zero():
    """The native SamplingParams constructor rewrites topK/topP and logs a
    warning for every sub-threshold temperature it is handed with topK != 1;
    normalizing here keeps it off that branch however often the request is
    re-validated."""
    from experimental.server.engine import SamplingParams
    params = SamplingParams(temperature=0.0)
    assert (params.top_p, params.top_k) == (1.0, 1)
    assert params.temperature == 0.0


def test_greedy_requests_share_one_batch_key():
    """top_p/top_k are batch-compatibility fields: an un-normalized
    temperature=0 request could not batch with an explicitly greedy one."""
    from experimental.server.batching import BATCH_COMPATIBILITY_FIELDS
    from experimental.server.engine import SamplingParams

    bare = SamplingParams(temperature=0.0)
    explicit = SamplingParams(temperature=0.0, top_p=1.0, top_k=1)
    assert (bare.top_p, bare.top_k) == (explicit.top_p, explicit.top_k)
    for field in ("temperature", "top_p", "top_k"):
        assert field in BATCH_COMPATIBILITY_FIELDS


def test_audio_params_normalizes_greedy_talker_temperature():
    """The talker path builds a native SamplingParams per decode step, so an
    un-normalized greedy request warns once per step."""
    from experimental.server.engine import AudioParams
    params = AudioParams(talker_temperature=0.0)
    assert (params.talker_top_p, params.talker_top_k) == (1.0, 1)


def test_talker_knobs_renormalize_after_setattr():
    from experimental.server.api_server import _apply_talker_knobs
    from experimental.server.engine import AudioParams

    params = AudioParams()
    assert _apply_talker_knobs({"talker_temperature": 0}, params, "") is None
    assert (params.talker_top_p, params.talker_top_k) == (1.0, 1)


def test_chat_temperature_zero_reaches_the_runtime_as_greedy(client_and_llm):
    client, llm = client_and_llm
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{
                "role": "user",
                "content": "hi"
            }],
            "max_tokens": 8,
            "temperature": 0,
        },
    )
    assert response.status_code == 200, response.text
    assert llm.captured_params.temperature == 0.0
    assert llm.captured_params.top_p == 1.0
    assert llm.captured_params.top_k == 1


def test_usage_uses_runtime_prompt_token_counts():
    """The runtime count wins over the HF-template estimate, which undercounts
    multimodal placeholders; the estimate is the fallback when it is absent."""
    from experimental.server.api_server import _runtime_prompt_tokens

    class _Resp:
        prompt_token_counts = [7, 9]

    assert _runtime_prompt_tokens(_Resp(), 1, 4) == 9
    assert _runtime_prompt_tokens(object(), 0, 4) == 4


def test_stream_usage_chunk_when_requested(client_and_llm):
    """stream_options.include_usage adds a final choices-less usage chunk."""

    client, llm = client_and_llm

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hi"
            }],
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
        },
    )
    assert resp.status_code == 200
    chunks = [
        json.loads(line[len("data: "):]) for line in resp.text.splitlines()
        if line.startswith("data: ") and "[DONE]" not in line
    ]
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["choices"] == []
    assert usage_chunks[0]["usage"]["prompt_tokens"] == 7


def test_stream_usage_absent_by_default(client_and_llm):
    client, _ = client_and_llm
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hi"
            }],
            "stream": True
        },
    )
    assert resp.status_code == 200
    assert '"usage"' not in resp.text


def test_serve_forwards_allowed_local_media_path(monkeypatch):
    """LLM.serve must accept and forward every run_server kwarg the CLI passes."""
    import inspect

    from experimental.server import api_server
    from experimental.server.engine import LLM

    serve_params = inspect.signature(LLM.serve).parameters
    run_params = inspect.signature(api_server.run_server).parameters
    for name in run_params:
        if name in ("llm_instance", "host", "port"):
            continue
        assert name in serve_params, f"LLM.serve is missing {name}"


def test_batch_slice_carries_prompt_token_counts():
    """Batched rows must keep their own prompt length, or usage reports 0."""
    from experimental.server.batching import _copy_response_rows

    class _Resp:
        output_texts = ["a", "b", "c"]
        output_ids = [[1], [2, 3], [4]]
        finish_reasons = ["stop"] * 3
        logprobs = [[], [], []]
        prompt_token_counts = [10, 615, 59]

    sliced = _copy_response_rows(_Resp(), 1, 2)
    assert sliced.prompt_token_counts == [615, 59]


def test_cli_main_wires_through_to_app(monkeypatch):
    """Chain test for the CLI layer: argv -> main() -> LLM.serve() ->
    run_server() -> _create_app(). The other tests enter at _create_app, so
    only this one catches a kwarg dropped in an intermediate layer."""
    import sys

    from experimental.server import api_server
    from experimental.server.engine import LLM

    captured = {}

    def _fake_llm_init(self, **kwargs):
        self._eagle_engine_dir = ""
        self._tool_template_formatter = None
        self._model_id = "test"
        self._multimodal_engine_dir = ""
        self._runtime = None
        self._rt = None
        captured["llm"] = kwargs

    def _fake_uvicorn_run(app, **kwargs):
        captured["served"] = kwargs

    monkeypatch.setattr(LLM, "__init__", _fake_llm_init)
    monkeypatch.setattr(api_server, "_create_app",
                        lambda llm, **kw: captured.setdefault("app", kw))
    fake_uvicorn = type("_U", (), {"run": staticmethod(_fake_uvicorn_run)})
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--model", "hf/model", "--port", "9",
        "--allowed-local-media-path", "/srv/media"
    ])

    api_server.main()

    # The flag survived every layer down to the app factory.
    assert captured["app"]["allowed_local_media_path"] == "/srv/media"
    assert captured["served"]["port"] == 9


def test_oversized_json_body_rejected(client_and_llm, monkeypatch):
    """Chat bodies are capped by bytes received, not by Content-Length."""
    from experimental.server import api_server

    client, _ = client_and_llm
    monkeypatch.setattr(api_server, "MAX_REQUEST_BODY_BYTES", 512)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{
            "role": "user",
            "content": "x" * 4096
        }]},
    )
    assert resp.status_code == 413


def test_normal_body_still_accepted(client_and_llm):
    client, _ = client_and_llm
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{
            "role": "user",
            "content": "hi"
        }]},
    )
    assert resp.status_code == 200


def test_tool_stream_usage_chunk_when_requested(client_and_llm):
    """The tool-calling stream is a separate generator; it must emit usage too."""

    from experimental.server.engine import StreamDelta

    client, llm = client_and_llm

    def fake_stream(messages, params, **kw):
        yield StreamDelta(text="ok",
                          token_ids=[1],
                          finished=True,
                          finish_reason="stop")

    llm.generate_stream = fake_stream
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hi"
            }],
            "stream":
            True,
            "stream_options": {
                "include_usage": True
            },
            "tools": [{
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }],
        },
    )
    assert resp.status_code == 200
    chunks = [
        json.loads(line[len("data: "):]) for line in resp.text.splitlines()
        if line.startswith("data: ") and "[DONE]" not in line
    ]
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"]["prompt_tokens"] == 7


@pytest.mark.parametrize("output", [
    "call:weather{location:Berkeley, California}It is sunny.",
    "call:unknown{}Success.",
    "<|tool_call>call:unknown{}<tool_call|>Success.",
    "<|tool_call>call:weather{location:Berkeley}",
])
@pytest.mark.parametrize("stream", [False, True])
def test_tool_protocol_error_http_never_returns_model_content(
        client_and_llm, output, stream):
    from experimental.server.engine import StreamDelta

    client, llm = client_and_llm
    llm.model_dir = "/fake/gemma4"
    llm._runtime._resp_text = output
    closed = []

    def fake_stream(messages, params, **kw):
        handoff = kw["admission_handoff"]
        handoff.worker_started()
        try:
            for char in output:
                yield StreamDelta(text=char, token_ids=[1], finished=False)
            yield StreamDelta(text="", finished=True, finish_reason="stop")
        finally:
            closed.append(True)
            handoff.release()

    llm.generate_stream = fake_stream
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "Weather?"
                               }],
                               "stream":
                               stream,
                               "tools": [{
                                   "type": "function",
                                   "function": {
                                       "name": "weather",
                                       "parameters": {
                                           "type": "object",
                                           "properties": {
                                               "location": {
                                                   "type": "string"
                                               }
                                           }
                                       }
                                   }
                               }],
                           })
    if not stream:
        assert response.status_code == 502
        payload = response.json()
        assert set(payload) == {"error"}
        assert payload["error"]["code"] == "invalid_tool_call"
    else:
        assert response.status_code == 200
        payloads = [
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        assert [p["error"]["code"] for p in payloads
                if "error" in p] == ["invalid_tool_call"]
        choices = [p["choices"][0] for p in payloads if "choices" in p]
        assert not any("content" in c["delta"] or "tool_calls" in c["delta"]
                       for c in choices)
        assert choices[-1]["finish_reason"] == "error"
        assert response.text.endswith("data: [DONE]\n\n")
    assert output not in response.text
    assert "Berkeley" not in response.text
    assert "Success" not in response.text
    if stream:
        assert closed == [True]
    llm._runtime._resp_text = "Okay."
    following = client.post(
        "/v1/chat/completions",
        json={"messages": [{
            "role": "user",
            "content": "Hi"
        }]})
    assert following.status_code == 200
    assert following.json()["choices"][0]["message"]["content"] == "Okay."


def test_local_media_frames_rejected_when_unset(tmp_path):
    """`{"type": "video", "frames": [...]}` is a local-path entry point too."""
    from experimental.server.api_server import enforce_local_media_policy

    msgs = [{
        "role":
        "user",
        "content": [{
            "type": "video",
            "frames": ["/etc/passwd", "/etc/hosts"],
            "fps": 1.0
        }]
    }]
    with pytest.raises(PermissionError):
        enforce_local_media_policy(msgs, None)

    root = tmp_path / "root"
    root.mkdir()
    inside = root / "f0.png"
    inside.write_bytes(b"")
    enforce_local_media_policy(
        [{
            "role": "user",
            "content": [{
                "type": "video",
                "frames": [str(inside)]
            }]
        }], str(root))

    outside = tmp_path / "out.png"
    outside.write_bytes(b"")
    with pytest.raises(PermissionError):
        enforce_local_media_policy(
            [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "frames": [str(outside)]
                }]
            }], str(root))


@pytest.mark.parametrize("body", [{"top_k": 1.9}, {"max_tokens": 2.9}])
def test_integer_params_reject_floats(body):
    from experimental.server.api_server import parse_sampling_params
    with pytest.raises(ValueError):
        parse_sampling_params(body, default_max_tokens=16)


def test_include_usage_rejects_non_bool(client_and_llm):
    client, _ = client_and_llm
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hi"
            }],
            "stream": True,
            "stream_options": {
                "include_usage": "false"
            },
        },
    )
    assert resp.status_code == 400
