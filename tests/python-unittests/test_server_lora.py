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
"""LoRA adapter serving: discovery, configuration, selection and switching.

Only the pybind module is faked; every test drives the real ``LLM`` built
from an engine tree under ``tmp_path``.
"""

import asyncio
import hashlib
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file as save_numpy_safetensors

from tensorrt_edgellm import lora_package

from experimental.server.config import (ApiConfig, LoraAdapterConfig,
                                        ModelConfig, ServerConfigError,
                                        parse_server_config)
from experimental.server.runtime.engine import (LLM, SamplingParams,
                                                finish_reason_name)

# ---------------------------------------------------------------------------
# Fake pybind runtime: records the adapter each request carries
# ---------------------------------------------------------------------------


class _FakeChunk:
    """One StreamChannel pop: a full turn in a single terminal delta."""

    text = "hello"
    token_ids = [1, 2]
    prompt_token_count = 7
    finished = True
    reason = None
    logprobs = []

    @property
    def metrics(self):
        return None


class _FakeChannel:
    """Minimal native StreamChannel: one chunk, then finished."""

    def __init__(self):
        self._chunks = [_FakeChunk()]
        self._finished = False

    def set_skip_special_tokens(self, _enabled):
        pass

    def wait_pop(self, timeout_ms=0):
        if self._chunks:
            return self._chunks.pop(0)
        self._finished = True
        return None

    def try_pop(self):
        return self._chunks.pop(0) if self._chunks else None

    def is_finished(self):
        return self._finished

    def is_cancelled(self):
        return False

    def cancel(self):
        self._finished = True


class _FakeLLMRuntime:
    """Blocking-path runtime: records ``lora_weights_name`` per request."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.switch_names = []
        self._lock = threading.Lock()

    def record(self, request):
        with self._lock:
            self.switch_names.append(request.lora_weights_name)

    def capture_decoding_cuda_graph(self):
        return True

    def has_draft_model(self):
        return False

    def handle_request(self, request):
        self.record(request)
        return SimpleNamespace(
            output_texts=["hello"],
            output_ids=[[1, 2, 3]],
            prompt_token_counts=[7],
            finish_reasons=["stop"],
            spec_verify_counts=[],
            spec_acceptance_lengths=[],
            logprobs=[],
        )


class _FakeContextCacheConfig:

    def __init__(self):
        self.enabled = False
        self.max_records = 0
        self.recurrent_snapshot_pool_bytes = 0
        self.partial_kv_snapshot_pool_bytes = 0
        self.encoder_embedding_cache_budget_bytes = -1


class _NativeValue:
    """pybind value type stand-in: accepts positional args and attributes."""

    def __init__(self, *args, **values):
        self.args = args
        self.__dict__.update(values)


class _FakeRuntimeModule:
    """The pybind surface ``LLM`` touches on the blocking request path."""

    LLMRuntime = _FakeLLMRuntime
    ContextCacheConfig = _FakeContextCacheConfig
    FinishReason = SimpleNamespace(NOT_FINISHED="not_finished",
                                   END_ID="end_id",
                                   LENGTH="length",
                                   CANCELLED="cancelled",
                                   ERROR="error",
                                   STOP_WORDS="stop_words")
    Message = _NativeValue
    MessageContent = _NativeValue
    MessageToolCall = _NativeValue
    LLMGenerationRequest = _NativeValue
    Request = _NativeValue
    ToolChoice = _NativeValue
    ToolDefinition = _NativeValue
    ToolChoiceMode = SimpleNamespace(NONE="none",
                                     AUTO="auto",
                                     REQUIRED="required",
                                     FUNCTION="function")
    ContextCacheLookupPolicy = SimpleNamespace(USE_CACHE="use",
                                               BYPASS="bypass")
    ContextCacheCommitPolicy = SimpleNamespace(
        INCLUDING_GENERATED_TOKENS="all", PREFILL_STATE_ONLY="prefill")

    class StreamChannel:

        @staticmethod
        def create(*, collect_metrics=False):
            return _FakeChannel()

    @staticmethod
    def load_image_from_path(_path):
        raise AssertionError("no media in LoRA tests")

    @staticmethod
    def load_image_from_bytes(_data):
        raise AssertionError("no media in LoRA tests")

    @staticmethod
    def load_audio_buffer_from_bytes(_data):
        raise AssertionError("no audio in LoRA tests")


#: Fixed served name: the engine tree's directory name would otherwise leak
#: tmp_path into every model alias.
SERVED_MODEL = "qwen3"

# ---------------------------------------------------------------------------
# Fixtures: a prebuilt engine tree, with or without adapter sidecars
# ---------------------------------------------------------------------------


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _engine_tree(tmp_path: Path, max_lora_rank: int = 0) -> Path:
    bundle = tmp_path / "engines"
    (bundle / "llm").mkdir(parents=True)
    (bundle / "llm" / "llm.engine").touch()
    _write_json(
        bundle / "llm" / "config.json",
        {
            "model": "qwen3",
            "model_type": "qwen3",
            "spec_decode_type": "none",
            "engine_role": "llm",
            "builder_config": {
                "max_input_len": 32,
                "max_batch_size": 1,
                "max_kv_cache_capacity": 128,
                "max_lora_rank": max_lora_rank,
            },
        },
    )
    lora_package.write_base_identity(str(bundle / "llm"))
    return bundle


def _adapter(engine_dir: Path,
             name: str,
             rank=None,
             *,
             manifest: bool = True,
             base_identity=None) -> Path:
    """A processed adapter directory: real FP16 factors for two modules, an
    ``edgellm_lora_processing`` record when ``rank`` is given, and a manifest
    bound to the engine directory's base identity (or ``base_identity``)."""
    directory = engine_dir / "lora_weights" / name
    directory.mkdir(parents=True)
    file_rank = rank or 4
    stems = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"]
    tensors = {}
    for stem in stems:
        tensors[f"{stem}.lora_A.weight"] = np.zeros((8, file_rank), np.float16)
        tensors[f"{stem}.lora_B.weight"] = np.zeros((file_rank, 8), np.float16)
    weights = directory / "processed_adapter_model.safetensors"
    save_numpy_safetensors(tensors, str(weights))
    if rank is not None:
        _write_json(
            directory / "config.json", {
                "edgellm_lora_processing": {
                    "processing_schema_version": 1,
                    "rank": rank,
                    "lora_scale": 1.0,
                    "max_lora_rank": 64,
                    "bound_modules": ["a", "b"],
                    "unbound_untargeted_bindings": ["c"],
                }
            })
    if manifest:
        if base_identity is None:
            base_identity = lora_package.read_base_identity(
                str(engine_dir))["digest"]
        _write_json(
            directory / lora_package.ADAPTER_MANIFEST_FILE, {
                "schema": 1,
                "version": f"{name}-v1",
                "base_identity": base_identity,
                "weights": {
                    "file": weights.name,
                    "size": weights.stat().st_size,
                    "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                },
                "rank": file_rank,
                "lora_scale": 1.0,
                "processing_schema_version": 1,
                "targets": sorted(stems),
                "source_adapter_sha256": None,
                "created_utc": "2026-11-12T00:00:00Z",
            })
    return weights


@pytest.fixture
def _fake_runtime(monkeypatch):
    """Swap the pybind module for the recording fake in every HTTP-path test."""
    monkeypatch.setattr("experimental.server.runtime.engine._import_runtime",
                        lambda: _FakeRuntimeModule)


def _build_llm(tmp_path: Path, *, max_lora_rank: int = 16) -> LLM:
    """Real LLM over the fake pybind module, with french + poetic adapters.

    ``french`` carries an ``edgellm_lora_processing`` record; ``poetic`` is a
    bare sidecar without one, so both health-report shapes are exercised.
    Requires the ``_fake_runtime`` fixture.
    """
    bundle = _engine_tree(tmp_path, max_lora_rank=max_lora_rank)
    _adapter(bundle / "llm", "french", rank=8)
    _adapter(bundle / "llm", "poetic")
    return LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovered_adapters_come_from_the_lora_weights_directory(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "french", rank=8)
    (bundle / "llm" / "lora_weights" / "empty").mkdir()
    (bundle / "llm" / "lora_weights" / "loose.safetensors").touch()

    llm = LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))

    assert llm.lora_names() == ["french"]
    assert llm.max_lora_rank == 16


def test_discovery_ignores_hidden_entries_and_plain_files(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "tool", rank=8)
    _adapter(bundle / "llm", ".tool-v1-prev", rank=8)
    (bundle / "llm" / "lora_weights" / "README").write_text("notes")

    llm = LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))

    assert llm.lora_names() == ["tool"]
    # The name -> path map handed to the native runtime.
    assert sorted(llm._runtime.args[2]) == ["tool"]


def test_discovery_stops_on_an_invalid_adapter_directory_name(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    invalid = bundle / "llm" / "lora_weights" / " pirate "
    invalid.mkdir(parents=True)
    (invalid / "processed_adapter_model.safetensors").touch()

    with pytest.raises(ServerConfigError, match=" pirate "):
        LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))


def test_discovery_skips_a_sidecarless_directory_with_a_warning(
        tmp_path, _fake_runtime, caplog):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "french", rank=8)
    stray = bundle / "llm" / "lora_weights" / "stray"
    stray.mkdir()
    (stray / "unrelated.bin").touch()

    with caplog.at_level(logging.WARNING, logger="edgellm.server"):
        llm = LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))

    assert llm.lora_names() == ["french"]
    assert any("stray" in record.message and record.levelno == logging.WARNING
               for record in caplog.records)


def test_adapters_without_a_sidecar_directory_are_not_discovered(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    llm = LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))
    assert llm.lora_names() == []
    assert llm.max_lora_rank == 16


def test_discovery_with_no_lora_weights_directory_yields_no_adapters(tmp_path):
    from experimental.server.runtime.engine import _resolve_lora_adapters
    assert _resolve_lora_adapters(str(tmp_path), []) == []


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------


def test_cli_parses_repeatable_lora_adapter_arguments():
    config = parse_server_config([
        "m",
        "--lora-adapter",
        "french=/adapters/fr.safetensors",
        "--lora-adapter",
        "gemma.v2=/adapters/gv2.safetensors",
    ])
    assert config.model.lora_adapters == (
        LoraAdapterConfig(name="french", path="/adapters/fr.safetensors"),
        LoraAdapterConfig(name="gemma.v2", path="/adapters/gv2.safetensors"),
    )


def test_cli_lora_adapter_kwargs_reach_the_runtime():
    config = ModelConfig(model="m",
                         lora_adapters=(LoraAdapterConfig(
                             "a", "/x.safetensors"), ))
    kwargs = config.llm_kwargs()
    assert kwargs["lora_adapters"] == [
        LoraAdapterConfig("a", "/x.safetensors")
    ]


@pytest.mark.parametrize("name", ["", "has space", "colon:name", "sl/ash"])
def test_cli_rejects_invalid_adapter_names(name):
    with pytest.raises(ServerConfigError, match="must match"):
        ModelConfig(model="m", lora_adapters=(LoraAdapterConfig(name, "/x"), ))


def test_cli_rejects_duplicate_adapter_names():
    with pytest.raises(ServerConfigError, match="more than once"):
        ModelConfig(
            model="m",
            lora_adapters=(LoraAdapterConfig("a", "/x"),
                           LoraAdapterConfig("a", "/y")),
        )


def test_cli_rejects_malformed_adapter_argument():
    with pytest.raises(SystemExit):
        parse_server_config(["m", "--lora-adapter", "no-path-separator"])


def test_cli_rejects_adapter_name_colliding_with_a_discovered_one(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "french", rank=8)
    with pytest.raises(ValueError, match="collides"):
        LLM(model=str(bundle),
            cache_dir=str(tmp_path / "cache"),
            lora_adapters=[LoraAdapterConfig("french", "/other.safetensors")])


# ---------------------------------------------------------------------------
# Startup refusals
# ---------------------------------------------------------------------------


def test_adapters_on_a_rank_zero_engine_refuse_to_start(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=0)
    _adapter(bundle / "llm", "french", rank=8)
    with pytest.raises(ValueError, match="max_lora_rank=0"):
        LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))


def test_adapter_rank_above_the_engine_refuses_to_start(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "wide", rank=32)
    with pytest.raises(ValueError, match="exceeds the engine's max_lora_rank"):
        LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))


def test_configured_adapters_on_a_rank_zero_engine_refuse_to_start(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=0)
    with pytest.raises(ValueError, match="max_lora_rank=0"):
        LLM(model=str(bundle),
            cache_dir=str(tmp_path / "cache"),
            lora_adapters=[LoraAdapterConfig("french", "/fr.safetensors")])


# ---------------------------------------------------------------------------
# Request-time selection over HTTP (real LLM + fake pybind module)
# ---------------------------------------------------------------------------


def _client_with(llm: LLM, config=None):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from experimental.server.api.app import create_app
    from experimental.server.runtime.engine_client import EngineClient

    config = config or ApiConfig(served_model_name=SERVED_MODEL)
    return TestClient(create_app(EngineClient(llm, config), config)), llm


def _chat(model=None, lora_name=None):
    body = {"messages": [{"role": "user", "content": "Hello"}]}
    if model is not None:
        body["model"] = model
    if lora_name is not None:
        body["lora_name"] = lora_name
    return body


def _switches(llm: LLM):
    """``lora_weights_name`` values the llm's runtime recorded, in order."""
    return llm._runtime.switch_names


def test_selector_accepts_both_spellings(tmp_path, _fake_runtime):
    client, llm = _client_with(_build_llm(tmp_path))
    served = SERVED_MODEL

    response = client.post("/v1/chat/completions",
                           json=_chat(model=f"{served}:french"))
    assert response.status_code == 200, response.text
    assert response.json()["model"] == f"{served}:french"

    response = client.post("/v1/chat/completions",
                           json=_chat(lora_name="poetic"))
    assert response.status_code == 200, response.text
    assert response.json()["model"] == f"{served}:poetic"

    response = client.post("/v1/chat/completions", json=_chat())
    assert response.status_code == 200, response.text
    assert response.json()["model"] == served

    response = client.post("/v1/chat/completions", json=_chat(model=served))
    assert response.status_code == 200, response.text
    assert response.json()["model"] == served

    assert _switches(llm) == ["french", "poetic", "", ""]


def test_selector_conflict_is_a_bad_request(tmp_path, _fake_runtime):
    client, _ = _client_with(_build_llm(tmp_path))
    response = client.post("/v1/chat/completions",
                           json=_chat(model=f"{SERVED_MODEL}:french",
                                      lora_name="poetic"))
    assert response.status_code == 400
    assert "french" in response.text and "poetic" in response.text


def test_selector_conflict_with_base_is_a_bad_request(tmp_path, _fake_runtime):
    client, _ = _client_with(_build_llm(tmp_path))
    response = client.post("/v1/chat/completions",
                           json=_chat(model=f"{SERVED_MODEL}:french",
                                      lora_name=""))
    assert response.status_code == 400


def test_selector_lora_name_with_bare_model_selects_the_adapter(
        tmp_path, _fake_runtime):
    # ``model`` is what OpenAI clients always send; it must not veto
    # ``lora_name``. The bare served name plus lora_name runs the adapter.
    client, llm = _client_with(_build_llm(tmp_path))
    response = client.post("/v1/chat/completions",
                           json=_chat(model=SERVED_MODEL, lora_name="french"))
    assert response.status_code == 200, response.text
    assert response.json()["model"] == f"{SERVED_MODEL}:french"
    assert _switches(llm) == ["french"]


def test_selector_lora_name_alone_selects_the_adapter(tmp_path, _fake_runtime):
    client, llm = _client_with(_build_llm(tmp_path))
    response = client.post("/v1/chat/completions",
                           json=_chat(lora_name="french"))
    assert response.status_code == 200, response.text
    assert response.json()["model"] == f"{SERVED_MODEL}:french"
    assert _switches(llm) == ["french"]


def test_unknown_adapter_is_a_bad_request_listing_known_names(
        tmp_path, _fake_runtime):
    client, _ = _client_with(_build_llm(tmp_path))
    response = client.post("/v1/chat/completions",
                           json=_chat(model=f"{SERVED_MODEL}:missing"))
    assert response.status_code == 400
    assert "french" in response.text and "poetic" in response.text


def test_adapter_selection_without_loaded_adapters_is_a_bad_request(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    llm = LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))
    client, _ = _client_with(llm)
    response = client.post("/v1/chat/completions", json=_chat(lora_name="x"))
    assert response.status_code == 400
    response = client.post("/v1/chat/completions",
                           json=_chat(model=f"{SERVED_MODEL}:x"))
    assert response.status_code == 400


def test_unknown_model_without_adapter_still_404s(tmp_path, _fake_runtime):
    client, _ = _client_with(_build_llm(tmp_path))
    response = client.post("/v1/chat/completions", json=_chat(model="other"))
    assert response.status_code == 404


def test_lora_name_field_is_accepted_by_the_protocol():
    from experimental.server.api.protocol import ChatCompletionRequest
    request = ChatCompletionRequest(
        messages=[{
            "role": "user",
            "content": "hi"
        }],
        lora_name="french",
    )
    assert request.lora_name == "french"


# ---------------------------------------------------------------------------
# In-flight batching refusal
# ---------------------------------------------------------------------------


def test_ifb_refuses_adapter_requests_at_build_time(tmp_path, _fake_runtime):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from experimental.server.api.app import create_app
    from experimental.server.runtime.engine_client import EngineClient

    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "french", rank=8)
    llm = LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))
    # Under IFB the runtime is a RequestEngine; mark the deployment so the
    # request-build refusal fires.
    llm._engine = object()
    client = TestClient(
        create_app(EngineClient(llm),
                   ApiConfig(served_model_name=SERVED_MODEL)))
    response = client.post("/v1/chat/completions",
                           json=_chat(lora_name="french"))
    assert response.status_code == 400
    assert "in-flight batching" in response.text


def test_ifb_refuses_adapter_requests_from_the_engine_api(monkeypatch):
    from experimental.server.runtime import engine as engine_module

    class NativeValue:

        def __init__(self, **values):
            self.__dict__.update(values)

    runtime = SimpleNamespace(
        LLMGenerationRequest=NativeValue,
        Request=NativeValue,
        ToolChoice=NativeValue,
        ToolChoiceMode=SimpleNamespace(NONE="none",
                                       AUTO="auto",
                                       REQUIRED="required",
                                       FUNCTION="function"),
        ContextCacheLookupPolicy=SimpleNamespace(USE_CACHE="use",
                                                 BYPASS="bypass"),
        ContextCacheCommitPolicy=SimpleNamespace(
            INCLUDING_GENERATED_TOKENS="all", PREFILL_STATE_ONLY="prefill"),
    )
    llm = LLM.__new__(LLM)
    llm._runtime = runtime
    llm._engine = object()  # IFB path: an engine, no blocking runtime
    llm._rt = runtime
    llm._context_cache_config = type("C", (), {"enabled": False})()
    llm._spec_decode_type = ""
    llm._prepare_messages_for_runtime = lambda _messages: ([], [])
    monkeypatch.setattr(engine_module, "_load_audio_buffers",
                        lambda *_args: [])
    with pytest.raises(ValueError, match="in-flight batching"):
        llm._make_generation_request([{
            "role": "user",
            "content": "hello"
        }], SamplingParams(lora_name="french"))


# ---------------------------------------------------------------------------
# /v1/models and /health (served by the real LLM)
# ---------------------------------------------------------------------------


def test_models_endpoint_lists_adapter_variants(tmp_path, _fake_runtime):
    client, _ = _client_with(_build_llm(tmp_path))
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [card["id"] for card in response.json()["data"]]
    assert ids == [
        SERVED_MODEL,
        f"{SERVED_MODEL}:french",
        f"{SERVED_MODEL}:poetic",
    ]


def test_health_reports_adapters_and_max_lora_rank(tmp_path, _fake_runtime):
    client, _ = _client_with(_build_llm(tmp_path))
    response = client.get("/health")
    assert response.status_code == 200
    lora = response.json()["lora"]
    assert lora["max_lora_rank"] == 16
    assert [a["name"] for a in lora["adapters"]] == ["french", "poetic"]
    french = lora["adapters"][0]
    assert french["rank"] == 8
    assert french["bound_modules"] == 2
    assert french["unbound_untargeted_bindings"] == 1
    assert french["processed"] is True


def test_health_reports_unprocessed_adapter_without_a_provenance_record(
        tmp_path, _fake_runtime):
    # ``poetic`` has no config.json, so lora_adapter_info must report both
    # the null coverage counts and processed=False for it.
    client, _ = _client_with(_build_llm(tmp_path))
    response = client.get("/health")
    assert response.status_code == 200
    lora = response.json()["lora"]
    poetic = lora["adapters"][1]
    assert poetic["name"] == "poetic"
    assert poetic["rank"] is None
    assert poetic["bound_modules"] is None
    assert poetic["unbound_untargeted_bindings"] is None
    assert poetic["processed"] is False


# ---------------------------------------------------------------------------
# Adapter switching through the real LLM and the recording fake runtime
# ---------------------------------------------------------------------------


def test_runtime_receives_the_adapter_switch_sequence(tmp_path, _fake_runtime):
    """base -> french -> poetic -> french -> base, in one session.

    The fake runtime records every ``lora_weights_name`` the real engine
    passes down, which is the wire the C++ LoRAManager switches on. Each
    response echoes the selector the request named.
    """
    client, llm = _client_with(_build_llm(tmp_path))
    served = SERVED_MODEL

    sequence = [
        (_chat(), served, ""),
        (_chat(model=f"{served}:french"), f"{served}:french", "french"),
        (_chat(lora_name="poetic"), f"{served}:poetic", "poetic"),
        (_chat(model=f"{served}:french",
               lora_name="french"), f"{served}:french", "french"),
        (_chat(), served, ""),
    ]
    for body, expected_model, expected_switch in sequence:
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["model"] == expected_model

    assert _switches(llm) == ["", "french", "poetic", "french", ""]


def test_streamed_response_echoes_the_adapter_model(tmp_path, _fake_runtime):
    client, _ = _client_with(_build_llm(tmp_path))
    body = _chat(model=f"{SERVED_MODEL}:french")
    body["stream"] = True
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    models = {
        json.loads(line.removeprefix("data: "))["model"]
        for line in response.text.splitlines() if line.startswith("data: {")
    }
    assert models == {f"{SERVED_MODEL}:french"}


def test_blocking_generation_reaches_the_fake_runtime_with_the_adapter(
        tmp_path, _fake_runtime):
    # Direct (non-HTTP) path: the real LLM builds the native request and the
    # fake runtime observes the selector on it. generate() takes a message
    # batch list, so the conversation is wrapped once more.
    llm = _build_llm(tmp_path)
    output = llm.generate([[{
        "role": "user",
        "content": "Hello"
    }]], SamplingParams(lora_name="french"))
    assert output[0].text == "hello"
    assert output[0].finish_reason == "stop"
    assert _switches(llm) == ["french"]


def test_finish_reason_maps_the_fake_enum():
    assert finish_reason_name(_FakeRuntimeModule, "stop_words") == "stop"
    assert finish_reason_name(_FakeRuntimeModule, "end_id") == "stop"
    assert finish_reason_name(_FakeRuntimeModule, "not_finished") is None


# ---------------------------------------------------------------------------
# Async engine-client path
# ---------------------------------------------------------------------------


def test_engine_client_carries_the_adapter_to_the_request(
        tmp_path, _fake_runtime):
    from experimental.server.runtime.engine_client import EngineClient

    llm = _build_llm(tmp_path)
    client = EngineClient(llm)

    async def exercise():
        prepared = await client.prepare_request([{
            "role": "user",
            "content": "hello"
        }], SamplingParams(lora_name="french"))
        return prepared

    prepared = asyncio.run(exercise())
    # Request preparation builds the native request; the adapter selector
    # rides on it ready for the runtime's LoRAManager.
    assert prepared.request.lora_weights_name == "french"


# ---------------------------------------------------------------------------
# Manifest and base-identity enforcement
# ---------------------------------------------------------------------------


def test_adapter_without_a_manifest_refuses_to_start(tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "tool", rank=8, manifest=False)

    with pytest.raises(ValueError, match="tool"):
        LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))


def test_adapter_packaged_for_another_base_refuses_to_start(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "tool", rank=8, base_identity="0" * 64)

    with pytest.raises(ValueError, match="deployed base"):
        LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))


def test_changed_base_file_refuses_to_start(tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "tool", rank=8)
    (bundle / "llm" / "tokenizer.json").write_text("{}")

    with pytest.raises(ValueError, match="tokenizer.json"):
        LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))


def test_unverified_adapters_are_served_only_when_allowed(
        tmp_path, _fake_runtime):
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "good", rank=8)
    _adapter(bundle / "llm", "stale", rank=8, base_identity="0" * 64)

    llm = LLM(model=str(bundle),
              cache_dir=str(tmp_path / "cache"),
              allow_unverified_lora_adapters=True)

    report = {entry["name"]: entry for entry in llm.lora_adapter_info()}
    assert report["good"]["verified"] is True
    assert report["good"]["version"] == "good-v1"
    assert report["stale"]["verified"] is False
    assert sorted(llm._runtime.args[2]) == ["good", "stale"]


def test_checkpoint_backed_bundle_is_not_covered_by_the_base_identity(tmp_path):
    from experimental.server.runtime.engine import _resolve_lora_adapters
    engine_dir = _engine_tree(tmp_path, max_lora_rank=16) / "llm"
    _adapter(engine_dir, "tool", rank=8)

    with pytest.raises(ValueError, match="checkpoint-backed"):
        _resolve_lora_adapters(str(engine_dir), [],
                               checkpoint_dir=str(tmp_path / "ckpt"))
    adapters = _resolve_lora_adapters(str(engine_dir), [],
                                      checkpoint_dir=str(tmp_path / "ckpt"),
                                      allow_unverified=True)
    assert [adapter.verified for adapter in adapters] == [False]


def test_cli_parses_lora_verification_flags():
    config = parse_server_config(
        ["m", "--lora-verify", "full", "--allow-unverified-lora-adapters"])
    assert config.model.lora_verify == "full"
    assert config.model.allow_unverified_lora_adapters is True
    defaults = parse_server_config(["m"])
    assert defaults.model.lora_verify == "sizes"
    assert defaults.model.allow_unverified_lora_adapters is False
