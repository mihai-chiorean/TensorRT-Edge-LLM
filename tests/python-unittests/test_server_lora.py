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
"""LoRA adapter serving: discovery, configuration, selection and switching."""

import asyncio
import json
from pathlib import Path

import pytest

from experimental.server.config import (ApiConfig, LoraAdapterConfig,
                                        ModelConfig, ServerConfigError,
                                        parse_server_config)
from experimental.server.runtime.engine import LLM, LoraAdapter, SamplingParams

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
    return bundle


def _adapter(engine_dir: Path, name: str, rank=None) -> Path:
    directory = engine_dir / "lora_weights" / name
    directory.mkdir(parents=True)
    (directory / "processed_adapter_model.safetensors").touch()
    if rank is not None:
        _write_json(
            directory / "config.json", {
                "edgellm_lora_processing": {
                    "rank": rank,
                    "lora_scale": 1.0,
                    "max_lora_rank": 64,
                    "bound_modules": ["a", "b"],
                    "unbound_untargeted_bindings": ["c"],
                }
            })
    return directory / "processed_adapter_model.safetensors"


class _FakeRuntime:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def capture_decoding_cuda_graph(self):
        return True

    def has_draft_model(self):
        return False


class _FakeBindings:
    LLMRuntime = _FakeRuntime
    ContextCacheConfig = type(
        "Config", (), {
            "__init__": lambda self: setattr(self, "enabled", False),
        })


def _monkeypatch_runtime(monkeypatch):
    monkeypatch.setattr("experimental.server.runtime.engine._import_runtime",
                        lambda: _FakeBindings)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovered_adapters_come_from_the_lora_weights_directory(
        tmp_path, monkeypatch):
    _monkeypatch_runtime(monkeypatch)
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "french", rank=8)
    _adapter(bundle / "llm", " pirate ", rank=8)  # invalid name, skipped
    (bundle / "llm" / "lora_weights" / "empty").mkdir()
    (bundle / "llm" / "lora_weights" / "loose.safetensors").touch()

    llm = LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))

    assert llm.lora_names() == ["french"]
    assert llm.max_lora_rank == 16


def test_adapters_without_a_sidecar_directory_are_not_discovered(
        tmp_path, monkeypatch):
    _monkeypatch_runtime(monkeypatch)
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
        tmp_path, monkeypatch):
    _monkeypatch_runtime(monkeypatch)
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "french", rank=8)
    with pytest.raises(ValueError, match="collides"):
        LLM(model=str(bundle),
            cache_dir=str(tmp_path / "cache"),
            lora_adapters=[LoraAdapterConfig("french", "/other.safetensors")])


# ---------------------------------------------------------------------------
# Startup refusals
# ---------------------------------------------------------------------------


def test_adapters_on_a_rank_zero_engine_refuse_to_start(tmp_path, monkeypatch):
    _monkeypatch_runtime(monkeypatch)
    bundle = _engine_tree(tmp_path, max_lora_rank=0)
    _adapter(bundle / "llm", "french", rank=8)
    with pytest.raises(ValueError, match="max_lora_rank=0"):
        LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))


def test_adapter_rank_above_the_engine_refuses_to_start(tmp_path, monkeypatch):
    _monkeypatch_runtime(monkeypatch)
    bundle = _engine_tree(tmp_path, max_lora_rank=16)
    _adapter(bundle / "llm", "wide", rank=32)
    with pytest.raises(ValueError, match="exceeds the engine's max_lora_rank"):
        LLM(model=str(bundle), cache_dir=str(tmp_path / "cache"))


def test_configured_adapters_on_a_rank_zero_engine_refuse_to_start(
        tmp_path, monkeypatch):
    _monkeypatch_runtime(monkeypatch)
    bundle = _engine_tree(tmp_path, max_lora_rank=0)
    with pytest.raises(ValueError, match="max_lora_rank=0"):
        LLM(model=str(bundle),
            cache_dir=str(tmp_path / "cache"),
            lora_adapters=[LoraAdapterConfig("french", "/fr.safetensors")])


# ---------------------------------------------------------------------------
# Request-time selection over HTTP
# ---------------------------------------------------------------------------


class _FakeLLM:
    runtime_kind = "chat"
    video_capable = False
    has_draft_model = False
    max_batch_size = 1

    def __init__(self, root: Path, adapters=(), max_lora_rank=16):
        self.model_dir = str(root / "qwen3")
        self.bundle_dir = str(root / "engines")
        self.model_id = "fake-model"
        self._lora_adapters = tuple(adapters)
        self._max_lora_rank = max_lora_rank
        self.prepared_messages = None
        self.last_sampling_params = None
        self.next_text = "answer"
        self.switches = []  # lora_weights_name sequence seen by the runtime
        Path(self.model_dir).mkdir(parents=True)
        Path(self.bundle_dir).mkdir(parents=True)
        (Path(self.bundle_dir) / "config.json").write_text(
            json.dumps({
                "builder_config": {
                    "max_input_len": 32,
                    "max_batch_size": 1,
                    "max_kv_cache_capacity": 128,
                    "max_lora_rank": max_lora_rank,
                },
                "kv_cache_dtype": "fp16",
            }),
            encoding="utf-8",
        )
        from experimental.server.runtime.engine_layout import inspect_bundle
        self.bundle_layout = inspect_bundle(self.bundle_dir)

    @property
    def lora_adapters(self):
        return self._lora_adapters

    @property
    def max_lora_rank(self):
        return self._max_lora_rank

    def lora_names(self):
        return [adapter.name for adapter in self._lora_adapters]

    def lora_adapter_info(self):
        report = []
        for adapter in self._lora_adapters:
            provenance = adapter.provenance or {}
            bound = provenance.get("bound_modules")
            unbound = provenance.get("unbound_untargeted_bindings")
            report.append({
                "name":
                adapter.name,
                "rank":
                adapter.rank,
                "bound_modules":
                len(bound) if isinstance(bound, list) else None,
                "unbound_untargeted_bindings":
                len(unbound) if isinstance(unbound, list) else None,
                "processed":
                provenance is not None,
            })
        return report

    def _make_generation_request(self, messages, params, **kwargs):
        self.prepared_messages = messages
        self.last_sampling_params = params
        self.switches.append(params.lora_name)
        if getattr(self, "_engine", None) is not None and params.lora_name:
            raise ValueError(
                "LoRA adapters are not supported under in-flight batching")
        return object()

    def _count_prepared_prompt_tokens(self, _request):
        return 7

    def _complete_prepared_request(self, _request, _params, tool_config,
                                   **_kwargs):
        from experimental.server.runtime.engine import CompletionOutput
        return CompletionOutput(text=self.next_text,
                                token_ids=[1, 2, 3],
                                prompt_tokens=7,
                                finish_reason="stop")

    def generate_stream(self, _messages, _params, *, tools=None, **_kwargs):
        from experimental.server.runtime.engine import StreamDelta
        yield StreamDelta(text="hello",
                          token_ids=[1, 2],
                          prompt_tokens=7,
                          finished=True,
                          finish_reason="stop")

    def close(self):
        pass


def _client_with(llm, config=None):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from experimental.server.api.app import create_app
    from experimental.server.runtime.engine_client import EngineClient

    config = config or ApiConfig()
    return TestClient(create_app(EngineClient(llm, config), config)), llm


def _lora_llm(tmp_path):
    return _FakeLLM(
        tmp_path,
        adapters=(
            LoraAdapter(
                name="french",
                path="/x/fr.safetensors",
                rank=8,
                provenance={
                    "rank": 8,
                    "bound_modules": ["a", "b"],
                    "unbound_untargeted_bindings": ["c"],
                },
            ),
            LoraAdapter(name="poetic", path="/x/po.safetensors", rank=16),
        ),
        max_lora_rank=16,
    )


def _chat(model=None, lora_name=None):
    body = {"messages": [{"role": "user", "content": "Hello"}]}
    if model is not None:
        body["model"] = model
    if lora_name is not None:
        body["lora_name"] = lora_name
    return body


def _switches(llm):
    return llm.switches


def test_selector_accepts_both_spellings(tmp_path):
    client, llm = _client_with(_lora_llm(tmp_path))

    response = client.post("/v1/chat/completions",
                           json=_chat(model="fake-model:french"))
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "fake-model:french"

    response = client.post("/v1/chat/completions",
                           json=_chat(lora_name="poetic"))
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "fake-model:poetic"

    response = client.post("/v1/chat/completions", json=_chat())
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "fake-model"

    response = client.post("/v1/chat/completions",
                           json=_chat(model="fake-model"))
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "fake-model"

    assert _switches(llm) == ["french", "poetic", "", ""]


def test_selector_conflict_is_a_bad_request(tmp_path):
    client, _ = _client_with(_lora_llm(tmp_path))
    response = client.post("/v1/chat/completions",
                           json=_chat(model="fake-model:french",
                                      lora_name="poetic"))
    assert response.status_code == 400
    assert "french" in response.text and "poetic" in response.text


def test_selector_conflict_with_base_is_a_bad_request(tmp_path):
    client, _ = _client_with(_lora_llm(tmp_path))
    response = client.post("/v1/chat/completions",
                           json=_chat(model="fake-model:french", lora_name=""))
    assert response.status_code == 400


def test_selector_base_vs_adapter_conflict_via_lora_name(tmp_path):
    # ``model`` explicitly names the base; ``lora_name`` names an adapter.
    # Both are present and disagree, so the request is refused.
    client, _ = _client_with(_lora_llm(tmp_path))
    response = client.post("/v1/chat/completions",
                           json=_chat(model="fake-model", lora_name="french"))
    assert response.status_code == 400
    assert "french" in response.text


def test_unknown_adapter_is_a_bad_request_listing_known_names(tmp_path):
    client, _ = _client_with(_lora_llm(tmp_path))
    response = client.post("/v1/chat/completions",
                           json=_chat(model="fake-model:missing"))
    assert response.status_code == 400
    assert "french" in response.text and "poetic" in response.text


def test_adapter_selection_without_loaded_adapters_is_a_bad_request(tmp_path):
    client, _ = _client_with(_FakeLLM(tmp_path))
    response = client.post("/v1/chat/completions", json=_chat(lora_name="x"))
    assert response.status_code == 400
    response = client.post("/v1/chat/completions",
                           json=_chat(model="fake-model:x"))
    assert response.status_code == 400


def test_unknown_model_without_adapter_still_404s(tmp_path):
    client, _ = _client_with(_lora_llm(tmp_path))
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


def test_ifb_refuses_adapter_requests_at_build_time(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from experimental.server.api.app import create_app
    from experimental.server.runtime.engine_client import EngineClient

    llm = _FakeLLM(tmp_path,
                   adapters=(LoraAdapter(name="french",
                                         path="/x/fr.safetensors",
                                         rank=8), ),
                   max_lora_rank=16)
    # Under IFB the runtime is a RequestEngine; the fake marks it via _engine.
    llm._engine = object()
    client = TestClient(create_app(EngineClient(llm), ApiConfig()))
    response = client.post("/v1/chat/completions",
                           json=_chat(lora_name="french"))
    assert response.status_code == 400
    assert "in-flight batching" in response.text


def test_ifb_refuses_adapter_requests_from_the_engine_api(monkeypatch):
    from types import SimpleNamespace

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
# /v1/models and /health
# ---------------------------------------------------------------------------


def test_models_endpoint_lists_adapter_variants(tmp_path):
    client, _ = _client_with(_lora_llm(tmp_path))
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [card["id"] for card in response.json()["data"]]
    assert ids == ["fake-model", "fake-model:french", "fake-model:poetic"]


def test_health_reports_adapters_and_max_lora_rank(tmp_path):
    client, _ = _client_with(_lora_llm(tmp_path))
    response = client.get("/health")
    assert response.status_code == 200
    lora = response.json()["lora"]
    assert lora["max_lora_rank"] == 16
    assert [a["name"] for a in lora["adapters"]] == ["french", "poetic"]
    french = lora["adapters"][0]
    assert french["rank"] == 8
    assert french["bound_modules"] == 2
    assert french["unbound_untargeted_bindings"] == 1


def test_health_reports_null_coverage_without_a_provenance_record(tmp_path):
    client, _ = _client_with(
        _FakeLLM(tmp_path,
                 adapters=(LoraAdapter(name="bare",
                                       path="/x/bare.safetensors"), ),
                 max_lora_rank=8))
    response = client.get("/health")
    lora = response.json()["lora"]
    bare = lora["adapters"][0]
    assert bare["rank"] is None
    assert bare["bound_modules"] is None
    assert bare["unbound_untargeted_bindings"] is None


# ---------------------------------------------------------------------------
# Adapter switching through a recording runtime
# ---------------------------------------------------------------------------


def test_runtime_receives_the_adapter_switch_sequence(tmp_path):
    """base -> french -> poetic -> french -> base, in one session.

    The fake runtime records every ``lora_weights_name`` the engine passes
    down, which is the wire the C++ LoRAManager switches on. Each response
    echoes the selector the request named.
    """
    client, llm = _client_with(_lora_llm(tmp_path))

    sequence = [
        (_chat(), "fake-model", ""),
        (_chat(model="fake-model:french"), "fake-model:french", "french"),
        (_chat(lora_name="poetic"), "fake-model:poetic", "poetic"),
        (_chat(model="fake-model:french",
               lora_name="french"), "fake-model:french", "french"),
        (_chat(), "fake-model", ""),
    ]
    for body, expected_model, expected_switch in sequence:
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["model"] == expected_model
        assert llm.switches[-1] == expected_switch

    assert llm.switches == ["", "french", "poetic", "french", ""]


def test_streamed_response_echoes_the_adapter_model(tmp_path):
    client, _ = _client_with(_lora_llm(tmp_path))
    body = _chat(model="fake-model:french")
    body["stream"] = True
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    models = {
        json.loads(line.removeprefix("data: "))["model"]
        for line in response.text.splitlines() if line.startswith("data: {")
    }
    assert models == {"fake-model:french"}


# ---------------------------------------------------------------------------
# Async engine-client path
# ---------------------------------------------------------------------------


def test_engine_client_carries_the_adapter_to_the_request(tmp_path):
    from experimental.server.runtime.engine_client import EngineClient

    llm = _FakeLLM(tmp_path)
    client = EngineClient(llm)

    async def exercise():
        await client.prepare_request([{
            "role": "user",
            "content": "hello"
        }], SamplingParams(lora_name="french"))

    asyncio.run(exercise())
    assert llm.switches == ["french"]
