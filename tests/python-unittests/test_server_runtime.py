# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import threading
import time
from types import SimpleNamespace

import pytest

from experimental.server.engine import LLM, SamplingParams
from experimental.server.tool_calling import (ToolProtocolError,
                                              make_stream_parser,
                                              parse_assistant_output,
                                              validate_tool_request)
from experimental.server.tool_chat_template import ToolChatTemplateFormatter


class _ConcurrentRuntime:

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def handle_request(self, request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

        time.sleep(0.02)

        with self.lock:
            self.active -= 1
        return request


def test_runtime_requests_are_serialized():
    runtime = _ConcurrentRuntime()
    llm = SimpleNamespace(_runtime=runtime)
    results = []

    threads = [
        threading.Thread(target=lambda request=request: results.append(
            LLM._handle_request(llm, request))) for request in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == list(range(8))
    assert runtime.max_active == 1


class _GemmaTokenDecoder:
    pieces = {
        48: "<|tool_call>",
        49: "<tool_call|>",
        52: '<|"|>',
        1: "call:lookup{city:",
        2: "Berkeley, California",
        3: "}",
        4: "<turn|>",
        5: "<|channel>analysis",
        6: "Unverified prose.",
        7: "STOPignored",
    }
    special_ids = {48, 49, 52, 4, 5}

    def decode(self, ids, *, skip_special_tokens,
               clean_up_tokenization_spaces):
        assert clean_up_tokenization_spaces is False
        return "".join(
            self.pieces[token_id] for token_id in ids
            if not skip_special_tokens or token_id not in self.special_ids)


def _gemma_tool_fixture():
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    }
                }
            }
        }
    }]
    config = validate_tool_request([], tools)
    tokenizer = _GemmaTokenDecoder()
    formatter = ToolChatTemplateFormatter([], template_owner=tokenizer)
    llm = LLM.__new__(LLM)
    llm._model_dir = "/fake/gemma4"
    llm._get_tool_template_formatter = lambda: formatter
    ids = [48, 1, 52, 2, 52, 3, 49, 4]
    return llm, config, tokenizer, ids


def test_gemma_nonstream_restores_native_stripped_protocol():
    llm, config, tokenizer, ids = _gemma_tool_fixture()
    stripped = tokenizer.decode(ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    assert stripped == "call:lookup{city:Berkeley, California}"
    with pytest.raises(ToolProtocolError):
        parse_assistant_output(stripped, config, llm.model_dir)

    restored = llm.decode_tool_output(stripped, ids, config)
    assert '<|"|>Berkeley, California<|"|>' in restored
    assert not restored.endswith("<turn|>")
    parsed = parse_assistant_output(restored, config, llm.model_dir)
    assert parsed.content == ""
    assert json.loads(parsed.tool_calls[0].arguments) == {
        "city": "Berkeley, California"
    }


def test_gemma_restoration_preserves_channel_markers_and_stop_truncation():
    llm, config, _, ids = _gemma_tool_fixture()
    assert llm.decode_tool_output(
        "stripped", [5, 6, 4],
        config) == ("<|channel>analysisUnverified prose.")
    assert llm.decode_tool_output("stripped", [6, 7, 4],
                                  config,
                                  stop_strings=("ignored",
                                                "STOP")) == "Unverified prose."
    stopped = llm.decode_tool_output("stripped",
                                     ids,
                                     config,
                                     stop_strings=("California", ))
    assert stopped.endswith("Berkeley, ")
    with pytest.raises(ToolProtocolError):
        parse_assistant_output(stopped, config, llm.model_dir)


def test_gemma_restoration_is_tool_only_and_fails_closed():
    llm, config, _, ids = _gemma_tool_fixture()
    no_tools = validate_tool_request([])
    assert llm.decode_tool_output("unchanged", ids, no_tools) == "unchanged"
    llm._model_dir = "/fake/qwen3"
    assert llm.decode_tool_output("unchanged", ids, config) == "unchanged"
    llm._model_dir = "/fake/gemma4"
    with pytest.raises(ToolProtocolError, match="invalid tool call"):
        llm.decode_tool_output("stripped protocol", [], config)

    def broken_formatter():
        raise ValueError("private model text")

    llm._get_tool_template_formatter = broken_formatter
    with pytest.raises(ToolProtocolError) as raised:
        llm.decode_tool_output("stripped protocol", ids, config)
    assert raised.value.reason == "decode_failed"
    assert "private" not in str(raised.value)


@pytest.mark.parametrize("model, tools_enabled, choice, expected_skip", [
    ("gemma4", True, "auto", False),
    ("gemma4", True, "none", True),
    ("gemma4", False, "none", True),
    ("qwen3", True, "auto", True),
])
def test_stream_channel_preserves_only_gemma_tool_protocol(
        model, tools_enabled, choice, expected_skip):
    llm, config, tokenizer, ids = _gemma_tool_fixture()
    llm._model_dir = "/fake/" + model

    class _Channel:
        index = 0
        skip = None

        def set_skip_special_tokens(self, flag):
            self.skip = flag

        def wait_pop(self, timeout_ms):
            if self.index == len(ids):
                return None
            token_id = ids[self.index]
            self.index += 1
            # Matches native emitDelta -> idToPiece(token, skipSpecial).
            text = tokenizer.decode([token_id],
                                    skip_special_tokens=self.skip,
                                    clean_up_tokenization_spaces=False)
            return SimpleNamespace(text=text,
                                   token_ids=[token_id],
                                   finished=False,
                                   reason=None,
                                   logprobs=[])

        def is_finished(self):
            return self.index == len(ids)

        def is_cancelled(self):
            return False

        def cancel(self):
            raise AssertionError("Completed channel must not be cancelled")

    channel = _Channel()
    llm._rt = SimpleNamespace(StreamChannel=SimpleNamespace(
        create=lambda: channel))
    llm._handle_request = lambda request: None
    deltas = list(
        llm.generate_stream(
            [],
            tools=config.tools if tools_enabled else [],
            tool_choice=choice,
            prebuilt_request=SimpleNamespace(stream_channels=[])))
    assert channel.skip is expected_skip
    assert [token_id for delta in deltas
            for token_id in delta.token_ids] == ids
    joined = "".join(delta.text for delta in deltas)
    assert joined == tokenizer.decode(ids,
                                      skip_special_tokens=expected_skip,
                                      clean_up_tokenization_spaces=False)
    if not expected_skip:
        parser = make_stream_parser(config,
                                    llm.model_dir,
                                    strip_tokens=("<turn|>", ))
        events = [
            event for delta in deltas for event in parser.feed(delta.text)
        ]
        events.extend(parser.finish())
        assert [event["type"] for event in events] == ["tool_call"]
        assert json.loads(events[0]["tool_call"].arguments) == {
            "city": "Berkeley, California"
        }


def test_direct_generate_restores_tokens_before_tool_parsing():
    llm, config, tokenizer, ids = _gemma_tool_fixture()
    llm._rt = None
    llm._make_generation_request = lambda *args, **kwargs: object()
    llm._handle_request = lambda request: SimpleNamespace(output_texts=[
        tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ],
                                                          output_ids=[ids],
                                                          finish_reasons=[],
                                                          logprobs=[])
    output, = llm.generate("Look it up", SamplingParams(), tools=config.tools)
    assert output.text == ""
    assert json.loads(output.tool_calls[0]["function"]["arguments"]) == {
        "city": "Berkeley, California"
    }
