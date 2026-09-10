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

import json

import pytest

from experimental.server.api_server import (_build_message_body,
                                            _generate_stream_sse,
                                            _ThinkingStateMachine)
from experimental.server.engine import StreamDelta
from experimental.server.tool_calling import (ToolProtocolError,
                                              make_stream_parser,
                                              parse_assistant_output,
                                              validate_tool_request)


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    },
                },
            },
        },
    }]


def _tool_config(tool_choice="auto"):
    return validate_tool_request([{
        "role": "user",
        "content": "Weather?"
    }], _tools(), tool_choice)


def test_validates_tool_request():
    messages = [{
        "role":
        "assistant",
        "content":
        None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}",
            },
        }],
    }, {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": {
            "temperature": 22
        },
    }]

    config = validate_tool_request(messages, _tools(), "auto")

    assert config.tool_choice == "auto"

    with pytest.raises(ValueError, match="Unknown forced tool name"):
        validate_tool_request([{
            "role": "user",
            "content": "hi"
        }], _tools(), {
            "type": "function",
            "function": {
                "name": "missing"
            },
        })

    with pytest.raises(ValueError, match="'tools' must be an array"):
        validate_tool_request([{
            "role": "user",
            "content": "hi"
        }], {"type": "function"})

    with pytest.raises(ValueError, match="Dangling tool_call_id"):
        validate_tool_request([{
            "role": "tool",
            "tool_call_id": "call_missing",
            "content": "42",
        }])


def test_parses_tool_calls(tmp_path):
    json_text = ("Let me check.\n"
                 "<tool_call>{\"name\":\"get_weather\","
                 "\"arguments\":{\"city\":\"Paris\"}}</tool_call>")

    parsed = parse_assistant_output(json_text, _tool_config(), str(tmp_path))

    assert parsed.content == "Let me check.\n"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Paris"}

    qwen_text = (
        "<think>plan</think>Before"
        "<function=get_weather><parameter=city>Paris</parameter></function>"
        "After")

    parsed = parse_assistant_output(qwen_text, _tool_config(), str(tmp_path))

    assert [event["type"] for event in parsed.events
            ] == ["reasoning", "content", "tool_call", "content"]
    assert parsed.reasoning == "plan"
    assert parsed.content == "BeforeAfter"
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Paris"}


def test_parses_gemma4_tool_calls(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": "gemma4_text"}))
    tools = [{
        "type": "function",
        "function": {
            "name": "set_volume",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {
                        "type": "integer"
                    },
                    "mode": {
                        "type": "string"
                    },
                },
            },
        },
    }]
    config = validate_tool_request([{
        "role": "user",
        "content": "Set volume"
    }], tools, "required")

    parsed = parse_assistant_output('call:set_volume{mode:louder,percent:75}',
                                    config, str(tmp_path))

    assert parsed.content == ""
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "set_volume"
    assert json.loads(parsed.tool_calls[0].arguments) == {
        "mode": "louder",
        "percent": 75,
    }

    tagged = parse_assistant_output(
        '<|tool_call>call:set_volume{mode:<|"|>a,b<|"|>,percent:40}'
        '<tool_call|>', config, str(tmp_path))
    assert json.loads(tagged.tool_calls[0].arguments) == {
        "mode": "a,b",
        "percent": 40,
    }

    premature = parse_assistant_output(
        "call:set_volume{percent:75,mode:louder}"
        "I have already changed the volume.",
        config,
        str(tmp_path),
    )
    assert premature.content == ""
    assert len(premature.tool_calls) == 1
    assert premature.tool_calls[0].name == "set_volume"
    assert json.loads(premature.tool_calls[0].arguments) == {
        "mode": "louder",
        "percent": 75,
    }


def test_rejects_malformed_gemma4_tool_call(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": "gemma4_text"}))
    config = _tool_config()
    text = "call:get_weather{city:Paris,city:London}"

    with pytest.raises(ToolProtocolError):
        parse_assistant_output(text, config, str(tmp_path))


def test_filters_forced_tool(tmp_path):
    text = "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>"
    parsed = parse_assistant_output(
        text,
        _tool_config({
            "type": "function",
            "function": {
                "name": "get_weather"
            },
        }),
        str(tmp_path),
    )

    assert parsed.tool_calls == []
    assert parsed.content == text


class _FakeLLM:

    def __init__(self, model_dir):
        self.model_dir = str(model_dir)
        self._model_id = "fake-model"

    def _make_generation_request(self, messages, params, **kw):
        return object()  # the streaming path prebuilds (and reuses) a request

    def generate_stream(self,
                        messages,
                        params,
                        *,
                        tools=None,
                        tool_choice=None,
                        tool_config=None,
                        prebuilt_request=None,
                        admission_handoff=None):
        yield StreamDelta(text="<think>plan</think>", finished=False)
        yield StreamDelta(
            text="<tool_call>{\"name\":\"get_weather\","
            "\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
            finished=True,
            finish_reason="stop",
        )


def test_builds_tool_response_shapes(tmp_path):
    config = _tool_config()
    message, has_tool_calls = _build_message_body(
        "<tool_call>{\"name\":\"get_weather\","
        "\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
        config,
        str(tmp_path),
    )

    assert has_tool_calls
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    chunks = list(
        _generate_stream_sse(
            _FakeLLM(tmp_path),
            [{
                "role": "user",
                "content": "Weather?"
            }],
            object(),
            "chatcmpl-test",
            False,
            tool_config=config,
        ))
    payloads = [
        json.loads(chunk.removeprefix("data: ")) for chunk in chunks
        if chunk.startswith("data: {")
    ]

    assert payloads[1]["choices"][0]["delta"]["reasoning"] == "plan"
    assert "tool_calls" in payloads[2]["choices"][0]["delta"]
    assert payloads[3]["choices"][0]["delta"]["tool_calls"][0]["function"][
        "arguments"] == '{"city": "Paris"}'
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


class _FakeTokenLLM:
    """Fake LLM whose deltas carry token ids, for usage accounting tests."""

    def __init__(self, model_dir):
        self.model_dir = str(model_dir)
        self._model_id = "fake-model"

    def _make_generation_request(self, messages, params, **kw):
        return object()

    def generate_stream(self,
                        messages,
                        params,
                        *,
                        tools=None,
                        tool_choice=None,
                        tool_config=None,
                        prebuilt_request=None,
                        admission_handoff=None):
        yield StreamDelta(text="Hello ", token_ids=[1, 2, 3], finished=False)
        yield StreamDelta(text="world",
                          token_ids=[4, 5],
                          finished=True,
                          finish_reason="stop")


def test_stream_usage_chunk(tmp_path):
    """include_usage adds exactly one final usage chunk; absent otherwise."""

    def run(**kwargs):
        return list(
            _generate_stream_sse(
                _FakeTokenLLM(tmp_path),
                [{
                    "role": "user",
                    "content": "Hi"
                }],
                object(),
                "chatcmpl-run",
                False,
                tool_config=None,
                **kwargs,
            ))

    chunks = run(include_usage=True, prompt_tokens=7)
    assert chunks[-1] == "data: [DONE]\n\n"
    usage_payload = json.loads(chunks[-2].removeprefix("data: "))
    assert usage_payload["choices"] == []
    assert usage_payload["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
    }

    plain = [
        json.loads(c.removeprefix("data: ")) for c in run()
        if c.startswith("data: {")
    ]
    assert all("usage" not in payload for payload in plain)


def test_tool_stream_usage_chunk(tmp_path):
    config = _tool_config()
    chunks = list(
        _generate_stream_sse(
            _FakeLLM(tmp_path),
            [{
                "role": "user",
                "content": "Weather?"
            }],
            object(),
            "chatcmpl-toolusage",
            False,
            tool_config=config,
            include_usage=True,
            prompt_tokens=100,
        ))
    assert chunks[-1] == "data: [DONE]\n\n"
    usage_payload = json.loads(chunks[-2].removeprefix("data: "))
    assert usage_payload["usage"]["prompt_tokens"] == 100
    assert usage_payload["usage"]["total_tokens"] == (
        100 + usage_payload["usage"]["completion_tokens"])


def _gemma_config(tmp_path, tool_choice="auto"):
    (tmp_path / "config.json").write_text(json.dumps({"model": "gemma4_text"}))
    tools = [{
        "type": "function",
        "function": {
            "name": "set_volume",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {
                        "type": "integer"
                    },
                },
            },
        },
    }]
    return validate_tool_request([{
        "role": "user",
        "content": "Set volume"
    }], tools, tool_choice)


def _feed_all(parser, deltas):
    """Return per-delta event lists plus the finish events."""
    out = [parser.feed(d) for d in deltas]
    out.append(parser.finish())
    return out


def _texts(events):
    return [e["text"] for e in events if e["type"] == "content"]


def test_gemma4_stream_parser_releases_prose_incrementally(tmp_path):
    config = _gemma_config(tmp_path)
    parser = make_stream_parser(config,
                                str(tmp_path),
                                strip_tokens=("<|im_end|>", ))

    per_delta = _feed_all(parser, ["Owls ", "can ", "rotate<", "|to", "wn"])
    assert _texts(per_delta[0]) == ["Owls "]
    assert _texts(per_delta[1]) == ["can "]
    # "<" and "<|to" may still open a tool-call block: held back.
    assert _texts(per_delta[2]) == ["rotate"]
    assert per_delta[3] == []
    assert _texts(per_delta[4]) == ["<|town"]
    assert per_delta[5] == []


def test_gemma4_stream_parser_block_after_prose(tmp_path):
    config = _gemma_config(tmp_path)
    parser = make_stream_parser(config, str(tmp_path))

    per_delta = _feed_all(parser, [
        "Sure. <|tool_call>call:set_vol",
        "ume{percent:40}<tool_ca",
        "ll|> Done.",
    ])
    assert _texts(per_delta[0]) == ["Sure. "]
    assert per_delta[1] == []
    kinds = [e["type"] for e in per_delta[2]]
    assert kinds == ["tool_call"]
    call = per_delta[2][0]["tool_call"]
    assert call.name == "set_volume"
    assert json.loads(call.arguments) == {"percent": 40}
    assert per_delta[3] == []


def test_gemma4_stream_parser_bare_call_holds_and_drops_trailing_prose(
        tmp_path):
    config = _gemma_config(tmp_path, "required")
    parser = make_stream_parser(config, str(tmp_path))

    per_delta = _feed_all(
        parser,
        ["ca", "ll:set_volume{per", "cent:75}", "I have changed the volume."])
    assert per_delta[:4] == [[], [], [], []]
    assert [e["type"] for e in per_delta[4]] == ["tool_call"]
    assert json.loads(per_delta[4][0]["tool_call"].arguments) == {
        "percent": 75
    }


def test_gemma4_stream_parser_start_holdback_only_for_call_prefix(tmp_path):
    config = _gemma_config(tmp_path)
    parser = make_stream_parser(config, str(tmp_path))
    per_delta = _feed_all(parser, ["  ", "ca", "ts fly"])
    assert per_delta[0] == [] and per_delta[1] == []
    assert _texts(per_delta[2]) == ["  cats fly"]

    parser = make_stream_parser(config, str(tmp_path))
    assert _texts(parser.feed("Owls")) == ["Owls"]

    parser = make_stream_parser(config, str(tmp_path))
    assert parser.feed("ca") == []
    assert _texts(parser.finish()) == ["ca"]


def test_gemma4_stream_parser_malformed_block_fails_closed(tmp_path):
    config = _gemma_config(tmp_path)
    parser = make_stream_parser(config, str(tmp_path))
    block = "<|tool_call>call:set_volume{percent:1,percent:2}<tool_call|>"
    assert _texts(parser.feed("Hi ")) == ["Hi "]
    with pytest.raises(ToolProtocolError):
        parser.feed(block)
    with pytest.raises(ToolProtocolError):
        parser.feed("Success!")
    with pytest.raises(ToolProtocolError):
        parser.finish()


def test_gemma4_stream_parser_unterminated_block_fails_closed(tmp_path):
    config = _gemma_config(tmp_path)
    parser = make_stream_parser(config, str(tmp_path))
    assert _texts(parser.feed("Hi <|tool_call>call:set_volume{")) == ["Hi "]
    with pytest.raises(ToolProtocolError) as raised:
        parser.finish()
    assert raised.value.reason == "incomplete"


def test_gemma4_stream_parser_strips_tokens(tmp_path):
    config = _gemma_config(tmp_path)
    parser = make_stream_parser(config,
                                str(tmp_path),
                                strip_tokens=("<|im_end|>", ))
    per_delta = _feed_all(parser, ["Bye<|im_", "end|>", "<|im_end|>"])
    assert _texts(per_delta[0]) == ["Bye"]
    assert per_delta[1] == [] and per_delta[2] == [] and per_delta[3] == []

    parser = make_stream_parser(config,
                                str(tmp_path),
                                strip_tokens=("<|im_end|>", ))
    per_delta = _feed_all(parser, ["call:set_volume{percent:5}<|im_end|>"])
    assert [e["type"] for e in per_delta[1]] == ["tool_call"]


def test_gemma4_stream_parser_matches_whole_output_parser(tmp_path):
    """Chunking must not change the event sequence the whole-output parser
    produces, for every split point of representative outputs."""
    config = _gemma_config(tmp_path)
    samples = [
        "Owls can rotate their heads. <|tool_call>call:set_volume{percent:40}"
        "<tool_call|> Done.",
        "call:set_volume{percent:75}I have changed the volume.",
        "  call:set_volume{percent:75}",
        "Plain <| prose with < angle brackets<|im_end|>",
        "<|tool_call>call:set_volume{percent:3}<tool_call|>"
        "<|tool_call>call:set_volume{percent:4}<tool_call|>",
    ]
    for sample in samples:
        expected = parse_assistant_output(sample.replace("<|im_end|>", ""),
                                          config, str(tmp_path))
        for split in range(len(sample) + 1):
            parser = make_stream_parser(config,
                                        str(tmp_path),
                                        strip_tokens=("<|im_end|>", ))
            events = parser.feed(sample[:split]) + parser.feed(
                sample[split:]) + parser.finish()
            content = "".join(_texts(events))
            calls = [(e["tool_call"].name, e["tool_call"].arguments)
                     for e in events if e["type"] == "tool_call"]
            assert content == expected.content, (sample, split)
            assert calls == [(c.name, c.arguments)
                             for c in expected.tool_calls], (sample, split)


def test_generic_stream_parser_buffers_until_finish(tmp_path):
    config = _tool_config()
    parser = make_stream_parser(config, str(tmp_path))
    per_delta = _feed_all(parser, [
        "<tool_call>{\"name\":\"get_weather\",",
        "\"arguments\":{\"city\":\"Paris\"}}</tool_call>"
    ])
    assert per_delta[0] == [] and per_delta[1] == []
    assert [e["type"] for e in per_delta[2]] == ["tool_call"]


_INVALID_GEMMA_TURNS = [
    "call:get_weather{city:Berkeley, California}",
    "call:get_weather{city:Berkeley, California}It is sunny.",
    'call:get_weather{city:"Berkeley, California}',
    'call:get_weather{city:<|"|>Berkeley, California}',
    'call:get_weather{city:<|"|>Paris<|"|>,city:<|"|>London<|"|>}',
    'call:get_weather{"city":"Paris","city":"London"}',
    'call:unknown{city:"Paris"}',
    "call:",
    "call:!not_a_name{}Success.",
    "call:get_weather{",
    "<|tool_call>",
    "<|tool_call>call:get_weather{city:Paris}",
    "<|tool_call>nonsense<tool_call|>Success.",
    "<|tool_call>call:unknown{}<tool_call|>Success.",
    "<|tool_call>call:get_weather{city:Paris,city:London}<tool_call|>Success.",
    "call:nothing here <|tool_call>call:get_weather{city:Paris}<tool_call|>",
    "<|tool_call>call:get_weather{city:Paris,city:London}<tool_call|>"
    "<|tool_call>call:get_weather{city:Paris}<tool_call|>",
]


@pytest.mark.parametrize("sample", _INVALID_GEMMA_TURNS)
def test_gemma4_protocol_failures_never_become_content(tmp_path, sample):
    _gemma_config(tmp_path)
    config = _tool_config()
    with pytest.raises(ToolProtocolError):
        parse_assistant_output(sample, config, str(tmp_path))
    with pytest.raises(ToolProtocolError):
        _build_message_body(sample, config, str(tmp_path))

    chunks = [[sample[:split], sample[split:]]
              for split in range(len(sample) + 1)] + [list(sample)]
    for pieces in chunks:
        parser = make_stream_parser(config, str(tmp_path))
        events = []
        with pytest.raises(ToolProtocolError) as raised:
            for piece in pieces:
                events.extend(parser.feed(piece))
            events.extend(parser.finish())
        assert events == [], pieces
        assert raised.value.to_openai()["code"] == "invalid_tool_call"
        assert sample not in str(raised.value)
        with pytest.raises(ToolProtocolError):
            parser.feed("The operation succeeded.")
        with pytest.raises(ToolProtocolError):
            parser.finish()


@pytest.mark.parametrize("arguments, expected", [
    ('{city:<|"|>Berkeley, California<|"|>}', {
        "city": "Berkeley, California"
    }),
    ('{city:"Berkeley, California"}', {
        "city": "Berkeley, California"
    }),
    ('{"city":"Berkeley, California"}', {
        "city": "Berkeley, California"
    }),
    ('{city:"A } comma, quote \\" and slash \\\\"}', {
        "city": 'A } comma, quote " and slash \\'
    }),
    ('{city:<|"|>A } comma, quote " and colon :<|"|>}', {
        "city": 'A } comma, quote " and colon :'
    }),
    ('{nested:{items:[1,true,null,{label:"a,b"}]}}', {
        "nested": {
            "items": [1, True, None, {
                "label": "a,b"
            }]
        }
    }),
])
@pytest.mark.parametrize("tagged", [False, True])
def test_gemma4_quoted_arguments_every_split(tmp_path, arguments, expected,
                                             tagged):
    _gemma_config(tmp_path)
    config = _tool_config()
    sample = "call:get_weather" + arguments
    if tagged:
        sample = "<|tool_call>" + sample + "<tool_call|>"
    sample += "I already did it."
    parsed = parse_assistant_output(sample, config, str(tmp_path))
    assert parsed.content == ""
    assert json.loads(parsed.tool_calls[0].arguments) == expected
    chunks = [[sample[:split], sample[split:]]
              for split in range(len(sample) + 1)] + [list(sample)]
    for pieces in chunks:
        parser = make_stream_parser(config, str(tmp_path))
        events = sum(_feed_all(parser, pieces), [])
        assert [e["type"] for e in events] == ["tool_call"]
        assert json.loads(events[0]["tool_call"].arguments) == expected


def test_gemma4_forced_tool_rejects_other_declared_name(tmp_path):
    config = _gemma_config(tmp_path, "set_volume")
    config.tools.extend(_tools())
    with pytest.raises(ToolProtocolError) as raised:
        parse_assistant_output("call:get_weather{city:Paris}", config,
                               str(tmp_path))
    assert raised.value.reason == "unauthorized"


def test_gemma4_illustrative_mid_prose_call_is_not_protocol(tmp_path):
    config = _gemma_config(tmp_path)
    sample = "For example, call:set_volume{percent:40} is tool syntax."
    assert parse_assistant_output(sample, config,
                                  str(tmp_path)).content == sample
    for split in range(len(sample) + 1):
        parser = make_stream_parser(config, str(tmp_path))
        events = sum(_feed_all(parser, [sample[:split], sample[split:]]), [])
        assert "".join(_texts(events)) == sample


def test_gemma4_leading_stripped_token_preserves_protocol_classification(
        tmp_path):
    config = _gemma_config(tmp_path)
    sample = " <|im_end|> call:unknown{}Success."
    for split in range(len(sample) + 1):
        parser = make_stream_parser(config,
                                    str(tmp_path),
                                    strip_tokens=("<|im_end|>", ))
        events = []
        with pytest.raises(ToolProtocolError):
            events.extend(parser.feed(sample[:split]))
            events.extend(parser.feed(sample[split:]))
            events.extend(parser.finish())
        assert events == []


class _FakeGemmaLLM:

    def __init__(self, model_dir, pieces):
        self.model_dir = str(model_dir)
        self._model_id = "fake-gemma"
        self._pieces = pieces

    def _make_generation_request(self, messages, params, **kw):
        return object()

    def generate_stream(self, messages, params, **kw):
        for piece in self._pieces[:-1]:
            yield StreamDelta(text=piece, token_ids=[1], finished=False)
        yield StreamDelta(text=self._pieces[-1],
                          token_ids=[1],
                          finished=True,
                          finish_reason="stop")


def _stream_payloads(llm, config):
    chunks = list(
        _generate_stream_sse(llm, [{
            "role": "user",
            "content": "Owls?"
        }],
                             object(),
                             "chatcmpl-gemma",
                             False,
                             tool_config=config))
    assert chunks[-1] == "data: [DONE]\n\n"
    return [
        json.loads(c.removeprefix("data: "))["choices"][0] for c in chunks
        if c.startswith("data: {")
    ]


def test_tool_stream_sse_streams_content_per_delta(tmp_path):
    config = _gemma_config(tmp_path)
    pieces = ["Owls ", "can ", "rotate ", "their ", "heads."]
    choices = _stream_payloads(_FakeGemmaLLM(tmp_path, pieces), config)
    contents = [
        c["delta"]["content"] for c in choices if "content" in c["delta"]
    ]
    assert contents == pieces
    assert choices[-1]["finish_reason"] == "stop"


def test_tool_stream_sse_prose_then_block_call(tmp_path):
    config = _gemma_config(tmp_path)
    pieces = [
        "Sure. ", "<|tool_call>call:set_volume", "{percent:40}", "<tool_call|>"
    ]
    choices = _stream_payloads(_FakeGemmaLLM(tmp_path, pieces), config)
    deltas = [c["delta"] for c in choices]
    assert deltas[1] == {"content": "Sure. "}
    head, args = deltas[2]["tool_calls"][0], deltas[3]["tool_calls"][0]
    assert head["index"] == 0 and head["type"] == "function"
    assert head["function"] == {"name": "set_volume", "arguments": ""}
    assert args == {"index": 0, "function": {"arguments": '{"percent": 40}'}}
    assert choices[-1]["finish_reason"] == "tool_calls"


def test_tool_stream_sse_forced_bare_call_has_no_content(tmp_path):
    config = _gemma_config(tmp_path, {
        "type": "function",
        "function": {
            "name": "set_volume"
        }
    })
    pieces = ["call:set_", "volume{percent:75}", "Volume set."]
    choices = _stream_payloads(_FakeGemmaLLM(tmp_path, pieces), config)
    deltas = [c["delta"] for c in choices]
    assert not any("content" in d for d in deltas)
    assert [d["tool_calls"][0]["index"] for d in deltas
            if "tool_calls" in d] == [0, 0]
    assert choices[-1]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("sample", _INVALID_GEMMA_TURNS)
def test_tool_stream_sse_protocol_errors_are_typed_and_private(
        tmp_path, sample):
    _gemma_config(tmp_path)
    config = _tool_config()
    for split in range(len(sample) + 1):
        llm = _FakeGemmaLLM(tmp_path, [sample[:split], sample[split:]])
        chunks = list(
            _generate_stream_sse(llm, [],
                                 object(),
                                 "test",
                                 False,
                                 tool_config=config))
        assert chunks[-1] == "data: [DONE]\n\n"
        payloads = [json.loads(c.removeprefix("data: ")) for c in chunks[:-1]]
        errors = [p["error"] for p in payloads if "error" in p]
        assert len(errors) == 1
        assert errors[0]["type"] == "tool_protocol_error"
        assert errors[0]["code"] == "invalid_tool_call"
        assert errors[0][
            "message"] == "The model generated an invalid tool call."
        choices = [p["choices"][0] for p in payloads if "choices" in p]
        assert not any("content" in c["delta"] or "tool_calls" in c["delta"]
                       for c in choices)
        assert choices[-1]["finish_reason"] == "error"


def test_tool_stream_sse_preceding_prose_cannot_be_retracted(tmp_path):
    config = _gemma_config(tmp_path)
    llm = _FakeGemmaLLM(tmp_path, [
        "Let me check. ",
        "<|tool_call>call:set_volume{percent:1,percent:2}<tool_call|>",
        "Success.",
    ])
    chunks = list(
        _generate_stream_sse(llm, [],
                             object(),
                             "test",
                             False,
                             tool_config=config))
    payloads = [json.loads(c.removeprefix("data: ")) for c in chunks[:-1]]
    choices = [p["choices"][0] for p in payloads if "choices" in p]
    assert [c["delta"]["content"] for c in choices
            if "content" in c["delta"]] == ["Let me check. "]
    assert choices[-1]["finish_reason"] == "error"
    assert len([p for p in payloads if "error" in p]) == 1


def test_tool_stream_sse_error_overrides_prior_tool_finish(tmp_path):
    config = _gemma_config(tmp_path)
    llm = _FakeGemmaLLM(tmp_path, [
        "<|tool_call>call:set_volume{percent:1}<tool_call|>",
        "<|tool_call>call:unknown{}<tool_call|>",
    ])
    chunks = list(
        _generate_stream_sse(llm, [],
                             object(),
                             "test",
                             False,
                             tool_config=config))
    payloads = [json.loads(c.removeprefix("data: ")) for c in chunks[:-1]]
    assert payloads[-2]["error"]["code"] == "invalid_tool_call"
    assert payloads[-1]["choices"][0]["finish_reason"] == "error"


def test_thinking_state_machine_holds_only_partial_tags():
    sm = _ThinkingStateMachine(True)
    assert list(sm.feed("Owls can rotate")) == [("content", "Owls can rotate")]
    assert list(sm.feed(" <th")) == [("content", " ")]
    assert list(sm.feed("ink>plan</th")) == [("reasoning", "plan")]
    assert list(sm.feed("ink>done")) == [("content", "done")]
    assert list(sm.flush()) == []
