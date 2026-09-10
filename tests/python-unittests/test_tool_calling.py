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

from experimental.server.parsing.tool_calling import (
    ToolProtocolError, _parser_name_for_model, _select_parser,
    list_tool_parsers, parse_assistant_output, stream_assistant_output,
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
                    }
                },
            },
        },
    }]


def _config(tool_choice="auto", parallel=True):
    return validate_tool_request(
        [{
            "role": "user",
            "content": "Weather?"
        }],
        _tools(),
        tool_choice,
        parallel,
    )


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
                "arguments": '{"city":"Paris"}',
            },
        }],
    }, {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": {
            "temperature": 22
        },
    }]
    config = validate_tool_request(messages, _tools(), "required", False)
    assert config.tool_choice == "required"
    assert not config.parallel_tool_calls

    with pytest.raises(ValueError, match="Unknown forced tool name"):
        validate_tool_request(messages, _tools(), {
            "type": "function",
            "function": {
                "name": "missing"
            },
        })
    with pytest.raises(ValueError, match="Dangling tool_call_id"):
        validate_tool_request([{
            "role": "tool",
            "tool_call_id": "missing",
            "content": "42",
        }])


def test_parses_reasoning_and_multiple_tool_calls(tmp_path):
    text = (
        "<think>plan</think>Before"
        "<function=get_weather><parameter=city>Paris</parameter></function>"
        '<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}'
        "</tool_call>After")
    parsed = parse_assistant_output(text,
                                    _config(),
                                    str(tmp_path),
                                    reasoning_parser="qwen3")
    assert parsed.reasoning == "plan"
    assert parsed.content == "BeforeAfter"
    assert [json.loads(call.arguments)["city"]
            for call in parsed.tool_calls] == ["Paris", "Tokyo"]


def test_normal_output_is_content_without_reasoning_parser(tmp_path):
    parsed = parse_assistant_output("ordinary answer", _config("none"),
                                    str(tmp_path))
    assert parsed.content == "ordinary answer"
    assert parsed.reasoning == ""


def test_filters_forced_tool(tmp_path):
    text = '<tool_call>{"name":"other","arguments":{}}</tool_call>'
    parsed = parse_assistant_output(
        text,
        _config({
            "type": "function",
            "function": {
                "name": "get_weather"
            },
        }),
        str(tmp_path),
    )
    assert parsed.tool_calls == []
    assert parsed.content == text


def test_streams_content_reasoning_and_tools_across_chunk_boundaries(tmp_path):
    parser = stream_assistant_output(_config(),
                                     str(tmp_path),
                                     reasoning_parser="qwen3")
    events = []
    for chunk in (
            "<th",
            "ink>plan</think>Before<tool_",
            'call>{"name":"get_weather","arguments":{"city":',
            '"Paris"}}</tool_call>After',
    ):
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())

    assert "".join(event["text"] for event in events
                   if event["type"] == "reasoning") == "plan"
    assert "".join(event["text"] for event in events
                   if event["type"] == "content") == "BeforeAfter"
    calls = [
        event["tool_call"] for event in events if event["type"] == "tool_call"
    ]
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Paris"}


def test_stream_parser_flushes_untagged_provider_format(tmp_path):
    parser = stream_assistant_output(_config(), str(tmp_path))
    events = list(parser.feed('get_weather(city="Paris")'))
    events.extend(parser.flush())

    assert len(events) == 1
    assert events[0]["type"] == "tool_call"
    assert json.loads(events[0]["tool_call"].arguments) == {"city": "Paris"}


# ---------------------------------------------------------------------------
# Gemma 4 native tool protocol
# ---------------------------------------------------------------------------


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


def _gemma_stream(config, model_dir, strip_tokens=()):
    return _select_parser(str(model_dir),
                          "auto").stream_events(config, strip_tokens)


def _feed_all(parser, deltas):
    """Return per-delta event lists plus the finish events."""
    out = [parser.feed(d) for d in deltas]
    out.append(parser.finish())
    return out


def _texts(events):
    return [e["text"] for e in events if e["type"] == "content"]


def test_gemma4_parser_is_selected_from_runtime_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": "gemma4_text"}))
    assert _parser_name_for_model(str(tmp_path)) == "gemma4"
    assert "gemma4" in list_tool_parsers()


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
        "I have already changed the volume.", config, str(tmp_path))
    assert premature.content == ""
    assert len(premature.tool_calls) == 1
    assert json.loads(premature.tool_calls[0].arguments) == {
        "mode": "louder",
        "percent": 75,
    }


def test_rejects_malformed_gemma4_tool_call(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": "gemma4_text"}))
    with pytest.raises(ToolProtocolError):
        parse_assistant_output("call:get_weather{city:Paris,city:London}",
                               _config(), str(tmp_path))


def test_gemma4_stream_parser_releases_prose_incrementally(tmp_path):
    config = _gemma_config(tmp_path)
    parser = _gemma_stream(config, tmp_path, strip_tokens=("<|im_end|>", ))

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
    parser = _gemma_stream(config, tmp_path)

    per_delta = _feed_all(parser, [
        "Sure. <|tool_call>call:set_vol",
        "ume{percent:40}<tool_ca",
        "ll|> Done.",
    ])
    assert _texts(per_delta[0]) == ["Sure. "]
    assert per_delta[1] == []
    assert [e["type"] for e in per_delta[2]] == ["tool_call"]
    call = per_delta[2][0]["tool_call"]
    assert call.name == "set_volume"
    assert json.loads(call.arguments) == {"percent": 40}
    assert per_delta[3] == []


def test_gemma4_stream_parser_bare_call_holds_and_drops_trailing_prose(
        tmp_path):
    config = _gemma_config(tmp_path, "required")
    parser = _gemma_stream(config, tmp_path)

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
    parser = _gemma_stream(config, tmp_path)
    per_delta = _feed_all(parser, ["  ", "ca", "ts fly"])
    assert per_delta[0] == [] and per_delta[1] == []
    assert _texts(per_delta[2]) == ["  cats fly"]

    parser = _gemma_stream(config, tmp_path)
    assert _texts(parser.feed("Owls")) == ["Owls"]

    parser = _gemma_stream(config, tmp_path)
    assert parser.feed("ca") == []
    assert _texts(parser.finish()) == ["ca"]


def test_gemma4_stream_parser_malformed_block_fails_closed(tmp_path):
    config = _gemma_config(tmp_path)
    parser = _gemma_stream(config, tmp_path)
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
    parser = _gemma_stream(config, tmp_path)
    assert _texts(parser.feed("Hi <|tool_call>call:set_volume{")) == ["Hi "]
    with pytest.raises(ToolProtocolError) as raised:
        parser.finish()
    assert raised.value.reason == "incomplete"


def test_gemma4_stream_parser_strips_tokens(tmp_path):
    config = _gemma_config(tmp_path)
    parser = _gemma_stream(config, tmp_path, strip_tokens=("<|im_end|>", ))
    per_delta = _feed_all(parser, ["Bye<|im_", "end|>", "<|im_end|>"])
    assert _texts(per_delta[0]) == ["Bye"]
    assert per_delta[1] == [] and per_delta[2] == [] and per_delta[3] == []

    parser = _gemma_stream(config, tmp_path, strip_tokens=("<|im_end|>", ))
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
            parser = _gemma_stream(config,
                                   tmp_path,
                                   strip_tokens=("<|im_end|>", ))
            events = parser.feed(sample[:split]) + parser.feed(
                sample[split:]) + parser.finish()
            content = "".join(_texts(events))
            calls = [(e["tool_call"].name, e["tool_call"].arguments)
                     for e in events if e["type"] == "tool_call"]
            assert content == expected.content, (sample, split)
            assert calls == [(c.name, c.arguments)
                             for c in expected.tool_calls], (sample, split)


def test_gemma4_openai_stream_events_match_generic_shape(tmp_path):
    config = _gemma_config(tmp_path)
    parser = _select_parser(str(tmp_path), "gemma4").stream(config)
    events = list(parser.feed("Sure. <|tool_call>call:set_volume{percent:40}"))
    events += list(parser.feed("<tool_call|>"))
    events += list(parser.flush())
    assert [e.kind for e in events
            ] == ["content", "tool_head", "tool_args", "tool_done"]
    assert events[0].text == "Sure. "
    assert events[1].name == "set_volume" and events[1].index == 0
    assert json.loads(events[2].text) == {"percent": 40}


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
    config = _config()
    with pytest.raises(ToolProtocolError):
        parse_assistant_output(sample, config, str(tmp_path))

    chunks = [[sample[:split], sample[split:]]
              for split in range(len(sample) + 1)] + [list(sample)]
    for pieces in chunks:
        parser = _gemma_stream(config, tmp_path)
        events = []
        with pytest.raises(ToolProtocolError) as raised:
            for piece in pieces:
                events.extend(parser.feed(piece))
            events.extend(parser.finish())
        assert events == [], pieces
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
    config = _config()
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
        parser = _gemma_stream(config, tmp_path)
        events = sum(_feed_all(parser, pieces), [])
        assert [e["type"] for e in events] == ["tool_call"]
        assert json.loads(events[0]["tool_call"].arguments) == expected


def test_gemma4_forced_tool_rejects_other_declared_name(tmp_path):
    config = _gemma_config(tmp_path, {
        "type": "function",
        "function": {
            "name": "set_volume"
        }
    })
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
        parser = _gemma_stream(config, tmp_path)
        events = sum(_feed_all(parser, [sample[:split], sample[split:]]), [])
        assert "".join(_texts(events)) == sample


def test_gemma4_leading_stripped_token_preserves_protocol_classification(
        tmp_path):
    config = _gemma_config(tmp_path)
    sample = " <|im_end|> call:unknown{}Success."
    for split in range(len(sample) + 1):
        parser = _gemma_stream(config, tmp_path, strip_tokens=("<|im_end|>", ))
        events = []
        with pytest.raises(ToolProtocolError):
            events.extend(parser.feed(sample[:split]))
            events.extend(parser.feed(sample[split:]))
            events.extend(parser.finish())
        assert events == []
