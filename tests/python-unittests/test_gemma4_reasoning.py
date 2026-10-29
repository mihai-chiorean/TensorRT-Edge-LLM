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
"""Gemma 4 ``<|channel>thought\\n...<channel|>`` reasoning parser."""

import json

import pytest

from experimental.server.config import parse_server_config
from experimental.server.parsing import tool_calling
from experimental.server.parsing.reasoning import (REASONING_PARSERS,
                                                   Gemma4ReasoningParser)
from experimental.server.parsing.tool_calling import (parse_assistant_output,
                                                      stream_assistant_output,
                                                      validate_tool_request)

_THOUGHT = "<|channel>thought\nCheck the weather.\nUse the tool.<channel|>"
_CALL = '<|tool_call>call:weather{location:<|"|>Paris<|"|>}<tool_call|>'


def _gemma_dir(tmp_path, config=None, name="gemma-4-e4b"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    if config is not None:
        (model_dir / "config.json").write_text(json.dumps(config),
                                               encoding="utf-8")
    return str(model_dir)


def _tools(tool_choice="auto"):
    return validate_tool_request([{
        "role": "user",
        "content": "Weather?"
    }], [{
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string"
                    }
                },
            },
        },
    }], tool_choice)


def _no_tools():
    return validate_tool_request([{"role": "user", "content": "hi"}])


def _split(deltas):
    deltas = list(deltas)
    reasoning = "".join(d.text for d in deltas if d.field == "reasoning")
    content = "".join(d.text for d in deltas if d.field == "content")
    return reasoning or None, content or None


def _stream(chunks, starts_open=False):
    stream = Gemma4ReasoningParser().stream(allow_implicit=starts_open)
    deltas = []
    for chunk in chunks:
        deltas.extend(stream.feed(chunk))
    deltas.extend(stream.flush())
    return _split(deltas)


@pytest.mark.parametrize(
    "config,name",
    [
        ({
            "model": "gemma4_text"
        }, "engines"),
        ({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"]
        }, "renamed"),
        # Nemotron Bolt is a Gemma 4 checkpoint with another family's name.
        ({
            "model_type": "gemma4"
        }, "NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"),
        (None, "gemma-4-e4b-it"),
    ])
def test_auto_selects_gemma4_by_model_family(tmp_path, config, name):
    model_dir = _gemma_dir(tmp_path, config, name)
    assert isinstance(REASONING_PARSERS.resolve("auto", model_dir),
                      Gemma4ReasoningParser)


def test_bolt_checkpoint_gets_the_gemma4_tool_parser(tmp_path):
    model_dir = _gemma_dir(tmp_path, {"model_type": "gemma4"},
                           "NVIDIA-Nemotron-3.5-Bolt-E4B-BF16")
    assert tool_calling._parser_name_for_model(model_dir) == "gemma4"


def test_other_families_keep_their_parsers(tmp_path):
    nemotron = _gemma_dir(tmp_path, {"model_type": "nemotron_h"},
                          "nemotron-nano")
    assert not isinstance(REASONING_PARSERS.resolve("auto", nemotron),
                          Gemma4ReasoningParser)
    assert REASONING_PARSERS.resolve("auto", nemotron).start_token == "<think>"


def test_cli_accepts_explicit_gemma4_parser():
    config = parse_server_config(
        ["/models/bolt", "--reasoning-parser", "gemma4"])
    assert config.api.reasoning_parser == "gemma4"
    assert isinstance(REASONING_PARSERS.resolve("gemma4", ""),
                      Gemma4ReasoningParser)


@pytest.mark.parametrize(
    "text,expected",
    [
        (_THOUGHT + "It is sunny.",
         ("Check the weather.\nUse the tool.", "It is sunny.")),
        ("It is sunny.", (None, "It is sunny.")),
        # Unterminated at max_tokens or at a stop string.
        ("<|channel>thought\nStill thinking", ("Still thinking", None)),
        ("<|channel>thought\n<channel|>Hi.", (None, "Hi.")),
        ("<|channel>thought<channel|>Hi.", (None, "Hi.")),
        ("<|channel>thought", (None, None)),
        # The generation prompt already opened the block.
        ("Compare the results.<channel|>Paris is warmer.",
         ("Compare the results.", "Paris is warmer.")),
        # Markers after the answer started are literal content.
        ("Hi <|channel>thought\nx<channel|>y",
         (None, "Hi <|channel>thought\nx<channel|>y")),
    ])
def test_extract_splits_the_thought_block(text, expected):
    assert Gemma4ReasoningParser().extract(text) == expected


def test_extract_follows_the_prompt_hint():
    parser = Gemma4ReasoningParser()
    assert parser.extract("Unfinished",
                          starts_open=True) == ("Unfinished", None)
    assert parser.extract("a<channel|>b", starts_open=True) == ("a", "b")
    assert parser.extract("a<channel|>b",
                          starts_open=False) == (None, "a<channel|>b")
    assert parser.extract(
        _THOUGHT + "x",
        starts_open=False) == ("Check the weather.\nUse the tool.", "x")


@pytest.mark.parametrize("text", [
    _THOUGHT + "It is sunny.",
    "It is sunny.",
    "<|channel>thought\nStill thinking",
    "<|channel>thought\n<channel|>Hi.",
    "<|channel>thought<channel|>Hi.",
    "<|chan",
    "<|channel>tho",
])
def test_stream_matches_extract_at_every_split(text):
    expected = Gemma4ReasoningParser().extract(text)
    assert _stream(list(text)) == expected
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            assert _stream([text[:i], text[i:j], text[j:]]) == expected, (i, j)


def test_stream_releases_plain_answers_and_reasoning_without_lag():
    stream = Gemma4ReasoningParser().stream()
    assert _split(stream.feed("Hello")) == (None, "Hello")

    stream = Gemma4ReasoningParser().stream()
    assert _split(stream.feed("<|channel>thought\nabc<chan")) == ("abc", None)
    assert _split(stream.feed("nel|>Done")) == (None, "Done")


def test_stream_that_starts_inside_the_block():
    assert _stream(["Compare", " the results.<chan", "nel|>Paris."],
                   starts_open=True) == ("Compare the results.", "Paris.")
    assert _stream(["Unfinished"], starts_open=True) == ("Unfinished", None)
    # The model may still reopen the block explicitly.
    assert _stream(["<|channel>thought\nx<channel|>y"],
                   starts_open=True) == ("x", "y")


@pytest.mark.parametrize("messages,templated,generation_prompt,expected", [
    ([{
        "role": "user",
        "content": "hi"
    }], True, True, False),
    ([{
        "role": "user",
        "content": "hi"
    }, {
        "role": "assistant",
        "tool_calls": []
    }, {
        "role": "tool",
        "content": "sunny"
    }], True, True, True),
    ([{
        "role": "assistant",
        "tool_responses": [{
            "name": "weather",
            "response": "sunny"
        }]
    }], True, True, True),
    ([{
        "role": "tool",
        "content": "sunny"
    }], True, False, False),
    ([{
        "role": "tool",
        "content": "sunny"
    }], False, True, False),
    ([{
        "role": "user",
        "content": "<|turn>model\n<|channel>thought\n"
    }], False, True, True),
    ([{
        "role": "user",
        "content": [{
            "type": "text",
            "text": "<|channel>thought\nx<channel|>"
        }]
    }], False, True, False),
    ([{
        "role":
        "user",
        "content": [
            "<|turn>model\n", {
                "text": "<|channel>"
            }, {
                "type": "input_text",
                "text": "thought\n"
            }
        ]
    }], False, True, True),
])
def test_prompt_opens_reasoning_after_a_tool_response(messages, templated,
                                                      generation_prompt,
                                                      expected):
    assert Gemma4ReasoningParser().prompt_opens_reasoning(
        messages, templated=templated,
        add_generation_prompt=generation_prompt) is expected


@pytest.mark.parametrize("call", [_CALL, 'call:weather{location:Paris}'])
def test_tool_call_after_thought_is_parsed_from_the_answer(tmp_path, call):
    model_dir = _gemma_dir(tmp_path, {"model": "gemma4_text"})
    parsed = parse_assistant_output(_THOUGHT + call,
                                    _tools(),
                                    model_dir,
                                    reasoning_parser="gemma4")
    assert parsed.reasoning == "Check the weather.\nUse the tool."
    assert parsed.content == ""
    assert [c.name for c in parsed.tool_calls] == ["weather"]
    assert json.loads(parsed.tool_calls[0].arguments)["location"] == "Paris"


def test_tool_markup_inside_the_thought_is_not_a_call(tmp_path):
    model_dir = _gemma_dir(tmp_path, {"model": "gemma4_text"})
    text = ("<|channel>thought\nMaybe " + _CALL +
            "? No.<channel|>It is sunny.")
    parsed = parse_assistant_output(text,
                                    _tools(),
                                    model_dir,
                                    reasoning_parser="gemma4")
    assert parsed.tool_calls == []
    assert parsed.reasoning == "Maybe " + _CALL + "? No."
    assert parsed.content == "It is sunny."


def test_unterminated_thought_with_tools_is_reasoning_only(tmp_path):
    model_dir = _gemma_dir(tmp_path, {"model": "gemma4_text"})
    parsed = parse_assistant_output("<|channel>thought\nStill going",
                                    _tools(),
                                    model_dir,
                                    reasoning_parser="gemma4")
    assert parsed.reasoning == "Still going"
    assert parsed.content == "" and parsed.tool_calls == []


@pytest.mark.parametrize("call", [_CALL, 'call:weather{location:Paris}'])
def test_assistant_stream_parses_tool_after_split_thought(tmp_path, call):
    model_dir = _gemma_dir(tmp_path, {"model": "gemma4_text"})
    text = _THOUGHT + call
    for size in (1, 3, 7, 11):
        parser = stream_assistant_output(_tools(),
                                         model_dir,
                                         reasoning_parser="gemma4")
        events = []
        for i in range(0, len(text), size):
            events.extend(parser.feed(text[i:i + size]))
        events.extend(parser.flush())
        assert "".join(e["text"] for e in events
                       if e["type"] == "reasoning") == (
                           "Check the weather.\nUse the tool.")
        assert not [e for e in events if e["type"] == "content" and e["text"]]
        calls = [e["tool_call"] for e in events if e["type"] == "tool_call"]
        assert [c.name for c in calls] == ["weather"]
        assert [e["type"] for e in events
                if e["type"] != "content"][-1] == "tool_call"


def test_assistant_stream_without_tools_uses_the_prompt_hint(tmp_path):
    model_dir = _gemma_dir(tmp_path, {"model": "gemma4_text"})
    for reasoning_open, expected in ((None, ("", "plan<channel|>answer")),
                                     (True, ("plan", "answer"))):
        parser = stream_assistant_output(_no_tools(),
                                         model_dir,
                                         reasoning_parser="gemma4",
                                         reasoning_open=reasoning_open)
        events = list(parser.feed("plan<chan")) + list(
            parser.feed("nel|>answer")) + list(parser.flush())
        assert ("".join(e["text"] for e in events if e["type"] == "reasoning"),
                "".join(e["text"] for e in events
                        if e["type"] == "content")) == expected


def test_python_api_applies_the_prompt_hint(tmp_path):
    import contextlib

    from experimental.server.runtime.engine import LLM, SamplingParams

    model_dir = _gemma_dir(tmp_path, {"model": "gemma4_text"})
    llm = LLM.__new__(LLM)
    llm._model_dir = model_dir
    llm._admission = contextlib.nullcontext
    llm._ensure_open = lambda: None
    llm._make_generation_request = lambda *args, **kwargs: object()
    seen = []

    def complete(_request, params, tool_config, **kwargs):
        seen.append((kwargs["reasoning_open"], params.skip_special_tokens))
        return llm._parse_generation_output(
            "Maybe " + _CALL, [1],
            7,
            "length",
            tool_config,
            reasoning_parser=kwargs["reasoning_parser"],
            reasoning_open=kwargs["reasoning_open"],
            skip_special_tokens=params.skip_special_tokens)

    llm._complete_prepared_request = complete
    messages = [{
        "role": "user",
        "content": "Weather?"
    }, {
        "role":
        "assistant",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": {
                    "location": "Paris"
                }
            },
        }],
    }, {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "sunny"
    }]
    tools = [{
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object"
            }
        },
    }]
    output = llm.chat(messages,
                      SamplingParams(enable_thinking=True),
                      tools=tools,
                      reasoning_parser="gemma4")
    assert seen == [(True, False)]
    assert output.reasoning == "Maybe " + _CALL
    assert output.tool_calls == []
