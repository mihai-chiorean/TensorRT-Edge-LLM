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
"""OpenAI-compatible tool request validation and output parsing.

References the OpenAI-compatible tool-calling API shape documented by vLLM:
https://docs.vllm.ai/en/stable/features/tool_calling/
"""

import ast
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass
class ToolConfig:
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "none"
    forced_name: Optional[str] = None

    @property
    def parse_output(self) -> bool:
        return bool(self.tools) and self.tool_choice != "none"

    @property
    def names(self) -> set[str]:
        return {
            tool["function"]["name"]
            for tool in self.tools if isinstance(tool.get("function"), dict)
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


class ToolProtocolError(ValueError):
    """A recognized tool turn that cannot safely produce an executable call."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__("The model generated an invalid tool call.")

    def to_openai(self) -> Dict[str, str]:
        return {
            "message": str(self),
            "type": "tool_protocol_error",
            "code": "invalid_tool_call",
            "reason": self.reason,
        }


@dataclass
class ParsedAssistantOutput:
    events: List[Dict[str, Any]]
    malformed: bool = False

    @property
    def content(self) -> str:
        return "".join(e["text"] for e in self.events
                       if e["type"] == "content")

    @property
    def reasoning(self) -> str:
        return "".join(e["text"] for e in self.events
                       if e["type"] == "reasoning")

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [
            e["tool_call"] for e in self.events if e["type"] == "tool_call"
        ]


def validate_tool_request(
    messages: Sequence[Dict[str, Any]],
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> ToolConfig:
    """Validate OpenAI-style tool fields and message links."""
    if tools is None:
        tool_list: List[Dict[str, Any]] = []
    elif isinstance(tools, list):
        tool_list = list(tools)
    else:
        raise ValueError("'tools' must be an array")

    names = _validate_tools(tool_list)
    choice, forced_name = _validate_tool_choice(tool_choice, names, tool_list)
    _validate_tool_messages(messages)
    return ToolConfig(tools=tool_list,
                      tool_choice=choice,
                      forced_name=forced_name)


def parse_assistant_output(text: str, tool_config: ToolConfig,
                           model_dir: str) -> ParsedAssistantOutput:
    """Parse model text into ordered content, reasoning, and tool-call events."""
    if not tool_config.parse_output:
        return ParsedAssistantOutput(_split_reasoning_events(text))

    parser = _select_parser(model_dir)
    events, malformed = parser.parse(text, tool_config)
    expanded: List[Dict[str, Any]] = []
    for event in events:
        if event["type"] == "content":
            expanded.extend(_split_reasoning_events(event["text"]))
        else:
            expanded.append(event)
    return ParsedAssistantOutput(expanded, malformed=malformed)


def _validate_tools(tools: Sequence[Dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{idx}] must be an object")
        if tool.get("type") != "function":
            raise ValueError(f"tools[{idx}].type must be 'function'")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"tools[{idx}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tools[{idx}].function.name must be a string")
        if name in names:
            raise ValueError(f"Duplicate tool name: {name}")
        names.add(name)
        desc = function.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ValueError(
                f"tools[{idx}].function.description must be a string")
        params = function.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise ValueError(
                f"tools[{idx}].function.parameters must be an object")
        strict = function.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise ValueError(f"tools[{idx}].function.strict must be a bool")
    return names


def _validate_tool_choice(
        tool_choice: Optional[Union[str, Dict[str, Any]]], names: set[str],
        tools: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    if tool_choice is None:
        return ("auto" if tools else "none"), None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            if tool_choice in {"auto", "required"} and not tools:
                raise ValueError("'tool_choice' requires at least one tool")
            return tool_choice, None
        if tool_choice not in names:
            raise ValueError(f"Unknown forced tool name: {tool_choice}")
        return "function", tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError("tool_choice.type must be 'function'")
        function = tool_choice.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool_choice.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice.function.name must be a string")
        if name not in names:
            raise ValueError(f"Unknown forced tool name: {name}")
        return "function", name
    raise ValueError(
        "'tool_choice' must be 'auto', 'none', 'required', or a function choice"
    )


def _validate_tool_messages(messages: Sequence[Dict[str, Any]]) -> None:
    seen: set[str] = set()
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{idx}] must be an object")
        role = msg.get("role")
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError(
                    f"messages[{idx}].tool_calls must be an array")
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    raise ValueError(
                        f"messages[{idx}].tool_calls entries must be objects")
                tc_id = tc.get("id")
                if not isinstance(tc_id, str) or not tc_id:
                    raise ValueError(
                        f"messages[{idx}].tool_calls[].id must be a string")
                function = tc.get("function")
                if not isinstance(function, dict):
                    raise ValueError(
                        f"messages[{idx}].tool_calls[].function must be an object"
                    )
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"messages[{idx}].tool_calls[].function.name must be a string"
                    )
                seen.add(tc_id)
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if not isinstance(tc_id, str) or not tc_id:
                raise ValueError(
                    f"messages[{idx}].tool_call_id must be a string")
            if tc_id not in seen:
                raise ValueError(f"Dangling tool_call_id: {tc_id}")
            content = msg.get("content", "")
            if content is not None and not isinstance(content,
                                                      (str, dict, list)):
                raise ValueError(
                    f"messages[{idx}].content must be text or JSON")


def _split_reasoning_events(text: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    pos = 0
    for match in re.finditer(r"<think>(.*?)</think>", text, flags=re.S):
        if match.start() > pos:
            events.append({"type": "content", "text": text[pos:match.start()]})
        events.append({"type": "reasoning", "text": match.group(1).strip()})
        pos = match.end()
    if pos < len(text):
        tail = text[pos:]
        if tail:
            events.append({"type": "content", "text": tail})
    return events or [{"type": "content", "text": ""}]


class _GenericToolParser:

    def stream_parser(
        self, tool_config: "ToolConfig", strip_tokens: Sequence[str] = ()
    ) -> "ToolStreamParser":
        # The generic grammar has no fixed opener for its bare forms (raw
        # JSON, code fences, pythonic lines), so it cannot release content
        # before the whole output is known.
        return _BufferedToolStreamParser(self, tool_config, strip_tokens)

    _BLOCK_RE = re.compile(
        r"(<tool_call>.*?</tool_call>|<tool_calls>.*?</tool_calls>|"
        r"<toolcall>.*?</toolcall>|<toolcalls>.*?</toolcalls>|"
        r"<function_call>.*?</function_call>|"
        r"<function_calls>.*?</function_calls>|"
        r"<function=[^>]+>.*?</function>|"
        r"\[TOOL_CALLS?\].*?(?:\[/TOOL_CALLS?\]|$))",
        re.S,
    )

    def parse(self, text: str,
              tool_config: ToolConfig) -> Tuple[List[Dict[str, Any]], bool]:
        events: List[Dict[str, Any]] = []
        malformed = False
        pos = 0
        matched = False
        for match in self._BLOCK_RE.finditer(text):
            matched = True
            if match.start() > pos:
                events.append({
                    "type": "content",
                    "text": text[pos:match.start()]
                })
            calls = _parse_tool_block(match.group(0), tool_config)
            if calls:
                events.extend({
                    "type": "tool_call",
                    "tool_call": c
                } for c in calls)
            else:
                malformed = True
                events.append({"type": "content", "text": match.group(0)})
            pos = match.end()
        if pos < len(text):
            events.append({"type": "content", "text": text[pos:]})
        if matched:
            return events, malformed

        calls = _parse_tool_block(text, tool_config)
        if calls:
            return ([{
                "type": "tool_call",
                "tool_call": c
            } for c in calls], False)
        return [{"type": "content", "text": text}], False


class _Gemma4ToolParser:

    BLOCK_OPENER = "<|tool_call>"
    BLOCK_CLOSER = "<tool_call|>"
    BARE_PREFIX = "call:"

    def stream_parser(
        self, tool_config: "ToolConfig", strip_tokens: Sequence[str] = ()
    ) -> "ToolStreamParser":
        return _Gemma4ToolStreamParser(self, tool_config, strip_tokens)

    _CALL_RE = re.compile(
        r"^call:(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        r"(?P<arguments>\{.*\})$",
        re.S,
    )
    _CALL_PREFIX_RE = re.compile(
        r"^\s*call:[A-Za-z_][A-Za-z0-9_]*(?P<arguments>\{)",
        re.S,
    )

    def parse(self, text: str,
              tool_config: ToolConfig) -> Tuple[List[Dict[str, Any]], bool]:
        stream = self.stream_parser(tool_config)
        return stream.feed(text) + stream.finish(), False


def make_stream_parser(
    tool_config: ToolConfig, model_dir: str,
    strip_tokens: Sequence[str] = ()) -> "ToolStreamParser":
    """Incremental counterpart of :func:`parse_assistant_output`.

    ``strip_tokens`` are removed from every content span and from tool-call
    bodies before parsing, matching the whole-output path. ``<think>`` tags are
    left in content for the caller to split.
    """
    return _select_parser(model_dir).stream_parser(tool_config, strip_tokens)


def partial_marker_len(text: str, markers: Sequence[str]) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of one
    of ``markers``: the span a streaming splitter must hold back because the
    next delta may complete a marker."""
    best = 0
    for marker in markers:
        for size in range(min(len(text), len(marker) - 1), best, -1):
            if text.endswith(marker[:size]):
                best = size
                break
    return best


def _strip_tokens(text: str, tokens: Sequence[str]) -> str:
    for token in tokens:
        text = text.replace(token, "")
    return text


def _content_event(text: str) -> Dict[str, Any]:
    return {"type": "content", "text": text}


class ToolStreamParser:
    """Splits a token stream into ordered content and tool-call events.

    ``feed`` returns the events that became final with the new text;
    ``finish`` returns whatever is still held back at end of generation.
    Events have the shape produced by the parser ``parse`` methods.
    """

    def feed(self, text: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def finish(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class _BufferedToolStreamParser(ToolStreamParser):

    def __init__(self, parser, tool_config: ToolConfig,
                 strip_tokens: Sequence[str]):
        self._parser = parser
        self._tool_config = tool_config
        self._strip_tokens = tuple(strip_tokens)
        self._parts: List[str] = []

    def feed(self, text: str) -> List[Dict[str, Any]]:
        self._parts.append(text)
        return []

    def finish(self) -> List[Dict[str, Any]]:
        text = _strip_tokens("".join(self._parts), self._strip_tokens)
        self._parts = []
        events, _ = self._parser.parse(text, self._tool_config)
        return events


class _Gemma4ToolStreamParser(ToolStreamParser):
    """Releases content as soon as it can no longer become a tool call.

    Block form (``<|tool_call>...<tool_call|>``) may appear anywhere: content
    streams up to the opener, the block is buffered to its closer. The
    bare form (``call:NAME{...}``) is only recognised at the very start, so
    nothing is released until the first non-blank text rules it out. Once it
    is seen the output is buffered until generation finishes; only the first
    complete call is returned, and following prose is dropped. Recognized protocol
    never returns to content: malformed or incomplete calls raise a typed
    error, and prose following valid calls is suppressed. Prose emitted before
    a block opener cannot be retracted without buffering ordinary conversation.
    """

    def __init__(self, parser, tool_config: ToolConfig,
                 strip_tokens: Sequence[str]):
        self._parser = parser
        self._tool_config = tool_config
        self._strip_tokens = tuple(strip_tokens)
        self._opener = parser.BLOCK_OPENER
        self._closer = parser.BLOCK_CLOSER
        self._bare_prefix = parser.BARE_PREFIX
        self._scan_markers = (self._opener, ) + self._strip_tokens
        self._buf = ""
        self._at_start = True
        self._in_block = False
        self._bare = False
        self._done = False
        self._in_protocol = False
        self._error: Optional[ToolProtocolError] = None

    def feed(self, text: str) -> List[Dict[str, Any]]:
        if self._error is not None:
            raise self._error
        if self._done:
            return []
        self._buf += text
        events: List[Dict[str, Any]] = []
        if self._bare:
            return []
        while self._buf:
            if self._in_block:
                idx = self._buf.find(self._closer, len(self._opener))
                if idx == -1:
                    break
                end = idx + len(self._closer)
                try:
                    events.append(self._block_event(self._buf[:end]))
                except ToolProtocolError as exc:
                    self._fail(exc)
                self._buf = self._buf[end:]
                self._in_block = False
                continue
            if self._at_start:
                head = self._buf.lstrip()
                stripped = next((token for token in self._strip_tokens
                                 if head.startswith(token)), None)
                if stripped is not None:
                    self._buf = (self._buf[:len(self._buf) - len(head)] +
                                 head[len(stripped):])
                    continue
                if any(token.startswith(head) for token in self._strip_tokens):
                    break
                if head.startswith(self._bare_prefix):
                    self._bare = True
                    self._in_protocol = True
                    break
                if self._bare_prefix.startswith(head):
                    break
                self._at_start = False
            idx, marker = self._earliest_marker()
            if marker is None:
                hold = partial_marker_len(self._buf, self._scan_markers)
                release = self._buf[:len(self._buf) - hold]
                if release and not self._in_protocol:
                    events.append(_content_event(release))
                self._buf = self._buf[len(release):]
                break
            if idx > 0 and not self._in_protocol:
                events.append(_content_event(self._buf[:idx]))
            if marker == self._opener:
                self._buf = self._buf[idx:]
                self._in_block = True
                self._in_protocol = True
            else:
                self._buf = self._buf[idx + len(marker):]
        return events

    def finish(self) -> List[Dict[str, Any]]:
        if self._error is not None:
            raise self._error
        if self._bare and not self._done:
            events = self._bare_progress()
            if events:
                return events
        buf, self._buf = self._buf, ""
        if self._done:
            return []
        text = _strip_tokens(buf, self._strip_tokens)
        self._done = True
        if self._bare or self._in_block:
            self._fail(ToolProtocolError("incomplete"))
        if self._in_protocol:
            return []
        return [_content_event(text)] if text else []

    def _bare_progress(self) -> List[Dict[str, Any]]:
        try:
            call = _parse_gemma4_call_prefix(
                _strip_tokens(self._buf, self._strip_tokens),
                self._tool_config)
        except ToolProtocolError as exc:
            self._fail(exc)
        if call is None:
            return []
        self._done = True
        self._buf = ""
        return [{"type": "tool_call", "tool_call": call}]

    def _fail(self, error: ToolProtocolError) -> None:
        self._error = error
        self._buf = ""
        raise error

    def _earliest_marker(self) -> Tuple[int, Optional[str]]:
        best_idx, best = -1, None
        for marker in self._scan_markers:
            idx = self._buf.find(marker)
            if idx != -1 and (best is None or idx < best_idx):
                best_idx, best = idx, marker
        return best_idx, best

    def _block_event(self, block: str) -> Dict[str, Any]:
        block = _strip_tokens(block, self._strip_tokens)
        body = block[len(self._opener):-len(self._closer)]
        call = _parse_gemma4_call(body, self._tool_config)
        return {"type": "tool_call", "tool_call": call}


class _ToolParserRegistry:

    def __init__(self):
        parser = _GenericToolParser()
        self._parsers = {
            "generic": parser,
            "hermes": parser,
            "qwen3_xml": parser,
            "nemotron": parser,
            "openai": parser,
            "gemma4": _Gemma4ToolParser(),
        }

    def get(self, model_dir: str):
        return self._parsers.get(_parser_name_for_model(model_dir),
                                 self._parsers["generic"])


_PARSERS = _ToolParserRegistry()


def _select_parser(model_dir: str):
    return _PARSERS.get(model_dir)


def _parser_name_for_model(model_dir: str) -> str:
    model_type = ""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
            model_type = str(
                config.get("model_type") or config.get("model") or "").lower()
    except (OSError, ValueError):
        pass
    name = f"{model_type} {os.path.basename(model_dir).lower()}"
    if "qwen3" in name and "coder" in name:
        return "qwen3_xml"
    if "qwen" in name:
        return "hermes"
    if "nemotron" in name:
        return "nemotron"
    if "openai" in name or "gpt-oss" in name:
        return "openai"
    if "gemma4" in name:
        return "gemma4"
    return "generic"


def _parse_gemma4_call(text: str, tool_config: ToolConfig) -> ToolCall:
    match = _Gemma4ToolParser._CALL_RE.fullmatch(text.strip())
    if match is None:
        raise ToolProtocolError("malformed")
    name = match.group("name")
    if not _tool_name_allowed(name, tool_config):
        raise ToolProtocolError("unauthorized")
    try:
        arguments = _parse_gemma4_object(match.group("arguments"),
                                         _param_types_for(name, tool_config))
    except (ValueError, RecursionError):
        raise ToolProtocolError("malformed") from None
    return ToolCall(id=_new_call_id(),
                    name=name,
                    arguments=_arguments_to_json(arguments))


def _parse_gemma4_call_prefix(
    text: str,
    tool_config: ToolConfig,
) -> Optional[ToolCall]:
    match = _Gemma4ToolParser._CALL_PREFIX_RE.match(text)
    if match is None:
        return None
    start = match.start("arguments")
    end = _gemma4_object_end(text, start)
    if end is None:
        return None
    return _parse_gemma4_call(text[:end], tool_config)


def _gemma4_object_end(text: str, start: int) -> Optional[int]:
    """Return the end of one balanced Gemma object outside string tokens."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    try:
        for index, char in _gemma4_unquoted_chars(text[start:]):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return start + index + 1
    except ValueError:
        return None
    return None


def _gemma4_unquoted_chars(text: str) -> Iterable[Tuple[int, str]]:
    """Yield syntax outside Gemma-token or JSON strings without decoding it."""
    delimiter = '<|"|>'
    quote: Optional[str] = None
    index = 0
    while index < len(text):
        if quote == '"':
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == '"':
                quote = None
        elif text.startswith(delimiter, index):
            quote = None if quote == delimiter else delimiter
            index += len(delimiter)
            continue
        elif quote is None:
            if text[index] == '"':
                quote = '"'
            else:
                yield index, text[index]
        index += 1
    if quote is not None:
        raise ValueError("Unterminated Gemma tool string")


def _parse_gemma4_object(text: str,
                         parameter_types: Optional[Dict[str,
                                                        str]] = None) -> dict:
    body = text.strip()
    if not (body.startswith("{") and body.endswith("}")):
        raise ValueError("Gemma tool arguments must be an object")
    body = body[1:-1].strip()
    if not body:
        return {}

    result = {}
    for item in _split_gemma4_fields(body):
        key, value = _split_gemma4_key_value(item)
        if key in result:
            raise ValueError("Duplicate Gemma tool argument")
        expected_type = (parameter_types or {}).get(key)
        result[key] = _parse_gemma4_value(value, expected_type)
    return result


def _split_gemma4_fields(text: str) -> List[str]:
    fields = []
    start = 0
    stack = []
    for index, char in _gemma4_unquoted_chars(text):
        if char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != {"]": "[", "}": "{"}[char]:
                raise ValueError("Unbalanced Gemma tool arguments")
        elif char == "," and not stack:
            fields.append(text[start:index].strip())
            start = index + 1
    if stack:
        raise ValueError("Unbalanced Gemma tool arguments")
    fields.append(text[start:].strip())
    if any(not field for field in fields):
        raise ValueError("Empty Gemma tool argument")
    return fields


def _split_gemma4_key_value(field: str) -> Tuple[str, str]:
    for index, char in _gemma4_unquoted_chars(field):
        if char == ":":
            key = field[:index].strip()
            value = field[index + 1:].strip()
            if key.startswith('"'):
                key = json.loads(key)
            elif not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                break
            if not isinstance(key, str) or not value:
                break
            return key, value
    raise ValueError("Invalid Gemma tool argument")


def _parse_gemma4_value(text: str, expected_type: Optional[str] = None) -> Any:
    value = text.strip()
    delimiter = '<|"|>'
    if value.startswith(delimiter) and value.endswith(delimiter):
        inner = value[len(delimiter):-len(delimiter)]
        if len(value) < 2 * len(delimiter) or delimiter in inner:
            raise ValueError("Invalid Gemma tool string")
        return inner
    if value.startswith("{"):
        return _parse_gemma4_object(value)
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return ([] if not body else [
            _parse_gemma4_value(item) for item in _split_gemma4_fields(body)
        ])
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if (expected_type == "string"
                and not any(char in value for char in '\"\'{}[]:')
                and delimiter not in value):
            return value
        raise ValueError("Invalid Gemma tool argument value")


def _parse_tool_block(block: str, tool_config: ToolConfig) -> List[ToolCall]:
    body = _strip_tool_tags(block)
    calls = _parse_qwen_xml_calls(body, tool_config)
    if calls:
        return calls
    payload = _loads_payload(body)
    if payload is None:
        calls = _parse_pythonic_calls(body, tool_config)
        return calls
    return list(_calls_from_payload(payload, tool_config))


def _strip_tool_tags(text: str) -> str:
    body = text.strip()
    for pattern in [
            r"^<tool_call>\s*(.*?)\s*</tool_call>$",
            r"^<tool_calls>\s*(.*?)\s*</tool_calls>$",
            r"^<toolcall>\s*(.*?)\s*</toolcall>$",
            r"^<toolcalls>\s*(.*?)\s*</toolcalls>$",
            r"^<function_call>\s*(.*?)\s*</function_call>$",
            r"^<function_calls>\s*(.*?)\s*</function_calls>$",
            r"^\[TOOL_CALLS?\]\s*(.*?)\s*(?:\[/TOOL_CALLS?\])?$",
    ]:
        match = re.match(pattern, body, flags=re.S)
        if match:
            return match.group(1).strip()
    return _strip_code_fence(body)


def _strip_code_fence(text: str) -> str:
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text.strip(), flags=re.S)
    return match.group(1).strip() if match else text.strip()


def _loads_payload(text: str) -> Optional[Any]:
    body = _strip_code_fence(text)
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(body)
        except (ValueError, SyntaxError, TypeError):
            pass
    return None


def _calls_from_payload(payload: Any,
                        tool_config: ToolConfig) -> Iterable[ToolCall]:
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"),
                                                list):
        payload = payload["tool_calls"]
    if isinstance(payload, dict):
        call = _call_from_dict(payload, tool_config)
        if call:
            yield call
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                call = _call_from_dict(item, tool_config)
                if call:
                    yield call


def _call_from_dict(data: Dict[str, Any],
                    tool_config: ToolConfig) -> Optional[ToolCall]:
    function = data.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = data.get("name")
        arguments = data.get("arguments", data.get("parameters", {}))
    if not isinstance(name, str) or not _tool_name_allowed(name, tool_config):
        return None
    return ToolCall(
        id=data.get("id")
        if isinstance(data.get("id"), str) else _new_call_id(),
        name=name,
        arguments=_arguments_to_json(arguments),
    )


def _param_types_for(name: str, tool_config: ToolConfig) -> Dict[str, str]:
    """Map each parameter name -> its JSON-schema ``type`` for the named function."""
    for tool in tool_config.tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties")
            if isinstance(props, dict):
                return {
                    k: v.get("type")
                    for k, v in props.items()
                    if isinstance(v, dict) and isinstance(v.get("type"), str)
                }
    return {}


def _coerce_param(value: str, ptype: Optional[str]) -> Any:
    """Coerce an XML-extracted string to its declared JSON-schema type.

    Qwen's ``<parameter=...>...</parameter>`` values are always text; without
    this the arguments stay stringly-typed (``base="10"``) and fail strict
    consumers such as BFCL AST matching, which expects ``base=10``.
    """
    v = value.strip()
    try:
        if ptype == "integer":
            return int(v)
        if ptype == "number":
            f = float(v)
            return int(f) if f.is_integer() else f
        if ptype == "boolean":
            return v.lower() in ("true", "1", "yes")
        if ptype in ("array", "object"):
            return json.loads(v)
    except (ValueError, json.JSONDecodeError):
        return value
    return value


def _parse_qwen_xml_calls(text: str,
                          tool_config: ToolConfig) -> List[ToolCall]:
    calls = []
    for match in re.finditer(r"<function=([^>]+)>(.*?)</function>",
                             text,
                             flags=re.S):
        name = match.group(1).strip()
        if not _tool_name_allowed(name, tool_config):
            continue
        ptypes = _param_types_for(name, tool_config)
        args: Dict[str, Any] = {}
        for param in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>",
                                 match.group(2),
                                 flags=re.S):
            pname = param.group(1).strip()
            args[pname] = _coerce_param(
                param.group(2).strip(), ptypes.get(pname))
        calls.append(
            ToolCall(id=_new_call_id(),
                     name=name,
                     arguments=_arguments_to_json(args)))
    return calls


def _parse_pythonic_calls(text: str,
                          tool_config: ToolConfig) -> List[ToolCall]:
    calls = []
    for line in text.strip().splitlines():
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*,?\s*$", line)
        if not match:
            continue
        name = match.group(1)
        if not _tool_name_allowed(name, tool_config):
            continue
        calls.append(
            ToolCall(id=_new_call_id(),
                     name=name,
                     arguments=_arguments_to_json(
                         _parse_python_args(match.group(2)))))
    return calls


def _parse_python_args(args_src: str) -> Dict[str, Any]:
    try:
        expr = ast.parse(f"f({args_src})", mode="eval").body
    except SyntaxError:
        return {"__raw__": args_src}
    if not isinstance(expr, ast.Call):
        return {"__raw__": args_src}
    args: Dict[str, Any] = {}
    for kw in expr.keywords:
        if kw.arg is not None:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                args[kw.arg] = ast.unparse(kw.value)
    return args


def _tool_name_allowed(name: str, tool_config: ToolConfig) -> bool:
    if tool_config.forced_name and name != tool_config.forced_name:
        return False
    names = tool_config.names
    return not names or name in names


def _arguments_to_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            json.loads(arguments)
            return arguments
        except ValueError:
            return json.dumps({"__raw__": arguments}, ensure_ascii=False)
    return json.dumps(arguments or {}, ensure_ascii=False)


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"
