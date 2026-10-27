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
"""Terminal special tokens that must not reach clients as content.

When special tokens are preserved in decoded text (tool parsing, reasoning
parsers), the runtime also emits the model's end-of-turn piece, such as
``<|im_end|>``, ``<turn|>`` or ``<eos>``. Generation stops at that piece, so
only a trailing run of terminal tokens is removed; an earlier occurrence is
literal text (for example inside a tool argument) and is kept.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional, Sequence, Tuple

IM_END_TOKEN = "<|im_end|>"


@lru_cache(maxsize=16)
def terminal_special_tokens(model_dir: str) -> Tuple[str, ...]:
    """``<|im_end|>`` plus the tokenizer's ``eos_token``/``eot_token``, longest
    first."""
    tokens = {IM_END_TOKEN}
    try:
        with open(os.path.join(model_dir, "tokenizer_config.json"),
                  encoding="utf-8") as stream:
            config = json.load(stream)
        for key in ("eos_token", "eot_token"):
            value = config.get(key)
            if isinstance(value, dict):
                value = value.get("content")
            if isinstance(value, str) and value:
                tokens.add(value)
    except (OSError, ValueError, AttributeError):
        pass
    return tuple(sorted(tokens, key=len, reverse=True))


def _trailing_tokens_len(text: str, tokens: Sequence[str]) -> int:
    """Length of the suffix of ``text`` made only of complete ``tokens``."""
    end = len(text)
    while True:
        for token in tokens:
            if token and text.endswith(token, 0, end):
                end -= len(token)
                break
        else:
            return len(text) - end


def ended_on_stop(finish_reason: Optional[str]) -> bool:
    """Only a stopped generation can end in a terminal token; a length limit
    or a cancellation leaves literal text at the end."""
    return finish_reason in (None, "stop")


def strip_terminal_tokens(text: str, tokens: Sequence[str]) -> str:
    """Remove the trailing run of ``tokens`` from ``text``."""
    return text[:len(text) - _trailing_tokens_len(text, tokens)]


def partial_token_len(text: str, tokens: Sequence[str]) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of one
    of ``tokens``."""
    best = 0
    for token in tokens:
        for size in range(min(len(text), len(token) - 1), best, -1):
            if text.endswith(token[:size]):
                best = size
                break
    return best


class TerminalTokenStripper:
    """Remove the trailing terminal tokens from a delta stream.

    ``feed`` returns the text that is safe to forward. A suffix of complete
    tokens, optionally followed by a prefix of one, is held back: it is
    released unchanged if more text follows, and dropped by ``flush`` when
    the generation stopped.
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        self._tokens = tuple(tokens)
        self._held = ""

    def feed(self, text: str) -> str:
        if not self._tokens:
            return text
        buffer = self._held + text
        hold = partial_token_len(buffer, self._tokens)
        hold += _trailing_tokens_len(buffer[:len(buffer) - hold], self._tokens)
        self._held = buffer[len(buffer) - hold:] if hold else ""
        return buffer[:len(buffer) - hold]

    def flush(self, stopped: bool = True) -> str:
        held, self._held = self._held, ""
        return strip_terminal_tokens(held, self._tokens) if stopped else held
