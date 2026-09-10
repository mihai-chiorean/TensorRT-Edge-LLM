"""Terminal special tokens that must not reach clients as content.

When special tokens are preserved in decoded text (tool parsing, reasoning
parsers), the runtime also emits the model's end-of-turn piece, such as
``<|im_end|>``, ``<turn|>`` or ``<eos>``. These helpers remove it from
non-streamed text and from a delta stream without leaking a token that is
split across deltas.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Sequence, Tuple

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


def strip_terminal_tokens(text: str, tokens: Sequence[str]) -> str:
    """Remove every occurrence of ``tokens`` and trailing whitespace left
    behind by a terminal token at the end of the text."""
    for token in tokens:
        text = text.replace(token, "")
    return text


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
    """Remove terminal tokens from a delta stream.

    ``feed`` returns the text that is safe to forward; a suffix that could
    still grow into one of the tokens is held back until the next delta or
    ``flush`` settles it.
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        self._tokens = tuple(tokens)
        self._held = ""

    def feed(self, text: str) -> str:
        if not self._tokens:
            return text
        buffer = strip_terminal_tokens(self._held + text, self._tokens)
        hold = partial_token_len(buffer, self._tokens)
        self._held = buffer[len(buffer) - hold:] if hold else ""
        return buffer[:len(buffer) - hold]

    def flush(self) -> str:
        held, self._held = self._held, ""
        return strip_terminal_tokens(held, self._tokens)
