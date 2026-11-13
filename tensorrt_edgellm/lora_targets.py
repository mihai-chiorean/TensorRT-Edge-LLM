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
"""LoRA target selection shared by the ONNX insertion and the direct builder.

A target names one or more whole, contiguous dot-separated segments of a module
stem: ``self_attn`` selects ``model.layers.3.self_attn.q_proj``, ``q_proj``
selects every q_proj, ``layers.3.mlp`` selects layer 3's MLP projections, and
``proj`` selects nothing. Kept free of torch/onnx imports so the direct builder
and the server can use it.
"""

import re
from typing import Iterable, List, Optional, Sequence, Tuple

_TARGET_RE = re.compile(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*")


def parse_lora_targets(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Parse a comma-separated target list; ``None``/``""`` mean every eligible GEMM."""
    if value is None or value == "":
        return None
    targets = sorted({part.strip() for part in value.split(",") if part.strip()})
    if not targets:
        raise ValueError(f"--lora-targets {value!r} names no targets")
    for target in targets:
        if not _TARGET_RE.fullmatch(target):
            raise ValueError(
                f"invalid LoRA target {target!r}: expected dot-separated "
                "module names such as self_attn, mlp or layers.3.mlp")
    return tuple(targets)


def stem_matches_targets(stem: str,
                         targets: Optional[Sequence[str]]) -> bool:
    """Whether ``stem`` is selected; every stem is selected when ``targets`` is None."""
    if targets is None:
        return True
    padded = f".{stem}."
    return any(f".{target}." in padded for target in targets)


def unmatched_targets(stems: Iterable[str],
                      targets: Sequence[str]) -> List[str]:
    """Targets that select none of ``stems``."""
    stems = list(stems)
    return sorted(target for target in targets
                  if not any(stem_matches_targets(stem, (target, ))
                             for stem in stems))
