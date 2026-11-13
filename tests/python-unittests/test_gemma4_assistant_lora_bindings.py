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
"""Dynamic-LoRA insertion contracts for token-major ONNX Gemm nodes and
the INT4 groupwise GEMM plugins."""
"""Spike for MIT-2120: which projections of the Gemma 4 MTP assistant (draft)
would get runtime LoRA inputs if the draft were built with --max-lora-rank.

The module tree is constructed with the geometry of
google/gemma-4-E4B-it-assistant (config.json: hidden_size 256,
backbone_hidden_size 2560, 4 layers, layer 3 full attention) and a fake
checkpoint; ``Linear._uses_lora`` decides per projection. Only projections
the assistant's forward pass calls create engine inputs, so the expected set
lists those (the shared-KV attention never calls k_proj / v_proj).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")

from experimental.builder.models.gemma4.modeling_gemma4_assistant import \
    Gemma4AssistantForCausalLM  # noqa: E402
from experimental.builder.ops.linear import Linear  # noqa: E402
from experimental.builder.ops.module import (BuildContext,  # noqa: E402
                                             BuildOptions)

_LAYERS = 4
_FULL_LAYER = 3


def _assistant(options):
    cfg = SimpleNamespace(
        backbone_hidden_size=2560,
        num_hidden_layers=_LAYERS,
        rms_norm_eps=1e-6,
        layer_head_dim=lambda i: 512 if i == _FULL_LAYER else 256,
        layer_num_kv_heads=lambda i: 2,
        attention_type=lambda i: "full_attention"
        if i == _FULL_LAYER else "sliding_attention",
        attention_k_eq_v=False,
        num_kv_shared_layers=_LAYERS,
        use_ordered_embeddings=True,
        num_centroids=2048,
        centroid_intermediate_top_k=32,
        vocab_size=262144,
        tp_size=1,
        tp_rank=0,
    )
    weights = SimpleNamespace(
        has=lambda key: False,
        causal_lm_head_prefix=lambda: "lm_head",
        linear_adapter=lambda prefix: None,
    )
    ctx = BuildContext(net=SimpleNamespace(),
                       cfg=cfg,
                       weights=weights,
                       options=options,
                       bundle=SimpleNamespace(),
                       args=SimpleNamespace())
    return Gemma4AssistantForCausalLM(ctx)


def _linears(module, seen=None):
    seen = set() if seen is None else seen
    if id(module) in seen:
        return []
    seen.add(id(module))
    found = [module] if isinstance(module, Linear) else []
    for value in vars(module).values():
        children = value if isinstance(value, (list, tuple)) else [value]
        for child in children:
            if hasattr(child, "ctx") and hasattr(child, "prefix"):
                found += _linears(child, seen)
    return found


def _called_in_forward(prefix):
    """Projections Gemma4AssistantForCausalLM.forward reaches."""
    return not prefix.endswith((".k_proj", ".v_proj"))


_DRAFT_TARGETS = ("q_proj", "o_proj", "mlp", "pre_projection",
                  "post_projection")


def _expected_branches():
    stems = {"pre_projection", "post_projection"}
    for layer in range(_LAYERS):
        stems |= {
            f"model.layers.{layer}.self_attn.q_proj",
            f"model.layers.{layer}.self_attn.o_proj",
            f"model.layers.{layer}.mlp.gate_proj",
            f"model.layers.{layer}.mlp.up_proj",
            f"model.layers.{layer}.mlp.down_proj",
        }
    return stems


def test_draft_targets_select_every_called_projection_and_nothing_else():
    model = _assistant(
        BuildOptions(max_lora_rank=16, lora_targets=_DRAFT_TARGETS))

    adapted = {
        linear.prefix
        for linear in _linears(model)
        if linear._uses_lora() and _called_in_forward(linear.prefix)
    }

    assert adapted == _expected_branches()
    assert len(adapted) == 22  # 44 engine inputs


def test_without_targets_the_ordered_embedding_centroids_would_be_adapted():
    model = _assistant(BuildOptions(max_lora_rank=16))

    adapted = {
        linear.prefix
        for linear in _linears(model)
        if linear._uses_lora() and _called_in_forward(linear.prefix)
    }

    assert adapted == _expected_branches() | {"masked_embedding.centroids"}
    assert "lm_head" not in adapted


def test_shared_kv_layers_construct_unused_key_value_projections():
    model = _assistant(BuildOptions(max_lora_rank=16))
    prefixes = {linear.prefix for linear in _linears(model)}
    assert "model.layers.0.self_attn.k_proj" in prefixes
    assert "model.layers.0.self_attn.v_proj" in prefixes
