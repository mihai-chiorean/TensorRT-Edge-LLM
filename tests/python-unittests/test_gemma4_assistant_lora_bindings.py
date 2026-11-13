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
"""Spike for MIT-2120: which projections of the Gemma 4 MTP assistant (draft)
would get runtime LoRA inputs if the draft were built with --max-lora-rank.

The module tree is constructed with the geometry of
google/gemma-4-E4B-it-assistant (config.json: hidden_size 256,
backbone_hidden_size 2560, 4 layers, layer 3 full attention, ordered
embeddings) and ``Gemma4AssistantForCausalLM.forward`` is run over mocked
graph operations, recording every ``Linear`` the forward pass calls. Only a
called projection creates engine inputs (``Linear.forward`` ->
``F.dynamic_lora``), so ``_uses_lora`` on the called set is the draft's LoRA
binding set.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from experimental.builder.models.gemma4 import \
    modeling_gemma4_assistant as assistant_module  # noqa: E402
from experimental.builder.models.gemma4 import \
    modeling_gemma4_layers as layers_module  # noqa: E402
from experimental.builder.models.gemma4 import \
    modeling_gemma4_text as text_module  # noqa: E402
from experimental.builder.ops.linear import Linear  # noqa: E402
from experimental.builder.ops.module import (BuildContext,  # noqa: E402
                                             BuildOptions)

_LAYERS = 4
_FULL_LAYER = 3
_VOCAB = 262144
_DRAFT_TARGETS = ("q_proj", "o_proj", "mlp", "pre_projection",
                  "post_projection")


class _Config(SimpleNamespace):
    """The assistant geometry; graph-only fields the trace never inspects
    resolve to mocks."""

    def __getattr__(self, name):
        return mock.MagicMock()


def _assistant(options):
    cfg = _Config(
        backbone_hidden_size=2560,
        hidden_size=256,
        num_attention_heads=4,
        intermediate_size=2048,
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
        vocab_size=_VOCAB,
        tp_size=1,
        tp_rank=0,
    )
    weights = mock.MagicMock()
    weights.has.return_value = False
    weights.causal_lm_head_prefix.return_value = "lm_head"
    weights.linear_adapter.return_value = None
    weights.array.side_effect = lambda key: np.arange(_VOCAB, dtype=np.int32)
    ctx = BuildContext(net=mock.MagicMock(),
                       cfg=cfg,
                       weights=weights,
                       options=options,
                       bundle=SimpleNamespace(),
                       args=SimpleNamespace())
    return assistant_module.Gemma4AssistantForCausalLM(ctx)


def _graph_ops():
    ops = mock.MagicMock()
    ops.attention.return_value = (mock.MagicMock(), mock.MagicMock())
    ops.topk.return_value = (mock.MagicMock(), mock.MagicMock())
    return ops


def _called_linears(options):
    """Every Linear module the assistant's forward pass calls, in order."""
    called = []

    def record(self, hidden_states, *args, **kwargs):
        called.append(self)
        return mock.MagicMock()

    with mock.patch.object(Linear, "forward", record), \
            mock.patch.object(assistant_module, "F", _graph_ops()), \
            mock.patch.object(text_module, "F", _graph_ops()), \
            mock.patch.object(layers_module, "F", _graph_ops()), \
            mock.patch.object(assistant_module, "RaggedDecoderInputs",
                              mock.MagicMock()):
        model = _assistant(options)
        model.forward(inputs_embeds=mock.MagicMock(),
                      hidden_states_input=mock.MagicMock(),
                      past=[mock.MagicMock()] * _LAYERS,
                      rope_full=mock.MagicMock(),
                      rope_sliding=mock.MagicMock())
    return called


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


def test_forward_calls_no_key_or_value_projection():
    called = {linear.prefix for linear in _called_linears(BuildOptions())}

    assert called == _expected_branches() | {
        "lm_head", "masked_embedding.centroids"
    }


def test_draft_targets_select_every_called_projection_except_the_heads():
    adapted = {
        linear.prefix
        for linear in _called_linears(
            BuildOptions(max_lora_rank=16, lora_targets=_DRAFT_TARGETS))
        if linear._uses_lora()
    }

    assert adapted == _expected_branches()
    assert len(adapted) == 22  # 44 engine inputs


def test_without_targets_the_ordered_embedding_centroids_would_be_adapted():
    adapted = {
        linear.prefix
        for linear in _called_linears(BuildOptions(max_lora_rank=16))
        if linear._uses_lora()
    }

    assert adapted == _expected_branches() | {"masked_embedding.centroids"}
