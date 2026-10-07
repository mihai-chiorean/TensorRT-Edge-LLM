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

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from tensorrt_edgellm.lora.lora import (_match_fp8_gemm, _match_fp16_gemm,
                                        _match_int4_gemm, insert_lora_and_save)


def _gemm_graph(weight_name="_model.layers.0.mlp.up_proj.weight", **attrs):
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    weight = gs.Constant(weight_name,
                         values=np.zeros((32, 16), dtype=np.float16))
    output = gs.Variable("output", dtype=np.float16, shape=["tokens", 32])
    gemm_attrs = {"alpha": 1.0, "transA": 0, "transB": 1}
    gemm_attrs.update(attrs)
    node = gs.Node(op="Gemm",
                   name="node_Gemm_0",
                   attrs=gemm_attrs,
                   inputs=[hidden, weight],
                   outputs=[output])
    return gs.Graph(nodes=[node], inputs=[hidden], outputs=[output])


def _fp8_gemm_graph():
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    activation_scale = gs.Constant("activation_scale",
                                   values=np.array(1.0, dtype=np.float16))
    hidden_fp8 = gs.Variable("hidden_fp8", shape=["tokens", 16])
    hidden_dq = gs.Variable("hidden_dq",
                            dtype=np.float16,
                            shape=["tokens", 16])
    weight = gs.Constant("_model.layers.0.mlp.up_proj.weight",
                         values=np.zeros((32, 16), dtype=np.uint8))
    weight_scale = gs.Constant("weight_scale",
                               values=np.array(1.0, dtype=np.float16))
    weight_dq = gs.Variable("weight_dq", dtype=np.float16, shape=[32, 16])
    output = gs.Variable("output", dtype=np.float16, shape=["tokens", 32])
    nodes = [
        gs.Node(op="QuantizeLinear",
                attrs={"output_dtype": 17},
                inputs=[hidden, activation_scale],
                outputs=[hidden_fp8]),
        gs.Node(op="DequantizeLinear",
                inputs=[hidden_fp8, activation_scale],
                outputs=[hidden_dq]),
        gs.Node(op="DequantizeLinear",
                inputs=[weight, weight_scale],
                outputs=[weight_dq]),
        gs.Node(op="Gemm",
                name="node_Gemm_0",
                attrs={
                    "alpha": 1.0,
                    "transA": 0,
                    "transB": 1
                },
                inputs=[hidden_dq, weight_dq],
                outputs=[output]),
    ]
    return gs.Graph(nodes=nodes, inputs=[hidden], outputs=[output])


def test_fp16_gemm_match_preserves_transposed_weight_semantics():
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    weight = gs.Constant("_model.layers.0.mlp.up_proj.weight",
                         values=np.zeros((32, 16), dtype=np.float16))
    bias = gs.Constant("bias", values=np.zeros((32, ), dtype=np.float16))
    output = gs.Variable("output", dtype=np.float16, shape=["tokens", 32])
    node = gs.Node(op="Gemm",
                   name="node_Gemm_0",
                   attrs={
                       "alpha": 1.0,
                       "beta": 1.0,
                       "transA": 0,
                       "transB": 1
                   },
                   inputs=[hidden, weight, bias],
                   outputs=[output])
    graph = gs.Graph(nodes=[node], inputs=[hidden], outputs=[output])

    matches = _match_fp16_gemm(graph)

    assert len(matches) == 1
    assert matches[0].input is hidden
    assert matches[0].output is output
    assert matches[0].weight_shape == (16, 32)
    assert matches[0].name == "/layers/0/mlp/up_proj/MatMul"
    assert node.attrs == {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1}


def test_fp8_gemm_match_supports_token_major_gemm_export():
    matches = _match_fp8_gemm(_fp8_gemm_graph())

    assert len(matches) == 1
    assert matches[0].input.name == "hidden"
    assert matches[0].output.name == "output"
    assert matches[0].weight_shape == (16, 32)
    assert matches[0].name == "/layers/0/mlp/up_proj/MatMul"


def test_insert_lora_rejects_graph_without_eligible_linear(tmp_path):
    value = onnx.helper.make_tensor_value_info("value",
                                               onnx.TensorProto.FLOAT16,
                                               ["tokens", 16])
    output = onnx.helper.make_tensor_value_info("output",
                                                onnx.TensorProto.FLOAT16,
                                                ["tokens", 16])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["value"], ["output"])],
        "no-linear", [value], [output])
    onnx.save(onnx.helper.make_model(graph), tmp_path / "model.onnx")

    with pytest.raises(ValueError, match="no eligible linear layers"):
        insert_lora_and_save(str(tmp_path))


def test_fp16_gemm_match_skips_scaled_base_output():
    assert _match_fp16_gemm(_gemm_graph(alpha=0.5)) == []


def test_fp16_gemm_match_skips_noncanonical_weight_initializer():
    assert _match_fp16_gemm(_gemm_graph(weight_name="weight")) == []


def _matmul_graph(weight_name, node_name="node_linear"):
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    weight = gs.Constant(weight_name,
                         values=np.zeros((16, 32), dtype=np.float16))
    output = gs.Variable("logits", dtype=np.float16, shape=["tokens", 32])
    node = gs.Node(op="MatMul",
                   name=node_name,
                   inputs=[hidden, weight],
                   outputs=[output])
    return gs.Graph(nodes=[node], inputs=[hidden], outputs=[output])


def test_fp16_matmul_match_uses_weight_initializer_stem():
    matches = _match_fp16_gemm(
        _matmul_graph("_model.layers.0.mlp.up_proj.weight", "node_MatMul_3"))

    assert len(matches) == 1
    assert matches[0].name == "/layers/0/mlp/up_proj/MatMul"
    assert matches[0].weight_shape == (16, 32)


def test_fp16_matmul_with_anonymous_weight_gets_no_lora_inputs(tmp_path):
    """The dynamo exporter emits a tied lm_head as ``MatMul(x, val_N)`` named
    ``node_linear``; a node-name stem would bypass the lm_head filter."""
    graph = _matmul_graph("val_2631")

    assert _match_fp16_gemm(graph) == []

    onnx.save(gs.export_onnx(graph), tmp_path / "model.onnx")
    with pytest.raises(ValueError, match="no eligible linear layers"):
        insert_lora_and_save(str(tmp_path))
    assert not (tmp_path / "lora_model.onnx").exists()


def test_fp16_matmul_named_lm_head_is_excluded(tmp_path):
    graph = _matmul_graph("_model.lm_head.weight")
    assert [match.name
            for match in _match_fp16_gemm(graph)] == ["/lm_head/MatMul"]
    onnx.save(gs.export_onnx(graph), tmp_path / "model.onnx")

    with pytest.raises(ValueError, match="no eligible linear layers"):
        insert_lora_and_save(str(tmp_path))


def test_canonical_lm_head_is_excluded_actionably(tmp_path):
    graph = _gemm_graph(weight_name="_model.lm_head.weight")
    assert [match.name
            for match in _match_fp16_gemm(graph)] == ["/lm_head/MatMul"]
    onnx.save(gs.export_onnx(graph), tmp_path / "model.onnx")

    with pytest.raises(ValueError, match="no eligible linear layers"):
        insert_lora_and_save(str(tmp_path))


@pytest.mark.parametrize("graph", [
    _gemm_graph(alpha=0.5),
    _gemm_graph(weight_name="weight"),
])
def test_insert_lora_rejects_when_every_gemm_is_skipped(tmp_path, graph):
    onnx.save(gs.export_onnx(graph), tmp_path / "model.onnx")

    with pytest.raises(ValueError, match="no eligible linear layers"):
        insert_lora_and_save(str(tmp_path))


_INT4_STEM = "model.layers.0.self_attn.q_proj"
_INT4_GEMM_NAME = "/model/layers/0/self_attn/q_proj/MatMul"


def _int4_plugin(activation, with_output_scale=False):
    """``Int4GroupwiseGemmPluginV2`` consuming ``activation`` with the Gemma 4
    q_proj geometry scaled down to k=16, n=32. With ``with_output_scale`` the
    GEMM output feeds an architectural ``hidden_size**-0.5`` Mul the way the
    PLE model projection does."""
    weight = gs.Constant(f"_model.{_INT4_STEM}.weight",
                         values=np.zeros((16, 4), dtype=np.int8))
    weight_scale = gs.Constant(f"_model.{_INT4_STEM}.weight_scale",
                               values=np.ones((1, 32), dtype=np.float16))
    gemm_out = gs.Variable("gemm_out", dtype=np.float16, shape=["tokens", 32])
    plugin = gs.Node(op="Int4GroupwiseGemmPluginV2",
                     name="node_int4_gemm",
                     attrs={
                         "gemm_k": 16,
                         "gemm_n": 32,
                         "group_size": 128
                     },
                     inputs=[activation, weight, weight_scale],
                     outputs=[gemm_out])
    if not with_output_scale:
        # Real exports never end on a GEMM; mirror down_proj's trailing Cast.
        casted = gs.Variable("casted", dtype=np.float16, shape=["tokens", 32])
        cast = gs.Node(op="Cast",
                       name="node_output_cast",
                       attrs={"to": 10},
                       inputs=[gemm_out],
                       outputs=[casted])
        return [plugin, cast], casted
    out_scale = gs.Constant("_to_copy",
                            values=np.array(16**-0.5, dtype=np.float16))
    scaled = gs.Variable("scaled", dtype=np.float16, shape=["tokens", 32])
    mul = gs.Node(op="Mul",
                  name="node_output_scale",
                  inputs=[gemm_out, out_scale],
                  outputs=[scaled])
    return [plugin, mul], scaled


def _int4_awq_graph(scale_first=False,
                    scale_name=f"_model.{_INT4_STEM}.pre_quant_scale",
                    with_output_scale=False):
    """Checkpoint-based ModelOpt AWQ export: ``Mul(x, pre_quant_scale)`` feeds
    the plugin, with non-uniform per-channel scales."""
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    scale = gs.Constant(scale_name,
                        values=np.linspace(0.4, 2.5, 16, dtype=np.float16))
    smoothed = gs.Variable("smoothed", dtype=np.float16, shape=["tokens", 16])
    mul = gs.Node(op="Mul",
                  name="node_mul",
                  inputs=[scale, hidden] if scale_first else [hidden, scale],
                  outputs=[smoothed])
    nodes, output = _int4_plugin(smoothed, with_output_scale)
    return gs.Graph(nodes=[mul] + nodes, inputs=[hidden], outputs=[output])


def _int4_legacy_awq_graph():
    """Legacy ModelOpt-traced export: ``Mul`` named after the input quantizer
    followed by a ``Cast``."""
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    scale = gs.Constant("smooth", values=np.full((16, ), 2.0, np.float32))
    smoothed = gs.Variable("/layers/0/self_attn/q_proj/input_quantizer/Mul",
                           dtype=np.float32,
                           shape=["tokens", 16])
    casted = gs.Variable("casted", dtype=np.float16, shape=["tokens", 16])
    nodes = [
        gs.Node(op="Mul", inputs=[hidden, scale], outputs=[smoothed]),
        gs.Node(op="Cast",
                attrs={"to": 10},
                inputs=[smoothed],
                outputs=[casted]),
    ]
    plugin_nodes, output = _int4_plugin(casted)
    return gs.Graph(nodes=nodes + plugin_nodes,
                    inputs=[hidden],
                    outputs=[output])


def _int4_graph_input_graph():
    """``per_layer_model_projection`` style: the plugin reads a graph input."""
    hidden = gs.Variable("inputs_embeds",
                         dtype=np.float16,
                         shape=["tokens", 16])
    nodes, output = _int4_plugin(hidden)
    return gs.Graph(nodes=nodes, inputs=[hidden], outputs=[output])


def _int4_unsmoothed_product_graph():
    """GPTQ-style export: the plugin is fed by a ``Mul`` of two activations
    (gate * up) with no scaling constant."""
    gate = gs.Variable("gate", dtype=np.float16, shape=["tokens", 16])
    up = gs.Variable("up", dtype=np.float16, shape=["tokens", 16])
    product = gs.Variable("product", dtype=np.float16, shape=["tokens", 16])
    mul = gs.Node(op="Mul", inputs=[gate, up], outputs=[product])
    nodes, output = _int4_plugin(product)
    return gs.Graph(nodes=[mul] + nodes, inputs=[gate, up], outputs=[output])


def _lora_nodes(onnx_dir):
    graph = gs.import_onnx(onnx.load(onnx_dir / "lora_model.onnx"))
    return graph, {node.name: node for node in graph.nodes}


@pytest.mark.parametrize("scale_first", [False, True])
def test_int4_awq_lora_branch_reads_the_unsmoothed_activation(
        tmp_path, scale_first):
    graph = _int4_awq_graph(scale_first=scale_first)

    matches = _match_int4_gemm(graph)

    assert len(matches) == 1
    assert matches[0].input.name == "hidden"
    assert matches[0].name == _INT4_GEMM_NAME
    assert matches[0].weight_shape == (16, 32)

    onnx.save(gs.export_onnx(graph), tmp_path / "model.onnx")
    insert_lora_and_save(str(tmp_path))
    lora_graph, nodes = _lora_nodes(tmp_path)

    assert [inp.name for inp in lora_graph.inputs] == [
        "hidden", f"{_INT4_STEM}.lora_A.weight", f"{_INT4_STEM}.lora_B.weight"
    ]
    assert nodes[f"{_INT4_GEMM_NAME}/lora_matmul_A"].inputs[0].name == "hidden"
    assert nodes["node_mul"].inputs[0 if not scale_first else 1].name == \
        "hidden"
    assert nodes["node_int4_gemm"].inputs[0].name == "smoothed"
    add = nodes[f"{_INT4_GEMM_NAME}/lora_add"]
    assert add.inputs[0].name == "gemm_out"
    assert nodes["node_output_cast"].inputs[0].name == \
        f"{_INT4_GEMM_NAME}/lora_add_output"


def test_int4_legacy_awq_lora_branch_reads_the_unsmoothed_activation():
    matches = _match_int4_gemm(_int4_legacy_awq_graph())

    assert len(matches) == 1
    assert matches[0].input.name == "hidden"


def test_int4_graph_input_activation_is_used_directly(tmp_path):
    graph = _int4_graph_input_graph()

    matches = _match_int4_gemm(graph)

    assert len(matches) == 1
    assert matches[0].input.name == "inputs_embeds"

    onnx.save(gs.export_onnx(graph), tmp_path / "model.onnx")
    insert_lora_and_save(str(tmp_path))
    _, nodes = _lora_nodes(tmp_path)

    assert nodes[f"{_INT4_GEMM_NAME}/lora_matmul_A"].inputs[0].name == \
        "inputs_embeds"


def test_int4_unsmoothed_product_activation_is_used_directly():
    matches = _match_int4_gemm(_int4_unsmoothed_product_graph())

    assert len(matches) == 1
    assert matches[0].input.name == "product"


def test_int4_unrecognized_scaling_mul_is_rejected():
    graph = _int4_awq_graph(scale_name="node_scale")

    with pytest.raises(ValueError, match="pre_quant_scale"):
        _match_int4_gemm(graph)


def test_int4_lora_add_precedes_architectural_output_scale(tmp_path):
    graph = _int4_awq_graph(with_output_scale=True)
    onnx.save(gs.export_onnx(graph), tmp_path / "model.onnx")

    insert_lora_and_save(str(tmp_path))
    _, nodes = _lora_nodes(tmp_path)

    add = nodes[f"{_INT4_GEMM_NAME}/lora_add"]
    assert add.inputs[0].name == "gemm_out"
    assert nodes["node_output_scale"].inputs[0].name == \
        f"{_INT4_GEMM_NAME}/lora_add_output"
