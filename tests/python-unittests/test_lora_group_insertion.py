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
"""insert-lora with --lora-group-plugin: which branches share a LoraGroupPlugin node, in what
input order, and that every former consumer (including a graph output) reads the plugin
output."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from tensorrt_edgellm.lora.lora import insert_lora_and_save


def _int4(stem, activation, n, k=16):
    weight = gs.Constant(f"_model.{stem}.weight",
                         values=np.zeros((16, 4), dtype=np.int8))
    scale = gs.Constant(f"_model.{stem}.weight_scale",
                        values=np.ones((1, n), dtype=np.float16))
    out = gs.Variable(f"{stem}.out", dtype=np.float16, shape=["tokens", n])
    node = gs.Node(op="Int4GroupwiseGemmPluginV2",
                   name=f"node_{stem}",
                   attrs={
                       "gemm_k": k,
                       "gemm_n": n,
                       "group_size": 128
                   },
                   inputs=[activation, weight, scale],
                   outputs=[out])
    return node, out


def _layer_graph():
    """One decoder layer's linear structure: q/k/v share `hidden`, o reads the attention
    output, gate/up share `post`, down is a graph output, and a per-layer projection that
    --lora-targets self_attn,mlp skips."""
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    attn = gs.Variable("attn", dtype=np.float16, shape=["tokens", 16])
    post = gs.Variable("post", dtype=np.float16, shape=["tokens", 16])
    act = gs.Variable("act", dtype=np.float16, shape=["tokens", 16])
    nodes, outs = [], {}
    for stem, inp, n in (("model.layers.0.self_attn.q_proj", hidden, 32),
                         ("model.layers.0.self_attn.k_proj", hidden, 8),
                         ("model.layers.0.self_attn.v_proj", hidden, 8),
                         ("model.layers.0.self_attn.o_proj", attn, 16),
                         ("model.layers.0.mlp.gate_proj", post, 64),
                         ("model.layers.0.mlp.up_proj", post, 64),
                         ("model.layers.0.mlp.down_proj", act, 16),
                         ("model.layers.0.per_layer_projection", hidden, 16)):
        node, out = _int4(stem, inp, n)
        nodes.append(node)
        outs[stem.rsplit(".", 1)[-1]] = out
    # Every GEMM output but down's feeds a consumer; down is the graph output.
    graph_outputs = [outs["down_proj"]]
    for name, out in outs.items():
        if name == "down_proj":
            continue
        consumed = gs.Variable(f"{name}.cast", dtype=np.float16)
        nodes.append(
            gs.Node(op="Cast",
                    name=f"cast_{name}",
                    attrs={"to": 10},
                    inputs=[out],
                    outputs=[consumed]))
        graph_outputs.append(consumed)
    return gs.Graph(nodes=nodes,
                    inputs=[hidden, attn, post, act],
                    outputs=graph_outputs)


def _insert(tmp_path, **kwargs):
    onnx.save(gs.export_onnx(_layer_graph()), tmp_path / "model.onnx")
    insert_lora_and_save(str(tmp_path), **kwargs)
    return gs.import_onnx(onnx.load(tmp_path / "lora_model.onnx"))


def _groups(graph):
    return {
        node.name: node
        for node in graph.nodes if node.op == "LoraGroupPlugin"
    }


def test_siblings_sharing_an_input_share_one_in_place_node(tmp_path):
    graph = _insert(tmp_path,
                    lora_group_plugin=True,
                    lora_targets=["self_attn", "mlp"])
    groups = _groups(graph)
    sizes = sorted(int(n.attrs["num_targets"]) for n in groups.values())
    assert sizes == [1, 1, 2, 3]
    assert all(int(n.attrs["inplace"]) == 1 for n in groups.values())
    assert not [n for n in graph.nodes if n.op in ("MatMul", "Add")]

    qkv = groups["/model/layers/0/self_attn/q_proj/MatMul/lora_group"]
    names = [t.name for t in qkv.inputs]
    prefix = "model.layers.0.self_attn"
    assert names == [
        f"{prefix}.q_proj.out", f"{prefix}.k_proj.out", f"{prefix}.v_proj.out",
        "hidden", f"{prefix}.q_proj.lora_A.weight",
        f"{prefix}.k_proj.lora_A.weight", f"{prefix}.v_proj.lora_A.weight",
        f"{prefix}.q_proj.lora_B.weight", f"{prefix}.k_proj.lora_B.weight",
        f"{prefix}.v_proj.lora_B.weight"
    ]
    # The consumers now read the plugin outputs, in the same target order.
    for out, proj in zip(qkv.outputs, ("q_proj", "k_proj", "v_proj")):
        assert [c.name for c in out.outputs] == [f"cast_{proj}"]


def test_graph_output_and_skipped_projection(tmp_path):
    graph = _insert(tmp_path,
                    lora_group_plugin=True,
                    lora_targets=["self_attn", "mlp"])
    down = _groups(graph)["/model/layers/0/mlp/down_proj/MatMul/lora_group"]
    assert down.outputs[0].name in [t.name for t in graph.outputs]
    assert "model.layers.0.mlp.down_proj.out" not in [
        t.name for t in graph.outputs
    ]
    # --lora-targets self_attn,mlp leaves the per-layer projection without LoRA inputs.
    assert not [t for t in graph.inputs if "per_layer_projection" in t.name]
    assert len([t for t in graph.inputs if ".lora_" in t.name]) == 14


def test_without_the_filter_every_int4_projection_gets_a_target(tmp_path):
    graph = _insert(tmp_path, lora_group_plugin=True)
    groups = _groups(graph)
    # per_layer_projection shares `hidden` with q/k/v: a fourth sibling starts a new node.
    assert sum(int(n.attrs["num_targets"]) for n in groups.values()) == 8
    assert sorted(int(n.attrs["num_targets"])
                  for n in groups.values()) == [1, 1, 1, 2, 3]
