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
"""insert-lora with --lora-fused-int4: the saved graph the engine builder reads. Selected INT4
GEMMs that share an activation are fed by one LoraShrinkPlugin (reading the unsmoothed
activation and their lora_A inputs); each becomes an Int4GroupwiseGemmLoraPlugin with its
shrink output and lora_B appended; GEMM outputs and consumers are untouched; no MatMul/Add
branch remains for them; other GEMMs keep the requested branch."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from tensorrt_edgellm.lora.lora import insert_lora_and_save

_PREFIX = "model.layers.0"


def _int4(stem, activation, n, k=16):
    weight = gs.Constant(f"_model.{stem}.weight",
                         values=np.zeros((16, 4), dtype=np.int8))
    scale = gs.Constant(f"_model.{stem}.weight_scale",
                        values=np.ones((1, n), dtype=np.float16))
    out = gs.Variable(f"{stem}.out", dtype=np.float16, shape=["tokens", n])
    node = gs.Node(op="Int4GroupwiseGemmPluginV2",
                   name=f"node_{stem}",
                   attrs={"gemm_k": k, "gemm_n": n, "group_size": 32},
                   inputs=[activation, weight, scale],
                   outputs=[out])
    return node, out


def _graph():
    """q and k read an AWQ-smoothed activation (Mul by a pre_quant_scale), o reads `attn`, and
    an FP16 MatMul (not INT4) also reads `attn`."""
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    attn = gs.Variable("attn", dtype=np.float16, shape=["tokens", 16])
    smoothed = gs.Variable("hidden_smoothed", dtype=np.float16, shape=["tokens", 16])
    pqs = gs.Constant(f"_model.{_PREFIX}.self_attn.q_proj.pre_quant_scale",
                      values=np.ones((16,), dtype=np.float16))
    nodes = [gs.Node(op="Mul", name="smooth", inputs=[hidden, pqs], outputs=[smoothed])]
    q, q_out = _int4(f"{_PREFIX}.self_attn.q_proj", smoothed, 32)
    k, _ = _int4(f"{_PREFIX}.self_attn.k_proj", smoothed, 8)
    o, o_out = _int4(f"{_PREFIX}.self_attn.o_proj", attn, 16)
    w = gs.Constant(f"_model.{_PREFIX}.mlp.fp16_proj.weight",
                    values=np.zeros((16, 8), dtype=np.float16))
    fp16_out = gs.Variable("fp16.out", dtype=np.float16, shape=["tokens", 8])
    mm = gs.Node(op="MatMul", name="fp16_mm", inputs=[attn, w], outputs=[fp16_out])
    nodes += [q, k, o, mm]
    q_cast = gs.Variable("q.cast", dtype=np.float16)
    nodes.append(gs.Node(op="Cast", name="cast_q", attrs={"to": 10}, inputs=[q_out],
                         outputs=[q_cast]))
    return gs.Graph(nodes=nodes, inputs=[hidden, attn],
                    outputs=[q_cast, o_out, fp16_out, k.outputs[0]])


def _insert(tmp_path, **kwargs):
    onnx.save(gs.export_onnx(_graph()), tmp_path / "model.onnx")
    insert_lora_and_save(str(tmp_path), **kwargs)
    return gs.import_onnx(onnx.load(tmp_path / "lora_model.onnx"))


def test_int4_targets_share_a_shrink_and_become_fused_nodes(tmp_path):
    graph = _insert(tmp_path, lora_fused_int4=True)
    fused = {n.name: n for n in graph.nodes if n.op == "Int4GroupwiseGemmLoraPlugin"}
    assert sorted(fused) == [f"node_{_PREFIX}.self_attn.{p}_proj" for p in ("k", "o", "q")]
    shrinks = [n for n in graph.nodes if n.op == "LoraShrinkPlugin"]
    by_input = {n.inputs[0].name: n for n in shrinks}
    # q and k share the unsmoothed activation (not the plugin input, which is smoothed).
    qk = by_input["hidden"]
    assert int(qk.attrs["num_targets"]) == 2
    assert [t.name for t in qk.inputs[1:]] == [
        f"{_PREFIX}.self_attn.q_proj.lora_A.weight", f"{_PREFIX}.self_attn.k_proj.lora_A.weight"
    ]
    assert int(by_input["attn"].attrs["num_targets"]) == 1
    q = fused[f"node_{_PREFIX}.self_attn.q_proj"]
    assert [t.name for t in q.inputs[:3]] == [
        "hidden_smoothed", f"_model.{_PREFIX}.self_attn.q_proj.weight",
        f"_model.{_PREFIX}.self_attn.q_proj.weight_scale"
    ]
    assert q.inputs[3] is qk.outputs[0]
    assert q.inputs[4].name == f"{_PREFIX}.self_attn.q_proj.lora_B.weight"
    assert fused[f"node_{_PREFIX}.self_attn.k_proj"].inputs[3] is qk.outputs[1]
    # The fused GEMM's output is still what its consumers and the graph outputs read.
    assert [c.name for c in q.outputs[0].outputs] == ["cast_q"]
    assert f"{_PREFIX}.self_attn.o_proj.out" in [t.name for t in graph.outputs]
    assert int(q.attrs["group_size"]) == 32


def test_non_int4_targets_keep_a_matmul_branch(tmp_path):
    graph = _insert(tmp_path, lora_fused_int4=True)
    matmul_inputs = {t.name for n in graph.nodes if n.op == "MatMul" for t in n.inputs}
    assert f"{_PREFIX}.mlp.fp16_proj.lora_A.weight" in matmul_inputs
    assert not any(".self_attn." in name and ".lora_" in name for name in matmul_inputs)
    assert len([n for n in graph.nodes if n.op == "Add"]) == 1
    # Binding names and the dynamic rank are those of every other backend.
    lora_inputs = {t.name: t.shape for t in graph.inputs if ".lora_" in t.name}
    assert lora_inputs[f"{_PREFIX}.self_attn.q_proj.lora_A.weight"] == [
        16, f"{_PREFIX}.self_attn.q_proj.rank"
    ]
    assert lora_inputs[f"{_PREFIX}.self_attn.q_proj.lora_B.weight"] == [
        f"{_PREFIX}.self_attn.q_proj.rank", 32
    ]


def test_int4_target_fed_by_an_adapted_fp16_gemm_shrinks_the_sum(tmp_path):
    """An FP16 target whose output feeds an INT4 target: the INT4 GEMM and its shrink both read
    the FP16 GEMM's LoRA sum, not its raw output."""
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    mid = gs.Variable("mid", dtype=np.float16, shape=["tokens", 16])
    w = gs.Constant(f"_model.{_PREFIX}.mlp.up_proj.weight",
                    values=np.zeros((16, 16), dtype=np.float16))
    up = gs.Node(op="MatMul", name="up_mm", inputs=[hidden, w], outputs=[mid])
    down, down_out = _int4(f"{_PREFIX}.mlp.down_proj", mid, 16)
    onnx.save(gs.export_onnx(gs.Graph(nodes=[up, down], inputs=[hidden], outputs=[down_out])),
              tmp_path / "model.onnx")
    insert_lora_and_save(str(tmp_path), lora_fused_int4=True)
    graph = gs.import_onnx(onnx.load(tmp_path / "lora_model.onnx"))
    fused = next(n for n in graph.nodes if n.op == "Int4GroupwiseGemmLoraPlugin")
    shrink = next(n for n in graph.nodes if n.op == "LoraShrinkPlugin")
    assert fused.inputs[0].inputs[0].op == "Add"
    assert shrink.inputs[0] is fused.inputs[0]
