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

import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from safetensors import safe_open

from tensorrt_edgellm._safetensors_io import save_file
from tensorrt_edgellm.checkpoint import loader as checkpoint_loader

from .phi4mm_utils import load_phi4mm_model

logger = logging.getLogger(__name__)

GEMMInfo = namedtuple("GEMMInfo", ["input", "output", "name", "weight_shape"])


def _find_matmul_node(quantize_linear_node: gs.Node) -> gs.Node:
    """
    Find the MatMul node after the quantize linear node. Usually it is 2-3 levels deep.
    """
    node = quantize_linear_node
    max_depth = 5
    depth = 0
    while node.op != "MatMul" and depth < max_depth:
        node = node.outputs[0].outputs[0]
        depth += 1
    if depth >= max_depth:
        raise ValueError(
            f"MatMul node not found after {max_depth} levels of quantization for {quantize_linear_node.name}. Please check the ONNX graph."
        )
    return node


_WEIGHT_DQ_OPS = {
    "DequantizeLinear",
    "TRT_MXFP8DequantizeLinear",
}


def _find_weight_shape(gemm_node: gs.Node) -> tuple:
    """
    Find the weight shape of the GEMM node. The weight shape is not intuitive because of the quantization and transpose nodes.
    """
    # Weights are always on the second of a GEMM node.
    node = gemm_node.inputs[1].inputs[0]
    max_depth = 5
    depth = 0
    attrs = gemm_node.attrs or {}
    num_transpose = (int(attrs.get("transB", 0))
                     if gemm_node.op == "Gemm" else 0)
    while node.op not in _WEIGHT_DQ_OPS and depth < max_depth:
        if node.op == "Transpose":
            num_transpose += 1
        node = node.inputs[0].inputs[0]
        depth += 1
    if depth >= max_depth:
        raise ValueError(
            f"DequantizeLinear node not found above {max_depth} levels of GEMM for {gemm_node.name}. Please check the ONNX graph."
        )
    weight = node.inputs[0]
    if num_transpose % 2 == 1:
        weight_shape = (weight.shape[1], weight.shape[0])
    else:
        weight_shape = weight.shape
    return tuple(weight_shape)


# ONNX FP8 dtype codes: FLOAT8E4M3FN=17, FLOAT8E5M2=18 (TensorProto enum).
_FP8_OUTPUT_DTYPES = {17, 18}


def _is_fp8_quantize_node(node: gs.Node) -> bool:
    """Standard ONNX FP8 producer with ``output_dtype`` set to an FP8 code."""
    if node.op == "QuantizeLinear":
        out_dtype = node.attrs.get("output_dtype") if node.attrs else None
        return isinstance(out_dtype, int) and out_dtype in _FP8_OUTPUT_DTYPES
    return False


_TRANSPARENT_OPS = {"Cast", "Reshape", "Identity"}


def _linear_consumers_after_dq(quantize_node: gs.Node, max_hops: int = 3):
    """Yield every MatMul or Gemm reached through activation Q/DQ. The DQ may
    fan out to several linear consumers (Q/K/V share one dequantized hidden
    state). Walks through ``max_hops`` levels of transparent ops
    (``Cast``/``Reshape``/``Identity``) between DQ and the linear op so a future
    modelopt emit pattern with intermediate ops keeps binding correctly instead
    of silently dropping the LoRA slot."""
    for dq in list(quantize_node.outputs[0].outputs):
        if dq.op != "DequantizeLinear":
            continue
        frontier = list(dq.outputs[0].outputs)
        seen = set()
        for _ in range(max_hops + 1):
            next_frontier = []
            for cons in frontier:
                cons_id = id(cons)
                if cons_id in seen:
                    continue
                seen.add(cons_id)
                if cons.op in ("MatMul", "Gemm"):
                    yield cons
                elif cons.op in _TRANSPARENT_OPS and cons.outputs:
                    next_frontier.extend(cons.outputs[0].outputs)
            if not next_frontier:
                break
            frontier = next_frontier


def _stem_from_init_name(name: str) -> str:
    """Convert an initializer name like ``_model.<...>.weight`` into the
    module-path stem used by the adapter safetensors (``model.<...>``).
    Returns ``""`` when the name is not in the expected
    ``_model.<...>.weight`` form so the caller refuses to synthesize a
    LoRA binding (silent garbage names produce dummy bindings at runtime)."""
    if not (name.startswith("_model.") and name.endswith(".weight")):
        return ""
    return name[len("_model."):-len(".weight")]


def _synth_gemm_name(stem: str, fallback: str) -> str:
    """Encode ``stem`` as a path so the downstream
    ``gemm_name.replace("/", ".").rsplit(".",1)[0][1:]`` in
    ``insert_lora_and_save`` collapses back to ``stem``. Fall back to the
    raw MatMul/plugin node name when the stem is unrecoverable — the
    runtime will then drop the binding by name mismatch rather than wire
    a silent garbage tensor."""
    return ("/" + stem.replace(".", "/") + "/MatMul") if stem else fallback


def _stem_from_weight_init(linear_node: gs.Node) -> str:
    """Walk up a linear op's weight side to its originating ``_model.…weight``
    initializer and return its module-path stem. Returns ``""`` when the walk
    does not terminate at an initializer named ``_model.<...>.weight`` — the
    caller must then refuse to synthesize a LoRA binding name (silent garbage
    names produce dummy bindings at runtime)."""
    if not linear_node.inputs[1].inputs:
        return ""
    node = linear_node.inputs[1].inputs[0]
    depth = 0
    while node is not None and node.op not in _WEIGHT_DQ_OPS and depth < 5:
        if not node.inputs or not node.inputs[0].inputs:
            return ""
        node = node.inputs[0].inputs[0]
        depth += 1
    if node is None or not node.inputs:
        return ""
    return _stem_from_init_name(getattr(node.inputs[0], "name", "") or "")


def _match_fp8_gemm(graph: gs.Graph):
    """
    Match FP8 GEMM nodes in the graph.

    A single Quantize may fan out to multiple linear ops through one
    DequantizeLinear; each linear op becomes its own GEMM, and ``name`` is
    rewritten to a path-style stem derived from the weight initializer so the
    downstream LoRA input names match the adapter safetensors.
    """
    fp8_gemm_infos = []
    seen_linear_ids = set()
    for node in graph.nodes:
        if not _is_fp8_quantize_node(node):
            continue
        input_node = node.inputs[0]
        linear_nodes = list(_linear_consumers_after_dq(node))

        for linear_node in linear_nodes:
            attrs = linear_node.attrs or {}
            if (linear_node.op == "Gemm"
                    and (int(attrs.get("transA", 0)) != 0
                         or float(attrs.get("alpha", 1.0)) != 1.0)):
                continue
            if id(linear_node) in seen_linear_ids:
                continue
            seen_linear_ids.add(id(linear_node))
            stem = _stem_from_weight_init(linear_node)
            fp8_gemm_infos.append(
                GEMMInfo(input=input_node,
                         output=linear_node.outputs[0],
                         name=_synth_gemm_name(stem, linear_node.name),
                         weight_shape=_find_weight_shape(linear_node)))
    return fp8_gemm_infos


def _match_nvfp4_gemm(graph: gs.Graph):
    """
    Match NVFP4 GEMM nodes in the graph.

    Same dynamo-naming hazard as the FP8 path: the downstream MatMul node
    is named ``node_MatMul_N`` by the dynamo exporter, so derive the GEMM
    name from the weight initializer instead.
    """
    nvfp4_gemm_infos = []
    nvfp4_quantize_linear_nodes = [
        node for node in graph.nodes if node.op == "TRT_FP4DynamicQuantize"
    ]
    for node in nvfp4_quantize_linear_nodes:
        input_node = node.inputs[0]
        matmul_node = _find_matmul_node(node)
        weight_shape = _find_weight_shape(matmul_node)
        stem = _stem_from_weight_init(matmul_node)
        nvfp4_gemm_infos.append(
            GEMMInfo(input=input_node,
                     output=matmul_node.outputs[0],
                     name=_synth_gemm_name(stem, matmul_node.name),
                     weight_shape=weight_shape))
    return nvfp4_gemm_infos


_PRE_QUANT_SCALE_SUFFIX = ".pre_quant_scale"


def _unsmoothed_int4_activation(node: gs.Node) -> gs.Tensor:
    """Return the activation the LoRA branch of an INT4 plugin must consume.

    ModelOpt AWQ stores ``Q(W / s)`` per input channel and the exported graph
    feeds the plugin ``Mul(x, s)``, so the base GEMM computes ``x * s * W / s
    = x * W``. A PEFT adapter is trained as ``x * A * B`` on the unsmoothed
    model, so its branch must read ``x``, the operand of that ``Mul`` that is
    not the ``*.pre_quant_scale`` constant (either operand order). Legacy
    ModelOpt-traced exports smooth through ``Cast <- Mul(*input_quantizer*)``.
    Exports without activation smoothing in the graph (GPTQ, column-packed
    AWQ, or a plugin fed by a graph input such as
    ``per_layer_model_projection`` on ``inputs_embeds``) use the plugin input
    directly. A scaling op that cannot be classified is an error: taking the
    scaled tensor would silently bind the adapter to the wrong activation.
    """
    activation = node.inputs[0]
    if not activation.inputs:
        return activation
    producer = activation.inputs[0]
    if producer.op == "Mul":
        constants = [
            operand for operand in producer.inputs
            if isinstance(operand, gs.Constant)
        ]
        if not constants:
            return activation
        if len(producer.inputs) == 2 and len(constants) == 1 and (
                constants[0].name or "").endswith(_PRE_QUANT_SCALE_SUFFIX):
            scale = constants[0]
            return next(operand for operand in producer.inputs
                        if operand is not scale)
        raise ValueError(
            f"INT4 GEMM {node.name} is fed by Mul {producer.name} with "
            f"constant operand(s) {[c.name for c in constants]} that are not "
            "a *.pre_quant_scale; cannot locate the unsmoothed activation "
            "for the LoRA branch")
    if producer.op == "Cast":
        cast_input = producer.inputs[0]
        if "input_quantizer" in (cast_input.name or "") and cast_input.inputs:
            mul_node = cast_input.inputs[0]
            if mul_node.op == "Mul":
                return mul_node.inputs[0]
        if cast_input.inputs and cast_input.inputs[0].op == "Mul" and any(
                isinstance(operand, gs.Constant)
                for operand in cast_input.inputs[0].inputs):
            raise ValueError(
                f"INT4 GEMM {node.name} is fed by Cast {producer.name} over "
                f"an unrecognized scaling Mul {cast_input.inputs[0].name}; "
                "cannot locate the unsmoothed activation for the LoRA branch")
    return activation


def _match_int4_gemm(graph: gs.Graph):
    """
    Match INT4 GEMM nodes in the graph.

    Both ``Int4GroupwiseGemmPlugin`` (V1) and ``Int4GroupwiseGemmPluginV2``
    (V2, the default backend) carry their weight initializer directly as
    ``node.inputs[1]`` (no DequantizeLinear chain) and share the same
    ``gemm_k``/``gemm_n`` attributes, so the stem can be derived in one step
    for either. V2 must be matched too; otherwise the default INT4 backend's
    GEMMs get no LoRA inputs and adapters are silently dropped at runtime.
    The LoRA input is the activation before any AWQ smoothing scale; see
    :func:`_unsmoothed_int4_activation`.
    """
    int4_gemm_infos = []
    int4_gemm_nodes = [
        node for node in graph.nodes
        if node.op in ("Int4GroupwiseGemmPlugin", "Int4GroupwiseGemmPluginV2")
    ]
    for node in int4_gemm_nodes:
        input_node = _unsmoothed_int4_activation(node)
        weight_shape = (node.attrs["gemm_k"], node.attrs["gemm_n"])
        weight_init_name = getattr(node.inputs[1], "name", "") or ""
        stem = _stem_from_init_name(weight_init_name)
        int4_gemm_infos.append(
            GEMMInfo(input=input_node,
                     output=node.outputs[0],
                     name=_synth_gemm_name(stem, node.name),
                     weight_shape=weight_shape))
    return int4_gemm_infos


def _match_nvfp4_a16_gemm(graph: gs.Graph):
    """
    Match weight-only NVFP4 (W4A16) GEMM nodes in the graph.

    ``Nvfp4A16GemmPlugin`` (the Marlin weight-only backend) carries its
    packed weight initializer directly as ``node.inputs[1]`` and the GEMM
    shape in its ``gemm_k``/``gemm_n`` attributes, mirroring the INT4
    groupwise plugin (activation stays FP16, so there is no smoothing
    Mul/Cast on the input). This op is distinct from ``TRT_FP4DynamicQuantize``
    (the W4A4 QDQ path); without matching it the weight-only NVFP4 GEMMs get
    no LoRA inputs and adapters are silently dropped at runtime.
    """
    nvfp4_a16_gemm_infos = []
    nvfp4_a16_nodes = [
        node for node in graph.nodes if node.op == "Nvfp4A16GemmPlugin"
    ]
    for node in nvfp4_a16_nodes:
        input_node = node.inputs[0]
        weight_shape = (node.attrs["gemm_k"], node.attrs["gemm_n"])
        weight_init_name = getattr(node.inputs[1], "name", "") or ""
        stem = _stem_from_init_name(weight_init_name)
        nvfp4_a16_gemm_infos.append(
            GEMMInfo(input=input_node,
                     output=node.outputs[0],
                     name=_synth_gemm_name(stem, node.name),
                     weight_shape=weight_shape))
    return nvfp4_a16_gemm_infos


def _match_mxfp8_gemm(graph: gs.Graph):
    """
    Match MXFP8 GEMM nodes in the graph.
    """
    mxfp8_gemm_infos = []
    mxfp8_quantize_linear_nodes = [
        node for node in graph.nodes if node.op == "TRT_MXFP8DynamicQuantize"
    ]
    for node in mxfp8_quantize_linear_nodes:
        input_node = node.inputs[0]
        matmul_node = _find_matmul_node(node)
        weight_shape = _find_weight_shape(matmul_node)
        gemm_info = GEMMInfo(input=input_node,
                             output=matmul_node.outputs[0],
                             name=matmul_node.name,
                             weight_shape=weight_shape)
        mxfp8_gemm_infos.append(gemm_info)
    return mxfp8_gemm_infos


def _match_fp16_gemm(graph: gs.Graph):
    """
    Match FP16 MatMul and Gemm nodes in the graph.

    Both ops take their binding stem from a ``_model.<stem>.weight``
    initializer. A linear whose weight is an anonymous constant is skipped:
    the dynamo exporter emits the tied lm_head as ``MatMul(x, val_N)`` named
    ``node_linear``, and a node-name stem would bypass the ``lm_head``
    exclusion and bind a ``[hidden, r]`` x ``[r, vocab]`` branch on the
    logits under a garbage name.
    """
    fp16_gemm_infos = []
    fp16_gemm_nodes = [
        node for node in graph.nodes if node.op in ("MatMul", "Gemm")
    ]
    for node in fp16_gemm_nodes:
        input_node = node.inputs[0]
        if not isinstance(node.inputs[1], gs.Constant):
            continue
        weight_shape = tuple(node.inputs[1].shape)
        if node.op == "Gemm":
            attrs = node.attrs or {}
            if (int(attrs.get("transA", 0)) != 0
                    or float(attrs.get("alpha", 1.0)) != 1.0):
                continue
            if int(attrs.get("transB", 0)) != 0:
                weight_shape = (weight_shape[1], weight_shape[0])
        stem = _stem_from_init_name(node.inputs[1].name)
        if not stem:
            logger.info(
                "Skipping LoRA insertion for %s %s: weight %r is not a "
                "_model.<stem>.weight initializer", node.op, node.name,
                node.inputs[1].name)
            continue
        gemm_info = GEMMInfo(input=input_node,
                             output=node.outputs[0],
                             name=_synth_gemm_name(stem, node.name),
                             weight_shape=weight_shape)
        fp16_gemm_infos.append(gemm_info)
    return fp16_gemm_infos


def _match_gemm_infos(graph: gs.Graph):
    """
    Match all GEMM nodes in the graph.
    """
    gemm_infos = []
    gemm_infos.extend(_match_fp8_gemm(graph))
    gemm_infos.extend(_match_nvfp4_gemm(graph))
    gemm_infos.extend(_match_nvfp4_a16_gemm(graph))
    gemm_infos.extend(_match_int4_gemm(graph))
    gemm_infos.extend(_match_mxfp8_gemm(graph))
    gemm_infos.extend(_match_fp16_gemm(graph))
    return gemm_infos


# Helper functions for LoRA weight processing
_BASE_MODEL_PREFIX = "base_model.model."
_LORA_KEY_RE = re.compile(r"^(?P<stem>.+)\.lora_(?P<factor>[AB])\.weight$")

# PEFT features the runtime does not implement. Processing folds one global
# lora_alpha / r into B and binds plain lora_A / lora_B factors, so an adapter
# that sets any of these would load and silently compute something else.
_UNSUPPORTED_ADAPTER_FEATURES = {
    "use_rslora":
    "rsLoRA scaling (lora_alpha / sqrt(r)) is not applied",
    "use_dora":
    "DoRA magnitude vectors have no engine binding",
    "rank_pattern":
    "per-module ranks; the runtime assumes one global r",
    "alpha_pattern":
    "per-module alphas; the runtime assumes one global "
    "lora_alpha",
    "modules_to_save":
    "full-weight module copies have no engine binding",
    "layers_to_transform":
    "layer subsets; declared-target coverage cannot "
    "be checked",
    "layers_pattern":
    "layer subsets; declared-target coverage cannot be "
    "checked",
}

# Modules that are never LoRA-inserted: embeddings and the LM head live
# outside the inserted GEMM set (sidecars and the lm_head filter) and the
# vision/audio towers run in separate engines.
_UNSUPPORTED_MODULE_PARTS = frozenset({
    "embed_tokens",
    "embed_tokens_per_layer",
    "lm_head",
    "vision_tower",
    "audio_tower",
    "visual",
    "multi_modal_projector",
})

_EXPORT_CONFIG_PROVENANCE_KEYS = ("model", "edgellm_version", "hidden_size",
                                  "num_hidden_layers", "vocab_size",
                                  "quantization", "quant_method")

_SIDECAR_FILES = ("embedding.safetensors", "ple_embedding.safetensors")


@dataclasses.dataclass
class LoraProcessingReport:
    """What ``process_lora_weights_and_save`` wrote and what it left out."""
    output_path: str
    rank: int
    lora_scale: float
    key_prefix: Tuple[str, str]
    bound: List[str]
    """Adapter modules written to the processed file."""
    unused: List[str]
    """Adapter modules dropped because the engine has no such module (for
    example k/v_proj of the KV-shared Gemma 4 layers)."""
    unbound_untargeted: List[str]
    """Engine LoRA bindings outside the adapter's declared targets; the
    runtime binds rank-1 zeros for them."""


def _read_adapter_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def _check_adapter_config(config: dict) -> Tuple[float, int]:
    """Return ``(lora_alpha, r)`` after refusing adapter features the runtime
    does not implement."""
    unsupported = [
        f"{key} ({reason})"
        for key, reason in _UNSUPPORTED_ADAPTER_FEATURES.items()
        if config.get(key)
    ]
    if config.get("bias", "none") != "none":
        unsupported.append(
            "bias != none (bias updates have no engine binding)")
    if unsupported:
        raise ValueError("adapter_config.json uses unsupported features: " +
                         "; ".join(unsupported))
    r = int(config["r"])
    if r <= 0:
        raise ValueError(f"adapter_config.json has non-positive rank r={r}")
    return float(config["lora_alpha"]), r


def _process_tensor_name(key: str,
                         strip_prefix: str = "",
                         insert_prefix: str = "") -> str:
    """Map a PEFT tensor name onto the engine binding stem: drop
    ``base_model.model.``, apply the checkpoint loader's wrapper-prefix rule
    (``model.language_model.`` -> ``model.`` for Gemma 4 / Qwen3-VL style
    checkpoints), and ensure the ``model.`` root."""
    if key.startswith(_BASE_MODEL_PREFIX):
        key = key[len(_BASE_MODEL_PREFIX):]
    if strip_prefix and key.startswith(strip_prefix):
        key = insert_prefix + key[len(strip_prefix):]
    if not key.startswith('model.'):
        key = 'model.' + key
    return key


def _normalize_adapter_keys(keys) -> Tuple[Dict[str, str], Tuple[str, str]]:
    """Return ``{source key: engine key}`` and the detected prefix pair,
    rejecting two source keys that collapse onto one engine key."""
    stripped = [
        key[len(_BASE_MODEL_PREFIX):]
        if key.startswith(_BASE_MODEL_PREFIX) else key for key in keys
    ]
    prefix = checkpoint_loader._detect_key_prefix(stripped)
    mapping: Dict[str, str] = {}
    owners: Dict[str, str] = {}
    for key in keys:
        new_key = _process_tensor_name(key, *prefix)
        if new_key in owners:
            raise ValueError(f"adapter tensors {owners[new_key]!r} and "
                             f"{key!r} both map to {new_key!r}")
        owners[new_key] = key
        mapping[key] = new_key
    return mapping, prefix


def _hf_module_name(stem: str, key_prefix: Tuple[str, str]) -> str:
    """Inverse of the prefix mapping, so regex ``target_modules`` written
    against HF module paths can be matched against engine stems."""
    strip_prefix, insert_prefix = key_prefix
    if stem.startswith(insert_prefix):
        return strip_prefix + stem[len(insert_prefix):]
    return stem


def _matches_module_spec(spec, stem: str, hf_name: str) -> bool:
    """PEFT target semantics: a string is a full-match regex, a list matches
    a module path exactly or by ``.<suffix>``."""
    if not spec:
        return False
    names = (stem, hf_name)
    if isinstance(spec, str):
        return any(re.fullmatch(spec, name) for name in names)
    return any(name == target or name.endswith("." + target) for name in names
               for target in spec)


def _is_declared_target(stem: str, config: dict, key_prefix: Tuple[str, str],
                        adapter_suffixes: Set[str]) -> bool:
    hf_name = _hf_module_name(stem, key_prefix)
    if _matches_module_spec(config.get("exclude_modules"), stem, hf_name):
        return False
    target_modules = config.get("target_modules")
    if target_modules:
        return _matches_module_spec(target_modules, stem, hf_name)
    return stem.rsplit(".", 1)[-1] in adapter_suffixes


def _read_lora_bindings(
        onnx_dir: str) -> Tuple[Dict[str, Tuple[int, int]], Set[str]]:
    """Return ``{stem: (k, n)}`` for the ``<stem>.lora_A/B.weight`` inputs of
    ``lora_model.onnx`` and the set of module stems that own a
    ``_model.<stem>.weight`` initializer (modules the engine has, whether or
    not LoRA was inserted on them)."""
    lora_model_path = os.path.join(onnx_dir, "lora_model.onnx")
    model = onnx.load(lora_model_path, load_external_data=False)
    dims_a: Dict[str, int] = {}
    dims_b: Dict[str, int] = {}
    for graph_input in model.graph.input:
        match = _LORA_KEY_RE.match(graph_input.name)
        if not match:
            continue
        dims = graph_input.type.tensor_type.shape.dim
        if len(dims) != 2:
            raise ValueError(f"LoRA input {graph_input.name} is not rank-2")
        if match["factor"] == "A":
            dims_a[match["stem"]] = dims[0].dim_value
        else:
            dims_b[match["stem"]] = dims[1].dim_value
    if set(dims_a) != set(dims_b):
        raise ValueError(f"{lora_model_path} has unpaired LoRA inputs: "
                         f"{sorted(set(dims_a) ^ set(dims_b))}")
    if not dims_a:
        raise ValueError(f"{lora_model_path} declares no LoRA inputs; run "
                         "tensorrt-edgellm-insert-lora first")
    bindings = {stem: (dims_a[stem], dims_b[stem]) for stem in dims_a}
    for stem, (k, n) in bindings.items():
        if k <= 0 or n <= 0:
            raise ValueError(f"LoRA binding {stem} has invalid dims {(k, n)}")
    engine_modules = {
        _stem_from_init_name(init.name)
        for init in model.graph.initializer
    }
    engine_modules.discard("")
    return bindings, engine_modules


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _engine_provenance(onnx_dir: str) -> dict:
    provenance = {
        "onnx_dir": os.path.abspath(onnx_dir),
        "lora_model_sha256": _sha256(os.path.join(onnx_dir,
                                                  "lora_model.onnx")),
    }
    data_path = os.path.join(onnx_dir, "model.onnx.data")
    if os.path.exists(data_path):
        provenance["model_data_sha256"] = _sha256(data_path)
    config_path = os.path.join(onnx_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            export_config = json.load(f)
        provenance["export_config"] = {
            key: export_config[key]
            for key in _EXPORT_CONFIG_PROVENANCE_KEYS if key in export_config
        }
    sidecars = {}
    for name in _SIDECAR_FILES:
        path = os.path.join(onnx_dir, name)
        if os.path.exists(path):
            sidecars[name] = _sha256(path)
    if sidecars:
        provenance["sidecar_sha256"] = sidecars
    return provenance


def _process_tensor(tensor: torch.Tensor, key: str, lora_alpha: float,
                    r: int) -> torch.Tensor:
    """
    Process tensor according to requirements:
    1. Convert bf16 to fp16
    2. Multiply lora_B.weight by lora_alpha/r
    3. Ensure correct shapes for lora_A and lora_B

    Args:
        tensor (torch.Tensor): Input tensor
        key (str): Tensor name
        lora_alpha (float): LoRA alpha value
        r (int): LoRA rank

    Returns:
        torch.Tensor: Processed tensor
    """

    # Handle lora_B.weight multiplication
    if 'lora_B.weight' in key:
        tensor = tensor * (lora_alpha / r)

    # Ensure correct shapes
    if 'lora_A.weight' in key:
        if tensor.shape[-1] != r:
            tensor = tensor.transpose(-2, -1)
    elif 'lora_B.weight' in key:
        if tensor.shape[0] != r:
            tensor = tensor.transpose(-2, -1)

    # Convert to fp16
    tensor = tensor.to(torch.float16).contiguous()

    return tensor


def _check_lora_add_precedes_consumers(gemm_name: str,
                                       output_tensor: gs.Tensor,
                                       final_output: gs.Tensor,
                                       gemm_consumers) -> None:
    """Every former consumer of the GEMM output must now read the LoRA sum.
    This keeps the Add ahead of architectural output scales that follow a
    GEMM (the Gemma 4 PLE model projection feeds a ``hidden_size**-0.5``
    Mul), so the adapter delta is scaled exactly like the base output."""
    for out_node in gemm_consumers:
        reads_sum = any(inp is final_output for inp in out_node.inputs)
        reads_raw = any(inp is output_tensor for inp in out_node.inputs)
        if reads_raw or not reads_sum:
            raise RuntimeError(
                f"LoRA Add for {gemm_name} was not wired before consumer "
                f"{out_node.op} {out_node.name}")


# Main functions for external use
def insert_lora_and_save(onnx_dir: str):
    """
    Insert LoRA patterns into ONNX models.

    Args:
        onnx_dir (str): Directory containing model.onnx and config.json.
            The modified graph is written to lora_model.onnx in the same directory.
    """
    start_time = time.time()
    # Load ONNX model
    onnx_model_path = os.path.join(onnx_dir, "model.onnx")
    logger.info("Loading original ONNX model from %s", onnx_model_path)

    # The LoRA model will share the same data as the base model
    onnx_model = onnx.load(onnx_model_path, load_external_data=False)
    graph = gs.import_onnx(onnx_model)

    # Insert dynamic LoRA patterns
    logger.info("Inserting dynamic LoRA patterns")
    # Track all GEMM nodes that need LoRA
    gemm_infos = [
        gemm_info for gemm_info in _match_gemm_infos(graph)
        if "lm_head" not in gemm_info.name
    ]
    if not gemm_infos:
        raise ValueError(
            "LoRA insertion found no eligible linear layers in model.onnx; "
            "expected a supported MatMul, Gemm, or quantized GEMM node")

    # Insert LoRA patterns for each GEMM
    for gemm_info in gemm_infos:
        input_tensor = gemm_info.input
        output_tensor = gemm_info.output
        gemm_name = gemm_info.name
        weight_shape = gemm_info.weight_shape
        k, n = weight_shape
        # Create dynamic input tensors for LoRA weights
        gemm_name_for_lora = gemm_name.replace("/", ".").rsplit(".", 1)[0][1:]

        lora_a = gs.Variable(f"{gemm_name_for_lora}.lora_A.weight",
                             dtype=np.float16,
                             shape=[k, f"{gemm_name_for_lora}.rank"])
        lora_b = gs.Variable(f"{gemm_name_for_lora}.lora_B.weight",
                             dtype=np.float16,
                             shape=[f"{gemm_name_for_lora}.rank", n])
        graph.inputs.extend([lora_a, lora_b])

        # First MatMul: input @ lora_A
        lora_mid = gs.Variable(f"{gemm_name}/lora_mid", dtype=np.float16)
        graph.layer(name=f"{gemm_name}/lora_matmul_A",
                    op="MatMul",
                    inputs=[input_tensor, lora_a],
                    outputs=[lora_mid])

        # Second MatMul: (input @ lora_A) @ lora_B
        lora_out = gs.Variable(f"{gemm_name}/lora_gemm_out", dtype=np.float16)
        graph.layer(name=f"{gemm_name}/lora_matmul_B",
                    op="MatMul",
                    inputs=[lora_mid, lora_b],
                    outputs=[lora_out])

        # Add LoRA output to original output
        final_output = gs.Variable(f"{gemm_name}/lora_add_output",
                                   dtype=np.float16)
        # Before the Add node: it also consumes output_tensor. Only rewire consumers
        # that existed for the GEMM, or we would replace the Add's input with
        # final_output (the Add output) and create a graph cycle.
        gemm_consumers = list(output_tensor.outputs)
        graph.layer(name=f"{gemm_name}/lora_add",
                    op="Add",
                    inputs=[output_tensor, lora_out],
                    outputs=[final_output])

        # Replace at the same input index; remove+append broke multi-input ops (e.g.
        # Reshape must keep data vs shape tensor order for TensorRT shape inference).
        for out_node in gemm_consumers:
            for idx, inp in enumerate(out_node.inputs):
                if inp is output_tensor:
                    out_node.inputs[idx] = final_output
        _check_lora_add_precedes_consumers(gemm_name, output_tensor,
                                           final_output, gemm_consumers)

    graph.cleanup().toposort().fold_constants().cleanup()

    # Save modified ONNX model
    output_model_path = os.path.join(onnx_dir, "lora_model.onnx")
    logger.info("Saving modified ONNX model to %s", output_model_path)

    modified_onnx_model = gs.export_onnx(graph)
    onnx.save_model(modified_onnx_model, output_model_path)

    end_time = time.time()
    logger.info("LoRA model saved to %s", output_model_path)
    logger.info("LoRA insertion completed in %.2fs", end_time - start_time)


def _model_type_from_config(model_dir: str) -> str:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return ""
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("model_type", "")
    except (OSError, ValueError):
        return ""


def _path_contains(parent: Path, child: Path) -> bool:
    return parent == child or parent in child.parents


def _prepare_merge_output_dir(output_dir: str, model_dir: str,
                              lora_dir: str) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    model_path = Path(model_dir).expanduser().resolve()
    lora_path = Path(lora_dir).expanduser().resolve()
    protected_paths = {Path("/").resolve(), Path.home().resolve()}
    try:
        protected_paths.add(Path.cwd().resolve())
    except OSError:
        pass

    if output_path in protected_paths:
        raise ValueError(f"Refusing to remove protected output_dir: "
                         f"{output_path}")
    if _path_contains(output_path, model_path) or _path_contains(
            output_path, lora_path):
        raise ValueError("Refusing to use an output_dir that contains the "
                         "input model or LoRA adapter directory")

    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(f"output_dir exists and is not a directory: "
                             f"{output_path}")
        logger.warning("Removing existing LoRA merge output directory: %s",
                       output_path)
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def merge_lora_and_save(model_dir: str,
                        lora_dir: str,
                        output_dir: str,
                        device: str = "cuda",
                        torch_dtype: str = "float16") -> None:
    """Merge a PEFT LoRA adapter into a HuggingFace checkpoint."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")

    _prepare_merge_output_dir(output_dir, model_dir, lora_dir)

    model_type = _model_type_from_config(model_dir)
    is_phi4mm = model_type in ("phi4mm", "phi4_multimodal")
    if is_phi4mm:
        model = load_phi4mm_model(model_dir,
                                  dtype_map[torch_dtype],
                                  patch_peft_generation=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype_map[torch_dtype],
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    if device:
        model.to(device)

    lora_model = PeftModel.from_pretrained(model, lora_dir)
    merged_model = lora_model.merge_and_unload()
    if is_phi4mm:
        merged_model.config.vision_lora = None
        merged_model.config.speech_lora = None
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    try:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True)
    except (OSError, ValueError):
        processor = None
    if processor is not None:
        if model_type in ("phi4mm", "phi4_multimodal"):
            for name in ("preprocessor_config.json", "processor_config.json",
                         "processing_phi4mm.py"):
                src = os.path.join(model_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(output_dir, name))
        else:
            processor.save_pretrained(output_dir)

    logger.info("Merged LoRA adapter %s into %s", lora_dir, output_dir)


def process_lora_weights_and_save(
        input_dir: str,
        output_dir: str,
        onnx_dir: Optional[str] = None,
        max_lora_rank: Optional[int] = None) -> LoraProcessingReport:
    """Convert a PEFT LoRA adapter into the runtime's binding layout.

    Every tensor must be a ``lora_A`` / ``lora_B`` weight pair of one module;
    names are mapped onto the engine stems (``model.layers.N...``) including
    multimodal wrapper prefixes; B is scaled by ``lora_alpha / r``; both
    factors are stored FP16 as ``[k, r]`` and ``[r, n]``. Adapter features
    the runtime does not implement are refused (see
    ``_UNSUPPORTED_ADAPTER_FEATURES``), as are embedding, lm_head and
    vision/audio updates.

    With ``onnx_dir`` (the directory holding ``lora_model.onnx``) the adapter
    is validated against the engine's LoRA inputs: each kept pair must match
    its binding's ``(k, n)`` exactly; pairs for modules the engine does not
    have are dropped and reported; a pair for a module the engine has but
    never LoRA-inserted is an error; every binding inside the adapter's
    declared ``target_modules`` must be covered. Nothing is written when any
    check fails.

    Args:
        input_dir: Directory with ``adapter_config.json`` and
            ``adapter_model.safetensors``.
        output_dir: Destination of ``processed_adapter_model.safetensors``
            and ``config.json`` (the adapter config plus an
            ``edgellm_lora_processing`` provenance record).
        onnx_dir: Optional directory with ``lora_model.onnx``.
        max_lora_rank: Optional engine ``--maxLoraRank``; the adapter rank
            must not exceed it.

    Returns:
        LoraProcessingReport: bound, unused and unbound module lists.
    """
    config_path = os.path.join(input_dir, 'adapter_config.json')
    config = _read_adapter_config(config_path)
    lora_alpha, r = _check_adapter_config(config)
    if max_lora_rank is not None:
        if max_lora_rank <= 0:
            raise ValueError(f"max_lora_rank must be positive, got "
                             f"{max_lora_rank}")
        if r > max_lora_rank:
            raise ValueError(f"adapter rank r={r} exceeds the engine's "
                             f"max_lora_rank={max_lora_rank}")

    safetensor_path = os.path.join(input_dir, 'adapter_model.safetensors')
    with safe_open(safetensor_path, framework="pt") as f:
        keys = list(f.keys())
        mapping, key_prefix = _normalize_adapter_keys(keys)

        pairs: Dict[str, Dict[str, str]] = {}
        for key, new_key in mapping.items():
            match = _LORA_KEY_RE.match(new_key)
            if not match:
                raise ValueError(
                    f"unsupported adapter tensor {key!r}: only lora_A/lora_B "
                    "weights of linear modules can be bound")
            stem = match["stem"]
            unsupported = _UNSUPPORTED_MODULE_PARTS.intersection(
                stem.split("."))
            if unsupported:
                raise ValueError(
                    f"adapter tensor {key!r} updates {sorted(unsupported)}, "
                    "which the engine cannot adapt")
            pairs.setdefault(stem, {})[match["factor"]] = key
        unpaired = sorted(stem for stem, factors in pairs.items()
                          if set(factors) != {"A", "B"})
        if unpaired:
            raise ValueError(f"adapter modules without both lora_A and "
                             f"lora_B: {unpaired}")

        bindings: Optional[Dict[str, Tuple[int, int]]] = None
        unused: List[str] = []
        unbound_untargeted: List[str] = []
        if onnx_dir is not None:
            bindings, engine_modules = _read_lora_bindings(onnx_dir)
            not_inserted = sorted(
                stem for stem in pairs
                if stem not in bindings and stem in engine_modules)
            if not_inserted:
                raise ValueError(
                    "adapter targets modules the engine has but did not "
                    f"LoRA-insert: {not_inserted}")
            unused = sorted(stem for stem in pairs if stem not in bindings)
            bound = sorted(stem for stem in pairs if stem in bindings)
            if not bound:
                raise ValueError(
                    "no adapter module matches an engine LoRA binding "
                    f"(adapter stems look like {sorted(pairs)[:3]}, bindings "
                    f"like {sorted(bindings)[:3]})")
            adapter_suffixes = {stem.rsplit(".", 1)[-1] for stem in pairs}
            missing = []
            for stem in bindings:
                if stem in pairs:
                    continue
                if _is_declared_target(stem, config, key_prefix,
                                       adapter_suffixes):
                    missing.append(stem)
                else:
                    unbound_untargeted.append(stem)
            if missing:
                raise ValueError(
                    f"{len(missing)} engine binding(s) inside the adapter's "
                    f"declared targets have no adapter tensors: "
                    f"{sorted(missing)[:10]}")
        else:
            bound = sorted(pairs)
            if not bound:
                raise ValueError("adapter has no lora_A/lora_B pairs")

        processed_tensors = {}
        for stem in bound:
            tensors = {}
            for factor in ("A", "B"):
                key = pairs[stem][factor]
                tensor = _process_tensor(f.get_tensor(key), key, lora_alpha, r)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"adapter tensor {key!r} has non-finite "
                                     "values after FP16 conversion")
                tensors[factor] = tensor
            k, n = bindings[stem] if bindings else (tensors["A"].shape[0],
                                                    tensors["B"].shape[1])
            if tuple(tensors["A"].shape) != (k, r) or tuple(
                    tensors["B"].shape) != (r, n):
                raise ValueError(
                    f"adapter module {stem} has lora_A {tuple(tensors['A'].shape)} "
                    f"and lora_B {tuple(tensors['B'].shape)}; expected "
                    f"[{k}, {r}] and [{r}, {n}]")
            processed_tensors[f"{stem}.lora_A.weight"] = tensors["A"]
            processed_tensors[f"{stem}.lora_B.weight"] = tensors["B"]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,
                               'processed_adapter_model.safetensors')
    save_file(processed_tensors, output_path)

    provenance = {
        "source_adapter_dir": os.path.abspath(input_dir),
        "adapter_model_sha256": _sha256(safetensor_path),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "key_prefix": {
            "strip": key_prefix[0],
            "insert": key_prefix[1]
        },
        "rank": r,
        "lora_scale": lora_alpha / r,
        "max_lora_rank": max_lora_rank,
        "bound_modules": bound,
        "unused_adapter_modules": unused,
        "unbound_untargeted_bindings": sorted(unbound_untargeted),
    }
    if onnx_dir is not None:
        provenance["engine"] = _engine_provenance(onnx_dir)
    output_config = dict(config)
    output_config["edgellm_lora_processing"] = provenance
    with open(os.path.join(output_dir, 'config.json'), 'w') as out:
        json.dump(output_config, out, indent=2)

    logger.info("Processed %d LoRA modules to %s (rank %d, scale %.4f)",
                len(bound), output_path, r, lora_alpha / r)
    return LoraProcessingReport(output_path=output_path,
                                rank=r,
                                lora_scale=lora_alpha / r,
                                key_prefix=key_prefix,
                                bound=bound,
                                unused=unused,
                                unbound_untargeted=sorted(unbound_untargeted))
