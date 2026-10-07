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
"""Adapter contract of ``process_lora_weights_and_save``: PEFT key mapping,
validation against the engine's LoRA bindings, and refusal of adapter
features the runtime does not implement."""

import json

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
import torch
from safetensors.torch import save_file

from tensorrt_edgellm.lora.lora import process_lora_weights_and_save

RANK = 4
ALPHA = 8.0
HIDDEN = 16
ATTN = 8
INTER = 24
PLE = 6

# The engine has k/v_proj on layers 0 and 1 only; layer 2 is KV-shared.
_ENGINE_SHAPES = {}
for _layer in range(3):
    _ENGINE_SHAPES[f"model.layers.{_layer}.self_attn.q_proj"] = (HIDDEN, ATTN)
    _ENGINE_SHAPES[f"model.layers.{_layer}.self_attn.o_proj"] = (ATTN, HIDDEN)
    _ENGINE_SHAPES[f"model.layers.{_layer}.mlp.gate_proj"] = (HIDDEN, INTER)
    _ENGINE_SHAPES[f"model.layers.{_layer}.mlp.up_proj"] = (HIDDEN, INTER)
    _ENGINE_SHAPES[f"model.layers.{_layer}.mlp.down_proj"] = (INTER, HIDDEN)
    if _layer < 2:
        _ENGINE_SHAPES[f"model.layers.{_layer}.self_attn.k_proj"] = (HIDDEN,
                                                                     ATTN)
        _ENGINE_SHAPES[f"model.layers.{_layer}.self_attn.v_proj"] = (HIDDEN,
                                                                     ATTN)
_ENGINE_SHAPES["model.layers.0.per_layer_input_gate"] = (HIDDEN, PLE)
_ENGINE_SHAPES["model.per_layer_model_projection"] = (HIDDEN, PLE)

# A module the engine has but the inserter skipped (no LoRA inputs).
_NOT_INSERTED = "model.layers.0.mlp.skipped_proj"

_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]
_HF_PREFIX = "base_model.model.model.language_model."


def _write_lora_model(onnx_dir, bindings=_ENGINE_SHAPES):
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", HIDDEN])
    nodes, inputs, outputs = [], [hidden], []
    for stem, (k, n) in bindings.items():
        lora_a = gs.Variable(f"{stem}.lora_A.weight",
                             dtype=np.float16,
                             shape=[k, f"{stem}.rank"])
        lora_b = gs.Variable(f"{stem}.lora_B.weight",
                             dtype=np.float16,
                             shape=[f"{stem}.rank", n])
        weight = gs.Constant(f"_model.{stem}.weight",
                             values=np.zeros((n, k), dtype=np.float16))
        out = gs.Variable(f"{stem}/out", dtype=np.float16)
        nodes.append(
            gs.Node(op="Gemm",
                    attrs={"transB": 1},
                    inputs=[hidden, weight],
                    outputs=[out]))
        inputs.extend([lora_a, lora_b])
        outputs.append(out)
    skipped = gs.Constant(f"_model.{_NOT_INSERTED}.weight",
                          values=np.zeros((INTER, HIDDEN), dtype=np.float16))
    skipped_out = gs.Variable("skipped/out", dtype=np.float16)
    nodes.append(
        gs.Node(op="Gemm",
                attrs={"transB": 1},
                inputs=[hidden, skipped],
                outputs=[skipped_out]))
    outputs.append(skipped_out)
    graph = gs.Graph(nodes=nodes, inputs=inputs, outputs=outputs)
    onnx.save(gs.export_onnx(graph), onnx_dir / "lora_model.onnx")
    (onnx_dir / "config.json").write_text(
        json.dumps({
            "model": "gemma4_text",
            "edgellm_version": "0.11.0"
        }))


def _hf_key(stem, factor, prefix=_HF_PREFIX):
    return f"{prefix}{stem[len('model.'):]}.lora_{factor}.weight"


def _adapter_tensors(stems=None,
                     rank=RANK,
                     prefix=_HF_PREFIX,
                     dtype=torch.bfloat16):
    """PEFT layout: lora_A ``[r, in]``, lora_B ``[out, r]``. Layer 2 carries
    k/v_proj pairs the engine has no module for (KV-shared layer)."""
    if stems is None:
        stems = [s for s in _ENGINE_SHAPES if "per_layer" not in s]
        stems += [
            "model.layers.2.self_attn.k_proj",
            "model.layers.2.self_attn.v_proj"
        ]
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for stem in stems:
        k, n = _ENGINE_SHAPES.get(stem, (HIDDEN, ATTN))
        tensors[_hf_key(stem, "A", prefix)] = torch.randn(
            (rank, k), generator=generator).to(dtype)
        tensors[_hf_key(stem, "B", prefix)] = torch.randn(
            (n, rank), generator=generator).to(dtype)
    return tensors


def _write_adapter(adapter_dir, tensors=None, **config_overrides):
    adapter_dir.mkdir(exist_ok=True)
    config = {
        "base_model_name_or_path": "google/gemma-4-E4B-it",
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": ALPHA,
        "target_modules": list(_TARGETS),
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
    }
    config.update(config_overrides)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
    save_file(tensors if tensors is not None else _adapter_tensors(),
              str(adapter_dir / "adapter_model.safetensors"))
    return adapter_dir


@pytest.fixture
def onnx_dir(tmp_path):
    path = tmp_path / "llm"
    path.mkdir()
    _write_lora_model(path)
    return path


def test_wrapper_keys_map_to_engine_bindings_and_unused_pairs_are_dropped(
        tmp_path, onnx_dir):
    adapter = _write_adapter(tmp_path / "adapter")
    out = tmp_path / "out"

    report = process_lora_weights_and_save(str(adapter),
                                           str(out),
                                           onnx_dir=str(onnx_dir),
                                           max_lora_rank=16)

    expected_bound = sorted(s for s in _ENGINE_SHAPES if "per_layer" not in s)
    assert report.bound == expected_bound
    assert report.unused == [
        "model.layers.2.self_attn.k_proj", "model.layers.2.self_attn.v_proj"
    ]
    assert report.unbound_untargeted == [
        "model.layers.0.per_layer_input_gate",
        "model.per_layer_model_projection"
    ]
    assert report.key_prefix == ("model.language_model.", "model.")

    from safetensors.torch import load_file
    processed = load_file(str(out / "processed_adapter_model.safetensors"))
    assert sorted(processed) == sorted(f"{s}.lora_{f}.weight"
                                       for s in expected_bound for f in "AB")
    raw = _adapter_tensors()
    stem = "model.layers.1.self_attn.q_proj"
    lora_a = processed[f"{stem}.lora_A.weight"]
    lora_b = processed[f"{stem}.lora_B.weight"]
    assert lora_a.dtype == torch.float16 and lora_b.dtype == torch.float16
    assert tuple(lora_a.shape) == (HIDDEN, RANK)
    assert tuple(lora_b.shape) == (RANK, ATTN)
    torch.testing.assert_close(lora_a.float(), raw[_hf_key(stem,
                                                           "A")].float().T)
    torch.testing.assert_close(
        lora_b.float(), raw[_hf_key(stem, "B")].float().T * ALPHA / RANK)

    config = json.loads((out / "config.json").read_text())
    assert config["r"] == RANK
    provenance = config["edgellm_lora_processing"]
    assert provenance["bound_modules"] == expected_bound
    assert provenance["unused_adapter_modules"] == report.unused
    assert provenance["lora_scale"] == ALPHA / RANK
    assert provenance["max_lora_rank"] == 16
    assert len(provenance["adapter_model_sha256"]) == 64
    assert provenance["engine"]["export_config"]["model"] == "gemma4_text"
    assert len(provenance["engine"]["lora_model_sha256"]) == 64


def test_causal_lm_keys_map_unchanged(tmp_path, onnx_dir):
    tensors = _adapter_tensors(prefix="base_model.model.model.")
    adapter = _write_adapter(tmp_path / "adapter", tensors)

    report = process_lora_weights_and_save(str(adapter),
                                           str(tmp_path / "out"),
                                           onnx_dir=str(onnx_dir))

    assert report.key_prefix == ("", "")
    assert "model.layers.0.self_attn.q_proj" in report.bound


def test_without_onnx_dir_every_pair_is_kept(tmp_path):
    adapter = _write_adapter(tmp_path / "adapter")

    report = process_lora_weights_and_save(str(adapter), str(tmp_path / "out"))

    assert "model.layers.2.self_attn.k_proj" in report.bound
    assert report.unused == [] and report.unbound_untargeted == []


def test_regex_target_modules_are_matched_against_hf_names(tmp_path, onnx_dir):
    adapter = _write_adapter(
        tmp_path / "adapter",
        target_modules=r"model\.language_model\.layers\.\d+\.self_attn\.q_proj"
    )

    report = process_lora_weights_and_save(str(adapter),
                                           str(tmp_path / "out"),
                                           onnx_dir=str(onnx_dir))

    assert "model.layers.0.self_attn.q_proj" in report.bound


def test_over_rank_adapter_fails_before_writing(tmp_path, onnx_dir):
    adapter = _write_adapter(tmp_path / "adapter",
                             _adapter_tensors(rank=17),
                             r=17)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="r=17 exceeds"):
        process_lora_weights_and_save(str(adapter),
                                      str(out),
                                      onnx_dir=str(onnx_dir),
                                      max_lora_rank=16)
    assert not out.exists()


@pytest.mark.parametrize("key, value", [
    ("use_rslora", True),
    ("use_dora", True),
    ("rank_pattern", {
        "q_proj": 8
    }),
    ("alpha_pattern", {
        "q_proj": 16
    }),
    ("modules_to_save", ["embed_tokens", "lm_head"]),
    ("layers_to_transform", [0, 1]),
    ("bias", "all"),
])
def test_unsupported_adapter_features_are_refused(tmp_path, onnx_dir, key,
                                                  value):
    adapter = _write_adapter(tmp_path / "adapter", **{key: value})
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=key):
        process_lora_weights_and_save(str(adapter),
                                      str(out),
                                      onnx_dir=str(onnx_dir))
    assert not out.exists()


@pytest.mark.parametrize("key, match", [
    (f"{_HF_PREFIX}embed_tokens.weight", "only lora_A/lora_B"),
    (f"{_HF_PREFIX}embed_tokens.lora_embedding_A", "only lora_A/lora_B"),
    (f"{_HF_PREFIX}layers.0.self_attn.q_proj.lora_magnitude_vector",
     "only lora_A/lora_B"),
    ("base_model.model.lm_head.lora_A.weight", "lm_head"),
    ("base_model.model.model.vision_tower.blocks.0.attn.q.lora_A.weight",
     "vision_tower"),
])
def test_non_linear_lora_tensors_are_refused(tmp_path, onnx_dir, key, match):
    tensors = _adapter_tensors()
    tensors[key] = torch.zeros((RANK, HIDDEN), dtype=torch.bfloat16)
    adapter = _write_adapter(tmp_path / "adapter", tensors)

    with pytest.raises(ValueError, match=match):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))


def test_unpaired_factor_is_refused(tmp_path, onnx_dir):
    tensors = _adapter_tensors()
    del tensors[_hf_key("model.layers.0.self_attn.q_proj", "B")]
    adapter = _write_adapter(tmp_path / "adapter", tensors)

    with pytest.raises(ValueError, match="without both lora_A and lora_B"):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))


def test_dimension_mismatch_is_refused(tmp_path, onnx_dir):
    tensors = _adapter_tensors()
    tensors[_hf_key("model.layers.0.self_attn.q_proj", "A")] = torch.zeros(
        (RANK, HIDDEN + 1), dtype=torch.bfloat16)
    adapter = _write_adapter(tmp_path / "adapter", tensors)

    with pytest.raises(ValueError, match="expected \\[16, 4\\]"):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))


def test_non_finite_values_are_refused(tmp_path, onnx_dir):
    tensors = _adapter_tensors()
    key = _hf_key("model.layers.0.mlp.up_proj", "B")
    tensors[key] = tensors[key].clone()
    tensors[key][0, 0] = float("nan")
    adapter = _write_adapter(tmp_path / "adapter", tensors)

    with pytest.raises(ValueError, match="non-finite"):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))


def test_fp16_overflow_is_refused(tmp_path, onnx_dir):
    tensors = _adapter_tensors()
    key = _hf_key("model.layers.0.mlp.up_proj", "B")
    tensors[key] = torch.full_like(tensors[key], 1e5)
    adapter = _write_adapter(tmp_path / "adapter", tensors)

    with pytest.raises(ValueError, match="non-finite"):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))


def test_name_collision_after_mapping_is_refused(tmp_path, onnx_dir):
    tensors = _adapter_tensors()
    stem = "model.layers.0.self_attn.q_proj"
    tensors[_hf_key(stem, "A", "base_model.model.model.")] = tensors[_hf_key(
        stem, "A")].clone()
    adapter = _write_adapter(tmp_path / "adapter", tensors)

    with pytest.raises(ValueError, match="both map to"):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))


def test_missing_declared_target_is_refused(tmp_path, onnx_dir):
    tensors = _adapter_tensors()
    for factor in "AB":
        del tensors[_hf_key("model.layers.1.self_attn.q_proj", factor)]
    adapter = _write_adapter(tmp_path / "adapter", tensors)

    with pytest.raises(ValueError, match="model.layers.1.self_attn.q_proj"):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))


def test_untargeted_bindings_are_not_required(tmp_path, onnx_dir):
    stems = [f"model.layers.{i}.self_attn.q_proj" for i in range(3)]
    adapter = _write_adapter(tmp_path / "adapter",
                             _adapter_tensors(stems),
                             target_modules=["q_proj"])

    report = process_lora_weights_and_save(str(adapter),
                                           str(tmp_path / "out"),
                                           onnx_dir=str(onnx_dir))

    assert report.bound == stems
    assert "model.layers.0.self_attn.k_proj" in report.unbound_untargeted


def test_zero_matched_pairs_is_refused(tmp_path, onnx_dir):
    stems = [
        "model.layers.2.self_attn.k_proj", "model.layers.2.self_attn.v_proj"
    ]
    adapter = _write_adapter(tmp_path / "adapter",
                             _adapter_tensors(stems),
                             target_modules=["k_proj", "v_proj"])

    with pytest.raises(ValueError, match="no adapter module matches"):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))


def test_module_without_lora_insertion_is_refused(tmp_path, onnx_dir):
    tensors = _adapter_tensors()
    tensors[_hf_key(_NOT_INSERTED, "A")] = torch.zeros((RANK, HIDDEN),
                                                       dtype=torch.bfloat16)
    tensors[_hf_key(_NOT_INSERTED, "B")] = torch.zeros((INTER, RANK),
                                                       dtype=torch.bfloat16)
    adapter = _write_adapter(tmp_path / "adapter", tensors)

    with pytest.raises(ValueError, match="did not LoRA-insert"):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))


def test_missing_lora_model_is_actionable(tmp_path, onnx_dir):
    (onnx_dir / "lora_model.onnx").unlink()
    adapter = _write_adapter(tmp_path / "adapter")

    with pytest.raises((FileNotFoundError, OSError)):
        process_lora_weights_and_save(str(adapter),
                                      str(tmp_path / "out"),
                                      onnx_dir=str(onnx_dir))
