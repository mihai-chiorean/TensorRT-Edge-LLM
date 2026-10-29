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
"""CPU tests for the Q4_0-grid QAT checkpoint import."""

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tensorrt_edgellm.checkpoint.q4_0_grid import (convert_q4_0_checkpoint,
                                                   grid_quantize,
                                                   pack_modelopt_int4,
                                                   unpack_modelopt_int4)

G = 32


def _grid_weight(n: int, k: int, seed: int) -> torch.Tensor:
    """bf16 weight whose every group of G is code * d with an extreme at -8 or +7."""
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(-8, 8, (n, k // G, G), generator=gen)
    # Force one extreme per group so the scale rule is identifiable.
    sign = torch.randint(0, 2, (n, k // G), generator=gen)
    codes[..., 0] = torch.where(sign == 0, torch.full_like(sign, -8),
                                torch.full_like(sign, 7))
    d = torch.rand(n, k // G, 1, generator=gen) * 0.03 + 0.001
    return (codes * d).reshape(n, k).to(torch.bfloat16)


def test_grid_quantize_recovers_codes_and_scale():
    w = _grid_weight(128, 256, seed=1)
    codes, scale, stats = grid_quantize(w, G)
    assert codes.dtype == torch.int8 and scale.shape == (128, 256 // G)
    assert codes.min() >= -8 and codes.max() <= 7
    assert stats.groups_on_grid == stats.groups == 128 * 256 // G
    deq = (codes.float().reshape(128, -1, G) * scale.unsqueeze(-1)).reshape(
        128, 256)
    # bf16 storage of code*d is the only error source: one bf16 ulp at most.
    assert torch.allclose(deq, w.float(), rtol=2**-7, atol=0)
    assert stats.max_rel_err < 2**-7


def test_off_grid_weight_is_reported():
    w = torch.randn(64, 128)
    _, _, stats = grid_quantize(w, G)
    assert stats.groups_on_grid < stats.groups


def test_pack_roundtrip_matches_modelopt_layout():
    codes = torch.randint(-8, 8, (6, 64), dtype=torch.int8)
    packed = pack_modelopt_int4(codes)
    assert packed.dtype == torch.uint8 and packed.shape == (3, 64)
    # Even rows in the low nibble, odd rows in the high nibble, two's complement.
    assert int(packed[0, 0] & 0xF) == int(codes[0, 0]) & 0xF
    assert int(packed[0, 0] >> 4) == int(codes[1, 0]) & 0xF
    assert torch.equal(unpack_modelopt_int4(packed), codes)


def test_convert_writes_modelopt_prepacked_checkpoint(tmp_path: Path):
    src = tmp_path / "hf"
    src.mkdir()
    tensors = {
        "model.language_model.embed_tokens.weight":
        _grid_weight(256, 128, seed=2),
        "model.language_model.layers.0.self_attn.q_proj.weight":
        _grid_weight(128, 128, seed=3),
        "model.language_model.layers.0.mlp.down_proj.weight":
        _grid_weight(128, 192, seed=4),
        "model.language_model.layers.0.input_layernorm.weight":
        torch.ones(128, dtype=torch.bfloat16),
        "model.vision_tower.encoder.layers.0.fc.weight":
        torch.randn(32, 32, dtype=torch.bfloat16),
    }
    save_file(tensors, src / "model.safetensors", metadata={"format": "pt"})
    (src / "config.json").write_text(
        json.dumps({
            "dtype": "bfloat16",
            "text_config": {
                "dtype": "bfloat16"
            },
            "quantization_config": {
                "x": 1
            }
        }))
    (src / "tokenizer_config.json").write_text("{}")

    out = tmp_path / "out"
    report = convert_q4_0_checkpoint(str(src),
                                     str(out),
                                     group_size=G,
                                     keep_fp16=["*.mlp.down_proj"],
                                     int4_lm_head=True)
    assert report["int4"].tensors == [
        "model.language_model.layers.0.self_attn.q_proj"
    ]
    assert report["lm_head"].groups_on_grid == report["lm_head"].groups

    with safe_open(out / "model.safetensors", "pt") as sf:
        keys = set(sf.keys())
        q = sf.get_tensor(
            "model.language_model.layers.0.self_attn.q_proj.weight")
        s = sf.get_tensor(
            "model.language_model.layers.0.self_attn.q_proj.weight_scale")
        assert q.dtype == torch.uint8 and q.shape == (64, 128)
        assert s.dtype == torch.float32 and s.shape == (128, 128 // G)
        ref = tensors[
            "model.language_model.layers.0.self_attn.q_proj.weight"].float()
        deq = (unpack_modelopt_int4(q).float().reshape(128, -1, G) *
               s.unsqueeze(-1)).reshape(128, 128)
        assert torch.allclose(deq, ref, rtol=2**-7, atol=0)
        # down_proj kept fp16 without a scale; norms/vision cast to fp16; head INT4, embedding fp16.
        dp = sf.get_tensor(
            "model.language_model.layers.0.mlp.down_proj.weight")
        assert dp.dtype == torch.float16 and dp.shape == (128, 192)
        assert "model.language_model.layers.0.mlp.down_proj.weight_scale" not in keys
        assert sf.get_tensor(
            "model.language_model.layers.0.input_layernorm.weight"
        ).dtype == torch.float16
        assert sf.get_tensor("model.vision_tower.encoder.layers.0.fc.weight"
                             ).dtype == torch.float16
        assert sf.get_tensor("lm_head.weight").shape == (128, 128)
        assert sf.get_tensor("lm_head.weight_scale").shape == (256, 128 // G)
        assert sf.get_tensor(
            "model.language_model.embed_tokens.weight").dtype == torch.float16

    hq = json.loads((out / "hf_quant_config.json").read_text())["quantization"]
    assert hq["quant_algo"] == "W4A16_AWQ" and hq["group_size"] == G
    assert hq["has_zero_point"] is False and hq["pre_quant_scale"] is False
    assert hq["exclude_modules"] == [
        "model.language_model.layers.0.mlp.down_proj"
    ]
    cfg = json.loads((out / "config.json").read_text())
    assert "quantization_config" not in cfg
    assert cfg["dtype"] == "float16" and cfg["text_config"][
        "dtype"] == "float16"
    assert (out / "tokenizer_config.json").exists()


def test_convert_without_output_only_reports(tmp_path: Path):
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.up_proj.weight":
            _grid_weight(64, 64, seed=5)
        }, src / "model.safetensors")
    report = convert_q4_0_checkpoint(str(src), None, group_size=G)
    assert report["int4"].groups == 64 * 64 // G
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("linear_prefix,keep_fp16,n", [
    ("missing.", (), 64),
    ("model.language_model.", ("*", ), 64),
    ("model.language_model.", (), 32),
])
def test_convert_rejects_output_without_quantized_linears(
        tmp_path: Path, linear_prefix, keep_fp16, n):
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.up_proj.weight":
            _grid_weight(n, 64, seed=13)
        }, src / "model.safetensors")
    (src / "config.json").write_text("{}")
    out = tmp_path / "out"
    report = convert_q4_0_checkpoint(str(src),
                                     None,
                                     linear_prefix=linear_prefix,
                                     keep_fp16=keep_fp16)
    assert report["int4"].tensors == []
    with pytest.raises(ValueError, match="no linear tensor.*was quantized"):
        convert_q4_0_checkpoint(str(src),
                                str(out),
                                linear_prefix=linear_prefix,
                                keep_fp16=keep_fp16)
    assert not out.exists()


def test_int4_lm_head_requires_the_embedding(tmp_path: Path):
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.up_proj.weight":
            _grid_weight(64, 64, seed=6)
        }, src / "model.safetensors")
    with pytest.raises(ValueError, match="int4_lm_head"):
        convert_q4_0_checkpoint(str(src),
                                None,
                                group_size=G,
                                int4_lm_head=True)


def test_int4_lm_head_rejects_unaligned_vocab(tmp_path: Path):
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.embed_tokens.weight":
            _grid_weight(130, 64, seed=7)
        }, src / "model.safetensors")
    with pytest.raises(ValueError, match="64-aligned"):
        convert_q4_0_checkpoint(str(src),
                                None,
                                group_size=G,
                                int4_lm_head=True)


def test_int4_lm_head_rejects_an_untied_checkpoint(tmp_path: Path):
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.embed_tokens.weight":
            _grid_weight(128, 64, seed=8),
            "lm_head.weight":
            _grid_weight(128, 64, seed=9)
        }, src / "model.safetensors")
    with pytest.raises(ValueError, match="tied embedding"):
        convert_q4_0_checkpoint(str(src),
                                None,
                                group_size=G,
                                int4_lm_head=True)


def test_fused_expert_tensors_are_rejected(tmp_path: Path):
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.up_proj.weight":
            _grid_weight(64, 64, seed=8),
            "model.language_model.layers.0.experts.gate_up_proj":
            torch.zeros(2, 128, 64, dtype=torch.bfloat16),
        }, src / "model.safetensors")
    with pytest.raises(ValueError, match="fused expert"):
        convert_q4_0_checkpoint(str(src), str(tmp_path / "out"), group_size=G)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("k", [48, 100])
def test_grid_quantize_rejects_unaligned_k(k):
    with pytest.raises(ValueError):
        grid_quantize(torch.zeros(64, k), G)


@pytest.mark.parametrize("group_size", [16, 64, 96, 129])
def test_unsupported_group_size_is_rejected(tmp_path: Path, group_size: int):
    # 64 is aligned and divisible here, so only the group-size gate rejects it.
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.up_proj.weight":
            _grid_weight(64, 64, seed=9)
        }, src / "model.safetensors")
    (src / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="group_size"):
        convert_q4_0_checkpoint(str(src),
                                str(tmp_path / "out"),
                                group_size=group_size)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("group_size", [32, 128])
def test_supported_group_sizes_are_accepted(tmp_path: Path, group_size: int):
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.up_proj.weight":
            _grid_weight(64, 128, seed=10)
        }, src / "model.safetensors")
    report = convert_q4_0_checkpoint(str(src), None, group_size=group_size)
    assert report["int4"].tensors == [
        "model.language_model.layers.0.mlp.up_proj"
    ]


def test_sharded_checkpoint_is_rejected(tmp_path: Path):
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.up_proj.weight":
            _grid_weight(64, 64, seed=11)
        }, src / "model-00001-of-00001.safetensors")
    (src / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {
                "total_size": 0
            },
            "weight_map": {
                "model.language_model.layers.0.mlp.up_proj.weight":
                "model-00001-of-00001.safetensors"
            },
        }))
    with pytest.raises(ValueError, match="sharded"):
        convert_q4_0_checkpoint(str(src), None, group_size=G)


@pytest.mark.parametrize("group_size,exit_code", [("64", 2), ("32", None)])
def test_cli_group_size_choices(tmp_path: Path, monkeypatch, group_size: str,
                                exit_code):
    from tensorrt_edgellm.scripts import import_q4_0_qat
    src = tmp_path / "hf"
    src.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.up_proj.weight":
            _grid_weight(64, 64, seed=12)
        }, src / "model.safetensors")
    monkeypatch.setattr(sys, "argv", [
        "import_q4_0_qat", "--model_dir",
        str(src), "--group_size", group_size
    ])
    if exit_code is None:
        import_q4_0_qat.main()
    else:
        # 2 is argparse's usage error; the converter's own failures exit 1.
        with pytest.raises(SystemExit) as excinfo:
            import_q4_0_qat.main()
        assert excinfo.value.code == exit_code
