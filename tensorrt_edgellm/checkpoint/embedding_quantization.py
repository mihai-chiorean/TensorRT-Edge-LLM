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
"""Embedding quantization for tensorrt_edgellm runtime sidecars."""

from __future__ import annotations

import json
import logging
import os
from typing import Tuple

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

FP8_E4M3_MAX = 448.0
INT8_SYMMETRIC_MAX = 127.0
# Rows are quantized in chunks of this many elements so the FP32 working copy
# stays bounded (64 MiB) for 262k-row vocabularies.
INT8_QUANTIZATION_CHUNK_ELEMENTS = 16 * 1024 * 1024
INT8_SIDECAR_FORMAT = "int8_symmetric_per_row"
INT8_SIDECAR_VERSION = "1"
INT8_SIDECAR_ROLES = frozenset(("embedding", ))
# Packed INT4: two columns per byte, one FP16 scale per 32 contiguous columns.
INT4_GROUP_SIZE = 32
INT4_SYMMETRIC_MAX = 7.0
INT4_SIDECAR_VERSION = "1"
INT4_SIDECAR_FORMAT = "int4_symmetric_group32_packed"
INT4_SIDECAR_ROLES = frozenset(("embedding", ))
# "absmax": d = amax / 7, codes in [-7, 7].
# "q4_0_grid": codes in [-8, 7] with a signed step per group, since Q4_0 maps
# the largest-magnitude value to -8 and so stores a negative d when that value
# is positive. Candidates d = max(max / 7, -min / 8) and
# d = -max(max / 8, -min / 7); the one with the smaller FP16 reconstruction
# error is kept (ties keep the positive step). Reproduces tables that already
# lie on a Q4_0 group grid (QAT Q4_0 checkpoints).
INT4_SCALE_MODES = ("absmax", "q4_0_grid")


def int8_sidecar_metadata(tensor_role: str) -> dict[str, str]:
    """Return the versioned safetensors metadata for an INT8 sidecar."""
    if tensor_role not in INT8_SIDECAR_ROLES:
        raise ValueError(
            f"Unsupported INT8 sidecar tensor role: {tensor_role!r}")
    return {
        "trt_edge_llm_quantization": INT8_SIDECAR_FORMAT,
        "trt_edge_llm_quantization_version": INT8_SIDECAR_VERSION,
        "trt_edge_llm_tensor_role": tensor_role,
    }


def int4_sidecar_metadata(tensor_role: str) -> dict[str, str]:
    """Return the versioned safetensors metadata for a packed INT4 sidecar."""
    if tensor_role not in INT4_SIDECAR_ROLES:
        raise ValueError(
            f"Unsupported INT4 sidecar tensor role: {tensor_role!r}")
    return {
        "trt_edge_llm_quantization": INT4_SIDECAR_FORMAT,
        "trt_edge_llm_quantization_version": INT4_SIDECAR_VERSION,
        "trt_edge_llm_tensor_role": tensor_role,
    }


def _int4_quantize_groups(
        grouped: torch.Tensor, group_step: torch.Tensor,
        code_min: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round ``grouped`` against the FP16 form of ``group_step`` (the scale the
    runtime sees); a step that is zero in FP16 becomes scale 1.0."""
    group_scales = group_step.to(torch.float16)
    group_scales = torch.where(group_scales != 0, group_scales,
                               torch.ones_like(group_scales))
    quantized = torch.round(grouped / group_scales.float().unsqueeze(2))
    return group_scales, quantized.clamp(code_min, INT4_SYMMETRIC_MAX)


def _int4_group_error(grouped: torch.Tensor, group_scales: torch.Tensor,
                      quantized: torch.Tensor) -> torch.Tensor:
    """Squared error of the runtime's FP16 decode, summed per group."""
    decoded = (quantized * group_scales.float().unsqueeze(2)).to(
        torch.float16).float()
    return (decoded - grouped).square().sum(dim=2)


def quantize_embedding_to_int4(
    embedding_weight: torch.Tensor,
    *,
    group_size: int = INT4_GROUP_SIZE,
    scale_mode: str = "absmax",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an embedding table to packed symmetric INT4 with FP16 scales.

    Every ``group_size`` contiguous columns of a row share one FP16 scale.
    ``scale_mode`` selects the scale and code range (see
    ``INT4_SCALE_MODES``); values are rounded with the FP16-rounded scale, so
    the runtime's ``fp16((nibble - 8) * scale)`` reproduces the quantizer's
    own grid and zero stays exact. Two columns are packed per byte: the even
    column in the low nibble, the odd column in the high nibble, each stored
    as ``value + 8``. Returns ``(uint8 [vocab, width // 2],
    fp16 [vocab, width // group_size])``. An all-zero group gets scale 1.0.
    """
    if scale_mode not in INT4_SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {INT4_SCALE_MODES}, "
                         f"got {scale_mode!r}")
    if embedding_weight.dim() != 2:
        raise ValueError(
            f"Embedding must be 2D, got {embedding_weight.dim()}D")
    if embedding_weight.shape[0] <= 0 or embedding_weight.shape[1] <= 0:
        raise ValueError(
            f"Embedding dimensions must be positive, got {list(embedding_weight.shape)}"
        )
    if not embedding_weight.is_floating_point():
        raise ValueError(
            f"Embedding must have a floating-point dtype, got {embedding_weight.dtype}"
        )
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(
            f"group_size must be a positive even number, got {group_size}")
    if embedding_weight.shape[1] % group_size != 0:
        raise ValueError(
            f"Embedding width {embedding_weight.shape[1]} must be divisible "
            f"by group_size {group_size}")

    vocab_size, width = embedding_weight.shape
    num_groups = width // group_size
    rows_per_chunk = max(1, INT8_QUANTIZATION_CHUNK_ELEMENTS // width)
    packed = torch.empty((vocab_size, width // 2),
                         dtype=torch.uint8,
                         device=embedding_weight.device)
    scales = torch.empty((vocab_size, num_groups),
                         dtype=torch.float16,
                         device=embedding_weight.device)

    for row_start in range(0, vocab_size, rows_per_chunk):
        row_end = min(vocab_size, row_start + rows_per_chunk)
        weight_fp32 = embedding_weight[row_start:row_end].float()
        if not torch.isfinite(weight_fp32).all():
            raise ValueError("Embedding contains non-finite values")

        grouped = weight_fp32.view(row_end - row_start, num_groups, group_size)
        if scale_mode == "absmax":
            group_scales, quantized = _int4_quantize_groups(
                grouped,
                grouped.abs().amax(dim=2) / INT4_SYMMETRIC_MAX,
                -INT4_SYMMETRIC_MAX)
        else:
            group_max = grouped.amax(dim=2)
            group_min = grouped.amin(dim=2)
            code_min = -(INT4_SYMMETRIC_MAX + 1)
            pos_scales, pos_quantized = _int4_quantize_groups(
                grouped,
                torch.maximum(group_max / INT4_SYMMETRIC_MAX,
                              -group_min / (INT4_SYMMETRIC_MAX + 1)), code_min)
            neg_scales, neg_quantized = _int4_quantize_groups(
                grouped, -torch.maximum(group_max / (INT4_SYMMETRIC_MAX + 1),
                                        -group_min / INT4_SYMMETRIC_MAX),
                code_min)
            use_negative = _int4_group_error(
                grouped, neg_scales,
                neg_quantized) < _int4_group_error(grouped, pos_scales,
                                                   pos_quantized)
            group_scales = torch.where(use_negative, neg_scales, pos_scales)
            quantized = torch.where(use_negative.unsqueeze(2), neg_quantized,
                                    pos_quantized)
        decoded = (quantized * group_scales.float().unsqueeze(2)).to(
            torch.float16)
        if not torch.isfinite(decoded).all():
            raise ValueError(
                "INT4 quantization would decode to non-finite FP16 values")
        offset = (quantized + 8).to(torch.uint8).view(row_end - row_start,
                                                      width)
        packed[row_start:row_end] = offset[:, 0::2] | (offset[:, 1::2] << 4)
        scales[row_start:row_end] = group_scales

    logger.info(
        "Quantized embedding to packed INT4: [%d, %d] -> %s, scales: %s",
        vocab_size, width, list(packed.shape), list(scales.shape))
    return packed, scales


def dequantize_int4_embedding(
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        group_size: int = INT4_GROUP_SIZE) -> torch.Tensor:
    """Reference inverse of :func:`quantize_embedding_to_int4` (FP16 result)."""
    vocab_size, half_width = packed.shape
    values = torch.empty((vocab_size, half_width * 2),
                         dtype=torch.float32,
                         device=packed.device)
    values[:, 0::2] = (packed & 0x0F).float() - 8.0
    values[:, 1::2] = (packed >> 4).float() - 8.0
    grouped = values.view(vocab_size, -1, group_size)
    return (grouped * scales.float().unsqueeze(2)).view(vocab_size,
                                                        -1).to(torch.float16)


def quantize_embedding_to_int8(
        embedding_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an embedding table to symmetric INT8 with one FP32 scale per row.

    The payload uses ``[-127, 127]`` so zero stays exact and the runtime
    dequantizes as ``fp16(int8_value * row_scale)``. An all-zero row gets
    scale 1.0 and an all-zero payload.
    """
    if embedding_weight.dim() != 2:
        raise ValueError(
            f"Embedding must be 2D, got {embedding_weight.dim()}D")
    if embedding_weight.shape[0] <= 0 or embedding_weight.shape[1] <= 0:
        raise ValueError(
            f"Embedding dimensions must be positive, got {list(embedding_weight.shape)}"
        )
    if not embedding_weight.is_floating_point():
        raise ValueError(
            f"Embedding must have a floating-point dtype, got {embedding_weight.dtype}"
        )

    vocab_size, width = embedding_weight.shape
    rows_per_chunk = max(1, INT8_QUANTIZATION_CHUNK_ELEMENTS // width)
    embedding_int8 = torch.empty_like(embedding_weight, dtype=torch.int8)
    scales = torch.empty(vocab_size,
                         dtype=torch.float32,
                         device=embedding_weight.device)

    for row_start in range(0, vocab_size, rows_per_chunk):
        row_end = min(vocab_size, row_start + rows_per_chunk)
        weight_fp32 = embedding_weight[row_start:row_end].float()
        if not torch.isfinite(weight_fp32).all():
            raise ValueError("Embedding contains non-finite values")

        row_amax = weight_fp32.abs().amax(dim=1)
        row_scales = torch.where(row_amax > 0, row_amax / INT8_SYMMETRIC_MAX,
                                 torch.ones_like(row_amax))
        quantized = torch.round(weight_fp32 / row_scales.unsqueeze(1))
        embedding_int8[row_start:row_end] = quantized.clamp(
            -INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).to(torch.int8)
        scales[row_start:row_end] = row_scales

    logger.info("Quantized embedding to INT8: [%d, %d], scales: [%d]",
                vocab_size, width, vocab_size)
    return embedding_int8, scales


def prepare_ple_export(model,
                       model_dir: str,
                       output_dir: str,
                       fp8_ple: bool = False) -> None:
    """Select the PLE pool dtype and scale before exporting its graph."""
    if model.config.ple_enabled:
        _prepare_gemma_ple(model, model_dir, output_dir, fp8_ple)
    elif model.config.ngram_layers:
        _prepare_ngram_ple(model, model_dir, output_dir, fp8_ple)
    elif fp8_ple:
        raise ValueError("--fp8-ple requires a PLE table")


def _prepare_gemma_ple(model, model_dir: str, output_dir: str,
                       fp8_ple: bool) -> None:
    """Read the packed table's dtype and scale, quantizing it first when asked."""
    from .checkpoint_utils import _gemma_ple_checkpoint_metadata

    metadata = _gemma_ple_checkpoint_metadata(model_dir, model.config)
    if fp8_ple and metadata["dtype"] != "F8_E4M3":
        _quantize_gemma_ple(model, model_dir, output_dir,
                            metadata["source"]["tensor_key"])
        metadata = _gemma_ple_checkpoint_metadata(model._ple_checkpoint_dir,
                                                  model.config)
    model._ple_pool_scale = (metadata["scale"]
                             if metadata["dtype"] == "F8_E4M3" else None)


def _quantize_gemma_ple(model, model_dir: str, output_dir: str,
                        key: str) -> None:
    """Write the packed table as per-tensor FP8 under ``<output_dir>/ple``."""
    from tensorrt_edgellm._safetensors_io import save_file

    from .loader import _build_shard_map

    with safe_open(_build_shard_map(model_dir)[key],
                   framework="pt",
                   device="cpu") as file:
        weight = file.get_tensor(key)
    packed, scale = quantize_embedding_to_fp8(weight)
    ple_dir = os.path.join(output_dir, "ple")
    os.makedirs(ple_dir, exist_ok=True)
    save_file({
        key: packed,
        key + "_scale": scale.reshape(())
    }, os.path.join(ple_dir, "model.safetensors"))
    model._ple_checkpoint_dir = ple_dir


def _prepare_ngram_ple(model, model_dir: str, output_dir: str,
                       fp8_ple: bool) -> None:
    """The n-gram scale is a checkpoint buffer; only FP8 conversion needs preparing."""
    if fp8_ple and _ngram_table_dtype(model_dir) != "F8_E4M3":
        _quantize_ngram_table(model, model_dir, output_dir)


def _ngram_table_dtype(model_dir: str) -> str:
    from . import ple_source
    from .loader import _build_shard_map

    shards = _build_shard_map(model_dir)
    key = ple_source.ngram_table_keys(shards)[1][0]
    with safe_open(shards[key], framework="pt", device="cpu") as file:
        return file.get_slice(key).get_dtype()


def _quantize_ngram_table(model, model_dir: str, output_dir: str) -> None:
    """Write the n-gram table as FP8 shards under one F32 scale, beside its hash constants."""
    from tensorrt_edgellm._safetensors_io import save_file

    from . import ple_source
    from .loader import _build_shard_map

    shards = _build_shard_map(model_dir)
    prefix, keys = ple_source.ngram_table_keys(shards)

    def read(key: str) -> torch.Tensor:
        with safe_open(shards[key], framework="pt", device="cpu") as file:
            return file.get_tensor(key)

    packed, scale = quantize_embedding_to_fp8(
        torch.cat([read(key) for key in keys]))
    tensors = {
        f"{prefix}.shard_{index}.weight": shard
        for index, shard in enumerate(
            packed.chunk(model.config.split_ngram_parts or 1))
    }
    tensors[prefix + ".weight_scale"] = scale.reshape(1)
    for key in ple_source.ngram_hash_constant_keys(model_dir).values():
        tensors[key] = read(key)
    ple_dir = os.path.join(output_dir, "ple")
    os.makedirs(ple_dir, exist_ok=True)
    save_file(tensors, os.path.join(ple_dir, "model.safetensors"))
    for name, buffer in model.named_buffers():
        if name.endswith(".ngram_embedding.weight_scale"):
            buffer.copy_(scale.reshape(1))
    model._ple_checkpoint_dir = ple_dir


def quantize_embedding_to_fp8(
    embedding_weight: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize token or PLE rows with one FP32 scale, without a full FP32 copy."""
    if embedding_weight.dim() != 2:
        raise ValueError(
            f"Embedding must be 2D, got {embedding_weight.dim()}D")

    scale = embedding_weight.abs().amax().float().clamp(
        min=1e-4) / FP8_E4M3_MAX
    packed = torch.empty_like(embedding_weight, dtype=torch.float8_e4m3fn)
    for start in range(0, embedding_weight.shape[0], 4096):
        rows = embedding_weight[start:start + 4096].float() / scale
        packed[start:start + 4096] = rows.clamp(
            -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return packed, scale.reshape(1, 1)


def write_modelopt_embedding(model_dir: str, embedding: torch.nn.Embedding,
                             embedding_scale: float, output_path: str) -> bool:
    """Write a ModelOpt-quantized input table in the runtime's lookup format.

    The Edge-LLM model does not own embedding scale buffers, so read them from
    the original checkpoint rather than from its loaded embedding parameter.
    """
    from tensorrt_edgellm._safetensors_io import save_file

    from .loader import _build_shard_map

    shards = _build_shard_map(model_dir)
    candidates = [
        key for key in shards if (key == "embed_tokens.weight" or key.endswith(
            ".embed_tokens.weight")) and key + "_scale" in shards
    ]
    if not candidates:
        return False
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one quantized token embedding, found {candidates}")
    weight_key = candidates[0]

    def read(key: str) -> torch.Tensor:
        path = shards[key]
        if not path.endswith(".safetensors"):
            raise ValueError("ModelOpt embedding weights must be safetensors")
        with safe_open(path, framework="pt", device="cpu") as file:
            return file.get_tensor(key)

    weight = read(weight_key).contiguous()
    scales = read(weight_key + "_scale").contiguous()
    vocab, hidden = embedding.num_embeddings, embedding.embedding_dim
    if weight.dtype == torch.float8_e4m3fn:
        if weight.shape != (vocab, hidden):
            raise ValueError("ModelOpt FP8 embedding has an invalid shape")
        if scales.dtype != torch.float32 or scales.numel() != 1:
            raise ValueError(
                "ModelOpt FP8 embedding requires one FP32 tensor scale")
        save_file(
            {
                "embedding": weight,
                "embedding_scale":
                scales.reshape(1, 1).float() * embedding_scale
            }, output_path)
    elif weight.dtype == torch.uint8:
        if hidden % 16 or weight.shape != (vocab, hidden // 2):
            raise ValueError(
                "ModelOpt NVFP4 embedding has invalid packed shape")
        multiplier_key = weight_key + "_scale_2"
        if multiplier_key not in shards:
            raise ValueError(
                "ModelOpt NVFP4 embedding is missing its multiplier scale")
        multiplier = read(multiplier_key)
        if scales.dtype != torch.float8_e4m3fn or scales.shape != (
                vocab, hidden // 16):
            raise ValueError(
                "ModelOpt NVFP4 embedding has invalid block scales")
        if multiplier.dtype != torch.float32 or multiplier.numel() != 1:
            raise ValueError(
                "ModelOpt NVFP4 embedding multiplier must be one FP32 value")
        descriptor = {
            "logical_shape": [vocab, hidden],
            "group_size": 16,
            "model_output_scale": embedding_scale
        }
        save_file(
            {
                "embedding": weight,
                "embedding_block_scale": scales,
                "embedding_multiplier_scale": multiplier.reshape(1)
            },
            output_path,
            metadata={
                "tensorrt_edge_llm.embedding_descriptor":
                json.dumps(descriptor)
            })
    else:
        raise ValueError(
            f"Unsupported ModelOpt embedding dtype: {weight.dtype}")
    logger.info("Wrote ModelOpt %s embedding to %s", weight.dtype, output_path)
    return True
