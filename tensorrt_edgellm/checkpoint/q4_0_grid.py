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
"""Re-express weights that already sit on a symmetric INT4 group grid as a W4A16 checkpoint.

Quantization-aware-trained checkpoints released as half-precision weights (for example Google's
``gemma-4-*-qat-q4_0-unquantized``) hold values of the form ``code * d`` per group of ``group_size``
input channels, with ``code`` in ``[-8, 7]`` and ``d = max(wmax / 7, -wmin / 8)``: the group's extreme
element is code -8 when it is negative and +7 when it is positive. Recovering ``code`` and ``d`` from
the weights reproduces the trained grid; re-quantizing with another group size or scale rule would
not. The output uses the ModelOpt W4A16 prepacked layout that :mod:`tensorrt_edgellm.scripts.export`
already consumes (``weight`` uint8 ``[N//2, K]`` two's-complement nibbles, ``weight_scale`` fp32
``[N, K//group_size]``, ``hf_quant_config.json`` with ``quant_algo: W4A16_AWQ``), so no AWQ smoothing
or calibration is involved.
"""

import fnmatch
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

_COPIED_FILES = ("tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "generation_config.json",
                 "processor_config.json", "preprocessor_config.json",
                 "special_tokens_map.json")


@dataclass
class GridStats:
    """Grid fit of the quantized tensors (``groups_on_grid / groups`` should be ~1.0)."""
    groups: int = 0
    groups_on_grid: int = 0
    elements: int = 0
    elements_exact_bf16: int = 0
    max_rel_err: float = 0.0
    tensors: List[str] = field(default_factory=list)

    def add(self, name: str, other: "GridStats") -> None:
        self.groups += other.groups
        self.groups_on_grid += other.groups_on_grid
        self.elements += other.elements
        self.elements_exact_bf16 += other.elements_exact_bf16
        self.max_rel_err = max(self.max_rel_err, other.max_rel_err)
        self.tensors.append(name)


def grid_quantize(
        weight: torch.Tensor,
        group_size: int) -> Tuple[torch.Tensor, torch.Tensor, GridStats]:
    """Recover INT4 codes and per-group scales from a weight on the symmetric [-8, 7] grid.

    Args:
        weight: ``[N, K]`` tensor, ``K % group_size == 0``.
        group_size: input channels per scale.

    Returns:
        ``codes`` int8 ``[N, K]`` in ``[-8, 7]``, ``scale`` fp32 ``[N, K // group_size]`` and the fit
        statistics. A group whose residual to the nearest code exceeds 0.1 step anywhere counts as off
        grid (the conversion still rounds it).
    """
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} is not a multiple of group_size={group_size}")
    g = weight.reshape(n, k // group_size, group_size).float()
    d = torch.maximum(g.max(dim=-1).values / 7.0, -g.min(dim=-1).values / 8.0)
    d = torch.where(d > 0, d, torch.ones_like(d))
    q = g / d.unsqueeze(-1)
    codes = q.round().clamp(-8, 7)
    resid = (q - codes).abs().amax(dim=-1)
    # The export stores scales in fp16, so judge the reconstruction the kernel will see.
    deq = codes * d.half().float().unsqueeze(-1)
    amax = g.abs().max()
    stats = GridStats(
        groups=n * k // group_size,
        groups_on_grid=int((resid < 0.1).sum()),
        elements=n * k,
        elements_exact_bf16=int((deq.bfloat16().float() == g).sum()),
        max_rel_err=float((deq - g).abs().max() / amax) if amax > 0 else 0.0,
    )
    return codes.reshape(n, k).to(torch.int8), d.contiguous(), stats


def pack_modelopt_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack int8 codes ``[N, K]`` in ``[-8, 7]`` into ModelOpt's uint8 ``[N//2, K]`` layout.

    Even output rows go to the low nibble, odd rows to the high nibble, as two's-complement nibbles
    (``code & 0xF``); this is what ``checkpoint.repacking._cast_modelopt_awq_prepacked`` unpacks.
    """
    if codes.shape[0] % 2 != 0:
        raise ValueError("N must be even to pack two output rows per byte")
    u = (codes.to(torch.int16) & 0xF).to(torch.uint8)
    return (u[0::2] | (u[1::2] << 4)).contiguous()


def unpack_modelopt_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_modelopt_int4` (int8 codes ``[N, K]``)."""
    n_half, k = packed.shape
    nib = torch.empty(n_half * 2, k, dtype=torch.int16)
    nib[0::2] = packed.to(torch.int16) & 0xF
    nib[1::2] = (packed.to(torch.int16) >> 4) & 0xF
    return torch.where(nib >= 8, nib - 16, nib).to(torch.int8)


def _set_fp16(cfg: dict) -> None:
    for key, value in cfg.items():
        if isinstance(value, dict):
            _set_fp16(value)
        elif key in ("dtype", "torch_dtype"):
            cfg[key] = "float16"


def _is_candidate(name: str, shape: torch.Size, linear_prefix: str,
                  skip_substrings: Iterable[str]) -> bool:
    return (name.startswith(linear_prefix) and name.endswith(".weight")
            and len(shape) == 2
            and not any(s in name for s in skip_substrings))


def convert_q4_0_checkpoint(
    model_dir: str,
    output_dir: Optional[str],
    group_size: int = 32,
    linear_prefix: str = "model.language_model.",
    skip_substrings: Iterable[str] = ("embed_tokens", "norm", "layer_scalar",
                                      "projection_scale"),
    keep_fp16: Iterable[str] = (),
    int4_lm_head: bool = False,
    embed_name: Optional[str] = None,
) -> Dict[str, GridStats]:
    """Convert a single-file safetensors checkpoint. ``output_dir=None`` only reports statistics.

    Every 2-D ``<linear_prefix>*.weight`` tensor whose name contains none of ``skip_substrings``, and
    whose N and K are multiples of 64 (the cuteDSL fragment layout) with ``K % group_size == 0``, is
    written as INT4 unless its module matches a ``keep_fp16`` glob. With ``int4_lm_head`` the tied
    embedding is additionally quantized as ``lm_head`` (the embedding itself stays fp16 for lookups).
    Remaining bf16 tensors are cast to fp16, as ModelOpt does.

    Returns the per-category statistics: ``"int4"`` for the quantized linears and, when requested,
    ``"lm_head"``.
    """
    keep_fp16 = list(keep_fp16)
    if embed_name is None:
        embed_name = linear_prefix + "embed_tokens.weight"
    report: Dict[str, GridStats] = {"int4": GridStats()}
    out: Dict[str, torch.Tensor] = {}
    excluded: List[str] = []
    checkpoint = os.path.join(model_dir, "model.safetensors")
    with safe_open(checkpoint, "pt") as sf:
        # Fused expert tensors ([E, N, K]) would stay unquantized while hf_quant_config declares
        # W4A16, which routes the exporter to the prequantized-expert path that cannot load them.
        fused = [
            name for name in sf.keys() if name.startswith(linear_prefix)
            and len(sf.get_slice(name).get_shape()) >= 3
        ]
        if fused:
            raise ValueError(
                f"fused expert / >2-D tensors under {linear_prefix} are not supported: "
                f"{', '.join(sorted(fused)[:4])}")
        for name in sorted(sf.keys()):
            tensor = sf.get_tensor(name)
            if int4_lm_head and name == embed_name:
                n, k = tensor.shape
                if n % 64 != 0 or k % 64 != 0 or k % group_size != 0:
                    raise ValueError(
                        f"--int4_lm_head needs a 64-aligned [{n}, {k}] embedding with "
                        f"K % {group_size} == 0; keep the head fp16 instead")
                codes, scale, stats = grid_quantize(tensor, group_size)
                report["lm_head"] = GridStats()
                report["lm_head"].add("lm_head", stats)
                out["lm_head.weight"] = pack_modelopt_int4(codes)
                out["lm_head.weight_scale"] = scale
            if _is_candidate(name, tensor.shape, linear_prefix,
                             skip_substrings):
                module = name[:-len(".weight")]
                n, k = tensor.shape
                alignable = n % 64 == 0 and k % 64 == 0 and k % group_size == 0
                if not alignable or any(
                        fnmatch.fnmatch(module, pat) for pat in keep_fp16):
                    if not alignable:
                        logger.warning(
                            "%s [%d, %d] is not 64-aligned; kept fp16", module,
                            n, k)
                    out[name] = tensor.to(torch.float16)
                    excluded.append(module)
                    continue
                codes, scale, stats = grid_quantize(tensor, group_size)
                report["int4"].add(module, stats)
                out[name] = pack_modelopt_int4(codes)
                out[module + ".weight_scale"] = scale
                continue
            out[name] = tensor.to(
                torch.float16) if tensor.dtype == torch.bfloat16 else tensor

    if int4_lm_head and "lm_head" not in report:
        raise ValueError(
            f"--int4_lm_head requested but {embed_name} is not in {checkpoint}"
        )
    for category, stats in report.items():
        logger.info(
            "%s: %d tensors, groups on grid %d/%d (%.6f), bf16-exact elements "
            "%d/%d, max |dequant - w| / max|w| %.3e", category,
            len(stats.tensors), stats.groups_on_grid, stats.groups,
            stats.groups_on_grid / max(stats.groups, 1),
            stats.elements_exact_bf16, stats.elements, stats.max_rel_err)
    if output_dir is None:
        return report

    os.makedirs(output_dir, exist_ok=True)
    save_file(out,
              os.path.join(output_dir, "model.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    _set_fp16(cfg)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    exclude_modules = sorted(
        set(([] if int4_lm_head else ["lm_head"]) + excluded))
    hf_quant_config = {
        "producer": {
            "name": "tensorrt_edgellm.checkpoint.q4_0_grid",
            "version": "1"
        },
        "quantization": {
            "quant_algo": "W4A16_AWQ",
            "kv_cache_quant_algo": None,
            "group_size": group_size,
            "has_zero_point": False,
            "pre_quant_scale": False,
            "exclude_modules": exclude_modules,
        },
    }
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant_config, f, indent=4)
    for fname in _COPIED_FILES:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, output_dir)
    return report
