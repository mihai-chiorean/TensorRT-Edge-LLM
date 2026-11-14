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

"""Device helpers for the dynamic-LoRA expand epilogue of the INT4 GEMM/GEMV kernels.

The shrink ``t16 = fp16(x @ A)`` runs before the GEMM, in ``LoraShrinkPlugin``
(``cpp/kernels/loraKernels/loraShrinkKernels.cuh``), which writes ``t16`` as
``[M, 16]`` fp16 with zeros past the rank. The INT4 kernel then completes

    d16 = fp16(D),  D = fma.rn.f32 chain over rank j in order of t16[m, j] * B[j, n]
    out = fp16(float(base16) + float(d16))

where ``base16`` is the kernel's unchanged fp16 output. ``t16`` and ``B`` are
ready before the kernel starts, so the CTA copies its rows of ``t16`` and its B
tile to shared memory right after issuing the main-loop prologue, and the
epilogue reads no global memory beyond the base kernel's own.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float16, Float32, Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import dsl_user_op

LORA_MAX_RANK = 16


@dsl_user_op
def fma_f32(a, b, c, *, loc=None, ip=None):
    """``fma.rn.f32 a * b + c``; explicit so the rounding is not left to contraction."""
    f32 = Float32.mlir_type
    r = llvm.inline_asm(
        f32,
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
         Float32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(r)


@dsl_user_op
def _ld_v4_b32(ptr, *, loc=None, ip=None):
    """One 128-bit global load (16-byte aligned) as a vector of four 32-bit words."""
    return cute.arch.load(ptr, ir.VectorType.get([4], Int32.mlir_type), loc=loc, ip=ip)


@dsl_user_op
def _st_v4_b32(ptr, value, *, loc=None, ip=None):
    cute.arch.store(ptr, value, ss="cta", loc=loc, ip=ip)


@dsl_user_op
def _zero_v4_b32(*, loc=None, ip=None):
    return vector.broadcast(
        ir.VectorType.get([4], Int32.mlir_type), Int32(0).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )


def _ptr2(tensor, i, j):
    return cute.domain_offset((i, j), tensor).iterator


@cute.jit
def lora_fill_smem(
    tidx: Int32,
    num_threads: cutlass.Constexpr,
    mT: cute.Tensor,
    mLB: cute.Tensor,
    sT: cute.Tensor,
    sB: cute.Tensor,
    row0: Int32,
    rows: cutlass.Constexpr,
    col_groups,
    rank: Int32,
):
    """Copy ``t16[row0 : row0 + rows, 0:16]`` to ``sT`` [rows, 16] (row-major), and the B columns of
    ``col_groups`` (``(start, count)`` pairs, starts and counts multiples of 8, starts
    dynamic) side by side into ``sB`` [16, total count] (row-major). Rows of t past M, rows of B past
    the rank and columns past N are zero. 128-bit loads; no barrier (the caller's next CTA
    barrier orders these stores before any read)."""
    M = mT.shape[0]
    N = mLB.shape[1]
    # t: two 16-byte pieces per row.
    for it in cutlass.range_constexpr((rows * 2 + num_threads - 1) // num_threads):
        p = tidx + Int32(it * num_threads)
        if p < Int32(rows * 2):
            r = p // Int32(2)
            half = p % Int32(2)
            v = _zero_v4_b32()
            if row0 + r < M:
                v = _ld_v4_b32(_ptr2(mT, row0 + r, half * Int32(8)))
            _st_v4_b32(_ptr2(sT, r, half * Int32(8)), v)
    base_col = 0
    for start, count in col_groups:
        pieces = count // 8
        for it in cutlass.range_constexpr((LORA_MAX_RANK * pieces + num_threads - 1) // num_threads):
            p = tidx + Int32(it * num_threads)
            if p < Int32(LORA_MAX_RANK * pieces):
                j = p // Int32(pieces)
                c8 = (p % Int32(pieces)) * Int32(8)
                v = _zero_v4_b32()
                if j < rank and start + c8 < N:
                    v = _ld_v4_b32(_ptr2(mLB, j, start + c8))
                _st_v4_b32(_ptr2(sB, j, Int32(base_col) + c8), v)
        base_col += count


@cute.jit
def lora_delta8_smem(sT: cute.Tensor, m, sB: cute.Tensor, c0, rank: Int32, d16: cute.Tensor):
    """``d16[e]`` for tile row ``m`` and shared-B columns ``c0 + e``, e < 8, in rank order."""
    acc = cute.make_rmem_tensor(cute.make_layout(8), Float32)
    acc.fill(0.0)
    for j in cutlass.range_constexpr(LORA_MAX_RANK):
        if Int32(j) < rank:
            t = Float32(sT[m, j])
            for e in cutlass.range_constexpr(8):
                acc[e] = fma_f32(t, Float32(sB[j, c0 + Int32(e)]), acc[e])
    for e in cutlass.range_constexpr(8):
        d16[e] = acc[e].to(Float16)


@cute.jit
def lora_delta1_smem(sT: cute.Tensor, m, sB: cute.Tensor, c, rank: Int32):
    """``d16`` for tile row ``m`` and shared-B column ``c``."""
    acc = Float32(0.0)
    for j in cutlass.range_constexpr(LORA_MAX_RANK):
        if Int32(j) < rank:
            acc = fma_f32(Float32(sT[m, j]), Float32(sB[j, c]), acc)
    return acc.to(Float16)


@cute.jit
def lora_add(base16, d16):
    return (Float32(base16) + Float32(d16)).to(Float16)
