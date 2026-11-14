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

"""Host emulation of the dynamic-LoRA epilogue contract (see ``int4_lora.py``).

Reproduces the kernel's documented reduction tree in NumPy so the kernel output
can be checked bit for bit.  ``fma.rn.f32`` is emulated as an exact float64
product-sum rounded to float32: the product of two FP16 values is exact in
float64 and the sum is exact unless the operands are more than about 29 binary
orders of magnitude apart, which these tests do not produce.
"""

from __future__ import annotations

import numpy as np


def _fma32(a, b, c):
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)


def lora_t16(x16: np.ndarray, a16: np.ndarray, chunk_k: int) -> np.ndarray:
    """``t16 [M, r]`` for ``x16 [M, K]`` and ``a16 [K, r]`` (both float16), with the decode shrink
    kernel's tree (``cpp/kernels/loraKernels/loraShrinkKernels.h``):
    - per chunk and 8-column half: slot q in [0, 64) chains k = chunk0 + q + 64 i; each warp
      xor-butterflies its 16 slots (distances 8, 4, 2, 1); P = ((W0 + W1) + W2) + W3;
    - chunks: Q[g] sums P[g], P[g + 8], ... in order, T sums Q[0..7] in order."""
    M, K = x16.shape
    r = a16.shape[1]
    x = x16.astype(np.float32)
    a = np.zeros((K, 16), np.float32)
    a[:, :r] = a16.astype(np.float32)
    num_chunks = (K + chunk_k - 1) // chunk_k
    parts = []
    for ch in range(num_chunks):
        c0 = ch * chunk_k
        S = np.zeros((64, M, 16), np.float32)
        for i in range(chunk_k // 64):
            if c0 + 64 * i >= K:
                break
            ks = c0 + np.arange(64) + 64 * i
            S = _fma32(x[:, ks].T[:, :, None], a[ks][:, None, :], S)
        W = S.reshape(4, 16, M, 16)
        for d in (8, 4, 2, 1):
            W = (W + W[:, np.arange(16) ^ d]).astype(np.float32)
        W = W[:, 0]
        parts.append((((W[0] + W[1]).astype(np.float32) + W[2]).astype(np.float32) + W[3]).astype(np.float32))
    groups = []
    for g in range(min(8, num_chunks)):
        q = parts[g]
        for j in range(g + 8, num_chunks, 8):
            q = (q + parts[j]).astype(np.float32)
        groups.append(q)
    T = groups[0]
    for q in groups[1:]:
        T = (T + q).astype(np.float32)
    return T[:, :r].astype(np.float16)


def lora_out(base16: np.ndarray, t16: np.ndarray, b16: np.ndarray) -> np.ndarray:
    """``out [M, N]`` from the kernel's own ``base16``, ``t16 [M, r]`` and ``b16 [r, N]``."""
    M, N = base16.shape
    r = t16.shape[1]
    t = t16.astype(np.float32)
    b = b16.astype(np.float32)
    D = np.zeros((M, N), np.float32)
    for j in range(r):
        D = _fma32(t[:, j:j + 1], b[j:j + 1, :], D)
    d16 = D.astype(np.float16)
    return (base16.astype(np.float32) + d16.astype(np.float32)).astype(np.float16)
