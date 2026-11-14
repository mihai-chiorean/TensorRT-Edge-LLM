/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

//! Dynamic-LoRA shrink for the fused INT4 expand epilogue: t16[i] = fp16(x @ A[i]) for up to
//! kLoraShrinkMaxTargets projections that read the same activation x [M, K]. Each t16 is
//! [M, 16] fp16 with zeros past that target's rank; a target bound at rank 0 is skipped.
//!
//! Summation tree (fixed by M, K and the rank; independent of the grid and of timing):
//!   decode path (M <= 8), chunk c (CK = 256), row m, slot q in [0, 64), half h:
//!     S[q, h] = fma.rn chain over i of x[m, k] * A[k, 8h + e],  k = c * CK + q + 64 i
//!     W[w, h] = xor-butterfly over the 16 slots q = 16w .. 16w + 15 (lane distances 16, 8, 4, 2)
//!     P[c]    = ((W[0] + W[1]) + W[2]) + W[3]
//!   T = Q[0] + ... + Q[7] in order,  Q[g] = P[g] + P[g + 8] + ... in order;  t16 = fp16(T).
//! The prefill path (M >= kLoraShrinkPrefillMinM) is loraShrinkPrefillBody: tensor-core tiles, deterministic, its own
//! summation order.

#include <cuda_fp16.h>
#ifndef __CUDACC_RTC__
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#endif

namespace trt_edgellm
{
namespace kernels
{

constexpr int kLoraShrinkMaxTargets = 4;
constexpr int kLoraShrinkMaxRank = 16;
constexpr int kLoraShrinkThreads = 128;
//! Rows of x per block; also the granularity of the per-(target, row block) counters.
constexpr int kLoraShrinkRowsPerBlock = 16;
constexpr int kLoraShrinkGroups = 8;
//! M from which the tensor-core prefill shrink runs.
constexpr int kLoraShrinkPrefillMinM = 9;
//! Tensor-core path K chunk per block. While M is small (MTP verify, short prompts) short chunks
//! put enough blocks on the GPU, but not so many that the last block's ordered chunk sum
//! dominates (K = 10240); long prompts use long chunks.
constexpr int kLoraShrinkPrefillSmallMaxM = 64;
constexpr int kLoraShrinkPrefillShortChunkMaxK = 4096;
constexpr int kLoraShrinkPrefillChunk = 512;

//! K per decode-path block (M < kLoraShrinkPrefillMinM). Shape-only, so part of the tree. 256 was the
//! fastest in-engine on the AGX (A arrives from DRAM; more blocks keep more loads in flight).
__host__ __device__ inline int loraShrinkChunkK(int /*m*/)
{
    return 256;
}

//! K per tensor-core-path block (M >= kLoraShrinkPrefillMinM); a multiple of 64. Shape-only.
__host__ __device__ inline int loraShrinkPrefillChunkK(int m, int k)
{
    if (m > kLoraShrinkPrefillSmallMaxM)
    {
        return kLoraShrinkPrefillChunk;
    }
    return k <= kLoraShrinkPrefillShortChunkMaxK ? 128 : 256;
}

struct LoraShrinkTarget
{
    __half const* a; //!< [K, rank] row-major
    __half* t;       //!< [M, 16]
    int rank;        //!< 0 skips the target
};

struct LoraShrinkParams
{
    __half const* x; //!< [M, K] row-major
    int m;
    int k;
    int numTargets;
    LoraShrinkTarget target[kLoraShrinkMaxTargets];
    float* partial;   //!< [numTargets, numChunks, M, 16]
    int32_t* counter; //!< [numTargets, rowBlocks], zero between launches (the last block resets)
};

#ifndef __CUDACC_RTC__
//! Number of K chunks for an M x K shrink.
int32_t loraShrinkNumChunks(int64_t m, int32_t k);

//! Workspace bytes for the partials of `numTargets` targets at up to `maxM` rows and any K up to
//! `maxK` (a multiple of 64).
size_t loraShrinkWorkspaceBytes(int64_t maxM, int32_t maxK, int32_t numTargets);

//! Counter words (int32, zero between launches) for `numTargets` targets at up to `maxM` rows.
size_t loraShrinkCounterWords(int64_t maxM, int32_t numTargets);

//! Launches the shrink for the targets in `p` (rank-0 targets are skipped; nothing launches when
//! all are). `p.partial` and `p.counter` must be sized by the helpers above. Returns 0 on success.
int32_t launchLoraShrink(LoraShrinkParams const& p, cudaStream_t stream);
#endif

} // namespace kernels
} // namespace trt_edgellm
