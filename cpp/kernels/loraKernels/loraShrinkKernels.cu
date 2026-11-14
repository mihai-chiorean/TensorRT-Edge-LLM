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

#include "loraShrinkKernels.cuh"

#include <algorithm>

namespace trt_edgellm
{
namespace kernels
{

int32_t loraShrinkNumChunks(int64_t m, int32_t k)
{
    int const mi = static_cast<int>(m);
    int32_t const ck = m >= kLoraShrinkPrefillMinM ? loraShrinkPrefillChunkK(mi, k) : loraShrinkChunkK(mi);
    return (k + ck - 1) / ck;
}

size_t loraShrinkWorkspaceBytes(int64_t maxM, int32_t maxK, int32_t numTargets)
{
    // Partials are rows x chunks. The chunk changes with M (and, on the tensor-core path, with
    // K), so the count is not monotonic in either: take the top of every M bucket at the top
    // of every K bucket.
    int64_t best = 0;
    for (int32_t const k : {std::min<int32_t>(maxK, kLoraShrinkPrefillShortChunkMaxK), maxK})
    {
        for (int64_t const m : {std::min<int64_t>(maxM, kLoraShrinkPrefillMinM - 1),
                 std::min<int64_t>(maxM, kLoraShrinkPrefillSmallMaxM), maxM})
        {
            best = std::max(best, m * loraShrinkNumChunks(m, k));
        }
    }
    return static_cast<size_t>(best * numTargets * kLoraShrinkMaxRank) * sizeof(float);
}

size_t loraShrinkCounterWords(int64_t maxM, int32_t numTargets)
{
    // Decode: one counter per (target, 16-row block); prefill: one per 64-row block.
    return static_cast<size_t>(numTargets * ((maxM + kLoraShrinkRowsPerBlock - 1) / kLoraShrinkRowsPerBlock));
}

int32_t launchLoraShrink(LoraShrinkParams const& p, cudaStream_t stream)
{
    bool any = false;
    for (int32_t i = 0; i < p.numTargets; ++i)
    {
        any = any || p.target[i].rank > 0;
    }
    if (!any || p.m <= 0)
    {
        return 0;
    }
    if (p.k % 64 != 0 || p.numTargets > kLoraShrinkMaxTargets)
    {
        return -1;
    }
    if (p.m >= kLoraShrinkPrefillMinM)
    {
        dim3 const grid(loraShrinkNumChunks(p.m, p.k), (p.m + 63) / 64);
        loraShrinkPrefillKernel<<<grid, kLoraShrinkThreads, 0, stream>>>(p);
    }
    else
    {
        dim3 const grid(loraShrinkNumChunks(p.m, p.k), p.numTargets,
            (p.m + kLoraShrinkRowsPerBlock - 1) / kLoraShrinkRowsPerBlock);
        loraShrinkKernel<<<grid, kLoraShrinkThreads, 0, stream>>>(p);
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

} // namespace kernels
} // namespace trt_edgellm
