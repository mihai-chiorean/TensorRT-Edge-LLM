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

#include "gemma4MTPRuntimeKernels.h"

#include "common/checkMacros.h"
#include "runtime/decoding/requestStableRng.h"

#include <cfloat>
#include <cmath>
#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kMaxSampleTopKDevice = kGemma4MTPMaxSampleTopK;

__global__ void buildVerifyTokensKernel(int32_t const* __restrict__ seedTokenIds,
    int32_t const* __restrict__ draftTokenIds, int32_t* __restrict__ verifyTokenIds, int32_t draftingStep,
    int32_t verifySize, int32_t totalElements)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements)
    {
        return;
    }

    int32_t const batchIdx = idx / verifySize;
    int32_t const posIdx = idx % verifySize;
    if (posIdx == 0)
    {
        verifyTokenIds[batchIdx * verifySize] = seedTokenIds[batchIdx];
    }
    else
    {
        verifyTokenIds[batchIdx * verifySize + posIdx] = draftTokenIds[batchIdx * draftingStep + posIdx - 1];
    }
}

__global__ void gatherSeedHiddenKernel(uint8_t const* __restrict__ sourceHiddenStates,
    uint8_t* __restrict__ seedHiddenStates, int32_t const* __restrict__ sourceTokenIndices, int64_t sourceSeqLen,
    int64_t rowBytes, int64_t totalBytes)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= totalBytes)
    {
        return;
    }

    int32_t const batchIdx = static_cast<int32_t>(idx / rowBytes);
    int64_t const byteIdx = idx % rowBytes;
    int64_t const sourceTokenIdx = sourceTokenIndices[batchIdx];
    int64_t const srcOffset = (static_cast<int64_t>(batchIdx) * sourceSeqLen + sourceTokenIdx) * rowBytes + byteIdx;
    seedHiddenStates[idx] = sourceHiddenStates[srcOffset];
}

__global__ void storeDraftTokenKernel(int32_t const* __restrict__ selectedTokenIds, int32_t* __restrict__ draftTokenIds,
    int32_t draftingStep, int32_t step, int32_t batchSize)
{
    int32_t const batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize)
    {
        return;
    }

    draftTokenIds[batchIdx * draftingStep + step] = selectedTokenIds[batchIdx];
}

__global__ void verifyUniformsKernel(uint64_t const* __restrict__ requestSeeds,
    uint64_t const* __restrict__ nextAbsolutePositions, int8_t const* __restrict__ treeMask,
    float* __restrict__ rowUniforms, int32_t verifySize, int32_t totalRows)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalRows)
    {
        return;
    }
    int32_t const batchIdx = idx / verifySize;
    int32_t const rowIdx = idx % verifySize;
    int32_t depth = rowIdx;
    if (treeMask != nullptr)
    {
        // Depth below the root: ancestors-plus-self of this node minus those of the root. Padding rows
        // (empty mask rows) clamp to depth 0; their samples are never reached by the accept walk.
        int8_t const* batchMask = treeMask + static_cast<int64_t>(batchIdx) * verifySize * verifySize;
        int32_t rowCount = 0;
        int32_t rootCount = 0;
        for (int32_t j = 0; j < verifySize; ++j)
        {
            rowCount += batchMask[rowIdx * verifySize + j] == 1 ? 1 : 0;
            rootCount += batchMask[j] == 1 ? 1 : 0;
        }
        depth = max(rowCount - rootCount, 0);
    }
    uint64_t const position = nextAbsolutePositions[batchIdx] + static_cast<uint64_t>(depth);
    rowUniforms[idx] = rt::requestStableUniform(requestSeeds[batchIdx], position, rt::SpecRandomPurpose::kTarget, 0);
}

constexpr int32_t kSampleThreads = 1024;

//! Map a float to an unsigned key with the same ordering (larger float -> larger key).
__device__ __forceinline__ uint32_t orderedFloatKey(float value)
{
    uint32_t const bits = __float_as_uint(value);
    return (bits & 0x80000000U) != 0U ? ~bits : (bits | 0x80000000U);
}

//! One block per row. A 4 x 8-bit radix select finds the k-th largest logit, the k candidates are compacted
//! and sorted descending in shared memory, and thread 0 runs the top-k sampler's softmax / top-p /
//! inverse-CDF walk on them.
__global__ void __launch_bounds__(kSampleThreads)
    sampleTopKRowsKernel(float const* __restrict__ logits, int32_t vocabSize, int32_t topK, float invTemp, float topP,
        float const* __restrict__ rowUniforms, int32_t* __restrict__ sampledTokenIds)
{
    __shared__ uint32_t hist[256];
    __shared__ uint32_t sPrefix;
    __shared__ uint32_t sMask;
    __shared__ int32_t sRemaining;
    __shared__ int32_t sAbove;
    __shared__ int32_t sTies;
    __shared__ float candValues[kMaxSampleTopKDevice];
    __shared__ int32_t candIndices[kMaxSampleTopKDevice];

    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const lane = tid & 31;
    float const* rowLogits = logits + static_cast<int64_t>(blockIdx.x) * vocabSize;

    if (tid == 0)
    {
        sPrefix = 0U;
        sMask = 0U;
        sRemaining = topK;
        sAbove = 0;
        sTies = 0;
    }

    for (int32_t shift = 24; shift >= 0; shift -= 8)
    {
        if (tid < 256)
        {
            hist[tid] = 0U;
        }
        __syncthreads();
        uint32_t const prefix = sPrefix;
        uint32_t const mask = sMask;
        for (int32_t base = 0; base < vocabSize; base += kSampleThreads)
        {
            int32_t const v = base + tid;
            bool active = false;
            uint32_t digit = 0U;
            if (v < vocabSize)
            {
                uint32_t const key = orderedFloatKey(rowLogits[v]);
                active = (key & mask) == prefix;
                digit = (key >> shift) & 0xFFU;
            }
            // Logits crowd into a few high digits; aggregate equal digits per warp before the shared atomic.
            uint32_t const activeMask = __ballot_sync(0xFFFFFFFFU, active);
            if (active)
            {
                uint32_t const peers = __match_any_sync(activeMask, digit);
                if (lane == __ffs(peers) - 1)
                {
                    atomicAdd(&hist[digit], static_cast<uint32_t>(__popc(peers)));
                }
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            uint32_t const remaining = static_cast<uint32_t>(sRemaining);
            uint32_t above = 0U;
            int32_t digit = 255;
            for (; digit > 0; --digit)
            {
                if (above + hist[digit] >= remaining)
                {
                    break;
                }
                above += hist[digit];
            }
            sRemaining = static_cast<int32_t>(remaining - above);
            sPrefix = prefix | (static_cast<uint32_t>(digit) << shift);
            sMask = mask | (0xFFU << shift);
        }
        __syncthreads();
    }

    // sPrefix is now the k-th largest key: every larger key is in, plus sRemaining keys equal to it.
    uint32_t const threshold = sPrefix;
    int32_t const tiesNeeded = sRemaining;
    int32_t const numAbove = topK - tiesNeeded;
    for (int32_t v = tid; v < vocabSize; v += kSampleThreads)
    {
        float const value = rowLogits[v];
        uint32_t const key = orderedFloatKey(value);
        if (key > threshold)
        {
            int32_t const slot = atomicAdd(&sAbove, 1);
            candValues[slot] = value;
            candIndices[slot] = v;
        }
        else if (key == threshold)
        {
            int32_t const tie = atomicAdd(&sTies, 1);
            if (tie < tiesNeeded)
            {
                candValues[numAbove + tie] = value;
                candIndices[numAbove + tie] = v;
            }
        }
    }
    int32_t sortSize = 1;
    while (sortSize < topK)
    {
        sortSize <<= 1;
    }
    for (int32_t i = topK + tid; i < sortSize; i += kSampleThreads)
    {
        // -inf with the largest id sorts after every real candidate, including masked (-inf) logits.
        candValues[i] = -INFINITY;
        candIndices[i] = INT32_MAX;
    }
    __syncthreads();

    // Bitonic sort: value descending, then token id ascending.
    for (int32_t size = 2; size <= sortSize; size <<= 1)
    {
        for (int32_t stride = size >> 1; stride > 0; stride >>= 1)
        {
            for (int32_t i = tid; i < sortSize; i += kSampleThreads)
            {
                int32_t const partner = i ^ stride;
                if (partner > i)
                {
                    bool const descending = (i & size) == 0;
                    float const a = candValues[i];
                    float const b = candValues[partner];
                    int32_t const ia = candIndices[i];
                    int32_t const ib = candIndices[partner];
                    bool const aFirst = a > b || (a == b && ia < ib);
                    if (aFirst != descending)
                    {
                        candValues[i] = b;
                        candValues[partner] = a;
                        candIndices[i] = ib;
                        candIndices[partner] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0)
    {
        // Mirrors topKStage1/topKStage2Sampling: scale by 1/T, exp relative to the max, optional nucleus
        // cut on the sorted weights, then walk the weights with uniform * retained mass.
        // __fmul_rn keeps the product rounded (no FMA contraction), as the generic sampler stores it.
        float const maxScaled = __fmul_rn(candValues[0], invTemp);
        float sum = 0.0F;
        for (int32_t i = 0; i < topK; ++i)
        {
            float const weight = __expf(__fmul_rn(candValues[i], invTemp) - maxScaled);
            candValues[i] = weight;
            sum += weight;
        }
        float retainedSum = sum;
        if (topP < 1.0F)
        {
            retainedSum = 0.0F;
            float const nucleusThreshold = topP * sum;
            for (int32_t i = 0; i < topK; ++i)
            {
                retainedSum += candValues[i];
                if (retainedSum >= nucleusThreshold)
                {
                    break;
                }
            }
        }
        float const uniform = fminf(fmaxf(rowUniforms[blockIdx.x], 0.0F), 0.99999994F);
        float randNum = uniform * retainedSum;
        // If rounding leaves mass after the walk, take the last candidate with nonzero weight (the generic
        // sampler takes the k-th, which can be a masked token).
        int32_t selected = candIndices[0];
        for (int32_t i = 0; i < topK; ++i)
        {
            if (candValues[i] > 0.0F)
            {
                selected = candIndices[i];
            }
            randNum -= candValues[i];
            if (randNum <= 0.0F)
            {
                selected = candIndices[i];
                break;
            }
        }
        sampledTokenIds[blockIdx.x] = selected;
    }
}

} // namespace

void launchGemma4MTPSampleTopKRows(float const* logits, int32_t rows, int32_t vocabSize, int32_t topK,
    float temperature, float topP, float const* rowUniforms, int32_t* sampledTokenIds, cudaStream_t stream)
{
    check::check(topK > 0 && topK <= vocabSize && topK <= kGemma4MTPMaxSampleTopK,
        "Gemma4 MTP top-k row sampling requires 0 < topK <= min(vocabSize, kGemma4MTPMaxSampleTopK).");
    if (rows <= 0)
    {
        return;
    }
    float const invTemp = temperature < 1e-3F ? 1000.0F : 1.0F / temperature;
    sampleTopKRowsKernel<<<rows, kSampleThreads, 0, stream>>>(
        logits, vocabSize, topK, invTemp, topP, rowUniforms, sampledTokenIds);
    CUDA_CHECK(cudaGetLastError());
}

void launchGemma4MTPVerifyUniforms(uint64_t const* requestSeeds, uint64_t const* nextAbsolutePositions,
    int8_t const* treeMask, float* rowUniforms, int32_t batchSize, int32_t verifySize, cudaStream_t stream)
{
    int32_t const totalRows = batchSize * verifySize;
    if (totalRows <= 0)
    {
        return;
    }
    constexpr int32_t kThreads = 128;
    int32_t const blocks = (totalRows + kThreads - 1) / kThreads;
    verifyUniformsKernel<<<blocks, kThreads, 0, stream>>>(
        requestSeeds, nextAbsolutePositions, treeMask, rowUniforms, verifySize, totalRows);
    CUDA_CHECK(cudaGetLastError());
}

void launchGemma4MTPBuildVerifyTokens(int32_t const* seedTokenIds, int32_t const* draftTokenIds,
    int32_t* verifyTokenIds, int32_t batchSize, int32_t draftingStep, cudaStream_t stream)
{
    if (batchSize == 0 || draftingStep < 0)
    {
        return;
    }

    int32_t const verifySize = draftingStep + 1;
    int32_t const totalElements = batchSize * verifySize;
    constexpr int32_t kThreads = 256;
    int32_t const blocks = (totalElements + kThreads - 1) / kThreads;
    buildVerifyTokensKernel<<<blocks, kThreads, 0, stream>>>(
        seedTokenIds, draftTokenIds, verifyTokenIds, draftingStep, verifySize, totalElements);
    CUDA_CHECK(cudaGetLastError());
}

void launchGemma4MTPGatherSeedHidden(void const* sourceHiddenStates, void* seedHiddenStates,
    int32_t const* sourceTokenIndices, int32_t batchSize, int64_t sourceSeqLen, int64_t hiddenSize, size_t elementBytes,
    cudaStream_t stream)
{
    if (batchSize == 0 || hiddenSize == 0 || elementBytes == 0)
    {
        return;
    }

    int64_t const rowBytes = hiddenSize * static_cast<int64_t>(elementBytes);
    int64_t const totalBytes = static_cast<int64_t>(batchSize) * rowBytes;
    constexpr int32_t kThreads = 256;
    int64_t const blockCount = (totalBytes + kThreads - 1) / kThreads;
    check::check(blockCount <= std::numeric_limits<int32_t>::max(),
        "Gemma4 MTP seed-hidden gather exceeds CUDA grid dimension limit.");
    int32_t const blocks = static_cast<int32_t>(blockCount);
    gatherSeedHiddenKernel<<<blocks, kThreads, 0, stream>>>(static_cast<uint8_t const*>(sourceHiddenStates),
        static_cast<uint8_t*>(seedHiddenStates), sourceTokenIndices, sourceSeqLen, rowBytes, totalBytes);
    CUDA_CHECK(cudaGetLastError());
}

void launchGemma4MTPStoreDraftToken(int32_t const* selectedTokenIds, int32_t* draftTokenIds, int32_t batchSize,
    int32_t draftingStep, int32_t step, cudaStream_t stream)
{
    if (batchSize == 0)
    {
        return;
    }

    check::check(step >= 0 && step < draftingStep, "Gemma4 MTP draft-token store step is out of range.");
    constexpr int32_t kThreads = 256;
    int32_t const blocks = (batchSize + kThreads - 1) / kThreads;
    storeDraftTokenKernel<<<blocks, kThreads, 0, stream>>>(
        selectedTokenIds, draftTokenIds, draftingStep, step, batchSize);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
