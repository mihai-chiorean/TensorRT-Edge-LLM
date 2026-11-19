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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Build linear target-verification token IDs: [seed, draft_0, ..., draft_{N-1}].
void launchGemma4MTPBuildVerifyTokens(int32_t const* seedTokenIds, int32_t const* draftTokenIds,
    int32_t* verifyTokenIds, int32_t batchSize, int32_t draftingStep, cudaStream_t stream);

//! Gather one target hidden-state row per batch into the assistant seed-hidden input.
void launchGemma4MTPGatherSeedHidden(void const* sourceHiddenStates, void* seedHiddenStates,
    int32_t const* sourceTokenIndices, int32_t batchSize, int64_t sourceSeqLen, int64_t hiddenSize, size_t elementBytes,
    cudaStream_t stream);

//! Store sampled assistant token IDs into the draft-token matrix at the current draft step.
void launchGemma4MTPStoreDraftToken(int32_t const* selectedTokenIds, int32_t* draftTokenIds, int32_t batchSize,
    int32_t draftingStep, int32_t step, cudaStream_t stream);

//! Fill one request-stable target-sampling uniform per verify row for sampled (non-greedy) verification.
//!
//! Row (b, i) predicts the token at absolute position nextAbsolutePositions[b] + depth(i), and draws
//! requestStableUniform(requestSeeds[b], that position, kTarget, lane 0): the same uniform the vanilla
//! decoder draws for that position. depth(i) is i for a linear chain (treeMask == nullptr) and the node's
//! ancestor count below the root for a tree ([B, verifySize, verifySize] INT8 mask, self included).
void launchGemma4MTPVerifyUniforms(uint64_t const* requestSeeds, uint64_t const* nextAbsolutePositions,
    int8_t const* treeMask, float* rowUniforms, int32_t batchSize, int32_t verifySize, cudaStream_t stream);

//! Largest top-k handled by launchGemma4MTPSampleTopKRows; callers fall back to the generic sampler above it.
inline constexpr int32_t kGemma4MTPMaxSampleTopK = 1024;

//! Sample one token per logits row from the top-k (then top-p) filtered, temperature-scaled softmax, using
//! the caller's uniform per row. Same distribution and inverse-CDF walk as topKtopPSamplingFromLogits'
//! top-k path (tokens agree with it except where logits tie at the k-th place), but the top-k set comes
//! from a radix select instead of k full passes over the vocabulary, which matters for the B * verifySize
//! rows a verification round samples. Requires 0 < topK <= min(vocabSize, kGemma4MTPMaxSampleTopK).
void launchGemma4MTPSampleTopKRows(float const* logits, int32_t rows, int32_t vocabSize, int32_t topK,
    float temperature, float topP, float const* rowUniforms, int32_t* sampledTokenIds, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
