/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/tensor.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "runtime/llmRuntimeUtils.h"
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

using namespace trt_edgellm;

namespace
{

// Helper to create a CPU INT32 tensor from a flat vector with shape [batchSize, seqLen].
rt::Tensor makeCpuIds(std::vector<int32_t> const& ids, int64_t batchSize, int64_t seqLen)
{
    rt::Tensor t({batchSize, seqLen}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
    std::memcpy(t.rawPointer(), ids.data(), ids.size() * sizeof(int32_t));
    return t;
}

// Helper to read the result tensor into a flat vector.
std::vector<int32_t> toVec(rt::Tensor const& t)
{
    auto const shape = t.getShape();
    int64_t const n = shape[0] * shape[1];
    std::vector<int32_t> v(n);
    std::memcpy(v.data(), t.dataPointer<int32_t>(), n * sizeof(int32_t));
    return v;
}

} // namespace

// Audio-only tokens (audioTokenId set, no image tokens)
TEST(GenerateMultimodalIndices, AudioOnly)
{
    int32_t constexpr kAudioTok = 99;
    // batch=1, seqLen=5: two audio tokens at positions 1 and 3
    auto ids = makeCpuIds({10, kAudioTok, 20, kAudioTok, 30}, 1, 5);
    auto result = rt::generateMultimodalIndices(ids, kAudioTok, std::nullopt);
    auto v = toVec(result);
    EXPECT_EQ(v, (std::vector<int32_t>{0, 0, 0, 1, 0}));
}

// Image-only tokens (explicit imageTokenId)
TEST(GenerateMultimodalIndices, ImageOnlyExplicitId)
{
    int32_t constexpr kImageTok = 50;
    auto ids = makeCpuIds({10, kImageTok, 20, kImageTok, kImageTok}, 1, 5);
    auto result = rt::generateMultimodalIndices(ids, std::nullopt, kImageTok);
    auto v = toVec(result);
    EXPECT_EQ(v, (std::vector<int32_t>{0, 0, 0, 1, 2}));
}

// Mixed audio + image tokens
TEST(GenerateMultimodalIndices, MixedAudioImage)
{
    int32_t constexpr kAudioTok = 99;
    int32_t constexpr kImageTok = 50;
    auto ids = makeCpuIds({kAudioTok, kImageTok, kAudioTok, kImageTok, 10}, 1, 5);
    auto result = rt::generateMultimodalIndices(ids, kAudioTok, kImageTok);
    auto v = toVec(result);
    // audio indices: 0, 1; image indices: 0, 1
    EXPECT_EQ(v, (std::vector<int32_t>{0, 0, 1, 1, 0}));
}

// No multimodal tokens (all normal text)
TEST(GenerateMultimodalIndices, NoMultimodalTokens)
{
    auto ids = makeCpuIds({10, 20, 30, 40}, 1, 4);
    auto result = rt::generateMultimodalIndices(ids, std::nullopt, std::nullopt);
    auto v = toVec(result);
    EXPECT_EQ(v, (std::vector<int32_t>{0, 0, 0, 0}));
}

// Multi-batch with global indexing across batches
TEST(GenerateMultimodalIndices, MultiBatchGlobalIndexing)
{
    int32_t constexpr kAudioTok = 99;
    int32_t constexpr kImageTok = 50;
    // batch=2, seqLen=3
    // batch 0: [99, 10, 50]  -> audio idx 0, text, image idx 0
    // batch 1: [99, 50, 10]  -> audio idx 1, image idx 1, text
    auto ids = makeCpuIds({kAudioTok, 10, kImageTok, kAudioTok, kImageTok, 10}, 2, 3);
    auto result = rt::generateMultimodalIndices(ids, kAudioTok, kImageTok);
    auto v = toVec(result);
    EXPECT_EQ(v, (std::vector<int32_t>{0, 0, 0, 1, 1, 0}));
}

// Contiguous image runs get one block id each; audio tokens stay causal (-1).
TEST(GenerateVisionBlockIds, ImageRunsGrouped)
{
    int32_t constexpr kImageTok = 50;
    int32_t constexpr kAudioTok = 52;
    auto ids = makeCpuIds({10, kImageTok, kImageTok, 20, kImageTok, kAudioTok, kImageTok, 30}, 1, 8);
    auto result = rt::generateVisionBlockIds(ids, kImageTok);
    auto v = toVec(result);
    // Adjacent image placeholders form one vision run, matching HF's
    // block grouping. Audio remains -1 (causal).
    EXPECT_EQ(v, (std::vector<int32_t>{-1, 0, 0, -1, 1, -1, 2, -1}));
}

// Block numbering restarts at 0 for each batch entry.
TEST(GenerateVisionBlockIds, BlockIdsRestartPerBatch)
{
    int32_t constexpr kImageTok = 50;
    auto ids = makeCpuIds({kImageTok, 10, kImageTok, 20, kImageTok, kImageTok}, 2, 3);
    auto result = rt::generateVisionBlockIds(ids, kImageTok);
    auto v = toVec(result);
    EXPECT_EQ(v, (std::vector<int32_t>{0, -1, 1, -1, 0, 0}));
}

// Row bases restart each row's counters at the encoder rows its placeholders occupy.
TEST(GenerateMultimodalIndices, RowBasesOffsetSuffixRows)
{
    int32_t constexpr kAudioTok = 99;
    int32_t constexpr kImageTok = 50;
    // Row 0 is a suffix whose reused prefix already consumed 3 image rows; row 1 begins after
    // row 0's full input (5 image rows) plus one reused audio row.
    auto ids = makeCpuIds({kImageTok, kImageTok, 10, kAudioTok, kImageTok, kAudioTok}, 2, 3);
    rt::MultimodalRowBases const bases{{3, 5}, {0, 1}};
    auto result = rt::generateMultimodalIndices(ids, kAudioTok, kImageTok, &bases);
    auto v = toVec(result);
    EXPECT_EQ(v, (std::vector<int32_t>{3, 4, 0, 1, 5, 2}));
}

TEST(ComputeMultimodalRowBases, SuffixInsideMediaRunSkipsReusedRows)
{
    int32_t constexpr kImageTok = 50;
    int32_t constexpr kAudioTok = 99;
    // [sys, sys, img x6, txt] with a reused prefix of 5 tokens: 3 image rows lie inside the prefix.
    std::vector<std::vector<int32_t>> const inputs{
        {1, 2, kImageTok, kImageTok, kImageTok, kImageTok, kImageTok, kImageTok, 3},
        {kAudioTok, kAudioTok, kImageTok, 4},
    };
    auto const bases = rt::computeMultimodalRowBases(inputs, {5, 1}, kImageTok, kAudioTok);
    EXPECT_EQ(bases.image, (std::vector<int32_t>{3, 6}));
    EXPECT_EQ(bases.audio, (std::vector<int32_t>{0, 1}));
}

TEST(ComputeMultimodalRowBases, NoReuseMatchesGlobalCounters)
{
    int32_t constexpr kImageTok = 50;
    std::vector<std::vector<int32_t>> const inputs{{kImageTok, 1, kImageTok}, {2, kImageTok}, {3}};
    auto const bases = rt::computeMultimodalRowBases(inputs, {0, 0, 0}, kImageTok, std::nullopt);
    EXPECT_EQ(bases.image, (std::vector<int32_t>{0, 2, 3}));
    EXPECT_EQ(bases.audio, (std::vector<int32_t>{0, 0, 0}));
}

TEST(ComputeMultimodalRowBases, RejectsPrefixBeyondInput)
{
    std::vector<std::vector<int32_t>> const inputs{{1, 2, 3}};
    EXPECT_THROW(rt::computeMultimodalRowBases(inputs, {4}, 50, std::nullopt), std::exception);
}

TEST(SuffixContainsToken, ReportsPlaceholdersAtOrAfterPrefillStart)
{
    int32_t constexpr kImageTok = 50;
    std::vector<std::vector<int32_t>> const inputs{{1, kImageTok, kImageTok, 2}, {3, 4}};
    EXPECT_TRUE(rt::suffixContainsToken(inputs, nullptr, kImageTok));
    std::vector<int32_t> const insideRun{2, 0};
    EXPECT_TRUE(rt::suffixContainsToken(inputs, &insideRun, kImageTok));
    std::vector<int32_t> const pastRun{3, 0};
    EXPECT_FALSE(rt::suffixContainsToken(inputs, &pastRun, kImageTok));
    std::vector<int32_t> const atEnd{4, 2};
    EXPECT_FALSE(rt::suffixContainsToken(inputs, &atEnd, kImageTok));
    EXPECT_FALSE(rt::suffixContainsToken(inputs, nullptr, 99));
}

// The device kernel agrees with the host reference, with and without row bases.
TEST(GenerateMultimodalIndices, DeviceKernelMatchesHostReference)
{
    int32_t constexpr kAudioTok = 99;
    int32_t constexpr kImageTok = 50;
    std::vector<int32_t> const tokens{kImageTok, kImageTok, 10, kAudioTok, kImageTok, kAudioTok, kImageTok, 11};
    auto hostIds = makeCpuIds(tokens, 2, 4);
    rt::MultimodalRowBases const bases{{7, 9}, {2, 4}};

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    rt::Tensor deviceIds({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor deviceIndices({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor deviceBases({2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    std::vector<int32_t> const flatBases{bases.image[0], bases.audio[0], bases.image[1], bases.audio[1]};
    ASSERT_EQ(
        cudaMemcpy(deviceIds.rawPointer(), tokens.data(), tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(deviceBases.rawPointer(), flatBases.data(), flatBases.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice),
        cudaSuccess);

    auto readBack = [&]() {
        std::vector<int32_t> out(tokens.size());
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(
            cudaMemcpy(out.data(), deviceIndices.rawPointer(), out.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);
        return out;
    };

    kernel::generateMultimodalIndices(deviceIds, deviceIndices, kImageTok, kAudioTok, stream);
    EXPECT_EQ(readBack(), toVec(rt::generateMultimodalIndices(hostIds, kAudioTok, kImageTok)));

    kernel::generateMultimodalIndices(
        deviceIds, deviceIndices, kImageTok, kAudioTok, stream, rt::OptionalInputTensor{std::ref(deviceBases)});
    EXPECT_EQ(readBack(), toVec(rt::generateMultimodalIndices(hostIds, kAudioTok, kImageTok, &bases)));
    EXPECT_EQ(readBack(), (std::vector<int32_t>{7, 8, 0, 2, 9, 4, 10, 0}));

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(LLMRuntimeUtils, ClampMaxGenerateLengthForKVCapacitySingleBatch)
{
    EXPECT_EQ(rt::clampMaxGenerateLengthForKVCapacity({100}, 80, 256, 0), 80);
    EXPECT_EQ(rt::clampMaxGenerateLengthForKVCapacity({220}, 80, 256, 0), 36);
    EXPECT_EQ(rt::clampMaxGenerateLengthForKVCapacity({246}, 80, 256, 10), 0);
}

TEST(LLMRuntimeUtils, ClampMaxGenerateLengthForKVCapacityMixedBatch)
{
    EXPECT_EQ(rt::clampMaxGenerateLengthForKVCapacity({100, 180, 150}, 90, 256, 20), 56);
}
