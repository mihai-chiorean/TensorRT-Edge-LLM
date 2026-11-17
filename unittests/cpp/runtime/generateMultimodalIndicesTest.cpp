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
#include "runtime/llmRuntimeUtils.h"
#include <gtest/gtest.h>
#include <optional>
#include <vector>

using namespace trt_edgellm;

namespace
{

// Helper to create a CPU INT32 tensor from a flat vector with shape [batchSize, seqLen].
rt::Tensor makeCpuIds(std::vector<int32_t> const& ids, int64_t batchSize, int64_t seqLen)
{
    rt::Tensor t({batchSize, seqLen}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
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
    EXPECT_EQ(v, (std::vector<int32_t>{-1, 0, -1, 1, -1}));
}

// Image-only tokens (explicit imageTokenId)
TEST(GenerateMultimodalIndices, ImageOnlyExplicitId)
{
    int32_t constexpr kImageTok = 50;
    auto ids = makeCpuIds({10, kImageTok, 20, kImageTok, kImageTok}, 1, 5);
    auto result = rt::generateMultimodalIndices(ids, std::nullopt, kImageTok);
    auto v = toVec(result);
    EXPECT_EQ(v, (std::vector<int32_t>{-1, 0, -1, 1, 2}));
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
    EXPECT_EQ(v, (std::vector<int32_t>{0, 0, 1, 1, -1}));
}

// No multimodal tokens (all normal text)
TEST(GenerateMultimodalIndices, NoMultimodalTokens)
{
    auto ids = makeCpuIds({10, 20, 30, 40}, 1, 4);
    auto result = rt::generateMultimodalIndices(ids, std::nullopt, std::nullopt);
    auto v = toVec(result);
    EXPECT_EQ(v, (std::vector<int32_t>{-1, -1, -1, -1}));
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
    EXPECT_EQ(v, (std::vector<int32_t>{0, -1, 0, 1, 1, -1}));
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

namespace
{

int32_t constexpr kSliceImage = 77;
int32_t constexpr kSliceAudio = 99;

struct SliceIndices
{
    std::vector<int32_t> indices;
    rt::MediaRowCounts next;
};

SliceIndices sliceIndices(std::vector<int32_t> const& prompt, int32_t sliceBegin, std::vector<int32_t> const& slice,
    rt::MediaRowCounts rowBase = {})
{
    SliceIndices result;
    result.indices.assign(slice.size(), 12345);
    result.next = rt::generateSliceMultimodalIndices(prompt, sliceBegin, slice.data(),
        static_cast<int32_t>(slice.size()), kSliceImage, kSliceAudio, rowBase, result.indices.data());
    return result;
}

} // namespace

TEST(GenerateSliceMultimodalIndices, WholePromptNumbersFromZero)
{
    std::vector<int32_t> const prompt{1, kSliceImage, kSliceImage, 2, kSliceAudio, 3};
    auto const result = sliceIndices(prompt, 0, prompt);
    EXPECT_EQ(result.indices, (std::vector<int32_t>{-1, 0, 1, -1, 0, -1}));
    EXPECT_EQ(result.next.image, 2);
    EXPECT_EQ(result.next.audio, 1);
}

// A suffix after a reused prefix reads the rows that follow the prefix's media, not row zero.
TEST(GenerateSliceMultimodalIndices, SuffixSkipsRowsOfThePrefix)
{
    std::vector<int32_t> const prompt{1, kSliceImage, kSliceImage, kSliceAudio, 2, kSliceImage, kSliceAudio, 3};
    auto const result = sliceIndices(prompt, 4, {2, kSliceImage, kSliceAudio, 3});
    EXPECT_EQ(result.indices, (std::vector<int32_t>{-1, 2, 1, -1}));
    EXPECT_EQ(result.next.image, 3);
    EXPECT_EQ(result.next.audio, 2);
}

TEST(GenerateSliceMultimodalIndices, LaterRequestStartsAfterEarlierRequestsWholePrompts)
{
    // Request 0 reuses its image; its suffix is text only. Request 1's image is still the second row.
    std::vector<int32_t> const first{1, kSliceImage, 2, 3};
    std::vector<int32_t> const second{4, kSliceImage, 5};
    auto const row0 = sliceIndices(first, 2, {2, 3});
    EXPECT_EQ(row0.indices, (std::vector<int32_t>{-1, -1}));
    auto const row1 = sliceIndices(second, 0, second, row0.next);
    EXPECT_EQ(row1.indices, (std::vector<int32_t>{-1, 1, -1}));
    EXPECT_EQ(row1.next.image, 2);
}

TEST(GenerateSliceMultimodalIndices, SliceEndingBeforeThePromptEndStopsCounting)
{
    std::vector<int32_t> const prompt{1, kSliceImage, 2, kSliceImage, 3};
    auto const result = sliceIndices(prompt, 0, {1, kSliceImage, 2});
    EXPECT_EQ(result.indices, (std::vector<int32_t>{-1, 0, -1}));
    EXPECT_EQ(result.next.image, 1);
}

// A draft prefill feeds the base prefill's tokens from the second on.
TEST(GenerateSliceMultimodalIndices, DraftSliceStartsOnePastTheBaseSlice)
{
    std::vector<int32_t> const prompt{1, kSliceImage, kSliceImage, kSliceImage, 2};
    auto const base = sliceIndices(prompt, 2, {kSliceImage, kSliceImage, 2});
    EXPECT_EQ(base.indices, (std::vector<int32_t>{1, 2, -1}));
    auto const draft = sliceIndices(prompt, 3, {kSliceImage, 2});
    EXPECT_EQ(draft.indices, (std::vector<int32_t>{2, -1}));
}

// Visual-token pruning compacts a cold request's tokens and rows together, so the slice is counted as fed.
TEST(GenerateSliceMultimodalIndices, CompactedSliceCountsItsOwnTokens)
{
    std::vector<int32_t> const prompt{1, kSliceImage, kSliceImage, kSliceImage, 2};
    auto const result = sliceIndices(prompt, 1, {kSliceImage, 2, 0, 0});
    EXPECT_EQ(result.indices, (std::vector<int32_t>{0, -1, -1, -1}));
    EXPECT_EQ(result.next.image, 1);
}

TEST(GenerateSliceMultimodalIndices, UnusedPlaceholderIdsMatchNothing)
{
    std::vector<int32_t> const prompt{1, kSliceImage, kSliceAudio};
    std::vector<int32_t> indices(prompt.size(), 12345);
    auto const next = rt::generateSliceMultimodalIndices(prompt, 0, prompt.data(), 3, -1, -1, {}, indices.data());
    EXPECT_EQ(indices, (std::vector<int32_t>{-1, -1, -1}));
    EXPECT_EQ(next.image, 0);
    EXPECT_EQ(next.audio, 0);
}

TEST(GenerateSliceMultimodalIndices, SliceMustStartInsideThePrompt)
{
    std::vector<int32_t> const prompt{1, kSliceImage, 2};
    EXPECT_NO_THROW(sliceIndices(prompt, 3, {4}));
    EXPECT_THROW(sliceIndices(prompt, 4, {4}), std::runtime_error);
    EXPECT_THROW(sliceIndices(prompt, -1, {1}), std::runtime_error);
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

TEST(LLMRuntimeUtils, ClampMaxGenerateLengthForKVCapacityReservesFullDiffusionCanvas)
{
    int32_t constexpr kPromptLength = 186;
    int32_t constexpr kCanvasLength = 256;

    // Without the canvas reservation, a one-token generation would incorrectly be admitted.
    EXPECT_EQ(rt::clampMaxGenerateLengthForKVCapacity({kPromptLength}, 1, 442, 0), 1);
    // Block diffusion must reserve every canvas position before entering denoise.
    EXPECT_EQ(rt::clampMaxGenerateLengthForKVCapacity({kPromptLength}, 1, 442, kCanvasLength), 0);
    EXPECT_EQ(rt::clampMaxGenerateLengthForKVCapacity({kPromptLength}, 1, 443, kCanvasLength), 1);
}
