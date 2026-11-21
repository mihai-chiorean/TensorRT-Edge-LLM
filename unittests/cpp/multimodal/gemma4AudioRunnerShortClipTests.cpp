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

//! Engine-backed tests for Gemma4AudioRunner with clips shorter than the encoder engine's profile minimum.
//!
//! The encoder is a stand-in built in the test with the real encoder's interface and a profile of 16 to 64 frames.
//! It turns 4 frames of 128 mel bins into one row of 64 values:
//!   last_hidden_state[r, c] = mean over the 4 frames and 2 bins of (r, c) of (input_features[t, d] + 8 * mask[t])
//!                             + 0.5 * (number of frames the mask marks valid) + (largest input_features value)
//! The last two terms make every row depend on every input frame and mask entry, padding included, and keep a row
//! made only of padding away from 0.

#include "common/checkMacros.h"
#include "common/safetensorsUtils.h"
#include "common/trtInterop.h"
#include "multimodal/gemma4/gemma4AudioRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "scratchDir.h"

#include <NvInfer.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using ::testing::Each;
using ::testing::ElementsAre;

namespace
{
constexpr int64_t kMEL_BINS = 128;
constexpr int64_t kHIDDEN_DIM = 64;
constexpr int64_t kBINS_PER_VALUE = kMEL_BINS / kHIDDEN_DIM;
constexpr int64_t kSUBSAMPLING_FACTOR = 4;
constexpr int64_t kPROFILE_MIN_FRAMES = 16;
constexpr int64_t kPROFILE_OPT_FRAMES = 32;
constexpr int64_t kPROFILE_MAX_FRAMES = 64;
constexpr int32_t kAUDIO_TOKEN_ID = 5;
constexpr int32_t kBEGIN_AUDIO_TOKEN_ID = 6;
constexpr int32_t kEND_AUDIO_TOKEN_ID = 7;
constexpr int64_t kMEL_PATTERN_PERIOD = 13;
constexpr float kMEL_PATTERN_STEP = 0.5F;
constexpr float kVALID_FRAME_WEIGHT = 8.0F;
constexpr float kVALID_COUNT_WEIGHT = 0.5F;
constexpr float kSENTINEL = 777.0F;

//! Mel value of one frame and bin of a clip. Multiples of 0.5 no larger than clipBase + 6, so every value the
//! stand-in encoder computes from the clips used here is exact in FP16.
float melValue(float clipBase, int64_t frame, int64_t bin)
{
    return clipBase + kMEL_PATTERN_STEP * static_cast<float>((frame + 3 * bin) % kMEL_PATTERN_PERIOD);
}

//! Rows the stand-in encoder owes a clip of `frames` frames: ceil(frames / 4) rows of kHIDDEN_DIM values.
std::vector<float> expectedRows(float clipBase, int64_t frames)
{
    int64_t const rows = (frames + kSUBSAMPLING_FACTOR - 1) / kSUBSAMPLING_FACTOR;
    float const largestMel = clipBase + kMEL_PATTERN_STEP * static_cast<float>(kMEL_PATTERN_PERIOD - 1);
    float const clipTerms = kVALID_COUNT_WEIGHT * static_cast<float>(frames) + largestMel;
    std::vector<float> result;
    result.reserve(static_cast<size_t>(rows * kHIDDEN_DIM));
    for (int64_t row = 0; row < rows; ++row)
    {
        int64_t const lastFrame = std::min((row + 1) * kSUBSAMPLING_FACTOR, frames);
        for (int64_t column = 0; column < kHIDDEN_DIM; ++column)
        {
            float sum = 0.0F;
            for (int64_t frame = row * kSUBSAMPLING_FACTOR; frame < lastFrame; ++frame)
            {
                for (int64_t bin = column * kBINS_PER_VALUE; bin < (column + 1) * kBINS_PER_VALUE; ++bin)
                {
                    sum += melValue(clipBase, frame, bin) + kVALID_FRAME_WEIGHT;
                }
            }
            result.push_back(sum / static_cast<float>(kSUBSAMPLING_FACTOR * kBINS_PER_VALUE) + clipTerms);
        }
    }
    return result;
}

class Gemma4AudioRunnerShortClipTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
        mDir = makeScratchDir("gemma4AudioRunnerShortClipTests");
        writeEngine();
        std::ofstream(mDir / "config.json")
            << R"({"audio_config": {"num_mel_bins": 128}, "audio_token_id": 5, "boa_token_id": 6, "eoa_token_id": 7})";

        mRunner = std::make_unique<Gemma4AudioRunner>(mDir.string(), mStream);
        mContextMemory = Tensor({std::max<int64_t>(mRunner->getRequiredContextMemorySize(), 1)}, DeviceType::kGPU,
            trt_edgellm::DataType::kUINT8, "Gemma4AudioRunnerShortClipTest::mContextMemory");
        ASSERT_TRUE(mRunner->setContextMemory(mContextMemory));
        fillEmbeddingCapacity(kSENTINEL);
    }

    void TearDown() override
    {
        mRunner.reset();
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mDir);
    }

    void writeEngine()
    {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(TrtLogger::instance()));
        ELLM_CHECK(builder != nullptr, "Test builder creation failed");
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        ELLM_CHECK(network != nullptr && config != nullptr, "Test network creation failed");

        auto* features
            = network->addInput("input_features", nvinfer1::DataType::kHALF, nvinfer1::Dims3{1, -1, kMEL_BINS});
        auto* mask = network->addInput("input_features_mask", nvinfer1::DataType::kBOOL, nvinfer1::Dims2{1, -1});
        ELLM_CHECK(features != nullptr && mask != nullptr, "Test input creation failed");

        auto* features4d = network->addShuffle(*features);
        features4d->setReshapeDimensions(nvinfer1::Dims4{1, 1, -1, kMEL_BINS});
        auto* maskHalf = network->addCast(*mask, nvinfer1::DataType::kHALF);
        auto* mask4d = network->addShuffle(*maskHalf->getOutput(0));
        mask4d->setReshapeDimensions(nvinfer1::Dims4{1, 1, -1, 1});

        __half const frameWeightValue = __float2half(kVALID_FRAME_WEIGHT);
        __half const countWeightValue = __float2half(kVALID_COUNT_WEIGHT);
        nvinfer1::Dims4 const kSCALAR_DIMS{1, 1, 1, 1};
        auto* frameWeight = network->addConstant(kSCALAR_DIMS, {nvinfer1::DataType::kHALF, &frameWeightValue, 1});
        auto* countWeight = network->addConstant(kSCALAR_DIMS, {nvinfer1::DataType::kHALF, &countWeightValue, 1});

        auto* weightedMask = network->addElementWise(
            *mask4d->getOutput(0), *frameWeight->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
        auto* frames = network->addElementWise(
            *features4d->getOutput(0), *weightedMask->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        nvinfer1::Dims2 const kPOOL_WINDOW{kSUBSAMPLING_FACTOR, kBINS_PER_VALUE};
        auto* pooled = network->addPoolingNd(*frames->getOutput(0), nvinfer1::PoolingType::kAVERAGE, kPOOL_WINDOW);
        pooled->setStrideNd(kPOOL_WINDOW);

        uint32_t const kFRAME_AXIS = 1U << 2;
        uint32_t const kBIN_AXIS = 1U << 3;
        auto* validCount
            = network->addReduce(*mask4d->getOutput(0), nvinfer1::ReduceOperation::kSUM, kFRAME_AXIS, true);
        auto* weightedCount = network->addElementWise(
            *validCount->getOutput(0), *countWeight->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
        auto* largest = network->addReduce(
            *features4d->getOutput(0), nvinfer1::ReduceOperation::kMAX, kFRAME_AXIS | kBIN_AXIS, true);

        auto* withCount = network->addElementWise(
            *pooled->getOutput(0), *weightedCount->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        auto* withLargest = network->addElementWise(
            *withCount->getOutput(0), *largest->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        auto* rows = network->addShuffle(*withLargest->getOutput(0));
        rows->setReshapeDimensions(nvinfer1::Dims3{1, -1, kHIDDEN_DIM});
        rows->getOutput(0)->setName("last_hidden_state");
        network->markOutput(*rows->getOutput(0));

        auto* profile = builder->createOptimizationProfile();
        for (auto const& [selector, frameCount] : {std::pair{nvinfer1::OptProfileSelector::kMIN, kPROFILE_MIN_FRAMES},
                 std::pair{nvinfer1::OptProfileSelector::kOPT, kPROFILE_OPT_FRAMES},
                 std::pair{nvinfer1::OptProfileSelector::kMAX, kPROFILE_MAX_FRAMES}})
        {
            profile->setDimensions("input_features", selector, nvinfer1::Dims3{1, frameCount, kMEL_BINS});
            profile->setDimensions("input_features_mask", selector, nvinfer1::Dims2{1, frameCount});
        }
        config->addOptimizationProfile(profile);
        config->setBuilderOptimizationLevel(0);

        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        ELLM_CHECK(plan != nullptr, "Test engine build failed");
        std::ofstream file(mDir / "audio_encoder.engine", std::ios::binary);
        file.write(static_cast<char const*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    }

    //! A precomputed-mel clip of `frames` frames whose values are melValue(clipBase, frame, bin).
    audioUtils::AudioData makeClip(float clipBase, int64_t frames)
    {
        std::vector<Tensor> tensors;
        Tensor& mel
            = tensors.emplace_back(Coords{1, frames, kMEL_BINS}, DeviceType::kCPU, trt_edgellm::DataType::kHALF, "mel");
        auto* const values = mel.dataPointer<__half>();
        for (int64_t frame = 0; frame < frames; ++frame)
        {
            for (int64_t bin = 0; bin < kMEL_BINS; ++bin)
            {
                values[frame * kMEL_BINS + bin] = __float2half(melValue(clipBase, frame, bin));
            }
        }
        auto const path = mDir / ("clip" + std::to_string(mClipCount++) + ".safetensors");
        ELLM_CHECK(safetensors::saveSafetensors(path, tensors, mStream), "Test mel file failed to save");

        audioUtils::AudioData clip;
        clip.melSpectrogramPath = path.string();
        clip.melSpectrogramFormat = "safetensors";
        return clip;
    }

    //! Overwrite the whole allocation behind the output embedding, including rows past its current shape.
    void fillEmbeddingCapacity(float value)
    {
        Tensor& embedding = mRunner->getOutputEmbedding();
        std::vector<__half> const host(
            static_cast<size_t>(embedding.getMemoryCapacity()) / sizeof(__half), __float2half(value));
        CUDA_CHECK(
            cudaMemcpy(embedding.rawPointer(), host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice));
    }

    //! Run one request holding `clips` through the runner, with one audio placeholder per clip. The token ids the
    //! runner expands the placeholders to are left in mInputIds.
    bool encode(std::vector<audioUtils::AudioData> clips)
    {
        LLMGenerationRequest request;
        mInputIds = {std::vector<int32_t>(clips.size(), kAUDIO_TOKEN_ID)};
        request.requests.emplace_back().audioBuffers = std::move(clips);
        bool const ok = mRunner->preprocess(request, mInputIds, nullptr, std::nullopt, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        return ok;
    }

    std::vector<int64_t> embeddingShape()
    {
        Coords const shape = mRunner->getOutputEmbedding().getShape();
        std::vector<int64_t> dims;
        for (int32_t i = 0; i < shape.getNumDims(); ++i)
        {
            dims.push_back(shape[i]);
        }
        return dims;
    }

    //! `count` rows of the output embedding's allocation starting at `firstRow`, whatever its current shape.
    std::vector<float> readRows(int64_t firstRow, int64_t count)
    {
        std::vector<__half> host(static_cast<size_t>(count * kHIDDEN_DIM));
        auto const* const base = static_cast<__half const*>(mRunner->getOutputEmbedding().rawPointer());
        CUDA_CHECK(cudaMemcpy(
            host.data(), base + firstRow * kHIDDEN_DIM, host.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        std::vector<float> result;
        result.reserve(host.size());
        for (__half const value : host)
        {
            result.push_back(__half2float(value));
        }
        return result;
    }

    cudaStream_t mStream{};
    std::filesystem::path mDir;
    std::unique_ptr<Gemma4AudioRunner> mRunner;
    Tensor mContextMemory{};
    std::vector<std::vector<int32_t>> mInputIds;
    int32_t mClipCount{0};
};
} // namespace

// Given an encoder engine whose profile starts at 16 frames, and an output buffer filled with a sentinel
// When a request holds one 5-frame clip
// Then the request succeeds, its placeholder becomes 2 audio tokens between the begin and end tokens, the output
//      has 2 rows holding the clip's own encoding, and the rows after them still hold the sentinel
TEST_F(Gemma4AudioRunnerShortClipTest, ShortClipKeepsOnlyItsOwnRows)
{
    ASSERT_TRUE(encode({makeClip(10.0F, 5)}));

    EXPECT_THAT(mRunner->getLastMediaTokenLengths(), ElementsAre(2));
    EXPECT_THAT(mInputIds,
        ElementsAre(ElementsAre(kBEGIN_AUDIO_TOKEN_ID, kAUDIO_TOKEN_ID, kAUDIO_TOKEN_ID, kEND_AUDIO_TOKEN_ID)));
    EXPECT_THAT(embeddingShape(), ElementsAre(2, kHIDDEN_DIM));
    EXPECT_EQ(readRows(0, 2), expectedRows(10.0F, 5));
    EXPECT_THAT(readRows(2, 4), Each(kSENTINEL));
}

// Given an encoder engine whose profile starts at 16 frames, and an output buffer filled with a sentinel
// When a request holds a 40-frame clip followed by a 5-frame clip
// Then the clips get 10 and 2 audio tokens, the output has 12 rows holding each clip's own encoding in request
//      order, and the rows after the last clip still hold the sentinel
TEST_F(Gemma4AudioRunnerShortClipTest, ShortClipLastLeavesTheRowsAfterItUntouched)
{
    ASSERT_TRUE(encode({makeClip(20.0F, 40), makeClip(10.0F, 5)}));

    EXPECT_THAT(mRunner->getLastMediaTokenLengths(), ElementsAre(10, 2));
    EXPECT_THAT(embeddingShape(), ElementsAre(12, kHIDDEN_DIM));
    EXPECT_EQ(readRows(0, 10), expectedRows(20.0F, 40));
    EXPECT_EQ(readRows(10, 2), expectedRows(10.0F, 5));
    EXPECT_THAT(readRows(12, 4), Each(kSENTINEL));
}

// Given an encoder engine whose profile starts at 16 frames
// When a request holds a 5-frame clip followed by a 40-frame clip
// Then the clips get 2 and 10 audio tokens, and the output has 12 rows holding each clip's own encoding in
//      request order
TEST_F(Gemma4AudioRunnerShortClipTest, ShortClipFirstDoesNotDisplaceTheNextClip)
{
    ASSERT_TRUE(encode({makeClip(10.0F, 5), makeClip(20.0F, 40)}));

    EXPECT_THAT(mRunner->getLastMediaTokenLengths(), ElementsAre(2, 10));
    EXPECT_THAT(embeddingShape(), ElementsAre(12, kHIDDEN_DIM));
    EXPECT_EQ(readRows(0, 2), expectedRows(10.0F, 5));
    EXPECT_EQ(readRows(2, 10), expectedRows(20.0F, 40));
}

// Given an encoder engine whose profile starts at 16 frames, and a runner that has already encoded a 40-frame clip
// When a request holds one 17-frame clip, longer than the profile minimum and not a multiple of 4
// Then the clip gets 5 audio tokens, and the output has 5 rows holding the clip's own encoding, with 17 frames
//      marked valid and nothing left of the earlier clip
TEST_F(Gemma4AudioRunnerShortClipTest, ClipAboveTheMinimumIsEncodedWithItsOwnMask)
{
    ASSERT_TRUE(encode({makeClip(20.0F, 40)}));
    ASSERT_TRUE(encode({makeClip(30.0F, 17)}));

    EXPECT_THAT(mRunner->getLastMediaTokenLengths(), ElementsAre(5));
    EXPECT_THAT(embeddingShape(), ElementsAre(5, kHIDDEN_DIM));
    EXPECT_EQ(readRows(0, 5), expectedRows(30.0F, 17));
}
