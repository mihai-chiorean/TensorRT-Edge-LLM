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

#include "common/cudaUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/specPleStaging.h"
#include "scratchDir.h"
#include "substituteEngine.h"
#include "testUtils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kVocab{16};
constexpr int32_t kHidden{64};
constexpr int32_t kBatch{2};
constexpr int32_t kVerifyWidth{3};

//! A speculative runtime context whose token table is either device resident or host-mapped from the same INT8
//! sidecar; the PLE preprocessors are absent.
class HostMappedSpecLookupTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mDir = makeScratchDir("hostMappedSpecLookupTests");
        mSidecar = mDir / "embedding.safetensors";
        std::vector<Tensor> tensors;
        Tensor& table = tensors.emplace_back(
            Coords{kVocab, kHidden}, DeviceType::kCPU, trt_edgellm::DataType::kINT8, "embedding");
        for (int32_t idx = 0; idx < kVocab * kHidden; ++idx)
        {
            table.dataPointer<int8_t>()[idx] = static_cast<int8_t>((idx * 37) % 251 - 125);
        }
        Tensor& scales
            = tensors.emplace_back(Coords{kVocab}, DeviceType::kCPU, trt_edgellm::DataType::kFLOAT, "embedding_scale");
        for (int32_t row = 0; row < kVocab; ++row)
        {
            scales.dataPointer<float>()[row] = 0.0625F * static_cast<float>(row + 1);
        }
        ASSERT_TRUE(safetensors::saveSafetensors(mSidecar, tensors, nullptr));
        mDeviceEmbedding = loadEmbeddingTable(mSidecar, mStream);
        mHostEmbedding = mapEmbeddingTable(mSidecar);

        HybridCacheManager::Config cacheConfig{};
        cacheConfig.maxBatchSize = kBatch;
        cacheConfig.layerTypes = {HybridCacheManager::LayerType::kAttention};
        cacheConfig.kvConfig = {1, 2, 128, {{1, 64}}, trt_edgellm::DataType::kHALF};
        mCache = std::make_unique<HybridCacheManager>(cacheConfig, mStream);
        mPreprocessor = std::make_unique<EmbeddingPreprocessor>(mHostEmbedding, mDeployment.base);
        mRuntime.reset(new DecodingRuntimeContext{mDeployment, kBatch, mCheckpoint, mCheckpoint,
            {mExecutor, mTensorMap, mShared, *mCache, mIo, {}},
            {mStepPreparer, *mPreprocessor, mHostEmbedding, mUnused, mUnused, mUnused, nullptr, nullptr, nullptr},
            mTokenizer, mBias, mGuided,
            {mUnused, mUnused, mUnused, mUnused, mUnused, mUnused, mUnused, mUnused, mUnused},
            {mUnused, mUnused, mUnused, mUnused, mUnused}});
    }

    static DeploymentConfig makeDeployment()
    {
        DeploymentConfig deployment{};
        deployment.base.hiddenSize = kHidden;
        deployment.base.maxSupportedBatchSize = kBatch;
        deployment.base.maxPhysicalTokens = kBatch * kVerifyWidth;
        deployment.base.imageTokenId = -1;
        deployment.base.audioTokenId = -1;
        deployment.specConfig.emplace();
        return deployment;
    }

    void TearDown() override
    {
        mRuntime.reset();
        mPreprocessor.reset();
        mCache.reset();
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mDir);
    }

    //! The rows a device-resident table gives for `ids`, as raw FP16 bits.
    std::vector<uint16_t> deviceRows(Tensor const& ids)
    {
        Tensor expected({ids.getShape().volume(), kHidden}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        trt_edgellm::kernel::embeddingLookup(
            ids, mDeviceEmbedding.table, mDeviceEmbedding.scalesAsOptional(), expected, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        return copyDeviceToHost<uint16_t>(expected);
    }

    cudaStream_t mStream{};
    std::filesystem::path mDir;
    std::filesystem::path mSidecar;
    EmbeddingData mDeviceEmbedding;
    EmbeddingData mHostEmbedding;
    std::unique_ptr<HybridCacheManager> mCache;
    DeploymentConfig mDeployment{makeDeployment()};
    SharedResources mShared;
    PipelineIO mIo;
    substitute_engine::MockEngineExecutor mExecutor;
    TensorMap mTensorMap;
    StepPreparer mStepPreparer{mDeployment.base};
    trt_edgellm::tokenizer::Tokenizer mTokenizer;
    GuidedDecoder mGuided;
    LogitBias mBias;
    Tensor mUnused;
    std::filesystem::path mCheckpoint;
    std::unique_ptr<EmbeddingPreprocessor> mPreprocessor;
    std::unique_ptr<DecodingRuntimeContext> mRuntime;
};

} // namespace

// Given a host-mapped token table and verify ids that exist only on the device, as a draft round leaves them
// When the speculative staging feeds the verify rows
// Then one host copy of the ids yields the rows the device-resident table gives, and a graph-capture context,
// which carries no request tokens, embeds nothing
TEST_F(HostMappedSpecLookupTest, VerifyRowsEmbedFromOneHostCopyOfTheIds)
{
    std::vector<int32_t> const verifyIds{3, 15, 0, 7, -1, kVocab + 2};
    Tensor ids({kBatch, kVerifyWidth}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice(ids, verifyIds);
    mIo.queryLengths = Tensor({kBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<int32_t>(mIo.queryLengths, {kVerifyWidth, kVerifyWidth - 1});
    mIo.inputsEmbeds = Tensor({kBatch * kVerifyWidth, kHidden}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(mIo.inputsEmbeds.rawPointer(), 0xFF, mIo.inputsEmbeds.getMemoryCapacity(), mStream));

    SpecPleStaging staging(*mRuntime, kBatch * kVerifyWidth, SpecVerifyLayout::kChain);
    DecodingInferenceContext capture;
    capture.stream = mStream;
    staging.embed(*mRuntime, capture, ids, kBatch, kVerifyWidth, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    for (uint16_t const bits : copyDeviceToHost<uint16_t>(mIo.inputsEmbeds))
    {
        ASSERT_EQ(bits, 0xFFFF);
    }

    DecodingInferenceContext context;
    context.stream = mStream;
    context.tokenIds.assign(kBatch, std::vector<int32_t>{1});
    staging.embed(*mRuntime, context, ids, kBatch, kVerifyWidth, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_EQ(copyDeviceToHost<uint16_t>(mIo.inputsEmbeds), deviceRows(ids));
}

// Given a host-mapped token table and draft ids sampled on the device
// When a draft step embeds them without host ids
// Then they are copied to the host first and embed to the device-resident table's rows; a copy past the host
// buffer is refused
TEST_F(HostMappedSpecLookupTest, DeviceSampledDraftIdsEmbedLikeTheDeviceTable)
{
    std::vector<int32_t> const sampled{4, 4};
    Tensor ids({kBatch, 1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice(ids, sampled);
    mIo.inputsEmbeds = Tensor({kBatch, kHidden}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    Tensor scratch({kBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);

    decoder_utils::embedTokenIds(*mRuntime, ids, nullptr, scratch, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_EQ(std::vector<int32_t>(scratch.dataPointer<int32_t>(), scratch.dataPointer<int32_t>() + kBatch), sampled);
    EXPECT_EQ(copyDeviceToHost<uint16_t>(mIo.inputsEmbeds), deviceRows(ids));

    Tensor wide({kBatch, kVerifyWidth}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    EXPECT_THROW(decoder_utils::copyTokenIdsToHost(wide, kBatch * kVerifyWidth, scratch, mStream), std::runtime_error);
}

// Given a device-resident token table
// When a draft step embeds device ids without host ids and without a host buffer
// Then the lookup runs on the device and needs neither
TEST_F(HostMappedSpecLookupTest, DeviceTableNeedsNoHostIds)
{
    EmbeddingPreprocessor devicePreprocessor(mDeviceEmbedding, mDeployment.base);
    DecodingRuntimeContext deviceRuntime{mDeployment, kBatch, mCheckpoint, mCheckpoint,
        {mExecutor, mTensorMap, mShared, *mCache, mIo, {}},
        {mStepPreparer, devicePreprocessor, mDeviceEmbedding, mUnused, mUnused, mUnused, nullptr, nullptr, nullptr},
        mTokenizer, mBias, mGuided, {mUnused, mUnused, mUnused, mUnused, mUnused, mUnused, mUnused, mUnused, mUnused},
        {mUnused, mUnused, mUnused, mUnused, mUnused}};
    Tensor ids({kBatch, 1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<int32_t>(ids, {9, 2});
    mIo.inputsEmbeds = Tensor({kBatch, kHidden}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    Tensor noScratch;

    decoder_utils::embedTokenIds(deviceRuntime, ids, nullptr, noScratch, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_EQ(copyDeviceToHost<uint16_t>(mIo.inputsEmbeds), deviceRows(ids));
}
