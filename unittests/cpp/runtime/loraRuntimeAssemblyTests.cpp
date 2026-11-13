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

// Assembling LLMInferenceRuntime against an engine that exposes LoRA bindings
// and adapter weights on disk: the end-to-end load path (sharedResources ->
// LoRAManager::loadWeights -> initializeEngineBindings) that the LoRAManager
// unit tests cannot reach.
//
// The rank ceiling comes from the deployment config's `max_lora_rank`, so an
// adapter whose rank exceeds it must fail runtime construction, and one that
// fits must assemble cleanly.

#include "substituteEngine.h"

#include "common/checkMacros.h"
#include "runtime/llmInferenceRuntime.h"

#include <string>
#include <unordered_map>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using namespace substitute_engine;

namespace
{

//! The deployment's `max_lora_rank`, and the rank the valid adapter is written with.
constexpr int64_t kMaxLoraRank{8};
//! The rank of the adapter that must be refused at load.
constexpr int64_t kOverRank{16};
//! Hidden dimension shared by the engine bindings and the adapter tensors.
constexpr int64_t kHiddenSize{64};

//! A vanilla deployment with LoRA enabled: `max_lora_rank` drives
//! SharedResources' LoRAManager construction and every rank check behind it.
class LoraRuntimeAssemblyTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        Json config = makeTinyVanillaConfig();
        config["builder_config"]["max_lora_rank"] = kMaxLoraRank;
        return writeModelDir("loraRuntimeAssemblyTests", config);
    }

    //! Write a two-tensor adapter (lora_A_layer_0 + lora_B_layer_0) in
    //! safetensors form, the way a deployed adapter file arrives.
    static void writeAdapter(std::filesystem::path const& path, int64_t rank, cudaStream_t stream)
    {
        std::vector<uint16_t> ones(static_cast<size_t>(rank * kHiddenSize), 0x3800);
        rt::Tensor a({kHiddenSize, rank}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "lora_A_layer_0");
        rt::Tensor b({rank, kHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "lora_B_layer_0");
        CUDA_CHECK(cudaMemcpyAsync(
            a.rawPointer(), ones.data(), ones.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            b.rawPointer(), ones.data(), ones.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<rt::Tensor> tensors;
        tensors.push_back(std::move(a));
        tensors.push_back(std::move(b));
        ASSERT_TRUE(rt::safetensors::saveSafetensors(path, tensors, stream));
    }

    //! An engine exposing one LoRA pair shaped for the deployment's max rank.
    //! Every non-LoRA query keeps the NiceMock defaults from makeEngine().
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeLoraEngine()
    {
        using ::testing::_;
        using ::testing::Return;
        using ::testing::StrEq;

        nvinfer1::Dims aShape;
        aShape.nbDims = 2;
        aShape.d[0] = kHiddenSize;
        aShape.d[1] = kMaxLoraRank;
        nvinfer1::Dims bShape;
        bShape.nbDims = 2;
        bShape.d[0] = kMaxLoraRank;
        bShape.d[1] = kHiddenSize;

        auto engine = makeEngine();
        ON_CALL(*engine, getNumIOTensors()).WillByDefault(Return(2));
        ON_CALL(*engine, getIOTensorName(0)).WillByDefault(Return("lora_A_layer_0"));
        ON_CALL(*engine, getIOTensorName(1)).WillByDefault(Return("lora_B_layer_0"));
        ON_CALL(*engine, getProfileShape(StrEq("lora_A_layer_0"), _, _)).WillByDefault(Return(aShape));
        ON_CALL(*engine, getProfileShape(StrEq("lora_B_layer_0"), _, _)).WillByDefault(Return(bShape));
        return engine;
    }

    //! Assemble the artifacts over a LoRA-enabled engine and adapter directory.
    rt::ModelArtifacts makeArtifacts(std::unique_ptr<rt::EngineExecutor> engine)
    {
        return makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    }

    //! The runtime under test, with the model directory's adapter file
    //! registered under the name the load path reads.
    rt::LLMInferenceRuntime makeRuntime(rt::ModelArtifacts artifacts, std::string const& adapterName)
    {
        std::unordered_map<std::string, std::string> const loraWeightsMap{
            {adapterName, (mModelDir / (adapterName + ".safetensors")).string()}};
        return rt::LLMInferenceRuntime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"",
            loraWeightsMap, std::nullopt, mStream};
    }
};

// Given a deployment whose config declares max_lora_rank 8
// When the runtime is assembled over an adapter whose rank exceeds it
// Then construction throws: the adapter can never bind to the engine
TEST_F(LoraRuntimeAssemblyTest, OverRankAdapterFailsRuntimeConstruction)
{
    std::filesystem::path const adapterPath = mModelDir / "adapter_over_rank.safetensors";
    writeAdapter(adapterPath, kOverRank, mStream);

    auto engine = makeLoraEngine();
    EXPECT_THROW(makeRuntime(makeArtifacts(std::move(engine)), "adapter_over_rank"), std::runtime_error);
}

// Given the same deployment
// When the runtime is assembled over an adapter that fits the max rank
// Then it comes up as a working single-engine deployment
TEST_F(LoraRuntimeAssemblyTest, RankFittingAdapterAssembles)
{
    std::filesystem::path const adapterPath = mModelDir / "adapter_rank8.safetensors";
    writeAdapter(adapterPath, kMaxLoraRank, mStream);

    auto runtime = makeRuntime(makeArtifacts(makeLoraEngine()), "adapter_rank8");

    EXPECT_FALSE(runtime.hasDraftModel());
    EXPECT_STREQ(runtime.getSpeculativeDecodingStrategyName(), "vanilla");
}

//! System-prompt warm-up for an adapter the runtime never loaded is refused, not thrown, so a caller's bad
//! adapter name cannot take down the process (the request path fails the same way).
TEST_F(LoraRuntimeAssemblyTest, SystemPromptWarmupForUnknownAdapterIsRefused)
{
    std::filesystem::path const adapterPath = mModelDir / "adapter_rank8.safetensors";
    writeAdapter(adapterPath, kMaxLoraRank, mStream);
    auto runtime = makeRuntime(makeArtifacts(makeLoraEngine()), "adapter_rank8");

    bool saved = true;
    EXPECT_NO_THROW(saved = runtime.genAndSaveSystemPromptKVCache("You are a robot.", "missing", mStream));
    EXPECT_FALSE(saved);
}

//! The same deployment built without LoRA inputs.
class RankZeroLoraRuntimeAssemblyTest : public LoraRuntimeAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("loraRuntimeAssemblyTests", makeTinyVanillaConfig());
    }
};

// Given a deployment whose config declares max_lora_rank 0
// When the runtime is assembled with an adapter
// Then construction throws instead of serving the base model under the adapter's name
TEST_F(RankZeroLoraRuntimeAssemblyTest, AdapterOnRankZeroEngineFailsRuntimeConstruction)
{
    writeAdapter(mModelDir / "adapter_rank4.safetensors", 4, mStream);

    EXPECT_THROW(makeRuntime(makeArtifacts(makeEngine()), "adapter_rank4"), std::runtime_error);
}

} // namespace
