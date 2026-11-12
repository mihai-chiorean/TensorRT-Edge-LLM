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

#include "runtime/state/loraManager.h"

#include "common/checkMacros.h"
#include "substituteEngine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using namespace substitute_engine;

namespace
{

//! A full engine's `max_lora_rank` as the tests build adapters against it.
constexpr int32_t kMaxLoraRank{8};
//! The hidden dimension every test binding and adapter tensor shares.
constexpr int64_t kHiddenSize{64};

class LoRAManagerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    //! Helper to create a simple GPU tensor with the given shape and a name.
    static rt::Tensor makeGpuTensor(rt::Coords const& shape, std::string const& name)
    {
        rt::Tensor tensor(shape, DeviceType::kGPU, nvinfer1::DataType::kHALF, name);
        // Zero-fill so the finiteness scan over a GPU payload reads deterministic
        // values rather than whatever cudaMalloc returned.
        CUDA_CHECK(cudaMemset(tensor.rawPointer(), 0, tensor.getMemoryCapacity()));
        return tensor;
    }

    //! Build a weight map with an A/B pair of the given rank for testing.
    static std::map<std::string, rt::Tensor> buildTestWeights(std::string const& prefix, int64_t rank = 4)
    {
        std::map<std::string, rt::Tensor> weights;
        weights.emplace("lora_A_layer_0", makeGpuTensor({kHiddenSize, rank}, prefix + "_A0"));
        weights.emplace("lora_B_layer_0", makeGpuTensor({rank, kHiddenSize}, prefix + "_B0"));
        return weights;
    }

    //! Write a two-tensor adapter (lora_A_layer_0 + lora_B_layer_0) in safetensors
    //! form, the way a deployed adapter file arrives. `poison` fills the B payload
    //! with an FP16 NaN to exercise host-side finiteness validation.
    void writeAdapterFile(std::filesystem::path const& path, int64_t rank, bool poison = false)
    {
        std::vector<uint16_t> aHost(static_cast<size_t>(kHiddenSize * rank), 0x3800);
        std::vector<uint16_t> bHost(
            static_cast<size_t>(rank * kHiddenSize), poison ? uint16_t{0x7E00} : uint16_t{0x3800});
        rt::Tensor a({kHiddenSize, rank}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "lora_A_layer_0");
        rt::Tensor b({rank, kHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "lora_B_layer_0");
        CUDA_CHECK(cudaMemcpyAsync(
            a.rawPointer(), aHost.data(), aHost.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(
            b.rawPointer(), bHost.data(), bHost.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<rt::Tensor> tensors;
        tensors.push_back(std::move(a));
        tensors.push_back(std::move(b));
        ASSERT_TRUE(safetensors::saveSafetensors(path, tensors, mStream));
    }

    //! A mock engine whose I/O list is exactly one LoRA pair: the shape contract
    //! `initializeEngineBindings` and the runtime assembly enforce.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeLoraEngine(int64_t maxRank = kMaxLoraRank)
    {
        using ::testing::_;
        using ::testing::Return;
        using ::testing::StrEq;

        nvinfer1::Dims aShape;
        aShape.nbDims = 2;
        aShape.d[0] = kHiddenSize;
        aShape.d[1] = maxRank;
        nvinfer1::Dims bShape;
        bShape.nbDims = 2;
        bShape.d[0] = maxRank;
        bShape.d[1] = kHiddenSize;

        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, getNumIOTensors()).WillByDefault(Return(2));
        ON_CALL(*engine, getIOTensorName(0)).WillByDefault(Return("lora_A_layer_0"));
        ON_CALL(*engine, getIOTensorName(1)).WillByDefault(Return("lora_B_layer_0"));
        ON_CALL(*engine, getProfileShape(StrEq("lora_A_layer_0"), _, _)).WillByDefault(Return(aShape));
        ON_CALL(*engine, getProfileShape(StrEq("lora_B_layer_0"), _, _)).WillByDefault(Return(bShape));
        return engine;
    }

    cudaStream_t mStream{};
    //! The tensor map refreshTensorMap populates, keyed by engine binding name.
    TensorMap mTensorMap;
};

//! An adapter tensor is useless unless its engine binding exists: it would ride
//! the rank-1 dummy forever. Registration must refuse it loudly.
TEST_F(LoRAManagerTest, AdapterTensorWithoutEngineBindingThrowsAtInitializeEngineBindings)
{
    LoRAManager mgr(kMaxLoraRank);
    // The engine's bindings are lora_{A,B}_layer_0; an extra unpaired adapter
    // pair on another layer names no binding and can never be read by the engine.
    std::map<std::string, rt::Tensor> weights = buildTestWeights("a1");
    weights.emplace("lora_A_layer_9", makeGpuTensor({kHiddenSize, 4}, "a1_A9"));
    weights.emplace("lora_B_layer_9", makeGpuTensor({4, kHiddenSize}, "a1_B9"));
    mgr.addWeights("adapter1", std::move(weights));

    auto engine = makeLoraEngine();
    EXPECT_THROW(mgr.initializeEngineBindings(*engine), std::runtime_error);
}

//! A matching pair binds cleanly and every binding reaches the adapter tensor.
TEST_F(LoRAManagerTest, MatchingAdapterPassesInitializeEngineBindings)
{
    LoRAManager mgr(kMaxLoraRank);
    mgr.addWeights("adapter1", buildTestWeights("a1"));

    auto engine = makeLoraEngine();
    mgr.initializeEngineBindings(*engine);

    mgr.switchWeights("adapter1");
    EXPECT_EQ(mgr.getActiveAdapterName(), "adapter1");
    mgr.refreshTensorMap(mTensorMap);

    rt::Tensor* const bound = mTensorMap.get("lora_A_layer_0");
    ASSERT_NE(bound, nullptr);
    EXPECT_EQ(bound->getShape()[0], kHiddenSize);
    EXPECT_EQ(bound->getShape()[1], 4);
}

//! Every tensor an adapter carries must be a lora_A or lora_B factor.
TEST_F(LoRAManagerTest, NonLoraTensorNameThrows)
{
    LoRAManager mgr(kMaxLoraRank);

    std::map<std::string, rt::Tensor> weights = buildTestWeights("a1");
    weights.emplace("hidden_weights_layer_0", makeGpuTensor({kHiddenSize, 4}, "a1_H0"));

    EXPECT_THROW(mgr.addWeights("adapter1", std::move(weights)), std::runtime_error);
}

//! Rank above the engine's max is refused at registration, before any engine exists.
TEST_F(LoRAManagerTest, OverRankAdapterThrows)
{
    LoRAManager mgr(kMaxLoraRank);

    EXPECT_THROW(mgr.addWeights("adapter16", buildTestWeights("a16", /*rank=*/16)), std::runtime_error);
}

//! The over-rank rule also applies to file loading, against the same limit.
TEST_F(LoRAManagerTest, OverRankAdapterFileThrows)
{
    auto const dir = makeScratchDir("loraManagerTest");
    writeAdapterFile(dir / "rank16.safetensors", 16);

    LoRAManager mgr(kMaxLoraRank);
    EXPECT_THROW(mgr.loadWeights("adapter16", dir / "rank16.safetensors", /*stream=*/nullptr), std::runtime_error);

    std::filesystem::remove_all(dir);
}

//! A lone lora_A without its lora_B partner would produce a one-sided update.
TEST_F(LoRAManagerTest, UnpairedLoraAThrows)
{
    LoRAManager mgr(kMaxLoraRank);

    std::map<std::string, rt::Tensor> weights;
    weights.emplace("lora_A_layer_0", makeGpuTensor({kHiddenSize, 4}, "a1_A0"));

    EXPECT_THROW(mgr.addWeights("adapter1", std::move(weights)), std::runtime_error);
}

//! FP16 is the only dtype the engine's LoRA bindings accept.
TEST_F(LoRAManagerTest, NonFp16AdapterThrows)
{
    LoRAManager mgr(kMaxLoraRank);

    std::map<std::string, rt::Tensor> weights;
    weights.emplace(
        "lora_A_layer_0", rt::Tensor({kHiddenSize, 4}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "a1_A0"));
    weights.emplace(
        "lora_B_layer_0", rt::Tensor({4, kHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "a1_B0"));

    EXPECT_THROW(mgr.addWeights("adapter1", std::move(weights)), std::runtime_error);
}

//! NaN in an adapter would poison the engine's whole update, and decode-graph
//! capture only warns on failure, so garbage values are refused at load.
TEST_F(LoRAManagerTest, NonFiniteAdapterValueThrows)
{
    LoRAManager mgr(kMaxLoraRank);

    std::map<std::string, rt::Tensor> weights;
    weights.emplace("lora_A_layer_0", makeGpuTensor({kHiddenSize, 4}, "a1_A0"));
    rt::Tensor bTensor = makeGpuTensor({4, kHiddenSize}, "a1_B0");
    uint16_t const nanBits{0x7E00};
    std::vector<uint16_t> host(4 * static_cast<size_t>(kHiddenSize), nanBits);
    CUDA_CHECK(cudaMemcpyAsync(
        bTensor.rawPointer(), host.data(), host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, /*stream=*/nullptr));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    weights.emplace("lora_B_layer_0", std::move(bTensor));

    EXPECT_THROW(mgr.addWeights("adapter1", std::move(weights)), std::runtime_error);
}

//! A NaN payload in a safetensors file must be caught from the host mapping,
//! before any adapter GPU memory is allocated.
TEST_F(LoRAManagerTest, NonFiniteAdapterFileValueThrows)
{
    auto const dir = makeScratchDir("loraManagerTest");
    writeAdapterFile(dir / "nan.safetensors", 4, /*poison=*/true);

    LoRAManager mgr(kMaxLoraRank);
    EXPECT_THROW(mgr.loadWeights("adapter_nan", dir / "nan.safetensors", /*stream=*/nullptr), std::runtime_error);

    std::filesystem::remove_all(dir);
}

//! A/B partners must agree on rank, or the contraction is shape-broken.
TEST_F(LoRAManagerTest, PartnerRankMismatchThrows)
{
    LoRAManager mgr(kMaxLoraRank);

    std::map<std::string, rt::Tensor> weights;
    weights.emplace("lora_A_layer_0", makeGpuTensor({kHiddenSize, 4}, "a1_A0"));
    weights.emplace("lora_B_layer_0", makeGpuTensor({8, kHiddenSize}, "a1_B0"));

    EXPECT_THROW(mgr.addWeights("adapter1", std::move(weights)), std::runtime_error);
}

//! Adapters are immutable for the runtime's lifetime: a second registration
//! under the same name must fail rather than silently replace the weights.
TEST_F(LoRAManagerTest, DuplicateAdapterNameThrows)
{
    LoRAManager mgr(kMaxLoraRank);
    mgr.addWeights("adapter1", buildTestWeights("a1"));
    EXPECT_THROW(mgr.addWeights("adapter1", buildTestWeights("a1b")), std::runtime_error);
}

//! When an active adapter covers only some engine bindings (fused QKV naming),
//! refreshTensorMap warns once naming the adapter and the unbound counts, and
//! never repeats the warning.
TEST_F(LoRAManagerTest, PartialAdapterWarnsOnceWithUnboundCounts)
{
    LoRAManager mgr(kMaxLoraRank);
    auto engine = makeLoraEngine();

    // The engine exposes a fused pair on layer_1, but the adapter only covers
    // the unfused layer_0 projections: layer_1 stays on the rank-1 dummy.
    using ::testing::_;
    using ::testing::Return;
    using ::testing::StrEq;
    nvinfer1::Dims layer1AShape;
    layer1AShape.nbDims = 2;
    layer1AShape.d[0] = kHiddenSize;
    layer1AShape.d[1] = kMaxLoraRank;
    nvinfer1::Dims layer1BShape;
    layer1BShape.nbDims = 2;
    layer1BShape.d[0] = kMaxLoraRank;
    layer1BShape.d[1] = kHiddenSize;

    ON_CALL(*engine, getNumIOTensors()).WillByDefault(Return(4));
    ON_CALL(*engine, getIOTensorName(2)).WillByDefault(Return("lora_A_layer_1"));
    ON_CALL(*engine, getIOTensorName(3)).WillByDefault(Return("lora_B_layer_1"));
    ON_CALL(*engine, getProfileShape(StrEq("lora_A_layer_1"), _, _)).WillByDefault(Return(layer1AShape));
    ON_CALL(*engine, getProfileShape(StrEq("lora_B_layer_1"), _, _)).WillByDefault(Return(layer1BShape));

    mgr.initializeEngineBindings(*engine);
    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.switchWeights("adapter1");

    ::testing::internal::CaptureStderr();
    mgr.refreshTensorMap(mTensorMap);
    std::string const first = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(first.find("adapter1"), std::string::npos) << "warning must name the adapter, got: " << first;
    EXPECT_NE(first.find("2 of 4"), std::string::npos) << "warning must count unbound of total, got: " << first;

    ::testing::internal::CaptureStderr();
    mgr.refreshTensorMap(mTensorMap);
    std::string const second = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(second.find("leaves"), std::string::npos) << "warning must not repeat, got: " << second;
}

//! A fully-bound adapter produces no warning at all.
TEST_F(LoRAManagerTest, FullyBoundAdapterDoesNotWarn)
{
    LoRAManager mgr(kMaxLoraRank);
    auto engine = makeLoraEngine();

    mgr.initializeEngineBindings(*engine);
    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.switchWeights("adapter1");

    ::testing::internal::CaptureStderr();
    mgr.refreshTensorMap(mTensorMap);
    std::string const stderrOut = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(stderrOut.find("leaves"), std::string::npos) << "no warning expected, got: " << stderrOut;
}

TEST_F(LoRAManagerTest, InitialState)
{
    LoRAManager mgr(kMaxLoraRank);
    EXPECT_FALSE(mgr.hasActiveAdapter());
    EXPECT_TRUE(mgr.getActiveAdapterName().empty());
    EXPECT_TRUE(mgr.getBindingNames().empty());
    EXPECT_TRUE(mgr.getAdapterNames().empty());
}

TEST_F(LoRAManagerTest, AddAndSwitchWeights)
{
    LoRAManager mgr(kMaxLoraRank);

    mgr.addWeights("adapter1", buildTestWeights("a1"));
    EXPECT_FALSE(mgr.hasActiveAdapter());

    mgr.switchWeights("adapter1");
    EXPECT_TRUE(mgr.hasActiveAdapter());
    EXPECT_EQ(mgr.getActiveAdapterName(), "adapter1");

    // Retrieve active weight and verify it is the expected tensor.
    rt::Tensor& w = mgr.getActiveWeight("lora_A_layer_0");
    EXPECT_FALSE(w.isEmpty());
    EXPECT_EQ(w.getShape()[0], kHiddenSize);
    EXPECT_EQ(w.getShape()[1], 4);
}

TEST_F(LoRAManagerTest, SwitchBetweenAdapters)
{
    LoRAManager mgr(kMaxLoraRank);

    mgr.addWeights("adapter1", buildTestWeights("a1"));

    // Create adapter2 with a different rank to distinguish them.
    mgr.addWeights("adapter2", buildTestWeights("a2", /*rank=*/8));

    mgr.switchWeights("adapter1");
    EXPECT_EQ(mgr.getActiveWeight("lora_A_layer_0").getShape()[0], kHiddenSize);

    mgr.switchWeights("adapter2");
    EXPECT_EQ(mgr.getActiveAdapterName(), "adapter2");
    EXPECT_EQ(mgr.getActiveWeight("lora_A_layer_0").getShape()[0], kHiddenSize);
    EXPECT_EQ(mgr.getActiveWeight("lora_A_layer_0").getShape()[1], 8);
}

TEST_F(LoRAManagerTest, ResetWeightsReturnsDummy)
{
    LoRAManager mgr(kMaxLoraRank);
    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.switchWeights("adapter1");
    EXPECT_TRUE(mgr.hasActiveAdapter());

    mgr.resetWeights();
    EXPECT_FALSE(mgr.hasActiveAdapter());
    EXPECT_TRUE(mgr.getActiveAdapterName().empty());

    // After reset, getActiveWeight should return a dummy tensor.
    rt::Tensor& dummy = mgr.getActiveWeight("lora_A_layer_0");
    // The dummy is a small tensor of shape [1].
    EXPECT_EQ(dummy.getShape().volume(), 1);
}

TEST_F(LoRAManagerTest, SwitchToUnknownAdapterThrows)
{
    LoRAManager mgr(kMaxLoraRank);
    EXPECT_THROW(mgr.switchWeights("nonexistent"), std::runtime_error);
}

TEST_F(LoRAManagerTest, GetActiveWeightMissingBindingThrows)
{
    LoRAManager mgr(kMaxLoraRank);
    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.switchWeights("adapter1");

    EXPECT_THROW(mgr.getActiveWeight("nonexistent_binding"), std::runtime_error);
}

TEST_F(LoRAManagerTest, GetBindingNamesCompleteness)
{
    LoRAManager mgr(kMaxLoraRank);

    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.addWeights("adapter2", buildTestWeights("a2"));

    auto const names = mgr.getBindingNames();

    // Should contain the union of all binding names.
    EXPECT_EQ(names.size(), 2U);
    EXPECT_TRUE(std::find(names.begin(), names.end(), "lora_A_layer_0") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "lora_B_layer_0") != names.end());
}

TEST_F(LoRAManagerTest, GetAdapterNames)
{
    LoRAManager mgr(kMaxLoraRank);
    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.addWeights("adapter2", buildTestWeights("a2"));

    auto const adapterNames = mgr.getAdapterNames();
    EXPECT_EQ(adapterNames.size(), 2U);
    EXPECT_TRUE(std::find(adapterNames.begin(), adapterNames.end(), "adapter1") != adapterNames.end());
    EXPECT_TRUE(std::find(adapterNames.begin(), adapterNames.end(), "adapter2") != adapterNames.end());
}

TEST_F(LoRAManagerTest, DummyTensorReturnedWhenNoAdapterActive)
{
    LoRAManager mgr(kMaxLoraRank);

    // Even without any adapters loaded, getActiveWeight should return a dummy.
    rt::Tensor& dummy = mgr.getActiveWeight("any_binding");
    EXPECT_EQ(dummy.getShape().volume(), 1);
    EXPECT_EQ(dummy.getDeviceType(), DeviceType::kGPU);
}

} // namespace
