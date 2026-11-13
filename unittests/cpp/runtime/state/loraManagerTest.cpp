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

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
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

//! A factor that is not 2-D has no rank dimension; it is refused before its shape is indexed.
TEST_F(LoRAManagerTest, OneDimensionalFactorThrows)
{
    LoRAManager mgr(kMaxLoraRank);

    std::map<std::string, rt::Tensor> weights;
    weights.emplace("lora_A_layer_0", makeGpuTensor({kHiddenSize}, "a1_A0"));
    weights.emplace("lora_B_layer_0", makeGpuTensor({4, kHiddenSize}, "a1_B0"));

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

//! An FP16 GPU tensor whose every element holds the bit pattern @p bits, with element @p poke
//! (when non-negative) set to @p pokeBits instead.
rt::Tensor makeFilledGpuTensor(rt::Coords const& shape, std::string const& name, uint16_t bits, int64_t poke = -1,
    uint16_t pokeBits = 0)
{
    std::vector<uint16_t> host(static_cast<size_t>(shape.volume()), bits);
    if (poke >= 0)
    {
        host[static_cast<size_t>(poke)] = pokeBits;
    }
    rt::Tensor tensor(shape, DeviceType::kGPU, nvinfer1::DataType::kHALF, name);
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), host.data(), host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    return tensor;
}

//! The two-tensor adapter writeAdapterFile() produces (both factors 0x3800), registered directly.
std::map<std::string, rt::Tensor> filledWeights(int64_t rank, std::string const& aName = "lora_A_layer_0",
    std::string const& bName = "lora_B_layer_0", int64_t pokeB = -1)
{
    std::map<std::string, rt::Tensor> weights;
    weights.emplace(aName, makeFilledGpuTensor({kHiddenSize, rank}, aName, 0x3800));
    weights.emplace(bName, makeFilledGpuTensor({rank, kHiddenSize}, bName, 0x3800, pokeB, 0x3C00));
    return weights;
}

//! Identity depends on content only: equal weights under two names share it.
TEST_F(LoRAManagerTest, AdapterIdentityIgnoresAdapterName)
{
    LoRAManager mgr(kMaxLoraRank);
    mgr.addWeights("tool", filledWeights(4));
    mgr.addWeights("copy", filledWeights(4));

    std::string const& identity = mgr.getAdapterIdentity("tool");
    EXPECT_EQ(identity.size(), 32u);
    EXPECT_EQ(identity.find_first_not_of("0123456789abcdef"), std::string::npos);
    EXPECT_EQ(identity, mgr.getAdapterIdentity("copy"));
}

//! One changed element is a different adapter: replacing weights under a name must change cache keys.
TEST_F(LoRAManagerTest, AdapterIdentityChangesWithOneElement)
{
    LoRAManager mgr(kMaxLoraRank);
    mgr.addWeights("v1", filledWeights(4));
    mgr.addWeights("v2", filledWeights(4, "lora_A_layer_0", "lora_B_layer_0", /*pokeB=*/17));

    EXPECT_NE(mgr.getAdapterIdentity("v1"), mgr.getAdapterIdentity("v2"));
}

//! The same values bound to different engine inputs are a different adapter.
TEST_F(LoRAManagerTest, AdapterIdentityChangesWithTensorName)
{
    LoRAManager mgr(kMaxLoraRank);
    mgr.addWeights("layer0", filledWeights(4));
    mgr.addWeights("layer1", filledWeights(4, "lora_A_layer_1", "lora_B_layer_1"));

    EXPECT_NE(mgr.getAdapterIdentity("layer0"), mgr.getAdapterIdentity("layer1"));
}

//! A file loaded from disk and the same tensors registered in memory have one identity.
TEST_F(LoRAManagerTest, AdapterIdentityMatchesBetweenFileAndMemory)
{
    auto const dir = makeScratchDir("loraManagerTest");
    writeAdapterFile(dir / "adapter.safetensors", 4);

    LoRAManager mgr(kMaxLoraRank);
    mgr.loadWeights("fromFile", dir / "adapter.safetensors", mStream);
    mgr.addWeights("fromMemory", filledWeights(4));

    EXPECT_EQ(mgr.getAdapterIdentity("fromFile"), mgr.getAdapterIdentity("fromMemory"));
    std::filesystem::remove_all(dir);
}

TEST_F(LoRAManagerTest, AdapterIdentityOfUnknownAdapterThrows)
{
    LoRAManager mgr(kMaxLoraRank);
    EXPECT_THROW(mgr.getAdapterIdentity("missing"), std::runtime_error);
}

//! A GPU tensor holding @p bytes of host data, for writing adapter files of any dtype.
rt::Tensor makeTensorFromHost(rt::Coords const& shape, nvinfer1::DataType dtype, std::string const& name,
    void const* bytes, size_t size)
{
    rt::Tensor tensor(shape, DeviceType::kGPU, dtype, name);
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), bytes, size, cudaMemcpyHostToDevice));
    return tensor;
}

uint16_t halfBits(float value)
{
    return __half_raw(__float2half_rn(value)).x;
}

//! INT8 transport adapter: one module, A [64, 4] and B [4, 64] with codes c(i) and per-column scales.
struct Int8Adapter
{
    int64_t rank{4};
    std::vector<int8_t> aCodes, bCodes;
    std::vector<uint16_t> aScale, bScale;

    Int8Adapter()
    {
        aCodes.resize(static_cast<size_t>(kHiddenSize * rank));
        bCodes.resize(static_cast<size_t>(rank * kHiddenSize));
        for (size_t i = 0; i < aCodes.size(); ++i)
        {
            aCodes[i] = static_cast<int8_t>(static_cast<int>(i % 255) - 127);
        }
        for (size_t i = 0; i < bCodes.size(); ++i)
        {
            bCodes[i] = static_cast<int8_t>(static_cast<int>((i * 7) % 255) - 127);
        }
        for (int64_t c = 0; c < rank; ++c)
        {
            aScale.push_back(halfBits(0.01F * static_cast<float>(c + 1)));
        }
        for (int64_t c = 0; c < kHiddenSize; ++c)
        {
            bScale.push_back(halfBits(0.002F * static_cast<float>(c + 1)));
        }
    }

    //! The FP16 bits the loader must produce for factor A or B.
    std::vector<uint16_t> dequantized(bool a) const
    {
        auto const& codes = a ? aCodes : bCodes;
        auto const& scale = a ? aScale : bScale;
        int64_t const columns = a ? rank : kHiddenSize;
        std::vector<uint16_t> out(codes.size());
        for (size_t i = 0; i < codes.size(); ++i)
        {
            __half_raw s{};
            s.x = scale[i % static_cast<size_t>(columns)];
            out[i] = halfBits(static_cast<float>(codes[i]) * __half2float(__half(s)));
        }
        return out;
    }

    std::vector<rt::Tensor> tensors(bool withAScale = true) const
    {
        std::vector<rt::Tensor> out;
        out.push_back(makeTensorFromHost(
            {kHiddenSize, rank}, nvinfer1::DataType::kINT8, "lora_A_layer_0", aCodes.data(), aCodes.size()));
        out.push_back(makeTensorFromHost(
            {rank, kHiddenSize}, nvinfer1::DataType::kINT8, "lora_B_layer_0", bCodes.data(), bCodes.size()));
        if (withAScale)
        {
            out.push_back(makeTensorFromHost({static_cast<int64_t>(aScale.size())}, nvinfer1::DataType::kHALF,
                "lora_A_layer_0_scale", aScale.data(), aScale.size() * sizeof(uint16_t)));
        }
        out.push_back(makeTensorFromHost({static_cast<int64_t>(bScale.size())}, nvinfer1::DataType::kHALF,
            "lora_B_layer_0_scale", bScale.data(), bScale.size() * sizeof(uint16_t)));
        return out;
    }
};

std::vector<uint16_t> readBack(rt::Tensor const& tensor)
{
    std::vector<uint16_t> host(static_cast<size_t>(tensor.getShape().volume()));
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.rawPointer(), host.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return host;
}

//! An INT8 transport file registers FP16 factors equal to float(code) * float(scale) rounded to FP16.
TEST_F(LoRAManagerTest, Int8AdapterIsDequantizedToFp16AtLoad)
{
    auto const dir = makeScratchDir("loraManagerTest");
    Int8Adapter const adapter;
    ASSERT_TRUE(safetensors::saveSafetensors(dir / "int8.safetensors", adapter.tensors(), mStream));

    LoRAManager mgr(kMaxLoraRank);
    mgr.loadWeights("int8", dir / "int8.safetensors", mStream);
    mgr.switchWeights("int8");

    rt::Tensor& a = mgr.getActiveWeight("lora_A_layer_0");
    rt::Tensor& b = mgr.getActiveWeight("lora_B_layer_0");
    EXPECT_EQ(a.getDataType(), nvinfer1::DataType::kHALF);
    EXPECT_EQ(readBack(a), adapter.dequantized(true));
    EXPECT_EQ(readBack(b), adapter.dequantized(false));
    EXPECT_EQ(mgr.getBindingNames().size(), 2u);
    std::filesystem::remove_all(dir);
}

//! The transport format does not change what the adapter is: its identity equals the FP16 file of its values.
TEST_F(LoRAManagerTest, Int8AdapterIdentityMatchesItsDequantizedFp16File)
{
    auto const dir = makeScratchDir("loraManagerTest");
    Int8Adapter const adapter;
    ASSERT_TRUE(safetensors::saveSafetensors(dir / "int8.safetensors", adapter.tensors(), mStream));
    auto const aHost = adapter.dequantized(true);
    auto const bHost = adapter.dequantized(false);
    std::vector<rt::Tensor> fp16;
    fp16.push_back(makeTensorFromHost({kHiddenSize, adapter.rank}, nvinfer1::DataType::kHALF, "lora_A_layer_0",
        aHost.data(), aHost.size() * sizeof(uint16_t)));
    fp16.push_back(makeTensorFromHost({adapter.rank, kHiddenSize}, nvinfer1::DataType::kHALF, "lora_B_layer_0",
        bHost.data(), bHost.size() * sizeof(uint16_t)));
    ASSERT_TRUE(safetensors::saveSafetensors(dir / "fp16.safetensors", fp16, mStream));

    LoRAManager mgr(kMaxLoraRank);
    mgr.loadWeights("int8", dir / "int8.safetensors", mStream);
    mgr.loadWeights("fp16", dir / "fp16.safetensors", mStream);

    EXPECT_EQ(mgr.getAdapterIdentity("int8"), mgr.getAdapterIdentity("fp16"));
    std::filesystem::remove_all(dir);
}

//! Each malformed INT8 file is refused at load.
TEST_F(LoRAManagerTest, MalformedInt8AdaptersThrow)
{
    auto const dir = makeScratchDir("loraManagerTest");
    Int8Adapter const good;
    auto const expectThrow = [&](std::vector<rt::Tensor> const& tensors, char const* label) {
        auto const file = dir / (std::string(label) + ".safetensors");
        ASSERT_TRUE(safetensors::saveSafetensors(file, tensors, mStream));
        LoRAManager mgr(kMaxLoraRank);
        EXPECT_THROW(mgr.loadWeights(label, file, mStream), std::runtime_error) << label;
    };

    // The well-formed file loads, so each refusal below is about its one defect.
    ASSERT_TRUE(safetensors::saveSafetensors(dir / "good.safetensors", good.tensors(), mStream));
    LoRAManager goodManager(kMaxLoraRank);
    EXPECT_NO_THROW(goodManager.loadWeights("good", dir / "good.safetensors", mStream));

    expectThrow(good.tensors(/*withAScale=*/false), "missingScale");

    Int8Adapter shortScale;
    shortScale.aScale.pop_back();
    expectThrow(shortScale.tensors(), "wrongScaleLength");

    Int8Adapter zeroScale;
    zeroScale.bScale[3] = halfBits(0.0F);
    expectThrow(zeroScale.tensors(), "zeroScale");

    Int8Adapter nanScale;
    nanScale.bScale[5] = 0x7E00;
    expectThrow(nanScale.tensors(), "nanScale");

    auto f32Scale = good.tensors();
    std::vector<float> f32(static_cast<size_t>(good.rank), 0.01F);
    f32Scale[2] = makeTensorFromHost(
        {good.rank}, nvinfer1::DataType::kFLOAT, "lora_A_layer_0_scale", f32.data(), f32.size() * sizeof(float));
    expectThrow(f32Scale, "f32Scale");

    auto mixed = good.tensors();
    std::vector<uint16_t> bFp16(static_cast<size_t>(good.rank * kHiddenSize), 0x3800);
    mixed[1] = makeTensorFromHost({good.rank, kHiddenSize}, nvinfer1::DataType::kHALF, "lora_B_layer_0",
        bFp16.data(), bFp16.size() * sizeof(uint16_t));
    mixed.pop_back(); // the B scale
    expectThrow(mixed, "mixedFp16Int8");

    std::filesystem::remove_all(dir);
}

} // namespace
