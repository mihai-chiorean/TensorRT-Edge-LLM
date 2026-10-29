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

/*
 * The group-size contract of the INT4 groupwise GEMM plugins, asserted at the
 * plugin boundary: the V1 plugin serves group 128 only, and the V2 plugin
 * serves group 32 only when the CuTe DSL artifact carries the _g32 variants.
 * The scales tensor (position 2) is the only input whose shape depends on the
 * group size, so each case pins every other position as accepted and varies
 * the expected verdict on that position alone.
 */

#include "testPluginLoader.h"

#include <NvInfer.h>
#include <gtest/gtest.h>

#include <array>
#include <memory>

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;

constexpr char const* kV1Name{"Int4GroupwiseGemmPlugin"};
constexpr char const* kV2Name{"Int4GroupwiseGemmPluginV2"};
constexpr char const* kPluginVersion{"1"};
constexpr char const* kPluginNamespace{""};
constexpr int32_t kGemmN{128};
constexpr int32_t kGemmK{256};
constexpr int32_t kNbInputs{3};
constexpr int32_t kNbOutputs{1};
constexpr int32_t kScalesPos{2};
constexpr int32_t kM{8};

int32_t divUp(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

std::unique_ptr<IPluginV3> createGemmPlugin(IPluginRegistry& registry, char const* name, int32_t groupSize)
{
    auto* creator = static_cast<IPluginCreatorV3One*>(registry.getCreator(name, kPluginVersion, kPluginNamespace));
    if (creator == nullptr)
    {
        return nullptr;
    }
    int32_t n = kGemmN;
    int32_t k = kGemmK;
    int32_t gs = groupSize;
    std::array<PluginField, 3> const fields{PluginField("gemm_n", &n, PluginFieldType::kINT32, 1),
        PluginField("gemm_k", &k, PluginFieldType::kINT32, 1),
        PluginField("group_size", &gs, PluginFieldType::kINT32, 1)};
    PluginFieldCollection const fc{static_cast<int32_t>(fields.size()), fields.data()};
    return std::unique_ptr<IPluginV3>(creator->createPlugin("group_size_contract", &fc, TensorRTPhase::kBUILD));
}

void expectPositionsAcceptGroupSize(
    IPluginV3OneBuild* build, int32_t groupSize, bool weightsUseFragmentLayout, bool groupSizeAccepted)
{
    std::array<DynamicPluginTensorDesc, kNbInputs + kNbOutputs> desc{};
    for (auto& tensor : desc)
    {
        tensor.desc.format = TensorFormat::kLINEAR;
    }
    desc[0].desc.type = desc[3].desc.type = DataType::kHALF;
    desc[0].desc.dims = Dims{2, {kM, kGemmK}};
    desc[3].desc.dims = Dims{2, {kM, kGemmN}};
    desc[1].desc.type = DataType::kINT8;
    desc[1].desc.dims = weightsUseFragmentLayout ? Dims{2, {divUp(kGemmN, 128) * divUp(kGemmK, 64) * 8, 512}}
                                                 : Dims{2, {kGemmN / 2, kGemmK}};
    desc[kScalesPos].desc.type = DataType::kHALF;
    desc[kScalesPos].desc.dims = Dims{2, {kGemmK / groupSize, kGemmN}};
    for (int32_t pos = 0; pos < kNbInputs + kNbOutputs; ++pos)
    {
        bool const expected = (pos == kScalesPos) ? groupSizeAccepted : true;
        EXPECT_EQ(build->supportsFormatCombination(pos, desc.data(), kNbInputs, kNbOutputs), expected)
            << "groupSize=" << groupSize << " position=" << pos;
    }
}

TEST(Int4GroupwiseGemmPluginV2Contract, GroupSizeGate)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    auto& registry = *getPluginRegistry();
    {
        auto plugin = createGemmPlugin(registry, kV2Name, 128);
        ASSERT_NE(plugin, nullptr);
        auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
        ASSERT_NE(build, nullptr);
        expectPositionsAcceptGroupSize(build, 128, /*weightsUseFragmentLayout=*/true,
            /*groupSizeAccepted=*/true);
    }
#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
    {
        auto plugin = createGemmPlugin(registry, kV2Name, 32);
        ASSERT_NE(plugin, nullptr);
        auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
        ASSERT_NE(build, nullptr);
#ifdef CUTE_DSL_INT4_FP16_GEMM_G32_ENABLED
        expectPositionsAcceptGroupSize(build, 32, /*weightsUseFragmentLayout=*/true, /*groupSizeAccepted=*/true);
#else
        expectPositionsAcceptGroupSize(build, 32, /*weightsUseFragmentLayout=*/true,
            /*groupSizeAccepted=*/false);
#endif
    }
    {
        auto plugin = createGemmPlugin(registry, kV2Name, 64);
        ASSERT_NE(plugin, nullptr);
        auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
        ASSERT_NE(build, nullptr);
        expectPositionsAcceptGroupSize(build, 64, /*weightsUseFragmentLayout=*/true, /*groupSizeAccepted=*/false);
    }
#endif
}

TEST(Int4GroupwiseGemmPluginContract, GroupSizeGate)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    auto& registry = *getPluginRegistry();
    {
        auto plugin = createGemmPlugin(registry, kV1Name, 128);
        ASSERT_NE(plugin, nullptr);
        auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
        ASSERT_NE(build, nullptr);
        expectPositionsAcceptGroupSize(build, 128, /*weightsUseFragmentLayout=*/false,
            /*groupSizeAccepted=*/true);
    }
    {
        auto plugin = createGemmPlugin(registry, kV1Name, 32);
        ASSERT_NE(plugin, nullptr);
        auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
        ASSERT_NE(build, nullptr);
        expectPositionsAcceptGroupSize(build, 32, /*weightsUseFragmentLayout=*/false, /*groupSizeAccepted=*/false);
    }
}
} // namespace
} // namespace trt_edgellm::plugins
