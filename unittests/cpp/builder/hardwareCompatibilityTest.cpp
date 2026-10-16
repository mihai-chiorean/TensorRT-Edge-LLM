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

#include "builder/builderUtils.h"
#include "builder/llmBuilder.h"
#include "builder/visualBuilder.h"
#include "common/logger.h"
#include "common/trtUtils.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

using namespace trt_edgellm;
using namespace trt_edgellm::builder;

TEST(HardwareCompatibilityTest, ParsesKnownNames)
{
    EXPECT_EQ(parseHardwareCompatibility("none"), HardwareCompatibility::kNONE);
    EXPECT_EQ(parseHardwareCompatibility("same_compute_capability"), HardwareCompatibility::kSAME_COMPUTE_CAPABILITY);
}

TEST(HardwareCompatibilityTest, RejectsUnknownNames)
{
    EXPECT_FALSE(parseHardwareCompatibility("").has_value());
    EXPECT_FALSE(parseHardwareCompatibility("same-cc").has_value());
    EXPECT_FALSE(parseHardwareCompatibility("SAME_COMPUTE_CAPABILITY").has_value());
    EXPECT_FALSE(parseHardwareCompatibility("ampere_plus").has_value());
}

TEST(HardwareCompatibilityTest, NamesRoundTrip)
{
    for (auto const compatibility : {HardwareCompatibility::kNONE, HardwareCompatibility::kSAME_COMPUTE_CAPABILITY})
    {
        EXPECT_EQ(parseHardwareCompatibility(hardwareCompatibilityName(compatibility)), compatibility);
    }
}

TEST(HardwareCompatibilityTest, DefaultBuilderConfigsDoNotSerializeTheLevel)
{
    LLMBuilderConfig const llmConfig;
    VisualBuilderConfig const visualConfig;

    EXPECT_FALSE(llmConfig.toJson().contains("hardware_compatibility"));
    EXPECT_FALSE(visualConfig.toJson().contains("hardware_compatibility"));
    EXPECT_EQ(LLMBuilderConfig::fromJson(llmConfig.toJson()).hardwareCompatibility, HardwareCompatibility::kNONE);
}

TEST(HardwareCompatibilityTest, SameComputeCapabilityRoundTripsThroughJson)
{
    LLMBuilderConfig llmConfig;
    llmConfig.hardwareCompatibility = HardwareCompatibility::kSAME_COMPUTE_CAPABILITY;
    VisualBuilderConfig visualConfig;
    visualConfig.hardwareCompatibility = HardwareCompatibility::kSAME_COMPUTE_CAPABILITY;

    EXPECT_EQ(llmConfig.toJson().at("hardware_compatibility"), "same_compute_capability");
    EXPECT_EQ(LLMBuilderConfig::fromJson(llmConfig.toJson()).hardwareCompatibility,
        HardwareCompatibility::kSAME_COMPUTE_CAPABILITY);
    EXPECT_EQ(VisualBuilderConfig::fromJson(visualConfig.toJson()).hardwareCompatibility,
        HardwareCompatibility::kSAME_COMPUTE_CAPABILITY);
}

TEST(HardwareCompatibilityTest, UnknownJsonValueIsRejected)
{
    Json const json = {{"hardware_compatibility", "same-cc"}};

    EXPECT_THROW(LLMBuilderConfig::fromJson(json), std::runtime_error);
    EXPECT_THROW(VisualBuilderConfig::fromJson(json), std::runtime_error);
}

TEST(HardwareCompatibilityTest, AppliesLevelToBuilderConfig)
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "No CUDA device available.";
    }
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    ASSERT_NE(builder, nullptr);
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    ASSERT_NE(config, nullptr);

    EXPECT_TRUE(applyHardwareCompatibility(*config, HardwareCompatibility::kNONE));
    EXPECT_EQ(config->getHardwareCompatibilityLevel(), nvinfer1::HardwareCompatibilityLevel::kNONE);

#if !IS_TRT_RTX && (NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 9))
    EXPECT_TRUE(applyHardwareCompatibility(*config, HardwareCompatibility::kSAME_COMPUTE_CAPABILITY));
    EXPECT_EQ(config->getHardwareCompatibilityLevel(), nvinfer1::HardwareCompatibilityLevel::kSAME_COMPUTE_CAPABILITY);
#else
    EXPECT_FALSE(applyHardwareCompatibility(*config, HardwareCompatibility::kSAME_COMPUTE_CAPABILITY));
#endif
}
