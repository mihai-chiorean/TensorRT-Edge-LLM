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

#include "common/checkMacros.h"
#include "kernels/speculative/gemma4MTPRuntimeKernels.h"
#include "runtime/decoding/requestStableRng.h"
#include "sampler/sampling.h"
#include "testUtils.h"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

using namespace trt_edgellm;

namespace
{

struct RowSamplingCase
{
    int32_t rows;
    int32_t vocabSize;
    int32_t topK;
    float temperature;
    float topP;
};

class Gemma4MtpRowSamplingTest : public ::testing::TestWithParam<RowSamplingCase>
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    cudaStream_t stream{};
};

// The verification sampler must pick the same token as the generic top-k sampler for the same uniform.
// Logits are drawn from a continuous distribution, so the k-th place does not tie.
TEST_P(Gemma4MtpRowSamplingTest, MatchesGenericTopKSamplerForTheSameUniforms)
{
    RowSamplingCase const c = GetParam();
    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
    std::vector<float> logits(static_cast<size_t>(c.rows) * c.vocabSize);
    for (auto& v : logits)
    {
        v = normal(rng);
    }
    std::vector<float> uniforms(c.rows);
    for (auto& u : uniforms)
    {
        u = uniform(rng);
    }

    rt::Tensor logitsTensor({c.rows, c.vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    rt::Tensor uniformsTensor({c.rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "uniforms");
    rt::Tensor generic({c.rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "generic");
    rt::Tensor fast({c.rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "fast");
    copyHostToDevice(logitsTensor, logits);
    copyHostToDevice(uniformsTensor, uniforms);

    SamplingParams const params(c.rows, c.vocabSize, c.temperature, c.topK, c.topP);
    rt::Tensor workspace({static_cast<int64_t>(getTopKtopPSamplingWorkspaceSize(c.rows, c.vocabSize, params))},
        rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "workspace");
    topKtopPSamplingFromLogits(logitsTensor, generic, params, workspace, stream, 0, 0, &uniformsTensor);
    kernel::launchGemma4MTPSampleTopKRows(logitsTensor.dataPointer<float>(), c.rows, c.vocabSize, c.topK, c.temperature,
        c.topP, uniformsTensor.dataPointer<float>(), fast.dataPointer<int32_t>(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const genericTokens = copyDeviceToHost<int32_t>(generic);
    auto const fastTokens = copyDeviceToHost<int32_t>(fast);
    EXPECT_EQ(genericTokens, fastTokens);
}

INSTANTIATE_TEST_SUITE_P(Shapes, Gemma4MtpRowSamplingTest,
    ::testing::Values(RowSamplingCase{12, 262144, 50, 0.7F, 0.9F}, RowSamplingCase{12, 262144, 50, 0.7F, 1.0F},
        RowSamplingCase{64, 4099, 1, 1.0F, 1.0F}, RowSamplingCase{64, 4099, 7, 1.3F, 0.5F},
        RowSamplingCase{16, 32000, 1024, 0.9F, 0.95F}, RowSamplingCase{8, 600, 600, 0.6F, 1.0F}));

// Masked (-inf) logits, as a grammar leaves them, are never sampled even when fewer than k remain.
TEST(Gemma4MtpRowSampling, NeverSamplesMaskedLogits)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    constexpr int32_t rows = 256;
    constexpr int32_t vocabSize = 1000;
    std::vector<float> logits(rows * vocabSize, -std::numeric_limits<float>::infinity());
    std::vector<float> uniforms(rows);
    for (int32_t r = 0; r < rows; ++r)
    {
        logits[r * vocabSize + 3] = 0.5F;
        logits[r * vocabSize + 500] = 0.0F;
        logits[r * vocabSize + 999] = -0.25F;
        uniforms[r] = (static_cast<float>(r) + 0.5F) / rows;
    }
    uniforms[rows - 1] = 1.0F;
    rt::Tensor logitsTensor({rows, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    rt::Tensor uniformsTensor({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "uniforms");
    rt::Tensor sampled({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "sampled");
    copyHostToDevice(logitsTensor, logits);
    copyHostToDevice(uniformsTensor, uniforms);
    kernel::launchGemma4MTPSampleTopKRows(logitsTensor.dataPointer<float>(), rows, vocabSize, /*topK=*/50,
        /*temperature=*/0.8F, /*topP=*/1.0F, uniformsTensor.dataPointer<float>(), sampled.dataPointer<int32_t>(),
        stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto const tokens = copyDeviceToHost<int32_t>(sampled);
    int32_t counts[3]{0, 0, 0};
    for (int32_t t : tokens)
    {
        ASSERT_TRUE(t == 3 || t == 500 || t == 999) << "sampled masked token " << t;
        counts[t == 3 ? 0 : (t == 500 ? 1 : 2)]++;
    }
    // Stratified uniforms: counts follow softmax([0.5, 0, -0.25] / 0.8) = [0.519, 0.278, 0.203] to within a row.
    EXPECT_NEAR(counts[0], 0.5190 * rows, 2.0);
    EXPECT_NEAR(counts[1], 0.2778 * rows, 2.0);
    EXPECT_NEAR(counts[2], 0.2032 * rows, 2.0);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// Verify rows draw the vanilla decoder's request-stable uniform for the position they predict:
// a linear row i predicts nextPosition + i, a tree node its depth below the root.
TEST(Gemma4MtpVerifyUniforms, UseTheVanillaUniformOfThePredictedPosition)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    constexpr int32_t batch = 2;
    constexpr int32_t verifySize = 4;
    std::vector<int64_t> rng{11, 22, 100, 7}; // seeds, then next absolute positions
    // Tree for both batches: root -> {1, 2}; 1 -> 3.
    std::vector<int8_t> mask{1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1};
    mask.insert(mask.end(), mask.begin(), mask.end());
    rt::Tensor rngTensor({2, batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "rng");
    rt::Tensor maskTensor({batch, verifySize, verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "mask");
    rt::Tensor linear({batch * verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "linear");
    rt::Tensor tree({batch * verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "tree");
    copyHostToDevice(rngTensor, rng);
    copyHostToDevice(maskTensor, mask);
    auto const* seeds = reinterpret_cast<uint64_t const*>(rngTensor.dataPointer<int64_t>());
    kernel::launchGemma4MTPVerifyUniforms(
        seeds, seeds + batch, nullptr, linear.dataPointer<float>(), batch, verifySize, stream);
    kernel::launchGemma4MTPVerifyUniforms(
        seeds, seeds + batch, maskTensor.dataPointer<int8_t>(), tree.dataPointer<float>(), batch, verifySize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto const linearHost = copyDeviceToHost<float>(linear);
    auto const treeHost = copyDeviceToHost<float>(tree);
    int32_t const treeDepth[verifySize]{0, 1, 1, 2};
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t i = 0; i < verifySize; ++i)
        {
            auto const seed = static_cast<uint64_t>(rng[b]);
            auto const next = static_cast<uint64_t>(rng[batch + b]);
            EXPECT_EQ(linearHost[b * verifySize + i],
                rt::requestStableUniform(seed, next + i, rt::SpecRandomPurpose::kTarget, 0));
            EXPECT_EQ(treeHost[b * verifySize + i],
                rt::requestStableUniform(seed, next + treeDepth[i], rt::SpecRandomPurpose::kTarget, 0));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace
