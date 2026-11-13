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
 * LoraGroupPlugin at its TensorRT boundary: attributes survive serialization, outputs alias
 * their base inputs only in place, and enqueue adds the residual for targets bound at a
 * positive rank while leaving rank-0 targets exactly as the base GEMM produced them (copied
 * when not in place).
 */

#include "testPluginLoader.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;

constexpr char const* kName{"LoraGroupPlugin"};

std::unique_ptr<IPluginV3> createGroup(int32_t numTargets, int32_t inplace)
{
    auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(kName, "1", ""));
    if (creator == nullptr)
    {
        return nullptr;
    }
    std::array<PluginField, 2> const fields{PluginField("num_targets", &numTargets, PluginFieldType::kINT32, 1),
        PluginField("inplace", &inplace, PluginFieldType::kINT32, 1)};
    PluginFieldCollection const fc{static_cast<int32_t>(fields.size()), fields.data()};
    return std::unique_ptr<IPluginV3>(creator->createPlugin("lora_group_test", &fc, TensorRTPhase::kRUNTIME));
}

std::unique_ptr<IPluginV3> roundTrip(IPluginV3& plugin)
{
    auto* runtime = static_cast<IPluginV3OneRuntime*>(plugin.getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    PluginFieldCollection const* fc = runtime->getFieldsToSerialize();
    auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(kName, "1", ""));
    return std::unique_ptr<IPluginV3>(creator->createPlugin("lora_group_test", fc, TensorRTPhase::kRUNTIME));
}

TEST(LoraGroupPluginContract, AttributesSurviveSerializationAndSetAliasing)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    for (int32_t const inplace : {0, 1})
    {
        auto plugin = createGroup(3, inplace);
        ASSERT_NE(plugin, nullptr);
        auto restored = roundTrip(*plugin);
        ASSERT_NE(restored, nullptr);
        auto* core = static_cast<IPluginV3OneCore*>(restored->getCapabilityInterface(PluginCapabilityType::kCORE));
        auto* build = static_cast<IPluginV3OneBuildV2*>(restored->getCapabilityInterface(PluginCapabilityType::kBUILD));
        EXPECT_EQ(build->getNbOutputs(), 3);
        for (int32_t i = 0; i < 3; ++i)
        {
            EXPECT_EQ(build->getAliasedInput(i), inplace ? i : -1) << "inplace " << inplace << " output " << i;
        }
        EXPECT_STREQ(core->getPluginName(), kName);
    }
    EXPECT_EQ(createGroup(4, 1), nullptr); // at most three siblings per node
}

PluginTensorDesc desc2d(int64_t rows, int64_t cols)
{
    PluginTensorDesc d{};
    d.dims.nbDims = 2;
    d.dims.d[0] = rows;
    d.dims.d[1] = cols;
    d.type = DataType::kHALF;
    d.format = TensorFormat::kLINEAR;
    return d;
}

std::vector<__half> randomHalf(size_t n, float scale, std::mt19937& gen)
{
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<__half> v(n);
    for (auto& e : v)
    {
        e = __float2half(dist(gen));
    }
    return v;
}

void* toDevice(std::vector<__half> const& h)
{
    void* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(__half)), cudaSuccess);
    if (!h.empty())
    {
        EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);
    }
    return d;
}

//! Two targets (q-like N=64 at rank 8, and a rank-0 sibling N=32), M=12, K=128. The active
//! output must equal base + fp16(fp16(x@A)@B); the rank-0 output must equal its base exactly.
void runEnqueue(int32_t inplace)
{
    constexpr int32_t M = 12;
    constexpr int32_t K = 128;
    std::array<int32_t, 2> const N{64, 32};
    std::array<int32_t, 2> const R{8, 0};
    auto plugin = createGroup(2, inplace);
    ASSERT_NE(plugin, nullptr);
    auto* runtime = static_cast<IPluginV3OneRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    // Execution contexts attach a clone; that is where the barrier state is allocated.
    std::unique_ptr<IPluginV3> attached(runtime->attachToContext(nullptr));
    ASSERT_NE(attached, nullptr);
    auto* rt = static_cast<IPluginV3OneRuntime*>(attached->getCapabilityInterface(PluginCapabilityType::kRUNTIME));

    std::mt19937 gen(5);
    auto const x = randomHalf(static_cast<size_t>(M) * K, 0.5f, gen);
    std::array<std::vector<__half>, 2> A, B, base;
    for (int32_t i = 0; i < 2; ++i)
    {
        A[i] = randomHalf(static_cast<size_t>(K) * R[i], 0.1f, gen);
        B[i] = randomHalf(static_cast<size_t>(R[i]) * N[i], 0.1f, gen);
        base[i] = randomHalf(static_cast<size_t>(M) * N[i], 1.f, gen);
    }
    std::array<PluginTensorDesc, 7> in{desc2d(M, N[0]), desc2d(M, N[1]), desc2d(M, K), desc2d(K, R[0]),
        desc2d(K, R[1]), desc2d(R[0], N[0]), desc2d(R[1], N[1])};
    std::array<PluginTensorDesc, 2> outDesc{desc2d(M, N[0]), desc2d(M, N[1])};
    std::array<void*, 7> dIn{toDevice(base[0]), toDevice(base[1]), toDevice(x), toDevice(A[0]), toDevice(A[1]),
        toDevice(B[0]), toDevice(B[1])};
    std::array<void*, 2> dOut{};
    for (int32_t i = 0; i < 2; ++i)
    {
        dOut[i] = inplace ? dIn[i] : toDevice(std::vector<__half>(static_cast<size_t>(M) * N[i]));
    }
    ASSERT_EQ(rt->onShapeChange(in.data(), 7, outDesc.data(), 2), 0);
    void* workspace = nullptr;
    ASSERT_EQ(cudaMalloc(&workspace, 8 << 20), cudaSuccess);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(rt->enqueue(in.data(), outDesc.data(), dIn.data(), dOut.data(), workspace, stream), 0);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int32_t i = 0; i < 2; ++i)
    {
        std::vector<__half> got(static_cast<size_t>(M) * N[i]);
        ASSERT_EQ(cudaMemcpy(got.data(), dOut[i], got.size() * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
        for (int32_t m = 0; m < M; ++m)
        {
            std::vector<float> t(R[i]);
            for (int32_t r = 0; r < R[i]; ++r)
            {
                double s = 0;
                for (int32_t k = 0; k < K; ++k)
                {
                    s += double(__half2float(x[m * K + k])) * __half2float(A[i][k * R[i] + r]);
                }
                t[r] = __half2float(__float2half(static_cast<float>(s)));
            }
            for (int32_t n = 0; n < N[i]; ++n)
            {
                double s = 0;
                for (int32_t r = 0; r < R[i]; ++r)
                {
                    s += double(t[r]) * __half2float(B[i][r * N[i] + n]);
                }
                float const b = __half2float(base[i][m * N[i] + n]);
                float const want = R[i] == 0
                    ? b
                    : __half2float(__float2half(b + __half2float(__float2half(static_cast<float>(s)))));
                float const tol = R[i] == 0 ? 0.f : 2e-3f + 2e-3f * std::fabs(want);
                ASSERT_NEAR(__half2float(got[m * N[i] + n]), want, tol)
                    << "inplace " << inplace << " target " << i << " m " << m << " n " << n;
            }
        }
    }
    for (void* p : dIn)
    {
        cudaFree(p);
    }
    if (!inplace)
    {
        cudaFree(dOut[0]);
        cudaFree(dOut[1]);
    }
    cudaFree(workspace);
    cudaStreamDestroy(stream);
}

TEST(LoraGroupPluginEnqueue, InPlaceAddsActiveAndLeavesRankZeroUntouched)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    runEnqueue(1);
}

TEST(LoraGroupPluginEnqueue, CopyModeWritesEveryOutput)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    runEnqueue(0);
}

} // namespace
} // namespace trt_edgellm::plugins
