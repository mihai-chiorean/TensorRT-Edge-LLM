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
 * LoraShrinkPlugin at its TensorRT boundary: for targets sharing x, t_i = fp16(x @ A_i) as
 * [M, 16] with zeros past the rank, at decode and prefill row counts and for ranks that are and are
 * not multiples of 8; a target bound at rank 0 leaves its output untouched; repeated launches give
 * the same bits (the counters reset); a plugin recreated from its serialized fields runs up to the
 * serialized maximum M and refuses a larger M; a K that is not a multiple of 64 is refused at build.
 */

#include "testPluginLoader.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;

std::unique_ptr<IPluginV3> createShrink(int32_t numTargets)
{
    auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("LoraShrinkPlugin", "1", ""));
    if (creator == nullptr)
    {
        return nullptr;
    }
    std::array<PluginField, 1> const fields{PluginField("num_targets", &numTargets, PluginFieldType::kINT32, 1)};
    PluginFieldCollection const fc{1, fields.data()};
    return std::unique_ptr<IPluginV3>(creator->createPlugin("lora_shrink_test", &fc, TensorRTPhase::kBUILD));
}

DynamicPluginTensorDesc dyn(Dims minD, Dims maxD)
{
    DynamicPluginTensorDesc d{};
    d.desc.type = DataType::kHALF;
    d.desc.format = TensorFormat::kLINEAR;
    d.desc.dims = maxD;
    d.min = minD;
    d.opt = maxD;
    d.max = maxD;
    return d;
}

PluginTensorDesc desc(int64_t rows, int64_t cols)
{
    PluginTensorDesc d{};
    d.dims.nbDims = 2;
    d.dims.d[0] = rows;
    d.dims.d[1] = cols;
    d.type = DataType::kHALF;
    d.format = TensorFormat::kLINEAR;
    return d;
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

//! One attached plugin (profile max M 512) runs every M in `ms` in order on the same counters and
//! workspace. Before each launch all outputs are poisoned, so a launch that elects no final block
//! (a counter not reset by the previous launch) leaves NaNs and fails the reference check. With
//! `viaSerialization`, the attached plugin is the one the runtime gets: recreated by the creator from
//! the configured plugin's serialized fields, never configured itself.
void runShrinkSequence(int32_t K, std::vector<int32_t> const& ms, std::array<int32_t, 3> const& R = {16, 0, 8},
    bool viaSerialization = false)
{
    auto plugin = createShrink(3);
    ASSERT_NE(plugin, nullptr);
    constexpr int32_t kMaxM = 512;
    std::array<DynamicPluginTensorDesc, 4> inD{dyn(Dims{2, {1, K}}, Dims{2, {kMaxM, K}}),
        dyn(Dims{2, {K, 0}}, Dims{2, {K, 16}}), dyn(Dims{2, {K, 0}}, Dims{2, {K, 16}}),
        dyn(Dims{2, {K, 0}}, Dims{2, {K, 16}})};
    std::array<DynamicPluginTensorDesc, 3> outD{dyn(Dims{2, {1, 16}}, Dims{2, {kMaxM, 16}}),
        dyn(Dims{2, {1, 16}}, Dims{2, {kMaxM, 16}}), dyn(Dims{2, {1, 16}}, Dims{2, {kMaxM, 16}})};
    auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
    ASSERT_EQ(build->configurePlugin(inD.data(), 4, outD.data(), 3), 0);
    size_t const ws = build->getWorkspaceSize(inD.data(), 4, outD.data(), 3);
    auto* runtime = static_cast<IPluginV3OneRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    std::unique_ptr<IPluginV3> restored;
    if (viaSerialization)
    {
        auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("LoraShrinkPlugin", "1", ""));
        ASSERT_NE(creator, nullptr);
        restored.reset(
            creator->createPlugin("lora_shrink_test", runtime->getFieldsToSerialize(), TensorRTPhase::kRUNTIME));
        ASSERT_NE(restored, nullptr);
        runtime = static_cast<IPluginV3OneRuntime*>(restored->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    }
    std::unique_ptr<IPluginV3> attached(runtime->attachToContext(nullptr));
    ASSERT_NE(attached, nullptr);
    auto* rt = static_cast<IPluginV3OneRuntime*>(attached->getCapabilityInterface(PluginCapabilityType::kRUNTIME));

    std::mt19937 gen(static_cast<uint32_t>(K));
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<__half> x(size_t(kMaxM) * K);
    for (auto& v : x)
    {
        v = __float2half(dist(gen) * 0.5f);
    }
    std::array<std::vector<__half>, 3> A;
    std::array<void*, 4> dIn{};
    dIn[0] = toDevice(x);
    for (int32_t i = 0; i < 3; ++i)
    {
        A[i].resize(size_t(K) * R[i]);
        for (auto& v : A[i])
        {
            v = __float2half(dist(gen) * 0.05f);
        }
        dIn[1 + i] = toDevice(A[i]);
    }
    std::array<void*, 3> dOutput{};
    for (auto& o : dOutput)
    {
        ASSERT_EQ(cudaMalloc(&o, size_t(kMaxM) * 16 * 2), cudaSuccess);
    }
    void* workspace = nullptr;
    ASSERT_EQ(cudaMalloc(&workspace, ws), cudaSuccess);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    std::vector<std::array<std::vector<__half>, 3>> firstByM(kMaxM + 1);
    for (int32_t const M : ms)
    {
        std::array<PluginTensorDesc, 4> in{desc(M, K), desc(K, R[0]), desc(K, R[1]), desc(K, R[2])};
        std::array<PluginTensorDesc, 3> out{desc(M, 16), desc(M, 16), desc(M, 16)};
        ASSERT_EQ(rt->onShapeChange(in.data(), 4, out.data(), 3), 0);
        for (auto& o : dOutput)
        {
            ASSERT_EQ(cudaMemsetAsync(o, 0xFF, size_t(M) * 16 * 2, stream), cudaSuccess);
        }
        ASSERT_EQ(rt->enqueue(in.data(), out.data(), const_cast<void const* const*>(dIn.data()), dOutput.data(),
                      workspace, stream),
            0);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        for (int32_t i = 0; i < 3; ++i)
        {
            std::vector<__half> got(size_t(M) * 16);
            ASSERT_EQ(cudaMemcpy(got.data(), dOutput[i], got.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
            // A repeated M must reproduce its first result bit for bit (the order is fixed).
            if (firstByM[M][i].empty())
            {
                firstByM[M][i] = got;
            }
            else
            {
                EXPECT_EQ(std::memcmp(got.data(), firstByM[M][i].data(), got.size() * 2), 0)
                    << "M " << M << " target " << i;
            }
            for (int32_t m = 0; m < M; ++m)
            {
                for (int32_t j = 0; j < 16; ++j)
                {
                    uint16_t bits;
                    std::memcpy(&bits, &got[size_t(m) * 16 + j], 2);
                    if (R[i] == 0)
                    {
                        ASSERT_EQ(bits, 0xFFFF) << "rank-0 target output was written, M " << M;
                        continue;
                    }
                    double s = 0;
                    for (int32_t k = 0; k < K && j < R[i]; ++k)
                    {
                        s += double(__half2float(x[size_t(m) * K + k])) * __half2float(A[i][size_t(k) * R[i] + j]);
                    }
                    float const g = __half2float(got[size_t(m) * 16 + j]);
                    ASSERT_NEAR(g, s, 1e-3 + 2e-3 * std::fabs(s))
                        << "M " << M << " target " << i << " m " << m << " j " << j;
                }
            }
        }
    }
    if (viaSerialization)
    {
        std::array<PluginTensorDesc, 4> in{desc(kMaxM + 1, K), desc(K, R[0]), desc(K, R[1]), desc(K, R[2])};
        std::array<PluginTensorDesc, 3> out{desc(kMaxM + 1, 16), desc(kMaxM + 1, 16), desc(kMaxM + 1, 16)};
        EXPECT_NE(rt->enqueue(in.data(), out.data(), const_cast<void const* const*>(dIn.data()), dOutput.data(),
                      workspace, stream),
            0)
            << "M above the serialized max_m must be refused";
    }
    for (void* p : dIn)
    {
        cudaFree(p);
    }
    for (void* p : dOutput)
    {
        cudaFree(p);
    }
    cudaFree(workspace);
    cudaStreamDestroy(stream);
}

TEST(LoraShrinkPlugin, DecodeAndPrefillRowsAndRankZero)
{
    // Given one attached shrink plugin with three targets at ranks 16, 0 and 8 that share x,
    // when it runs decode (M <= 8) and tensor-core (M >= 9) row counts alternately, each repeated, with every output
    // poisoned before each launch,
    // then each positive-rank target's output equals x @ A within FP64 tolerance with zeros past its rank, a repeated
    // M reproduces its first result bit for bit, and the rank-0 target's output is never written.
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    // Decode and prefill buckets alternate on one instance, each M repeated.
    runShrinkSequence(2560, {1, 1, 5, 80, 12, 1, 80, 5, 12, 200});
    runShrinkSequence(10240, {5, 1, 5});
    runShrinkSequence(2048 + 64, {80, 1, 80});
}

TEST(LoraShrinkPlugin, RanksThatAreNotMultiplesOfEight)
{
    // Given targets whose ranks are not multiples of 8 (12, 4, 3 and 1, 7, 15), which stage A element by element,
    // when the plugin runs decode and tensor-core row counts,
    // then each output equals x @ A within FP64 tolerance in every rank column and is zero past the rank.
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    runShrinkSequence(2560, {1, 8, 12, 80, 1}, {12, 4, 3});
    runShrinkSequence(10240, {3, 64, 3}, {1, 7, 15});
}

TEST(LoraShrinkPlugin, RecreatedFromSerializedFields)
{
    // Given a plugin configured for a profile with maximum M 512, and a second plugin created only from the first
    // one's serialized fields (the runtime's path),
    // when the second plugin is attached and run at M from 1 to 512 and then at M = 513,
    // then its outputs equal x @ A up to M = 512 and it refuses M = 513.
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    runShrinkSequence(2560, {1, 5, 200, 512, 1}, {16, 0, 8}, /*viaSerialization=*/true);
}

TEST(LoraShrinkPlugin, RefusesKThatIsNotAMultipleOf64AtBuild)
{
    // Given a shrink plugin,
    // when it is configured with K = 2560 + 32 (not a multiple of 64),
    // then configurePlugin refuses the profile, and with K = 2560 it accepts it.
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    for (int32_t const K : {2560 + 32, 2560})
    {
        auto plugin = createShrink(1);
        ASSERT_NE(plugin, nullptr);
        std::array<DynamicPluginTensorDesc, 2> inD{
            dyn(Dims{2, {1, K}}, Dims{2, {64, K}}), dyn(Dims{2, {K, 0}}, Dims{2, {K, 16}})};
        std::array<DynamicPluginTensorDesc, 1> outD{dyn(Dims{2, {1, 16}}, Dims{2, {64, 16}})};
        auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
        int32_t const status = build->configurePlugin(inD.data(), 2, outD.data(), 1);
        if (K % 64 != 0)
        {
            EXPECT_NE(status, 0) << "K " << K;
        }
        else
        {
            EXPECT_EQ(status, 0) << "K " << K;
        }
    }
}

} // namespace
} // namespace trt_edgellm::plugins
