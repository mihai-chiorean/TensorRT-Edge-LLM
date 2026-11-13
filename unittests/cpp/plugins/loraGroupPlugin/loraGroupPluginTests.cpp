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
#include <cstring>
#include <initializer_list>
#include <memory>
#include <random>
#include <vector>

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;

constexpr char const* kName{"LoraGroupPlugin"};

std::unique_ptr<IPluginV3> createGroup(int32_t numTargets, int32_t inplace, int32_t factorFormat = -1)
{
    auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(kName, "1", ""));
    if (creator == nullptr)
    {
        return nullptr;
    }
    // factorFormat < 0: attribute absent, as in engines built before it existed.
    std::array<PluginField, 3> const fields{PluginField("num_targets", &numTargets, PluginFieldType::kINT32, 1),
        PluginField("inplace", &inplace, PluginFieldType::kINT32, 1),
        PluginField("lora_factor_format", &factorFormat, PluginFieldType::kINT32, 1)};
    PluginFieldCollection const fc{factorFormat < 0 ? 2 : 3, fields.data()};
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
    std::array<PluginTensorDesc, 7> in{desc2d(M, N[0]), desc2d(M, N[1]), desc2d(M, K), desc2d(K, R[0]), desc2d(K, R[1]),
        desc2d(R[0], N[0]), desc2d(R[1], N[1])};
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
                float const want
                    = R[i] == 0 ? b : __half2float(__float2half(b + __half2float(__float2half(static_cast<float>(s)))));
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

PluginTensorDesc desc(DataType type, std::initializer_list<int64_t> dims)
{
    PluginTensorDesc d{};
    d.dims.nbDims = static_cast<int32_t>(dims.size());
    int32_t i = 0;
    for (int64_t v : dims)
    {
        d.dims.d[i++] = v;
    }
    d.type = type;
    d.format = TensorFormat::kLINEAR;
    return d;
}

template <typename T>
void* toDeviceBytes(std::vector<T> const& h)
{
    void* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, std::max<size_t>(h.size() * sizeof(T), 16)), cudaSuccess);
    if (!h.empty())
    {
        EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    }
    return d;
}

//! int8_per_channel_v1 factor with column scales, and the FP16 values the runtime registers for it
//! on an FP16 engine (fp16(float(code) * float(scale))).
void quantizedFactor(int32_t rows, int32_t cols, std::mt19937& gen, std::vector<int8_t>& codes,
    std::vector<__half>& scales, std::vector<__half>& dequantized)
{
    std::uniform_int_distribution<int32_t> code(-127, 127);
    std::uniform_real_distribution<float> scale(1e-3f, 4e-3f);
    codes.resize(static_cast<size_t>(rows) * cols);
    scales.resize(cols);
    dequantized.resize(codes.size());
    for (auto& s : scales)
    {
        s = __float2half(scale(gen));
    }
    for (size_t i = 0; i < codes.size(); ++i)
    {
        codes[i] = static_cast<int8_t>(code(gen));
        dequantized[i] = __float2half_rn(static_cast<float>(codes[i]) * __half2float(scales[i % cols]));
    }
}

TEST(LoraGroupPluginContract, FactorFormatSurvivesSerializationAndRejectsUnknownValues)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    auto int8 = createGroup(2, 1, 1);
    ASSERT_NE(int8, nullptr);
    auto restored = roundTrip(*int8);
    ASSERT_NE(restored, nullptr);
    auto* build = static_cast<IPluginV3OneBuildV2*>(restored->getCapabilityInterface(PluginCapabilityType::kBUILD));
    // INT8 layout: 2 bases, x, 2 A (INT8), 2 B (INT8), 2 A scales, 2 B scales, then 2 outputs.
    std::array<DynamicPluginTensorDesc, 13> io{};
    auto set = [&](int32_t pos, DataType type, int32_t nbDims) {
        io[pos].desc.type = type;
        io[pos].desc.format = TensorFormat::kLINEAR;
        io[pos].desc.dims.nbDims = nbDims;
    };
    for (int32_t pos = 0; pos < 3; ++pos)
    {
        set(pos, DataType::kHALF, 2);
    }
    for (int32_t pos = 3; pos < 7; ++pos)
    {
        set(pos, DataType::kINT8, 2);
    }
    for (int32_t pos = 7; pos < 11; ++pos)
    {
        set(pos, DataType::kHALF, 1);
    }
    set(11, DataType::kHALF, 2);
    set(12, DataType::kHALF, 2);
    for (int32_t pos = 0; pos < 13; ++pos)
    {
        EXPECT_TRUE(build->supportsFormatCombination(pos, io.data(), 11, 2)) << "position " << pos;
    }
    set(3, DataType::kHALF, 2); // an FP16 A factor on an INT8 node
    EXPECT_FALSE(build->supportsFormatCombination(3, io.data(), 11, 2));
    EXPECT_EQ(createGroup(2, 1, 2), nullptr);
}

//! The INT8-format plugin gives bit-identical output to the FP16-format plugin fed the same factors
//! dequantized on the host, at decode width (M = 1) and short-prefill width (M = 28), and leaves a
//! rank-0 sibling untouched.
TEST(LoraGroupPluginEnqueue, Int8FactorsMatchHostDequantizedFp16)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    constexpr int32_t K = 256;
    std::array<int32_t, 2> const N{64, 32};
    std::array<int32_t, 2> const R{16, 0};
    for (int32_t const M : {1, 28})
    {
        std::mt19937 gen(11 + M);
        auto const x = randomHalf(static_cast<size_t>(M) * K, 0.5f, gen);
        std::array<std::vector<int8_t>, 2> aq, bq;
        std::array<std::vector<__half>, 2> as, bs, ad, bd, base;
        for (int32_t i = 0; i < 2; ++i)
        {
            quantizedFactor(K, R[i], gen, aq[i], as[i], ad[i]);
            quantizedFactor(R[i], N[i], gen, bq[i], bs[i], bd[i]);
            base[i] = randomHalf(static_cast<size_t>(M) * N[i], 1.f, gen);
        }
        std::array<std::vector<__half>, 2> outs;
        for (int32_t const format : {0, 1})
        {
            auto plugin = createGroup(2, 1, format);
            ASSERT_NE(plugin, nullptr);
            auto* runtime
                = static_cast<IPluginV3OneRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
            std::unique_ptr<IPluginV3> attached(runtime->attachToContext(nullptr));
            ASSERT_NE(attached, nullptr);
            auto* rt
                = static_cast<IPluginV3OneRuntime*>(attached->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
            DataType const factorType = format == 1 ? DataType::kINT8 : DataType::kHALF;
            std::vector<PluginTensorDesc> in{desc(DataType::kHALF, {M, N[0]}), desc(DataType::kHALF, {M, N[1]}),
                desc(DataType::kHALF, {M, K}), desc(factorType, {K, R[0]}), desc(factorType, {K, R[1]}),
                desc(factorType, {R[0], N[0]}), desc(factorType, {R[1], N[1]})};
            std::vector<void*> dIn{toDeviceBytes(base[0]), toDeviceBytes(base[1]), toDeviceBytes(x)};
            if (format == 1)
            {
                in.push_back(desc(DataType::kHALF, {R[0]}));
                in.push_back(desc(DataType::kHALF, {R[1]}));
                in.push_back(desc(DataType::kHALF, {N[0]}));
                in.push_back(desc(DataType::kHALF, {N[1]}));
                for (void* p : {toDeviceBytes(aq[0]), toDeviceBytes(aq[1]), toDeviceBytes(bq[0]), toDeviceBytes(bq[1]),
                         toDeviceBytes(as[0]), toDeviceBytes(as[1]), toDeviceBytes(bs[0]), toDeviceBytes(bs[1])})
                {
                    dIn.push_back(p);
                }
            }
            else
            {
                for (void* p : {toDeviceBytes(ad[0]), toDeviceBytes(ad[1]), toDeviceBytes(bd[0]), toDeviceBytes(bd[1])})
                {
                    dIn.push_back(p);
                }
            }
            std::array<PluginTensorDesc, 2> outDesc{desc(DataType::kHALF, {M, N[0]}), desc(DataType::kHALF, {M, N[1]})};
            std::array<void*, 2> dOut{dIn[0], dIn[1]};
            int32_t const nbIn = static_cast<int32_t>(in.size());
            ASSERT_EQ(rt->onShapeChange(in.data(), nbIn, outDesc.data(), 2), 0);
            void* workspace = nullptr;
            ASSERT_EQ(cudaMalloc(&workspace, 16 << 20), cudaSuccess);
            cudaStream_t stream{};
            ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_EQ(rt->enqueue(in.data(), outDesc.data(), dIn.data(), dOut.data(), workspace, stream), 0);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            outs[format].resize(static_cast<size_t>(M) * (N[0] + N[1]));
            ASSERT_EQ(cudaMemcpy(outs[format].data(), dOut[0], static_cast<size_t>(M) * N[0] * sizeof(__half),
                          cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpy(outs[format].data() + static_cast<size_t>(M) * N[0], dOut[1],
                          static_cast<size_t>(M) * N[1] * sizeof(__half), cudaMemcpyDeviceToHost),
                cudaSuccess);
            for (void* p : dIn)
            {
                cudaFree(p);
            }
            cudaFree(workspace);
            cudaStreamDestroy(stream);
        }
        size_t changed = 0;
        for (size_t e = 0; e < outs[0].size(); ++e)
        {
            uint16_t a = 0;
            uint16_t b = 0;
            std::memcpy(&a, &outs[0][e], 2);
            std::memcpy(&b, &outs[1][e], 2);
            ASSERT_EQ(a, b) << "M " << M << " element " << e;
            if (e < static_cast<size_t>(M) * N[0])
            {
                uint16_t o = 0;
                std::memcpy(&o, &base[0][e], 2);
                changed += a != o;
            }
            else
            {
                uint16_t o = 0;
                std::memcpy(&o, &base[1][e - static_cast<size_t>(M) * N[0]], 2);
                ASSERT_EQ(a, o) << "rank-0 sibling changed, M " << M;
            }
        }
        EXPECT_GT(changed, static_cast<size_t>(M) * N[0] / 4) << "the delta was not applied";
    }
}

} // namespace
} // namespace trt_edgellm::plugins
