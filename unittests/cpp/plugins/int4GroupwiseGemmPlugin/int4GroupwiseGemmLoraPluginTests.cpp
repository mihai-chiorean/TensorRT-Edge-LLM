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
 * The fused INT4 LoRA path at its TensorRT boundary:
 * - Int4GroupwiseGemmLoraPlugin: a rank-0 binding produces exactly the Int4GroupwiseGemmPluginV2
 *   output; a positive rank produces fp16(float(base) + float(fp16(fma chain of t16 @ B))) bit for
 *   bit on both the GEMV and the GEMM routes; a profile above the rank cap is refused at build;
 *   the attributes survive serialization.
 * - LoraShrinkPlugin: t_i = fp16(x @ A_i) for each target (zeros past its rank), and a rank-0
 *   target's output is left untouched.
 * The shrink's exact summation tree is checked by kernelSrcs/int4_fp16_gemm_cutedsl/test_lora_epilogue.py.
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

#if defined(CUTE_DSL_INT4_FP16_GEMM_ENABLED) && defined(CUTE_DSL_INT4_FP16_GEMM_LORA_ENABLED)

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;

constexpr char const* kLoraName{"Int4GroupwiseGemmLoraPlugin"};
constexpr char const* kV2Name{"Int4GroupwiseGemmPluginV2"};

#ifdef CUTE_DSL_INT4_FP16_GEMM_G32_ENABLED
constexpr int32_t kGroup{32};
#else
constexpr int32_t kGroup{128};
#endif

int32_t divUp(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

std::unique_ptr<IPluginV3> create(char const* name, int32_t n, int32_t k, PluginFieldCollection const* extra = nullptr)
{
    auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(name, "1", ""));
    if (creator == nullptr)
    {
        return nullptr;
    }
    if (extra != nullptr)
    {
        return std::unique_ptr<IPluginV3>(creator->createPlugin("lora_int4_test", extra, TensorRTPhase::kRUNTIME));
    }
    int32_t gs = kGroup;
    std::array<PluginField, 3> const fields{PluginField("gemm_n", &n, PluginFieldType::kINT32, 1),
        PluginField("gemm_k", &k, PluginFieldType::kINT32, 1),
        PluginField("group_size", &gs, PluginFieldType::kINT32, 1)};
    PluginFieldCollection const fc{static_cast<int32_t>(fields.size()), fields.data()};
    return std::unique_ptr<IPluginV3>(creator->createPlugin("lora_int4_test", &fc, TensorRTPhase::kBUILD));
}

template <typename T>
T* iface(IPluginV3& p, PluginCapabilityType t)
{
    return static_cast<T*>(p.getCapabilityInterface(t));
}

DynamicPluginTensorDesc dyn(DataType type, Dims minD, Dims maxD)
{
    DynamicPluginTensorDesc d{};
    d.desc.type = type;
    d.desc.format = TensorFormat::kLINEAR;
    d.desc.dims = maxD;
    d.min = minD;
    d.opt = maxD;
    d.max = maxD;
    return d;
}

PluginTensorDesc desc(DataType type, int64_t rows, int64_t cols)
{
    PluginTensorDesc d{};
    d.dims.nbDims = 2;
    d.dims.d[0] = rows;
    d.dims.d[1] = cols;
    d.type = type;
    d.format = TensorFormat::kLINEAR;
    return d;
}

//! Configure with profile max M and max rank; returns configurePlugin's status.
int32_t configure(IPluginV3& p, int32_t n, int32_t k, int32_t maxM, int32_t maxRank, bool lora)
{
    int32_t const rows = divUp(n, 128) * divUp(k, 64) * 8;
    std::vector<DynamicPluginTensorDesc> in{dyn(DataType::kHALF, Dims{2, {1, k}}, Dims{2, {maxM, k}}),
        dyn(DataType::kINT8, Dims{2, {rows, 512}}, Dims{2, {rows, 512}}),
        dyn(DataType::kHALF, Dims{2, {k / kGroup, n}}, Dims{2, {k / kGroup, n}})};
    if (lora)
    {
        in.push_back(dyn(DataType::kHALF, Dims{2, {1, 16}}, Dims{2, {maxM, 16}}));
        in.push_back(dyn(DataType::kHALF, Dims{2, {0, n}}, Dims{2, {maxRank, n}}));
    }
    DynamicPluginTensorDesc out = dyn(DataType::kHALF, Dims{2, {1, n}}, Dims{2, {maxM, n}});
    auto* build = iface<IPluginV3OneBuild>(p, PluginCapabilityType::kBUILD);
    return build->configurePlugin(in.data(), static_cast<int32_t>(in.size()), &out, 1);
}

struct DeviceMem
{
    void* p{nullptr};
    explicit DeviceMem(size_t bytes)
    {
        EXPECT_EQ(cudaMalloc(&p, bytes < 16 ? 16 : bytes), cudaSuccess);
    }
    ~DeviceMem()
    {
        cudaFree(p);
    }
    DeviceMem(DeviceMem const&) = delete;
    DeviceMem& operator=(DeviceMem const&) = delete;
};

template <typename T>
void upload(DeviceMem& d, std::vector<T> const& h)
{
    ASSERT_EQ(cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
}

std::vector<__half> download(DeviceMem const& d, size_t n)
{
    std::vector<__half> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d.p, n * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
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

//! Runs V2 and the LoRA form on the same weights and compares. Ranks are run in the order given
//! on one attached plugin, so rank transitions (0 -> r -> 0) run on the same instance.
void runCase(int32_t M, int32_t N, int32_t K, std::vector<int32_t> const& ranks)
{
    std::mt19937 gen(static_cast<uint32_t>(M * 131 + N + K));
    int32_t const rows = divUp(N, 128) * divUp(K, 64) * 8;
    std::vector<int8_t> w(static_cast<size_t>(rows) * 512);
    std::uniform_int_distribution<int32_t> byte(-128, 127);
    for (auto& b : w)
    {
        b = static_cast<int8_t>(byte(gen));
    }
    std::vector<__half> scales(static_cast<size_t>(K / kGroup) * N);
    std::uniform_real_distribution<float> sd(0.01f, 0.05f);
    for (auto& s : scales)
    {
        s = __float2half(sd(gen));
    }
    auto const act = randomHalf(static_cast<size_t>(M) * K, 1.f, gen);
    constexpr int32_t kMaxRank = 16;
    auto const B = randomHalf(static_cast<size_t>(kMaxRank) * N, 0.05f, gen);

    DeviceMem dAct(act.size() * 2), dW(w.size()), dS(scales.size() * 2), dBase(size_t(M) * N * 2),
        dOut(size_t(M) * N * 2);
    upload(dAct, act);
    upload(dW, w);
    upload(dS, scales);

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto v2 = create(kV2Name, N, K);
    ASSERT_NE(v2, nullptr);
    ASSERT_EQ(configure(*v2, N, K, 64, 0, false), 0);
    std::unique_ptr<IPluginV3> v2rt(iface<IPluginV3OneRuntime>(*v2, PluginCapabilityType::kRUNTIME)->attachToContext(nullptr));
    auto* v2r = iface<IPluginV3OneRuntime>(*v2rt, PluginCapabilityType::kRUNTIME);
    std::array<PluginTensorDesc, 3> v2In{desc(DataType::kHALF, M, K), desc(DataType::kINT8, rows, 512),
        desc(DataType::kHALF, K / kGroup, N)};
    PluginTensorDesc outD = desc(DataType::kHALF, M, N);
    ASSERT_EQ(v2r->onShapeChange(v2In.data(), 3, &outD, 1), 0);
    std::array<void const*, 3> v2Ptr{dAct.p, dW.p, dS.p};
    std::array<void*, 1> basePtr{dBase.p};
    ASSERT_EQ(v2r->enqueue(v2In.data(), &outD, v2Ptr.data(), basePtr.data(), nullptr, stream), 0);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    auto const base = download(dBase, size_t(M) * N);

    auto lora = create(kLoraName, N, K);
    ASSERT_NE(lora, nullptr);
    ASSERT_EQ(configure(*lora, N, K, 64, kMaxRank, true), 0);
    std::unique_ptr<IPluginV3> rtPlugin(
        iface<IPluginV3OneRuntime>(*lora, PluginCapabilityType::kRUNTIME)->attachToContext(nullptr));
    auto* rt = iface<IPluginV3OneRuntime>(*rtPlugin, PluginCapabilityType::kRUNTIME);

    for (int32_t const r : ranks)
    {
        // t16 as LoraShrinkPlugin writes it: [M, 16], zeros past the rank.
        std::vector<__half> t(static_cast<size_t>(M) * 16, __float2half(0.f));
        std::uniform_real_distribution<float> td(-2.f, 2.f);
        for (int32_t m = 0; m < M; ++m)
        {
            for (int32_t j = 0; j < r; ++j)
            {
                t[size_t(m) * 16 + j] = __float2half(td(gen));
            }
        }
        std::vector<__half> Br(B.begin(), B.begin() + size_t(r) * N);
        // Rank-0 bindings are distinct non-null allocations, as the runtime binds them.
        DeviceMem dT(t.size() * 2), dB(std::max<size_t>(Br.size(), 1) * 2);
        upload(dT, t);
        if (r > 0)
        {
            upload(dB, Br);
        }
        ASSERT_EQ(cudaMemset(dOut.p, 0xFF, size_t(M) * N * 2), cudaSuccess); // poison (fp16 NaN)
        std::array<PluginTensorDesc, 5> in{v2In[0], v2In[1], v2In[2], desc(DataType::kHALF, M, 16),
            desc(DataType::kHALF, r, N)};
        ASSERT_EQ(rt->onShapeChange(in.data(), 5, &outD, 1), 0);
        std::array<void const*, 5> ptr{dAct.p, dW.p, dS.p, dT.p, dB.p};
        std::array<void*, 1> outPtr{dOut.p};
        ASSERT_EQ(rt->enqueue(in.data(), &outD, ptr.data(), outPtr.data(), nullptr, stream), 0) << "rank " << r;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        auto const got = download(dOut, size_t(M) * N);

        if (r == 0)
        {
            EXPECT_EQ(std::memcmp(got.data(), base.data(), got.size() * 2), 0)
                << "rank 0 must equal Int4GroupwiseGemmPluginV2 bit for bit, M " << M << " N " << N;
            continue;
        }
        int64_t bad = 0;
        for (int32_t m = 0; m < M; ++m)
        {
            for (int32_t n = 0; n < N; ++n)
            {
                float d = 0.f;
                for (int32_t j = 0; j < r; ++j)
                {
                    d = std::fma(__half2float(t[size_t(m) * 16 + j]), __half2float(Br[size_t(j) * N + n]), d);
                }
                __half const want = __float2half(__half2float(base[size_t(m) * N + n]) + __half2float(__float2half(d)));
                __half const g = got[size_t(m) * N + n];
                if (std::memcmp(&g, &want, 2) != 0 && bad++ < 5)
                {
                    ADD_FAILURE() << "rank " << r << " M " << M << " N " << N << " m " << m << " n " << n << " got "
                                  << __half2float(g) << " want " << __half2float(want);
                }
            }
        }
        EXPECT_EQ(bad, 0);
    }
    cudaStreamDestroy(stream);
}

TEST(Int4GroupwiseGemmLoraPlugin, GemvRouteMatchesV2AtRankZeroAndAddsTheDelta)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    runCase(1, 512, 2560, {0, 16, 0, 3});
    runCase(5, 256, 1024, {16, 0});
}

TEST(Int4GroupwiseGemmLoraPlugin, GemmRouteMatchesV2AtRankZeroAndAddsTheDelta)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    runCase(1, 2048, 2560, {16, 0, 7}); // N * K above the group-32 M = 1 GEMM threshold
    runCase(40, 384, 1024, {0, 16});
}

TEST(Int4GroupwiseGemmLoraPlugin, RefusesUnsupportedShapesAndSerializes)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    auto lora = create(kLoraName, 256, 1024);
    ASSERT_NE(lora, nullptr);
    EXPECT_NE(configure(*lora, 256, 1024, 64, 32, true), 0); // rank above the fused cap
    ASSERT_EQ(configure(*lora, 256, 1024, 8192, 16, true), 0);

    auto oddN = create(kLoraName, 130, 1024); // B rows would not be 16-byte aligned
    ASSERT_NE(oddN, nullptr);
    EXPECT_NE(configure(*oddN, 130, 1024, 64, 16, true), 0);

    // A deserialized plugin carries every serialized field unchanged (N, K, group size and the
    // lock storage sized from the profile above).
    auto* runtime = iface<IPluginV3OneRuntime>(*lora, PluginCapabilityType::kRUNTIME);
    PluginFieldCollection const* fields = runtime->getFieldsToSerialize();
    auto restored = create(kLoraName, 0, 0, fields);
    ASSERT_NE(restored, nullptr);
    EXPECT_STREQ(iface<IPluginV3OneCore>(*restored, PluginCapabilityType::kCORE)->getPluginName(), kLoraName);
    PluginFieldCollection const* again
        = iface<IPluginV3OneRuntime>(*restored, PluginCapabilityType::kRUNTIME)->getFieldsToSerialize();
    ASSERT_EQ(again->nbFields, fields->nbFields);
    auto intField = [&](char const* name) -> int64_t {
        for (int32_t i = 0; i < again->nbFields; ++i)
        {
            PluginField const& f = again->fields[i];
            if (std::strcmp(f.name, name) == 0)
            {
                return f.type == PluginFieldType::kINT64 ? *static_cast<int64_t const*>(f.data)
                                                         : *static_cast<int32_t const*>(f.data);
            }
        }
        ADD_FAILURE() << "field " << name << " not serialized";
        return -1;
    };
    EXPECT_EQ(intField("gemm_n"), 256);
    EXPECT_EQ(intField("gemm_k"), 1024);
    EXPECT_EQ(intField("group_size"), kGroup);
    EXPECT_GE(intField("max_lock_workspace_bytes"), 0);
    for (int32_t i = 0; i < fields->nbFields; ++i)
    {
        PluginField const& a = fields->fields[i];
        PluginField const& b = again->fields[i];
        EXPECT_STREQ(a.name, b.name);
        ASSERT_EQ(a.type, b.type);
        ASSERT_EQ(a.length, b.length);
        size_t const bytes = a.type == PluginFieldType::kINT64 ? 8 : 4;
        EXPECT_EQ(std::memcmp(a.data, b.data, bytes * static_cast<size_t>(a.length)), 0) << a.name;
    }
}

} // namespace
} // namespace trt_edgellm::plugins

#endif
