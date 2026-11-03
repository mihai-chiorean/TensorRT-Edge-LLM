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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "common/cudaUtils.h"
#include "kernels/gdnKernels/gdnPrefillSequential.h"
#include "testUtils.h"
#ifdef CUTE_DSL_GDN_ENABLED
#include "kernels/gdnKernels/cuteDslGDNRunner.h"
#endif

using namespace trt_edgellm;

namespace
{

constexpr int32_t kDim{128};

float lcg(uint32_t& s)
{
    s = s * 1664525U + 1013904223U;
    return (static_cast<float>(s >> 8) / static_cast<float>(1U << 24)) - 0.5F;
}

struct Case
{
    int32_t n;
    int32_t seqLen;
    int32_t h;
    int32_t hv;
    int32_t poolRows;
    std::vector<int32_t> contextLengths;
    std::vector<int32_t> stateIndices; // empty: row i_n
};

struct HostData
{
    std::vector<__half> q, k, v, a, b, dtBias;
    std::vector<float> logA, h0;
};

HostData makeInputs(Case const& c, uint32_t seed)
{
    size_t const qk = static_cast<size_t>(c.n) * c.seqLen * c.h * kDim;
    size_t const vo = static_cast<size_t>(c.n) * c.seqLen * c.hv * kDim;
    size_t const ab = static_cast<size_t>(c.n) * c.seqLen * c.hv;
    HostData d;
    d.q.resize(qk);
    d.k.resize(qk);
    d.v.resize(vo);
    d.a.resize(ab);
    d.b.resize(ab);
    d.dtBias.resize(c.hv);
    d.logA.resize(c.hv);
    d.h0.resize(static_cast<size_t>(c.poolRows) * c.hv * kDim * kDim);
    for (auto& x : d.q)
        x = __float2half_rn(4.0F * lcg(seed));
    for (auto& x : d.k)
        x = __float2half_rn(4.0F * lcg(seed));
    for (auto& x : d.v)
        x = __float2half_rn(2.0F * lcg(seed));
    for (auto& x : d.a)
        x = __float2half_rn(8.0F * lcg(seed));
    for (auto& x : d.b)
        x = __float2half_rn(8.0F * lcg(seed));
    for (int32_t i = 0; i < c.hv; ++i)
    {
        d.dtBias[i] = __float2half_rn(lcg(seed));
        d.logA[i] = 2.0F * lcg(seed);
    }
    for (auto& x : d.h0)
        x = 0.2F * lcg(seed);
    return d;
}

// Double-precision reference of the gated delta rule with the QK L2 norm (gdn_prefill semantics).
void reference(Case const& c, HostData const& d, std::vector<float>& h0, std::vector<float>& o)
{
    double const scale = 1.0 / std::sqrt(static_cast<double>(kDim));
    o.assign(static_cast<size_t>(c.n) * c.seqLen * c.hv * kDim, 0.0F);
    for (int32_t iN = 0; iN < c.n; ++iN)
    {
        int32_t const slot = c.stateIndices.empty() ? iN : c.stateIndices[iN];
        int32_t const valid = std::max(0, std::min(c.seqLen, c.contextLengths[iN]));
        for (int32_t iHv = 0; iHv < c.hv; ++iHv)
        {
            int32_t const iH = iHv / (c.hv / c.h);
            float* const hs = h0.data() + (static_cast<size_t>(slot) * c.hv + iHv) * kDim * kDim;
            std::vector<double> H(hs, hs + kDim * kDim);
            for (int32_t t = 0; t < valid; ++t)
            {
                size_t const tok = static_cast<size_t>(iN) * c.seqLen + t;
                __half const* kr = d.k.data() + (tok * c.h + iH) * kDim;
                __half const* qr = d.q.data() + (tok * c.h + iH) * kDim;
                double nk = 1e-6;
                double nq = 1e-6;
                for (int32_t i = 0; i < kDim; ++i)
                {
                    nk += static_cast<double>(__half2float(kr[i])) * __half2float(kr[i]);
                    nq += static_cast<double>(__half2float(qr[i])) * __half2float(qr[i]);
                }
                std::vector<double> kn(kDim), qn(kDim);
                for (int32_t i = 0; i < kDim; ++i)
                {
                    kn[i] = __half2float(kr[i]) / std::sqrt(nk);
                    qn[i] = __half2float(qr[i]) * scale / std::sqrt(nq);
                }
                size_t const abIdx = tok * c.hv + iHv;
                double const x = static_cast<double>(__half2float(d.a[abIdx])) + __half2float(d.dtBias[iHv]);
                double const sp = x <= 20.0 ? std::log1p(std::exp(x)) : x;
                double const g = std::exp(-std::exp(static_cast<double>(d.logA[iHv])) * sp);
                double const beta = 1.0 / (1.0 + std::exp(-static_cast<double>(__half2float(d.b[abIdx]))));
                for (auto& x2 : H)
                    x2 *= g;
                __half const* vr = d.v.data() + (tok * c.hv + iHv) * kDim;
                for (int32_t col = 0; col < kDim; ++col)
                {
                    double dot = 0.0;
                    for (int32_t r = 0; r < kDim; ++r)
                        dot += H[r * kDim + col] * kn[r];
                    double const delta = (__half2float(vr[col]) - dot) * beta;
                    double out = 0.0;
                    for (int32_t r = 0; r < kDim; ++r)
                    {
                        H[r * kDim + col] += kn[r] * delta;
                        out += H[r * kDim + col] * qn[r];
                    }
                    o[(tok * c.hv + iHv) * kDim + col] = static_cast<float>(out);
                }
            }
            for (int32_t i = 0; i < kDim * kDim; ++i)
                hs[i] = static_cast<float>(H[i]);
        }
    }
}

struct DeviceBuffers
{
    void* q{};
    void* k{};
    void* v{};
    void* a{};
    void* b{};
    void* logA{};
    void* dtBias{};
    void* h0{};
    void* ctx{};
    void* idx{};
    void* o{};

    template <typename T>
    static void* upload(std::vector<T> const& h)
    {
        void* p{};
        CUDA_CHECK(cudaMalloc(&p, h.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }

    DeviceBuffers(Case const& c, HostData const& d)
    {
        q = upload(d.q);
        k = upload(d.k);
        v = upload(d.v);
        a = upload(d.a);
        b = upload(d.b);
        logA = upload(d.logA);
        dtBias = upload(d.dtBias);
        h0 = upload(d.h0);
        ctx = upload(c.contextLengths);
        idx = c.stateIndices.empty() ? nullptr : upload(c.stateIndices);
        CUDA_CHECK(cudaMalloc(&o, d.v.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemset(o, 0xff, d.v.size() * sizeof(__half)));
    }

    ~DeviceBuffers()
    {
        for (void* p : {q, k, v, a, b, logA, dtBias, h0, ctx, idx, o})
        {
            if (p != nullptr)
            {
                cudaFree(p);
            }
        }
    }

    kernel::GdnPrefillSequentialParams params(Case const& c) const
    {
        kernel::GdnPrefillSequentialParams p{};
        p.q = q;
        p.k = k;
        p.v = v;
        p.a = a;
        p.b = b;
        p.logA = logA;
        p.dtBias = dtBias;
        p.h0Source = h0;
        p.stateIndices = idx;
        p.contextLengths = ctx;
        p.o = o;
        p.n = c.n;
        p.seqLen = c.seqLen;
        p.h = c.h;
        p.hv = c.hv;
        p.kDim = kDim;
        p.vDim = kDim;
        return p;
    }

#ifdef CUTE_DSL_GDN_ENABLED
    GDNParams cuteDslParams(Case const& c) const
    {
        GDNParams p{};
        p.q = q;
        p.k = k;
        p.v = v;
        p.a = a;
        p.b = b;
        p.A_log = logA;
        p.dt_bias = dtBias;
        p.h0_source = h0;
        p.context_lengths = ctx;
        p.state_indices = idx;
        p.o = o;
        p.n = c.n;
        p.seq_len = c.seqLen;
        p.h = c.h;
        p.hv = c.hv;
        p.k_dim = kDim;
        p.v_dim = kDim;
        p.state_pool_rows = c.poolRows;
        p.use_prefill = true;
        p.smVersion = getSMVersion();
        return p;
    }
#endif
};

template <typename T>
std::vector<T> download(void const* p, size_t count)
{
    std::vector<T> h(count);
    CUDA_CHECK(cudaMemcpy(h.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

void checkAgainstReference(Case const& c, uint32_t seed)
{
    HostData const d = makeInputs(c, seed);
    DeviceBuffers buf(c, d);
    ASSERT_EQ(kernel::launchGdnPrefillSequential(buf.params(c), nullptr), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto const o = download<__half>(buf.o, d.v.size());
    auto const h0 = download<float>(buf.h0, d.h0.size());

    std::vector<float> h0Ref(d.h0);
    std::vector<float> oRef;
    reference(c, d, h0Ref, oRef);
    for (size_t i = 0; i < oRef.size(); ++i)
    {
        ASSERT_TRUE(isclose(__half2float(o[i]), oRef[i], 2e-2F, 2e-3F))
            << "o[" << i << "] = " << __half2float(o[i]) << ", reference " << oRef[i];
    }
    for (size_t i = 0; i < h0Ref.size(); ++i)
    {
        ASSERT_TRUE(isclose(h0[i], h0Ref[i], 1e-3F, 1e-4F))
            << "h0[" << i << "] = " << h0[i] << ", reference " << h0Ref[i];
    }
}

} // namespace

TEST(GdnPrefillSequential, Supported)
{
    EXPECT_TRUE(kernel::gdnPrefillSequentialSupported(16, 32, 128, 128));
    EXPECT_TRUE(kernel::gdnPrefillSequentialSupported(8, 8, 128, 128));
    EXPECT_FALSE(kernel::gdnPrefillSequentialSupported(16, 24, 128, 128));
    EXPECT_FALSE(kernel::gdnPrefillSequentialSupported(16, 32, 64, 128));
    EXPECT_FALSE(kernel::gdnPrefillSequentialSupported(16, 32, 128, 256));
}

TEST(GdnPrefillSequential, MatchesReference)
{
    checkAgainstReference(Case{1, 67, 2, 4, 1, {67}, {}}, 7U);
}

TEST(GdnPrefillSequential, MatchesReferenceWithPaddingAndStateIndices)
{
    checkAgainstReference(Case{3, 40, 2, 4, 5, {40, 17, 0}, {4, 0, 2}}, 11U);
}

#ifdef CUTE_DSL_GDN_ENABLED
// The CuTe DSL sequential kernel is the prefill backend this kernel stands in for: outputs and
// final states must be identical bit for bit.
TEST(GdnPrefillSequential, BitExactAgainstCuteDslPrefill)
{
    std::vector<Case> const cases{
        Case{1, 1032, 16, 32, 1, {1032}, {}},
        Case{3, 40, 2, 4, 5, {40, 17, 0}, {4, 0, 2}},
        Case{2, 2, 8, 8, 2, {2, 1}, {1, 0}},
    };
    for (Case const& c : cases)
    {
        GDNParams probe{};
        probe.use_prefill = true;
        probe.smVersion = getSMVersion();
        probe.h = c.h;
        probe.hv = c.hv;
        if (CuteDslGDNRunner::selectBackend(probe) != GDNBackend::kPrefill)
        {
            GTEST_SKIP() << "the CuTe DSL sequential prefill does not run on SM " << probe.smVersion;
        }
        HostData const d = makeInputs(c, 1234U + static_cast<uint32_t>(c.seqLen));
        DeviceBuffers ours(c, d);
        DeviceBuffers dsl(c, d);
        ASSERT_EQ(kernel::launchGdnPrefillSequential(ours.params(c), nullptr), cudaSuccess);
        CuteDslGDNRunner runner;
        ASSERT_EQ(runner.run(dsl.cuteDslParams(c), nullptr), 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto const oOurs = download<uint16_t>(ours.o, d.v.size());
        auto const oDsl = download<uint16_t>(dsl.o, d.v.size());
        auto const hOurs = download<uint32_t>(ours.h0, d.h0.size());
        auto const hDsl = download<uint32_t>(dsl.h0, d.h0.size());
        size_t oDiff = 0;
        size_t hDiff = 0;
        for (size_t i = 0; i < oOurs.size(); ++i)
            oDiff += oOurs[i] != oDsl[i];
        for (size_t i = 0; i < hOurs.size(); ++i)
            hDiff += hOurs[i] != hDsl[i];
        EXPECT_EQ(oDiff, 0U) << "n=" << c.n << " seqLen=" << c.seqLen;
        EXPECT_EQ(hDiff, 0U) << "n=" << c.n << " seqLen=" << c.seqLen;
    }
}

TEST(GdnPrefillSequential, Bench)
{
    if (std::getenv("EDGELLM_GDN_PREFILL_BENCH") == nullptr)
    {
        GTEST_SKIP() << "set EDGELLM_GDN_PREFILL_BENCH=1 to time the prefill kernels";
    }
    Case const c{1, 1032, 16, 32, 1, {1032}, {}};
    HostData const d = makeInputs(c, 5U);
    DeviceBuffers buf(c, d);
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    constexpr int32_t kIters{20};
    auto timeIt = [&](auto&& fn) {
        fn();
        CUDA_CHECK(cudaEventRecord(start));
        for (int32_t i = 0; i < kIters; ++i)
            fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms / kIters;
    };
    float const oursMs = timeIt([&] { kernel::launchGdnPrefillSequential(buf.params(c), nullptr); });
    CuteDslGDNRunner runner;
    GDNParams const dslParams = buf.cuteDslParams(c);
    float const dslMs = timeIt([&] { runner.run(dslParams, nullptr); });
    std::printf("GDN prefill n=1 T=1032 h=16 hv=32: sequential %.3f ms, CuTe DSL %.3f ms\n", oursMs, dslMs);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
#endif
