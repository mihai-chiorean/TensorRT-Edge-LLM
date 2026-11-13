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

#include "kernels/loraKernels/loraGroupKernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

using namespace trt_edgellm::kernels;

namespace
{

#define CUDA_OK(call) ASSERT_EQ((call), cudaSuccess) << #call

struct GroupCase
{
    std::string name;
    int32_t M;
    int32_t K;
    std::vector<int32_t> N;
    std::vector<int32_t> r;
    //! Targets with active[i] == false are bound at rank 0: the plugin passes only the active
    //! ones to the launcher, and their base output must stay bit-identical.
    std::vector<bool> active;
};

std::vector<GroupCase> cases()
{
    // Gemma 4 E4B text: hidden 2560, intermediate 10240, 8 query heads and 2 KV heads of width
    // 256 (sliding) or 512 (global). Groups are the LoraGroupPlugin sibling sets.
    struct Shape
    {
        char const* name;
        int32_t K;
        std::vector<int32_t> N;
    };
    std::vector<Shape> const shapes{{"qkv_local", 2560, {2048, 512, 512}}, {"qkv_global", 2560, {4096, 1024, 1024}},
        {"q_shared", 2560, {2048}}, {"o_local", 2048, {2560}}, {"o_global", 4096, {2560}},
        {"gate_up", 2560, {10240, 10240}}, {"down", 10240, {2560}}};
    std::vector<GroupCase> out;
    for (auto const& s : shapes)
    {
        for (int32_t M : {1, 12, 64})
        {
            for (int32_t r : {1, 8, 16})
            {
                GroupCase c{std::string(s.name) + "_M" + std::to_string(M) + "_r" + std::to_string(r), M, s.K, s.N,
                    std::vector<int32_t>(s.N.size(), r), std::vector<bool>(s.N.size(), true)};
                out.push_back(c);
            }
        }
    }
    // Partial adapters: some siblings inactive (rank 0) must not be touched.
    out.push_back({"qkv_q_only_M12", 12, 2560, {2048, 512, 512}, {16, 16, 16}, {true, false, false}});
    out.push_back({"gate_up_up_only_M1", 1, 2560, {10240, 10240}, {16, 16}, {false, true}});
    // Mixed ranks in one group, rank above one tile, and shapes that are not multiples of 8.
    out.push_back({"qkv_mixed_ranks_M12", 12, 2560, {2048, 512, 512}, {8, 32, 16}, {true, true, true}});
    out.push_back({"qkv_rank64_M5", 5, 2560, {2048, 512, 512}, {64, 16, 16}, {true, true, true}});
    out.push_back({"odd_shapes_M3", 3, 137, {259, 33}, {7, 5}, {true, true}});
    out.push_back({"odd_shapes_M40", 40, 137, {259, 33}, {7, 5}, {true, true}});
    return out;
}

float roundHalf(double v)
{
    return __half2float(__float2half(static_cast<float>(v)));
}

class LoraGroupTest : public ::testing::TestWithParam<GroupCase>
{
};

TEST_P(LoraGroupTest, MatchesUnfusedFp16Branch)
{
    GroupCase const& c = GetParam();
    int32_t const n = static_cast<int32_t>(c.N.size());
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto fill = [&](std::vector<__half>& v, size_t count, float scale) {
        v.resize(count);
        for (auto& e : v)
        {
            e = __float2half(dist(gen) * scale);
        }
    };

    std::vector<__half> x;
    fill(x, static_cast<size_t>(c.M) * c.K, 0.5f);
    std::vector<std::vector<__half>> A(n), B(n), base(n);
    for (int32_t i = 0; i < n; ++i)
    {
        fill(A[i], static_cast<size_t>(c.K) * c.r[i], 0.05f);
        fill(B[i], static_cast<size_t>(c.r[i]) * c.N[i], 0.05f);
        fill(base[i], static_cast<size_t>(c.M) * c.N[i] + 1, 1.f); // +1: guard element after the output
    }

    // Reference: the insert-lora branch, MatMul(x, A) -> fp16, MatMul(., B) -> fp16, Add -> fp16.
    std::vector<std::vector<float>> expected(n);
    for (int32_t i = 0; i < n; ++i)
    {
        expected[i].resize(base[i].size());
        for (size_t e = 0; e < base[i].size(); ++e)
        {
            expected[i][e] = __half2float(base[i][e]);
        }
        if (!c.active[i])
        {
            continue;
        }
        int32_t const r = c.r[i];
        int32_t const N = c.N[i];
        for (int32_t m = 0; m < c.M; ++m)
        {
            std::vector<float> t(r);
            for (int32_t j = 0; j < r; ++j)
            {
                double s = 0;
                for (int32_t k = 0; k < c.K; ++k)
                {
                    s += static_cast<double>(__half2float(x[static_cast<size_t>(m) * c.K + k]))
                        * __half2float(A[i][static_cast<size_t>(k) * r + j]);
                }
                t[j] = roundHalf(s);
            }
            for (int32_t col = 0; col < N; ++col)
            {
                double s = 0;
                for (int32_t j = 0; j < r; ++j)
                {
                    s += static_cast<double>(t[j]) * __half2float(B[i][static_cast<size_t>(j) * N + col]);
                }
                size_t const idx = static_cast<size_t>(m) * N + col;
                expected[i][idx] = roundHalf(static_cast<double>(expected[i][idx]) + roundHalf(s));
            }
        }
    }

    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto upload = [&](std::vector<__half> const& h) {
        void* d = nullptr;
        EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(__half) + 16), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);
        return d;
    };
    void* dx = upload(x);
    std::vector<void*> dA(n), dB(n), dBase(n);
    std::vector<LoraTarget> targets;
    int32_t maxRank = 1;
    for (int32_t i = 0; i < n; ++i)
    {
        dA[i] = upload(A[i]);
        dB[i] = upload(B[i]);
        dBase[i] = upload(base[i]);
        maxRank = std::max(maxRank, c.r[i]);
        if (c.active[i])
        {
            LoraTarget t;
            t.A = dA[i];
            t.B = dB[i];
            t.base = dBase[i];
            t.out = dBase[i]; // in place, as the plugin runs it
            t.r = c.r[i];
            t.N = c.N[i];
            targets.push_back(t);
        }
    }
    void* workspace = nullptr;
    void* state = nullptr;
    CUDA_OK(cudaMalloc(&workspace, loraGroupWorkspaceBytes(c.M, n, maxRank)));
    CUDA_OK(cudaMalloc(&state, kLoraGroupStateBytes));
    CUDA_OK(cudaMemset(state, 0, kLoraGroupStateBytes));
    // Setup copies and the counter reset ran on the legacy default stream, which the
    // non-blocking test stream does not wait for.
    CUDA_OK(cudaDeviceSynchronize());

    // Twice through a captured graph, as the decode runtime replays it: the grid-barrier counter
    // must work across replays. Re-upload the base before the checked replay.
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    ASSERT_EQ(
        launchLoraGroup(dx, c.M, c.K, targets.data(), static_cast<int32_t>(targets.size()), workspace, state, stream),
        0);
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_OK(cudaGraphLaunch(exec, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    for (int32_t i = 0; i < n; ++i)
    {
        CUDA_OK(cudaMemcpy(dBase[i], base[i].data(), base[i].size() * sizeof(__half), cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGraphLaunch(exec, stream));
    CUDA_OK(cudaStreamSynchronize(stream));

    for (int32_t i = 0; i < n; ++i)
    {
        std::vector<__half> got(base[i].size());
        CUDA_OK(cudaMemcpy(got.data(), dBase[i], got.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        size_t const outCount = static_cast<size_t>(c.M) * c.N[i];
        for (size_t e = 0; e < got.size(); ++e)
        {
            float const g = __half2float(got[e]);
            float const want = expected[i][e];
            // Inactive targets and the guard element must be bit-identical; active outputs may
            // differ from the reference by fp32 summation order (a couple of fp16 ulps).
            float const tol = (c.active[i] && e < outCount) ? 2e-3f + 2e-3f * std::fabs(want) : 0.f;
            ASSERT_NEAR(g, want, tol) << c.name << " target " << i << " element " << e;
        }
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(dx);
    for (int32_t i = 0; i < n; ++i)
    {
        cudaFree(dA[i]);
        cudaFree(dB[i]);
        cudaFree(dBase[i]);
    }
    cudaFree(workspace);
    cudaFree(state);
    cudaStreamDestroy(stream);
}

INSTANTIATE_TEST_SUITE_P(Gemma4E4B, LoraGroupTest, ::testing::ValuesIn(cases()),
    [](::testing::TestParamInfo<GroupCase> const& info) { return info.param.name; });

TEST(LoraGroupLauncher, NoActiveTargetLaunchesNothing)
{
    // A group with every target at rank 0 is passed with zero targets: no work, no error, and
    // nothing is captured into a graph.
    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph{};
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    ASSERT_EQ(launchLoraGroup(nullptr, 12, 2560, nullptr, 0, nullptr, nullptr, stream), 0);
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    size_t nodes = 1;
    CUDA_OK(cudaGraphGetNodes(graph, nullptr, &nodes));
    EXPECT_EQ(nodes, 0u);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

//! out = base + fp16(fp16(x @ A) @ B), rounded like the unfused fp16 branch.
std::vector<float> referenceOut(std::vector<__half> const& x, std::vector<__half> const& A,
    std::vector<__half> const& B, std::vector<__half> const& base, int32_t M, int32_t K, int32_t N, int32_t r)
{
    std::vector<float> out(base.size());
    for (size_t e = 0; e < base.size(); ++e)
    {
        out[e] = __half2float(base[e]);
    }
    for (int32_t m = 0; m < M; ++m)
    {
        std::vector<float> t(r);
        for (int32_t j = 0; j < r; ++j)
        {
            double s = 0;
            for (int32_t k = 0; k < K; ++k)
            {
                s += static_cast<double>(__half2float(x[static_cast<size_t>(m) * K + k]))
                    * __half2float(A[static_cast<size_t>(k) * r + j]);
            }
            t[j] = roundHalf(s);
        }
        for (int32_t col = 0; col < N; ++col)
        {
            double s = 0;
            for (int32_t j = 0; j < r; ++j)
            {
                s += static_cast<double>(t[j]) * __half2float(B[static_cast<size_t>(j) * N + col]);
            }
            size_t const idx = static_cast<size_t>(m) * N + col;
            out[idx] = roundHalf(static_cast<double>(out[idx]) + roundHalf(s));
        }
    }
    return out;
}

TEST(LoraGroupLauncher, FastAndFallbackPathsShareOneStateAcrossReplays)
{
    // One plugin state serves both the cooperative group kernel (rank 16) and the per-target
    // fallback (rank 1 is not a multiple of 8), which launch different grid sizes on GPUs with
    // more than 32 SMs. Interleave them in one captured graph and replay it: every replay must
    // finish and give the reference result.
    int32_t const M = 12;
    int32_t const K = 2560;
    std::vector<int32_t> const N{2048, 512, 512};
    std::vector<int32_t> const ranks{16, 1, 16};
    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto fill = [&](size_t count, float scale) {
        std::vector<__half> v(count);
        for (auto& e : v)
        {
            e = __float2half(dist(gen) * scale);
        }
        return v;
    };
    std::vector<__half> const x = fill(static_cast<size_t>(M) * K, 0.5f);
    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    void* dx = nullptr;
    CUDA_OK(cudaMalloc(&dx, x.size() * sizeof(__half)));
    CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice));
    void* workspace = nullptr;
    void* state = nullptr;
    CUDA_OK(cudaMalloc(&workspace, loraGroupWorkspaceBytes(M, 3, 16)));
    CUDA_OK(cudaMalloc(&state, kLoraGroupStateBytes));
    CUDA_OK(cudaMemset(state, 0, kLoraGroupStateBytes));

    struct Launch
    {
        std::vector<std::vector<__half>> A, B, base;
        std::vector<void*> dA, dB, dBase, dOrig;
        std::vector<LoraTarget> targets;
    };
    std::vector<Launch> launches(ranks.size());
    for (size_t l = 0; l < ranks.size(); ++l)
    {
        auto& L = launches[l];
        for (size_t i = 0; i < N.size(); ++i)
        {
            L.A.push_back(fill(static_cast<size_t>(K) * ranks[l], 0.05f));
            L.B.push_back(fill(static_cast<size_t>(ranks[l]) * N[i], 0.05f));
            L.base.push_back(fill(static_cast<size_t>(M) * N[i], 1.f));
            void* p[4];
            for (auto*& q : p)
            {
                q = nullptr;
            }
            CUDA_OK(cudaMalloc(&p[0], L.A.back().size() * 2));
            CUDA_OK(cudaMalloc(&p[1], L.B.back().size() * 2));
            CUDA_OK(cudaMalloc(&p[2], L.base.back().size() * 2));
            CUDA_OK(cudaMalloc(&p[3], L.base.back().size() * 2));
            CUDA_OK(cudaMemcpy(p[0], L.A.back().data(), L.A.back().size() * 2, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(p[1], L.B.back().data(), L.B.back().size() * 2, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(p[3], L.base.back().data(), L.base.back().size() * 2, cudaMemcpyHostToDevice));
            L.dA.push_back(p[0]);
            L.dB.push_back(p[1]);
            L.dBase.push_back(p[2]);
            L.dOrig.push_back(p[3]);
            LoraTarget t;
            t.A = p[0];
            t.B = p[1];
            t.base = p[2];
            t.out = p[2];
            t.r = ranks[l];
            t.N = N[i];
            L.targets.push_back(t);
        }
    }
    CUDA_OK(cudaDeviceSynchronize());
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (auto& L : launches)
    {
        for (size_t i = 0; i < N.size(); ++i)
        {
            CUDA_OK(cudaMemcpyAsync(L.dBase[i], L.dOrig[i], L.base[i].size() * 2, cudaMemcpyDeviceToDevice, stream));
        }
        ASSERT_EQ(launchLoraGroup(dx, M, K, L.targets.data(), 3, workspace, state, stream), 0);
    }
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; ++replay)
    {
        CUDA_OK(cudaGraphLaunch(exec, stream));
        auto const start = std::chrono::steady_clock::now();
        cudaError_t status;
        while ((status = cudaStreamQuery(stream)) == cudaErrorNotReady)
        {
            ASSERT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(20))
                << "replay " << replay << " did not finish: grid barrier desynchronized";
        }
        ASSERT_EQ(status, cudaSuccess);
    }
    for (size_t l = 0; l < launches.size(); ++l)
    {
        auto& L = launches[l];
        for (size_t i = 0; i < N.size(); ++i)
        {
            std::vector<float> const want = referenceOut(x, L.A[i], L.B[i], L.base[i], M, K, N[i], ranks[l]);
            std::vector<__half> got(L.base[i].size());
            CUDA_OK(cudaMemcpy(got.data(), L.dBase[i], got.size() * 2, cudaMemcpyDeviceToHost));
            for (size_t e = 0; e < got.size(); ++e)
            {
                ASSERT_NEAR(__half2float(got[e]), want[e], 2e-3f + 2e-3f * std::fabs(want[e]))
                    << "launch " << l << " target " << i << " element " << e;
            }
        }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    for (auto& L : launches)
    {
        for (size_t i = 0; i < N.size(); ++i)
        {
            cudaFree(L.dA[i]);
            cudaFree(L.dB[i]);
            cudaFree(L.dBase[i]);
            cudaFree(L.dOrig[i]);
        }
    }
    cudaFree(dx);
    cudaFree(workspace);
    cudaFree(state);
    cudaStreamDestroy(stream);
}

//! Orin NX has 8 SMs: with one CTA per SM the gate/up group's expand tiles do not fit the
//! 160 KiB kernel budget in one go, and must be staged in rounds instead of falling back to
//! per-target launches. 16 SMs (AGX Orin) keeps a single round.
TEST(LoraGroupPlan, GateUpStaysFusedOnEightSms)
{
    constexpr size_t kOrinBudget = 160 * 1024;
    std::vector<LoraTarget> gateUp(2);
    for (auto& t : gateUp)
    {
        t.r = 16;
        t.N = 10240;
    }
    for (int32_t const M : {1, 12, 16})
    {
        LoraGroupPlan const nx = planLoraGroup(M, 2560, gateUp.data(), 2, 8, kOrinBudget);
        EXPECT_TRUE(nx.fused) << "M " << M;
        EXPECT_LE(nx.smemBytes, kOrinBudget);
        EXPECT_EQ(nx.tilesPerCta, 160);
        EXPECT_LT(nx.tilesPerRound, nx.tilesPerCta);

        LoraGroupPlan const agx = planLoraGroup(M, 2560, gateUp.data(), 2, 16, kOrinBudget);
        EXPECT_TRUE(agx.fused);
        EXPECT_EQ(agx.tilesPerRound, agx.tilesPerCta);
    }
    // Every Gemma 4 E4B group stays fused on 8 SMs.
    struct Group
    {
        int32_t K;
        std::vector<int32_t> N;
    };
    for (Group const& g : {Group{2560, {2048, 512, 512}}, Group{2560, {4096, 1024, 1024}}, Group{2048, {2560}},
             Group{4096, {2560}}, Group{10240, {2560}}})
    {
        std::vector<LoraTarget> targets(g.N.size());
        for (size_t i = 0; i < g.N.size(); ++i)
        {
            targets[i].r = 16;
            targets[i].N = g.N[i];
        }
        LoraGroupPlan const nx
            = planLoraGroup(12, g.K, targets.data(), static_cast<int32_t>(targets.size()), 8, kOrinBudget);
        EXPECT_TRUE(nx.fused) << "K " << g.K << " targets " << g.N.size();
        EXPECT_LE(nx.smemBytes, kOrinBudget);
    }
}

//! Runs gate/up through the fused kernel with the grid forced to 8 CTAs (and with a small
//! shared-memory budget, 96 KiB, to force more rounds) and checks it against the reference.
TEST(LoraGroupLauncher, MultiRoundExpandMatchesReference)
{
    int dev = 0;
    int optin = 0;
    CUDA_OK(cudaGetDevice(&dev));
    CUDA_OK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    int32_t const M = 12;
    int32_t const K = 2560;
    std::vector<int32_t> const N{10240, 10240};
    std::mt19937 gen(31);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto fill = [&](size_t count, float scale) {
        std::vector<__half> v(count);
        for (auto& e : v)
        {
            e = __float2half(dist(gen) * scale);
        }
        return v;
    };
    std::vector<__half> const x = fill(static_cast<size_t>(M) * K, 0.5f);
    std::vector<std::vector<__half>> A, B, base;
    for (int32_t const n : N)
    {
        A.push_back(fill(static_cast<size_t>(K) * 16, 0.05f));
        B.push_back(fill(static_cast<size_t>(16) * n, 0.05f));
        base.push_back(fill(static_cast<size_t>(M) * n, 1.f));
    }
    void* dx = nullptr;
    CUDA_OK(cudaMalloc(&dx, x.size() * 2));
    CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    std::vector<void*> dA(2), dB(2), dBase(2);
    std::vector<LoraTarget> targets(2);
    for (int32_t i = 0; i < 2; ++i)
    {
        CUDA_OK(cudaMalloc(&dA[i], A[i].size() * 2));
        CUDA_OK(cudaMalloc(&dB[i], B[i].size() * 2));
        CUDA_OK(cudaMalloc(&dBase[i], base[i].size() * 2));
        CUDA_OK(cudaMemcpy(dA[i], A[i].data(), A[i].size() * 2, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dB[i], B[i].data(), B[i].size() * 2, cudaMemcpyHostToDevice));
        targets[i].A = dA[i];
        targets[i].B = dB[i];
        targets[i].base = dBase[i];
        targets[i].out = dBase[i];
        targets[i].r = 16;
        targets[i].N = N[i];
    }
    void* workspace = nullptr;
    CUDA_OK(cudaMalloc(&workspace, loraGroupWorkspaceBytes(M, 2, 16)));
    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (size_t const budget : {std::min<size_t>(static_cast<size_t>(optin), 160 * 1024), size_t{96 * 1024}})
    {
        LoraGroupPlan const plan = planLoraGroup(M, K, targets.data(), 2, 8, budget);
        ASSERT_TRUE(plan.fused) << "budget " << budget;
        ASSERT_LT(plan.tilesPerRound, plan.tilesPerCta) << "budget " << budget;
        // Fresh barrier state per grid size.
        void* state = nullptr;
        CUDA_OK(cudaMalloc(&state, kLoraGroupStateBytes));
        CUDA_OK(cudaMemset(state, 0, kLoraGroupStateBytes));
        for (int32_t i = 0; i < 2; ++i)
        {
            CUDA_OK(cudaMemcpy(dBase[i], base[i].data(), base[i].size() * 2, cudaMemcpyHostToDevice));
        }
        CUDA_OK(cudaDeviceSynchronize());
        ASSERT_EQ(launchLoraGroupWithLimits(dx, M, K, targets.data(), 2, workspace, state, stream, 8, budget), 0);
        CUDA_OK(cudaStreamSynchronize(stream));
        for (int32_t i = 0; i < 2; ++i)
        {
            std::vector<float> const want = referenceOut(x, A[i], B[i], base[i], M, K, N[i], 16);
            std::vector<__half> got(base[i].size());
            CUDA_OK(cudaMemcpy(got.data(), dBase[i], got.size() * 2, cudaMemcpyDeviceToHost));
            for (size_t e = 0; e < got.size(); ++e)
            {
                ASSERT_NEAR(__half2float(got[e]), want[e], 2e-3f + 2e-3f * std::fabs(want[e]))
                    << "budget " << budget << " target " << i << " element " << e;
            }
        }
        cudaFree(state);
    }
    cudaFree(dx);
    for (int32_t i = 0; i < 2; ++i)
    {
        cudaFree(dA[i]);
        cudaFree(dB[i]);
        cudaFree(dBase[i]);
    }
    cudaFree(workspace);
    cudaStreamDestroy(stream);
}

} // namespace
