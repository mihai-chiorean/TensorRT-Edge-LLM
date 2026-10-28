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

// Kernel-level timing table for the INT4 (W4A16) FP16 CuTe DSL GEMV variants
// (M = 1..8) against every compiled GEMM tile variant, over the INT4 (N, K)
// shapes of the exported Gemma 4 E2B decoder and the M values the plugin sees
// (M = 1 decode, small M for speculative verification rows and short prefills).
// Each group size compiled into the artifact is timed. Drives the AOT artifact
// directly like the correctness tests and reports every output's error against
// an FP32 host reference, so the table also documents output equivalence. It is
// a measurement, not a regression test: it runs only when
// EDGELLM_INT4_CROSSOVER_BENCH is set.

#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED

#include <cuda.h>
#if defined(TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT)
#include <cuda_runtime.h>
#if CUDA_VERSION < 12800
typedef CUlibrary cudaLibrary_t;
static inline cudaError_t cudaLibraryUnload(cudaLibrary_t lib)
{
    return static_cast<cudaError_t>(cuLibraryUnload(lib));
}
#endif // CUDA_VERSION < 12800
#endif // TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT

#include "common/cudaUtils.h"
#include "cutedsl_all.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace
{

#define ASSERT_CUDA_OK(expr) ASSERT_EQ((expr), cudaSuccess) << cudaGetErrorString(cudaGetLastError())

constexpr int32_t kBN = 128;
constexpr int32_t kBK = 64;
constexpr int32_t kLaunchesPerSample = 20;
constexpr int32_t kSamples = 15;
// DVFS lowers the GPU clock while the host computes each shape's reference, so
// every kernel runs back to back before it is sampled.
constexpr float kWarmupMs = 30.f;
constexpr float kShapeWarmupMs = 500.f;
constexpr int32_t kReferenceMaxM = 16;
// Weight and scale copies are rotated so that every launch reads its operands
// from DRAM, as consecutive decoder layers do, instead of from a warm L2.
constexpr size_t kMinRotationBytes = size_t{64} << 20;
constexpr size_t kRotationL2Multiple = 4;

// Current GPU clock from the Jetson devfreq node; -1 where it does not exist.
int32_t gpuClockMHz()
{
    std::ifstream f("/sys/class/devfreq/17000000.gpu/cur_freq");
    int64_t hz = -1;
    if (f >> hz)
    {
        return static_cast<int32_t>(hz / 1000000);
    }
    return -1;
}

inline int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

template <typename T>
struct DeviceBuffer
{
    T* ptr{nullptr};
    DeviceBuffer() = default;
    ~DeviceBuffer()
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
        }
    }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    cudaError_t alloc(size_t count)
    {
        return cudaMalloc(&ptr, count * sizeof(T));
    }
};

// X(prefix, tile M, stages, split-K). Group 128 keeps the unsuffixed names;
// group 32 carries a `_g32` suffix and exists only in artifacts that list it.
#define INT4_FP16_GEMM_VARIANTS_G(X, SUFFIX)                                                                           \
    X(int4_fp16_gemm_16x128x64_s3_sk1##SUFFIX, 16, 3, 1)                                                               \
    X(int4_fp16_gemm_16x128x64_s4_sk1##SUFFIX, 16, 4, 1)                                                               \
    X(int4_fp16_gemm_16x128x64_s4_sk2##SUFFIX, 16, 4, 2)                                                               \
    X(int4_fp16_gemm_16x128x64_s4_sk4##SUFFIX, 16, 4, 4)                                                               \
    X(int4_fp16_gemm_32x128x64_s2_sk1##SUFFIX, 32, 2, 1)                                                               \
    X(int4_fp16_gemm_32x128x64_s3_sk2##SUFFIX, 32, 3, 2)                                                               \
    X(int4_fp16_gemm_32x128x64_s3_sk4##SUFFIX, 32, 3, 4)                                                               \
    X(int4_fp16_gemm_32x128x64_s4_sk1##SUFFIX, 32, 4, 1)                                                               \
    X(int4_fp16_gemm_64x128x64_s2_sk1##SUFFIX, 64, 2, 1)                                                               \
    X(int4_fp16_gemm_64x128x64_s3_sk1##SUFFIX, 64, 3, 1)                                                               \
    X(int4_fp16_gemm_64x128x64_s3_sk4##SUFFIX, 64, 3, 4)                                                               \
    X(int4_fp16_gemm_64x128x64_s4_sk2##SUFFIX, 64, 4, 2)                                                               \
    X(int4_fp16_gemm_64x128x64_s4_sk4##SUFFIX, 64, 4, 4)                                                               \
    X(int4_fp16_gemm_128x128x64_s2_sk1##SUFFIX, 128, 2, 1)                                                             \
    X(int4_fp16_gemm_128x128x64_s2_sk4##SUFFIX, 128, 2, 4)                                                             \
    X(int4_fp16_gemm_128x128x64_s4_sk1##SUFFIX, 128, 4, 1)

#define INT4_FP16_GEMV_VARIANTS_G(X, SUFFIX)                                                                           \
    X(int4_fp16_gemv_m1##SUFFIX, 1)                                                                                    \
    X(int4_fp16_gemv_m2##SUFFIX, 2)                                                                                    \
    X(int4_fp16_gemv_m3##SUFFIX, 3)                                                                                    \
    X(int4_fp16_gemv_m4##SUFFIX, 4)                                                                                    \
    X(int4_fp16_gemv_m5##SUFFIX, 5)                                                                                    \
    X(int4_fp16_gemv_m6##SUFFIX, 6)                                                                                    \
    X(int4_fp16_gemv_m7##SUFFIX, 7)                                                                                    \
    X(int4_fp16_gemv_m8##SUFFIX, 8)

#define INT4_FP16_GEMM_VARIANTS_G128(X) INT4_FP16_GEMM_VARIANTS_G(X, )
#define INT4_FP16_GEMV_VARIANTS_G128(X) INT4_FP16_GEMV_VARIANTS_G(X, )
#ifdef CUTE_DSL_INT4_FP16_GEMM_G32_ENABLED
#define INT4_FP16_GEMM_VARIANTS_G32(X) INT4_FP16_GEMM_VARIANTS_G(X, _g32)
#define INT4_FP16_GEMV_VARIANTS_G32(X) INT4_FP16_GEMV_VARIANTS_G(X, _g32)
constexpr bool kHasGroup32 = true;
#else
#define INT4_FP16_GEMM_VARIANTS_G32(X)
#define INT4_FP16_GEMV_VARIANTS_G32(X)
constexpr bool kHasGroup32 = false;
#endif

#define DECL_GEMM_MODULE(PREFIX, VM, VS, VK) PREFIX##_Kernel_Module_t g_##PREFIX{};
INT4_FP16_GEMM_VARIANTS_G128(DECL_GEMM_MODULE)
INT4_FP16_GEMM_VARIANTS_G32(DECL_GEMM_MODULE)
#undef DECL_GEMM_MODULE
#define DECL_GEMV_MODULE(PREFIX, VM) PREFIX##_Kernel_Module_t g_##PREFIX{};
INT4_FP16_GEMV_VARIANTS_G128(DECL_GEMV_MODULE)
INT4_FP16_GEMV_VARIANTS_G32(DECL_GEMV_MODULE)
#undef DECL_GEMV_MODULE

bool loadModules()
{
    cudaFree(nullptr);
    (void) cudaGetLastError();
    bool ok = true;
#define LOAD_GEMM_MODULE(PREFIX, VM, VS, VK)                                                                           \
    PREFIX##_Kernel_Module_Load(&g_##PREFIX);                                                                          \
    ok = ok && (g_##PREFIX.module != nullptr);
    INT4_FP16_GEMM_VARIANTS_G128(LOAD_GEMM_MODULE)
    INT4_FP16_GEMM_VARIANTS_G32(LOAD_GEMM_MODULE)
#undef LOAD_GEMM_MODULE
#define LOAD_GEMV_MODULE(PREFIX, VM)                                                                                   \
    PREFIX##_Kernel_Module_Load(&g_##PREFIX);                                                                          \
    ok = ok && (g_##PREFIX.module != nullptr);
    INT4_FP16_GEMV_VARIANTS_G128(LOAD_GEMV_MODULE)
    INT4_FP16_GEMV_VARIANTS_G32(LOAD_GEMV_MODULE)
#undef LOAD_GEMV_MODULE
    return ok && cudaGetLastError() == cudaSuccess;
}

#define SET_2D_TENSOR(t, p, d0, d1)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        (t).data = const_cast<void*>(static_cast<void const*>(p));                                                     \
        (t).dynamic_shapes[0] = (d0);                                                                                  \
        (t).dynamic_shapes[1] = (d1);                                                                                  \
        (t).dynamic_strides[0] = static_cast<int64_t>(d1);                                                             \
    } while (0)

#define SET_1D_TENSOR(t, p, d0)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        (t).data = const_cast<void*>(static_cast<void const*>(p));                                                     \
        (t).dynamic_shapes[0] = (d0);                                                                                  \
    } while (0)

struct GemmVariant
{
    int32_t tileM;
    int32_t stages;
    int32_t splitK;
};

std::vector<GemmVariant> gemmVariants()
{
    std::vector<GemmVariant> v;
#define ADD_VARIANT(PREFIX, VM, VS, VK) v.push_back(GemmVariant{VM, VS, VK});
    INT4_FP16_GEMM_VARIANTS_G128(ADD_VARIANT)
#undef ADD_VARIANT
    return v;
}

struct Problem
{
    void const* A;
    void const* QW;
    void const* scales;
    void* C;
    void* locks;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t groupSize;
};

int32_t qwRowsOf(int32_t N, int32_t K)
{
    return ceilDiv(N, kBN) * ceilDiv(K, kBK) * ((kBK / 16) * (kBN / 64));
}

#define LAUNCH_GEMM(PREFIX)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_mA_t tA{};                                                                                     \
        SET_2D_TENSOR(tA, p.A, p.M, p.K);                                                                              \
        PREFIX##_Tensor_mQW_t tQW{};                                                                                   \
        SET_2D_TENSOR(tQW, p.QW, qwRows, 128);                                                                         \
        PREFIX##_Tensor_mScales_t tS{};                                                                                \
        SET_2D_TENSOR(tS, p.scales, scaleRows, p.N);                                                                   \
        PREFIX##_Tensor_mC_t tC{};                                                                                     \
        SET_2D_TENSOR(tC, p.C, p.M, p.N);                                                                              \
        PREFIX##_Tensor_mWorkspace_t tW{};                                                                             \
        SET_2D_TENSOR(tW, p.C, p.M, p.N);                                                                              \
        PREFIX##_Tensor_mLocks_t tL{};                                                                                 \
        SET_1D_TENSOR(tL, p.locks, nTiles);                                                                            \
        ret = cute_dsl_##PREFIX##_wrapper(&g_##PREFIX, &tA, &tQW, &tS, &tC, &tW, &tL, 1 /* swizzle */, stream);        \
    } while (0)

int32_t launchGemm(GemmVariant const& v, Problem const& p, cudaStream_t stream)
{
    int32_t const qwRows = qwRowsOf(p.N, p.K);
    int32_t const scaleRows = ceilDiv(p.K, p.groupSize);
    int32_t const nTiles = ceilDiv(p.M, v.tileM) * ceilDiv(p.N, kBN);
    int32_t ret = -1;
#define DISPATCH_GEMM(PREFIX, VM, VS, VK)                                                                              \
    if (v.tileM == (VM) && v.stages == (VS) && v.splitK == (VK))                                                       \
    {                                                                                                                  \
        LAUNCH_GEMM(PREFIX);                                                                                           \
        return ret;                                                                                                    \
    }
    if (p.groupSize == 32)
    {
        INT4_FP16_GEMM_VARIANTS_G32(DISPATCH_GEMM)
    }
    else if (p.groupSize == 128)
    {
        INT4_FP16_GEMM_VARIANTS_G128(DISPATCH_GEMM)
    }
#undef DISPATCH_GEMM
    return -1;
}

#define LAUNCH_GEMV(PREFIX)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_mA_t tA{};                                                                                     \
        SET_2D_TENSOR(tA, p.A, p.M, p.K);                                                                              \
        PREFIX##_Tensor_mQW_t tQW{};                                                                                   \
        SET_2D_TENSOR(tQW, p.QW, qwRows, 128);                                                                         \
        PREFIX##_Tensor_mScales_t tS{};                                                                                \
        SET_2D_TENSOR(tS, p.scales, scaleRows, p.N);                                                                   \
        PREFIX##_Tensor_mOut_t tOut{};                                                                                 \
        SET_2D_TENSOR(tOut, p.C, p.M, p.N);                                                                            \
        ret = cute_dsl_##PREFIX##_wrapper(&g_##PREFIX, &tA, &tQW, &tS, &tOut, stream);                                 \
    } while (0)

int32_t launchGemv(Problem const& p, cudaStream_t stream)
{
    int32_t const qwRows = qwRowsOf(p.N, p.K);
    int32_t const scaleRows = ceilDiv(p.K, p.groupSize);
    int32_t ret = -1;
#define DISPATCH_GEMV(PREFIX, VM)                                                                                      \
    if (p.M == (VM))                                                                                                   \
    {                                                                                                                  \
        LAUNCH_GEMV(PREFIX);                                                                                           \
        return ret;                                                                                                    \
    }
    if (p.groupSize == 32)
    {
        INT4_FP16_GEMV_VARIANTS_G32(DISPATCH_GEMV)
    }
    else if (p.groupSize == 128)
    {
        INT4_FP16_GEMV_VARIANTS_G128(DISPATCH_GEMV)
    }
#undef DISPATCH_GEMV
    return -1;
}

inline int32_t signedNibble(std::vector<uint8_t> const& qweight, int32_t n, int32_t k, int32_t K)
{
    uint8_t const packed = qweight[static_cast<size_t>(n / 2) * K + k];
    int32_t const nib = (n % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    return nib >= 8 ? nib - 16 : nib;
}

void int4GemmReference(std::vector<half> const& A, std::vector<uint8_t> const& qweight, std::vector<half> const& scales,
    std::vector<float>& C, int32_t M, int32_t N, int32_t K, int32_t groupSize)
{
    std::vector<float> w(static_cast<size_t>(K));
    for (int32_t n = 0; n < N; ++n)
    {
        for (int32_t k = 0; k < K; ++k)
        {
            w[k] = static_cast<float>(signedNibble(qweight, n, k, K))
                * __half2float(scales[static_cast<size_t>(k / groupSize) * N + n]);
        }
        for (int32_t m = 0; m < M; ++m)
        {
            float acc = 0.f;
            half const* a = A.data() + static_cast<size_t>(m) * K;
            for (int32_t k = 0; k < K; ++k)
            {
                acc += __half2float(a[k]) * w[k];
            }
            C[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

// Fragment-order repack shared by the GEMV and all GEMM tiles (bN=128, bK=64).
std::vector<uint32_t> repackWeights(std::vector<uint8_t> const& qweight, int32_t N, int32_t K)
{
    constexpr int32_t kBlocks = kBK / 16;
    constexpr int32_t nPairs = kBN / 64;
    constexpr int32_t kn = kBlocks * nPairs;
    int32_t const numNBlocks = ceilDiv(N, kBN);
    int32_t const numKTiles = ceilDiv(K, kBK);
    std::vector<uint32_t> out(static_cast<size_t>(numNBlocks) * numKTiles * kn * 128, 0u);
    auto biased = [&](int32_t n, int32_t k) -> uint32_t {
        if (n >= N || k >= K)
        {
            return 8u;
        }
        uint8_t const packed = qweight[static_cast<size_t>(n / 2) * K + k];
        uint32_t const nib = (n % 2 == 0) ? (packed & 0xFu) : ((packed >> 4) & 0xFu);
        return (nib + 8u) & 0xFu;
    };
    struct Nibble
    {
        int32_t shift;
        bool hi;
        int32_t koff;
    };
    static constexpr Nibble kNibbles[8] = {
        {0, false, 0},
        {4, false, 8},
        {8, true, 0},
        {12, true, 8},
        {16, false, 1},
        {20, false, 9},
        {24, true, 1},
        {28, true, 9},
    };
    for (int32_t t = 0; t < 128; ++t)
    {
        int32_t const nBase = t / 4;
        int32_t const kb = 2 * (t % 4);
        for (int32_t nb = 0; nb < numNBlocks; ++nb)
        {
            for (int32_t kt = 0; kt < numKTiles; ++kt)
            {
                for (int32_t kbl = 0; kbl < kBlocks; ++kbl)
                {
                    for (int32_t p = 0; p < nPairs; ++p)
                    {
                        int32_t const nLo = nb * kBN + (nBase + 64 * p);
                        int32_t const k0 = kt * kBK + (16 * kbl + kb);
                        uint32_t word = 0u;
                        for (auto const& nbl : kNibbles)
                        {
                            word |= biased(nbl.hi ? nLo + 32 : nLo, k0 + nbl.koff) << nbl.shift;
                        }
                        int64_t const row = static_cast<int64_t>(nb * numKTiles + kt) * kn + (kbl * nPairs + p);
                        out[static_cast<size_t>(row) * 128 + t] = word;
                    }
                }
            }
        }
    }
    return out;
}

struct Shape
{
    char const* role;
    int32_t count; // INT4 layers of this shape in the exported E2B decoder
    int32_t N;
    int32_t K;
    int32_t maxM;
};

// The down_proj and lm_head shapes are INT4 only in recipes that quantize them.
std::vector<Shape> benchShapes()
{
    return {
        {"gate_up_wide", 40, 12288, 1536, 128},
        {"gate_up", 30, 6144, 1536, 128},
        {"q_local", 28, 2048, 1536, 128},
        {"o_local", 28, 1536, 2048, 128},
        {"kv_local_ple_gate", 59, 256, 1536, 128},
        {"ple_back", 35, 1536, 256, 128},
        {"q_global", 7, 4096, 1536, 128},
        {"o_global", 7, 1536, 4096, 128},
        {"kv_global", 6, 512, 1536, 128},
        {"ple_proj", 1, 8960, 1536, 128},
        {"down_wide", 20, 1536, 12288, 128},
        {"down", 15, 1536, 6144, 128},
        {"lm_head", 1, 262144, 1536, 16},
    };
}

std::vector<int32_t> benchMs()
{
    return {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128};
}

// Row 0 of A is the same for every M, so row 0 of any output can be compared
// bit for bit with the M = 1 GEMV (the vanilla decode path).
int32_t rowZeroMismatches(std::vector<half> const& out, std::vector<half> const& decodeRow, int32_t N)
{
    if (decodeRow.empty() || out.size() < static_cast<size_t>(N))
    {
        return -1;
    }
    int32_t mismatches = 0;
    for (int32_t n = 0; n < N; ++n)
    {
        mismatches += (std::memcmp(&out[n], &decodeRow[n], sizeof(half)) != 0) ? 1 : 0;
    }
    return mismatches;
}

// Max |out - ref| / max |ref| over the output; infinity if any value is not finite.
float relativeError(std::vector<half> const& out, std::vector<float> const& ref)
{
    float maxDiff = 0.f;
    float refMax = 0.f;
    for (size_t i = 0; i < out.size(); ++i)
    {
        float const o = __half2float(out[i]);
        if (!std::isfinite(o) || !std::isfinite(ref[i]))
        {
            return std::numeric_limits<float>::infinity();
        }
        maxDiff = std::max(maxDiff, std::abs(o - ref[i]));
        refMax = std::max(refMax, std::abs(ref[i]));
    }
    return maxDiff / (refMax + 1e-8f);
}

TEST(Int4Fp16GemmGemvCrossoverBench, TimingTable)
{
    if (std::getenv("EDGELLM_INT4_CROSSOVER_BENCH") == nullptr)
    {
        GTEST_SKIP() << "Set EDGELLM_INT4_CROSSOVER_BENCH=1 to run the timing table.";
    }
    int32_t smMajor{}, smMinor{};
    ASSERT_CUDA_OK(cudaDeviceGetAttribute(&smMajor, cudaDevAttrComputeCapabilityMajor, 0));
    ASSERT_CUDA_OK(cudaDeviceGetAttribute(&smMinor, cudaDevAttrComputeCapabilityMinor, 0));
    if (smMajor * 10 + smMinor < 80)
    {
        GTEST_SKIP() << "Ampere or newer required";
    }
    ASSERT_TRUE(loadModules()) << "Failed to load INT4 FP16 CuTe DSL kernel modules";
    int32_t l2Size{};
    ASSERT_CUDA_OK(cudaDeviceGetAttribute(&l2Size, cudaDevAttrL2CacheSize, 0));
    size_t const l2Bytes = static_cast<size_t>(l2Size);

    cudaStream_t stream{};
    ASSERT_CUDA_OK(cudaStreamCreate(&stream));
    cudaEvent_t start{}, stop{};
    ASSERT_CUDA_OK(cudaEventCreate(&start));
    ASSERT_CUDA_OK(cudaEventCreate(&stop));

    std::printf(
        "int4_crossover,sm,group,role,count,N,K,M,kernel,median_us,rel_err,ref,copies,row0_diff_vs_gemv_m1,gpu_mhz\n");
    std::vector<GemmVariant> const variants = gemmVariants();
    std::vector<int32_t> groupSizes{128};
    if (kHasGroup32)
    {
        groupSizes.push_back(32);
    }

    for (int32_t const groupSize : groupSizes)
    {
        for (Shape const& shape : benchShapes())
        {
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> aDist(-1.f, 1.f);
            std::uniform_real_distribution<float> scaleDist(0.01f, 0.2f);
            std::uniform_int_distribution<int32_t> byteDist(0, 255);
            int32_t const N = shape.N, K = shape.K, maxM = shape.maxM;
            int32_t const scaleRows = ceilDiv(K, groupSize);
            std::vector<half> hA(static_cast<size_t>(maxM) * K);
            std::vector<uint8_t> hQW(static_cast<size_t>((N + 1) / 2) * K);
            std::vector<half> hScales(static_cast<size_t>(scaleRows) * N);
            for (auto& v : hA)
                v = __float2half(aDist(rng));
            for (auto& v : hQW)
                v = static_cast<uint8_t>(byteDist(rng));
            for (auto& v : hScales)
                v = __float2half(scaleDist(rng));
            std::vector<uint32_t> const hRepacked = repackWeights(hQW, N, K);
            int32_t const refM = std::min(kReferenceMaxM, maxM);
            std::vector<float> hRef(static_cast<size_t>(refM) * N);
            int4GemmReference(hA, hQW, hScales, hRef, refM, N, K, groupSize);

            size_t const copyBytes = hRepacked.size() * sizeof(uint32_t) + hScales.size() * sizeof(half);
            size_t const rotationBytes = std::max(kMinRotationBytes, kRotationL2Multiple * l2Bytes);
            int32_t const copies
                = static_cast<int32_t>(std::max<size_t>((rotationBytes + copyBytes - 1) / copyBytes, 1));

            DeviceBuffer<half> dA, dScales, dC;
            DeviceBuffer<uint32_t> dQW;
            DeviceBuffer<int32_t> dLocks;
            int64_t const maxTiles = static_cast<int64_t>(ceilDiv(maxM, 16)) * ceilDiv(N, kBN);
            ASSERT_CUDA_OK(dA.alloc(hA.size()));
            ASSERT_CUDA_OK(dQW.alloc(hRepacked.size() * copies));
            ASSERT_CUDA_OK(dScales.alloc(hScales.size() * copies));
            ASSERT_CUDA_OK(dC.alloc(static_cast<size_t>(maxM) * N));
            ASSERT_CUDA_OK(dLocks.alloc(static_cast<size_t>(maxTiles)));
            ASSERT_CUDA_OK(cudaMemcpy(dA.ptr, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
            for (int32_t c = 0; c < copies; ++c)
            {
                ASSERT_CUDA_OK(cudaMemcpy(dQW.ptr + hRepacked.size() * c, hRepacked.data(),
                    hRepacked.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
                ASSERT_CUDA_OK(cudaMemcpy(dScales.ptr + hScales.size() * c, hScales.data(),
                    hScales.size() * sizeof(half), cudaMemcpyHostToDevice));
            }

            std::vector<half> decodeRow;
            bool firstKernelOfShape = true;
            for (int32_t const M : benchMs())
            {
                if (M > maxM)
                {
                    continue;
                }
                auto problemFor = [&](int32_t copy) {
                    return Problem{dA.ptr, dQW.ptr + hRepacked.size() * copy, dScales.ptr + hScales.size() * copy,
                        dC.ptr, dLocks.ptr, M, N, K, groupSize};
                };
                // Serial split-K reduces into C and needs zeroed locks before every launch (as the plugin does),
                // so the memset is part of the timed region. One sample replays a CUDA graph of the rotating
                // launches, as the runtime replays the engine, so host submission gaps are not timed. The graph
                // holds whole rotations, so consecutive replays keep every copy's reuse distance.
                int32_t const launchesPerSample = copies * ((kLaunchesPerSample + copies - 1) / copies);
                auto timed = [&](auto const& launch) -> float {
                    cudaGraph_t graph{};
                    cudaGraphExec_t graphExec{};
                    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
                    {
                        return -1.f;
                    }
                    bool launched = true;
                    for (int32_t i = 0; i < launchesPerSample; ++i)
                    {
                        launched = launched && launch(problemFor(i % copies)) == 0;
                    }
                    bool const captured = cudaStreamEndCapture(stream, &graph) == cudaSuccess;
                    if (!launched || !captured || trt_edgellm::instantiateCudaGraph(&graphExec, graph) != cudaSuccess)
                    {
                        if (graph != nullptr)
                        {
                            cudaGraphDestroy(graph);
                        }
                        (void) cudaGetLastError();
                        return -1.f;
                    }
                    auto batchMs = [&]() -> float {
                        cudaEventRecord(start, stream);
                        if (cudaGraphLaunch(graphExec, stream) != cudaSuccess)
                        {
                            return -1.f;
                        }
                        cudaEventRecord(stop, stream);
                        cudaEventSynchronize(stop);
                        float elapsedMs{};
                        cudaEventElapsedTime(&elapsedMs, start, stop);
                        return elapsedMs;
                    };
                    auto release = [&]() {
                        cudaGraphExecDestroy(graphExec);
                        cudaGraphDestroy(graph);
                    };
                    float const warmupMs = firstKernelOfShape ? kShapeWarmupMs : kWarmupMs;
                    firstKernelOfShape = false;
                    for (float warmedMs = 0.f; warmedMs < warmupMs;)
                    {
                        float const ms = batchMs();
                        if (ms < 0.f)
                        {
                            release();
                            return -1.f;
                        }
                        warmedMs += ms;
                    }
                    std::vector<float> perLaunchUs;
                    for (int32_t s = 0; s < kSamples; ++s)
                    {
                        float const ms = batchMs();
                        if (ms < 0.f)
                        {
                            release();
                            return -1.f;
                        }
                        perLaunchUs.push_back(ms * 1000.f / launchesPerSample);
                    }
                    release();
                    std::sort(perLaunchUs.begin(), perLaunchUs.end());
                    return perLaunchUs[perLaunchUs.size() / 2];
                };
                auto fetch = [&](std::vector<half>& out) {
                    out.resize(static_cast<size_t>(M) * N);
                    cudaMemcpy(out.data(), dC.ptr, out.size() * sizeof(half), cudaMemcpyDeviceToHost);
                };
                std::vector<float> ref;
                char const* refName = "fp32";
                if (M <= refM)
                {
                    ref.assign(hRef.begin(), hRef.begin() + static_cast<size_t>(M) * N);
                }
                auto report = [&](std::string const& kernel, float us, std::vector<half> const& out) {
                    float const err = (us < 0.f || ref.empty()) ? -1.f : relativeError(out, ref);
                    std::printf("int4_crossover,sm%d%d,%d,%s,%d,%d,%d,%d,%s,%.2f,%.5f,%s,%d,%d,%d\n", smMajor, smMinor,
                        groupSize, shape.role, shape.count, N, K, M, kernel.c_str(), us, err,
                        us < 0.f ? "launch_failed" : refName, copies, rowZeroMismatches(out, decodeRow, N),
                        gpuClockMHz());
                };

                if (M <= 8)
                {
                    float const us = timed([&](Problem const& p) { return launchGemv(p, stream); });
                    ASSERT_CUDA_OK(cudaStreamSynchronize(stream));
                    std::vector<half> out;
                    fetch(out);
                    if (M == 1 && us >= 0.f)
                    {
                        decodeRow = out;
                    }
                    report("gemv_m" + std::to_string(M), us, out);
                }
                for (GemmVariant const& v : variants)
                {
                    if (v.splitK > 1 && ceilDiv(K, kBK) % v.splitK != 0)
                    {
                        continue;
                    }
                    int32_t const nTiles = ceilDiv(M, v.tileM) * ceilDiv(N, kBN);
                    float const us = timed([&](Problem const& p) {
                        if (v.splitK > 1)
                        {
                            cudaMemsetAsync(p.locks, 0, static_cast<size_t>(nTiles) * sizeof(int32_t), stream);
                        }
                        return launchGemm(v, p, stream);
                    });
                    ASSERT_CUDA_OK(cudaStreamSynchronize(stream));
                    std::vector<half> out;
                    fetch(out);
                    if (ref.empty() && us >= 0.f)
                    {
                        ref.resize(out.size());
                        for (size_t i = 0; i < out.size(); ++i)
                        {
                            ref[i] = __half2float(out[i]);
                        }
                        refName = "first_gemm_variant";
                    }
                    report("gemm_" + std::to_string(v.tileM) + "x128x64_s" + std::to_string(v.stages) + "_sk"
                            + std::to_string(v.splitK),
                        us, out);
                }
                std::fflush(stdout);
            }
        }
    }
    ASSERT_CUDA_OK(cudaEventDestroy(start));
    ASSERT_CUDA_OK(cudaEventDestroy(stop));
    ASSERT_CUDA_OK(cudaStreamDestroy(stream));
}

} // namespace

#endif // CUTE_DSL_INT4_FP16_GEMM_ENABLED
