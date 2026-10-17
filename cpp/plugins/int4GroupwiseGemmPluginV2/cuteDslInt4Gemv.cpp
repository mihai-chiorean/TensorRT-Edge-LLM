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

#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED

#include "cuteDslInt4Gemv.h"
#include "common/cudaUtils.h"
#include "cuteDslInt4Gemm.h"

// The AOT-generated headers use cudaLibrary_t (CUDA 12.8+ runtime). On CUDA < 12.8
// the build defines TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT; route it to the driver type.
#include <cuda.h>
#if defined(TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT)
#include <cuda_runtime.h>
#if CUDA_VERSION < 12800
typedef CUlibrary cudaLibrary_t;
static inline cudaError_t cudaLibraryUnload(cudaLibrary_t lib)
{
    return static_cast<cudaError_t>(cuLibraryUnload(lib));
}
#endif
#endif

#include "cutedsl_all.h"

#include <mutex>

namespace trt_edgellm
{
namespace plugins
{

namespace
{

// The compiled GEMV variants (must match build_cutedsl.py's int4_fp16_gemv
// registration): one exported function per M, baked once per group size. The
// group-128 set keeps the unsuffixed names; group 32 carries a `_g32` suffix.
// X(prefix, M).
#define INT4_FP16_GEMV_VARIANTS_G(X, SUFFIX)                                                                           \
    X(int4_fp16_gemv_m1##SUFFIX, 1)                                                                                    \
    X(int4_fp16_gemv_m2##SUFFIX, 2)                                                                                    \
    X(int4_fp16_gemv_m3##SUFFIX, 3)                                                                                    \
    X(int4_fp16_gemv_m4##SUFFIX, 4)                                                                                    \
    X(int4_fp16_gemv_m5##SUFFIX, 5)                                                                                    \
    X(int4_fp16_gemv_m6##SUFFIX, 6)                                                                                    \
    X(int4_fp16_gemv_m7##SUFFIX, 7)                                                                                    \
    X(int4_fp16_gemv_m8##SUFFIX, 8)
#define INT4_FP16_GEMV_VARIANTS_G128(X) INT4_FP16_GEMV_VARIANTS_G(X, )
#ifdef CUTE_DSL_INT4_FP16_GEMM_G32_ENABLED
#define INT4_FP16_GEMV_VARIANTS_G32(X) INT4_FP16_GEMV_VARIANTS_G(X, _g32)
#else
#define INT4_FP16_GEMV_VARIANTS_G32(X)
#endif

// Module storage, loaded once per group size.
#define DECL_MODULE(prefix, M) prefix##_Kernel_Module_t g_##prefix{};
INT4_FP16_GEMV_VARIANTS_G128(DECL_MODULE)
INT4_FP16_GEMV_VARIANTS_G32(DECL_MODULE)
#undef DECL_MODULE

bool gLoadedG128 = false;
bool gLoadedG32 = false;
std::mutex gLoadMutex;

constexpr int32_t kBN = 128; // fragment N-block (shared with the GEMM)
constexpr int32_t kBK = 64;  // fragment K-tile

#define LOAD_MODULE(prefix, M)                                                                                         \
    prefix##_Kernel_Module_Load(&g_##prefix);                                                                          \
    ok = ok && (g_##prefix.module != nullptr);

bool loadModulesG128()
{
    bool ok = true;
    INT4_FP16_GEMV_VARIANTS_G128(LOAD_MODULE)
    return ok;
}

bool loadModulesG32()
{
    bool ok = true;
    INT4_FP16_GEMV_VARIANTS_G32(LOAD_MODULE)
    return ok;
}
#undef LOAD_MODULE
} // namespace

bool cuteDslInt4GemvSupported(int32_t M)
{
    return M >= 1 && M <= kGemvMaxM;
}

bool cuteDslInt4GemvLoadModules(int32_t groupSize)
{
    if (!cuteDslInt4GroupSizeSupported(groupSize))
    {
        return false;
    }
    std::lock_guard<std::mutex> lock(gLoadMutex);
    bool& loaded = (groupSize == 32) ? gLoadedG32 : gLoadedG128;
    if (loaded)
    {
        return true;
    }
    cudaFree(nullptr); // ensure a current context for the (driver-API on CUDA<12.8) loads
    (void) cudaGetLastError();
    bool const ok = (groupSize == 32) ? loadModulesG32() : loadModulesG128();
    if (!ok || cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    loaded = true;
    return true;
}

// Marshal tensor structs and call the M-variant's generated wrapper.
#define SET_2D(t, ptr, d0, d1)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        (t).data = const_cast<void*>(static_cast<void const*>(ptr));                                                   \
        (t).dynamic_shapes[0] = (d0);                                                                                  \
        (t).dynamic_shapes[1] = (d1);                                                                                  \
        (t).dynamic_strides[0] = static_cast<int64_t>(d1);                                                             \
    } while (0)

int32_t cuteDslInt4GemvLaunch(int32_t M, void const* A, void const* fragW, void const* scales, void* Out, int32_t N,
    int32_t K, int32_t groupSize, cudaStream_t stream)
{
    bool const loaded = (groupSize == 32) ? gLoadedG32 : (groupSize == 128) ? gLoadedG128 : false;
    if (!loaded)
    {
        return -1;
    }
    int32_t const qwRows = divUp(N, kBN) * divUp(K, kBK) * (kBK / 16) * (kBN / 64);
    int32_t const scaleRows = divUp(K, groupSize);
#define DISPATCH(prefix, VM)                                                                                           \
    if (M == (VM))                                                                                                     \
    {                                                                                                                  \
        prefix##_Tensor_mA_t tA{};                                                                                     \
        SET_2D(tA, A, M, K);                                                                                           \
        prefix##_Tensor_mQW_t tW{};                                                                                    \
        SET_2D(tW, fragW, qwRows, 128);                                                                                \
        prefix##_Tensor_mScales_t tS{};                                                                                \
        SET_2D(tS, scales, scaleRows, N);                                                                              \
        prefix##_Tensor_mOut_t tOut{};                                                                                 \
        SET_2D(tOut, Out, M, N);                                                                                       \
        return cute_dsl_##prefix##_wrapper(&g_##prefix, &tA, &tW, &tS, &tOut, stream);                                 \
    }
    if (groupSize == 32)
    {
        INT4_FP16_GEMV_VARIANTS_G32(DISPATCH)
    }
    else
    {
        INT4_FP16_GEMV_VARIANTS_G128(DISPATCH)
    }
#undef DISPATCH
    return -1; // M outside [1, kGemvMaxM]
}

#undef SET_2D
#undef INT4_FP16_GEMV_VARIANTS_G
#undef INT4_FP16_GEMV_VARIANTS_G128
#undef INT4_FP16_GEMV_VARIANTS_G32
#undef INT4_FP16_GEMV_VARIANTS

} // namespace plugins
} // namespace trt_edgellm

#endif // CUTE_DSL_INT4_FP16_GEMM_ENABLED
