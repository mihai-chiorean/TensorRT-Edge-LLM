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

#include "gdnPrefillSequential.h"

#include <cmath>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

constexpr int32_t kHeadDim{128};
// A V column's 128 state rows are spread over 8 lanes (16 rows each) and every lane group of
// 8 holds 4 columns, the lane mapping of gdn_prefill. Its butterfly reductions over the 8 row
// lanes (offsets 16, 8, 4) and the ascending 16-row FMA chains fix the summation order.
constexpr int32_t kRowLanes{8};
constexpr int32_t kColsPerGroup{4};
constexpr int32_t kRowsPerLane{kHeadDim / kRowLanes};
constexpr int32_t kLaneStride{kRowsPerLane + 4};
// Two 4-column groups per lane (two independent FMA chains) and 8 warps give 64 CTAs for the
// Qwen3.5 GDN geometry (hv = 32): four full waves on an 8-SM Orin at two CTAs per SM.
constexpr int32_t kGroups{2};
constexpr int32_t kWarps{8};
constexpr int32_t kMinBlocksPerSm{2};
constexpr int32_t kChunk{16};
constexpr int32_t kThreads{kWarps * 32};
constexpr int32_t kBlockCols{kWarps * kColsPerGroup * kGroups};
constexpr int32_t kColBlocks{kHeadDim / kBlockCols};
constexpr int32_t kTokensPerWarp{(kChunk + kWarps - 1) / kWarps};
constexpr uint32_t kFullMask{0xffffffffU};

__device__ __forceinline__ float warpButterflySum(float x)
{
#pragma unroll
    for (int32_t offset = 16; offset >= 1; offset /= 2)
    {
        x += __shfl_xor_sync(kFullMask, x, offset);
    }
    return x;
}

__device__ __forceinline__ float rowLaneSum(float x)
{
#pragma unroll
    for (int32_t offset = 16; offset >= 4; offset /= 2)
    {
        x += __shfl_xor_sync(kFullMask, x, offset);
    }
    return x;
}

// gdn_prefill reduces 32-element warp partials s0..s3 (s4..s7 are zero) with a second butterfly
// over 8 lanes, which leaves (s0 + s2) + (s1 + s3) in lane 0.
__device__ __forceinline__ float inverseL2Norm(float const (&x)[4])
{
    float s[4];
#pragma unroll
    for (int32_t w = 0; w < 4; ++w)
    {
        s[w] = warpButterflySum(x[w] * x[w]);
    }
    return rsqrtf(((s[0] + s[2]) + (s[1] + s[3])) + 1e-6F);
}

__device__ __forceinline__ void copyAsync16(void* dst, void const* src)
{
    uint32_t const dstAddr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dstAddr), "l"(src));
}

__device__ __forceinline__ void copyAsyncCommit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void copyAsyncWaitAll()
{
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// Stages the raw FP16 k/q/v rows of tokens [t0, t0 + len) with cp.async (16-byte pieces).
__device__ __forceinline__ void stageChunk(__half (*rawK)[kHeadDim], __half (*rawQ)[kHeadDim],
    __half (*rawV)[kBlockCols], __half const* k, __half const* q, __half const* v, int64_t tokenBase, int32_t t0,
    int32_t len, int32_t h, int32_t iH, int32_t hv, int32_t iHv, int32_t colBase)
{
    constexpr int32_t kPiecesPerRow{kHeadDim / 8};
    constexpr int32_t kPiecesPerV{kBlockCols / 8};
    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < len * kPiecesPerRow; idx += kThreads)
    {
        int32_t const j = idx / kPiecesPerRow;
        int32_t const piece = idx % kPiecesPerRow;
        int64_t const row = ((tokenBase + t0 + j) * h + iH) * kHeadDim + piece * 8;
        copyAsync16(&rawK[j][piece * 8], k + row);
        copyAsync16(&rawQ[j][piece * 8], q + row);
    }
    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < len * kPiecesPerV; idx += kThreads)
    {
        int32_t const j = idx / kPiecesPerV;
        int32_t const piece = idx % kPiecesPerV;
        copyAsync16(&rawV[j][piece * 8], v + ((tokenBase + t0 + j) * hv + iHv) * kHeadDim + colBase + piece * 8);
    }
    copyAsyncCommit();
}

__global__ void __launch_bounds__(kThreads, kMinBlocksPerSm) gdnPrefillSequentialKernel(__half const* __restrict__ q,
    __half const* __restrict__ k, __half const* __restrict__ v, __half const* __restrict__ a,
    __half const* __restrict__ b, float const* __restrict__ logA, __half const* __restrict__ dtBias,
    float* __restrict__ h0Source, int32_t const* __restrict__ stateIndices, int32_t const* __restrict__ contextLengths,
    __half* __restrict__ o, int32_t seqLen, int32_t h, int32_t hv, float scale)
{
    static_assert(kChunk <= 32, "one lane of warp 0 computes the gates of each chunk token");
    static_assert(kBlockCols % 8 == 0, "v rows are staged in 16-byte pieces");

    // Row r * 8 + l of a token's normalized k/q is stored at [l][r], so a lane reads its 16 rows with
    // four 16-byte loads; the 4-float pad keeps the 8 row lanes on distinct banks.
    __shared__ __align__(16) float sK[kChunk][kRowLanes][kLaneStride];
    __shared__ __align__(16) float sQ[kChunk][kRowLanes][kLaneStride];
    __shared__ __align__(16) __half rawK[kChunk][kHeadDim];
    __shared__ __align__(16) __half rawQ[kChunk][kHeadDim];
    __shared__ __align__(16) __half rawV[kChunk][kBlockCols];
    __shared__ float sV[kChunk][kBlockCols];
    __shared__ __half sO[kChunk][kBlockCols];
    __shared__ float sG[kChunk];
    __shared__ float sBeta[kChunk];

    int32_t const colBlock = static_cast<int32_t>(blockIdx.x) % kColBlocks;
    int32_t const seqHead = static_cast<int32_t>(blockIdx.x) / kColBlocks;
    int32_t const iN = seqHead / hv;
    int32_t const iHv = seqHead % hv;
    int32_t const iH = iHv / (hv / h);
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
    int32_t const rowLane = lane / kColsPerGroup;
    int32_t const colBase = colBlock * kBlockCols;
    int32_t const slot = stateIndices != nullptr ? stateIndices[iN] : iN;
    int32_t const validLen = max(0, min(seqLen, contextLengths[iN]));
    int64_t const tokenBase = static_cast<int64_t>(iN) * seqLen;

    int32_t colLocal[kGroups];
#pragma unroll
    for (int32_t g = 0; g < kGroups; ++g)
    {
        colLocal[g] = (warp * kGroups + g) * kColsPerGroup + lane % kColsPerGroup;
    }

    if (validLen > 0)
    {
        stageChunk(rawK, rawQ, rawV, k, q, v, tokenBase, 0, min(kChunk, validLen), h, iH, hv, iHv, colBase);
    }
    float const dtBiasValue = __half2float(dtBias[iHv]);
    float const logAValue = logA[iHv];
    float nextA = 0.0F;
    float nextB = 0.0F;
    if (warp == 0 && lane < min(kChunk, validLen))
    {
        nextA = __half2float(a[(tokenBase + lane) * hv + iHv]);
        nextB = __half2float(b[(tokenBase + lane) * hv + iHv]);
    }

    float* const state = h0Source + (static_cast<int64_t>(slot) * hv + iHv) * kHeadDim * kHeadDim;
    float hState[kGroups][kRowsPerLane];
#pragma unroll
    for (int32_t g = 0; g < kGroups; ++g)
    {
#pragma unroll
        for (int32_t r = 0; r < kRowsPerLane; ++r)
        {
            hState[g][r] = state[(r * kRowLanes + rowLane) * kHeadDim + colBase + colLocal[g]];
        }
    }

    for (int32_t t0 = 0; t0 < validLen; t0 += kChunk)
    {
        int32_t const len = min(kChunk, validLen - t0);
        copyAsyncWaitAll();
        __syncthreads();
        if (warp == 0 && lane < len)
        {
            float const x = nextA + dtBiasValue;
            float softplus = x;
            if (x <= 20.0F)
            {
                softplus = logf(1.0F + expf(x));
            }
            sG[lane] = expf(-expf(logAValue) * softplus);
            sBeta[lane] = 1.0F / (1.0F + expf(-nextB));
        }
        for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < len * kBlockCols; idx += kThreads)
        {
            sV[idx / kBlockCols][idx % kBlockCols] = __half2float(rawV[idx / kBlockCols][idx % kBlockCols]);
        }
#pragma unroll
        for (int32_t u = 0; u < kTokensPerWarp; ++u)
        {
            int32_t const j = warp + u * kWarps;
            if (j < len)
            {
                float kx[4];
                float qx[4];
#pragma unroll
                for (int32_t w = 0; w < 4; ++w)
                {
                    kx[w] = __half2float(rawK[j][w * 32 + lane]);
                    qx[w] = __half2float(rawQ[j][w * 32 + lane]);
                }
                float const invK = inverseL2Norm(kx);
                float const invQ = inverseL2Norm(qx);
#pragma unroll
                for (int32_t w = 0; w < 4; ++w)
                {
                    int32_t const row = w * 32 + lane;
                    sK[j][row % kRowLanes][row / kRowLanes] = kx[w] * invK;
                    sQ[j][row % kRowLanes][row / kRowLanes] = __fmul_rn(qx[w], scale) * invQ;
                }
            }
        }
        __syncthreads();
        int32_t const nextT0 = t0 + kChunk;
        if (nextT0 < validLen)
        {
            int32_t const nextLen = min(kChunk, validLen - nextT0);
            stageChunk(rawK, rawQ, rawV, k, q, v, tokenBase, nextT0, nextLen, h, iH, hv, iHv, colBase);
            if (warp == 0 && lane < nextLen)
            {
                nextA = __half2float(a[(tokenBase + nextT0 + lane) * hv + iHv]);
                nextB = __half2float(b[(tokenBase + nextT0 + lane) * hv + iHv]);
            }
        }

        for (int32_t j = 0; j < len; ++j)
        {
            float const gate = sG[j];
            float const beta = sBeta[j];
            float kr[kRowsPerLane];
            float qr[kRowsPerLane];
#pragma unroll
            for (int32_t i = 0; i < kRowsPerLane / 4; ++i)
            {
                float4 const kv4 = reinterpret_cast<float4 const*>(sK[j][rowLane])[i];
                float4 const qv4 = reinterpret_cast<float4 const*>(sQ[j][rowLane])[i];
                kr[4 * i] = kv4.x;
                kr[4 * i + 1] = kv4.y;
                kr[4 * i + 2] = kv4.z;
                kr[4 * i + 3] = kv4.w;
                qr[4 * i] = qv4.x;
                qr[4 * i + 1] = qv4.y;
                qr[4 * i + 2] = qv4.z;
                qr[4 * i + 3] = qv4.w;
            }
            // gdn_prefill decays the state (H * g) before both dot products; keeping the decayed
            // value is the same rounding as recomputing it.
            float dotK[kGroups];
#pragma unroll
            for (int32_t g = 0; g < kGroups; ++g)
            {
                dotK[g] = 0.0F;
            }
#pragma unroll
            for (int32_t r = 0; r < kRowsPerLane; ++r)
            {
#pragma unroll
                for (int32_t g = 0; g < kGroups; ++g)
                {
                    hState[g][r] = __fmul_rn(hState[g][r], gate);
                    dotK[g] = __fmaf_rn(hState[g][r], kr[r], dotK[g]);
                }
            }
            float delta[kGroups];
#pragma unroll
            for (int32_t g = 0; g < kGroups; ++g)
            {
                delta[g] = __fmul_rn(__fsub_rn(sV[j][colLocal[g]], rowLaneSum(dotK[g])), beta);
            }
            float dotQ[kGroups];
#pragma unroll
            for (int32_t g = 0; g < kGroups; ++g)
            {
                dotQ[g] = 0.0F;
            }
#pragma unroll
            for (int32_t r = 0; r < kRowsPerLane; ++r)
            {
#pragma unroll
                for (int32_t g = 0; g < kGroups; ++g)
                {
                    hState[g][r] = __fmaf_rn(kr[r], delta[g], hState[g][r]);
                    dotQ[g] = __fmaf_rn(hState[g][r], qr[r], dotQ[g]);
                }
            }
#pragma unroll
            for (int32_t g = 0; g < kGroups; ++g)
            {
                float const out = rowLaneSum(dotQ[g]);
                if (rowLane == 0)
                {
                    sO[j][colLocal[g]] = __float2half_rn(out);
                }
            }
        }
        __syncthreads();
        for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < len * kBlockCols; idx += kThreads)
        {
            int32_t const j = idx / kBlockCols;
            int32_t const c = idx % kBlockCols;
            o[((tokenBase + t0 + j) * hv + iHv) * kHeadDim + colBase + c] = sO[j][c];
        }
    }

    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < (seqLen - validLen) * kBlockCols; idx += kThreads)
    {
        int32_t const t = validLen + idx / kBlockCols;
        int32_t const c = idx % kBlockCols;
        o[((tokenBase + t) * hv + iHv) * kHeadDim + colBase + c] = __float2half_rn(0.0F);
    }

#pragma unroll
    for (int32_t g = 0; g < kGroups; ++g)
    {
#pragma unroll
        for (int32_t r = 0; r < kRowsPerLane; ++r)
        {
            state[(r * kRowLanes + rowLane) * kHeadDim + colBase + colLocal[g]] = hState[g][r];
        }
    }
}

} // namespace

bool gdnPrefillSequentialSupported(int32_t h, int32_t hv, int32_t kDim, int32_t vDim)
{
    return kDim == kHeadDim && vDim == kHeadDim && h > 0 && hv > 0 && hv % h == 0;
}

cudaError_t launchGdnPrefillSequential(GdnPrefillSequentialParams const& p, cudaStream_t stream)
{
    if (!gdnPrefillSequentialSupported(p.h, p.hv, p.kDim, p.vDim) || p.n <= 0 || p.seqLen <= 0)
    {
        return cudaErrorInvalidValue;
    }
    // gdn_prefill bakes 1/sqrt(128) as a Python double rounded to FP32.
    float const scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.kDim)));
    dim3 const grid(static_cast<uint32_t>(p.n * p.hv * kColBlocks));
    gdnPrefillSequentialKernel<<<grid, kThreads, 0, stream>>>(static_cast<__half const*>(p.q),
        static_cast<__half const*>(p.k), static_cast<__half const*>(p.v), static_cast<__half const*>(p.a),
        static_cast<__half const*>(p.b), static_cast<float const*>(p.logA), static_cast<__half const*>(p.dtBias),
        static_cast<float*>(p.h0Source), static_cast<int32_t const*>(p.stateIndices),
        static_cast<int32_t const*>(p.contextLengths), static_cast<__half*>(p.o), p.seqLen, p.h, p.hv, scale);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
