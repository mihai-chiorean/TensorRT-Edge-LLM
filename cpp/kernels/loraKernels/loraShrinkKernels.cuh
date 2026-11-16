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

#pragma once

//! Device side of the LoRA shrink (see loraShrinkKernels.h for the contract and summation tree).
//! Header-only so the kernel test harness can compile the same source with NVRTC.

#include "loraShrinkKernels.h"

#include <mma.h>

namespace trt_edgellm
{
namespace kernels
{

//! Grid (numChunks, numTargets, ceil(M / kLoraShrinkRowsPerBlock)), kLoraShrinkThreads threads.
__device__ inline void loraShrinkBody(LoraShrinkParams const& p)
{
    int const tg = blockIdx.y;
    LoraShrinkTarget const target = p.target[tg];
    if (target.rank <= 0)
    {
        return;
    }
    int const chunkK = loraShrinkChunkK(p.m);
    int const numChunks = (p.k + chunkK - 1) / chunkK;
    int const chunk = blockIdx.x;
    int const c0 = chunk * chunkK;
    int const row0 = blockIdx.z * kLoraShrinkRowsPerBlock;
    int const rows = min(kLoraShrinkRowsPerBlock, p.m - row0);
    int const tid = threadIdx.x;
    int const h = tid & 1;
    int const q = tid >> 1;
    int const w = tid >> 5;
    int const lane = tid & 31;
    int const rank = target.rank;
    // K % 64 == 0 and CK % 64 == 0: whole 64-row steps.
    int const nIter = min(chunkK, p.k - c0) / 64;

    __shared__ float sW[4][2][8];
    __shared__ float sQ[kLoraShrinkGroups][kLoraShrinkMaxRank];
    __shared__ int sLast;

    for (int r = 0; r < rows; ++r)
    {
        int const m = row0 + r;
        float acc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            acc[e] = 0.f;
        }
        __half const* xr = p.x + static_cast<int64_t>(m) * p.k;
        if ((rank & 7) == 0)
        {
            if (h * 8 < rank)
            {
                int i = 0;
                for (; i + 4 <= nIter; i += 4)
                {
                    uint4 av[4];
                    float xv[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                    {
                        int const k = c0 + q + (i + u) * 64;
                        av[u] = *reinterpret_cast<uint4 const*>(target.a + static_cast<int64_t>(k) * rank + h * 8);
                        xv[u] = __half2float(xr[k]);
                    }
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                    {
                        __half const* ah = reinterpret_cast<__half const*>(&av[u]);
#pragma unroll
                        for (int e = 0; e < 8; ++e)
                        {
                            acc[e] = __fmaf_rn(xv[u], __half2float(ah[e]), acc[e]);
                        }
                    }
                }
                for (; i < nIter; ++i)
                {
                    int const k = c0 + q + i * 64;
                    uint4 const av = *reinterpret_cast<uint4 const*>(target.a + static_cast<int64_t>(k) * rank + h * 8);
                    float const xv = __half2float(xr[k]);
                    __half const* ah = reinterpret_cast<__half const*>(&av);
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                    {
                        acc[e] = __fmaf_rn(xv, __half2float(ah[e]), acc[e]);
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < nIter; ++i)
            {
                int const k = c0 + q + i * 64;
                float const xv = __half2float(xr[k]);
#pragma unroll
                for (int e = 0; e < 8; ++e)
                {
                    int const col = h * 8 + e;
                    float const a = col < rank ? __half2float(target.a[static_cast<int64_t>(k) * rank + col]) : 0.f;
                    acc[e] = __fmaf_rn(xv, a, acc[e]);
                }
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            float v = acc[e];
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 16));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 8));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
            if (lane < 2)
            {
                sW[w][h][e] = v;
            }
        }
        __syncthreads();
        if (tid < kLoraShrinkMaxRank)
        {
            int const hh = tid >> 3;
            int const e = tid & 7;
            float const v = __fadd_rn(__fadd_rn(__fadd_rn(sW[0][hh][e], sW[1][hh][e]), sW[2][hh][e]), sW[3][hh][e]);
            p.partial[((static_cast<int64_t>(tg) * numChunks + chunk) * p.m + m) * kLoraShrinkMaxRank + tid] = v;
        }
        __syncthreads();
    }

    // The last block of this (target, row block) reduces the chunks in fixed order.
    __threadfence();
    __syncthreads();
    int32_t* const counter = p.counter + tg * gridDim.z + blockIdx.z;
    if (tid == 0)
    {
        sLast = atomicAdd(counter, 1) == numChunks - 1;
    }
    __syncthreads();
    if (!sLast)
    {
        return;
    }
    __threadfence();
    int const groups = min(kLoraShrinkGroups, numChunks);
    for (int r = 0; r < rows; ++r)
    {
        int const m = row0 + r;
        int const g = tid >> 4;
        int const c = tid & 15;
        if (g < groups)
        {
            float qsum = 0.f;
            for (int j = g; j < numChunks; j += kLoraShrinkGroups)
            {
                float const v
                    = __ldcg(&p.partial[((static_cast<int64_t>(tg) * numChunks + j) * p.m + m) * kLoraShrinkMaxRank + c]);
                qsum = j == g ? v : __fadd_rn(qsum, v);
            }
            sQ[g][c] = qsum;
        }
        __syncthreads();
        if (tid < kLoraShrinkMaxRank)
        {
            float t = sQ[0][tid];
            for (int gg = 1; gg < groups; ++gg)
            {
                t = __fadd_rn(t, sQ[gg][tid]);
            }
            target.t[static_cast<int64_t>(m) * kLoraShrinkMaxRank + tid] = __float2half_rn(t);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        *counter = 0;
    }
}

__global__ void __launch_bounds__(kLoraShrinkThreads) loraShrinkKernel(LoraShrinkParams p)
{
    loraShrinkBody(p);
}

//! Prefill shrink (M >= kLoraShrinkPrefillMinM): tensor-core tiles of 16 rows x 16 rank columns per warp, 64 rows per
//! block, one K chunk (loraShrinkPrefillChunkK) per block, in K tiles of 64 that the block stages
//! cooperatively with register double buffering; every target of the group shares the x tile.
//! Partials [target][chunk][M][16]; the last block of a row block sums the chunks in order and
//! writes t16. Deterministic, but not the decode tree (tensor-core summation).
//! Grid (numChunks, ceil(M / 64)), 128 threads.
__device__ inline void loraShrinkPrefillBody(LoraShrinkParams const& p)
{
    using namespace nvcuda;
    constexpr int kT = 64; // K per staged tile
    int const chunk = blockIdx.x;
    int const chunkK = loraShrinkPrefillChunkK(p.m, p.k);
    int const numChunks = (p.k + chunkK - 1) / chunkK;
    int const c0 = chunk * chunkK;
    int const nTiles = (min(p.k, c0 + chunkK) - c0) / kT; // K % 64 == 0
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const rb0 = blockIdx.y * 64;

    // x tile [64 rows][64 k] and A tiles [targets][64 k][16]; +8 halves of padding per row.
    __shared__ __align__(32) __half sX[64][kT + 8];
    __shared__ __align__(32) __half sA[kLoraShrinkMaxTargets][kT][16 + 8];
    __shared__ __align__(32) float sC[4][16 * 16];
    __shared__ int sLast;

    // Staging assignment: x: 64 rows x 8 pieces of 8 halves = 512 pieces, 4 per thread;
    // A: 64 k rows x 2 pieces per target = 128 pieces, 1 per thread per target.
    uint4 rx[4];
    uint4 ra[kLoraShrinkMaxTargets];
    auto loadTile = [&](int kt) {
        int const k0 = c0 + kt * kT;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int const piece = tid + i * kLoraShrinkThreads;
            int const r = piece >> 3;
            int const c = (piece & 7) * 8;
            rx[i] = make_uint4(0, 0, 0, 0);
            if (rb0 + r < p.m)
            {
                rx[i] = *reinterpret_cast<uint4 const*>(p.x + static_cast<int64_t>(rb0 + r) * p.k + k0 + c);
            }
        }
        int const ak = tid >> 1;
        int const ah = (tid & 1) * 8;
#pragma unroll
        for (int t = 0; t < kLoraShrinkMaxTargets; ++t)
        {
            ra[t] = make_uint4(0, 0, 0, 0);
            if (t < p.numTargets)
            {
                LoraShrinkTarget const& tg = p.target[t];
                if ((tg.rank & 7) == 0)
                {
                    if (ah < tg.rank)
                    {
                        ra[t] = *reinterpret_cast<uint4 const*>(tg.a + static_cast<int64_t>(k0 + ak) * tg.rank + ah);
                    }
                }
                else
                {
                    __half* h = reinterpret_cast<__half*>(&ra[t]);
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                    {
                        h[e] = ah + e < tg.rank ? tg.a[static_cast<int64_t>(k0 + ak) * tg.rank + ah + e]
                                                : __float2half(0.f);
                    }
                }
            }
        }
    };
    auto storeTile = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int const piece = tid + i * kLoraShrinkThreads;
            *reinterpret_cast<uint4*>(&sX[piece >> 3][(piece & 7) * 8]) = rx[i];
        }
#pragma unroll
        for (int t = 0; t < kLoraShrinkMaxTargets; ++t)
        {
            if (t < p.numTargets)
            {
                *reinterpret_cast<uint4*>(&sA[t][tid >> 1][(tid & 1) * 8]) = ra[t];
            }
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kLoraShrinkMaxTargets];
#pragma unroll
    for (int t = 0; t < kLoraShrinkMaxTargets; ++t)
    {
        wmma::fill_fragment(acc[t], 0.f);
    }
    if (nTiles > 0)
    {
        loadTile(0);
    }
    for (int kt = 0; kt < nTiles; ++kt)
    {
        storeTile();
        __syncthreads();
        if (kt + 1 < nTiles)
        {
            loadTile(kt + 1); // in flight while the tensor cores consume this tile
        }
#pragma unroll
        for (int kk = 0; kk < kT; kk += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fx;
            wmma::load_matrix_sync(fx, &sX[warp * 16][kk], kT + 8);
#pragma unroll
            for (int t = 0; t < kLoraShrinkMaxTargets; ++t)
            {
                if (t < p.numTargets && p.target[t].rank > 0)
                {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::load_matrix_sync(fa, &sA[t][kk][0], 16 + 8);
                    wmma::mma_sync(acc[t], fx, fa, acc[t]);
                }
            }
        }
        __syncthreads();
    }
    int const row0 = rb0 + warp * 16;
#pragma unroll
    for (int t = 0; t < kLoraShrinkMaxTargets; ++t)
    {
        if (t >= p.numTargets || p.target[t].rank <= 0)
        {
            continue;
        }
        wmma::store_matrix_sync(sC[warp], acc[t], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 16 * 16; i += 32)
        {
            int const rr = i >> 4;
            if (row0 + rr < p.m)
            {
                p.partial[((static_cast<int64_t>(t) * numChunks + chunk) * p.m + row0 + rr) * kLoraShrinkMaxRank
                    + (i & 15)]
                    = sC[warp][i];
            }
        }
        __syncwarp();
    }

    __threadfence();
    __syncthreads();
    int32_t* const counter = p.counter + blockIdx.y;
    if (tid == 0)
    {
        sLast = atomicAdd(counter, 1) == numChunks - 1;
    }
    __syncthreads();
    if (!sLast)
    {
        return;
    }
    __threadfence();
    int const rows = min(64, p.m - rb0);
    for (int t = 0; t < p.numTargets; ++t)
    {
        if (p.target[t].rank <= 0)
        {
            continue;
        }
        for (int i = tid; i < rows * kLoraShrinkMaxRank; i += kLoraShrinkThreads)
        {
            int const m = rb0 + i / kLoraShrinkMaxRank;
            int const c = i % kLoraShrinkMaxRank;
            float v = 0.f;
            for (int j = 0; j < numChunks; ++j)
            {
                float const pj
                    = __ldcg(&p.partial[((static_cast<int64_t>(t) * numChunks + j) * p.m + m) * kLoraShrinkMaxRank + c]);
                v = j == 0 ? pj : __fadd_rn(v, pj);
            }
            p.target[t].t[static_cast<int64_t>(m) * kLoraShrinkMaxRank + c] = __float2half_rn(v);
        }
    }
    if (tid == 0)
    {
        *counter = 0;
    }
}

__global__ void __launch_bounds__(kLoraShrinkThreads) loraShrinkPrefillKernel(LoraShrinkParams p)
{
    loraShrinkPrefillBody(p);
}

} // namespace kernels
} // namespace trt_edgellm
