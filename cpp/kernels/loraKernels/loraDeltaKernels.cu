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

#include "loraDeltaKernels.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <mma.h>
#include <mutex>
#include <unordered_map>

namespace trt_edgellm
{
namespace kernels
{

namespace
{

constexpr int32_t kSmallThreads{256};
constexpr int32_t kMaxGrid{128};

inline size_t alignUp(size_t v, size_t a)
{
    return (v + a - 1) / a * a;
}

__device__ __forceinline__ float roundHalf(float v)
{
    return __half2float(__float2half_rn(v));
}

__device__ __forceinline__ unsigned long long loadAcquire(unsigned long long const* p)
{
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// Grid-wide barrier for a launch whose CTAs are all co-resident. The counter only ever
// grows by gridDim.x per launch, so every launch starts at a multiple of gridDim.x and
// needs no reset.
__device__ __forceinline__ void gridBarrier(unsigned long long* counter)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        unsigned long long const old = atomicAdd(counter, 1ULL);
        unsigned long long const target = (old / gridDim.x + 1ULL) * gridDim.x;
        while (loadAcquire(counter) < target)
        {
            __nanosleep(32);
        }
        __threadfence();
    }
    __syncthreads();
}

// Single-launch path for M <= MMAX rows.
// Phase 1: CTA g reduces a K slice of x @ A into part[g][m][c] (fp32).
// Phase 2 (after the grid barrier): every CTA sums the partials into t (rounded to fp16),
// then computes its slice of output columns: out = base + fp16(t @ B).
template <int MMAX>
__global__ void __launch_bounds__(kSmallThreads) loraAddSmallKernel(half const* __restrict__ base,
    half const* __restrict__ x, half const* __restrict__ A, half const* __restrict__ B, half* __restrict__ out,
    int32_t M, int32_t K, int32_t N, int32_t r, float* __restrict__ part, unsigned long long* counter)
{
    __shared__ float red[MMAX * kSmallThreads];
    __shared__ float tS[MMAX * kLoraMaxRank];

    int32_t const G = gridDim.x;
    int32_t const tid = threadIdx.x;
    int32_t rp = 1;
    while (rp < r)
    {
        rp <<= 1;
    }
    int32_t const groups = kSmallThreads / rp;
    int32_t const kk = tid / rp;
    int32_t const rr = tid % rp;

    int32_t kPer = (K + G - 1) / G;
    kPer = (kPer + 7) & ~7;
    int32_t const k0 = blockIdx.x * kPer;
    int32_t const k1 = min(K, k0 + kPer);

    float acc[MMAX];
#pragma unroll
    for (int m = 0; m < MMAX; ++m)
    {
        acc[m] = 0.f;
    }
    if (rr < r)
    {
        for (int32_t k = k0 + kk; k < k1; k += groups)
        {
            float const a = __half2float(A[static_cast<size_t>(k) * r + rr]);
#pragma unroll
            for (int m = 0; m < MMAX; ++m)
            {
                if (m < M)
                {
                    acc[m] += __half2float(x[static_cast<size_t>(m) * K + k]) * a;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MMAX; ++m)
    {
        red[m * kSmallThreads + tid] = acc[m];
    }
    __syncthreads();
    for (int32_t o = tid; o < M * r; o += kSmallThreads)
    {
        int32_t const m = o / r;
        int32_t const c = o % r;
        float s = 0.f;
        for (int32_t g = 0; g < groups; ++g)
        {
            s += red[m * kSmallThreads + g * rp + c];
        }
        part[(static_cast<size_t>(blockIdx.x) * M + m) * r + c] = s;
    }

    gridBarrier(counter);

    for (int32_t o = tid; o < M * r; o += kSmallThreads)
    {
        int32_t const m = o / r;
        int32_t const c = o % r;
        float s = 0.f;
        for (int32_t g = 0; g < G; ++g)
        {
            s += __ldcg(&part[(static_cast<size_t>(g) * M + m) * r + c]);
        }
        tS[m * r + c] = roundHalf(s);
    }
    __syncthreads();

    int32_t const pairs = N / 2;
    int32_t const pPer = (pairs + G - 1) / G;
    int32_t const p0 = blockIdx.x * pPer;
    int32_t const p1 = min(pairs, p0 + pPer);
    half2 const* B2 = reinterpret_cast<half2 const*>(B);
    half2 const* base2 = reinterpret_cast<half2 const*>(base);
    half2* out2 = reinterpret_cast<half2*>(out);
    for (int32_t p = p0 + tid; p < p1; p += kSmallThreads)
    {
        float2 d[MMAX];
#pragma unroll
        for (int m = 0; m < MMAX; ++m)
        {
            d[m] = make_float2(0.f, 0.f);
        }
        for (int32_t c = 0; c < r; ++c)
        {
            float2 const b = __half22float2(B2[static_cast<size_t>(c) * pairs + p]);
#pragma unroll
            for (int m = 0; m < MMAX; ++m)
            {
                if (m < M)
                {
                    float const t = tS[m * r + c];
                    d[m].x += t * b.x;
                    d[m].y += t * b.y;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MMAX; ++m)
        {
            if (m < M)
            {
                size_t const idx = static_cast<size_t>(m) * pairs + p;
                float2 const bs = __half22float2(base2[idx]);
                out2[idx] = __floats2half2_rn(bs.x + roundHalf(d[m].x), bs.y + roundHalf(d[m].y));
            }
        }
    }
}

// Tensor-core single-launch path for 2 <= M <= 16 rows and rank <= 16 * RT.
// Phase 1: each CTA stages 128-row K chunks of x (16 rows, zero-padded) and A into shared
// memory; warp w runs one m16n16k16 step per chunk; warps are summed into part[g][m][c].
// Phase 2 (after the grid barrier): t = fp16(sum of partials); each warp owns 16-column
// output tiles: out = base + fp16(t @ B_tile).
constexpr int32_t kTcKChunk{128};
template <int RT>
__global__ void __launch_bounds__(kSmallThreads) loraAddSmallTcKernel(half const* __restrict__ base,
    half const* __restrict__ x, half const* __restrict__ A, half const* __restrict__ B, half* __restrict__ out,
    int32_t M, int32_t K, int32_t N, int32_t r, float* __restrict__ part, unsigned long long* counter)
{
    using namespace nvcuda;
    constexpr int RW = 16 * RT;
    __shared__ __align__(32) half xs[16][kTcKChunk + 8];
    __shared__ __align__(32) half as[kTcKChunk][RW + 8];
    __shared__ __align__(32) float scr[8][16][16];
    __shared__ __align__(32) half ts[16][RW + 8];
    __shared__ __align__(32) half bsw[8][RW][24];

    int32_t const G = gridDim.x;
    int32_t const tid = threadIdx.x;
    int32_t const warp = tid / 32;
    int32_t const lane = tid % 32;

    int32_t kPer = (K + G - 1) / G;
    kPer = (kPer + 15) & ~15;
    int32_t const k0 = blockIdx.x * kPer;
    int32_t const k1 = min(K, k0 + kPer);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[RT];
#pragma unroll
    for (int j = 0; j < RT; ++j)
    {
        wmma::fill_fragment(acc[j], 0.f);
    }
    bool const vecX = (K % 8) == 0;
    for (int32_t kc = k0; kc < k1; kc += kTcKChunk)
    {
        // x chunk: 16 rows x 128 halves, as 8-half vectors when K allows it.
        if (vecX)
        {
            for (int32_t i = tid; i < 16 * (kTcKChunk / 8); i += kSmallThreads)
            {
                int32_t const m = i / (kTcKChunk / 8);
                int32_t const kk = (i % (kTcKChunk / 8)) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (m < M && kc + kk < k1)
                {
                    v = *reinterpret_cast<uint4 const*>(&x[static_cast<size_t>(m) * K + kc + kk]);
                }
                *reinterpret_cast<uint4*>(&xs[m][kk]) = v;
            }
        }
        else
        {
            for (int32_t i = tid; i < 16 * kTcKChunk; i += kSmallThreads)
            {
                int32_t const m = i / kTcKChunk;
                int32_t const kk = i % kTcKChunk;
                xs[m][kk] = (m < M && kc + kk < k1) ? x[static_cast<size_t>(m) * K + kc + kk] : __float2half(0.f);
            }
        }
        for (int32_t i = tid; i < kTcKChunk * RW; i += kSmallThreads)
        {
            int32_t const kk = i / RW;
            int32_t const c = i % RW;
            as[kk][c] = (kc + kk < k1 && c < r) ? A[static_cast<size_t>(kc + kk) * r + c] : __float2half(0.f);
        }
        __syncthreads();
        {
            int32_t const ks = warp * 16;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, &xs[0][ks], kTcKChunk + 8);
#pragma unroll
            for (int j = 0; j < RT; ++j)
            {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, &as[ks][j * 16], RW + 8);
                wmma::mma_sync(acc[j], fa, fb, acc[j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < RT; ++j)
    {
        wmma::store_matrix_sync(&scr[warp][0][0], acc[j], 16, wmma::mem_row_major);
        __syncthreads();
        {
            int32_t const m = tid / 16;
            int32_t const c = j * 16 + tid % 16;
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w)
            {
                s += scr[w][m][tid % 16];
            }
            if (m < M && c < r)
            {
                part[(static_cast<size_t>(blockIdx.x) * M + m) * r + c] = s;
            }
        }
        __syncthreads();
    }

    gridBarrier(counter);

    for (int32_t i = tid; i < 16 * RW; i += kSmallThreads)
    {
        int32_t const m = i / RW;
        int32_t const c = i % RW;
        float s = 0.f;
        if (m < M && c < r)
        {
            for (int32_t g = 0; g < G; ++g)
            {
                s += __ldcg(&part[(static_cast<size_t>(g) * M + m) * r + c]);
            }
        }
        ts[m][c] = __float2half_rn(s);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[RT];
#pragma unroll
    for (int j = 0; j < RT; ++j)
    {
        wmma::load_matrix_sync(fa[j], &ts[0][j * 16], RW + 8);
    }
    int32_t const nTiles = (N + 15) / 16;
    int32_t const tPer = (nTiles + G - 1) / G;
    int32_t const t0 = blockIdx.x * tPer;
    int32_t const t1 = min(nTiles, t0 + tPer);
    bool const vecB = (N % 8) == 0;
    for (int32_t tile = t0 + warp; tile < t1; tile += 8)
    {
        int32_t const n0 = tile * 16;
        // B tile rows [0, RW) x 16 columns: two 8-half vectors per row.
        for (int32_t i = lane; i < RW * 2; i += 32)
        {
            int32_t const c = i / 2;
            int32_t const h8 = (i % 2) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c < r)
            {
                if (vecB && n0 + h8 + 8 <= N)
                {
                    v = *reinterpret_cast<uint4 const*>(&B[static_cast<size_t>(c) * N + n0 + h8]);
                }
                else
                {
                    half tmp[8];
                    for (int e = 0; e < 8; ++e)
                    {
                        int32_t const n = n0 + h8 + e;
                        tmp[e] = n < N ? B[static_cast<size_t>(c) * N + n] : __float2half(0.f);
                    }
                    v = *reinterpret_cast<uint4*>(tmp);
                }
            }
            *reinterpret_cast<uint4*>(&bsw[warp][c][h8]) = v;
        }
        __syncwarp();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> d;
        wmma::fill_fragment(d, 0.f);
#pragma unroll
        for (int j = 0; j < RT; ++j)
        {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
            wmma::load_matrix_sync(fb, &bsw[warp][j * 16][0], 24);
            wmma::mma_sync(d, fa[j], fb, d);
        }
        wmma::store_matrix_sync(&scr[warp][0][0], d, 16, wmma::mem_row_major);
        __syncwarp();
        for (int32_t i = lane; i < M * 16; i += 32)
        {
            int32_t const m = i / 16;
            int32_t const n = n0 + i % 16;
            if (n < N)
            {
                size_t const idx = static_cast<size_t>(m) * N + n;
                out[idx] = __float2half_rn(__half2float(base[idx]) + roundHalf(scr[warp][m][i % 16]));
            }
        }
        __syncwarp();
    }
}

// Large-M path, step 1: split-K partials p[z][M][cols] (fp32) of x @ [A_0 | A_1 | ...], each
// target's rank zero-padded to a multiple of 16 columns. Every CTA stages its 64-row x tile
// once and accumulates all column tiles from it, so siblings sharing x read it once.
// Grid (ceil(M/64), S); 4 warps, each warp owns 16 rows; blockIdx.y owns K range
// [y*kPer, (y+1)*kPer).
__global__ void __launch_bounds__(128) loraShrinkWmmaKernel(half const* __restrict__ x, PrefillTargets const t,
    float* __restrict__ p, int64_t M, int32_t K, int32_t kPer)
{
    using namespace nvcuda;
    constexpr int kMaxTiles = kPrefillMaxCols / 16;
    __shared__ __align__(32) half xs[64][72];
    __shared__ __align__(32) half as[64][kPrefillMaxCols + 8];
    __shared__ __align__(32) float cs[4][16][16];
    int32_t const cols = t.colOff[t.n];
    int32_t const tiles = cols / 16;
    int32_t const warp = threadIdx.x / 32;
    int32_t const lane = threadIdx.x % 32;
    int64_t const row0 = static_cast<int64_t>(blockIdx.x) * 64;
    int32_t const kBeg = blockIdx.y * kPer;
    int32_t const kEnd = min(K, kBeg + kPer);
    bool const vecX = (K % 8) == 0;
    __shared__ half const* chA[kPrefillMaxCols / 8];
    __shared__ int32_t chR[kPrefillMaxCols / 8];
    __shared__ int32_t chC[kPrefillMaxCols / 8];
    int32_t const nCh = cols / 8;
    if (threadIdx.x < nCh)
    {
        int32_t const cc = threadIdx.x * 8;
        int32_t ti = 0;
#pragma unroll
        for (int q = 1; q < kPrefillMaxTargets; ++q)
        {
            if (q < t.n && cc >= t.colOff[q])
            {
                ti = q;
            }
        }
        // Select through a constant-index chain so the parameter struct is never indexed dynamically.
        half const* a = t.A[0];
        int32_t rr = t.r[0];
        int32_t off = t.colOff[0];
        if (ti == 1)
        {
            a = t.A[1];
            rr = t.r[1];
            off = t.colOff[1];
        }
        if (ti == 2)
        {
            a = t.A[2];
            rr = t.r[2];
            off = t.colOff[2];
        }
        chA[threadIdx.x] = a;
        chR[threadIdx.x] = rr;
        chC[threadIdx.x] = cc - off;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kMaxTiles];
#pragma unroll
    for (int j = 0; j < kMaxTiles; ++j)
    {
        wmma::fill_fragment(acc[j], 0.f);
    }
    for (int32_t k0 = kBeg; k0 < kEnd; k0 += 64)
    {
        if (vecX)
        {
            for (int32_t i = threadIdx.x; i < 64 * 8; i += 128)
            {
                int32_t const rr = i / 8;
                int32_t const kk = (i % 8) * 8;
                int64_t const gm = row0 + rr;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (gm < M && k0 + kk < kEnd)
                {
                    v = *reinterpret_cast<uint4 const*>(&x[gm * K + k0 + kk]);
                }
                *reinterpret_cast<uint4*>(&xs[rr][kk]) = v;
            }
        }
        else
        {
            for (int32_t i = threadIdx.x; i < 64 * 64; i += 128)
            {
                int32_t const rr = i / 64;
                int32_t const kk = i % 64;
                int64_t const gm = row0 + rr;
                xs[rr][kk] = (gm < M && k0 + kk < kEnd) ? x[gm * K + k0 + kk] : __float2half(0.f);
            }
        }
        // A chunk: 64 rows x cols, in 8-column pieces mapped once per CTA (chA/chR/chC).
        for (int32_t i = threadIdx.x; i < 64 * nCh; i += 128)
        {
            int32_t const kk = i / nCh;
            int32_t const ch = i - kk * nCh;
            int32_t const gk = k0 + kk;
            int32_t const rr = chR[ch];
            int32_t const c = chC[ch];
            half const* a = chA[ch];
            half* dst = &as[kk][ch * 8];
            if (gk < kEnd && (rr % 8) == 0 && c < rr)
            {
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<uint4 const*>(&a[static_cast<size_t>(gk) * rr + c]);
            }
            else
            {
#pragma unroll
                for (int e = 0; e < 8; ++e)
                {
                    dst[e] = (gk < kEnd && c + e < rr) ? a[static_cast<size_t>(gk) * rr + c + e] : __float2half(0.f);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int32_t ks = 0; ks < 64; ks += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, &xs[warp * 16][ks], 72);
#pragma unroll
            for (int j = 0; j < kMaxTiles; ++j)
            {
                if (j < tiles)
                {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, &as[ks][j * 16], kPrefillMaxCols + 8);
                    wmma::mma_sync(acc[j], fa, fb, acc[j]);
                }
            }
        }
        __syncthreads();
    }
    float* pz = p + static_cast<size_t>(blockIdx.y) * M * cols;
#pragma unroll
    for (int j = 0; j < kMaxTiles; ++j)
    {
        if (j < tiles)
        {
            wmma::store_matrix_sync(&cs[warp][0][0], acc[j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int32_t i = lane; i < 256; i += 32)
            {
                int32_t const rr = i / 16;
                int32_t const cc = i % 16;
                int64_t const gm = row0 + warp * 16 + rr;
                if (gm < M)
                {
                    pz[gm * cols + j * 16 + cc] = cs[warp][rr][cc];
                }
            }
            __syncwarp();
        }
    }
}

// Large-M path, step 2 (streaming): out = base + fp16(fp16(sum_z p[z]) @ B).
// Each thread owns 8 consecutive output columns and keeps its slice of B in registers (rank
// <= 16; larger ranks read B through L1), then streams kRowsPer rows of base/out with
// 16-byte accesses. t for the CTA's rows is reduced from the split-K partials once.
constexpr int32_t kStreamThreads{128};
constexpr int32_t kStreamRowsPer{16};
__global__ void __launch_bounds__(kStreamThreads) loraExpandStreamKernel(half const* __restrict__ base,
    float const* __restrict__ p, int32_t S, int32_t pStride, int32_t pOff, half const* __restrict__ B,
    half* __restrict__ out, int64_t M, int32_t N, int32_t r)
{
    __shared__ float tS[kStreamRowsPer][kLoraMaxRank];
    int64_t const row0 = static_cast<int64_t>(blockIdx.y) * kStreamRowsPer;
    int32_t const rows = static_cast<int32_t>(M - row0 < kStreamRowsPer ? M - row0 : kStreamRowsPer);
    for (int32_t i = threadIdx.x; i < rows * r; i += kStreamThreads)
    {
        int32_t const m = i / r;
        int32_t const c = i % r;
        float v = 0.f;
        for (int32_t z = 0; z < S; ++z)
        {
            v += p[(static_cast<size_t>(z) * M + row0 + m) * pStride + pOff + c];
        }
        tS[m][c] = roundHalf(v);
    }
    __syncthreads();
    int32_t const c0 = (blockIdx.x * kStreamThreads + threadIdx.x) * 8;
    if (c0 >= N)
    {
        return;
    }
    bool const vec = (N % 8) == 0;
    if (vec && r <= 16)
    {
        uint4 bReg[16];
#pragma unroll
        for (int c = 0; c < 16; ++c)
        {
            bReg[c] = c < r ? *reinterpret_cast<uint4 const*>(&B[static_cast<size_t>(c) * N + c0]) : make_uint4(0, 0, 0, 0);
        }
        // Four rows per step: the four base loads are issued before any of them is used.
        constexpr int kU = 4;
        for (int32_t m0 = 0; m0 < rows; m0 += kU)
        {
            uint4 bv[kU];
#pragma unroll
            for (int u = 0; u < kU; ++u)
            {
                if (m0 + u < rows)
                {
                    bv[u] = *reinterpret_cast<uint4 const*>(&base[static_cast<size_t>(row0 + m0 + u) * N + c0]);
                }
            }
#pragma unroll
            for (int u = 0; u < kU; ++u)
            {
                int32_t const m = m0 + u;
                if (m >= rows)
                {
                    break;
                }
                float acc[8];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    acc[j] = 0.f;
                }
#pragma unroll
                for (int c = 0; c < 16; ++c)
                {
                    if (c < r)
                    {
                        float const t = tS[m][c];
                        half2 const* b2 = reinterpret_cast<half2 const*>(&bReg[c]);
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                        {
                            float2 const b = __half22float2(b2[j]);
                            acc[2 * j] += t * b.x;
                            acc[2 * j + 1] += t * b.y;
                        }
                    }
                }
                half2 const* bh = reinterpret_cast<half2 const*>(&bv[u]);
                uint4 ov;
                half2* oh = reinterpret_cast<half2*>(&ov);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float2 const b = __half22float2(bh[j]);
                    oh[j] = __floats2half2_rn(b.x + roundHalf(acc[2 * j]), b.y + roundHalf(acc[2 * j + 1]));
                }
                *reinterpret_cast<uint4*>(&out[static_cast<size_t>(row0 + m) * N + c0]) = ov;
            }
        }
        return;
    }
    // Generic: any rank up to kLoraMaxRank, any N.
    for (int32_t m = 0; m < rows; ++m)
    {
        for (int32_t j = 0; j < 8; ++j)
        {
            int32_t const n = c0 + j;
            if (n >= N)
            {
                break;
            }
            float acc = 0.f;
            for (int32_t c = 0; c < r; ++c)
            {
                acc += tS[m][c] * __half2float(B[static_cast<size_t>(c) * N + n]);
            }
            size_t const idx = static_cast<size_t>(row0 + m) * N + n;
            out[idx] = __float2half_rn(__half2float(base[idx]) + roundHalf(acc));
        }
    }
}

// Split-K factor for the large-M shrink: aim for >= 64 CTAs, keep S * mTiles <= 128 so the
// partials fit loraAddWorkspaceBytes, and keep >= 128 K per split.
int32_t largeSplit(int64_t M, int32_t K)
{
    int64_t const mTiles = (M + 63) / 64;
    int64_t const tiles = mTiles;
    int64_t s = (64 + tiles - 1) / tiles;
    s = std::min<int64_t>(s, std::max<int64_t>(1, K / 128));
    s = std::min<int64_t>(s, std::max<int64_t>(1, 128 / mTiles));
    s = std::min<int64_t>(s, 32);
    return static_cast<int32_t>(std::max<int64_t>(s, 1));
}

//! Cooperative grid for the current device (one CTA per SM, clamped), cached per device.
int32_t smCount()
{
    static std::mutex sMutex;
    static std::unordered_map<int, int32_t> sByDevice;
    int dev{0};
    if (cudaGetDevice(&dev) != cudaSuccess)
    {
        return 0;
    }
    std::lock_guard<std::mutex> lock(sMutex);
    auto const found = sByDevice.find(dev);
    if (found != sByDevice.end())
    {
        return found->second;
    }
    int count{0};
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
    {
        (void) cudaGetLastError();
        return 0;
    }
    return sByDevice[dev] = std::min<int32_t>(count, kMaxGrid);
}

template <typename Kernel>
int32_t launchCoop(Kernel kernel, half const* base, half const* x, half const* A, half const* B, half* out, int32_t M,
    int32_t K, int32_t N, int32_t r, float* part, unsigned long long* counter, int32_t G, cudaStream_t stream)
{
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(G);
    cfg.blockDim = dim3(kSmallThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, base, x, A, B, out, M, K, N, r, part, counter) == cudaSuccess ? 0 : -1;
}

template <int MMAX>
int32_t launchSmall(half const* base, half const* x, half const* A, half const* B, half* out, int32_t M, int32_t K,
    int32_t N, int32_t r, float* part, unsigned long long* counter, int32_t G, cudaStream_t stream)
{
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(G);
    cfg.blockDim = dim3(kSmallThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t const err
        = cudaLaunchKernelEx(&cfg, loraAddSmallKernel<MMAX>, base, x, A, B, out, M, K, N, r, part, counter);
    return err == cudaSuccess ? 0 : -1;
}

} // namespace

size_t loraAddWorkspaceBytes(int64_t maxM, int32_t maxRank) noexcept
{
    int32_t const rp16 = (std::max(maxRank, 1) + 15) / 16 * 16;
    size_t const small = alignUp(static_cast<size_t>(kMaxGrid) * kLoraSmallMaxM * kLoraMaxRank * sizeof(float), 256);
    size_t const large = loraPrefillWorkspaceBytes(maxM, rp16);
    return std::max(small, large);
}

size_t loraPrefillWorkspaceBytes(int64_t maxM, int32_t cols) noexcept
{
    // Split-K partials: S * mTiles <= 128, so S * M <= max(M, 128 * 64) rows of `cols` floats.
    return alignUp(static_cast<size_t>(std::max<int64_t>(maxM, 128 * 64)) * std::max(cols, 16) * sizeof(float), 256);
}

int32_t launchLoraPrefill(void const* x, int64_t M, int32_t K, int32_t n, void const* const* A,
    void const* const* B, void const* const* base, void* const* out, int32_t const* r, int32_t const* N,
    void* workspace, cudaStream_t stream) noexcept
{
    if (M <= 0 || n <= 0)
    {
        return 0;
    }
    if (n > kPrefillMaxTargets || workspace == nullptr || K <= 0)
    {
        return -1;
    }
    PrefillTargets t{};
    t.n = n;
    int32_t cols = 0;
    for (int32_t i = 0; i < n; ++i)
    {
        if (r[i] <= 0 || r[i] > kLoraMaxRank || N[i] <= 0)
        {
            return -1;
        }
        t.A[i] = static_cast<__half const*>(A[i]);
        t.r[i] = r[i];
        t.colOff[i] = cols;
        cols += (r[i] + 15) / 16 * 16;
    }
    t.colOff[n] = cols;
    if (cols > kPrefillMaxCols)
    {
        for (int32_t i = 0; i < n; ++i)
        {
            if (launchLoraPrefill(x, M, K, 1, &A[i], &B[i], &base[i], &out[i], &r[i], &N[i], workspace, stream) != 0)
            {
                return -1;
            }
        }
        return 0;
    }
    auto* p = static_cast<float*>(workspace);
    int32_t const S = largeSplit(M, K);
    int32_t kPer = (K + S - 1) / S;
    kPer = (kPer + 63) / 64 * 64;
    int32_t const sUsed = (K + kPer - 1) / kPer;
    dim3 const gridShrink(static_cast<uint32_t>((M + 63) / 64), static_cast<uint32_t>(sUsed));
    loraShrinkWmmaKernel<<<gridShrink, 128, 0, stream>>>(static_cast<half const*>(x), t, p, M, K, kPer);
    for (int32_t i = 0; i < n; ++i)
    {
        dim3 const gridExpand(static_cast<uint32_t>((N[i] + 8 * kStreamThreads - 1) / (8 * kStreamThreads)),
            static_cast<uint32_t>((M + kStreamRowsPer - 1) / kStreamRowsPer));
        loraExpandStreamKernel<<<gridExpand, kStreamThreads, 0, stream>>>(static_cast<half const*>(base[i]), p, sUsed,
            cols, t.colOff[i], static_cast<half const*>(B[i]), static_cast<half*>(out[i]), M, N[i], r[i]);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t launchLoraAdd(void const* base, void const* x, void const* loraA, void const* loraB, void* out, int64_t M,
    int32_t K, int32_t N, int32_t rank, void* workspace, void* barrierState, cudaStream_t stream) noexcept
{
    if (M <= 0)
    {
        return 0;
    }
    if (rank <= 0)
    {
        return cudaMemcpyAsync(out, base, static_cast<size_t>(M) * N * sizeof(half), cudaMemcpyDeviceToDevice, stream)
                == cudaSuccess
            ? 0
            : -1;
    }
    if (rank > kLoraMaxRank || K <= 0 || N <= 0 || workspace == nullptr)
    {
        return -1;
    }
    auto const* hb = static_cast<half const*>(base);
    auto const* hx = static_cast<half const*>(x);
    auto const* hA = static_cast<half const*>(loraA);
    auto const* hB = static_cast<half const*>(loraB);
    auto* ho = static_cast<half*>(out);
    if (M <= kLoraSmallMaxM && N % 2 == 0 && barrierState != nullptr)
    {
        int32_t const G = smCount();
        if (G <= 0)
        {
            return -1;
        }
        auto* part = static_cast<float*>(workspace);
        auto* counter = static_cast<unsigned long long*>(barrierState);
        int32_t const m = static_cast<int32_t>(M);
        // M = 1 is latency-bound and stays on CUDA cores; 2..16 rows use tensor cores.
        bool const tc = rank <= 32 && m >= 2;
        if (tc)
        {
            return rank <= 16
                ? launchCoop(loraAddSmallTcKernel<1>, hb, hx, hA, hB, ho, m, K, N, rank, part, counter, G, stream)
                : launchCoop(loraAddSmallTcKernel<2>, hb, hx, hA, hB, ho, m, K, N, rank, part, counter, G, stream);
        }
        if (m <= 1)
            return launchSmall<1>(hb, hx, hA, hB, ho, m, K, N, rank, part, counter, G, stream);
        if (m <= 2)
            return launchSmall<2>(hb, hx, hA, hB, ho, m, K, N, rank, part, counter, G, stream);
        if (m <= 4)
            return launchSmall<4>(hb, hx, hA, hB, ho, m, K, N, rank, part, counter, G, stream);
        if (m <= 8)
            return launchSmall<8>(hb, hx, hA, hB, ho, m, K, N, rank, part, counter, G, stream);
        return launchSmall<16>(hb, hx, hA, hB, ho, m, K, N, rank, part, counter, G, stream);
    }
    void const* As[1] = {loraA};
    void const* Bs[1] = {loraB};
    void const* bases[1] = {base};
    void* outs[1] = {out};
    int32_t const rs[1] = {rank};
    int32_t const Ns[1] = {N};
    return launchLoraPrefill(x, M, K, 1, As, Bs, bases, outs, rs, Ns, workspace, stream);
}

} // namespace kernels
} // namespace trt_edgellm
