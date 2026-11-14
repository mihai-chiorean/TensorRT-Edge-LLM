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
#include "loraGroupKernels.h"

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

#ifndef LORA_GROUP_THREADS
#define LORA_GROUP_THREADS 512
#endif
constexpr int32_t kThreads{LORA_GROUP_THREADS};
constexpr int32_t kWarps{kThreads / 32};
constexpr int32_t kMaxGrid{128};
// Grid of the cooperative kernel (one CTA per SM), clamped to this.
constexpr int32_t kMaxBarrierGrid{32};
constexpr size_t kMaxDynSmem{160 * 1024};

struct GroupParams
{
    half const* x;
    half const* A[kLoraGroupMaxTargets];
    half const* B[kLoraGroupMaxTargets];
    half const* base[kLoraGroupMaxTargets];
    half* out[kLoraGroupMaxTargets];
    int32_t r[kLoraGroupMaxTargets];
    int32_t N[kLoraGroupMaxTargets];
    int32_t colOff[kLoraGroupMaxTargets + 1];  // padded t-column offsets (multiples of 16)
    int32_t tileOff[kLoraGroupMaxTargets + 1]; // cumulative 16-column output tiles
    int32_t n;
    int32_t M;
    int32_t K;
    int32_t kPer;     // K rows per CTA (multiple of 16)
    int32_t tilesPer; // output tiles per CTA
    int32_t tileCap;  // output tiles staged in shared memory per round (<= tilesPer)
    int32_t rtMax;    // max over targets of ceil(r/16)
    float* part;      // [G][16][cols]
    unsigned long long* counter;
};

__device__ __forceinline__ float roundHalf(float v)
{
    return __half2float(__float2half_rn(v));
}

__device__ __forceinline__ void cpAsync16(void* smem, void const* gmem, bool pred)
{
    unsigned const s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    int const size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(size) : "memory");
}

__device__ __forceinline__ void cpAsyncCommit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cpAsyncWait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ unsigned long long loadAcquire(unsigned long long const* p)
{
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void gridBarrier(unsigned long long* counter)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        unsigned long long const old = atomicAdd(counter, 1ULL);
        unsigned long long const target = (old / gridDim.x + 1ULL) * gridDim.x;
        // Deliberately no __nanosleep: one thread per CTA polls, and every microsecond of
        // wake-up latency is paid by all 168 group kernels of a decode step.
        while (loadAcquire(counter) < target)
        {
        }
        __threadfence();
    }
    __syncthreads();
}

struct SmemLayout
{
    int32_t xsLd, asLd, cols;
    size_t xs, as, bs, bm, scr, ts, total;
};

__host__ __device__ inline size_t align128(size_t v)
{
    return (v + 127) / 128 * 128;
}

__host__ __device__ inline SmemLayout smemLayout(int32_t kPer, int32_t cols, int32_t tilesPer, int32_t rtMax)
{
    SmemLayout L{};
    L.xsLd = kPer + 8;
    L.asLd = cols + 8;
    L.cols = cols;
    size_t off = 0;
    L.xs = off;
    off = align128(off + size_t(16) * L.xsLd * 2);
    L.as = off;
    off = align128(off + size_t(kPer) * L.asLd * 2);
    L.bs = off;
    off = align128(off + size_t(tilesPer) * rtMax * 16 * 16 * 2);
    L.bm = off;
    off = align128(off + size_t(tilesPer) * 16 * 16 * 2);
    L.scr = off;
    off = align128(off + size_t(kWarps) * 16 * cols * 4);
    L.ts = off;
    off = align128(off + size_t(16) * L.asLd * 2);
    L.total = off;
    return L;
}

__device__ __forceinline__ int32_t targetOfTile(GroupParams const& p, int32_t gt)
{
    int32_t i = 0;
    while (i + 1 < p.n && gt >= p.tileOff[i + 1])
    {
        ++i;
    }
    return i;
}

__global__ void __launch_bounds__(kThreads, 1) loraGroupSmallKernel(GroupParams const p)
{
    using namespace nvcuda;
    extern __shared__ __align__(128) unsigned char smem[];
    int32_t const cols = p.colOff[p.n];
    SmemLayout const L = smemLayout(p.kPer, cols, p.tileCap, p.rtMax);
    half* xs = reinterpret_cast<half*>(smem + L.xs);
    half* as = reinterpret_cast<half*>(smem + L.as);
    half* bs = reinterpret_cast<half*>(smem + L.bs);
    half* bm = reinterpret_cast<half*>(smem + L.bm);
    float* scr = reinterpret_cast<float*>(smem + L.scr);
    half* ts = reinterpret_cast<half*>(smem + L.ts);

    int32_t const tid = threadIdx.x;
    int32_t const warp = tid / 32;
    int32_t const lane = tid % 32;
    int32_t const G = gridDim.x;
    int32_t const k0 = blockIdx.x * p.kPer;
    int32_t const kLen = max(0, min(p.K, k0 + p.kPer) - k0);
    int32_t const totalTiles = p.tileOff[p.n];
    int32_t const t0 = blockIdx.x * p.tilesPer;
    int32_t const nT = max(0, min(totalTiles, t0 + p.tilesPer) - t0);
    int32_t const bRows = p.rtMax * 16;

    // Stages B and base for tiles [first, first + count) of this CTA into the round buffers:
    // per tile, rtMax*16 B rows and 16 base rows, two 16-byte chunks each, in row-major order
    // (row, tile, half) so consecutive threads read consecutive 16 bytes.
    auto stageTiles = [&](int32_t first, int32_t count) {
        int32_t const perTile = (bRows + 16) * 2;
        int32_t const rowLen = count * 2;
        for (int32_t i = tid; i < count * perTile; i += kThreads)
        {
            int32_t const row = i / rowLen;
            int32_t const j = i - row * rowLen;
            int32_t const li = j >> 1;
            int32_t const h = (j & 1) * 8;
            int32_t const gt = t0 + first + li;
            int32_t const ti = targetOfTile(p, gt);
            int32_t const N = p.N[ti];
            int32_t const n0 = (gt - p.tileOff[ti]) * 16;
            if (row < bRows)
            {
                bool const ok = row < p.r[ti] && n0 + h < N;
                cpAsync16(&bs[(static_cast<size_t>(li) * bRows + row) * 16 + h],
                    ok ? &p.B[ti][static_cast<size_t>(row) * N + n0 + h] : p.x, ok);
            }
            else
            {
                int32_t const m = row - bRows;
                bool const ok = m < p.M && n0 + h < N;
                cpAsync16(&bm[(static_cast<size_t>(li) * 16 + m) * 16 + h],
                    ok ? &p.base[ti][static_cast<size_t>(m) * N + n0 + h] : p.x, ok);
            }
        }
    };

    // ---- Issue every global load this CTA needs, all at once. ----
    // x slice: 16 rows (rows >= M zero) x kPer columns.
    {
        int32_t const chunks = p.kPer / 8;
        for (int32_t i = tid; i < 16 * chunks; i += kThreads)
        {
            int32_t const m = i / chunks;
            int32_t const kk = (i % chunks) * 8;
            bool const ok = m < p.M && kk < kLen;
            cpAsync16(&xs[m * L.xsLd + kk], ok ? &p.x[static_cast<size_t>(m) * p.K + k0 + kk] : p.x, ok);
        }
    }
    // A slices: kPer rows x (per target, RT*16 columns).
    for (int32_t ti = 0; ti < p.n; ++ti)
    {
        int32_t const r = p.r[ti];
        int32_t const w = p.colOff[ti + 1] - p.colOff[ti];
        int32_t const chunks = w / 8;
        for (int32_t i = tid; i < p.kPer * chunks; i += kThreads)
        {
            int32_t const kk = i / chunks;
            int32_t const c = (i % chunks) * 8;
            bool const ok = kk < kLen && c < r;
            cpAsync16(
                &as[kk * L.asLd + p.colOff[ti] + c], ok ? &p.A[ti][static_cast<size_t>(k0 + kk) * r + c] : p.x, ok);
        }
    }
    cpAsyncCommit(); // group 0: x and A (phase 1)
    // B and base tiles of the first round (all tiles when they fit, which is the common case).
    stageTiles(0, min(nT, p.tileCap));
    cpAsyncCommit(); // group 1: B and base (phase 2), still in flight during phase 1 and the barrier
    cpAsyncWait<1>();
    __syncthreads();

    // ---- Phase 1: partial shrink over this CTA's K slice. ----
    int32_t const rtTot = cols / 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kLoraGroupMaxCols / 16];
#pragma unroll
    for (int j = 0; j < kLoraGroupMaxCols / 16; ++j)
    {
        wmma::fill_fragment(acc[j], 0.f);
    }
    for (int32_t ks = warp * 16; ks < p.kPer; ks += kWarps * 16)
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
        wmma::load_matrix_sync(fa, &xs[ks], L.xsLd);
#pragma unroll
        for (int j = 0; j < kLoraGroupMaxCols / 16; ++j)
        {
            if (j < rtTot)
            {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, &as[ks * L.asLd + j * 16], L.asLd);
                wmma::mma_sync(acc[j], fa, fb, acc[j]);
            }
        }
    }
    float* myScr = scr + static_cast<size_t>(warp) * 16 * cols;
#pragma unroll
    for (int j = 0; j < kLoraGroupMaxCols / 16; ++j)
    {
        if (j < rtTot)
        {
            wmma::store_matrix_sync(&myScr[j * 16], acc[j], cols, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int32_t i = tid; i < p.M * cols; i += kThreads)
    {
        int32_t const m = i / cols;
        int32_t const c = i % cols;
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            s += scr[(w * 16 + m) * cols + c];
        }
        p.part[(static_cast<size_t>(blockIdx.x) * 16 + m) * cols + c] = s;
    }

    gridBarrier(p.counter);
    cpAsyncWait<0>();
    __syncthreads();

    // ---- Phase 2: t = fp16(sum of partials); expand + add for this CTA's tiles. ----
    // Partials: one thread per output (m, c), all G loads in flight at once.
    for (int32_t i = tid; i < 16 * cols; i += kThreads)
    {
        int32_t const m = i / cols;
        int32_t const c = i % cols;
        float s = 0.f;
        if (m < p.M)
        {
            float v[kMaxBarrierGrid];
#pragma unroll
            for (int g = 0; g < kMaxBarrierGrid; ++g)
            {
                v[g] = g < G ? __ldcg(&p.part[(static_cast<size_t>(g) * 16 + m) * cols + c]) : 0.f;
            }
#pragma unroll
            for (int g = 0; g < kMaxBarrierGrid; ++g)
            {
                s += v[g];
            }
        }
        ts[m * L.asLd + c] = __float2half_rn(s);
    }
    __syncthreads();
    float* wScr = scr + static_cast<size_t>(warp) * 256;
    for (int32_t r0 = 0; r0 < nT; r0 += p.tileCap)
    {
        int32_t const nR = min(p.tileCap, nT - r0);
        if (r0 > 0)
        {
            // Later rounds (small GPUs): reuse the round buffers once every warp is done with them.
            __syncthreads();
            stageTiles(r0, nR);
            cpAsyncCommit();
            cpAsyncWait<0>();
            __syncthreads();
        }
        for (int32_t li = warp; li < nR; li += kWarps)
        {
            int32_t const gt = t0 + r0 + li;
            int32_t const ti = targetOfTile(p, gt);
            int32_t const N = p.N[ti];
            int32_t const n0 = (gt - p.tileOff[ti]) * 16;
            int32_t const rt = (p.colOff[ti + 1] - p.colOff[ti]) / 16;
            half const* bt = bs + static_cast<size_t>(li) * p.rtMax * 256;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> d;
            wmma::fill_fragment(d, 0.f);
            for (int32_t j = 0; j < rt; ++j)
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, &ts[p.colOff[ti] + j * 16], L.asLd);
                wmma::load_matrix_sync(fb, &bt[j * 256], 16);
                wmma::mma_sync(d, fa, fb, d);
            }
            wmma::store_matrix_sync(wScr, d, 16, wmma::mem_row_major);
            __syncwarp();
            half const* mt = bm + static_cast<size_t>(li) * 256;
            half* out = p.out[ti];
            for (int32_t e = lane; e < p.M * 8; e += 32)
            {
                int32_t const m = e / 8;
                int32_t const c = (e % 8) * 2;
                if (n0 + c < N)
                {
                    float2 const b = __half22float2(*reinterpret_cast<half2 const*>(&mt[m * 16 + c]));
                    *reinterpret_cast<half2*>(&out[static_cast<size_t>(m) * N + n0 + c])
                        = __floats2half2_rn(b.x + roundHalf(wScr[m * 16 + c]), b.y + roundHalf(wScr[m * 16 + c + 1]));
                }
            }
            __syncwarp();
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// Short prefill (kLoraSmallMaxM < M <= kLoraGroupMidMaxM): one cooperative launch per group.
//
// Same split as loraGroupSmallKernel (phase 1: split-K shrink over one CTA per SM; grid barrier;
// phase 2: every CTA reduces t and expands its own 16-column output tiles), with RT = M/16
// rounded up to 2 or 4 WMMA row tiles. x and A are too large to stage whole at these widths, so
// they stream through a ring of `stages` K chunks (every chunk is in flight at entry when the
// ring holds them all); B tiles are staged at entry as before; base is not staged: each lane
// loads its 16-byte slice of every row tile of a tile straight into registers before the MMA.
// ---------------------------------------------------------------------------------------------

struct MidLayout
{
    int32_t xcLd, acLd, tsLd;
    size_t stage, xc, ac, ring, scr, ts, bs, total;
};

__host__ __device__ inline int32_t midChunkRows(int32_t rowTiles)
{
    // kWarps = rowTiles x (kWarps / rowTiles) K lanes; one 16-row K step per warp per chunk.
    return 16 * (kWarps / rowTiles);
}

__host__ __device__ inline MidLayout midLayout(
    int32_t rowTiles, int32_t cols, int32_t stages, int32_t tileCap, int32_t rtMax)
{
    MidLayout L{};
    int32_t const kc = midChunkRows(rowTiles);
    L.xcLd = kc + 8;
    L.acLd = cols + 8;
    L.tsLd = cols + 8;
    L.xc = 0;
    L.ac = align128(size_t(16) * rowTiles * L.xcLd * 2);
    L.stage = align128(L.ac + size_t(kc) * L.acLd * 2);
    size_t off = 0;
    off += L.stage * stages;
    L.ring = 0;
    L.scr = align128(off);
    off = L.scr + size_t(kWarps) * 256 * 4;
    L.ts = align128(off);
    off = L.ts + size_t(16) * rowTiles * L.tsLd * 2;
    L.bs = align128(off);
    off = L.bs + size_t(tileCap) * rtMax * 16 * 16 * 2;
    L.total = align128(off);
    return L;
}

//! cp.async.wait_group with a run-time count (ring depth is a launch parameter).
__device__ __forceinline__ void cpAsyncWaitN(int32_t n)
{
    switch (n)
    {
    case 0: cpAsyncWait<0>(); break;
    case 1: cpAsyncWait<1>(); break;
    case 2: cpAsyncWait<2>(); break;
    case 3: cpAsyncWait<3>(); break;
    case 4: cpAsyncWait<4>(); break;
    default: cpAsyncWait<0>(); break; // conservative: wait for everything
    }
}

template <int RT>
__global__ void __launch_bounds__(kThreads, 1) loraGroupMidKernel(GroupParams const p, int32_t const stages)
{
    using namespace nvcuda;
    static_assert(kWarps % RT == 0, "every row tile needs the same number of K lanes");
    constexpr int32_t kWk = kWarps / RT; // warps per row tile along K
    constexpr int32_t kKc = 16 * kWk;    // K rows per chunk
    constexpr int32_t kRows = 16 * RT;   // padded rows
    constexpr int32_t kMaxCt = kLoraGroupMaxCols / 16;
    extern __shared__ __align__(128) unsigned char smem[];
    int32_t const cols = p.colOff[p.n];
    MidLayout const L = midLayout(RT, cols, stages, p.tileCap, p.rtMax);
    float* scr = reinterpret_cast<float*>(smem + L.scr);
    half* ts = reinterpret_cast<half*>(smem + L.ts);
    half* bs = reinterpret_cast<half*>(smem + L.bs);

    int32_t const tid = threadIdx.x;
    int32_t const warp = tid / 32;
    int32_t const lane = tid % 32;
    int32_t const G = gridDim.x;
    int32_t const k0 = blockIdx.x * p.kPer;
    int32_t const kLen = max(0, min(p.K, k0 + p.kPer) - k0);
    int32_t const nChunks = (kLen + kKc - 1) / kKc;
    int32_t const totalTiles = p.tileOff[p.n];
    int32_t const t0 = blockIdx.x * p.tilesPer;
    int32_t const nT = max(0, min(totalTiles, t0 + p.tilesPer) - t0);
    int32_t const bRows = p.rtMax * 16;

    // x rows [0, kRows) (rows >= M zero) and A rows of K chunk c into ring stage c % stages.
    auto stageChunk = [&](int32_t c) {
        unsigned char* st = smem + L.ring + static_cast<size_t>(c % stages) * L.stage;
        half* xc = reinterpret_cast<half*>(st + L.xc);
        half* ac = reinterpret_cast<half*>(st + L.ac);
        int32_t const kc0 = c * kKc; // relative to k0
        constexpr int32_t xChunks = kKc / 8;
        for (int32_t i = tid; i < kRows * xChunks; i += kThreads)
        {
            int32_t const m = i / xChunks;
            int32_t const kk = (i % xChunks) * 8;
            bool const ok = m < p.M && kc0 + kk < kLen;
            cpAsync16(&xc[m * L.xcLd + kk], ok ? &p.x[static_cast<size_t>(m) * p.K + k0 + kc0 + kk] : p.x, ok);
        }
        int32_t const aChunks = cols / 8;
        for (int32_t i = tid; i < kKc * aChunks; i += kThreads)
        {
            int32_t const kk = i / aChunks;
            int32_t const cc = (i % aChunks) * 8;
            int32_t ti = 0;
            while (ti + 1 < p.n && cc >= p.colOff[ti + 1])
            {
                ++ti;
            }
            int32_t const c = cc - p.colOff[ti];
            int32_t const r = p.r[ti];
            bool const ok = kc0 + kk < kLen && c < r;
            cpAsync16(&ac[kk * L.acLd + cc], ok ? &p.A[ti][static_cast<size_t>(k0 + kc0 + kk) * r + c] : p.x, ok);
        }
    };
    // B rows of tiles [first, first + count) of this CTA (base is read in phase 2).
    auto stageB = [&](int32_t first, int32_t count) {
        int32_t const rowLen = count * 2;
        for (int32_t i = tid; i < count * bRows * 2; i += kThreads)
        {
            int32_t const row = i / rowLen;
            int32_t const j = i - row * rowLen;
            int32_t const li = j >> 1;
            int32_t const h = (j & 1) * 8;
            int32_t const gt = t0 + first + li;
            int32_t const ti = targetOfTile(p, gt);
            int32_t const N = p.N[ti];
            int32_t const n0 = (gt - p.tileOff[ti]) * 16;
            bool const ok = row < p.r[ti] && n0 + h < N;
            cpAsync16(&bs[(static_cast<size_t>(li) * bRows + row) * 16 + h],
                ok ? &p.B[ti][static_cast<size_t>(row) * N + n0 + h] : p.x, ok);
        }
    };

    // ---- Entry: the first `stages` chunks, then the first round of B tiles. ----
    int32_t const pre = min(stages, nChunks);
    for (int32_t c = 0; c < pre; ++c)
    {
        stageChunk(c);
        cpAsyncCommit();
    }
    stageB(0, min(nT, p.tileCap));
    cpAsyncCommit();

    // ---- Phase 1: shrink over the ring. Warp w owns row tile w / kWk and K lane w % kWk. ----
    int32_t const myRt = warp / kWk;
    int32_t const myK = (warp % kWk) * 16;
    int32_t const ct = cols / 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kMaxCt];
#pragma unroll
    for (int j = 0; j < kMaxCt; ++j)
    {
        wmma::fill_fragment(acc[j], 0.f);
    }
    for (int32_t c = 0; c < nChunks; ++c)
    {
        // Groups committed after chunk c: the rest of the ring window, B, and refills so far.
        int32_t const committed = min(nChunks, c + stages);
        int32_t const pos = c < stages ? c : c + 1;
        cpAsyncWaitN(committed - pos);
        __syncthreads();
        unsigned char const* st = smem + L.ring + static_cast<size_t>(c % stages) * L.stage;
        half const* xc = reinterpret_cast<half const*>(st + L.xc);
        half const* ac = reinterpret_cast<half const*>(st + L.ac);
        if (c * kKc + myK < kLen)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, &xc[myRt * 16 * L.xcLd + myK], L.xcLd);
#pragma unroll
            for (int j = 0; j < kMaxCt; ++j)
            {
                if (j < ct)
                {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, &ac[myK * L.acLd + j * 16], L.acLd);
                    wmma::mma_sync(acc[j], fa, fb, acc[j]);
                }
            }
        }
        __syncthreads();
        if (c + stages < nChunks)
        {
            stageChunk(c + stages);
            cpAsyncCommit();
        }
    }
    // Sum the kWk K lanes of each row tile, one 16-column tile at a time, into part[g][m][c].
    float* wScr = scr + static_cast<size_t>(warp) * 256;
#pragma unroll
    for (int j = 0; j < kMaxCt; ++j)
    {
        if (j < ct)
        {
            wmma::store_matrix_sync(wScr, acc[j], 16, wmma::mem_row_major);
            __syncthreads();
            for (int32_t i = tid; i < kRows * 16; i += kThreads)
            {
                int32_t const m = i / 16;
                int32_t const cc = i % 16;
                if (m < p.M)
                {
                    int32_t const rt = m / 16;
                    float s = 0.f;
#pragma unroll
                    for (int w = 0; w < kWk; ++w)
                    {
                        s += scr[(rt * kWk + w) * 256 + (m % 16) * 16 + cc];
                    }
                    p.part[(static_cast<size_t>(blockIdx.x) * p.M + m) * cols + j * 16 + cc] = s;
                }
            }
            __syncthreads();
        }
    }

    // Base slice of CTA tile `tile` (all row tiles) for this lane: row l / 2, 8 columns at (l % 2) * 8.
    // Out of range (no such tile, rows >= M, columns >= N) reads nothing and stays zero.
    int32_t const lr = lane / 2;
    int32_t const lc = (lane % 2) * 8;
    auto loadBase = [&](int32_t tile, uint4* v) {
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
        {
            v[rt] = make_uint4(0, 0, 0, 0);
        }
        if (tile >= nT)
        {
            return;
        }
        int32_t const gt = t0 + tile;
        int32_t const ti = targetOfTile(p, gt);
        int32_t const N = p.N[ti];
        int32_t const n0 = (gt - p.tileOff[ti]) * 16;
        if (n0 + lc >= N)
        {
            return;
        }
        half const* base = p.base[ti];
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
        {
            int32_t const m = rt * 16 + lr;
            if (m < p.M)
            {
                v[rt] = *reinterpret_cast<uint4 const*>(&base[static_cast<size_t>(m) * N + n0 + lc]);
            }
        }
    };
    // The first tile's base does not depend on t: its loads overlap the barrier wait.
    uint4 bv[RT];
    loadBase(warp, bv);

    gridBarrier(p.counter);
    cpAsyncWait<0>();
    __syncthreads();

    // ---- Phase 2: t = fp16(sum of partials) for the columns this CTA's tiles use. ----
    int32_t cLo = 0;
    int32_t cHi = 0;
    if (nT > 0)
    {
        cLo = p.colOff[targetOfTile(p, t0)];
        cHi = p.colOff[targetOfTile(p, t0 + nT - 1) + 1];
    }
    int32_t const w = cHi - cLo;
    for (int32_t i = tid; i < kRows * w; i += kThreads)
    {
        int32_t const m = i / w;
        int32_t const c = cLo + i % w;
        float s = 0.f;
        if (m < p.M)
        {
            float v[kMaxBarrierGrid];
#pragma unroll
            for (int g = 0; g < kMaxBarrierGrid; ++g)
            {
                v[g] = g < G ? __ldcg(&p.part[(static_cast<size_t>(g) * p.M + m) * cols + c]) : 0.f;
            }
#pragma unroll
            for (int g = 0; g < kMaxBarrierGrid; ++g)
            {
                s += v[g];
            }
        }
        ts[m * L.tsLd + c] = __float2half_rn(s);
    }
    __syncthreads();

    // Expand + add: warp per tile; lane l owns row l / 2 and columns (l % 2) * 8 .. + 8 of each
    // 16 x 16 row tile. Base is the bandwidth term here (M rows of every output column, read and
    // written once), so each warp keeps two tiles' base slices in flight: the next tile's loads are
    // issued before the current tile's MMA, and the first tile's before the grid barrier above.
    for (int32_t r0 = 0; r0 < nT; r0 += p.tileCap)
    {
        int32_t const nR = min(p.tileCap, nT - r0);
        if (r0 > 0)
        {
            __syncthreads();
            stageB(r0, nR);
            cpAsyncCommit();
            cpAsyncWait<0>();
            __syncthreads();
            loadBase(r0 + warp, bv);
        }
        for (int32_t li = warp; li < nR; li += kWarps)
        {
            uint4 bvNext[RT];
            if (li + kWarps < nR)
            {
                loadBase(r0 + li + kWarps, bvNext);
            }
            int32_t const gt = t0 + r0 + li;
            int32_t const ti = targetOfTile(p, gt);
            int32_t const N = p.N[ti];
            int32_t const n0 = (gt - p.tileOff[ti]) * 16;
            int32_t const rtT = (p.colOff[ti + 1] - p.colOff[ti]) / 16;
            half const* bt = bs + static_cast<size_t>(li) * p.rtMax * 256;
            half* out = p.out[ti];
            bool const colOk = n0 + lc < N;
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
            {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> d;
                wmma::fill_fragment(d, 0.f);
                for (int32_t j = 0; j < rtT; ++j)
                {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fa, &ts[rt * 16 * L.tsLd + p.colOff[ti] + j * 16], L.tsLd);
                    wmma::load_matrix_sync(fb, &bt[j * 256], 16);
                    wmma::mma_sync(d, fa, fb, d);
                }
                wmma::store_matrix_sync(wScr, d, 16, wmma::mem_row_major);
                __syncwarp();
                int32_t const m = rt * 16 + lr;
                if (colOk && m < p.M)
                {
                    half2 const* bh = reinterpret_cast<half2 const*>(&bv[rt]);
                    uint4 ov;
                    half2* oh = reinterpret_cast<half2*>(&ov);
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        float2 const b = __half22float2(bh[e]);
                        oh[e] = __floats2half2_rn(b.x + roundHalf(wScr[lr * 16 + lc + 2 * e]),
                            b.y + roundHalf(wScr[lr * 16 + lc + 2 * e + 1]));
                    }
                    *reinterpret_cast<uint4*>(&out[static_cast<size_t>(m) * N + n0 + lc]) = ov;
                }
                __syncwarp();
            }
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
            {
                bv[rt] = bvNext[rt];
            }
        }
    }
}

struct DeviceLimits
{
    int32_t grid{-1};    //!< cooperative grid: one CTA per SM, clamped to kMaxBarrierGrid
    size_t smemBytes{0}; //!< dynamic shared memory the kernel may use on this device
};

//! Device attributes only (no context state), cached per device ordinal. Failed queries are
//! not cached. The kernel's shared-memory opt-in is context state and is applied per launch.

//! Queried once per process for the current device. SM87 (Orin) opts in to 163 KB per block,
//! SM121 (GB10) to about 99 KB; layouts above the limit take the per-target fallback.
//! Cached per CUDA device (the shared-memory opt-in is per device as well).
DeviceLimits deviceLimits()
{
    static std::mutex sMutex;
    static std::unordered_map<int, DeviceLimits> sByDevice;
    int dev{0};
    if (cudaGetDevice(&dev) != cudaSuccess)
    {
        return DeviceLimits{};
    }
    std::lock_guard<std::mutex> lock(sMutex);
    auto const found = sByDevice.find(dev);
    if (found != sByDevice.end())
    {
        return found->second;
    }
    DeviceLimits sLimits;
    {
        int count{0};
        int optin{0};
        if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
            || cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return sLimits;
        }
        sLimits.grid = std::min<int32_t>(count, kMaxBarrierGrid);
        sLimits.smemBytes = std::min<size_t>(static_cast<size_t>(optin), kMaxDynSmem);
    }
    sByDevice[dev] = sLimits;
    return sLimits;
}

} // namespace

size_t loraGroupWorkspaceBytes(int64_t maxM, int32_t nbTargets, int32_t maxRank) noexcept
{
    size_t const part
        = std::max(static_cast<size_t>(kMaxGrid) * 16, static_cast<size_t>(kMaxBarrierGrid) * kLoraGroupMidMaxM)
        * kLoraGroupMaxCols * sizeof(float);
    int32_t const rp16 = (std::max(maxRank, 1) + 15) / 16 * 16;
    int32_t const cols = std::min(nbTargets * rp16, kPrefillMaxCols);
    return std::max({part, loraAddWorkspaceBytes(maxM, maxRank), loraPrefillWorkspaceBytes(maxM, cols)});
}

LoraGroupPlan planLoraGroup(
    int64_t M, int32_t K, LoraTarget const* targets, int32_t nbTargets, int32_t grid, size_t smemLimit) noexcept
{
    LoraGroupPlan plan{};
    // One launch for short prefills up to kLoraGroupMidMaxM rows on 16+ SMs, up to 32 rows on smaller
    // grids: with half the CTAs each one streams twice the base rows, and above 32 rows the
    // shrink + expand path measured faster for gate/up, down and o (8-CTA grid emulated on AGX Orin).
    int32_t const maxM = grid >= 16 ? kLoraGroupMidMaxM : 32;
    if (grid <= 0 || nbTargets <= 0 || nbTargets > kLoraGroupMaxTargets || M <= 0 || M > maxM || K % 8 != 0)
    {
        return plan;
    }
    int32_t cols = 0;
    int32_t tiles = 0;
    int32_t rtMax = 0;
    for (int32_t i = 0; i < nbTargets; ++i)
    {
        auto const& t = targets[i];
        if (t.r <= 0 || t.r % 8 != 0 || t.N % 8 != 0)
        {
            return plan;
        }
        int32_t const rt = (t.r + 15) / 16;
        cols += rt * 16;
        tiles += (t.N + 15) / 16;
        rtMax = std::max(rtMax, rt);
    }
    if (cols > kLoraGroupMaxCols)
    {
        return plan;
    }
    int32_t kPer = (K + grid - 1) / grid;
    kPer = (kPer + 15) / 16 * 16;
    int32_t const tilesPer = (tiles + grid - 1) / grid;
    if (M > 16)
    {
        // Short prefill: 2 or 4 row tiles; the deepest K ring (up to every chunk, at most 4) with
        // all tiles staged, else a 2-deep ring (1-deep as a last resort) with the largest round of
        // tiles that fits.
        int32_t const rowTiles = M <= 32 ? 2 : 4;
        int32_t const kc = midChunkRows(rowTiles);
        int32_t const nChunks = std::max(1, (kPer + kc - 1) / kc);
        int32_t stages = std::min(nChunks, 4);
        while (stages > 2 && midLayout(rowTiles, cols, stages, tilesPer, rtMax).total > smemLimit)
        {
            --stages;
        }
        auto const fitCap = [&](int32_t ring) {
            int32_t cap = tilesPer;
            while (cap > 0 && midLayout(rowTiles, cols, ring, cap, rtMax).total > smemLimit)
            {
                cap = cap > kWarps ? std::max(kWarps, cap - kWarps) : cap - 1;
            }
            return cap;
        };
        int32_t cap = fitCap(stages);
        if (cap <= 0 && stages > 1)
        {
            stages = 1;
            cap = fitCap(stages);
        }
        if (cap <= 0)
        {
            return plan;
        }
        plan.fused = true;
        plan.grid = grid;
        plan.tilesPerCta = tilesPer;
        plan.tilesPerRound = cap;
        plan.rowTiles = rowTiles;
        plan.stages = stages;
        plan.smemBytes = midLayout(rowTiles, cols, stages, cap, rtMax).total;
        return plan;
    }
    // Stage every tile at once when it fits; otherwise the largest round that does (at least one
    // tile per warp keeps every warp busy in each round).
    int32_t cap = tilesPer;
    while (cap > 0 && smemLayout(kPer, cols, cap, rtMax).total > smemLimit)
    {
        cap = cap > kWarps ? std::max(kWarps, cap - kWarps) : cap - 1;
    }
    if (cap <= 0)
    {
        return plan;
    }
    plan.fused = true;
    plan.grid = grid;
    plan.tilesPerCta = tilesPer;
    plan.tilesPerRound = cap;
    plan.rowTiles = 1;
    plan.smemBytes = smemLayout(kPer, cols, cap, rtMax).total;
    return plan;
}

int32_t launchLoraGroup(void const* x, int64_t M, int32_t K, LoraTarget const* targets, int32_t nbTargets,
    void* workspace, void* state, cudaStream_t stream) noexcept
{
    DeviceLimits const limits = deviceLimits();
    return launchLoraGroupWithLimits(
        x, M, K, targets, nbTargets, workspace, state, stream, limits.grid, limits.smemBytes);
}

int32_t launchLoraGroupWithLimits(void const* x, int64_t M, int32_t K, LoraTarget const* targets, int32_t nbTargets,
    void* workspace, void* state, cudaStream_t stream, int32_t grid, size_t smemLimit) noexcept
{
    if (nbTargets <= 0 || M <= 0)
    {
        return 0;
    }
    if (nbTargets > kLoraGroupMaxTargets || workspace == nullptr || state == nullptr)
    {
        return -1;
    }
    LoraGroupPlan const plan = planLoraGroup(M, K, targets, nbTargets, grid, std::min(smemLimit, kMaxDynSmem));
    bool fast = plan.fused;
    if (fast)
    {
        GroupParams p{};
        p.x = static_cast<half const*>(x);
        p.n = nbTargets;
        p.M = static_cast<int32_t>(M);
        p.K = K;
        int32_t cols = 0;
        int32_t tiles = 0;
        int32_t rtMax = 0;
        for (int32_t i = 0; i < nbTargets; ++i)
        {
            auto const& t = targets[i];
            p.A[i] = static_cast<half const*>(t.A);
            p.B[i] = static_cast<half const*>(t.B);
            p.base[i] = static_cast<half const*>(t.base);
            p.out[i] = static_cast<half*>(t.out);
            p.r[i] = t.r;
            p.N[i] = t.N;
            p.colOff[i] = cols;
            p.tileOff[i] = tiles;
            int32_t const rt = (t.r + 15) / 16;
            cols += rt * 16;
            tiles += (t.N + 15) / 16;
            rtMax = std::max(rtMax, rt);
        }
        p.colOff[nbTargets] = cols;
        p.tileOff[nbTargets] = tiles;
        int32_t kPer = (K + plan.grid - 1) / plan.grid;
        p.kPer = (kPer + 15) / 16 * 16;
        p.tilesPer = plan.tilesPerCta;
        p.tileCap = plan.tilesPerRound;
        p.rtMax = rtMax;
        p.part = static_cast<float*>(workspace);
        // Word 0 of the state: this kernel's barrier. The per-target small-M fallback below uses
        // its own word, because its grid size differs and a shared counter would desynchronize.
        p.counter = static_cast<unsigned long long*>(state);
        void const* kernel = reinterpret_cast<void const*>(loraGroupSmallKernel);
        int32_t stages = 0;
        if (plan.rowTiles > 1)
        {
            // Short prefill: its own barrier word (same grid as the small kernel, kept separate
            // so the two paths never depend on each other's launch count).
            kernel = plan.rowTiles == 2 ? reinterpret_cast<void const*>(loraGroupMidKernel<2>)
                                        : reinterpret_cast<void const*>(loraGroupMidKernel<4>);
            stages = plan.stages;
            p.counter = reinterpret_cast<unsigned long long*>(static_cast<char*>(state) + kLoraMidCounterOffset);
        }
        // The opt-in above 48 KB is per context: set it for the current one before launching.
        // It is a host-side function attribute, legal during stream capture.
        if (plan.smemBytes > 48 * 1024
            && cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(plan.smemBytes))
                != cudaSuccess)
        {
            (void) cudaGetLastError();
            fast = false;
        }
        if (fast)
        {
            cudaLaunchConfig_t cfg{};
            cfg.gridDim = dim3(plan.grid);
            cfg.blockDim = dim3(kThreads);
            cfg.dynamicSmemBytes = plan.smemBytes;
            cfg.stream = stream;
            cudaLaunchAttribute attr[1];
            attr[0].id = cudaLaunchAttributeCooperative;
            attr[0].val.cooperative = 1;
            cfg.attrs = attr;
            cfg.numAttrs = 1;
            if (plan.rowTiles == 2)
            {
                return cudaLaunchKernelEx(&cfg, loraGroupMidKernel<2>, p, stages) == cudaSuccess ? 0 : -1;
            }
            if (plan.rowTiles == 4)
            {
                return cudaLaunchKernelEx(&cfg, loraGroupMidKernel<4>, p, stages) == cudaSuccess ? 0 : -1;
            }
            return cudaLaunchKernelEx(&cfg, loraGroupSmallKernel, p) == cudaSuccess ? 0 : -1;
        }
    }
    if (M > 16)
    {
        // Prefill: one grouped shrink (x read once), one streaming expand per target.
        void const* As[kLoraGroupMaxTargets];
        void const* Bs[kLoraGroupMaxTargets];
        void const* bases[kLoraGroupMaxTargets];
        void* outs[kLoraGroupMaxTargets];
        int32_t rs[kLoraGroupMaxTargets];
        int32_t Ns[kLoraGroupMaxTargets];
        for (int32_t i = 0; i < nbTargets; ++i)
        {
            As[i] = targets[i].A;
            Bs[i] = targets[i].B;
            bases[i] = targets[i].base;
            outs[i] = targets[i].out;
            rs[i] = targets[i].r;
            Ns[i] = targets[i].N;
        }
        return launchLoraPrefill(x, M, K, nbTargets, As, Bs, bases, outs, rs, Ns, workspace, stream);
    }
    // Small-M fallback (unaligned shapes, oversize groups): one target at a time.
    for (int32_t i = 0; i < nbTargets; ++i)
    {
        auto const& t = targets[i];
        if (launchLoraAdd(t.base, x, t.A, t.B, t.out, M, K, t.N, t.r, workspace,
                static_cast<char*>(state) + kLoraFallbackCounterOffset, stream)
            != 0)
        {
            return -1;
        }
    }
    return 0;
}

} // namespace kernels
} // namespace trt_edgellm
