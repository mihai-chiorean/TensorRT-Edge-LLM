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
    size_t const part = static_cast<size_t>(kMaxGrid) * 16 * kLoraGroupMaxCols * sizeof(float);
    int32_t const rp16 = (std::max(maxRank, 1) + 15) / 16 * 16;
    int32_t const cols = std::min(nbTargets * rp16, kPrefillMaxCols);
    return std::max({part, loraAddWorkspaceBytes(maxM, maxRank), loraPrefillWorkspaceBytes(maxM, cols)});
}

LoraGroupPlan planLoraGroup(
    int64_t M, int32_t K, LoraTarget const* targets, int32_t nbTargets, int32_t grid, size_t smemLimit) noexcept
{
    LoraGroupPlan plan{};
    if (grid <= 0 || nbTargets <= 0 || nbTargets > kLoraGroupMaxTargets || M <= 0 || M > 16 || K % 8 != 0)
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
        // The opt-in above 48 KB is per context: set it for the current one before launching.
        // It is a host-side function attribute, legal during stream capture.
        if (plan.smemBytes > 48 * 1024
            && cudaFuncSetAttribute(
                   loraGroupSmallKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(plan.smemBytes))
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
