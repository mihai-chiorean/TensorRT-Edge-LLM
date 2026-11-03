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

#include "gdnPrefillChunked.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

// Chunked (WY) form of the gated delta rule, one CTA per (sequence, value head). For a chunk of
// kChunk tokens with in-chunk cumulative log decay c_i (gamma_i = exp(c_i)) and state S0 [dk, dv]:
//   L_ij = beta_i exp(c_i - c_j) k_i.k_j (j < i)
//   U = (I + L)^-1 [beta (V - gamma K S0)]                 (FP32 forward substitution)
//   O = gamma * (Q S0) + P U,  P_ij = exp(c_i - c_j) q_i.k_j (j <= i)
//   S1 = gamma_last S0 + (exp(c_last - c) K)^T U
// The state stays in FP32 accumulator registers; products with S0 use FP16 tensor-core MMAs on a
// two-term split S0 = hi + lo, so the state operand keeps about 22 mantissa bits.
constexpr int32_t kHeadDim{128};
constexpr int32_t kChunk{32};
constexpr int32_t kWarps{8};
constexpr int32_t kThreads{kWarps * 32};
constexpr int32_t kHalfStride{kHeadDim + 8};
constexpr int32_t kPStride{kChunk + 8};
constexpr int32_t kWvStride{kHeadDim + 4};
constexpr int32_t kNTiles{kHeadDim / 8};
constexpr uint32_t kFullMask{0xffffffffU};
constexpr int32_t kMaxDevices{64};
constexpr int32_t kMaxStateShift{100};
constexpr int32_t kMaxUShift{15};

struct SharedLayout
{
    __half kHat[kChunk][kHalfStride];
    __half qHat[kChunk][kHalfStride];
    __half kDecayed[kChunk][kHalfStride]; // Decayed keys exp(c_last - c_j) k_j for the state update.
    __half vOrU[kChunk][kHalfStride];     // v, then U after the solve.
    __half sHi[kHeadDim][kHalfStride];
    __half sLo[kHeadDim][kHalfStride];
    __half rawK[kChunk][kHeadDim]; // Next chunk's inputs, staged by cp.async.
    __half rawQ[kChunk][kHeadDim];
    __half rawV[kChunk][kHeadDim];
    float wv[kChunk][kWvStride]; // R = beta (V - gamma K S0), the solve's right-hand side.
    float lt[kChunk][kChunk];    // L transposed: lt[j][i] = L_ij.
    __half p[kChunk][kPStride];
    float cumLog[kChunk];
    float beta[kChunk];
    float gamma[kChunk];
    float keyDecay[kChunk]; // exp(c_last - c_j)
    uint32_t stateMaxBits;  // max |S| and max |U| as FP32 bits, for the FP16 range shifts
    uint32_t uMaxBits;
};

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsmX4(uint32_t (&r)[4], void const* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smemAddr(p)));
}

__device__ __forceinline__ void ldsmX4Trans(uint32_t (&r)[4], void const* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smemAddr(p)));
}

__device__ __forceinline__ void ldsmX2(uint32_t (&r)[2], void const* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(smemAddr(p)));
}

__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A operand (16 x 16) of a row-major [m][k] tile at (m0, k0).
template <int32_t kStride>
__device__ __forceinline__ void loadA(
    uint32_t (&r)[4], __half const (*tile)[kStride], int32_t m0, int32_t k0, int32_t lane)
{
    ldsmX4(r, &tile[m0 + (lane & 15)][k0 + (lane >> 4) * 8]);
}

// A operand (16 x 16) whose storage is [k][m] (the transpose), at (m0, k0).
template <int32_t kStride>
__device__ __forceinline__ void loadATrans(
    uint32_t (&r)[4], __half const (*tile)[kStride], int32_t m0, int32_t k0, int32_t lane)
{
    ldsmX4Trans(r, &tile[k0 + (lane & 7) + (lane >> 4) * 8][m0 + ((lane >> 3) & 1) * 8]);
}

// B operands of two adjacent n8 tiles (16 x 16) stored [k][n] row-major, at (k0, n0): r[0..1] is tile n0,
// r[2..3] tile n0 + 8.
template <int32_t kStride>
__device__ __forceinline__ void loadBPair(
    uint32_t (&r)[4], __half const (*tile)[kStride], int32_t k0, int32_t n0, int32_t lane)
{
    ldsmX4Trans(r, &tile[k0 + (lane & 15)][n0 + (lane >> 4) * 8]);
}

// B operand of one n8 tile stored [n][k] (the transpose), at (k0, n0).
template <int32_t kStride>
__device__ __forceinline__ void loadBTrans(
    uint32_t (&r)[2], __half const (*tile)[kStride], int32_t k0, int32_t n0, int32_t lane)
{
    ldsmX2(r, &tile[n0 + (lane & 7)][k0 + ((lane >> 3) & 1) * 8]);
}

__device__ __forceinline__ float warpMax(float x)
{
#pragma unroll
    for (int32_t offset = 16; offset >= 1; offset /= 2)
    {
        x = fmaxf(x, __shfl_xor_sync(kFullMask, x, offset));
    }
    return x;
}

// Power-of-two shift that brings a block's largest magnitude below 2^15 for an FP16 MMA operand.
// It is zero in the usual range and exact otherwise: the state's factor is restored on the FP32
// accumulators, U's is folded into P and the decayed keys (both bounded by 1), which caps it at
// maxShift = 15, i.e. |U| up to about 2^29.
__device__ __forceinline__ int32_t fp16RangeShift(uint32_t maxBits, int32_t maxShift)
{
    float const maxAbs = __uint_as_float(maxBits);
    return maxAbs > 16384.0F ? min(ilogbf(maxAbs) - 14 + 1, maxShift) : 0;
}

__device__ __forceinline__ float warpSum(float x)
{
#pragma unroll
    for (int32_t offset = 16; offset >= 1; offset /= 2)
    {
        x += __shfl_xor_sync(kFullMask, x, offset);
    }
    return x;
}

__device__ __forceinline__ void copyAsync16(void* dst, void const* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smemAddr(dst)), "l"(src));
}

__device__ __forceinline__ void stageChunk(SharedLayout& sm, __half const* q, __half const* k, __half const* v,
    int64_t firstToken, int32_t len, int32_t h, int32_t iH, int32_t hv, int32_t iHv)
{
    constexpr int32_t kPieces{kHeadDim / 8};
    for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < len * kPieces; idx += kThreads)
    {
        int32_t const j = idx / kPieces;
        int32_t const col = (idx % kPieces) * 8;
        int64_t const token = firstToken + j;
        copyAsync16(&sm.rawK[j][col], k + (token * h + iH) * kHeadDim + col);
        copyAsync16(&sm.rawQ[j][col], q + (token * h + iH) * kHeadDim + col);
        copyAsync16(&sm.rawV[j][col], v + (token * hv + iHv) * kHeadDim + col);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

__global__ void __launch_bounds__(kThreads, 1) gdnPrefillChunkedKernel(__half const* __restrict__ q,
    __half const* __restrict__ k, __half const* __restrict__ v, __half const* __restrict__ a,
    __half const* __restrict__ b, float const* __restrict__ logA, __half const* __restrict__ dtBias,
    float* __restrict__ h0Source, int32_t const* __restrict__ stateIndices, int32_t const* __restrict__ contextLengths,
    __half* __restrict__ o, int32_t seqLen, int32_t h, int32_t hv, float scale)
{
    extern __shared__ __align__(16) unsigned char smemRaw[];
    SharedLayout& sm = *reinterpret_cast<SharedLayout*>(smemRaw);

    int32_t const iN = static_cast<int32_t>(blockIdx.x) / hv;
    int32_t const iHv = static_cast<int32_t>(blockIdx.x) % hv;
    int32_t const iH = iHv / (hv / h);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const lane = tid & 31;
    int32_t const warp = tid >> 5;
    int32_t const group = lane >> 2;
    int32_t const quad = lane & 3;
    int32_t const slot = stateIndices != nullptr ? stateIndices[iN] : iN;
    int32_t const validLen = max(0, min(seqLen, contextLengths[iN]));
    int64_t const tokenBase = static_cast<int64_t>(iN) * seqLen;
    float* const state = h0Source + (static_cast<int64_t>(slot) * hv + iHv) * kHeadDim * kHeadDim;

    // Warp w owns state rows [16w, 16w + 16) and all 128 columns: 16 m16n8 accumulator tiles.
    float s[kNTiles][4];
    int32_t const sRow0 = warp * 16 + group;
#pragma unroll
    for (int32_t nt = 0; nt < kNTiles; ++nt)
    {
        int32_t const col = nt * 8 + quad * 2;
        float2 const top = *reinterpret_cast<float2 const*>(&state[sRow0 * kHeadDim + col]);
        float2 const bottom = *reinterpret_cast<float2 const*>(&state[(sRow0 + 8) * kHeadDim + col]);
        s[nt][0] = top.x;
        s[nt][1] = top.y;
        s[nt][2] = bottom.x;
        s[nt][3] = bottom.y;
    }

    // Writes S * 2^-stateShift as hi + lo FP16 B operands; the block reduction of max |S| needs the
    // caller to have cleared sm.stateMaxBits behind a barrier.
    int32_t stateShift = 0;
    auto storeStateSplit = [&]() {
        float localMax = 0.0F;
#pragma unroll
        for (int32_t nt = 0; nt < kNTiles; ++nt)
        {
#pragma unroll
            for (int32_t e = 0; e < 4; ++e)
            {
                localMax = fmaxf(localMax, fabsf(s[nt][e]));
            }
        }
        localMax = warpMax(localMax);
        if (lane == 0)
        {
            atomicMax(&sm.stateMaxBits, __float_as_uint(localMax));
        }
        __syncthreads();
        stateShift = fp16RangeShift(sm.stateMaxBits, kMaxStateShift);
#pragma unroll
        for (int32_t nt = 0; nt < kNTiles; ++nt)
        {
            int32_t const col = nt * 8 + quad * 2;
#pragma unroll
            for (int32_t hr = 0; hr < 2; ++hr)
            {
                float const x0 = ldexpf(s[nt][2 * hr], -stateShift);
                float const x1 = ldexpf(s[nt][2 * hr + 1], -stateShift);
                __half2 const hi = __floats2half2_rn(x0, x1);
                float2 const hiF = __half22float2(hi);
                __half2 const lo = __floats2half2_rn(x0 - hiF.x, x1 - hiF.y);
                *reinterpret_cast<__half2*>(&sm.sHi[sRow0 + 8 * hr][col]) = hi;
                *reinterpret_cast<__half2*>(&sm.sLo[sRow0 + 8 * hr][col]) = lo;
            }
        }
    };
    if (tid == 0)
    {
        sm.stateMaxBits = 0U;
    }
    __syncthreads();
    storeStateSplit();

    float const negA = -expf(logA[iHv]);
    float const dtBiasValue = __half2float(dtBias[iHv]);
    float nextA = 0.0F;
    float nextB = 0.0F;
    if (validLen > 0)
    {
        stageChunk(sm, q, k, v, tokenBase, min(kChunk, validLen), h, iH, hv, iHv);
        if (warp == 0 && lane < min(kChunk, validLen))
        {
            nextA = __half2float(a[(tokenBase + lane) * hv + iHv]);
            nextB = __half2float(b[(tokenBase + lane) * hv + iHv]);
        }
    }

    for (int32_t t0 = 0; t0 < validLen; t0 += kChunk)
    {
        int32_t const len = min(kChunk, validLen - t0);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        if (tid == 0)
        {
            sm.stateMaxBits = 0U;
            sm.uMaxBits = 0U;
        }

        // Gates and the in-chunk cumulative log decay (padded tokens: g = 1, beta = 0).
        if (warp == 0)
        {
            float logG = 0.0F;
            float betaValue = 0.0F;
            if (lane < len)
            {
                float const x = nextA + dtBiasValue;
                float const softplus = x <= 20.0F ? log1pf(expf(x)) : x;
                logG = negA * softplus;
                betaValue = 1.0F / (1.0F + expf(-nextB));
            }
            float cum = logG;
#pragma unroll
            for (int32_t offset = 1; offset < 32; offset *= 2)
            {
                float const other = __shfl_up_sync(kFullMask, cum, offset);
                cum += lane >= offset ? other : 0.0F;
            }
            float const cLast = __shfl_sync(kFullMask, cum, kChunk - 1);
            sm.cumLog[lane] = cum;
            sm.beta[lane] = betaValue;
            sm.gamma[lane] = expf(cum);
            sm.keyDecay[lane] = expf(cLast - cum);
        }

        // Normalized q/k (FP16 MMA operands) and v; padded tokens are zero.
#pragma unroll
        for (int32_t u = 0; u < kChunk / kWarps; ++u)
        {
            int32_t const j = warp * (kChunk / kWarps) + u;
            float kx[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            float qx[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            float vx[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            if (j < len)
            {
#pragma unroll
                for (int32_t w = 0; w < 4; ++w)
                {
                    kx[w] = __half2float(sm.rawK[j][w * 32 + lane]);
                    qx[w] = __half2float(sm.rawQ[j][w * 32 + lane]);
                    vx[w] = __half2float(sm.rawV[j][w * 32 + lane]);
                }
            }
            float const invK = rsqrtf(warpSum(kx[0] * kx[0] + kx[1] * kx[1] + kx[2] * kx[2] + kx[3] * kx[3]) + 1e-6F);
            float const invQ = rsqrtf(warpSum(qx[0] * qx[0] + qx[1] * qx[1] + qx[2] * qx[2] + qx[3] * qx[3]) + 1e-6F);
#pragma unroll
            for (int32_t w = 0; w < 4; ++w)
            {
                int32_t const col = w * 32 + lane;
                sm.kHat[j][col] = __float2half_rn(kx[w] * invK);
                sm.qHat[j][col] = __float2half_rn(qx[w] * scale * invQ);
                sm.vOrU[j][col] = __float2half_rn(vx[w]);
            }
        }
        __syncthreads();

        int32_t const nextT0 = t0 + kChunk;
        if (nextT0 < validLen)
        {
            int32_t const nextLen = min(kChunk, validLen - nextT0);
            stageChunk(sm, q, k, v, tokenBase + nextT0, nextLen, h, iH, hv, iHv);
            if (warp == 0 && lane < nextLen)
            {
                nextA = __half2float(a[(tokenBase + nextT0 + lane) * hv + iHv]);
                nextB = __half2float(b[(tokenBase + nextT0 + lane) * hv + iHv]);
            }
        }

        // K K^T -> L, Q K^T -> P (warp w: the 16 x 8 tile at rows 16 (w / 4), cols 8 (w % 4)), and
        // Y = K S0, Z = Q S0 (warp w: rows 16 (w / 4), columns 32 (w % 4) .. + 32).
        int32_t const m0 = (warp >> 2) * 16;
        int32_t const c0 = (warp & 3) * 32;
        float accZ[4][4];
        {
            int32_t const n0 = (warp & 3) * 8;
            float kk[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            float qk[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            float accY[4][4];
#pragma unroll
            for (int32_t nt = 0; nt < 4; ++nt)
            {
#pragma unroll
                for (int32_t e = 0; e < 4; ++e)
                {
                    accY[nt][e] = 0.0F;
                    accZ[nt][e] = 0.0F;
                }
            }
#pragma unroll
            for (int32_t k0 = 0; k0 < kHeadDim; k0 += 16)
            {
                uint32_t aK[4];
                uint32_t aQ[4];
                loadA<kHalfStride>(aK, sm.kHat, m0, k0, lane);
                loadA<kHalfStride>(aQ, sm.qHat, m0, k0, lane);
                uint32_t bK[2];
                loadBTrans<kHalfStride>(bK, sm.kHat, k0, n0, lane);
                mma16816(kk, aK, bK[0], bK[1]);
                mma16816(qk, aQ, bK[0], bK[1]);
#pragma unroll
                for (int32_t pair = 0; pair < 2; ++pair)
                {
                    uint32_t bHi[4];
                    uint32_t bLo[4];
                    loadBPair<kHalfStride>(bHi, sm.sHi, k0, c0 + pair * 16, lane);
                    loadBPair<kHalfStride>(bLo, sm.sLo, k0, c0 + pair * 16, lane);
#pragma unroll
                    for (int32_t sub = 0; sub < 2; ++sub)
                    {
                        int32_t const nt = pair * 2 + sub;
                        mma16816(accY[nt], aK, bHi[2 * sub], bHi[2 * sub + 1]);
                        mma16816(accY[nt], aK, bLo[2 * sub], bLo[2 * sub + 1]);
                        mma16816(accZ[nt], aQ, bHi[2 * sub], bHi[2 * sub + 1]);
                        mma16816(accZ[nt], aQ, bLo[2 * sub], bLo[2 * sub + 1]);
                    }
                }
            }
            if (stateShift > 0)
            {
                float const unshift = ldexpf(1.0F, stateShift);
#pragma unroll
                for (int32_t nt = 0; nt < 4; ++nt)
                {
#pragma unroll
                    for (int32_t e = 0; e < 4; ++e)
                    {
                        accY[nt][e] *= unshift;
                        accZ[nt][e] *= unshift;
                    }
                }
            }
#pragma unroll
            for (int32_t e = 0; e < 4; ++e)
            {
                int32_t const i = m0 + group + (e >> 1) * 8;
                int32_t const j = n0 + quad * 2 + (e & 1);
                float const decay = expf(sm.cumLog[i] - sm.cumLog[j]);
                sm.lt[j][i] = j < i ? sm.beta[i] * decay * kk[e] : 0.0F;
                sm.p[i][j] = __float2half_rn(j <= i ? decay * qk[e] : 0.0F);
            }
            // Right-hand side R = beta (V - gamma K S0) of (I + L) U = R.
#pragma unroll
            for (int32_t nt = 0; nt < 4; ++nt)
            {
                int32_t const col = c0 + nt * 8 + quad * 2;
#pragma unroll
                for (int32_t hr = 0; hr < 2; ++hr)
                {
                    int32_t const row = m0 + group + hr * 8;
                    float const betaRow = sm.beta[row];
                    float const gammaRow = sm.gamma[row];
                    float2 const vv = __half22float2(*reinterpret_cast<__half2 const*>(&sm.vOrU[row][col]));
                    *reinterpret_cast<float2*>(&sm.wv[row][col])
                        = make_float2(betaRow * (vv.x - gammaRow * accY[nt][2 * hr]),
                            betaRow * (vv.y - gammaRow * accY[nt][2 * hr + 1]));
                }
            }
        }
        __syncthreads();

        float const cLast = sm.cumLog[kChunk - 1];
        float x[kChunk];
        if (tid < kHeadDim)
        {
            // U = (I + L)^-1 R by right-looking forward substitution, one column per thread: once u_j is
            // final it is folded into every later row, so each step waits on a single FMA.
            int32_t const col = tid;
#pragma unroll
            for (int32_t i = 0; i < kChunk; ++i)
            {
                x[i] = sm.wv[i][col];
            }
#pragma unroll
            for (int32_t j = 0; j < kChunk - 1; ++j)
            {
#pragma unroll
                for (int32_t i4 = (j + 1) / 4; i4 < kChunk / 4; ++i4)
                {
                    float4 const l4 = reinterpret_cast<float4 const*>(sm.lt[j])[i4];
                    float const lv[4] = {l4.x, l4.y, l4.z, l4.w};
#pragma unroll
                    for (int32_t e = 0; e < 4; ++e)
                    {
                        if (i4 * 4 + e > j)
                        {
                            x[i4 * 4 + e] = fmaf(-lv[e], x[j], x[i4 * 4 + e]);
                        }
                    }
                }
            }
            float localMax = 0.0F;
#pragma unroll
            for (int32_t i = 0; i < kChunk; ++i)
            {
                localMax = fmaxf(localMax, fabsf(x[i]));
            }
            localMax = warpMax(localMax);
            if (lane == 0)
            {
                atomicMax(&sm.uMaxBits, __float_as_uint(localMax));
            }
        }
        __syncthreads();
        int32_t const uShift = fp16RangeShift(sm.uMaxBits, kMaxUShift);
        if (tid < kHeadDim)
        {
#pragma unroll
            for (int32_t i = 0; i < kChunk; ++i)
            {
                sm.vOrU[i][tid] = __float2half_rn(ldexpf(x[i], -uShift));
            }
        }
        else
        {
            // Decayed keys exp(c_last - c_j) k_j for the state update, carrying the U shift.
            for (int32_t idx = tid - kHeadDim; idx < kChunk * kHeadDim / 2; idx += kThreads - kHeadDim)
            {
                int32_t const j = idx / (kHeadDim / 2);
                int32_t const col = (idx % (kHeadDim / 2)) * 2;
                float2 const kv = __half22float2(*reinterpret_cast<__half2 const*>(&sm.kHat[j][col]));
                float const decay = ldexpf(sm.keyDecay[j], uShift);
                *reinterpret_cast<__half2*>(&sm.kDecayed[j][col]) = __floats2half2_rn(kv.x * decay, kv.y * decay);
            }
        }
        __syncthreads();

        // O = gamma * Z + P U.
        {
            float const gTop = sm.gamma[m0 + group];
            float const gBottom = sm.gamma[m0 + group + 8];
#pragma unroll
            for (int32_t nt = 0; nt < 4; ++nt)
            {
                accZ[nt][0] *= gTop;
                accZ[nt][1] *= gTop;
                accZ[nt][2] *= gBottom;
                accZ[nt][3] *= gBottom;
            }
#pragma unroll
            for (int32_t k0 = 0; k0 < kChunk; k0 += 16)
            {
                uint32_t aP[4];
                loadA<kPStride>(aP, sm.p, m0, k0, lane);
                if (uShift > 0)
                {
                    __half2 const shift = __float2half2_rn(ldexpf(1.0F, uShift));
#pragma unroll
                    for (int32_t r = 0; r < 4; ++r)
                    {
                        __half2 scaled = __hmul2(*reinterpret_cast<__half2*>(&aP[r]), shift);
                        aP[r] = *reinterpret_cast<uint32_t*>(&scaled);
                    }
                }
#pragma unroll
                for (int32_t pair = 0; pair < 2; ++pair)
                {
                    uint32_t uFrag[4];
                    loadBPair<kHalfStride>(uFrag, sm.vOrU, k0, c0 + pair * 16, lane);
                    mma16816(accZ[pair * 2], aP, uFrag[0], uFrag[1]);
                    mma16816(accZ[pair * 2 + 1], aP, uFrag[2], uFrag[3]);
                }
            }
#pragma unroll
            for (int32_t nt = 0; nt < 4; ++nt)
            {
                int32_t const col = c0 + nt * 8 + quad * 2;
#pragma unroll
                for (int32_t hr = 0; hr < 2; ++hr)
                {
                    int32_t const row = m0 + group + hr * 8;
                    if (row < len)
                    {
                        *reinterpret_cast<__half2*>(&o[((tokenBase + t0 + row) * hv + iHv) * kHeadDim + col])
                            = __floats2half2_rn(accZ[nt][2 * hr], accZ[nt][2 * hr + 1]);
                    }
                }
            }
        }

        // S1 = gamma_last S0 + (exp(c_last - c) K)^T U.
        {
            float const gammaLast = expf(cLast);
#pragma unroll
            for (int32_t nt = 0; nt < kNTiles; ++nt)
            {
#pragma unroll
                for (int32_t e = 0; e < 4; ++e)
                {
                    s[nt][e] *= gammaLast;
                }
            }
#pragma unroll
            for (int32_t k0 = 0; k0 < kChunk; k0 += 16)
            {
                uint32_t aK[4];
                loadATrans<kHalfStride>(aK, sm.kDecayed, warp * 16, k0, lane);
#pragma unroll
                for (int32_t pair = 0; pair < kNTiles / 2; ++pair)
                {
                    uint32_t uFrag[4];
                    loadBPair<kHalfStride>(uFrag, sm.vOrU, k0, pair * 16, lane);
                    mma16816(s[pair * 2], aK, uFrag[0], uFrag[1]);
                    mma16816(s[pair * 2 + 1], aK, uFrag[2], uFrag[3]);
                }
            }
        }
        storeStateSplit();
    }

    for (int32_t idx = tid; idx < (seqLen - validLen) * kHeadDim; idx += kThreads)
    {
        int32_t const t = validLen + idx / kHeadDim;
        int32_t const col = idx % kHeadDim;
        o[((tokenBase + t) * hv + iHv) * kHeadDim + col] = __float2half_rn(0.0F);
    }
#pragma unroll
    for (int32_t nt = 0; nt < kNTiles; ++nt)
    {
        int32_t const col = nt * 8 + quad * 2;
        *reinterpret_cast<float2*>(&state[sRow0 * kHeadDim + col]) = make_float2(s[nt][0], s[nt][1]);
        *reinterpret_cast<float2*>(&state[(sRow0 + 8) * kHeadDim + col]) = make_float2(s[nt][2], s[nt][3]);
    }
}

} // namespace

size_t gdnPrefillChunkedSharedBytes()
{
    return sizeof(SharedLayout);
}

bool gdnPrefillChunkedSupported(int32_t h, int32_t hv, int32_t kDim, int32_t vDim, int32_t smVersion)
{
    if (kDim != kHeadDim || vDim != kHeadDim || h <= 0 || hv <= 0 || hv % h != 0 || smVersion < 80)
    {
        return false;
    }
    int32_t device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices)
    {
        return false;
    }
    // Per device: 0 = not queried yet, 1 = fits, -1 = does not fit.
    static std::array<std::atomic<int8_t>, kMaxDevices> fits{};
    int8_t state = fits[device].load(std::memory_order_relaxed);
    if (state == 0)
    {
        int32_t optinBytes = 0;
        bool const ok
            = cudaDeviceGetAttribute(&optinBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess
            && static_cast<size_t>(optinBytes) >= sizeof(SharedLayout);
        state = ok ? 1 : -1;
        fits[device].store(state, std::memory_order_relaxed);
    }
    return state > 0;
}

cudaError_t launchGdnPrefillChunked(GdnPrefillSequentialParams const& p, cudaStream_t stream)
{
    if (p.n <= 0 || p.seqLen <= 0 || p.kDim != kHeadDim || p.vDim != kHeadDim || p.h <= 0 || p.hv % p.h != 0)
    {
        return cudaErrorInvalidValue;
    }
    int32_t device = 0;
    cudaError_t const deviceStatus = cudaGetDevice(&device);
    if (deviceStatus != cudaSuccess)
    {
        return deviceStatus;
    }
    if (device < 0 || device >= kMaxDevices)
    {
        return cudaErrorInvalidDevice;
    }
    // The dynamic shared-memory opt-in is a per-device function attribute.
    static std::array<std::atomic<bool>, kMaxDevices> optedIn{};
    if (!optedIn[device].load(std::memory_order_acquire))
    {
        cudaError_t const attrStatus = cudaFuncSetAttribute(gdnPrefillChunkedKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int32_t>(sizeof(SharedLayout)));
        if (attrStatus != cudaSuccess)
        {
            return attrStatus;
        }
        optedIn[device].store(true, std::memory_order_release);
    }
    float const scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.kDim)));
    gdnPrefillChunkedKernel<<<static_cast<uint32_t>(p.n * p.hv), kThreads, sizeof(SharedLayout), stream>>>(
        static_cast<__half const*>(p.q), static_cast<__half const*>(p.k), static_cast<__half const*>(p.v),
        static_cast<__half const*>(p.a), static_cast<__half const*>(p.b), static_cast<float const*>(p.logA),
        static_cast<__half const*>(p.dtBias), static_cast<float*>(p.h0Source),
        static_cast<int32_t const*>(p.stateIndices), static_cast<int32_t const*>(p.contextLengths),
        static_cast<__half*>(p.o), p.seqLen, p.h, p.hv, scale);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
