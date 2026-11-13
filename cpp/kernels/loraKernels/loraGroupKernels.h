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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernels
{

//! Sibling projections that share one input (q/k/v, gate/up) handled by one launch.
constexpr int32_t kLoraGroupMaxTargets{3};
//! Sum over active targets of the rank rounded up to 16.
constexpr int32_t kLoraGroupMaxCols{128};

//! One active LoRA target: out = base + fp16(fp16(x @ A) @ B). out may equal base (in place).
struct LoraTarget
{
    void const* A{nullptr};   //!< [K, r] fp16
    void const* B{nullptr};   //!< [r, N] fp16
    void const* base{nullptr}; //!< [M, N] fp16
    void* out{nullptr};       //!< [M, N] fp16
    int32_t r{0};
    int32_t N{0};
};

//! Workspace bytes for launchLoraGroup with up to maxM rows and per-target ranks up to maxRank.
size_t loraGroupWorkspaceBytes(int64_t maxM, int32_t nbTargets, int32_t maxRank) noexcept;

//! Persistent per-plugin device state, zeroed once at allocation: grid-barrier counters, one
//! per launcher path (each path always launches the same grid size on a given device).
constexpr size_t kLoraGroupStateBytes{256};
constexpr size_t kLoraFallbackCounterOffset{128};

/*!
 * Applies the LoRA residual of every target in `targets` (all ranks > 0, shared x [M, K]).
 *
 * M <= 16 with K, N and r multiples of 8: one cooperative launch. Every load the CTA needs
 * (its K slice of x and A, its B and base output tiles) is issued as cp.async at entry, so
 * DRAM latency overlaps the grid barrier; shrink and expand use m16n16k16 tensor-core MMA.
 * Otherwise: per target, a split-K WMMA shrink and a WMMA expand (prefill).
 *
 * Rounding follows the unfused ONNX branch: the rank-r intermediate and the delta are each
 * rounded to fp16 and the final add is done in fp32 and rounded once.
 */
int32_t launchLoraGroup(void const* x, int64_t M, int32_t K, LoraTarget const* targets, int32_t nbTargets,
    void* workspace, void* state, cudaStream_t stream) noexcept;

} // namespace kernels
} // namespace trt_edgellm
