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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernels
{

//! Rows (flattened batch * sequence) served by the fused single-launch path.
constexpr int32_t kLoraSmallMaxM{16};
//! Largest LoRA rank the kernels accept.
constexpr int32_t kLoraMaxRank{64};

//! Up to this many targets and padded rank columns share one prefill shrink.
constexpr int32_t kPrefillMaxTargets{3};
constexpr int32_t kPrefillMaxCols{128};

//! Targets of one grouped prefill shrink (shared x): A_i [K, r_i], columns colOff[i]..colOff[i+1].
struct PrefillTargets
{
    __half const* A[kPrefillMaxTargets]{};
    int32_t r[kPrefillMaxTargets]{};
    int32_t colOff[kPrefillMaxTargets + 1]{};
    int32_t n{0};
};

/*!
 * Prefill (any M) residual for n <= kPrefillMaxTargets targets sharing x: one split-K WMMA shrink
 * over [A_0 | A_1 | ...] that reads x once, then one streaming expand + add per target.
 * out_i may equal base_i. Workspace: loraPrefillWorkspaceBytes(maxM, n * roundup16(maxRank)).
 */
int32_t launchLoraPrefill(void const* x, int64_t M, int32_t K, int32_t n, void const* const* A, void const* const* B,
    void const* const* base, void* const* out, int32_t const* r, int32_t const* N, void* workspace,
    cudaStream_t stream) noexcept;

//! Workspace bytes for launchLoraPrefill with up to maxM rows and `cols` padded rank columns.
size_t loraPrefillWorkspaceBytes(int64_t maxM, int32_t cols) noexcept;

//! Workspace bytes for one launch of launchLoraAdd with up to maxM rows and rank maxRank.
size_t loraAddWorkspaceBytes(int64_t maxM, int32_t maxRank) noexcept;

//! Bytes of persistent device state (grid-barrier counter) a launcher path owns.
constexpr size_t kLoraBarrierStateBytes{256};

/*!
 * out[M,N] = base[M,N] + (x[M,K] @ A[K,r]) @ B[r,N], fp16 in and out.
 *
 * Rounding follows the unfused ONNX branch it replaces: the rank-r intermediate and the
 * delta are each rounded to fp16 (fp32 accumulation), and the final add is done in fp32
 * and rounded once.
 *
 * M <= kLoraSmallMaxM: one cooperative launch (split-K shrink, grid barrier, expand + add).
 * barrierState must point to kLoraBarrierStateBytes of device memory zeroed once at
 * allocation and used by no other launch configuration.
 * M > kLoraSmallMaxM: two tensor-core (WMMA) launches through the workspace.
 *
 * Returns 0 on success.
 */
int32_t launchLoraAdd(void const* base, void const* x, void const* loraA, void const* loraB, void* out, int64_t M,
    int32_t K, int32_t N, int32_t rank, void* workspace, void* barrierState, cudaStream_t stream) noexcept;

} // namespace kernels
} // namespace trt_edgellm
