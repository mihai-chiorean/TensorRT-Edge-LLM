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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Arguments of the sequential GDN prefill. Layouts match the CuTe DSL gdn_prefill kernel:
//! q, k [n, seqLen, h, 128] FP16; v, o [n, seqLen, hv, 128] FP16; a, b [n, seqLen, hv] FP16;
//! logA [hv] FP32; dtBias [hv] FP16; h0Source [statePoolRows, hv, 128, 128] FP32 (K-major,
//! updated in place); stateIndices [n] INT32 or nullptr (row i_n); contextLengths [n] INT32.
struct GdnPrefillSequentialParams
{
    void const* q{nullptr};
    void const* k{nullptr};
    void const* v{nullptr};
    void const* a{nullptr};
    void const* b{nullptr};
    void const* logA{nullptr};
    void const* dtBias{nullptr};
    void* h0Source{nullptr};
    void const* stateIndices{nullptr};
    void const* contextLengths{nullptr};
    void* o{nullptr};
    int32_t n{0};
    int32_t seqLen{0};
    int32_t h{0};
    int32_t hv{0};
    int32_t kDim{0};
    int32_t vDim{0};
};

//! True when launchGdnPrefillSequential can run this head geometry.
bool gdnPrefillSequentialSupported(int32_t h, int32_t hv, int32_t kDim, int32_t vDim);

//! Gated delta rule over seqLen tokens with the QK L2 norm, scale 1/sqrt(kDim) and softplus
//! gate of the CuTe DSL gdn_prefill kernel. Each V column of the recurrent state evolves
//! independently, so every (sequence, value head, V-column group) gets its own CTA and keeps
//! its state slice in registers. Per element it performs the same FP32 operations in the same
//! order as gdn_prefill, so outputs and final state match it bit for bit.
cudaError_t launchGdnPrefillSequential(GdnPrefillSequentialParams const& params, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
