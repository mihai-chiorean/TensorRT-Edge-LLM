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

#include "gdnPrefillSequential.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Dynamic shared memory of one launchGdnPrefillChunked CTA.
size_t gdnPrefillChunkedSharedBytes();

//! True when launchGdnPrefillChunked can run this geometry on the current device (SM80+ and enough
//! opt-in shared memory per block).
bool gdnPrefillChunkedSupported(int32_t h, int32_t hv, int32_t kDim, int32_t vDim, int32_t smVersion);

//! Same contract as launchGdnPrefillSequential, computed in the chunked (WY) form with FP16 tensor-core
//! MMAs and FP32 state. Not bit-identical to gdn_prefill.
cudaError_t launchGdnPrefillChunked(GdnPrefillSequentialParams const& params, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
