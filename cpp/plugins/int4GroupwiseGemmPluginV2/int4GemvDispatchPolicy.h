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

namespace trt_edgellm::plugins
{

//! Largest M that Int4GroupwiseGemmPluginV2 dispatches to the CUDA-core GEMV when
//! the multi-row rule is off.
inline constexpr int32_t kInt4GemvDispatchMaxM{4};

//! Largest M the multi-row rule can send to the GEMV (the compiled GEMV variants).
inline constexpr int32_t kInt4GemvNarrowDispatchMaxM{8};

//! With the multi-row rule, M in [2, kInt4GemvNarrowDispatchMaxM] takes the GEMV
//! only when N is at most this. On SM87 Gemma 4 decoder shapes the GEMM's 16-row
//! tile time is flat from M = 1 to 16 while the GEMV's grows with M, so above
//! M = 1 the GEMV stays faster only for narrow outputs.
inline constexpr int32_t kInt4GemvNarrowMaxN{512};

//! At M = 1 and group size 32, weights with more than this many elements (N * K)
//! go to the GEMM: on SM87 Gemma 4 decoder shapes its 16-row tile measured faster
//! than the GEMV there. At group 128 the GEMV keeps M = 1, where the GEMM measured
//! no faster.
inline constexpr int64_t kInt4G32M1GemmMinWeights{int64_t{4} * 1024 * 1024};

//! GEMV-or-GEMM routing for one launch. m1LargeToGemm enables the group-32 M = 1
//! rule. multiRowToGemm enables the multi-row rule; with it off, every M in
//! [2, kInt4GemvDispatchMaxM] takes the GEMV.
constexpr bool int4RouteToGemv(
    int32_t m, int32_t n, int32_t k, int32_t groupSize, bool m1LargeToGemm, bool multiRowToGemm) noexcept
{
    if (m < 1)
    {
        return false;
    }
    if (m == 1)
    {
        return !(m1LargeToGemm && groupSize == 32
            && static_cast<int64_t>(n) * static_cast<int64_t>(k) > kInt4G32M1GemmMinWeights);
    }
    if (multiRowToGemm)
    {
        return m <= kInt4GemvNarrowDispatchMaxM && n <= kInt4GemvNarrowMaxN;
    }
    return m <= kInt4GemvDispatchMaxM;
}

} // namespace trt_edgellm::plugins
