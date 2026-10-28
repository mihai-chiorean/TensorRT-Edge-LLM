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

#include "plugins/int4GroupwiseGemmPluginV2/int4GemvDispatchPolicy.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace trt_edgellm::plugins
{
namespace
{

struct Shape
{
    int32_t n;
    int32_t k;
};

// Gemma 4 E2B attention projections (at or under 4 Mi weights) and the E2B and
// E4B MLP projections (above it).
constexpr std::array<Shape, 3> kSmallShapes{{{2048, 1536}, {1536, 2048}, {256, 1536}}};
constexpr std::array<Shape, 4> kLargeShapes{{{6144, 1536}, {1536, 6144}, {10240, 2560}, {2560, 10240}}};

TEST(Int4GemvDispatchPolicyTest, G32M1LargeWeightsGoToGemm)
{
    for (auto const& s : kLargeShapes)
    {
        EXPECT_FALSE(int4RouteToGemv(1, s.n, s.k, 32, true, true)) << s.n << "x" << s.k;
    }
}

TEST(Int4GemvDispatchPolicyTest, G32M1SmallWeightsStayOnGemv)
{
    for (auto const& s : kSmallShapes)
    {
        EXPECT_TRUE(int4RouteToGemv(1, s.n, s.k, 32, true, true)) << s.n << "x" << s.k;
    }
}

TEST(Int4GemvDispatchPolicyTest, ThresholdIsStrict)
{
    EXPECT_TRUE(int4RouteToGemv(1, 2048, 2048, 32, true, true));
    EXPECT_FALSE(int4RouteToGemv(1, 2048 + 128, 2048, 32, true, true));
    EXPECT_TRUE(int4RouteToGemv(1, 4096, 1024, 32, true, true));
    EXPECT_FALSE(int4RouteToGemv(1, 4096, 1024 + 64, 32, true, true));
}

TEST(Int4GemvDispatchPolicyTest, NoInt32OverflowOnHugeWeights)
{
    EXPECT_FALSE(int4RouteToGemv(1, 262144, 16384, 32, true, true));
}

TEST(Int4GemvDispatchPolicyTest, G128KeepsEveryM1OnGemv)
{
    for (auto const& s : kLargeShapes)
    {
        EXPECT_TRUE(int4RouteToGemv(1, s.n, s.k, 128, true, true)) << s.n << "x" << s.k;
    }
}

TEST(Int4GemvDispatchPolicyTest, SwitchOffKeepsEveryM1OnGemv)
{
    for (auto const& s : kLargeShapes)
    {
        EXPECT_TRUE(int4RouteToGemv(1, s.n, s.k, 32, false, true)) << s.n << "x" << s.k;
    }
}

TEST(Int4GemvDispatchPolicyTest, MultiRowRuleOffKeepsSmallMOnGemv)
{
    for (bool const m1Rule : {true, false})
    {
        for (int32_t const groupSize : {32, 128})
        {
            for (auto const& s : kLargeShapes)
            {
                for (int32_t m = 2; m <= kInt4GemvDispatchMaxM; ++m)
                {
                    EXPECT_TRUE(int4RouteToGemv(m, s.n, s.k, groupSize, m1Rule, false));
                }
                EXPECT_FALSE(int4RouteToGemv(kInt4GemvDispatchMaxM + 1, s.n, s.k, groupSize, m1Rule, false));
                EXPECT_FALSE(int4RouteToGemv(0, s.n, s.k, groupSize, m1Rule, false));
            }
        }
    }
}

TEST(Int4GemvDispatchPolicyTest, MultiRowNarrowOutputsStayOnGemv)
{
    for (int32_t const groupSize : {32, 128})
    {
        for (int32_t m = 2; m <= kInt4GemvNarrowDispatchMaxM; ++m)
        {
            EXPECT_TRUE(int4RouteToGemv(m, 256, 1536, groupSize, true, true)) << m;
            EXPECT_TRUE(int4RouteToGemv(m, kInt4GemvNarrowMaxN, 1536, groupSize, true, true)) << m;
        }
        EXPECT_FALSE(int4RouteToGemv(kInt4GemvNarrowDispatchMaxM + 1, 256, 1536, groupSize, true, true));
    }
}

TEST(Int4GemvDispatchPolicyTest, MultiRowWideOutputsGoToGemm)
{
    for (int32_t const groupSize : {32, 128})
    {
        for (int32_t m = 2; m <= kInt4GemvNarrowDispatchMaxM; ++m)
        {
            EXPECT_FALSE(int4RouteToGemv(m, kInt4GemvNarrowMaxN + 128, 1536, groupSize, true, true)) << m;
            for (auto const& s : kSmallShapes)
            {
                if (s.n > kInt4GemvNarrowMaxN)
                {
                    EXPECT_FALSE(int4RouteToGemv(m, s.n, s.k, groupSize, true, true)) << m;
                }
            }
            for (auto const& s : kLargeShapes)
            {
                EXPECT_FALSE(int4RouteToGemv(m, s.n, s.k, groupSize, true, true)) << m;
            }
        }
    }
}

TEST(Int4GemvDispatchPolicyTest, MultiRowRuleLeavesM1Alone)
{
    for (bool const m1Rule : {true, false})
    {
        for (int32_t const groupSize : {32, 128})
        {
            for (auto const& s : kLargeShapes)
            {
                EXPECT_EQ(int4RouteToGemv(1, s.n, s.k, groupSize, m1Rule, true),
                    int4RouteToGemv(1, s.n, s.k, groupSize, m1Rule, false));
            }
            for (auto const& s : kSmallShapes)
            {
                EXPECT_TRUE(int4RouteToGemv(1, s.n, s.k, groupSize, m1Rule, true));
            }
        }
    }
}

} // namespace
} // namespace trt_edgellm::plugins
