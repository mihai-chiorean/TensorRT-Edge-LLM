/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! Tests for the mel-frame count an audio encoder is run on when its engine profile has a minimum length.

#include "multimodal/common/audioUtils.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace au = trt_edgellm::rt::audioUtils;

// Given an encoder with subsampling factor 4 whose engine profile starts at 16 frames
// When a clip has fewer frames than the profile minimum
// Then the encoder is run on exactly the profile minimum
TEST(EncoderInputFramesTest, ShortClipIsRunAtTheProfileMinimum)
{
    EXPECT_EQ(au::encoderInputFrames(1, 4, 16), 16);
    EXPECT_EQ(au::encoderInputFrames(5, 4, 16), 16);
    EXPECT_EQ(au::encoderInputFrames(12, 4, 16), 16);
    EXPECT_EQ(au::encoderInputFrames(40, 4, 100), 100);
}

// Given an encoder with subsampling factor 4 whose engine profile starts at 16 frames
// When a clip has at least as many frames as the profile minimum
// Then the encoder is run on the clip's own frames rounded up to a multiple of 4
TEST(EncoderInputFramesTest, LongerClipIsOnlyAlignedToTheSubsamplingFactor)
{
    EXPECT_EQ(au::encoderInputFrames(16, 4, 16), 16);
    EXPECT_EQ(au::encoderInputFrames(17, 4, 16), 20);
    EXPECT_EQ(au::encoderInputFrames(1043, 4, 16), 1044);
}

// Given an engine profile minimum that is not a multiple of the subsampling factor
// When a clip is shorter than that minimum
// Then the encoder is run on the minimum rounded up to a multiple of the factor
TEST(EncoderInputFramesTest, UnalignedProfileMinimumIsRoundedUp)
{
    EXPECT_EQ(au::encoderInputFrames(3, 4, 10), 12);
}
