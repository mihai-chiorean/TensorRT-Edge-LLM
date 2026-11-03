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

#include "common/tensor.h"
#include "runtime/audioUtils.h"
#include "runtime/contextCacheRequest.h"
#include "runtime/imageUtils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kImageToken{900};
constexpr int32_t kTextToken{7};

imageUtils::ImageData makeMedia(int64_t frames, uint8_t fill, bool isVideo)
{
    Tensor pixels({frames, 2, 2, 3}, DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    std::memset(pixels.rawPointer(), fill, static_cast<size_t>(frames) * 2 * 2 * 3);
    imageUtils::ImageData image(std::move(pixels));
    image.isVideo = isVideo;
    return image;
}

// Text, then `runs` image runs of `runLength` tokens with a two-token text gap (a timestamp) before each.
std::vector<int32_t> promptWithRuns(int32_t runs, int32_t runLength)
{
    std::vector<int32_t> tokens(4, kTextToken);
    for (int32_t r = 0; r < runs; ++r)
    {
        tokens.insert(tokens.end(), 2, kTextToken);
        tokens.insert(tokens.end(), static_cast<size_t>(runLength), kImageToken);
    }
    tokens.insert(tokens.end(), 3, kTextToken);
    return tokens;
}

std::vector<Hash128> hashOf(std::vector<int32_t> const& tokens, std::vector<imageUtils::ImageData> const& images)
{
    return buildPerPositionMediaHash(tokens, {kImageToken}, images, {}, nullptr);
}

} // namespace

TEST(ContextCacheMediaHash, VideoTemporalGroupsShareTheVideoIdentity)
{
    auto const tokens = promptWithRuns(/*runs=*/4, /*runLength=*/5);
    std::vector<imageUtils::ImageData> videoA;
    videoA.push_back(makeMedia(8, 1, true));
    std::vector<Hash128> const hashA = hashOf(tokens, videoA);
    ASSERT_EQ(hashA.size(), tokens.size());

    Hash128 videoHash{};
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        if (tokens[i] == kImageToken)
        {
            if (videoHash == Hash128{})
            {
                videoHash = hashA[i];
            }
            EXPECT_EQ(hashA[i], videoHash) << "position " << i;
        }
        else
        {
            EXPECT_EQ(hashA[i], Hash128{}) << "position " << i;
        }
    }
    EXPECT_NE(videoHash, Hash128{});

    std::vector<imageUtils::ImageData> videoB;
    videoB.push_back(makeMedia(8, 2, true));
    std::vector<Hash128> const hashB = hashOf(tokens, videoB);
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        if (tokens[i] == kImageToken)
        {
            EXPECT_NE(hashA[i], hashB[i]) << "position " << i;
        }
    }
}

TEST(ContextCacheMediaHash, ImageBeforeVideoKeepsItsOwnIdentity)
{
    // One run for the image, then three temporal groups of the video.
    auto const tokens = promptWithRuns(/*runs=*/4, /*runLength=*/3);
    std::vector<imageUtils::ImageData> media;
    media.push_back(makeMedia(1, 3, false));
    media.push_back(makeMedia(6, 4, true));
    std::vector<Hash128> const mixed = hashOf(tokens, media);

    std::vector<imageUtils::ImageData> imageOnly;
    imageOnly.push_back(makeMedia(1, 3, false));
    std::vector<Hash128> const single = hashOf(promptWithRuns(1, 3), imageOnly);

    size_t const firstImage = 6;
    EXPECT_EQ(mixed[firstImage], single[firstImage]);
    size_t const firstVideo = firstImage + 3 + 2;
    ASSERT_EQ(tokens[firstVideo], kImageToken);
    EXPECT_NE(mixed[firstVideo], mixed[firstImage]);
    EXPECT_EQ(mixed[firstVideo], mixed[tokens.size() - 4]);
}

TEST(ContextCacheMediaHash, RunsAfterTwoVideosDependOnBoth)
{
    auto const tokens = promptWithRuns(/*runs=*/4, /*runLength=*/3);
    auto twoVideos = [](uint8_t second) {
        std::vector<imageUtils::ImageData> media;
        media.push_back(makeMedia(4, 6, true));
        media.push_back(makeMedia(4, second, true));
        return media;
    };
    std::vector<Hash128> const first = hashOf(tokens, twoVideos(7));
    std::vector<Hash128> const second = hashOf(tokens, twoVideos(8));
    std::vector<imageUtils::ImageData> oneVideo;
    oneVideo.push_back(makeMedia(4, 6, true));
    std::vector<Hash128> const single = hashOf(tokens, oneVideo);
    size_t const firstRun = 6;
    EXPECT_NE(first[firstRun], second[firstRun]);
    EXPECT_NE(first[firstRun], single[firstRun]);
    EXPECT_EQ(first[firstRun], first[tokens.size() - 4]);
}

TEST(ContextCacheMediaHash, ExtraImageRunsWithoutVideoStillFail)
{
    auto const tokens = promptWithRuns(/*runs=*/2, /*runLength=*/3);
    std::vector<imageUtils::ImageData> image;
    image.push_back(makeMedia(1, 5, false));
    EXPECT_THROW(hashOf(tokens, image), std::exception);
}
