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

#include "multimodal/encoderEmbeddingCache.h"

#include "common/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int64_t kHidden{8};

EncoderEmbeddingKeyParams makeParams()
{
    EncoderEmbeddingKeyParams params;
    params.height = 480;
    params.width = 640;
    params.channels = 3;
    params.targetHeight = 432;
    params.targetWidth = 576;
    params.patchSize = 16;
    params.poolingKernelSize = 3;
    return params;
}

// Rows of `source` hold row-major values row * 100 + column so any slice is recognisable.
Tensor makeSource(int64_t rows, cudaStream_t stream)
{
    std::vector<half> host(static_cast<size_t>(rows * kHidden));
    for (int64_t r = 0; r < rows; ++r)
    {
        for (int64_t c = 0; c < kHidden; ++c)
        {
            host[static_cast<size_t>(r * kHidden + c)] = __float2half(static_cast<float>(r * 100 + c));
        }
    }
    Tensor source({rows, kHidden}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    EXPECT_EQ(
        cudaMemcpyAsync(source.rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice, stream),
        cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    return source;
}

std::vector<float> readRows(EncoderEmbeddingCache::CachedRows const& rows, cudaStream_t stream)
{
    std::vector<half> host(static_cast<size_t>(rows.rows * kHidden));
    EXPECT_EQ(cudaMemcpyAsync(host.data(), rows.data, host.size() * sizeof(half), cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<float> values;
    values.reserve(host.size());
    for (half const value : host)
    {
        values.push_back(__half2float(value));
    }
    return values;
}

} // namespace

TEST(EncoderEmbeddingKey, ContentAndParametersBothMatter)
{
    std::string const pixelsA(3000, 'a');
    std::string const pixelsB(3000, 'b');
    EncoderEmbeddingKeyParams const params = makeParams();
    Hash128 const keyA = makeEncoderEmbeddingKey(pixelsA, params);
    EXPECT_EQ(keyA, makeEncoderEmbeddingKey(pixelsA, params));
    EXPECT_NE(keyA, makeEncoderEmbeddingKey(pixelsB, params));

    EncoderEmbeddingKeyParams noResize = params;
    noResize.doResize = false;
    EXPECT_NE(keyA, makeEncoderEmbeddingKey(pixelsA, noResize));

    EncoderEmbeddingKeyParams otherTarget = params;
    otherTarget.targetWidth = 528;
    EXPECT_NE(keyA, makeEncoderEmbeddingKey(pixelsA, otherTarget));

    EncoderEmbeddingKeyParams otherPooling = params;
    otherPooling.poolingKernelSize = 2;
    EXPECT_NE(keyA, makeEncoderEmbeddingKey(pixelsA, otherPooling));
}

TEST(EncoderEmbeddingCache, RoundTripsRowSlicesAndEvictsLeastRecentlyUsed)
{
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    Tensor const source = makeSource(8, stream);
    EncoderEmbeddingCache cache(/*slots=*/2, /*maxRows=*/4, kHidden);
    EXPECT_EQ(cache.capacity(), 2U);
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.bytesPerSlot(), 4 * kHidden * static_cast<int64_t>(sizeof(half)));

    Hash128 const keyA{1, 1};
    Hash128 const keyB{2, 2};
    Hash128 const keyC{3, 3};
    EXPECT_FALSE(cache.find(keyA).has_value());
    EXPECT_TRUE(cache.insert(keyA, source, 0, 3, stream));
    EXPECT_TRUE(cache.insert(keyB, source, 3, 2, stream));
    EXPECT_EQ(cache.size(), 2U);

    auto const hitA = cache.find(keyA);
    ASSERT_TRUE(hitA.has_value());
    EXPECT_EQ(hitA->rows, 3);
    std::vector<float> const valuesA = readRows(*hitA, stream);
    EXPECT_FLOAT_EQ(valuesA.front(), 0.0F);
    EXPECT_FLOAT_EQ(valuesA[static_cast<size_t>(kHidden)], 100.0F);
    EXPECT_FLOAT_EQ(valuesA.back(), 200.0F + static_cast<float>(kHidden - 1));

    auto const hitB = cache.find(keyB);
    ASSERT_TRUE(hitB.has_value());
    EXPECT_EQ(hitB->rows, 2);
    EXPECT_FLOAT_EQ(readRows(*hitB, stream).front(), 300.0F);

    // keyA was used before keyB, so a third key evicts keyA.
    EXPECT_TRUE(cache.insert(keyC, source, 5, 1, stream));
    EXPECT_FALSE(cache.find(keyA).has_value());
    ASSERT_TRUE(cache.find(keyB).has_value());
    auto const hitC = cache.find(keyC);
    ASSERT_TRUE(hitC.has_value());
    EXPECT_FLOAT_EQ(readRows(*hitC, stream).front(), 500.0F);

    // Re-inserting an existing key refreshes it in place instead of consuming another slot.
    EXPECT_TRUE(cache.insert(keyC, source, 6, 2, stream));
    EXPECT_EQ(cache.size(), 2U);
    ASSERT_TRUE(cache.find(keyB).has_value());
    EXPECT_EQ(cache.find(keyC)->rows, 2);

    // Rows beyond a slot's capacity are refused rather than truncated.
    EXPECT_FALSE(cache.insert(keyA, source, 0, 5, stream));
    EXPECT_EQ(cache.hits(), 6U);
    EXPECT_EQ(cache.misses(), 2U);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(EncoderEmbeddingCache, DefaultConstructedCacheNeverHits)
{
    EncoderEmbeddingCache cache;
    EXPECT_EQ(cache.capacity(), 0U);
    EXPECT_FALSE(cache.find(Hash128{7, 7}).has_value());
}
