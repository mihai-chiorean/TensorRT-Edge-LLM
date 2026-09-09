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

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <cuda_fp16.h>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

Hash128 makeEncoderEmbeddingKey(std::string_view pixelBytes, EncoderEmbeddingKeyParams const& params)
{
    Hash128 const content = hashOpaqueIdentity(pixelBytes);
    std::string identity;
    identity.reserve(sizeof(Hash128) + 9 * sizeof(int64_t));
    auto append = [&identity](int64_t value) { identity.append(reinterpret_cast<char const*>(&value), sizeof(value)); };
    append(static_cast<int64_t>(content.hi));
    append(static_cast<int64_t>(content.lo));
    append(params.frames);
    append(params.height);
    append(params.width);
    append(params.channels);
    append(params.doResize ? 1 : 0);
    append(params.targetHeight);
    append(params.targetWidth);
    append(params.patchSize);
    append(params.poolingKernelSize);
    return hashOpaqueIdentity(identity);
}

EncoderEmbeddingCache::EncoderEmbeddingCache(size_t slots, int64_t maxRows, int64_t hiddenSize)
    : mMaxRows(maxRows)
    , mHiddenSize(hiddenSize)
{
    check::check(maxRows > 0 && hiddenSize > 0, "EncoderEmbeddingCache slots must hold at least one row");
    mSlots.reserve(slots);
    for (size_t index = 0; index < slots; ++index)
    {
        Slot slot;
        slot.data
            = Tensor({maxRows, hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "EncoderEmbeddingCache::slot");
        mSlots.push_back(std::move(slot));
    }
}

EncoderEmbeddingCache::Slot* EncoderEmbeddingCache::findSlot(Hash128 const& key) noexcept
{
    for (Slot& slot : mSlots)
    {
        if (slot.rows > 0 && slot.key == key)
        {
            return &slot;
        }
    }
    return nullptr;
}

std::optional<EncoderEmbeddingCache::CachedRows> EncoderEmbeddingCache::find(Hash128 const& key)
{
    Slot* const slot = findSlot(key);
    if (slot == nullptr)
    {
        ++mMisses;
        return std::nullopt;
    }
    slot->lastUse = ++mClock;
    ++mHits;
    return CachedRows{slot->data.rawPointer(), slot->rows};
}

bool EncoderEmbeddingCache::insert(
    Hash128 const& key, Tensor const& source, int64_t rowOffset, int64_t rows, cudaStream_t stream)
{
    check::check(source.getDeviceType() == DeviceType::kGPU && source.getDataType() == nvinfer1::DataType::kHALF,
        "EncoderEmbeddingCache caches FP16 device rows");
    auto const shape = source.getShape();
    check::check(
        shape.getNumDims() == 2 && shape[1] == mHiddenSize, "EncoderEmbeddingCache source must be [rows, hiddenSize]");
    check::check(rowOffset >= 0 && rows > 0 && rowOffset + rows <= shape[0],
        "EncoderEmbeddingCache insert range is outside the source");
    if (mSlots.empty() || rows > mMaxRows)
    {
        return false;
    }

    Slot* target = findSlot(key);
    if (target == nullptr)
    {
        target = &*std::min_element(mSlots.begin(), mSlots.end(), [](Slot const& lhs, Slot const& rhs) {
            // Empty slots sort first, then the least recently used.
            if ((lhs.rows == 0) != (rhs.rows == 0))
            {
                return lhs.rows == 0;
            }
            return lhs.lastUse < rhs.lastUse;
        });
    }

    size_t const rowBytes = static_cast<size_t>(mHiddenSize) * sizeof(half);
    char const* const sourceRows
        = static_cast<char const*>(source.rawPointer()) + static_cast<size_t>(rowOffset) * rowBytes;
    CUDA_CHECK(cudaMemcpyAsync(
        target->data.rawPointer(), sourceRows, static_cast<size_t>(rows) * rowBytes, cudaMemcpyDeviceToDevice, stream));
    target->key = key;
    target->rows = rows;
    target->lastUse = ++mClock;
    return true;
}

size_t EncoderEmbeddingCache::size() const noexcept
{
    return static_cast<size_t>(
        std::count_if(mSlots.begin(), mSlots.end(), [](Slot const& slot) { return slot.rows > 0; }));
}

int64_t EncoderEmbeddingCache::bytesPerSlot() const noexcept
{
    return mMaxRows * mHiddenSize * static_cast<int64_t>(sizeof(half));
}

} // namespace rt
} // namespace trt_edgellm
