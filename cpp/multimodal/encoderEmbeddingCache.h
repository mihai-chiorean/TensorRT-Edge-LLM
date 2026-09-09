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

#include "common/tensor.h"
#include "runtime/state/contextCache/blockHash.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <optional>
#include <string_view>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Preprocessing parameters that, together with the pixel content, determine an encoder's output rows.
//! Runner-constant normalization (image mean/std) is deliberately absent: a cache belongs to one runner.
struct EncoderEmbeddingKeyParams
{
    int64_t frames{1};
    int64_t height{0};
    int64_t width{0};
    int64_t channels{0};
    bool doResize{true};
    int64_t targetHeight{0}; //!< Spatial size actually fed to the encoder after any resize
    int64_t targetWidth{0};
    int64_t patchSize{0};
    int64_t poolingKernelSize{0};
};

//! Identity of one encoder run over one image: content hash (shared with the context cache) mixed with params.
Hash128 makeEncoderEmbeddingKey(std::string_view pixelBytes, EncoderEmbeddingKeyParams const& params);

//! Bounded LRU of per-image encoder output rows held on the device.
//!
//! Every slot is allocated at construction with room for `maxRows` FP16 rows, so lookups and inserts never
//! allocate on the request path. All copies are enqueued on the caller's stream; a hit's rows must be
//! consumed on that same stream before any later insert can recycle the slot.
class EncoderEmbeddingCache
{
public:
    struct CachedRows
    {
        void const* data{nullptr};
        int64_t rows{0};
    };

    EncoderEmbeddingCache() = default;
    EncoderEmbeddingCache(size_t slots, int64_t maxRows, int64_t hiddenSize);

    //! Rows cached for key, or nullopt on a miss. A hit becomes the most recently used slot.
    std::optional<CachedRows> find(Hash128 const& key);

    //! Copy rows [rowOffset, rowOffset + rows) of `source` ([N, hiddenSize] FP16, device) into the slot holding
    //! `key`, else into a free slot, else into the least recently used one. Rows beyond `maxRows` are not cached.
    //! @return false when the rows do not fit a slot.
    bool insert(Hash128 const& key, Tensor const& source, int64_t rowOffset, int64_t rows, cudaStream_t stream);

    size_t capacity() const noexcept
    {
        return mSlots.size();
    }

    size_t size() const noexcept;

    //! Device bytes reserved per slot.
    int64_t bytesPerSlot() const noexcept;

    uint64_t hits() const noexcept
    {
        return mHits;
    }

    uint64_t misses() const noexcept
    {
        return mMisses;
    }

private:
    struct Slot
    {
        Hash128 key{};
        int64_t rows{0};
        uint64_t lastUse{0};
        Tensor data;
    };

    Slot* findSlot(Hash128 const& key) noexcept;

    std::vector<Slot> mSlots;
    int64_t mMaxRows{0};
    int64_t mHiddenSize{0};
    uint64_t mClock{0};
    uint64_t mHits{0};
    uint64_t mMisses{0};
};

} // namespace rt
} // namespace trt_edgellm
