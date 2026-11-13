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

#include "runtime/exec/tensorMap.h"
#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Minimal view of a TRT execution context that MapBindingCache needs. Lets tests record setter calls
//! without a TRT engine.
class BindingTarget
{
public:
    virtual ~BindingTarget() = default;

    //! Bind a named tensor's device address. Returns false on failure (the cache marks the entry invalid).
    virtual bool setTensorAddress(char const* name, void* address) = 0;

    //! Set the shape of a named input. Returns false on failure (the cache marks the entry invalid).
    virtual bool setInputShape(char const* name, nvinfer1::Dims const& dims) = 0;

    //! Return the current shape of a named tensor on the target.
    virtual nvinfer1::Dims getTensorShape(char const* name) const = 0;
};

//! An engine I/O tensor bound from the TensorMap rather than the registry (LoRA weights and a few others),
//! with the address and shape last set on the target. Valid until the profile changes.
struct MapBinding
{
    char const* name{nullptr};
    bool isInput{false};
    bool valid{false};
    uintptr_t address{0};
    nvinfer1::Dims dims{};
    Tensor* tensor{nullptr}; //!< TensorMap entry resolved for the generation in MapBindingCache
};

/*!
 * @brief Cache of TensorMap-bound engine bindings so unchanged entries skip the setter calls.
 *
 * TensorRT does not clear bindings on setOptimizationProfileAsync: shapes and addresses persist
 * across profile switches and across execute() calls. Rebinding every engine I/O tensor per step is
 * therefore redundant work; LoRA tensors (hundreds per rank-adapted engine) only change at prefill.
 *
 * Usage: initialize() once with the engine I/O tensor names and their input flags, in I/O order.
 * apply() then binds each map-bound tensor through @p target, skipping entries whose address and
 * dims match what was last set. invalidateAll() drops every cached entry (profile switch).
 */
class MapBindingCache
{
public:
    /*!
     * @brief Record the map-bound subset of the engine's I/O tensors.
     *
     * @param names Engine I/O tensor names in I/O order (pointers must outlive the cache)
     * @param isInputFlags Per-name input flag, same order and size as @p names
     * @param isMapBound Per-name predicate: true when the tensor is bound from the TensorMap rather
     *        than the registry
     */
    void initialize(std::vector<char const*> const& names, std::vector<bool> const& isInputFlags,
        std::vector<bool> const& isMapBound);

    //! @brief Whether initialize() has run.
    bool initialized() const noexcept
    {
        return !mMapBindingIndex.empty();
    }

    //! @brief Invalidate every cached entry; the next apply() rebinds all map bindings.
    void invalidateAll();

    /*!
     * @brief Bind all map-bound tensors through @p target, skipping unchanged entries.
     *
     * For each map binding: look up the tensor in @p map (a missing tensor is an error). If the
     * cached entry is valid and address and dims are unchanged, skip. Otherwise call setTensorAddress,
     * plus setInputShape for inputs, then cache the address and the dims (for outputs, the shape the
     * target reports after the bind). A failed setter leaves the entry invalid and returns false.
     *
     * @param map Name-to-tensor mapping holding the current addresses and shapes
     * @param target Binding surface to call through
     * @return True if every map binding was satisfied
     */
    bool apply(TensorMap const& map, BindingTarget& target);

    //! @brief Whether the last apply() called any setter.
    bool lastApplyChanged() const noexcept
    {
        return mLastApplyChanged;
    }

    /*!
     * @brief Cached address and dims for the map binding at I/O position @p ioIndex.
     *
     * @param ioIndex Engine I/O index, as used by initialize()'s @p names order
     * @return The applied address and dims, or nullopt when the position is registry-bound or the
     *         entry has not been applied since the last invalidate
     */
    std::optional<std::pair<uintptr_t, nvinfer1::Dims>> cached(int32_t ioIndex) const;

    //! @brief Name of the binding that made the last apply() fail, or nullptr after a successful apply().
    char const* lastFailedBinding() const noexcept
    {
        return mLastFailedBinding;
    }

    //! @brief Applied map bindings, in engine I/O order of the map-bound positions.
    std::vector<MapBinding> const& bindings() const noexcept
    {
        return mMapBindings;
    }

    //! @brief Position in bindings() of I/O index @p ioIndex, or -1 when registry-bound.
    int32_t mapIndex(int32_t ioIndex) const noexcept
    {
        size_t const index = static_cast<size_t>(ioIndex);
        return index < mMapBindingIndex.size() ? mMapBindingIndex[index] : -1;
    }

private:
    char const* mLastFailedBinding{nullptr};
    bool mLastApplyChanged{false};
    //! TensorMap and generation the cached Tensor pointers were resolved against.
    TensorMap const* mResolvedMap{nullptr};
    uint64_t mResolvedGeneration{0};
    std::vector<MapBinding> mMapBindings;
    //! Per I/O index: position in mMapBindings, or -1 for a registry-bound tensor.
    std::vector<int32_t> mMapBindingIndex;
};

} // namespace rt
} // namespace trt_edgellm
