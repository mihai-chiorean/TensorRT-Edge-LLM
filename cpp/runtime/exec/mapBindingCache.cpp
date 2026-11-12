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

#include "runtime/exec/mapBindingCache.h"

#include "common/trtUtils.h"

namespace trt_edgellm
{
namespace rt
{

void MapBindingCache::initialize(
    std::vector<char const*> const& names, std::vector<bool> const& isInputFlags, std::vector<bool> const& isMapBound)
{
    mMapBindings.clear();
    mMapBindingIndex.assign(names.size(), -1);
    for (size_t i = 0; i < names.size(); ++i)
    {
        if (!isMapBound[i])
        {
            continue;
        }
        mMapBindingIndex[i] = static_cast<int32_t>(mMapBindings.size());
        mMapBindings.push_back(MapBinding{names[i], isInputFlags[i]});
    }
}

void MapBindingCache::invalidateAll()
{
    for (auto& binding : mMapBindings)
    {
        binding.valid = false;
    }
}

bool MapBindingCache::apply(TensorMap const& map, BindingTarget& target)
{
    for (auto& binding : mMapBindings)
    {
        char const* name = binding.name;
        Tensor* tensor = map.get(name);
        if (tensor == nullptr)
        {
            return false;
        }
        auto const address = reinterpret_cast<uintptr_t>(tensor->rawPointer());
        nvinfer1::Dims const dims = tensor->getTRTDims();
        if (binding.valid && binding.address == address && dimsEqual(binding.dims, dims))
        {
            continue;
        }
        binding.valid = false;
        if (!target.setTensorAddress(name, tensor->rawPointer()))
        {
            return false;
        }
        if (binding.isInput && !target.setInputShape(name, dims))
        {
            return false;
        }
        binding.address = address;
        binding.dims = binding.isInput ? dims : target.getTensorShape(name);
        binding.valid = true;
    }
    return true;
}

std::optional<std::pair<uintptr_t, nvinfer1::Dims>> MapBindingCache::cached(int32_t ioIndex) const
{
    size_t const index = static_cast<size_t>(ioIndex);
    if (index >= mMapBindingIndex.size())
    {
        return std::nullopt;
    }
    int32_t const mapIndex = mMapBindingIndex[index];
    if (mapIndex < 0 || !mMapBindings[static_cast<size_t>(mapIndex)].valid)
    {
        return std::nullopt;
    }
    auto const& binding = mMapBindings[static_cast<size_t>(mapIndex)];
    return std::make_pair(binding.address, binding.dims);
}

} // namespace rt
} // namespace trt_edgellm
