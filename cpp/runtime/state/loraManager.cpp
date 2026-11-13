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

#include "runtime/state/loraManager.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/state/contextCache/blockHash.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <memory>
#include <set>
#include <stdexcept>
#include <string_view>

namespace trt_edgellm
{
namespace rt
{

namespace
{

//! Enumerate LoRA weight tensor names from an engine's I/O bindings.
std::vector<std::string> collectLoraWeightsTensorNames(EngineExecutor const& runner)
{
    std::vector<std::string> loraTensorNames;
    int32_t const numIO = runner.getNumIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        std::string const name(runner.getIOTensorName(i));
        if (binding_names::isLoraBinding(name))
        {
            loraTensorNames.push_back(name);
        }
    }
    return loraTensorNames;
}

//! The partner factor of a LoRA binding name: the name with its first `lora_A`
//! replaced by `lora_B`, or vice versa. Returns an empty string when the name
//! carries neither prefix.
std::string partnerName(std::string const& name)
{
    std::size_t const aPos = name.find(binding_names::kLoraAPrefix);
    std::size_t const bPos = name.find(binding_names::kLoraBPrefix);
    if (aPos != std::string::npos && (bPos == std::string::npos || aPos <= bPos))
    {
        return name.substr(0, aPos) + binding_names::kLoraBPrefix
            + name.substr(aPos + std::strlen(binding_names::kLoraAPrefix));
    }
    if (bPos != std::string::npos)
    {
        return name.substr(0, bPos) + binding_names::kLoraAPrefix
            + name.substr(bPos + std::strlen(binding_names::kLoraBPrefix));
    }
    return {};
}

//! Whether @p name is a LoRA A factor (its first A/B prefix hit is `lora_A`).
bool isLoraAName(std::string const& name)
{
    std::size_t const aPos = name.find(binding_names::kLoraAPrefix);
    std::size_t const bPos = name.find(binding_names::kLoraBPrefix);
    return aPos != std::string::npos && (bPos == std::string::npos || aPos <= bPos);
}

//! A LoRA factor must be a 2-D FP16 tensor whose rank dimension sits in the
//! position the engine binds: last dim for A ([k, r]), first dim for B ([r, n]).
int64_t loraRankDimension(std::string const& name, Coords const& shape)
{
    return isLoraAName(name) ? shape[shape.getNumDims() - 1] : shape[0];
}

//! Header-level validation of one adapter tensor, before its payload is read
//! or any GPU memory is allocated; returns its rank dimension.
int64_t validateAdapterTensorHeader(std::string const& adapterName, std::string const& tensorName, Coords const& shape)
{
    bool const isA = tensorName.find(binding_names::kLoraAPrefix) != std::string::npos;
    bool const isB = tensorName.find(binding_names::kLoraBPrefix) != std::string::npos;
    ELLM_CHECK(isA || isB,
        "LoRAManager: adapter '" + adapterName + "' tensor '" + tensorName
            + "' is neither a lora_A nor a lora_B weight");
    ELLM_CHECK(
        shape.getNumDims() == 2, "LoRAManager: adapter '" + adapterName + "' tensor '" + tensorName + "' must be 2-D");
    return loraRankDimension(tensorName, shape);
}

void validateAdapterTensorMetadata(std::string const& adapterName, safetensors::TensorMetadata const& tensor,
    std::map<std::string, int64_t> const& ranksOf, int32_t maxSupportedLoraRank)
{
    std::string const& name = tensor.name;
    ELLM_CHECK(tensor.dataType == nvinfer1::DataType::kHALF,
        "LoRAManager: adapter '" + adapterName + "' tensor '" + name + "' must be FP16");

    ELLM_CHECK(
        tensor.shape.getNumDims() == 2, "LoRAManager: adapter '" + adapterName + "' tensor '" + name + "' must be 2-D");

    std::string const partner = partnerName(name);
    ELLM_CHECK(!partner.empty() && ranksOf.count(partner) != 0,
        "LoRAManager: adapter '" + adapterName + "' tensor '" + name + "' has no partner "
            + (partner.empty() ? std::string("(name carries no lora_A/lora_B prefix)") : "'" + partner + "'"));

    int64_t const rank = loraRankDimension(name, tensor.shape);
    ELLM_CHECK(rank >= 1,
        "LoRAManager: adapter '" + adapterName + "' tensor '" + name + "' has rank " + std::to_string(rank)
            + "; must be at least 1");
    ELLM_CHECK(rank <= maxSupportedLoraRank,
        "LoRAManager: adapter '" + adapterName + "' tensor '" + name + "' has rank " + std::to_string(rank)
            + " exceeding the engine's max LoRA rank " + std::to_string(maxSupportedLoraRank));
    if (!partner.empty() && ranksOf.count(partner) != 0)
    {
        ELLM_CHECK(ranksOf.at(partner) == rank,
            "LoRAManager: adapter '" + adapterName + "' tensors '" + name + "' and '" + partner
                + "' disagree on rank: " + std::to_string(rank) + " vs " + std::to_string(ranksOf.at(partner)));
    }
}

//! Reject non-finite FP16 payloads. Any Inf or NaN bit pattern makes the
//! adapter's output undefined, and the decode CUDA-graph capture only warns on
//! failure, so the garbage would otherwise reach the server silently.
//! Safetensors payload offsets are not guaranteed to be 2-byte aligned, so words are read through memcpy.
void validateFiniteOnHost(
    std::string const& adapterName, std::string const& tensorName, void const* payload, size_t bytes)
{
    auto const* const data = static_cast<unsigned char const*>(payload);
    size_t const count = bytes / sizeof(uint16_t);
    for (size_t i = 0; i < count; ++i)
    {
        uint16_t bits{};
        std::memcpy(&bits, data + i * sizeof(uint16_t), sizeof(bits));
        ELLM_CHECK((bits & 0x7C00U) != 0x7C00U,
            "LoRAManager: adapter '" + adapterName + "' tensor '" + tensorName
                + "' has a non-finite (Inf/NaN) value at element " + std::to_string(i));
    }
}

//! Check every registered adapter against the engine bindings recorded in
//! @p maxShapes. A tensor that names no engine binding, or whose non-rank
//! dimensions disagree with the binding's max profile shape, cannot bind and
//! would silently fall back to the rank-1 dummy at every refresh.
void validateAdaptersAgainstBindings(std::string const& contextLabel, std::map<std::string, rt::Tensor> const& weights,
    std::string const& adapterName, std::map<std::string, nvinfer1::Dims> const& maxShapes)
{
    for (auto const& [tensorName, tensor] : weights)
    {
        auto const bindingIt = maxShapes.find(tensorName);
        ELLM_CHECK(bindingIt != maxShapes.end(),
            contextLabel + ": adapter '" + adapterName + "' tensor '" + tensorName
                + "' does not name an engine LoRA binding");

        nvinfer1::Dims const bindingShape = bindingIt->second;
        nvinfer1::Dims const tensorShape = tensor.getShape().getTRTDims();
        ELLM_CHECK(tensorShape.nbDims == bindingShape.nbDims,
            contextLabel + ": adapter '" + adapterName + "' tensor '" + tensorName + "' is rank "
                + std::to_string(tensorShape.nbDims) + " but its engine binding is rank "
                + std::to_string(bindingShape.nbDims));
        bool const tensorIsA = isLoraAName(tensorName);
        int32_t const rankDim = tensorIsA ? tensorShape.nbDims - 1 : 0;
        for (int32_t d = 0; d < tensorShape.nbDims; ++d)
        {
            if (d == rankDim)
            {
                ELLM_CHECK(tensorShape.d[d] <= bindingShape.d[d],
                    contextLabel + ": adapter '" + adapterName + "' tensor '" + tensorName + "' has rank "
                        + std::to_string(tensorShape.d[d]) + " exceeding its binding's max rank "
                        + std::to_string(bindingShape.d[d]));
            }
            else
            {
                ELLM_CHECK(tensorShape.d[d] == bindingShape.d[d],
                    contextLabel + ": adapter '" + adapterName + "' tensor '" + tensorName + "' dimension "
                        + std::to_string(d) + " is " + std::to_string(tensorShape.d[d])
                        + " but its engine binding's max shape is " + std::to_string(bindingShape.d[d]));
            }
        }
    }
}

//! One adapter factor as FP16 host bytes: either the file mapping itself or a dequantized copy.
struct HostFactor
{
    safetensors::TensorMetadata meta; //!< dataType is the stored type for FP16 files, kHALF after dequantization
    void const* data{nullptr};
    std::shared_ptr<std::vector<uint16_t>> storage; //!< Owns dequantized bytes; empty for FP16 files
};

constexpr char const* kScaleSuffix = "_scale";

bool isScaleName(std::string const& name)
{
    std::string_view const suffix{kScaleSuffix};
    return name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

//! Collect the adapter's factors as FP16 host data. An INT8 transport file holds I8 factors and one FP16 scale per
//! factor column (`<factor>_scale`, shape [columns]); it is dequantized as float(code) * float(scale) rounded to FP16.
//! Every malformed combination (mixed FP16/I8 factors, a missing, misshaped, non-FP16, non-finite or non-positive
//! scale, a scale without its factor) throws before any GPU allocation.
template <typename PayloadFn>
std::vector<HostFactor> readHostFactors(
    std::string const& adapterName, safetensors::FileMetadata const& metadata, PayloadFn const& payloadOf)
{
    std::map<std::string, safetensors::TensorMetadata const*> scales;
    std::vector<safetensors::TensorMetadata const*> factorEntries;
    for (auto const& entry : metadata.tensors)
    {
        if (isScaleName(entry.name))
        {
            scales[entry.name] = &entry;
        }
        else
        {
            factorEntries.push_back(&entry);
        }
    }
    bool const anyInt8 = std::any_of(factorEntries.begin(), factorEntries.end(),
        [](auto const* entry) { return entry->dataType == nvinfer1::DataType::kINT8; });
    if (!anyInt8)
    {
        ELLM_CHECK(scales.empty(),
            "LoRAManager: adapter '" + adapterName + "' has scale tensor '" + scales.begin()->first
                + "' but no INT8 factors");
        std::vector<HostFactor> factors;
        for (auto const* entry : factorEntries)
        {
            factors.push_back(HostFactor{*entry, payloadOf(*entry), nullptr});
        }
        return factors;
    }

    std::vector<HostFactor> factors;
    for (auto const* entry : factorEntries)
    {
        std::string const& name = entry->name;
        ELLM_CHECK(entry->dataType == nvinfer1::DataType::kINT8,
            "LoRAManager: adapter '" + adapterName + "' mixes FP16 and INT8 factors (tensor '" + name + "')");
        ELLM_CHECK(entry->shape.getNumDims() == 2,
            "LoRAManager: adapter '" + adapterName + "' tensor '" + name + "' must be 2-D");
        auto const scaleIt = scales.find(name + kScaleSuffix);
        ELLM_CHECK(scaleIt != scales.end(),
            "LoRAManager: adapter '" + adapterName + "' INT8 tensor '" + name + "' has no '" + name + kScaleSuffix
                + "'");
        safetensors::TensorMetadata const& scale = *scaleIt->second;
        int64_t const rows = entry->shape[0];
        int64_t const columns = entry->shape[1];
        ELLM_CHECK(scale.dataType == nvinfer1::DataType::kHALF && scale.shape.getNumDims() == 1
                && scale.shape[0] == columns,
            "LoRAManager: adapter '" + adapterName + "' scale '" + scale.name + "' must be FP16 with "
                + std::to_string(columns) + " elements");
        auto const* codes = static_cast<int8_t const*>(static_cast<void const*>(payloadOf(*entry)));
        auto const* scaleBytes = static_cast<unsigned char const*>(static_cast<void const*>(payloadOf(scale)));
        std::vector<float> scaleValues(static_cast<size_t>(columns));
        for (int64_t c = 0; c < columns; ++c)
        {
            __half_raw raw{};
            std::memcpy(&raw.x, scaleBytes + c * sizeof(uint16_t), sizeof(uint16_t));
            float const value = __half2float(__half(raw));
            ELLM_CHECK(std::isfinite(value) && value > 0.0F,
                "LoRAManager: adapter '" + adapterName + "' scale '" + scale.name + "' element " + std::to_string(c)
                    + " is not a positive finite value");
            scaleValues[static_cast<size_t>(c)] = value;
        }
        auto storage = std::make_shared<std::vector<uint16_t>>(static_cast<size_t>(rows * columns));
        for (int64_t r = 0; r < rows; ++r)
        {
            for (int64_t c = 0; c < columns; ++c)
            {
                size_t const i = static_cast<size_t>(r * columns + c);
                __half_raw const value
                    = __half_raw(__float2half_rn(static_cast<float>(codes[i]) * scaleValues[static_cast<size_t>(c)]));
                (*storage)[i] = value.x;
            }
        }
        safetensors::TensorMetadata meta{name, nvinfer1::DataType::kHALF, entry->shape, 0, storage->size() * sizeof(uint16_t)};
        factors.push_back(HostFactor{meta, storage->data(), storage});
        scales.erase(scaleIt);
    }
    ELLM_CHECK(scales.empty(),
        "LoRAManager: adapter '" + adapterName + "' scale '" + (scales.empty() ? std::string() : scales.begin()->first)
            + "' has no INT8 factor");
    return factors;
}

//! Accumulates the per-tensor lines that define an adapter's content identity.
//! Tensors must be added in ascending name order.
class AdapterIdentityBuilder
{
public:
    void add(std::string const& tensorName, nvinfer1::DataType dataType, Coords const& shape, void const* payload,
        size_t bytes)
    {
        Hash128 const payloadHash = hashOpaqueIdentity(std::string_view(static_cast<char const*>(payload), bytes));
        mManifest += tensorName;
        mManifest += '\t';
        mManifest += std::to_string(static_cast<int32_t>(dataType));
        mManifest += '\t';
        for (int32_t d = 0; d < shape.getNumDims(); ++d)
        {
            if (d > 0)
            {
                mManifest += 'x';
            }
            mManifest += std::to_string(shape[d]);
        }
        mManifest += '\t';
        mManifest += toHex(payloadHash);
        mManifest += '\n';
    }

    std::string finish() const
    {
        return toHex(hashOpaqueIdentity(mManifest));
    }

private:
    static std::string toHex(Hash128 const& hash)
    {
        char buffer[33];
        std::snprintf(buffer, sizeof(buffer), "%016" PRIx64 "%016" PRIx64, hash.hi, hash.lo);
        return std::string(buffer, 32);
    }

    std::string mManifest;
};

} // namespace

LoRAManager::LoRAManager(int32_t maxSupportedLoraRank) noexcept
    : mMaxSupportedLoraRank(maxSupportedLoraRank)
{
}

void LoRAManager::loadWeights(std::string const& name, std::filesystem::path const& path, cudaStream_t stream)
{
    ELLM_CHECK(mAdapters.find(name) == mAdapters.end(),
        "LoRAManager: adapter '" + name + "' is already registered; adapters are immutable for the runtime's lifetime");

    LOG_INFO("LoRAManager: loading adapter '%s' from %s", name.c_str(), path.string().c_str());

    // Parse the header from the file mapping and validate every tensor before
    // allocating adapter GPU memory: a rejected adapter must not leave partial
    // allocations behind.
    file_io::MmapReader reader(path);
    safetensors::FileMetadata const metadata
        = safetensors::parseMetadata(reader.getData(), reader.getSize(), path.string());
    auto const payloadOf = [&](safetensors::TensorMetadata const& entry) {
        return reader.getByteData() + metadata.dataOffset + entry.offset;
    };

    // FP16 factors are used as stored; an INT8 transport file (factors I8 plus one FP16 scale per column,
    // named <factor>_scale) is dequantized here so the registered adapter is FP16 either way.
    std::vector<HostFactor> const factors = readHostFactors(name, metadata, payloadOf);

    std::map<std::string, int64_t> ranksOf;
    for (auto const& factor : factors)
    {
        ELLM_CHECK(ranksOf.emplace(factor.meta.name, validateAdapterTensorHeader(name, factor.meta.name, factor.meta.shape))
                       .second,
            "LoRAManager: adapter '" + name + "' contains duplicate tensor '" + factor.meta.name + "'");
    }
    for (auto const& factor : factors)
    {
        validateAdapterTensorMetadata(name, factor.meta, ranksOf, mMaxSupportedLoraRank);
        validateFiniteOnHost(name, factor.meta.name, factor.data, factor.meta.bytes);
    }

    std::vector<HostFactor const*> byName;
    byName.reserve(factors.size());
    for (auto const& factor : factors)
    {
        byName.push_back(&factor);
    }
    std::sort(byName.begin(), byName.end(), [](auto const* a, auto const* b) { return a->meta.name < b->meta.name; });
    AdapterIdentityBuilder identity;
    for (auto const* factor : byName)
    {
        identity.add(factor->meta.name, factor->meta.dataType, factor->meta.shape, factor->data, factor->meta.bytes);
    }

    // Metadata and values are valid; allocate and copy the payloads (the loop from safetensors::loadSafetensors).
    WeightMap weights;
    for (auto const& factor : factors)
    {
        Tensor tensor(factor.meta.shape, DeviceType::kGPU, factor.meta.dataType, factor.meta.name);
        ELLM_CHECK(static_cast<size_t>(tensor.getMemoryCapacity()) == factor.meta.bytes,
            "LoRAManager: adapter '" + name + "' tensor '" + factor.meta.name
                + "' byte count does not match its metadata");
        CUDA_CHECK(
            cudaMemcpyAsync(tensor.rawPointer(), factor.data, factor.meta.bytes, cudaMemcpyHostToDevice, stream));
        weights.emplace(factor.meta.name, std::move(tensor));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mEngineBindingNames.empty())
    {
        validateAdaptersAgainstBindings("LoRAManager", weights, name, mEngineMaxBindingShapes);
    }

    LOG_INFO("LoRAManager: adapter '%s' loaded with %zu binding(s)", name.c_str(), weights.size());

    mAdapterIdentities[name] = identity.finish();
    mAdapters[name] = std::move(weights);
}

void LoRAManager::addWeights(std::string const& name, std::map<std::string, rt::Tensor> weights)
{
    ELLM_CHECK(mAdapters.find(name) == mAdapters.end(),
        "LoRAManager: adapter '" + name + "' is already registered; adapters are immutable for the runtime's lifetime");

    LOG_INFO("LoRAManager: adding adapter '%s' with %zu binding(s)", name.c_str(), weights.size());

    std::map<std::string, int64_t> ranksOf;
    for (auto const& [tensorName, tensor] : weights)
    {
        ELLM_CHECK(ranksOf.emplace(tensorName, validateAdapterTensorHeader(name, tensorName, tensor.getShape())).second,
            "LoRAManager: adapter '" + name + "' contains duplicate tensor '" + tensorName + "'");
    }
    AdapterIdentityBuilder identity; // std::map iterates in ascending name order
    for (auto const& [tensorName, tensor] : weights)
    {
        validateAdapterTensorMetadata(name,
            safetensors::TensorMetadata{tensorName, tensor.getDataType(), tensor.getShape(), 0, 0}, ranksOf,
            mMaxSupportedLoraRank);

        size_t const bytes = static_cast<size_t>(tensor.getShape().volume()) * sizeof(uint16_t);
        if (tensor.getDeviceType() == DeviceType::kGPU)
        {
            std::vector<uint16_t> host(bytes / sizeof(uint16_t));
            CUDA_CHECK(cudaMemcpy(host.data(), tensor.rawPointer(), bytes, cudaMemcpyDeviceToHost));
            validateFiniteOnHost(name, tensorName, host.data(), bytes);
            identity.add(tensorName, tensor.getDataType(), tensor.getShape(), host.data(), bytes);
        }
        else
        {
            validateFiniteOnHost(name, tensorName, tensor.rawPointer(), bytes);
            identity.add(tensorName, tensor.getDataType(), tensor.getShape(), tensor.rawPointer(), bytes);
        }
    }

    if (!mEngineBindingNames.empty())
    {
        validateAdaptersAgainstBindings("LoRAManager", weights, name, mEngineMaxBindingShapes);
    }

    mAdapterIdentities[name] = identity.finish();
    mAdapters[name] = std::move(weights);
}

void LoRAManager::switchWeights(std::string const& name)
{
    auto const it = mAdapters.find(name);
    ELLM_CHECK(it != mAdapters.end(), "LoRAManager: adapter '" + name + "' not found");
    mActiveAdapterName = name;
    LOG_INFO("LoRAManager: switched to adapter '%s'", name.c_str());
}

void LoRAManager::resetWeights()
{
    mActiveAdapterName.clear();
    LOG_INFO("LoRAManager: adapter deactivated");
}

rt::Tensor& LoRAManager::getActiveWeight(std::string const& bindingName)
{
    if (mActiveAdapterName.empty())
    {
        // Lazy-initialize dummy tensor on first use. Zero-fill so any TRT
        // engine that reads it (rank=1 LoRA contraction on inactive adapters)
        // sees deterministic zeros rather than uninitialized GPU memory.
        if (mDummyTensor.isEmpty())
        {
            mDummyTensor = rt::Tensor({1}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "LoRAManager::dummy");
            CUDA_CHECK(cudaMemset(mDummyTensor.rawPointer(), 0, mDummyTensor.getMemoryCapacity()));
        }
        return mDummyTensor;
    }

    auto const adapterIt = mAdapters.find(mActiveAdapterName);
    ELLM_CHECK(adapterIt != mAdapters.end(),
        "LoRAManager: active adapter '" + mActiveAdapterName + "' disappeared unexpectedly");

    auto const weightIt = adapterIt->second.find(bindingName);
    ELLM_CHECK(weightIt != adapterIt->second.end(),
        "LoRAManager: binding '" + bindingName + "' not found in adapter '" + mActiveAdapterName + "'");

    return weightIt->second;
}

std::string const& LoRAManager::getActiveAdapterName() const noexcept
{
    return mActiveAdapterName;
}

std::vector<std::string> LoRAManager::getBindingNames() const
{
    std::set<std::string> names;
    for (auto const& [adapterName, weightMap] : mAdapters)
    {
        for (auto const& [bindingName, tensor] : weightMap)
        {
            names.insert(bindingName);
        }
    }
    return {names.begin(), names.end()};
}

std::vector<std::string> LoRAManager::getAdapterNames() const
{
    std::vector<std::string> names;
    names.reserve(mAdapters.size());
    for (auto const& [adapterName, weightMap] : mAdapters)
    {
        names.push_back(adapterName);
    }
    return names;
}

std::string const& LoRAManager::getAdapterIdentity(std::string const& name) const
{
    auto const it = mAdapterIdentities.find(name);
    ELLM_CHECK(it != mAdapterIdentities.end(), "LoRAManager: adapter '" + name + "' not found");
    return it->second;
}

bool LoRAManager::hasActiveAdapter() const noexcept
{
    return !mActiveAdapterName.empty();
}

bool LoRAManager::hasWeightFor(std::string const& bindingName) const noexcept
{
    if (mActiveAdapterName.empty())
    {
        return false;
    }
    auto const adapterIt = mAdapters.find(mActiveAdapterName);
    if (adapterIt == mAdapters.end())
    {
        return false;
    }
    return adapterIt->second.find(bindingName) != adapterIt->second.end();
}

void LoRAManager::initializeEngineBindings(EngineExecutor const& runner)
{
    constexpr int32_t kEmptyLoraRank = 1;

    std::vector<std::string> names = collectLoraWeightsTensorNames(runner);
    std::map<std::string, rt::Tensor> dummyTensors;
    std::map<std::string, nvinfer1::Dims> maxShapes;

    for (auto const& name : names)
    {
        nvinfer1::Dims const maxShape
            = runner.getProfileShape(name.c_str(), /*profileIndex=*/0, nvinfer1::OptProfileSelector::kMAX);
        nvinfer1::Dims dummyShape = maxShape;
        if (name.find(binding_names::kLoraAPrefix) != std::string::npos)
        {
            // LoRA A: [k, rank] → set rank (last dim) to 1
            dummyShape.d[dummyShape.nbDims - 1] = kEmptyLoraRank;
        }
        else if (name.find(binding_names::kLoraBPrefix) != std::string::npos)
        {
            // LoRA B: [rank, n] → set rank (first dim) to 1
            dummyShape.d[0] = kEmptyLoraRank;
        }
        std::vector<int64_t> shape(dummyShape.nbDims);
        for (int32_t d = 0; d < dummyShape.nbDims; ++d)
        {
            shape[d] = dummyShape.d[d];
        }
        // Zero-fill: when no adapter is active these dummies are bound directly
        // to the engine's lora_A_*/lora_B_* inputs, so the rank-1 contraction
        // must read deterministic zeros, not uninitialized cudaMalloc memory.
        auto [it, inserted] = dummyTensors.emplace(
            name, rt::Tensor(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "loraDummy_" + name));
        CUDA_CHECK(cudaMemset(it->second.rawPointer(), 0, it->second.getMemoryCapacity()));
        maxShapes.emplace(name, maxShape);
    }

    mEngineBindingNames = std::move(names);
    mEngineDummyTensors = std::move(dummyTensors);
    mEngineMaxBindingShapes = std::move(maxShapes);
    LOG_INFO("LoRAManager: registered %zu engine LoRA binding name(s)", mEngineBindingNames.size());

    // Adapters are loaded before the bindings are known (step 6 vs step 8 of
    // runtime initialization). Reconcile them now, before any graph capture.
    for (auto const& [adapterName, weights] : mAdapters)
    {
        validateAdaptersAgainstBindings("LoRAManager", weights, adapterName, mEngineMaxBindingShapes);
    }
}

void LoRAManager::refreshTensorMap(TensorMap& map)
{
    for (auto const& engineBindingName : mEngineBindingNames)
    {
        // Exact name match works for non-fused projections. Fused engines use
        // a different naming convention (e.g. `qkv_proj.*` vs the adapter's
        // separate `q_proj.*` / `k_proj.*` / `v_proj.*`); for those bindings
        // we fall back to the per-binding dummy tensor.
        if (hasWeightFor(engineBindingName))
        {
            map.set(engineBindingName, getActiveWeight(engineBindingName));
            continue;
        }

        auto it = mEngineDummyTensors.find(engineBindingName);
        if (it != mEngineDummyTensors.end())
        {
            map.set(engineBindingName, it->second);
        }
    }

    if (mActiveAdapterName.empty() || mWarnedUnboundAdapters.count(mActiveAdapterName) != 0)
    {
        return;
    }

    std::vector<std::string> unbound;
    auto const adapterIt = mAdapters.find(mActiveAdapterName);
    if (adapterIt != mAdapters.end())
    {
        for (auto const& engineBindingName : mEngineBindingNames)
        {
            if (adapterIt->second.find(engineBindingName) == adapterIt->second.end())
            {
                unbound.push_back(engineBindingName);
            }
        }
    }
    if (unbound.empty())
    {
        return;
    }

    std::string samples;
    int32_t const numSamples = std::min<size_t>(unbound.size(), 3);
    for (int32_t i = 0; i < numSamples; ++i)
    {
        if (i != 0)
        {
            samples += ", ";
        }
        samples += "'" + unbound[static_cast<size_t>(i)] + "'";
    }
    LOG_WARNING(
        "LoRAManager: adapter '%s' leaves %zu of %zu engine LoRA binding(s) on the rank-1 zero dummy; "
        "they contribute no adapter update. Samples: %s.",
        mActiveAdapterName.c_str(), unbound.size(), mEngineBindingNames.size(), samples.c_str());
    mWarnedUnboundAdapters.emplace(mActiveAdapterName);
}

} // namespace rt
} // namespace trt_edgellm
