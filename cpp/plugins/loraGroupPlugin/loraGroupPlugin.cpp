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

#include "loraGroupPlugin.h"

#include "common/logger.h"
#include "kernels/loraKernels/loraGroupKernels.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kNAME{"LoraGroupPlugin"};
constexpr char const* kVERSION{"1"};


//! Allocates `bytes` of zeroed device memory for persistent plugin state. Safe to call while
//! another stream on this thread is capturing a CUDA graph: relaxed capture mode admits the
//! allocation, and the zeroing runs and completes on a private non-blocking stream (no legacy
//! default stream, no device-wide synchronization).
bool allocateZeroedState(void*& state, size_t bytes) noexcept
{
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    if (cudaThreadExchangeStreamCaptureMode(&mode) != cudaSuccess)
    {
        (void) cudaGetLastError();
        return false;
    }
    bool ok = cudaMalloc(&state, bytes) == cudaSuccess;
    if (ok)
    {
        cudaStream_t zeroStream{};
        ok = cudaStreamCreateWithFlags(&zeroStream, cudaStreamNonBlocking) == cudaSuccess;
        if (ok)
        {
            ok = cudaMemsetAsync(state, 0, bytes, zeroStream) == cudaSuccess
                && cudaStreamSynchronize(zeroStream) == cudaSuccess;
            cudaStreamDestroy(zeroStream);
        }
    }
    if (!ok)
    {
        if (state != nullptr)
        {
            cudaFree(state);
        }
        state = nullptr;
        (void) cudaGetLastError();
    }
    cudaThreadExchangeStreamCaptureMode(&mode);
    return ok;
}

int64_t leadingVolume(Dims const& dims)
{
    int64_t v{1};
    for (int32_t i = 0; i < dims.nbDims - 1; ++i)
    {
        v *= dims.d[i];
    }
    return v;
}
} // namespace

PluginFieldCollection LoraGroupPluginCreator::mFieldCollection{};
std::vector<PluginField> LoraGroupPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(LoraGroupPluginCreator);

LoraGroupPlugin::LoraGroupPlugin(std::string const& name, int32_t numTargets, int32_t inplace)
    : mLayerName(name)
    , mNumTargets(numTargets)
    , mInplace(inplace)
{
}

LoraGroupPlugin::~LoraGroupPlugin()
{
    if (mState != nullptr)
    {
        cudaFree(mState);
    }
}

IPluginCapability* LoraGroupPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuildV2*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* LoraGroupPlugin::clone() noexcept
{
    try
    {
        auto* p = new LoraGroupPlugin(mLayerName, mNumTargets, mInplace);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* LoraGroupPlugin::getPluginName() const noexcept
{
    return kNAME;
}

char const* LoraGroupPlugin::getPluginVersion() const noexcept
{
    return kVERSION;
}

char const* LoraGroupPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LoraGroupPlugin::setPluginNamespace(char const* ns) noexcept
{
    mNamespace = ns == nullptr ? "" : ns;
}

int32_t LoraGroupPlugin::getNbOutputs() const noexcept
{
    return mNumTargets;
}

int32_t LoraGroupPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept
{
    for (int32_t i = 0; i < nbOutputs; ++i)
    {
        outputTypes[i] = DataType::kHALF;
    }
    return 0;
}

int32_t LoraGroupPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /*shapeInputs*/,
    int32_t /*nbShapeInputs*/, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& /*exprBuilder*/) noexcept
{
    if (nbInputs != 3 * mNumTargets + 1 || nbOutputs != mNumTargets)
    {
        return -1;
    }
    for (int32_t i = 0; i < mNumTargets; ++i)
    {
        outputs[i] = inputs[i];
    }
    return 0;
}

bool LoraGroupPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    int32_t const n = mNumTargets;
    if (nbInputs != 3 * n + 1 || nbOutputs != n || pos < 0 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    auto const& d = inOut[pos].desc;
    if (d.type != DataType::kHALF || d.format != PluginFormat::kLINEAR)
    {
        return false;
    }
    if (pos <= n)
    {
        return d.dims.nbDims == 2 || d.dims.nbDims == 3; // base_i, x
    }
    if (pos <= 3 * n)
    {
        return d.dims.nbDims == 2; // A_i, B_i
    }
    return d.dims.nbDims == inOut[pos - 3 * n - 1].desc.dims.nbDims; // out_i like base_i
}

int32_t LoraGroupPlugin::configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    return 0;
}

size_t LoraGroupPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
{
    int32_t const n = mNumTargets;
    int64_t const maxM = leadingVolume(inputs[n].max);
    int32_t maxRank = 1;
    for (int32_t i = 0; i < n; ++i)
    {
        maxRank = std::max<int32_t>(maxRank, static_cast<int32_t>(inputs[n + 1 + i].max.d[1]));
    }
    return kernels::loraGroupWorkspaceBytes(maxM, n, maxRank);
}

int32_t LoraGroupPlugin::getAliasedInput(int32_t outputIndex) noexcept
{
    return mInplace != 0 && outputIndex >= 0 && outputIndex < mNumTargets ? outputIndex : -1;
}

int32_t LoraGroupPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /*outputDesc*/,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int32_t const n = mNumTargets;
    auto const& xDims = inputDesc[n].dims;
    int64_t const M = leadingVolume(xDims);
    int32_t const K = static_cast<int32_t>(xDims.d[xDims.nbDims - 1]);
    kernels::LoraTarget active[kernels::kLoraGroupMaxTargets];
    int32_t nbActive = 0;
    for (int32_t i = 0; i < n; ++i)
    {
        auto const& bd = inputDesc[i].dims;
        int32_t const N = static_cast<int32_t>(bd.d[bd.nbDims - 1]);
        int32_t const r = static_cast<int32_t>(inputDesc[n + 1 + i].dims.d[1]);
        if (leadingVolume(bd) != M || inputDesc[n + 1 + i].dims.d[0] != K || inputDesc[2 * n + 1 + i].dims.d[0] != r
            || inputDesc[2 * n + 1 + i].dims.d[1] != N)
        {
            LOG_ERROR("LoraGroupPlugin %s: inconsistent shapes for target %d", mLayerName.c_str(), i);
            return -1;
        }
        if (outputs[i] == nullptr || inputs[i] == nullptr)
        {
            LOG_ERROR("LoraGroupPlugin %s: null tensor for target %d", mLayerName.c_str(), i);
            return -1;
        }
        if (r == 0)
        {
            if (mInplace == 0 && M > 0
                && cudaMemcpyAsync(outputs[i], inputs[i], static_cast<size_t>(M) * N * 2, cudaMemcpyDeviceToDevice,
                       stream)
                    != cudaSuccess)
            {
                return -1;
            }
            continue;
        }
        auto& t = active[nbActive++];
        t.A = inputs[n + 1 + i];
        t.B = inputs[2 * n + 1 + i];
        t.base = inputs[i];
        t.out = outputs[i];
        t.r = r;
        t.N = N;
    }
    int32_t const status = kernels::launchLoraGroup(inputs[n], M, K, active, nbActive, workspace, mState, stream);
    if (status != 0)
    {
        LOG_ERROR("LoraGroupPlugin %s: launch failed (M=%lld K=%d active=%d): %s", mLayerName.c_str(),
            static_cast<long long>(M), K, nbActive, cudaGetErrorString(cudaPeekAtLastError()));
    }
    return status;
}

int32_t LoraGroupPlugin::onShapeChange(
    PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/, PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    // Normally allocated in attachToContext; this covers a plugin used without one.
    if (mState != nullptr || allocateZeroedState(mState, kernels::kLoraGroupStateBytes))
    {
        return 0;
    }
    LOG_ERROR("LoraGroupPlugin %s: could not allocate the barrier state", mLayerName.c_str());
    return -1;
}

IPluginV3* LoraGroupPlugin::attachToContext(IPluginResourceContext* /*context*/) noexcept
{
    // Execution contexts are created outside graph capture: allocate the per-context barrier
    // state here rather than at the first shape change.
    auto* plugin = static_cast<LoraGroupPlugin*>(clone());
    if (plugin != nullptr)
    {
        (void) allocateZeroedState(plugin->mState, kernels::kLoraGroupStateBytes);
    }
    return plugin;
}

PluginFieldCollection const* LoraGroupPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("num_targets", &mNumTargets, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("inplace", &mInplace, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

LoraGroupPluginCreator::LoraGroupPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_targets", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("inplace", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* LoraGroupPluginCreator::getPluginName() const noexcept
{
    return kNAME;
}

char const* LoraGroupPluginCreator::getPluginVersion() const noexcept
{
    return kVERSION;
}

PluginFieldCollection const* LoraGroupPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* LoraGroupPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LoraGroupPluginCreator::setPluginNamespace(char const* ns) noexcept
{
    mNamespace = ns == nullptr ? "" : ns;
}

IPluginV3* LoraGroupPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept
{
    try
    {
        int32_t numTargets = 1;
        int32_t inplace = 1;
        for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i)
        {
            std::string const f(fc->fields[i].name);
            if (f == "num_targets")
            {
                numTargets = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            else if (f == "inplace")
            {
                inplace = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }
        if (numTargets < 1 || numTargets > kernels::kLoraGroupMaxTargets)
        {
            return nullptr;
        }
        auto* p = new LoraGroupPlugin(name == nullptr ? "" : name, numTargets, inplace);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (...)
    {
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
