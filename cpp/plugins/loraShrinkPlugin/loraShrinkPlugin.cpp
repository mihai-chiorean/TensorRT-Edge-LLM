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

#include "loraShrinkPlugin.h"

#include "common/logger.h"
#include "kernels/loraKernels/loraShrinkKernels.h"

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
constexpr char const* kNAME{"LoraShrinkPlugin"};
constexpr char const* kVERSION{"1"};
constexpr char const* kMAX_M_FIELD{"max_m"};

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

PluginFieldCollection LoraShrinkPluginCreator::mFieldCollection{};
std::vector<PluginField> LoraShrinkPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(LoraShrinkPluginCreator);

LoraShrinkPlugin::LoraShrinkPlugin(std::string const& name, int32_t numTargets)
    : mLayerName(name)
    , mNumTargets(numTargets)
{
}

LoraShrinkPlugin::~LoraShrinkPlugin()
{
    if (mCounters != nullptr)
    {
        cudaFree(mCounters);
    }
}

bool LoraShrinkPlugin::ensureCounters() noexcept
{
    if (mCounters != nullptr)
    {
        return true;
    }
    if (mMaxM <= 0)
    {
        return false;
    }
    size_t const bytes = kernels::loraShrinkCounterWords(mMaxM, mNumTargets) * sizeof(int32_t);
    // Contexts are normally created outside capture; relaxed mode admits the allocation when another
    // stream on this thread is capturing, and the zeroing completes on a private stream.
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    if (cudaThreadExchangeStreamCaptureMode(&mode) != cudaSuccess)
    {
        (void) cudaGetLastError();
        return false;
    }
    bool ok = cudaMalloc(&mCounters, bytes) == cudaSuccess;
    if (ok)
    {
        cudaStream_t zeroStream{};
        ok = cudaStreamCreateWithFlags(&zeroStream, cudaStreamNonBlocking) == cudaSuccess;
        if (ok)
        {
            ok = cudaMemsetAsync(mCounters, 0, bytes, zeroStream) == cudaSuccess
                && cudaStreamSynchronize(zeroStream) == cudaSuccess;
            cudaStreamDestroy(zeroStream);
        }
    }
    (void) cudaThreadExchangeStreamCaptureMode(&mode);
    if (!ok)
    {
        if (mCounters != nullptr)
        {
            cudaFree(mCounters);
        }
        mCounters = nullptr;
        (void) cudaGetLastError();
        LOG_ERROR("LoraShrinkPlugin %s: could not allocate its counters", mLayerName.c_str());
    }
    return ok;
}

IPluginCapability* LoraShrinkPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* LoraShrinkPlugin::clone() noexcept
{
    try
    {
        auto* p = new LoraShrinkPlugin(mLayerName, mNumTargets);
        p->setPluginNamespace(mNamespace.c_str());
        p->mMaxM = mMaxM;
        return p;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* LoraShrinkPlugin::getPluginName() const noexcept
{
    return kNAME;
}

char const* LoraShrinkPlugin::getPluginVersion() const noexcept
{
    return kVERSION;
}

char const* LoraShrinkPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LoraShrinkPlugin::setPluginNamespace(char const* ns) noexcept
{
    mNamespace = ns == nullptr ? "" : ns;
}

int32_t LoraShrinkPlugin::getNbOutputs() const noexcept
{
    return mNumTargets;
}

int32_t LoraShrinkPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept
{
    for (int32_t i = 0; i < nbOutputs; ++i)
    {
        outputTypes[i] = DataType::kHALF;
    }
    return 0;
}

int32_t LoraShrinkPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /*shapeInputs*/,
    int32_t /*nbShapeInputs*/, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (nbInputs != mNumTargets + 1 || nbOutputs != mNumTargets)
    {
        return -1;
    }
    for (int32_t i = 0; i < mNumTargets; ++i)
    {
        outputs[i] = inputs[0];
        outputs[i].d[inputs[0].nbDims - 1] = exprBuilder.constant(kernels::kLoraShrinkMaxRank);
    }
    return 0;
}

bool LoraShrinkPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    int32_t const n = mNumTargets;
    if (nbInputs != n + 1 || nbOutputs != n || pos < 0 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    auto const& d = inOut[pos].desc;
    if (d.type != DataType::kHALF || d.format != PluginFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0)
    {
        return d.dims.nbDims == 2 || d.dims.nbDims == 3;
    }
    if (pos <= n)
    {
        return d.dims.nbDims == 2 && d.dims.d[0] == inOut[0].desc.dims.d[inOut[0].desc.dims.nbDims - 1];
    }
    return d.dims.nbDims == inOut[0].desc.dims.nbDims;
}

int32_t LoraShrinkPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    if (nbInputs != mNumTargets + 1 || in[0].max.nbDims < 2)
    {
        return -1;
    }
    mMaxM = std::max<int32_t>(mMaxM, static_cast<int32_t>(leadingVolume(in[0].max)));
    for (int32_t i = 1; i <= mNumTargets; ++i)
    {
        if (in[i].max.d[1] > kernels::kLoraShrinkMaxRank)
        {
            LOG_ERROR("LoraShrinkPlugin %s: max LoRA rank %lld exceeds %d", mLayerName.c_str(),
                static_cast<long long>(in[i].max.d[1]), kernels::kLoraShrinkMaxRank);
            return -1;
        }
    }
    return 0;
}

size_t LoraShrinkPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
{
    auto const& x = inputs[0].max;
    return kernels::loraShrinkWorkspaceBytes(
        leadingVolume(x), static_cast<int32_t>(x.d[x.nbDims - 1]), mNumTargets);
}

int32_t LoraShrinkPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /*outputDesc*/,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    auto const& xDims = inputDesc[0].dims;
    int64_t const M = leadingVolume(xDims);
    int32_t const K = static_cast<int32_t>(xDims.d[xDims.nbDims - 1]);
    kernels::LoraShrinkParams p{};
    p.x = static_cast<__half const*>(inputs[0]);
    p.m = static_cast<int32_t>(M);
    p.k = K;
    p.numTargets = mNumTargets;
    bool any = false;
    for (int32_t i = 0; i < mNumTargets; ++i)
    {
        int32_t const r = static_cast<int32_t>(inputDesc[1 + i].dims.d[1]);
        if (inputDesc[1 + i].dims.d[0] != K || r > kernels::kLoraShrinkMaxRank)
        {
            LOG_ERROR("LoraShrinkPlugin %s: lora_A %d is not [%d, <=16]", mLayerName.c_str(), i, K);
            return -1;
        }
        p.target[i] = kernels::LoraShrinkTarget{static_cast<__half const*>(inputs[1 + i]),
            static_cast<__half*>(outputs[i]), r};
        any = any || r > 0;
    }
    if (!any)
    {
        return 0;
    }
    if (M > mMaxM || (mCounters == nullptr && !ensureCounters()) || workspace == nullptr)
    {
        return -1;
    }
    p.partial = static_cast<float*>(workspace);
    p.counter = static_cast<int32_t*>(mCounters);
    int32_t const status = kernels::launchLoraShrink(p, stream);
    if (status != 0)
    {
        LOG_ERROR("LoraShrinkPlugin %s: launch failed (M=%lld K=%d): %s", mLayerName.c_str(),
            static_cast<long long>(M), K, cudaGetErrorString(cudaPeekAtLastError()));
    }
    return status;
}

int32_t LoraShrinkPlugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    // Normally allocated in attachToContext; this covers a plugin used without one.
    return ensureCounters() ? 0 : -1;
}

IPluginV3* LoraShrinkPlugin::attachToContext(IPluginResourceContext* /*context*/) noexcept
{
    // Execution contexts are created outside graph capture: allocate the counters here rather
    // than at the first shape change.
    auto* plugin = static_cast<LoraShrinkPlugin*>(clone());
    if (plugin != nullptr)
    {
        (void) plugin->ensureCounters();
    }
    return plugin;
}

PluginFieldCollection const* LoraShrinkPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("num_targets", &mNumTargets, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(kMAX_M_FIELD, &mMaxM, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

LoraShrinkPluginCreator::LoraShrinkPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_targets", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* LoraShrinkPluginCreator::getPluginName() const noexcept
{
    return kNAME;
}

char const* LoraShrinkPluginCreator::getPluginVersion() const noexcept
{
    return kVERSION;
}

PluginFieldCollection const* LoraShrinkPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* LoraShrinkPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LoraShrinkPluginCreator::setPluginNamespace(char const* ns) noexcept
{
    mNamespace = ns == nullptr ? "" : ns;
}

IPluginV3* LoraShrinkPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept
{
    try
    {
        int32_t numTargets = 1;
        int32_t maxM = 0;
        for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i)
        {
            std::string const f(fc->fields[i].name);
            if (f == "num_targets")
            {
                numTargets = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            else if (f == kMAX_M_FIELD)
            {
                maxM = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }
        if (numTargets < 1 || numTargets > kernels::kLoraShrinkMaxTargets)
        {
            return nullptr;
        }
        auto* p = new LoraShrinkPlugin(name == nullptr ? "" : name, numTargets);
        p->mMaxM = maxM;
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
