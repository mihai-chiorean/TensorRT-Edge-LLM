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

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

/*!
 * @brief LoRA shrink for projections that read one activation: t_i = fp16(x @ A_i) [.., 16].
 *
 * Feeds Int4GroupwiseGemmLoraPlugin, whose epilogue adds fp16(t_i @ B_i). Columns past a target's
 * rank are zero; a target bound at rank 0 is skipped (its t_i is not written and its consumer runs
 * the plain kernel), and nothing launches when every target is at rank 0.
 *
 * Inputs: [0] x [b, s, K] or [M, K] fp16 (unsmoothed); [1, n] lora_A_i [K, r_i] fp16.
 * Outputs: [0, n) t_i, shaped like x with the last dimension 16, fp16.
 * Attributes: num_targets (1..4).
 */
class LoraShrinkPlugin : public nvinfer1::IPluginV3,
                         public nvinfer1::IPluginV3OneCore,
                         public nvinfer1::IPluginV3OneBuild,
                         public nvinfer1::IPluginV3OneRuntime
{
public:
    LoraShrinkPlugin(std::string const& name, int32_t numTargets);
    LoraShrinkPlugin() = delete;
    LoraShrinkPlugin(LoraShrinkPlugin const&) = delete;
    ~LoraShrinkPlugin() override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    friend class LoraShrinkPluginCreator;
    bool ensureCounters() noexcept;

    std::string mLayerName;
    std::string mNamespace;
    int32_t mNumTargets{1};
    //! Largest flattened M over the profiles; sizes the persistent counters.
    int32_t mMaxM{0};
    //! Per-context completion counters (int32, zero between launches), allocated outside capture.
    void* mCounters{nullptr};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class LoraShrinkPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    LoraShrinkPluginCreator();
    ~LoraShrinkPluginCreator() override = default;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
