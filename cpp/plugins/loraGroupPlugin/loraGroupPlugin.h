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
 * @brief Grouped dynamic LoRA for sibling linear layers that share one input.
 *
 * out_i = base_i + (x @ A_i) @ B_i for i < num_targets (q/k/v, or gate/up, or one projection).
 * Replaces insert-lora's MatMul -> MatMul -> Add branches. With inplace = 1 each output
 * aliases its base input, so a target whose bound rank is 0 (no adapter, or a module the
 * adapter does not cover) costs nothing; when every target is inactive no kernel is launched.
 *
 * Inputs: [0, n) base_i [b,s,N_i] or [M,N_i] fp16; [n] x [b,s,K] or [M,K] fp16 (unsmoothed);
 * [n+1, 2n] lora_A_i [K, r_i] fp16; [2n+1, 3n] lora_B_i [r_i, N_i] fp16 (B carries alpha / r).
 * Outputs: [0, n) out_i, shaped like base_i.
 * Attributes: num_targets (1..3), inplace (0/1).
 */
class LoraGroupPlugin : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneBuildV2,
                        public nvinfer1::IPluginV3OneRuntime
{
public:
    LoraGroupPlugin(std::string const& name, int32_t numTargets, int32_t inplace);
    LoraGroupPlugin() = delete;
    LoraGroupPlugin(LoraGroupPlugin const&) = delete;
    ~LoraGroupPlugin() override;

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
    int32_t getAliasedInput(int32_t outputIndex) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    std::string mLayerName;
    std::string mNamespace;
    int32_t mNumTargets{1};
    int32_t mInplace{1};
    void* mState{nullptr};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class LoraGroupPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    LoraGroupPluginCreator();
    ~LoraGroupPluginCreator() override = default;
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
