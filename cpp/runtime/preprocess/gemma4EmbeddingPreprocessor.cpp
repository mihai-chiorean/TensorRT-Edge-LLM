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

#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

Gemma4EmbeddingPreprocessor::Gemma4EmbeddingPreprocessor(std::filesystem::path const& engineDir,
    LLMEngineConfig const& config, int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream,
    std::optional<Tensor> checkpointTable, EmbeddingTableResidency residency)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.pleEnabled, "Gemma4EmbeddingPreprocessor constructed while PLE is disabled");
    ELLM_CHECK(maxBatchSize > 0, "Gemma4EmbeddingPreprocessor requires positive max batch size");
    ELLM_CHECK(maxSeqLen > 0, "Gemma4EmbeddingPreprocessor requires positive max sequence length");

    std::filesystem::path const plePath = engineDir / binding_names::kPleEmbeddingFileName;
    nvinfer1::DataType tableDataType{};
    Coords tableShape;
    std::optional<Coords> scaleShape;
    if (residency == EmbeddingTableResidency::kHostMapped)
    {
        ELLM_CHECK(!checkpointTable.has_value(), "A checkpoint-backed PLE table cannot stay host-mapped");
        mHostTable = std::make_unique<HostRowTable>(plePath, "weight", "weight_scale");
        tableDataType = mHostTable->dataType();
        tableShape = mHostTable->shape();
        if (mHostTable->hasScales())
        {
            scaleShape = mHostTable->scaleShape();
        }
        mStaging = std::make_unique<RowStaging>(
            *mHostTable, static_cast<int64_t>(maxBatchSize) * maxSeqLen, "Gemma4EmbeddingPreprocessor::staging");
        mStagedIds.resize(static_cast<size_t>(maxBatchSize) * maxSeqLen);
    }
    else if (checkpointTable.has_value())
    {
        ELLM_CHECK(checkpointTable->getDataType() != nvinfer1::DataType::kINT8,
            "Checkpoint-backed INT8 PLE requires scale ownership and is not supported");
        mPleTable = std::move(*checkpointTable);
        tableDataType = mPleTable.getDataType();
        tableShape = mPleTable.getShape();
    }
    else
    {
        std::vector<Tensor> pleTensors;
        ELLM_CHECK(safetensors::loadSafetensors(plePath, pleTensors, stream),
            "Failed to load " + std::string(binding_names::kPleEmbeddingFileName)
                + " from model directory: " + engineDir.string());
        Tensor* weight = nullptr;
        Tensor* scales = nullptr;
        for (auto& tensor : pleTensors)
        {
            if (tensor.getName() == "weight")
            {
                ELLM_CHECK(weight == nullptr, "ple_embedding.safetensors contains duplicate weight tensors");
                weight = &tensor;
            }
            else if (tensor.getName() == "weight_scale")
            {
                ELLM_CHECK(scales == nullptr, "ple_embedding.safetensors contains duplicate weight_scale tensors");
                scales = &tensor;
            }
        }
        ELLM_CHECK(weight != nullptr, "ple_embedding.safetensors must contain a tensor named weight");
        if (weight->getDataType() == nvinfer1::DataType::kINT8 || weight->getDataType() == nvinfer1::DataType::kUINT8)
        {
            ELLM_CHECK(pleTensors.size() == 2 && scales != nullptr,
                "INT8/INT4 ple_embedding.safetensors must contain exactly weight and weight_scale");
            mPleScales = std::move(*scales);
            scaleShape = mPleScales.getShape();
        }
        else
        {
            ELLM_CHECK(pleTensors.size() == 1 && scales == nullptr,
                "FP16/BF16 ple_embedding.safetensors must contain exactly one tensor named weight");
        }
        mPleTable = std::move(*weight);
        tableDataType = mPleTable.getDataType();
        tableShape = mPleTable.getShape();
    }

    ELLM_CHECK(tableShape.getNumDims() == 2, "PLE table must be 2D [vocab, num_layers * hidden]");
    ELLM_CHECK(tableShape[0] > 0, "PLE table vocab dimension must be positive");
    int64_t const rowWidth = static_cast<int64_t>(mConfig.numPleInputs) * mConfig.pleHiddenSize;
    bool const isInt8Ple = tableDataType == nvinfer1::DataType::kINT8;
    bool const isInt4Ple = tableDataType == nvinfer1::DataType::kUINT8;
    ELLM_CHECK(tableShape[1] == (isInt4Ple ? rowWidth / 2 : rowWidth),
        "PLE table second dimension must equal num_ple_inputs * ple_hidden_size (halved for packed INT4)");
    mVocabSize = static_cast<int32_t>(tableShape[0]);
    if (mConfig.imageTokenId >= 0 || mConfig.audioTokenId >= 0)
    {
        ELLM_CHECK(mConfig.padTokenId >= 0 && mConfig.padTokenId < tableShape[0],
            "Gemma4 multimodal PLE pad_token_id is outside the PLE table vocabulary");
    }
    ELLM_CHECK(tableDataType == nvinfer1::DataType::kHALF || tableDataType == nvinfer1::DataType::kBF16 || isInt8Ple
            || isInt4Ple,
        "PLE table must be FP16, BF16, INT8, or packed INT4");
    nvinfer1::DataType const scaleDataType = isHostResident()
        ? (mHostTable->hasScales() ? mHostTable->scaleDataType() : nvinfer1::DataType::kFLOAT)
        : mPleScales.getDataType();
    if (isInt8Ple)
    {
        ELLM_CHECK(
            scaleShape.has_value() && scaleShape->getNumDims() == 2, "INT8 PLE scales must be 2D [vocab, num_layers]");
        ELLM_CHECK(scaleDataType == nvinfer1::DataType::kFLOAT, "INT8 PLE scales must be FP32");
        ELLM_CHECK((*scaleShape)[0] == tableShape[0], "INT8 PLE scale rows must match the table vocab size");
        ELLM_CHECK((*scaleShape)[1] == mConfig.numPleInputs, "INT8 PLE scale columns must match num_ple_inputs");
        mPleOutputDataType = nvinfer1::DataType::kHALF;
    }
    else if (isInt4Ple)
    {
        ELLM_CHECK(scaleShape.has_value() && scaleShape->getNumDims() == 2,
            "INT4 PLE scales must be 2D [vocab, num_layers * hidden / 32]");
        ELLM_CHECK(scaleDataType == nvinfer1::DataType::kHALF, "INT4 PLE scales must be FP16");
        ELLM_CHECK((*scaleShape)[0] == tableShape[0], "INT4 PLE scale rows must match the table vocab size");
        ELLM_CHECK(rowWidth % kInt4EmbeddingGroupSize == 0 && (*scaleShape)[1] == rowWidth / kInt4EmbeddingGroupSize,
            "INT4 PLE scale columns must match num_ple_inputs * ple_hidden_size / 32");
        mPleOutputDataType = nvinfer1::DataType::kHALF;
    }
    else
    {
        ELLM_CHECK(!scaleShape.has_value(), "PLE scales are only valid for an INT8 or INT4 table");
        mPleOutputDataType = tableDataType;
    }

    mPleOutputBuffer = Tensor({mConfig.numPleInputs, maxBatchSize, maxSeqLen, mConfig.pleHiddenSize}, DeviceType::kGPU,
        mPleOutputDataType, "Gemma4EmbeddingPreprocessor::mPleOutputBuffer");

    mPleOutputViews.reserve(mConfig.numPleInputs);
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews.emplace_back(makeTokenMajorOutputViewForLayer(idx, maxBatchSize * maxSeqLen));
        tensorMap.set(mPleOutputViews.back().getName(), mPleOutputViews.back());
    }

    LOG_INFO("Initialized Gemma4 PLE preprocessor: table=%s (%s) outputBuffer=%s numPleInputs=%d pleHiddenSize=%d",
        tableShape.formatString().c_str(), isHostResident() ? "host-mapped" : "device",
        mPleOutputBuffer.getShape().formatString().c_str(), mConfig.numPleInputs, mConfig.pleHiddenSize);
}

Tensor Gemma4EmbeddingPreprocessor::makeTokenMajorOutputViewForLayer(int32_t layerIdx, int64_t physicalTokens)
{
    ELLM_CHECK(layerIdx >= 0 && layerIdx < mConfig.numPleInputs, "Gemma4 PLE layer index out of range");
    auto const outputShape = mPleOutputBuffer.getShape();
    ELLM_CHECK(physicalTokens > 0, "Gemma4 PLE physical token count must be positive");
    ELLM_CHECK(
        physicalTokens <= outputShape[1] * outputShape[2], "Gemma4 PLE physical token count exceeds buffer capacity");

    int64_t const layerOutputCapacityBytes = outputShape[1] * outputShape[2] * mConfig.pleHiddenSize
        * static_cast<int64_t>(utils::getTypeSize(mPleOutputDataType));
    void* const layerOutputPtr
        = static_cast<void*>(static_cast<char*>(mPleOutputBuffer.rawPointer()) + layerIdx * layerOutputCapacityBytes);
    return Tensor(layerOutputPtr, Coords{physicalTokens, mConfig.pleHiddenSize}, DeviceType::kGPU, mPleOutputDataType,
        binding_names::formatPleTokenEmbedsName(layerIdx));
}

void Gemma4EmbeddingPreprocessor::reshapeOutputsTokenMajor(int64_t physicalTokens)
{
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews[idx] = makeTokenMajorOutputViewForLayer(idx, physicalTokens);
    }
}

void Gemma4EmbeddingPreprocessor::embed(Tensor const& tokenIds, int32_t const* hostTokenIds, cudaStream_t stream)
{
    auto const tokenShape = tokenIds.getShape();
    ELLM_CHECK(tokenShape.getNumDims() == 2, "Gemma4 PLE token IDs must be [batch, seq_len]");
    int64_t const physicalTokens = tokenShape[0] * tokenShape[1];
    if (mStaging)
    {
        ELLM_CHECK(hostTokenIds != nullptr, "A host-mapped PLE table needs the token ids in host memory");
        // Staged rows are indexed by position, so media placeholders must stage the PAD row here.
        int32_t const* stagedIds = hostTokenIds;
        if (mConfig.imageTokenId >= 0 || mConfig.audioTokenId >= 0)
        {
            for (int64_t i = 0; i < physicalTokens; ++i)
            {
                int32_t const id = hostTokenIds[i];
                bool const media = id == mConfig.imageTokenId || id == mConfig.audioTokenId;
                mStagedIds[i] = media ? mConfig.padTokenId : id;
            }
            stagedIds = mStagedIds.data();
        }
        mStaging->upload(stagedIds, physicalTokens, stream);
        kernel::gemma4PleGather(tokenIds, mStaging->rows(), mPleOutputBuffer, mConfig.numPleInputs,
            mConfig.pleHiddenSize, mConfig.imageTokenId, mConfig.audioTokenId, mConfig.padTokenId, stream,
            mStaging->scales(), mVocabSize);
    }
    else
    {
        OptionalInputTensor const scales
            = mPleScales.getShape().volume() > 0 ? OptionalInputTensor{mPleScales} : std::nullopt;
        kernel::gemma4PleGather(tokenIds, mPleTable, mPleOutputBuffer, mConfig.numPleInputs, mConfig.pleHiddenSize,
            mConfig.imageTokenId, mConfig.audioTokenId, mConfig.padTokenId, stream, scales);
    }
    reshapeOutputsTokenMajor(physicalTokens);
}

} // namespace rt
} // namespace trt_edgellm
