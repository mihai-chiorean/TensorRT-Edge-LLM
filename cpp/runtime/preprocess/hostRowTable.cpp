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

#include "runtime/preprocess/hostRowTable.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"

#include <cstring>
#include <limits>

namespace trt_edgellm
{
namespace rt
{

HostRowTable::HostRowTable(
    std::filesystem::path const& filePath, std::string_view tableName, std::string_view scaleName)
    : mFile(filePath)
    , mSource(filePath.string())
{
    safetensors::FileMetadata const metadata = safetensors::parseMetadata(mFile.getData(), mFile.getSize(), mSource);
    safetensors::TensorMetadata const* table = nullptr;
    safetensors::TensorMetadata const* scales = nullptr;
    for (auto const& entry : metadata.tensors)
    {
        if (entry.name == tableName)
        {
            ELLM_CHECK(table == nullptr, mSource + " contains duplicate " + std::string(tableName) + " tensors");
            table = &entry;
        }
        else if (entry.name == scaleName)
        {
            ELLM_CHECK(scales == nullptr, mSource + " contains duplicate " + std::string(scaleName) + " tensors");
            scales = &entry;
        }
    }
    ELLM_CHECK(table != nullptr, mSource + " must contain a tensor named " + std::string(tableName));
    ELLM_CHECK(table->shape.getNumDims() == 2, std::string(tableName) + " must be 2D [rows, columns]: " + mSource);
    ELLM_CHECK(table->shape[0] > 0 && table->shape[1] > 0, std::string(tableName) + " must be non-empty: " + mSource);
    ELLM_CHECK(table->shape[0] <= std::numeric_limits<int32_t>::max(),
        std::string(tableName) + " row count exceeds the int32 token id range: " + mSource);

    mDataType = table->dataType;
    mShape = table->shape;
    mRowBytes = mShape[1] * static_cast<int64_t>(utils::getTypeSize(mDataType));
    ELLM_CHECK(static_cast<size_t>(mRowBytes) * mShape[0] == table->bytes,
        std::string(tableName) + " byte count does not match its shape: " + mSource);
    mTable = static_cast<uint8_t const*>(mFile.getData()) + metadata.dataOffset + table->offset;

    if (scales != nullptr)
    {
        ELLM_CHECK(
            scales->dataType == nvinfer1::DataType::kFLOAT, std::string(scaleName) + " must be FP32: " + mSource);
        ELLM_CHECK(scales->shape.getNumDims() == 1 || scales->shape.getNumDims() == 2,
            std::string(scaleName) + " must be [rows] or [rows, scalesPerRow]: " + mSource);
        ELLM_CHECK(scales->shape[0] == mShape[0],
            std::string(scaleName) + " rows must match " + std::string(tableName) + " rows: " + mSource);
        mScaleShape = scales->shape;
        mScalesPerRow = scales->shape.getNumDims() == 2 ? scales->shape[1] : 1;
        ELLM_CHECK(mScalesPerRow > 0, std::string(scaleName) + " must have a positive column count: " + mSource);
        ELLM_CHECK(static_cast<size_t>(mScalesPerRow) * mShape[0] * sizeof(float) == scales->bytes,
            std::string(scaleName) + " byte count does not match its shape: " + mSource);
        // Kept as bytes: a safetensors header need not pad the payload to 4-byte alignment.
        mScales = static_cast<uint8_t const*>(mFile.getData()) + metadata.dataOffset + scales->offset;
    }

    if (!mFile.adviseRandomAccess())
    {
        LOG_WARNING(
            "madvise(MADV_RANDOM) failed for %s; page faults will read around the touched rows", mSource.c_str());
    }
    LOG_INFO("Mapped %s from %s: shape=%s rowBytes=%ld scalesPerRow=%ld", std::string(tableName).c_str(),
        mSource.c_str(), mShape.formatString().c_str(), mRowBytes, mScalesPerRow);
}

void HostRowTable::gather(int32_t const* ids, int64_t count, void* rows, float* scales) const
{
    ELLM_CHECK(ids != nullptr && rows != nullptr, "HostRowTable::gather requires ids and a row destination");
    ELLM_CHECK(count >= 0, "HostRowTable::gather requires a non-negative row count");
    auto* dstRows = static_cast<uint8_t*>(rows);
    bool const copyScales = mScales != nullptr && scales != nullptr;
    size_t const scaleBytes = static_cast<size_t>(mScalesPerRow) * sizeof(float);
    for (int64_t i = 0; i < count; ++i)
    {
        int32_t const id = ids[i];
        uint8_t* dstRow = dstRows + i * mRowBytes;
        float* dstScales = copyScales ? scales + i * mScalesPerRow : nullptr;
        if (id < 0 || id >= mShape[0])
        {
            std::memset(dstRow, 0, static_cast<size_t>(mRowBytes));
            if (copyScales)
            {
                std::memset(dstScales, 0, scaleBytes);
            }
            continue;
        }
        std::memcpy(dstRow, mTable + static_cast<int64_t>(id) * mRowBytes, static_cast<size_t>(mRowBytes));
        if (copyScales)
        {
            std::memcpy(dstScales, mScales + static_cast<int64_t>(id) * scaleBytes, scaleBytes);
        }
    }
}

RowStaging::RowStaging(HostRowTable const& table, int64_t capacity, std::string const& name)
    : mTable(table)
    , mCapacity(capacity)
{
    ELLM_CHECK(capacity > 0, "RowStaging requires a positive row capacity: " + name);
    Coords const rowShape{capacity, table.shape()[1]};
    mHostRows = Tensor(rowShape, DeviceType::kCPU, table.dataType(), name + "::hostRows");
    mDeviceRows = Tensor(rowShape, DeviceType::kGPU, table.dataType(), name + "::deviceRows");
    if (table.hasScales())
    {
        Coords const scaleShape
            = table.scaleShape().getNumDims() == 2 ? Coords{capacity, table.scalesPerRow()} : Coords{capacity};
        mHostScales = Tensor(scaleShape, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, name + "::hostScales");
        mDeviceScales = Tensor(scaleShape, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, name + "::deviceScales");
    }
}

void RowStaging::upload(int32_t const* ids, int64_t count, cudaStream_t stream)
{
    ELLM_CHECK(count > 0 && count <= mCapacity,
        "RowStaging::upload row count " + std::to_string(count) + " exceeds capacity " + std::to_string(mCapacity));
    mFence.wait();

    Coords const rowShape{count, mTable.shape()[1]};
    ELLM_CHECK(mHostRows.reshape(rowShape) && mDeviceRows.reshape(rowShape), "RowStaging row reshape failed");
    float* hostScales = nullptr;
    if (mTable.hasScales())
    {
        Coords const scaleShape
            = mTable.scaleShape().getNumDims() == 2 ? Coords{count, mTable.scalesPerRow()} : Coords{count};
        ELLM_CHECK(
            mHostScales.reshape(scaleShape) && mDeviceScales.reshape(scaleShape), "RowStaging scale reshape failed");
        hostScales = mHostScales.dataPointer<float>();
    }

    mTable.gather(ids, count, mHostRows.rawPointer(), hostScales);
    CUDA_CHECK(cudaMemcpyAsync(mDeviceRows.rawPointer(), mHostRows.rawPointer(),
        static_cast<size_t>(count) * mTable.rowBytes(), cudaMemcpyHostToDevice, stream));
    if (hostScales != nullptr)
    {
        CUDA_CHECK(cudaMemcpyAsync(mDeviceScales.rawPointer(), mHostScales.rawPointer(),
            static_cast<size_t>(count) * mTable.scalesPerRow() * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    mFence.record(stream);
}

} // namespace rt
} // namespace trt_edgellm
