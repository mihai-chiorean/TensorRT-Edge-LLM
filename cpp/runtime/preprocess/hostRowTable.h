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

#include "common/mmapReader.h"
#include "common/tensor.h"
#include "runtime/state/asyncHostStagingFence.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace trt_edgellm
{
namespace rt
{

//! Where the token-embedding and PLE sidecar tables live at runtime.
enum class EmbeddingTableResidency
{
    kDevice,     //!< Tables are copied into device memory at load.
    kHostMapped, //!< Tables stay mapped from their files; each step gathers its rows on the CPU and uploads them.
};

//! A 2D table read in place from a read-only mapping of its safetensors file, with an optional FP32 scale tensor
//! that has one row per table row. Rows are gathered on the CPU, so the table's pages stay file-backed and
//! reclaimable instead of occupying device memory.
class HostRowTable
{
public:
    //! Maps `filePath` and resolves `tableName` and, when present in the file, `scaleName`.
    //! The mapping is advised as random access: rows are gathered by token id, so read-around on a page fault
    //! would only pull unused neighbours into memory.
    HostRowTable(std::filesystem::path const& filePath, std::string_view tableName, std::string_view scaleName);

    HostRowTable(HostRowTable const&) = delete;
    HostRowTable& operator=(HostRowTable const&) = delete;

    nvinfer1::DataType dataType() const noexcept
    {
        return mDataType;
    }

    //! [rows, columns]
    Coords const& shape() const noexcept
    {
        return mShape;
    }

    int64_t rows() const noexcept
    {
        return mShape[0];
    }

    int64_t rowBytes() const noexcept
    {
        return mRowBytes;
    }

    bool hasScales() const noexcept
    {
        return mScales != nullptr;
    }

    //! Shape of the scale tensor as stored: [rows] or [rows, scalesPerRow]. Empty without scales.
    Coords const& scaleShape() const noexcept
    {
        return mScaleShape;
    }

    //! FP32 scales per table row; 0 without scales.
    int64_t scalesPerRow() const noexcept
    {
        return mScalesPerRow;
    }

    //! Copies table row ids[i] into row i of `rows` and its scales into `scales` (when the table has scales and
    //! `scales` is non-null). Ids outside [0, rows()) produce zero rows and zero scales.
    void gather(int32_t const* ids, int64_t count, void* rows, float* scales) const;

private:
    file_io::MmapReader mFile;
    std::string mSource;
    nvinfer1::DataType mDataType{};
    Coords mShape;
    int64_t mRowBytes{0};
    uint8_t const* mTable{nullptr};
    uint8_t const* mScales{nullptr}; //!< FP32 payload addressed as bytes; the file need not align it.
    Coords mScaleShape;
    int64_t mScalesPerRow{0};
};

//! Pinned and device staging for rows gathered from a HostRowTable, sized for `capacity` rows.
//!
//! `upload` reuses the pinned buffer, so it first waits for the previous upload's copy to finish reading it,
//! then gathers on the CPU and enqueues the host-to-device copies on the caller's stream. The device tensors it
//! exposes are shaped to the uploaded row count and are consumed by the staged-rows mode of the lookup kernels.
class RowStaging
{
public:
    RowStaging(HostRowTable const& table, int64_t capacity, std::string const& name);

    RowStaging(RowStaging const&) = delete;
    RowStaging& operator=(RowStaging const&) = delete;

    int64_t capacity() const noexcept
    {
        return mCapacity;
    }

    //! Gathers `count` rows for `ids` and copies them to the device on `stream`.
    void upload(int32_t const* ids, int64_t count, cudaStream_t stream);

    //! Device rows of the last upload, shaped [count, columns].
    Tensor const& rows() const noexcept
    {
        return mDeviceRows;
    }

    //! Device scales of the last upload, shaped [count] or [count, scalesPerRow]; nullopt for a table without scales.
    OptionalInputTensor scales() const noexcept
    {
        return mTable.hasScales() ? OptionalInputTensor{mDeviceScales} : std::nullopt;
    }

private:
    HostRowTable const& mTable;
    int64_t mCapacity{0};
    Tensor mHostRows;
    Tensor mDeviceRows;
    Tensor mHostScales;
    Tensor mDeviceScales;
    AsyncHostStagingFence mFence;
};

} // namespace rt
} // namespace trt_edgellm
