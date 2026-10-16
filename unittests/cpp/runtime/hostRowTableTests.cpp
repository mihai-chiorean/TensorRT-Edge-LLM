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

#include "common/cudaUtils.h"
#include "common/safetensorsUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/preprocess/hostRowTable.h"
#include "testUtils.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

constexpr int32_t kRows = 5;
constexpr int32_t kColumns = 24;
constexpr int32_t kScalesPerRow = 3;

struct SidecarFixture
{
    std::filesystem::path dir;
    std::filesystem::path file;
    std::vector<int8_t> table;
    std::vector<float> scales;

    explicit SidecarFixture(std::string const& name, bool withScales = true)
        : dir(std::filesystem::temp_directory_path() / name)
        , file(dir / "embedding.safetensors")
        , table(static_cast<size_t>(kRows) * kColumns)
        , scales(static_cast<size_t>(kRows) * kScalesPerRow)
    {
        for (size_t idx = 0; idx < table.size(); ++idx)
        {
            table[idx] = static_cast<int8_t>(static_cast<int32_t>((idx * 11) % 251) - 125);
        }
        for (size_t idx = 0; idx < scales.size(); ++idx)
        {
            scales[idx] = 0.25F * static_cast<float>(idx + 1);
        }
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        std::vector<rt::Tensor> tensors;
        tensors.emplace_back(rt::Coords{kRows, kColumns}, rt::DeviceType::kCPU, DataType::kINT8, "embedding");
        std::memcpy(tensors.back().rawPointer(), table.data(), table.size());
        if (withScales)
        {
            tensors.emplace_back(
                rt::Coords{kRows, kScalesPerRow}, rt::DeviceType::kCPU, DataType::kFLOAT, "embedding_scale");
            std::memcpy(tensors.back().rawPointer(), scales.data(), scales.size() * sizeof(float));
        }
        if (!rt::safetensors::saveSafetensors(file, tensors, nullptr))
        {
            throw std::runtime_error("saveSafetensors failed");
        }
    }

    ~SidecarFixture()
    {
        std::filesystem::remove_all(dir);
    }
};

} // namespace

TEST(HostRowTableTest, MapsTableAndGathersRowsWithScales)
{
    SidecarFixture const fixture("trt_edgellm_host_row_table_test");
    rt::HostRowTable const hostTable(fixture.file, "embedding", "embedding_scale");
    EXPECT_EQ(hostTable.dataType(), DataType::kINT8);
    EXPECT_EQ(hostTable.rows(), kRows);
    EXPECT_EQ(hostTable.shape()[1], kColumns);
    EXPECT_EQ(hostTable.rowBytes(), kColumns);
    ASSERT_TRUE(hostTable.hasScales());
    EXPECT_EQ(hostTable.scalesPerRow(), kScalesPerRow);
    EXPECT_EQ(hostTable.scaleShape().getNumDims(), 2);

    std::vector<int32_t> const ids{4, -1, 0, kRows, 2};
    std::vector<int8_t> rows(ids.size() * kColumns, 99);
    std::vector<float> scales(ids.size() * kScalesPerRow, 99.0F);
    hostTable.gather(ids.data(), static_cast<int64_t>(ids.size()), rows.data(), scales.data());

    for (size_t pos = 0; pos < ids.size(); ++pos)
    {
        int32_t const id = ids[pos];
        for (int32_t col = 0; col < kColumns; ++col)
        {
            int8_t const expected
                = (id >= 0 && id < kRows) ? fixture.table[static_cast<size_t>(id) * kColumns + col] : 0;
            EXPECT_EQ(rows[pos * kColumns + col], expected) << "pos " << pos << " col " << col;
        }
        for (int32_t s = 0; s < kScalesPerRow; ++s)
        {
            float const expected
                = (id >= 0 && id < kRows) ? fixture.scales[static_cast<size_t>(id) * kScalesPerRow + s] : 0.0F;
            EXPECT_EQ(scales[pos * kScalesPerRow + s], expected) << "pos " << pos << " scale " << s;
        }
    }
}

TEST(HostRowTableTest, TableWithoutScalesReportsNone)
{
    SidecarFixture const fixture("trt_edgellm_host_row_table_noscale_test", /*withScales=*/false);
    rt::HostRowTable const hostTable(fixture.file, "embedding", "embedding_scale");
    EXPECT_FALSE(hostTable.hasScales());
    EXPECT_EQ(hostTable.scalesPerRow(), 0);
    std::vector<int32_t> const ids{1};
    std::vector<int8_t> rows(kColumns);
    hostTable.gather(ids.data(), 1, rows.data(), nullptr);
    EXPECT_EQ(std::memcmp(rows.data(), fixture.table.data() + kColumns, kColumns), 0);
}

TEST(HostRowTableTest, RejectsMissingTensor)
{
    SidecarFixture const fixture("trt_edgellm_host_row_table_missing_test");
    EXPECT_THROW(rt::HostRowTable(fixture.file, "weight", "weight_scale"), std::runtime_error);
}

TEST(HostRowTableTest, RowStagingUploadsGatheredRows)
{
    SidecarFixture const fixture("trt_edgellm_row_staging_test");
    rt::HostRowTable const hostTable(fixture.file, "embedding", "embedding_scale");
    rt::RowStaging staging(hostTable, /*capacity=*/4, "test");
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<int32_t> const ids{3, kRows + 7, 1};
    staging.upload(ids.data(), static_cast<int64_t>(ids.size()), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(staging.rows().getShape()[0], static_cast<int64_t>(ids.size()));
    EXPECT_EQ(staging.rows().getShape()[1], kColumns);
    ASSERT_TRUE(staging.scales().has_value());
    EXPECT_EQ(staging.scales()->get().getShape()[0], static_cast<int64_t>(ids.size()));
    EXPECT_EQ(staging.scales()->get().getShape()[1], kScalesPerRow);

    auto const rows = copyDeviceToHost<int8_t>(staging.rows());
    auto const scales = copyDeviceToHost<float>(staging.scales()->get());
    EXPECT_EQ(std::memcmp(rows.data(), fixture.table.data() + 3 * kColumns, kColumns), 0);
    EXPECT_EQ(std::memcmp(rows.data() + 2 * kColumns, fixture.table.data() + kColumns, kColumns), 0);
    for (int32_t col = 0; col < kColumns; ++col)
    {
        EXPECT_EQ(rows[kColumns + col], 0);
    }
    EXPECT_EQ(scales[0], fixture.scales[3 * kScalesPerRow]);
    EXPECT_EQ(scales[kScalesPerRow], 0.0F);
    EXPECT_EQ(scales[2 * kScalesPerRow + 1], fixture.scales[kScalesPerRow + 1]);

    // A second upload reuses the pinned buffers and reshapes to the new count.
    std::vector<int32_t> const moreIds{0, 4, 2, 1};
    staging.upload(moreIds.data(), static_cast<int64_t>(moreIds.size()), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(staging.rows().getShape()[0], 4);
    auto const moreRows = copyDeviceToHost<int8_t>(staging.rows());
    EXPECT_EQ(std::memcmp(moreRows.data() + kColumns, fixture.table.data() + 4 * kColumns, kColumns), 0);

    EXPECT_THROW(staging.upload(moreIds.data(), 5, stream), std::runtime_error);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(HostRowTableTest, MapEmbeddingTableValidatesContract)
{
    SidecarFixture const fixture("trt_edgellm_map_embedding_table_test");
    // The embedding contract wants one scale per row for INT8; a [rows, 3] scale tensor must be rejected.
    EXPECT_THROW(rt::mapEmbeddingTable(fixture.file), std::runtime_error);

    auto const dir = std::filesystem::temp_directory_path() / "trt_edgellm_map_embedding_table_ok_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::vector<rt::Tensor> tensors;
    tensors.emplace_back(rt::Coords{kRows, kColumns}, rt::DeviceType::kCPU, DataType::kINT8, "embedding");
    std::memcpy(tensors.back().rawPointer(), fixture.table.data(), fixture.table.size());
    tensors.emplace_back(rt::Coords{kRows}, rt::DeviceType::kCPU, DataType::kFLOAT, "embedding_scale");
    std::memcpy(tensors.back().rawPointer(), fixture.scales.data(), kRows * sizeof(float));
    ASSERT_TRUE(rt::safetensors::saveSafetensors(dir / "embedding.safetensors", tensors, nullptr));

    rt::EmbeddingData const embedding = rt::mapEmbeddingTable(dir / "embedding.safetensors");
    ASSERT_TRUE(embedding.isHostResident());
    EXPECT_EQ(embedding.table.getShape().volume(), 0);
    EXPECT_FALSE(embedding.scalesAsOptional().has_value());
    EXPECT_EQ(embedding.hostTable->rows(), kRows);
    EXPECT_EQ(embedding.hostTable->scalesPerRow(), 1);
    std::filesystem::remove_all(dir);
}
