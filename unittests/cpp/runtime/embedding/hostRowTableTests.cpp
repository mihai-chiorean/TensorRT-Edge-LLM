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

#include "common/checkMacros.h"
#include "common/safetensorsUtils.h"
#include "runtime/embedding/hostRowTable.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/state/pipelineIO.h"
#include "scratchDir.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kRows = 5;
constexpr int32_t kColumns = 24;
constexpr int32_t kScalesPerRow = 3;

std::vector<int8_t> int8Table(int64_t rows, int64_t columns)
{
    std::vector<int8_t> table(static_cast<size_t>(rows * columns));
    for (size_t idx = 0; idx < table.size(); ++idx)
    {
        table[idx] = static_cast<int8_t>(static_cast<int32_t>((idx * 11) % 251) - 125);
    }
    return table;
}

std::vector<float> fp32Scales(size_t count)
{
    std::vector<float> scales(count);
    for (size_t idx = 0; idx < count; ++idx)
    {
        scales[idx] = 0.25F * static_cast<float>(idx + 1);
    }
    return scales;
}

template <typename T>
rt::Tensor hostTensor(
    rt::Coords const& shape, trt_edgellm::DataType dataType, std::string const& name, std::vector<T> const& values)
{
    rt::Tensor tensor(shape, rt::DeviceType::kCPU, dataType, name);
    EXPECT_EQ(static_cast<size_t>(tensor.getShape().volume()), values.size());
    std::memcpy(tensor.rawPointer(), values.data(), values.size() * sizeof(T));
    return tensor;
}

void writeSidecar(std::filesystem::path const& file, std::vector<rt::Tensor> const& tensors,
    std::unordered_map<std::string, std::string> const& metadata = {})
{
    ASSERT_TRUE(rt::safetensors::saveSafetensors(file, tensors, nullptr, metadata));
}

std::string thrownMessage(std::function<void()> const& fn)
{
    try
    {
        fn();
    }
    catch (std::exception const& error)
    {
        return error.what();
    }
    return {};
}

class HostRowTableTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDir = makeScratchDir("hostRowTableTests");
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mDir);
    }

    std::filesystem::path mDir;
    cudaStream_t mStream{};
};

} // namespace

TEST_F(HostRowTableTest, MapsTableAndGathersRowsWithScales)
{
    auto const table = int8Table(kRows, kColumns);
    auto const scales = fp32Scales(static_cast<size_t>(kRows) * kScalesPerRow);
    auto const file = mDir / "embedding.safetensors";
    std::vector<rt::Tensor> tensors;
    tensors.push_back(hostTensor(rt::Coords{kRows, kColumns}, trt_edgellm::DataType::kINT8, "embedding", table));
    tensors.push_back(
        hostTensor(rt::Coords{kRows, kScalesPerRow}, trt_edgellm::DataType::kFLOAT, "embedding_scale", scales));
    writeSidecar(file, tensors);

    rt::HostRowTable const hostTable(file, "embedding", "embedding_scale");
    EXPECT_EQ(hostTable.dataType(), trt_edgellm::DataType::kINT8);
    EXPECT_EQ(hostTable.rows(), kRows);
    EXPECT_EQ(hostTable.shape()[1], kColumns);
    EXPECT_EQ(hostTable.rowBytes(), kColumns);
    ASSERT_TRUE(hostTable.hasScales());
    EXPECT_EQ(hostTable.scaleDataType(), trt_edgellm::DataType::kFLOAT);
    EXPECT_EQ(hostTable.scalesPerRow(), kScalesPerRow);
    EXPECT_EQ(hostTable.scaleRowBytes(), kScalesPerRow * static_cast<int64_t>(sizeof(float)));

    std::vector<int32_t> const ids{4, -1, 0, kRows, 2};
    std::vector<int8_t> rows(ids.size() * kColumns, 99);
    std::vector<float> gatheredScales(ids.size() * kScalesPerRow, 99.0F);
    hostTable.gather(ids.data(), static_cast<int64_t>(ids.size()), rows.data(), gatheredScales.data());

    for (size_t pos = 0; pos < ids.size(); ++pos)
    {
        int32_t const id = ids[pos];
        bool const valid = id >= 0 && id < kRows;
        for (int32_t col = 0; col < kColumns; ++col)
        {
            int8_t const expected = valid ? table[static_cast<size_t>(id) * kColumns + col] : 0;
            EXPECT_EQ(rows[pos * kColumns + col], expected) << "pos " << pos << " col " << col;
        }
        for (int32_t s = 0; s < kScalesPerRow; ++s)
        {
            float const expected = valid ? scales[static_cast<size_t>(id) * kScalesPerRow + s] : 0.0F;
            EXPECT_EQ(gatheredScales[pos * kScalesPerRow + s], expected) << "pos " << pos << " scale " << s;
        }
    }
}

TEST_F(HostRowTableTest, TableWithoutScalesReportsNone)
{
    auto const table = int8Table(kRows, kColumns);
    auto const file = mDir / "embedding.safetensors";
    std::vector<rt::Tensor> tensors;
    tensors.push_back(hostTensor(rt::Coords{kRows, kColumns}, trt_edgellm::DataType::kINT8, "embedding", table));
    writeSidecar(file, tensors);

    rt::HostRowTable const hostTable(file, "embedding", "embedding_scale");
    EXPECT_FALSE(hostTable.hasScales());
    EXPECT_EQ(hostTable.scalesPerRow(), 0);
    std::vector<int32_t> const ids{1};
    std::vector<int8_t> rows(kColumns);
    hostTable.gather(ids.data(), 1, rows.data(), nullptr);
    EXPECT_EQ(std::memcmp(rows.data(), table.data() + kColumns, kColumns), 0);
}

TEST_F(HostRowTableTest, RejectsMissingTensor)
{
    auto const file = mDir / "embedding.safetensors";
    std::vector<rt::Tensor> tensors;
    tensors.push_back(
        hostTensor(rt::Coords{kRows, kColumns}, trt_edgellm::DataType::kINT8, "embedding", int8Table(kRows, kColumns)));
    writeSidecar(file, tensors);
    EXPECT_THROW(rt::HostRowTable(file, "weight", "weight_scale"), std::runtime_error);
}

// The packed INT4 sidecar stores two columns per byte and one FP16 scale per 32 columns.
TEST_F(HostRowTableTest, GathersPackedInt4RowsWithFp16Scales)
{
    constexpr int32_t kPackedColumns = 32;
    constexpr int32_t kGroups = 2;
    std::vector<uint8_t> packed(static_cast<size_t>(kRows) * kPackedColumns);
    for (size_t idx = 0; idx < packed.size(); ++idx)
    {
        packed[idx] = static_cast<uint8_t>((idx * 37) % 256);
    }
    std::vector<half> scales(static_cast<size_t>(kRows) * kGroups);
    for (size_t idx = 0; idx < scales.size(); ++idx)
    {
        scales[idx] = __float2half(0.125F * static_cast<float>(idx + 1));
    }
    auto const file = mDir / "embedding.safetensors";
    std::vector<rt::Tensor> tensors;
    tensors.push_back(
        hostTensor(rt::Coords{kRows, kPackedColumns}, trt_edgellm::DataType::kUINT8, "embedding", packed));
    tensors.push_back(hostTensor(rt::Coords{kRows, kGroups}, trt_edgellm::DataType::kHALF, "embedding_scale", scales));
    writeSidecar(file, tensors);

    rt::HostRowTable const hostTable(file, "embedding", "embedding_scale");
    EXPECT_EQ(hostTable.scaleDataType(), trt_edgellm::DataType::kHALF);
    EXPECT_EQ(hostTable.scaleRowBytes(), kGroups * static_cast<int64_t>(sizeof(half)));
    std::vector<int32_t> const ids{3, 9};
    std::vector<uint8_t> rows(ids.size() * kPackedColumns, 1);
    std::vector<half> gathered(ids.size() * kGroups, __float2half(7.0F));
    hostTable.gather(ids.data(), static_cast<int64_t>(ids.size()), rows.data(), gathered.data());
    EXPECT_EQ(std::memcmp(rows.data(), packed.data() + 3 * kPackedColumns, kPackedColumns), 0);
    EXPECT_EQ(std::memcmp(gathered.data(), scales.data() + 3 * kGroups, kGroups * sizeof(half)), 0);
    for (int32_t col = 0; col < kPackedColumns; ++col)
    {
        EXPECT_EQ(rows[kPackedColumns + col], 0);
    }
    EXPECT_EQ(__half2float(gathered[kGroups]), 0.0F);
}

TEST_F(HostRowTableTest, RowStagingUploadsGatheredRows)
{
    auto const table = int8Table(kRows, kColumns);
    auto const scales = fp32Scales(static_cast<size_t>(kRows) * kScalesPerRow);
    auto const file = mDir / "embedding.safetensors";
    std::vector<rt::Tensor> tensors;
    tensors.push_back(hostTensor(rt::Coords{kRows, kColumns}, trt_edgellm::DataType::kINT8, "embedding", table));
    tensors.push_back(
        hostTensor(rt::Coords{kRows, kScalesPerRow}, trt_edgellm::DataType::kFLOAT, "embedding_scale", scales));
    writeSidecar(file, tensors);
    rt::HostRowTable const hostTable(file, "embedding", "embedding_scale");
    rt::RowStaging staging(hostTable, /*capacity=*/4, "test");

    std::vector<int32_t> const ids{3, kRows + 7, 1};
    staging.upload(ids.data(), static_cast<int64_t>(ids.size()), mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_EQ(staging.rows().getShape()[0], static_cast<int64_t>(ids.size()));
    EXPECT_EQ(staging.rows().getShape()[1], kColumns);
    ASSERT_TRUE(staging.scales().has_value());
    EXPECT_EQ(staging.scales()->get().getShape()[0], static_cast<int64_t>(ids.size()));
    EXPECT_EQ(staging.scales()->get().getShape()[1], kScalesPerRow);

    auto const rows = copyDeviceToHost<int8_t>(staging.rows());
    auto const stagedScales = copyDeviceToHost<float>(staging.scales()->get());
    EXPECT_EQ(std::memcmp(rows.data(), table.data() + 3 * kColumns, kColumns), 0);
    EXPECT_EQ(std::memcmp(rows.data() + 2 * kColumns, table.data() + kColumns, kColumns), 0);
    for (int32_t col = 0; col < kColumns; ++col)
    {
        EXPECT_EQ(rows[kColumns + col], 0);
    }
    EXPECT_EQ(stagedScales[0], scales[3 * kScalesPerRow]);
    EXPECT_EQ(stagedScales[kScalesPerRow], 0.0F);
    EXPECT_EQ(stagedScales[2 * kScalesPerRow + 1], scales[kScalesPerRow + 1]);

    // A second upload reuses the pinned buffers and reshapes to the new count.
    std::vector<int32_t> const moreIds{0, 4, 2, 1};
    staging.upload(moreIds.data(), static_cast<int64_t>(moreIds.size()), mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_EQ(staging.rows().getShape()[0], 4);
    auto const moreRows = copyDeviceToHost<int8_t>(staging.rows());
    EXPECT_EQ(std::memcmp(moreRows.data() + kColumns, table.data() + 4 * kColumns, kColumns), 0);

    EXPECT_THROW(staging.upload(moreIds.data(), 5, mStream), std::runtime_error);
}

TEST_F(HostRowTableTest, MapEmbeddingTableValidatesContract)
{
    auto const table = int8Table(kRows, kColumns);
    auto const scales = fp32Scales(static_cast<size_t>(kRows) * kScalesPerRow);

    // INT8 wants one FP32 scale per row; a [rows, 3] scale tensor is refused.
    auto const wrongScales = mDir / "wrong.safetensors";
    {
        std::vector<rt::Tensor> tensors;
        tensors.push_back(hostTensor(rt::Coords{kRows, kColumns}, trt_edgellm::DataType::kINT8, "embedding", table));
        tensors.push_back(
            hostTensor(rt::Coords{kRows, kScalesPerRow}, trt_edgellm::DataType::kFLOAT, "embedding_scale", scales));
        writeSidecar(wrongScales, tensors);
    }
    EXPECT_THROW(rt::mapEmbeddingTable(wrongScales), std::runtime_error);

    auto const int8File = mDir / "int8.safetensors";
    {
        std::vector<rt::Tensor> tensors;
        tensors.push_back(hostTensor(rt::Coords{kRows, kColumns}, trt_edgellm::DataType::kINT8, "embedding", table));
        tensors.push_back(hostTensor(rt::Coords{kRows}, trt_edgellm::DataType::kFLOAT, "embedding_scale",
            std::vector<float>(scales.begin(), scales.begin() + kRows)));
        writeSidecar(int8File, tensors);
    }
    rt::EmbeddingData const embedding = rt::mapEmbeddingTable(int8File);
    ASSERT_TRUE(embedding.isHostMapped());
    EXPECT_EQ(embedding.table.getShape().volume(), 0);
    EXPECT_FALSE(embedding.scalesAsOptional().has_value());
    EXPECT_EQ(embedding.hostTable->rows(), kRows);
    EXPECT_EQ(embedding.hostTable->scalesPerRow(), 1);

    auto const int4File = mDir / "int4.safetensors";
    {
        std::vector<rt::Tensor> tensors;
        tensors.push_back(hostTensor(rt::Coords{kRows, 16}, trt_edgellm::DataType::kUINT8, "embedding",
            std::vector<uint8_t>(static_cast<size_t>(kRows) * 16, 0x88)));
        tensors.push_back(hostTensor(rt::Coords{kRows, 1}, trt_edgellm::DataType::kHALF, "embedding_scale",
            std::vector<half>(kRows, __float2half(1.0F))));
        writeSidecar(int4File, tensors);
    }
    EXPECT_TRUE(rt::mapEmbeddingTable(int4File).isHostMapped());

    auto const fp8File = mDir / "fp8.safetensors";
    {
        std::vector<rt::Tensor> tensors;
        tensors.push_back(hostTensor(rt::Coords{kRows, 128}, trt_edgellm::DataType::kFP8, "embedding",
            std::vector<uint8_t>(static_cast<size_t>(kRows) * 128, 0)));
        tensors.push_back(
            hostTensor(rt::Coords{1, 1}, trt_edgellm::DataType::kFLOAT, "embedding_scale", std::vector<float>{1.0F}));
        writeSidecar(fp8File, tensors);
    }
    EXPECT_NE(
        thrownMessage([&]() { rt::mapEmbeddingTable(fp8File); }).find("FP16, INT8, or packed INT4"), std::string::npos);

    auto const nvfp4File = mDir / "nvfp4.safetensors";
    {
        std::vector<rt::Tensor> tensors;
        tensors.push_back(hostTensor(rt::Coords{kRows, 16}, trt_edgellm::DataType::kUINT8, "embedding",
            std::vector<uint8_t>(static_cast<size_t>(kRows) * 16, 0)));
        writeSidecar(nvfp4File, tensors, {{rt::kEmbeddingDescriptorMetadataKey, "{}"}});
    }
    EXPECT_NE(thrownMessage([&]() { rt::mapEmbeddingTable(nvfp4File); }).find("NVFP4"), std::string::npos);
}

namespace
{

struct SidecarCase
{
    char const* name;
    trt_edgellm::DataType dataType;
};

class HostMappedEmbeddingPreprocessorTest : public HostRowTableTest, public ::testing::WithParamInterface<SidecarCase>
{
protected:
    static constexpr int32_t kVocab{12};
    static constexpr int32_t kHidden{64};
    static constexpr int32_t kImageTokenId{7};

    //! Writes an embedding.safetensors of the parameter's format with distinct rows.
    std::filesystem::path writeTable() const
    {
        auto const file = mDir / "embedding.safetensors";
        std::vector<rt::Tensor> tensors;
        switch (GetParam().dataType)
        {
        case trt_edgellm::DataType::kHALF:
        {
            std::vector<half> values(static_cast<size_t>(kVocab) * kHidden);
            for (size_t idx = 0; idx < values.size(); ++idx)
            {
                values[idx] = __float2half(0.01F * static_cast<float>(static_cast<int32_t>(idx % 97) - 48));
            }
            tensors.push_back(
                hostTensor(rt::Coords{kVocab, kHidden}, trt_edgellm::DataType::kHALF, "embedding", values));
            break;
        }
        case trt_edgellm::DataType::kINT8:
            tensors.push_back(hostTensor(
                rt::Coords{kVocab, kHidden}, trt_edgellm::DataType::kINT8, "embedding", int8Table(kVocab, kHidden)));
            tensors.push_back(
                hostTensor(rt::Coords{kVocab}, trt_edgellm::DataType::kFLOAT, "embedding_scale", fp32Scales(kVocab)));
            break;
        default:
        {
            std::vector<uint8_t> packed(static_cast<size_t>(kVocab) * kHidden / 2);
            for (size_t idx = 0; idx < packed.size(); ++idx)
            {
                packed[idx] = static_cast<uint8_t>((idx * 29) % 256);
            }
            std::vector<half> scales(static_cast<size_t>(kVocab) * kHidden / 32);
            for (size_t idx = 0; idx < scales.size(); ++idx)
            {
                scales[idx] = __float2half(0.03125F * static_cast<float>(idx + 1));
            }
            tensors.push_back(
                hostTensor(rt::Coords{kVocab, kHidden / 2}, trt_edgellm::DataType::kUINT8, "embedding", packed));
            tensors.push_back(
                hostTensor(rt::Coords{kVocab, kHidden / 32}, trt_edgellm::DataType::kHALF, "embedding_scale", scales));
            break;
        }
        }
        writeSidecar(file, tensors);
        return file;
    }
};

} // namespace

// Given the same sidecar loaded to the device and mapped in host memory
// When a prefill with an image placeholder, padding and an out-of-range id is embedded through each
// Then the host-mapped path gives bit-identical embeddings, and it refuses to run without host token ids
TEST_P(HostMappedEmbeddingPreprocessorTest, MatchesTheDeviceTableBitForBit)
{
    auto const file = writeTable();
    rt::EmbeddingData const deviceEmbedding = rt::loadEmbeddingTable(file, mStream);
    rt::EmbeddingData const hostEmbedding = rt::mapEmbeddingTable(file);
    ASSERT_TRUE(hostEmbedding.isHostMapped());

    rt::LLMEngineConfig config{};
    config.hiddenSize = kHidden;
    config.maxPhysicalTokens = 16;
    config.imageTokenId = kImageTokenId;
    config.audioTokenId = -1;
    rt::EmbeddingPreprocessor devicePreprocessor(deviceEmbedding, config);
    rt::EmbeddingPreprocessor hostPreprocessor(hostEmbedding, config);
    EXPECT_FALSE(devicePreprocessor.needsHostTokenIds());
    EXPECT_TRUE(hostPreprocessor.needsHostTokenIds());

    constexpr int64_t kBatch = 2;
    constexpr int64_t kSeq = 5;
    std::vector<int32_t> const ids{1, kImageTokenId, kImageTokenId, 11, -1, 0, kVocab + 3, 5, 5, kImageTokenId};
    std::vector<half> image(static_cast<size_t>(3) * kHidden);
    for (size_t idx = 0; idx < image.size(); ++idx)
    {
        image[idx] = __float2half(5.0F + static_cast<float>(idx % 13));
    }
    rt::Tensor deviceIds({kBatch, kSeq}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor imageRows({3, kHidden}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    copyHostToDevice(deviceIds, ids);
    copyHostToDevice(imageRows, image);

    rt::PipelineIO deviceIo{};
    deviceIo.inputsEmbeds = rt::Tensor({kBatch, kSeq, kHidden}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    rt::PipelineIO hostIo{};
    hostIo.inputsEmbeds = rt::Tensor({kBatch, kSeq, kHidden}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);

    devicePreprocessor.embed(deviceIds, nullptr, rt::OptionalInputTensor{imageRows}, std::nullopt, deviceIo, mStream);
    hostPreprocessor.embed(deviceIds, ids.data(), rt::OptionalInputTensor{imageRows}, std::nullopt, hostIo, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_EQ(copyDeviceToHost<uint16_t>(hostIo.inputsEmbeds), copyDeviceToHost<uint16_t>(deviceIo.inputsEmbeds));

    // Text-only decode shape through the same preprocessor: the staging reshapes to the step's row count.
    rt::Tensor decodeIds({kBatch, 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    std::vector<int32_t> const decode{3, 9};
    copyHostToDevice(decodeIds, decode);
    ASSERT_TRUE(deviceIo.inputsEmbeds.reshape({kBatch, 1, kHidden}));
    ASSERT_TRUE(hostIo.inputsEmbeds.reshape({kBatch, 1, kHidden}));
    devicePreprocessor.embed(decodeIds, nullptr, std::nullopt, std::nullopt, deviceIo, mStream);
    hostPreprocessor.embed(decodeIds, decode.data(), std::nullopt, std::nullopt, hostIo, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_EQ(copyDeviceToHost<uint16_t>(hostIo.inputsEmbeds), copyDeviceToHost<uint16_t>(deviceIo.inputsEmbeds));

    EXPECT_THROW(
        hostPreprocessor.embed(decodeIds, nullptr, std::nullopt, std::nullopt, hostIo, mStream), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(Formats, HostMappedEmbeddingPreprocessorTest,
    ::testing::Values(SidecarCase{"Fp16", trt_edgellm::DataType::kHALF},
        SidecarCase{"Int8", trt_edgellm::DataType::kINT8}, SidecarCase{"PackedInt4", trt_edgellm::DataType::kUINT8}),
    [](::testing::TestParamInfo<SidecarCase> const& info) { return std::string(info.param.name); });

// Rows beyond the base engine's physical tokens, as a shared speculative draft can embed, need explicit staging rows.
TEST_F(HostRowTableTest, PreprocessorStagingCapacityFollowsTheRequestedRows)
{
    auto const file = mDir / "embedding.safetensors";
    std::vector<rt::Tensor> tensors;
    tensors.push_back(
        hostTensor(rt::Coords{kRows, 64}, trt_edgellm::DataType::kINT8, "embedding", int8Table(kRows, 64)));
    tensors.push_back(
        hostTensor(rt::Coords{kRows}, trt_edgellm::DataType::kFLOAT, "embedding_scale", fp32Scales(kRows)));
    writeSidecar(file, tensors);
    rt::EmbeddingData const embedding = rt::mapEmbeddingTable(file);
    rt::LLMEngineConfig config{};
    config.hiddenSize = 64;
    config.maxPhysicalTokens = 2;
    config.imageTokenId = -1;
    config.audioTokenId = -1;

    std::vector<int32_t> const ids{0, 1, 2, 3};
    rt::Tensor deviceIds({1, 4}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice(deviceIds, ids);
    rt::PipelineIO io{};
    io.inputsEmbeds = rt::Tensor({1, 4, 64}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);

    rt::EmbeddingPreprocessor baseSized(embedding, config);
    EXPECT_THROW(baseSized.embed(deviceIds, ids.data(), std::nullopt, std::nullopt, io, mStream), std::runtime_error);
    rt::EmbeddingPreprocessor draftSized(embedding, config, /*hostStagingRows=*/4);
    EXPECT_NO_THROW(draftSized.embed(deviceIds, ids.data(), std::nullopt, std::nullopt, io, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
}
