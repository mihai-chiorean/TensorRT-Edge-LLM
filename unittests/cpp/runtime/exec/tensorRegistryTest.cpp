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

#include "runtime/exec/tensorRegistry.h"
#include "common/tensor.h"
#include "runtime/config/inferenceDims.h"
#include "runtime/exec/tensorMap.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace trt_edgellm::rt;

TEST(TensorRegistryTest, ResolveShapeAllFixed)
{
    TensorRegistry reg;
    reg.addTensor({"logits", TensorIO::kOutput, nvinfer1::DataType::kFLOAT, {fixed(4), fixed(32000)}});
    auto specs = reg.allExpandedSpecs();
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].name, "logits");

    InferenceDims const dims{/*.batch=*/0, /*.seqLen=*/0, /*.kvLen=*/0, /*.selectLen=*/0, /*.attnMaskSeqLen=*/0,
        /*.ropeBatch=*/0, /*.packedMaskLen=*/0, /*.contextMaskSelectorLen=*/0, /*.startIndexLen=*/0,
        /*.executionPhaseLen=*/0, /*.skipSoftmaxScaleLen=*/0, /*.swaKVCacheModeLen=*/0, /*.queryOffsetLen=*/0,
        /*.contextSequenceCount=*/0};
    auto resolved = reg.resolveShape(specs[0].shape, dims);
    EXPECT_EQ(resolved.nbDims, 2);
    EXPECT_EQ(resolved.d[0], 4);
    EXPECT_EQ(resolved.d[1], 32000);
}

TEST(TensorRegistryTest, ResolveShapeWithSymbolicDims)
{
    TensorRegistry reg;
    reg.addTensor({"inputs_embeds", TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(4096)}});

    InferenceDims const dims{/*.batch=*/4, /*.seqLen=*/128, /*.kvLen=*/1, /*.selectLen=*/1, /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/1, /*.packedMaskLen=*/1, /*.contextMaskSelectorLen=*/0, /*.startIndexLen=*/4,
        /*.executionPhaseLen=*/0, /*.skipSoftmaxScaleLen=*/0, /*.swaKVCacheModeLen=*/0, /*.queryOffsetLen=*/5,
        /*.contextSequenceCount=*/0};
    auto specs = reg.allExpandedSpecs();
    auto resolved = reg.resolveShape(specs[0].shape, dims);
    EXPECT_EQ(resolved.nbDims, 3);
    EXPECT_EQ(resolved.d[0], 4);
    EXPECT_EQ(resolved.d[1], 128);
    EXPECT_EQ(resolved.d[2], 4096);
}

TEST(TensorRegistryTest, PerLayerExpansion)
{
    TensorRegistry reg;
    reg.addTensor({"past_key_values_%d", TensorIO::kInput, nvinfer1::DataType::kHALF,
        {fixed(4), fixed(2), fixed(8), fixed(2048), fixed(128)},
        /*perLayer=*/3});
    auto specs = reg.allExpandedSpecs();
    ASSERT_EQ(specs.size(), 3u);
    EXPECT_EQ(specs[0].name, "past_key_values_0");
    EXPECT_EQ(specs[1].name, "past_key_values_1");
    EXPECT_EQ(specs[2].name, "past_key_values_2");
}

TEST(TensorRegistryTest, AllTensorNames)
{
    TensorRegistry reg;
    reg.addTensor({"inputs_embeds", TensorIO::kInput, nvinfer1::DataType::kHALF, {fixed(4), fixed(128), fixed(4096)}});
    reg.addTensor({"kv_%d", TensorIO::kInput, nvinfer1::DataType::kHALF, {fixed(4)}, /*perLayer=*/2});
    auto names = reg.allTensorNames();
    ASSERT_EQ(names.size(), 3u); // inputs_embeds, kv_0, kv_1
}

TEST(TensorRegistryTest, ReferencedMembersDedupedAcrossTensors)
{
    TensorRegistry reg;
    // Two tensors that both reference batch; batch should appear once in the set.
    reg.addTensor(
        {"a", TensorIO::kInput, nvinfer1::DataType::kHALF, {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen)}});
    reg.addTensor(
        {"b", TensorIO::kInput, nvinfer1::DataType::kHALF, {sym(&InferenceDims::batch), sym(&InferenceDims::kvLen)}});

    auto const& refs = reg.referencedMembers();
    ASSERT_EQ(refs.size(), 3u);
    EXPECT_NE(std::find(refs.begin(), refs.end(), &InferenceDims::batch), refs.end());
    EXPECT_NE(std::find(refs.begin(), refs.end(), &InferenceDims::seqLen), refs.end());
    EXPECT_NE(std::find(refs.begin(), refs.end(), &InferenceDims::kvLen), refs.end());
}

TEST(TensorRegistryTest, ReferencedMembersAllFixedIsEmpty)
{
    TensorRegistry reg;
    reg.addTensor({"logits", TensorIO::kOutput, nvinfer1::DataType::kFLOAT, {fixed(4), fixed(32000)}});
    EXPECT_TRUE(reg.referencedMembers().empty());
}

namespace
{
//! Records the setters the caching bindAll calls, without TensorRT.
class CountingTarget final : public BindingTarget
{
public:
    std::vector<std::string> addresses;
    std::vector<std::string> shapes;
    std::vector<nvinfer1::Dims> shapeDims;
    bool setTensorAddress(char const* name, void* /*address*/) override
    {
        addresses.emplace_back(name);
        return true;
    }
    bool setInputShape(char const* name, nvinfer1::Dims const& dims) override
    {
        shapes.emplace_back(name);
        shapeDims.push_back(dims);
        return true;
    }
    nvinfer1::Dims getTensorShape(char const* /*name*/) const override
    {
        return {};
    }
    void clear()
    {
        addresses.clear();
        shapes.clear();
        shapeDims.clear();
    }
};

InferenceDims dimsWithBatch(int64_t batch)
{
    return InferenceDims{/*.batch=*/batch, /*.seqLen=*/1, /*.kvLen=*/1, /*.selectLen=*/1, /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/1, /*.packedMaskLen=*/1, /*.contextMaskSelectorLen=*/0, /*.startIndexLen=*/1,
        /*.executionPhaseLen=*/0, /*.skipSoftmaxScaleLen=*/0, /*.swaKVCacheModeLen=*/0, /*.queryOffsetLen=*/1,
        /*.contextSequenceCount=*/0};
}
} // namespace

//! The per-step decode bind: unchanged addresses and resolved shapes call no setter and report no
//! change; a changed symbolic dim rebinds only the input it shapes; a repointed tensor rebinds.
TEST(TensorRegistryTest, CachedBindAllCallsOnlyChangedSetters)
{
    TensorRegistry reg;
    reg.addTensor({"inputs_embeds", TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), fixed(1), fixed(64)}});
    reg.addTensor({"position_ids", TensorIO::kInput, nvinfer1::DataType::kINT32, {fixed(1), fixed(1)}});
    reg.addTensor({"logits", TensorIO::kOutput, nvinfer1::DataType::kFLOAT, {fixed(4), fixed(32)}});
    Tensor embeds{Coords{4, 1, 64}, DeviceType::kCPU, nvinfer1::DataType::kHALF};
    Tensor positions{Coords{1, 1}, DeviceType::kCPU, nvinfer1::DataType::kINT32};
    Tensor logits{Coords{4, 32}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT};
    TensorMap map;
    map.set("inputs_embeds", embeds);
    map.set("position_ids", positions);
    map.set("logits", logits);
    TensorRegistry::BindingCache cache;
    CountingTarget target;
    bool changed = false;

    ASSERT_TRUE(reg.bindAll(target, map, dimsWithBatch(1), cache, changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(target.addresses.size(), 3u);
    EXPECT_EQ(target.shapes.size(), 2u);

    target.clear();
    ASSERT_TRUE(reg.bindAll(target, map, dimsWithBatch(1), cache, changed));
    EXPECT_FALSE(changed);
    EXPECT_TRUE(target.addresses.empty());
    EXPECT_TRUE(target.shapes.empty());

    target.clear();
    ASSERT_TRUE(reg.bindAll(target, map, dimsWithBatch(2), cache, changed));
    EXPECT_TRUE(changed);
    ASSERT_EQ(target.shapes.size(), 1u);
    EXPECT_EQ(target.shapes[0], "inputs_embeds");
    EXPECT_EQ(target.shapeDims[0].d[0], 2);

    target.clear();
    Tensor otherLogits{Coords{4, 32}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT};
    map.set("logits", otherLogits);
    ASSERT_TRUE(reg.bindAll(target, map, dimsWithBatch(2), cache, changed));
    EXPECT_TRUE(changed);
    ASSERT_EQ(target.addresses.size(), 1u);
    EXPECT_EQ(target.addresses[0], "logits");
    EXPECT_TRUE(target.shapes.empty());

    target.clear();
    cache.invalidate(); // a profile switch: everything is bound again
    ASSERT_TRUE(reg.bindAll(target, map, dimsWithBatch(2), cache, changed));
    EXPECT_EQ(target.addresses.size(), 3u);
}
