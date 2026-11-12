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

#include "runtime/exec/mapBindingCache.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/exec/tensorMap.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace trt_edgellm;
using trt_edgellm::rt::BindingTarget;
using trt_edgellm::rt::Coords;
using trt_edgellm::rt::DeviceType;
using trt_edgellm::rt::MapBindingCache;
using trt_edgellm::rt::Tensor;
using trt_edgellm::rt::TensorMap;

namespace
{
//! Records every setter call without touching TensorRT.
class RecordingTarget final : public BindingTarget
{
public:
    RecordingTarget()
    {
        // 2-D so a cached output dims differs from a default-constructed Dims.
        reportedShape.nbDims = 2;
        reportedShape.d[0] = 4;
        reportedShape.d[1] = 32000;
    }

    struct Call
    {
        std::string name;
        void* address{nullptr};
        nvinfer1::Dims dims{};
    };

    std::vector<Call> addressCalls;
    std::vector<Call> shapeCalls;
    //! Next setTensorAddress for a name in this set returns false (and is still recorded).
    std::vector<std::string> failingNames;
    //! Shape reported by getTensorShape for every name; 2-D so a cached output differs from a
    //! default-constructed Dims.
    nvinfer1::Dims reportedShape{};

    bool setTensorAddress(char const* name, void* address) override
    {
        addressCalls.push_back(Call{name, address, {}});
        for (auto const& failing : failingNames)
        {
            if (failing == name)
            {
                return false;
            }
        }
        return true;
    }

    bool setInputShape(char const* name, nvinfer1::Dims const& dims) override
    {
        shapeCalls.push_back(Call{name, nullptr, dims});
        return true;
    }

    nvinfer1::Dims getTensorShape(char const* name) const override
    {
        (void) name;
        return reportedShape;
    }
    void reset()
    {
        addressCalls.clear();
        shapeCalls.clear();
        failingNames.clear();
    }

    size_t totalCalls() const
    {
        return addressCalls.size() + shapeCalls.size();
    }
};

//! Two map-bound inputs followed by one map-bound output, the layout a rank-adapted engine has.
struct Fixture
{
    static constexpr char const* kIn0 = "lora_in_0";
    static constexpr char const* kIn1 = "lora_in_1";
    static constexpr char const* kOut = "logits";

    Tensor in0{Coords{4, 128, 4096}, DeviceType::kCPU, nvinfer1::DataType::kHALF};
    Tensor in1{Coords{4, 128, 4096}, DeviceType::kCPU, nvinfer1::DataType::kHALF};
    Tensor out{Coords{4, 32000}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT};
    TensorMap map;
    MapBindingCache cache;
    RecordingTarget target;

    Fixture()
    {
        map.set(kIn0, in0);
        map.set(kIn1, in1);
        map.set(kOut, out);
        cache.initialize({kIn0, kIn1, kOut}, {true, true, false}, {true, true, true});
    }
};
} // namespace

TEST(MapBindingCacheTest, FirstApplySetsEveryBinding)
{
    Fixture f;
    EXPECT_TRUE(f.cache.apply(f.map, f.target));

    ASSERT_EQ(f.target.addressCalls.size(), 3u);
    EXPECT_EQ(f.target.addressCalls[0].name, Fixture::kIn0);
    EXPECT_EQ(f.target.addressCalls[0].address, f.in0.rawPointer());
    EXPECT_EQ(f.target.addressCalls[1].name, Fixture::kIn1);
    EXPECT_EQ(f.target.addressCalls[2].name, Fixture::kOut);
    ASSERT_EQ(f.target.shapeCalls.size(), 2u);
    EXPECT_EQ(f.target.shapeCalls[0].name, Fixture::kIn0);
    EXPECT_EQ(f.target.shapeCalls[1].name, Fixture::kIn1);
    EXPECT_TRUE(dimsEqual(f.target.shapeCalls[0].dims, f.in0.getTRTDims()));
}

TEST(MapBindingCacheTest, SecondApplyWithSameMapSkipsAllSetters)
{
    Fixture f;
    ASSERT_TRUE(f.cache.apply(f.map, f.target));
    f.target.reset();
    EXPECT_TRUE(f.cache.apply(f.map, f.target));
    EXPECT_EQ(f.target.totalCalls(), 0u);
}

TEST(MapBindingCacheTest, ChangedAddressRebindsOnlyThatEntry)
{
    Fixture f;
    ASSERT_TRUE(f.cache.apply(f.map, f.target));
    f.target.reset();

    Tensor replacement{Coords{4, 128, 4096}, DeviceType::kCPU, nvinfer1::DataType::kHALF};
    f.map.set(Fixture::kIn1, replacement);

    EXPECT_TRUE(f.cache.apply(f.map, f.target));
    EXPECT_EQ(f.target.totalCalls(), 2u);
    ASSERT_EQ(f.target.addressCalls.size(), 1u);
    EXPECT_EQ(f.target.addressCalls[0].name, Fixture::kIn1);
    EXPECT_EQ(f.target.addressCalls[0].address, replacement.rawPointer());
    ASSERT_EQ(f.target.shapeCalls.size(), 1u);
    EXPECT_EQ(f.target.shapeCalls[0].name, Fixture::kIn1);
}

TEST(MapBindingCacheTest, ReshapedInputRebindsShapeOnlyForThatName)
{
    Fixture f;
    ASSERT_TRUE(f.cache.apply(f.map, f.target));
    f.target.reset();

    EXPECT_TRUE(f.in1.reshape(Coords{2, 128, 4096}));
    EXPECT_TRUE(f.cache.apply(f.map, f.target));

    ASSERT_EQ(f.target.addressCalls.size(), 1u);
    EXPECT_EQ(f.target.addressCalls[0].name, Fixture::kIn1);
    ASSERT_EQ(f.target.shapeCalls.size(), 1u);
    EXPECT_EQ(f.target.shapeCalls[0].name, Fixture::kIn1);
    EXPECT_TRUE(dimsEqual(f.target.shapeCalls[0].dims, f.in1.getTRTDims()));
}

TEST(MapBindingCacheTest, InvalidateAllRebindsEverything)
{
    Fixture f;
    ASSERT_TRUE(f.cache.apply(f.map, f.target));
    f.target.reset();

    f.cache.invalidateAll();
    EXPECT_TRUE(f.cache.apply(f.map, f.target));

    EXPECT_EQ(f.target.addressCalls.size(), 3u);
    EXPECT_EQ(f.target.shapeCalls.size(), 2u);
}

TEST(MapBindingCacheTest, MissingTensorFails)
{
    Fixture f;
    TensorMap partial;
    partial.set(Fixture::kIn0, f.in0);
    EXPECT_FALSE(f.cache.apply(partial, f.target));
    ASSERT_NE(f.cache.lastFailedBinding(), nullptr);
    EXPECT_EQ(std::string(f.cache.lastFailedBinding()), Fixture::kIn1);
}

TEST(MapBindingCacheTest, FailedAddressSetterFailsAndRetriesNextApply)
{
    Fixture f;
    f.target.failingNames = {Fixture::kIn1};
    EXPECT_FALSE(f.cache.apply(f.map, f.target));

    // The failed entry stays invalid, so the next apply retries exactly that name plus the
    // bindings after it that were never reached.
    f.target.failingNames.clear();
    f.target.addressCalls.clear();
    f.target.shapeCalls.clear();
    EXPECT_TRUE(f.cache.apply(f.map, f.target));
    EXPECT_EQ(f.target.totalCalls(), 3u);
    ASSERT_EQ(f.target.addressCalls.size(), 2u);
    EXPECT_EQ(f.target.addressCalls[0].name, Fixture::kIn1);
    EXPECT_EQ(f.target.addressCalls[1].name, Fixture::kOut);
    ASSERT_EQ(f.target.shapeCalls.size(), 1u);
    EXPECT_EQ(f.target.shapeCalls[0].name, Fixture::kIn1);
}

TEST(MapBindingCacheTest, CachedReturnsAppliedState)
{
    Fixture f;
    EXPECT_FALSE(f.cache.cached(0).has_value());
    EXPECT_FALSE(f.cache.cached(2).has_value());

    ASSERT_TRUE(f.cache.apply(f.map, f.target));

    auto const in0 = f.cache.cached(0);
    ASSERT_TRUE(in0.has_value());
    EXPECT_EQ(in0->first, reinterpret_cast<uintptr_t>(f.in0.rawPointer()));
    EXPECT_TRUE(dimsEqual(in0->second, f.in0.getTRTDims()));

    auto const out = f.cache.cached(2);
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->first, reinterpret_cast<uintptr_t>(f.out.rawPointer()));
    // Outputs cache the shape the target reports, not the TensorMap tensor's dims.
    EXPECT_TRUE(dimsEqual(out->second, f.target.reportedShape));

    f.cache.invalidateAll();
    EXPECT_FALSE(f.cache.cached(0).has_value());
}

TEST(MapBindingCacheTest, CachedReturnsNothingForRegistryBoundPositions)
{
    Fixture f;
    f.cache.initialize({Fixture::kIn0, Fixture::kIn1, Fixture::kOut}, {true, true, false}, {false, true, true});
    ASSERT_TRUE(f.cache.apply(f.map, f.target));

    EXPECT_FALSE(f.cache.cached(0).has_value());
    EXPECT_TRUE(f.cache.cached(1).has_value());
    EXPECT_TRUE(f.cache.cached(2).has_value());
    EXPECT_FALSE(f.cache.cached(3).has_value());
}

TEST(MapBindingCacheTest, OutOfRangeIndexReturnsNothing)
{
    Fixture f;
    EXPECT_FALSE(f.cache.cached(99).has_value());
}
