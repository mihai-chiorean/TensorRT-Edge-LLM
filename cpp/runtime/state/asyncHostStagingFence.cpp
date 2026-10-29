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

#include "runtime/state/asyncHostStagingFence.h"

#include "common/cudaUtils.h"

#include <utility>

namespace trt_edgellm
{
namespace rt
{

AsyncHostStagingFence::AsyncHostStagingFence(AsyncHostStagingFence&& other) noexcept
    : mEvent(std::exchange(other.mEvent, nullptr))
    , mPending(std::exchange(other.mPending, false))
{
}

AsyncHostStagingFence& AsyncHostStagingFence::operator=(AsyncHostStagingFence&& other) noexcept
{
    if (this != &other)
    {
        release();
        mEvent = std::exchange(other.mEvent, nullptr);
        mPending = std::exchange(other.mPending, false);
    }
    return *this;
}

AsyncHostStagingFence::~AsyncHostStagingFence()
{
    release();
}

void AsyncHostStagingFence::wait()
{
    if (mPending)
    {
        CUDA_CHECK(cudaEventSynchronize(mEvent));
        mPending = false;
    }
}

void AsyncHostStagingFence::record(cudaStream_t stream)
{
    if (mEvent == nullptr)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&mEvent, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(mEvent, stream));
    mPending = true;
}

void AsyncHostStagingFence::release() noexcept
{
    if (mPending && mEvent != nullptr)
    {
        static_cast<void>(cudaEventSynchronize(mEvent));
    }
    if (mEvent != nullptr)
    {
        static_cast<void>(cudaEventDestroy(mEvent));
    }
    mEvent = nullptr;
    mPending = false;
}

} // namespace rt
} // namespace trt_edgellm
