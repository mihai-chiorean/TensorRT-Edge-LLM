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

#include <cstdint>
#include <optional>
#include <string>

namespace trt_edgellm
{
namespace builder
{

//! Engine hardware compatibility requested from the builder.
enum class HardwareCompatibility : int32_t
{
    kNONE = 0,                    //!< Engine targets the build GPU only (TensorRT default)
    kSAME_COMPUTE_CAPABILITY = 1, //!< Engine runs on any GPU with the build GPU's compute capability
};

//! Parse a hardware compatibility name: "none" or "same_compute_capability".
//! @return The parsed value, or std::nullopt for an unknown name
std::optional<HardwareCompatibility> parseHardwareCompatibility(std::string const& name) noexcept;

//! @return The name accepted by parseHardwareCompatibility for this value
char const* hardwareCompatibilityName(HardwareCompatibility compatibility) noexcept;

} // namespace builder
} // namespace trt_edgellm
