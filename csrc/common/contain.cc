/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/common/contain.h"

#include <cstdint>
#include <memory>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"

ABSL_FLAG(  // NONSTANDARD_FLAG_NAME_OK=false_positive
    bool, torch_tpu_internal_enable_compilation_container, false,
    "Enable usage of gcontain for capturing peak memory during compilation.");

namespace torch_tpu {

struct ScopedMemMeasuringContainer::Impl {
  // Empty implementation for OSS
};

ScopedMemMeasuringContainer::ScopedMemMeasuringContainer()
    : impl_(std::make_unique<Impl>()) {}

ScopedMemMeasuringContainer::~ScopedMemMeasuringContainer() = default;

absl::StatusOr<int64_t> ContainerPeakHostMemoryBytes() { return 0; }

void CleanUpContainer() {}

}  // namespace torch_tpu
