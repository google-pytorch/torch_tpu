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

#ifndef TORCH_TPU_CSRC_COMMON_TPU_SUBSLICE_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_TPU_SUBSLICE_UTILS_H_

#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {

// Returns the chip coordinates (x, y, z) of the TPU subslice that
// CompileAndLoadMaybeSubslice compiles `options` for, or nullopt if it compiles
// them for the full slice. An executable compiled for a subslice must be loaded
// at this origin.
absl::StatusOr<std::optional<xla::PjRtDeviceDimensions>> GetSubsliceOrigin(
    const xla::PjRtClient& client, const xla::CompileOptions& options);

// Compiles `module` and loads it. If the device assignment of `options` is one
// replica over every core of a box of chips smaller than the slice, compiles it
// for that subslice and loads it at the subslice origin.
//
// On single-slice TPU, compiles to an unloaded `PjRtExecutable`, serializes it,
// drops the compile-time executable, and loads from serialized bytes so
// compile-time compiler state is not retained in Host DRAM alongside the loaded
// executable. Multi-slice (MegaScale) and non-TPU clients use `CompileAndLoad`.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
CompileAndLoadMaybeSubslice(xla::PjRtClient& client,
                            xla::MaybeOwningMlirModule module,
                            xla::CompileOptions options);

// Loads a serialized executable at `subslice_origin` (see GetSubsliceOrigin),
// or for the full slice if nullopt.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
LoadSerializedExecutableAtOrigin(
    xla::PjRtClient& client, std::string_view serialized,
    const std::optional<xla::PjRtDeviceDimensions>& subslice_origin);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_TPU_SUBSLICE_UTILS_H_
