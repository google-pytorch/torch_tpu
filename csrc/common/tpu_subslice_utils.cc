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

#include "csrc/common/tpu_subslice_utils.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"

namespace torch_tpu {
namespace {

struct Subslice {
  std::unique_ptr<xla::PjRtTopologyDescription> topology;
  // Chip coordinates (x, y, z) of the subslice in the slice.
  xla::PjRtDeviceDimensions origin;
  // The device assignment, with the device ids of the subslice.
  xla::DeviceAssignment device_assignment;
};

// The chip coordinates and core index of a device.
struct ChipAndCore {
  xla::PjRtDeviceDimensions chip;
  int32_t core;
};

// Returns the subslice for `options` (see CompileAndLoadMaybeSubslice), or
// nullopt to compile for the full slice.
absl::StatusOr<std::optional<Subslice>> MaybeMakeSubslice(
    const xla::PjRtClient& client, const xla::CompileOptions& options) {
  const xla::ExecutableBuildOptions& build = options.executable_build_options;
  if (client.platform_id() != xla::TpuId() || !build.has_device_assignment()) {
    return std::nullopt;
  }
  const xla::DeviceAssignment& device_assignment = build.device_assignment();
  const int num_partitions = device_assignment.computation_count();
  if (device_assignment.replica_count() != 1 || num_partitions < 2) {
    return std::nullopt;
  }
  TT_ASSIGN_OR_RETURN(const xla::PjRtTopologyDescription* const topology,
                      client.GetTopologyDescription());
  TT_ASSIGN_OR_RETURN(const int num_devices,
                      topology->LogicalDeviceCountOfDefaultType());
  if (num_partitions >= num_devices) {
    return std::nullopt;
  }
  TT_ASSIGN_OR_RETURN(const int cores_per_chip,
                      topology->LogicalDeviceCountOfDefaultTypePerChip());

  // The chip and core of each partition, and the box of their chips.
  std::vector<ChipAndCore> chip_and_core;
  chip_and_core.reserve(num_partitions);
  for (int p = 0; p < num_partitions; ++p) {
    const int64_t device_id = device_assignment(0, p);
    // In multi-slice topologies, GetTopologyDescription() describes a single
    // slice of `num_devices` logical devices while device IDs encode the slice
    // offset outside [0, num_devices).
    if (device_id < 0 || device_id >= num_devices) {
      return std::nullopt;
    }
    TT_ASSIGN_OR_RETURN(
        const auto location,
        topology->ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
            xla::GlobalDeviceId(device_id)));
    chip_and_core.push_back({location.first, location.second});
  }
  xla::PjRtDeviceDimensions min_chip = chip_and_core[0].chip;
  xla::PjRtDeviceDimensions max_chip = chip_and_core[0].chip;
  for (const auto& [chip, core] : chip_and_core) {
    for (int i = 0; i < 3; ++i) {
      min_chip[i] = std::min(min_chip[i], chip[i]);
      max_chip[i] = std::max(max_chip[i], chip[i]);
    }
  }
  const xla::PjRtDeviceDimensions box{max_chip[0] - min_chip[0] + 1,
                                      max_chip[1] - min_chip[1] + 1,
                                      max_chip[2] - min_chip[2] + 1};
  // The partitions are on distinct devices of the box, so they cover all its
  // cores if there are as many partitions as cores.
  if (box[0] * box[1] * box[2] * cores_per_chip != num_partitions) {
    return std::nullopt;
  }

  // torch_tpu runs one process per core with chips-per-host bounds 1,1,1 (see
  // environment.cc), so the box is given as host bounds.
  absl::StatusOr<std::unique_ptr<xla::PjRtTopologyDescription>>
      subslice_topology = topology->Subslice(
          /*chips_per_host_bounds=*/{1, 1, 1}, /*host_bounds=*/box);
  if (!subslice_topology.ok()) {
    ABSL_VLOG(1) << "Compiling for the full slice: no subslice of chips "
                 << box.ToString() << ": " << subslice_topology.status();
    return std::nullopt;
  }
  Subslice subslice{*std::move(subslice_topology), min_chip,
                    xla::DeviceAssignment(1, num_partitions)};
  for (int p = 0; p < num_partitions; ++p) {
    const auto& [chip, core] = chip_and_core[p];
    const xla::PjRtDeviceDimensions subslice_chip{
        chip[0] - min_chip[0], chip[1] - min_chip[1], chip[2] - min_chip[2]};
    TT_ASSIGN_OR_RETURN(
        const xla::GlobalDeviceId device_id,
        subslice.topology
            ->LogicalDeviceOfDefaultTypeIdFromChipCoordAndCoreIndex(
                subslice_chip, core));
    subslice.device_assignment(0, p) = device_id.value();
  }
  return subslice;
}

}  // namespace

absl::StatusOr<std::optional<xla::PjRtDeviceDimensions>> GetSubsliceOrigin(
    const xla::PjRtClient& client, const xla::CompileOptions& options) {
  TT_ASSIGN_OR_RETURN(std::optional<Subslice> subslice,
                      MaybeMakeSubslice(client, options));
  if (!subslice.has_value()) {
    return std::nullopt;
  }
  return subslice->origin;
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
CompileAndLoadMaybeSubslice(xla::PjRtClient& client,
                            xla::MaybeOwningMlirModule module,
                            xla::CompileOptions options) {
  TT_ASSIGN_OR_RETURN(std::optional<Subslice> subslice,
                      MaybeMakeSubslice(client, options));
  if (!subslice.has_value()) {
    return client.CompileAndLoad(std::move(module), std::move(options));
  }
  options.executable_build_options.set_device_assignment(
      subslice->device_assignment);
  TT_ASSIGN_OR_RETURN(const std::unique_ptr<xla::PjRtExecutable> executable,
                      xla::PjRtCompile(std::move(options), std::move(module),
                                       *subslice->topology, &client));
  // PjRtCApiClient::Load() drops LoadOptions, so load via serialization, which
  // honors computation_origin.
  TT_ASSIGN_OR_RETURN(const std::string serialized,
                      executable->SerializeExecutable());
  return LoadSerializedExecutableAtOrigin(client, serialized, subslice->origin);
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
LoadSerializedExecutableAtOrigin(
    xla::PjRtClient& client, const std::string_view serialized,
    const std::optional<xla::PjRtDeviceDimensions>& subslice_origin) {
  xla::LoadOptions load_options;
  load_options.computation_origin = subslice_origin;
  return client.LoadSerializedExecutable(serialized, /*options=*/std::nullopt,
                                         load_options);
}

}  // namespace torch_tpu
