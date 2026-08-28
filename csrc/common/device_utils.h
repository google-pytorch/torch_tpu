// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TORCH_TPU_CSRC_COMMON_DEVICE_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_DEVICE_UTILS_H_

#include <string>
#include <string_view>

namespace torch_tpu {

// Returns true if `device_indices` represents a single non-negative device
// index (e.g. "0" or "2"). Returns false for empty strings, multi-device
// strings ("0,1"), or non-integers.
[[nodiscard]] bool IsSingleDeviceSpecified(std::string_view device_indices);

// Returns a comma-separated string of `1`s matching the dimensionality of
// `topology` (i.e. `"1,1,1"` for any 3D `topology`, or `"1,1,1,1"` for any 4D
// `topology`), representing the per-process single-chip sub-mesh bounds.
//
// `topology` is a comma-separated string of positive integers specifying the
// full TPU mesh dimensions across all workers:
//   - Input `topology = "2,2,1"` (3D mesh, e.g. TPU v5e) -> returns `"1,1,1"`.
//   - Input `topology = "2,2,1,2"` (4D mesh, e.g. TPU v5p/v7) -> returns
//     `"1,1,1,1"`.
[[nodiscard]] std::string GetChipsPerHostBounds(std::string_view topology);

// Selects the single TPU device visible to this process: `TPU_VISIBLE_DEVICES`
// if it specifies a single device index, or `default_device` otherwise.
//
// TPU_VISIBLE_DEVICES is the source of truth for device visibility. Because
// libtpu prioritizes TPU_VISIBLE_CHIPS over TPU_VISIBLE_DEVICES when
// non-empty, both env vars are overwritten with the selected device so that
// external settings do not override device selection. TPU_VISIBLE_DEVICES is
// read with std::getenv() rather than GetEnvOnce<>(), since it is written
// here.
void ConfigureVisibleDevice(std::string_view default_device);

// Sets the libtpu host/process bounds environment variables to `host_bounds`
// and `chips_bounds` and enables concurrent libtpu loads across processes on
// the same host.
void SetTpuBoundsEnv(const std::string& host_bounds,
                     const std::string& chips_bounds);

// Sets the boolean libtpu flag `flag_name` to `value` in the
// whitespace-separated flags `args`: replaces each setting of the flag (as
// `--<flag_name>`, `--<flag_name>=<value>` or `--no<flag_name>`, with one or
// two leading dashes) with `--<flag_name>=<value>`, or appends that if `args`
// doesn't set the flag.
void SetLibtpuInitFlag(std::string* args, std::string_view flag_name,
                       bool value);

// Configures the TPU topology, slicebuilder address/port, task ID and visible
// device environment variables of a distributed worker of global rank `rank`.
// The visible device is selected with ConfigureVisibleDevice(default_device).
// `sb_addrs` lists the comma-separated slicebuilder addresses of all processes
// of the slice.
void ConfigureDistributedTopologyEnv(int rank, std::string_view default_device,
                                     const std::string& topology,
                                     const std::string& sb_addrs,
                                     const std::string& sb_port);

// Returns whether the TPU device paths that libtpu should use are configured
// explicitly, in `TPU_VISIBLE_DEVICE_PATHS` or as
// `--deepsea_hal_included_devicepaths` in `LIBTPU_INIT_ARGS`. Such paths may
// name device nodes outside of the default locations checked by
// HasTpuDeviceNodes() in tpu_hardware.h. Does not initialize the TPU runtime
// and does not cache the result.
[[nodiscard]] bool HasCustomTpuDevicePaths();

// Returns `configured_devices` if it specifies a single device index, or
// `default_device` otherwise.
[[nodiscard]] inline std::string GetSingleDeviceOrDefault(
    std::string_view configured_devices, std::string_view default_device) {
  return IsSingleDeviceSpecified(configured_devices)
             ? std::string(configured_devices)
             : std::string(default_device);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_DEVICE_UTILS_H_
