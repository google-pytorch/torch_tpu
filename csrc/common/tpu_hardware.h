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
#ifndef TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_
#define TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_

#include <string_view>

#include "absl/status/statusor.h"

namespace torch_tpu {

// Returns whether any device node through which libtpu could reach a TPU
// exists:
// - a VFIO group node `<dev_dir>/vfio/<number>` (not the container node
//   `<dev_dir>/vfio/vfio`);
// - a VFIO per-device (iommufd) node, i.e. any entry of
//   `<dev_dir>/vfio/devices/`, e.g. `vfio0`;
// - an accel node `<dev_dir>/accel*`, `<dev_borg_dir>/accel*`,
//   `<dev_dir>/accel/*` or `<dev_borg_dir>/accel/*` (the layout of the kernel's
//   compute accelerator subsystem, e.g. /dev/accel/accel0) that is a character
//   device whose major is registered in `proc_devices` (i.e. backed by a
//   driver).
// This does not depend on PCI device IDs, so it also covers unknown TPU
// generations. Returns true if a directory exists
// but cannot be listed, or if an accel character device exists but
// `proc_devices` cannot be read, since the absence of TPU device nodes cannot
// be established then. Does not initialize the TPU runtime.
//
// A process without any of these nodes cannot use a TPU through the default
// device paths. Hosts with other VFIO or accel devices (e.g. a NIC bound to
// VFIO) also yield true, so true does not mean that a TPU is present.
[[nodiscard]] bool HasTpuDeviceNodes(
    std::string_view dev_dir = "/dev",
    std::string_view dev_borg_dir = "/dev_borg",
    std::string_view proc_devices = "/proc/devices");

// Returns whether `accelerator_type` is a multi-device-per-chip TPU without
// megacore that requires 4D TPU bounds ("1,1,1,1") instead of 3D ("1,1,1") for
// single-device initialization.
//
// `accelerator_type` must be in the case-insensitive format
// `<generation>-<device_count>`, where `<device_count>` is a positive integer
// (e.g., "v5litepod-4", "v7x-8") and `<generation>` is one of:
// - Single-device-per-chip or megacore generations (returns false):
//   "v4", "v5e", "v5litepod", "v5p", "v6e", "v8t", "tpu8t".
// - Multi-device-per-chip generations without megacore (returns true):
//   "v7", "v7x", "tpu7", "tpu7x", "v8i", "tpu8i".
//
// Returns an InvalidArgumentError if `accelerator_type` is empty, does not
// match `<generation>-<device_count>`, or has an unrecognized `<generation>`.
absl::StatusOr<bool> IsMultiDevicePerChipAcceleratorType(
    std::string_view accelerator_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_
