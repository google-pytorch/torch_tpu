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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

namespace torch_tpu {

// SparseCore specs of a TPU generation.
//
// SparseCores are TPU processors specialized for sparse operations such as
// embedding lookups.
struct SparseCoreSpec {
  // Number of SparseCores per TPU device. On generations that expose each
  // TensorCore of a chip as a separate device (e.g. TPU v7), this counts the
  // SparseCores of one device, not of the chip.
  int64_t sparse_cores_per_device = 0;
  // SIMD width (number of vector lanes) of each SparseCore.
  int64_t simd_width = 0;

  bool operator==(const SparseCoreSpec& other) const = default;
};

// Specs of a TPU generation that TorchTPU supports.
//
// TorchTPU doesn't run on TPU v4 and older generations, so they have no
// constants on `TpuSpec`.
struct TpuSpec {
  using Topology = std::vector<int64_t>;  // INT_VEC_OK
  using TopologyMap = absl::flat_hash_map<int64_t, Topology>;

  // Human-readable name of the generation, as returned by
  // `torch.tpu.get_device_properties().name`.
  std::string name;
  // Number of separately-addressable devices the chip exposes to the runtime,
  // used as the divisor that turns a per-chip quantity into what one device
  // sees. This is megacore-aware (not the physical TensorCore count): 1 for
  // single-core chips (v5e, v6e, v8t) and megacore chips (v5p) that fuse their
  // two physical cores into one addressable device, and 2 for dual-core chips
  // without megacore (v7, v8i).
  int64_t devices_per_chip = 1;
  // Physical HBM capacity per chip, in bytes.
  int64_t hbm_bytes_per_chip = 0;
  // Peak dense bf16 compute per chip, in FLOP/s.
  int64_t bf16_flops_per_chip = 0;
  // Topology dimensions of each number of devices that a host can use.
  TopologyMap topologies;
  // SparseCore specs of the generation, or std::nullopt if it has no
  // SparseCores.
  std::optional<SparseCoreSpec> sparse_core_spec;

  // Physical HBM capacity, in bytes, visible to a single device.
  [[nodiscard]] int64_t hbm_bytes_per_device() const {
    return hbm_bytes_per_chip / devices_per_chip;
  }

  // Peak dense bf16 compute, in FLOP/s, for a single device.
  [[nodiscard]] int64_t bf16_flops_per_device() const {
    return bf16_flops_per_chip / devices_per_chip;
  }

  // Returns the topology dimensions for `world_size` (or for the number of
  // VFIO-exposed TPU devices if `world_size` is std::nullopt), or an
  // InvalidArgumentError if no topology is defined for that device count.
  absl::StatusOr<Topology> GetTopology(
      std::optional<int64_t> world_size = std::nullopt,
      std::string_view pci_devices_dir = "/sys/bus/pci/devices",
      std::string_view vfio_dir = "/dev/vfio") const;

  bool operator==(const TpuSpec& other) const = default;

  static const TpuSpec& V5p();
  static const TpuSpec& V5e();
  static const TpuSpec& V6e();
  static const TpuSpec& V7();
  static const TpuSpec& V8i();
  static const TpuSpec& V8t();

  // Returns the mapping from PCI device ID (e.g. "0x0076") to the
  // corresponding `TpuSpec`.
  [[nodiscard]] static const absl::flat_hash_map<std::string,
                                                 const TpuSpec* absl_nonnull>&
  SpecsByPciDeviceId();

  // Returns the `TpuSpec` of the attached TPU device by scanning PCI devices
  // under `pci_devices_dir` for a Google TPU, or nullptr if no supported TPU is
  // found. Does not require VFIO exposure and does not initialize any TPU
  // runtime.
  [[nodiscard]] static const TpuSpec* absl_nullable Get(
      std::string_view pci_devices_dir = "/sys/bus/pci/devices");
};

// Returns the number of supported Google TPU PCI devices under
// `pci_devices_dir` whose IOMMU group is exposed under `vfio_dir` as a
// character device that this process can read and write.
//
// This function does not initialize any TPU runtime and ignores placeholders at
// TPU VFIO paths that a sandbox doesn't allocate to this process, such as
// unreadable regular files.
[[nodiscard]] int64_t GetTpuDeviceCount(
    std::string_view pci_devices_dir = "/sys/bus/pci/devices",
    std::string_view vfio_dir = "/dev/vfio");

// Like GetTpuDeviceCount(), but if `TPU_VISIBLE_DEVICES` is a comma-separated
// list of chip indices, caps the count at the number of devices on that many
// chips, like `CUDA_VISIBLE_DEVICES` caps the GPU count. Reads
// `TPU_VISIBLE_DEVICES` on every call, since it may change until the TPU
// runtime is initialized. TorchTPU itself may rewrite it when it initializes
// the runtime, so this is meant for use before then.
[[nodiscard]] int64_t GetVisibleTpuDeviceCount(
    std::string_view pci_devices_dir = "/sys/bus/pci/devices",
    std::string_view vfio_dir = "/dev/vfio");

// Returns the comma-separated topology string (e.g. "2,2,1") for the
// VFIO-exposed TPU devices, or std::nullopt if no VFIO-exposed TPU device is
// found. Returns an InvalidArgumentError if a TPU is found but no topology is
// defined for `world_size` (or the detected device count when `world_size` is
// std::nullopt).
absl::StatusOr<std::optional<std::string>> GetTpuTopology(
    std::optional<int64_t> world_size = std::nullopt,
    std::string_view pci_devices_dir = "/sys/bus/pci/devices",
    std::string_view vfio_dir = "/dev/vfio");

// Returns the single-device TPU bounds ({1, 1, 1, 1} for TPU generations with
// multiple devices per chip without megacore, or {1, 1, 1} otherwise) for the
// TPU generation attached under `pci_devices_dir`.
[[nodiscard]] TpuSpec::Topology GetSingleDeviceTpuBounds(
    std::string_view pci_devices_dir = "/sys/bus/pci/devices");

// Returns the comma-separated single-device TPU bounds string ("1,1,1,1" or
// "1,1,1") for the TPU generation attached under `pci_devices_dir`.
[[nodiscard]] std::string GetSingleDeviceTpuBoundsString(
    std::string_view pci_devices_dir = "/sys/bus/pci/devices");

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

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_
