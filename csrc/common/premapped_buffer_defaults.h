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
#ifndef TORCH_TPU_CSRC_COMMON_PREMAPPED_BUFFER_DEFAULTS_H_
#define TORCH_TPU_CSRC_COMMON_PREMAPPED_BUFFER_DEFAULTS_H_

// Implementation details of the TPU_PREMAPPED_BUFFER_SIZE default computed by
// GetPremappedBufferSizeFromEnvOnce() in discovery.h. This library is private
// to the csrc/common package and exposed here only so that it can be unit
// tested.

#include <cstdint>
#include <optional>
#include <string_view>

namespace torch_tpu {

// Filesystem paths inspected when probing host and container memory limits.
// Tests can override these fields to isolate probing from the host cgroup.
struct MemoryLimitProbePaths {
  std::string_view proc_self_cgroup = "/proc/self/cgroup";
  std::string_view cgroup_v2_root = "/sys/fs/cgroup";
  std::string_view cgroup_v1_borg_root = "/dev/cgroup/memory";
  std::string_view cgroup_v1_sysfs_root = "/sys/fs/cgroup/memory";
  std::optional<int64_t> physical_memory_bytes_override = std::nullopt;
};

// Returns the effective host or container memory limit in bytes (minimum of
// physical RAM and any finite Linux cgroup v1/v2 memory limit along the
// process's cgroup hierarchy), or std::nullopt if no limit is readable.
[[nodiscard]] std::optional<int64_t> GetEffectiveMemoryLimitBytes(
    const MemoryLimitProbePaths& paths = {});

// Estimates how many TorchTPU worker processes share this host/container
// based on distributed env vars and locally exposed TPU devices.
[[nodiscard]] int64_t EstimateLocalWorkerCount(
    std::string_view pci_devices_dir = "/sys/bus/pci/devices",
    std::string_view vfio_dir = "/dev/vfio");

// Computes a default TPU premapped buffer size in bytes for one worker given
// `memory_limit_bytes` shared across `local_workers` processes.
[[nodiscard]] int64_t ComputeDefaultPremappedBufferSizeBytes(
    std::optional<int64_t> memory_limit_bytes, int64_t local_workers);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_PREMAPPED_BUFFER_DEFAULTS_H_
