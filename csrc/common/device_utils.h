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

// Resolves the single TPU device visible to this process and configures
// `TPU_VISIBLE_CHIPS` for libtpu.
//
// Contract for `TPU_VISIBLE_DEVICES` vs `TPU_VISIBLE_CHIPS`:
// - `TPU_VISIBLE_DEVICES` (`SymbolStage::Stable()`) is a read-only input knob
//   set by the user or external job launcher (`torchrun`, GKE, etc.). TorchTPU
//   reads it via `GetEnvOnce<kTpuVisibleDevicesEnvVar>()` and never modifies
//   it, preserving consistency with `GetEnvOnce`.
// - `TPU_VISIBLE_CHIPS` (`SymbolStage::InternalImplementation()`) is a
//   write-only output environment variable used by TorchTPU to pass the
//   resolved single chip index (`TPU_VISIBLE_DEVICES` if it specifies a single
//   valid device index, otherwise `fallback_device`) to `libtpu`, which
//   prioritizes non-empty `TPU_VISIBLE_CHIPS` over `TPU_VISIBLE_DEVICES`.
//
// Returns the resolved device index string.
std::string ConfigureVisibleDevice(std::string_view fallback_device);

// Sets or overwrites a boolean `--<flag_name>=<value>` entry in `args`.
void SetLibtpuInitFlag(std::string* args, std::string_view flag_name,
                       std::string_view value);

// Configures common TPU topology, slicebuilder address/port, and visible device
// environment variables for a distributed worker.
void ConfigureDistributedTopologyEnv(int rank, int local_rank,
                                     const std::string& topology,
                                     const std::string& sb_addrs,
                                     const std::string& sb_port);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_DEVICE_UTILS_H_
