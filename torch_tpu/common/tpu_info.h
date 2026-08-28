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

#ifndef TORCH_TPU_COMMON_TPU_INFO_H_
#define TORCH_TPU_COMMON_TPU_INFO_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <string_view>

namespace torch_tpu {

// Enum representing TPU chip versions matching JAX's ChipVersion.
enum class ChipVersion {
  kTPUv2,
  kTPUv3,
  kTPUv4i,
  kTPUv4,
  kTPUv5e,
  kTPUv5p,
  kTPUv6e,
  kTPU7,
  kTPU7x,
  kTPU8i,
  kTPU8t,
};

// Information about SparseCore on TPU.
struct SparseCoreInfo {
  int num_cores;
  int num_subcores;
  int num_lanes;
  int64_t vmem_capacity_bytes;
  int64_t dma_granule_size_bytes;
};

// Information about TPU chip hardware specifications and memory capacities.
struct TpuInfo {
  ChipVersion chip_version;
  int generation;
  int num_cores;
  int num_lanes;
  int num_sublanes;
  int mxu_column_size;
  int num_mxus;
  int num_accumulators;
  int64_t vmem_capacity_bytes;
  int64_t cmem_capacity_bytes;
  int64_t smem_capacity_bytes;
  int64_t hbm_capacity_bytes;
  int64_t mem_bw_bytes_per_second;
  int64_t bf16_ops_per_second;
  int64_t int8_ops_per_second;
  int64_t fp8_ops_per_second;
  int64_t int4_ops_per_second;
  std::optional<SparseCoreInfo> sparse_core;
  bool is_lite;
  bool is_split_chip;
  bool is_megacore;
};

// Maps a device kind string (e.g. "TPU v4", "TPU v5e", "TPU v6e", "TPU7") to
// ChipVersion.
std::optional<ChipVersion> ChipVersionFromDeviceKind(
    std::string_view device_kind);

// Returns TpuInfo for a given chip version.
std::optional<TpuInfo> GetTpuInfoForChip(ChipVersion chip_version);

// Returns TpuInfo for a given device kind string.
std::optional<TpuInfo> GetTpuInfo(std::string_view device_kind);

// Stream operators formatting representations with indented entries.
std::ostream& operator<<(std::ostream& os, ChipVersion chip_version);
std::ostream& operator<<(std::ostream& os, const SparseCoreInfo& sc_info);
std::ostream& operator<<(std::ostream& os, const TpuInfo& tpu_info);

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_TPU_INFO_H_
