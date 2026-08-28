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

#include "torch_tpu/common/tpu_info.h"

#include <cstdint>  // IWYU pragma: keep
#include <optional>
#include <ostream>
#include <string_view>

namespace torch_tpu {

#include "torch_tpu/common/tpu_info_data.inc"

std::optional<TpuInfo> GetTpuInfo(std::string_view device_kind) {
  auto chip_version = ChipVersionFromDeviceKind(device_kind);
  if (!chip_version.has_value()) {
    return std::nullopt;
  }
  return GetTpuInfoForChip(*chip_version);
}

std::ostream& operator<<(std::ostream& os, ChipVersion chip_version) {
  return os << ToString(chip_version);
}

std::ostream& operator<<(std::ostream& os, const SparseCoreInfo& sc_info) {
  os << "SparseCoreInfo {\n"
     << "  num_cores: " << sc_info.num_cores << ",\n"
     << "  num_subcores: " << sc_info.num_subcores << ",\n"
     << "  num_lanes: " << sc_info.num_lanes << ",\n"
     << "  vmem_capacity_bytes: " << sc_info.vmem_capacity_bytes << ",\n"
     << "  dma_granule_size_bytes: " << sc_info.dma_granule_size_bytes << ",\n"
     << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const TpuInfo& info) {
  os << "TpuInfo {\n"
     << "  chip_version: " << ToString(info.chip_version) << ",\n"
     << "  generation: " << info.generation << ",\n"
     << "  num_cores: " << info.num_cores << ",\n"
     << "  num_lanes: " << info.num_lanes << ",\n"
     << "  num_sublanes: " << info.num_sublanes << ",\n"
     << "  mxu_column_size: " << info.mxu_column_size << ",\n"
     << "  num_mxus: " << info.num_mxus << ",\n"
     << "  num_accumulators: " << info.num_accumulators << ",\n"
     << "  vmem_capacity_bytes: " << info.vmem_capacity_bytes << ",\n"
     << "  cmem_capacity_bytes: " << info.cmem_capacity_bytes << ",\n"
     << "  smem_capacity_bytes: " << info.smem_capacity_bytes << ",\n"
     << "  hbm_capacity_bytes: " << info.hbm_capacity_bytes << ",\n"
     << "  mem_bw_bytes_per_second: " << info.mem_bw_bytes_per_second << ",\n"
     << "  bf16_ops_per_second: " << info.bf16_ops_per_second << ",\n"
     << "  int8_ops_per_second: " << info.int8_ops_per_second << ",\n"
     << "  fp8_ops_per_second: " << info.fp8_ops_per_second << ",\n"
     << "  int4_ops_per_second: " << info.int4_ops_per_second << ",\n";
  if (info.sparse_core.has_value()) {
    const SparseCoreInfo& sc_info = *info.sparse_core;
    os << "  sparse_core: SparseCoreInfo {\n"
       << "    num_cores: " << sc_info.num_cores << ",\n"
       << "    num_subcores: " << sc_info.num_subcores << ",\n"
       << "    num_lanes: " << sc_info.num_lanes << ",\n"
       << "    vmem_capacity_bytes: " << sc_info.vmem_capacity_bytes << ",\n"
       << "    dma_granule_size_bytes: " << sc_info.dma_granule_size_bytes
       << ",\n"
       << "  },\n";
  } else {
    os << "  sparse_core: N/A,\n";
  }
  os << "  is_lite: " << (info.is_lite ? "true" : "false") << ",\n"
     << "  is_split_chip: " << (info.is_split_chip ? "true" : "false") << ",\n"
     << "  is_megacore: " << (info.is_megacore ? "true" : "false") << ",\n"
     << "}";
  return os;
}

}  // namespace torch_tpu
