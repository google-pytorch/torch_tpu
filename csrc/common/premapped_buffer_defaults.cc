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
#include "csrc/common/premapped_buffer_defaults.h"

#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/tpu_hardware.h"

namespace torch_tpu {
namespace {

// Cgroup v1 reports sentinel limits near INT64_MAX (commonly
// PAGE_COUNTER_MAX * PAGE_SIZE ~= 8 EiB) when unlimited. Treat anything at or
// above 1 EiB as unlimited.
constexpr int64_t kUnlimitedCgroupMemoryBytes = int64_t{1} << 60;
constexpr int64_t kMaxDefaultPremappedBufferBytes = int64_t{4} << 30;
constexpr int64_t kMinPerWorkerMemoryBytes = int64_t{16} << 30;
constexpr int kMaxCgroupHierarchyDepth = 64;

void UpdateMinLimit(std::optional<int64_t>& current,
                    const std::optional<int64_t> candidate) {
  if (!candidate.has_value() || *candidate <= 0) {
    return;
  }
  current = current.has_value() ? std::min(*current, *candidate) : *candidate;
}

[[nodiscard]] bool HasCommaToken(std::string_view csv, std::string_view token) {
  for (const std::string_view item : absl::StrSplit(csv, ',')) {
    if (item == token) {
      return true;
    }
  }
  return false;
}

// Cgroup paths of this process, relative to the cgroup filesystem roots.
struct ProcessCgroupPaths {
  // Path in the cgroup v2 unified hierarchy.
  std::string v2_rel_path;
  // Path in the cgroup v1 memory controller hierarchy.
  std::string v1_memory_rel_path;
};

[[nodiscard]] ProcessCgroupPaths ParseProcessCgroupPaths(
    std::string_view proc_self_cgroup) {
  ProcessCgroupPaths paths;
  std::ifstream file{std::string(proc_self_cgroup)};
  if (!file.is_open()) {
    return paths;
  }
  std::string line;
  while (std::getline(file, line)) {
    const std::vector<std::string_view> fields = absl::StrSplit(
        absl::StripTrailingAsciiWhitespace(line), absl::MaxSplits(':', 2));
    if (fields.size() != 3) {
      continue;
    }
    if (fields[0] == "0" && fields[1].empty()) {
      paths.v2_rel_path = std::string(fields[2]);
    } else if (HasCommaToken(fields[1], "memory")) {
      paths.v1_memory_rel_path = std::string(fields[2]);
    }
  }
  return paths;
}

[[nodiscard]] bool HasParentTraversalSegment(std::string_view rel_path) {
  for (const std::string_view segment : absl::StrSplit(rel_path, '/')) {
    if (segment == "..") {
      return true;
    }
  }
  return false;
}

[[nodiscard]] std::optional<int64_t> ParseCgroupLimitValue(
    std::string_view text, const bool is_v2) {
  if (is_v2 && text == "max") {
    return std::nullopt;
  }
  int64_t val = 0;
  if (!absl::SimpleAtoi(text, &val) || val <= 0) {
    return std::nullopt;
  }
  if (!is_v2 && val >= kUnlimitedCgroupMemoryBytes) {
    return std::nullopt;
  }
  return val;
}

[[nodiscard]] std::optional<int64_t> FindMinLimitAlongHierarchy(
    std::string_view root, std::string_view rel_path,
    std::string_view limit_filename, const bool is_v2) {
  while (root.size() > 1 && root.back() == '/') {
    root.remove_suffix(1);
  }
  if (root.empty()) {
    return std::nullopt;
  }
  while (!rel_path.empty() && rel_path.front() == '/') {
    rel_path.remove_prefix(1);
  }
  while (!rel_path.empty() && rel_path.back() == '/') {
    rel_path.remove_suffix(1);
  }

  std::string current_dir(root);
  if (!rel_path.empty() && !HasParentTraversalSegment(rel_path)) {
    current_dir = root == "/" ? absl::StrCat("/", rel_path)
                              : absl::StrCat(root, "/", rel_path);
  }

  std::optional<int64_t> min_limit;
  for (int depth = 0;
       depth < kMaxCgroupHierarchyDepth && current_dir.size() >= root.size();
       ++depth) {
    const std::string limit_path =
        current_dir == "/" ? absl::StrCat("/", limit_filename)
                           : absl::StrCat(current_dir, "/", limit_filename);
    if (const std::optional<std::string> raw = ReadTrimmedFile(limit_path);
        raw.has_value()) {
      UpdateMinLimit(min_limit, ParseCgroupLimitValue(*raw, is_v2));
    }
    if (current_dir == root) {
      break;
    }
    const size_t slash_pos = current_dir.rfind('/');
    if (slash_pos == std::string::npos) {
      break;
    }
    current_dir.resize(slash_pos == 0 ? 1 : slash_pos);
  }
  return min_limit;
}

[[nodiscard]] std::optional<int64_t> GetCgroupMemoryLimitBytes(
    const MemoryLimitProbePaths& paths) {
  const ProcessCgroupPaths cgroup_paths =
      ParseProcessCgroupPaths(paths.proc_self_cgroup);

  std::optional<int64_t> min_limit;
  UpdateMinLimit(min_limit, FindMinLimitAlongHierarchy(
                                paths.cgroup_v2_root, cgroup_paths.v2_rel_path,
                                "memory.max", /*is_v2=*/true));
  for (const std::string_view v1_root :
       {paths.cgroup_v1_borg_root, paths.cgroup_v1_sysfs_root}) {
    UpdateMinLimit(min_limit, FindMinLimitAlongHierarchy(
                                  v1_root, cgroup_paths.v1_memory_rel_path,
                                  "memory.limit_in_bytes",
                                  /*is_v2=*/false));
  }
  return min_limit;
}

[[nodiscard]] std::optional<int64_t> GetPhysicalMemoryBytes() {
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0) {
    return std::nullopt;
  }
  const int64_t pages64 = static_cast<int64_t>(pages);
  const int64_t page_size64 = static_cast<int64_t>(page_size);
  if (pages64 > std::numeric_limits<int64_t>::max() / page_size64) {
    return std::nullopt;
  }
  return pages64 * page_size64;
}

// Reads an integer env var without memoizing it. The default is computed
// during PjRt initialization, and memoizing distributed env vars here (e.g.
// with GetIntegerEnvOnce<>()) would freeze them before process group creation
// reads them, ignoring any later updates.
[[nodiscard]] std::optional<int64_t> ReadIntegerEnvWithoutMemoizing(
    const char* name) {
  const char* const raw = std::getenv(name);  // GETENV_OK=must not memoize.
  int64_t value = 0;
  if (raw == nullptr || !absl::SimpleAtoi(raw, &value)) {
    return std::nullopt;
  }
  return value;
}

}  // namespace

std::optional<int64_t> GetEffectiveMemoryLimitBytes(
    const MemoryLimitProbePaths& paths) {
  std::optional<int64_t> limit;
  if (paths.physical_memory_bytes_override.has_value()) {
    UpdateMinLimit(limit, paths.physical_memory_bytes_override);
  } else {
    UpdateMinLimit(limit, GetPhysicalMemoryBytes());
  }
  UpdateMinLimit(limit, GetCgroupMemoryLimitBytes(paths));
  return limit;
}

int64_t EstimateLocalWorkerCount(const std::string_view pci_devices_dir,
                                 const std::string_view vfio_dir) {
  const std::optional<int64_t> world_size =
      ReadIntegerEnvWithoutMemoizing(kWorldSizeEnvVar);
  if (!world_size.has_value() || *world_size <= 1) {
    return 1;
  }
  if (const std::optional<int64_t> devices_per_host =
          ReadIntegerEnvWithoutMemoizing(kTorchTpuDevicesPerHostEnvVar);
      devices_per_host.has_value() && *devices_per_host > 0) {
    return std::min(*world_size, *devices_per_host);
  }
  if (const std::optional<int64_t> nnodes =
          ReadIntegerEnvWithoutMemoizing(kNnodesEnvVar);
      nnodes.has_value() && *nnodes > 1) {
    // Ceiling division that cannot overflow, since world_size > 1 and
    // nnodes > 1 here.
    const int64_t per_node = 1 + (*world_size - 1) / *nnodes;
    return std::max<int64_t>(1, per_node);
  }
  const int64_t tpu_devices = GetTpuDeviceCount(pci_devices_dir, vfio_dir);
  if (tpu_devices > 0) {
    return std::min(*world_size, tpu_devices);
  }
  return *world_size;
}

int64_t ComputeDefaultPremappedBufferSizeBytes(
    const std::optional<int64_t> memory_limit_bytes,
    const int64_t local_workers) {
  if (!memory_limit_bytes.has_value() || *memory_limit_bytes <= 0 ||
      local_workers <= 0) {
    return 0;
  }
  const int64_t per_worker_limit = *memory_limit_bytes / local_workers;
  if (per_worker_limit < kMinPerWorkerMemoryBytes) {
    return 0;
  }
  const int64_t budget =
      std::min(kMaxDefaultPremappedBufferBytes, per_worker_limit / 16);
  return static_cast<int64_t>(std::bit_floor(static_cast<uint64_t>(budget)));
}

}  // namespace torch_tpu
