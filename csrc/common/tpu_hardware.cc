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
#include "csrc/common/tpu_hardware.h"

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"  // IWYU pragma: keep

namespace torch_tpu {
namespace {

constexpr std::string_view kGooglePciVendorId = "0x1ae0";

// The result of listing a directory.
enum class DirListing { kListed, kMissing, kUnlistable };

// Calls `visit` with the name of each entry of `dir`. Returns whether `dir`
// was listed, does not exist, or exists but cannot be (fully) listed.
[[nodiscard]] DirListing ForEachDirEntry(
    std::string_view dir, absl::FunctionRef<void(std::string_view)> visit) {
  const std::unique_ptr<DIR, decltype(&closedir)> d(
      opendir(std::string(dir).c_str()), &closedir);
  if (d == nullptr) {
    return errno == ENOENT || errno == ENOTDIR ? DirListing::kMissing
                                               : DirListing::kUnlistable;
  }
  while (true) {
    // readdir() returns nullptr both at the end and on error; only the latter
    // sets errno.
    errno = 0;
    const dirent* const entry = readdir(d.get());
    if (entry == nullptr) {
      return errno == 0 ? DirListing::kListed : DirListing::kUnlistable;
    }
    visit(entry->d_name);
  }
}

// Returns the character device majors registered in `proc_devices`, which has
// the format of /proc/devices, or std::nullopt if it cannot be read.
[[nodiscard]] std::optional<std::vector<int>> ReadCharDeviceMajors(
    std::string_view proc_devices) {
  std::ifstream file{std::string(proc_devices)};
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::vector<int> majors;
  bool in_char_section = false;
  std::string line;
  while (std::getline(file, line)) {
    const std::string_view stripped = absl::StripAsciiWhitespace(line);
    if (stripped == "Character devices:") {
      in_char_section = true;
      continue;
    }
    // A blank line or another section header ("Block devices:") ends it.
    if (stripped.empty() || absl::EndsWith(stripped, ":")) {
      in_char_section = false;
      continue;
    }
    int major_number;
    if (in_char_section &&
        absl::SimpleAtoi(stripped.substr(0, stripped.find(' ')),
                         &major_number)) {
      majors.push_back(major_number);
    }
  }
  if (file.bad()) {
    return std::nullopt;
  }
  return majors;
}

// Returns true if `name` is the name of a VFIO group node, e.g. "12".
[[nodiscard]] bool IsVfioGroupNodeName(std::string_view name) {
  return !name.empty() && absl::c_all_of(name, absl::ascii_isdigit);
}

// Returns true if `name` is the name of an accel node, e.g. "accel0".
[[nodiscard]] bool IsAccelNodeName(std::string_view name) {
  return absl::StartsWith(name, "accel");
}

// Returns true if `name` is the name of a real directory entry, i.e. neither
// "." nor "..".
[[nodiscard]] bool IsRealEntryName(std::string_view name) {
  return name != "." && name != "..";
}

// Returns whether `vfio_dir` may hold a node through which libtpu could reach
// a TPU: a VFIO group node `<vfio_dir>/<number>` (not the container node
// `<vfio_dir>/vfio`), or any entry of `<vfio_dir>/devices`, which holds the
// VFIO per-device (iommufd) nodes such as `vfio0`. Also returns true if either
// directory exists but cannot be listed.
[[nodiscard]] bool HasVfioNodes(std::string_view vfio_dir) {
  bool has_vfio_node = false;
  if (ForEachDirEntry(
          vfio_dir,
          [&](std::string_view name) {  // REF_CAPTURE_OK=called synchronously
            has_vfio_node |= IsVfioGroupNodeName(name);
          }) == DirListing::kUnlistable ||
      has_vfio_node) {
    return true;
  }
  return ForEachDirEntry(absl::StrCat(vfio_dir, "/devices"),
                         [&]  // REF_CAPTURE_OK=called synchronously
                         (std::string_view name) {
                           has_vfio_node |= IsRealEntryName(name);
                         }) == DirListing::kUnlistable ||
         has_vfio_node;
}

// Appends to `majors` the major of each character device in `dir` whose name
// satisfies `is_candidate`. Returns whether `dir` was listed, does not exist,
// or exists but cannot be (fully) listed.
[[nodiscard]] DirListing AppendCharDeviceMajors(
    std::string_view dir,
    absl::FunctionRef<bool(std::string_view)> is_candidate,
    std::vector<int>& majors) {
  return ForEachDirEntry(
      dir, [&](std::string_view name) {  // REF_CAPTURE_OK=called synchronously
        if (!is_candidate(name)) {
          return;
        }
        struct stat st;
        if (stat(absl::StrCat(dir, "/", name).c_str(), &st) == 0 &&
            S_ISCHR(st.st_mode)) {
          majors.push_back(static_cast<int>(major(st.st_rdev)));
        }
      });
}

// Reads the entire contents of `path` and returns it with leading and trailing
// ASCII whitespace stripped, or std::nullopt if `path` cannot be opened or
// read.
[[nodiscard]] std::optional<std::string> ReadTrimmedFile(
    std::string_view path) {
  std::ifstream file{std::string(path)};
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::string contents((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  if (file.bad()) {
    return std::nullopt;
  }
  return std::string(absl::StripAsciiWhitespace(contents));
}

// Returns the full paths of all entries in `pci_devices_dir` (excluding "." and
// "..") in lexicographical order, or an empty vector if `pci_devices_dir` does
// not exist or cannot be listed.
[[nodiscard]] std::vector<std::string> ListSortedPciDevicePaths(
    std::string_view pci_devices_dir) {
  std::vector<std::string> paths;
  (void)ForEachDirEntry(  // VOID_CAST_OK=missing/unlistable dir yields empty
      pci_devices_dir,
      [&](std::string_view name) {  // REF_CAPTURE_OK=called synchronously
        if (IsRealEntryName(name)) {
          paths.push_back(absl::StrCat(pci_devices_dir, "/", name));
        }
      });
  absl::c_sort(paths);
  return paths;
}

// Returns the `TpuSpec` matching the sysfs PCI device directory `device_path`
// if its `vendor` file matches Google's PCI vendor ID and its `device` file
// matches a supported TPU PCI device ID, or nullptr otherwise.
[[nodiscard]] const TpuSpec* absl_nullable MatchSupportedTpuDevice(
    std::string_view device_path) {
  const std::optional<std::string> vendor_id =
      ReadTrimmedFile(absl::StrCat(device_path, "/vendor"));
  if (!vendor_id.has_value() || *vendor_id != kGooglePciVendorId) {
    return nullptr;
  }
  const std::optional<std::string> device_id =
      ReadTrimmedFile(absl::StrCat(device_path, "/device"));
  if (!device_id.has_value()) {
    return nullptr;
  }
  const auto& spec_map = TpuSpec::SpecsByPciDeviceId();
  const auto it = spec_map.find(*device_id);
  return it != spec_map.end() ? it->second : nullptr;
}

// Returns whether `path` resolves to a character device that this process has
// read and write access to.
//
// Sandboxes may put placeholders at the paths of devices they don't allocate
// to the process. For example, a sandbox with fewer TPU chips than its host may
// have empty, unreadable regular files at the `/dev/vfio/<group>` paths of the
// other chips. Those are not usable, so they should not count as devices.
//
// This function does not open any node because only one process at a time can
// open a VFIO group. Opening a node could fail or prevent the TPU runtime of
// this or another process from claiming it.
[[nodiscard]] bool IsAccessibleCharDevice(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISCHR(st.st_mode) &&
         faccessat(AT_FDCWD, path.c_str(), R_OK | W_OK, AT_EACCESS) == 0;
}

// Result of scanning PCI devices for VFIO-exposed TPUs.
struct PciTpuScanResult {
  int64_t device_count = 0;
  const TpuSpec* absl_nullable spec = nullptr;
};

// Scans PCI devices under `pci_devices_dir` for supported Google TPUs whose
// IOMMU group is exposed under `vfio_dir` as a character device that this
// process can read and write. Returns the count of VFIO-exposed TPU devices and
// the `TpuSpec` of the first one found (or nullptr if none).
[[nodiscard]] PciTpuScanResult ScanPciTpus(std::string_view pci_devices_dir,
                                           std::string_view vfio_dir) {
  PciTpuScanResult result;
  for (const std::string& device_path :
       ListSortedPciDevicePaths(pci_devices_dir)) {
    const TpuSpec* absl_nullable const spec =
        MatchSupportedTpuDevice(device_path);
    if (spec == nullptr) {
      continue;
    }
    // Some devices have multiple TensorCores sharing the same bus.
    // We count only those that are exposed via VFIO.
    char link_buf[PATH_MAX];
    const ssize_t len =
        readlink(absl::StrCat(device_path, "/iommu_group").c_str(), link_buf,
                 sizeof(link_buf));
    if (len <= 0 || static_cast<size_t>(len) >= sizeof(link_buf)) {
      continue;
    }
    const std::string_view group_target(link_buf, static_cast<size_t>(len));
    const size_t slash_pos = group_target.rfind('/');
    const std::string_view group_id = slash_pos == std::string_view::npos
                                          ? group_target
                                          : group_target.substr(slash_pos + 1);
    if (group_id.empty()) {
      continue;
    }
    if (!IsAccessibleCharDevice(absl::StrCat(vfio_dir, "/", group_id))) {
      continue;
    }
    ++result.device_count;
    if (result.spec == nullptr) {
      result.spec = spec;
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<TpuSpec::Topology> TpuSpec::GetTopology(
    std::optional<int64_t> world_size, std::string_view pci_devices_dir,
    std::string_view vfio_dir) const {
  const int64_t count = world_size.has_value()
                            ? *world_size
                            : GetTpuDeviceCount(pci_devices_dir, vfio_dir);
  const auto it = topologies.find(count);
  // torchtpu_agents' engine_compatibility_check.py matches this message.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=covered in tpu_hardware_test.
      it != topologies.end(), error::kInvalidArgument)
      << "No TPU topology found for count: " << count;
  return it->second;
}

// HBM and bf16 FLOP/s sources:
// - v5e, v5p, v6e, v7: https://cloud.google.com/tpu/docs/{v5e,v5p,v6e,tpu7x}
// - v8i, v8t: `hbm_capacity_bytes` and `bf16_ops_per_second` (per TensorCore)
//   times TensorCores per chip in
//   https://github.com/jax-ml/jax/blob/main/jax/_src/tpu_info.py
// SparseCore specs come from `SparseCoreInfo` in
// https://github.com/jax-ml/jax/blob/main/jax/_src/tpu_info.py.
const TpuSpec& TpuSpec::V5p() {
  static const absl::NoDestructor<TpuSpec> kSpec(TpuSpec{
      .name = "TPU v5p",
      .devices_per_chip = 1,
      .hbm_bytes_per_chip = 95LL * 1024 * 1024 * 1024,
      .bf16_flops_per_chip = 459'000'000'000'000LL,
      .topologies =
          {
              {1, {1, 1, 1}},
              {2, {1, 2, 1}},
              {4, {2, 2, 1}},
              {8, {2, 2, 2}},
          },
      .sparse_core_spec =
          SparseCoreSpec{.sparse_cores_per_device = 4, .simd_width = 8},
  });
  return *kSpec;
}

const TpuSpec& TpuSpec::V5e() {
  static const absl::NoDestructor<TpuSpec> kSpec(TpuSpec{
      .name = "TPU v5e",
      .devices_per_chip = 1,
      .hbm_bytes_per_chip = 16LL * 1024 * 1024 * 1024,
      .bf16_flops_per_chip = 197'000'000'000'000LL,
      .topologies =
          {
              {1, {1, 1, 1}},
              {4, {2, 2, 1}},
              // TPU v5e (8-chip) topology differs between environments:
              // - Cloud TPU VMs: The physical hardware interconnect is wired as
              //   a 2D mesh (2x4):
              //   https://docs.cloud.google.com/tpu/docs/v5e#tpu-v5e-config
              // - Internal test environments: A virtual 3D cube mapping (2x2x2)
              //   is used to prevent coordinate collisions during local
              //   discovery in the absence of physical wrap cabling.
              {8, TT_IS_INTERNAL_TORCH_TPU ? Topology{2, 2, 2}
                                           : Topology{2, 4, 1}},
          },
      .sparse_core_spec = std::nullopt,
  });
  return *kSpec;
}

const TpuSpec& TpuSpec::V6e() {
  static const absl::NoDestructor<TpuSpec> kSpec(TpuSpec{
      .name = "TPU v6e",
      .devices_per_chip = 1,
      .hbm_bytes_per_chip = 32LL * 1024 * 1024 * 1024,
      .bf16_flops_per_chip = 918'000'000'000'000LL,
      .topologies =
          {
              {1, {1, 1, 1}},
              {4, {2, 2, 1}},
              {8, {2, 4, 1}},
          },
      .sparse_core_spec =
          SparseCoreSpec{.sparse_cores_per_device = 2, .simd_width = 8},
  });
  return *kSpec;
}

const TpuSpec& TpuSpec::V7() {
  static const absl::NoDestructor<TpuSpec> kSpec(TpuSpec{
      .name = "TPU v7",
      .devices_per_chip = 2,
      .hbm_bytes_per_chip = 192LL * 1024 * 1024 * 1024,
      .bf16_flops_per_chip = 2'307'000'000'000'000LL,
      .topologies =
          {
              {2, {1, 1, 1, 2}},
              {4, {1, 2, 1, 2}},
              {8, {2, 2, 1, 2}},
          },
      .sparse_core_spec =
          SparseCoreSpec{.sparse_cores_per_device = 2, .simd_width = 16},
  });
  return *kSpec;
}

const TpuSpec& TpuSpec::V8i() {
  static const absl::NoDestructor<TpuSpec> kSpec(TpuSpec{
      .name = "TPU v8i",
      .devices_per_chip = 2,
      .hbm_bytes_per_chip = 309'000'000'000LL,
      .bf16_flops_per_chip = 1'101'000'000'000'000LL,
      .topologies =
          {
              {2, {1, 1, 1, 2}},
              {4, {2, 1, 1, 2}},
              {8, {4, 1, 1, 2}},
          },
      .sparse_core_spec =
          SparseCoreSpec{.sparse_cores_per_device = 1, .simd_width = 16},
  });
  return *kSpec;
}

const TpuSpec& TpuSpec::V8t() {
  static const absl::NoDestructor<TpuSpec> kSpec(TpuSpec{
      .name = "TPU v8t",
      .devices_per_chip = 1,
      .hbm_bytes_per_chip = 231'000'000'000LL,
      .bf16_flops_per_chip = 996'100'000'000'000LL,
      .topologies =
          {
              {1, {1, 1, 1}},
              {2, {1, 2, 1}},
              {4, {2, 2, 1}},
          },
      .sparse_core_spec =
          SparseCoreSpec{.sparse_cores_per_device = 2, .simd_width = 16},
  });
  return *kSpec;
}

const absl::flat_hash_map<std::string, const TpuSpec* absl_nonnull>&
TpuSpec::SpecsByPciDeviceId() {
  // Verified IDs from util/platforminfo/pci_ids.h.
  static const absl::NoDestructor<
      absl::flat_hash_map<std::string, const TpuSpec* absl_nonnull>>
      kMap({
          {"0x0062", &TpuSpec::V5p()},
          {"0x0063", &TpuSpec::V5e()},
          {"0x006f", &TpuSpec::V6e()},
          {"0x0076", &TpuSpec::V7()},
          {"0x007c", &TpuSpec::V8t()},
          {"0x0083", &TpuSpec::V8i()},
      });
  return *kMap;
}

const TpuSpec* absl_nullable TpuSpec::Get(std::string_view pci_devices_dir) {
  for (const std::string& device_path :
       ListSortedPciDevicePaths(pci_devices_dir)) {
    if (const TpuSpec* absl_nullable const spec =
            MatchSupportedTpuDevice(device_path);
        spec != nullptr) {
      return spec;
    }
  }
  return nullptr;
}

int64_t GetTpuDeviceCount(std::string_view pci_devices_dir,
                          std::string_view vfio_dir) {
  return ScanPciTpus(pci_devices_dir, vfio_dir).device_count;
}

absl::StatusOr<std::optional<std::string>> GetTpuTopology(
    std::optional<int64_t> world_size, std::string_view pci_devices_dir,
    std::string_view vfio_dir) {
  const PciTpuScanResult scan = ScanPciTpus(pci_devices_dir, vfio_dir);
  if (scan.spec == nullptr) {
    return std::nullopt;
  }
  const int64_t target_count =
      world_size.has_value() ? *world_size : scan.device_count;
  TT_ASSIGN_OR_RETURN(
      const TpuSpec::Topology topology,
      scan.spec->GetTopology(target_count, pci_devices_dir, vfio_dir));
  return absl::StrJoin(topology, ",");
}

TpuSpec::Topology GetSingleDeviceTpuBounds(std::string_view pci_devices_dir) {
  const TpuSpec* absl_nullable const spec = TpuSpec::Get(pci_devices_dir);
  return (spec != nullptr && spec->devices_per_chip > 1)
             ? TpuSpec::Topology{1, 1, 1, 1}
             : TpuSpec::Topology{1, 1, 1};
}

std::string GetSingleDeviceTpuBoundsString(std::string_view pci_devices_dir) {
  return absl::StrJoin(GetSingleDeviceTpuBounds(pci_devices_dir), ",");
}

bool HasTpuDeviceNodes(std::string_view dev_dir, std::string_view dev_borg_dir,
                       std::string_view proc_devices) {
  if (HasVfioNodes(absl::StrCat(dev_dir, "/vfio"))) {
    return true;
  }

  // Accel nodes count only if they are character devices backed by a driver,
  // that is, whose major is registered in /proc/devices: some static /dev trees
  // (e.g. in sandboxed build and test environments) contain hundreds of
  // driverless /dev/accel* nodes. The kernel's compute accelerator subsystem
  // puts its nodes in an `accel` directory instead, e.g. /dev/accel/accel0.
  std::vector<int> accel_majors;
  for (const std::string_view dir : {dev_dir, dev_borg_dir}) {
    if (AppendCharDeviceMajors(dir, IsAccelNodeName, accel_majors) ==
            DirListing::kUnlistable ||
        AppendCharDeviceMajors(absl::StrCat(dir, "/accel"), IsRealEntryName,
                               accel_majors) == DirListing::kUnlistable) {
      return true;
    }
  }
  if (accel_majors.empty()) {
    return false;
  }
  const std::optional<std::vector<int>> registered_majors =
      ReadCharDeviceMajors(proc_devices);
  if (!registered_majors.has_value()) {
    return true;
  }
  return absl::c_any_of(
      accel_majors,
      [&](const int accel_major) {  // REF_CAPTURE_OK=called synchronously
        return absl::c_linear_search(*registered_majors, accel_major);
      });
}

}  // namespace torch_tpu
