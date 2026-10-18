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
#include <sys/stat.h>
#include <sys/sysmacros.h>

#include <cerrno>
#include <cstddef>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/error_utils.h"

namespace torch_tpu {
namespace {

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

}  // namespace

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

absl::StatusOr<bool> IsMultiDevicePerChipAcceleratorType(
    std::string_view accelerator_type) {
  const size_t dash_pos = accelerator_type.find('-');
  int device_count = 0;
  if (dash_pos == std::string_view::npos || dash_pos == 0 ||
      !absl::SimpleAtoi(accelerator_type.substr(dash_pos + 1), &device_count) ||
      device_count <= 0) {
    return TT_ERROR(error::kInvalidArgument)
           << "expected accelerator type in format "
              "'<generation>-<device_count>' (e.g., 'v4-8', 'v5litepod-4', "
              "'v7x-8'), got '"
           << accelerator_type << "'";
  }
  const std::string generation =
      absl::AsciiStrToLower(accelerator_type.substr(0, dash_pos));
  // TPU v7 (e.g. "v7x-8", "tpu7x-8") and TPU v8i (e.g. "v8i-8", "tpu8i-8")
  // have 2 devices per chip without megacore.
  if (generation == "v7" || generation == "v7x" || generation == "tpu7" ||
      generation == "tpu7x" || generation == "v8i" || generation == "tpu8i") {
    return true;
  }
  if (generation == "v4" || generation == "v5e" || generation == "v5litepod" ||
      generation == "v5p" || generation == "v6e" || generation == "v8t" ||
      generation == "tpu8t") {
    return false;
  }
  return TT_ERROR(error::kInvalidArgument)
         << "unrecognized TPU generation '" << generation
         << "' in accelerator type '" << accelerator_type << "'";
}

}  // namespace torch_tpu
