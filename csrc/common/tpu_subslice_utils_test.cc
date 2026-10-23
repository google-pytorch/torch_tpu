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

#include "csrc/common/tpu_subslice_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOkAndHolds;
using testing::Eq;
using testing::Optional;

// Fake single-slice 2x2x1 TPU topology (4 chips, 1 core per chip = 4 devices)
// that rejects logical device IDs outside [0, num_devices).
class FakeTpuTopologyDescription : public xla::PjRtTopologyDescription {
 public:
  explicit FakeTpuTopologyDescription(
      xla::PjRtDeviceDimensions chip_bounds = {2, 2, 1}, int cores_per_chip = 1)
      : chip_bounds_(chip_bounds), cores_per_chip_(cores_per_chip) {}

  xla::PjRtPlatformId platform_id() const override { return xla::TpuId(); }
  std::string_view platform_name() const override { return xla::TpuName(); }
  std::string_view platform_version() const override { return "test"; }

  std::vector<std::unique_ptr<const xla::PjRtDeviceDescription>>
  DeviceDescriptions() const override {
    return {};
  }

  absl::StatusOr<uint64_t> Fingerprint() const override { return 0; }

  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  absl::StatusOr<xla::Layout> GetDefaultLayout(
      xla::PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return xla::Layout();
  }

  absl::StatusOr<int> LogicalDeviceCountOfDefaultType() const override {
    return num_devices();
  }

  absl::StatusOr<int> LogicalDeviceCountOfDefaultTypePerChip() const override {
    return cores_per_chip_;
  }

  absl::StatusOr<std::pair<xla::PjRtDeviceDimensions, int32_t>>
  ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
      xla::GlobalDeviceId device_id) const override {
    const int64_t id = device_id.value();
    if (id < 0 || id >= num_devices()) {
      return TT_ERROR(error::kInvalidArgument) << "Invalid device id: " << id;
    }
    const int32_t core = static_cast<int32_t>(id % cores_per_chip_);
    int64_t chip_idx = id / cores_per_chip_;
    const int32_t x = static_cast<int32_t>(chip_idx % chip_bounds_[0]);
    chip_idx /= chip_bounds_[0];
    const int32_t y = static_cast<int32_t>(chip_idx % chip_bounds_[1]);
    const int32_t z = static_cast<int32_t>(chip_idx / chip_bounds_[1]);
    return std::make_pair(xla::PjRtDeviceDimensions{x, y, z}, core);
  }

  absl::StatusOr<xla::GlobalDeviceId>
  LogicalDeviceOfDefaultTypeIdFromChipCoordAndCoreIndex(
      const xla::PjRtDeviceDimensions& chip, int core_index) const override {
    const int64_t chip_idx =
        (chip[2] * chip_bounds_[1] + chip[1]) * chip_bounds_[0] + chip[0];
    return xla::GlobalDeviceId(chip_idx * cores_per_chip_ + core_index);
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtTopologyDescription>> Subslice(
      const xla::PjRtDeviceDimensions& chips_per_host_bounds,
      const xla::PjRtDeviceDimensions& host_bounds) const override {
    return std::make_unique<FakeTpuTopologyDescription>(host_bounds,
                                                        cores_per_chip_);
  }

 private:
  int num_devices() const {
    return chip_bounds_[0] * chip_bounds_[1] * chip_bounds_[2] *
           cores_per_chip_;
  }

  xla::PjRtDeviceDimensions chip_bounds_;
  int cores_per_chip_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

class FakeTpuClient : public xla::PjRtClient {
 public:
  explicit FakeTpuClient(xla::PjRtDeviceDimensions chip_bounds = {2, 2, 1})
      : topology_(chip_bounds) {}

  int process_index() const override { return 0; }
  int device_count() const override {
    return *topology_.LogicalDeviceCountOfDefaultType();
  }
  int addressable_device_count() const override { return 1; }
  absl::Span<xla::PjRtDevice* const> devices() const override { return {}; }
  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return {};
  }
  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    return {};
  }
  xla::PjRtPlatformId platform_id() const override { return xla::TpuId(); }
  std::string_view platform_name() const override { return xla::TpuName(); }
  std::string_view platform_version() const override { return "test"; }

  absl::StatusOr<const xla::PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return &topology_;
  }

 private:
  FakeTpuTopologyDescription topology_;
};

xla::CompileOptions MakeCompileOptions(
    absl::Span<const int64_t> partition_device_ids) {
  xla::CompileOptions options;
  xla::DeviceAssignment da(1, partition_device_ids.size());
  for (size_t p = 0; p < partition_device_ids.size(); ++p) {
    da(0, p) = partition_device_ids[p];
  }
  options.executable_build_options.set_device_assignment(da);
  return options;
}

TEST(TpuSubsliceUtilsTest, SingleSliceSubgroupReturnsSubsliceOrigin) {
  FakeTpuClient client({2, 2, 1});
  // Devices 2 and 3 occupy chips (0, 1, 0) and (1, 1, 0), forming a 2x1x1
  // subslice with origin (0, 1, 0).
  const xla::CompileOptions options = MakeCompileOptions({2, 3});
  EXPECT_THAT(GetSubsliceOrigin(client, options),
              IsOkAndHolds(Optional(Eq(xla::PjRtDeviceDimensions{0, 1, 0}))));
}

TEST(TpuSubsliceUtilsTest, FullSingleSliceReturnsNullopt) {
  FakeTpuClient client({2, 2, 1});
  const xla::CompileOptions options = MakeCompileOptions({0, 1, 2, 3});
  EXPECT_THAT(GetSubsliceOrigin(client, options),
              IsOkAndHolds(Eq(std::nullopt)));
}

TEST(TpuSubsliceUtilsTest, MultiSliceWorldAssignmentReturnsNullopt) {
  // In a multi-slice setup, GetTopologyDescription() describes a single slice
  // (4 devices) while a 2-slice world program has 8 partitions with
  // slice-offset device IDs (100000..100003 on slice 0, 200000..200003 on slice
  // 1).
  FakeTpuClient client({2, 2, 1});
  const xla::CompileOptions options = MakeCompileOptions(
      {100000, 100001, 100002, 100003, 200000, 200001, 200002, 200003});
  EXPECT_THAT(GetSubsliceOrigin(client, options),
              IsOkAndHolds(Eq(std::nullopt)));
}

TEST(TpuSubsliceUtilsTest,
     MultiSliceSubgroupWithOffsetDeviceIdsReturnsNullopt) {
  // Even when a subgroup has fewer partitions (2) than the single-slice device
  // count (4), multi-slice device IDs outside [0, num_devices) must not be
  // looked up in the single-slice topology description.
  FakeTpuClient client({2, 2, 1});
  EXPECT_THAT(GetSubsliceOrigin(client, MakeCompileOptions({100000, 200000})),
              IsOkAndHolds(Eq(std::nullopt)));
  EXPECT_THAT(GetSubsliceOrigin(client, MakeCompileOptions({100000, 100001})),
              IsOkAndHolds(Eq(std::nullopt)));
}

}  // namespace
}  // namespace torch_tpu
