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

#include <cstddef>
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
#include "xla/pjrt/maybe_owning_mlir_module.h"
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

using absl_testing::IsOk;
using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using testing::Eq;
using testing::Optional;

// Fake single-slice 2x2x1 TPU topology (4 chips, 1 core per chip = 4 devices)
// that rejects logical device IDs outside [0, num_devices).
class FakeTpuTopologyDescription : public xla::PjRtTopologyDescription {
 public:
  explicit FakeTpuTopologyDescription(
      xla::PjRtDeviceDimensions chip_bounds = {2, 2, 1}, int cores_per_chip = 1,
      xla::PjRtCompiler* compiler = nullptr)
      : chip_bounds_(chip_bounds),
        cores_per_chip_(cores_per_chip),
        compiler_(compiler) {}

  std::optional<xla::PjRtCompiler*> compiler() const override {
    if (compiler_ == nullptr) {
      return std::nullopt;
    }
    return compiler_;
  }

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
    return std::make_unique<FakeTpuTopologyDescription>(
        host_bounds, cores_per_chip_, compiler_);
  }

  int num_devices() const {
    return chip_bounds_[0] * chip_bounds_[1] * chip_bounds_[2] *
           cores_per_chip_;
  }

 private:
  xla::PjRtDeviceDimensions chip_bounds_;
  int cores_per_chip_;
  xla::PjRtCompiler* compiler_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

class FakeTpuClient : public xla::PjRtClient {
 public:
  explicit FakeTpuClient(xla::PjRtDeviceDimensions chip_bounds = {2, 2, 1},
                         xla::PjRtCompiler* compiler = nullptr)
      : topology_(chip_bounds, /*cores_per_chip=*/1, compiler) {}

  int process_index() const override { return 0; }
  int device_count() const override { return topology_.num_devices(); }
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

class FakePjRtExecutable final : public xla::PjRtExecutable {
 public:
  FakePjRtExecutable(bool& destroyed, std::optional<std::string> serialized)
      : destroyed_(destroyed), serialized_(std::move(serialized)) {}

  ~FakePjRtExecutable() override { destroyed_ = true; }

  int num_replicas() const override { return 1; }
  int num_partitions() const override { return 1; }
  int64_t SizeOfGeneratedCodeInBytes() const override { return 0; }
  std::string_view name() const override { return "fake"; }
  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetParameterMemoryKinds() const override {
    return std::vector<std::vector<std::string_view>>{};
  }
  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const override {
    return std::vector<std::vector<std::string_view>>{};
  }
  absl::StatusOr<std::string> SerializeExecutable() const override {
    if (!serialized_.has_value()) {
      return TT_ERROR(error::kInternal) << "Serialize failed.";
    }
    return *serialized_;
  }

 private:
  bool& destroyed_;
  std::optional<std::string> serialized_;
};

class FakePjRtCompiler final : public xla::PjRtCompiler {
 public:
  explicit FakePjRtCompiler(
      bool& executable_destroyed,
      std::string serialized_output = "fake-subslice-executable")
      : executable_destroyed_(executable_destroyed),
        serialized_output_(std::move(serialized_output)) {}

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      xla::CompileOptions options, const xla::XlaComputation& computation,
      const xla::PjRtTopologyDescription& topology,
      xla::PjRtClient* client) override {
    return TT_ERROR(error::kInternal)
           << "XlaComputation compile is not implemented.";
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      xla::CompileOptions options, xla::MaybeOwningMlirModule module,
      const xla::PjRtTopologyDescription& topology,
      xla::PjRtClient* client) override {
    ++compile_calls_;
    if (options.executable_build_options.has_device_assignment()) {
      last_device_assignment_ =
          options.executable_build_options.device_assignment();
    }
    executable_destroyed_ = false;
    return std::make_unique<FakePjRtExecutable>(executable_destroyed_,
                                                serialized_output_);
  }

  int compile_calls() const { return compile_calls_; }
  const std::optional<xla::DeviceAssignment>& last_device_assignment() const {
    return last_device_assignment_;
  }

 private:
  bool& executable_destroyed_;
  std::string serialized_output_;
  int compile_calls_ = 0;
  std::optional<xla::DeviceAssignment> last_device_assignment_;
};

class FakeCompileClient final : public FakeTpuClient {
 public:
  struct CallLog {
    int compile_calls = 0;
    int compile_and_load_calls = 0;
    int load_serialized_calls = 0;
    bool executable_destroyed_before_load = false;
    std::string last_serialized;
    std::optional<xla::PjRtDeviceDimensions> last_computation_origin;
  };

  explicit FakeCompileClient(
      xla::PjRtPlatformId platform_id = xla::TpuId(),
      std::string_view platform_name = xla::TpuName(),
      xla::PjRtCompiler* compiler = nullptr,
      std::optional<int> device_count_override = std::nullopt,
      bool* external_executable_destroyed = nullptr)
      : FakeTpuClient({2, 2, 1}, compiler),
        platform_id_(platform_id),
        platform_name_(platform_name),
        device_count_override_(device_count_override),
        external_executable_destroyed_(external_executable_destroyed) {}

  int device_count() const override {
    return device_count_override_.value_or(FakeTpuClient::device_count());
  }
  xla::PjRtPlatformId platform_id() const override { return platform_id_; }
  std::string_view platform_name() const override { return platform_name_; }

  using FakeTpuClient::Compile;
  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      xla::MaybeOwningMlirModule module, xla::CompileOptions options) override {
    ++log_.compile_calls;
    executable_destroyed_ = false;
    return std::make_unique<FakePjRtExecutable>(executable_destroyed_,
                                                serialized_output_);
  }

  using FakeTpuClient::CompileAndLoad;
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileAndLoad(
      xla::MaybeOwningMlirModule module, xla::CompileOptions options) override {
    ++log_.compile_and_load_calls;
    return nullptr;
  }

  using FakeTpuClient::LoadSerializedExecutable;
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
  LoadSerializedExecutable(std::string_view serialized,
                           std::optional<xla::CompileOptions> options,
                           const xla::LoadOptions& load_options) override {
    ++log_.load_serialized_calls;
    log_.executable_destroyed_before_load =
        external_executable_destroyed_ != nullptr
            ? *external_executable_destroyed_
            : executable_destroyed_;
    log_.last_serialized = std::string(serialized);
    log_.last_computation_origin = load_options.computation_origin;
    return nullptr;
  }

  void set_serialized_output(std::optional<std::string> serialized_output) {
    serialized_output_ = std::move(serialized_output);
  }

  const CallLog& log() const { return log_; }

 private:
  xla::PjRtPlatformId platform_id_;
  std::string platform_name_;
  std::optional<int> device_count_override_;
  bool* external_executable_destroyed_;
  std::optional<std::string> serialized_output_ = "fake-serialized-executable";
  bool executable_destroyed_ = false;
  CallLog log_;
};

TEST(TpuSubsliceUtilsTest,
     FullSliceTpuSerializesDropsExecutableAndLoadsFromBytes) {
  FakeCompileClient client;
  EXPECT_THAT(CompileAndLoadMaybeSubslice(client, xla::MaybeOwningMlirModule{},
                                          MakeCompileOptions({0, 1, 2, 3})),
              IsOk());
  EXPECT_EQ(client.log().compile_calls, 1);
  EXPECT_EQ(client.log().compile_and_load_calls, 0);
  EXPECT_EQ(client.log().load_serialized_calls, 1);
  EXPECT_TRUE(client.log().executable_destroyed_before_load);
  EXPECT_EQ(client.log().last_serialized, "fake-serialized-executable");
  EXPECT_EQ(client.log().last_computation_origin, std::nullopt);
}

TEST(TpuSubsliceUtilsTest,
     SubsliceTpuSerializesDropsExecutableAndLoadsAtOrigin) {
  bool executable_destroyed = false;
  FakePjRtCompiler compiler(executable_destroyed);
  FakeCompileClient client(xla::TpuId(), xla::TpuName(), &compiler,
                           /*device_count_override=*/std::nullopt,
                           &executable_destroyed);
  EXPECT_THAT(CompileAndLoadMaybeSubslice(client, xla::MaybeOwningMlirModule{},
                                          MakeCompileOptions({2, 3})),
              IsOk());
  EXPECT_EQ(compiler.compile_calls(), 1);
  ASSERT_TRUE(compiler.last_device_assignment().has_value());
  EXPECT_EQ(compiler.last_device_assignment()->replica_count(), 1);
  EXPECT_EQ(compiler.last_device_assignment()->computation_count(), 2);
  EXPECT_EQ((*compiler.last_device_assignment())(0, 0), 0);
  EXPECT_EQ((*compiler.last_device_assignment())(0, 1), 1);
  EXPECT_EQ(client.log().compile_calls, 0);
  EXPECT_EQ(client.log().compile_and_load_calls, 0);
  EXPECT_EQ(client.log().load_serialized_calls, 1);
  EXPECT_TRUE(client.log().executable_destroyed_before_load);
  EXPECT_EQ(client.log().last_serialized, "fake-subslice-executable");
  EXPECT_THAT(client.log().last_computation_origin,
              Optional(Eq(xla::PjRtDeviceDimensions{0, 1, 0})));
}

TEST(TpuSubsliceUtilsTest, MultiSliceMegaScaleTpuFallsBackToCompileAndLoad) {
  FakeCompileClient client(xla::TpuId(), xla::TpuName(), /*compiler=*/nullptr,
                           /*device_count_override=*/8);
  EXPECT_THAT(CompileAndLoadMaybeSubslice(
                  client, xla::MaybeOwningMlirModule{},
                  MakeCompileOptions({100000, 100001, 100002, 100003, 200000,
                                      200001, 200002, 200003})),
              IsOk());
  EXPECT_EQ(client.log().compile_calls, 0);
  EXPECT_EQ(client.log().compile_and_load_calls, 1);
  EXPECT_EQ(client.log().load_serialized_calls, 0);
}

TEST(TpuSubsliceUtilsTest, TpuSerializeFailureSkipsLoadSerializedExecutable) {
  FakeCompileClient client;
  client.set_serialized_output(std::nullopt);
  EXPECT_THAT(CompileAndLoadMaybeSubslice(client, xla::MaybeOwningMlirModule{},
                                          MakeCompileOptions({0, 1, 2, 3})),
              StatusIs(error::kInternal));
  EXPECT_EQ(client.log().compile_calls, 1);
  EXPECT_EQ(client.log().compile_and_load_calls, 0);
  EXPECT_EQ(client.log().load_serialized_calls, 0);
}

TEST(TpuSubsliceUtilsTest, NonTpuClientUsesCompileAndLoadDirectly) {
  FakeCompileClient client(xla::CpuId(), xla::CpuName());
  EXPECT_THAT(CompileAndLoadMaybeSubslice(client, xla::MaybeOwningMlirModule{},
                                          MakeCompileOptions({0, 1, 2, 3})),
              IsOk());
  EXPECT_EQ(client.log().compile_calls, 0);
  EXPECT_EQ(client.log().compile_and_load_calls, 1);
  EXPECT_EQ(client.log().load_serialized_calls, 0);
}

}  // namespace
}  // namespace torch_tpu
