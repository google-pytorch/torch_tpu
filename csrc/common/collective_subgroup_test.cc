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

#include "csrc/common/collective_subgroup.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/computation_placer.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

constexpr std::string_view kPairAllReduce = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
        replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>,
        use_global_device_ids} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";

constexpr std::string_view kWorldAllGather = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<32xf32> {
    %0 = "stablehlo.all_gather"(%arg0) {all_gather_dim = 0 : i64,
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        use_global_device_ids} : (tensor<4xf32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}
)mlir";

constexpr std::string_view kRingPermute = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.collective_permute"(%arg0) {
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
        source_target_pairs = dense<[[0, 1], [1, 0], [2, 3], [3, 2]]> : tensor<4x2xi64>
        } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";

constexpr std::string_view kCrossPermute = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.collective_permute"(%arg0) {
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
        source_target_pairs = dense<[[0, 2], [2, 4], [4, 6], [6, 0]]> : tensor<4x2xi64>
        } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";

constexpr std::string_view kRaggedAllToAll = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.custom_call"(%arg0) {
        call_target_name = "ragged_all_to_all"
        } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";

// An inlined kernel keeps the shardings of its partitioning over the slice:
// here a (stage, pair) mesh of eight devices in rows (0,1), (2,3), (4,5),
// (6,7), with the argument sharded over the pair axis and the add replicated.
constexpr std::string_view kShardedKernel = R"mlir(
module attributes {mhlo.spmd_output_sharding = "{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} {
  func.func @main(%arg0: tensor<8x4xf32> {mhlo.sharding = "{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}"}) -> (tensor<8x4xf32> {mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.add %arg0, %arg0 {mhlo.sharding = "{devices=[1,8]<=[8] last_tile_dim_replicate}"} : tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}
)mlir";

xla::HloSharding Sharding(std::string_view text) {
  auto sharding = xla::ParseSharding(text);
  EXPECT_THAT(sharding, IsOk()) << text;
  return *sharding;
}

class CollectiveSubgroupTest : public testing::Test {
 protected:
  CollectiveSubgroupTest() {
    context_.loadDialect<mlir::stablehlo::StablehloDialect>();
    context_.loadDialect<mlir::func::FuncDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> Parse(std::string_view text) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(text, &context_);
    EXPECT_TRUE(module);
    return module;
  }

  // The rows of the named attribute of the first op carrying it.
  std::vector<std::vector<int64_t>> Rows(mlir::ModuleOp module,  // INT_VEC_OK
                                         std::string_view attr_name) {
    std::vector<std::vector<int64_t>> rows;  // INT_VEC_OK
    module.walk([&](mlir::Operation* op) {
      auto attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(attr_name);
      if (attr == nullptr || !rows.empty()) {
        return;
      }
      const auto type = mlir::cast<mlir::ShapedType>(attr.getType());
      auto it = attr.value_begin<int64_t>();
      for (int64_t r = 0; r < type.getDimSize(0); ++r) {
        std::vector<int64_t> row;  // INT_VEC_OK
        for (int64_t c = 0; c < type.getDimSize(1); ++c) {
          row.push_back(*it++);
        }
        rows.push_back(row);
      }
    });
    return rows;
  }

  // The named sharding attribute of the first op carrying it.
  std::string OpSharding(mlir::ModuleOp module) {
    std::string result;
    module.walk([&](mlir::Operation* op) {
      auto attr = op->getAttrOfType<mlir::StringAttr>("mhlo.sharding");
      if (attr != nullptr && result.empty()) {
        result = attr.getValue().str();
      }
    });
    return result;
  }

  std::string ArgSharding(mlir::ModuleOp module, unsigned index) {
    std::string result;
    module.walk([&](mlir::func::FuncOp func) {
      auto attr =
          func.getArgAttrOfType<mlir::StringAttr>(index, "mhlo.sharding");
      if (attr != nullptr) {
        result = attr.getValue().str();
      }
    });
    return result;
  }

  mlir::MLIRContext context_;
};

TEST_F(CollectiveSubgroupTest, CommonGroupSkipsEmptyEntries) {
  std::vector<std::vector<int64_t>> groups = {
      {}, {2, 3}, {}, {2, 3}};  // INT_VEC_OK
  EXPECT_THAT(CommonCollectiveDeviceGroup(groups), ElementsAre(2, 3));
}

TEST_F(CollectiveSubgroupTest, CommonGroupIsEmptyWithoutCollectives) {
  std::vector<std::vector<int64_t>> groups = {{}, {}};  // INT_VEC_OK
  EXPECT_THAT(CommonCollectiveDeviceGroup(groups), IsEmpty());
}

TEST_F(CollectiveSubgroupTest, CommonGroupIsEmptyForMixedGroups) {
  std::vector<std::vector<int64_t>> groups = {// INT_VEC_OK
                                              {0, 1},
                                              {0, 1, 2, 3, 4, 5, 6, 7}};
  EXPECT_THAT(CommonCollectiveDeviceGroup(groups), IsEmpty());
}

TEST_F(CollectiveSubgroupTest, RestrictsOptionsToThePartitionedGroup) {
  xla::ExecutableBuildOptions options;
  const std::vector<int64_t> global_ids = {10, 11, 12, 13,
                                           14, 15, 16, 17};  // INT_VEC_OK
  ASSERT_THAT(RestrictBuildOptionsToDeviceGroup(options, {2, 3}, global_ids,
                                                /*replicated=*/false),
              IsOk());
  EXPECT_EQ(options.num_replicas(), 1);
  EXPECT_EQ(options.num_partitions(), 2);
  ASSERT_TRUE(options.has_device_assignment());
  EXPECT_EQ(options.device_assignment()(0, 0), 12);
  EXPECT_EQ(options.device_assignment()(0, 1), 13);
}

TEST_F(CollectiveSubgroupTest, RestrictsOptionsToTheReplicatedGroup) {
  xla::ExecutableBuildOptions options;
  const std::vector<int64_t> global_ids = {10, 11, 12, 13,
                                           14, 15, 16, 17};  // INT_VEC_OK
  ASSERT_THAT(RestrictBuildOptionsToDeviceGroup(options, {6, 7}, global_ids,
                                                /*replicated=*/true),
              IsOk());
  EXPECT_EQ(options.num_replicas(), 2);
  EXPECT_EQ(options.num_partitions(), 1);
  ASSERT_TRUE(options.has_device_assignment());
  EXPECT_EQ(options.device_assignment()(0, 0), 16);
  EXPECT_EQ(options.device_assignment()(1, 0), 17);
}

TEST_F(CollectiveSubgroupTest, RestrictRejectsDevicesOutsideTheSlice) {
  xla::ExecutableBuildOptions options;
  const std::vector<int64_t> global_ids = {10, 11};  // INT_VEC_OK
  EXPECT_THAT(RestrictBuildOptionsToDeviceGroup(options, {1, 2}, global_ids,
                                                /*replicated=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CollectiveSubgroupTest, RenumbersReplicaGroupsToTheGroup) {
  auto module = Parse(kPairAllReduce);
  ASSERT_THAT(RewriteCollectivesForDeviceGroup(*module, {2, 3}), IsOk());
  EXPECT_THAT(Rows(*module, "replica_groups"), ElementsAre(ElementsAre(0, 1)));
}

TEST_F(CollectiveSubgroupTest, RejectsAGroupThatSplitsACollective) {
  auto module = Parse(kWorldAllGather);
  EXPECT_THAT(RewriteCollectivesForDeviceGroup(*module, {2, 3}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not the program's device group")));
}

TEST_F(CollectiveSubgroupTest, RejectsACollectiveOutsideTheGroup) {
  auto module = Parse(kPairAllReduce);
  EXPECT_THAT(
      RewriteCollectivesForDeviceGroup(*module, {8, 9}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("involves none")));
}

TEST_F(CollectiveSubgroupTest, KeepsOnlyThePermutePairsInsideTheGroup) {
  auto module = Parse(kRingPermute);
  ASSERT_THAT(RewriteCollectivesForDeviceGroup(*module, {2, 3}), IsOk());
  EXPECT_THAT(Rows(*module, "source_target_pairs"),
              ElementsAre(ElementsAre(0, 1), ElementsAre(1, 0)));
}

TEST_F(CollectiveSubgroupTest, RejectsAPermuteThatCrossesTheGroup) {
  auto module = Parse(kCrossPermute);
  EXPECT_THAT(
      RewriteCollectivesForDeviceGroup(*module, {0, 1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("crosses")));
}

TEST_F(CollectiveSubgroupTest, ProjectsATiledShardingOntoTheGroup) {
  // Tile 0 lives on devices {0, 2, 4, 6} and tile 1 on {1, 3, 5, 7}: every
  // pair holds one device of each tile.
  const xla::HloSharding sharding =
      Sharding("{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}");
  for (const std::vector<int64_t>& group :  // INT_VEC_OK
       {std::vector<int64_t>{0, 1}, std::vector<int64_t>{6, 7}}) {
    EXPECT_THAT(ProjectShardingToDeviceGroup(sharding, group),
                IsOkAndHolds(Sharding("{devices=[2,1]<=[2]}")));
  }
}

TEST_F(CollectiveSubgroupTest, ProjectsAReplicatedTileToReplication) {
  auto projected = ProjectShardingToDeviceGroup(
      Sharding("{devices=[1,8]<=[8] last_tile_dim_replicate}"), {6, 7});
  ASSERT_THAT(projected, IsOk());
  EXPECT_TRUE(projected->IsReplicated());
}

TEST_F(CollectiveSubgroupTest, KeepsDevicelessShardingsAndRenumbersMaximal) {
  EXPECT_THAT(ProjectShardingToDeviceGroup(Sharding("{replicated}"), {6, 7}),
              IsOkAndHolds(Sharding("{replicated}")));
  EXPECT_THAT(ProjectShardingToDeviceGroup(Sharding("{manual}"), {6, 7}),
              IsOkAndHolds(Sharding("{manual}")));
  EXPECT_THAT(
      ProjectShardingToDeviceGroup(Sharding("{maximal device=7}"), {6, 7}),
      IsOkAndHolds(Sharding("{maximal device=1}")));
}

TEST_F(CollectiveSubgroupTest, RejectsAShardingTheGroupCannotHold) {
  EXPECT_THAT(
      ProjectShardingToDeviceGroup(Sharding("{devices=[2,4]<=[8]}"), {0, 1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unevenly")));
  EXPECT_THAT(
      ProjectShardingToDeviceGroup(Sharding("{devices=[2,1]<=[2]}"), {6, 7}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("holds none")));
}

TEST_F(CollectiveSubgroupTest, RewritesTheShardingsOfAnInlinedKernel) {
  auto module = Parse(kShardedKernel);
  ASSERT_THAT(RewriteCollectivesForDeviceGroup(*module, {6, 7}), IsOk());
  EXPECT_EQ(Sharding(ArgSharding(*module, 0)),
            Sharding("{devices=[2,1]<=[2]}"));
  EXPECT_EQ(Sharding(OpSharding(*module)), Sharding("{replicated}"));
  EXPECT_EQ(Sharding((*module)
                         ->getAttrOfType<mlir::StringAttr>(
                             "mhlo.spmd_output_sharding")
                         .getValue()
                         .str()),
            Sharding("{devices=[2,1]<=[2]}"));
}

TEST_F(CollectiveSubgroupTest, RejectsARaggedAllToAll) {
  auto module = Parse(kRaggedAllToAll);
  EXPECT_THAT(RewriteCollectivesForDeviceGroup(*module, {0, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ragged all-to-all")));
}

}  // namespace
}  // namespace torch_tpu
