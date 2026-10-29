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
#include "csrc/distributed/utils.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/distributed/allgather.h"
#include "csrc/distributed/alltoall.h"
#include "csrc/distributed/reduce_scatter.h"
#include "csrc/distributed/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Types.hpp"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::AllOf;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::Not;

// ---------------------------------------------------------------------------
// Pure folding helpers (virtual multi-host).
// ---------------------------------------------------------------------------

TEST(FoldVirtualMultiHostGroupTest, GroupOnOneSliceIsNotFolded) {
  TT_ASSERT_OK_AND_ASSIGN(const VirtualMultiHostFold fold,
                          FoldVirtualMultiHostGroup({0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(fold.fold, 1);
  EXPECT_THAT(fold.physical_device_ids, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));

  TT_ASSERT_OK_AND_ASSIGN(const VirtualMultiHostFold single,
                          FoldVirtualMultiHostGroup({5}));
  EXPECT_EQ(single.fold, 1);
  EXPECT_THAT(single.physical_device_ids, ElementsAre(5));
}

TEST(FoldVirtualMultiHostGroupTest, WorldOverTwoSlicesFoldsByTwo) {
  // 16 ranks over two 8-device slices: rank r runs on device r % 8.
  TT_ASSERT_OK_AND_ASSIGN(const VirtualMultiHostFold fold,
                          FoldVirtualMultiHostGroup({0, 1, 2, 3, 4, 5, 6, 7, 0,
                                                     1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(fold.fold, 2);
  EXPECT_THAT(fold.physical_device_ids, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(FoldVirtualMultiHostGroupTest, SubgroupSpanningSlicesFolds) {
  // An EP group of 8 ranks taking devices 0..3 of each of two slices.
  TT_ASSERT_OK_AND_ASSIGN(const VirtualMultiHostFold fold,
                          FoldVirtualMultiHostGroup({0, 1, 2, 3, 0, 1, 2, 3}));
  EXPECT_EQ(fold.fold, 2);
  EXPECT_THAT(fold.physical_device_ids, ElementsAre(0, 1, 2, 3));
  // Device ids need not start at zero or be contiguous.
  TT_ASSERT_OK_AND_ASSIGN(const VirtualMultiHostFold four,
                          FoldVirtualMultiHostGroup({6, 2, 6, 2, 6, 2, 6, 2}));
  EXPECT_EQ(four.fold, 4);
  EXPECT_THAT(four.physical_device_ids, ElementsAre(6, 2));
}

TEST(FoldVirtualMultiHostGroupTest, RejectsLayoutsThatAreNotTiles) {
  // Interleaved slices: the per-slice device order differs between tiles.
  EXPECT_THAT(
      FoldVirtualMultiHostGroup({0, 1, 1, 0}),
      StatusIs(error::kInvalidArgument, HasSubstr("not the 2 devices")));
  // A rank repeats a device before the slice is complete.
  EXPECT_THAT(FoldVirtualMultiHostGroup({0, 0, 1, 1}),
              StatusIs(error::kInvalidArgument));
  // Partial last tile.
  EXPECT_THAT(FoldVirtualMultiHostGroup({0, 1, 2, 0, 1}),
              StatusIs(error::kInvalidArgument));
  // A different device in the second tile.
  EXPECT_THAT(FoldVirtualMultiHostGroup({0, 1, 2, 3, 0, 1, 2, 4}),
              StatusIs(error::kInvalidArgument));
  EXPECT_THAT(FoldVirtualMultiHostGroup({}),
              StatusIs(error::kInvalidArgument, HasSubstr("without members")));
}

TEST(UniqueDeviceIdsTest, KeepsFirstAppearanceOrder) {
  EXPECT_THAT(UniqueDeviceIds({}), ElementsAre());
  EXPECT_THAT(UniqueDeviceIds({3, 1, 2}), ElementsAre(3, 1, 2));
  // The global device assignment of two EP subgroups over two slices.
  EXPECT_THAT(UniqueDeviceIds({0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7}),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(FoldReplicaGroupsTest, FoldsEachGroupToItsFirstPartitions) {
  const DeviceGroupList logical({{0, 1, 2, 3, 4, 5, 6, 7}});
  EXPECT_EQ(FoldReplicaGroups(logical), logical);
  EXPECT_EQ(PhysicalGroupSize(logical), 8);

  const DeviceGroupList folded = logical.WithVirtualFold(2);
  EXPECT_EQ(PhysicalGroupSize(folded), 4);
  const DeviceGroupList physical = FoldReplicaGroups(folded);
  ASSERT_EQ(physical.size(), 1);
  EXPECT_THAT(physical[0], ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(physical.virtual_fold(), 1);

  const DeviceGroupList two_groups =
      DeviceGroupList({{0, 1, 2, 3}, {4, 5, 6, 7}}).WithVirtualFold(2);
  const DeviceGroupList two_physical = FoldReplicaGroups(two_groups);
  ASSERT_EQ(two_physical.size(), 2);
  EXPECT_THAT(two_physical[0], ElementsAre(0, 1));
  EXPECT_THAT(two_physical[1], ElementsAre(4, 5));
}

// ---------------------------------------------------------------------------
// The StableHLO the collectives lower to.
// ---------------------------------------------------------------------------

// An MLIRContext with the StableHLO and func dialects loaded; a base class so
// it is ready before the builders below are constructed.
class LoadedContext {
 protected:
  LoadedContext() {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    registry.insert<mlir::func::FuncDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::MLIRContext context_;
};

class CollectiveBuilderTest : public LoadedContext, public testing::Test {
 protected:
  CollectiveBuilderTest()
      : module_builder_(context_, mlir::unknownLoc(context_)),
        function_builder_(module_builder_, "main") {}

  mlir::MlirOp F32Argument(llvm::ArrayRef<int64_t> shape) {
    return mlir::func::Argument(
        function_builder_,
        mlir::RankedTensorType::get(
            shape, function_builder_.getOpBuilder().getF32Type()));
  }

  llvm::ArrayRef<int64_t> Shape(const mlir::MlirOp& op) {
    return mlir::cast<mlir::RankedTensorType>(op.getType()).getShape();
  }

  // The module text so far (the ops built into `main`).
  std::string ModuleText() {
    mlir::func::Return(function_builder_, llvm::ArrayRef<mlir::MlirOp>());
    mlir::OwningOpRef<mlir::ModuleOp> module = module_builder_.build();
    std::string text;
    llvm::raw_string_ostream os(text);
    module->print(os);
    return text;
  }

  mlir::ModuleBuilder module_builder_;
  mlir::func::FunctionBuilder function_builder_;
};

TEST_F(CollectiveBuilderTest, ReplicaGroupsAttrOfFoldedGroupsIsPhysical) {
  const DeviceGroupList logical({{0, 1, 2, 3, 4, 5, 6, 7}});
  mlir::DenseIntElementsAttr attr =
      BuildReplicaGroupsAttr(function_builder_, logical);
  EXPECT_THAT(attr.getType().getShape(), ElementsAre(1, 8));

  mlir::DenseIntElementsAttr folded =
      BuildReplicaGroupsAttr(function_builder_, logical.WithVirtualFold(2));
  EXPECT_THAT(folded.getType().getShape(), ElementsAre(1, 4));
  const std::vector<int64_t> ids(  // INT_VEC_OK
      folded.value_begin<int64_t>(), folded.value_end<int64_t>());
  EXPECT_THAT(ids, ElementsAre(0, 1, 2, 3));
}

TEST_F(CollectiveBuilderTest, AllGatherKeepsTheLogicalShapeAndTiles) {
  const DeviceGroupList logical({{0, 1, 2, 3, 4, 5, 6, 7}});
  TT_ASSERT_OK_AND_ASSIGN(auto regular, BuildDistributedAllGatherShlo(
                                            F32Argument({2, 4}), logical,
                                            AllGatherOutputMode::kConcat));
  ASSERT_EQ(regular.size(), 1);
  EXPECT_THAT(Shape(regular[0]), ElementsAre(16, 4));

  TT_ASSERT_OK_AND_ASSIGN(
      auto folded, BuildDistributedAllGatherShlo(F32Argument({2, 4}),
                                                 logical.WithVirtualFold(2),
                                                 AllGatherOutputMode::kConcat));
  ASSERT_EQ(folded.size(), 1);
  // Same logical shape as the regular gather ...
  EXPECT_THAT(Shape(folded[0]), ElementsAre(16, 4));
  // ... produced by a gather over the local four devices, tiled twice.
  const std::string text = ModuleText();
  EXPECT_THAT(text, HasSubstr("replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, "
                              "7]]>"));
  EXPECT_THAT(text, HasSubstr("replica_groups = dense<[[0, 1, 2, 3]]>"));
  EXPECT_THAT(text, HasSubstr("tensor<8x4xf32>"));
  EXPECT_THAT(text, HasSubstr("stablehlo.concatenate"));
}

TEST_F(CollectiveBuilderTest, AllGatherStackAndListModesUseTheLogicalSize) {
  const DeviceGroupList folded =
      DeviceGroupList({{0, 1, 2, 3, 4, 5, 6, 7}}).WithVirtualFold(4);
  TT_ASSERT_OK_AND_ASSIGN(
      auto stacked, BuildDistributedAllGatherShlo(F32Argument({2, 4}), folded,
                                                  AllGatherOutputMode::kStack));
  ASSERT_EQ(stacked.size(), 1);
  EXPECT_THAT(Shape(stacked[0]), ElementsAre(8, 2, 4));
  TT_ASSERT_OK_AND_ASSIGN(
      auto listed, BuildDistributedAllGatherShlo(F32Argument({2, 4}), folded,
                                                 AllGatherOutputMode::kList));
  EXPECT_EQ(listed.size(), 8);
  for (const mlir::MlirOp& op : listed) {
    EXPECT_THAT(Shape(op), ElementsAre(2, 4));
  }
  EXPECT_THAT(ModuleText(), HasSubstr("replica_groups = dense<[[0, 1]]>"));
}

TEST_F(CollectiveBuilderTest, ReduceScatterScattersTheLocalShareOnly) {
  const DeviceGroupList logical({{0, 1, 2, 3, 4, 5, 6, 7}});
  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp regular,
      BuildDistributedReduceScatterShlo(F32Argument({16, 4}),
                                        c10d::ReduceOp::SUM, logical));
  EXPECT_THAT(Shape(regular), ElementsAre(2, 4));
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp folded,
                          BuildDistributedReduceScatterShlo(
                              F32Argument({16, 4}), c10d::ReduceOp::SUM,
                              logical.WithVirtualFold(2)));
  // The shard shape is the logical one ...
  EXPECT_THAT(Shape(folded), ElementsAre(2, 4));
  const std::string text = ModuleText();
  // ... computed from the first 2 * 4 rows over the local four devices.
  EXPECT_THAT(text, HasSubstr("stablehlo.slice"));
  EXPECT_THAT(text, HasSubstr("tensor<8x4xf32>"));
  EXPECT_THAT(text, HasSubstr("replica_groups = dense<[[0, 1, 2, 3]]>"));
}

TEST_F(CollectiveBuilderTest, AllToAllSplitsAmongTheLocalMembers) {
  const DeviceGroupList logical({{0, 1, 2, 3, 4, 5, 6, 7}});
  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp regular,
      BuildDistributedAllToAllBaseShlo(F32Argument({16, 4}), logical));
  EXPECT_THAT(Shape(regular), ElementsAre(16, 4));
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp folded, BuildDistributedAllToAllBaseShlo(
                                                   F32Argument({16, 4}),
                                                   logical.WithVirtualFold(2)));
  EXPECT_THAT(Shape(folded), ElementsAre(16, 4));
  const std::string text = ModuleText();
  EXPECT_THAT(text, HasSubstr("split_count = 8"));
  EXPECT_THAT(text, HasSubstr("split_count = 4"));
  EXPECT_THAT(text, HasSubstr("replica_groups = dense<[[0, 1, 2, 3]]>"));
}

TEST_F(CollectiveBuilderTest, RaggedAllToAllSendsNothingAcrossSlices) {
  const DeviceGroupList folded =
      DeviceGroupList({{0, 1, 2, 3, 4, 5, 6, 7}}).WithVirtualFold(2);
  const std::vector<int32_t> offsets = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<int32_t> sizes = {1, 1, 1, 1, 1, 1, 1, 1};
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp result,
                          BuildDistributedAllToAllBaseUnevenSplitsShlo(
                              F32Argument({8, 4}), F32Argument({8, 4}), offsets,
                              sizes, offsets, sizes, folded));
  EXPECT_THAT(Shape(result), ElementsAre(8, 4));
  const std::string text = ModuleText();
  // Offsets and sizes are cut down to the four local members, and the sizes
  // zeroed (x - x), so the placeholder exchange moves nothing.
  EXPECT_THAT(text, HasSubstr("tensor<4xi32>"));
  EXPECT_THAT(text, HasSubstr("stablehlo.subtract"));
  EXPECT_THAT(text, HasSubstr("replica_groups = dense<[[0, 1, 2, 3]]>"));
  EXPECT_THAT(text, Not(HasSubstr("replica_groups = dense<[[0, 1, 2, 3, 4")));
}

}  // namespace
}  // namespace torch_tpu
