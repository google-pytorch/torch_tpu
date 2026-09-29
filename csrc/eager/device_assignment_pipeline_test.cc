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

#include "csrc/eager/device_assignment_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Exception.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_names.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

std::vector<int64_t> ExtractDenseIntAttrValues(  // INT_VEC_OK
    mlir::Operation* op, std::string_view attr_name) {
  std::vector<int64_t> vals;  // INT_VEC_OK
  if (auto attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(attr_name)) {
    for (const llvm::APInt& v : attr.getValues<llvm::APInt>()) {
      vals.push_back(v.getSExtValue());
    }
  }
  return vals;
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentPermutationRemapsWithWarning) {
  const std::vector<int64_t>  // INT_VEC_OK
      pallas_da = {0, 1, 4, 5, 6, 7, 2, 3};
  const std::vector<int64_t>  // INT_VEC_OK
      collective_da = {0, 1, 2, 3, 4, 5, 6, 7};

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved_da,  // INT_VEC_OK
      ResolveDeviceAssignment(
          {
              OpDeviceAssignment{OpName::kDistributedAllReduce, collective_da},
          },
          {
              PallasDeviceAssignment{pallas_da},
          }));

  EXPECT_EQ(resolved_da, pallas_da);

  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_permutation_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp collective_op = mlir::stablehlo::Neg(arg);
  mlir::Operation* raw_op = collective_op.getValue().getDefiningOp();
  auto rg_type = mlir::RankedTensorType::get(
      {2, 4}, mlir::IntegerType::get(&mlir_context, 64));
  raw_op->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          rg_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));

  mlir::func::Return(fb, {collective_op});
  auto module = mb.build();

  EXPECT_TRUE(
      RewriteAndTileOperationForSubgroup(raw_op, collective_da, resolved_da)
          .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(raw_op, "replica_groups"),
              testing::ElementsAre(0, 1, 6, 7, 2, 3, 4, 5));
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentPreservesSyntheticNonZeroDeviceIds) {
  const std::vector<int64_t> synthetic_da = {2, 3};             // INT_VEC_OK
  TT_ASSERT_OK_AND_ASSIGN(const std::vector<int64_t> resolved,  // INT_VEC_OK
                          ResolveDeviceAssignment({
                              OpDeviceAssignment{OpName::kAdd, synthetic_da},
                          }));
  EXPECT_THAT(resolved, testing::ElementsAre(2, 3));
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentThrowsOnDifferingPallasMeshes) {
  std::vector<int64_t> da1(128);  // INT_VEC_OK
  std::vector<int64_t> da2(128);  // INT_VEC_OK
  for (int i = 0; i < 128; ++i) {
    da1[i] = i;
    da2[i] = 127 - i;
  }
  bool caught = false;
  try {
    (void)ResolveDeviceAssignment({}, {
                                          PallasDeviceAssignment{da1},
                                          PallasDeviceAssignment{da2},
                                      });
  } catch (const c10::Error& e) {
    caught = true;
    EXPECT_THAT(e.what(),
                testing::HasSubstr("conflicting device assignments detected"));
  } catch (const torch_tpu::TtError& e) {
    caught = true;
    EXPECT_THAT(e.what(),
                testing::HasSubstr("conflicting device assignments detected"));
  }
  EXPECT_TRUE(caught);
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentLogsWarningAndFailsOnDifferingDAs) {
  const std::vector<int64_t> da1 = {0, 1, 2, 3, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t>                                  // INT_VEC_OK
      da2 = {8, 9, 10, 11, 12, 13, 14, 15};

  const absl::StatusOr<std::vector<int64_t>> resolved =  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, da1},
          OpDeviceAssignment{OpName::kDistributedAllReduce, da2},
      });

  EXPECT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      resolved.status().message(),
      testing::HasSubstr("device assignment mismatch across operations"));
  EXPECT_THAT(
      resolved.status().message(),
      testing::HasSubstr("All operations in the traversal must share the same "
                         "world-sized device assignment"));

  const std::vector<int64_t> da3 = {0, 1, 2, 3};  // INT_VEC_OK
  const absl::StatusOr<std::vector<int64_t>>      // INT_VEC_OK
      resolved_size_mismatch = ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, da1},
          OpDeviceAssignment{OpName::kDistributedAllReduce, da3},
      });

  EXPECT_FALSE(resolved_size_mismatch.ok());
  EXPECT_EQ(resolved_size_mismatch.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      resolved_size_mismatch.status().message(),
      testing::HasSubstr("device assignment mismatch across operations"));
  EXPECT_THAT(
      resolved_size_mismatch.status().message(),
      testing::HasSubstr("All operations in the traversal must share the same "
                         "world-sized device assignment"));

  const absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
      resolved_pallas_mismatch = ResolveDeviceAssignment(
          {
              OpDeviceAssignment{OpName::kDistributedAllReduce, da2},
          },
          {
              PallasDeviceAssignment{da1},
          });

  EXPECT_FALSE(resolved_pallas_mismatch.ok());
  EXPECT_EQ(resolved_pallas_mismatch.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      resolved_pallas_mismatch.status().message(),
      testing::HasSubstr("device assignment mismatch across operations"));
  EXPECT_THAT(
      resolved_pallas_mismatch.status().message(),
      testing::HasSubstr("All operations in the traversal must share the same "
                         "world-sized device assignment"));
}

TEST(DeviceAssignmentPipelineTest,
     RewriteAndTileOperationForSubgroupRewritesOrthogonalReplicaGroups) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp shard_op = mlir::stablehlo::Neg(arg);
  auto shard_rg_type = mlir::RankedTensorType::get(
      {2, 4}, mlir::IntegerType::get(&mlir_context, 64));
  shard_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          shard_rg_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));

  mlir::MlirOp replicate_op = mlir::stablehlo::Abs(shard_op);
  auto rep_rg_type = mlir::RankedTensorType::get(
      {4, 2}, mlir::IntegerType::get(&mlir_context, 64));
  replicate_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          rep_rg_type, llvm::ArrayRef<int64_t>({0, 4, 1, 5, 2, 6, 3, 7})));

  mlir::func::Return(fb, {replicate_op});
  auto module = mb.build();

  const std::vector<int64_t> op_da = {0, 1, 2, 3, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t>                                    // INT_VEC_OK
      resolved_da = {2, 3, 0, 1, 6, 7, 4, 5};
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  shard_op.getValue().getDefiningOp(), op_da, resolved_da)
                  .ok());
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  replicate_op.getValue().getDefiningOp(), op_da, resolved_da)
                  .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(shard_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(2, 3, 0, 1, 6, 7, 4, 5));
  EXPECT_THAT(ExtractDenseIntAttrValues(replicate_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(2, 6, 3, 7, 0, 4, 1, 5));
}

TEST(DeviceAssignmentPipelineTest,
     RemapOperationForSubgroupRemapsReplicaGroupsAndSourceTargetPairs) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_tile_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp op = mlir::stablehlo::Neg(arg);
  mlir::Operation* raw_op = op.getValue().getDefiningOp();
  auto rg_type = mlir::RankedTensorType::get(
      {2, 4}, mlir::IntegerType::get(&mlir_context, 64));
  raw_op->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          rg_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));
  auto st_type = mlir::RankedTensorType::get(
      {4, 2}, mlir::IntegerType::get(&mlir_context, 64));
  raw_op->setAttr(
      "source_target_pairs",
      mlir::DenseIntElementsAttr::get(
          st_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));

  mlir::func::Return(fb, {op});
  auto module = mb.build();
  mlir::func::FuncOp func_op = module->lookupSymbol<mlir::func::FuncOp>("main");

  const std::vector<int64_t> op_da = {0, 1, 2, 3, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t>                                    // INT_VEC_OK
      resolved_da = {0, 1, 6, 7, 2, 3, 4, 5};
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(func_op.getOperation(), op_da,
                                                 resolved_da)
                  .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(raw_op, "replica_groups"),
              testing::ElementsAre(0, 1, 4, 5, 6, 7, 2, 3));
  EXPECT_THAT(ExtractDenseIntAttrValues(raw_op, "source_target_pairs"),
              testing::ElementsAre(0, 1, 4, 5, 6, 7, 2, 3));
}

TEST(DeviceAssignmentPipelineTest,
     RewriteAndTileOperationForSubgroupHandlesPermutedPallasMesh) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_hsdp_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp replicate_op = mlir::stablehlo::Abs(arg);
  auto rep_rg_type = mlir::RankedTensorType::get(
      {4, 2}, mlir::IntegerType::get(&mlir_context, 64));
  replicate_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          rep_rg_type, llvm::ArrayRef<int64_t>({0, 4, 1, 5, 2, 6, 3, 7})));

  mlir::func::Return(fb, {replicate_op});
  auto module = mb.build();

  const std::vector<int64_t> op_da = {0, 1, 2, 3, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t>                                    // INT_VEC_OK
      resolved_da = {3, 2, 1, 0, 4, 5, 6, 7};
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  replicate_op.getValue().getDefiningOp(), op_da, resolved_da)
                  .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(replicate_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(3, 4, 2, 5, 1, 6, 0, 7));
}

TEST(DeviceAssignmentPipelineTest,
     RewriteAndTileOperationForSubgroupHsdpReplicateSymmetricAcrossRanks) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_hsdp_symmetric_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  std::vector<int64_t> rep_rg_vals;  // INT_VEC_OK
  rep_rg_vals.reserve(128);
  for (int64_t i = 0; i < 64; ++i) {
    rep_rg_vals.push_back(i);
    rep_rg_vals.push_back(i + 64);
  }
  auto rep_rg_type = mlir::RankedTensorType::get(
      {64, 2}, mlir::IntegerType::get(&mlir_context, 64));

  mlir::MlirOp replicate_op_rank0 = mlir::stablehlo::Abs(arg);
  replicate_op_rank0.getValue().getDefiningOp()->setAttr(
      "replica_groups", mlir::DenseIntElementsAttr::get(
                            rep_rg_type, llvm::ArrayRef<int64_t>(rep_rg_vals)));

  mlir::MlirOp replicate_op_rank8 = mlir::stablehlo::Abs(replicate_op_rank0);
  replicate_op_rank8.getValue().getDefiningOp()->setAttr(
      "replica_groups", mlir::DenseIntElementsAttr::get(
                            rep_rg_type, llvm::ArrayRef<int64_t>(rep_rg_vals)));

  mlir::func::Return(fb, {replicate_op_rank8});
  auto module = mb.build();

  const std::vector<int64_t>  // INT_VEC_OK
      shard_subgroup_0 = {
          0,  1,  32, 33, 64, 65, 96,  97,  8,  9,  40, 41, 72, 73, 104, 105,
          16, 17, 48, 49, 80, 81, 112, 113, 24, 25, 56, 57, 88, 89, 120, 121,
          2,  3,  34, 35, 66, 67, 98,  99,  10, 11, 42, 43, 74, 75, 106, 107,
          18, 19, 50, 51, 82, 83, 114, 115, 26, 27, 58, 59, 90, 91, 122, 123};

  auto antipodal_partner = [](int64_t d) {
    const int64_t c = d % 2;
    const int64_t chip = d / 2;
    const int64_t z = chip % 4;
    const int64_t y = (chip / 4) % 4;
    const int64_t x = (chip / 16) % 4;
    const int64_t x2 = (x + 2) % 4;
    const int64_t y2 = (y + 2) % 4;
    const int64_t z2 = (z + 2) % 4;
    const int64_t chip2 = x2 * 16 + y2 * 4 + z2;
    return chip2 * 2 + c;
  };

  std::vector<int64_t> canonical_world_da = shard_subgroup_0;  // INT_VEC_OK
  canonical_world_da.reserve(128);
  for (int64_t d : shard_subgroup_0) {
    canonical_world_da.push_back(antipodal_partner(d));
  }

  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  replicate_op_rank0.getValue().getDefiningOp(),
                  /*op_da=*/canonical_world_da, canonical_world_da)
                  .ok());
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  replicate_op_rank8.getValue().getDefiningOp(),
                  /*op_da=*/canonical_world_da, canonical_world_da)
                  .ok());

  const std::vector<int64_t> rg_rank0 =  // INT_VEC_OK
      ExtractDenseIntAttrValues(replicate_op_rank0.getValue().getDefiningOp(),
                                "replica_groups");
  const std::vector<int64_t> rg_rank8 =  // INT_VEC_OK
      ExtractDenseIntAttrValues(replicate_op_rank8.getValue().getDefiningOp(),
                                "replica_groups");

  EXPECT_EQ(rg_rank0.size(), 128);
  EXPECT_EQ(rg_rank8.size(), 128);
  EXPECT_EQ(
      ExtractDenseIntAttrValues(replicate_op_rank0.getValue().getDefiningOp(),
                                "replica_groups"),
      ExtractDenseIntAttrValues(replicate_op_rank8.getValue().getDefiningOp(),
                                "replica_groups"));

  auto contains_device_pair = [](absl::Span<const int64_t> rg,
                                 absl::Span<const int64_t> resolved_da,
                                 int64_t dev_1, int64_t dev_2) {
    for (size_t i = 0; i + 1 < rg.size(); i += 2) {
      const int64_t d_a = resolved_da[rg[i]];
      const int64_t d_b = resolved_da[rg[i + 1]];
      if ((d_a == dev_1 && d_b == dev_2) || (d_a == dev_2 && d_b == dev_1)) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(contains_device_pair(rg_rank0, canonical_world_da, 0, 84));
  EXPECT_TRUE(contains_device_pair(rg_rank8, canonical_world_da, 0, 84));

  auto has_consecutive_pair = [](absl::Span<const int64_t> rg, int64_t p1,
                                 int64_t p2) {
    for (size_t i = 0; i + 1 < rg.size(); i += 2) {
      if ((rg[i] == p1 && rg[i + 1] == p2) ||
          (rg[i] == p2 && rg[i + 1] == p1)) {
        return true;
      }
    }
    return false;
  };
  EXPECT_FALSE(has_consecutive_pair(rg_rank0, 44, 45));
  EXPECT_FALSE(has_consecutive_pair(rg_rank8, 44, 45));
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentHsdpSymmetricAcrossRanks) {
  const std::vector<int64_t>  // INT_VEC_OK
      shard_subgroup_0 = {
          0,  1,  32, 33, 64, 65, 96,  97,  8,  9,  40, 41, 72, 73, 104, 105,
          16, 17, 48, 49, 80, 81, 112, 113, 24, 25, 56, 57, 88, 89, 120, 121,
          2,  3,  34, 35, 66, 67, 98,  99,  10, 11, 42, 43, 74, 75, 106, 107,
          18, 19, 50, 51, 82, 83, 114, 115, 26, 27, 58, 59, 90, 91, 122, 123};
  const std::vector<int64_t>  // INT_VEC_OK
      shard_subgroup_1 = {
          4,  5,  36, 37, 68, 69, 100, 101, 12, 13, 44, 45, 76, 77, 108, 109,
          20, 21, 52, 53, 84, 85, 116, 117, 28, 29, 60, 61, 92, 93, 124, 125,
          6,  7,  38, 39, 70, 71, 102, 103, 14, 15, 46, 47, 78, 79, 110, 111,
          22, 23, 54, 55, 86, 87, 118, 119, 30, 31, 62, 63, 94, 95, 126, 127};

  std::vector<int64_t> canonical_world_da = shard_subgroup_0;  // INT_VEC_OK
  canonical_world_da.insert(canonical_world_da.end(), shard_subgroup_1.begin(),
                            shard_subgroup_1.end());

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved_rank0,  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, canonical_world_da},
          OpDeviceAssignment{OpName::kDistributedAllReduce, canonical_world_da},
      }));

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved_rank6,  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, canonical_world_da},
          OpDeviceAssignment{OpName::kDistributedAllReduce, canonical_world_da},
      }));

  EXPECT_EQ(resolved_rank0.size(), 128);
  EXPECT_EQ(resolved_rank6.size(), 128);
  EXPECT_EQ(resolved_rank0, canonical_world_da);
  EXPECT_EQ(resolved_rank6, canonical_world_da);
  EXPECT_EQ(resolved_rank0, resolved_rank6);
}

}  // namespace
}  // namespace torch_tpu
