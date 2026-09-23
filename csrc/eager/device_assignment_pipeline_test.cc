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

#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_names.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
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

TEST(DeviceAssignmentPipelineTest, IsCompatibleDeviceAssignmentTile) {
  const std::vector<int64_t> full_da = {2, 3, 1, 0, 4, 5, 6, 7};  // INT_VEC_OK
  EXPECT_TRUE(IsCompatibleDeviceAssignmentTile({2, 3, 1, 0}, full_da));
  EXPECT_TRUE(IsCompatibleDeviceAssignmentTile({4, 5, 6, 7}, full_da));
  EXPECT_TRUE(IsCompatibleDeviceAssignmentTile({2, 3}, full_da));
  EXPECT_TRUE(IsCompatibleDeviceAssignmentTile({6, 7}, full_da));
  EXPECT_TRUE(IsCompatibleDeviceAssignmentTile(full_da, full_da));

  // Unaligned or non-contiguous tiles are incompatible.
  EXPECT_FALSE(IsCompatibleDeviceAssignmentTile({3, 1}, full_da));
  EXPECT_FALSE(IsCompatibleDeviceAssignmentTile({2, 4}, full_da));
  EXPECT_FALSE(IsCompatibleDeviceAssignmentTile({0, 1, 2, 3}, full_da));
  EXPECT_FALSE(IsCompatibleDeviceAssignmentTile({}, full_da));
  EXPECT_FALSE(IsCompatibleDeviceAssignmentTile({2, 3, 1}, full_da));
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentPromotesCompatiblePallasAndStableHloTiles) {
  const std::vector<int64_t> pallas_sub = {2, 3, 1, 0};  // INT_VEC_OK
  const std::vector<int64_t> stablehlo_full = {          // INT_VEC_OK
                                               2, 3, 1, 0, 4, 5, 6, 7};
  const std::vector<int64_t> orthogonal = {2, 4};  // INT_VEC_OK
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved,  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kCustomKernel, pallas_sub},
          OpDeviceAssignment{OpName::kDistributedAllReduce, stablehlo_full},
          OpDeviceAssignment{OpName::kDistributedAllReduce, orthogonal},
      }));
  EXPECT_THAT(resolved, testing::ElementsAre(2, 3, 1, 0, 4, 5, 6, 7));
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
     IncompatiblePallasKernelsReturnDiagnosticError) {
  const std::vector<int64_t> da_a = {2, 3, 1, 0, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t> da_b = {7, 6, 5, 4, 0, 1, 3, 2};  // INT_VEC_OK
  const absl::StatusOr<std::vector<int64_t>> resolved =        // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kCustomKernel, da_a},
          OpDeviceAssignment{OpName::kCustomKernel, da_b},
      });
  EXPECT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status().code(), error::kInvalidArgument);
  EXPECT_THAT(resolved.status().message(),
              testing::HasSubstr("Disjoint or conflicting device assignments"));
  EXPECT_THAT(resolved.status().message(),
              testing::HasSubstr("op_da=[7, 6, 5, 4, 0, 1, 3, 2]"));
  EXPECT_THAT(resolved.status().message(),
              testing::HasSubstr("pallas_da=[2, 3, 1, 0, 4, 5, 6, 7]"));
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
      {1, 4}, mlir::IntegerType::get(&mlir_context, 64));
  shard_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(shard_rg_type,
                                      llvm::ArrayRef<int64_t>({0, 1, 2, 3})));

  mlir::MlirOp replicate_op = mlir::stablehlo::Abs(shard_op);
  auto rep_rg_type = mlir::RankedTensorType::get(
      {1, 2}, mlir::IntegerType::get(&mlir_context, 64));
  replicate_op.getValue().getDefiningOp()->setAttr(
      "replica_groups", mlir::DenseIntElementsAttr::get(
                            rep_rg_type, llvm::ArrayRef<int64_t>({0, 1})));

  mlir::func::Return(fb, {replicate_op});
  auto module = mb.build();

  const std::vector<int64_t> resolved_da = {// INT_VEC_OK
                                            2, 3, 1, 0, 4, 5, 6, 7};
  RewriteAndTileOperationForSubgroup(shard_op.getValue().getDefiningOp(),
                                     /*op_da=*/{2, 3, 1, 0}, resolved_da);
  RewriteAndTileOperationForSubgroup(replicate_op.getValue().getDefiningOp(),
                                     /*op_da=*/{2, 4}, resolved_da);

  EXPECT_THAT(ExtractDenseIntAttrValues(shard_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
  EXPECT_THAT(ExtractDenseIntAttrValues(replicate_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(0, 4, 1, 5, 2, 6, 3, 7));
}

TEST(DeviceAssignmentPipelineTest,
     TileOperationForSubgroupTilesReplicaGroupsAndSourceTargetPairs) {
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
      {1, 4}, mlir::IntegerType::get(&mlir_context, 64));
  raw_op->setAttr("replica_groups",
                  mlir::DenseIntElementsAttr::get(
                      rg_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3})));
  auto st_type = mlir::RankedTensorType::get(
      {2, 2}, mlir::IntegerType::get(&mlir_context, 64));
  raw_op->setAttr("source_target_pairs",
                  mlir::DenseIntElementsAttr::get(
                      st_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3})));

  mlir::func::Return(fb, {op});
  auto module = mb.build();
  mlir::func::FuncOp func_op = module->lookupSymbol<mlir::func::FuncOp>("main");

  RewriteAndTileOperationForSubgroup(func_op.getOperation(),
                                     /*op_da=*/{0, 1, 4, 5},
                                     /*resolved_da=*/{0, 1, 4, 5, 6, 7, 2, 3});

  EXPECT_THAT(ExtractDenseIntAttrValues(raw_op, "replica_groups"),
              testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
  EXPECT_THAT(ExtractDenseIntAttrValues(raw_op, "source_target_pairs"),
              testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

}  // namespace
}  // namespace torch_tpu
