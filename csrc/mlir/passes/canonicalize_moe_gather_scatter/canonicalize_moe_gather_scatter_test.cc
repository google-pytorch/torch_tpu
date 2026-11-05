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

#include "csrc/mlir/passes/canonicalize_moe_gather_scatter/canonicalize_moe_gather_scatter.h"  // NOLINT

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"  // NOLINT
#include "mlir/IR/BuiltinAttributes.h"         // NOLINT
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"  // NOLINT
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

template <typename OpTy>
int CountOps(mlir::ModuleOp module) {
  int count = 0;
  module.walk([&](OpTy) { ++count; });
  return count;
}

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 7 + 3) % 47) * 0.125f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape,
                                       float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    float f = base + static_cast<float>((i * 5 + 1) % 31) * 0.25f;
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

mlir::DenseElementsAttr MakeI32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<int32_t> vals) {
  auto type =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 32));
  return mlir::DenseElementsAttr::get(type, vals);
}

// Mechanism 2: Fold identity or constant gather.
TEST(CanonicalizeMoEGatherScatterTest, FoldIdentityOrConstantGatherNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @identity_gather(%operand: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}>
      : () -> tensor<4x1xi32>
  %out = "stablehlo.gather"(%operand, %iota) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 8>
  }> : (tensor<4x8xf32>, tensor<4x1xi32>) -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr operand = MakeF32Tensor(context, {4, 8}, 2.0f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {operand}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {operand}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

// Mechanism 3: Sink row ops past gather (generalized rank-3 variant).
TEST(CanonicalizeMoEGatherScatterTest, SinkRowOpsPastGatherRank3Numerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @sink_row_ops_3d(
    %input: tensor<4x2x8xbf16>,
    %scale: tensor<4xf32>,
    %indices: tensor<6x1xi32>) -> tensor<6x2x8xf32> {
  %conv = stablehlo.convert %input : (tensor<4x2x8xbf16>) -> tensor<4x2x8xf32>
  %bcast = stablehlo.broadcast_in_dim %scale, dims = [0]
      : (tensor<4xf32>) -> tensor<4x2x8xf32>
  %scaled = stablehlo.multiply %conv, %bcast : tensor<4x2x8xf32>
  %out = "stablehlo.gather"(%scaled, %indices) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 2, 8>
  }> : (tensor<4x2x8xf32>, tensor<6x1xi32>) -> tensor<6x2x8xf32>
  return %out : tensor<6x2x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr input = MakeBF16Tensor(context, {4, 2, 8}, 1.5f);
  mlir::DenseElementsAttr scale = MakeF32Tensor(context, {4}, 0.5f);
  mlir::DenseElementsAttr indices =
      MakeI32Tensor(context, {6, 1}, {3, 1, 0, 2, 1, 3});

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {input, scale, indices}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {input, scale, indices},
                                            interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

// Mechanism 4a: Bijective inverse permutation scatter into zeros (3D tensor).
TEST(CanonicalizeMoEGatherScatterTest,
     BijectiveInversePermutationScatter3DNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @bijective_scatter_3d(
    %updates: tensor<4x2x8xf32>) -> tensor<4x2x8xf32> {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x2x8xf32>
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %perm = stablehlo.constant dense<[2, 0, 3, 1]> : tensor<4xi32>
  %inv_perm = "stablehlo.scatter"(%iota, %perm, %iota) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    stablehlo.return %rhs : tensor<i32>
  }) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  %out = "stablehlo.scatter"(%zeros, %inv_perm, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<4x2x8xf32>, tensor<4xi32>, tensor<4x2x8xf32>)
      -> tensor<4x2x8xf32>
  return %out : tensor<4x2x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr updates = MakeF32Tensor(context, {4, 2, 8}, 3.0f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {updates}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {updates}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

// Mechanism 3 Negative: Scatter into non-zero tensor is not lowered.
TEST(CanonicalizeMoEGatherScatterTest, ScatterIntoNonZeroTensorNotLowered) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @scatter_into_nonzero(
    %base: tensor<3x4xf32>,
    %updates: tensor<3x4xf32>,
    %perm: tensor<3xi32>) -> tensor<3x4xf32> {
  %out = "stablehlo.scatter"(%base, %perm, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<3x4xf32>, tensor<3xi32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %out : tensor<3x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);
}

// Argsort inverse permutation lowering positive test.
TEST(CanonicalizeMoEGatherScatterTest, BijectiveArgsortScatterLoweredToGather) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @argsort_scatter(
    %updates: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %keys = stablehlo.constant dense<[2, 0, 3, 1]> : tensor<4xi32>
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %sorted:2 = "stablehlo.sort"(%keys, %iota) <{
    dimension = 0 : i64,
    is_stable = true
  }> ({
  ^bb0(%k1: tensor<i32>, %k2: tensor<i32>, %v1: tensor<i32>, %v2: tensor<i32>):
    %cmp = "stablehlo.compare"(%k1, %k2) <{
      comparison_direction = #stablehlo<comparison_direction LT>
    }> : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)

  %out = "stablehlo.scatter"(%zeros, %sorted#1, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<4x4xf32>, tensor<4xi32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %out : tensor<4x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr updates = MakeF32Tensor(context, {4, 4}, 2.5f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {updates}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {updates}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

// Negative test 1: Descending sort comparator must not be lowered.
TEST(CanonicalizeMoEGatherScatterTest, DescendingSortComparatorNotLowered) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @descending_sort_scatter(
    %updates: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %keys = stablehlo.constant dense<[2, 0, 3, 1]> : tensor<4xi32>
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %sorted:2 = "stablehlo.sort"(%keys, %iota) <{
    dimension = 0 : i64,
    is_stable = true
  }> ({
  ^bb0(%k1: tensor<i32>, %k2: tensor<i32>, %v1: tensor<i32>, %v2: tensor<i32>):
    %cmp = "stablehlo.compare"(%k1, %k2) <{
      comparison_direction = #stablehlo<comparison_direction GT>
    }> : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)

  %out = "stablehlo.scatter"(%zeros, %sorted#1, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<4x4xf32>, tensor<4xi32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %out : tensor<4x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);
}

// Negative test 2: Float keys sort must not be lowered.
TEST(CanonicalizeMoEGatherScatterTest, FloatKeysSortNotLowered) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @float_keys_sort_scatter(
    %updates: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %keys = stablehlo.constant
      dense<[2.000000e+00, 0.000000e+00, 3.000000e+00, 1.000000e+00]>
      : tensor<4xf32>
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %sorted:2 = "stablehlo.sort"(%keys, %iota) <{
    dimension = 0 : i64,
    is_stable = true
  }> ({
  ^bb0(%k1: tensor<f32>, %k2: tensor<f32>, %v1: tensor<i32>, %v2: tensor<i32>):
    %cmp = "stablehlo.compare"(%k1, %k2) <{
      comparison_direction = #stablehlo<comparison_direction LT>
    }> : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) : (tensor<4xf32>, tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>)

  %out = "stablehlo.scatter"(%zeros, %sorted#1, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<4x4xf32>, tensor<4xi32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %out : tensor<4x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);
}

// Negative test 3: Duplicate int keys must not be lowered.
TEST(CanonicalizeMoEGatherScatterTest, DuplicateIntKeysSortNotLowered) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @duplicate_int_keys_sort_scatter(
    %updates: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %keys = stablehlo.constant dense<[1, 1, 0, 2]> : tensor<4xi32>
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %sorted:2 = "stablehlo.sort"(%keys, %iota) <{
    dimension = 0 : i64,
    is_stable = true
  }> ({
  ^bb0(%k1: tensor<i32>, %k2: tensor<i32>, %v1: tensor<i32>, %v2: tensor<i32>):
    %cmp = "stablehlo.compare"(%k1, %k2) <{
      comparison_direction = #stablehlo<comparison_direction LT>
    }> : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)

  %out = "stablehlo.scatter"(%zeros, %sorted#1, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<4x4xf32>, tensor<4xi32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %out : tensor<4x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);
}

TEST(CanonicalizeMoEGatherScatterTest,
     FoldConstantGatherSliceTransposeBroadcastNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @const_gather_slice_transpose_bcast(
    %operand: tensor<4x3x8xf32>) -> tensor<2x8x4xf32> {
  %iota = "stablehlo.iota"() <{iota_dimension = 1 : i64}>
      : () -> tensor<2x4x1xi32>
  %c1 = stablehlo.constant dense<1> : tensor<2x4x1xi32>
  %indices = "stablehlo.concatenate"(%iota, %c1) <{dimension = 2 : i64}>
      : (tensor<2x4x1xi32>, tensor<2x4x1xi32>) -> tensor<2x4x2xi32>
  %out = "stablehlo.gather"(%operand, %indices) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 1, 8>
  }> : (tensor<4x3x8xf32>, tensor<2x4x2xi32>) -> tensor<2x8x4xf32>
  return %out : tensor<2x8x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr operand = MakeF32Tensor(context, {4, 3, 8}, 1.25f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {operand}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {operand}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

TEST(CanonicalizeMoEGatherScatterTest, OutOfBoundsConstantGatherNotEliminated) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @oob_const_gather(%operand: tensor<4x8xf32>) -> tensor<1x8xf32> {
  %c5 = stablehlo.constant dense<[[5]]> : tensor<1x1xi32>
  %out = "stablehlo.gather"(%operand, %c5) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 8>
  }> : (tensor<4x8xf32>, tensor<1x1xi32>) -> tensor<1x8xf32>
  return %out : tensor<1x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);
}

TEST(CanonicalizeMoEGatherScatterTest, ZeroElementGatherNotEliminated) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @zero_elem_gather(%operand: tensor<4x0xf32>) -> tensor<4x0xf32> {
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}>
      : () -> tensor<4x1xi32>
  %out = "stablehlo.gather"(%operand, %iota) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 0>
  }> : (tensor<4x0xf32>, tensor<4x1xi32>) -> tensor<4x0xf32>
  return %out : tensor<4x0xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);
}

TEST(CanonicalizeMoEGatherScatterTest, FoldIdentityRank1GatherNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @identity_rank1_gather(%operand: tensor<4xf32>) -> tensor<4xf32> {
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}>
      : () -> tensor<4x1xi32>
  %out = "stablehlo.gather"(%operand, %iota) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1>
  }> : (tensor<4xf32>, tensor<4x1xi32>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr operand = MakeF32Tensor(context, {4}, 3.5f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {operand}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {operand}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

TEST(CanonicalizeMoEGatherScatterTest,
     SinkRowMaskZeroSelectOnTrueAndOnFalseNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @sink_row_mask_zero_select(
    %input1: tensor<4x8xbf16>,
    %input2: tensor<4x8xbf16>,
    %mask1: tensor<4xi1>,
    %mask2: tensor<4xi1>,
    %indices: tensor<4x1xi32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x8xf32>
  %widen1 = "stablehlo.convert"(%input1)
      : (tensor<4x8xbf16>) -> tensor<4x8xf32>
  %bcast_mask1 = "stablehlo.broadcast_in_dim"(%mask1) <{
    broadcast_dimensions = array<i64: 0>
  }> : (tensor<4xi1>) -> tensor<4x8xi1>
  %sel1 = "stablehlo.select"(%bcast_mask1, %zeros, %widen1)
      : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %out1 = "stablehlo.gather"(%sel1, %indices) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 8>
  }> : (tensor<4x8xf32>, tensor<4x1xi32>) -> tensor<4x8xf32>

  %widen2 = "stablehlo.convert"(%input2)
      : (tensor<4x8xbf16>) -> tensor<4x8xf32>
  %bcast_mask2 = "stablehlo.broadcast_in_dim"(%mask2) <{
    broadcast_dimensions = array<i64: 0>
  }> : (tensor<4xi1>) -> tensor<4x8xi1>
  %sel2 = "stablehlo.select"(%bcast_mask2, %widen2, %zeros)
      : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %out2 = "stablehlo.gather"(%sel2, %indices) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 8>
  }> : (tensor<4x8xf32>, tensor<4x1xi32>) -> tensor<4x8xf32>

  return %out1, %out2 : tensor<4x8xf32>, tensor<4x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr input1 = MakeBF16Tensor(context, {4, 8}, 1.0f);
  mlir::DenseElementsAttr input2 = MakeBF16Tensor(context, {4, 8}, 2.0f);
  auto i1_type =
      mlir::RankedTensorType::get({4}, mlir::IntegerType::get(&context, 1));
  bool mask1_vals[] = {true, false, true, false};
  bool mask2_vals[] = {false, true, false, true};
  auto mask1 = mlir::DenseElementsAttr::get(
      i1_type, llvm::ArrayRef<bool>(mask1_vals, 4));
  auto mask2 = mlir::DenseElementsAttr::get(
      i1_type, llvm::ArrayRef<bool>(mask2_vals, 4));
  mlir::DenseElementsAttr indices =
      MakeI32Tensor(context, {4, 1}, {2, 0, 3, 1});

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {input1, input2, mask1, mask2, indices}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 4);

  auto actual = mlir::stablehlo::evalModule(
      *opt_mod, {input1, input2, mask1, mask2, indices}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (size_t res_idx = 0; res_idx < 2; ++res_idx) {
    auto exp_it = (*expected)[res_idx].value_begin<float>();
    auto act_it = (*actual)[res_idx].value_begin<float>();
    for (; exp_it != (*expected)[res_idx].value_end<float>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ(*exp_it, *act_it);
    }
  }
}

TEST(CanonicalizeMoEGatherScatterTest,
     GatherOrUnwrapInversePerm1DCancellationNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @gather_inverse_perm_cancel(
    %operand: tensor<4x8xbf16>,
    %scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %perm = stablehlo.constant dense<[2, 0, 3, 1]> : tensor<4xi32>
  %inv_perm = "stablehlo.scatter"(%iota, %perm, %iota) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    stablehlo.return %rhs : tensor<i32>
  }) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  %perm_2d = "stablehlo.reshape"(%perm) : (tensor<4xi32>) -> tensor<4x1xi32>
  %v_gather = "stablehlo.gather"(%scale, %perm_2d) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1>
  }> : (tensor<4xf32>, tensor<4x1xi32>) -> tensor<4xf32>

  %scale_bcast = "stablehlo.broadcast_in_dim"(%v_gather) <{
    broadcast_dimensions = array<i64: 0>
  }> : (tensor<4xf32>) -> tensor<4x8xf32>
  %widen = "stablehlo.convert"(%operand)
      : (tensor<4x8xbf16>) -> tensor<4x8xf32>
  %scaled = "stablehlo.multiply"(%widen, %scale_bcast)
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

  %inv_perm_2d = "stablehlo.reshape"(%inv_perm)
      : (tensor<4xi32>) -> tensor<4x1xi32>
  %out = "stablehlo.gather"(%scaled, %inv_perm_2d) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 8>
  }> : (tensor<4x8xf32>, tensor<4x1xi32>) -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr operand = MakeBF16Tensor(context, {4, 8}, 2.0f);
  mlir::DenseElementsAttr scale = MakeF32Tensor(context, {4}, 0.75f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {operand, scale}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  // The gather on scale cancelled, leaving only the sunk operand gather.
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {operand, scale}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

TEST(CanonicalizeMoEGatherScatterTest, SinkRowOpsPastGatherMultiUseRejected) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @multi_use_reject(
    %operand: tensor<4x8xbf16>,
    %indices: tensor<4x1xi32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %widen = "stablehlo.convert"(%operand)
      : (tensor<4x8xbf16>) -> tensor<4x8xf32>
  %out = "stablehlo.gather"(%widen, %indices) <{
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 8>
  }> : (tensor<4x8xf32>, tensor<4x1xi32>) -> tensor<4x8xf32>
  return %out, %widen : tensor<4x8xf32>, tensor<4x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);
  opt_mod->walk([&](mlir::stablehlo::GatherOp op) {
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
    EXPECT_TRUE(in_type.getElementType().isF32());
  });
}

TEST(CanonicalizeMoEGatherScatterTest,
     BijectiveScatterBroadcastedTokensUnwrappingNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @scatter_bcast_tokens(
    %tokens: tensor<2x8xf32>) -> tensor<4x8xf32> {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x8xf32>
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %perm = stablehlo.constant dense<[2, 0, 3, 1]> : tensor<4xi32>
  %inv_perm = "stablehlo.scatter"(%iota, %perm, %iota) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    stablehlo.return %rhs : tensor<i32>
  }) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  %bcast = "stablehlo.broadcast_in_dim"(%tokens) <{
    broadcast_dimensions = array<i64: 0, 2>
  }> : (tensor<2x8xf32>) -> tensor<2x2x8xf32>
  %updates = "stablehlo.reshape"(%bcast)
      : (tensor<2x2x8xf32>) -> tensor<4x8xf32>

  %out = "stablehlo.scatter"(%zeros, %inv_perm, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<4x8xf32>, tensor<4xi32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr tokens = MakeF32Tensor(context, {2, 8}, 4.0f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {tokens}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {tokens}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

TEST(CanonicalizeMoEGatherScatterTest,
     BijectiveScatterTupleWrappedIndexNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @scatter_tuple_wrapped_index(
    %updates: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x8xf32>
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %perm = stablehlo.constant dense<[2, 0, 3, 1]> : tensor<4xi32>
  %inv_perm = "stablehlo.scatter"(%iota, %perm, %iota) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    stablehlo.return %rhs : tensor<i32>
  }) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  %dummy = stablehlo.constant dense<0> : tensor<4xi32>
  %tup = "stablehlo.tuple"(%inv_perm, %dummy)
      : (tensor<4xi32>, tensor<4xi32>) -> tuple<tensor<4xi32>, tensor<4xi32>>
  %gte = "stablehlo.get_tuple_element"(%tup) <{index = 0 : i32}>
      : (tuple<tensor<4xi32>, tensor<4xi32>>) -> tensor<4xi32>

  %out = "stablehlo.scatter"(%zeros, %gte, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<4x8xf32>, tensor<4xi32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr updates = MakeF32Tensor(context, {4, 8}, 5.0f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {updates}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {updates}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

TEST(CanonicalizeMoEGatherScatterTest, NonPermutationScatterNotLowered) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @scatter_non_perm(
    %updates: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x8xf32>
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %non_perm = stablehlo.constant dense<[0, 1, 1, 2]> : tensor<4xi32>
  %scatter1 = "stablehlo.scatter"(%iota, %non_perm, %iota) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    stablehlo.return %rhs : tensor<i32>
  }) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  %out = "stablehlo.scatter"(%zeros, %scatter1, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<4x8xf32>, tensor<4xi32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeMoEGatherScatterPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 0);
}

TEST(CanonicalizeMoEGatherScatterTest,
     FullPipelineExecutionViaRunTorchTpuOptimizationPasses) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @pipeline_moe_scatter(
    %updates: tensor<4x2x8xf32>) -> tensor<4x2x8xf32> {
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<4x2x8xf32>
  %iota = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  %perm = stablehlo.constant dense<[2, 0, 3, 1]> : tensor<4xi32>
  %inv_perm = "stablehlo.scatter"(%iota, %perm, %iota) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    stablehlo.return %rhs : tensor<i32>
  }) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  %out = "stablehlo.scatter"(%zeros, %inv_perm, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  }> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<4x2x8xf32>, tensor<4xi32>, tensor<4x2x8xf32>)
      -> tensor<4x2x8xf32>
  return %out : tensor<4x2x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr updates = MakeF32Tensor(context, {4, 2, 8}, 3.0f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {updates}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {updates}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*exp_it, *act_it);
  }
}

}  // namespace
}  // namespace torch_tpu
