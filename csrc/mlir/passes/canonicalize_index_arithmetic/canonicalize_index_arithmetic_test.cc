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

#include "csrc/mlir/passes/canonicalize_index_arithmetic/canonicalize_index_arithmetic.h"  // NOLINT

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"  // NOLINT
#include "mlir/Dialect/Func/IR/FuncOps.h"      // NOLINT
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
  module.walk([&count](OpTy) { ++count; });
  return count;
}

mlir::DenseElementsAttr MakeSequentialF32Tensor(mlir::MLIRContext& ctx,
                                                llvm::ArrayRef<int64_t> shape,
                                                float base = 0.5f) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 7 + 1) % 43) * 0.1f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeI32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<int32_t> vals) {
  auto type =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 32));
  return mlir::DenseElementsAttr::get(type, vals);
}

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape,
                                       llvm::ArrayRef<float> f_vals) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(f_vals.size());
  for (float f : f_vals) {
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

mlir::DenseElementsAttr MakeI64Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<int64_t> vals) {
  auto type =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 64));
  return mlir::DenseElementsAttr::get(type, vals);
}

mlir::DenseElementsAttr MakeUi64Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape,
                                       llvm::ArrayRef<uint64_t> vals) {
  auto type = mlir::RankedTensorType::get(
      shape, mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Unsigned));
  return mlir::DenseElementsAttr::get(type, vals);
}

void ExpectAttrsClose(mlir::DenseElementsAttr actual,
                      mlir::DenseElementsAttr expected, float tol = 1e-4f) {
  ASSERT_EQ(actual.getType(), expected.getType());
  auto act_vals = actual.getValues<float>();
  auto exp_vals = expected.getValues<float>();
  auto act_it = act_vals.begin();
  auto exp_it = exp_vals.begin();
  for (; exp_it != exp_vals.end(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, tol);
  }
}

void ExpectBf16AttrsClose(mlir::DenseElementsAttr actual,
                          mlir::DenseElementsAttr expected, float tol = 1e-2f) {
  ASSERT_EQ(actual.getType(), expected.getType());
  auto act_it = actual.value_begin<llvm::APFloat>();
  auto exp_it = expected.value_begin<llvm::APFloat>();
  for (; exp_it != expected.value_end<llvm::APFloat>(); ++exp_it, ++act_it) {
    EXPECT_NEAR((*act_it).convertToFloat(), (*exp_it).convertToFloat(), tol);
  }
}

TEST(CanonicalizeIndexArithmeticTest,
     SingleRowAdditiveScatterToMaskedReduceNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%updates: tensor<16x4xf32>) -> tensor<1x4xf32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<1x4xf32>
    %idx = stablehlo.constant dense<0> : tensor<16xi32>
    %0 = "stablehlo.scatter"(%c0, %idx, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0],
        update_window_dims = [1],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = false
    }> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %1 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x4xf32>, tensor<16xi32>, tensor<16x4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*opt_module), 1);

  auto in_updates = MakeSequentialF32Tensor(context, {16, 4});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {in_updates}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_updates}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeIndexArithmeticTest,
     SingleRowAdditiveScatterNonConstantIndicesNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %indices: tensor<16xi32>,
      %updates: tensor<16x4xf32>) -> tensor<1x4xf32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<1x4xf32>
    %0 = "stablehlo.scatter"(%c0, %indices, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0],
        update_window_dims = [1],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = false
    }> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %1 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x4xf32>, tensor<16xi32>, tensor<16x4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*opt_module), 1);

  std::vector<int32_t> idx_vals =  // INT_VEC_OK
      {0, 0, 1, 0, -1, 0, 0, 2, 0, 0, 0, -5, 0, 0, 0, 0};
  auto in_idx = MakeI32Tensor(context, {16}, idx_vals);
  auto in_updates = MakeSequentialF32Tensor(context, {16, 4});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {in_idx, in_updates}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {in_idx, in_updates}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeIndexArithmeticTest, MultiRowAdditiveScatterRemainsScatter) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %indices: tensor<16xi32>,
      %updates: tensor<16x4xf32>) -> tensor<8x4xf32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>
    %0 = "stablehlo.scatter"(%c0, %indices, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0],
        update_window_dims = [1],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = false
    }> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %1 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<8x4xf32>, tensor<16xi32>, tensor<16x4xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Multi-row indicator contraction is dropped; multi-row scatter remains
  // a ScatterOp.
  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 0);
}

TEST(CanonicalizeIndexArithmeticTest, NegativeCaseNonzeroBaseRemainsScatter) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %base: tensor<3x4xf32>,
      %updates: tensor<2x4xf32>) -> tensor<3x4xf32> {
    %idx = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
    %0 = "stablehlo.scatter"(%base, %idx, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0],
        update_window_dims = [1],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = false
    }> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %1 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<3x4xf32>, tensor<2xi32>, tensor<2x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*module), 1);
}

TEST(CanonicalizeIndexArithmeticTest, SqueezeTrailingUnitGatherDimNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%table: tensor<10x20xf32>) -> tensor<4x20xf32> {
    %idx = stablehlo.constant dense<[[0], [2], [5], [9]]> : tensor<4x1xi64>
    %0 = "stablehlo.gather"(%table, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 20>
    }> : (tensor<10x20xf32>, tensor<4x1xi64>) -> tensor<4x20xf32>
    return %0 : tensor<4x20xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Structural assert: gather index rank squeezed from 2 to 1, and
  // index_vector_dim updated.
  opt_module->walk([](mlir::stablehlo::GatherOp gather) {
    auto idx_type =
        mlir::cast<mlir::RankedTensorType>(gather.getStartIndices().getType());
    EXPECT_EQ(idx_type.getRank(), 1);
    EXPECT_EQ(idx_type.getDimSize(0), 4);
    EXPECT_EQ(gather.getDimensionNumbers().getIndexVectorDim(), 1);
  });

  auto in_table = MakeSequentialF32Tensor(context, {10, 20});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_table}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_table}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeIndexArithmeticTest, StepAIotaHoistConvertNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main() -> tensor<8x4xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0]
        : (tensor<8xi64>) -> tensor<8x4xi64>
    %2 = stablehlo.convert %1 : (tensor<8x4xi64>) -> tensor<8x4xf32>
    return %2 : tensor<8x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(
      *ref_module, llvm::ArrayRef<mlir::DenseElementsAttr>{}, config);
  auto actual = mlir::stablehlo::evalModule(
      *opt_module, llvm::ArrayRef<mlir::DenseElementsAttr>{}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeIndexArithmeticTest,
     NegativeCaseUnsignedIndexWrapPreservesUi64) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%operand: tensor<100x32xf32>, %raw_idx: tensor<4xui64>)
      -> tensor<4x32xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<4xui64>
    %c100 = stablehlo.constant dense<100> : tensor<4xui64>
    %cmp = stablehlo.compare LT, %raw_idx, %c0 :
        (tensor<4xui64>, tensor<4xui64>) -> tensor<4xi1>
    %add = stablehlo.add %raw_idx, %c100 : tensor<4xui64>
    %wrapped = stablehlo.select %cmp, %add, %raw_idx :
        tensor<4xi1>, tensor<4xui64>
    %0 = "stablehlo.gather"(%operand, %wrapped) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 32>
    }> : (tensor<100x32xf32>, tensor<4xui64>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Verifies that unsigned ui64 index wrap is NOT narrowed to i32.
  opt_module->walk([](mlir::stablehlo::GatherOp gather) {
    auto idx_type =
        mlir::cast<mlir::RankedTensorType>(gather.getStartIndices().getType());
    auto int_type = mlir::cast<mlir::IntegerType>(idx_type.getElementType());
    EXPECT_TRUE(int_type.isUnsigned());
    EXPECT_EQ(int_type.getWidth(), 64);
  });
}

TEST(CanonicalizeIndexArithmeticTest,
     NegativeCaseNonMatchingWrapComparePreservesI64) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%operand: tensor<100x32xf32>, %raw_idx: tensor<4xi64>)
      -> tensor<4x32xf32> {
    %c5 = stablehlo.constant dense<5> : tensor<4xi64>
    %c100 = stablehlo.constant dense<100> : tensor<4xi64>
    %cmp = stablehlo.compare LT, %raw_idx, %c5 :
        (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    %add = stablehlo.add %raw_idx, %c100 : tensor<4xi64>
    %wrapped = stablehlo.select %cmp, %add, %raw_idx :
        tensor<4xi1>, tensor<4xi64>
    %0 = "stablehlo.gather"(%operand, %wrapped) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 32>
    }> : (tensor<100x32xf32>, tensor<4xi64>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Verifies that compare against 5 (not 0 or -1) does NOT narrow to i32.
  opt_module->walk([](mlir::stablehlo::GatherOp gather) {
    auto idx_type =
        mlir::cast<mlir::RankedTensorType>(gather.getStartIndices().getType());
    auto int_type = mlir::cast<mlir::IntegerType>(idx_type.getElementType());
    EXPECT_TRUE(int_type.isSignless());
    EXPECT_EQ(int_type.getWidth(), 64);
  });
}

TEST(CanonicalizeIndexArithmeticTest,
     NarrowNegativeIndexWrapOutOfBoundsI64Numerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%operand: tensor<10x20xf32>, %raw_idx: tensor<4xi64>)
      -> tensor<4x20xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<4xi64>
    %c10 = stablehlo.constant dense<10> : tensor<4xi64>
    %cmp = stablehlo.compare LT, %raw_idx, %c0 :
        (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    %add = stablehlo.add %raw_idx, %c10 : tensor<4xi64>
    %wrapped = stablehlo.select %cmp, %add, %raw_idx :
        tensor<4xi1>, tensor<4xi64>
    %0 = "stablehlo.gather"(%operand, %wrapped) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 20>
    }> : (tensor<10x20xf32>, tensor<4xi64>) -> tensor<4x20xf32>
    return %0 : tensor<4x20xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Verifies that gather was narrowed to i32 with clamp.
  opt_module->walk([](mlir::stablehlo::GatherOp gather) {
    auto idx_type =
        mlir::cast<mlir::RankedTensorType>(gather.getStartIndices().getType());
    EXPECT_TRUE(idx_type.getElementType().isInteger(32));
  });

  auto in_operand = MakeSequentialF32Tensor(context, {10, 20});
  // Indices include out-of-bounds 64-bit values that would wrap into valid i32
  // if unclamped.
  std::vector<int64_t> idx_vals = {// INT_VEC_OK
                                   2, -3, 2147483648LL, -4294967294LL};
  auto in_idx = MakeI64Tensor(context, {4}, idx_vals);

  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {in_operand, in_idx}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {in_operand, in_idx}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeIndexArithmeticTest, NarrowBoundedI64IndexSubgraph) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main() -> tensor<4xi1> {
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %c1 = stablehlo.constant dense<1> : tensor<4xi64>
    %add = stablehlo.add %iota, %c1 : tensor<4xi64>
    %c3 = stablehlo.constant dense<3> : tensor<4xi64>
    %cmp = stablehlo.compare LT, %add, %c3 :
        (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Confirms iota and add are narrowed from i64 to i32.
  opt_module->walk([](mlir::stablehlo::IotaOp iota) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(iota.getType());
    EXPECT_TRUE(rtt.getElementType().isInteger(32));
  });

  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(
      *ref_module, llvm::ArrayRef<mlir::DenseElementsAttr>{}, config);
  auto actual = mlir::stablehlo::evalModule(
      *opt_module, llvm::ArrayRef<mlir::DenseElementsAttr>{}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  EXPECT_EQ((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeIndexArithmeticTest,
     SingleRowScatterWideWidthToReduceNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%updates: tensor<16x256xf32>) -> tensor<1x256xf32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<1x256xf32>
    %idx = stablehlo.constant dense<0> : tensor<16xi32>
    %0 = "stablehlo.scatter"(%c0, %idx, %updates) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0],
        update_window_dims = [1],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = false
    }> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %1 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x256xf32>, tensor<16xi32>, tensor<16x256xf32>)
        -> tensor<1x256xf32>
    return %0 : tensor<1x256xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ScatterOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*opt_module), 1);

  auto in_updates = MakeSequentialF32Tensor(context, {16, 256});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {in_updates}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_updates}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeIndexArithmeticTest,
     NegativeCaseClampUnsoundCounterexampleRemainsI64) {
  // Counterexample: x in [0, 1], min = iota in [0, 3], max = 1000.
  // With unsound clamp interval formula lo = min(max(x.lo, mn.lo), mx.lo) and
  // hi = min(max(x.hi, mn.lo), mx.hi), hi would evaluate to
  // min(max(1, 0), 1000) = 1, making clamp output appear to be in [0, 1].
  // But actual clamp max is 3 (iota=3). When multiplied by 2^30, 3 * 2^30
  // overflows signed 32-bit int, whereas 1 * 2^30 would fit. Sound interval
  // correctly recognizes potential overflow and leaves the graph as i64.
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4xi64>) -> tensor<4xi1> {
    %c0 = stablehlo.constant dense<0> : tensor<4xi64>
    %c1 = stablehlo.constant dense<1> : tensor<4xi64>
    %x_clamped = stablehlo.clamp %c0, %x, %c1 : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %c1000 = stablehlo.constant dense<1000> : tensor<4xi64>
    %clamped = stablehlo.clamp %iota, %x_clamped, %c1000 : tensor<4xi64>
    %c_large = stablehlo.constant dense<1073741824> : tensor<4xi64>
    %mul = stablehlo.multiply %clamped, %c_large : tensor<4xi64>
    %cmp = stablehlo.compare LT, %mul, %c_large :
        (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  opt_module->walk([](mlir::stablehlo::MulOp mul_op) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(mul_op.getType());
    EXPECT_TRUE(rtt.getElementType().isSignlessInteger(64));
  });
}

TEST(CanonicalizeIndexArithmeticTest, NegativeCaseIntervalOverflowRemainsI64) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main() -> tensor<4xi1> {
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %c_overflow = stablehlo.constant dense<3000000000> : tensor<4xi64>
    %add = stablehlo.add %iota, %c_overflow : tensor<4xi64>
    %c_large = stablehlo.constant dense<4000000000> : tensor<4xi64>
    %cmp = stablehlo.compare LT, %add, %c_large :
        (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  opt_module->walk([](mlir::stablehlo::AddOp add_op) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(add_op.getType());
    EXPECT_TRUE(rtt.getElementType().isSignlessInteger(64));
  });
}

TEST(CanonicalizeIndexArithmeticTest, NegativeCaseUnsupportedSinkRemainsI64) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main() -> tensor<4xi64> {
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %c1 = stablehlo.constant dense<1> : tensor<4xi64>
    %add = stablehlo.add %iota, %c1 : tensor<4xi64>
    return %add : tensor<4xi64>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  opt_module->walk([](mlir::stablehlo::AddOp add_op) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(add_op.getType());
    EXPECT_TRUE(rtt.getElementType().isSignlessInteger(64));
  });
}

TEST(CanonicalizeIndexArithmeticTest, GuardedFloatToIntIntervalNarrowedToI32) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main() -> tensor<4xi1> {
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %f = stablehlo.convert %iota : (tensor<4xi64>) -> tensor<4xf32>
    %conv = stablehlo.convert %f : (tensor<4xf32>) -> tensor<4xi64>
    %c0_i64 = stablehlo.constant dense<0> : tensor<4xi64>
    %cmp_guard = stablehlo.compare LT, %iota, %c0_i64 :
        (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    %guarded_i64 = stablehlo.select %cmp_guard, %c0_i64, %conv :
        tensor<4xi1>, tensor<4xi64>
    %c10_i64 = stablehlo.constant dense<10> : tensor<4xi64>
    %add = stablehlo.add %guarded_i64, %c10_i64 : tensor<4xi64>
    %c50_i64 = stablehlo.constant dense<50> : tensor<4xi64>
    %cmp = stablehlo.compare LT, %add, %c50_i64 :
        (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %cmp : tensor<4xi1>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeIndexArithmeticPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  opt_module->walk([](mlir::stablehlo::AddOp add_op) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(add_op.getType());
    EXPECT_TRUE(rtt.getElementType().isInteger(32));
  });
}

}  // namespace
}  // namespace torch_tpu
