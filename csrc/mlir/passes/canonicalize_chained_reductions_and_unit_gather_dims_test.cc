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

#include "csrc/mlir/passes/canonicalize_chained_reductions_and_unit_gather_dims.h"

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOk;

void RegisterDialects(mlir::MLIRContext& context) {
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

TEST(CanonicalizeChainedReductionsAndUnitGatherDimsTest,
     FusesChainedKeepDimMeanReductionsAcrossFloatRoundtrip) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(%x: tensor<1x96x32x32xbf16>) -> tensor<1x96x1x1xbf16> {
      %x_f32 = stablehlo.convert %x : (tensor<1x96x32x32xbf16>) -> tensor<1x96x32x32xf32>
      %zero = stablehlo.constant dense<0.0> : tensor<f32>
      %r1 = stablehlo.reduce(%x_f32 init: %zero) across dimensions = [2] : (tensor<1x96x32x32xf32>, tensor<f32>) -> tensor<1x96x32xf32>
        reducer(%a: tensor<f32>, %b: tensor<f32>) {
          %sum = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }
      %r1_kd = stablehlo.reshape %r1 : (tensor<1x96x32xf32>) -> tensor<1x96x1x32xf32>
      %c32_1 = stablehlo.constant dense<32.0> : tensor<1x96x1x32xf32>
      %div1 = stablehlo.divide %r1_kd, %c32_1 : tensor<1x96x1x32xf32>
      %mid_bf16 = stablehlo.convert %div1 : (tensor<1x96x1x32xf32>) -> tensor<1x96x1x32xbf16>
      %mid_f32 = stablehlo.convert %mid_bf16 : (tensor<1x96x1x32xbf16>) -> tensor<1x96x1x32xf32>
      %r2 = stablehlo.reduce(%mid_f32 init: %zero) across dimensions = [3] : (tensor<1x96x1x32xf32>, tensor<f32>) -> tensor<1x96x1xf32>
        reducer(%a: tensor<f32>, %b: tensor<f32>) {
          %sum = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }
      %r2_kd = stablehlo.reshape %r2 : (tensor<1x96x1xf32>) -> tensor<1x96x1x1xf32>
      %c32_2 = stablehlo.constant dense<32.0> : tensor<1x96x1x1xf32>
      %div2 = stablehlo.divide %r2_kd, %c32_2 : tensor<1x96x1x1xf32>
      %out = stablehlo.convert %div2 : (tensor<1x96x1x1xf32>) -> tensor<1x96x1x1xbf16>
      return %out : tensor<1x96x1x1xbf16>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConvertOp>(*module), 2);

  module->walk([&](mlir::stablehlo::ReduceOp op) {
    EXPECT_THAT(op.getDimensions(), ::testing::ElementsAre(2, 3));
  });
  module->walk([&](mlir::stablehlo::DivOp op) {
    auto cst_op = op.getRhs().getDefiningOp<mlir::stablehlo::ConstantOp>();
    ASSERT_TRUE(cst_op);
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst_op.getValue());
    ASSERT_TRUE(dense && dense.isSplat());
    EXPECT_DOUBLE_EQ(dense.getSplatValue<llvm::APFloat>().convertToDouble(),
                     1024.0);
  });
}

TEST(CanonicalizeChainedReductionsAndUnitGatherDimsTest,
     FusesChainedDropDimAdditiveReductions) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(%x: tensor<2x8x16x32xf32>) -> tensor<2x32xf32> {
      %zero = stablehlo.constant dense<0.0> : tensor<f32>
      %r1 = stablehlo.reduce(%x init: %zero) across dimensions = [1] : (tensor<2x8x16x32xf32>, tensor<f32>) -> tensor<2x16x32xf32>
        reducer(%a: tensor<f32>, %b: tensor<f32>) {
          %sum = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }
      %r2 = stablehlo.reduce(%r1 init: %zero) across dimensions = [1] : (tensor<2x16x32xf32>, tensor<f32>) -> tensor<2x32xf32>
        reducer(%a: tensor<f32>, %b: tensor<f32>) {
          %sum = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }
      return %r2 : tensor<2x32xf32>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*module), 1);
  module->walk([&](mlir::stablehlo::ReduceOp op) {
    EXPECT_THAT(op.getDimensions(), ::testing::ElementsAre(1, 2));
  });
}

TEST(CanonicalizeChainedReductionsAndUnitGatherDimsTest,
     PreservesChainedReductionWhenInitIsNonZero) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(%x: tensor<2x8x16xf32>) -> tensor<2xf32> {
      %one = stablehlo.constant dense<1.0> : tensor<f32>
      %r1 = stablehlo.reduce(%x init: %one) across dimensions = [1] : (tensor<2x8x16xf32>, tensor<f32>) -> tensor<2x16xf32>
        reducer(%a: tensor<f32>, %b: tensor<f32>) {
          %sum = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }
      %r2 = stablehlo.reduce(%r1 init: %one) across dimensions = [1] : (tensor<2x16xf32>, tensor<f32>) -> tensor<2xf32>
        reducer(%a: tensor<f32>, %b: tensor<f32>) {
          %sum = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }
      return %r2 : tensor<2xf32>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*module), 2);
}

TEST(CanonicalizeChainedReductionsAndUnitGatherDimsTest,
     SqueezesUnitGatherDimAndNarrowsEntryNegativeIndexWrapToI32) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(
        %table: tensor<225x16xbf16>,
        %idx_2d: tensor<64x64xi64>) -> tensor<4096x16xbf16> {
      %idx_flat = stablehlo.reshape %idx_2d : (tensor<64x64xi64>) -> tensor<4096xi64>
      %zero = stablehlo.constant dense<0> : tensor<4096xi64>
      %cmp = stablehlo.compare LT, %idx_flat, %zero : (tensor<4096xi64>, tensor<4096xi64>) -> tensor<4096xi1>
      %extent = stablehlo.constant dense<225> : tensor<4096xi64>
      %shifted = stablehlo.add %idx_flat, %extent : tensor<4096xi64>
      %wrapped = stablehlo.select %cmp, %shifted, %idx_flat : tensor<4096xi1>, tensor<4096xi64>
      %indices = stablehlo.reshape %wrapped : (tensor<4096xi64>) -> tensor<4096x1xi64>
      %out = "stablehlo.gather"(%table, %indices) <{
        dimension_numbers = #stablehlo.gather<
          offset_dims = [1],
          collapsed_slice_dims = [0],
          start_index_map = [0],
          index_vector_dim = 1
        >,
        indices_are_sorted = false,
        slice_sizes = array<i64: 1, 16>
      }> : (tensor<225x16xbf16>, tensor<4096x1xi64>) -> tensor<4096x16xbf16>
      return %out : tensor<4096x16xbf16>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*module), 1);
  module->walk([&](mlir::stablehlo::GatherOp gather_op) {
    auto indices_type = mlir::cast<mlir::RankedTensorType>(
        gather_op.getStartIndices().getType());
    EXPECT_EQ(indices_type.getRank(), 1);
    EXPECT_TRUE(indices_type.getElementType().isInteger(32));
    EXPECT_EQ(gather_op.getDimensionNumbers().getIndexVectorDim(), 1);
  });
}

TEST(CanonicalizeChainedReductionsAndUnitGatherDimsTest,
     SqueezesUnitGatherDimWithoutNegativeIndexWrap) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(
        %table: tensor<225x16xbf16>,
        %indices: tensor<4096x1xi32>) -> tensor<4096x16xbf16> {
      %out = "stablehlo.gather"(%table, %indices) <{
        dimension_numbers = #stablehlo.gather<
          offset_dims = [1],
          collapsed_slice_dims = [0],
          start_index_map = [0],
          index_vector_dim = 1
        >,
        indices_are_sorted = false,
        slice_sizes = array<i64: 1, 16>
      }> : (tensor<225x16xbf16>, tensor<4096x1xi32>) -> tensor<4096x16xbf16>
      return %out : tensor<4096x16xbf16>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*module), 1);
  module->walk([&](mlir::stablehlo::GatherOp gather_op) {
    auto indices_type = mlir::cast<mlir::RankedTensorType>(
        gather_op.getStartIndices().getType());
    EXPECT_EQ(indices_type.getRank(), 1);
    EXPECT_TRUE(indices_type.getElementType().isInteger(32));
    EXPECT_EQ(gather_op.getDimensionNumbers().getIndexVectorDim(), 1);
  });
}

TEST(CanonicalizeChainedReductionsAndUnitGatherDimsTest,
     PreservesNegativeIndexWrapWhenShiftConstantMismatchesTableExtent) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(
        %table: tensor<225x16xbf16>,
        %idx_flat: tensor<4096xi64>) -> tensor<4096x16xbf16> {
      %zero = stablehlo.constant dense<0> : tensor<4096xi64>
      %cmp = stablehlo.compare LT, %idx_flat, %zero :
          (tensor<4096xi64>, tensor<4096xi64>) -> tensor<4096xi1>
      %extent = stablehlo.constant dense<100> : tensor<4096xi64>
      %shifted = stablehlo.add %idx_flat, %extent : tensor<4096xi64>
      %wrapped = stablehlo.select %cmp, %shifted, %idx_flat :
          tensor<4096xi1>, tensor<4096xi64>
      %out = "stablehlo.gather"(%table, %wrapped) <{
        dimension_numbers = #stablehlo.gather<
          offset_dims = [1],
          collapsed_slice_dims = [0],
          start_index_map = [0],
          index_vector_dim = 1
        >,
        indices_are_sorted = false,
        slice_sizes = array<i64: 1, 16>
      }> : (tensor<225x16xbf16>, tensor<4096xi64>) -> tensor<4096x16xbf16>
      return %out : tensor<4096x16xbf16>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::GatherOp>(*module), 1);
  module->walk([&](mlir::stablehlo::GatherOp gather_op) {
    auto indices_type = mlir::cast<mlir::RankedTensorType>(
        gather_op.getStartIndices().getType());
    EXPECT_TRUE(indices_type.getElementType().isInteger(64));
  });
}

TEST(CanonicalizeChainedReductionsAndUnitGatherDimsTest,
     PreservesChainedReductionWhenFirstReductionHasMultipleUses) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(
        %x: tensor<2x8x16xf32>) -> (tensor<2xf32>, tensor<2x16xf32>) {
      %zero = stablehlo.constant dense<0.0> : tensor<f32>
      %r1 = stablehlo.reduce(%x init: %zero) across dimensions = [1] :
          (tensor<2x8x16xf32>, tensor<f32>) -> tensor<2x16xf32>
        reducer(%a: tensor<f32>, %b: tensor<f32>) {
          %sum = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }
      %r2 = stablehlo.reduce(%r1 init: %zero) across dimensions = [1] :
          (tensor<2x16xf32>, tensor<f32>) -> tensor<2xf32>
        reducer(%a: tensor<f32>, %b: tensor<f32>) {
          %sum = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }
      return %r2, %r1 : tensor<2xf32>, tensor<2x16xf32>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*module), 2);
}

}  // namespace
}  // namespace torch_tpu
