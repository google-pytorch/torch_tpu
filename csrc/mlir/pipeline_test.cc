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

#include "csrc/mlir/pipeline.h"

#include "absl/status/status_matchers.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"  // NOLINT
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

// IR-level tests of the optimization pipeline are lit tests run through
// `torch-tpu-optimization-pipeline`. This file only covers the C++ entry
// point and smoke tests.

namespace torch_tpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using testing::HasSubstr;

void InitMlirContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

TEST(PipelineTest, RunOptimizationPassesFailsOnInvalidModule) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create a non-external FuncOp with an empty block (missing terminator),
  // which fails MLIR post-pass verification during pm.run().
  auto func = mlir::func::FuncOp::create(builder.getUnknownLoc(), "invalid_fn",
                                         builder.getFunctionType({}, {}));
  func.addEntryBlock();
  module->push_back(func);

  EXPECT_THAT(
      RunTorchTpuOptimizationPasses(*module, &context),
      StatusIs(error::kInternal,
               HasSubstr("Failed to run custom MLIR optimization passes")));
}

TEST(PipelineTest, CancelsInverseTransposeAroundSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x8x16xf32>) -> tensor<1x4x8x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x4x8x16xf32>) -> tensor<8x4x16x1xf32>
    %sl = stablehlo.slice %t1 [0:8, 0:4, 0:8, 0:1]
        : (tensor<8x4x16x1xf32>) -> tensor<8x4x8x1xf32>
    %t2 = stablehlo.transpose %sl, dims = [3, 1, 0, 2]
        : (tensor<8x4x8x1xf32>) -> tensor<1x4x8x8xf32>
    return %t2 : tensor<1x4x8x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitMlirContext(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int transpose_count = 0;
  int slice_count = 0;
  module->walk([&](mlir::stablehlo::TransposeOp) { ++transpose_count; });
  module->walk([&](mlir::stablehlo::SliceOp) { ++slice_count; });
  EXPECT_EQ(transpose_count, 0);
  EXPECT_EQ(slice_count, 1);
}

TEST(PipelineTest, RunOptimizationPassesSucceedsOnValidModule) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @trivial_fn(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    return %arg0 : tensor<2x3xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());
}

TEST(PipelineTest, CanonicalizesIndexArithmeticScatterSmokeTest) {
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
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x4xf32>, tensor<16xi32>,
          tensor<16x4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());

  int scatter_count = 0;
  int reduce_count = 0;
  module->walk([&](mlir::stablehlo::ScatterOp) { ++scatter_count; });
  module->walk([&](mlir::stablehlo::ReduceOp) { ++reduce_count; });
  EXPECT_EQ(scatter_count, 0);
  EXPECT_EQ(reduce_count, 1);
}

TEST(PipelineTest, CanonicalizesReductionsInPipeline) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();

  constexpr llvm::StringRef kIr = R"mlir(
module {
  func.func @test_reduction(%arg0: tensor<4x16xbf16>) -> tensor<4xf32> {
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<4x16xbf16>) -> tensor<4x16xf32>
    %1 = stablehlo.tanh %0 : tensor<4x16xf32>
    %2 = stablehlo.reduce(%1 init: %neg_inf) across dimensions = [1]
        : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    return %2 : tensor<4xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());

  auto func = module->lookupSymbol<mlir::func::FuncOp>("test_reduction");
  ASSERT_TRUE(func);
  bool saw_hoisted_tanh = false;
  func.walk([&](mlir::stablehlo::TanhOp tanh_op) {
    auto ty = mlir::cast<mlir::RankedTensorType>(tanh_op.getType());
    if (ty.getRank() == 1) {
      saw_hoisted_tanh = true;
    }
  });
  EXPECT_TRUE(saw_hoisted_tanh);
}
}  // namespace
}  // namespace torch_tpu
