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
#include "mlir/Dialect/Func/IR/FuncDialect.h"  // NOLINT
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  context
      .loadDialect<mlir::func::FuncDialect>();  // NOLINT(misc-include-cleaner)

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

}  // namespace
}  // namespace torch_tpu
