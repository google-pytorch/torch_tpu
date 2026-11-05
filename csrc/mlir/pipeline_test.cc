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

#include <cstdlib>

#include "absl/status/status_matchers.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/mlir/registration.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

// IR-level tests of the optimization pipeline are lit tests run through
// `torch-tpu-optimization-pipeline`. This file only covers the C++ entry
// point.

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::HasSubstr;

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

TEST(PipelineTest,
     RegisterTorchTpuPassesFindsCanonicalizeGatherScatterReductions) {
  RegisterTorchTpuPasses();
  const mlir::PassInfo* info = mlir::PassInfo::lookup(
      "torch-tpu-canonicalize-gather-scatter-reductions");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getPassArgument(),
            "torch-tpu-canonicalize-gather-scatter-reductions");
  EXPECT_FALSE(info->getPassDescription().empty());

  mlir::OpPassManager func_pm(mlir::func::FuncOp::getOperationName());
  EXPECT_TRUE(mlir::succeeded(mlir::parsePassPipeline(
      "torch-tpu-canonicalize-gather-scatter-reductions", func_pm)));
  ASSERT_FALSE(func_pm.empty());
  EXPECT_EQ(func_pm.getPasses().begin()->getArgument(),
            "torch-tpu-canonicalize-gather-scatter-reductions");
}

TEST(PipelineDeathTest, CanonicalizeGatherScatterReductionsDisabledByEnvVar) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %idx_col: tensor<4x1x1xi64>,
      %updates: tensor<4x1xf32>) -> tensor<4xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<4x8xf32>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %bcast_iota = stablehlo.broadcast_in_dim %iota, dims = [0]
        : (tensor<4xi64>) -> tensor<4x1xi64>
    %row_iota = stablehlo.reshape %bcast_iota
        : (tensor<4x1xi64>) -> tensor<4x1x1xi64>
    %concat_indices = stablehlo.concatenate %row_iota, %idx_col, dim = 2
        : (tensor<4x1x1xi64>, tensor<4x1x1xi64>) -> tensor<4x1x2xi64>
    %scattered = "stablehlo.scatter"(%base, %concat_indices, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [], inserted_window_dims = [0, 1],
        input_batching_dims = [], scatter_indices_batching_dims = [],
        scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) : (tensor<4x8xf32>, tensor<4x1x2xi64>,
          tensor<4x1xf32>) -> tensor<4x8xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<4xf32>
  }
}
)mlir";

        SafeSetEnv<
            kTorchTpuInternalEnableCanonicalizeGatherScatterReductionsEnvVar>(
            "0");
        mlir::MLIRContext context;
        InitContext(context);

        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
        if (!module) {
          std::exit(1);
        }
        if (!RunTorchTpuOptimizationPasses(*module, &context).ok()) {
          std::exit(2);
        }
        if (CountOps<mlir::stablehlo::ScatterOp>(*module) == 0) {
          std::exit(3);
        }
        std::exit(0);
      },
      testing::ExitedWithCode(0), "");
}
}  // namespace
}  // namespace torch_tpu
