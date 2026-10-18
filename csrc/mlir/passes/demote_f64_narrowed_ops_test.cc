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

#include <string>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

std::string PrintModule(mlir::ModuleOp module) {
  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  return out;
}

TEST(DemoteF64NarrowedOpsTest, DemotesCompiledAdamScalarBiasCorrectionChain) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @adam_scalar_chain(%step: tensor<f32>) -> (tensor<f32>, tensor<bf16>, tensor<bf16>) {
  %c1_f64 = stablehlo.constant dense<1.0> : tensor<f64>
  %lr_f64 = stablehlo.constant dense<1.0e-4> : tensor<f64>
  %eps_f64 = stablehlo.constant dense<1.0e-8> : tensor<f64>
  %b1 = stablehlo.constant dense<0.9> : tensor<f32>
  %b2 = stablehlo.constant dense<0.999> : tensor<f32>
  %half = stablehlo.constant dense<0.5> : tensor<f32>
  %one_f32 = stablehlo.constant dense<1.0> : tensor<f32>

  %step_f64 = stablehlo.convert %step : (tensor<f32>) -> tensor<f64>
  %step_inc_f64 = stablehlo.add %step_f64, %c1_f64 : tensor<f64>
  %step_inc = stablehlo.convert %step_inc_f64 : (tensor<f64>) -> tensor<f32>

  %pow1 = stablehlo.power %b1, %step_inc : tensor<f32>
  %pow1_f64 = stablehlo.convert %pow1 : (tensor<f32>) -> tensor<f64>
  %bc1_f64 = stablehlo.subtract %c1_f64, %pow1_f64 : tensor<f64>
  %bc1 = stablehlo.convert %bc1_f64 : (tensor<f64>) -> tensor<f32>
  %inv_bc1 = stablehlo.divide %one_f32, %bc1 : tensor<f32>
  %inv_bc1_f64 = stablehlo.convert %inv_bc1 : (tensor<f32>) -> tensor<f64>
  %step_size_f64 = stablehlo.multiply %inv_bc1_f64, %lr_f64 : tensor<f64>
  %step_size = stablehlo.convert %step_size_f64 : (tensor<f64>) -> tensor<f32>

  %pow2 = stablehlo.power %b2, %step_inc : tensor<f32>
  %pow2_f64 = stablehlo.convert %pow2 : (tensor<f32>) -> tensor<f64>
  %bc2_f64 = stablehlo.subtract %c1_f64, %pow2_f64 : tensor<f64>
  %bc2 = stablehlo.convert %bc2_f64 : (tensor<f64>) -> tensor<f32>
  %bc2_sqrt = stablehlo.power %bc2, %half : tensor<f32>

  %full_step = stablehlo.multiply %step_size, %bc2_sqrt : tensor<f32>
  %full_step_bf16 = stablehlo.convert %full_step : (tensor<f32>) -> tensor<bf16>

  %bc2_sqrt_f64 = stablehlo.convert %bc2_sqrt : (tensor<f32>) -> tensor<f64>
  %eps_scaled_f64 = stablehlo.multiply %bc2_sqrt_f64, %eps_f64 : tensor<f64>
  %eps_scaled = stablehlo.convert %eps_scaled_f64 : (tensor<f64>) -> tensor<f32>
  %eps_bf16 = stablehlo.convert %eps_scaled : (tensor<f32>) -> tensor<bf16>

  return %step_inc, %full_step_bf16, %eps_bf16 : tensor<f32>, tensor<bf16>, tensor<bf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());

  std::string out = PrintModule(*module);
  EXPECT_THAT(out, ::testing::Not(::testing::HasSubstr("f64")));
  EXPECT_THAT(out, ::testing::HasSubstr("stablehlo.sqrt"));
}

TEST(DemoteF64NarrowedOpsTest, PreservesGenuineF64InputsAndOutputs) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @preserve_f64(%x: tensor<4xf64>, %y: tensor<4xf32>) -> (tensor<4xf64>, tensor<4xf32>) {
  %y_f64 = stablehlo.convert %y : (tensor<4xf32>) -> tensor<4xf64>
  %sum = stablehlo.add %x, %y_f64 : tensor<4xf64>
  %sum_f32 = stablehlo.convert %sum : (tensor<4xf64>) -> tensor<4xf32>
  return %sum, %sum_f32 : tensor<4xf64>, tensor<4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());

  std::string out = PrintModule(*module);
  EXPECT_THAT(out, ::testing::HasSubstr("tensor<4xf64>"));
}

}  // namespace
}  // namespace torch_tpu
