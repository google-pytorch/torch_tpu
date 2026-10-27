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

#include "csrc/mlir/passes/canonicalize_pointwise_conv_and_finite_sdpa_biases.h"

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
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

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     LowersUnitBatchPointwiseConvAndCommutesTransposesAcrossGluAndSilu) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(
        %x: tensor<1x128x1024xbf16>,
        %w1: tensor<2048x1024x1xbf16>,
        %b1: tensor<2048xbf16>,
        %ln_out: tensor<1x128x1024xbf16>,
        %w2: tensor<1024x1024x1xbf16>,
        %b2: tensor<1024xbf16>)
        -> (tensor<1x1024x128xbf16>, tensor<1x128x1024xbf16>) {
      %r0 = stablehlo.reshape %x : (tensor<1x128x1024xbf16>) -> tensor<128x1024x1xbf16>
      %t0 = stablehlo.transpose %r0, dims = [2, 1, 0] : (tensor<128x1024x1xbf16>) -> tensor<1x1024x128xbf16>
      %c1 = stablehlo.convolution(%t0, %w1) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x128xbf16>, tensor<2048x1024x1xbf16>) -> tensor<1x2048x128xbf16>
      %b1_bcast = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<2048xbf16>) -> tensor<1x2048x128xbf16>
      %c1_b = stablehlo.add %c1, %b1_bcast : tensor<1x2048x128xbf16>
      %glu_in = stablehlo.convert %c1_b : (tensor<1x2048x128xbf16>) -> tensor<1x2048x128xf32>
      %s0 = stablehlo.slice %glu_in [0:1, 0:1024, 0:128] : (tensor<1x2048x128xf32>) -> tensor<1x1024x128xf32>
      %s1 = stablehlo.slice %glu_in [0:1, 1024:2048, 0:128] : (tensor<1x2048x128xf32>) -> tensor<1x1024x128xf32>
      %neg1 = stablehlo.negate %s1 : tensor<1x1024x128xf32>
      %exp1 = stablehlo.exponential %neg1 : tensor<1x1024x128xf32>
      %one = stablehlo.constant dense<1.0> : tensor<1x1024x128xf32>
      %add1 = stablehlo.add %exp1, %one : tensor<1x1024x128xf32>
      %div1 = stablehlo.divide %one, %add1 : tensor<1x1024x128xf32>
      %mul1 = stablehlo.multiply %s0, %div1 : tensor<1x1024x128xf32>
      %glu_out = stablehlo.convert %mul1 : (tensor<1x1024x128xf32>) -> tensor<1x1024x128xbf16>

      %r_ln = stablehlo.reshape %ln_out : (tensor<1x128x1024xbf16>) -> tensor<128x1024x1xbf16>
      %t_ln = stablehlo.transpose %r_ln, dims = [2, 1, 0] : (tensor<128x1024x1xbf16>) -> tensor<1x1024x128xbf16>
      %silu_in = stablehlo.convert %t_ln : (tensor<1x1024x128xbf16>) -> tensor<1x1024x128xf32>
      %neg2 = stablehlo.negate %silu_in : tensor<1x1024x128xf32>
      %exp2 = stablehlo.exponential %neg2 : tensor<1x1024x128xf32>
      %add2 = stablehlo.add %exp2, %one : tensor<1x1024x128xf32>
      %div2 = stablehlo.divide %silu_in, %add2 : tensor<1x1024x128xf32>
      %silu_out = stablehlo.convert %div2 : (tensor<1x1024x128xf32>) -> tensor<1x1024x128xbf16>
      %c2 = stablehlo.convolution(%silu_out, %w2) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1024x128xbf16>, tensor<1024x1024x1xbf16>) -> tensor<1x1024x128xbf16>
      %b2_bcast = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<1024xbf16>) -> tensor<1x1024x128xbf16>
      %c2_b = stablehlo.add %c2, %b2_bcast : tensor<1x1024x128xbf16>
      %r_out = stablehlo.reshape %c2_b : (tensor<1x1024x128xbf16>) -> tensor<1024x128x1xbf16>
      %t_out = stablehlo.transpose %r_out, dims = [2, 1, 0] : (tensor<1024x128x1xbf16>) -> tensor<1x128x1024xbf16>
      return %glu_out, %t_out : tensor<1x1024x128xbf16>, tensor<1x128x1024xbf16>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::ConvolutionOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 1);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     FoldsFiniteRelativePositionalSdpaMaskAndHoistsSharedQueryScale) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(
        %q_in: tensor<128x1024xbf16>,
        %k_in: tensor<1x16x128x64xbf16>,
        %r_in: tensor<128x64x128xbf16>,
        %probs: tensor<1x16x128x128xf32>)
        -> (tensor<1x16x128x128xbf16>, tensor<1x16x128x128xf32>) {
      %iot = stablehlo.iota dim = 0 : tensor<128xi64>
      %zero_i64 = stablehlo.constant dense<0> : tensor<128xi64>
      %cmp_iot = stablehlo.compare GE, %iot, %zero_i64 : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
      %b_cmp = stablehlo.broadcast_in_dim %cmp_iot, dims = [2] : (tensor<128xi1>) -> tensor<1x16x128x128xi1>

      %q_r1 = stablehlo.reshape %q_in : (tensor<128x1024xbf16>) -> tensor<128x16x64xbf16>
      %rel_dot = stablehlo.dot_general %q_r1, %r_in, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x16x64xbf16>, tensor<128x64x128xbf16>) -> tensor<128x16x128xbf16>
      %rel_r = stablehlo.reshape %rel_dot : (tensor<128x16x128xbf16>) -> tensor<1x128x16x128xbf16>
      %rel_t = stablehlo.transpose %rel_r, dims = [0, 2, 1, 3] : (tensor<1x128x16x128xbf16>) -> tensor<1x16x128x128xbf16>
      %scale = stablehlo.constant dense<0.125> : tensor<1x16x128x128xbf16>
      %rel_scaled = stablehlo.multiply %rel_t, %scale : tensor<1x16x128x128xbf16>
      %neg_max = stablehlo.constant dense<-3.38953139e+38> : tensor<1x16x128x128xbf16>
      %mask = stablehlo.select %b_cmp, %rel_scaled, %neg_max : tensor<1x16x128x128xi1>, tensor<1x16x128x128xbf16>

      %neg_inf = stablehlo.constant dense<0xFF80> : tensor<1x16x128x128xbf16>
      %is_neg_inf = stablehlo.compare EQ, %mask, %neg_inf : (tensor<1x16x128x128xbf16>, tensor<1x16x128x128xbf16>) -> tensor<1x16x128x128xi1>
      %init_true = stablehlo.constant dense<true> : tensor<i1>
      %row_all = stablehlo.reduce(%is_neg_inf init: %init_true) across dimensions = [3] : (tensor<1x16x128x128xi1>, tensor<i1>) -> tensor<1x16x128xi1>
        reducer(%a: tensor<i1>, %b: tensor<i1>) {
          %and = stablehlo.and %a, %b : tensor<i1>
          stablehlo.return %and : tensor<i1>
        }
      %row_bcast = stablehlo.broadcast_in_dim %row_all, dims = [0, 1, 2] : (tensor<1x16x128xi1>) -> tensor<1x16x128x128xi1>
      %zeros = stablehlo.constant dense<0.0> : tensor<1x16x128x128xf32>
      %masked_probs = stablehlo.select %row_bcast, %zeros, %probs : tensor<1x16x128x128xi1>, tensor<1x16x128x128xf32>

      %q_r2 = stablehlo.reshape %q_in : (tensor<128x1024xbf16>) -> tensor<128x16x64x1xbf16>
      %q_t2 = stablehlo.transpose %q_r2, dims = [3, 1, 0, 2] : (tensor<128x16x64x1xbf16>) -> tensor<1x16x128x64xbf16>
      %scale_q = stablehlo.constant dense<0.125> : tensor<1x16x128x64xbf16>
      %q_scaled = stablehlo.multiply %q_t2, %scale_q : tensor<1x16x128x64xbf16>
      %qk = stablehlo.dot_general %q_scaled, %k_in, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x16x128x64xbf16>, tensor<1x16x128x64xbf16>) -> tensor<1x16x128x128xbf16>
      %clamped = stablehlo.maximum %mask, %neg_max : tensor<1x16x128x128xbf16>
      %scores = stablehlo.add %qk, %clamped : tensor<1x16x128x128xbf16>
      return %scores, %masked_probs : tensor<1x16x128x128xbf16>, tensor<1x16x128x128xf32>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::CompareOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SelectOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::MaxOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::MulOp>(*module), 1);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     PreservesMaskChecksWhenNegInfinityIsPossible) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(
        %pred: tensor<1x8x16x16xi1>,
        %probs: tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32> {
      %zero = stablehlo.constant dense<0.0> : tensor<1x8x16x16xbf16>
      %neg_inf = stablehlo.constant dense<0xFF80> : tensor<1x8x16x16xbf16>
      %mask = stablehlo.select %pred, %zero, %neg_inf : tensor<1x8x16x16xi1>, tensor<1x8x16x16xbf16>
      %is_neg_inf = stablehlo.compare EQ, %mask, %neg_inf : (tensor<1x8x16x16xbf16>, tensor<1x8x16x16xbf16>) -> tensor<1x8x16x16xi1>
      %init_true = stablehlo.constant dense<true> : tensor<i1>
      %row_all = stablehlo.reduce(%is_neg_inf init: %init_true) across dimensions = [3] : (tensor<1x8x16x16xi1>, tensor<i1>) -> tensor<1x8x16xi1>
        reducer(%a: tensor<i1>, %b: tensor<i1>) {
          %and = stablehlo.and %a, %b : tensor<i1>
          stablehlo.return %and : tensor<i1>
        }
      %row_bcast = stablehlo.broadcast_in_dim %row_all, dims = [0, 1, 2] : (tensor<1x8x16xi1>) -> tensor<1x8x16x16xi1>
      %zeros = stablehlo.constant dense<0.0> : tensor<1x8x16x16xf32>
      %out = stablehlo.select %row_bcast, %zeros, %probs : tensor<1x8x16x16xi1>, tensor<1x8x16x16xf32>
      return %out : tensor<1x8x16x16xf32>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*module), 1);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     PreservesNonUnitBatchOrSpatialWindow1DConvolutions) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(
        %x_batched: tensor<4x64x16xbf16>,
        %w_pointwise: tensor<128x64x1xbf16>,
        %x_unit: tensor<1x64x16xbf16>,
        %w_spatial: tensor<128x64x3xbf16>)
        -> (tensor<4x128x16xbf16>, tensor<1x128x16xbf16>) {
      %c_batched = stablehlo.convolution(%x_batched, %w_pointwise) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x64x16xbf16>, tensor<128x64x1xbf16>) -> tensor<4x128x16xbf16>
      %c_spatial = stablehlo.convolution(%x_unit, %w_spatial) dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], window = {stride = [1], pad = [[1, 1]], lhs_dilate = [1], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x16xbf16>, tensor<128x64x3xbf16>) -> tensor<1x128x16xbf16>
      return %c_batched, %c_spatial : tensor<4x128x16xbf16>, tensor<1x128x16xbf16>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::ConvolutionOp>(*module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module), 0);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     SimplifiesTransposeSliceTransposeWithNonInvolutoryPermutation) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(%x: tensor<2x3x4xf32>) -> tensor<2x2x2xf32> {
      %t0 = stablehlo.transpose %x, dims = [1, 2, 0] : (tensor<2x3x4xf32>) -> tensor<3x4x2xf32>
      %s0 = stablehlo.slice %t0 [0:2, 1:3, 0:2] : (tensor<3x4x2xf32>) -> tensor<2x2x2xf32>
      %t1 = stablehlo.transpose %s0, dims = [2, 0, 1] : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
      return %t1 : tensor<2x2x2xf32>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 1);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     HandlesBlockArgumentsGracefullyInFiniteBiasCheck) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(
        %arg_bias: tensor<1x16x128x128xbf16>,
        %pred: tensor<1x16x128x128xi1>) -> tensor<1x16x128x128xi1> {
      %neg_max = stablehlo.constant dense<-3.38953139e+38> : tensor<1x16x128x128xbf16>
      %mask = stablehlo.select %pred, %arg_bias, %neg_max : tensor<1x16x128x128xi1>, tensor<1x16x128x128xbf16>
      %neg_inf = stablehlo.constant dense<0xFF80> : tensor<1x16x128x128xbf16>
      %cmp = stablehlo.compare EQ, %mask, %neg_inf : (tensor<1x16x128x128xbf16>, tensor<1x16x128x128xbf16>) -> tensor<1x16x128x128xi1>
      return %cmp : tensor<1x16x128x128xi1>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  // %arg_bias cannot be proven finite at compile time, so the compare survives
  // without crashing on the block argument.
  EXPECT_EQ(CountOps<mlir::stablehlo::CompareOp>(*module), 1);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     HandlesZeroResultOperationsWithoutCrashing) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func private @side_effect() -> ()
    func.func @main(%x: tensor<1x1024x16xbf16>) -> tensor<1x1024x16xbf16> {
      func.call @side_effect() : () -> ()
      return %x : tensor<1x1024x16xbf16>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());
  EXPECT_EQ(CountOps<mlir::func::CallOp>(*module), 1);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     PreservesAndReductionAcrossZeroExtentDimension) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(%x: tensor<1x16x0xi1>) -> tensor<1x16xi1> {
      %false_cst = stablehlo.constant dense<false> : tensor<1x16x0xi1>
      %init_true = stablehlo.constant dense<true> : tensor<i1>
      %red = stablehlo.reduce(%false_cst init: %init_true) across dimensions = [2] : (tensor<1x16x0xi1>, tensor<i1>) -> tensor<1x16xi1>
        reducer(%a: tensor<i1>, %b: tensor<i1>) {
          %and = stablehlo.and %a, %b : tensor<i1>
          stablehlo.return %and : tensor<i1>
        }
      return %red : tensor<1x16xi1>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizePointwiseConvAndFiniteSdpaBiasesPass());
  EXPECT_TRUE(mlir::succeeded(pm.run(*module)));

  // Empty dimension reduction returns initValue (true); must not fold to false.
  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*module), 1);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     PreservesLogisticOpWhenCommutingAcrossTranspose) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(%x_btc: tensor<1x16x32xf32>) -> tensor<1x16x32xf32> {
      %x = stablehlo.transpose %x_btc, dims = [0, 2, 1] : (tensor<1x16x32xf32>) -> tensor<1x32x16xf32>
      %log = stablehlo.logistic %x : tensor<1x32x16xf32>
      %silu = stablehlo.multiply %x, %log : tensor<1x32x16xf32>
      %tr = stablehlo.transpose %silu, dims = [0, 2, 1] : (tensor<1x32x16xf32>) -> tensor<1x16x32xf32>
      return %tr : tensor<1x16x32xf32>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::LogisticOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::MulOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ExpOp>(*module), 0);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     DoesNotFoldMaskWithNegativeInfinitySentinelsUnderAdd) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  // When SelectOp has -FLT_MAX on false branch and non-uniform predicate,
  // AddOp of two such selects can overflow to -inf; compare against -inf
  // must NOT be folded to false.
  constexpr const char* kIr = R"mlir(
    func.func @main(%pred1: tensor<1x8x16xi1>, %pred2: tensor<1x8x16xi1>) -> tensor<1x8x16xi1> {
      %c0 = stablehlo.constant dense<0.000000e+00> : tensor<1x8x16xf32>
      %c_lowest = stablehlo.constant dense<-3.40282347e+38> : tensor<1x8x16xf32>
      %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<1x8x16xf32>
      %s1 = stablehlo.select %pred1, %c0, %c_lowest : tensor<1x8x16xi1>, tensor<1x8x16xf32>
      %s2 = stablehlo.select %pred2, %c0, %c_lowest : tensor<1x8x16xi1>, tensor<1x8x16xf32>
      %add = stablehlo.add %s1, %s2 : tensor<1x8x16xf32>
      %cmp = stablehlo.compare EQ, %add, %neg_inf : (tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xi1>
      return %cmp : tensor<1x8x16xi1>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  // Compare against -inf must not be folded away.
  EXPECT_EQ(CountOps<mlir::stablehlo::CompareOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConstantOp>(*module), 3);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     DoesNotFoldUnsignedCompareOnAffineIotaWithNegativeConstant) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  // In unsigned comparison, -1 is 0xFFFFFFFF (UINT_MAX), so iota < -1 is true
  // for small iotas, but signed evaluation would say false. Must not fold.
  constexpr const char* kIr = R"mlir(
    func.func @main() -> tensor<1x16xi1> {
      %iota = stablehlo.iota dim = 1 : tensor<1x16xi32>
      %c_neg1 = stablehlo.constant dense<-1> : tensor<1x16xi32>
      %cmp = stablehlo.compare LT, %iota, %c_neg1, UNSIGNED : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi1>
      return %cmp : tensor<1x16xi1>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  // Unsigned comparison must not be folded by affine iota evaluator.
  EXPECT_EQ(CountOps<mlir::stablehlo::CompareOp>(*module), 1);
}

TEST(CanonicalizePointwiseConvAndFiniteSdpaBiasesTest,
     PreservesLogisticOpWhenCommutingAcrossTransposeInGlu) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  constexpr const char* kIr = R"mlir(
    func.func @main(%x_btc: tensor<1x16x32xf32>) -> tensor<1x16x16xf32> {
      %x = stablehlo.transpose %x_btc, dims = [0, 2, 1] : (tensor<1x16x32xf32>) -> tensor<1x32x16xf32>
      %s0 = stablehlo.slice %x [0:1, 0:16, 0:16] : (tensor<1x32x16xf32>) -> tensor<1x16x16xf32>
      %s1 = stablehlo.slice %x [0:1, 16:32, 0:16] : (tensor<1x32x16xf32>) -> tensor<1x16x16xf32>
      %log = stablehlo.logistic %s1 : tensor<1x16x16xf32>
      %mul = stablehlo.multiply %s0, %log : tensor<1x16x16xf32>
      %tr = stablehlo.transpose %mul, dims = [0, 2, 1] : (tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
      return %tr : tensor<1x16x16xf32>
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  // GLU commutes to [B, T, 2C] -> slice -> logistic -> mul. Transposes cancel.
  // Verify LogisticOp is preserved and dead ops are erased (no 6x cloned dead
  // IR).
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::LogisticOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::MulOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ExpOp>(*module), 0);
}

}  // namespace
}  // namespace torch_tpu
