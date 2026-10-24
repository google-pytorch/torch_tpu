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

#include "csrc/mlir/passes/hoist_monotonic_reduce_and_barrier_multi_dot_grad.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/ChloOps.h"
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
    vals[i] = base + static_cast<float>((i * 11 + 3) % 29) * 0.125f - 1.5f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    float f = static_cast<float>((i * 13 + 5) % 31) * 0.25f - 3.5f;
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

TEST(HoistMonotonicReduceAndBarrierMultiDotGradTest,
     HoistsWidenedSoftcapChainFromSharedLogSoftmaxReduceMax) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%logits: tensor<4x16xbf16>) -> (tensor<4xf32>, tensor<4x16xf32>) {
  %f32 = stablehlo.convert %logits : (tensor<4x16xbf16>) -> tensor<4x16xf32>
  %s = stablehlo.constant dense<0.25> : tensor<4x16xf32>
  %scaled = stablehlo.multiply %f32, %s : tensor<4x16xf32>
  %c = stablehlo.constant dense<20.0> : tensor<4x16xf32>
  %div = stablehlo.divide %scaled, %c : tensor<4x16xf32>
  %t = stablehlo.tanh %div : tensor<4x16xf32>
  %capped = stablehlo.multiply %t, %c : tensor<4x16xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %max = stablehlo.reduce(%capped init: %neg_inf) across dimensions = [1]
      : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
  %bcast_max = stablehlo.broadcast_in_dim %max, dims = [0]
      : (tensor<4xf32>) -> tensor<4x16xf32>
  %shifted = stablehlo.subtract %capped, %bcast_max : tensor<4x16xf32>
  return %max, %shifted : tensor<4xf32>, tensor<4x16xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr logits = MakeBF16Tensor(context, {4, 16});
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {logits}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // The ReduceOp now operates directly on tensor<4x16xbf16>.
  int bf16_reduce_count = 0;
  opt_mod->walk([&](mlir::stablehlo::ReduceOp rop) {
    auto in_ty =
        llvm::cast<mlir::RankedTensorType>(rop.getInputs()[0].getType());
    if (in_ty.getElementType().isBF16()) {
      ++bf16_reduce_count;
    }
  });
  EXPECT_EQ(bf16_reduce_count, 1);
  // Consecutive (x * 0.25) / 20.0 is folded into a single multiply, so DivideOp
  // is completely eliminated.
  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*opt_mod), 0);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {logits}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (int out_idx = 0; out_idx < 2; ++out_idx) {
    auto exp_it = (*expected)[out_idx].value_begin<float>();
    auto act_it = (*actual)[out_idx].value_begin<float>();
    for (; exp_it != (*expected)[out_idx].value_end<float>();
         ++exp_it, ++act_it) {
      EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
    }
  }
}

TEST(HoistMonotonicReduceAndBarrierMultiDotGradTest,
     HoistsSingleUseAffineAndTranscendentalChainFromReduceMin) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<3x8xf32>, %bias: tensor<3xf32>) -> tensor<3xf32> {
  %c_mul = stablehlo.constant dense<2.0> : tensor<3x8xf32>
  %scaled = stablehlo.multiply %x, %c_mul : tensor<3x8xf32>
  %bcast_bias = stablehlo.broadcast_in_dim %bias, dims = [0]
      : (tensor<3xf32>) -> tensor<3x8xf32>
  %shifted = stablehlo.add %scaled, %bcast_bias : tensor<3x8xf32>
  %sig = stablehlo.logistic %shifted : tensor<3x8xf32>
  %pos_inf = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %min = stablehlo.reduce(%sig init: %pos_inf) across dimensions = [1]
      : (tensor<3x8xf32>, tensor<f32>) -> tensor<3xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.minimum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
  return %min : tensor<3xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {3, 8}, 0.5f);
  mlir::DenseElementsAttr bias = MakeF32Tensor(context, {3}, -0.25f);
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x, bias}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // The LogisticOp now runs on 1D tensor<3xf32> instead of 2D tensor<3x8xf32>.
  int rank1_logistic = 0;
  opt_mod->walk([&](mlir::stablehlo::LogisticOp lop) {
    auto ty = llvm::cast<mlir::RankedTensorType>(lop.getType());
    if (ty.getRank() == 1) {
      ++rank1_logistic;
    }
  });
  EXPECT_EQ(rank1_logistic, 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*opt_mod), 0);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, bias}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
  }
}

TEST(HoistMonotonicReduceAndBarrierMultiDotGradTest,
     InsertsOptimizationBarrierOnNarrowedMultiDotSoftcapSoftmaxGrad) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %logits_bf16: tensor<8x32xbf16>,
    %dy_f32: tensor<8x32xf32>,
    %w: tensor<32x16xbf16>,
    %h: tensor<8x16xbf16>) -> (tensor<8x16xbf16>, tensor<32x16xbf16>) {
  %logits_f32 = stablehlo.convert %logits_bf16
      : (tensor<8x32xbf16>) -> tensor<8x32xf32>
  %t = stablehlo.tanh %logits_f32 : tensor<8x32xf32>
  %e = stablehlo.exponential %t : tensor<8x32xf32>
  %diff = stablehlo.subtract %dy_f32, %e : tensor<8x32xf32>
  %t2 = stablehlo.multiply %t, %t : tensor<8x32xf32>
  %one = stablehlo.constant dense<1.0> : tensor<8x32xf32>
  %dt = stablehlo.subtract %one, %t2 : tensor<8x32xf32>
  %grad_f32 = stablehlo.multiply %diff, %dt : tensor<8x32xf32>
  %grad_bf16 = stablehlo.convert %grad_f32
      : (tensor<8x32xf32>) -> tensor<8x32xbf16>
  %dx = "stablehlo.dot_general"(%grad_bf16, %w) <{
      dot_dimension_numbers = #stablehlo.dot<
          lhs_contracting_dimensions = [1],
          rhs_contracting_dimensions = [0]>
  }> : (tensor<8x32xbf16>, tensor<32x16xbf16>) -> tensor<8x16xbf16>
  %grad_t = stablehlo.transpose %grad_bf16, dims = [1, 0]
      : (tensor<8x32xbf16>) -> tensor<32x8xbf16>
  %dw = "stablehlo.dot_general"(%grad_t, %h) <{
      dot_dimension_numbers = #stablehlo.dot<
          lhs_contracting_dimensions = [1],
          rhs_contracting_dimensions = [0]>
  }> : (tensor<32x8xbf16>, tensor<8x16xbf16>) -> tensor<32x16xbf16>
  return %dx, %dw : tensor<8x16xbf16>, tensor<32x16xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::OptimizationBarrierOp>(*module), 1);
}

TEST(HoistMonotonicReduceAndBarrierMultiDotGradTest,
     DoesNotHoistNegativeMultiplierOrReductionDependentBroadcast) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<4x8xf32>, %col_bias: tensor<8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %neg_c = stablehlo.constant dense<-2.0> : tensor<4x8xf32>
  %neg_scaled = stablehlo.multiply %x, %neg_c : tensor<4x8xf32>
  %bcast_col = stablehlo.broadcast_in_dim %col_bias, dims = [1]
      : (tensor<8xf32>) -> tensor<4x8xf32>
  %col_shifted = stablehlo.add %x, %bcast_col : tensor<4x8xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %max1 = stablehlo.reduce(%neg_scaled init: %neg_inf) across dimensions = [1]
      : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
  %max2 = stablehlo.reduce(%col_shifted init: %neg_inf) across dimensions = [1]
      : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
  return %max1, %max2 : tensor<4xf32>, tensor<4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*module), 1);
}

TEST(HoistMonotonicReduceAndBarrierMultiDotGradTest,
     HoistsSubtractWithInvariantRhsAndExpFromReduceMax) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<4x8xf32>, %bias: tensor<4xf32>) -> tensor<4xf32> {
  %bcast_bias = stablehlo.broadcast_in_dim %bias, dims = [0]
      : (tensor<4xf32>) -> tensor<4x8xf32>
  %sub = stablehlo.subtract %x, %bcast_bias : tensor<4x8xf32>
  %exp = stablehlo.exponential %sub : tensor<4x8xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %max = stablehlo.reduce(%exp init: %neg_inf) across dimensions = [1]
      : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
  return %max : tensor<4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {4, 8}, 0.5f);
  mlir::DenseElementsAttr bias = MakeF32Tensor(context, {4}, 0.25f);
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x, bias}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // In opt_mod, exponential now operates on 1D tensor<4xf32> instead of 2D.
  int rank1_exp = 0;
  opt_mod->walk([&](mlir::stablehlo::ExpOp exp_op) {
    auto ty = llvm::cast<mlir::RankedTensorType>(exp_op.getType());
    if (ty.getRank() == 1) {
      ++rank1_exp;
    }
  });
  EXPECT_EQ(rank1_exp, 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*opt_mod), 0);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, bias}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
  }
}

TEST(HoistMonotonicReduceAndBarrierMultiDotGradTest,
     HoistsChloErfFromReduceMax) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<4x8xf32>) -> tensor<4xf32> {
  %erf = chlo.erf %x : tensor<4x8xf32> -> tensor<4x8xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %max = stablehlo.reduce(%erf init: %neg_inf) across dimensions = [1]
      : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
  return %max : tensor<4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // In opt_mod, chlo.erf operates on 1D tensor<4xf32>, and ReduceOp operates
  // directly on the 2D argument tensor<4x8xf32>.
  int rank1_erf = 0;
  opt_mod->walk([&](mlir::chlo::ErfOp erf_op) {
    auto ty = llvm::cast<mlir::RankedTensorType>(erf_op.getType());
    if (ty.getRank() == 1) {
      ++rank1_erf;
    }
  });
  EXPECT_EQ(rank1_erf, 1);

  int direct_reduce = 0;
  opt_mod->walk([&](mlir::stablehlo::ReduceOp rop) {
    auto in_ty =
        llvm::cast<mlir::RankedTensorType>(rop.getInputs()[0].getType());
    if (in_ty.getRank() == 2 &&
        llvm::isa<mlir::BlockArgument>(rop.getInputs()[0])) {
      ++direct_reduce;
    }
  });
  EXPECT_EQ(direct_reduce, 1);
}

TEST(HoistMonotonicReduceAndBarrierMultiDotGradTest,
     HoistsAddWithInvariantLhsFromReduceMax) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<3x8xf32>, %bias: tensor<3xf32>) -> tensor<3xf32> {
  %bcast_bias = stablehlo.broadcast_in_dim %bias, dims = [0]
      : (tensor<3xf32>) -> tensor<3x8xf32>
  %shifted = stablehlo.add %bcast_bias, %x : tensor<3x8xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %max = stablehlo.reduce(%shifted init: %neg_inf) across dimensions = [1]
      : (tensor<3x8xf32>, tensor<f32>) -> tensor<3xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
  return %max : tensor<3xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {3, 8}, 0.5f);
  mlir::DenseElementsAttr bias = MakeF32Tensor(context, {3}, -0.25f);
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x, bias}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // In opt_mod, AddOp operates on 1D tensor<3xf32>.
  int rank1_add = 0;
  opt_mod->walk([&](mlir::stablehlo::AddOp add_op) {
    auto ty = llvm::cast<mlir::RankedTensorType>(add_op.getType());
    if (ty.getRank() == 1) {
      ++rank1_add;
    }
  });
  EXPECT_EQ(rank1_add, 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*opt_mod), 0);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, bias}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
  }
}

TEST(HoistMonotonicReduceAndBarrierMultiDotGradTest,
     DoesNotHoistLogOrSqrtDueToPotentialNanPoisoning) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<4x8xf32>, %y: tensor<4x8xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  %log = stablehlo.log %x : tensor<4x8xf32>
  %sqrt = stablehlo.sqrt %y : tensor<4x8xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %max1 = stablehlo.reduce(%log init: %neg_inf) across dimensions = [1]
      : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
  %max2 = stablehlo.reduce(%sqrt init: %neg_inf) across dimensions = [1]
      : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
  return %max1, %max2 : tensor<4xf32>, tensor<4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());

  // Both ReduceOps must still take 2D tensor<4x8xf32> inputs from log and sqrt.
  module->walk([&](mlir::stablehlo::ReduceOp rop) {
    auto in_ty =
        llvm::cast<mlir::RankedTensorType>(rop.getInputs()[0].getType());
    EXPECT_EQ(in_ty.getRank(), 2);
  });
  EXPECT_EQ(CountOps<mlir::stablehlo::LogOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SqrtOp>(*module), 1);
}

}  // namespace
}  // namespace torch_tpu
