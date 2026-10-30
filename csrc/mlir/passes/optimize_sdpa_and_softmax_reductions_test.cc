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

#include "csrc/mlir/passes/optimize_sdpa_and_softmax_reductions.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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
                                      float scale = 0.05f) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = static_cast<float>((i * 19 + 7) % 97 - 48) * scale;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

TEST(OptimizeSdpaAndSoftmaxReductionsTest,
     RewritesUnalignedMultiTileSdpaDivAndBackwardRowsumNumericallyExact) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %logits: tensor<1x8x142x142xf32>,
    %ds: tensor<1x8x142x142xf32>) -> (tensor<1x8x142x142xf32>, tensor<1x8x142x142xf32>) {
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %max_red = stablehlo.reduce(%logits init: %neg_inf)
      applies stablehlo.maximum across dimensions = [3]
      : (tensor<1x8x142x142xf32>, tensor<f32>) -> tensor<1x8x142xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max_red, dims = [0, 1, 2]
      : (tensor<1x8x142xf32>) -> tensor<1x8x142x142xf32>
  %shifted = stablehlo.subtract %logits, %max_bcast : tensor<1x8x142x142xf32>
  %exp = stablehlo.exponential %shifted : tensor<1x8x142x142xf32>
  %sum_red = stablehlo.reduce(%exp init: %zero)
      applies stablehlo.add across dimensions = [3]
      : (tensor<1x8x142x142xf32>, tensor<f32>) -> tensor<1x8x142xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum_red, dims = [0, 1, 2]
      : (tensor<1x8x142xf32>) -> tensor<1x8x142x142xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<1x8x142x142xf32>
  %s_mul_ds = stablehlo.multiply %probs, %ds : tensor<1x8x142x142xf32>
  %rowsum = stablehlo.reduce(%s_mul_ds init: %zero)
      applies stablehlo.add across dimensions = [3]
      : (tensor<1x8x142x142xf32>, tensor<f32>) -> tensor<1x8x142xf32>
  %rowsum_bcast = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2]
      : (tensor<1x8x142xf32>) -> tensor<1x8x142x142xf32>
  %shifted_ds = stablehlo.subtract %ds, %rowsum_bcast : tensor<1x8x142x142xf32>
  %dp = stablehlo.multiply %probs, %shifted_ds : tensor<1x8x142x142xf32>
  return %probs, %dp : tensor<1x8x142x142xf32>, tensor<1x8x142x142xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr logits =
      MakeF32Tensor(context, {1, 8, 142, 142}, 0.03f);
  mlir::DenseElementsAttr ds = MakeF32Tensor(context, {1, 8, 142, 142}, 0.02f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {logits, ds}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createOptimizeSdpaAndSoftmaxReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));
  EXPECT_EQ(CountOps<mlir::stablehlo::SubtractOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::NegOp>(*opt_mod), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {logits, ds}, cfg);
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

TEST(OptimizeSdpaAndSoftmaxReductionsTest,
     NormalizesForwardGqaAndSinksRowMaskSelectForOnlineSoftmax) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %logits: tensor<1x32x2048x2048xf32>,
    %v: tensor<1x8x2048x64xf32>,
    %row_mask: tensor<1x32x2048xi1>) -> tensor<1x32x2048x64xf32> {
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero_s = stablehlo.constant dense<0.0> : tensor<f32>
  %zero_4d = stablehlo.constant dense<0.0> : tensor<1x32x2048x2048xf32>
  %max_red = stablehlo.reduce(%logits init: %neg_inf)
      applies stablehlo.maximum across dimensions = [3]
      : (tensor<1x32x2048x2048xf32>, tensor<f32>) -> tensor<1x32x2048xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max_red, dims = [0, 1, 2]
      : (tensor<1x32x2048xf32>) -> tensor<1x32x2048x2048xf32>
  %shifted = stablehlo.subtract %logits, %max_bcast : tensor<1x32x2048x2048xf32>
  %exp = stablehlo.exponential %shifted : tensor<1x32x2048x2048xf32>
  %sum_red = stablehlo.reduce(%exp init: %zero_s)
      applies stablehlo.add across dimensions = [3]
      : (tensor<1x32x2048x2048xf32>, tensor<f32>) -> tensor<1x32x2048xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum_red, dims = [0, 1, 2]
      : (tensor<1x32x2048xf32>) -> tensor<1x32x2048x2048xf32>
  %probs_unmasked = stablehlo.divide %exp, %sum_bcast : tensor<1x32x2048x2048xf32>
  %mask_bcast = stablehlo.broadcast_in_dim %row_mask, dims = [0, 1, 2]
      : (tensor<1x32x2048xi1>) -> tensor<1x32x2048x2048xi1>
  %probs = stablehlo.select %mask_bcast, %zero_4d, %probs_unmasked
      : tensor<1x32x2048x2048xi1>, tensor<1x32x2048x2048xf32>
  %probs_5d = stablehlo.reshape %probs
      : (tensor<1x32x2048x2048xf32>) -> tensor<1x8x4x2048x2048xf32>
  %out_5d = stablehlo.dot_general %probs_5d, %v,
      batching_dims = [0, 1] x [0, 1],
      contracting_dims = [4] x [2]
      : (tensor<1x8x4x2048x2048xf32>, tensor<1x8x2048x64xf32>)
      -> tensor<1x8x4x2048x64xf32>
  %out_4d = stablehlo.reshape %out_5d
      : (tensor<1x8x4x2048x64xf32>) -> tensor<1x32x2048x64xf32>
  return %out_4d : tensor<1x32x2048x64xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createOptimizeSdpaAndSoftmaxReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SelectOp>(*opt_mod), 1);

  opt_mod->walk([&](mlir::stablehlo::DotGeneralOp dg) {
    auto lhs_t = mlir::cast<mlir::RankedTensorType>(dg.getLhs().getType());
    auto rhs_t = mlir::cast<mlir::RankedTensorType>(dg.getRhs().getType());
    EXPECT_EQ(lhs_t.getShape(), llvm::ArrayRef<int64_t>({1, 32, 2048, 2048}));
    EXPECT_EQ(rhs_t.getShape(), llvm::ArrayRef<int64_t>({1, 32, 2048, 64}));
    auto dnums = dg.getDotDimensionNumbers();
    EXPECT_EQ(llvm::ArrayRef<int64_t>(dnums.getLhsBatchingDimensions()),
              llvm::ArrayRef<int64_t>({0, 1}));
    EXPECT_EQ(llvm::ArrayRef<int64_t>(dnums.getRhsBatchingDimensions()),
              llvm::ArrayRef<int64_t>({0, 1}));
    EXPECT_EQ(llvm::ArrayRef<int64_t>(dnums.getLhsContractingDimensions()),
              llvm::ArrayRef<int64_t>({3}));
    EXPECT_EQ(llvm::ArrayRef<int64_t>(dnums.getRhsContractingDimensions()),
              llvm::ArrayRef<int64_t>({2}));

    auto resh = dg.getRhs().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    ASSERT_TRUE(resh != nullptr);
    auto bcast =
        resh.getOperand().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    ASSERT_TRUE(bcast != nullptr);
    EXPECT_EQ(bcast.getBroadcastDimensions(),
              llvm::ArrayRef<int64_t>({0, 1, 3, 4}));
  });

  opt_mod->walk([&](mlir::stablehlo::SelectOp sel) {
    auto sel_t = mlir::cast<mlir::RankedTensorType>(sel.getType());
    EXPECT_EQ(sel_t.getShape(), llvm::ArrayRef<int64_t>({1, 32, 2048, 64}));
    auto const_zero =
        sel.getOnTrue().getDefiningOp<mlir::stablehlo::ConstantOp>();
    ASSERT_TRUE(const_zero != nullptr);
    auto dense_attr =
        mlir::dyn_cast<mlir::DenseElementsAttr>(const_zero.getValue());
    ASSERT_TRUE(dense_attr && dense_attr.isSplat());
    EXPECT_TRUE(dense_attr.getSplatValue<llvm::APFloat>().isZero());

    auto pred_bcast =
        sel.getPred().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    ASSERT_TRUE(pred_bcast != nullptr);
    EXPECT_EQ(pred_bcast.getBroadcastDimensions(),
              llvm::ArrayRef<int64_t>({0, 1, 2}));

    auto dot_op =
        sel.getOnFalse().getDefiningOp<mlir::stablehlo::DotGeneralOp>();
    EXPECT_TRUE(dot_op != nullptr);
  });
}

TEST(OptimizeSdpaAndSoftmaxReductionsTest,
     RewritesLargePreNormalizedReciprocalSoftmaxSubtractions) {
  mlir::MLIRContext context;
  InitContext(context);

  // 1. Numerical verification on unaligned pre-normalized softmax shape.
  const char* kUnalignedIr = R"mlir(
func.func @main(
    %logits: tensor<1x8x142x142xf32>,
    %ds: tensor<1x8x142x142xf32>) -> (tensor<1x8x142x142xf32>, tensor<1x8x142x142xf32>) {
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %one_3d = stablehlo.constant dense<1.0> : tensor<1x8x142xf32>
  %max_red = stablehlo.reduce(%logits init: %neg_inf)
      applies stablehlo.maximum across dimensions = [3]
      : (tensor<1x8x142x142xf32>, tensor<f32>) -> tensor<1x8x142xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max_red, dims = [0, 1, 2]
      : (tensor<1x8x142xf32>) -> tensor<1x8x142x142xf32>
  %shifted = stablehlo.subtract %logits, %max_bcast : tensor<1x8x142x142xf32>
  %exp = stablehlo.exponential %shifted : tensor<1x8x142x142xf32>
  %sum_red = stablehlo.reduce(%exp init: %zero)
      applies stablehlo.add across dimensions = [3]
      : (tensor<1x8x142x142xf32>, tensor<f32>) -> tensor<1x8x142xf32>
  %inv_sum = stablehlo.divide %one_3d, %sum_red : tensor<1x8x142xf32>
  %inv_bcast = stablehlo.broadcast_in_dim %inv_sum, dims = [0, 1, 2]
      : (tensor<1x8x142xf32>) -> tensor<1x8x142x142xf32>
  %probs = stablehlo.multiply %exp, %inv_bcast : tensor<1x8x142x142xf32>
  %s_mul_ds = stablehlo.multiply %probs, %ds : tensor<1x8x142x142xf32>
  %rowsum = stablehlo.reduce(%s_mul_ds init: %zero)
      applies stablehlo.add across dimensions = [3]
      : (tensor<1x8x142x142xf32>, tensor<f32>) -> tensor<1x8x142xf32>
  %rowsum_bcast = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2]
      : (tensor<1x8x142xf32>) -> tensor<1x8x142x142xf32>
  %shifted_ds = stablehlo.subtract %ds, %rowsum_bcast : tensor<1x8x142x142xf32>
  %dp = stablehlo.multiply %probs, %shifted_ds : tensor<1x8x142x142xf32>
  return %probs, %dp : tensor<1x8x142x142xf32>, tensor<1x8x142x142xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_unaligned_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kUnalignedIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_unaligned_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kUnalignedIr, &context);
  ASSERT_TRUE(orig_unaligned_mod && opt_unaligned_mod);

  mlir::DenseElementsAttr logits =
      MakeF32Tensor(context, {1, 8, 142, 142}, 0.03f);
  mlir::DenseElementsAttr ds = MakeF32Tensor(context, {1, 8, 142, 142}, 0.02f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_unaligned_mod, {logits, ds}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm_unaligned(&context);
  pm_unaligned.addNestedPass<mlir::func::FuncOp>(
      createOptimizeSdpaAndSoftmaxReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm_unaligned.run(*opt_unaligned_mod)));
  EXPECT_EQ(CountOps<mlir::stablehlo::SubtractOp>(*opt_unaligned_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::NegOp>(*opt_unaligned_mod), 2);

  auto actual =
      mlir::stablehlo::evalModule(*opt_unaligned_mod, {logits, ds}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (int out_idx = 0; out_idx < 2; ++out_idx) {
    auto exp_it = (*expected)[out_idx].value_begin<float>();
    auto act_it = (*actual)[out_idx].value_begin<float>();
    for (; exp_it != (*expected)[out_idx].value_end<float>();
         ++exp_it, ++act_it) {
      EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
    }
  }

  // 2. Structural verification on large 16 MiB shape.
  const char* kLargeIr = R"mlir(
func.func @main(
    %logits: tensor<1x16x512x512xf32>,
    %ds: tensor<1x16x512x512xf32>) -> (tensor<1x16x512x512xf32>, tensor<1x16x512x512xf32>) {
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %one_3d = stablehlo.constant dense<1.0> : tensor<1x16x512xf32>
  %max_red = stablehlo.reduce(%logits init: %neg_inf)
      applies stablehlo.maximum across dimensions = [3]
      : (tensor<1x16x512x512xf32>, tensor<f32>) -> tensor<1x16x512xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max_red, dims = [0, 1, 2]
      : (tensor<1x16x512xf32>) -> tensor<1x16x512x512xf32>
  %shifted = stablehlo.subtract %logits, %max_bcast : tensor<1x16x512x512xf32>
  %exp = stablehlo.exponential %shifted : tensor<1x16x512x512xf32>
  %sum_red = stablehlo.reduce(%exp init: %zero)
      applies stablehlo.add across dimensions = [3]
      : (tensor<1x16x512x512xf32>, tensor<f32>) -> tensor<1x16x512xf32>
  %inv_sum = stablehlo.divide %one_3d, %sum_red : tensor<1x16x512xf32>
  %inv_bcast = stablehlo.broadcast_in_dim %inv_sum, dims = [0, 1, 2]
      : (tensor<1x16x512xf32>) -> tensor<1x16x512x512xf32>
  %probs = stablehlo.multiply %exp, %inv_bcast : tensor<1x16x512x512xf32>
  %s_mul_ds = stablehlo.multiply %probs, %ds : tensor<1x16x512x512xf32>
  %rowsum = stablehlo.reduce(%s_mul_ds init: %zero)
      applies stablehlo.add across dimensions = [3]
      : (tensor<1x16x512x512xf32>, tensor<f32>) -> tensor<1x16x512xf32>
  %rowsum_bcast = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2]
      : (tensor<1x16x512xf32>) -> tensor<1x16x512x512xf32>
  %shifted_ds = stablehlo.subtract %ds, %rowsum_bcast : tensor<1x16x512x512xf32>
  %dp = stablehlo.multiply %probs, %shifted_ds : tensor<1x16x512x512xf32>
  return %probs, %dp : tensor<1x16x512x512xf32>, tensor<1x16x512x512xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_large_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kLargeIr, &context);
  ASSERT_TRUE(opt_large_mod);

  mlir::PassManager pm_large(&context);
  pm_large.addNestedPass<mlir::func::FuncOp>(
      createOptimizeSdpaAndSoftmaxReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm_large.run(*opt_large_mod)));
  EXPECT_EQ(CountOps<mlir::stablehlo::SubtractOp>(*opt_large_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::NegOp>(*opt_large_mod), 2);
}

TEST(OptimizeSdpaAndSoftmaxReductionsTest,
     HandlesZeroResultUserWithoutCrashing) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %logits: tensor<1x8x142x142xf32>) -> tensor<1x8x142x142xf32> {
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %max_red = stablehlo.reduce(%logits init: %neg_inf)
      applies stablehlo.maximum across dimensions = [3]
      : (tensor<1x8x142x142xf32>, tensor<f32>) -> tensor<1x8x142xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max_red, dims = [0, 1, 2]
      : (tensor<1x8x142xf32>) -> tensor<1x8x142x142xf32>
  %shifted = stablehlo.subtract %logits, %max_bcast : tensor<1x8x142x142xf32>
  %exp = stablehlo.exponential %shifted : tensor<1x8x142x142xf32>
  %sum_red = stablehlo.reduce(%exp init: %zero)
      applies stablehlo.add across dimensions = [3]
      : (tensor<1x8x142x142xf32>, tensor<f32>) -> tensor<1x8x142xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum_red, dims = [0, 1, 2]
      : (tensor<1x8x142xf32>) -> tensor<1x8x142x142xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<1x8x142x142xf32>
  return %probs : tensor<1x8x142x142xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createOptimizeSdpaAndSoftmaxReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));
  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::MulOp>(*opt_mod), 1);
}

}  // namespace
}  // namespace torch_tpu
