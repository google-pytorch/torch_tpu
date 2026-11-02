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

#include "csrc/mlir/passes/fold_concat_softmax.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
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
  module.walk([&](OpTy) { ++count; });
  return count;
}

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float scale = 0.1f) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = static_cast<float>((i * 17 + 5) % 53 - 26) * scale;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape,
                                       float base = 0.1f) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    float f = base + static_cast<float>((i * 5 + 1) % 31) * 0.05f;
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

bool RunPassDirectly(mlir::ModuleOp module, mlir::MLIRContext& context) {
  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createFoldConcatSoftmaxPass());
  return mlir::succeeded(pm.run(module));
}

TEST(FoldConcatSoftmaxTest,
     PartitionsReciprocalConcatSoftmaxWithTwoResultMaxPreSubtraction) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<2x8xf32>,
    %rhs: tensor<2x1xf32>) -> tensor<2x8xf32> {
  %cat = stablehlo.concatenate %lhs, %rhs, dim = 1
      : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x9xf32>
  %idx = stablehlo.iota dim = 1 : tensor<2x9xi64>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero_i64 = stablehlo.constant dense<0> : tensor<i64>
  %zero_f = stablehlo.constant dense<0.0> : tensor<f32>
  %one_2x1 = stablehlo.constant dense<1.0> : tensor<2x1xf32>
  %max1, %argmax1 = "stablehlo.reduce"(%cat, %idx, %neg_inf, %zero_i64) <{
      dimensions = array<i64: 1>
  }> ({
    ^bb0(%v0: tensor<f32>, %i0: tensor<i64>,
         %v1: tensor<f32>, %i1: tensor<i64>):
      %m = stablehlo.maximum %v0, %v1 : tensor<f32>
      %cmp = stablehlo.compare GE, %v0, %v1
          : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %idx_m = stablehlo.select %cmp, %i0, %i1 : tensor<i1>, tensor<i64>
      stablehlo.return %m, %idx_m : tensor<f32>, tensor<i64>
  }) : (tensor<2x9xf32>, tensor<2x9xi64>, tensor<f32>, tensor<i64>)
      -> (tensor<2xf32>, tensor<2xi64>)
  %bcast1 = stablehlo.broadcast_in_dim %max1, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %sub1 = stablehlo.subtract %cat, %bcast1 : tensor<2x9xf32>
  %max2 = stablehlo.reduce(%sub1 init: %neg_inf)
      applies stablehlo.maximum across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %bcast2 = stablehlo.broadcast_in_dim %max2, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %sub2 = stablehlo.subtract %sub1, %bcast2 : tensor<2x9xf32>
  %exp = stablehlo.exponential %sub2 : tensor<2x9xf32>
  %sum = stablehlo.reduce(%exp init: %zero_f)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %sum_2x1 = stablehlo.reshape %sum : (tensor<2xf32>) -> tensor<2x1xf32>
  %inv = stablehlo.divide %one_2x1, %sum_2x1 : tensor<2x1xf32>
  %inv_bcast = stablehlo.broadcast_in_dim %inv, dims = [0, 1]
      : (tensor<2x1xf32>) -> tensor<2x9xf32>
  %probs = stablehlo.multiply %exp, %inv_bcast : tensor<2x9xf32>
  %slice = stablehlo.slice %probs [0:2, 0:8]
      : (tensor<2x9xf32>) -> tensor<2x8xf32>
  return %slice : tensor<2x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr lhs = MakeF32Tensor(context, {2, 8}, 0.2f);
  mlir::DenseElementsAttr rhs = MakeF32Tensor(context, {2, 1}, 0.3f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {lhs, rhs}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPassDirectly(*opt_mod, context));
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod), 0);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {lhs, rhs}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
  }
}

TEST(FoldConcatSoftmaxTest, PartitionsConcatSoftmaxBackwardExact) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %p_lhs: tensor<2x8xf32>,
    %p_rhs: tensor<2x1xf32>,
    %g_lhs: tensor<2x8xf32>,
    %g_rhs: tensor<2x1xf32>) -> (tensor<2x8xf32>, tensor<2x1xf32>) {
  %p_cat = stablehlo.concatenate %p_lhs, %p_rhs, dim = 1
      : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x9xf32>
  %g_cat = stablehlo.concatenate %g_lhs, %g_rhs, dim = 1
      : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x9xf32>
  %dot = stablehlo.dot_general %p_cat, %g_cat,
      batching_dims = [0] x [0],
      contracting_dims = [1] x [1]
      : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2xf32>
  %dot_bcast = stablehlo.broadcast_in_dim %dot, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %sub = stablehlo.subtract %g_cat, %dot_bcast : tensor<2x9xf32>
  %dx = stablehlo.multiply %p_cat, %sub : tensor<2x9xf32>
  %dx_lhs = stablehlo.slice %dx [0:2, 0:8]
      : (tensor<2x9xf32>) -> tensor<2x8xf32>
  %dx_rhs = stablehlo.slice %dx [0:2, 8:9]
      : (tensor<2x9xf32>) -> tensor<2x1xf32>
  return %dx_lhs, %dx_rhs : tensor<2x8xf32>, tensor<2x1xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr p_lhs = MakeF32Tensor(context, {2, 8}, 0.1f);
  mlir::DenseElementsAttr p_rhs = MakeF32Tensor(context, {2, 1}, 0.2f);
  mlir::DenseElementsAttr g_lhs = MakeF32Tensor(context, {2, 8}, 0.15f);
  mlir::DenseElementsAttr g_rhs = MakeF32Tensor(context, {2, 1}, 0.05f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {p_lhs, p_rhs, g_lhs, g_rhs}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPassDirectly(*opt_mod, context));
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod), 0);

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {p_lhs, p_rhs, g_lhs, g_rhs}, cfg);
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

TEST(FoldConcatSoftmaxTest, PartitionsArbitraryRankAndTailDivConcatSoftmax) {
  mlir::MLIRContext context;
  InitContext(context);

  // Tests 3D tensor with 32-wide tail (exceeding legacy 16 threshold),
  // concatenated along reduction dim 2.
  const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<2x3x64xf32>,
    %rhs: tensor<2x3x32xf32>) -> tensor<2x3x64xf32> {
  %cat = stablehlo.concatenate %lhs, %rhs, dim = 2
      : (tensor<2x3x64xf32>, tensor<2x3x32xf32>) -> tensor<2x3x96xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero_f = stablehlo.constant dense<0.0> : tensor<f32>
  %max = stablehlo.reduce(%cat init: %neg_inf)
      applies stablehlo.maximum across dimensions = [2]
      : (tensor<2x3x96xf32>, tensor<f32>) -> tensor<2x3xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0, 1]
      : (tensor<2x3xf32>) -> tensor<2x3x96xf32>
  %sub = stablehlo.subtract %cat, %max_bcast : tensor<2x3x96xf32>
  %exp = stablehlo.exponential %sub : tensor<2x3x96xf32>
  %sum = stablehlo.reduce(%exp init: %zero_f)
      applies stablehlo.add across dimensions = [2]
      : (tensor<2x3x96xf32>, tensor<f32>) -> tensor<2x3xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0, 1]
      : (tensor<2x3xf32>) -> tensor<2x3x96xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<2x3x96xf32>
  %slice = stablehlo.slice %probs [0:2, 0:3, 0:64]
      : (tensor<2x3x96xf32>) -> tensor<2x3x64xf32>
  return %slice : tensor<2x3x64xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr lhs = MakeF32Tensor(context, {2, 3, 64}, 0.2f);
  mlir::DenseElementsAttr rhs = MakeF32Tensor(context, {2, 3, 32}, 0.3f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {lhs, rhs}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPassDirectly(*opt_mod, context));
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod), 0);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {lhs, rhs}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
  }
}

TEST(FoldConcatSoftmaxTest,
     DoesNotPartitionWhenConcatDimDiffersFromReductionDim) {
  mlir::MLIRContext context;
  InitContext(context);

  // Concat along dim 0, reduction along dim 1. Pass must not partition.
  const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<2x8xf32>,
    %rhs: tensor<2x8xf32>) -> tensor<4x8xf32> {
  %cat = stablehlo.concatenate %lhs, %rhs, dim = 0
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<4x8xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero_f = stablehlo.constant dense<0.0> : tensor<f32>
  %max = stablehlo.reduce(%cat init: %neg_inf)
      applies stablehlo.maximum across dimensions = [1]
      : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0]
      : (tensor<4xf32>) -> tensor<4x8xf32>
  %sub = stablehlo.subtract %cat, %max_bcast : tensor<4x8xf32>
  %exp = stablehlo.exponential %sub : tensor<4x8xf32>
  %sum = stablehlo.reduce(%exp init: %zero_f)
      applies stablehlo.add across dimensions = [1]
      : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<4xf32>) -> tensor<4x8xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<4x8xf32>
  return %probs : tensor<4x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  ASSERT_TRUE(RunPassDirectly(*opt_mod, context));
  // ConcatOp remains because concat dimension 0 != reduction dimension 1.
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 1);
}

TEST(FoldConcatSoftmaxTest, RunTorchTpuOptimizationPassesRunsPassByDefault) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<2x8xf32>,
    %rhs: tensor<2x1xf32>) -> tensor<2x8xf32> {
  %cat = stablehlo.concatenate %lhs, %rhs, dim = 1
      : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x9xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero_f = stablehlo.constant dense<0.0> : tensor<f32>
  %max = stablehlo.reduce(%cat init: %neg_inf)
      applies stablehlo.maximum across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %sub = stablehlo.subtract %cat, %max_bcast : tensor<2x9xf32>
  %exp = stablehlo.exponential %sub : tensor<2x9xf32>
  %sum = stablehlo.reduce(%exp init: %zero_f)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<2x9xf32>
  %slice = stablehlo.slice %probs [0:2, 0:8]
      : (tensor<2x9xf32>) -> tensor<2x8xf32>
  return %slice : tensor<2x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 0);
}

#if GTEST_HAS_DEATH_TEST
TEST(FoldConcatSoftmaxTest, PassDisabledViaEnvVarInSubprocess) {
  EXPECT_EXIT(
      {
        setenv("TORCH_TPU_INTERNAL_ENABLE_FOLD_CONCAT_SOFTMAX", "0", 1);
        mlir::MLIRContext context;
        InitContext(context);
        const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<2x8xf32>,
    %rhs: tensor<2x1xf32>) -> tensor<2x8xf32> {
  %cat = stablehlo.concatenate %lhs, %rhs, dim = 1
      : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x9xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero_f = stablehlo.constant dense<0.0> : tensor<f32>
  %max = stablehlo.reduce(%cat init: %neg_inf)
      applies stablehlo.maximum across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %sub = stablehlo.subtract %cat, %max_bcast : tensor<2x9xf32>
  %exp = stablehlo.exponential %sub : tensor<2x9xf32>
  %sum = stablehlo.reduce(%exp init: %zero_f)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<2x9xf32>
  %slice = stablehlo.slice %probs [0:2, 0:8]
      : (tensor<2x9xf32>) -> tensor<2x8xf32>
  return %slice : tensor<2x8xf32>
}
)mlir";
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
        if (!module || !RunTorchTpuOptimizationPasses(*module, &context).ok()) {
          std::exit(1);
        }
        if (CountOps<mlir::stablehlo::ConcatenateOp>(*module) != 1) {
          std::exit(2);
        }
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}
#endif

TEST(FoldConcatSoftmaxTest, PartitionsRank1ConcatSoftmaxAlongDim0) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<8xf32>,
    %rhs: tensor<1xf32>) -> tensor<8xf32> {
  %cat = stablehlo.concatenate %lhs, %rhs, dim = 0
      : (tensor<8xf32>, tensor<1xf32>) -> tensor<9xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero_f = stablehlo.constant dense<0.0> : tensor<f32>
  %max = stablehlo.reduce(%cat init: %neg_inf)
      applies stablehlo.maximum across dimensions = [0]
      : (tensor<9xf32>, tensor<f32>) -> tensor<f32>
  %max_bcast = stablehlo.broadcast_in_dim %max, dims = []
      : (tensor<f32>) -> tensor<9xf32>
  %sub = stablehlo.subtract %cat, %max_bcast : tensor<9xf32>
  %exp = stablehlo.exponential %sub : tensor<9xf32>
  %sum = stablehlo.reduce(%exp init: %zero_f)
      applies stablehlo.add across dimensions = [0]
      : (tensor<9xf32>, tensor<f32>) -> tensor<f32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = []
      : (tensor<f32>) -> tensor<9xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<9xf32>
  %slice = stablehlo.slice %probs [0:8]
      : (tensor<9xf32>) -> tensor<8xf32>
  return %slice : tensor<8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr lhs = MakeF32Tensor(context, {8}, 0.2f);
  mlir::DenseElementsAttr rhs = MakeF32Tensor(context, {1}, 0.3f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {lhs, rhs}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPassDirectly(*opt_mod, context));
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod), 0);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {lhs, rhs}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
  }
}

TEST(FoldConcatSoftmaxTest,
     PartitionsBf16ConcatSoftmaxBackwardWithF32DotGeneralAccumulation) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %p_lhs: tensor<2x8xbf16>,
    %p_rhs: tensor<2x1xbf16>,
    %g_lhs: tensor<2x8xbf16>,
    %g_rhs: tensor<2x1xbf16>) -> (tensor<2x8xbf16>, tensor<2x1xbf16>) {
  %p_cat = stablehlo.concatenate %p_lhs, %p_rhs, dim = 1
      : (tensor<2x8xbf16>, tensor<2x1xbf16>) -> tensor<2x9xbf16>
  %g_cat = stablehlo.concatenate %g_lhs, %g_rhs, dim = 1
      : (tensor<2x8xbf16>, tensor<2x1xbf16>) -> tensor<2x9xbf16>
  %dot = stablehlo.dot_general %p_cat, %g_cat,
      batching_dims = [0] x [0],
      contracting_dims = [1] x [1]
      : (tensor<2x9xbf16>, tensor<2x9xbf16>) -> tensor<2xf32>
  %dot_bf16 = stablehlo.convert %dot : (tensor<2xf32>) -> tensor<2xbf16>
  %dot_bcast = stablehlo.broadcast_in_dim %dot_bf16, dims = [0]
      : (tensor<2xbf16>) -> tensor<2x9xbf16>
  %sub = stablehlo.subtract %g_cat, %dot_bcast : tensor<2x9xbf16>
  %dx = stablehlo.multiply %p_cat, %sub : tensor<2x9xbf16>
  %dx_lhs = stablehlo.slice %dx [0:2, 0:8]
      : (tensor<2x9xbf16>) -> tensor<2x8xbf16>
  %dx_rhs = stablehlo.slice %dx [0:2, 8:9]
      : (tensor<2x9xbf16>) -> tensor<2x1xbf16>
  return %dx_lhs, %dx_rhs : tensor<2x8xbf16>, tensor<2x1xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr p_lhs = MakeBF16Tensor(context, {2, 8}, 0.1f);
  mlir::DenseElementsAttr p_rhs = MakeBF16Tensor(context, {2, 1}, 0.2f);
  mlir::DenseElementsAttr g_lhs = MakeBF16Tensor(context, {2, 8}, 0.15f);
  mlir::DenseElementsAttr g_rhs = MakeBF16Tensor(context, {2, 1}, 0.05f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {p_lhs, p_rhs, g_lhs, g_rhs}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPassDirectly(*opt_mod, context));
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod), 0);

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {p_lhs, p_rhs, g_lhs, g_rhs}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (int out_idx = 0; out_idx < 2; ++out_idx) {
    auto exp_it = (*expected)[out_idx].value_begin<llvm::APFloat>();
    auto act_it = (*actual)[out_idx].value_begin<llvm::APFloat>();
    for (; exp_it != (*expected)[out_idx].value_end<llvm::APFloat>();
         ++exp_it, ++act_it) {
      EXPECT_NEAR((*exp_it).convertToFloat(), (*act_it).convertToFloat(),
                  0.05f);
    }
  }
}

TEST(FoldConcatSoftmaxTest, DoesNotPartitionWhenConcatSoftmaxOutputHasNoSlice) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<2x8xf32>,
    %rhs: tensor<2x1xf32>) -> tensor<2x9xf32> {
  %cat = stablehlo.concatenate %lhs, %rhs, dim = 1
      : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x9xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero_f = stablehlo.constant dense<0.0> : tensor<f32>
  %max = stablehlo.reduce(%cat init: %neg_inf)
      applies stablehlo.maximum across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %sub = stablehlo.subtract %cat, %max_bcast : tensor<2x9xf32>
  %exp = stablehlo.exponential %sub : tensor<2x9xf32>
  %sum = stablehlo.reduce(%exp init: %zero_f)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<2x9xf32>
  return %probs : tensor<2x9xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  ASSERT_TRUE(RunPassDirectly(*opt_mod, context));
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 1);
}

TEST(FoldConcatSoftmaxTest, DoesNotPartitionWhenMaxReduceInitIsFinite) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<2x8xf32>,
    %rhs: tensor<2x1xf32>) -> tensor<2x8xf32> {
  %cat = stablehlo.concatenate %lhs, %rhs, dim = 1
      : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x9xf32>
  %zero_f = stablehlo.constant dense<0.0> : tensor<f32>
  %max = stablehlo.reduce(%cat init: %zero_f)
      applies stablehlo.maximum across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %sub = stablehlo.subtract %cat, %max_bcast : tensor<2x9xf32>
  %exp = stablehlo.exponential %sub : tensor<2x9xf32>
  %sum = stablehlo.reduce(%exp init: %zero_f)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<2x9xf32>
  %slice = stablehlo.slice %probs [0:2, 0:8]
      : (tensor<2x9xf32>) -> tensor<2x8xf32>
  return %slice : tensor<2x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  ASSERT_TRUE(RunPassDirectly(*opt_mod, context));
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 1);
}

TEST(FoldConcatSoftmaxTest, DoesNotPartitionWhenSumReduceInitIsNonzero) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<2x8xf32>,
    %rhs: tensor<2x1xf32>) -> tensor<2x8xf32> {
  %cat = stablehlo.concatenate %lhs, %rhs, dim = 1
      : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x9xf32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %one_f = stablehlo.constant dense<1.0> : tensor<f32>
  %max = stablehlo.reduce(%cat init: %neg_inf)
      applies stablehlo.maximum across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %sub = stablehlo.subtract %cat, %max_bcast : tensor<2x9xf32>
  %exp = stablehlo.exponential %sub : tensor<2x9xf32>
  %sum = stablehlo.reduce(%exp init: %one_f)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x9xf32>, tensor<f32>) -> tensor<2xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x9xf32>
  %probs = stablehlo.divide %exp, %sum_bcast : tensor<2x9xf32>
  %slice = stablehlo.slice %probs [0:2, 0:8]
      : (tensor<2x9xf32>) -> tensor<2x8xf32>
  return %slice : tensor<2x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  ASSERT_TRUE(RunPassDirectly(*opt_mod, context));
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 1);
}

}  // namespace
}  // namespace torch_tpu
