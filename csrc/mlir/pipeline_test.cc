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

#include <cstdint>

#include "absl/status/status_matchers.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using testing::HasSubstr;

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

TEST(PipelineTest, FoldsShiftInvariantPreSubtractionBeforeSoftmax) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%arg0: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %max1 = stablehlo.reduce(%arg0 init: %min_f32) across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %bcast_max1 = stablehlo.broadcast_in_dim %max1, dims = [0] : (tensor<4xf32>) -> tensor<4x16xf32>
    %x = stablehlo.subtract %arg0, %bcast_max1 : tensor<4x16xf32>
    %max2 = stablehlo.reduce(%x init: %min_f32) across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %bcast_max2 = stablehlo.broadcast_in_dim %max2, dims = [0] : (tensor<4xf32>) -> tensor<4x16xf32>
    %shifted = stablehlo.subtract %x, %bcast_max2 : tensor<4x16xf32>
    %exp = stablehlo.exponential %shifted : tensor<4x16xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_sum = stablehlo.broadcast_in_dim %sum, dims = [0] : (tensor<4xf32>) -> tensor<4x16xf32>
    %out = stablehlo.divide %exp, %bcast_sum : tensor<4x16xf32>
    return %out : tensor<4x16xf32>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int max_reduce_count = 0;
  int sub_count = 0;
  module->walk([&](mlir::stablehlo::ReduceOp op) {
    op.getBody().walk([&](mlir::stablehlo::MaxOp) { ++max_reduce_count; });
  });
  module->walk([&](mlir::stablehlo::SubtractOp) { ++sub_count; });

  EXPECT_EQ(max_reduce_count, 1);
  EXPECT_EQ(sub_count, 1);
}

TEST(PipelineTest, FoldsGptOssAttentionSinkConcatSoftmaxSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%attn_weights: tensor<2x4x8x16xbf16>, %sinks: tensor<2x4x8x1xbf16>) -> tensor<2x4x8x16xbf16> {
    %min_bf16 = stablehlo.constant dense<0xFF7F> : tensor<bf16>
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %concat = stablehlo.concatenate %attn_weights, %sinks, dim = 3 : (tensor<2x4x8x16xbf16>, tensor<2x4x8x1xbf16>) -> tensor<2x4x8x17xbf16>
    %max1 = stablehlo.reduce(%concat init: %min_bf16) across dimensions = [3] : (tensor<2x4x8x17xbf16>, tensor<bf16>) -> tensor<2x4x8xbf16>
     reducer(%lhs: tensor<bf16>, %rhs: tensor<bf16>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<bf16>
      stablehlo.return %m : tensor<bf16>
    }
    %max1_keepdim = stablehlo.reshape %max1 : (tensor<2x4x8xbf16>) -> tensor<2x4x8x1xbf16>
    %bcast_max1 = stablehlo.broadcast_in_dim %max1_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xbf16>) -> tensor<2x4x8x17xbf16>
    %centered = stablehlo.subtract %concat, %bcast_max1 : tensor<2x4x8x17xbf16>
    %centered_f32 = stablehlo.convert %centered : (tensor<2x4x8x17xbf16>) -> tensor<2x4x8x17xf32>
    %max2 = stablehlo.reduce(%centered_f32 init: %min_f32) across dimensions = [3] : (tensor<2x4x8x17xf32>, tensor<f32>) -> tensor<2x4x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %max2_keepdim = stablehlo.reshape %max2 : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
    %bcast_max2 = stablehlo.broadcast_in_dim %max2_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8x17xf32>
    %shifted = stablehlo.subtract %centered_f32, %bcast_max2 : tensor<2x4x8x17xf32>
    %exp = stablehlo.exponential %shifted : tensor<2x4x8x17xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<2x4x8x17xf32>, tensor<f32>) -> tensor<2x4x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %sum_keepdim = stablehlo.reshape %sum : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
    %bcast_sum = stablehlo.broadcast_in_dim %sum_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8x17xf32>
    %probs_f32 = stablehlo.divide %exp, %bcast_sum : tensor<2x4x8x17xf32>
    %probs_bf16 = stablehlo.convert %probs_f32 : (tensor<2x4x8x17xf32>) -> tensor<2x4x8x17xbf16>
    %scores = stablehlo.slice %probs_bf16 [0:2, 0:4, 0:8, 0:16] : (tensor<2x4x8x17xbf16>) -> tensor<2x4x8x16xbf16>
    return %scores : tensor<2x4x8x16xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int concat_count = 0;
  int slice_count = 0;
  int dim17_value_count = 0;
  int max_reduce_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    }
    for (mlir::Value res : op->getResults()) {
      if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(res.getType())) {
        for (int64_t d : rtt.getShape()) {
          if (d == 17) {
            ++dim17_value_count;
          }
        }
      }
    }
    if (auto reduce = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
      reduce.getBody().walk(
          [&](mlir::stablehlo::MaxOp) { ++max_reduce_count; });
    }
  });

  EXPECT_EQ(concat_count, 0);
  EXPECT_EQ(slice_count, 0);
  EXPECT_EQ(dim17_value_count, 0);
  EXPECT_EQ(max_reduce_count, 1);
}

TEST(PipelineTest, FoldsGptOssAttentionSinkConcatSoftmaxSliceMultiTokenSink) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%attn_weights: tensor<2x4x8x16xbf16>, %sinks: tensor<2x4x8x4xbf16>) -> tensor<2x4x8x16xbf16> {
    %min_bf16 = stablehlo.constant dense<0xFF7F> : tensor<bf16>
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %concat = stablehlo.concatenate %attn_weights, %sinks, dim = 3 : (tensor<2x4x8x16xbf16>, tensor<2x4x8x4xbf16>) -> tensor<2x4x8x20xbf16>
    %max1 = stablehlo.reduce(%concat init: %min_bf16) across dimensions = [3] : (tensor<2x4x8x20xbf16>, tensor<bf16>) -> tensor<2x4x8xbf16>
     reducer(%lhs: tensor<bf16>, %rhs: tensor<bf16>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<bf16>
      stablehlo.return %m : tensor<bf16>
    }
    %max1_keepdim = stablehlo.reshape %max1 : (tensor<2x4x8xbf16>) -> tensor<2x4x8x1xbf16>
    %bcast_max1 = stablehlo.broadcast_in_dim %max1_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xbf16>) -> tensor<2x4x8x20xbf16>
    %centered = stablehlo.subtract %concat, %bcast_max1 : tensor<2x4x8x20xbf16>
    %centered_f32 = stablehlo.convert %centered : (tensor<2x4x8x20xbf16>) -> tensor<2x4x8x20xf32>
    %max2 = stablehlo.reduce(%centered_f32 init: %min_f32) across dimensions = [3] : (tensor<2x4x8x20xf32>, tensor<f32>) -> tensor<2x4x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %max2_keepdim = stablehlo.reshape %max2 : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
    %bcast_max2 = stablehlo.broadcast_in_dim %max2_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8x20xf32>
    %shifted = stablehlo.subtract %centered_f32, %bcast_max2 : tensor<2x4x8x20xf32>
    %exp = stablehlo.exponential %shifted : tensor<2x4x8x20xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<2x4x8x20xf32>, tensor<f32>) -> tensor<2x4x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %sum_keepdim = stablehlo.reshape %sum : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
    %bcast_sum = stablehlo.broadcast_in_dim %sum_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8x20xf32>
    %probs_f32 = stablehlo.divide %exp, %bcast_sum : tensor<2x4x8x20xf32>
    %probs_bf16 = stablehlo.convert %probs_f32 : (tensor<2x4x8x20xf32>) -> tensor<2x4x8x20xbf16>
    %scores = stablehlo.slice %probs_bf16 [0:2, 0:4, 0:8, 0:16] : (tensor<2x4x8x20xbf16>) -> tensor<2x4x8x16xbf16>
    return %scores : tensor<2x4x8x16xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int concat_count = 0;
  int slice_count = 0;
  int dim20_value_count = 0;
  int max_reduce_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    }
    for (mlir::Value res : op->getResults()) {
      if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(res.getType())) {
        for (int64_t d : rtt.getShape()) {
          if (d == 20) {
            ++dim20_value_count;
          }
        }
      }
    }
    if (auto reduce = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
      reduce.getBody().walk(
          [&](mlir::stablehlo::MaxOp) { ++max_reduce_count; });
    }
  });

  EXPECT_EQ(concat_count, 0);
  EXPECT_EQ(slice_count, 0);
  EXPECT_EQ(dim20_value_count, 0);
  EXPECT_EQ(max_reduce_count, 2);
}

TEST(PipelineTest,
     CanonicalizesDecomposedGemmaRmsNormAndReassociatesMixedDtypeWeight) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x8x16xbf16>, %w_f32: tensor<16xf32>) -> tensor<2x8x16xbf16> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %two = stablehlo.constant dense<2.000000e+00> : tensor<2x8x16xf32>
    %n = stablehlo.constant dense<1.600000e+01> : tensor<2x8x1xf32>
    %eps = stablehlo.constant dense<1.000000e-06> : tensor<2x8x1xf32>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<16xf32>
    %x_f32 = stablehlo.convert %x : (tensor<2x8x16xbf16>) -> tensor<2x8x16xf32>
    %sq = stablehlo.power %x_f32, %two : tensor<2x8x16xf32>
    %sum = stablehlo.reduce(%sq init: %zero) across dimensions = [2] : (tensor<2x8x16xf32>, tensor<f32>) -> tensor<2x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %keepdim = stablehlo.reshape %sum : (tensor<2x8xf32>) -> tensor<2x8x1xf32>
    %var = stablehlo.divide %keepdim, %n : tensor<2x8x1xf32>
    %var_eps = stablehlo.add %var, %eps : tensor<2x8x1xf32>
    %rstd = stablehlo.rsqrt %var_eps : tensor<2x8x1xf32>
    %bcast_rstd = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<2x8x1xf32>) -> tensor<2x8x16xf32>
    %norm_f32 = stablehlo.multiply %x_f32, %bcast_rstd : tensor<2x8x16xf32>
    %norm_bf16 = stablehlo.convert %norm_f32 : (tensor<2x8x16xf32>) -> tensor<2x8x16xbf16>
    %one_plus_w_f32 = stablehlo.add %one, %w_f32 : tensor<16xf32>
    %one_plus_w_bf16 = stablehlo.convert %one_plus_w_f32 : (tensor<16xf32>) -> tensor<16xbf16>
    %bcast_w = stablehlo.broadcast_in_dim %one_plus_w_bf16, dims = [2] : (tensor<16xbf16>) -> tensor<2x8x16xbf16>
    %out = stablehlo.multiply %norm_bf16, %bcast_w : tensor<2x8x16xbf16>
    return %out : tensor<2x8x16xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int pow_count = 0;
  int div_count = 0;
  int one_const_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::PowOp>(op)) {
      ++pow_count;
    }
    if (mlir::isa<mlir::stablehlo::DivOp>(op)) {
      ++div_count;
    }
    if (auto const_op = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
      if (auto attr =
              mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue())) {
        if (attr.isSplat() &&
            mlir::isa<mlir::FloatType>(attr.getElementType()) &&
            attr.getSplatValue<llvm::APFloat>().isExactlyValue(1.0)) {
          ++one_const_count;
        }
      }
    }
  });

  EXPECT_EQ(pow_count, 0);
  EXPECT_EQ(div_count, 0);
  EXPECT_EQ(one_const_count, 0);
}

TEST(PipelineTest, CommutesInverse4DTransposePairAroundQkRmsNorm) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x8x4x16xbf16>, %w: tensor<16xbf16>) -> tensor<2x8x4x16xbf16> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %two = stablehlo.constant dense<2.000000e+00> : tensor<2x4x8x16xf32>
    %n = stablehlo.constant dense<1.600000e+01> : tensor<2x4x8x1xf32>
    %eps = stablehlo.constant dense<1.000000e-06> : tensor<2x4x8x1xf32>
    %t_in = stablehlo.transpose %x, dims = [0, 2, 1, 3] : (tensor<2x8x4x16xbf16>) -> tensor<2x4x8x16xbf16>
    %x_f32 = stablehlo.convert %t_in : (tensor<2x4x8x16xbf16>) -> tensor<2x4x8x16xf32>
    %sq = stablehlo.power %x_f32, %two : tensor<2x4x8x16xf32>
    %sum = stablehlo.reduce(%sq init: %zero) across dimensions = [3] : (tensor<2x4x8x16xf32>, tensor<f32>) -> tensor<2x4x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %keepdim = stablehlo.reshape %sum : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
    %var = stablehlo.divide %keepdim, %n : tensor<2x4x8x1xf32>
    %var_eps = stablehlo.add %var, %eps : tensor<2x4x8x1xf32>
    %rstd = stablehlo.rsqrt %var_eps : tensor<2x4x8x1xf32>
    %bcast_rstd = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8x16xf32>
    %norm_f32 = stablehlo.multiply %x_f32, %bcast_rstd : tensor<2x4x8x16xf32>
    %norm_bf16 = stablehlo.convert %norm_f32 : (tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xbf16>
    %bcast_w = stablehlo.broadcast_in_dim %w, dims = [3] : (tensor<16xbf16>) -> tensor<2x4x8x16xbf16>
    %scaled = stablehlo.multiply %norm_bf16, %bcast_w : tensor<2x4x8x16xbf16>
    %t_out = stablehlo.transpose %scaled, dims = [0, 2, 1, 3] : (tensor<2x4x8x16xbf16>) -> tensor<2x8x4x16xbf16>
    return %t_out : tensor<2x8x4x16xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int transpose_count = 0;
  int pow_count = 0;
  int div_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::TransposeOp>(op)) {
      ++transpose_count;
    }
    if (mlir::isa<mlir::stablehlo::PowOp>(op)) {
      ++pow_count;
    }
    if (mlir::isa<mlir::stablehlo::DivOp>(op)) {
      ++div_count;
    }
  });

  EXPECT_EQ(transpose_count, 0);
  EXPECT_EQ(pow_count, 0);
  EXPECT_EQ(div_count, 0);
}

TEST(PipelineTest, FoldsComplementaryZeroPadAddIntoConcatenate) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%a: tensor<2x4x8x32xbf16>, %b: tensor<2x4x8x32xbf16>) -> tensor<2x4x8x64xbf16> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %pad_a = stablehlo.pad %a, %zero, low = [0, 0, 0, 0], high = [0, 0, 0, 32], interior = [0, 0, 0, 0] : (tensor<2x4x8x32xbf16>, tensor<bf16>) -> tensor<2x4x8x64xbf16>
    %pad_b = stablehlo.pad %b, %zero, low = [0, 0, 0, 32], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<2x4x8x32xbf16>, tensor<bf16>) -> tensor<2x4x8x64xbf16>
    %sum = stablehlo.add %pad_a, %pad_b : tensor<2x4x8x64xbf16>
    return %sum : tensor<2x4x8x64xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int pad_count = 0;
  int add_count = 0;
  int concat_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::PadOp>(op)) {
      ++pad_count;
    }
    if (mlir::isa<mlir::stablehlo::AddOp>(op)) {
      ++add_count;
    }
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
  });

  EXPECT_EQ(pad_count, 0);
  EXPECT_EQ(add_count, 0);
  EXPECT_EQ(concat_count, 1);
}

TEST(PipelineTest, FoldsCommutedAnd3WayComplementaryZeroPadAddIntoConcatenate) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x0: tensor<2x16xf32>, %x1: tensor<2x16xf32>, %x2: tensor<2x32xf32>) -> tensor<2x64xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %p0 = stablehlo.pad %x0, %zero, low = [0, 0], high = [0, 48], interior = [0, 0] : (tensor<2x16xf32>, tensor<f32>) -> tensor<2x64xf32>
    %p1 = stablehlo.pad %x1, %zero, low = [0, 16], high = [0, 32], interior = [0, 0] : (tensor<2x16xf32>, tensor<f32>) -> tensor<2x64xf32>
    %p2 = stablehlo.pad %x2, %zero, low = [0, 32], high = [0, 0], interior = [0, 0] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2x64xf32>
    %sum20 = stablehlo.add %p2, %p0 : tensor<2x64xf32>
    %sum201 = stablehlo.add %sum20, %p1 : tensor<2x64xf32>
    return %sum201 : tensor<2x64xf32>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int pad_count = 0;
  int add_count = 0;
  int concat_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::PadOp>(op)) {
      ++pad_count;
    }
    if (mlir::isa<mlir::stablehlo::AddOp>(op)) {
      ++add_count;
    }
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
  });

  EXPECT_EQ(pad_count, 0);
  EXPECT_EQ(add_count, 0);
  EXPECT_GE(concat_count, 1);
}

}  // namespace
}  // namespace torch_tpu
