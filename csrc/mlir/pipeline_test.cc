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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
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
     CanonicalizesOuterProductBroadcastContractionWithGroupDedup) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%c_in: tensor<512x8x128xf32>, %b_in: tensor<512x8x128xf32>, %l_in: tensor<1x64x4x128x128xf32>, %h_in: tensor<1x4x128x64x64xf32>) -> tensor<1x4x128x64x64xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_rep = stablehlo.broadcast_in_dim %c_in, dims = [1, 2, 4] : (tensor<512x8x128xf32>) -> tensor<1x512x8x8x128xf32>
    %c_r1 = stablehlo.reshape %c_rep : (tensor<1x512x8x8x128xf32>) -> tensor<1x4x128x1x64x128xf32>
    %c_b1 = stablehlo.broadcast_in_dim %c_r1, dims = [0, 1, 2, 3, 4, 5] : (tensor<1x4x128x1x64x128xf32>) -> tensor<1x4x128x1x64x128xf32>
    %c_r2 = stablehlo.reshape %c_b1 : (tensor<1x4x128x1x64x128xf32>) -> tensor<1x4x128x64x128xf32>
    %c_b2 = stablehlo.broadcast_in_dim %c_r2, dims = [0, 1, 2, 4, 5] : (tensor<1x4x128x64x128xf32>) -> tensor<1x4x128x128x64x128xf32>
    %b_rep = stablehlo.broadcast_in_dim %b_in, dims = [1, 2, 4] : (tensor<512x8x128xf32>) -> tensor<1x512x8x8x128xf32>
    %b_r1 = stablehlo.reshape %b_rep : (tensor<1x512x8x8x128xf32>) -> tensor<1x4x1x128x64x128xf32>
    %b_b1 = stablehlo.broadcast_in_dim %b_r1, dims = [0, 1, 2, 3, 4, 5] : (tensor<1x4x1x128x64x128xf32>) -> tensor<1x4x1x128x64x128xf32>
    %b_r2 = stablehlo.reshape %b_b1 : (tensor<1x4x1x128x64x128xf32>) -> tensor<1x4x128x64x128xf32>
    %b_b2 = stablehlo.broadcast_in_dim %b_r2, dims = [0, 1, 3, 4, 5] : (tensor<1x4x128x64x128xf32>) -> tensor<1x4x128x128x64x128xf32>
    %g_mul = stablehlo.multiply %c_b2, %b_b2 : tensor<1x4x128x128x64x128xf32>
    %g_sum = stablehlo.reduce(%g_mul init: %zero) across dimensions = [5] : (tensor<1x4x128x128x64x128xf32>, tensor<f32>) -> tensor<1x4x128x128x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %g_6d = stablehlo.reshape %g_sum : (tensor<1x4x128x128x64xf32>) -> tensor<1x4x128x128x64x1xf32>
    %l_r = stablehlo.reshape %l_in : (tensor<1x64x4x128x128xf32>) -> tensor<64x4x128x128x1x1xf32>
    %l_tr = stablehlo.transpose %l_r, dims = [4, 1, 2, 3, 0, 5] : (tensor<64x4x128x128x1x1xf32>) -> tensor<1x4x128x128x64x1xf32>
    %m_mul = stablehlo.multiply %g_6d, %l_tr : tensor<1x4x128x128x64x1xf32>
    %m_sum = stablehlo.reduce(%m_mul init: %zero) across dimensions = [5] : (tensor<1x4x128x128x64x1xf32>, tensor<f32>) -> tensor<1x4x128x128x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %m_b = stablehlo.broadcast_in_dim %m_sum, dims = [0, 1, 2, 3, 4] : (tensor<1x4x128x128x64xf32>) -> tensor<1x4x128x128x64x64xf32>
    %h_b = stablehlo.broadcast_in_dim %h_in, dims = [0, 1, 3, 4, 5] : (tensor<1x4x128x64x64xf32>) -> tensor<1x4x128x128x64x64xf32>
    %y_mul = stablehlo.multiply %m_b, %h_b : tensor<1x4x128x128x64x64xf32>
    %y_sum = stablehlo.reduce(%y_mul init: %zero) across dimensions = [3] : (tensor<1x4x128x128x64x64xf32>, tensor<f32>) -> tensor<1x4x128x64x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %y_sum : tensor<1x4x128x64x64xf32>
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

  int reduce_count = 0;
  int dot_count = 0;
  int unrepeated_dot_count = 0;
  module->walk([&](mlir::stablehlo::ReduceOp) { ++reduce_count; });
  module->walk([&](mlir::stablehlo::DotGeneralOp dot) {
    ++dot_count;
    auto rtt = mlir::cast<mlir::RankedTensorType>(dot.getType());
    if (rtt.getShape() == llvm::ArrayRef<int64_t>({1, 4, 8, 128, 128})) {
      ++unrepeated_dot_count;
    }
  });

  EXPECT_EQ(reduce_count, 0);
  EXPECT_EQ(dot_count, 2);
  EXPECT_EQ(unrepeated_dot_count, 1);
}

TEST(PipelineTest, FoldsSingleKeySoftmaxAndSqueezesInteriorUnitLayerNorm) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%enc_in: tensor<1022x1x1024xbf16>, %w: tensor<1024xbf16>, %b: tensor<1024xbf16>, %q: tensor<16x1x64xbf16>, %k: tensor<16x64x1xbf16>, %v: tensor<16x1x64xbf16>) -> (tensor<1022x1x1024xbf16>, tensor<16x1x64xbf16>) {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %one_f32 = stablehlo.constant dense<1.000000e+00> : tensor<16x1x1xf32>
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %inv_n = stablehlo.constant dense<9.765625e-04> : tensor<1022x1xf32>
    %zero_2d = stablehlo.constant dense<0.000000e+00> : tensor<1022x1xf32>
    %eps_2d = stablehlo.constant dense<1.000000e-05> : tensor<1022x1xf32>

    %x = stablehlo.convert %enc_in : (tensor<1022x1x1024xbf16>) -> tensor<1022x1x1024xf32>
    %x2 = stablehlo.multiply %x, %x : tensor<1022x1x1024xf32>
    %ln:2 = stablehlo.reduce(%x init: %zero_f32), (%x2 init: %zero_f32) across dimensions = [2] : (tensor<1022x1x1024xf32>, tensor<1022x1x1024xf32>, tensor<f32>, tensor<f32>) -> (tensor<1022x1xf32>, tensor<1022x1xf32>)
     reducer(%a0: tensor<f32>, %b0: tensor<f32>) (%a1: tensor<f32>, %b1: tensor<f32>) {
      %s0 = stablehlo.add %a0, %b0 : tensor<f32>
      %s1 = stablehlo.add %a1, %b1 : tensor<f32>
      stablehlo.return %s0, %s1 : tensor<f32>, tensor<f32>
    }
    %mean = stablehlo.multiply %ln#0, %inv_n : tensor<1022x1xf32>
    %mean2 = stablehlo.multiply %ln#1, %inv_n : tensor<1022x1xf32>
    %mean_sq = stablehlo.multiply %mean, %mean : tensor<1022x1xf32>
    %var_raw = stablehlo.subtract %mean2, %mean_sq : tensor<1022x1xf32>
    %var = stablehlo.maximum %var_raw, %zero_2d : tensor<1022x1xf32>
    %var_eps = stablehlo.add %var, %eps_2d : tensor<1022x1xf32>
    %rstd = stablehlo.rsqrt %var_eps : tensor<1022x1xf32>
    %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1] : (tensor<1022x1xf32>) -> tensor<1022x1x1024xf32>
    %rstd_b = stablehlo.broadcast_in_dim %rstd, dims = [0, 1] : (tensor<1022x1xf32>) -> tensor<1022x1x1024xf32>
    %centered = stablehlo.subtract %x, %mean_b : tensor<1022x1x1024xf32>
    %normed = stablehlo.multiply %centered, %rstd_b : tensor<1022x1x1024xf32>
    %normed_bf16 = stablehlo.convert %normed : (tensor<1022x1x1024xf32>) -> tensor<1022x1x1024xbf16>
    %w_b = stablehlo.broadcast_in_dim %w, dims = [2] : (tensor<1024xbf16>) -> tensor<1022x1x1024xbf16>
    %scaled = stablehlo.multiply %normed_bf16, %w_b : tensor<1022x1x1024xbf16>
    %b_b = stablehlo.broadcast_in_dim %b, dims = [2] : (tensor<1024xbf16>) -> tensor<1022x1x1024xbf16>
    %ln_out = stablehlo.add %scaled, %b_b : tensor<1022x1x1024xbf16>

    %scores = stablehlo.dot_general %q, %k, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x1x64xbf16>, tensor<16x64x1xbf16>) -> tensor<16x1x1xbf16>
    %scores_f32 = stablehlo.convert %scores : (tensor<16x1x1xbf16>) -> tensor<16x1x1xf32>
    %max = stablehlo.reduce(%scores_f32 init: %min_f32) across dimensions = [2] : (tensor<16x1x1xf32>, tensor<f32>) -> tensor<16x1xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %max_r = stablehlo.reshape %max : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %shifted = stablehlo.subtract %scores_f32, %max_r : tensor<16x1x1xf32>
    %exp = stablehlo.exponential %shifted : tensor<16x1x1xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [2] : (tensor<16x1x1xf32>, tensor<f32>) -> tensor<16x1xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %sum_r = stablehlo.reshape %sum : (tensor<16x1xf32>) -> tensor<16x1x1xf32>
    %inv_sum = stablehlo.divide %one_f32, %sum_r : tensor<16x1x1xf32>
    %probs_f32 = stablehlo.multiply %exp, %inv_sum : tensor<16x1x1xf32>
    %probs = stablehlo.convert %probs_f32 : (tensor<16x1x1xf32>) -> tensor<16x1x1xbf16>
    %attn_out = stablehlo.dot_general %probs, %v, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x1x1xbf16>, tensor<16x1x64xbf16>) -> tensor<16x1x64xbf16>
    return %ln_out, %attn_out : tensor<1022x1x1024xbf16>, tensor<16x1x64xbf16>
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

  int dot_count = 0;
  int reduce_1d_count = 0;
  module->walk([&](mlir::stablehlo::DotGeneralOp) { ++dot_count; });
  module->walk([&](mlir::stablehlo::ReduceOp red) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(red.getResultTypes()[0]);
    if (rtt.getShape() == llvm::ArrayRef<int64_t>({1022})) {
      ++reduce_1d_count;
    }
  });

  EXPECT_EQ(dot_count, 0);
  EXPECT_EQ(reduce_1d_count, 1);
}

TEST(PipelineTest, CanonicalizesUnitDimensionAddReduceWithoutContractions) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<16x1x64xf32>) -> tensor<16x64xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%x init: %zero) across dimensions = [1] : (tensor<16x1x64xf32>, tensor<f32>) -> tensor<16x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %sum : tensor<16x64xf32>
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

  int reduce_count = 0;
  int reshape_count = 0;
  module->walk([&reduce_count](mlir::stablehlo::ReduceOp) { ++reduce_count; });
  module->walk(
      [&reshape_count](mlir::stablehlo::ReshapeOp) { ++reshape_count; });

  EXPECT_EQ(reduce_count, 0);
  EXPECT_EQ(reshape_count, 1);
}

TEST(PipelineTest, AbsorbsBatchTransposeAcrossMulInContraction) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%g_in: tensor<1x4x64x128x128xf32>, %l_in: tensor<1x64x4x128x128xf32>, %h_in: tensor<1x4x128x64x64xf32>) -> tensor<1x4x128x64x64xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %g_tr = stablehlo.transpose %g_in, dims = [0, 1, 3, 4, 2] : (tensor<1x4x64x128x128xf32>) -> tensor<1x4x128x128x64xf32>
    %g_6d = stablehlo.reshape %g_tr : (tensor<1x4x128x128x64xf32>) -> tensor<1x4x128x128x64x1xf32>
    %l_r = stablehlo.reshape %l_in : (tensor<1x64x4x128x128xf32>) -> tensor<64x4x128x128x1x1xf32>
    %l_tr = stablehlo.transpose %l_r, dims = [4, 1, 2, 3, 0, 5] : (tensor<64x4x128x128x1x1xf32>) -> tensor<1x4x128x128x64x1xf32>
    %m_mul = stablehlo.multiply %g_6d, %l_tr : tensor<1x4x128x128x64x1xf32>
    %m_sum = stablehlo.reduce(%m_mul init: %zero) across dimensions = [5] : (tensor<1x4x128x128x64x1xf32>, tensor<f32>) -> tensor<1x4x128x128x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %m_b = stablehlo.broadcast_in_dim %m_sum, dims = [0, 1, 2, 3, 4] : (tensor<1x4x128x128x64xf32>) -> tensor<1x4x128x128x64x64xf32>
    %h_b = stablehlo.broadcast_in_dim %h_in, dims = [0, 1, 3, 4, 5] : (tensor<1x4x128x64x64xf32>) -> tensor<1x4x128x128x64x64xf32>
    %y_mul = stablehlo.multiply %m_b, %h_b : tensor<1x4x128x128x64x64xf32>
    %y_sum = stablehlo.reduce(%y_mul init: %zero) across dimensions = [3] : (tensor<1x4x128x128x64x64xf32>, tensor<f32>) -> tensor<1x4x128x64x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %y_sum : tensor<1x4x128x64x64xf32>
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

  int dot_count = 0;
  int reduce_count = 0;
  bool l_in_used_by_transpose = false;
  module->walk([&dot_count](mlir::stablehlo::DotGeneralOp) { ++dot_count; });
  module->walk([&reduce_count](mlir::stablehlo::ReduceOp) { ++reduce_count; });
  module->walk([&l_in_used_by_transpose](mlir::stablehlo::TransposeOp tr) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(tr.getOperand().getType());
    if (rtt.getShape() == llvm::ArrayRef<int64_t>({64, 4, 128, 128, 1, 1})) {
      l_in_used_by_transpose = true;
    }
  });

  EXPECT_EQ(dot_count, 1);
  EXPECT_EQ(reduce_count, 0);
  EXPECT_FALSE(l_in_used_by_transpose);
}

TEST(PipelineTest, FoldsUnitDotGeneralWithBooleanSplat) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%v: tensor<16x1x64xi1>) -> tensor<16x1x64xi1> {
    %one_i1 = stablehlo.constant dense<true> : tensor<16x1x1xi1>
    %attn_out = stablehlo.dot_general %one_i1, %v, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x1x1xi1>, tensor<16x1x64xi1>) -> tensor<16x1x64xi1>
    return %attn_out : tensor<16x1x64xi1>
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

  int dot_count = 0;
  int reshape_count = 0;
  module->walk([&dot_count](mlir::stablehlo::DotGeneralOp) { ++dot_count; });
  module->walk(
      [&reshape_count](mlir::stablehlo::ReshapeOp) { ++reshape_count; });

  EXPECT_EQ(dot_count, 0);
  EXPECT_EQ(reshape_count, 1);
}

}  // namespace
}  // namespace torch_tpu
