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

TEST(PipelineTest, CanonicalizesUnmaskedSdpaAndBackwardSoftmax) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %q: tensor<1x4x64x32xbf16>,
      %k: tensor<1x4x64x32xbf16>,
      %v: tensor<1x4x64x32xbf16>,
      %do: tensor<1x4x64x32xbf16>) -> (tensor<1x4x64x32xbf16>, tensor<1x4x64x64xbf16>) {
    %min_bf16 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero_3d = stablehlo.constant dense<0.000000e+00> : tensor<1x4x64xf32>
    %one_3d = stablehlo.constant dense<1.000000e+00> : tensor<1x4x64xf32>
    %scores = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x64x32xbf16>, tensor<1x4x64x32xbf16>) -> tensor<1x4x64x64xbf16>
    %max = stablehlo.reduce(%scores init: %min_bf16) across dimensions = [3] : (tensor<1x4x64x64xbf16>, tensor<bf16>) -> tensor<1x4x64xbf16>
     reducer(%lhs: tensor<bf16>, %rhs: tensor<bf16>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<bf16>
      stablehlo.return %m : tensor<bf16>
    }
    %bcast_max = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2] : (tensor<1x4x64xbf16>) -> tensor<1x4x64x64xbf16>
    %shifted_bf16 = stablehlo.subtract %scores, %bcast_max : tensor<1x4x64x64xbf16>
    %shifted_f32 = stablehlo.convert %shifted_bf16 : (tensor<1x4x64x64xbf16>) -> tensor<1x4x64x64xf32>
    %exp = stablehlo.exponential %shifted_f32 : tensor<1x4x64x64xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<1x4x64x64xf32>, tensor<f32>) -> tensor<1x4x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_sum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x4x64xf32>) -> tensor<1x4x64x64xf32>
    %probs_fwd = stablehlo.divide %exp, %bcast_sum : tensor<1x4x64x64xf32>
    %probs_fwd_bf16 = stablehlo.convert %probs_fwd : (tensor<1x4x64x64xf32>) -> tensor<1x4x64x64xbf16>
    %out = stablehlo.dot_general %probs_fwd_bf16, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x64x64xbf16>, tensor<1x4x64x32xbf16>) -> tensor<1x4x64x32xbf16>
    %is_zero = stablehlo.compare EQ, %sum, %zero_3d : (tensor<1x4x64xf32>, tensor<1x4x64xf32>) -> tensor<1x4x64xi1>
    %safe_sum = stablehlo.select %is_zero, %one_3d, %sum : tensor<1x4x64xi1>, tensor<1x4x64xf32>
    %bcast_safe = stablehlo.broadcast_in_dim %safe_sum, dims = [0, 1, 2] : (tensor<1x4x64xf32>) -> tensor<1x4x64x64xf32>
    %probs_bwd = stablehlo.divide %exp, %bcast_safe : tensor<1x4x64x64xf32>
    %ds_bf16 = stablehlo.dot_general %do, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x64x32xbf16>, tensor<1x4x64x32xbf16>) -> tensor<1x4x64x64xbf16>
    %ds = stablehlo.convert %ds_bf16 : (tensor<1x4x64x64xbf16>) -> tensor<1x4x64x64xf32>
    %p_ds = stablehlo.multiply %probs_bwd, %ds : tensor<1x4x64x64xf32>
    %rowsum = stablehlo.reduce(%p_ds init: %zero_f32) across dimensions = [3] : (tensor<1x4x64x64xf32>, tensor<f32>) -> tensor<1x4x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_rowsum = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2] : (tensor<1x4x64xf32>) -> tensor<1x4x64x64xf32>
    %ds_shifted = stablehlo.subtract %ds, %bcast_rowsum : tensor<1x4x64x64xf32>
    %dp_f32 = stablehlo.multiply %probs_bwd, %ds_shifted : tensor<1x4x64x64xf32>
    %dp = stablehlo.convert %dp_f32 : (tensor<1x4x64x64xf32>) -> tensor<1x4x64x64xbf16>
    return %out, %dp : tensor<1x4x64x32xbf16>, tensor<1x4x64x64xbf16>
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

  int compare_count = 0;
  int select_count = 0;
  int subtract_count = 0;
  int div_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::stablehlo::CompareOp>(op)) {
      ++compare_count;
    }
    if (llvm::isa<mlir::stablehlo::SelectOp>(op)) {
      ++select_count;
    }
    if (llvm::isa<mlir::stablehlo::SubtractOp>(op)) {
      ++subtract_count;
    }
    if (llvm::isa<mlir::stablehlo::DivOp>(op)) {
      ++div_count;
    }
  });

  EXPECT_EQ(compare_count, 0);
  EXPECT_EQ(select_count, 0);
  EXPECT_EQ(subtract_count, 0);
  EXPECT_EQ(div_count, 1);
}

TEST(PipelineTest, DecouplesAlignedBackwardSdpaFromForwardOnlineSoftmax) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %q: tensor<1x2x128x64xbf16>,
      %k: tensor<1x2x128x64xbf16>,
      %v: tensor<1x2x128x64xbf16>,
      %do: tensor<1x2x128x64xbf16>) -> (tensor<1x2x128x64xbf16>, tensor<1x2x128x128xbf16>) {
    %min_bf16 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero_3d = stablehlo.constant dense<0.000000e+00> : tensor<1x2x128xf32>
    %one_3d = stablehlo.constant dense<1.000000e+00> : tensor<1x2x128xf32>
    %scores = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x128x64xbf16>, tensor<1x2x128x64xbf16>) -> tensor<1x2x128x128xbf16>
    %max = stablehlo.reduce(%scores init: %min_bf16) across dimensions = [3] : (tensor<1x2x128x128xbf16>, tensor<bf16>) -> tensor<1x2x128xbf16>
     reducer(%lhs: tensor<bf16>, %rhs: tensor<bf16>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<bf16>
      stablehlo.return %m : tensor<bf16>
    }
    %bcast_max = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2] : (tensor<1x2x128xbf16>) -> tensor<1x2x128x128xbf16>
    %shifted_bf16 = stablehlo.subtract %scores, %bcast_max : tensor<1x2x128x128xbf16>
    %shifted_f32 = stablehlo.convert %shifted_bf16 : (tensor<1x2x128x128xbf16>) -> tensor<1x2x128x128xf32>
    %exp = stablehlo.exponential %shifted_f32 : tensor<1x2x128x128xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<1x2x128x128xf32>, tensor<f32>) -> tensor<1x2x128xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_sum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x2x128xf32>) -> tensor<1x2x128x128xf32>
    %probs_fwd = stablehlo.divide %exp, %bcast_sum : tensor<1x2x128x128xf32>
    %probs_fwd_bf16 = stablehlo.convert %probs_fwd : (tensor<1x2x128x128xf32>) -> tensor<1x2x128x128xbf16>
    %out = stablehlo.dot_general %probs_fwd_bf16, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x2x128x128xbf16>, tensor<1x2x128x64xbf16>) -> tensor<1x2x128x64xbf16>
    %is_zero = stablehlo.compare EQ, %sum, %zero_3d : (tensor<1x2x128xf32>, tensor<1x2x128xf32>) -> tensor<1x2x128xi1>
    %safe_sum = stablehlo.select %is_zero, %one_3d, %sum : tensor<1x2x128xi1>, tensor<1x2x128xf32>
    %bcast_safe = stablehlo.broadcast_in_dim %safe_sum, dims = [0, 1, 2] : (tensor<1x2x128xf32>) -> tensor<1x2x128x128xf32>
    %probs_bwd = stablehlo.divide %exp, %bcast_safe : tensor<1x2x128x128xf32>
    %ds_bf16 = stablehlo.dot_general %do, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x128x64xbf16>, tensor<1x2x128x64xbf16>) -> tensor<1x2x128x128xbf16>
    %ds = stablehlo.convert %ds_bf16 : (tensor<1x2x128x128xbf16>) -> tensor<1x2x128x128xf32>
    %p_ds = stablehlo.multiply %probs_bwd, %ds : tensor<1x2x128x128xf32>
    %rowsum = stablehlo.reduce(%p_ds init: %zero_f32) across dimensions = [3] : (tensor<1x2x128x128xf32>, tensor<f32>) -> tensor<1x2x128xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_rowsum = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2] : (tensor<1x2x128xf32>) -> tensor<1x2x128x128xf32>
    %ds_shifted = stablehlo.subtract %ds, %bcast_rowsum : tensor<1x2x128x128xf32>
    %dp_f32 = stablehlo.multiply %probs_bwd, %ds_shifted : tensor<1x2x128x128xf32>
    %dp = stablehlo.convert %dp_f32 : (tensor<1x2x128x128xf32>) -> tensor<1x2x128x128xbf16>
    return %out, %dp : tensor<1x2x128x64xbf16>, tensor<1x2x128x128xbf16>
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

  int compare_count = 0;
  int select_count = 0;
  int subtract_count = 0;
  int neg_count = 0;
  int div_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::stablehlo::CompareOp>(op)) {
      ++compare_count;
    }
    if (llvm::isa<mlir::stablehlo::SelectOp>(op)) {
      ++select_count;
    }
    if (llvm::isa<mlir::stablehlo::SubtractOp>(op)) {
      ++subtract_count;
    }
    if (llvm::isa<mlir::stablehlo::NegOp>(op)) {
      ++neg_count;
    }
    if (llvm::isa<mlir::stablehlo::DivOp>(op)) {
      ++div_count;
    }
  });

  EXPECT_EQ(compare_count, 0);
  EXPECT_EQ(select_count, 0);
  EXPECT_EQ(subtract_count, 1);
  EXPECT_EQ(neg_count, 2);
  EXPECT_EQ(div_count, 2);
}

TEST(PipelineTest, CanonicalizesCausalSdpaAndBackwardSoftmax) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %q: tensor<1x4x64x32xbf16>,
      %k: tensor<1x4x64x32xbf16>,
      %v: tensor<1x4x64x32xbf16>,
      %do: tensor<1x4x64x32xbf16>,
      %mask: tensor<1x4x64x64xi1>) -> (tensor<1x4x64x32xbf16>, tensor<1x4x64x64xbf16>) {
    %min_bf16 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero_3d = stablehlo.constant dense<0.000000e+00> : tensor<1x4x64xf32>
    %one_3d = stablehlo.constant dense<1.000000e+00> : tensor<1x4x64xf32>
    %neg_inf_4d = stablehlo.constant dense<0xFF80> : tensor<1x4x64x64xbf16>
    %scores = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x64x32xbf16>, tensor<1x4x64x32xbf16>) -> tensor<1x4x64x64xbf16>
    %masked_scores = stablehlo.select %mask, %scores, %neg_inf_4d : tensor<1x4x64x64xi1>, tensor<1x4x64x64xbf16>
    %max = stablehlo.reduce(%masked_scores init: %min_bf16) across dimensions = [3] : (tensor<1x4x64x64xbf16>, tensor<bf16>) -> tensor<1x4x64xbf16>
     reducer(%lhs: tensor<bf16>, %rhs: tensor<bf16>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<bf16>
      stablehlo.return %m : tensor<bf16>
    }
    %bcast_max = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2] : (tensor<1x4x64xbf16>) -> tensor<1x4x64x64xbf16>
    %shifted_bf16 = stablehlo.subtract %masked_scores, %bcast_max : tensor<1x4x64x64xbf16>
    %shifted_f32 = stablehlo.convert %shifted_bf16 : (tensor<1x4x64x64xbf16>) -> tensor<1x4x64x64xf32>
    %exp = stablehlo.exponential %shifted_f32 : tensor<1x4x64x64xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<1x4x64x64xf32>, tensor<f32>) -> tensor<1x4x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_sum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x4x64xf32>) -> tensor<1x4x64x64xf32>
    %probs_fwd = stablehlo.divide %exp, %bcast_sum : tensor<1x4x64x64xf32>
    %probs_fwd_bf16 = stablehlo.convert %probs_fwd : (tensor<1x4x64x64xf32>) -> tensor<1x4x64x64xbf16>
    %out = stablehlo.dot_general %probs_fwd_bf16, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x64x64xbf16>, tensor<1x4x64x32xbf16>) -> tensor<1x4x64x32xbf16>
    %is_zero = stablehlo.compare EQ, %sum, %zero_3d : (tensor<1x4x64xf32>, tensor<1x4x64xf32>) -> tensor<1x4x64xi1>
    %safe_sum = stablehlo.select %is_zero, %one_3d, %sum : tensor<1x4x64xi1>, tensor<1x4x64xf32>
    %bcast_safe = stablehlo.broadcast_in_dim %safe_sum, dims = [0, 1, 2] : (tensor<1x4x64xf32>) -> tensor<1x4x64x64xf32>
    %probs_bwd = stablehlo.divide %exp, %bcast_safe : tensor<1x4x64x64xf32>
    %ds_bf16 = stablehlo.dot_general %do, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x64x32xbf16>, tensor<1x4x64x32xbf16>) -> tensor<1x4x64x64xbf16>
    %ds = stablehlo.convert %ds_bf16 : (tensor<1x4x64x64xbf16>) -> tensor<1x4x64x64xf32>
    %p_ds = stablehlo.multiply %probs_bwd, %ds : tensor<1x4x64x64xf32>
    %rowsum = stablehlo.reduce(%p_ds init: %zero_f32) across dimensions = [3] : (tensor<1x4x64x64xf32>, tensor<f32>) -> tensor<1x4x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_rowsum = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2] : (tensor<1x4x64xf32>) -> tensor<1x4x64x64xf32>
    %ds_shifted = stablehlo.subtract %ds, %bcast_rowsum : tensor<1x4x64x64xf32>
    %dp_f32 = stablehlo.multiply %probs_bwd, %ds_shifted : tensor<1x4x64x64xf32>
    %dp = stablehlo.convert %dp_f32 : (tensor<1x4x64x64xf32>) -> tensor<1x4x64x64xbf16>
    return %out, %dp : tensor<1x4x64x32xbf16>, tensor<1x4x64x64xbf16>
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

  int compare_count = 0;
  int select_count = 0;
  int subtract_count = 0;
  int div_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::stablehlo::CompareOp>(op)) {
      ++compare_count;
    }
    if (llvm::isa<mlir::stablehlo::SelectOp>(op)) {
      ++select_count;
    }
    if (llvm::isa<mlir::stablehlo::SubtractOp>(op)) {
      ++subtract_count;
    }
    if (llvm::isa<mlir::stablehlo::DivOp>(op)) {
      ++div_count;
    }
  });

  EXPECT_EQ(compare_count, 0);
  EXPECT_EQ(select_count, 1);
  EXPECT_EQ(subtract_count, 0);
  EXPECT_EQ(div_count, 1);
}

TEST(PipelineTest, CanonicalizesUnalignedRectangularCrossAttention) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %q: tensor<1x2x64x32xbf16>,
      %k: tensor<1x2x128x32xbf16>,
      %v: tensor<1x2x128x32xbf16>,
      %do: tensor<1x2x64x32xbf16>) -> (tensor<1x2x64x32xbf16>, tensor<1x2x64x128xbf16>) {
    %min_bf16 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero_3d = stablehlo.constant dense<0.000000e+00> : tensor<1x2x64xf32>
    %one_3d = stablehlo.constant dense<1.000000e+00> : tensor<1x2x64xf32>
    %scores = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x64x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x64x128xbf16>
    %max = stablehlo.reduce(%scores init: %min_bf16) across dimensions = [3] : (tensor<1x2x64x128xbf16>, tensor<bf16>) -> tensor<1x2x64xbf16>
     reducer(%lhs: tensor<bf16>, %rhs: tensor<bf16>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<bf16>
      stablehlo.return %m : tensor<bf16>
    }
    %bcast_max = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2] : (tensor<1x2x64xbf16>) -> tensor<1x2x64x128xbf16>
    %shifted_bf16 = stablehlo.subtract %scores, %bcast_max : tensor<1x2x64x128xbf16>
    %shifted_f32 = stablehlo.convert %shifted_bf16 : (tensor<1x2x64x128xbf16>) -> tensor<1x2x64x128xf32>
    %exp = stablehlo.exponential %shifted_f32 : tensor<1x2x64x128xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<1x2x64x128xf32>, tensor<f32>) -> tensor<1x2x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_sum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x2x64xf32>) -> tensor<1x2x64x128xf32>
    %probs_fwd = stablehlo.divide %exp, %bcast_sum : tensor<1x2x64x128xf32>
    %probs_fwd_bf16 = stablehlo.convert %probs_fwd : (tensor<1x2x64x128xf32>) -> tensor<1x2x64x128xbf16>
    %out = stablehlo.dot_general %probs_fwd_bf16, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x2x64x128xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x64x32xbf16>
    %is_zero = stablehlo.compare EQ, %sum, %zero_3d : (tensor<1x2x64xf32>, tensor<1x2x64xf32>) -> tensor<1x2x64xi1>
    %safe_sum = stablehlo.select %is_zero, %one_3d, %sum : tensor<1x2x64xi1>, tensor<1x2x64xf32>
    %bcast_safe = stablehlo.broadcast_in_dim %safe_sum, dims = [0, 1, 2] : (tensor<1x2x64xf32>) -> tensor<1x2x64x128xf32>
    %probs_bwd = stablehlo.divide %exp, %bcast_safe : tensor<1x2x64x128xf32>
    %ds_bf16 = stablehlo.dot_general %do, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x64x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x64x128xbf16>
    %ds = stablehlo.convert %ds_bf16 : (tensor<1x2x64x128xbf16>) -> tensor<1x2x64x128xf32>
    %p_ds = stablehlo.multiply %probs_bwd, %ds : tensor<1x2x64x128xf32>
    %rowsum = stablehlo.reduce(%p_ds init: %zero_f32) across dimensions = [3] : (tensor<1x2x64x128xf32>, tensor<f32>) -> tensor<1x2x64xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_rowsum = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2] : (tensor<1x2x64xf32>) -> tensor<1x2x64x128xf32>
    %ds_shifted = stablehlo.subtract %ds, %bcast_rowsum : tensor<1x2x64x128xf32>
    %dp_f32 = stablehlo.multiply %probs_bwd, %ds_shifted : tensor<1x2x64x128xf32>
    %dp = stablehlo.convert %dp_f32 : (tensor<1x2x64x128xf32>) -> tensor<1x2x64x128xbf16>
    return %out, %dp : tensor<1x2x64x32xbf16>, tensor<1x2x64x128xbf16>
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

  int compare_count = 0;
  int select_count = 0;
  int subtract_count = 0;
  int div_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::stablehlo::CompareOp>(op)) {
      ++compare_count;
    }
    if (llvm::isa<mlir::stablehlo::SelectOp>(op)) {
      ++select_count;
    }
    if (llvm::isa<mlir::stablehlo::SubtractOp>(op)) {
      ++subtract_count;
    }
    if (llvm::isa<mlir::stablehlo::DivOp>(op)) {
      ++div_count;
    }
  });

  EXPECT_EQ(compare_count, 0);
  EXPECT_EQ(select_count, 0);
  EXPECT_EQ(subtract_count, 0);
  EXPECT_EQ(div_count, 1);
}

TEST(PipelineTest, DecouplesAlignedRectangularCrossAttention) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %q: tensor<1x2x128x64xbf16>,
      %k: tensor<1x2x256x64xbf16>,
      %v: tensor<1x2x256x64xbf16>,
      %do: tensor<1x2x128x64xbf16>) -> (tensor<1x2x128x64xbf16>, tensor<1x2x128x256xbf16>) {
    %min_bf16 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero_3d = stablehlo.constant dense<0.000000e+00> : tensor<1x2x128xf32>
    %one_3d = stablehlo.constant dense<1.000000e+00> : tensor<1x2x128xf32>
    %scores = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x128x64xbf16>, tensor<1x2x256x64xbf16>) -> tensor<1x2x128x256xbf16>
    %max = stablehlo.reduce(%scores init: %min_bf16) across dimensions = [3] : (tensor<1x2x128x256xbf16>, tensor<bf16>) -> tensor<1x2x128xbf16>
     reducer(%lhs: tensor<bf16>, %rhs: tensor<bf16>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<bf16>
      stablehlo.return %m : tensor<bf16>
    }
    %bcast_max = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2] : (tensor<1x2x128xbf16>) -> tensor<1x2x128x256xbf16>
    %shifted_bf16 = stablehlo.subtract %scores, %bcast_max : tensor<1x2x128x256xbf16>
    %shifted_f32 = stablehlo.convert %shifted_bf16 : (tensor<1x2x128x256xbf16>) -> tensor<1x2x128x256xf32>
    %exp = stablehlo.exponential %shifted_f32 : tensor<1x2x128x256xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<1x2x128x256xf32>, tensor<f32>) -> tensor<1x2x128xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_sum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x2x128xf32>) -> tensor<1x2x128x256xf32>
    %probs_fwd = stablehlo.divide %exp, %bcast_sum : tensor<1x2x128x256xf32>
    %probs_fwd_bf16 = stablehlo.convert %probs_fwd : (tensor<1x2x128x256xf32>) -> tensor<1x2x128x256xbf16>
    %out = stablehlo.dot_general %probs_fwd_bf16, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x2x128x256xbf16>, tensor<1x2x256x64xbf16>) -> tensor<1x2x128x64xbf16>
    %is_zero = stablehlo.compare EQ, %sum, %zero_3d : (tensor<1x2x128xf32>, tensor<1x2x128xf32>) -> tensor<1x2x128xi1>
    %safe_sum = stablehlo.select %is_zero, %one_3d, %sum : tensor<1x2x128xi1>, tensor<1x2x128xf32>
    %bcast_safe = stablehlo.broadcast_in_dim %safe_sum, dims = [0, 1, 2] : (tensor<1x2x128xf32>) -> tensor<1x2x128x256xf32>
    %probs_bwd = stablehlo.divide %exp, %bcast_safe : tensor<1x2x128x256xf32>
    %ds_bf16 = stablehlo.dot_general %do, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x128x64xbf16>, tensor<1x2x256x64xbf16>) -> tensor<1x2x128x256xbf16>
    %ds = stablehlo.convert %ds_bf16 : (tensor<1x2x128x256xbf16>) -> tensor<1x2x128x256xf32>
    %p_ds = stablehlo.multiply %probs_bwd, %ds : tensor<1x2x128x256xf32>
    %rowsum = stablehlo.reduce(%p_ds init: %zero_f32) across dimensions = [3] : (tensor<1x2x128x256xf32>, tensor<f32>) -> tensor<1x2x128xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_rowsum = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2] : (tensor<1x2x128xf32>) -> tensor<1x2x128x256xf32>
    %ds_shifted = stablehlo.subtract %ds, %bcast_rowsum : tensor<1x2x128x256xf32>
    %dp_f32 = stablehlo.multiply %probs_bwd, %ds_shifted : tensor<1x2x128x256xf32>
    %dp = stablehlo.convert %dp_f32 : (tensor<1x2x128x256xf32>) -> tensor<1x2x128x256xbf16>
    return %out, %dp : tensor<1x2x128x64xbf16>, tensor<1x2x128x256xbf16>
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

  int compare_count = 0;
  int select_count = 0;
  int subtract_count = 0;
  int neg_count = 0;
  int div_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::stablehlo::CompareOp>(op)) {
      ++compare_count;
    }
    if (llvm::isa<mlir::stablehlo::SelectOp>(op)) {
      ++select_count;
    }
    if (llvm::isa<mlir::stablehlo::SubtractOp>(op)) {
      ++subtract_count;
    }
    if (llvm::isa<mlir::stablehlo::NegOp>(op)) {
      ++neg_count;
    }
    if (llvm::isa<mlir::stablehlo::DivOp>(op)) {
      ++div_count;
    }
  });

  EXPECT_EQ(compare_count, 0);
  EXPECT_EQ(select_count, 0);
  EXPECT_EQ(subtract_count, 1);
  EXPECT_EQ(neg_count, 2);
  EXPECT_EQ(div_count, 2);
}

}  // namespace
}  // namespace torch_tpu
