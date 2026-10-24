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

#include "csrc/mlir/passes/canonicalize_gqa_and_masked_online_softmax.h"

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOk;

class CanonicalizeGqaAndMaskedOnlineSoftmaxTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    registry.insert<mlir::func::FuncDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> ParseAndOptimize(const char* ir) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &context_);
    if (!module) {
      ADD_FAILURE() << "Failed to parse MLIR source string";
      return nullptr;
    }
    EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context_), IsOk());
    return module;
  }

  mlir::MLIRContext context_;
};

TEST_F(CanonicalizeGqaAndMaskedOnlineSoftmaxTest,
       NormalizesBoth5dGqaDotsTo4dAcrossHeadDims) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%q: tensor<1x24x2048x256xbf16>,
                      %k: tensor<1x4x2048x256xbf16>,
                      %v: tensor<1x4x2048x256xbf16>,
                      %causal_pred: tensor<1x24x2048x2048xi1>)
          -> tensor<1x24x2048x256xbf16> {
        %q5 = stablehlo.reshape %q
            : (tensor<1x24x2048x256xbf16>) -> tensor<1x4x6x2048x256xbf16>
        %logits5 = stablehlo.dot_general %q5, %k,
            batching_dims = [0, 1] x [0, 1],
            contracting_dims = [4] x [3]
            : (tensor<1x4x6x2048x256xbf16>, tensor<1x4x2048x256xbf16>)
              -> tensor<1x4x6x2048x2048xbf16>
        %logits4 = stablehlo.reshape %logits5
            : (tensor<1x4x6x2048x2048xbf16>) -> tensor<1x24x2048x2048xbf16>
        %min_val = stablehlo.constant dense<-3.389530e+38>
            : tensor<1x24x2048x2048xbf16>
        %masked = stablehlo.select %causal_pred, %min_val, %logits4
            : tensor<1x24x2048x2048xi1>, tensor<1x24x2048x2048xbf16>
        %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
        %max = stablehlo.reduce(%masked init: %neg_inf)
            applies stablehlo.maximum across dimensions = [3]
            : (tensor<1x24x2048x2048xbf16>, tensor<bf16>)
              -> tensor<1x24x2048xbf16>
        %bmax = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2]
            : (tensor<1x24x2048xbf16>) -> tensor<1x24x2048x2048xbf16>
        %sub = stablehlo.subtract %masked, %bmax
            : tensor<1x24x2048x2048xbf16>
        %sub_f32 = stablehlo.convert %sub
            : (tensor<1x24x2048x2048xbf16>) -> tensor<1x24x2048x2048xf32>
        %exp = stablehlo.exponential %sub_f32 : tensor<1x24x2048x2048xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %sum = stablehlo.reduce(%exp init: %zero)
            applies stablehlo.add across dimensions = [3]
            : (tensor<1x24x2048x2048xf32>, tensor<f32>)
              -> tensor<1x24x2048xf32>
        %bsum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2]
            : (tensor<1x24x2048xf32>) -> tensor<1x24x2048x2048xf32>
        %div = stablehlo.divide %exp, %bsum : tensor<1x24x2048x2048xf32>
        %probs = stablehlo.convert %div
            : (tensor<1x24x2048x2048xf32>) -> tensor<1x24x2048x2048xbf16>
        %probs5 = stablehlo.reshape %probs
            : (tensor<1x24x2048x2048xbf16>) -> tensor<1x4x6x2048x2048xbf16>
        %out5 = stablehlo.dot_general %probs5, %v,
            batching_dims = [0, 1] x [0, 1],
            contracting_dims = [4] x [2]
            : (tensor<1x4x6x2048x2048xbf16>, tensor<1x4x2048x256xbf16>)
              -> tensor<1x4x6x2048x256xbf16>
        %out4 = stablehlo.reshape %out5
            : (tensor<1x4x6x2048x256xbf16>) -> tensor<1x24x2048x256xbf16>
        return %out4 : tensor<1x24x2048x256xbf16>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());
  int dot_4d_count = 0;
  int dot_5d_count = 0;
  llvm::SmallVector<mlir::RankedTensorType> dot_4d_types;
  module->walk([&](mlir::stablehlo::DotGeneralOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getType());
    if (t.getRank() == 4) {
      ++dot_4d_count;
      dot_4d_types.push_back(t);
    } else if (t.getRank() == 5) {
      ++dot_5d_count;
    }
  });
  EXPECT_EQ(dot_4d_count, 2);
  EXPECT_EQ(dot_5d_count, 0);
  ASSERT_EQ(dot_4d_types.size(), 2);
  EXPECT_EQ(dot_4d_types[0].getShape(),
            llvm::ArrayRef<int64_t>({1, 24, 2048, 2048}));
  EXPECT_EQ(dot_4d_types[1].getShape(),
            llvm::ArrayRef<int64_t>({1, 24, 2048, 256}));
}

TEST_F(CanonicalizeGqaAndMaskedOnlineSoftmaxTest,
       FoldsPackedSequenceScanAndSlidingWindowRowMaskReduction) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%logits: tensor<2x32x1024x1024xbf16>,
                      %v: tensor<2x32x1024x256xbf16>)
          -> tensor<2x32x1024x256xbf16> {
        %iota = stablehlo.iota dim = 0 : tensor<1024xi64>
        %pos = stablehlo.broadcast_in_dim %iota, dims = [1]
            : (tensor<1024xi64>) -> tensor<2x1024xi64>
        %first = stablehlo.slice %iota [0:1]
            : (tensor<1024xi64>) -> tensor<1xi64>
        %first_b = stablehlo.broadcast_in_dim %first, dims = [1]
            : (tensor<1xi64>) -> tensor<2x1xi64>
        %c1_2_1 = stablehlo.constant dense<1> : tensor<2x1xi64>
        %dummy = stablehlo.subtract %first_b, %c1_2_1 : tensor<2x1xi64>
        %cat = stablehlo.concatenate %dummy, %pos, dim = 1
            : (tensor<2x1xi64>, tensor<2x1024xi64>) -> tensor<2x1025xi64>
        %prev = stablehlo.slice %cat [0:2, 0:1024]
            : (tensor<2x1025xi64>) -> tensor<2x1024xi64>
        %diff = stablehlo.subtract %pos, %prev : tensor<2x1024xi64>
        %c1 = stablehlo.constant dense<1> : tensor<2x1024xi64>
        %ne = stablehlo.compare NE, %diff, %c1 : (tensor<2x1024xi64>,
            tensor<2x1024xi64>) -> tensor<2x1024xi1>
        %ne_i64 = stablehlo.convert %ne
            : (tensor<2x1024xi1>) -> tensor<2x1024xi64>
        %init0 = stablehlo.constant dense<0> : tensor<2xi64>
        %scan:2 = "chlo.scan"(%ne_i64, %init0) ({
        ^bb0(%a0: tensor<2xi64>, %a1: tensor<2xi64>):
          %s = stablehlo.add %a0, %a1 : tensor<2xi64>
          stablehlo.return %s, %s : tensor<2xi64>, tensor<2xi64>
        }) {dimension = 1 : i64, is_associative = true,
            operandSegmentSizes = array<i32: 1, 1>,
            resultSegmentSizes = array<i32: 1, 1>}
            : (tensor<2x1024xi64>, tensor<2xi64>)
              -> (tensor<2x1024xi64>, tensor<2xi64>)
        %idx = stablehlo.constant dense<0> : tensor<2x1x1024x1x2xi64>
        %g_q = "stablehlo.gather"(%scan#0, %idx) {
          dimension_numbers = #stablehlo.gather<
            offset_dims = [],
            collapsed_slice_dims = [0, 1],
            start_index_map = [0, 1],
            index_vector_dim = 4>,
          slice_sizes = array<i64: 1, 1>
        } : (tensor<2x1024xi64>, tensor<2x1x1024x1x2xi64>)
            -> tensor<2x1x1024x1xi64>
        %g_q_b = stablehlo.broadcast_in_dim %g_q, dims = [0, 1, 2, 3]
            : (tensor<2x1x1024x1xi64>) -> tensor<2x1x1024x1024xi64>
        %g_k_b = stablehlo.broadcast_in_dim %g_q, dims = [0, 1, 3, 2]
            : (tensor<2x1x1024x1xi64>) -> tensor<2x1x1024x1024xi64>
        %eq = stablehlo.compare EQ, %g_q_b, %g_k_b
            : (tensor<2x1x1024x1024xi64>, tensor<2x1x1024x1024xi64>)
              -> tensor<2x1x1024x1024xi1>

        %q_idx = stablehlo.broadcast_in_dim %iota, dims = [2]
            : (tensor<1024xi64>) -> tensor<1x1x1024x1024xi64>
        %k_idx = stablehlo.broadcast_in_dim %iota, dims = [3]
            : (tensor<1024xi64>) -> tensor<1x1x1024x1024xi64>
        %win = stablehlo.constant dense<1024> : tensor<1x1x1024x1024xi64>
        %q_minus_w = stablehlo.subtract %q_idx, %win
            : tensor<1x1x1024x1024xi64>
        %gt = stablehlo.compare GT, %k_idx, %q_minus_w
            : (tensor<1x1x1024x1024xi64>, tensor<1x1x1024x1024xi64>)
              -> tensor<1x1x1024x1024xi1>
        %le = stablehlo.compare LE, %k_idx, %q_idx
            : (tensor<1x1x1024x1024xi64>, tensor<1x1x1024x1024xi64>)
              -> tensor<1x1x1024x1024xi1>
        %sw = stablehlo.and %gt, %le : tensor<1x1x1024x1024xi1>
        %sw_b = stablehlo.broadcast_in_dim %sw, dims = [0, 1, 2, 3]
            : (tensor<1x1x1024x1024xi1>) -> tensor<2x1x1024x1024xi1>
        %keep = stablehlo.and %sw_b, %eq : tensor<2x1x1024x1024xi1>
        %z_mask = stablehlo.constant dense<0.0> : tensor<2x1x1024x1024xbf16>
        %ninf_mask = stablehlo.constant dense<0xFF80>
            : tensor<2x1x1024x1024xbf16>
        %mask_1 = stablehlo.select %keep, %z_mask, %ninf_mask
            : tensor<2x1x1024x1024xi1>, tensor<2x1x1024x1024xbf16>
        %mask = stablehlo.broadcast_in_dim %mask_1, dims = [0, 1, 2, 3]
            : (tensor<2x1x1024x1024xbf16>) -> tensor<2x32x1024x1024xbf16>

        %ninf_4d = stablehlo.constant dense<0xFF80>
            : tensor<2x32x1024x1024xbf16>
        %is_ninf = stablehlo.compare EQ, %mask, %ninf_4d
            : (tensor<2x32x1024x1024xbf16>, tensor<2x32x1024x1024xbf16>)
              -> tensor<2x32x1024x1024xi1>
        %true_s = stablehlo.constant dense<true> : tensor<i1>
        %row_masked = stablehlo.reduce(%is_ninf init: %true_s)
            applies stablehlo.and across dimensions = [3]
            : (tensor<2x32x1024x1024xi1>, tensor<i1>)
              -> tensor<2x32x1024xi1>
        %row_masked_b = stablehlo.broadcast_in_dim %row_masked,
            dims = [0, 1, 2]
            : (tensor<2x32x1024xi1>) -> tensor<2x32x1024x1024xi1>

        %fmin = stablehlo.constant dense<-3.389530e+38>
            : tensor<2x32x1024x1024xbf16>
        %clamped = stablehlo.maximum %mask, %fmin
            : tensor<2x32x1024x1024xbf16>
        %masked_logits = stablehlo.add %logits, %clamped
            : tensor<2x32x1024x1024xbf16>
        %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
        %max = stablehlo.reduce(%masked_logits init: %neg_inf)
            applies stablehlo.maximum across dimensions = [3]
            : (tensor<2x32x1024x1024xbf16>, tensor<bf16>)
              -> tensor<2x32x1024xbf16>
        %bmax = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2]
            : (tensor<2x32x1024xbf16>) -> tensor<2x32x1024x1024xbf16>
        %sub = stablehlo.subtract %masked_logits, %bmax
            : tensor<2x32x1024x1024xbf16>
        %sub_f32 = stablehlo.convert %sub
            : (tensor<2x32x1024x1024xbf16>) -> tensor<2x32x1024x1024xf32>
        %exp = stablehlo.exponential %sub_f32 : tensor<2x32x1024x1024xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %sum = stablehlo.reduce(%exp init: %zero)
            applies stablehlo.add across dimensions = [3]
            : (tensor<2x32x1024x1024xf32>, tensor<f32>)
              -> tensor<2x32x1024xf32>
        %bsum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2]
            : (tensor<2x32x1024xf32>) -> tensor<2x32x1024x1024xf32>
        %div = stablehlo.divide %exp, %bsum : tensor<2x32x1024x1024xf32>
        %z_probs = stablehlo.constant dense<0.0> : tensor<2x32x1024x1024xf32>
        %sel = stablehlo.select %row_masked_b, %z_probs, %div
            : tensor<2x32x1024x1024xi1>, tensor<2x32x1024x1024xf32>
        %probs = stablehlo.convert %sel
            : (tensor<2x32x1024x1024xf32>) -> tensor<2x32x1024x1024xbf16>
        %out = stablehlo.dot_general %probs, %v,
            batching_dims = [0, 1] x [0, 1],
            contracting_dims = [3] x [2]
            : (tensor<2x32x1024x1024xbf16>, tensor<2x32x1024x256xbf16>)
              -> tensor<2x32x1024x256xbf16>
        return %out : tensor<2x32x1024x256xbf16>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());
  int scan_count = 0;
  int gather_count = 0;
  int reduce_count = 0;
  int select_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::chlo::ScanOp>(op)) {
      ++scan_count;
    } else if (mlir::isa<mlir::stablehlo::GatherOp>(op)) {
      ++gather_count;
    } else if (mlir::isa<mlir::stablehlo::ReduceOp>(op)) {
      ++reduce_count;
    } else if (mlir::isa<mlir::stablehlo::SelectOp>(op)) {
      ++select_count;
    }
  });
  EXPECT_EQ(scan_count, 0);
  EXPECT_EQ(gather_count, 0);
  EXPECT_EQ(reduce_count, 2);
  EXPECT_EQ(select_count, 1);
}

TEST_F(CanonicalizeGqaAndMaskedOnlineSoftmaxTest,
       PreservesMultiUseTrainingSumReduce) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%q: tensor<1x32x2048x128xbf16>,
                      %k: tensor<1x2x2048x128xbf16>,
                      %v: tensor<1x2x2048x128xbf16>)
          -> (tensor<1x32x2048x128xbf16>, tensor<1x32x2048xf32>) {
        %q5 = stablehlo.reshape %q
            : (tensor<1x32x2048x128xbf16>) -> tensor<1x2x16x2048x128xbf16>
        %logits5 = stablehlo.dot_general %q5, %k,
            batching_dims = [0, 1] x [0, 1],
            contracting_dims = [4] x [3]
            : (tensor<1x2x16x2048x128xbf16>, tensor<1x2x2048x128xbf16>)
              -> tensor<1x2x16x2048x2048xbf16>
        %logits4 = stablehlo.reshape %logits5
            : (tensor<1x2x16x2048x2048xbf16>) -> tensor<1x32x2048x2048xbf16>
        %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
        %max = stablehlo.reduce(%logits4 init: %neg_inf)
            applies stablehlo.maximum across dimensions = [3]
            : (tensor<1x32x2048x2048xbf16>, tensor<bf16>)
              -> tensor<1x32x2048xbf16>
        %bmax = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2]
            : (tensor<1x32x2048xbf16>) -> tensor<1x32x2048x2048xbf16>
        %sub = stablehlo.subtract %logits4, %bmax
            : tensor<1x32x2048x2048xbf16>
        %sub_f32 = stablehlo.convert %sub
            : (tensor<1x32x2048x2048xbf16>) -> tensor<1x32x2048x2048xf32>
        %exp = stablehlo.exponential %sub_f32 : tensor<1x32x2048x2048xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %sum = stablehlo.reduce(%exp init: %zero)
            applies stablehlo.add across dimensions = [3]
            : (tensor<1x32x2048x2048xf32>, tensor<f32>)
              -> tensor<1x32x2048xf32>
        %bsum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2]
            : (tensor<1x32x2048xf32>) -> tensor<1x32x2048x2048xf32>
        %div = stablehlo.divide %exp, %bsum : tensor<1x32x2048x2048xf32>
        %probs = stablehlo.convert %div
            : (tensor<1x32x2048x2048xf32>) -> tensor<1x32x2048x2048xbf16>
        %probs5 = stablehlo.reshape %probs
            : (tensor<1x32x2048x2048xbf16>) -> tensor<1x2x16x2048x2048xbf16>
        %out5 = stablehlo.dot_general %probs5, %v,
            batching_dims = [0, 1] x [0, 1],
            contracting_dims = [4] x [2]
            : (tensor<1x2x16x2048x2048xbf16>, tensor<1x2x2048x128xbf16>)
              -> tensor<1x2x16x2048x128xbf16>
        %out4 = stablehlo.reshape %out5
            : (tensor<1x2x16x2048x128xbf16>) -> tensor<1x32x2048x128xbf16>
        return %out4, %sum : tensor<1x32x2048x128xbf16>, tensor<1x32x2048xf32>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());
  int dot_5d_count = 0;
  module->walk([&](mlir::stablehlo::DotGeneralOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getType());
    if (t.getRank() == 5) {
      ++dot_5d_count;
    }
  });
  EXPECT_EQ(dot_5d_count, 2);
}

TEST_F(CanonicalizeGqaAndMaskedOnlineSoftmaxTest,
       SinksRowMaskSelectPastDotGeneralWhenNotProvablyFalse) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%logits: tensor<2x32x1024x1024xbf16>,
                      %v: tensor<2x32x1024x256xbf16>,
                      %is_masked: tensor<2x32x1024x1024xi1>)
          -> tensor<2x32x1024x256xbf16> {
        %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
        %max = stablehlo.reduce(%logits init: %neg_inf)
            applies stablehlo.maximum across dimensions = [3]
            : (tensor<2x32x1024x1024xbf16>, tensor<bf16>)
              -> tensor<2x32x1024xbf16>
        %bmax = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2]
            : (tensor<2x32x1024xbf16>) -> tensor<2x32x1024x1024xbf16>
        %sub = stablehlo.subtract %logits, %bmax
            : tensor<2x32x1024x1024xbf16>
        %sub_f32 = stablehlo.convert %sub
            : (tensor<2x32x1024x1024xbf16>) -> tensor<2x32x1024x1024xf32>
        %exp = stablehlo.exponential %sub_f32 : tensor<2x32x1024x1024xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %sum = stablehlo.reduce(%exp init: %zero)
            applies stablehlo.add across dimensions = [3]
            : (tensor<2x32x1024x1024xf32>, tensor<f32>)
              -> tensor<2x32x1024xf32>
        %bsum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2]
            : (tensor<2x32x1024xf32>) -> tensor<2x32x1024x1024xf32>
        %div = stablehlo.divide %exp, %bsum : tensor<2x32x1024x1024xf32>
        %true_s = stablehlo.constant dense<true> : tensor<i1>
        %row_masked = stablehlo.reduce(%is_masked init: %true_s)
            applies stablehlo.and across dimensions = [3]
            : (tensor<2x32x1024x1024xi1>, tensor<i1>)
              -> tensor<2x32x1024xi1>
        %row_masked_b = stablehlo.broadcast_in_dim %row_masked,
            dims = [0, 1, 2]
            : (tensor<2x32x1024xi1>) -> tensor<2x32x1024x1024xi1>
        %z_probs = stablehlo.constant dense<0.0> : tensor<2x32x1024x1024xf32>
        %sel = stablehlo.select %row_masked_b, %z_probs, %div
            : tensor<2x32x1024x1024xi1>, tensor<2x32x1024x1024xf32>
        %probs = stablehlo.convert %sel
            : (tensor<2x32x1024x1024xf32>) -> tensor<2x32x1024x1024xbf16>
        %out = stablehlo.dot_general %probs, %v,
            batching_dims = [0, 1] x [0, 1],
            contracting_dims = [3] x [2]
            : (tensor<2x32x1024x1024xbf16>, tensor<2x32x1024x256xbf16>)
              -> tensor<2x32x1024x256xbf16>
        return %out : tensor<2x32x1024x256xbf16>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());

  int select_count = 0;
  int dot_count = 0;
  mlir::stablehlo::SelectOp sunk_select;
  mlir::stablehlo::DotGeneralOp dot_op;
  module->walk([&](mlir::stablehlo::SelectOp op) {
    ++select_count;
    sunk_select = op;
  });
  module->walk([&](mlir::stablehlo::DotGeneralOp op) {
    ++dot_count;
    dot_op = op;
  });
  EXPECT_EQ(select_count, 1);
  EXPECT_EQ(dot_count, 1);
  ASSERT_TRUE(static_cast<bool>(sunk_select));
  ASSERT_TRUE(static_cast<bool>(dot_op));
  auto sel_type = mlir::cast<mlir::RankedTensorType>(sunk_select.getType());
  EXPECT_EQ(sel_type.getShape(), llvm::ArrayRef<int64_t>({2, 32, 1024, 256}));
  EXPECT_EQ(sunk_select.getOnFalse(), dot_op.getResult());
}

}  // namespace
}  // namespace torch_tpu
