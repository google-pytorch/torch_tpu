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

#include "csrc/mlir/passes/canonicalize_rect_sdpa_and_shift_invariant_softmax.h"

#include <cstdint>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOk;

class CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxTest
    : public testing::Test {
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

TEST_F(CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxTest,
       CanonicalizesRectangularLaneAlignedKeySdpaForwardAndBackward) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%logits: tensor<1x24x64x512xbf16>,
                      %dp: tensor<1x24x64x512xf32>)
          -> (tensor<1x24x64x512xbf16>, tensor<1x24x64x512xbf16>) {
        %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
        %max = stablehlo.reduce(%logits init: %neg_inf)
            applies stablehlo.maximum across dimensions = [3]
            : (tensor<1x24x64x512xbf16>, tensor<bf16>)
              -> tensor<1x24x64xbf16>
        %bmax = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2]
            : (tensor<1x24x64xbf16>) -> tensor<1x24x64x512xbf16>
        %sub = stablehlo.subtract %logits, %bmax : tensor<1x24x64x512xbf16>
        %sub_f32 = stablehlo.convert %sub
            : (tensor<1x24x64x512xbf16>) -> tensor<1x24x64x512xf32>
        %exp = stablehlo.exponential %sub_f32 : tensor<1x24x64x512xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %sum = stablehlo.reduce(%exp init: %zero)
            applies stablehlo.add across dimensions = [3]
            : (tensor<1x24x64x512xf32>, tensor<f32>) -> tensor<1x24x64xf32>
        %bsum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2]
            : (tensor<1x24x64xf32>) -> tensor<1x24x64x512xf32>
        %probs_f32 = stablehlo.divide %exp, %bsum : tensor<1x24x64x512xf32>
        %probs = stablehlo.convert %probs_f32
            : (tensor<1x24x64x512xf32>) -> tensor<1x24x64x512xbf16>

        %z_3d = stablehlo.constant dense<0.0> : tensor<1x24x64xf32>
        %o_3d = stablehlo.constant dense<1.0> : tensor<1x24x64xf32>
        %is_z = stablehlo.compare EQ, %sum, %z_3d
            : (tensor<1x24x64xf32>, tensor<1x24x64xf32>) -> tensor<1x24x64xi1>
        %safe_sum = stablehlo.select %is_z, %o_3d, %sum
            : tensor<1x24x64xi1>, tensor<1x24x64xf32>
        %bsafe = stablehlo.broadcast_in_dim %safe_sum, dims = [0, 1, 2]
            : (tensor<1x24x64xf32>) -> tensor<1x24x64x512xf32>
        %bwd_probs = stablehlo.divide %exp, %bsafe : tensor<1x24x64x512xf32>
        %prod = stablehlo.multiply %bwd_probs, %dp : tensor<1x24x64x512xf32>
        %rowsum = stablehlo.reduce(%prod init: %zero)
            applies stablehlo.add across dimensions = [3]
            : (tensor<1x24x64x512xf32>, tensor<f32>) -> tensor<1x24x64xf32>
        %browsum = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2]
            : (tensor<1x24x64xf32>) -> tensor<1x24x64x512xf32>
        %dp_sub = stablehlo.subtract %dp, %browsum : tensor<1x24x64x512xf32>
        %ds_f32 = stablehlo.multiply %bwd_probs, %dp_sub
            : tensor<1x24x64x512xf32>
        %ds = stablehlo.convert %ds_f32
            : (tensor<1x24x64x512xf32>) -> tensor<1x24x64x512xbf16>
        return %probs, %ds : tensor<1x24x64x512xbf16>, tensor<1x24x64x512xbf16>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());

  int sub_count = 0;
  int div_4d_count = 0;
  int div_3d_count = 0;
  int neg_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::SubtractOp>(op)) {
      ++sub_count;
    } else if (auto div = mlir::dyn_cast<mlir::stablehlo::DivOp>(op)) {
      auto t = mlir::cast<mlir::RankedTensorType>(div.getType());
      if (t.getRank() == 4) {
        ++div_4d_count;
      } else if (t.getRank() == 3) {
        ++div_3d_count;
      }
    } else if (mlir::isa<mlir::stablehlo::NegOp>(op)) {
      ++neg_count;
    }
  });
  EXPECT_EQ(sub_count, 0);
  EXPECT_EQ(div_4d_count, 0);
  EXPECT_EQ(div_3d_count, 2);
  EXPECT_EQ(neg_count, 2);
}

TEST_F(CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxTest,
       EliminatesShiftInvariantPreSoftmaxAndZeroSumArgmaxScatter) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%x: tensor<1x64x1024x1025xbf16>,
                      %dp_bf16: tensor<1x64x1024x1025xbf16>)
          -> (tensor<1x64x1024x1025xbf16>, tensor<1x64x1024x1025xbf16>) {
        %iota = stablehlo.iota dim = 3 : tensor<1x64x1024x1025xi64>
        %ninf_bf16 = stablehlo.constant dense<-3.389530e+38> : tensor<bf16>
        %z_i64 = stablehlo.constant dense<0> : tensor<i64>
        %max_pair:2 = "stablehlo.reduce"(%x, %iota, %ninf_bf16, %z_i64) ({
        ^bb0(%v0: tensor<bf16>, %i0: tensor<i64>,
             %v1: tensor<bf16>, %i1: tensor<i64>):
          %m = stablehlo.maximum %v0, %v1 : tensor<bf16>
          %ge = stablehlo.compare GE, %v0, %v1
              : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
          %idx = stablehlo.select %ge, %i0, %i1 : tensor<i1>, tensor<i64>
          stablehlo.return %m, %idx : tensor<bf16>, tensor<i64>
        }) {dimensions = array<i64: 3>}
            : (tensor<1x64x1024x1025xbf16>, tensor<1x64x1024x1025xi64>,
               tensor<bf16>, tensor<i64>)
              -> (tensor<1x64x1024xbf16>, tensor<1x64x1024xi64>)
        %bmax = stablehlo.broadcast_in_dim %max_pair#0, dims = [0, 1, 2]
            : (tensor<1x64x1024xbf16>) -> tensor<1x64x1024x1025xbf16>
        %y_bf16 = stablehlo.subtract %x, %bmax : tensor<1x64x1024x1025xbf16>
        %y_f32 = stablehlo.convert %y_bf16
            : (tensor<1x64x1024x1025xbf16>) -> tensor<1x64x1024x1025xf32>
        %ninf_f32 = stablehlo.constant dense<-3.40282347e+38> : tensor<f32>
        %max2 = stablehlo.reduce(%y_f32 init: %ninf_f32)
            applies stablehlo.maximum across dimensions = [3]
            : (tensor<1x64x1024x1025xf32>, tensor<f32>)
              -> tensor<1x64x1024xf32>
        %bmax2 = stablehlo.broadcast_in_dim %max2, dims = [0, 1, 2]
            : (tensor<1x64x1024xf32>) -> tensor<1x64x1024x1025xf32>
        %shifted = stablehlo.subtract %y_f32, %bmax2
            : tensor<1x64x1024x1025xf32>
        %exp = stablehlo.exponential %shifted : tensor<1x64x1024x1025xf32>
        %zero_f32 = stablehlo.constant dense<0.0> : tensor<f32>
        %sum = stablehlo.reduce(%exp init: %zero_f32)
            applies stablehlo.add across dimensions = [3]
            : (tensor<1x64x1024x1025xf32>, tensor<f32>)
              -> tensor<1x64x1024xf32>
        %sum_4d = stablehlo.reshape %sum
            : (tensor<1x64x1024xf32>) -> tensor<1x64x1024x1xf32>
        %one_4d = stablehlo.constant dense<1.0> : tensor<1x64x1024x1xf32>
        %inv_4d = stablehlo.divide %one_4d, %sum_4d : tensor<1x64x1024x1xf32>
        %inv_3d = stablehlo.reshape %inv_4d
            : (tensor<1x64x1024x1xf32>) -> tensor<1x64x1024xf32>
        %binv = stablehlo.broadcast_in_dim %inv_3d, dims = [0, 1, 2]
            : (tensor<1x64x1024xf32>) -> tensor<1x64x1024x1025xf32>
        %probs_f32 = stablehlo.multiply %exp, %binv
            : tensor<1x64x1024x1025xf32>
        %probs = stablehlo.convert %probs_f32
            : (tensor<1x64x1024x1025xf32>) -> tensor<1x64x1024x1025xbf16>

        %probs_bwd = stablehlo.convert %probs
            : (tensor<1x64x1024x1025xbf16>) -> tensor<1x64x1024x1025xf32>
        %dp = stablehlo.convert %dp_bf16
            : (tensor<1x64x1024x1025xbf16>) -> tensor<1x64x1024x1025xf32>
        %rowsum = stablehlo.dot_general %probs_bwd, %dp,
            batching_dims = [0, 1, 2] x [0, 1, 2],
            contracting_dims = [3] x [3]
            : (tensor<1x64x1024x1025xf32>, tensor<1x64x1024x1025xf32>)
              -> tensor<1x64x1024xf32>
        %browsum = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2]
            : (tensor<1x64x1024xf32>) -> tensor<1x64x1024x1025xf32>
        %dp_sub = stablehlo.subtract %dp, %browsum
            : tensor<1x64x1024x1025xf32>
        %ds_f32 = stablehlo.multiply %probs_bwd, %dp_sub
            : tensor<1x64x1024x1025xf32>
        %ds = stablehlo.convert %ds_f32
            : (tensor<1x64x1024x1025xf32>) -> tensor<1x64x1024x1025xbf16>

        %idx_5d = stablehlo.reshape %max_pair#1
            : (tensor<1x64x1024xi64>) -> tensor<1x64x1024x1x1xi64>
        %neg_ds = stablehlo.negate %ds : tensor<1x64x1024x1025xbf16>
        %z_bf16 = stablehlo.constant dense<0.0> : tensor<bf16>
        %sum_neg_ds = stablehlo.reduce(%neg_ds init: %z_bf16)
            applies stablehlo.add across dimensions = [3]
            : (tensor<1x64x1024x1025xbf16>, tensor<bf16>)
              -> tensor<1x64x1024xbf16>
        %upd = stablehlo.reshape %sum_neg_ds
            : (tensor<1x64x1024xbf16>) -> tensor<1x64x1024x1xbf16>
        %zeros_4d = stablehlo.constant dense<0.0>
            : tensor<1x64x1024x1025xbf16>
        %scat = "stablehlo.scatter"(%zeros_4d, %idx_5d, %upd) ({
        ^bb0(%a0: tensor<bf16>, %a1: tensor<bf16>):
          %s = stablehlo.add %a0, %a1 : tensor<bf16>
          stablehlo.return %s : tensor<bf16>
        }) {
          scatter_dimension_numbers = #stablehlo.scatter<
            update_window_dims = [],
            inserted_window_dims = [3],
            input_batching_dims = [0, 1, 2],
            scatter_indices_batching_dims = [0, 1, 2],
            scatter_dims_to_operand_dims = [3],
            index_vector_dim = 4>
        } : (tensor<1x64x1024x1025xbf16>, tensor<1x64x1024x1x1xi64>,
             tensor<1x64x1024x1xbf16>) -> tensor<1x64x1024x1025xbf16>
        %dx = stablehlo.add %ds, %scat : tensor<1x64x1024x1025xbf16>
        return %probs, %dx
            : tensor<1x64x1024x1025xbf16>, tensor<1x64x1024x1025xbf16>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());

  int scatter_count = 0;
  int iota_count = 0;
  int reduce_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ScatterOp>(op)) {
      ++scatter_count;
    } else if (mlir::isa<mlir::stablehlo::IotaOp>(op)) {
      ++iota_count;
    } else if (mlir::isa<mlir::stablehlo::ReduceOp>(op)) {
      ++reduce_count;
    }
  });
  EXPECT_EQ(scatter_count, 0);
  EXPECT_EQ(iota_count, 0);
  EXPECT_EQ(reduce_count, 2);
}

TEST_F(CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxTest,
       FoldsStandaloneRowAllMaskedReductionWithReflexiveGatherEquality) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%cumsum: tensor<2x1024xi64>,
                      %probs: tensor<2x32x1024x1024xf32>)
          -> tensor<2x32x1024x1024xf32> {
        %b_iota = stablehlo.iota dim = 0 : tensor<2x1x1x1xi64>
        %z_b = stablehlo.constant dense<0> : tensor<2x1x1x1xi64>
        %two_b = stablehlo.constant dense<2> : tensor<2x1x1x1xi64>
        %lt_b = stablehlo.compare LT, %b_iota, %z_b
            : (tensor<2x1x1x1xi64>, tensor<2x1x1x1xi64>) -> tensor<2x1x1x1xi1>
        %add_b = stablehlo.add %b_iota, %two_b : tensor<2x1x1x1xi64>
        %sel_b = stablehlo.select %lt_b, %add_b, %b_iota
            : tensor<2x1x1x1xi1>, tensor<2x1x1x1xi64>

        %b_q = stablehlo.broadcast_in_dim %sel_b, dims = [0, 1, 2, 3]
            : (tensor<2x1x1x1xi64>) -> tensor<2x1x1024x1xi64>
        %b_q_5d = stablehlo.reshape %b_q
            : (tensor<2x1x1024x1xi64>) -> tensor<2x1x1024x1x1xi64>

        %q_iota = stablehlo.iota dim = 2 : tensor<1x1x1024x1xi64>
        %z_q = stablehlo.constant dense<0> : tensor<1x1x1024x1xi64>
        %n_q = stablehlo.constant dense<1024> : tensor<1x1x1024x1xi64>
        %lt_q = stablehlo.compare LT, %q_iota, %z_q
            : (tensor<1x1x1024x1xi64>, tensor<1x1x1024x1xi64>)
              -> tensor<1x1x1024x1xi1>
        %add_q = stablehlo.add %q_iota, %n_q : tensor<1x1x1024x1xi64>
        %sel_q = stablehlo.select %lt_q, %add_q, %q_iota
            : tensor<1x1x1024x1xi1>, tensor<1x1x1024x1xi64>
        %q_b = stablehlo.broadcast_in_dim %sel_q, dims = [0, 1, 2, 3]
            : (tensor<1x1x1024x1xi64>) -> tensor<2x1x1024x1xi64>
        %q_5d = stablehlo.reshape %q_b
            : (tensor<2x1x1024x1xi64>) -> tensor<2x1x1024x1x1xi64>
        %idx_q = stablehlo.concatenate %b_q_5d, %q_5d, dim = 4
            : (tensor<2x1x1024x1x1xi64>, tensor<2x1x1024x1x1xi64>)
              -> tensor<2x1x1024x1x2xi64>
        %g_q = "stablehlo.gather"(%cumsum, %idx_q) {
          dimension_numbers = #stablehlo.gather<
            offset_dims = [],
            collapsed_slice_dims = [0, 1],
            start_index_map = [0, 1],
            index_vector_dim = 4>,
          slice_sizes = array<i64: 1, 1>
        } : (tensor<2x1024xi64>, tensor<2x1x1024x1x2xi64>)
            -> tensor<2x1x1024x1xi64>
        %g_q_4d = stablehlo.broadcast_in_dim %g_q, dims = [0, 1, 2, 3]
            : (tensor<2x1x1024x1xi64>) -> tensor<2x1x1024x1024xi64>

        %b_k = stablehlo.broadcast_in_dim %sel_b, dims = [0, 1, 2, 3]
            : (tensor<2x1x1x1xi64>) -> tensor<2x1x1x1024xi64>
        %b_k_5d = stablehlo.reshape %b_k
            : (tensor<2x1x1x1024xi64>) -> tensor<2x1x1x1024x1xi64>
        %k_iota = stablehlo.iota dim = 3 : tensor<1x1x1x1024xi64>
        %z_k = stablehlo.constant dense<0> : tensor<1x1x1x1024xi64>
        %n_k = stablehlo.constant dense<1024> : tensor<1x1x1x1024xi64>
        %lt_k = stablehlo.compare LT, %k_iota, %z_k
            : (tensor<1x1x1x1024xi64>, tensor<1x1x1x1024xi64>)
              -> tensor<1x1x1x1024xi1>
        %add_k = stablehlo.add %k_iota, %n_k : tensor<1x1x1x1024xi64>
        %sel_k = stablehlo.select %lt_k, %add_k, %k_iota
            : tensor<1x1x1x1024xi1>, tensor<1x1x1x1024xi64>
        %k_b = stablehlo.broadcast_in_dim %sel_k, dims = [0, 1, 2, 3]
            : (tensor<1x1x1x1024xi64>) -> tensor<2x1x1x1024xi64>
        %k_5d = stablehlo.reshape %k_b
            : (tensor<2x1x1x1024xi64>) -> tensor<2x1x1x1024x1xi64>
        %idx_k = stablehlo.concatenate %b_k_5d, %k_5d, dim = 4
            : (tensor<2x1x1x1024x1xi64>, tensor<2x1x1x1024x1xi64>)
              -> tensor<2x1x1x1024x2xi64>
        %g_k = "stablehlo.gather"(%cumsum, %idx_k) {
          dimension_numbers = #stablehlo.gather<
            offset_dims = [],
            collapsed_slice_dims = [0, 1],
            start_index_map = [0, 1],
            index_vector_dim = 4>,
          slice_sizes = array<i64: 1, 1>
        } : (tensor<2x1024xi64>, tensor<2x1x1x1024x2xi64>)
            -> tensor<2x1x1x1024xi64>
        %g_k_4d = stablehlo.broadcast_in_dim %g_k, dims = [0, 1, 2, 3]
            : (tensor<2x1x1x1024xi64>) -> tensor<2x1x1024x1024xi64>

        %eq = stablehlo.compare EQ, %g_q_4d, %g_k_4d
            : (tensor<2x1x1024x1024xi64>, tensor<2x1x1024x1024xi64>)
              -> tensor<2x1x1024x1024xi1>
        %q_4d = stablehlo.broadcast_in_dim %q_iota, dims = [0, 1, 2, 3]
            : (tensor<1x1x1024x1xi64>) -> tensor<2x1x1024x1024xi64>
        %k_4d = stablehlo.broadcast_in_dim %k_iota, dims = [0, 1, 2, 3]
            : (tensor<1x1x1x1024xi64>) -> tensor<2x1x1024x1024xi64>
        %le = stablehlo.compare LE, %k_4d, %q_4d
            : (tensor<2x1x1024x1024xi64>, tensor<2x1x1024x1024xi64>)
              -> tensor<2x1x1024x1024xi1>
        %keep = stablehlo.and %le, %eq : tensor<2x1x1024x1024xi1>
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
            : (tensor<2x32x1024x1024xi1>, tensor<i1>) -> tensor<2x32x1024xi1>
        %row_masked_b = stablehlo.broadcast_in_dim %row_masked, dims = [0, 1, 2]
            : (tensor<2x32x1024xi1>) -> tensor<2x32x1024x1024xi1>
        %z_probs = stablehlo.constant dense<0.0> : tensor<2x32x1024x1024xf32>
        %out = stablehlo.select %row_masked_b, %z_probs, %probs
            : tensor<2x32x1024x1024xi1>, tensor<2x32x1024x1024xf32>
        return %out : tensor<2x32x1024x1024xf32>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());

  int reduce_count = 0;
  int select_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ReduceOp>(op)) {
      ++reduce_count;
    } else if (mlir::isa<mlir::stablehlo::SelectOp>(op)) {
      ++select_count;
    }
  });
  EXPECT_EQ(reduce_count, 0);
  EXPECT_EQ(select_count, 0);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(static_cast<bool>(func));
  EXPECT_EQ(func.getBody().front().getTerminator()->getOperand(0),
            func.getArgument(1));
}

TEST_F(CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxTest,
       CanonicalizesBackwardRowsumSubtractWithDotGeneral) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%logits: tensor<1x24x64x512xbf16>,
                      %dp: tensor<1x24x64x512xf32>)
          -> (tensor<1x24x64x512xbf16>, tensor<1x24x64x512xbf16>) {
        %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
        %max = stablehlo.reduce(%logits init: %neg_inf)
            applies stablehlo.maximum across dimensions = [3]
            : (tensor<1x24x64x512xbf16>, tensor<bf16>)
              -> tensor<1x24x64xbf16>
        %bmax = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2]
            : (tensor<1x24x64xbf16>) -> tensor<1x24x64x512xbf16>
        %sub = stablehlo.subtract %logits, %bmax : tensor<1x24x64x512xbf16>
        %sub_f32 = stablehlo.convert %sub
            : (tensor<1x24x64x512xbf16>) -> tensor<1x24x64x512xf32>
        %exp = stablehlo.exponential %sub_f32 : tensor<1x24x64x512xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %sum = stablehlo.reduce(%exp init: %zero)
            applies stablehlo.add across dimensions = [3]
            : (tensor<1x24x64x512xf32>, tensor<f32>) -> tensor<1x24x64xf32>
        %bsum = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2]
            : (tensor<1x24x64xf32>) -> tensor<1x24x64x512xf32>
        %probs_f32 = stablehlo.divide %exp, %bsum : tensor<1x24x64x512xf32>
        %probs = stablehlo.convert %probs_f32
            : (tensor<1x24x64x512xf32>) -> tensor<1x24x64x512xbf16>

        %z_3d = stablehlo.constant dense<0.0> : tensor<1x24x64xf32>
        %o_3d = stablehlo.constant dense<1.0> : tensor<1x24x64xf32>
        %is_z = stablehlo.compare EQ, %sum, %z_3d
            : (tensor<1x24x64xf32>, tensor<1x24x64xf32>) -> tensor<1x24x64xi1>
        %safe_sum = stablehlo.select %is_z, %o_3d, %sum
            : tensor<1x24x64xi1>, tensor<1x24x64xf32>
        %bsafe = stablehlo.broadcast_in_dim %safe_sum, dims = [0, 1, 2]
            : (tensor<1x24x64xf32>) -> tensor<1x24x64x512xf32>
        %bwd_probs = stablehlo.divide %exp, %bsafe : tensor<1x24x64x512xf32>
        %rowsum = stablehlo.dot_general %bwd_probs, %dp,
            batching_dims = [0, 1, 2] x [0, 1, 2],
            contracting_dims = [3] x [3]
            : (tensor<1x24x64x512xf32>, tensor<1x24x64x512xf32>)
              -> tensor<1x24x64xf32>
        %browsum = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2]
            : (tensor<1x24x64xf32>) -> tensor<1x24x64x512xf32>
        %dp_sub = stablehlo.subtract %dp, %browsum : tensor<1x24x64x512xf32>
        %ds_f32 = stablehlo.multiply %bwd_probs, %dp_sub
            : tensor<1x24x64x512xf32>
        %ds = stablehlo.convert %ds_f32
            : (tensor<1x24x64x512xf32>) -> tensor<1x24x64x512xbf16>
        return %probs, %ds : tensor<1x24x64x512xbf16>, tensor<1x24x64x512xbf16>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());

  int sub_count = 0;
  int div_4d_count = 0;
  int div_3d_count = 0;
  int neg_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::SubtractOp>(op)) {
      ++sub_count;
    } else if (auto div = mlir::dyn_cast<mlir::stablehlo::DivOp>(op)) {
      auto t = mlir::cast<mlir::RankedTensorType>(div.getType());
      if (t.getRank() == 4) {
        ++div_4d_count;
      } else if (t.getRank() == 3) {
        ++div_3d_count;
      }
    } else if (mlir::isa<mlir::stablehlo::NegOp>(op)) {
      ++neg_count;
    }
  });
  EXPECT_EQ(sub_count, 0);
  EXPECT_EQ(div_4d_count, 0);
  EXPECT_EQ(div_3d_count, 2);
  EXPECT_EQ(neg_count, 2);
}

TEST_F(CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxTest,
       PreservesScatterWhenCombinerIsNotAdd) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%x: tensor<1x64x1024x1025xbf16>,
                      %dp_bf16: tensor<1x64x1024x1025xbf16>)
          -> (tensor<1x64x1024x1025xbf16>, tensor<1x64x1024x1025xbf16>) {
        %iota = stablehlo.iota dim = 3 : tensor<1x64x1024x1025xi64>
        %ninf_bf16 = stablehlo.constant dense<-3.389530e+38> : tensor<bf16>
        %z_i64 = stablehlo.constant dense<0> : tensor<i64>
        %max_pair:2 = "stablehlo.reduce"(%x, %iota, %ninf_bf16, %z_i64) ({
        ^bb0(%v0: tensor<bf16>, %i0: tensor<i64>,
             %v1: tensor<bf16>, %i1: tensor<i64>):
          %m = stablehlo.maximum %v0, %v1 : tensor<bf16>
          %ge = stablehlo.compare GE, %v0, %v1
              : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
          %idx = stablehlo.select %ge, %i0, %i1 : tensor<i1>, tensor<i64>
          stablehlo.return %m, %idx : tensor<bf16>, tensor<i64>
        }) {dimensions = array<i64: 3>}
            : (tensor<1x64x1024x1025xbf16>, tensor<1x64x1024x1025xi64>,
               tensor<bf16>, tensor<i64>)
              -> (tensor<1x64x1024xbf16>, tensor<1x64x1024xi64>)
        %bmax = stablehlo.broadcast_in_dim %max_pair#0, dims = [0, 1, 2]
            : (tensor<1x64x1024xbf16>) -> tensor<1x64x1024x1025xbf16>
        %y_bf16 = stablehlo.subtract %x, %bmax : tensor<1x64x1024x1025xbf16>
        %y_f32 = stablehlo.convert %y_bf16
            : (tensor<1x64x1024x1025xbf16>) -> tensor<1x64x1024x1025xf32>
        %ninf_f32 = stablehlo.constant dense<-3.40282347e+38> : tensor<f32>
        %max2 = stablehlo.reduce(%y_f32 init: %ninf_f32)
            applies stablehlo.maximum across dimensions = [3]
            : (tensor<1x64x1024x1025xf32>, tensor<f32>)
              -> tensor<1x64x1024xf32>
        %bmax2 = stablehlo.broadcast_in_dim %max2, dims = [0, 1, 2]
            : (tensor<1x64x1024xf32>) -> tensor<1x64x1024x1025xf32>
        %shifted = stablehlo.subtract %y_f32, %bmax2
            : tensor<1x64x1024x1025xf32>
        %exp = stablehlo.exponential %shifted : tensor<1x64x1024x1025xf32>
        %zero_f32 = stablehlo.constant dense<0.0> : tensor<f32>
        %sum = stablehlo.reduce(%exp init: %zero_f32)
            applies stablehlo.add across dimensions = [3]
            : (tensor<1x64x1024x1025xf32>, tensor<f32>)
              -> tensor<1x64x1024xf32>
        %sum_4d = stablehlo.reshape %sum
            : (tensor<1x64x1024xf32>) -> tensor<1x64x1024x1xf32>
        %one_4d = stablehlo.constant dense<1.0> : tensor<1x64x1024x1xf32>
        %inv_4d = stablehlo.divide %one_4d, %sum_4d : tensor<1x64x1024x1xf32>
        %inv_3d = stablehlo.reshape %inv_4d
            : (tensor<1x64x1024x1xf32>) -> tensor<1x64x1024xf32>
        %binv = stablehlo.broadcast_in_dim %inv_3d, dims = [0, 1, 2]
            : (tensor<1x64x1024xf32>) -> tensor<1x64x1024x1025xf32>
        %probs_f32 = stablehlo.multiply %exp, %binv
            : tensor<1x64x1024x1025xf32>
        %probs = stablehlo.convert %probs_f32
            : (tensor<1x64x1024x1025xf32>) -> tensor<1x64x1024x1025xbf16>

        %probs_bwd = stablehlo.convert %probs
            : (tensor<1x64x1024x1025xbf16>) -> tensor<1x64x1024x1025xf32>
        %dp = stablehlo.convert %dp_bf16
            : (tensor<1x64x1024x1025xbf16>) -> tensor<1x64x1024x1025xf32>
        %rowsum = stablehlo.dot_general %probs_bwd, %dp,
            batching_dims = [0, 1, 2] x [0, 1, 2],
            contracting_dims = [3] x [3]
            : (tensor<1x64x1024x1025xf32>, tensor<1x64x1024x1025xf32>)
              -> tensor<1x64x1024xf32>
        %browsum = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2]
            : (tensor<1x64x1024xf32>) -> tensor<1x64x1024x1025xf32>
        %dp_sub = stablehlo.subtract %dp, %browsum
            : tensor<1x64x1024x1025xf32>
        %ds_f32 = stablehlo.multiply %probs_bwd, %dp_sub
            : tensor<1x64x1024x1025xf32>
        %ds = stablehlo.convert %ds_f32
            : (tensor<1x64x1024x1025xf32>) -> tensor<1x64x1024x1025xbf16>

        %idx_5d = stablehlo.reshape %max_pair#1
            : (tensor<1x64x1024xi64>) -> tensor<1x64x1024x1x1xi64>
        %neg_ds = stablehlo.negate %ds : tensor<1x64x1024x1025xbf16>
        %z_bf16 = stablehlo.constant dense<0.0> : tensor<bf16>
        %sum_neg_ds = stablehlo.reduce(%neg_ds init: %z_bf16)
            applies stablehlo.add across dimensions = [3]
            : (tensor<1x64x1024x1025xbf16>, tensor<bf16>)
              -> tensor<1x64x1024xbf16>
        %upd = stablehlo.reshape %sum_neg_ds
            : (tensor<1x64x1024xbf16>) -> tensor<1x64x1024x1xbf16>
        %zeros_4d = stablehlo.constant dense<0.0>
            : tensor<1x64x1024x1025xbf16>
        %scat = "stablehlo.scatter"(%zeros_4d, %idx_5d, %upd) ({
        ^bb0(%a0: tensor<bf16>, %a1: tensor<bf16>):
          %s = stablehlo.maximum %a0, %a1 : tensor<bf16>
          stablehlo.return %s : tensor<bf16>
        }) {
          scatter_dimension_numbers = #stablehlo.scatter<
            update_window_dims = [],
            inserted_window_dims = [3],
            input_batching_dims = [0, 1, 2],
            scatter_indices_batching_dims = [0, 1, 2],
            scatter_dims_to_operand_dims = [3],
            index_vector_dim = 4>
        } : (tensor<1x64x1024x1025xbf16>, tensor<1x64x1024x1x1xi64>,
             tensor<1x64x1024x1xbf16>) -> tensor<1x64x1024x1025xbf16>
        %dx = stablehlo.add %ds, %scat : tensor<1x64x1024x1025xbf16>
        return %probs, %dx
            : tensor<1x64x1024x1025xbf16>, tensor<1x64x1024x1025xbf16>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());

  int scatter_count = 0;
  module->walk([&](mlir::stablehlo::ScatterOp op) { ++scatter_count; });
  EXPECT_EQ(scatter_count, 1);
}

TEST_F(CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxTest,
       DoesNotFoldRowAllMaskedWhenReverseOpReversesKeyOrQuery) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%cumsum: tensor<2x1024xi64>,
                      %probs: tensor<2x32x1024x1024xf32>)
          -> tensor<2x32x1024x1024xf32> {
        %iota = stablehlo.iota dim = 2 : tensor<2x1x1024x1024xi64>
        %iota_k = stablehlo.iota dim = 3 : tensor<2x1x1024x1024xi64>
        %eq = stablehlo.compare EQ, %iota, %iota_k
            : (tensor<2x1x1024x1024xi64>, tensor<2x1x1024x1024xi64>)
              -> tensor<2x1x1024x1024xi1>
        %z_mask = stablehlo.constant dense<0.0> : tensor<2x1x1024x1024xbf16>
        %ninf_mask = stablehlo.constant dense<0xFF80>
            : tensor<2x1x1024x1024xbf16>
        %mask_1 = stablehlo.select %eq, %z_mask, %ninf_mask
            : tensor<2x1x1024x1024xi1>, tensor<2x1x1024x1024xbf16>
        %mask = stablehlo.broadcast_in_dim %mask_1, dims = [0, 1, 2, 3]
            : (tensor<2x1x1024x1024xbf16>) -> tensor<2x32x1024x1024xbf16>
        %ninf_4d = stablehlo.constant dense<0xFF80>
            : tensor<2x32x1024x1024xbf16>
        %is_ninf = stablehlo.compare EQ, %mask, %ninf_4d
            : (tensor<2x32x1024x1024xbf16>, tensor<2x32x1024x1024xbf16>)
              -> tensor<2x32x1024x1024xi1>
        %is_ninf_rev = "stablehlo.reverse"(%is_ninf) {dimensions = array<i64: 3>}
            : (tensor<2x32x1024x1024xi1>) -> tensor<2x32x1024x1024xi1>
        %true_s = stablehlo.constant dense<true> : tensor<i1>
        %row_masked = stablehlo.reduce(%is_ninf_rev init: %true_s)
            applies stablehlo.and across dimensions = [3]
            : (tensor<2x32x1024x1024xi1>, tensor<i1>) -> tensor<2x32x1024xi1>
        %row_masked_b = stablehlo.broadcast_in_dim %row_masked, dims = [0, 1, 2]
            : (tensor<2x32x1024xi1>) -> tensor<2x32x1024x1024xi1>
        %z_probs = stablehlo.constant dense<0.0> : tensor<2x32x1024x1024xf32>
        %out = stablehlo.select %row_masked_b, %z_probs, %probs
            : tensor<2x32x1024x1024xi1>, tensor<2x32x1024x1024xf32>
        return %out : tensor<2x32x1024x1024xf32>
      }
    }
  )mlir";

  auto module = ParseAndOptimize(kIr);
  ASSERT_TRUE(static_cast<bool>(module));
  ASSERT_FALSE(HasFailure());

  int reduce_count = 0;
  int select_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ReduceOp>(op)) {
      ++reduce_count;
    } else if (mlir::isa<mlir::stablehlo::SelectOp>(op)) {
      ++select_count;
    }
  });
  EXPECT_GT(reduce_count, 0);
  EXPECT_GT(select_count, 0);
}

}  // namespace
}  // namespace torch_tpu
