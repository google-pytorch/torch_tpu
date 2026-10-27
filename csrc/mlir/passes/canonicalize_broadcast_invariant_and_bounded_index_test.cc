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

#include "csrc/mlir/passes/canonicalize_broadcast_invariant_and_bounded_index.h"

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
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOk;

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

TEST(CanonicalizeBroadcastInvariantAndBoundedIndexTest,
     CollapsesSingletonExpandedWeightNormAndStripsUnitBroadcastDims) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%v: tensor<8x4x1xbf16>, %g: tensor<8x1x1xbf16>)
      -> (tensor<8x4x1xbf16>, tensor<8x1x1xf32>) {
    %v_f32 = stablehlo.convert %v : (tensor<8x4x1xbf16>) -> tensor<8x4x1xf32>
    %g_f32 = stablehlo.convert %g : (tensor<8x1x1xbf16>) -> tensor<8x1x1xf32>
    %sq = stablehlo.multiply %v_f32, %v_f32 : tensor<8x4x1xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %red = stablehlo.reduce(%sq init: %zero) across dimensions = [1, 2]
        : (tensor<8x4x1xf32>, tensor<f32>) -> tensor<8xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
        %sum = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %sum : tensor<f32>
      }
    %red_3d = stablehlo.reshape %red : (tensor<8xf32>) -> tensor<8x1x1xf32>
    %norm_3d = stablehlo.sqrt %red_3d : tensor<8x1x1xf32>
    %div_3d = stablehlo.divide %g_f32, %norm_3d : tensor<8x1x1xf32>
    %bcast_id = stablehlo.broadcast_in_dim %div_3d, dims = [0, 1, 2]
        : (tensor<8x1x1xf32>) -> tensor<8x1x1xf32>
    %div_2d = stablehlo.reshape %bcast_id
        : (tensor<8x1x1xf32>) -> tensor<8x1xf32>
    %bcast_v = stablehlo.broadcast_in_dim %div_2d, dims = [0, 2]
        : (tensor<8x1xf32>) -> tensor<8x4x1xf32>
    %out_f32 = stablehlo.multiply %v_f32, %bcast_v : tensor<8x4x1xf32>
    %out_bf16 = stablehlo.convert %out_f32
        : (tensor<8x4x1xf32>) -> tensor<8x4x1xbf16>
    return %out_bf16, %norm_3d : tensor<8x4x1xbf16>, tensor<8x1x1xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int sqrt_1d_count = 0;
  int sqrt_3d_count = 0;
  int div_1d_count = 0;
  int div_3d_count = 0;
  module->walk([&](mlir::stablehlo::SqrtOp op) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(op.getType());
    if (rtt.getRank() == 1) {
      ++sqrt_1d_count;
    } else if (rtt.getRank() == 3) {
      ++sqrt_3d_count;
    }
  });
  module->walk([&](mlir::stablehlo::DivOp op) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(op.getType());
    if (rtt.getRank() == 1) {
      ++div_1d_count;
    } else if (rtt.getRank() == 3) {
      ++div_3d_count;
    }
  });

  EXPECT_EQ(sqrt_1d_count, 1);
  EXPECT_EQ(sqrt_3d_count, 0);
  EXPECT_EQ(div_1d_count, 1);
  EXPECT_EQ(div_3d_count, 0);
}

TEST(CanonicalizeBroadcastInvariantAndBoundedIndexTest,
     FoldsBroadcastInvariantBatchedDotAndMultiBlockRotarySliceConcat) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%inv_freq: tensor<8xf32>) -> tensor<1x16x16xbf16> {
    %lhs = stablehlo.broadcast_in_dim %inv_freq, dims = [1]
        : (tensor<8xf32>) -> tensor<3x8x1xf32>
    %iota_i64 = stablehlo.iota dim = 0 : tensor<16xi64>
    %bcast_iota = stablehlo.broadcast_in_dim %iota_i64, dims = [3]
        : (tensor<16xi64>) -> tensor<3x1x1x16xi64>
    %flt_iota = stablehlo.convert %bcast_iota
        : (tensor<3x1x1x16xi64>) -> tensor<3x1x1x16xf32>
    %rhs = stablehlo.reshape %flt_iota
        : (tensor<3x1x1x16xf32>) -> tensor<3x1x16xf32>
    %dot = stablehlo.dot_general %lhs, %rhs,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<3x8x1xf32>, tensor<3x1x16xf32>) -> tensor<3x8x16xf32>
    %res4d = stablehlo.reshape %dot
        : (tensor<3x8x16xf32>) -> tensor<3x8x16x1xf32>
    %tr = stablehlo.transpose %res4d, dims = [0, 3, 2, 1]
        : (tensor<3x8x16x1xf32>) -> tensor<3x1x16x8xf32>
    %cat = stablehlo.concatenate %tr, %tr, dim = 3
        : (tensor<3x1x16x8xf32>, tensor<3x1x16x8xf32>) -> tensor<3x1x16x16xf32>
    %cos = stablehlo.cosine %cat : tensor<3x1x16x16xf32>
    %cos_bf16 = stablehlo.convert %cos
        : (tensor<3x1x16x16xf32>) -> tensor<3x1x16x16xbf16>
    %s0 = stablehlo.slice %cos_bf16 [0:1, 0:1, 0:16, 0:4]
        : (tensor<3x1x16x16xbf16>) -> tensor<1x1x16x4xbf16>
    %r0 = stablehlo.reshape %s0
        : (tensor<1x1x16x4xbf16>) -> tensor<1x16x4xbf16>
    %s1 = stablehlo.slice %cos_bf16 [1:2, 0:1, 0:16, 4:8]
        : (tensor<3x1x16x16xbf16>) -> tensor<1x1x16x4xbf16>
    %r1 = stablehlo.reshape %s1
        : (tensor<1x1x16x4xbf16>) -> tensor<1x16x4xbf16>
    %s2 = stablehlo.slice %cos_bf16 [2:3, 0:1, 0:16, 8:16]
        : (tensor<3x1x16x16xbf16>) -> tensor<1x1x16x8xbf16>
    %r2 = stablehlo.reshape %s2
        : (tensor<1x1x16x8xbf16>) -> tensor<1x16x8xbf16>
    %out = stablehlo.concatenate %r0, %r1, %r2, dim = 2
        : (tensor<1x16x4xbf16>, tensor<1x16x4xbf16>, tensor<1x16x8xbf16>)
        -> tensor<1x16x16xbf16>
    return %out : tensor<1x16x16xbf16>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int slice_count = 0;
  int i64_iota_count = 0;
  int dim3_cos_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    }
    if (auto iota = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
      if (mlir::cast<mlir::RankedTensorType>(iota.getType())
              .getElementType()
              .isInteger(64)) {
        ++i64_iota_count;
      }
    }
    if (auto cos_op = mlir::dyn_cast<mlir::stablehlo::CosineOp>(op)) {
      auto rtt = mlir::cast<mlir::RankedTensorType>(cos_op.getType());
      if (rtt.getDimSize(0) == 3) {
        ++dim3_cos_count;
      }
    }
  });

  EXPECT_EQ(slice_count, 0);
  EXPECT_EQ(i64_iota_count, 0);
  EXPECT_EQ(dim3_cos_count, 0);
}

TEST(CanonicalizeBroadcastInvariantAndBoundedIndexTest,
     NarrowsRelativePositionBucketI64SubgraphToI32) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%table: tensor<32x12xbf16>) -> tensor<64x64x12xbf16> {
    %iota = stablehlo.iota dim = 0 : tensor<64xi64>
    %b0 = stablehlo.broadcast_in_dim %iota, dims = [1]
        : (tensor<64xi64>) -> tensor<64x64xi64>
    %b1 = stablehlo.broadcast_in_dim %iota, dims = [0]
        : (tensor<64xi64>) -> tensor<64x64xi64>
    %sub = stablehlo.subtract %b0, %b1 : tensor<64x64xi64>
    %zero = stablehlo.constant dense<0> : tensor<64x64xi64>
    %min0 = stablehlo.minimum %sub, %zero : tensor<64x64xi64>
    %neg = stablehlo.negate %min0 : tensor<64x64xi64>
    %c16 = stablehlo.constant dense<16> : tensor<64x64xi64>
    %lt = stablehlo.compare LT, %neg, %c16
        : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %f_neg = stablehlo.convert %neg : (tensor<64x64xi64>) -> tensor<64x64xf32>
    %fc16 = stablehlo.constant dense<16.000000e+00> : tensor<64x64xf32>
    %div1 = stablehlo.divide %f_neg, %fc16 : tensor<64x64xf32>
    %log = stablehlo.log %div1 : tensor<64x64xf32>
    %fc_scale = stablehlo.constant dense<2.07944155e+00> : tensor<64x64xf32>
    %div2 = stablehlo.divide %log, %fc_scale : tensor<64x64xf32>
    %mul = stablehlo.multiply %div2, %fc16 : tensor<64x64xf32>
    %fc_min = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf32>
    %fc_max = stablehlo.constant dense<15.000000e+00> : tensor<64x64xf32>
    %clamp_f = stablehlo.clamp %fc_min, %mul, %fc_max : tensor<64x64xf32>
    %to_i64 = stablehlo.convert %clamp_f : (tensor<64x64xf32>) -> tensor<64x64xi64>
    %add = stablehlo.add %to_i64, %c16 : tensor<64x64xi64>
    %c31 = stablehlo.constant dense<31> : tensor<64x64xi64>
    %min1 = stablehlo.minimum %add, %c31 : tensor<64x64xi64>
    %bucket = stablehlo.select %lt, %neg, %min1
        : tensor<64x64xi1>, tensor<64x64xi64>
    %emb = "stablehlo.gather"(%table, %bucket) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 2
      >,
      slice_sizes = array<i64: 1, 12>
    } : (tensor<32x12xbf16>, tensor<64x64xi64>) -> tensor<64x64x12xbf16>
    return %emb : tensor<64x64x12xbf16>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int i64_op_count = 0;
  int i32_iota_count = 0;
  module->walk([&](mlir::Operation* op) {
    for (mlir::Value res : op->getResults()) {
      if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(res.getType())) {
        if (rtt.getElementType().isInteger(64)) {
          ++i64_op_count;
        }
      }
    }
    if (auto iota = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
      if (mlir::cast<mlir::RankedTensorType>(iota.getType())
              .getElementType()
              .isInteger(32)) {
        ++i32_iota_count;
      }
    }
  });

  EXPECT_EQ(i64_op_count, 0);
  EXPECT_EQ(i32_iota_count, 1);
}

TEST(CanonicalizeBroadcastInvariantAndBoundedIndexTest,
     RunsOnIsolatedPassManager) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%inv_freq: tensor<8xf32>) -> tensor<1x16x16xbf16> {
    %lhs = stablehlo.broadcast_in_dim %inv_freq, dims = [1]
        : (tensor<8xf32>) -> tensor<3x8x1xf32>
    %iota_i64 = stablehlo.iota dim = 0 : tensor<16xi64>
    %bcast_iota = stablehlo.broadcast_in_dim %iota_i64, dims = [3]
        : (tensor<16xi64>) -> tensor<3x1x1x16xi64>
    %flt_iota = stablehlo.convert %bcast_iota
        : (tensor<3x1x1x16xi64>) -> tensor<3x1x1x16xf32>
    %rhs = stablehlo.reshape %flt_iota
        : (tensor<3x1x1x16xf32>) -> tensor<3x1x16xf32>
    %dot = stablehlo.dot_general %lhs, %rhs,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<3x8x1xf32>, tensor<3x1x16xf32>) -> tensor<3x8x16xf32>
    %res4d = stablehlo.reshape %dot
        : (tensor<3x8x16xf32>) -> tensor<3x8x16x1xf32>
    %tr = stablehlo.transpose %res4d, dims = [0, 3, 2, 1]
        : (tensor<3x8x16x1xf32>) -> tensor<3x1x16x8xf32>
    %cat = stablehlo.concatenate %tr, %tr, dim = 3
        : (tensor<3x1x16x8xf32>, tensor<3x1x16x8xf32>) -> tensor<3x1x16x16xf32>
    %cos = stablehlo.cosine %cat : tensor<3x1x16x16xf32>
    %cos_bf16 = stablehlo.convert %cos
        : (tensor<3x1x16x16xf32>) -> tensor<3x1x16x16xbf16>
    %s0 = stablehlo.slice %cos_bf16 [0:1, 0:1, 0:16, 0:4]
        : (tensor<3x1x16x16xbf16>) -> tensor<1x1x16x4xbf16>
    %r0 = stablehlo.reshape %s0
        : (tensor<1x1x16x4xbf16>) -> tensor<1x16x4xbf16>
    %s1 = stablehlo.slice %cos_bf16 [1:2, 0:1, 0:16, 4:8]
        : (tensor<3x1x16x16xbf16>) -> tensor<1x1x16x4xbf16>
    %r1 = stablehlo.reshape %s1
        : (tensor<1x1x16x4xbf16>) -> tensor<1x16x4xbf16>
    %s2 = stablehlo.slice %cos_bf16 [2:3, 0:1, 0:16, 8:16]
        : (tensor<3x1x16x16xbf16>) -> tensor<1x1x16x8xbf16>
    %r2 = stablehlo.reshape %s2
        : (tensor<1x1x16x8xbf16>) -> tensor<1x16x8xbf16>
    %out = stablehlo.concatenate %r0, %r1, %r2, dim = 2
        : (tensor<1x16x4xbf16>, tensor<1x16x4xbf16>, tensor<1x16x8xbf16>)
        -> tensor<1x16x16xbf16>
    return %out : tensor<1x16x16xbf16>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeBroadcastInvariantAndBoundedIndexPass());
  EXPECT_TRUE(mlir::succeeded(pm.run(*module)));
}

TEST(CanonicalizeBroadcastInvariantAndBoundedIndexTest,
     SafelyFallsBackForTransposeWithNonLeadingPermutation) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8x16xf32>) -> tensor<1x4x16xf32> {
    %bcast = stablehlo.broadcast_in_dim %x, dims = [1, 0, 2]
        : (tensor<4x8x16xf32>) -> tensor<8x4x16xf32>
    %tr = stablehlo.transpose %bcast, dims = [1, 0, 2]
        : (tensor<8x4x16xf32>) -> tensor<4x8x16xf32>
    %s = stablehlo.slice %tr [0:1, 0:4, 0:16]
        : (tensor<4x8x16xf32>) -> tensor<1x4x16xf32>
    return %s : tensor<1x4x16xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeBroadcastInvariantAndBoundedIndexPass());
  EXPECT_TRUE(mlir::succeeded(pm.run(*module)));
}

TEST(CanonicalizeBroadcastInvariantAndBoundedIndexTest,
     PreservesNonContiguousSlicesFeedingConcatenate) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<16xf32>) -> tensor<4xf32> {
    %s0 = stablehlo.slice %x [0:2] : (tensor<16xf32>) -> tensor<2xf32>
    %s1 = stablehlo.slice %x [4:6] : (tensor<16xf32>) -> tensor<2xf32>
    %cat = stablehlo.concatenate %s0, %s1, dim = 0
        : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
    return %cat : tensor<4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeBroadcastInvariantAndBoundedIndexPass());
  EXPECT_TRUE(mlir::succeeded(pm.run(*module)));

  int slice_count = 0;
  int concat_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    }
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
  });
  EXPECT_EQ(slice_count, 2);
  EXPECT_EQ(concat_count, 1);
}

}  // namespace
}  // namespace torch_tpu
