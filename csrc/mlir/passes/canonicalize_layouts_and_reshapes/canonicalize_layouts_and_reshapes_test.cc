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

#include "csrc/mlir/passes/canonicalize_layouts_and_reshapes/canonicalize_layouts_and_reshapes.h"  // NOLINT

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"  // NOLINT
#include "mlir/Dialect/Func/IR/FuncOps.h"      // NOLINT
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float base = 1.0f) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 17 + 5) % 251) * 0.05f -
              static_cast<float>((i * 7) % 13) * 0.1f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

void AssertModulesEvaluateNumericallyEqual(
    mlir::ModuleOp orig_mod, mlir::ModuleOp opt_mod,
    llvm::ArrayRef<mlir::DenseElementsAttr> args) {
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(orig_mod, args, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));
  auto actual = mlir::stablehlo::evalModule(opt_mod, args, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ((*expected).size(), (*actual).size());
  for (size_t res_idx = 0; res_idx < (*expected).size(); ++res_idx) {
    auto exp_vals = (*expected)[res_idx].getValues<float>();
    auto act_vals = (*actual)[res_idx].getValues<float>();
    ASSERT_EQ(exp_vals.size(), act_vals.size());
    for (size_t i = 0; i < exp_vals.size(); ++i) {
      EXPECT_NEAR(exp_vals[i], act_vals[i], 1e-4f);
    }
  }
}

void InitMlirContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::chlo::ChloDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

template <typename OpTy>
int CountOps(mlir::ModuleOp module) {
  int count = 0;
  module.walk([&count](OpTy) { ++count; });
  return count;
}

mlir::LogicalResult RunPass(mlir::ModuleOp module, mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeLayoutsAndReshapesPass());
  return pm.run(module);
}

// =============================================================================
// GQA Broadcast Unwrapping Tests
// =============================================================================

TEST(CanonicalizeLayoutsAndReshapesTest, UnwrapGqaBroadcastsRank4) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_rank4(%q: tensor<1x8x16x64xf32>,
                              %k: tensor<1x2x16x64xf32>)
      -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 3, 4]
        : (tensor<1x2x16x64xf32>) -> tensor<1x2x4x16x64xf32>
    %1 = stablehlo.reshape %0
        : (tensor<1x2x4x16x64xf32>) -> tensor<1x8x16x64xf32>
    %2 = stablehlo.dot_general %q, %1,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x16x64xf32>, tensor<1x8x16x64xf32>)
        -> tensor<1x8x16x16xf32>
    return %2 : tensor<1x8x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 0);
  mod->walk([](mlir::stablehlo::DotGeneralOp op) {
    const auto rhs_ty =
        llvm::cast<mlir::RankedTensorType>(op.getRhs().getType());
    EXPECT_EQ(rhs_ty.getRank(), 4);
    EXPECT_EQ(rhs_ty.getDimSize(1), 2);
  });

  auto q = MakeF32Tensor(context, {1, 8, 16, 64}, 1.0f);
  auto k = MakeF32Tensor(context, {1, 2, 16, 64}, 0.5f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {q, k});
}

TEST(CanonicalizeLayoutsAndReshapesTest, UnwrapGqaBroadcastsRank5) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Generalized to arbitrary rank (rank 5 input).
  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_rank5(%q: tensor<2x1x8x16x64xf32>,
                              %k: tensor<2x1x2x16x64xf32>)
      -> tensor<2x1x8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 2, 4, 5]
        : (tensor<2x1x2x16x64xf32>) -> tensor<2x1x2x4x16x64xf32>
    %1 = stablehlo.reshape %0
        : (tensor<2x1x2x4x16x64xf32>) -> tensor<2x1x8x16x64xf32>
    %2 = stablehlo.dot_general %q, %1,
        batching_dims = [0, 1, 2] x [0, 1, 2],
        contracting_dims = [4] x [4]
        : (tensor<2x1x8x16x64xf32>, tensor<2x1x8x16x64xf32>)
        -> tensor<2x1x8x16x16xf32>
    return %2 : tensor<2x1x8x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 0);

  auto q = MakeF32Tensor(context, {2, 1, 8, 16, 64}, 1.0f);
  auto k = MakeF32Tensor(context, {2, 1, 2, 16, 64}, 0.5f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {q, k});
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     UnwrapGqaBroadcastsWithTransposedLastTwoAndReshape) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // GQA KV broadcast with transpose permuting last two dimensions and
  // unsqueeze + broadcast + reshape pattern.
  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_transpose(%q: tensor<1x8x16x64xf32>,
                                  %k: tensor<1x2x32x64xf32>)
      -> tensor<1x8x16x32xf32> {
    %0 = stablehlo.reshape %k
        : (tensor<1x2x32x64xf32>) -> tensor<1x2x1x32x64xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3, 4]
        : (tensor<1x2x1x32x64xf32>) -> tensor<1x2x4x32x64xf32>
    %2 = stablehlo.reshape %1
        : (tensor<1x2x4x32x64xf32>) -> tensor<1x8x32x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1, 3, 2]
        : (tensor<1x8x32x64xf32>) -> tensor<1x8x64x32xf32>
    %4 = stablehlo.dot_general %q, %3,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2]
        : (tensor<1x8x16x64xf32>, tensor<1x8x64x32xf32>)
        -> tensor<1x8x16x32xf32>
    return %4 : tensor<1x8x16x32xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 0);

  auto q = MakeF32Tensor(context, {1, 8, 16, 64}, 1.0f);
  auto k = MakeF32Tensor(context, {1, 2, 32, 64}, 0.5f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {q, k});
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     UnwrapGqaBroadcastsNegativeNonCanonicalBatch) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Non-canonical batch dims [0, 2] instead of leading iota [0, 1].
  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_negative_layout(%q: tensor<1x16x8x64xf32>,
                                        %k: tensor<1x16x2x64xf32>)
      -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 2, 4]
        : (tensor<1x16x2x64xf32>) -> tensor<1x16x2x4x64xf32>
    %1 = stablehlo.reshape %0
        : (tensor<1x16x2x4x64xf32>) -> tensor<1x16x8x64xf32>
    %2 = stablehlo.dot_general %q, %1,
        batching_dims = [0, 2] x [0, 2],
        contracting_dims = [3] x [3]
        : (tensor<1x16x8x64xf32>, tensor<1x16x8x64xf32>)
        -> tensor<1x8x16x16xf32>
    return %2 : tensor<1x8x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  // Non-canonical batch dims layout must be rejected.
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 1);
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     UnwrapGqaBroadcastsNegativeGroupSizeOne) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Group size 1 is MHA, not GQA; must not fire.
  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_negative_group_one(%q: tensor<1x8x16x64xf32>,
                                          %k: tensor<1x8x16x64xf32>)
      -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 3, 4]
        : (tensor<1x8x16x64xf32>) -> tensor<1x8x1x16x64xf32>
    %1 = stablehlo.reshape %0
        : (tensor<1x8x1x16x64xf32>) -> tensor<1x8x16x64xf32>
    %2 = stablehlo.dot_general %q, %1,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x16x64xf32>, tensor<1x8x16x64xf32>)
        -> tensor<1x8x16x16xf32>
    return %2 : tensor<1x8x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 1);
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     UnwrapGqaBroadcastsPrefillSequenceLengthGenerically) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // 1. Prefill sequence length 128 is unwrapped generically without magic
  // sequence length thresholds.
  static constexpr const char* kIrPrefill = R"mlir(
module {
  func.func @prefill_gqa(%q: tensor<1x8x128x64xf32>,
                         %k: tensor<1x2x128x64xf32>)
      -> tensor<1x8x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 3, 4]
        : (tensor<1x2x128x64xf32>) -> tensor<1x2x4x128x64xf32>
    %1 = stablehlo.reshape %0
        : (tensor<1x2x4x128x64xf32>) -> tensor<1x8x128x64xf32>
    %2 = stablehlo.dot_general %q, %1,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>)
        -> tensor<1x8x128x128xf32>
    return %2 : tensor<1x8x128x128xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_prefill_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIrPrefill, &context);
  mlir::OwningOpRef<mlir::ModuleOp> prefill_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIrPrefill, &context);
  ASSERT_TRUE(orig_prefill_mod && prefill_mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*prefill_mod, &context)));
  // Prefill length 128 is unwrapped; BroadcastInDimOp eliminated.
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*prefill_mod), 0);

  auto q_pre = MakeF32Tensor(context, {1, 8, 128, 64}, 1.0f);
  auto k_pre = MakeF32Tensor(context, {1, 2, 128, 64}, 0.5f);
  AssertModulesEvaluateNumericallyEqual(*orig_prefill_mod, *prefill_mod,
                                        {q_pre, k_pre});

  // 2. Decode sequence length 1: unwrapping also occurs.
  static constexpr const char* kIrDecode = R"mlir(
module {
  func.func @decode_gqa(%q: tensor<1x8x1x64xf32>,
                        %k: tensor<1x2x128x64xf32>)
      -> tensor<1x8x1x128xf32> {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 3, 4]
        : (tensor<1x2x128x64xf32>) -> tensor<1x2x4x128x64xf32>
    %1 = stablehlo.reshape %0
        : (tensor<1x2x4x128x64xf32>) -> tensor<1x8x128x64xf32>
    %2 = stablehlo.dot_general %q, %1,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x1x64xf32>, tensor<1x8x128x64xf32>)
        -> tensor<1x8x1x128xf32>
    return %2 : tensor<1x8x1x128xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_decode_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIrDecode, &context);
  mlir::OwningOpRef<mlir::ModuleOp> decode_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIrDecode, &context);
  ASSERT_TRUE(orig_decode_mod && decode_mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*decode_mod, &context)));
  // Decode length 1 IS unwrapped; BroadcastInDimOp eliminated.
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*decode_mod), 0);

  auto q_dec = MakeF32Tensor(context, {1, 8, 1, 64}, 1.0f);
  auto k_dec = MakeF32Tensor(context, {1, 2, 128, 64}, 0.5f);
  AssertModulesEvaluateNumericallyEqual(*orig_decode_mod, *decode_mod,
                                        {q_dec, k_dec});
}

// =============================================================================
// Projection Reshape Sinking Tests
// =============================================================================

TEST(CanonicalizeLayoutsAndReshapesTest, SinkProjectionReshapesDotGeneral) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @sink_proj_reshape_dg(%x: tensor<16x32xf32>,
                                  %w: tensor<32x64xf32>,
                                  %bias: tensor<64xf32>)
      -> tensor<4x4x64xf32> {
    %0 = stablehlo.dot_general %x, %w,
        contracting_dims = [1] x [0]
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<64xf32>) -> tensor<16x64xf32>
    %1 = stablehlo.add %0, %bcast : tensor<16x64xf32>
    %2 = stablehlo.reshape %1 : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    return %2 : tensor<4x4x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  ASSERT_EQ(CountOps<mlir::stablehlo::AddOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::AddOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    ASSERT_EQ(ty.getRank(), 3);
    EXPECT_EQ(ty.getDimSize(0), 4);
    EXPECT_EQ(ty.getDimSize(1), 4);
    EXPECT_EQ(ty.getDimSize(2), 64);
  });

  auto x = MakeF32Tensor(context, {16, 32}, 0.5f);
  auto w = MakeF32Tensor(context, {32, 64}, 0.25f);
  auto bias = MakeF32Tensor(context, {64}, 1.0f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {x, w, bias});
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     SinkProjectionReshapeNegativeNonTrailingBias) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Bias broadcast is on token dimension [0], not trailing dimension.
  constexpr const char* kIr = R"mlir(
module {
  func.func @sink_proj_negative_bias(%x: tensor<16x32xf32>,
                                     %w: tensor<32x64xf32>,
                                     %bias: tensor<16xf32>)
      -> tensor<4x4x64xf32> {
    %0 = stablehlo.dot %x, %w
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [0]
        : (tensor<16xf32>) -> tensor<16x64xf32>
    %1 = stablehlo.add %0, %bcast : tensor<16x64xf32>
    %2 = stablehlo.reshape %1 : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    return %2 : tensor<4x4x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  // Non-trailing broadcast must not be sunk.
  ASSERT_EQ(CountOps<mlir::stablehlo::AddOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::AddOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getRank(), 2);
    EXPECT_EQ(ty.getDimSize(0), 16);
    EXPECT_EQ(ty.getDimSize(1), 64);
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     SinkProjectionReshapeNegativeNonTrailingSizeOneBroadcast) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Bias has shape <16x1xf32> broadcast to <16x64xf32> with dims = [0, 1].
  // The reshape target <64x16x1xf32> matches the trailing shape of the bias
  // (<16x1>), but the pre-reshape shape <16x64> does not.
  constexpr const char* kIr = R"mlir(
module {
  func.func @sink_proj_negative_size_one_broadcast(
      %x: tensor<16x32xf32>, %w: tensor<32x64xf32>, %bias: tensor<16x1xf32>)
      -> tensor<64x16x1xf32> {
    %0 = stablehlo.dot %x, %w
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [0, 1]
        : (tensor<16x1xf32>) -> tensor<16x64xf32>
    %1 = stablehlo.add %0, %bcast : tensor<16x64xf32>
    %2 = stablehlo.reshape %1 : (tensor<16x64xf32>) -> tensor<64x16x1xf32>
    return %2 : tensor<64x16x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  ASSERT_EQ(CountOps<mlir::stablehlo::AddOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::AddOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getRank(), 2);
    EXPECT_EQ(ty.getDimSize(0), 16);
    EXPECT_EQ(ty.getDimSize(1), 64);
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     SinkProjectionReshapesConvertF32ToBF16) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @sink_proj_reshape_convert(%x: tensor<16x32xf32>,
                                       %w: tensor<32x64xf32>,
                                       %bias: tensor<64xbf16>)
      -> tensor<4x4x64xbf16> {
    %0 = stablehlo.dot_general %x, %w,
        contracting_dims = [1] x [0]
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %cvt = stablehlo.convert %0 : (tensor<16x64xf32>) -> tensor<16x64xbf16>
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<64xbf16>) -> tensor<16x64xbf16>
    %1 = stablehlo.add %cvt, %bcast : tensor<16x64xbf16>
    %2 = stablehlo.reshape %1 : (tensor<16x64xbf16>) -> tensor<4x4x64xbf16>
    return %2 : tensor<4x4x64xbf16>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  ASSERT_EQ(CountOps<mlir::stablehlo::AddOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::DotGeneralOp op) {
    auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getRank(), 2);
    EXPECT_TRUE(ty.getElementType().isF32());
  });
  mod->walk([](mlir::stablehlo::ReshapeOp op) {
    auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getShape(), llvm::ArrayRef<int64_t>({4, 4, 64}));
    EXPECT_TRUE(ty.getElementType().isF32());
  });
  mod->walk([](mlir::stablehlo::ConvertOp op) {
    auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getShape(), llvm::ArrayRef<int64_t>({4, 4, 64}));
    EXPECT_TRUE(ty.getElementType().isBF16());
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest, SinkProjectionReshapesScalarSplat) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @sink_proj_reshape_scalar(%x: tensor<16x32xf32>,
                                      %w: tensor<32x64xf32>)
      -> tensor<4x4x64xf32> {
    %0 = stablehlo.dot_general %x, %w,
        contracting_dims = [1] x [0]
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %cst = stablehlo.constant dense<2.0> : tensor<f32>
    %bcast = stablehlo.broadcast_in_dim %cst, dims = []
        : (tensor<f32>) -> tensor<16x64xf32>
    %1 = stablehlo.multiply %0, %bcast : tensor<16x64xf32>
    %2 = stablehlo.reshape %1 : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    return %2 : tensor<4x4x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  ASSERT_EQ(CountOps<mlir::stablehlo::MulOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::MulOp op) {
    auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getShape(), llvm::ArrayRef<int64_t>({4, 4, 64}));
  });

  auto x = MakeF32Tensor(context, {16, 32}, 0.5f);
  auto w = MakeF32Tensor(context, {32, 64}, 0.25f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {x, w});
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     SinkProjectionReshapeBailsOnDynamicShapes) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @sink_proj_dyn(%x: tensor<?x32xf32>,
                           %w: tensor<32x64xf32>)
      -> tensor<?x64xf32> {
    %0 = stablehlo.dot_general %x, %w,
        contracting_dims = [1] x [0]
        : (tensor<?x32xf32>, tensor<32x64xf32>) -> tensor<?x64xf32>
    return %0 : tensor<?x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     SinkProjectionReshapeBailsOnZeroElements) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @sink_proj_zero_elem(%x: tensor<0x32xf32>,
                                 %w: tensor<32x64xf32>,
                                 %bias: tensor<64xf32>)
      -> tensor<0x4x16xf32> {
    %0 = stablehlo.dot_general %x, %w,
        contracting_dims = [1] x [0]
        : (tensor<0x32xf32>, tensor<32x64xf32>) -> tensor<0x64xf32>
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<64xf32>) -> tensor<0x64xf32>
    %1 = stablehlo.add %0, %bcast : tensor<0x64xf32>
    %2 = stablehlo.reshape %1 : (tensor<0x64xf32>) -> tensor<0x4x16xf32>
    return %2 : tensor<0x4x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*mod), 1);
}

TEST(CanonicalizeLayoutsAndReshapesTest, PipelineIntegrationTest) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @pipeline_integration(%x: tensor<16x32xf32>,
                                  %w: tensor<32x64xf32>,
                                  %bias: tensor<64xf32>)
      -> tensor<4x4x64xf32> {
    %0 = stablehlo.dot_general %x, %w,
        contracting_dims = [1] x [0]
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<64xf32>) -> tensor<16x64xf32>
    %1 = stablehlo.add %0, %bcast : tensor<16x64xf32>
    %2 = stablehlo.reshape %1 : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    return %2 : tensor<4x4x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_THAT(RunTorchTpuOptimizationPasses(*mod, &context),
              absl_testing::IsOk());

  ASSERT_EQ(CountOps<mlir::stablehlo::AddOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::AddOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getRank(), 3);
    EXPECT_EQ(ty.getDimSize(0), 4);
    EXPECT_EQ(ty.getDimSize(1), 4);
    EXPECT_EQ(ty.getDimSize(2), 64);
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest, PipelineIntegrationGqaUnwrapTest) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @main(%q: tensor<1x8x16x64xf32>, %k: tensor<1x2x16x64xf32>)
      -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 3, 4]
        : (tensor<1x2x16x64xf32>) -> tensor<1x2x4x16x64xf32>
    %1 = stablehlo.reshape %0
        : (tensor<1x2x4x16x64xf32>) -> tensor<1x8x16x64xf32>
    %2 = stablehlo.dot_general %q, %1,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x16x64xf32>, tensor<1x8x16x64xf32>)
        -> tensor<1x8x16x16xf32>
    return %2 : tensor<1x8x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_THAT(RunTorchTpuOptimizationPasses(*mod, &context),
              absl_testing::IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 0);
}

TEST(CanonicalizeLayoutsAndReshapesTest, UnwrapGqaBroadcastsRank3Transposed) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Unbatched 3D GQA KV broadcast with transpose on last two dimensions.
  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_rank3_transpose(%q: tensor<8x16x64xf32>,
                                        %k: tensor<2x32x64xf32>)
      -> tensor<8x16x32xf32> {
    %0 = stablehlo.reshape %k
        : (tensor<2x32x64xf32>) -> tensor<2x1x32x64xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3]
        : (tensor<2x1x32x64xf32>) -> tensor<2x4x32x64xf32>
    %2 = stablehlo.reshape %1
        : (tensor<2x4x32x64xf32>) -> tensor<8x32x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1]
        : (tensor<8x32x64xf32>) -> tensor<8x64x32xf32>
    %4 = stablehlo.dot_general %q, %3,
        batching_dims = [0] x [0],
        contracting_dims = [2] x [1]
        : (tensor<8x16x64xf32>, tensor<8x64x32xf32>)
        -> tensor<8x16x32xf32>
    return %4 : tensor<8x16x32xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 0);

  auto q = MakeF32Tensor(context, {8, 16, 64}, 1.0f);
  auto k = MakeF32Tensor(context, {2, 32, 64}, 0.5f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {q, k});
}

TEST(CanonicalizeLayoutsAndReshapesTest, UnwrapGqaBroadcastsWithConvert) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // GQA KV broadcast with convert from bf16 to f32.
  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_convert(%q: tensor<1x8x16x64xf32>,
                                %k: tensor<1x2x16x64xbf16>)
      -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.convert %k
        : (tensor<1x2x16x64xbf16>) -> tensor<1x2x16x64xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 3, 4]
        : (tensor<1x2x16x64xf32>) -> tensor<1x2x4x16x64xf32>
    %2 = stablehlo.reshape %1
        : (tensor<1x2x4x16x64xf32>) -> tensor<1x8x16x64xf32>
    %3 = stablehlo.dot_general %q, %2,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x16x64xf32>, tensor<1x8x16x64xf32>)
        -> tensor<1x8x16x16xf32>
    return %3 : tensor<1x8x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConvertOp>(*mod), 1);
}

TEST(CanonicalizeLayoutsAndReshapesTest, UnwrapGqaBroadcastsSeenConvertLimit) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Two chained converts: f16 -> bf16 -> f32. StripOptionalConvert stops
  // stripping at the first non-noop cast because seen_convert is true.
  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_double_convert(%q: tensor<1x8x16x64xf32>,
                                       %k: tensor<1x2x16x64xf16>)
      -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.convert %k
        : (tensor<1x2x16x64xf16>) -> tensor<1x2x16x64xbf16>
    %1 = stablehlo.convert %0
        : (tensor<1x2x16x64xbf16>) -> tensor<1x2x16x64xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 3, 4]
        : (tensor<1x2x16x64xf32>) -> tensor<1x2x4x16x64xf32>
    %3 = stablehlo.reshape %2
        : (tensor<1x2x4x16x64xf32>) -> tensor<1x8x16x64xf32>
    %4 = stablehlo.dot_general %q, %3,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x16x64xf32>, tensor<1x8x16x64xf32>)
        -> tensor<1x8x16x16xf32>
    return %4 : tensor<1x8x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConvertOp>(*mod), 2);
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     UnwrapGqaBroadcastsZeroSequenceLength) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // s_q == 0 guard: GQA unwrapping bails out on empty sequence length.
  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_zero_seq(%q: tensor<1x8x0x64xf32>,
                                 %k: tensor<1x2x16x64xf32>)
      -> tensor<1x8x0x16xf32> {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 3, 4]
        : (tensor<1x2x16x64xf32>) -> tensor<1x2x4x16x64xf32>
    %1 = stablehlo.reshape %0
        : (tensor<1x2x4x16x64xf32>) -> tensor<1x8x16x64xf32>
    %2 = stablehlo.dot_general %q, %1,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x0x64xf32>, tensor<1x8x16x64xf32>)
        -> tensor<1x8x0x16xf32>
    return %2 : tensor<1x8x0x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 1);
}

TEST(CanonicalizeLayoutsAndReshapesTest, UnwrapGqaBroadcastsDynamicShapes) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // GQA unwrapping bails out on non-static tensor shapes.
  constexpr const char* kIr = R"mlir(
module {
  func.func @unwrap_gqa_dynamic(%q: tensor<1x8x?x64xf32>,
                                %k: tensor<1x2x16x64xf32>)
      -> tensor<1x8x?x16xf32> {
    %0 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 3, 4]
        : (tensor<1x2x16x64xf32>) -> tensor<1x2x4x16x64xf32>
    %1 = stablehlo.reshape %0
        : (tensor<1x2x4x16x64xf32>) -> tensor<1x8x16x64xf32>
    %2 = stablehlo.dot_general %q, %1,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x?x64xf32>, tensor<1x8x16x64xf32>)
        -> tensor<1x8x?x16xf32>
    return %2 : tensor<1x8x?x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 1);
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     UnwrapGqaBroadcastsNonConvergenceFailure) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // 11 chained group_size = 2 GQA broadcasts exceeding kMaxIterations = 10,
  // triggering signalPassFailure().
  constexpr const char* kIr = R"mlir(
module {
  func.func @non_converging(%q: tensor<1x2048x16x64xf32>,
                            %k0: tensor<1x1x16x64xf32>)
      -> tensor<1x2048x16x16xf32> {
    %b1 = stablehlo.broadcast_in_dim %k0, dims = [0, 1, 3, 4]
        : (tensor<1x1x16x64xf32>) -> tensor<1x1x2x16x64xf32>
    %k1 = stablehlo.reshape %b1
        : (tensor<1x1x2x16x64xf32>) -> tensor<1x2x16x64xf32>
    %b2 = stablehlo.broadcast_in_dim %k1, dims = [0, 1, 3, 4]
        : (tensor<1x2x16x64xf32>) -> tensor<1x2x2x16x64xf32>
    %k2 = stablehlo.reshape %b2
        : (tensor<1x2x2x16x64xf32>) -> tensor<1x4x16x64xf32>
    %b3 = stablehlo.broadcast_in_dim %k2, dims = [0, 1, 3, 4]
        : (tensor<1x4x16x64xf32>) -> tensor<1x4x2x16x64xf32>
    %k3 = stablehlo.reshape %b3
        : (tensor<1x4x2x16x64xf32>) -> tensor<1x8x16x64xf32>
    %b4 = stablehlo.broadcast_in_dim %k3, dims = [0, 1, 3, 4]
        : (tensor<1x8x16x64xf32>) -> tensor<1x8x2x16x64xf32>
    %k4 = stablehlo.reshape %b4
        : (tensor<1x8x2x16x64xf32>) -> tensor<1x16x16x64xf32>
    %b5 = stablehlo.broadcast_in_dim %k4, dims = [0, 1, 3, 4]
        : (tensor<1x16x16x64xf32>) -> tensor<1x16x2x16x64xf32>
    %k5 = stablehlo.reshape %b5
        : (tensor<1x16x2x16x64xf32>) -> tensor<1x32x16x64xf32>
    %b6 = stablehlo.broadcast_in_dim %k5, dims = [0, 1, 3, 4]
        : (tensor<1x32x16x64xf32>) -> tensor<1x32x2x16x64xf32>
    %k6 = stablehlo.reshape %b6
        : (tensor<1x32x2x16x64xf32>) -> tensor<1x64x16x64xf32>
    %b7 = stablehlo.broadcast_in_dim %k6, dims = [0, 1, 3, 4]
        : (tensor<1x64x16x64xf32>) -> tensor<1x64x2x16x64xf32>
    %k7 = stablehlo.reshape %b7
        : (tensor<1x64x2x16x64xf32>) -> tensor<1x128x16x64xf32>
    %b8 = stablehlo.broadcast_in_dim %k7, dims = [0, 1, 3, 4]
        : (tensor<1x128x16x64xf32>) -> tensor<1x128x2x16x64xf32>
    %k8 = stablehlo.reshape %b8
        : (tensor<1x128x2x16x64xf32>) -> tensor<1x256x16x64xf32>
    %b9 = stablehlo.broadcast_in_dim %k8, dims = [0, 1, 3, 4]
        : (tensor<1x256x16x64xf32>) -> tensor<1x256x2x16x64xf32>
    %k9 = stablehlo.reshape %b9
        : (tensor<1x256x2x16x64xf32>) -> tensor<1x512x16x64xf32>
    %b10 = stablehlo.broadcast_in_dim %k9, dims = [0, 1, 3, 4]
        : (tensor<1x512x16x64xf32>) -> tensor<1x512x2x16x64xf32>
    %k10 = stablehlo.reshape %b10
        : (tensor<1x512x2x16x64xf32>) -> tensor<1x1024x16x64xf32>
    %b11 = stablehlo.broadcast_in_dim %k10, dims = [0, 1, 3, 4]
        : (tensor<1x1024x16x64xf32>) -> tensor<1x1024x2x16x64xf32>
    %k11 = stablehlo.reshape %b11
        : (tensor<1x1024x2x16x64xf32>) -> tensor<1x2048x16x64xf32>
    %dot = stablehlo.dot_general %q, %k11,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x2048x16x64xf32>, tensor<1x2048x16x64xf32>)
        -> tensor<1x2048x16x16xf32>
    return %dot : tensor<1x2048x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  EXPECT_TRUE(mlir::failed(RunPass(*mod, &context)));
}

TEST(CanonicalizeLayoutsAndReshapesTest, CombinedGqaAndProjectionReshapeSink) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Combined test: K/V projection (dot -> add -> reshape) feeds into a GQA
  // KV broadcast dot. Both GQA unwrap and projection sink fire in same pass.
  constexpr const char* kIr = R"mlir(
module {
  func.func @combined_gqa_and_proj(%x: tensor<32x32xf32>,
                                   %w_k: tensor<32x64xf32>,
                                   %bias_k: tensor<64xf32>,
                                   %q: tensor<1x8x16x64xf32>)
      -> tensor<1x8x16x16xf32> {
    %0 = stablehlo.dot_general %x, %w_k,
        contracting_dims = [1] x [0]
        : (tensor<32x32xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %bias_bcast = stablehlo.broadcast_in_dim %bias_k, dims = [1]
        : (tensor<64xf32>) -> tensor<32x64xf32>
    %1 = stablehlo.add %0, %bias_bcast : tensor<32x64xf32>
    %k = stablehlo.reshape %1
        : (tensor<32x64xf32>) -> tensor<1x2x16x64xf32>
    %2 = stablehlo.broadcast_in_dim %k, dims = [0, 1, 3, 4]
        : (tensor<1x2x16x64xf32>) -> tensor<1x2x4x16x64xf32>
    %3 = stablehlo.reshape %2
        : (tensor<1x2x4x16x64xf32>) -> tensor<1x8x16x64xf32>
    %out = stablehlo.dot_general %q, %3,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<1x8x16x64xf32>, tensor<1x8x16x64xf32>)
        -> tensor<1x8x16x16xf32>
    return %out : tensor<1x8x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  // GQA broadcast eliminated; only the bias broadcast remains.
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 1);
  // Projection reshape sunk: AddOp operates on 4D tensor <1x2x16x64xf32>.
  ASSERT_EQ(CountOps<mlir::stablehlo::AddOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::AddOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getRank(), 4);
    EXPECT_EQ(ty.getShape(), llvm::ArrayRef<int64_t>({1, 2, 16, 64}));
  });

  auto x = MakeF32Tensor(context, {32, 32}, 0.5f);
  auto w_k = MakeF32Tensor(context, {32, 64}, 0.25f);
  auto bias_k = MakeF32Tensor(context, {64}, 1.0f);
  auto q = MakeF32Tensor(context, {1, 8, 16, 64}, 1.0f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {x, w_k, bias_k, q});
}

}  // namespace
}  // namespace torch_tpu
