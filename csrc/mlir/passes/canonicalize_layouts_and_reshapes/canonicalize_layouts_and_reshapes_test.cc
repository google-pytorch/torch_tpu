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

#include <memory>
#include <vector>

#include "absl/status/status_matchers.h"
#include "gtest/gtest.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"  // NOLINT
#include "mlir/Dialect/Func/IR/FuncOps.h"      // NOLINT
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOk;

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
// Mechanism 1: Fold Transpose(BroadcastInDim)
// =============================================================================

TEST(CanonicalizeLayoutsAndReshapesTest, FoldTransposeOfBroadcast) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @fold_transpose_bcast(%arg0: tensor<2x3xf32>)
      -> tensor<1x4x2x3xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 3]
        : (tensor<2x3xf32>) -> tensor<1x2x4x3xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3]
        : (tensor<1x2x4x3xf32>) -> tensor<1x4x2x3xf32>
    return %1 : tensor<1x4x2x3xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 1);

  auto arg0 = MakeF32Tensor(context, {2, 3}, 1.0f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {arg0});
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     FoldTransposeOfBroadcastNegativeDescending) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @fold_transpose_bcast_descending(%arg0: tensor<2x3xf32>)
      -> tensor<1x3x4x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 3]
        : (tensor<2x3xf32>) -> tensor<1x2x4x3xf32>
    %1 = stablehlo.transpose %0, dims = [0, 3, 2, 1]
        : (tensor<1x2x4x3xf32>) -> tensor<1x3x4x2xf32>
    return %1 : tensor<1x3x4x2xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  // Descending permutation cannot fold without data permutation.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
}

// =============================================================================
// Mechanism 2: Commute Pointwise DAG Across Transpose
// =============================================================================

TEST(CanonicalizeLayoutsAndReshapesTest,
     CommutePointwiseDagAcrossTransposeRank4) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_pointwise(%arg0: tensor<1x16x16x32xf32>)
      -> tensor<1x32x16x16xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2]
        : (tensor<1x16x16x32xf32>) -> tensor<1x32x16x16xf32>
    %1 = stablehlo.exponential %0 : tensor<1x32x16x16xf32>
    return %1 : tensor<1x32x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::ExpOp>(*mod), 1);

  // Exp commuted before transpose: exponential input is 1x16x16x32.
  mod->walk([](mlir::stablehlo::ExpOp op) {
    const auto ty =
        llvm::cast<mlir::RankedTensorType>(op.getOperand().getType());
    EXPECT_EQ(ty.getDimSize(1), 16);
    EXPECT_EQ(ty.getDimSize(3), 32);
  });

  auto arg0 = MakeF32Tensor(context, {1, 16, 16, 32}, 0.5f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {arg0});
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     CommutePointwiseDagAcrossTransposeRank3) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_pointwise_rank3(%arg0: tensor<4x8x16xf32>)
      -> tensor<4x16x8xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1]
        : (tensor<4x8x16xf32>) -> tensor<4x16x8xf32>
    %1 = stablehlo.negate %0 : tensor<4x16x8xf32>
    return %1 : tensor<4x16x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  ASSERT_EQ(CountOps<mlir::stablehlo::NegOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::NegOp op) {
    const auto ty =
        llvm::cast<mlir::RankedTensorType>(op.getOperand().getType());
    EXPECT_EQ(ty.getDimSize(1), 8);
    EXPECT_EQ(ty.getDimSize(2), 16);
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     CommutePointwiseDagNegativeCostModelWidening) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_negative_widening(%arg0: tensor<1x16x16x32xf16>)
      -> tensor<1x32x16x16xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2]
        : (tensor<1x16x16x32xf16>) -> tensor<1x32x16x16xf16>
    %1 = stablehlo.convert %0
        : (tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf32>
    return %1 : tensor<1x32x16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  // Widening f16 -> f32 increases volume before transpose; must be rejected.
  ASSERT_EQ(CountOps<mlir::stablehlo::ConvertOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::ConvertOp op) {
    const auto ty =
        llvm::cast<mlir::RankedTensorType>(op.getOperand().getType());
    EXPECT_EQ(ty.getDimSize(1), 32);
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     CommutePointwiseDagComplexDoesNotCrash) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_pointwise_complex(%arg0: tensor<4x8x16xcomplex<f32>>)
      -> tensor<4x16x8xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1]
        : (tensor<4x8x16xcomplex<f32>>) -> tensor<4x16x8xcomplex<f32>>
    %1 = stablehlo.negate %0 : tensor<4x16x8xcomplex<f32>>
    return %1 : tensor<4x16x8xcomplex<f32>>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  // Must handle complex element types without compiler crash and rewrite.
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  ASSERT_EQ(CountOps<mlir::stablehlo::NegOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::NegOp op) {
    const auto ty =
        llvm::cast<mlir::RankedTensorType>(op.getOperand().getType());
    EXPECT_EQ(ty.getDimSize(1), 8);
    EXPECT_EQ(ty.getDimSize(2), 16);
  });
}

// =============================================================================
// Mechanism 3: Unwrap GQA Broadcasts
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

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*mod), 0);
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

// =============================================================================
// Mechanism 4: Commute Token Permutation Across Linear Projections
// =============================================================================

TEST(CanonicalizeLayoutsAndReshapesTest, CommuteTokenPermutationAcrossLinear) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_token_perm(%x: tensor<4x4x32xf32>,
                                %w: tensor<32x64xf32>)
      -> tensor<16x64xf32> {
    %x_2d = stablehlo.reshape %x : (tensor<4x4x32xf32>) -> tensor<16x32xf32>
    %proj = stablehlo.dot %x_2d, %w
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %r_in = stablehlo.reshape %proj
        : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    %tr = stablehlo.transpose %r_in, dims = [1, 0, 2]
        : (tensor<4x4x64xf32>) -> tensor<4x4x64xf32>
    %r_out = stablehlo.reshape %tr
        : (tensor<4x4x64xf32>) -> tensor<16x64xf32>
    return %r_out : tensor<16x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::TransposeOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getDimSize(2), 32);
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest, CommuteTokenPermutationDotGeneral) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_token_perm_dg(%x: tensor<4x4x32xf32>,
                                   %w: tensor<32x64xf32>)
      -> tensor<16x64xf32> {
    %x_2d = stablehlo.reshape %x : (tensor<4x4x32xf32>) -> tensor<16x32xf32>
    %proj = stablehlo.dot_general %x_2d, %w,
        contracting_dims = [1] x [0]
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %r_in = stablehlo.reshape %proj
        : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    %tr = stablehlo.transpose %r_in, dims = [1, 0, 2]
        : (tensor<4x4x64xf32>) -> tensor<4x4x64xf32>
    %r_out = stablehlo.reshape %tr
        : (tensor<4x4x64xf32>) -> tensor<16x64xf32>
    return %r_out : tensor<16x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::TransposeOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getDimSize(2), 32);
  });

  auto x = MakeF32Tensor(context, {4, 4, 32}, 1.0f);
  auto w = MakeF32Tensor(context, {32, 64}, 0.5f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {x, w});
}

TEST(CanonicalizeLayoutsAndReshapesTest, CommuteTokenPermutationWithBias) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_token_perm_bias(%x: tensor<4x4x32xf32>,
                                     %w: tensor<32x64xf32>,
                                     %bias: tensor<64xf32>)
      -> tensor<16x64xf32> {
    %x_2d = stablehlo.reshape %x : (tensor<4x4x32xf32>) -> tensor<16x32xf32>
    %proj = stablehlo.dot %x_2d, %w
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<64xf32>) -> tensor<16x64xf32>
    %add = stablehlo.add %proj, %bcast : tensor<16x64xf32>
    %r_in = stablehlo.reshape %add
        : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    %tr = stablehlo.transpose %r_in, dims = [1, 0, 2]
        : (tensor<4x4x64xf32>) -> tensor<4x4x64xf32>
    %r_out = stablehlo.reshape %tr
        : (tensor<4x4x64xf32>) -> tensor<16x64xf32>
    return %r_out : tensor<16x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::TransposeOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getDimSize(2), 32);
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest, CommuteTokenPermutationQkvSharing) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // x_2d feeds 3 projections (Q, K, V).
  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_token_perm_qkv(%x: tensor<4x4x32xf32>,
                                    %wq: tensor<32x64xf32>,
                                    %wk: tensor<32x64xf32>,
                                    %wv: tensor<32x64xf32>)
      -> (tensor<16x64xf32>, tensor<16x64xf32>, tensor<16x64xf32>) {
    %x_2d = stablehlo.reshape %x : (tensor<4x4x32xf32>) -> tensor<16x32xf32>

    %q = stablehlo.dot_general %x_2d, %wq,
        contracting_dims = [1] x [0]
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %rq_in = stablehlo.reshape %q : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    %tq = stablehlo.transpose %rq_in, dims = [1, 0, 2]
        : (tensor<4x4x64xf32>) -> tensor<4x4x64xf32>
    %rq_out = stablehlo.reshape %tq : (tensor<4x4x64xf32>) -> tensor<16x64xf32>

    %k = stablehlo.dot_general %x_2d, %wk,
        contracting_dims = [1] x [0]
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %rk_in = stablehlo.reshape %k : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    %tk = stablehlo.transpose %rk_in, dims = [1, 0, 2]
        : (tensor<4x4x64xf32>) -> tensor<4x4x64xf32>
    %rk_out = stablehlo.reshape %tk : (tensor<4x4x64xf32>) -> tensor<16x64xf32>

    %v = stablehlo.dot_general %x_2d, %wv,
        contracting_dims = [1] x [0]
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %rv_in = stablehlo.reshape %v : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    %tv = stablehlo.transpose %rv_in, dims = [1, 0, 2]
        : (tensor<4x4x64xf32>) -> tensor<4x4x64xf32>
    %rv_out = stablehlo.reshape %tv : (tensor<4x4x64xf32>) -> tensor<16x64xf32>

    return %rq_out, %rk_out, %rv_out
        : tensor<16x64xf32>, tensor<16x64xf32>, tensor<16x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  // Must succeed without UAF on x_2d!
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  int transformed = 0;
  mod->walk([&transformed](mlir::stablehlo::TransposeOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    if (ty.getDimSize(2) == 32) {
      ++transformed;
    }
  });
  EXPECT_EQ(transformed, 3);

  auto x = MakeF32Tensor(context, {4, 4, 32}, 1.0f);
  auto wq = MakeF32Tensor(context, {32, 64}, 0.25f);
  auto wk = MakeF32Tensor(context, {32, 64}, 0.5f);
  auto wv = MakeF32Tensor(context, {32, 64}, 0.75f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {x, wq, wk, wv});
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     CommuteTokenPermutationNegativeTokenBias) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Bias broadcast along token dimension {0}, not channel dimension {1}.
  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_token_perm_neg_bias(%x: tensor<4x4x32xf32>,
                                         %w: tensor<32x64xf32>,
                                         %bias: tensor<16xf32>)
      -> tensor<16x64xf32> {
    %x_2d = stablehlo.reshape %x : (tensor<4x4x32xf32>) -> tensor<16x32xf32>
    %proj = stablehlo.dot %x_2d, %w
        : (tensor<16x32xf32>, tensor<32x64xf32>) -> tensor<16x64xf32>
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [0]
        : (tensor<16xf32>) -> tensor<16x64xf32>
    %add = stablehlo.add %proj, %bcast : tensor<16x64xf32>
    %r_in = stablehlo.reshape %add
        : (tensor<16x64xf32>) -> tensor<4x4x64xf32>
    %tr = stablehlo.transpose %r_in, dims = [1, 0, 2]
        : (tensor<4x4x64xf32>) -> tensor<4x4x64xf32>
    %r_out = stablehlo.reshape %tr
        : (tensor<4x4x64xf32>) -> tensor<16x64xf32>
    return %r_out : tensor<16x64xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  // Per-token bias cannot commute across permutation.
  ASSERT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::TransposeOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getDimSize(2), 64);
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     CommuteTokenPermutationNegativeCostModel) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Expansion projection: c_in=64 > c_out=16.
  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_token_perm_cost_negative(%x: tensor<4x4x64xf32>,
                                              %w: tensor<64x16xf32>)
      -> tensor<16x16xf32> {
    %x_2d = stablehlo.reshape %x : (tensor<4x4x64xf32>) -> tensor<16x64xf32>
    %proj = stablehlo.dot %x_2d, %w
        : (tensor<16x64xf32>, tensor<64x16xf32>) -> tensor<16x16xf32>
    %r_in = stablehlo.reshape %proj
        : (tensor<16x16xf32>) -> tensor<4x4x16xf32>
    %tr = stablehlo.transpose %r_in, dims = [1, 0, 2]
        : (tensor<4x4x16xf32>) -> tensor<4x4x16xf32>
    %r_out = stablehlo.reshape %tr
        : (tensor<4x4x16xf32>) -> tensor<16x16xf32>
    return %r_out : tensor<16x16xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  ASSERT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::TransposeOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getDimSize(2), 16);
  });
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     CommuteTokenPermutationComplexDoesNotCrash) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @commute_token_perm_complex(%x: tensor<4x4x32xcomplex<f32>>,
                                        %w: tensor<32x64xcomplex<f32>>)
      -> tensor<16x64xcomplex<f32>> {
    %x_2d = stablehlo.reshape %x
        : (tensor<4x4x32xcomplex<f32>>) -> tensor<16x32xcomplex<f32>>
    %proj = stablehlo.dot %x_2d, %w
        : (tensor<16x32xcomplex<f32>>, tensor<32x64xcomplex<f32>>)
        -> tensor<16x64xcomplex<f32>>
    %r_in = stablehlo.reshape %proj
        : (tensor<16x64xcomplex<f32>>) -> tensor<4x4x64xcomplex<f32>>
    %tr = stablehlo.transpose %r_in, dims = [1, 0, 2]
        : (tensor<4x4x64xcomplex<f32>>) -> tensor<4x4x64xcomplex<f32>>
    %r_out = stablehlo.reshape %tr
        : (tensor<4x4x64xcomplex<f32>>) -> tensor<16x64xcomplex<f32>>
    return %r_out : tensor<16x64xcomplex<f32>>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  // Must handle complex bitwidth gracefully without compiler abort and rewrite.
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::TransposeOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(ty.getDimSize(2), 32);
  });
}

// =============================================================================
// Mechanism 5: Flatten Spatial Reduction
// =============================================================================

TEST(CanonicalizeLayoutsAndReshapesTest, FlattenSpatialReductionRank3) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @flatten_reduction_rank3(%x: tensor<2x4x8xf32>) -> tensor<8x8xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %red = stablehlo.reduce(%x init: %zero) across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
      reducer(%in0: tensor<f32>, %in1: tensor<f32>) {
        %sum = stablehlo.add %in0, %in1 : tensor<f32>
        stablehlo.return %sum : tensor<f32>
      }
    %bcast = stablehlo.broadcast_in_dim %red, dims = [0, 1]
        : (tensor<2x4xf32>) -> tensor<2x4x8xf32>
    %sub = stablehlo.subtract %x, %bcast : tensor<2x4x8xf32>
    %flat = stablehlo.reshape %sub : (tensor<2x4x8xf32>) -> tensor<8x8xf32>
    return %flat : tensor<8x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::ReduceOp op) {
    const auto arg_ty =
        llvm::cast<mlir::RankedTensorType>(op.getInputs()[0].getType());
    EXPECT_EQ(arg_ty.getRank(), 2);
    EXPECT_EQ(arg_ty.getDimSize(0), 8);
    EXPECT_EQ(arg_ty.getDimSize(1), 8);
    EXPECT_EQ(op.getDimensions().size(), 1);
    EXPECT_EQ(op.getDimensions()[0], 1);
  });

  auto x = MakeF32Tensor(context, {2, 4, 8}, 1.0f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {x});
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     FlattenSpatialReductionRank4MomentsDag) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  // Normal layernorm lowering where moments DAG operates on reduced stats
  // shape.
  constexpr const char* kIr = R"mlir(
module {
  func.func @layernorm_moments_dag(%x: tensor<1x2x4x8xf32>) -> tensor<8x8xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%x init: %zero) across dimensions = [3]
        : (tensor<1x2x4x8xf32>, tensor<f32>) -> tensor<1x2x4xf32>
      reducer(%in0: tensor<f32>, %in1: tensor<f32>) {
        %add = stablehlo.add %in0, %in1 : tensor<f32>
        stablehlo.return %add : tensor<f32>
      }
    %c_inv = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %c_inv_bcast = stablehlo.broadcast_in_dim %c_inv, dims = []
        : (tensor<f32>) -> tensor<1x2x4xf32>
    %mean = stablehlo.multiply %sum, %c_inv_bcast : tensor<1x2x4xf32>
    %mean_bcast = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2]
        : (tensor<1x2x4xf32>) -> tensor<1x2x4x8xf32>
    %diff = stablehlo.subtract %x, %mean_bcast : tensor<1x2x4x8xf32>
    %flat = stablehlo.reshape %diff
        : (tensor<1x2x4x8xf32>) -> tensor<8x8xf32>
    return %flat : tensor<8x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  ASSERT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*mod), 1);
  ASSERT_EQ(CountOps<mlir::stablehlo::MulOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::ReduceOp op) {
    const auto arg_ty =
        llvm::cast<mlir::RankedTensorType>(op.getInputs()[0].getType());
    EXPECT_EQ(arg_ty.getRank(), 2);
    EXPECT_EQ(arg_ty.getDimSize(0), 8);
    EXPECT_EQ(arg_ty.getDimSize(1), 8);
  });
  mod->walk([](mlir::stablehlo::MulOp op) {
    const auto res_ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(res_ty.getRank(), 1);
    EXPECT_EQ(res_ty.getDimSize(0), 8);
  });

  auto x = MakeF32Tensor(context, {1, 2, 4, 8}, 1.0f);
  AssertModulesEvaluateNumericallyEqual(*orig_mod, *mod, {x});
}

TEST(CanonicalizeLayoutsAndReshapesTest,
     FlattenSpatialReductionNegativeEscapingMoment) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @escaping_reduction(%x: tensor<2x4x8xf32>, %eps: tensor<f32>)
      -> (tensor<8x8xf32>, tensor<2x4xf32>) {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %red = stablehlo.reduce(%x init: %zero) across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
      reducer(%in0: tensor<f32>, %in1: tensor<f32>) {
        %sum = stablehlo.add %in0, %in1 : tensor<f32>
        stablehlo.return %sum : tensor<f32>
      }
    %eps_bcast = stablehlo.broadcast_in_dim %eps, dims = []
        : (tensor<f32>) -> tensor<2x4xf32>
    %added = stablehlo.add %red, %eps_bcast : tensor<2x4xf32>
    %rsqrt = stablehlo.rsqrt %added : tensor<2x4xf32>
    %bcast = stablehlo.broadcast_in_dim %rsqrt, dims = [0, 1]
        : (tensor<2x4xf32>) -> tensor<2x4x8xf32>
    %norm = stablehlo.multiply %x, %bcast : tensor<2x4x8xf32>
    %flat = stablehlo.reshape %norm
        : (tensor<2x4x8xf32>) -> tensor<8x8xf32>
    return %flat, %red : tensor<8x8xf32>, tensor<2x4xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  // Escaping intermediate moment %red prevents flattening.
  ASSERT_EQ(CountOps<mlir::stablehlo::ReduceOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::ReduceOp op) {
    EXPECT_EQ(op.getDimensions().size(), 1);
    EXPECT_EQ(op.getDimensions()[0], 2);
  });
}

// =============================================================================
// Mechanism 6: Sink Projection Reshapes
// =============================================================================

TEST(CanonicalizeLayoutsAndReshapesTest,
     SinkProjectionReshapesElementwiseChain) {
  mlir::MLIRContext context;
  InitMlirContext(context);

  constexpr const char* kIr = R"mlir(
module {
  func.func @sink_proj_reshape(%x: tensor<16x32xf32>,
                               %w: tensor<32x64xf32>,
                               %bias: tensor<64xf32>)
      -> tensor<4x4x64xf32> {
    %0 = stablehlo.dot %x, %w
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
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));

  ASSERT_EQ(CountOps<mlir::stablehlo::AddOp>(*mod), 1);
  mod->walk([](mlir::stablehlo::AddOp op) {
    const auto ty = llvm::cast<mlir::RankedTensorType>(op.getType());
    ASSERT_EQ(ty.getRank(), 3);
    EXPECT_EQ(ty.getDimSize(0), 4);
    EXPECT_EQ(ty.getDimSize(1), 4);
    EXPECT_EQ(ty.getDimSize(2), 64);
  });
}

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

}  // namespace
}  // namespace torch_tpu
