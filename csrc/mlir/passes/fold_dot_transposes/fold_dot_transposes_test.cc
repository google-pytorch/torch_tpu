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

#include "csrc/mlir/passes/fold_dot_transposes/fold_dot_transposes.h"  // NOLINT

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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
  module.walk([&count](OpTy) { ++count; });
  return count;
}

bool RunPass(mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());
  pm.addNestedPass<mlir::func::FuncOp>(createFoldDotTransposesPass());
  return mlir::succeeded(pm.run(module));
}

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      const float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 17 + 5) % 251) * 0.03125f -
              static_cast<float>((i * 7) % 13) * 0.0625f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeI1Tensor(mlir::MLIRContext& ctx,
                                     llvm::ArrayRef<int64_t> shape) {
  auto type =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 1));
  const int64_t n = type.getNumElements();
  auto vals = std::make_unique<bool[]>(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = ((i * 3 + 1) % 5) != 0;
  }
  return mlir::DenseElementsAttr::get(type,
                                      llvm::ArrayRef<bool>(vals.get(), n));
}

void CheckDenseTensorNear(mlir::DenseElementsAttr expected,
                          mlir::DenseElementsAttr actual,
                          const float tol = 1e-4f) {
  ASSERT_EQ(expected.getNumElements(), actual.getNumElements());
  auto exp_it = expected.value_begin<float>();
  auto act_it = actual.value_begin<float>();
  for (; exp_it != expected.value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*exp_it, *act_it, tol);
  }
}

TEST(FoldDotTransposesTest, JointForwardBackwardHeadMergedContractionNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<1x2x8x128xf32>,
      %w: tensor<256x128xf32>,
      %dy: tensor<128x8xf32>) -> (tensor<8x128xf32>, tensor<128x256xf32>) {
    %t = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x2x8x128xf32>) -> tensor<8x2x128x1xf32>
    %r = stablehlo.reshape %t : (tensor<8x2x128x1xf32>) -> tensor<8x256xf32>
    %y = "stablehlo.dot_general"(%r, %w) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<8x256xf32>, tensor<256x128xf32>) -> tensor<8x128xf32>
    %dw = "stablehlo.dot_general"(%dy, %r) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<128x8xf32>, tensor<8x256xf32>) -> tensor<128x256xf32>
    return %y, %dw : tensor<8x128xf32>, tensor<128x256xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {1, 2, 8, 128}, 0.5f);
  mlir::DenseElementsAttr w = MakeF32Tensor(context, {256, 128}, 0.25f);
  mlir::DenseElementsAttr dy = MakeF32Tensor(context, {128, 8}, 0.1f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {x, w, dy}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPass(*opt_mod));
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 2);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, w, dy}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), actual->size());
  CheckDenseTensorNear((*expected)[0], (*actual)[0]);
  CheckDenseTensorNear((*expected)[1], (*actual)[1]);
}

TEST(FoldDotTransposesTest, FoldTransposeReshapeDotNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(
      %dy: tensor<2x2x4x128xf32>,
      %w: tensor<256x128xf32>) -> tensor<2x4x128xf32> {
    %t = stablehlo.transpose %dy, dims = [1, 2, 0, 3]
        : (tensor<2x2x4x128xf32>) -> tensor<2x4x2x128xf32>
    %r = stablehlo.reshape %t : (tensor<2x4x2x128xf32>) -> tensor<8x256xf32>
    %dot = "stablehlo.dot_general"(%r, %w) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<8x256xf32>, tensor<256x128xf32>) -> tensor<8x128xf32>
    %out = stablehlo.reshape %dot : (tensor<8x128xf32>) -> tensor<2x4x128xf32>
    return %out : tensor<2x4x128xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr dy = MakeF32Tensor(context, {2, 2, 4, 128}, 0.05f);
  mlir::DenseElementsAttr w = MakeF32Tensor(context, {256, 128}, 0.02f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {dy, w}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPass(*opt_mod));
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {dy, w}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), actual->size());
  CheckDenseTensorNear((*expected)[0], (*actual)[0]);
}

TEST(FoldDotTransposesTest, UnflattenBatchedDotGeneral4DNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(
      %q: tensor<2x8x16x64xf32>,
      %k: tensor<2x16x64x8xf32>) -> tensor<2x16x8x8xf32> {
    %tr_q = stablehlo.transpose %q, dims = [0, 2, 1, 3]
        : (tensor<2x8x16x64xf32>) -> tensor<2x16x8x64xf32>
    %r_q = stablehlo.reshape %tr_q
        : (tensor<2x16x8x64xf32>) -> tensor<32x8x64xf32>
    %r_k = stablehlo.reshape %k : (tensor<2x16x64x8xf32>) -> tensor<32x64x8xf32>
    %dot = "stablehlo.dot_general"(%r_q, %r_k) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_batching_dimensions = [0],
            rhs_batching_dimensions = [0],
            lhs_contracting_dimensions = [2],
            rhs_contracting_dimensions = [1]>
    } : (tensor<32x8x64xf32>, tensor<32x64x8xf32>) -> tensor<32x8x8xf32>
    %out = stablehlo.reshape %dot : (tensor<32x8x8xf32>) -> tensor<2x16x8x8xf32>
    return %out : tensor<2x16x8x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr q = MakeF32Tensor(context, {2, 8, 16, 64}, 0.1f);
  mlir::DenseElementsAttr k = MakeF32Tensor(context, {2, 16, 64, 8}, 0.2f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {q, k}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPass(*opt_mod));
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {q, k}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), actual->size());
  CheckDenseTensorNear((*expected)[0], (*actual)[0]);
}

TEST(FoldDotTransposesTest, NegativeCostModelRejectionPreservesOperations) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<1x8x4x128xf32>,
      %w: tensor<512x1024xf32>) -> tensor<8x1024xf32> {
    %tr = stablehlo.transpose %x, dims = [0, 1, 2, 3]
        : (tensor<1x8x4x128xf32>) -> tensor<1x8x4x128xf32>
    %r = stablehlo.reshape %tr : (tensor<1x8x4x128xf32>) -> tensor<8x512xf32>
    %out = "stablehlo.dot"(%r, %w)
        : (tensor<8x512xf32>, tensor<512x1024xf32>) -> tensor<8x1024xf32>
    return %out : tensor<8x1024xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);

  EXPECT_EQ(CountOps<mlir::stablehlo::DotOp>(*mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);

  ASSERT_TRUE(RunPass(*mod));

  EXPECT_EQ(CountOps<mlir::stablehlo::DotOp>(*mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
}

// =============================================================================
// Unit Tests for Audit Bug Fixes
// =============================================================================

TEST(FoldDotTransposesTest,
     ErasePeeledTransposesHandlesDuplicateEntriesSafely) {
  mlir::MLIRContext context;
  InitContext(context);

  // Shared %wt feeds two dots consuming the same transposed-reshaped %x.
  // Both contraction rewrites register %wt into peeled_trs; deduplication
  // must prevent duplicate erase UAF. A backward %dw contraction provides
  // the joint fwd-bwd eligibility.
  const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<1x2x256x128xf32>,
      %w: tensor<128x256xf32>,
      %dy: tensor<128x256xf32>)
      -> (tensor<256x128xf32>, tensor<256x128xf32>, tensor<128x256xf32>) {
    %t = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x2x256x128xf32>) -> tensor<256x2x128x1xf32>
    %r = stablehlo.reshape %t : (tensor<256x2x128x1xf32>) -> tensor<256x256xf32>
    %wt = stablehlo.transpose %w, dims = [1, 0]
        : (tensor<128x256xf32>) -> tensor<256x128xf32>
    %y1 = "stablehlo.dot_general"(%r, %wt) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<256x256xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %y2 = "stablehlo.dot_general"(%r, %wt) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<256x256xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %dw = "stablehlo.dot_general"(%dy, %r) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<128x256xf32>, tensor<256x256xf32>) -> tensor<128x256xf32>
    return %y1, %y2, %dw
        : tensor<256x128xf32>, tensor<256x128xf32>, tensor<128x256xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(RunPass(*mod));
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*mod), 3);
}

TEST(FoldDotTransposesTest,
     PeelIdentityBroadcastRejectsSquareTransposePermutation) {
  mlir::MLIRContext context;
  InitContext(context);

  // A square 8x8 broadcast with non-identity dimensions [0, 1, 3, 2] is a
  // transpose, NOT an identity broadcast, and must not be peeled away.
  const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<2x2x8x8xf32>,
      %w: tensor<16x128xf32>) -> tensor<8x2x128xf32> {
    %b = stablehlo.broadcast_in_dim %x, dims = [0, 1, 3, 2]
        : (tensor<2x2x8x8xf32>) -> tensor<2x2x8x8xf32>
    %t = stablehlo.transpose %b, dims = [2, 1, 0, 3]
        : (tensor<2x2x8x8xf32>) -> tensor<8x2x2x8xf32>
    %r = stablehlo.reshape %t
        : (tensor<8x2x2x8xf32>) -> tensor<16x16xf32>
    %dot = "stablehlo.dot_general"(%r, %w) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<16x16xf32>, tensor<16x128xf32>) -> tensor<16x128xf32>
    %out = stablehlo.reshape %dot
        : (tensor<16x128xf32>) -> tensor<8x2x128xf32>
    return %out : tensor<8x2x128xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {2, 2, 8, 8}, 0.1f);
  mlir::DenseElementsAttr w = MakeF32Tensor(context, {16, 128}, 0.2f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x, w}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPass(*opt_mod));
  // BroadcastInDimOp must NOT be peeled/eliminated.
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, w}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), actual->size());
  CheckDenseTensorNear((*expected)[0], (*actual)[0]);
}

TEST(FoldDotTransposesTest, BuildUnflattenedOtherPreservesPredHardwareMasking) {
  mlir::MLIRContext context;
  InitContext(context);

  // Other operand is converted from predicate i1 mask. Unflattening must
  // preserve the ConvertOp directly preceding the dot contraction.
  const char* kIr = R"mlir(
module {
  func.func @pred_mask(
      %x: tensor<1x2x8x128xf32>,
      %mask: tensor<256x128xi1>,
      %dy: tensor<128x8xf32>)
      -> (tensor<8x128xf32>, tensor<128x256xf32>) {
    %t = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x2x8x128xf32>) -> tensor<8x2x128x1xf32>
    %r = stablehlo.reshape %t : (tensor<8x2x128x1xf32>) -> tensor<8x256xf32>
    %w = stablehlo.convert %mask : (tensor<256x128xi1>) -> tensor<256x128xf32>
    %y = "stablehlo.dot_general"(%r, %w) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<8x256xf32>, tensor<256x128xf32>) -> tensor<8x128xf32>
    %dw = "stablehlo.dot_general"(%dy, %r) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<128x8xf32>, tensor<8x256xf32>) -> tensor<128x256xf32>
    return %y, %dw : tensor<8x128xf32>, tensor<128x256xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {1, 2, 8, 128}, 0.5f);
  mlir::DenseElementsAttr mask = MakeI1Tensor(context, {256, 128});
  mlir::DenseElementsAttr dy = MakeF32Tensor(context, {128, 8}, 0.1f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {x, mask, dy}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPass(*opt_mod));
  // Transpose folded, dot general emitted, and ConvertOp preserved on i1 pred.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConvertOp>(*opt_mod), 1);

  mlir::stablehlo::DotGeneralOp dg_y = nullptr;
  opt_mod->walk([&](mlir::stablehlo::DotGeneralOp op) {
    auto out_ty = llvm::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (out_ty && out_ty.getShape() == llvm::ArrayRef<int64_t>({8, 128})) {
      dg_y = op;
    }
  });
  ASSERT_NE(dg_y, nullptr);
  auto cvt_op = dg_y.getRhs().getDefiningOp<mlir::stablehlo::ConvertOp>();
  ASSERT_NE(cvt_op, nullptr);
  auto cvt_type = llvm::dyn_cast<mlir::RankedTensorType>(cvt_op.getType());
  ASSERT_NE(cvt_type, nullptr);
  EXPECT_EQ(cvt_type.getShape(), llvm::ArrayRef<int64_t>({2, 128, 128}));
  EXPECT_TRUE(cvt_type.getElementType().isF32());
  auto pred_ty =
      llvm::dyn_cast<mlir::RankedTensorType>(cvt_op.getOperand().getType());
  ASSERT_NE(pred_ty, nullptr);
  EXPECT_TRUE(pred_ty.getElementType().isInteger(1));
  EXPECT_EQ(pred_ty.getShape(), llvm::ArrayRef<int64_t>({2, 128, 128}));

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {x, mask, dy}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), actual->size());
  CheckDenseTensorNear((*expected)[0], (*actual)[0]);
  CheckDenseTensorNear((*expected)[1], (*actual)[1]);
}

TEST(FoldDotTransposesTest, Phase3BiasAddFusionWithoutPostDotPermNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(
      %dy: tensor<2x2x4x128xf32>,
      %w: tensor<256x128xf32>,
      %bias: tensor<128xf32>) -> tensor<2x4x128xf32> {
    %t = stablehlo.transpose %dy, dims = [1, 2, 0, 3]
        : (tensor<2x2x4x128xf32>) -> tensor<2x4x2x128xf32>
    %r = stablehlo.reshape %t : (tensor<2x4x2x128xf32>) -> tensor<8x256xf32>
    %dot = "stablehlo.dot_general"(%r, %w) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<8x256xf32>, tensor<256x128xf32>) -> tensor<8x128xf32>
    %bias_bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<128xf32>) -> tensor<8x128xf32>
    %add = stablehlo.add %dot, %bias_bcast : tensor<8x128xf32>
    %out = stablehlo.reshape %add : (tensor<8x128xf32>) -> tensor<2x4x128xf32>
    return %out : tensor<2x4x128xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr dy = MakeF32Tensor(context, {2, 2, 4, 128}, 0.05f);
  mlir::DenseElementsAttr w = MakeF32Tensor(context, {256, 128}, 0.02f);
  mlir::DenseElementsAttr bias = MakeF32Tensor(context, {128}, 0.01f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {dy, w, bias}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPass(*opt_mod));
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*opt_mod), 1);

  mlir::stablehlo::AddOp add_op = nullptr;
  opt_mod->walk([&](mlir::stablehlo::AddOp op) { add_op = op; });
  ASSERT_NE(add_op, nullptr);
  auto add_type = llvm::dyn_cast<mlir::RankedTensorType>(add_op.getType());
  ASSERT_NE(add_type, nullptr);
  EXPECT_EQ(add_type.getShape(), llvm::ArrayRef<int64_t>({2, 4, 128}));

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {dy, w, bias}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), actual->size());
  CheckDenseTensorNear((*expected)[0], (*actual)[0]);
}

TEST(FoldDotTransposesTest, Phase3BiasAddFusionWithPostDotPermNumerics) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<2x4x2x128xf32>,
      %w: tensor<256x128xf32>,
      %bias: tensor<128xf32>) -> tensor<2x4x128xf32> {
    %t = stablehlo.transpose %x, dims = [2, 1, 0, 3]
        : (tensor<2x4x2x128xf32>) -> tensor<2x4x2x128xf32>
    %r = stablehlo.reshape %t : (tensor<2x4x2x128xf32>) -> tensor<8x256xf32>
    %dot = "stablehlo.dot_general"(%r, %w) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<8x256xf32>, tensor<256x128xf32>) -> tensor<8x128xf32>
    %bias_bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<128xf32>) -> tensor<8x128xf32>
    %add = stablehlo.add %dot, %bias_bcast : tensor<8x128xf32>
    %out = stablehlo.reshape %add : (tensor<8x128xf32>) -> tensor<2x4x128xf32>
    return %out : tensor<2x4x128xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {2, 4, 2, 128}, 0.05f);
  mlir::DenseElementsAttr w = MakeF32Tensor(context, {256, 128}, 0.02f);
  mlir::DenseElementsAttr bias = MakeF32Tensor(context, {128}, 0.01f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {x, w, bias}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(RunPass(*opt_mod));
  // TransposeOp exists for post_dot_perm.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*opt_mod), 1);

  mlir::stablehlo::AddOp add_op = nullptr;
  opt_mod->walk([&](mlir::stablehlo::AddOp op) { add_op = op; });
  ASSERT_NE(add_op, nullptr);
  auto add_type = llvm::dyn_cast<mlir::RankedTensorType>(add_op.getType());
  ASSERT_NE(add_type, nullptr);
  // Add operates on pre-transpose dot output shape [4, 2, 128].
  EXPECT_EQ(add_type.getShape(), llvm::ArrayRef<int64_t>({4, 2, 128}));

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, w, bias}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), actual->size());
  CheckDenseTensorNear((*expected)[0], (*actual)[0]);
}

TEST(FoldDotTransposesTest, Phase2CustomCallSingleDirectionAcceptedWhenMGeN) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(%w: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %x = stablehlo.custom_call @opaque_source()
        : () -> tensor<1x2x256x128xf32>
    %t = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x2x256x128xf32>) -> tensor<256x2x128x1xf32>
    %r = stablehlo.reshape %t
        : (tensor<256x2x128x1xf32>) -> tensor<256x256xf32>
    %y = "stablehlo.dot_general"(%r, %w) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<256x256xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    return %y : tensor<256x128xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(RunPass(*mod));
  // M (256) >= N (128): single-direction folding accepted.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::CustomCallOp>(*mod), 1);
}

TEST(FoldDotTransposesTest, Phase2CustomCallSingleDirectionRejectedWhenMLtN) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(%w: tensor<256x128xf32>) -> tensor<8x128xf32> {
    %x = stablehlo.custom_call @opaque_source()
        : () -> tensor<1x2x8x128xf32>
    %t = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x2x8x128xf32>) -> tensor<8x2x128x1xf32>
    %r = stablehlo.reshape %t
        : (tensor<8x2x128x1xf32>) -> tensor<8x256xf32>
    %y = "stablehlo.dot_general"(%r, %w) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<8x256xf32>, tensor<256x128xf32>) -> tensor<8x128xf32>
    return %y : tensor<8x128xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(RunPass(*mod));
  // M (8) < N (128): unfavorable weight expansion, rejected.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::CustomCallOp>(*mod), 1);
}

TEST(FoldDotTransposesTest, SafeNoOpBailoutOnDynamicShapes) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<?x2x8x128xf32>) -> tensor<8x2x128x?xf32> {
    %t = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<?x2x8x128xf32>) -> tensor<8x2x128x?xf32>
    return %t : tensor<8x2x128x?xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(RunPass(*mod));
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
}

TEST(FoldDotTransposesTest, SafeNoOpBailoutOnZeroSizeDimensions) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<0x2x8x128xf32>,
      %w: tensor<0x128xf32>) -> tensor<8x128xf32> {
    %t = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<0x2x8x128xf32>) -> tensor<8x2x128x0xf32>
    %r = stablehlo.reshape %t : (tensor<8x2x128x0xf32>) -> tensor<8x0xf32>
    %dot = "stablehlo.dot_general"(%r, %w) {
        dot_dimension_numbers = #stablehlo.dot<
            lhs_contracting_dimensions = [1],
            rhs_contracting_dimensions = [0]>
    } : (tensor<8x0xf32>, tensor<0x128xf32>) -> tensor<8x128xf32>
    return %dot : tensor<8x128xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(RunPass(*mod));
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*mod), 1);
}

TEST(FoldDotTransposesTest, SafeNoOpBailoutOnRank0OrRank1Contractions) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<128xf32>, %y: tensor<128xf32>) -> tensor<f32> {
    %dot = "stablehlo.dot"(%x, %y)
        : (tensor<128xf32>, tensor<128xf32>) -> tensor<f32>
    return %dot : tensor<f32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(RunPass(*mod));
  EXPECT_EQ(CountOps<mlir::stablehlo::DotOp>(*mod), 1);
}

}  // namespace
}  // namespace torch_tpu
