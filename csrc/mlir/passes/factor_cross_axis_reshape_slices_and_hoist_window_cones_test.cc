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

#include "csrc/mlir/passes/factor_cross_axis_reshape_slices_and_hoist_window_cones.h"

#include <cstdint>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
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

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 11 + 3) % 29) * 0.0625f - 0.75f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

TEST(FactorCrossAxisReshapeSlicesAndHoistWindowConesTest,
     FactorsCrossAxisReshapeSlicesAndHoistsAcrossResidualMlpAndMultiSlice) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %x: tensor<1x8x16xf32>,
    %w1: tensor<16x16xf32>,
    %w2: tensor<16x40xf32>,
    %w_res: tensor<16x40xf32>,
    %scale: tensor<1x8xf32>) -> (tensor<1x4x1xf32>, tensor<1x4x1xf32>) {
  %x2d = stablehlo.reshape %x : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
  %mm1 = "stablehlo.dot_general"(%x2d, %w1) <{
      dot_dimension_numbers = #stablehlo.dot<
          lhs_contracting_dimensions = [1],
          rhs_contracting_dimensions = [0]>
  }> : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %mm1_3d = stablehlo.reshape %mm1 : (tensor<8x16xf32>) -> tensor<1x8x16xf32>
  %neg = stablehlo.negate %mm1_3d : tensor<1x8x16xf32>
  %exp = stablehlo.exponential %neg : tensor<1x8x16xf32>
  %one = stablehlo.constant dense<1.0> : tensor<1x8x16xf32>
  %den = stablehlo.add %exp, %one : tensor<1x8x16xf32>
  %silu = stablehlo.divide %mm1_3d, %den : tensor<1x8x16xf32>
  %silu2d = stablehlo.reshape %silu : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
  %mm2 = "stablehlo.dot_general"(%silu2d, %w2) <{
      dot_dimension_numbers = #stablehlo.dot<
          lhs_contracting_dimensions = [1],
          rhs_contracting_dimensions = [0]>
  }> : (tensor<8x16xf32>, tensor<16x40xf32>) -> tensor<8x40xf32>
  %mm2_3d = stablehlo.reshape %mm2 : (tensor<8x40xf32>) -> tensor<1x8x40xf32>
  %mm_res = "stablehlo.dot_general"(%x2d, %w_res) <{
      dot_dimension_numbers = #stablehlo.dot<
          lhs_contracting_dimensions = [1],
          rhs_contracting_dimensions = [0]>
  }> : (tensor<8x16xf32>, tensor<16x40xf32>) -> tensor<8x40xf32>
  %mm_res_3d = stablehlo.reshape %mm_res : (tensor<8x40xf32>) -> tensor<1x8x40xf32>
  %sum = stablehlo.add %mm2_3d, %mm_res_3d : tensor<1x8x40xf32>
  %bcast_scale = stablehlo.broadcast_in_dim %scale, dims = [0, 1]
      : (tensor<1x8xf32>) -> tensor<1x8x40xf32>
  %out = stablehlo.multiply %sum, %bcast_scale : tensor<1x8x40xf32>
  %flat = stablehlo.reshape %out : (tensor<1x8x40xf32>) -> tensor<64x5xf32>
  %s_head = stablehlo.slice %flat [56:64, 0:1]
      : (tensor<64x5xf32>) -> tensor<8x1xf32>
  %s_head_3d = stablehlo.reshape %s_head : (tensor<8x1xf32>) -> tensor<1x8x1xf32>
  %last_row = stablehlo.slice %out [0:1, 7:8, 0:40]
      : (tensor<1x8x40xf32>) -> tensor<1x1x40xf32>
  %last_2d = stablehlo.reshape %last_row : (tensor<1x1x40xf32>) -> tensor<8x5xf32>
  %s_tail = stablehlo.slice %last_2d [0:8, 1:5]
      : (tensor<8x5xf32>) -> tensor<8x4xf32>
  %s_tail_3d = stablehlo.reshape %s_tail : (tensor<8x4xf32>) -> tensor<1x8x4xf32>
  %rev = stablehlo.reverse %s_tail_3d, dims = [2] : tensor<1x8x4xf32>
  %cat = stablehlo.concatenate %s_head_3d, %rev, dim = 2
      : (tensor<1x8x1xf32>, tensor<1x8x4xf32>) -> tensor<1x8x5xf32>
  %q1 = stablehlo.slice %cat [0:1, 0:4, 1:2]
      : (tensor<1x8x5xf32>) -> tensor<1x4x1xf32>
  %q3 = stablehlo.slice %cat [0:1, 0:4, 3:4]
      : (tensor<1x8x5xf32>) -> tensor<1x4x1xf32>
  return %q1, %q3 : tensor<1x4x1xf32>, tensor<1x4x1xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {1, 8, 16}, 0.2f);
  mlir::DenseElementsAttr w1 = MakeF32Tensor(context, {16, 16}, -0.1f);
  mlir::DenseElementsAttr w2 = MakeF32Tensor(context, {16, 40}, 0.3f);
  mlir::DenseElementsAttr w_res = MakeF32Tensor(context, {16, 40}, -0.2f);
  mlir::DenseElementsAttr scale = MakeF32Tensor(context, {1, 8}, 1.0f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {x, w1, w2, w_res, scale}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // All 3 DotGeneralOp contractions now have LHS row count 1 (down from 8),
  // and the two output projections have RHS column count 20 (down from 40).
  int dot_count = 0;
  int shrunk_lhs_dots = 0;
  int shrunk_rhs_dots = 0;
  opt_mod->walk([&](mlir::stablehlo::DotGeneralOp dot) {
    ++dot_count;
    auto out_ty = mlir::cast<mlir::RankedTensorType>(dot.getType());
    if (out_ty.getDimSize(0) == 1) {
      ++shrunk_lhs_dots;
    }
    if (out_ty.getDimSize(1) == 20) {
      ++shrunk_rhs_dots;
    }
  });
  EXPECT_EQ(dot_count, 3);
  EXPECT_EQ(shrunk_lhs_dots, 3);
  EXPECT_EQ(shrunk_rhs_dots, 2);

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {x, w1, w2, w_res, scale}, cfg);
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

TEST(FactorCrossAxisReshapeSlicesAndHoistWindowConesTest,
     PreservesUnalignedCrossAxisReshapeSlice) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<1x8x40xf32>) -> tensor<8x1xf32> {
  %flat = stablehlo.reshape %x : (tensor<1x8x40xf32>) -> tensor<64x5xf32>
  %s = stablehlo.slice %flat [5:13, 0:1] : (tensor<64x5xf32>) -> tensor<8x1xf32>
  return %s : tensor<8x1xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());

  int dim64_reshape_count = 0;
  module->walk([&](mlir::stablehlo::ReshapeOp rop) {
    auto ty = mlir::cast<mlir::RankedTensorType>(rop.getType());
    if (ty.getDimSize(0) == 64) {
      ++dim64_reshape_count;
    }
  });
  EXPECT_EQ(dim64_reshape_count, 1);
}

TEST(FactorCrossAxisReshapeSlicesAndHoistWindowConesTest,
     HoistsSliceAcrossTransposeOp) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<12x8xf32>) -> tensor<4x4xf32> {
  %t = stablehlo.transpose %x, dims = [1, 0]
      : (tensor<12x8xf32>) -> tensor<8x12xf32>
  %s = stablehlo.slice %t [2:6, 3:7]
      : (tensor<8x12xf32>) -> tensor<4x4xf32>
  return %s : tensor<4x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {12, 8}, 0.5f);
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // Transpose input should now be a sliced tensor of shape tensor<4x4xf32>.
  int transpose_count = 0;
  opt_mod->walk([&](mlir::stablehlo::TransposeOp top) {
    ++transpose_count;
    auto in_ty = mlir::cast<mlir::RankedTensorType>(top.getOperand().getType());
    EXPECT_EQ(in_ty.getShape(), llvm::ArrayRef<int64_t>({4, 4}));
  });
  EXPECT_EQ(transpose_count, 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
  }
}

TEST(FactorCrossAxisReshapeSlicesAndHoistWindowConesTest,
     HoistsSliceAcross2DDotOp) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %lhs: tensor<16x32xf32>,
    %rhs: tensor<32x24xf32>) -> tensor<8x12xf32> {
  %dot = "stablehlo.dot"(%lhs, %rhs)
      : (tensor<16x32xf32>, tensor<32x24xf32>) -> tensor<16x24xf32>
  %s = stablehlo.slice %dot [4:12, 6:18]
      : (tensor<16x24xf32>) -> tensor<8x12xf32>
  return %s : tensor<8x12xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // DotOp output should now directly produce tensor<8x12xf32>, and its LHS/RHS
  // should be sliced before contraction.
  int dot_count = 0;
  opt_mod->walk([&](mlir::stablehlo::DotOp dot) {
    ++dot_count;
    auto out_ty = mlir::cast<mlir::RankedTensorType>(dot.getType());
    EXPECT_EQ(out_ty.getShape(), llvm::ArrayRef<int64_t>({8, 12}));

    auto lhs_slice = dot.getLhs().getDefiningOp<mlir::stablehlo::SliceOp>();
    ASSERT_TRUE(lhs_slice);
    EXPECT_EQ(lhs_slice.getStartIndices(), llvm::ArrayRef<int64_t>({4, 0}));
    EXPECT_EQ(lhs_slice.getLimitIndices(), llvm::ArrayRef<int64_t>({12, 32}));

    auto rhs_slice = dot.getRhs().getDefiningOp<mlir::stablehlo::SliceOp>();
    ASSERT_TRUE(rhs_slice);
    EXPECT_EQ(rhs_slice.getStartIndices(), llvm::ArrayRef<int64_t>({0, 6}));
    EXPECT_EQ(rhs_slice.getLimitIndices(), llvm::ArrayRef<int64_t>({32, 18}));
  });
  EXPECT_EQ(dot_count, 1);

  opt_mod->walk([&](mlir::func::ReturnOp ret) {
    for (mlir::Value v : ret.getOperands()) {
      EXPECT_TRUE(v.getDefiningOp<mlir::stablehlo::DotOp>());
    }
  });
}

TEST(FactorCrossAxisReshapeSlicesAndHoistWindowConesTest,
     FactorsCaseBMultiRowAlignedCrossAxisReshapeSlice) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<4x8xf32>) -> tensor<16xf32> {
  %flat = stablehlo.reshape %x : (tensor<4x8xf32>) -> tensor<32xf32>
  %s = stablehlo.slice %flat [8:24] : (tensor<32xf32>) -> tensor<16xf32>
  return %s : tensor<16xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {4, 8}, 1.5f);
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // Full 4x8 -> 32 reshape should be replaced with 2x8 -> 16 reshape on sliced
  // %x.
  int dim32_reshape_count = 0;
  int dim16_reshape_count = 0;
  opt_mod->walk([&](mlir::stablehlo::ReshapeOp rop) {
    auto out_ty = mlir::cast<mlir::RankedTensorType>(rop.getType());
    if (out_ty.getDimSize(0) == 32) {
      ++dim32_reshape_count;
    }
    if (out_ty.getDimSize(0) == 16) {
      ++dim16_reshape_count;
      auto in_ty =
          mlir::cast<mlir::RankedTensorType>(rop.getOperand().getType());
      EXPECT_EQ(in_ty.getShape(), llvm::ArrayRef<int64_t>({2, 8}));
    }
  });
  EXPECT_EQ(dim32_reshape_count, 0);
  EXPECT_EQ(dim16_reshape_count, 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
  }
}

TEST(FactorCrossAxisReshapeSlicesAndHoistWindowConesTest,
     HoistsSliceAcrossPointwiseOpWithIdenticalOperands) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<16x16xf32>) -> tensor<4x4xf32> {
  %sq = stablehlo.multiply %x, %x : tensor<16x16xf32>
  %s = stablehlo.slice %sq [2:6, 4:8] : (tensor<16x16xf32>) -> tensor<4x4xf32>
  return %s : tensor<4x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {16, 16}, 0.7f);
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // There should be exactly 1 SliceOp created for %x (not duplicated).
  int slice_count = 0;
  opt_mod->walk([&](mlir::stablehlo::SliceOp sop) {
    ++slice_count;
    auto in_ty = mlir::cast<mlir::RankedTensorType>(sop.getOperand().getType());
    EXPECT_EQ(in_ty.getShape(), llvm::ArrayRef<int64_t>({16, 16}));
  });
  EXPECT_EQ(slice_count, 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-5f);
  }
}

}  // namespace
}  // namespace torch_tpu
