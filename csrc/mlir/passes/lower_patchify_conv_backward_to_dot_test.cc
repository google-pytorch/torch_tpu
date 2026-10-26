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

#include "csrc/mlir/passes/lower_patchify_conv_backward_to_dot.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
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

std::string PrintModule(mlir::ModuleOp module) {
  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  return out;
}

template <typename OpTy>
int CountOps(mlir::ModuleOp module) {
  int count = 0;
  module.walk([&](OpTy) { ++count; });
  return count;
}

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 17 + 5) % 31) * 0.25f -
              static_cast<float>((i * 7) % 13) * 0.5f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape,
                                       int64_t seed = 3) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(n);
  const float table[7] = {-1.0f, -0.5f, 0.5f, 1.0f, -1.0f, 0.5f, 1.0f};
  for (int64_t i = 0; i < n; ++i) {
    float f = table[(i * 5 + seed) % 7];
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

TEST(LowerPatchifyConvBackwardToDotTest,
     PairedForwardAndBackward2DPatchifyWithBiasAndAdamSinksReshape) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %x: tensor<2x3x8x8xbf16>,
    %w: tensor<16x3x4x4xbf16>,
    %bias: tensor<16xbf16>,
    %dy_tokens: tensor<2x4x16xbf16>,
    %scale: tensor<bf16>)
    -> (tensor<2x4x16xbf16>, tensor<16x3x4x4xbf16>,
        tensor<16x3x4x4xbf16>, tensor<16xbf16>) {
  %zero = stablehlo.constant dense<0.0> : tensor<bf16>
  %fwd = stablehlo.convolution(%x, %w)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]],
                rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<2x3x8x8xbf16>, tensor<16x3x4x4xbf16>) -> tensor<2x16x2x2xbf16>
  %bias_bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
      : (tensor<16xbf16>) -> tensor<2x16x2x2xbf16>
  %fwd_biased = stablehlo.add %fwd, %bias_bcast : tensor<2x16x2x2xbf16>
  %fwd_flat = stablehlo.reshape %fwd_biased
      : (tensor<2x16x2x2xbf16>) -> tensor<2x16x4xbf16>
  %fwd_tokens = stablehlo.transpose %fwd_flat, dims = [0, 2, 1]
      : (tensor<2x16x4xbf16>) -> tensor<2x4x16xbf16>

  %dy_flat = stablehlo.transpose %dy_tokens, dims = [0, 2, 1]
      : (tensor<2x4x16xbf16>) -> tensor<2x16x4xbf16>
  %dy = stablehlo.reshape %dy_flat
      : (tensor<2x16x4xbf16>) -> tensor<2x16x2x2xbf16>
  %db = stablehlo.reduce(%dy init: %zero) across dimensions = [0, 2, 3]
      : (tensor<2x16x2x2xbf16>, tensor<bf16>) -> tensor<16xbf16>
      reducer(%a: tensor<bf16>, %b: tensor<bf16>) {
    %s = stablehlo.add %a, %b : tensor<bf16>
    stablehlo.return %s : tensor<bf16>
  }
  %dw = stablehlo.convolution(%x, %dy)
      dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[f, b, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
                rhs_dilate = [4, 4]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<2x3x8x8xbf16>, tensor<2x16x2x2xbf16>) -> tensor<16x3x4x4xbf16>
  %scale_bcast = stablehlo.broadcast_in_dim %scale, dims = []
      : (tensor<bf16>) -> tensor<16x3x4x4xbf16>
  %dw_scaled = stablehlo.multiply %dw, %scale_bcast : tensor<16x3x4x4xbf16>
  %dw_sq = stablehlo.multiply %dw, %dw : tensor<16x3x4x4xbf16>
  return %fwd_tokens, %dw_scaled, %dw_sq, %db
      : tensor<2x4x16xbf16>, tensor<16x3x4x4xbf16>,
        tensor<16x3x4x4xbf16>, tensor<16xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeBF16Tensor(context, {2, 3, 8, 8});
  mlir::DenseElementsAttr w = MakeBF16Tensor(context, {16, 3, 4, 4});
  mlir::DenseElementsAttr bias = MakeBF16Tensor(context, {16});
  mlir::DenseElementsAttr dy_tokens = MakeBF16Tensor(context, {2, 4, 16});
  mlir::DenseElementsAttr scale = MakeBF16Tensor(context, {});

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {x, w, bias, dy_tokens, scale}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::ConvolutionOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 2);
  // Both forward and backward share a single input patch-extraction transpose;
  // all intermediate spatial transposes are folded away.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 1);

  // Verify that the backward DotGeneralOp has NO direct ReshapeOp users
  // (ReshapeOp was sunk past multiply ops so XLA TpuBroadcastRewriter does not
  // split the 2D dot back into a 4D padded convolution).
  opt_mod->walk([&](mlir::stablehlo::DotGeneralOp dot) {
    if (dot.getType().getShape() == llvm::ArrayRef<int64_t>({16, 48})) {
      for (mlir::Operation* user : dot->getUsers()) {
        EXPECT_FALSE(mlir::isa<mlir::stablehlo::ReshapeOp>(user));
      }
    }
  });

  auto actual = mlir::stablehlo::evalModule(
      *opt_mod, {x, w, bias, dy_tokens, scale}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), actual->size());

  for (size_t out_idx = 0; out_idx < expected->size(); ++out_idx) {
    auto exp_it = (*expected)[out_idx].value_begin<llvm::APFloat>();
    auto act_it = (*actual)[out_idx].value_begin<llvm::APFloat>();
    for (; exp_it != (*expected)[out_idx].value_end<llvm::APFloat>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ((*act_it).convertToFloat(), (*exp_it).convertToFloat());
    }
  }
}

TEST(LowerPatchifyConvBackwardToDotTest,
     ExactNumericalMatchPadded2DForwardPatchifyWithOverhangF32) {
  mlir::MLIRContext context;
  InitContext(context);

  // Matches VitPose-style padded patchify stem: in=6x5, pad=[[1, 2], [1, 1]],
  // padded=9x7, k=4x3, stride=4x3 -> out=2x2 (with overhang 1 on H and 1 on W).
  const char* kPaddedIr = R"mlir(
func.func @main(
    %x: tensor<2x3x6x5xf32>,
    %w: tensor<8x3x4x3xf32>) -> tensor<2x8x2x2xf32> {
  %out = stablehlo.convolution(%x, %w)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 3], pad = [[1, 2], [1, 1]],
                rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
       precision_config = [#stablehlo<precision HIGHEST>,
                           #stablehlo<precision HIGHEST>]}
      : (tensor<2x3x6x5xf32>, tensor<8x3x4x3xf32>) -> tensor<2x8x2x2xf32>
  return %out : tensor<2x8x2x2xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kPaddedIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kPaddedIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {2, 3, 6, 5}, 0.5f);
  mlir::DenseElementsAttr w = MakeF32Tensor(context, {8, 3, 4, 3}, -0.25f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x, w}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::ConvolutionOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 1);
  EXPECT_THAT(PrintModule(*opt_mod),
              testing::HasSubstr("precision = [HIGHEST, HIGHEST]"));

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, w}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));

  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*act_it, *exp_it);
  }
}

TEST(LowerPatchifyConvBackwardToDotTest,
     ExactNumericalMatch3DBackwardPatchifyF32) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kBwd3DIr = R"mlir(
func.func @main(
    %x: tensor<2x3x4x6x6xf32>,
    %dy: tensor<2x8x2x2x2xf32>) -> tensor<8x3x2x3x3xf32> {
  %dw = stablehlo.convolution(%x, %dy)
      dim_numbers = [f, b, 0, 1, 2]x[i, o, 0, 1, 2]->[f, b, 0, 1, 2],
      window = {stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]],
                rhs_dilate = [2, 3, 3]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
       precision_config = [#stablehlo<precision HIGHEST>,
                           #stablehlo<precision HIGHEST>]}
      : (tensor<2x3x4x6x6xf32>, tensor<2x8x2x2x2xf32>)
      -> tensor<8x3x2x3x3xf32>
  return %dw : tensor<8x3x2x3x3xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kBwd3DIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kBwd3DIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr x = MakeF32Tensor(context, {2, 3, 4, 6, 6}, 1.0f);
  mlir::DenseElementsAttr dy = MakeF32Tensor(context, {2, 8, 2, 2, 2}, -0.5f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x, dy}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::ConvolutionOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, dy}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));

  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*act_it, *exp_it);
  }
}

TEST(LowerPatchifyConvBackwardToDotTest,
     OverlappingStrideOrLargeInChannelsPreserved) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kNonPatchifyIr = R"mlir(
func.func @non_patchify(
    %x_overlap: tensor<1x3x8x8xbf16>,
    %w_overlap: tensor<16x3x4x4xbf16>,
    %x_deep: tensor<1x16x8x8xbf16>,
    %w_deep: tensor<32x16x4x4xbf16>)
    -> (tensor<1x16x3x3xbf16>, tensor<1x32x2x2xbf16>) {
  %c1 = stablehlo.convolution(%x_overlap, %w_overlap)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]],
                rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x3x8x8xbf16>, tensor<16x3x4x4xbf16>) -> tensor<1x16x3x3xbf16>
  %c2 = stablehlo.convolution(%x_deep, %w_deep)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [4, 4], pad = [[0, 0], [0, 0]],
                rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x16x8x8xbf16>, tensor<32x16x4x4xbf16>) -> tensor<1x32x2x2xbf16>
  return %c1, %c2 : tensor<1x16x3x3xbf16>, tensor<1x32x2x2xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceString<mlir::ModuleOp>(kNonPatchifyIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_THAT(RunTorchTpuOptimizationPasses(*mod, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::ConvolutionOp>(*mod), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*mod), 0);
}

}  // namespace
}  // namespace torch_tpu
