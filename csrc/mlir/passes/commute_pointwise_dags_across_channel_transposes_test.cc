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

#include "csrc/mlir/passes/commute_pointwise_dags_across_channel_transposes.h"

#include <cstdint>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/ChloOps.h"
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
  module.walk([&](OpTy) { ++count; });
  return count;
}

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float scale) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = static_cast<float>((i * 13 + 7) % 29 - 14) * scale;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

void ExpectNumericallyClose(mlir::DenseElementsAttr actual,
                            mlir::DenseElementsAttr expected, float tol) {
  auto act_vals = actual.getValues<float>();
  auto exp_vals = expected.getValues<float>();
  ASSERT_EQ(actual.getNumElements(), expected.getNumElements());
  auto it_a = act_vals.begin();
  auto it_e = exp_vals.begin();
  for (; it_a != act_vals.end(); ++it_a, ++it_e) {
    EXPECT_NEAR(*it_a, *it_e, tol);
  }
}

TEST(CommutePointwiseDagsAcrossChannelTransposesTest,
     CommutesGeluDiamondBeforeDepthwiseConvTranspose) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<1x4x3x3xf32>, %w: tensor<4x1x3x3xf32>, %b: tensor<4xf32>) -> tensor<1x9x4xf32> {
  %conv = stablehlo.convolution(%x, %w)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 4 : i64}
      : (tensor<1x4x3x3xf32>, tensor<4x1x3x3xf32>) -> tensor<1x4x3x3xf32>
  %b_bcast = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<4xf32>) -> tensor<1x4x3x3xf32>
  %biased = stablehlo.add %conv, %b_bcast : tensor<1x4x3x3xf32>
  %r = stablehlo.reshape %biased : (tensor<1x4x3x3xf32>) -> tensor<4x9x1xf32>
  %tr = stablehlo.transpose %r, dims = [2, 1, 0] : (tensor<4x9x1xf32>) -> tensor<1x9x4xf32>
  %half = stablehlo.constant dense<0.5> : tensor<1x9x4xf32>
  %inv_sqrt2 = stablehlo.constant dense<0.707106769> : tensor<1x9x4xf32>
  %one = stablehlo.constant dense<1.0> : tensor<1x9x4xf32>
  %x_half = stablehlo.multiply %tr, %half : tensor<1x9x4xf32>
  %x_scaled = stablehlo.multiply %tr, %inv_sqrt2 : tensor<1x9x4xf32>
  %erf = chlo.erf %x_scaled : tensor<1x9x4xf32> -> tensor<1x9x4xf32>
  %erf_plus_one = stablehlo.add %erf, %one : tensor<1x9x4xf32>
  %gelu = stablehlo.multiply %x_half, %erf_plus_one : tensor<1x9x4xf32>
  return %gelu : tensor<1x9x4xf32>
}
)mlir";

  auto module_after = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_after);

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_after, &context));

  // Verify that chlo::ErfOp now runs on the pre-transpose shape
  // tensor<1x4x3x3xf32>.
  int erf_on_conv_shape = 0;
  module_after->walk([&](mlir::chlo::ErfOp op) {
    auto t = llvm::cast<mlir::RankedTensorType>(op.getType());
    if (t.getShape() == llvm::ArrayRef<int64_t>{1, 4, 3, 3}) {
      ++erf_on_conv_shape;
    }
  });
  EXPECT_EQ(erf_on_conv_shape, 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module_after), 1);
}

TEST(CommutePointwiseDagsAcrossChannelTransposesTest,
     CommutesConformerPointwiseConv1GluAndPointwiseConv2ToChannelMinor) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %x: tensor<1x6x4xf32>,
    %w1: tensor<8x4x1xf32>,
    %b1: tensor<8xf32>,
    %w_dw: tensor<4x1x3xf32>,
    %w2: tensor<4x4x1xf32>,
    %b2: tensor<4xf32>) -> tensor<1x6x4xf32> {
  %r_in = stablehlo.reshape %x : (tensor<1x6x4xf32>) -> tensor<6x4x1xf32>
  %tr_in = stablehlo.transpose %r_in, dims = [2, 1, 0] : (tensor<6x4x1xf32>) -> tensor<1x4x6xf32>
  %conv1 = stablehlo.convolution(%tr_in, %w1)
      dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
      window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x4x6xf32>, tensor<8x4x1xf32>) -> tensor<1x8x6xf32>
  %b1_bcast = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<8xf32>) -> tensor<1x8x6xf32>
  %conv1_b = stablehlo.add %conv1, %b1_bcast : tensor<1x8x6xf32>
  %h0 = stablehlo.slice %conv1_b [0:1, 0:4, 0:6] : (tensor<1x8x6xf32>) -> tensor<1x4x6xf32>
  %h1 = stablehlo.slice %conv1_b [0:1, 4:8, 0:6] : (tensor<1x8x6xf32>) -> tensor<1x4x6xf32>
  %one = stablehlo.constant dense<1.0> : tensor<1x4x6xf32>
  %neg1 = stablehlo.negate %h1 : tensor<1x4x6xf32>
  %exp1 = stablehlo.exponential %neg1 : tensor<1x4x6xf32>
  %den1 = stablehlo.add %exp1, %one : tensor<1x4x6xf32>
  %sig1 = stablehlo.divide %one, %den1 : tensor<1x4x6xf32>
  %glu = stablehlo.multiply %h0, %sig1 : tensor<1x4x6xf32>
  %dw = stablehlo.convolution(%glu, %w_dw)
      dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
      window = {stride = [1], pad = [[1, 1]], rhs_dilate = [1]}
      {batch_group_count = 1 : i64, feature_group_count = 4 : i64}
      : (tensor<1x4x6xf32>, tensor<4x1x3xf32>) -> tensor<1x4x6xf32>
  %neg2 = stablehlo.negate %dw : tensor<1x4x6xf32>
  %exp2 = stablehlo.exponential %neg2 : tensor<1x4x6xf32>
  %den2 = stablehlo.add %exp2, %one : tensor<1x4x6xf32>
  %silu = stablehlo.divide %dw, %den2 : tensor<1x4x6xf32>
  %conv2 = stablehlo.convolution(%silu, %w2)
      dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
      window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x4x6xf32>, tensor<4x4x1xf32>) -> tensor<1x4x6xf32>
  %b2_bcast = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<4xf32>) -> tensor<1x4x6xf32>
  %conv2_b = stablehlo.add %conv2, %b2_bcast : tensor<1x4x6xf32>
  %r_out = stablehlo.reshape %conv2_b : (tensor<1x4x6xf32>) -> tensor<4x6x1xf32>
  %tr_out = stablehlo.transpose %r_out, dims = [2, 1, 0] : (tensor<4x6x1xf32>) -> tensor<1x6x4xf32>
  return %tr_out : tensor<1x6x4xf32>
}
)mlir";

  auto module_before = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_before);
  auto module_after = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_after);

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_after, &context));

  // Both pointwise_conv1 and pointwise_conv2 are lowered to 2D DotGeneralOp,
  // leaving only the depthwise convolution as a ConvolutionOp.
  EXPECT_EQ(CountOps<mlir::stablehlo::ConvolutionOp>(*module_after), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_after), 2);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {1, 6, 4}, 0.1f),
      MakeF32Tensor(context, {8, 4, 1}, 0.05f),
      MakeF32Tensor(context, {8}, 0.02f),
      MakeF32Tensor(context, {4, 1, 3}, 0.05f),
      MakeF32Tensor(context, {4, 4, 1}, 0.05f),
      MakeF32Tensor(context, {4}, 0.02f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_before, inputs, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  auto opt_out = mlir::stablehlo::evalModule(*module_after, inputs, config);
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ExpectNumericallyClose((*opt_out)[0], (*ref_out)[0], 1e-5f);
}

TEST(CommutePointwiseDagsAcrossChannelTransposesTest,
     CommutesBatchNormInferenceAndSiluBeforePointwiseConv2To2d) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %dw: tensor<1x4x6xf32>,
    %scale: tensor<4xf32>,
    %offset: tensor<4xf32>,
    %mean: tensor<4xf32>,
    %var: tensor<4xf32>,
    %w2: tensor<4x4x1xf32>,
    %b2: tensor<4xf32>) -> tensor<1x6x4xf32> {
  %bn = "stablehlo.batch_norm_inference"(%dw, %scale, %offset, %mean, %var)
      {epsilon = 1.0e-05 : f32, feature_index = 1 : i64}
      : (tensor<1x4x6xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x6xf32>
  %one = stablehlo.constant dense<1.0> : tensor<1x4x6xf32>
  %neg2 = stablehlo.negate %bn : tensor<1x4x6xf32>
  %exp2 = stablehlo.exponential %neg2 : tensor<1x4x6xf32>
  %den2 = stablehlo.add %exp2, %one : tensor<1x4x6xf32>
  %silu = stablehlo.divide %bn, %den2 : tensor<1x4x6xf32>
  %conv2 = stablehlo.convolution(%silu, %w2)
      dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
      window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x4x6xf32>, tensor<4x4x1xf32>) -> tensor<1x4x6xf32>
  %b2_bcast = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<4xf32>) -> tensor<1x4x6xf32>
  %conv2_b = stablehlo.add %conv2, %b2_bcast : tensor<1x4x6xf32>
  %r_out = stablehlo.reshape %conv2_b : (tensor<1x4x6xf32>) -> tensor<4x6x1xf32>
  %tr_out = stablehlo.transpose %r_out, dims = [2, 1, 0] : (tensor<4x6x1xf32>) -> tensor<1x6x4xf32>
  return %tr_out : tensor<1x6x4xf32>
}
)mlir";

  auto module_after = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_after);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_after, &context));

  EXPECT_EQ(CountOps<mlir::stablehlo::ConvolutionOp>(*module_after), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_after), 1);
  int bn_2d_count = 0;
  module_after->walk([&](mlir::stablehlo::BatchNormInferenceOp op) {
    auto t = llvm::cast<mlir::RankedTensorType>(op.getType());
    if (t.getShape() == llvm::ArrayRef<int64_t>{6, 4} &&
        op.getFeatureIndex() == 1) {
      ++bn_2d_count;
    }
  });
  EXPECT_EQ(bn_2d_count, 1);
  module_after->walk([&](mlir::stablehlo::DotGeneralOp op) {
    auto dims = op.getDotDimensionNumbers();
    EXPECT_EQ(dims.getLhsContractingDimensions(), (llvm::ArrayRef<int64_t>{1}));
    EXPECT_EQ(dims.getRhsContractingDimensions(), (llvm::ArrayRef<int64_t>{1}));
  });
}

TEST(CommutePointwiseDagsAcrossChannelTransposesTest,
     CommutesPointwiseConv1GluWithConvertOpToChannelMinor) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %x: tensor<1x6x4xbf16>,
    %w1: tensor<8x4x1xbf16>,
    %b1: tensor<8xbf16>) -> tensor<1x4x6xbf16> {
  %tr_in = stablehlo.transpose %x, dims = [0, 2, 1] : (tensor<1x6x4xbf16>) -> tensor<1x4x6xbf16>
  %conv1 = stablehlo.convolution(%tr_in, %w1)
      dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
      window = {stride = [1], pad = [[0, 0]], rhs_dilate = [1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x4x6xbf16>, tensor<8x4x1xbf16>) -> tensor<1x8x6xbf16>
  %b1_bcast = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<8xbf16>) -> tensor<1x8x6xbf16>
  %conv1_b = stablehlo.add %conv1, %b1_bcast : tensor<1x8x6xbf16>
  %conv1_f32 = stablehlo.convert %conv1_b : (tensor<1x8x6xbf16>) -> tensor<1x8x6xf32>
  %h0 = stablehlo.slice %conv1_f32 [0:1, 0:4, 0:6] : (tensor<1x8x6xf32>) -> tensor<1x4x6xf32>
  %h1 = stablehlo.slice %conv1_f32 [0:1, 4:8, 0:6] : (tensor<1x8x6xf32>) -> tensor<1x4x6xf32>
  %one = stablehlo.constant dense<1.0> : tensor<1x4x6xf32>
  %neg1 = stablehlo.negate %h1 : tensor<1x4x6xf32>
  %exp1 = stablehlo.exponential %neg1 : tensor<1x4x6xf32>
  %den1 = stablehlo.add %exp1, %one : tensor<1x4x6xf32>
  %sig1 = stablehlo.divide %one, %den1 : tensor<1x4x6xf32>
  %glu = stablehlo.multiply %h0, %sig1 : tensor<1x4x6xf32>
  %cvt = stablehlo.convert %glu : (tensor<1x4x6xf32>) -> tensor<1x4x6xbf16>
  return %cvt : tensor<1x4x6xbf16>
}
)mlir";

  auto module_after = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_after);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_after, &context));

  // Pointwise conv1 was lowered to 2D DotGeneralOp, and GLU ConvertOp commuted.
  EXPECT_EQ(CountOps<mlir::stablehlo::ConvolutionOp>(*module_after), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_after), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module_after), 1);
  module_after->walk([&](mlir::stablehlo::TransposeOp op) {
    auto t = llvm::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(t.getElementType(), mlir::BFloat16Type::get(&context));
    EXPECT_EQ(t.getShape(), (llvm::ArrayRef<int64_t>{1, 4, 6}));
  });
}

}  // namespace
}  // namespace torch_tpu
