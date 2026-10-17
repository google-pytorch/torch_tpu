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

#include "csrc/mlir/passes/trim_convolution_padding_overhang.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 17 + 5) % 31) * 0.125f - 1.5f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

class TrimConvolutionPaddingOverhangTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    registry.insert<mlir::func::FuncDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  std::string RunPass(const char* ir) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(ir, &context_);
    EXPECT_TRUE(static_cast<bool>(module));
    if (!module) {
      return "";
    }
    mlir::PassManager pm(&context_);
    pm.addNestedPass<mlir::func::FuncOp>(
        createTrimConvolutionPaddingOverhangPass());
    EXPECT_TRUE(mlir::succeeded(pm.run(*module)));
    std::string out;
    llvm::raw_string_ostream os(out);
    module->print(os);
    return out;
  }

  mlir::MLIRContext context_;
};

TEST_F(TrimConvolutionPaddingOverhangTest,
       TrimsOverhangingWeightNormPositionalConvAndPushesSliceToInputs) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @pos_conv_weight_norm(
          %x: tensor<1x1024x6xbf16>,
          %v: tensor<1024x64x128xbf16>,
          %g: tensor<1x1x128xbf16>,
          %bias: tensor<1024xbf16>) -> tensor<1x1024x6xbf16> {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %v_f32 = stablehlo.convert %v : (tensor<1024x64x128xbf16>) -> tensor<1024x64x128xf32>
        %g_f32 = stablehlo.convert %g : (tensor<1x1x128xbf16>) -> tensor<1x1x128xf32>
        %abs = stablehlo.abs %v_f32 : tensor<1024x64x128xf32>
        %sq = stablehlo.multiply %abs, %abs : tensor<1024x64x128xf32>
        %red = stablehlo.reduce(%sq init: %zero)
            applies stablehlo.add across dimensions = [0, 1]
            : (tensor<1024x64x128xf32>, tensor<f32>) -> tensor<128xf32>
        %keep = stablehlo.reshape %red : (tensor<128xf32>) -> tensor<1x1x128xf32>
        %vnorm = stablehlo.sqrt %keep : tensor<1x1x128xf32>
        %inv = stablehlo.divide %g_f32, %vnorm : tensor<1x1x128xf32>
        %inv_b = stablehlo.broadcast_in_dim %inv, dims = [0, 1, 2]
            : (tensor<1x1x128xf32>) -> tensor<1024x64x128xf32>
        %w_f32 = stablehlo.multiply %v_f32, %inv_b : tensor<1024x64x128xf32>
        %w = stablehlo.convert %w_f32 : (tensor<1024x64x128xf32>) -> tensor<1024x64x128xbf16>
        %conv = stablehlo.convolution(%x, %w)
            dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
            window = {stride = [1], pad = [[64, 64]], lhs_dilate = [1], rhs_dilate = [1], reverse = [false]}
            {batch_group_count = 1 : i64, feature_group_count = 16 : i64}
            : (tensor<1x1024x6xbf16>, tensor<1024x64x128xbf16>) -> tensor<1x1024x7xbf16>
        %bias_b = stablehlo.broadcast_in_dim %bias, dims = [1]
            : (tensor<1024xbf16>) -> tensor<1x1024x7xbf16>
        %add = stablehlo.add %conv, %bias_b : tensor<1x1024x7xbf16>
        %out = stablehlo.slice %add [0:1, 0:1024, 0:6]
            : (tensor<1x1024x7xbf16>) -> tensor<1x1024x6xbf16>
        return %out : tensor<1x1024x6xbf16>
      }
    }
  )mlir";

  std::string out = RunPass(kIr);
  EXPECT_THAT(out, HasSubstr("[0:1024, 0:64, 59:70]"));
  EXPECT_THAT(out, HasSubstr("[0:1, 0:1, 59:70]"));
  EXPECT_THAT(out, HasSubstr("pad = [[5, 5]]"));
  EXPECT_THAT(out, HasSubstr("tensor<1024x64x11xbf16>"));
  EXPECT_THAT(out, HasSubstr("tensor<11xf32>"));
  EXPECT_THAT(out, Not(HasSubstr("tensor<1x1024x7xbf16>")));
}

TEST_F(TrimConvolutionPaddingOverhangTest,
       CanonicalizesUnitDimWeightNormRetainingAlignedNormOutput) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @weight_norm_with_norm_output(
          %g: tensor<384xf32>,
          %v: tensor<384x192x5xf32>) -> (tensor<384x192x5xf32>, tensor<384x1x1xf32>) {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %g_3d = stablehlo.reshape %g : (tensor<384xf32>) -> tensor<384x1x1xf32>
        %abs = stablehlo.abs %v : tensor<384x192x5xf32>
        %sq = stablehlo.multiply %abs, %abs : tensor<384x192x5xf32>
        %red = stablehlo.reduce(%sq init: %zero)
            applies stablehlo.add across dimensions = [1, 2]
            : (tensor<384x192x5xf32>, tensor<f32>) -> tensor<384xf32>
        %keep = stablehlo.reshape %red : (tensor<384xf32>) -> tensor<384x1x1xf32>
        %vnorm = stablehlo.sqrt %keep : tensor<384x1x1xf32>
        %inv = stablehlo.divide %g_3d, %vnorm : tensor<384x1x1xf32>
        %inv_b = stablehlo.broadcast_in_dim %inv, dims = [0, 1, 2]
            : (tensor<384x1x1xf32>) -> tensor<384x192x5xf32>
        %w = stablehlo.multiply %v, %inv_b : tensor<384x192x5xf32>
        return %w, %vnorm : tensor<384x192x5xf32>, tensor<384x1x1xf32>
      }
    }
  )mlir";

  std::string out = RunPass(kIr);
  EXPECT_THAT(out, HasSubstr("stablehlo.sqrt"));
  EXPECT_THAT(
      out,
      HasSubstr("dims = [0] : (tensor<384xf32>) -> tensor<384x192x5xf32>"));
  EXPECT_THAT(out, Not(HasSubstr("dims = [0, 1, 2]")));
}

TEST_F(TrimConvolutionPaddingOverhangTest,
       ExactNumericalEquivalenceOnReferenceInterpreter) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(
          %x: tensor<1x8x6xf32>,
          %v: tensor<8x4x32xf32>,
          %g: tensor<1x1x32xf32>,
          %bias: tensor<8xf32>) -> tensor<1x8x6xf32> {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %abs = stablehlo.abs %v : tensor<8x4x32xf32>
        %sq = stablehlo.multiply %abs, %abs : tensor<8x4x32xf32>
        %red = stablehlo.reduce(%sq init: %zero)
            applies stablehlo.add across dimensions = [0, 1]
            : (tensor<8x4x32xf32>, tensor<f32>) -> tensor<32xf32>
        %keep = stablehlo.reshape %red : (tensor<32xf32>) -> tensor<1x1x32xf32>
        %vnorm = stablehlo.sqrt %keep : tensor<1x1x32xf32>
        %inv = stablehlo.divide %g, %vnorm : tensor<1x1x32xf32>
        %inv_b = stablehlo.broadcast_in_dim %inv, dims = [0, 1, 2]
            : (tensor<1x1x32xf32>) -> tensor<8x4x32xf32>
        %w = stablehlo.multiply %v, %inv_b : tensor<8x4x32xf32>
        %conv = stablehlo.convolution(%x, %w)
            dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
            window = {stride = [1], pad = [[16, 16]], lhs_dilate = [1], rhs_dilate = [1], reverse = [false]}
            {batch_group_count = 1 : i64, feature_group_count = 2 : i64}
            : (tensor<1x8x6xf32>, tensor<8x4x32xf32>) -> tensor<1x8x7xf32>
        %bias_b = stablehlo.broadcast_in_dim %bias, dims = [1]
            : (tensor<8xf32>) -> tensor<1x8x7xf32>
        %add = stablehlo.add %conv, %bias_b : tensor<1x8x7xf32>
        %out = stablehlo.slice %add [0:1, 0:8, 0:6]
            : (tensor<1x8x7xf32>) -> tensor<1x8x6xf32>
        return %out : tensor<1x8x6xf32>
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context_);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context_);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(
      createTrimConvolutionPaddingOverhangPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_mod)));

  mlir::DenseElementsAttr x = MakeF32Tensor(context_, {1, 8, 6}, 0.2f);
  mlir::DenseElementsAttr v = MakeF32Tensor(context_, {8, 4, 32}, 0.5f);
  mlir::DenseElementsAttr g = MakeF32Tensor(context_, {1, 1, 32}, 1.1f);
  mlir::DenseElementsAttr bias = MakeF32Tensor(context_, {8}, -0.3f);

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x, v, g, bias}, cfg);
  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, v, g, bias}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ((*expected)[0].getNumElements(), (*actual)[0].getNumElements());

  auto exp_it = (*expected)[0].value_begin<llvm::APFloat>();
  auto act_it = (*actual)[0].value_begin<llvm::APFloat>();
  for (; exp_it != (*expected)[0].value_end<llvm::APFloat>();
       ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ((*act_it).convertToFloat(), (*exp_it).convertToFloat());
  }
}

TEST_F(TrimConvolutionPaddingOverhangTest,
       DoesNotFoldSliceWhenPaddingWouldBecomeNegative) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @zero_pad_conv_slice(
          %x: tensor<1x16x10xf32>,
          %w: tensor<16x16x3xf32>) -> tensor<1x16x6xf32> {
        %conv = stablehlo.convolution(%x, %w)
            dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
            window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1], reverse = [false]}
            {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
            : (tensor<1x16x10xf32>, tensor<16x16x3xf32>) -> tensor<1x16x8xf32>
        %out = stablehlo.slice %conv [0:1, 0:16, 2:8]
            : (tensor<1x16x8xf32>) -> tensor<1x16x6xf32>
        return %out : tensor<1x16x6xf32>
      }
    }
  )mlir";

  std::string out = RunPass(kIr);
  // Pad is [[0, 0]], subtracting o_start * 1 = 2 would give negative padding,
  // so the slice must NOT be folded into convolution padding.
  EXPECT_THAT(out, HasSubstr("pad = [[0, 0]]"));
  EXPECT_THAT(out, HasSubstr("stablehlo.slice"));
}

TEST_F(TrimConvolutionPaddingOverhangTest,
       DoesNotCanonicalizeAmbiguousUnitDimChannelBroadcast) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @ambiguous_weight_norm(
          %g: tensor<1xf32>,
          %v: tensor<1x16x1xf32>) -> tensor<1x16x1xf32> {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %g_3d = stablehlo.reshape %g : (tensor<1xf32>) -> tensor<1x1x1xf32>
        %abs = stablehlo.abs %v : tensor<1x16x1xf32>
        %sq = stablehlo.multiply %abs, %abs : tensor<1x16x1xf32>
        %red = stablehlo.reduce(%sq init: %zero)
            applies stablehlo.add across dimensions = [0, 1]
            : (tensor<1x16x1xf32>, tensor<f32>) -> tensor<1xf32>
        %keep = stablehlo.reshape %red : (tensor<1xf32>) -> tensor<1x1x1xf32>
        %vnorm = stablehlo.sqrt %keep : tensor<1x1x1xf32>
        %inv = stablehlo.divide %g_3d, %vnorm : tensor<1x1x1xf32>
        %inv_b = stablehlo.broadcast_in_dim %inv, dims = [0, 1, 2]
            : (tensor<1x1x1xf32>) -> tensor<1x16x1xf32>
        %w = stablehlo.multiply %v, %inv_b : tensor<1x16x1xf32>
        return %w : tensor<1x16x1xf32>
      }
    }
  )mlir";

  std::string out = RunPass(kIr);
  // When channels == 1 and out_type has multiple unit dimensions (dims 0 and
  // 2), the channel dimension is ambiguous, so the broadcast remains
  // uncanonicalized.
  EXPECT_THAT(out, HasSubstr("dims = [0, 1, 2]"));
}

}  // namespace
}  // namespace torch_tpu
