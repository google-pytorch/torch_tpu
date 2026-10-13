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

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

class FuseRotaryAndViewTransposeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_
        .loadDialect<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
                     mlir::chlo::ChloDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> ParseModule(const char* ir) {
    return mlir::parseSourceString<mlir::ModuleOp>(ir, &context_);
  }

  mlir::MLIRContext context_;
};

TEST_F(FuseRotaryAndViewTransposeTest,
       RewritesStridedViewRopeToSingleTransposeAndDot) {
  const char* kIr = R"mlir(
    module {
      func.func @main(
          %mm: tensor<2048x16x256xbf16>,
          %cos: tensor<1x16x2048x256xbf16>,
          %sin: tensor<1x16x2048x256xbf16>) -> tensor<1x16x2048x256xbf16> {
        %r0 = stablehlo.reshape %mm : (tensor<2048x16x256xbf16>) -> tensor<2048x16x256x1xbf16>
        %t0 = stablehlo.transpose %r0, dims = [3, 1, 0, 2] : (tensor<2048x16x256x1xbf16>) -> tensor<1x16x2048x256xbf16>
        %m1 = stablehlo.multiply %t0, %cos : tensor<1x16x2048x256xbf16>

        %shi = stablehlo.slice %mm [0:2048, 0:16, 128:256] : (tensor<2048x16x256xbf16>) -> tensor<2048x16x128xbf16>
        %rhi = stablehlo.reshape %shi : (tensor<2048x16x128xbf16>) -> tensor<2048x16x128x1xbf16>
        %thi = stablehlo.transpose %rhi, dims = [3, 1, 0, 2] : (tensor<2048x16x128x1xbf16>) -> tensor<1x16x2048x128xbf16>
        %nhi = stablehlo.negate %thi : tensor<1x16x2048x128xbf16>

        %slo = stablehlo.slice %mm [0:2048, 0:16, 0:128] : (tensor<2048x16x256xbf16>) -> tensor<2048x16x128xbf16>
        %rlo = stablehlo.reshape %slo : (tensor<2048x16x128xbf16>) -> tensor<2048x16x128x1xbf16>
        %tlo = stablehlo.transpose %rlo, dims = [3, 1, 0, 2] : (tensor<2048x16x128x1xbf16>) -> tensor<1x16x2048x128xbf16>

        %cat = stablehlo.concatenate %nhi, %tlo, dim = 3 : (tensor<1x16x2048x128xbf16>, tensor<1x16x2048x128xbf16>) -> tensor<1x16x2048x256xbf16>
        %m2 = stablehlo.multiply %cat, %sin : tensor<1x16x2048x256xbf16>
        %out = stablehlo.add %m1, %m2 : tensor<1x16x2048x256xbf16>
        return %out : tensor<1x16x2048x256xbf16>
      }
    }
  )mlir";

  auto module = ParseModule(kIr);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context_));

  int transpose_count = 0;
  int concat_count = 0;
  int slice_count = 0;
  int dot_count = 0;
  module->walk([&](mlir::stablehlo::TransposeOp) { ++transpose_count; });
  module->walk([&](mlir::stablehlo::ConcatenateOp) { ++concat_count; });
  module->walk([&](mlir::stablehlo::SliceOp) { ++slice_count; });
  module->walk([&](mlir::stablehlo::DotGeneralOp) { ++dot_count; });
  EXPECT_EQ(transpose_count, 1);
  EXPECT_EQ(concat_count, 0);
  EXPECT_EQ(slice_count, 0);
  EXPECT_EQ(dot_count, 1);
}

TEST_F(FuseRotaryAndViewTransposeTest,
       EliminatesInverseTransposePairOn4DRmsNorm) {
  const char* kIr = R"mlir(
    module {
      func.func @main(
          %mm: tensor<4105x1024xbf16>,
          %nw: tensor<64xbf16>) -> tensor<4105x16x64xbf16> {
        %r_in = stablehlo.reshape %mm : (tensor<4105x1024xbf16>) -> tensor<4105x16x64x1xbf16>
        %t_in = stablehlo.transpose %r_in, dims = [3, 1, 0, 2] : (tensor<4105x16x64x1xbf16>) -> tensor<1x16x4105x64xbf16>
        %xf = stablehlo.convert %t_in : (tensor<1x16x4105x64xbf16>) -> tensor<1x16x4105x64xf32>
        %two = stablehlo.constant dense<2.000000e+00> : tensor<1x16x4105x64xf32>
        %sq = stablehlo.power %xf, %two : tensor<1x16x4105x64xf32>
        %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %red = stablehlo.reduce(%sq init: %init) applies stablehlo.add across dimensions = [3] : (tensor<1x16x4105x64xf32>, tensor<f32>) -> tensor<1x16x4105xf32>
        %red4d = stablehlo.reshape %red : (tensor<1x16x4105xf32>) -> tensor<1x16x4105x1xf32>
        %c64 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
        %c64b = stablehlo.broadcast_in_dim %c64, dims = [] : (tensor<f32>) -> tensor<1x16x4105x1xf32>
        %var = stablehlo.divide %red4d, %c64b : tensor<1x16x4105x1xf32>
        %ceps = stablehlo.constant dense<1.000000e-06> : tensor<f32>
        %cepsb = stablehlo.broadcast_in_dim %ceps, dims = [] : (tensor<f32>) -> tensor<1x16x4105x1xf32>
        %add_eps = stablehlo.add %var, %cepsb : tensor<1x16x4105x1xf32>
        %rsq = stablehlo.rsqrt %add_eps : tensor<1x16x4105x1xf32>
        %rsq3 = stablehlo.reshape %rsq : (tensor<1x16x4105x1xf32>) -> tensor<1x16x4105xf32>
        %rsqb = stablehlo.broadcast_in_dim %rsq3, dims = [0, 1, 2] : (tensor<1x16x4105xf32>) -> tensor<1x16x4105x64xf32>
        %normf = stablehlo.multiply %xf, %rsqb : tensor<1x16x4105x64xf32>
        %normbf = stablehlo.convert %normf : (tensor<1x16x4105x64xf32>) -> tensor<1x16x4105x64xbf16>
        %nwb = stablehlo.broadcast_in_dim %nw, dims = [3] : (tensor<64xbf16>) -> tensor<1x16x4105x64xbf16>
        %out4 = stablehlo.multiply %nwb, %normbf : tensor<1x16x4105x64xbf16>
        %t_out = stablehlo.transpose %out4, dims = [2, 1, 3, 0] : (tensor<1x16x4105x64xbf16>) -> tensor<4105x16x64x1xbf16>
        %out3 = stablehlo.reshape %t_out : (tensor<4105x16x64x1xbf16>) -> tensor<4105x16x64xbf16>
        return %out3 : tensor<4105x16x64xbf16>
      }
    }
  )mlir";

  auto module = ParseModule(kIr);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context_));

  int transpose_count = 0;
  int pow_count = 0;
  module->walk([&](mlir::stablehlo::TransposeOp) { ++transpose_count; });
  module->walk([&](mlir::stablehlo::PowOp) { ++pow_count; });
  EXPECT_EQ(transpose_count, 0);
  EXPECT_EQ(pow_count, 0);
}

}  // namespace
}  // namespace torch_tpu
