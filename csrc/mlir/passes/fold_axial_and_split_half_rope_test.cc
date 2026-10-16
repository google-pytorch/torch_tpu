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

#include "csrc/mlir/passes/fold_axial_and_split_half_rope.h"

#include <cstdint>
#include <vector>

#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
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

mlir::LogicalResult RunFoldAxialAndSplitHalfRopePass(
    mlir::ModuleOp module, mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addNestedPass<mlir::func::FuncOp>(createFoldAxialAndSplitHalfRopePass());
  return pm.run(module);
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
    vals[i] = base + static_cast<float>((i * 17 + 5) % 251) * 0.0625f -
              static_cast<float>((i * 7) % 13) * 0.125f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape,
                                       float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    float f = base + static_cast<float>((i * 17 + 5) % 251) * 0.0625f -
              static_cast<float>((i * 7) % 13) * 0.125f;
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

void ExpectAttrsClose(mlir::DenseElementsAttr actual,
                      mlir::DenseElementsAttr expected, float tol) {
  ASSERT_EQ(actual.getType(), expected.getType());
  auto act_vals = actual.getValues<llvm::APFloat>();
  auto exp_vals = expected.getValues<llvm::APFloat>();
  auto act_it = act_vals.begin();
  auto exp_it = exp_vals.begin();
  for (; act_it != act_vals.end(); ++act_it, ++exp_it) {
    EXPECT_NEAR((*act_it).convertToFloat(), (*exp_it).convertToFloat(), tol);
  }
}

TEST(FoldAxialAndSplitHalfRopeTest,
     Folds3DAxialPairwiseRopeWithPassThroughAndSinksProjection) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @axial_3d(
    %x_2d: tensor<128x128xf32>,
    %w: tensor<128x128xf32>,
    %bias: tensor<128xf32>,
    %cos_d: tensor<1x128x20xf32>,
    %sin_d: tensor<1x128x20xf32>,
    %cos_h: tensor<1x128x20xf32>,
    %sin_h: tensor<1x128x20xf32>,
    %cos_w: tensor<1x128x20xf32>,
    %sin_w: tensor<1x128x20xf32>) -> tensor<1x2x128x64xf32> {
  %dot = stablehlo.dot_general %x_2d, %w, contracting_dims = [1] x [0]
      : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  %b2d = stablehlo.broadcast_in_dim %bias, dims = [1]
      : (tensor<128xf32>) -> tensor<128x128xf32>
  %addmm = stablehlo.add %b2d, %dot : tensor<128x128xf32>
  %r3 = stablehlo.reshape %addmm : (tensor<128x128xf32>) -> tensor<128x2x64xf32>
  %r4_pair = stablehlo.reshape %addmm : (tensor<128x128xf32>) -> tensor<128x2x32x2xf32>

  %cd = stablehlo.broadcast_in_dim %cos_d, dims = [0, 2, 3]
      : (tensor<1x128x20xf32>) -> tensor<1x2x128x20xf32>
  %sd = stablehlo.broadcast_in_dim %sin_d, dims = [0, 2, 3]
      : (tensor<1x128x20xf32>) -> tensor<1x2x128x20xf32>
  %ch = stablehlo.broadcast_in_dim %cos_h, dims = [0, 2, 3]
      : (tensor<1x128x20xf32>) -> tensor<1x2x128x20xf32>
  %sh = stablehlo.broadcast_in_dim %sin_h, dims = [0, 2, 3]
      : (tensor<1x128x20xf32>) -> tensor<1x2x128x20xf32>
  %cw = stablehlo.broadcast_in_dim %cos_w, dims = [0, 2, 3]
      : (tensor<1x128x20xf32>) -> tensor<1x2x128x20xf32>
  %sw = stablehlo.broadcast_in_dim %sin_w, dims = [0, 2, 3]
      : (tensor<1x128x20xf32>) -> tensor<1x2x128x20xf32>

  %sl_d = stablehlo.slice %r3 [0:128, 0:2, 0:20] : (tensor<128x2x64xf32>) -> tensor<128x2x20xf32>
  %r4_d = stablehlo.reshape %sl_d : (tensor<128x2x20xf32>) -> tensor<128x2x20x1xf32>
  %qd = stablehlo.transpose %r4_d, dims = [3, 1, 0, 2] : (tensor<128x2x20x1xf32>) -> tensor<1x2x128x20xf32>
  %md_c = stablehlo.multiply %qd, %cd : tensor<1x2x128x20xf32>
  %sl_do = stablehlo.slice %r4_pair [0:128, 0:2, 0:10, 1:2] : (tensor<128x2x32x2xf32>) -> tensor<128x2x10x1xf32>
  %tr_do = stablehlo.transpose %sl_do, dims = [3, 1, 0, 2] : (tensor<128x2x10x1xf32>) -> tensor<1x2x128x10xf32>
  %neg_do = stablehlo.negate %tr_do : tensor<1x2x128x10xf32>
  %st_do = stablehlo.reshape %neg_do : (tensor<1x2x128x10xf32>) -> tensor<1x2x128x10x1xf32>
  %sl_de = stablehlo.slice %r4_pair [0:128, 0:2, 0:10, 0:1] : (tensor<128x2x32x2xf32>) -> tensor<128x2x10x1xf32>
  %tr_de = stablehlo.transpose %sl_de, dims = [3, 1, 0, 2] : (tensor<128x2x10x1xf32>) -> tensor<1x2x128x10xf32>
  %st_de = stablehlo.reshape %tr_de : (tensor<1x2x128x10xf32>) -> tensor<1x2x128x10x1xf32>
  %cat_d = stablehlo.concatenate %st_do, %st_de, dim = 4 : (tensor<1x2x128x10x1xf32>, tensor<1x2x128x10x1xf32>) -> tensor<1x2x128x10x2xf32>
  %rot_d = stablehlo.reshape %cat_d : (tensor<1x2x128x10x2xf32>) -> tensor<1x2x128x20xf32>
  %md_s = stablehlo.multiply %rot_d, %sd : tensor<1x2x128x20xf32>
  %out_d = stablehlo.add %md_c, %md_s : tensor<1x2x128x20xf32>

  %sl_h = stablehlo.slice %r3 [0:128, 0:2, 20:40] : (tensor<128x2x64xf32>) -> tensor<128x2x20xf32>
  %r4_h = stablehlo.reshape %sl_h : (tensor<128x2x20xf32>) -> tensor<128x2x20x1xf32>
  %qh = stablehlo.transpose %r4_h, dims = [3, 1, 0, 2] : (tensor<128x2x20x1xf32>) -> tensor<1x2x128x20xf32>
  %mh_c = stablehlo.multiply %qh, %ch : tensor<1x2x128x20xf32>
  %sl_ho = stablehlo.slice %r4_pair [0:128, 0:2, 10:20, 1:2] : (tensor<128x2x32x2xf32>) -> tensor<128x2x10x1xf32>
  %tr_ho = stablehlo.transpose %sl_ho, dims = [3, 1, 0, 2] : (tensor<128x2x10x1xf32>) -> tensor<1x2x128x10xf32>
  %neg_ho = stablehlo.negate %tr_ho : tensor<1x2x128x10xf32>
  %st_ho = stablehlo.reshape %neg_ho : (tensor<1x2x128x10xf32>) -> tensor<1x2x128x10x1xf32>
  %sl_he = stablehlo.slice %r4_pair [0:128, 0:2, 10:20, 0:1] : (tensor<128x2x32x2xf32>) -> tensor<128x2x10x1xf32>
  %tr_he = stablehlo.transpose %sl_he, dims = [3, 1, 0, 2] : (tensor<128x2x10x1xf32>) -> tensor<1x2x128x10xf32>
  %st_he = stablehlo.reshape %tr_he : (tensor<1x2x128x10xf32>) -> tensor<1x2x128x10x1xf32>
  %cat_h = stablehlo.concatenate %st_ho, %st_he, dim = 4 : (tensor<1x2x128x10x1xf32>, tensor<1x2x128x10x1xf32>) -> tensor<1x2x128x10x2xf32>
  %rot_h = stablehlo.reshape %cat_h : (tensor<1x2x128x10x2xf32>) -> tensor<1x2x128x20xf32>
  %mh_s = stablehlo.multiply %rot_h, %sh : tensor<1x2x128x20xf32>
  %out_h = stablehlo.add %mh_c, %mh_s : tensor<1x2x128x20xf32>

  %sl_w = stablehlo.slice %r3 [0:128, 0:2, 40:60] : (tensor<128x2x64xf32>) -> tensor<128x2x20xf32>
  %r4_w = stablehlo.reshape %sl_w : (tensor<128x2x20xf32>) -> tensor<128x2x20x1xf32>
  %qw = stablehlo.transpose %r4_w, dims = [3, 1, 0, 2] : (tensor<128x2x20x1xf32>) -> tensor<1x2x128x20xf32>
  %mw_c = stablehlo.multiply %qw, %cw : tensor<1x2x128x20xf32>
  %sl_wo = stablehlo.slice %r4_pair [0:128, 0:2, 20:30, 1:2] : (tensor<128x2x32x2xf32>) -> tensor<128x2x10x1xf32>
  %tr_wo = stablehlo.transpose %sl_wo, dims = [3, 1, 0, 2] : (tensor<128x2x10x1xf32>) -> tensor<1x2x128x10xf32>
  %neg_wo = stablehlo.negate %tr_wo : tensor<1x2x128x10xf32>
  %st_wo = stablehlo.reshape %neg_wo : (tensor<1x2x128x10xf32>) -> tensor<1x2x128x10x1xf32>
  %sl_we = stablehlo.slice %r4_pair [0:128, 0:2, 20:30, 0:1] : (tensor<128x2x32x2xf32>) -> tensor<128x2x10x1xf32>
  %tr_we = stablehlo.transpose %sl_we, dims = [3, 1, 0, 2] : (tensor<128x2x10x1xf32>) -> tensor<1x2x128x10xf32>
  %st_we = stablehlo.reshape %tr_we : (tensor<1x2x128x10xf32>) -> tensor<1x2x128x10x1xf32>
  %cat_w = stablehlo.concatenate %st_wo, %st_we, dim = 4 : (tensor<1x2x128x10x1xf32>, tensor<1x2x128x10x1xf32>) -> tensor<1x2x128x10x2xf32>
  %rot_w = stablehlo.reshape %cat_w : (tensor<1x2x128x10x2xf32>) -> tensor<1x2x128x20xf32>
  %mw_s = stablehlo.multiply %rot_w, %sw : tensor<1x2x128x20xf32>
  %out_w = stablehlo.add %mw_c, %mw_s : tensor<1x2x128x20xf32>

  %sl_r = stablehlo.slice %r3 [0:128, 0:2, 60:64] : (tensor<128x2x64xf32>) -> tensor<128x2x4xf32>
  %r4_r = stablehlo.reshape %sl_r : (tensor<128x2x4xf32>) -> tensor<128x2x4x1xf32>
  %out_r = stablehlo.transpose %r4_r, dims = [3, 1, 0, 2] : (tensor<128x2x4x1xf32>) -> tensor<1x2x128x4xf32>

  %out = stablehlo.concatenate %out_d, %out_h, %out_w, %out_r, dim = 3
      : (tensor<1x2x128x20xf32>, tensor<1x2x128x20xf32>, tensor<1x2x128x20xf32>, tensor<1x2x128x4xf32>) -> tensor<1x2x128x64xf32>
  return %out : tensor<1x2x128x64xf32>
}
)mlir";

  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  ASSERT_TRUE(
      mlir::succeeded(RunFoldAxialAndSplitHalfRopePass(*opt_module, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 2);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {128, 128}, 0.1f),
      MakeF32Tensor(context, {128, 128}, -0.2f),
      MakeF32Tensor(context, {128}, 0.05f),
      MakeF32Tensor(context, {1, 128, 20}, 0.2f),
      MakeF32Tensor(context, {1, 128, 20}, -0.1f),
      MakeF32Tensor(context, {1, 128, 20}, 0.3f),
      MakeF32Tensor(context, {1, 128, 20}, 0.15f),
      MakeF32Tensor(context, {1, 128, 20}, -0.25f),
      MakeF32Tensor(context, {1, 128, 20}, 0.05f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, inputs, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, inputs, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0], 1e-4f);
}

TEST(FoldAxialAndSplitHalfRopeTest,
     FoldsSplitHalfComplexRopeAndSinksProjection) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @complex_rope(
    %x_2d: tensor<256x128xf32>,
    %w: tensor<128x128xf32>,
    %bias: tensor<128xf32>,
    %cos_2d: tensor<128x32xf32>,
    %sin_2d: tensor<128x32xf32>) -> tensor<2x2x128x64xf32> {
  %dot = stablehlo.dot_general %x_2d, %w, contracting_dims = [1] x [0]
      : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
  %b2d = stablehlo.broadcast_in_dim %bias, dims = [1]
      : (tensor<128xf32>) -> tensor<256x128xf32>
  %addmm = stablehlo.add %b2d, %dot : tensor<256x128xf32>
  %r4 = stablehlo.reshape %addmm : (tensor<256x128xf32>) -> tensor<2x128x2x64xf32>
  %sl_lo = stablehlo.slice %r4 [0:2, 0:128, 0:2, 0:32]
      : (tensor<2x128x2x64xf32>) -> tensor<2x128x2x32xf32>
  %x_lo = stablehlo.transpose %sl_lo, dims = [0, 2, 1, 3]
      : (tensor<2x128x2x32xf32>) -> tensor<2x2x128x32xf32>
  %sl_hi = stablehlo.slice %r4 [0:2, 0:128, 0:2, 32:64]
      : (tensor<2x128x2x64xf32>) -> tensor<2x128x2x32xf32>
  %x_hi = stablehlo.transpose %sl_hi, dims = [0, 2, 1, 3]
      : (tensor<2x128x2x32xf32>) -> tensor<2x2x128x32xf32>
  %c_bcast = stablehlo.broadcast_in_dim %cos_2d, dims = [2, 3]
      : (tensor<128x32xf32>) -> tensor<2x2x128x32xf32>
  %s_bcast = stablehlo.broadcast_in_dim %sin_2d, dims = [2, 3]
      : (tensor<128x32xf32>) -> tensor<2x2x128x32xf32>
  %m1 = stablehlo.multiply %x_lo, %c_bcast : tensor<2x2x128x32xf32>
  %m2 = stablehlo.multiply %x_hi, %s_bcast : tensor<2x2x128x32xf32>
  %sub = stablehlo.subtract %m1, %m2 : tensor<2x2x128x32xf32>
  %m3 = stablehlo.multiply %x_hi, %c_bcast : tensor<2x2x128x32xf32>
  %m4 = stablehlo.multiply %x_lo, %s_bcast : tensor<2x2x128x32xf32>
  %add = stablehlo.add %m3, %m4 : tensor<2x2x128x32xf32>
  %out = stablehlo.concatenate %sub, %add, dim = 3
      : (tensor<2x2x128x32xf32>, tensor<2x2x128x32xf32>) -> tensor<2x2x128x64xf32>
  return %out : tensor<2x2x128x64xf32>
}
)mlir";

  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  ASSERT_TRUE(
      mlir::succeeded(RunFoldAxialAndSplitHalfRopePass(*opt_module, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 2);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {256, 128}, 0.1f),
      MakeF32Tensor(context, {128, 128}, -0.2f),
      MakeF32Tensor(context, {128}, 0.05f),
      MakeF32Tensor(context, {128, 32}, 0.25f),
      MakeF32Tensor(context, {128, 32}, -0.15f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, inputs, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, inputs, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0], 1e-4f);
}

TEST(FoldAxialAndSplitHalfRopeTest, FoldsSplitHalfComplexRopeBf16) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @complex_rope_bf16(
    %x_2d: tensor<256x128xbf16>,
    %w: tensor<128x128xbf16>,
    %bias: tensor<128xbf16>,
    %cos_2d: tensor<128x32xbf16>,
    %sin_2d: tensor<128x32xbf16>) -> tensor<2x2x128x64xbf16> {
  %dot = stablehlo.dot_general %x_2d, %w, contracting_dims = [1] x [0]
      : (tensor<256x128xbf16>, tensor<128x128xbf16>) -> tensor<256x128xbf16>
  %b2d = stablehlo.broadcast_in_dim %bias, dims = [1]
      : (tensor<128xbf16>) -> tensor<256x128xbf16>
  %addmm = stablehlo.add %b2d, %dot : tensor<256x128xbf16>
  %r4 = stablehlo.reshape %addmm : (tensor<256x128xbf16>) -> tensor<2x128x2x64xbf16>
  %sl_lo = stablehlo.slice %r4 [0:2, 0:128, 0:2, 0:32]
      : (tensor<2x128x2x64xbf16>) -> tensor<2x128x2x32xbf16>
  %x_lo = stablehlo.transpose %sl_lo, dims = [0, 2, 1, 3]
      : (tensor<2x128x2x32xbf16>) -> tensor<2x2x128x32xbf16>
  %sl_hi = stablehlo.slice %r4 [0:2, 0:128, 0:2, 32:64]
      : (tensor<2x128x2x64xbf16>) -> tensor<2x128x2x32xbf16>
  %x_hi = stablehlo.transpose %sl_hi, dims = [0, 2, 1, 3]
      : (tensor<2x128x2x32xbf16>) -> tensor<2x2x128x32xbf16>
  %c_bcast = stablehlo.broadcast_in_dim %cos_2d, dims = [2, 3]
      : (tensor<128x32xbf16>) -> tensor<2x2x128x32xbf16>
  %s_bcast = stablehlo.broadcast_in_dim %sin_2d, dims = [2, 3]
      : (tensor<128x32xbf16>) -> tensor<2x2x128x32xbf16>
  %m1 = stablehlo.multiply %x_lo, %c_bcast : tensor<2x2x128x32xbf16>
  %m2 = stablehlo.multiply %x_hi, %s_bcast : tensor<2x2x128x32xbf16>
  %sub = stablehlo.subtract %m1, %m2 : tensor<2x2x128x32xbf16>
  %m3 = stablehlo.multiply %x_hi, %c_bcast : tensor<2x2x128x32xbf16>
  %m4 = stablehlo.multiply %x_lo, %s_bcast : tensor<2x2x128x32xbf16>
  %add = stablehlo.add %m3, %m4 : tensor<2x2x128x32xbf16>
  %out = stablehlo.concatenate %sub, %add, dim = 3
      : (tensor<2x2x128x32xbf16>, tensor<2x2x128x32xbf16>) -> tensor<2x2x128x64xbf16>
  return %out : tensor<2x2x128x64xbf16>
}
)mlir";

  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  ASSERT_TRUE(
      mlir::succeeded(RunFoldAxialAndSplitHalfRopePass(*opt_module, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 2);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeBF16Tensor(context, {256, 128}, 0.1f),
      MakeBF16Tensor(context, {128, 128}, -0.2f),
      MakeBF16Tensor(context, {128}, 0.05f),
      MakeBF16Tensor(context, {128, 32}, 0.25f),
      MakeBF16Tensor(context, {128, 32}, -0.15f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, inputs, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, inputs, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0], 1e-2f);
}

TEST(FoldAxialAndSplitHalfRopeTest, FoldsBshdPreTransposeRotateHalf) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @bshd_rope(
    %x: tensor<1x64x4x32xf32>,
    %cos_2d: tensor<64x32xf32>,
    %sin_2d: tensor<64x32xf32>) -> tensor<1x4x64x32xf32> {
  %c_bcast = stablehlo.broadcast_in_dim %cos_2d, dims = [1, 3]
      : (tensor<64x32xf32>) -> tensor<1x64x4x32xf32>
  %s_bcast = stablehlo.broadcast_in_dim %sin_2d, dims = [1, 3]
      : (tensor<64x32xf32>) -> tensor<1x64x4x32xf32>
  %x_lo = stablehlo.slice %x [0:1, 0:64, 0:4, 0:16]
      : (tensor<1x64x4x32xf32>) -> tensor<1x64x4x16xf32>
  %x_hi = stablehlo.slice %x [0:1, 0:64, 0:4, 16:32]
      : (tensor<1x64x4x32xf32>) -> tensor<1x64x4x16xf32>
  %neg_hi = stablehlo.negate %x_hi : tensor<1x64x4x16xf32>
  %rot = stablehlo.concatenate %neg_hi, %x_lo, dim = 3
      : (tensor<1x64x4x16xf32>, tensor<1x64x4x16xf32>) -> tensor<1x64x4x32xf32>
  %m_cos = stablehlo.multiply %x, %c_bcast : tensor<1x64x4x32xf32>
  %m_sin = stablehlo.multiply %rot, %s_bcast : tensor<1x64x4x32xf32>
  %added = stablehlo.add %m_cos, %m_sin : tensor<1x64x4x32xf32>
  %out = stablehlo.transpose %added, dims = [0, 2, 1, 3]
      : (tensor<1x64x4x32xf32>) -> tensor<1x4x64x32xf32>
  return %out : tensor<1x4x64x32xf32>
}
)mlir";

  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  ASSERT_TRUE(
      mlir::succeeded(RunFoldAxialAndSplitHalfRopePass(*opt_module, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {1, 64, 4, 32}, 0.1f),
      MakeF32Tensor(context, {64, 32}, 0.25f),
      MakeF32Tensor(context, {64, 32}, -0.15f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, inputs, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, inputs, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0], 1e-4f);
}

TEST(FoldAxialAndSplitHalfRopeTest, FoldsSingleHeadDecodeBshdRoPE) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @single_head_decode(
    %x: tensor<1x1x1x32xf32>,
    %cos_2d: tensor<1x32xf32>,
    %sin_2d: tensor<1x32xf32>) -> tensor<1x1x1x32xf32> {
  %c_bcast = stablehlo.broadcast_in_dim %cos_2d, dims = [1, 3]
      : (tensor<1x32xf32>) -> tensor<1x1x1x32xf32>
  %s_bcast = stablehlo.broadcast_in_dim %sin_2d, dims = [1, 3]
      : (tensor<1x32xf32>) -> tensor<1x1x1x32xf32>
  %x_lo = stablehlo.slice %x [0:1, 0:1, 0:1, 0:16]
      : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x16xf32>
  %x_hi = stablehlo.slice %x [0:1, 0:1, 0:1, 16:32]
      : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x16xf32>
  %neg_hi = stablehlo.negate %x_hi : tensor<1x1x1x16xf32>
  %rot = stablehlo.concatenate %neg_hi, %x_lo, dim = 3
      : (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x32xf32>
  %m_cos = stablehlo.multiply %x, %c_bcast : tensor<1x1x1x32xf32>
  %m_sin = stablehlo.multiply %rot, %s_bcast : tensor<1x1x1x32xf32>
  %added = stablehlo.add %m_cos, %m_sin : tensor<1x1x1x32xf32>
  %out = stablehlo.transpose %added, dims = [0, 2, 1, 3]
      : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
  return %out : tensor<1x1x1x32xf32>
}
)mlir";

  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  ASSERT_TRUE(
      mlir::succeeded(RunFoldAxialAndSplitHalfRopePass(*opt_module, &context)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {1, 1, 1, 32}, 0.1f),
      MakeF32Tensor(context, {1, 32}, 0.25f),
      MakeF32Tensor(context, {1, 32}, -0.15f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, inputs, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, inputs, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0], 1e-4f);
}

}  // namespace
}  // namespace torch_tpu
