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

#include <cmath>
#include <complex>
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
                                      float seed_offset) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] =
        seed_offset + static_cast<float>((i * 19 + 7) % 32) * 0.125f - 2.0f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape,
                                       float seed_offset) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    float f =
        seed_offset + static_cast<float>((i * 13 + 3) % 32) * 0.125f - 2.0f;
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

mlir::DenseElementsAttr MakeComplex64Tensor(mlir::MLIRContext& ctx,
                                            llvm::ArrayRef<int64_t> shape) {
  auto c_ty = mlir::ComplexType::get(mlir::Float32Type::get(&ctx));
  auto type = mlir::RankedTensorType::get(shape, c_ty);
  const int64_t n = type.getNumElements();
  std::vector<std::complex<float>> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    float re = static_cast<float>((i * 11 + 5) % 16) * 0.125f - 1.0f;
    float im = static_cast<float>((i * 7 + 9) % 16) * 0.125f - 1.0f;
    vals[i] = std::complex<float>(re, im);
  }
  return mlir::DenseElementsAttr::get(
      type, llvm::ArrayRef<std::complex<float>>(vals));
}

TEST(FactorHeadNormAndComplexRopeSublanesTest,
     FactorsForwardHeadNormAndComplexRopeTo4DSublanes) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @forward_qnorm_rope(
    %wq_out: tensor<2048x4096xbf16>,
    %freqs_cis: tensor<2048x64xcomplex<f32>>) -> tensor<1x32x2048x128xbf16> {
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %inv_d = stablehlo.constant dense<0.0078125> : tensor<f32>
  %eps = stablehlo.constant dense<1.0e-5> : tensor<f32>
  %scale = stablehlo.constant dense<3.875> : tensor<bf16>
  %b_inv_d = stablehlo.broadcast_in_dim %inv_d, dims = [] : (tensor<f32>) -> tensor<2048x32xf32>
  %b_eps = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<2048x32xf32>
  %b_scale = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<bf16>) -> tensor<2048x32x128xbf16>
  %x_3d = stablehlo.reshape %wq_out : (tensor<2048x4096xbf16>) -> tensor<2048x32x128xbf16>
  %x_f32 = stablehlo.convert %x_3d : (tensor<2048x32x128xbf16>) -> tensor<2048x32x128xf32>
  %sq = stablehlo.multiply %x_f32, %x_f32 : tensor<2048x32x128xf32>
  %red = stablehlo.reduce(%sq init: %zero) across dimensions = [2] : (tensor<2048x32x128xf32>, tensor<f32>) -> tensor<2048x32xf32>
    reducer(%a: tensor<f32>, %b: tensor<f32>) {
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }
  %mean = stablehlo.multiply %red, %b_inv_d : tensor<2048x32xf32>
  %var_eps = stablehlo.add %mean, %b_eps : tensor<2048x32xf32>
  %rstd = stablehlo.rsqrt %var_eps : tensor<2048x32xf32>
  %b_rstd = stablehlo.broadcast_in_dim %rstd, dims = [0, 1] : (tensor<2048x32xf32>) -> tensor<2048x32x128xf32>
  %norm_f32 = stablehlo.multiply %x_f32, %b_rstd : tensor<2048x32x128xf32>
  %norm_bf16 = stablehlo.convert %norm_f32 : (tensor<2048x32x128xf32>) -> tensor<2048x32x128xbf16>
  %scaled = stablehlo.multiply %norm_bf16, %b_scale : tensor<2048x32x128xbf16>
  %rope_in = stablehlo.convert %scaled : (tensor<2048x32x128xbf16>) -> tensor<2048x32x128xf32>
  %s_even = stablehlo.slice %rope_in [0:2048:1, 0:32:1, 0:128:2] : (tensor<2048x32x128xf32>) -> tensor<2048x32x64xf32>
  %s_odd = stablehlo.slice %rope_in [0:2048:1, 0:32:1, 1:128:2] : (tensor<2048x32x128xf32>) -> tensor<2048x32x64xf32>
  %z_in = stablehlo.complex %s_even, %s_odd : tensor<2048x32x64xcomplex<f32>>
  %cis_3d = stablehlo.reshape %freqs_cis : (tensor<2048x64xcomplex<f32>>) -> tensor<2048x1x64xcomplex<f32>>
  %cis_b1 = stablehlo.broadcast_in_dim %cis_3d, dims = [0, 1, 2] : (tensor<2048x1x64xcomplex<f32>>) -> tensor<2048x1x64xcomplex<f32>>
  %cis_2d = stablehlo.reshape %cis_b1 : (tensor<2048x1x64xcomplex<f32>>) -> tensor<2048x64xcomplex<f32>>
  %cis_b2 = stablehlo.broadcast_in_dim %cis_2d, dims = [0, 2] : (tensor<2048x64xcomplex<f32>>) -> tensor<2048x32x64xcomplex<f32>>
  %z_out = stablehlo.multiply %z_in, %cis_b2 : tensor<2048x32x64xcomplex<f32>>
  %re = stablehlo.real %z_out : (tensor<2048x32x64xcomplex<f32>>) -> tensor<2048x32x64xf32>
  %re_4d = stablehlo.reshape %re : (tensor<2048x32x64xf32>) -> tensor<2048x32x64x1xf32>
  %im = stablehlo.imag %z_out : (tensor<2048x32x64xcomplex<f32>>) -> tensor<2048x32x64xf32>
  %im_4d = stablehlo.reshape %im : (tensor<2048x32x64xf32>) -> tensor<2048x32x64x1xf32>
  %cat = stablehlo.concatenate %re_4d, %im_4d, dim = 3 : (tensor<2048x32x64x1xf32>, tensor<2048x32x64x1xf32>) -> tensor<2048x32x64x2xf32>
  %rope_f32 = stablehlo.reshape %cat : (tensor<2048x32x64x2xf32>) -> tensor<2048x32x128xf32>
  %rope_bf16 = stablehlo.convert %rope_f32 : (tensor<2048x32x128xf32>) -> tensor<2048x32x128xbf16>
  %rope_4d = stablehlo.reshape %rope_bf16 : (tensor<2048x32x128xbf16>) -> tensor<2048x32x128x1xbf16>
  %out = stablehlo.transpose %rope_4d, dims = [3, 1, 0, 2] : (tensor<2048x32x128x1xbf16>) -> tensor<1x32x2048x128xbf16>
  return %out : tensor<1x32x2048x128xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());
  const std::string printed = PrintModule(*module);

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ComplexOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 1);
  EXPECT_THAT(printed, testing::HasSubstr("tensor<256x8x32x128xf32>"));
  EXPECT_THAT(printed, testing::HasSubstr("tensor<256x8x32xf32>"));
}

TEST(
    FactorHeadNormAndComplexRopeSublanesTest,
    ExactNumericalEquivalenceForwardAndBackwardWithConjugateRopeAndWeightGrad) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %wq_out: tensor<16x1024xbf16>,
    %freqs_cis: tensor<16x64xcomplex<f32>>,
    %weight: tensor<128xf32>,
    %dout_4d: tensor<1x8x16x128xbf16>)
    -> (tensor<1x8x16x128xbf16>, tensor<16x1024xbf16>, tensor<128xf32>) {
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %inv_d = stablehlo.constant dense<0.0078125> : tensor<f32>
  %eps = stablehlo.constant dense<1.0e-5> : tensor<f32>
  %b_inv_d = stablehlo.broadcast_in_dim %inv_d, dims = [] : (tensor<f32>) -> tensor<16x8xf32>
  %b_eps = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x8xf32>
  %b_w = stablehlo.broadcast_in_dim %weight, dims = [2] : (tensor<128xf32>) -> tensor<16x8x128xf32>
  %x_3d = stablehlo.reshape %wq_out : (tensor<16x1024xbf16>) -> tensor<16x8x128xbf16>
  %x_f32 = stablehlo.convert %x_3d : (tensor<16x8x128xbf16>) -> tensor<16x8x128xf32>
  %sq = stablehlo.multiply %x_f32, %x_f32 : tensor<16x8x128xf32>
  %red = stablehlo.reduce(%sq init: %zero) across dimensions = [2] : (tensor<16x8x128xf32>, tensor<f32>) -> tensor<16x8xf32>
    reducer(%a: tensor<f32>, %b: tensor<f32>) {
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }
  %mean = stablehlo.multiply %red, %b_inv_d : tensor<16x8xf32>
  %var_eps = stablehlo.add %mean, %b_eps : tensor<16x8xf32>
  %rstd = stablehlo.rsqrt %var_eps : tensor<16x8xf32>
  %b_rstd = stablehlo.broadcast_in_dim %rstd, dims = [0, 1] : (tensor<16x8xf32>) -> tensor<16x8x128xf32>
  %norm_f32 = stablehlo.multiply %x_f32, %b_rstd : tensor<16x8x128xf32>
  %weighted_f32 = stablehlo.multiply %norm_f32, %b_w : tensor<16x8x128xf32>
  %scaled_bf16 = stablehlo.convert %weighted_f32 : (tensor<16x8x128xf32>) -> tensor<16x8x128xbf16>
  %rope_in = stablehlo.convert %scaled_bf16 : (tensor<16x8x128xbf16>) -> tensor<16x8x128xf32>
  %s_even = stablehlo.slice %rope_in [0:16:1, 0:8:1, 0:128:2] : (tensor<16x8x128xf32>) -> tensor<16x8x64xf32>
  %s_odd = stablehlo.slice %rope_in [0:16:1, 0:8:1, 1:128:2] : (tensor<16x8x128xf32>) -> tensor<16x8x64xf32>
  %z_in = stablehlo.complex %s_even, %s_odd : tensor<16x8x64xcomplex<f32>>
  %cis_3d = stablehlo.reshape %freqs_cis : (tensor<16x64xcomplex<f32>>) -> tensor<16x1x64xcomplex<f32>>
  %cis_2d = stablehlo.reshape %cis_3d : (tensor<16x1x64xcomplex<f32>>) -> tensor<16x64xcomplex<f32>>
  %cis_b = stablehlo.broadcast_in_dim %cis_2d, dims = [0, 2] : (tensor<16x64xcomplex<f32>>) -> tensor<16x8x64xcomplex<f32>>
  %z_out = stablehlo.multiply %z_in, %cis_b : tensor<16x8x64xcomplex<f32>>
  %re = stablehlo.real %z_out : (tensor<16x8x64xcomplex<f32>>) -> tensor<16x8x64xf32>
  %re_4d = stablehlo.reshape %re : (tensor<16x8x64xf32>) -> tensor<16x8x64x1xf32>
  %im = stablehlo.imag %z_out : (tensor<16x8x64xcomplex<f32>>) -> tensor<16x8x64xf32>
  %im_4d = stablehlo.reshape %im : (tensor<16x8x64xf32>) -> tensor<16x8x64x1xf32>
  %cat = stablehlo.concatenate %re_4d, %im_4d, dim = 3 : (tensor<16x8x64x1xf32>, tensor<16x8x64x1xf32>) -> tensor<16x8x64x2xf32>
  %rope_f32 = stablehlo.reshape %cat : (tensor<16x8x64x2xf32>) -> tensor<16x8x128xf32>
  %rope_bf16 = stablehlo.convert %rope_f32 : (tensor<16x8x128xf32>) -> tensor<16x8x128xbf16>
  %rope_4d = stablehlo.reshape %rope_bf16 : (tensor<16x8x128xbf16>) -> tensor<16x8x128x1xbf16>
  %fwd_out = stablehlo.transpose %rope_4d, dims = [3, 1, 0, 2] : (tensor<16x8x128x1xbf16>) -> tensor<1x8x16x128xbf16>

  // Backward pass with conjugate cis and RMSNorm backward + weight grad.
  %d_t = stablehlo.transpose %dout_4d, dims = [2, 1, 3, 0] : (tensor<1x8x16x128xbf16>) -> tensor<16x8x128x1xbf16>
  %d_3d = stablehlo.reshape %d_t : (tensor<16x8x128x1xbf16>) -> tensor<16x8x128xbf16>
  %d_f32 = stablehlo.convert %d_3d : (tensor<16x8x128xbf16>) -> tensor<16x8x128xf32>
  %ds_even = stablehlo.slice %d_f32 [0:16:1, 0:8:1, 0:128:2] : (tensor<16x8x128xf32>) -> tensor<16x8x64xf32>
  %ds_odd = stablehlo.slice %d_f32 [0:16:1, 0:8:1, 1:128:2] : (tensor<16x8x128xf32>) -> tensor<16x8x64xf32>
  %dz_in = stablehlo.complex %ds_even, %ds_odd : tensor<16x8x64xcomplex<f32>>
  %c_re = stablehlo.real %cis_3d : (tensor<16x1x64xcomplex<f32>>) -> tensor<16x1x64xf32>
  %c_im = stablehlo.imag %cis_3d : (tensor<16x1x64xcomplex<f32>>) -> tensor<16x1x64xf32>
  %c_neg_im = stablehlo.negate %c_im : tensor<16x1x64xf32>
  %cis_conj_3d = stablehlo.complex %c_re, %c_neg_im : tensor<16x1x64xcomplex<f32>>
  %cis_conj_2d = stablehlo.reshape %cis_conj_3d : (tensor<16x1x64xcomplex<f32>>) -> tensor<16x64xcomplex<f32>>
  %cis_conj_b = stablehlo.broadcast_in_dim %cis_conj_2d, dims = [0, 2] : (tensor<16x64xcomplex<f32>>) -> tensor<16x8x64xcomplex<f32>>
  %dz_out = stablehlo.multiply %dz_in, %cis_conj_b : tensor<16x8x64xcomplex<f32>>
  %dre = stablehlo.real %dz_out : (tensor<16x8x64xcomplex<f32>>) -> tensor<16x8x64xf32>
  %dre_4d = stablehlo.reshape %dre : (tensor<16x8x64xf32>) -> tensor<16x8x64x1xf32>
  %dim = stablehlo.imag %dz_out : (tensor<16x8x64xcomplex<f32>>) -> tensor<16x8x64xf32>
  %dim_4d = stablehlo.reshape %dim : (tensor<16x8x64xf32>) -> tensor<16x8x64x1xf32>
  %dcat = stablehlo.concatenate %dre_4d, %dim_4d, dim = 3 : (tensor<16x8x64x1xf32>, tensor<16x8x64x1xf32>) -> tensor<16x8x64x2xf32>
  %drope_f32 = stablehlo.reshape %dcat : (tensor<16x8x64x2xf32>) -> tensor<16x8x128xf32>
  %drope_bf16 = stablehlo.convert %drope_f32 : (tensor<16x8x128xf32>) -> tensor<16x8x128xbf16>
  %dy = stablehlo.convert %drope_bf16 : (tensor<16x8x128xbf16>) -> tensor<16x8x128xf32>
  %dw_prod = stablehlo.multiply %dy, %norm_f32 : tensor<16x8x128xf32>
  %dw = stablehlo.reduce(%dw_prod init: %zero) across dimensions = [0, 1] : (tensor<16x8x128xf32>, tensor<f32>) -> tensor<128xf32>
    reducer(%a: tensor<f32>, %b: tensor<f32>) {
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }
  %dy_w = stablehlo.multiply %dy, %b_w : tensor<16x8x128xf32>
  %rstd_3d = stablehlo.reshape %rstd : (tensor<16x8xf32>) -> tensor<16x8x1xf32>
  %rstd_b1 = stablehlo.broadcast_in_dim %rstd_3d, dims = [0, 1, 2] : (tensor<16x8x1xf32>) -> tensor<16x8x1xf32>
  %rstd_2d = stablehlo.reshape %rstd_b1 : (tensor<16x8x1xf32>) -> tensor<16x8xf32>
  %rstd_b2 = stablehlo.broadcast_in_dim %rstd_2d, dims = [0, 1] : (tensor<16x8xf32>) -> tensor<16x8x128xf32>
  %x_hat = stablehlo.multiply %x_f32, %rstd_b2 : tensor<16x8x128xf32>
  %prod_bwd = stablehlo.multiply %dy_w, %x_hat : tensor<16x8x128xf32>
  %red_bwd = stablehlo.reduce(%prod_bwd init: %zero) across dimensions = [2] : (tensor<16x8x128xf32>, tensor<f32>) -> tensor<16x8xf32>
    reducer(%a: tensor<f32>, %b: tensor<f32>) {
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }
  %mean_bwd = stablehlo.multiply %red_bwd, %b_inv_d : tensor<16x8xf32>
  %b_mean_bwd = stablehlo.broadcast_in_dim %mean_bwd, dims = [0, 1] : (tensor<16x8xf32>) -> tensor<16x8x128xf32>
  %proj_bwd = stablehlo.multiply %x_hat, %b_mean_bwd : tensor<16x8x128xf32>
  %diff_bwd = stablehlo.subtract %dy_w, %proj_bwd : tensor<16x8x128xf32>
  %dx_f32 = stablehlo.multiply %diff_bwd, %rstd_b2 : tensor<16x8x128xf32>
  %dx_bf16 = stablehlo.convert %dx_f32 : (tensor<16x8x128xf32>) -> tensor<16x8x128xbf16>
  %dx_2d = stablehlo.reshape %dx_bf16 : (tensor<16x8x128xbf16>) -> tensor<16x1024xbf16>
  return %fwd_out, %dx_2d, %dw : tensor<1x8x16x128xbf16>, tensor<16x1024xbf16>, tensor<128xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr wq_out = MakeBF16Tensor(context, {16, 1024}, 0.25f);
  mlir::DenseElementsAttr freqs_cis = MakeComplex64Tensor(context, {16, 64});
  mlir::DenseElementsAttr weight = MakeF32Tensor(context, {128}, 1.0f);
  mlir::DenseElementsAttr dout_4d =
      MakeBF16Tensor(context, {1, 8, 16, 128}, -0.125f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {wq_out, freqs_cis, weight, dout_4d}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ComplexOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_mod), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 2);
  EXPECT_THAT(PrintModule(*opt_mod),
              testing::HasSubstr("tensor<2x8x8x128xf32>"));

  auto actual = mlir::stablehlo::evalModule(
      *opt_mod, {wq_out, freqs_cis, weight, dout_4d}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), 3u);
  ASSERT_EQ(actual->size(), 3u);

  for (int out_idx = 0; out_idx < 2; ++out_idx) {
    auto exp_it = (*expected)[out_idx].value_begin<llvm::APFloat>();
    auto act_it = (*actual)[out_idx].value_begin<llvm::APFloat>();
    for (; exp_it != (*expected)[out_idx].value_end<llvm::APFloat>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ((*act_it).convertToFloat(), (*exp_it).convertToFloat());
    }
  }
  auto exp_dw_it = (*expected)[2].value_begin<float>();
  auto act_dw_it = (*actual)[2].value_begin<float>();
  for (; exp_dw_it != (*expected)[2].value_end<float>();
       ++exp_dw_it, ++act_dw_it) {
    EXPECT_NEAR(*act_dw_it, *exp_dw_it, 1e-4f);
  }
}

TEST(FactorHeadNormAndComplexRopeSublanesTest,
     PreservesSubgraphWhenHeadCountSmallerThanSublaneTile) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kSmallHeadsIr = R"mlir(
func.func @small_heads_kv(
    %wk_out: tensor<2048x256xbf16>) -> tensor<1x2x2048x128xbf16> {
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %x_3d = stablehlo.reshape %wk_out : (tensor<2048x256xbf16>) -> tensor<2048x2x128xbf16>
  %x_f32 = stablehlo.convert %x_3d : (tensor<2048x2x128xbf16>) -> tensor<2048x2x128xf32>
  %sq = stablehlo.multiply %x_f32, %x_f32 : tensor<2048x2x128xf32>
  %red = stablehlo.reduce(%sq init: %zero) across dimensions = [2] : (tensor<2048x2x128xf32>, tensor<f32>) -> tensor<2048x2xf32>
    reducer(%a: tensor<f32>, %b: tensor<f32>) {
      %s = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }
  %rstd = stablehlo.rsqrt %red : tensor<2048x2xf32>
  %b_rstd = stablehlo.broadcast_in_dim %rstd, dims = [0, 1] : (tensor<2048x2xf32>) -> tensor<2048x2x128xf32>
  %norm_f32 = stablehlo.multiply %x_f32, %b_rstd : tensor<2048x2x128xf32>
  %norm_bf16 = stablehlo.convert %norm_f32 : (tensor<2048x2x128xf32>) -> tensor<2048x2x128xbf16>
  %norm_4d = stablehlo.reshape %norm_bf16 : (tensor<2048x2x128xbf16>) -> tensor<2048x2x128x1xbf16>
  %out = stablehlo.transpose %norm_4d, dims = [3, 1, 0, 2] : (tensor<2048x2x128x1xbf16>) -> tensor<1x2x2048x128xbf16>
  return %out : tensor<1x2x2048x128xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kSmallHeadsIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());
  const std::string printed = PrintModule(*module);
  EXPECT_THAT(printed, testing::HasSubstr("tensor<2048x2x128xf32>"));
  EXPECT_THAT(printed,
              testing::Not(testing::HasSubstr("tensor<256x8x2x128xf32>")));
}

}  // namespace
}  // namespace torch_tpu
