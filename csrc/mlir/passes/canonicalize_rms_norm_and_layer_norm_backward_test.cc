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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
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
#include "mlir/Support/LLVM.h"
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
    vals[i] = base + static_cast<float>((i * 17 + 5) % 31) * 0.0625f;
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
    float f = base + static_cast<float>((i * 13 + 3) % 19) * 0.125f - 1.0f;
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

TEST(CanonicalizeRmsNormAndLayerNormBackwardTest,
     Canonicalizes3DBf16RmsNormBackwardAndFusesDgamma) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %dy: tensor<1x4x8xbf16>,
      %x: tensor<1x4x8xbf16>,
      %rstd: tensor<1x4x1xf32>,
      %gamma: tensor<8xbf16>)
      -> (tensor<1x4x8xbf16>, tensor<8xbf16>) {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero_bf16 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %three_s = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %three = stablehlo.broadcast_in_dim %three_s, dims = [] : (tensor<f32>) -> tensor<1x4x1xf32>
    %eight_s = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %eight = stablehlo.broadcast_in_dim %eight_s, dims = [] : (tensor<f32>) -> tensor<1x4x8xf32>
    %gamma_bcast = stablehlo.broadcast_in_dim %gamma, dims = [2] : (tensor<8xbf16>) -> tensor<1x4x8xbf16>
    %dy_gamma_bf16 = stablehlo.multiply %dy, %gamma_bcast : tensor<1x4x8xbf16>
    %dy_gamma_f32 = stablehlo.convert %dy_gamma_bf16 : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %rstd_bcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %term1_f32 = stablehlo.multiply %rstd_bcast, %dy_gamma_f32 : tensor<1x4x8xf32>
    %x_f32 = stablehlo.convert %x : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %rstd_cubed = stablehlo.power %rstd, %three : tensor<1x4x1xf32>
    %rstd_cubed_bcast = stablehlo.broadcast_in_dim %rstd_cubed, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %x_rstd3 = stablehlo.multiply %x_f32, %rstd_cubed_bcast : tensor<1x4x8xf32>
    %x_rstd3_div = stablehlo.divide %x_rstd3, %eight : tensor<1x4x8xf32>
    %prod_bf16 = stablehlo.multiply %x, %dy_gamma_bf16 : tensor<1x4x8xbf16>
    %prod_f32 = stablehlo.convert %prod_bf16 : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %dot_red_f32 = stablehlo.reduce(%prod_f32 init: %zero_f32) across dimensions = [2] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %dot_keepdim_f32 = stablehlo.reshape %dot_red_f32 : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %dot_keepdim = stablehlo.convert %dot_keepdim_f32 : (tensor<1x4x1xf32>) -> tensor<1x4x1xbf16>
    %dot_f32 = stablehlo.convert %dot_keepdim : (tensor<1x4x1xbf16>) -> tensor<1x4x1xf32>
    %dot_bcast = stablehlo.broadcast_in_dim %dot_f32, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %term2_f32 = stablehlo.multiply %x_rstd3_div, %dot_bcast : tensor<1x4x8xf32>
    %dx_f32 = stablehlo.subtract %term1_f32, %term2_f32 : tensor<1x4x8xf32>
    %dx = stablehlo.convert %dx_f32 : (tensor<1x4x8xf32>) -> tensor<1x4x8xbf16>
    %dy_f32 = stablehlo.convert %dy : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %x_hat = stablehlo.multiply %x_f32, %rstd_bcast : tensor<1x4x8xf32>
    %dgamma_prod_f32 = stablehlo.multiply %dy_f32, %x_hat : tensor<1x4x8xf32>
    %dgamma_f32 = stablehlo.reduce(%dgamma_prod_f32 init: %zero_f32) across dimensions = [0, 1] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %dgamma = stablehlo.convert %dgamma_f32 : (tensor<8xf32>) -> tensor<8xbf16>
    return %dx, %dgamma : tensor<1x4x8xbf16>, tensor<8xbf16>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr dy = MakeBF16Tensor(context, {1, 4, 8}, 0.25f);
  mlir::DenseElementsAttr x = MakeBF16Tensor(context, {1, 4, 8}, -0.5f);
  mlir::DenseElementsAttr rstd = MakeF32Tensor(context, {1, 4, 1}, 0.8f);
  mlir::DenseElementsAttr gamma = MakeBF16Tensor(context, {8}, 0.9f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {dy, x, rstd, gamma}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  int pow_count = 0;
  int div_count = 0;
  int reduce_2d_count = 0;
  int reduce_3d_count = 0;
  opt_mod->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::PowOp>(op)) {
      ++pow_count;
    } else if (mlir::isa<mlir::stablehlo::DivOp>(op)) {
      ++div_count;
    } else if (auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
      auto in_type =
          mlir::cast<mlir::RankedTensorType>(red.getInputs()[0].getType());
      if (in_type.getRank() == 2) {
        ++reduce_2d_count;
      } else if (in_type.getRank() == 3) {
        ++reduce_3d_count;
      }
    }
  });
  EXPECT_EQ(pow_count, 0);
  EXPECT_EQ(div_count, 0);
  EXPECT_EQ(reduce_2d_count, 2);
  EXPECT_EQ(reduce_3d_count, 0);

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {dy, x, rstd, gamma}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(actual->size(), expected->size());

  for (size_t idx = 0; idx < expected->size(); ++idx) {
    ASSERT_EQ((*actual)[idx].getNumElements(),
              (*expected)[idx].getNumElements());
    auto exp_it = (*expected)[idx].value_begin<llvm::APFloat>();
    auto act_it = (*actual)[idx].value_begin<llvm::APFloat>();
    for (; exp_it != (*expected)[idx].value_end<llvm::APFloat>();
         ++exp_it, ++act_it) {
      EXPECT_NEAR((*act_it).convertToFloat(), (*exp_it).convertToFloat(),
                  5e-2f);
    }
  }
}

TEST(CanonicalizeRmsNormAndLayerNormBackwardTest,
     Canonicalizes2DF32RmsNormBackwardWithoutWeight) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %dy: tensor<4x8xf32>,
      %x: tensor<4x8xf32>,
      %rstd: tensor<4x1xf32>)
      -> tensor<4x8xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %three_s = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %three = stablehlo.broadcast_in_dim %three_s, dims = [] : (tensor<f32>) -> tensor<4x1xf32>
    %eight_s = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %eight = stablehlo.broadcast_in_dim %eight_s, dims = [] : (tensor<f32>) -> tensor<4x8xf32>
    %rstd_bcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x8xf32>
    %term1 = stablehlo.multiply %dy, %rstd_bcast : tensor<4x8xf32>
    %rstd_cubed = stablehlo.power %rstd, %three : tensor<4x1xf32>
    %rstd_cubed_bcast = stablehlo.broadcast_in_dim %rstd_cubed, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x8xf32>
    %x_rstd3 = stablehlo.multiply %x, %rstd_cubed_bcast : tensor<4x8xf32>
    %x_rstd3_div = stablehlo.divide %x_rstd3, %eight : tensor<4x8xf32>
    %prod = stablehlo.multiply %x, %dy : tensor<4x8xf32>
    %dot = stablehlo.reduce(%prod init: %zero) across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %dot_keepdim = stablehlo.reshape %dot : (tensor<4xf32>) -> tensor<4x1xf32>
    %dot_bcast = stablehlo.broadcast_in_dim %dot_keepdim, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x8xf32>
    %term2 = stablehlo.multiply %x_rstd3_div, %dot_bcast : tensor<4x8xf32>
    %dx = stablehlo.subtract %term1, %term2 : tensor<4x8xf32>
    return %dx : tensor<4x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr dy = MakeF32Tensor(context, {4, 8}, 0.25f);
  mlir::DenseElementsAttr x = MakeF32Tensor(context, {4, 8}, -0.5f);
  mlir::DenseElementsAttr rstd = MakeF32Tensor(context, {4, 1}, 0.8f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {dy, x, rstd}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  int pow_count = 0;
  int div_count = 0;
  opt_mod->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::PowOp>(op)) {
      ++pow_count;
    } else if (mlir::isa<mlir::stablehlo::DivOp>(op)) {
      ++div_count;
    }
  });
  EXPECT_EQ(pow_count, 0);
  EXPECT_EQ(div_count, 0);

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {dy, x, rstd}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(actual->size(), expected->size());

  auto exp_it = (*expected)[0].value_begin<llvm::APFloat>();
  auto act_it = (*actual)[0].value_begin<llvm::APFloat>();
  for (; exp_it != (*expected)[0].value_end<llvm::APFloat>();
       ++exp_it, ++act_it) {
    EXPECT_NEAR((*act_it).convertToFloat(), (*exp_it).convertToFloat(), 1e-5f);
  }
}

TEST(CanonicalizeRmsNormAndLayerNormBackwardTest,
     SqueezesUnitBatchDimIn3DBf16LayerNormBackward) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %dy_2d: tensor<4x8xbf16>,
      %x: tensor<1x4x8xbf16>,
      %mean: tensor<1x4x1xf32>,
      %rstd: tensor<1x4x1xf32>,
      %gamma: tensor<8xbf16>)
      -> (tensor<1x4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>) {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %inv_n_s = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %inv_n = stablehlo.broadcast_in_dim %inv_n_s, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %dy = stablehlo.reshape %dy_2d : (tensor<4x8xbf16>) -> tensor<1x4x8xbf16>
    %x_f32 = stablehlo.convert %x : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %dy_f32 = stablehlo.convert %dy : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %gamma_f32 = stablehlo.convert %gamma : (tensor<8xbf16>) -> tensor<8xf32>
    %mean_bcast = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %rstd_bcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %x_sub = stablehlo.subtract %x_f32, %mean_bcast : tensor<1x4x8xf32>
    %x_hat = stablehlo.multiply %x_sub, %rstd_bcast : tensor<1x4x8xf32>
    %gamma_bcast = stablehlo.broadcast_in_dim %gamma_f32, dims = [2] : (tensor<8xf32>) -> tensor<1x4x8xf32>
    %dgamma_full = stablehlo.multiply %dy_f32, %x_hat : tensor<1x4x8xf32>
    %dgamma_f32 = stablehlo.reduce(%dgamma_full init: %zero) across dimensions = [0, 1] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %dbeta_f32 = stablehlo.reduce(%dy_f32 init: %zero) across dimensions = [0, 1] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %dy_gamma = stablehlo.multiply %dy_f32, %gamma_bcast : tensor<1x4x8xf32>
    %a1 = stablehlo.reduce(%dy_gamma init: %zero) across dimensions = [2] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a2_full = stablehlo.multiply %dy_gamma, %x_hat : tensor<1x4x8xf32>
    %a2 = stablehlo.reduce(%a2_full init: %zero) across dimensions = [2] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a1_scaled = stablehlo.multiply %a1, %inv_n : tensor<1x4xf32>
    %a2_scaled = stablehlo.multiply %a2, %inv_n : tensor<1x4xf32>
    %a1_bcast = stablehlo.broadcast_in_dim %a1_scaled, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x8xf32>
    %a2_bcast = stablehlo.broadcast_in_dim %a2_scaled, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x8xf32>
    %dx_sub1 = stablehlo.subtract %dy_gamma, %a1_bcast : tensor<1x4x8xf32>
    %x_hat_a2 = stablehlo.multiply %x_hat, %a2_bcast : tensor<1x4x8xf32>
    %dx_sub2 = stablehlo.subtract %dx_sub1, %x_hat_a2 : tensor<1x4x8xf32>
    %dx_f32 = stablehlo.multiply %dx_sub2, %rstd_bcast : tensor<1x4x8xf32>
    %dx = stablehlo.convert %dx_f32 : (tensor<1x4x8xf32>) -> tensor<1x4x8xbf16>
    %dgamma = stablehlo.convert %dgamma_f32 : (tensor<8xf32>) -> tensor<8xbf16>
    %dbeta = stablehlo.convert %dbeta_f32 : (tensor<8xf32>) -> tensor<8xbf16>
    return %dx, %dgamma, %dbeta : tensor<1x4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr dy_2d = MakeBF16Tensor(context, {4, 8}, 0.25f);
  mlir::DenseElementsAttr x = MakeBF16Tensor(context, {1, 4, 8}, -0.5f);
  mlir::DenseElementsAttr mean = MakeF32Tensor(context, {1, 4, 1}, 0.1f);
  mlir::DenseElementsAttr rstd = MakeF32Tensor(context, {1, 4, 1}, 0.8f);
  mlir::DenseElementsAttr gamma = MakeBF16Tensor(context, {8}, 0.9f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {dy_2d, x, mean, rstd, gamma}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  int reduce_2d_count = 0;
  int reduce_3d_count = 0;
  opt_mod->walk([&](mlir::stablehlo::ReduceOp red) {
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(red.getInputs()[0].getType());
    if (in_type.getRank() == 2) {
      ++reduce_2d_count;
    } else if (in_type.getRank() == 3) {
      ++reduce_3d_count;
    }
  });
  EXPECT_EQ(reduce_2d_count, 4);
  EXPECT_EQ(reduce_3d_count, 0);

  auto actual = mlir::stablehlo::evalModule(
      *opt_mod, {dy_2d, x, mean, rstd, gamma}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(actual->size(), expected->size());

  for (size_t idx = 0; idx < expected->size(); ++idx) {
    ASSERT_EQ((*actual)[idx].getNumElements(),
              (*expected)[idx].getNumElements());
    auto exp_it = (*expected)[idx].value_begin<llvm::APFloat>();
    auto act_it = (*actual)[idx].value_begin<llvm::APFloat>();
    for (; exp_it != (*expected)[idx].value_end<llvm::APFloat>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ((*act_it).convertToFloat(), (*exp_it).convertToFloat());
    }
  }
}

TEST(CanonicalizeRmsNormAndLayerNormBackwardTest,
     SqueezesUnitBatchDimIn3DBf16LayerNormBackwardWithDirectBf16Gamma) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %dy_2d: tensor<4x8xbf16>,
      %x: tensor<1x4x8xbf16>,
      %mean: tensor<1x4x1xf32>,
      %rstd: tensor<1x4x1xf32>,
      %gamma: tensor<8xbf16>)
      -> (tensor<1x4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>) {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %inv_n_s = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %inv_n = stablehlo.broadcast_in_dim %inv_n_s, dims = []
        : (tensor<f32>) -> tensor<1x4xf32>
    %dy = stablehlo.reshape %dy_2d
        : (tensor<4x8xbf16>) -> tensor<1x4x8xbf16>
    %x_f32 = stablehlo.convert %x
        : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %dy_f32 = stablehlo.convert %dy
        : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %gamma_bcast_bf16 = stablehlo.broadcast_in_dim %gamma, dims = [2]
        : (tensor<8xbf16>) -> tensor<1x4x8xbf16>
    %gamma_bcast = stablehlo.convert %gamma_bcast_bf16
        : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %mean_bcast = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %rstd_bcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2]
        : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %x_sub = stablehlo.subtract %x_f32, %mean_bcast : tensor<1x4x8xf32>
    %x_hat = stablehlo.multiply %x_sub, %rstd_bcast : tensor<1x4x8xf32>
    %dgamma_full = stablehlo.multiply %dy_f32, %x_hat : tensor<1x4x8xf32>
    %dgamma_f32 = stablehlo.reduce(%dgamma_full init: %zero)
        across dimensions = [0, 1]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %dbeta_f32 = stablehlo.reduce(%dy_f32 init: %zero)
        across dimensions = [0, 1]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %dy_gamma = stablehlo.multiply %dy_f32, %gamma_bcast : tensor<1x4x8xf32>
    %a1 = stablehlo.reduce(%dy_gamma init: %zero)
        across dimensions = [2]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a2_full = stablehlo.multiply %dy_gamma, %x_hat : tensor<1x4x8xf32>
    %a2 = stablehlo.reduce(%a2_full init: %zero)
        across dimensions = [2]
        : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a1_scaled = stablehlo.multiply %a1, %inv_n : tensor<1x4xf32>
    %a2_scaled = stablehlo.multiply %a2, %inv_n : tensor<1x4xf32>
    %a1_bcast = stablehlo.broadcast_in_dim %a1_scaled, dims = [0, 1]
        : (tensor<1x4xf32>) -> tensor<1x4x8xf32>
    %a2_bcast = stablehlo.broadcast_in_dim %a2_scaled, dims = [0, 1]
        : (tensor<1x4xf32>) -> tensor<1x4x8xf32>
    %dx_sub1 = stablehlo.subtract %dy_gamma, %a1_bcast : tensor<1x4x8xf32>
    %x_hat_a2 = stablehlo.multiply %x_hat, %a2_bcast : tensor<1x4x8xf32>
    %dx_sub2 = stablehlo.subtract %dx_sub1, %x_hat_a2 : tensor<1x4x8xf32>
    %dx_f32 = stablehlo.multiply %dx_sub2, %rstd_bcast : tensor<1x4x8xf32>
    %dx = stablehlo.convert %dx_f32 : (tensor<1x4x8xf32>) -> tensor<1x4x8xbf16>
    %dgamma = stablehlo.convert %dgamma_f32 : (tensor<8xf32>) -> tensor<8xbf16>
    %dbeta = stablehlo.convert %dbeta_f32 : (tensor<8xf32>) -> tensor<8xbf16>
    return %dx, %dgamma, %dbeta
        : tensor<1x4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr dy_2d = MakeBF16Tensor(context, {4, 8}, 0.25f);
  mlir::DenseElementsAttr x = MakeBF16Tensor(context, {1, 4, 8}, -0.5f);
  mlir::DenseElementsAttr mean = MakeF32Tensor(context, {1, 4, 1}, 0.1f);
  mlir::DenseElementsAttr rstd = MakeF32Tensor(context, {1, 4, 1}, 0.8f);
  mlir::DenseElementsAttr gamma = MakeBF16Tensor(context, {8}, 0.9f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {dy_2d, x, mean, rstd, gamma}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  int reduce_2d_count = 0;
  int reduce_3d_count = 0;
  opt_mod->walk([&](mlir::stablehlo::ReduceOp red) {
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(red.getInputs()[0].getType());
    if (in_type.getRank() == 2) {
      ++reduce_2d_count;
    } else if (in_type.getRank() == 3) {
      ++reduce_3d_count;
    }
  });
  EXPECT_EQ(reduce_2d_count, 4);
  EXPECT_EQ(reduce_3d_count, 0);

  auto actual = mlir::stablehlo::evalModule(
      *opt_mod, {dy_2d, x, mean, rstd, gamma}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(actual->size(), expected->size());

  for (size_t idx = 0; idx < expected->size(); ++idx) {
    ASSERT_EQ((*actual)[idx].getNumElements(),
              (*expected)[idx].getNumElements());
    auto exp_it = (*expected)[idx].value_begin<llvm::APFloat>();
    auto act_it = (*actual)[idx].value_begin<llvm::APFloat>();
    for (; exp_it != (*expected)[idx].value_end<llvm::APFloat>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ((*act_it).convertToFloat(), (*exp_it).convertToFloat());
    }
  }
}

TEST(CanonicalizeRmsNormAndLayerNormBackwardTest,
     SqueezesAdaLnLayerNormBackwardWithForwardXHatAndPreceedingGrads) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %dy: tensor<1x4x8xbf16>,
      %x: tensor<1x4x8xbf16>,
      %mean_raw: tensor<1x4x1xf32>,
      %rstd_raw: tensor<1x4x1xf32>,
      %scale: tensor<1x1x8xf32>)
      -> (tensor<1x4x8xbf16>, tensor<1x8xf32>, tensor<1x8xf32>) {
    %zero_early = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %x_f32 = stablehlo.convert %x : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %mean_fwd_bcast = stablehlo.broadcast_in_dim %mean_raw, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %rstd_fwd_bcast = stablehlo.broadcast_in_dim %rstd_raw, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %x_sub_fwd = stablehlo.subtract %x_f32, %mean_fwd_bcast : tensor<1x4x8xf32>
    %x_hat_fwd_f32 = stablehlo.multiply %x_sub_fwd, %rstd_fwd_bcast : tensor<1x4x8xf32>
    %x_hat_fwd_bf16 = stablehlo.convert %x_hat_fwd_f32 : (tensor<1x4x8xf32>) -> tensor<1x4x8xbf16>
    %x_hat_fwd = stablehlo.convert %x_hat_fwd_bf16 : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %dy_f32 = stablehlo.convert %dy : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %dshift = stablehlo.reduce(%dy_f32 init: %zero_early) across dimensions = [1] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %dscale_full = stablehlo.multiply %dy_f32, %x_hat_fwd : tensor<1x4x8xf32>
    %dscale = stablehlo.reduce(%dscale_full init: %zero_early) across dimensions = [1] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %scale_bcast = stablehlo.broadcast_in_dim %scale, dims = [0, 1, 2] : (tensor<1x1x8xf32>) -> tensor<1x4x8xf32>
    %dy_scaled_f32 = stablehlo.multiply %dy_f32, %scale_bcast : tensor<1x4x8xf32>
    %dy_scaled_bf16 = stablehlo.convert %dy_scaled_f32 : (tensor<1x4x8xf32>) -> tensor<1x4x8xbf16>
    %dy_scaled = stablehlo.convert %dy_scaled_bf16 : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %inv_n_s = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %inv_n = stablehlo.broadcast_in_dim %inv_n_s, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %mean_bwd_bcast = stablehlo.broadcast_in_dim %mean_raw, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %rstd_bwd_bcast = stablehlo.broadcast_in_dim %rstd_raw, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %x_sub_bwd = stablehlo.subtract %x_f32, %mean_bwd_bcast : tensor<1x4x8xf32>
    %x_hat_bwd = stablehlo.multiply %x_sub_bwd, %rstd_bwd_bcast : tensor<1x4x8xf32>
    %a1 = stablehlo.reduce(%dy_scaled init: %zero) across dimensions = [2] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a2_full = stablehlo.multiply %dy_scaled, %x_hat_bwd : tensor<1x4x8xf32>
    %a2 = stablehlo.reduce(%a2_full init: %zero) across dimensions = [2] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a1_scaled = stablehlo.multiply %a1, %inv_n : tensor<1x4xf32>
    %a2_scaled = stablehlo.multiply %a2, %inv_n : tensor<1x4xf32>
    %a1_bcast = stablehlo.broadcast_in_dim %a1_scaled, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x8xf32>
    %a2_bcast = stablehlo.broadcast_in_dim %a2_scaled, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x8xf32>
    %dx_sub1 = stablehlo.subtract %dy_scaled, %a1_bcast : tensor<1x4x8xf32>
    %x_hat_a2 = stablehlo.multiply %x_hat_bwd, %a2_bcast : tensor<1x4x8xf32>
    %dx_sub2 = stablehlo.subtract %dx_sub1, %x_hat_a2 : tensor<1x4x8xf32>
    %dx_f32 = stablehlo.multiply %dx_sub2, %rstd_bwd_bcast : tensor<1x4x8xf32>
    %dx = stablehlo.convert %dx_f32 : (tensor<1x4x8xf32>) -> tensor<1x4x8xbf16>
    return %dx, %dscale, %dshift : tensor<1x4x8xbf16>, tensor<1x8xf32>, tensor<1x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr dy = MakeBF16Tensor(context, {1, 4, 8}, 0.25f);
  mlir::DenseElementsAttr x = MakeBF16Tensor(context, {1, 4, 8}, -0.5f);
  mlir::DenseElementsAttr mean = MakeF32Tensor(context, {1, 4, 1}, 0.1f);
  mlir::DenseElementsAttr rstd = MakeF32Tensor(context, {1, 4, 1}, 0.8f);
  mlir::DenseElementsAttr scale = MakeF32Tensor(context, {1, 1, 8}, 1.2f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {dy, x, mean, rstd, scale}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  int reduce_2d_count = 0;
  int reduce_3d_count = 0;
  opt_mod->walk([&](mlir::stablehlo::ReduceOp red) {
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(red.getInputs()[0].getType());
    if (in_type.getRank() == 2) {
      ++reduce_2d_count;
    } else if (in_type.getRank() == 3) {
      ++reduce_3d_count;
    }
  });
  EXPECT_EQ(reduce_2d_count, 4);
  EXPECT_EQ(reduce_3d_count, 0);

  auto actual = mlir::stablehlo::evalModule(
      *opt_mod, {dy, x, mean, rstd, scale}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(actual->size(), expected->size());

  for (size_t idx = 0; idx < expected->size(); ++idx) {
    ASSERT_EQ((*actual)[idx].getNumElements(),
              (*expected)[idx].getNumElements());
    auto exp_it = (*expected)[idx].value_begin<llvm::APFloat>();
    auto act_it = (*actual)[idx].value_begin<llvm::APFloat>();
    for (; exp_it != (*expected)[idx].value_end<llvm::APFloat>();
         ++exp_it, ++act_it) {
      EXPECT_NEAR((*act_it).convertToFloat(), (*exp_it).convertToFloat(),
                  1.5e-1f);
    }
  }
}

TEST(CanonicalizeRmsNormAndLayerNormBackwardTest,
     SqueezesTransposedInputFeedingLayerNormBackwardWithoutMutatingTranspose) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %dy: tensor<1x4x8xf32>,
      %conv_out: tensor<1x8x2x2xbf16>,
      %mean: tensor<1x4x1xf32>,
      %rstd: tensor<1x4x1xf32>,
      %gamma: tensor<8xf32>)
      -> tensor<1x4x8xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %inv_n_s = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %inv_n = stablehlo.broadcast_in_dim %inv_n_s, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %reshaped = stablehlo.reshape %conv_out : (tensor<1x8x2x2xbf16>) -> tensor<8x4x1xbf16>
    %x_bf16 = stablehlo.transpose %reshaped, dims = [2, 1, 0] : (tensor<8x4x1xbf16>) -> tensor<1x4x8xbf16>
    %x = stablehlo.convert %x_bf16 : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %mean_bcast = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %rstd_bcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %x_sub = stablehlo.subtract %x, %mean_bcast : tensor<1x4x8xf32>
    %x_hat = stablehlo.multiply %x_sub, %rstd_bcast : tensor<1x4x8xf32>
    %gamma_bcast = stablehlo.broadcast_in_dim %gamma, dims = [2] : (tensor<8xf32>) -> tensor<1x4x8xf32>
    %dy_gamma = stablehlo.multiply %dy, %gamma_bcast : tensor<1x4x8xf32>
    %a1 = stablehlo.reduce(%dy_gamma init: %zero) across dimensions = [2] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a2_full = stablehlo.multiply %dy_gamma, %x_hat : tensor<1x4x8xf32>
    %a2 = stablehlo.reduce(%a2_full init: %zero) across dimensions = [2] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a1_scaled = stablehlo.multiply %a1, %inv_n : tensor<1x4xf32>
    %a2_scaled = stablehlo.multiply %a2, %inv_n : tensor<1x4xf32>
    %a1_bcast = stablehlo.broadcast_in_dim %a1_scaled, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x8xf32>
    %a2_bcast = stablehlo.broadcast_in_dim %a2_scaled, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x8xf32>
    %dx_sub1 = stablehlo.subtract %dy_gamma, %a1_bcast : tensor<1x4x8xf32>
    %x_hat_a2 = stablehlo.multiply %x_hat, %a2_bcast : tensor<1x4x8xf32>
    %dx_sub2 = stablehlo.subtract %dx_sub1, %x_hat_a2 : tensor<1x4x8xf32>
    %dx = stablehlo.multiply %dx_sub2, %rstd_bcast : tensor<1x4x8xf32>
    return %dx : tensor<1x4x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr dy = MakeF32Tensor(context, {1, 4, 8}, 0.2f);
  mlir::DenseElementsAttr conv_out =
      MakeBF16Tensor(context, {1, 8, 2, 2}, -0.3f);
  mlir::DenseElementsAttr mean = MakeF32Tensor(context, {1, 4, 1}, 0.05f);
  mlir::DenseElementsAttr rstd = MakeF32Tensor(context, {1, 4, 1}, 1.1f);
  mlir::DenseElementsAttr gamma = MakeF32Tensor(context, {8}, 0.9f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {dy, conv_out, mean, rstd, gamma}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  int reduce_2d_count = 0;
  int reduce_3d_count = 0;
  opt_mod->walk([&](mlir::Operation* op) {
    if (auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
      auto in_type =
          mlir::cast<mlir::RankedTensorType>(red.getInputs()[0].getType());
      if (in_type.getRank() == 2) {
        ++reduce_2d_count;
      } else if (in_type.getRank() == 3) {
        ++reduce_3d_count;
      }
    }
  });
  EXPECT_EQ(reduce_2d_count, 2);
  EXPECT_EQ(reduce_3d_count, 0);

  auto actual = mlir::stablehlo::evalModule(
      *opt_mod, {dy, conv_out, mean, rstd, gamma}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(actual->size(), expected->size());

  auto exp_it = (*expected)[0].value_begin<llvm::APFloat>();
  auto act_it = (*actual)[0].value_begin<llvm::APFloat>();
  for (; exp_it != (*expected)[0].value_end<llvm::APFloat>();
       ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ((*act_it).convertToFloat(), (*exp_it).convertToFloat());
  }
}

TEST(CanonicalizeRmsNormAndLayerNormBackwardTest,
     SqueezesTwoLeadingUnitBatchDimsIn4DF32WithoutGamma) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %dy: tensor<1x1x4x8xf32>,
      %x: tensor<1x1x4x8xf32>,
      %mean: tensor<1x1x4x1xf32>,
      %rstd: tensor<1x1x4x1xf32>)
      -> (tensor<1x1x4x8xf32>, tensor<8xf32>) {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %inv_n_s = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %inv_n = stablehlo.broadcast_in_dim %inv_n_s, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
    %mean_bcast = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2, 3] : (tensor<1x1x4x1xf32>) -> tensor<1x1x4x8xf32>
    %rstd_bcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2, 3] : (tensor<1x1x4x1xf32>) -> tensor<1x1x4x8xf32>
    %x_sub = stablehlo.subtract %x, %mean_bcast : tensor<1x1x4x8xf32>
    %x_hat = stablehlo.multiply %x_sub, %rstd_bcast : tensor<1x1x4x8xf32>
    %dbeta = stablehlo.reduce(%dy init: %zero) across dimensions = [0, 1, 2] : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a1 = stablehlo.reduce(%dy init: %zero) across dimensions = [3] : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a2_full = stablehlo.multiply %dy, %x_hat : tensor<1x1x4x8xf32>
    %a2 = stablehlo.reduce(%a2_full init: %zero) across dimensions = [3] : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a1_scaled = stablehlo.multiply %a1, %inv_n : tensor<1x1x4xf32>
    %a2_scaled = stablehlo.multiply %a2, %inv_n : tensor<1x1x4xf32>
    %a1_bcast = stablehlo.broadcast_in_dim %a1_scaled, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x1x4x8xf32>
    %a2_bcast = stablehlo.broadcast_in_dim %a2_scaled, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x1x4x8xf32>
    %dx_sub1 = stablehlo.subtract %dy, %a1_bcast : tensor<1x1x4x8xf32>
    %x_hat_a2 = stablehlo.multiply %x_hat, %a2_bcast : tensor<1x1x4x8xf32>
    %dx_sub2 = stablehlo.subtract %dx_sub1, %x_hat_a2 : tensor<1x1x4x8xf32>
    %dx = stablehlo.multiply %dx_sub2, %rstd_bcast : tensor<1x1x4x8xf32>
    return %dx, %dbeta : tensor<1x1x4x8xf32>, tensor<8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr dy = MakeF32Tensor(context, {1, 1, 4, 8}, 0.2f);
  mlir::DenseElementsAttr x = MakeF32Tensor(context, {1, 1, 4, 8}, -0.3f);
  mlir::DenseElementsAttr mean = MakeF32Tensor(context, {1, 1, 4, 1}, 0.05f);
  mlir::DenseElementsAttr rstd = MakeF32Tensor(context, {1, 1, 4, 1}, 1.1f);

  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {dy, x, mean, rstd}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  int reduce_2d_count = 0;
  int reduce_4d_count = 0;
  opt_mod->walk([&](mlir::stablehlo::ReduceOp red) {
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(red.getInputs()[0].getType());
    if (in_type.getRank() == 2) {
      ++reduce_2d_count;
    } else if (in_type.getRank() == 4) {
      ++reduce_4d_count;
    }
  });
  EXPECT_EQ(reduce_2d_count, 3);
  EXPECT_EQ(reduce_4d_count, 0);

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {dy, x, mean, rstd}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(actual->size(), expected->size());

  for (size_t idx = 0; idx < expected->size(); ++idx) {
    ASSERT_EQ((*actual)[idx].getNumElements(),
              (*expected)[idx].getNumElements());
    auto exp_it = (*expected)[idx].value_begin<llvm::APFloat>();
    auto act_it = (*actual)[idx].value_begin<llvm::APFloat>();
    for (; exp_it != (*expected)[idx].value_end<llvm::APFloat>();
         ++exp_it, ++act_it) {
      EXPECT_NEAR((*act_it).convertToFloat(), (*exp_it).convertToFloat(),
                  1e-5f);
    }
  }
}

TEST(CanonicalizeRmsNormAndLayerNormBackwardTest,
     LeavesNonUnitLeadingBatchDimUntouched) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %dy: tensor<2x4x8xf32>,
      %x: tensor<2x4x8xf32>,
      %mean: tensor<2x4x1xf32>,
      %rstd: tensor<2x4x1xf32>)
      -> tensor<2x4x8xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %inv_n_s = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %inv_n = stablehlo.broadcast_in_dim %inv_n_s, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %mean_bcast = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x8xf32>
    %rstd_bcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<2x4x1xf32>) -> tensor<2x4x8xf32>
    %x_sub = stablehlo.subtract %x, %mean_bcast : tensor<2x4x8xf32>
    %x_hat = stablehlo.multiply %x_sub, %rstd_bcast : tensor<2x4x8xf32>
    %a1 = stablehlo.reduce(%dy init: %zero) across dimensions = [2] : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a2_full = stablehlo.multiply %dy, %x_hat : tensor<2x4x8xf32>
    %a2 = stablehlo.reduce(%a2_full init: %zero) across dimensions = [2] : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a1_scaled = stablehlo.multiply %a1, %inv_n : tensor<2x4xf32>
    %a2_scaled = stablehlo.multiply %a2, %inv_n : tensor<2x4xf32>
    %a1_bcast = stablehlo.broadcast_in_dim %a1_scaled, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x8xf32>
    %a2_bcast = stablehlo.broadcast_in_dim %a2_scaled, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x8xf32>
    %dx_sub1 = stablehlo.subtract %dy, %a1_bcast : tensor<2x4x8xf32>
    %x_hat_a2 = stablehlo.multiply %x_hat, %a2_bcast : tensor<2x4x8xf32>
    %dx_sub2 = stablehlo.subtract %dx_sub1, %x_hat_a2 : tensor<2x4x8xf32>
    %dx = stablehlo.multiply %dx_sub2, %rstd_bcast : tensor<2x4x8xf32>
    return %dx : tensor<2x4x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);
  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  int reduce_3d_count = 0;
  opt_mod->walk([&](mlir::stablehlo::ReduceOp red) {
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(red.getInputs()[0].getType());
    if (in_type.getRank() == 3) {
      ++reduce_3d_count;
    }
  });
  EXPECT_EQ(reduce_3d_count, 2);
}

TEST(CanonicalizeRmsNormAndLayerNormBackwardTest, LeavesRank2Untouched) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %dy: tensor<1x8xf32>,
      %x: tensor<1x8xf32>,
      %mean: tensor<1x1xf32>,
      %rstd: tensor<1x1xf32>)
      -> tensor<1x8xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %inv_n_s = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %inv_n = stablehlo.broadcast_in_dim %inv_n_s, dims = []
        : (tensor<f32>) -> tensor<1xf32>
    %mean_bcast = stablehlo.broadcast_in_dim %mean, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x8xf32>
    %rstd_bcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1]
        : (tensor<1x1xf32>) -> tensor<1x8xf32>
    %x_sub = stablehlo.subtract %x, %mean_bcast : tensor<1x8xf32>
    %x_hat = stablehlo.multiply %x_sub, %rstd_bcast : tensor<1x8xf32>
    %a1 = stablehlo.reduce(%dy init: %zero) across dimensions = [1]
        : (tensor<1x8xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a2_full = stablehlo.multiply %dy, %x_hat : tensor<1x8xf32>
    %a2 = stablehlo.reduce(%a2_full init: %zero) across dimensions = [1]
        : (tensor<1x8xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }
    %a1_scaled = stablehlo.multiply %a1, %inv_n : tensor<1xf32>
    %a2_scaled = stablehlo.multiply %a2, %inv_n : tensor<1xf32>
    %a1_bcast = stablehlo.broadcast_in_dim %a1_scaled, dims = [0]
        : (tensor<1xf32>) -> tensor<1x8xf32>
    %a2_bcast = stablehlo.broadcast_in_dim %a2_scaled, dims = [0]
        : (tensor<1xf32>) -> tensor<1x8xf32>
    %dx_sub1 = stablehlo.subtract %dy, %a1_bcast : tensor<1x8xf32>
    %x_hat_a2 = stablehlo.multiply %x_hat, %a2_bcast : tensor<1x8xf32>
    %dx_sub2 = stablehlo.subtract %dx_sub1, %x_hat_a2 : tensor<1x8xf32>
    %dx = stablehlo.multiply %dx_sub2, %rstd_bcast : tensor<1x8xf32>
    return %dx : tensor<1x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);
  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  int reduce_2d_count = 0;
  opt_mod->walk([&](mlir::stablehlo::ReduceOp red) {
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(red.getInputs()[0].getType());
    if (in_type.getRank() == 2) {
      ++reduce_2d_count;
    }
  });
  EXPECT_EQ(reduce_2d_count, 2);
}

}  // namespace
}  // namespace torch_tpu
