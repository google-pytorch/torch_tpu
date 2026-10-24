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

#include "csrc/mlir/passes/propagate_single_sink_slices_and_zero_pads.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
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
  registry.insert<mlir::func::FuncDialect, mlir::chlo::ChloDialect>();
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
    vals[i] = base + static_cast<float>((i * 17 + 5) % 37) * 0.0625f -
              static_cast<float>((i * 7) % 11) * 0.125f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

void ExpectDenseAttrsNear(mlir::DenseElementsAttr actual,
                          mlir::DenseElementsAttr expected, float tol = 1e-4f) {
  ASSERT_EQ(actual.getType(), expected.getType());
  auto actual_vals = actual.getValues<float>();
  auto expected_vals = expected.getValues<float>();
  auto it_a = actual_vals.begin();
  auto it_e = expected_vals.begin();
  for (; it_a != actual_vals.end(); ++it_a, ++it_e) {
    EXPECT_NEAR(*it_a, *it_e, tol);
  }
}

TEST(PropagateSingleSinkSlicesAndZeroPadsTest,
     HoistsSliceAcrossLayerNormAndMlpSubgraph) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%x: tensor<1x8x4xf32>, %w1: tensor<4x6xf32>,
                      %w2: tensor<6x4xf32>, %gamma: tensor<4xf32>,
                      %beta: tensor<4xf32>) -> tensor<1x1x4xf32> {
        %r_x = stablehlo.reshape %x : (tensor<1x8x4xf32>) -> tensor<8x4xf32>
        %h1 = stablehlo.dot_general %r_x, %w1,
            contracting_dims = [1] x [0]
            : (tensor<8x4xf32>, tensor<4x6xf32>) -> tensor<8x6xf32>
        %h1_3d = stablehlo.reshape %h1 : (tensor<8x6xf32>) -> tensor<1x8x6xf32>
        %half = stablehlo.constant dense<0.5> : tensor<f32>
        %one = stablehlo.constant dense<1.0> : tensor<f32>
        %b_half = stablehlo.broadcast_in_dim %half, dims = []
            : (tensor<f32>) -> tensor<1x8x6xf32>
        %b_one = stablehlo.broadcast_in_dim %one, dims = []
            : (tensor<f32>) -> tensor<1x8x6xf32>
        %th = stablehlo.tanh %h1_3d : tensor<1x8x6xf32>
        %th1 = stablehlo.add %th, %b_one : tensor<1x8x6xf32>
        %h_half = stablehlo.multiply %h1_3d, %b_half : tensor<1x8x6xf32>
        %act = stablehlo.multiply %h_half, %th1 : tensor<1x8x6xf32>
        %act_2d = stablehlo.reshape %act
            : (tensor<1x8x6xf32>) -> tensor<8x6xf32>
        %h2 = stablehlo.dot_general %act_2d, %w2,
            contracting_dims = [1] x [0]
            : (tensor<8x6xf32>, tensor<6x4xf32>) -> tensor<8x4xf32>
        %h2_3d = stablehlo.reshape %h2 : (tensor<8x4xf32>) -> tensor<1x8x4xf32>
        %res = stablehlo.add %x, %h2_3d : tensor<1x8x4xf32>
        %res_sq = stablehlo.multiply %res, %res : tensor<1x8x4xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %sums:2 = stablehlo.reduce(%res init: %zero), (%res_sq init: %zero)
            across dimensions = [2]
            : (tensor<1x8x4xf32>, tensor<1x8x4xf32>, tensor<f32>, tensor<f32>)
              -> (tensor<1x8xf32>, tensor<1x8xf32>)
          reducer(%a0: tensor<f32>, %b0: tensor<f32>)
                 (%a1: tensor<f32>, %b1: tensor<f32>) {
            %s0 = stablehlo.add %a0, %b0 : tensor<f32>
            %s1 = stablehlo.add %a1, %b1 : tensor<f32>
            stablehlo.return %s0, %s1 : tensor<f32>, tensor<f32>
          }
        %inv_n = stablehlo.constant dense<0.25> : tensor<f32>
        %eps = stablehlo.constant dense<1.0e-5> : tensor<f32>
        %b_inv = stablehlo.broadcast_in_dim %inv_n, dims = []
            : (tensor<f32>) -> tensor<1x8xf32>
        %b_eps = stablehlo.broadcast_in_dim %eps, dims = []
            : (tensor<f32>) -> tensor<1x8xf32>
        %mean = stablehlo.multiply %sums#0, %b_inv : tensor<1x8xf32>
        %b_mean = stablehlo.broadcast_in_dim %mean, dims = [0, 1]
            : (tensor<1x8xf32>) -> tensor<1x8x4xf32>
        %cent = stablehlo.subtract %res, %b_mean : tensor<1x8x4xf32>
        %ex2 = stablehlo.multiply %sums#1, %b_inv : tensor<1x8xf32>
        %m2 = stablehlo.multiply %mean, %mean : tensor<1x8xf32>
        %var = stablehlo.subtract %ex2, %m2 : tensor<1x8xf32>
        %var_eps = stablehlo.add %var, %b_eps : tensor<1x8xf32>
        %rstd = stablehlo.rsqrt %var_eps : tensor<1x8xf32>
        %b_rstd = stablehlo.broadcast_in_dim %rstd, dims = [0, 1]
            : (tensor<1x8xf32>) -> tensor<1x8x4xf32>
        %norm = stablehlo.multiply %cent, %b_rstd : tensor<1x8x4xf32>
        %b_gamma = stablehlo.broadcast_in_dim %gamma, dims = [2]
            : (tensor<4xf32>) -> tensor<1x8x4xf32>
        %b_beta = stablehlo.broadcast_in_dim %beta, dims = [2]
            : (tensor<4xf32>) -> tensor<1x8x4xf32>
        %sc = stablehlo.multiply %norm, %b_gamma : tensor<1x8x4xf32>
        %ln_out = stablehlo.add %sc, %b_beta : tensor<1x8x4xf32>
        %cls = stablehlo.slice %ln_out [0:1, 0:1, 0:4]
            : (tensor<1x8x4xf32>) -> tensor<1x1x4xf32>
        return %cls : tensor<1x1x4xf32>
      }
    }
  )mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto before_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  auto after_mod = mlir::OwningOpRef<mlir::ModuleOp>(before_mod->clone());

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createPropagateSingleSinkSlicesAndZeroPadsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*after_mod)));

  // Both DotGeneralOps and the ReduceOp should now operate on 1 token instead
  // of 8 tokens.
  after_mod->walk([&](mlir::stablehlo::DotGeneralOp dot) {
    auto out_ty = llvm::cast<mlir::RankedTensorType>(dot.getType());
    EXPECT_EQ(out_ty.getDimSize(0), 1);
  });
  after_mod->walk([&](mlir::stablehlo::ReduceOp red) {
    auto out_ty =
        llvm::cast<mlir::RankedTensorType>(red.getResult(0).getType());
    EXPECT_EQ(out_ty.getDimSize(1), 1);
  });

  llvm::SmallVector<mlir::DenseElementsAttr> args = {
      MakeF32Tensor(context, {1, 8, 4}, 0.1f),
      MakeF32Tensor(context, {4, 6}, -0.2f),
      MakeF32Tensor(context, {6, 4}, 0.05f),
      MakeF32Tensor(context, {4}, 0.9f),
      MakeF32Tensor(context, {4}, -0.1f),
  };
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*before_mod, args, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));
  auto actual = mlir::stablehlo::evalModule(*after_mod, args, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectDenseAttrsNear((*actual)[0], (*expected)[0]);
}

TEST(PropagateSingleSinkSlicesAndZeroPadsTest,
     PropagatesZeroPaddedSliceAcrossLayerNormAndLinearBackward) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%grad_cls: tensor<1x1x4xf32>, %x_hat: tensor<1x8x4xf32>,
                      %rstd: tensor<1x8xf32>, %gamma: tensor<4xf32>,
                      %h: tensor<8x6xf32>, %w: tensor<4x6xf32>)
          -> (tensor<4xf32>, tensor<4xf32>, tensor<4x6xf32>, tensor<1x8x6xf32>) {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %dy = stablehlo.pad %grad_cls, %zero,
            low = [0, 0, 0], high = [0, 7, 0], interior = [0, 0, 0]
            : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1x8x4xf32>
        %b_gamma = stablehlo.broadcast_in_dim %gamma, dims = [2]
            : (tensor<4xf32>) -> tensor<1x8x4xf32>
        %dy_g = stablehlo.multiply %dy, %b_gamma : tensor<1x8x4xf32>
        %s_dy_g = stablehlo.reduce(%dy_g init: %zero) across dimensions = [2]
            : (tensor<1x8x4xf32>, tensor<f32>) -> tensor<1x8xf32>
          reducer(%a: tensor<f32>, %b: tensor<f32>) {
            %s = stablehlo.add %a, %b : tensor<f32>
            stablehlo.return %s : tensor<f32>
          }
        %inv_n = stablehlo.constant dense<0.25> : tensor<f32>
        %b_inv = stablehlo.broadcast_in_dim %inv_n, dims = []
            : (tensor<f32>) -> tensor<1x8xf32>
        %m_dy_g = stablehlo.multiply %s_dy_g, %b_inv : tensor<1x8xf32>
        %bm_dy_g = stablehlo.broadcast_in_dim %m_dy_g, dims = [0, 1]
            : (tensor<1x8xf32>) -> tensor<1x8x4xf32>
        %diff1 = stablehlo.subtract %dy_g, %bm_dy_g : tensor<1x8x4xf32>
        %dy_g_xh = stablehlo.multiply %dy_g, %x_hat : tensor<1x8x4xf32>
        %s_xh = stablehlo.reduce(%dy_g_xh init: %zero) across dimensions = [2]
            : (tensor<1x8x4xf32>, tensor<f32>) -> tensor<1x8xf32>
          reducer(%a: tensor<f32>, %b: tensor<f32>) {
            %s = stablehlo.add %a, %b : tensor<f32>
            stablehlo.return %s : tensor<f32>
          }
        %m_xh = stablehlo.multiply %s_xh, %b_inv : tensor<1x8xf32>
        %bm_xh = stablehlo.broadcast_in_dim %m_xh, dims = [0, 1]
            : (tensor<1x8xf32>) -> tensor<1x8x4xf32>
        %term2 = stablehlo.multiply %x_hat, %bm_xh : tensor<1x8x4xf32>
        %diff2 = stablehlo.subtract %diff1, %term2 : tensor<1x8x4xf32>
        %b_rstd = stablehlo.broadcast_in_dim %rstd, dims = [0, 1]
            : (tensor<1x8xf32>) -> tensor<1x8x4xf32>
        %dx = stablehlo.multiply %diff2, %b_rstd : tensor<1x8x4xf32>
        %dy_xh = stablehlo.multiply %dy, %x_hat : tensor<1x8x4xf32>
        %dgamma = stablehlo.reduce(%dy_xh init: %zero)
            across dimensions = [0, 1]
            : (tensor<1x8x4xf32>, tensor<f32>) -> tensor<4xf32>
          reducer(%a: tensor<f32>, %b: tensor<f32>) {
            %s = stablehlo.add %a, %b : tensor<f32>
            stablehlo.return %s : tensor<f32>
          }
        %dbeta = stablehlo.reduce(%dy init: %zero)
            across dimensions = [0, 1]
            : (tensor<1x8x4xf32>, tensor<f32>) -> tensor<4xf32>
          reducer(%a: tensor<f32>, %b: tensor<f32>) {
            %s = stablehlo.add %a, %b : tensor<f32>
            stablehlo.return %s : tensor<f32>
          }
        %dx_2d = stablehlo.reshape %dx : (tensor<1x8x4xf32>) -> tensor<8x4xf32>
        %dx_t = stablehlo.transpose %dx_2d, dims = [1, 0]
            : (tensor<8x4xf32>) -> tensor<4x8xf32>
        %dw = stablehlo.dot_general %dx_t, %h,
            contracting_dims = [1] x [0]
            : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
        %dh_2d = stablehlo.dot_general %dx_2d, %w,
            contracting_dims = [1] x [0]
            : (tensor<8x4xf32>, tensor<4x6xf32>) -> tensor<8x6xf32>
        %dh = stablehlo.reshape %dh_2d : (tensor<8x6xf32>) -> tensor<1x8x6xf32>
        return %dgamma, %dbeta, %dw, %dh
            : tensor<4xf32>, tensor<4xf32>, tensor<4x6xf32>, tensor<1x8x6xf32>
      }
    }
  )mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto before_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  auto after_mod = mlir::OwningOpRef<mlir::ModuleOp>(before_mod->clone());

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createPropagateSingleSinkSlicesAndZeroPadsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*after_mod)));

  // All ReduceOps and DotGeneralOps should now run on 1 token instead of 8.
  after_mod->walk([&](mlir::stablehlo::DotGeneralOp dot) {
    auto lhs_ty = llvm::cast<mlir::RankedTensorType>(dot.getLhs().getType());
    EXPECT_TRUE(lhs_ty.getDimSize(0) == 1 || lhs_ty.getDimSize(1) == 1);
  });

  llvm::SmallVector<mlir::DenseElementsAttr> args = {
      MakeF32Tensor(context, {1, 1, 4}, 0.3f),
      MakeF32Tensor(context, {1, 8, 4}, -0.1f),
      MakeF32Tensor(context, {1, 8}, 0.8f),
      MakeF32Tensor(context, {4}, 1.1f),
      MakeF32Tensor(context, {8, 6}, 0.2f),
      MakeF32Tensor(context, {4, 6}, -0.15f),
  };
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*before_mod, args, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));
  auto actual = mlir::stablehlo::evalModule(*after_mod, args, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (std::size_t i = 0; i < 4; ++i) {
    ExpectDenseAttrsNear((*actual)[i], (*expected)[i]);
  }
}

TEST(PropagateSingleSinkSlicesAndZeroPadsTest,
     FoldsCausalCumSumFinalSliceToReduce) {
  constexpr const char* kIr = R"mlir(
    module {
      func.func @main(%x: tensor<2x8xf32>) -> tensor<2x1xf32> {
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %cumsum = "stablehlo.reduce_window"(%x, %zero) ({
        ^bb0(%a: tensor<f32>, %b: tensor<f32>):
          %s = stablehlo.add %a, %b : tensor<f32>
          stablehlo.return %s : tensor<f32>
        }) {
          window_dimensions = array<i64: 1, 8>,
          window_strides = array<i64: 1, 1>,
          base_dilations = array<i64: 1, 1>,
          window_dilations = array<i64: 1, 1>,
          padding = dense<[[0, 0], [7, 0]]> : tensor<2x2xi64>
        } : (tensor<2x8xf32>, tensor<f32>) -> tensor<2x8xf32>
        %last = stablehlo.slice %cumsum [0:2, 7:8]
            : (tensor<2x8xf32>) -> tensor<2x1xf32>
        return %last : tensor<2x1xf32>
      }
    }
  )mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto before_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  auto after_mod = mlir::OwningOpRef<mlir::ModuleOp>(before_mod->clone());

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createPropagateSingleSinkSlicesAndZeroPadsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*after_mod)));

  int rw_count = 0;
  int red_count = 0;
  after_mod->walk([&](mlir::stablehlo::ReduceWindowOp) { ++rw_count; });
  after_mod->walk([&](mlir::stablehlo::ReduceOp) { ++red_count; });
  EXPECT_EQ(rw_count, 0);
  EXPECT_EQ(red_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> args = {
      MakeF32Tensor(context, {2, 8}, 0.25f),
  };
  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*before_mod, args, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));
  auto actual = mlir::stablehlo::evalModule(*after_mod, args, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectDenseAttrsNear((*actual)[0], (*expected)[0]);
}

}  // namespace
}  // namespace torch_tpu
