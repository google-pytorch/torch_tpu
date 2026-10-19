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

#include "csrc/mlir/passes/sink_3d_head_split_and_flatten_layernorm.h"

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
                                      float scale) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = static_cast<float>((i * 11 + 5) % 23 - 11) * scale;
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

TEST(Sink3dHeadSplitAndFlattenLayerNormTest,
     Sinks3dHeadSplitQkvAndFlattensMiddleUnitAxisLayerNorm) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %x: tensor<6x1x8xf32>,
    %wq: tensor<8x8xf32>,
    %bq: tensor<8xf32>,
    %wk: tensor<8x8xf32>,
    %bk: tensor<8xf32>,
    %gamma: tensor<8xf32>,
    %beta: tensor<8xf32>) -> (tensor<2x6x6xf32>, tensor<6x1x8xf32>) {
  %x_sq = stablehlo.multiply %x, %x : tensor<6x1x8xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %red:2 = stablehlo.reduce(%x init: %zero), (%x_sq init: %zero)
      across dimensions = [2] : (tensor<6x1x8xf32>, tensor<6x1x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<6x1xf32>, tensor<6x1xf32>)
    reducer(%a0: tensor<f32>, %b0: tensor<f32>) (%a1: tensor<f32>, %b1: tensor<f32>) {
      %s0 = stablehlo.add %a0, %b0 : tensor<f32>
      %s1 = stablehlo.add %a1, %b1 : tensor<f32>
      stablehlo.return %s0, %s1 : tensor<f32>, tensor<f32>
    }
  %inv_n = stablehlo.constant dense<0.125> : tensor<6x1xf32>
  %mean = stablehlo.multiply %red#0, %inv_n : tensor<6x1xf32>
  %mean_sq = stablehlo.multiply %red#1, %inv_n : tensor<6x1xf32>
  %mu_sq = stablehlo.multiply %mean, %mean : tensor<6x1xf32>
  %var_raw = stablehlo.subtract %mean_sq, %mu_sq : tensor<6x1xf32>
  %z2 = stablehlo.constant dense<0.0> : tensor<6x1xf32>
  %var = stablehlo.maximum %var_raw, %z2 : tensor<6x1xf32>
  %eps = stablehlo.constant dense<1.0e-05> : tensor<6x1xf32>
  %var_eps = stablehlo.add %var, %eps : tensor<6x1xf32>
  %rstd = stablehlo.rsqrt %var_eps : tensor<6x1xf32>
  %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1] : (tensor<6x1xf32>) -> tensor<6x1x8xf32>
  %rstd_b = stablehlo.broadcast_in_dim %rstd, dims = [0, 1] : (tensor<6x1xf32>) -> tensor<6x1x8xf32>
  %sub = stablehlo.subtract %x, %mean_b : tensor<6x1x8xf32>
  %norm = stablehlo.multiply %sub, %rstd_b : tensor<6x1x8xf32>
  %g_b = stablehlo.broadcast_in_dim %gamma, dims = [2] : (tensor<8xf32>) -> tensor<6x1x8xf32>
  %b_b = stablehlo.broadcast_in_dim %beta, dims = [2] : (tensor<8xf32>) -> tensor<6x1x8xf32>
  %ln_w = stablehlo.multiply %norm, %g_b : tensor<6x1x8xf32>
  %ln_out = stablehlo.add %ln_w, %b_b : tensor<6x1x8xf32>
  %ln_2d = stablehlo.reshape %ln_out : (tensor<6x1x8xf32>) -> tensor<6x8xf32>
  %dq = stablehlo.dot_general %ln_2d, %wq, contracting_dims = [1] x [0] : (tensor<6x8xf32>, tensor<8x8xf32>) -> tensor<6x8xf32>
  %bq_b = stablehlo.broadcast_in_dim %bq, dims = [1] : (tensor<8xf32>) -> tensor<6x8xf32>
  %q_add = stablehlo.add %bq_b, %dq : tensor<6x8xf32>
  %q_3d = stablehlo.reshape %q_add : (tensor<6x8xf32>) -> tensor<6x1x8xf32>
  %scale = stablehlo.constant dense<0.5> : tensor<6x1x8xf32>
  %q_sc = stablehlo.multiply %q_3d, %scale : tensor<6x1x8xf32>
  %q_hsd = stablehlo.reshape %q_sc : (tensor<6x1x8xf32>) -> tensor<6x2x4xf32>
  %q_tr = stablehlo.transpose %q_hsd, dims = [1, 0, 2] : (tensor<6x2x4xf32>) -> tensor<2x6x4xf32>
  %dk = stablehlo.dot_general %ln_2d, %wk, contracting_dims = [1] x [0] : (tensor<6x8xf32>, tensor<8x8xf32>) -> tensor<6x8xf32>
  %bk_b = stablehlo.broadcast_in_dim %bk, dims = [1] : (tensor<8xf32>) -> tensor<6x8xf32>
  %k_add = stablehlo.add %bk_b, %dk : tensor<6x8xf32>
  %k_hsd = stablehlo.reshape %k_add : (tensor<6x8xf32>) -> tensor<6x2x4xf32>
  %k_tr = stablehlo.transpose %k_hsd, dims = [1, 2, 0] : (tensor<6x2x4xf32>) -> tensor<2x4x6xf32>
  %scores = stablehlo.dot_general %q_tr, %k_tr, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x6x4xf32>, tensor<2x4x6xf32>) -> tensor<2x6x6xf32>
  return %scores, %ln_out : tensor<2x6x6xf32>, tensor<6x1x8xf32>
}
)mlir";

  auto module_before = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_before);
  auto module_after = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_after);

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_after, &context));

  // Verify LayerNorm ReduceOp now produces 1D tensor<6xf32> moments.
  int red_1d_count = 0;
  module_after->walk([&](mlir::stablehlo::ReduceOp op) {
    auto t = llvm::cast<mlir::RankedTensorType>(op.getResult(0).getType());
    if (t.getRank() == 1 && t.getDimSize(0) == 6) {
      ++red_1d_count;
    }
  });
  EXPECT_EQ(red_1d_count, 1);

  // Verify both Q and K TransposeOps now directly consume
  // ReshapeOp(DotGeneral).
  int sunk_tr_count = 0;
  module_after->walk([&](mlir::stablehlo::TransposeOp op) {
    if (auto r = op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (r.getOperand().getDefiningOp<mlir::stablehlo::DotGeneralOp>()) {
        ++sunk_tr_count;
      }
    }
  });
  EXPECT_EQ(sunk_tr_count, 2);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {6, 1, 8}, 0.1f),
      MakeF32Tensor(context, {8, 8}, 0.05f),
      MakeF32Tensor(context, {8}, 0.02f),
      MakeF32Tensor(context, {8, 8}, 0.04f),
      MakeF32Tensor(context, {8}, 0.03f),
      MakeF32Tensor(context, {8}, 0.1f),
      MakeF32Tensor(context, {8}, 0.01f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_before, inputs, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  auto opt_out = mlir::stablehlo::evalModule(*module_after, inputs, config);
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ExpectNumericallyClose((*opt_out)[0], (*ref_out)[0], 1e-4f);
  ExpectNumericallyClose((*opt_out)[1], (*ref_out)[1], 1e-4f);
}

TEST(Sink3dHeadSplitAndFlattenLayerNormTest,
     SinksUnitBatch4DHeadSplitTranspose) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %x: tensor<4x8xf32>,
    %w: tensor<8x8xf32>,
    %b: tensor<8xf32>) -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
  %d = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>
  %b_b = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<8xf32>) -> tensor<4x8xf32>
  %add = stablehlo.add %b_b, %d : tensor<4x8xf32>
  %scale = stablehlo.constant dense<0.5> : tensor<4x8xf32>
  %mul = stablehlo.multiply %add, %scale : tensor<4x8xf32>
  %r = stablehlo.reshape %mul : (tensor<4x8xf32>) -> tensor<4x2x4x1xf32>
  %tr = stablehlo.transpose %r, dims = [3, 1, 0, 2] : (tensor<4x2x4x1xf32>) -> tensor<1x2x4x4xf32>
  %out1 = stablehlo.add %tr, %tr : tensor<1x2x4x4xf32>
  %out2 = stablehlo.multiply %tr, %tr : tensor<1x2x4x4xf32>
  return %out1, %out2 : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
}
)mlir";

  auto module_before = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_before);
  auto module_after = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_after);

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_after, &context));

  int sunk_tr_count = 0;
  module_after->walk([&](mlir::stablehlo::TransposeOp op) {
    if (auto r = op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (r.getOperand().getDefiningOp<mlir::stablehlo::DotGeneralOp>()) {
        ++sunk_tr_count;
      }
    }
  });
  EXPECT_EQ(sunk_tr_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}, 0.1f),
      MakeF32Tensor(context, {8, 8}, 0.05f),
      MakeF32Tensor(context, {8}, 0.02f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_before, inputs, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  auto opt_out = mlir::stablehlo::evalModule(*module_after, inputs, config);
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ExpectNumericallyClose((*opt_out)[0], (*ref_out)[0], 1e-4f);
  ExpectNumericallyClose((*opt_out)[1], (*ref_out)[1], 1e-4f);
}

TEST(Sink3dHeadSplitAndFlattenLayerNormTest,
     CancelsInverseViewTransposesAcrossUnitDimReshapes) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<4x2x8xf32>, %y: tensor<8xf32>) -> (tensor<4x2x1x8xf32>, tensor<8x2xf32>) {
  %tr1 = stablehlo.transpose %x, dims = [1, 0, 2] : (tensor<4x2x8xf32>) -> tensor<2x4x8xf32>
  %r1 = stablehlo.reshape %tr1 : (tensor<2x4x8xf32>) -> tensor<2x1x4x8xf32>
  %tr2 = stablehlo.transpose %r1, dims = [2, 0, 1, 3] : (tensor<2x1x4x8xf32>) -> tensor<4x2x1x8xf32>
  %bcast = stablehlo.broadcast_in_dim %y, dims = [1] : (tensor<8xf32>) -> tensor<2x8xf32>
  %tr3 = stablehlo.transpose %bcast, dims = [1, 0] : (tensor<2x8xf32>) -> tensor<8x2xf32>
  return %tr2, %tr3 : tensor<4x2x1x8xf32>, tensor<8x2xf32>
}
)mlir";

  auto module_before = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_before);
  auto module_after = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_after);

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_after, &context));

  int tr_count = 0;
  module_after->walk([&](mlir::stablehlo::TransposeOp) { ++tr_count; });
  EXPECT_EQ(tr_count, 0);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 2, 8}, 0.1f),
      MakeF32Tensor(context, {8}, 0.05f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_before, inputs, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  auto opt_out = mlir::stablehlo::evalModule(*module_after, inputs, config);
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ExpectNumericallyClose((*opt_out)[0], (*ref_out)[0], 1e-4f);
  ExpectNumericallyClose((*opt_out)[1], (*ref_out)[1], 1e-4f);
}

TEST(Sink3dHeadSplitAndFlattenLayerNormTest,
     Pushes2DToNDReshapeThroughElementwiseActivation) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%x: tensor<6x8xf32>, %bias: tensor<6x1x8xf32>) -> tensor<6x1x8xf32> {
  %r = stablehlo.reshape %x : (tensor<6x8xf32>) -> tensor<6x1x8xf32>
  %tanh = stablehlo.tanh %r : tensor<6x1x8xf32>
  %scale = stablehlo.constant dense<0.5> : tensor<6x1x8xf32>
  %mul = stablehlo.multiply %tanh, %scale : tensor<6x1x8xf32>
  %add = stablehlo.add %mul, %bias : tensor<6x1x8xf32>
  return %add : tensor<6x1x8xf32>
}
)mlir";

  auto module_before = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_before);
  auto module_after = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_after);

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_after, &context));

  int elem_2d_count = 0;
  module_after->walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::stablehlo::TanhOp, mlir::stablehlo::MulOp>(op)) {
      auto t = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
      if (t.getRank() == 2 && t.getDimSize(0) == 6 && t.getDimSize(1) == 8) {
        ++elem_2d_count;
      }
    }
  });
  EXPECT_EQ(elem_2d_count, 2);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {6, 8}, 0.1f),
      MakeF32Tensor(context, {6, 1, 8}, 0.05f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_before, inputs, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  auto opt_out = mlir::stablehlo::evalModule(*module_after, inputs, config);
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ExpectNumericallyClose((*opt_out)[0], (*ref_out)[0], 1e-4f);
}

TEST(Sink3dHeadSplitAndFlattenLayerNormTest,
     FlattensLeadingUnitAxisLayerNormWithResidual) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %x: tensor<1x6x8xf32>,
    %gamma: tensor<8xf32>,
    %beta: tensor<8xf32>) -> (tensor<1x6x8xf32>, tensor<1x6x8xf32>) {
  %x_sq = stablehlo.multiply %x, %x : tensor<1x6x8xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %red:2 = stablehlo.reduce(%x init: %zero), (%x_sq init: %zero)
      across dimensions = [2] : (tensor<1x6x8xf32>, tensor<1x6x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<1x6xf32>, tensor<1x6xf32>)
    reducer(%a0: tensor<f32>, %b0: tensor<f32>) (%a1: tensor<f32>, %b1: tensor<f32>) {
      %s0 = stablehlo.add %a0, %b0 : tensor<f32>
      %s1 = stablehlo.add %a1, %b1 : tensor<f32>
      stablehlo.return %s0, %s1 : tensor<f32>, tensor<f32>
    }
  %inv_n = stablehlo.constant dense<0.125> : tensor<1x6xf32>
  %mean = stablehlo.multiply %red#0, %inv_n : tensor<1x6xf32>
  %mean_sq = stablehlo.multiply %red#1, %inv_n : tensor<1x6xf32>
  %mu_sq = stablehlo.multiply %mean, %mean : tensor<1x6xf32>
  %var_raw = stablehlo.subtract %mean_sq, %mu_sq : tensor<1x6xf32>
  %z2 = stablehlo.constant dense<0.0> : tensor<1x6xf32>
  %var = stablehlo.maximum %var_raw, %z2 : tensor<1x6xf32>
  %eps = stablehlo.constant dense<1.0e-05> : tensor<1x6xf32>
  %var_eps = stablehlo.add %var, %eps : tensor<1x6xf32>
  %rstd = stablehlo.rsqrt %var_eps : tensor<1x6xf32>
  %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1] : (tensor<1x6xf32>) -> tensor<1x6x8xf32>
  %rstd_b = stablehlo.broadcast_in_dim %rstd, dims = [0, 1] : (tensor<1x6xf32>) -> tensor<1x6x8xf32>
  %sub = stablehlo.subtract %x, %mean_b : tensor<1x6x8xf32>
  %norm = stablehlo.multiply %sub, %rstd_b : tensor<1x6x8xf32>
  %g_b = stablehlo.broadcast_in_dim %gamma, dims = [2] : (tensor<8xf32>) -> tensor<1x6x8xf32>
  %b_b = stablehlo.broadcast_in_dim %beta, dims = [2] : (tensor<8xf32>) -> tensor<1x6x8xf32>
  %ln_w = stablehlo.multiply %norm, %g_b : tensor<1x6x8xf32>
  %ln_out = stablehlo.add %ln_w, %b_b : tensor<1x6x8xf32>
  %residual_out = stablehlo.add %ln_out, %x : tensor<1x6x8xf32>
  return %ln_out, %residual_out : tensor<1x6x8xf32>, tensor<1x6x8xf32>
}
)mlir";

  auto module_before = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_before);
  auto module_after = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_after);

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_after, &context));

  int red_1d_count = 0;
  module_after->walk([&](mlir::stablehlo::ReduceOp op) {
    auto t = llvm::cast<mlir::RankedTensorType>(op.getResult(0).getType());
    if (t.getRank() == 1 && t.getDimSize(0) == 6) {
      ++red_1d_count;
    }
  });
  EXPECT_EQ(red_1d_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {1, 6, 8}, 0.1f),
      MakeF32Tensor(context, {8}, 0.1f),
      MakeF32Tensor(context, {8}, 0.01f),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_before, inputs, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  auto opt_out = mlir::stablehlo::evalModule(*module_after, inputs, config);
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ExpectNumericallyClose((*opt_out)[0], (*ref_out)[0], 1e-4f);
  ExpectNumericallyClose((*opt_out)[1], (*ref_out)[1], 1e-4f);
}

}  // namespace
}  // namespace torch_tpu
