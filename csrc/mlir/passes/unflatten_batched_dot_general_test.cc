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

#include <cstdint>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
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
    vals[i] = base + static_cast<float>((i * 17 + 5) % 251) * 0.03125f -
              static_cast<float>((i * 7) % 13) * 0.125f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

void VerifyNumerics(mlir::ModuleOp baseline, mlir::ModuleOp actual_module,
                    llvm::ArrayRef<mlir::DenseElementsAttr> inputs) {
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(baseline, inputs, config);
  auto actual = mlir::stablehlo::evalModule(actual_module, inputs, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), 1u);
  ASSERT_EQ(actual->size(), 1u);

  auto exp_vals = (*expected)[0].getValues<float>();
  auto act_vals = (*actual)[0].getValues<float>();
  ASSERT_EQ(exp_vals.size(), act_vals.size());
  auto exp_it = exp_vals.begin();
  auto act_it = act_vals.begin();
  for (; exp_it != exp_vals.end() && act_it != act_vals.end();
       ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, 1e-4f);
  }
}

TEST(UnflattenBatchedDotGeneralTest,
     Unflattens3DQkAndPvAndAbsorbs4DHeadSplitTransposes) {
  mlir::MLIRContext context;
  InitContext(context);

  constexpr char kIr[] = R"mlir(
func.func @main(%q_in: tensor<4x8x3x6xf32>,
                %k_in: tensor<4x8x3x6xf32>,
                %v_in: tensor<4x8x3x6xf32>) -> tensor<4x3x8x6xf32> {
  %q_tr = stablehlo.transpose %q_in, dims = [0, 2, 1, 3]
      : (tensor<4x8x3x6xf32>) -> tensor<4x3x8x6xf32>
  %q_3d = stablehlo.reshape %q_tr
      : (tensor<4x3x8x6xf32>) -> tensor<12x8x6xf32>
  %k_tr = stablehlo.transpose %k_in, dims = [0, 2, 3, 1]
      : (tensor<4x8x3x6xf32>) -> tensor<4x3x6x8xf32>
  %k_3d = stablehlo.reshape %k_tr
      : (tensor<4x3x6x8xf32>) -> tensor<12x6x8xf32>
  %qk_3d = stablehlo.dot_general %q_3d, %k_3d,
      batching_dims = [0] x [0],
      contracting_dims = [2] x [1]
      : (tensor<12x8x6xf32>, tensor<12x6x8xf32>) -> tensor<12x8x8xf32>
  %qk_4d = stablehlo.reshape %qk_3d
      : (tensor<12x8x8xf32>) -> tensor<4x3x8x8xf32>
  %scale = stablehlo.constant dense<0.25> : tensor<4x3x8x8xf32>
  %probs_4d = stablehlo.multiply %qk_4d, %scale : tensor<4x3x8x8xf32>
  %probs_3d = stablehlo.reshape %probs_4d
      : (tensor<4x3x8x8xf32>) -> tensor<12x8x8xf32>
  %v_tr = stablehlo.transpose %v_in, dims = [0, 2, 1, 3]
      : (tensor<4x8x3x6xf32>) -> tensor<4x3x8x6xf32>
  %v_3d = stablehlo.reshape %v_tr
      : (tensor<4x3x8x6xf32>) -> tensor<12x8x6xf32>
  %pv_3d = stablehlo.dot_general %probs_3d, %v_3d,
      batching_dims = [0] x [0],
      contracting_dims = [2] x [1]
      : (tensor<12x8x8xf32>, tensor<12x8x6xf32>) -> tensor<12x8x6xf32>
  %pv_4d = stablehlo.reshape %pv_3d
      : (tensor<12x8x6xf32>) -> tensor<4x3x8x6xf32>
  return %pv_4d : tensor<4x3x8x6xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_ref);
  mlir::OwningOpRef<mlir::ModuleOp> baseline_ref = module_ref->clone();

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_ref, &context));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module_ref), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module_ref), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_ref), 2);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8, 3, 6}, 0.1f),
      MakeF32Tensor(context, {4, 8, 3, 6}, -0.2f),
      MakeF32Tensor(context, {4, 8, 3, 6}, 0.3f),
  };
  VerifyNumerics(*baseline_ref, *module_ref, inputs);
}

TEST(UnflattenBatchedDotGeneralTest,
     LeavesDotWithoutAbsorbableTransposeUntouched) {
  mlir::MLIRContext context;
  InitContext(context);

  constexpr char kIr[] = R"mlir(
func.func @main(%a: tensor<4x3x8x6xf32>,
                %b: tensor<4x3x6x8xf32>) -> tensor<12x8x8xf32> {
  %a_3d = stablehlo.reshape %a : (tensor<4x3x8x6xf32>) -> tensor<12x8x6xf32>
  %b_3d = stablehlo.reshape %b : (tensor<4x3x6x8xf32>) -> tensor<12x6x8xf32>
  %dot = stablehlo.dot_general %a_3d, %b_3d,
      batching_dims = [0] x [0],
      contracting_dims = [2] x [1]
      : (tensor<12x8x6xf32>, tensor<12x6x8xf32>) -> tensor<12x8x8xf32>
  return %dot : tensor<12x8x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_ref);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_ref, &context));

  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_ref), 1);
  module_ref->walk([&](mlir::stablehlo::DotGeneralOp dot_op) {
    auto out_ty = mlir::cast<mlir::RankedTensorType>(dot_op.getType());
    EXPECT_EQ(out_ty.getRank(), 3);
  });
}

TEST(UnflattenBatchedDotGeneralTest, UnflattensWithUnitBatchDimension) {
  mlir::MLIRContext context;
  InitContext(context);

  constexpr char kIr[] = R"mlir(
func.func @main(%q_in: tensor<1x8x3x6xf32>,
                %k_in: tensor<1x8x3x6xf32>) -> tensor<1x3x8x8xf32> {
  %q_tr = stablehlo.transpose %q_in, dims = [0, 2, 1, 3]
      : (tensor<1x8x3x6xf32>) -> tensor<1x3x8x6xf32>
  %q_3d = stablehlo.reshape %q_tr
      : (tensor<1x3x8x6xf32>) -> tensor<3x8x6xf32>
  %k_tr = stablehlo.transpose %k_in, dims = [0, 2, 3, 1]
      : (tensor<1x8x3x6xf32>) -> tensor<1x3x6x8xf32>
  %k_3d = stablehlo.reshape %k_tr
      : (tensor<1x3x6x8xf32>) -> tensor<3x6x8xf32>
  %qk_3d = stablehlo.dot_general %q_3d, %k_3d,
      batching_dims = [0] x [0],
      contracting_dims = [2] x [1]
      : (tensor<3x8x6xf32>, tensor<3x6x8xf32>) -> tensor<3x8x8xf32>
  %qk_4d = stablehlo.reshape %qk_3d
      : (tensor<3x8x8xf32>) -> tensor<1x3x8x8xf32>
  return %qk_4d : tensor<1x3x8x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_ref);
  mlir::OwningOpRef<mlir::ModuleOp> baseline_ref = module_ref->clone();

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_ref, &context));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module_ref), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module_ref), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_ref), 1);
  module_ref->walk([&](mlir::stablehlo::DotGeneralOp dot_op) {
    auto out_ty = mlir::cast<mlir::RankedTensorType>(dot_op.getType());
    EXPECT_EQ(out_ty.getRank(), 4);
  });

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {1, 8, 3, 6}, 0.5f),
      MakeF32Tensor(context, {1, 8, 3, 6}, -0.7f),
  };
  VerifyNumerics(*baseline_ref, *module_ref, inputs);
}

TEST(UnflattenBatchedDotGeneralTest, UnflattensWith3DTransposePeel) {
  mlir::MLIRContext context;
  InitContext(context);

  constexpr char kIr[] = R"mlir(
func.func @main(%x_in: tensor<2x4x3x5xf32>,
                %y_in: tensor<2x3x5x7xf32>) -> tensor<2x3x4x7xf32> {
  %x_tr4d = stablehlo.transpose %x_in, dims = [0, 2, 1, 3]
      : (tensor<2x4x3x5xf32>) -> tensor<2x3x4x5xf32>
  %x_resh = stablehlo.reshape %x_tr4d
      : (tensor<2x3x4x5xf32>) -> tensor<6x4x5xf32>
  %x_tr3d = stablehlo.transpose %x_resh, dims = [0, 2, 1]
      : (tensor<6x4x5xf32>) -> tensor<6x5x4xf32>
  %y_3d = stablehlo.reshape %y_in
      : (tensor<2x3x5x7xf32>) -> tensor<6x5x7xf32>
  %dot = stablehlo.dot_general %x_tr3d, %y_3d,
      batching_dims = [0] x [0],
      contracting_dims = [1] x [1]
      : (tensor<6x5x4xf32>, tensor<6x5x7xf32>) -> tensor<6x4x7xf32>
  %dot_4d = stablehlo.reshape %dot
      : (tensor<6x4x7xf32>) -> tensor<2x3x4x7xf32>
  return %dot_4d : tensor<2x3x4x7xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_ref);
  mlir::OwningOpRef<mlir::ModuleOp> baseline_ref = module_ref->clone();

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_ref, &context));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module_ref), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module_ref), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_ref), 1);
  module_ref->walk([&](mlir::stablehlo::DotGeneralOp dot_op) {
    auto out_ty = mlir::cast<mlir::RankedTensorType>(dot_op.getType());
    EXPECT_EQ(out_ty.getRank(), 4);
  });

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {2, 4, 3, 5}, 0.2f),
      MakeF32Tensor(context, {2, 3, 5, 7}, -0.3f),
  };
  VerifyNumerics(*baseline_ref, *module_ref, inputs);
}

TEST(UnflattenBatchedDotGeneralTest,
     ReshapesBackTo3DWhenDotHasNonReshapeConsumers) {
  mlir::MLIRContext context;
  InitContext(context);

  constexpr char kIr[] = R"mlir(
func.func @main(%q_in: tensor<4x8x3x6xf32>,
                %k_in: tensor<4x8x3x6xf32>) -> tensor<12x8x8xf32> {
  %q_tr = stablehlo.transpose %q_in, dims = [0, 2, 1, 3]
      : (tensor<4x8x3x6xf32>) -> tensor<4x3x8x6xf32>
  %q_3d = stablehlo.reshape %q_tr
      : (tensor<4x3x8x6xf32>) -> tensor<12x8x6xf32>
  %k_tr = stablehlo.transpose %k_in, dims = [0, 2, 3, 1]
      : (tensor<4x8x3x6xf32>) -> tensor<4x3x6x8xf32>
  %k_3d = stablehlo.reshape %k_tr
      : (tensor<4x3x6x8xf32>) -> tensor<12x6x8xf32>
  %qk_3d = stablehlo.dot_general %q_3d, %k_3d,
      batching_dims = [0] x [0],
      contracting_dims = [2] x [1]
      : (tensor<12x8x6xf32>, tensor<12x6x8xf32>) -> tensor<12x8x8xf32>
  return %qk_3d : tensor<12x8x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_ref);
  mlir::OwningOpRef<mlir::ModuleOp> baseline_ref = module_ref->clone();

  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_ref, &context));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module_ref), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module_ref), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_ref), 1);
  module_ref->walk([&](mlir::stablehlo::DotGeneralOp dot_op) {
    auto out_ty = mlir::cast<mlir::RankedTensorType>(dot_op.getType());
    EXPECT_EQ(out_ty.getRank(), 4);
  });

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8, 3, 6}, 0.4f),
      MakeF32Tensor(context, {4, 8, 3, 6}, -0.1f),
  };
  VerifyNumerics(*baseline_ref, *module_ref, inputs);
}

TEST(UnflattenBatchedDotGeneralTest,
     LeavesDotUntouchedWhenBatchFactorizationsMismatch) {
  mlir::MLIRContext context;
  InitContext(context);

  constexpr char kIr[] = R"mlir(
func.func @main(%a_in: tensor<2x4x6x8xf32>,
                %b_in: tensor<4x3x8x5xf32>) -> tensor<12x4x5xf32> {
  %a_tr = stablehlo.transpose %a_in, dims = [0, 2, 1, 3]
      : (tensor<2x4x6x8xf32>) -> tensor<2x6x4x8xf32>
  %a_3d = stablehlo.reshape %a_tr
      : (tensor<2x6x4x8xf32>) -> tensor<12x4x8xf32>
  %b_3d = stablehlo.reshape %b_in
      : (tensor<4x3x8x5xf32>) -> tensor<12x8x5xf32>
  %dot = stablehlo.dot_general %a_3d, %b_3d,
      batching_dims = [0] x [0],
      contracting_dims = [2] x [1]
      : (tensor<12x4x8xf32>, tensor<12x8x5xf32>) -> tensor<12x4x5xf32>
  return %dot : tensor<12x4x5xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_ref);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module_ref, &context));

  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*module_ref), 1);
  module_ref->walk([&](mlir::stablehlo::DotGeneralOp dot_op) {
    auto out_ty = mlir::cast<mlir::RankedTensorType>(dot_op.getType());
    EXPECT_EQ(out_ty.getRank(), 3);
  });
}

}  // namespace
}  // namespace torch_tpu
