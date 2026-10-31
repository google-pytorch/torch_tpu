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

#include "csrc/ops/linalg/cross/linalg_cross.h"

#include <cstdint>

// Disable clang-format to include StableHLO Reference headers before
// StablehloBuilder headers (see scan_builder_test.cc).
// clang-format off
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"
// clang-format on

#include "ATen/ops/linalg_cross.h"
#include "ATen/ops/tensor.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::Pointwise;

class LinalgCrossTest : public ::testing::Test {
 protected:
  LinalgCrossTest()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::MlirOp MakeInt32Constant(mlir::MlirBuilder& builder,
                                 llvm::ArrayRef<int64_t> shape,
                                 llvm::ArrayRef<int32_t> values) {
    const mlir::RankedTensorType type =
        mlir::RankedTensorType::get(shape, builder.getOpBuilder().getI32Type());
    const mlir::DenseElementsAttr attr =
        mlir::DenseElementsAttr::get(type, values);
    return mlir::stablehlo::Constant(builder, attr);
  }

  mlir::MlirOp MakeFloat32Constant(mlir::MlirBuilder& builder,
                                   llvm::ArrayRef<int64_t> shape,
                                   llvm::ArrayRef<float> values) {
    const mlir::RankedTensorType type =
        mlir::RankedTensorType::get(shape, builder.getOpBuilder().getF32Type());
    const mlir::DenseElementsAttr attr =
        mlir::DenseElementsAttr::get(type, values);
    return mlir::stablehlo::Constant(builder, attr);
  }

  template <typename T>
  llvm::SmallVector<T> Eval(mlir::OwningOpRef<mlir::ModuleOp> mod) {
    const mlir::stablehlo::InterpreterConfiguration config;
    const llvm::SmallVector<mlir::DenseElementsAttr> empty_inputs;
    const mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>>
        eval_result = mlir::stablehlo::evalModule(*mod, empty_inputs, config);
    EXPECT_TRUE(mlir::succeeded(eval_result));
    return llvm::SmallVector<T>((*eval_result)[0].getValues<T>());
  }

  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

TEST_F(LinalgCrossTest, Evaluates1DInt32) {
  {
    mlir::func::FunctionBuilder fb(module_builder_, "main");
    mlir::MlirOp a = MakeInt32Constant(fb, {3}, {1, 2, 3});
    mlir::MlirOp b = MakeInt32Constant(fb, {3}, {4, 5, 6});
    TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp out, BuildLinalgCrossShlo(a, b, 0));
    mlir::func::Return(fb, {out});
  }
  EXPECT_THAT(Eval<int32_t>(module_builder_.build()), ElementsAre(-3, 6, -3));
}

TEST_F(LinalgCrossTest, MatchesAtLinalgCrossWithBroadcastAndMiddleDim) {
  const llvm::SmallVector<float> a_vals = {
      1.5f, -2.0f, 0.5f, 3.25f, 1.0f, -0.75f,
  };
  const llvm::SmallVector<float> b_vals = {
      -0.5f, 4.0f, 2.5f, 1.25f, -3.0f, 0.25f,
  };
  {
    mlir::func::FunctionBuilder fb(module_builder_, "main");
    // a: (1, 3, 2), b: (2, 3, 1), cross along dim=1 -> (2, 3, 2)
    mlir::MlirOp a = MakeFloat32Constant(fb, {1, 3, 2}, a_vals);
    mlir::MlirOp b = MakeFloat32Constant(fb, {2, 3, 1}, b_vals);
    TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp out, BuildLinalgCrossShlo(a, b, 1));
    EXPECT_THAT(GetTensorTypeOrDie(out).getShape(), ElementsAre(2, 3, 2));
    mlir::func::Return(fb, {out});
  }

  at::Tensor a_cpu = at::tensor(a_vals).reshape({1, 3, 2});
  at::Tensor b_cpu = at::tensor(b_vals).reshape({2, 3, 1});
  at::Tensor expected_cpu =
      at::linalg_cross(a_cpu, b_cpu, /*dim=*/1).contiguous();
  llvm::ArrayRef<float> expected(expected_cpu.data_ptr<float>(),
                                 expected_cpu.numel());
  EXPECT_THAT(Eval<float>(module_builder_.build()),
              Pointwise(FloatEq(), expected));
}

TEST_F(LinalgCrossTest, PreservesBoundedDynamicBatchDim) {
  mlir::func::FunctionBuilder fb(module_builder_, "main");
  mlir::OpBuilder& op_builder = fb.getOpBuilder();
  mlir::MlirOp lhs = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({8, 3}, op_builder.getF32Type()));
  mlir::MlirOp rhs = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({8, 3}, op_builder.getF32Type()));
  mlir::MlirOp dyn_size = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({}, op_builder.getI32Type()));
  lhs = mlir::stablehlo::SetDimensionSize(lhs, dyn_size, 0);
  rhs = mlir::stablehlo::SetDimensionSize(rhs, dyn_size, 0);

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp out, BuildLinalgCrossShlo(lhs, rhs, 1));
  const mlir::RankedTensorType out_type = GetTensorTypeOrDie(out);
  EXPECT_TRUE(out_type.isDynamicDim(0));
  EXPECT_FALSE(out_type.isDynamicDim(1));
  EXPECT_THAT(GetStaticShape(out), ElementsAre(8, 3));
}

}  // namespace
}  // namespace torch_tpu
