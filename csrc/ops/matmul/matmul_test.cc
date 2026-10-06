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

#include "csrc/ops/matmul/matmul.h"

#include <cstdint>

// We disable clang-format here to force the StableHLO Reference headers to
// be included before the TorchTPU StablehloBuilder headers. This prevents
// a namespace collision where both packages attempt to define 'Tuple'.
// clang-format off
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"
// clang-format on

#include "csrc/common/status_test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

constexpr mlir::stablehlo::Precision kDefaultPrecision =
    mlir::stablehlo::Precision::DEFAULT;  // EXPLICIT_PRECISION_OK=unit test

mlir::MlirOp MakeDenseI32Constant(mlir::MlirBuilder& builder,
                                  llvm::ArrayRef<int64_t> shape,
                                  llvm::ArrayRef<int32_t> values) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType type =
      mlir::RankedTensorType::get(shape, op_builder.getI32Type());
  const mlir::DenseElementsAttr attr =
      mlir::DenseElementsAttr::get(type, values);
  return mlir::stablehlo::Constant(builder, attr);
}

class MatmulBuilderTest : public testing::Test {
 protected:
  MatmulBuilderTest()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::ModuleBuilder& module_builder() { return module_builder_; }

  void EvaluateAndVerifyOutput(mlir::ModuleOp module,
                               llvm::ArrayRef<int64_t> expected_shape,
                               llvm::ArrayRef<int32_t> expected_values) {
    const mlir::stablehlo::InterpreterConfiguration config;
    const llvm::SmallVector<mlir::DenseElementsAttr> empty_inputs;
    const mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>>
        eval_result = mlir::stablehlo::evalModule(module, empty_inputs, config);
    ASSERT_TRUE(mlir::succeeded(eval_result));
    ASSERT_EQ(eval_result->size(), 1);

    const mlir::DenseElementsAttr res_attr = (*eval_result)[0];
    EXPECT_THAT(res_attr.getType().getShape(),
                testing::ElementsAreArray(expected_shape));
    const llvm::SmallVector<int32_t> res_vals(res_attr.getValues<int32_t>());
    EXPECT_THAT(res_vals, testing::ElementsAreArray(expected_values));
  }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

TEST_F(MatmulBuilderTest, Forward1Dx1D) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
  const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {3}, {4, 5, 6});
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {}, {32});
}

TEST_F(MatmulBuilderTest, Forward2Dx1D) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs = MakeDenseI32Constant(fn, {2, 3}, {1, 2, 3, 4, 5, 6});
  const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2}, {14, 32});
}

TEST_F(MatmulBuilderTest, Forward1Dx2D) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
  const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2}, {22, 28});
}

TEST_F(MatmulBuilderTest, Forward3Dx2D) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs =
      MakeDenseI32Constant(fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
  const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2, 2, 2}, {22, 28, 49, 64, 6, 8, 3, 4});
}

TEST_F(MatmulBuilderTest, Forward3Dx3DBroadcastBatch) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs =
      MakeDenseI32Constant(fn, {1, 2, 3}, {1, 2, 3, 4, 5, 6});
  const mlir::MlirOp rhs =
      MakeDenseI32Constant(fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0});
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2, 2, 2}, {4, 5, 10, 11, 2, 4, 8, 10});
}

TEST_F(MatmulBuilderTest, Backward3Dx2DGradOther) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad =
      MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
  const mlir::MlirOp self =
      MakeDenseI32Constant(fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
  const mlir::MlirOp other =
      MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_other,
      BuildMatmulGradOtherShlo(grad, self, other, kDefaultPrecision));
  mlir::func::Return(fn, grad_other);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  // self^T @ grad summed over batch:
  // Batch 0: [[1, 4], [2, 5], [3, 6]] @ [[1, 2], [3, 4]]
  //        = [[13, 18], [17, 24], [21, 30]]
  // Batch 1: [[1, 0], [0, 1], [1, 0]] @ [[1, 0], [0, 1]]
  //        = [[1, 0], [0, 1], [1, 0]]
  // Sum: [[14, 18], [17, 25], [22, 30]]
  EvaluateAndVerifyOutput(*module, {3, 2}, {14, 18, 17, 25, 22, 30});
}

TEST_F(MatmulBuilderTest, Backward3Dx3DBroadcastGradSelf) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad =
      MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
  const mlir::MlirOp self =
      MakeDenseI32Constant(fn, {1, 2, 3}, {1, 2, 3, 4, 5, 6});
  const mlir::MlirOp other =
      MakeDenseI32Constant(fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self,
      BuildMatmulGradSelfShlo(grad, self, other, kDefaultPrecision));
  mlir::func::Return(fn, grad_self);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  // Batch 0: [[1, 2], [3, 4]] @ [[1, 0, 1], [0, 1, 1]] = [[1, 2, 3], [3, 4, 7]]
  // Batch 1: [[1, 0], [0, 1]] @ [[2, 0, 0], [0, 2, 0]] = [[2, 0, 0], [0, 2, 0]]
  // Sum over batch dim 0 (since self_shape[0] == 1): [[3, 2, 3], [3, 6, 7]]
  EvaluateAndVerifyOutput(*module, {1, 2, 3}, {3, 2, 3, 3, 6, 7});
}

}  // namespace
}  // namespace torch_tpu
