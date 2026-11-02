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

#include <complex>
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

mlir::MlirOp MakeDenseComplexF32Constant(
    mlir::MlirBuilder& builder, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<std::complex<float>> values) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType type = mlir::RankedTensorType::get(
      shape, mlir::ComplexType::get(op_builder.getF32Type()));
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

  void EvaluateAndVerifyComplexOutput(
      mlir::ModuleOp module, llvm::ArrayRef<int64_t> expected_shape,
      llvm::ArrayRef<std::complex<float>> expected_values) {
    const mlir::stablehlo::InterpreterConfiguration config;
    const llvm::SmallVector<mlir::DenseElementsAttr> empty_inputs;
    const mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>>
        eval_result = mlir::stablehlo::evalModule(module, empty_inputs, config);
    ASSERT_TRUE(mlir::succeeded(eval_result));
    ASSERT_EQ(eval_result->size(), 1);

    const mlir::DenseElementsAttr res_attr = (*eval_result)[0];
    EXPECT_THAT(res_attr.getType().getShape(),
                testing::ElementsAreArray(expected_shape));
    const llvm::SmallVector<std::complex<float>> res_vals(
        res_attr.getValues<std::complex<float>>());
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

TEST_F(MatmulBuilderTest, Forward2Dx2DWithTransposeOptions) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs = MakeDenseI32Constant(fn, {3, 2}, {1, 4, 2, 5, 3, 6});
  const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {2, 3}, {1, 3, 5, 2, 4, 6});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out,
      BuildMatmulShlo(lhs, rhs, kDefaultPrecision,
                      {.transpose_lhs = MatrixTranspose::kTransposeLastTwo,
                       .transpose_rhs = MatrixTranspose::kTransposeLastTwo}));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2, 2}, {22, 28, 49, 64});
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

TEST_F(MatmulBuilderTest, Forward3Dx1D) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs =
      MakeDenseI32Constant(fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
  const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2, 2}, {14, 32, 4, 2});
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

TEST_F(MatmulBuilderTest, Forward4Dx4DBroadcastBothSides) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs = MakeDenseI32Constant(fn, {2, 1, 1, 2}, {1, 2, 3, 4});
  const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {1, 2, 2, 1}, {1, 1, 2, 0});
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2, 2, 1, 1}, {3, 2, 7, 6});
}

TEST_F(MatmulBuilderTest, Backward3Dx2DGradSelfAndGradOther) {
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
  EvaluateAndVerifyOutput(*module, {3, 2}, {14, 18, 17, 25, 22, 30});
}

TEST_F(MatmulBuilderTest, Backward3Dx3DBroadcastGradSelfWithStaticDimsOnly) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad =
      MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
  const mlir::MlirOp other =
      MakeDenseI32Constant(fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self,
      BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{1, 2, 3}, other,
                              kDefaultPrecision));
  mlir::func::Return(fn, grad_self);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {1, 2, 3}, {3, 2, 3, 3, 6, 7});
}

TEST_F(MatmulBuilderTest, Backward3Dx3DBroadcastGradOtherWithStaticDimsOnly) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad =
      MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
  const mlir::MlirOp self =
      MakeDenseI32Constant(fn, {1, 2, 3}, {1, 2, 3, 4, 5, 6});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_other,
      BuildMatmulGradOtherShlo(grad, self, /*other_static_dims=*/{2, 3, 2},
                               kDefaultPrecision));
  mlir::func::Return(fn, grad_other);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2, 3, 2},
                          {13, 18, 17, 24, 21, 30, 1, 4, 2, 5, 3, 6});
}

TEST_F(MatmulBuilderTest, Forward1Dx1DBoundedDynamicProducesScalarType) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const auto bounds_attr =
      mlir::DenseI64ArrayAttr::get(op_builder.getContext(), {int64_t{30}});
  const auto encoding = mlir::stablehlo::TypeExtensionsAttr::get(
      op_builder.getContext(), bounds_attr);
  const auto bounded_1d_type = mlir::RankedTensorType::get(
      {mlir::ShapedType::kDynamic}, op_builder.getF32Type(), encoding);
  const mlir::MlirOp lhs = mlir::func::Argument(fn, bounded_1d_type);
  const mlir::MlirOp rhs = mlir::func::Argument(fn, bounded_1d_type);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  const auto result_type = mlir::cast<mlir::RankedTensorType>(out.getType());
  EXPECT_EQ(result_type.getRank(), 0);
  EXPECT_EQ(result_type.getEncoding(), nullptr);
}

TEST_F(MatmulBuilderTest,
       Forward1Dx1DBoundedDynamicI64ProducesValidScalarType) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const auto bounds_attr =
      mlir::DenseI64ArrayAttr::get(op_builder.getContext(), {int64_t{30}});
  const auto encoding = mlir::stablehlo::TypeExtensionsAttr::get(
      op_builder.getContext(), bounds_attr);
  const auto bounded_i64_1d_type = mlir::RankedTensorType::get(
      {mlir::ShapedType::kDynamic}, op_builder.getI64Type(), encoding);
  const mlir::MlirOp lhs = mlir::func::Argument(fn, bounded_i64_1d_type);
  const mlir::MlirOp rhs = mlir::func::Argument(fn, bounded_i64_1d_type);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, out);
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  ASSERT_TRUE(mlir::succeeded(module->verify()));

  const auto result_type = mlir::cast<mlir::RankedTensorType>(out.getType());
  EXPECT_EQ(result_type.getRank(), 0);
  EXPECT_TRUE(result_type.getElementType().isInteger(64));
  EXPECT_EQ(result_type.getEncoding(), nullptr);

  int reshape_count = 0;
  int dot_general_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (auto reshape_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
      ++reshape_count;
      const auto in_type =
          mlir::cast<mlir::RankedTensorType>(reshape_op.getOperand().getType());
      const auto out_type =
          mlir::cast<mlir::RankedTensorType>(reshape_op.getResult().getType());
      EXPECT_TRUE(in_type.getElementType().isF64());
      EXPECT_TRUE(out_type.getElementType().isF64());
    } else if (auto dot_op =
                   mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_general_count;
      const auto dot_type =
          mlir::cast<mlir::RankedTensorType>(dot_op.getResult().getType());
      EXPECT_TRUE(dot_type.getElementType().isF64());
    }
  });
  EXPECT_EQ(reshape_count, 2);
  EXPECT_EQ(dot_general_count, 1);
}

TEST_F(MatmulBuilderTest, ForwardZeroContractingDimProducesZeroTensor) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs = MakeDenseI32Constant(fn, {2, 0}, {});
  const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {0, 3}, {});
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2, 3}, {0, 0, 0, 0, 0, 0});
}

TEST_F(MatmulBuilderTest, ForwardComplexWithConjugateOptions) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp lhs =
      MakeDenseComplexF32Constant(fn, {2}, {{1.0f, 2.0f}, {3.0f, -1.0f}});
  const mlir::MlirOp rhs =
      MakeDenseComplexF32Constant(fn, {2}, {{2.0f, 1.0f}, {1.0f, 4.0f}});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out,
      BuildMatmulShlo(lhs, rhs, kDefaultPrecision,
                      {.conjugate_lhs = ConjugateMode::kConjugate}));
  mlir::func::Return(fn, out);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyComplexOutput(*module, {}, {{3.0f, 10.0f}});
}

TEST_F(MatmulBuilderTest, BackwardComplexWirtingerGradSelf) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad =
      MakeDenseComplexF32Constant(fn, {1, 1}, {{1.0f, 1.0f}});
  const mlir::MlirOp other =
      MakeDenseComplexF32Constant(fn, {2, 1}, {{2.0f, 1.0f}, {1.0f, 4.0f}});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self,
      BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{1, 2}, other,
                              kDefaultPrecision));
  mlir::func::Return(fn, grad_self);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyComplexOutput(*module, {1, 2},
                                 {{3.0f, 1.0f}, {5.0f, -3.0f}});
}

TEST_F(MatmulBuilderTest, BackwardComplexWirtingerGradOther) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad =
      MakeDenseComplexF32Constant(fn, {1, 1}, {{1.0f, 1.0f}});
  const mlir::MlirOp self =
      MakeDenseComplexF32Constant(fn, {1, 2}, {{1.0f, 2.0f}, {3.0f, -1.0f}});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_other,
      BuildMatmulGradOtherShlo(grad, self, /*other_static_dims=*/{2, 1},
                               kDefaultPrecision));
  mlir::func::Return(fn, grad_other);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyComplexOutput(*module, {2, 1},
                                 {{3.0f, -1.0f}, {2.0f, 4.0f}});
}

TEST_F(MatmulBuilderTest, Backward1Dx3DGradSelf) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
  const mlir::MlirOp other =
      MakeDenseI32Constant(fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self,
      BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{3}, other,
                              kDefaultPrecision));
  mlir::func::Return(fn, grad_self);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {3}, {7, 10, 3});
}

TEST_F(MatmulBuilderTest, Backward1Dx3DGradOther) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
  const mlir::MlirOp self = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_other,
      BuildMatmulGradOtherShlo(grad, self, /*other_static_dims=*/{2, 3, 2},
                               kDefaultPrecision));
  mlir::func::Return(fn, grad_other);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2, 3, 2},
                          {1, 2, 2, 4, 3, 6, 3, 4, 6, 8, 9, 12});
}

TEST_F(MatmulBuilderTest, Backward3Dx1DGradSelf) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
  const mlir::MlirOp other = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self,
      BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{2, 2, 3}, other,
                              kDefaultPrecision));
  mlir::func::Return(fn, grad_self);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {2, 2, 3},
                          {1, 2, 3, 2, 4, 6, 3, 6, 9, 4, 8, 12});
}

TEST_F(MatmulBuilderTest, Backward3Dx1DGradOther) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
  const mlir::MlirOp self =
      MakeDenseI32Constant(fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_other,
      BuildMatmulGradOtherShlo(grad, self, /*other_static_dims=*/{3},
                               kDefaultPrecision));
  mlir::func::Return(fn, grad_other);
  const mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  EvaluateAndVerifyOutput(*module, {3}, {12, 16, 18});
}

TEST_F(MatmulBuilderTest,
       ForwardBF16WithF32TargetElementTypeUsesPreferredElementType) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const auto lhs_type =
      mlir::RankedTensorType::get({2, 3}, op_builder.getBF16Type());
  const auto rhs_type =
      mlir::RankedTensorType::get({3, 4}, op_builder.getBF16Type());
  const mlir::MlirOp lhs = mlir::func::Argument(fn, lhs_type);
  const mlir::MlirOp rhs = mlir::func::Argument(fn, rhs_type);
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out,
      BuildMatmulShlo(lhs, rhs, kDefaultPrecision,
                      {.target_element_type = mlir::ElementType::F32}));
  mlir::func::Return(fn, out);
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  const auto result_type = mlir::cast<mlir::RankedTensorType>(out.getType());
  EXPECT_TRUE(result_type.getElementType().isF32());
  int convert_count = 0;
  int dot_general_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ConvertOp>(op)) {
      ++convert_count;
    } else if (auto dot_op =
                   mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_general_count;
      EXPECT_TRUE(mlir::cast<mlir::RankedTensorType>(dot_op.getLhs().getType())
                      .getElementType()
                      .isBF16());
      EXPECT_TRUE(mlir::cast<mlir::RankedTensorType>(dot_op.getRhs().getType())
                      .getElementType()
                      .isBF16());
      EXPECT_TRUE(
          mlir::cast<mlir::RankedTensorType>(dot_op.getResult().getType())
              .getElementType()
              .isF32());
    }
  });
  EXPECT_EQ(convert_count, 0);
  EXPECT_EQ(dot_general_count, 1);
}

TEST_F(MatmulBuilderTest,
       ComplexF64ForwardAndBackward3Dx2DDecomposesToRealF64DotGeneral) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const auto c128_type = mlir::ComplexType::get(op_builder.getF64Type());
  const auto lhs_type = mlir::RankedTensorType::get({2, 3, 4}, c128_type);
  const auto rhs_type = mlir::RankedTensorType::get({4, 5}, c128_type);
  const auto grad_type = mlir::RankedTensorType::get({2, 3, 5}, c128_type);
  const mlir::MlirOp lhs = mlir::func::Argument(fn, lhs_type);
  const mlir::MlirOp rhs = mlir::func::Argument(fn, rhs_type);
  const mlir::MlirOp grad = mlir::func::Argument(fn, grad_type);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self,
      BuildMatmulGradSelfShlo(grad, lhs, rhs, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_other,
      BuildMatmulGradOtherShlo(grad, lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, {out, grad_self, grad_other});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  int broadcast_count = 0;
  int dot_general_count = 0;
  int reshape_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::BroadcastInDimOp>(op)) {
      ++broadcast_count;
    } else if (mlir::isa<mlir::stablehlo::ReshapeOp>(op)) {
      ++reshape_count;
    } else if (auto dot_op =
                   mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_general_count;
      const auto dot_type =
          mlir::cast<mlir::RankedTensorType>(dot_op.getResult().getType());
      EXPECT_TRUE(dot_type.getElementType().isF64());
    }
  });
  EXPECT_EQ(broadcast_count, 0);
  EXPECT_EQ(reshape_count, 0);
  EXPECT_EQ(dot_general_count, 12);
}

TEST_F(MatmulBuilderTest,
       ComplexF64BackwardSingletonBatchDimUsesSliceReductionWithoutReshape) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const auto c128_type = mlir::ComplexType::get(op_builder.getF64Type());
  const auto lhs_type = mlir::RankedTensorType::get({1, 3, 4}, c128_type);
  const auto rhs_type = mlir::RankedTensorType::get({2, 4, 5}, c128_type);
  const auto grad_type = mlir::RankedTensorType::get({2, 3, 5}, c128_type);
  const mlir::MlirOp lhs = mlir::func::Argument(fn, lhs_type);
  const mlir::MlirOp rhs = mlir::func::Argument(fn, rhs_type);
  const mlir::MlirOp grad = mlir::func::Argument(fn, grad_type);
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self,
      BuildMatmulGradSelfShlo(grad, lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, {grad_self});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  int slice_count = 0;
  int reshape_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    } else if (mlir::isa<mlir::stablehlo::ReshapeOp>(op)) {
      ++reshape_count;
    }
  });
  EXPECT_EQ(slice_count, 2);
  EXPECT_EQ(reshape_count, 0);
}

}  // namespace
}  // namespace torch_tpu
