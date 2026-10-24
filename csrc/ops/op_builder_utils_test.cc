/*
 * Copyright 2025 Google LLC
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

#include "csrc/ops/op_builder_utils.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/ones.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "c10/core/DefaultDtype.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/eager_mode.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;

class OpBuilderUtilsBuilder {
 public:
  OpBuilderUtilsBuilder()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::ModuleBuilder& get() { return module_builder_; }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

mlir::ElementType GetDefaultMlirDType() {
  auto default_dtype =
      ConvertTo<mlir::ElementType>(c10::get_default_dtype_as_scalartype());
  if (!default_dtype.ok()) {
    return mlir::ElementType::F32;
  }
  return *default_dtype;
}

// TODO(b/433265252): add tests for broadcasting on MlirOps.

TEST(OpBuilderUtils, ConvertIfIntegers_TwoOperands_Int) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  TT_ASSERT_OK_AND_ASSIGN(auto default_mlir_type,
                          ConvertTo<mlir::ElementType>(default_dtype));

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::MlirOp op1 = MakeScalarConstant(builder, 1, mlir::ElementType::I32);
  mlir::MlirOp op2 = MakeScalarConstant(builder, 1, mlir::ElementType::I32);

  TT_ASSERT_OK_AND_ASSIGN(ConvertedOps ops,
                          ConvertIfIntegers(op1, op2, default_mlir_type));

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::Type mlir_type,
      GetMlirType(builder.getContext(), GetDefaultMlirDType()));
  mlir::RankedTensorType default_type =
      mlir::RankedTensorType::get({}, mlir_type);
  EXPECT_EQ(ops.op1.getType(), default_type);
  EXPECT_EQ(ops.op2.getType(), default_type);
}

TEST(OpBuilderUtils, ConvertIfIntegers_TwoOperands_Float) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  TT_ASSERT_OK_AND_ASSIGN(auto default_mlir_type,
                          ConvertTo<mlir::ElementType>(default_dtype));

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType type =
      mlir::RankedTensorType::get({}, op_builder.getF64Type());
  mlir::MlirOp op1 = MakeScalarConstant(builder, 1, mlir::ElementType::F64);
  mlir::MlirOp op2 = MakeScalarConstant(builder, 1, mlir::ElementType::F64);

  TT_ASSERT_OK_AND_ASSIGN(ConvertedOps ops,
                          ConvertIfIntegers(op1, op2, default_mlir_type));
  EXPECT_EQ(ops.op1.getType(), type);
  EXPECT_EQ(ops.op2.getType(), type);
}

TEST(OpBuilderUtils, ConvertIfIntegers_TwoOperands_Int_Float) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  TT_ASSERT_OK_AND_ASSIGN(auto default_mlir_type,
                          ConvertTo<mlir::ElementType>(default_dtype));

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType int_type =
      mlir::RankedTensorType::get({}, op_builder.getI32Type());
  mlir::RankedTensorType float_type =
      mlir::RankedTensorType::get({}, op_builder.getF32Type());
  mlir::MlirOp op1 = MakeScalarConstant(builder, 1, int_type.getElementType());
  mlir::MlirOp op2 =
      MakeScalarConstant(builder, 1, float_type.getElementType());

  TT_ASSERT_OK_AND_ASSIGN(ConvertedOps ops,
                          ConvertIfIntegers(op1, op2, default_mlir_type));
  EXPECT_EQ(ops.op1.getType(), float_type);
  EXPECT_EQ(ops.op2.getType(), float_type);
}

TEST(OpBuilderUtils, ConvertIfIntegers_TwoOperands_Float_Int) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  TT_ASSERT_OK_AND_ASSIGN(auto default_mlir_type,
                          ConvertTo<mlir::ElementType>(default_dtype));

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType int_type =
      mlir::RankedTensorType::get({}, op_builder.getI32Type());
  mlir::RankedTensorType float_type =
      mlir::RankedTensorType::get({}, op_builder.getF32Type());
  mlir::MlirOp op1 =
      MakeScalarConstant(builder, 1, float_type.getElementType());
  mlir::MlirOp op2 = MakeScalarConstant(builder, 1, int_type.getElementType());

  TT_ASSERT_OK_AND_ASSIGN(ConvertedOps ops,
                          ConvertIfIntegers(op1, op2, default_mlir_type));
  EXPECT_EQ(ops.op1.getType(), float_type);
  EXPECT_EQ(ops.op2.getType(), float_type);
}

TEST(OpBuilderUtils, ConvertIfIntegers_TwoIntegerOperands) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  TT_ASSERT_OK_AND_ASSIGN(auto default_mlir_type,
                          ConvertTo<mlir::ElementType>(default_dtype));

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType int_type =
      mlir::RankedTensorType::get({}, op_builder.getI32Type());
  mlir::RankedTensorType float_type =
      mlir::RankedTensorType::get({}, op_builder.getF32Type());
  mlir::MlirOp op1 = MakeScalarConstant(builder, 1, int_type.getElementType());
  mlir::MlirOp op2 = MakeScalarConstant(builder, 1, int_type.getElementType());

  TT_ASSERT_OK_AND_ASSIGN(ConvertedOps ops,
                          ConvertIfIntegers(op1, op2, default_mlir_type));
  EXPECT_EQ(ops.op1.getType(), float_type);
  EXPECT_EQ(ops.op2.getType(), float_type);
}

TEST(OpBuilderUtils, ConvertIfInteger_OneOperand_Int) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  TT_ASSERT_OK_AND_ASSIGN(auto default_mlir_type,
                          ConvertTo<mlir::ElementType>(default_dtype));

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType type =
      mlir::RankedTensorType::get({}, op_builder.getI32Type());
  mlir::MlirOp op = MakeScalarConstant(builder, 1, type.getElementType());

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp converted_op,
                          ConvertIfInteger(op, default_mlir_type));

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::Type mlir_type,
      GetMlirType(builder.getContext(), GetDefaultMlirDType()));
  mlir::RankedTensorType default_type =
      mlir::RankedTensorType::get({}, mlir_type);
  EXPECT_EQ(converted_op.getType(), default_type);
}

TEST(OpBuilderUtils, ConvertIfInteger_OneOperand_Float) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  TT_ASSERT_OK_AND_ASSIGN(auto default_mlir_type,
                          ConvertTo<mlir::ElementType>(default_dtype));

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType type =
      mlir::RankedTensorType::get({}, op_builder.getF64Type());
  mlir::MlirOp op = MakeScalarConstant(builder, 1, type.getElementType());

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp converted_op,
                          ConvertIfInteger(op, default_mlir_type));
  EXPECT_EQ(converted_op.getType(), type);
}

TEST(OpBuilderUtils, GetMinFiniteValueAttr_Float32) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto result_attr = GetMinFiniteValueAttr(op_builder.getF32Type(), op_builder);
  std::string result_str;
  llvm::raw_string_ostream sstream(result_str);
  result_attr.print(sstream);
  EXPECT_EQ(result_str, "-3.40282347E+38 : f32");
}

TEST(OpBuilderUtils, GetMinFiniteValueAttr_Int32) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto result_attr = GetMinFiniteValueAttr(op_builder.getI32Type(), op_builder);
  std::string result_str;
  llvm::raw_string_ostream sstream(result_str);
  result_attr.print(sstream);
  EXPECT_EQ(result_str, "-2147483648 : i32");
}

TEST(OpBuilderUtils, GetMinFiniteValueAttr_Bool) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto result_attr = GetMinFiniteValueAttr(op_builder.getI1Type(), op_builder);
  std::string result_str;
  llvm::raw_string_ostream sstream(result_str);
  result_attr.print(sstream);
  EXPECT_EQ(result_str, "false");
}

TEST(OpBuilderUtils, GetMinFiniteValueAttr_UnsupportedType) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto result_attr =
      GetMinFiniteValueAttr(op_builder.getNoneType(), op_builder);
  EXPECT_EQ(result_attr, nullptr);
}

TEST(OpBuilderUtils, GetMaxFiniteValueAttr_Float32) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto result_attr = GetMaxFiniteValueAttr(op_builder.getF32Type(), op_builder);
  std::string result_str;
  llvm::raw_string_ostream sstream(result_str);
  result_attr.print(sstream);
  EXPECT_EQ(result_str, "3.40282347E+38 : f32");
}

TEST(OpBuilderUtils, GetMaxFiniteValueAttr_Int32) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto result_attr = GetMaxFiniteValueAttr(op_builder.getI32Type(), op_builder);
  std::string result_str;
  llvm::raw_string_ostream sstream(result_str);
  result_attr.print(sstream);
  EXPECT_EQ(result_str, "2147483647 : i32");
}

TEST(OpBuilderUtils, GetMaxFiniteValueAttr_Bool) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto result_attr = GetMaxFiniteValueAttr(op_builder.getI1Type(), op_builder);
  std::string result_str;
  llvm::raw_string_ostream sstream(result_str);
  result_attr.print(sstream);
  EXPECT_EQ(result_str, "true");
}

TEST(OpBuilderUtils, GetMaxFiniteValueAttr_UnsupportedType) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  auto result_attr =
      GetMaxFiniteValueAttr(op_builder.getNoneType(), op_builder);
  EXPECT_EQ(result_attr, nullptr);
}

TEST(OpBuilderUtils, InferSize_Broadcastable_Check) {
  Dimensions dims1_broadcastable = {2, 1};
  Dimensions dims2_broadcastible = {1, 3};
  TT_ASSERT_OK_AND_ASSIGN(auto broadcastable_size,
                          InferSize(dims1_broadcastable, dims2_broadcastible));
  EXPECT_THAT(broadcastable_size, ElementsAre(2, 3));

  Dimensions dims1_non_broadcastable = {2};
  Dimensions dims2_non_broadcastable = {3};
  auto result = InferSize(dims1_non_broadcastable, dims2_non_broadcastable);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("must match the size of tensor"));
}

TEST(InferSize, WorksWithTwoTensors) {
  at::Tensor tensor1 = at::ones({2, 1});
  at::Tensor tensor2 = at::ones({1, 3});
  TT_ASSERT_OK_AND_ASSIGN(auto result, InferSize(tensor1, tensor2));
  EXPECT_THAT(result, ElementsAre(2, 3));
}

TEST(InferSize, WorksWithMoreThanTwoTensors) {
  at::Tensor tensor1 = at::ones({4, 2, 1});
  at::Tensor tensor2 = at::ones({1, 3});
  at::Tensor tensor3 = at::ones({1, 2, 1});
  TT_ASSERT_OK_AND_ASSIGN(auto result, InferSize(tensor1, tensor2, tensor3));
  EXPECT_THAT(result, ElementsAre(4, 2, 3));
}

TEST(InferSize, WorksWithDimsAndTensors) {
  at::Tensor tensor1 = at::ones({4, 2, 1});
  Dimensions dims2 = {1, 3};
  at::Tensor tensor3 = at::ones({1, 2, 1});
  TT_ASSERT_OK_AND_ASSIGN(auto result, InferSize(tensor1, dims2, tensor3));
  EXPECT_THAT(result, ElementsAre(4, 2, 3));
}

TEST(BroadcastIfNeeded, Scalar) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op = MakeConstant(builder, 1.0f, op_builder.getF32Type(), {});
  Dimensions target_shape = {2, 3};
  TT_ASSERT_OK_AND_ASSIGN(auto result, BroadcastIfNeeded(op, target_shape));
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(2, 3));
}

TEST(BroadcastIfNeeded, SameShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {2, 3});
  Dimensions target_shape = {2, 3};
  TT_ASSERT_OK_AND_ASSIGN(auto result, BroadcastIfNeeded(op, target_shape));
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(2, 3));
}

TEST(BroadcastIfNeeded, BroadcastDim) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op = MakeConstant(builder, 1.0f, op_builder.getF32Type(), {3});
  Dimensions target_shape = {2, 3};
  TT_ASSERT_OK_AND_ASSIGN(auto result, BroadcastIfNeeded(op, target_shape));
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(2, 3));
}

TEST(BroadcastIfNeeded, Broadcast1x3to2x3) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {1, 3});
  Dimensions target_shape = {2, 3};
  TT_ASSERT_OK_AND_ASSIGN(auto result, BroadcastIfNeeded(op, target_shape));
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(2, 3));
}

TEST(BroadcastIfNeeded, StablehloDims) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op = MakeConstant(builder, 5, op_builder.getI32Type(), {1, 3});
  mlir::stablehlo::Dimensions target_shape = {{2}, {3}};
  TT_ASSERT_OK_AND_ASSIGN(auto result, BroadcastIfNeeded(op, target_shape));
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(2, 3));
}

TEST(BroadcastIfNeeded, StablehloDimsWithBoundOp) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op = MakeConstant(builder, 5, op_builder.getI32Type(), {3});
  mlir::MlirOp bound_op =
      MakeConstant(builder, 5, op_builder.getI32Type(), {1});
  mlir::stablehlo::Dimensions target_shape = {{10, bound_op.getValue(), 0},
                                              {3}};
  TT_ASSERT_OK_AND_ASSIGN(auto result, BroadcastIfNeeded(op, target_shape));
  auto dims = GetDimensions(result);
  EXPECT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].size, 10);  // check padded size
  EXPECT_EQ(dims[1].size, 3);   // check static size

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_TRUE(result_type.isDynamicDim(0));
  EXPECT_FALSE(result_type.isDynamicDim(1));
  EXPECT_EQ(result_type.getDimSize(1), 3);
}

TEST(BroadcastIfNeeded, Failure) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op = MakeConstant(builder, 5, op_builder.getI32Type(), {2});
  mlir::stablehlo::Dimensions target_shape = {{3}};
  auto result = BroadcastIfNeeded(op, target_shape);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("failed to broadcast tensor: "));
}

TEST(BroadcastIfNeeded, ExplicitBroadcastDims) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {2, 3});
  mlir::MlirOp target =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {2, 5, 3});
  TT_ASSERT_OK_AND_ASSIGN(auto result, BroadcastIfNeeded(op, target, {0, 2}));
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(2, 5, 3));
}

TEST(BroadcastIfNeeded, ExplicitBroadcastDimsBothDynamic) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  mlir::MlirOp bound_op = MakeConstant(builder, 5, op_builder.getI32Type(), {});

  mlir::MlirOp base_input =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {10, 3});
  mlir::Value input_val = mlir::stablehlo::SetDimensionSizeOp::create(
      op_builder, op_builder.getUnknownLoc(), base_input.getValue(),
      bound_op.getValue(), 0);
  mlir::MlirOp input_op(builder, input_val);

  mlir::MlirOp base_target =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {10, 4, 3});
  mlir::Value target_val = mlir::stablehlo::SetDimensionSizeOp::create(
      op_builder, op_builder.getUnknownLoc(), base_target.getValue(),
      bound_op.getValue(), 0);
  mlir::MlirOp target_op(builder, target_val);

  TT_ASSERT_OK_AND_ASSIGN(auto result,
                          BroadcastIfNeeded(input_op, target_op, {0, 2}));

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_EQ(result_type.getRank(), 3);
  EXPECT_TRUE(result_type.isDynamicDim(0));
  EXPECT_FALSE(result_type.isDynamicDim(1));
  EXPECT_FALSE(result_type.isDynamicDim(2));
  EXPECT_EQ(result_type.getDimSize(1), 4);
  EXPECT_EQ(result_type.getDimSize(2), 3);

  auto dims = GetDimensions(result);
  EXPECT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0].size, 10);
  EXPECT_TRUE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 4);
  EXPECT_EQ(dims[2].size, 3);
}

TEST(BroadcastIfNeeded, ExplicitBroadcastDimsFailure) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {2, 3});
  mlir::MlirOp target =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {2, 5, 3});
  // Mismatched rank size for broadcast_dimensions.
  auto result = BroadcastIfNeeded(op, target, {0});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kInvalidArgument);
}

TEST(BroadcastIfNeeded,
     ExplicitBroadcastDimsBoundedInputToStaticTargetFailure) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  mlir::MlirOp bound_op = MakeConstant(builder, 5, op_builder.getI32Type(), {});

  mlir::MlirOp base_input =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {10, 3});
  mlir::Value input_val = mlir::stablehlo::SetDimensionSizeOp::create(
      op_builder, op_builder.getUnknownLoc(), base_input.getValue(),
      bound_op.getValue(), 0);
  mlir::MlirOp input_op(builder, input_val);

  // Target is static shape [10, 4, 3]
  mlir::MlirOp target_op =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {10, 4, 3});

  auto result = BroadcastIfNeeded(input_op, target_op, {0, 2});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("cannot mix bounded and static dimensions"));
}

TEST(BroadcastIfNeeded, ExplicitBroadcastDimsStaticInputToDynamicTarget) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  mlir::MlirOp bound_op = MakeConstant(builder, 5, op_builder.getI32Type(), {});

  // Input is static shape [3, 1]
  mlir::MlirOp input_op =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {3, 1});

  // Target shape: [3, 4, ?] with bound 10 on dim 2 (last dim)
  mlir::MlirOp base_target =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {3, 4, 10});
  mlir::Value target_val = mlir::stablehlo::SetDimensionSizeOp::create(
      op_builder, op_builder.getUnknownLoc(), base_target.getValue(),
      bound_op.getValue(), 2);
  mlir::MlirOp target_op(builder, target_val);

  // Map input dim 0 -> target dim 0, input dim 1 -> target dim 1 (non-numpy
  // alignment)
  TT_ASSERT_OK_AND_ASSIGN(auto result,
                          BroadcastIfNeeded(input_op, target_op, {0, 1}));

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  ASSERT_TRUE(result_type);
  EXPECT_EQ(result_type.getRank(), 3);
  EXPECT_FALSE(result_type.isDynamicDim(0));
  EXPECT_FALSE(result_type.isDynamicDim(1));
  EXPECT_TRUE(result_type.isDynamicDim(2));
  EXPECT_EQ(result_type.getDimSize(0), 3);
  EXPECT_EQ(result_type.getDimSize(1), 4);

  auto dims = GetDimensions(result);
  EXPECT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0].size, 3);
  EXPECT_EQ(dims[1].size, 4);
  EXPECT_EQ(dims[2].size, 10);
  EXPECT_TRUE(dims[2].boundOp.has_value());
}

TEST(CastIfNeeded, I32ToF32) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType int_type =
      mlir::RankedTensorType::get({}, op_builder.getI32Type());
  mlir::MlirOp op = MakeScalarConstant(builder, 1, int_type.getElementType());
  TT_ASSERT_OK_AND_ASSIGN(auto result,
                          CastIfNeeded(op, mlir::ElementType::F32));
  mlir::RankedTensorType float_type =
      mlir::RankedTensorType::get({}, op_builder.getF32Type());
  EXPECT_EQ(result.getType(), float_type);
}

TEST(InferSize, WorksWithDimsAndScalars) {
  at::Tensor tensor1 = at::ones({4, 2, 1});
  Dimensions dims2 = {1, 3};
  at::Scalar scalar3 = 1;
  TT_ASSERT_OK_AND_ASSIGN(auto result, InferSize(tensor1, dims2, scalar3));
  EXPECT_THAT(result, ElementsAre(4, 2, 3));
}

TEST(AnnotateBufferDonations, SmokeTest) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::OpBuilder& op_builder = fb.getOpBuilder();
  auto arg = mlir::func::Argument(
      fb, mlir::RankedTensorType::get({}, op_builder.getF32Type()));
  mlir::func::Return(fb, arg);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  AnnotateBufferDonations(module.get(), {0});
  std::optional<mlir::ArrayAttr> arg_attrs =
      module->lookupSymbol<mlir::func::FuncOp>("main").getArgAttrs();

  EXPECT_TRUE(arg_attrs.has_value());
  EXPECT_EQ(arg_attrs.value().size(), 1);
  auto dict_attr = mlir::dyn_cast<mlir::DictionaryAttr>(arg_attrs.value()[0]);
  auto donor_attr = dict_attr.getAs<mlir::BoolAttr>("jax.buffer_donor");
  EXPECT_TRUE(donor_attr);             // dict has value
  EXPECT_TRUE(donor_attr.getValue());  // value is true
}

TEST(EagerModeTest, IsDeferNeverMode) {
  EXPECT_TRUE(IsDeferNeverMode(EagerMode::kDeferNever));
  EXPECT_TRUE(IsDeferNeverMode(EagerMode::kDeferNeverAndLaunchBlocking));
  EXPECT_FALSE(IsDeferNeverMode(EagerMode::kDeferAndFuse));
  EXPECT_FALSE(IsDeferNeverMode(EagerMode::kInternalDeferAll));
  EXPECT_FALSE(IsDeferNeverMode(EagerMode::kInternalCompileFxGraph));
}

TEST(ReshapeFromStaticDimensions, StaticShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = mb.getOpBuilder();
  mlir::MlirOp op = MakeConstant(mb, 1.0f, op_builder.getF32Type(), {2, 3});
  Dimensions static_input_shape = {2, 3};
  Dimensions static_output_shape = {6};
  TT_ASSERT_OK_AND_ASSIGN(
      auto reshaped,
      ReshapeFromStaticDimensions(op, static_input_shape, static_output_shape));
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(reshaped.getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(6));
}

TEST(BroadcastIfNeeded, BroadcastLikeOtherOp) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = mb.getOpBuilder();
  mlir::MlirOp cst = MakeConstant(mb, 1.0f, op_builder.getF32Type(), {});
  mlir::MlirOp op = MakeConstant(mb, 2.0f, op_builder.getF32Type(), {2, 2});
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp cst_bcast, BroadcastIfNeeded(cst, op));
  mlir::RankedTensorType cst_bcast_type =
      mlir::cast<mlir::RankedTensorType>(cst_bcast.getType());
  EXPECT_THAT(cst_bcast_type.getShape(), ElementsAre(2, 2));
}

TEST(ApplyBroadcastIfNeeded, Span) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = mb.getOpBuilder();
  mlir::MlirOp op1 = MakeConstant(mb, 1.0f, op_builder.getF32Type(), {3, 1});
  mlir::MlirOp op2 = MakeConstant(mb, 2.0f, op_builder.getF32Type(), {1, 5});
  std::vector<mlir::MlirOp> ops = {op1, op2};
  TT_ASSERT_OK_AND_ASSIGN(const auto& bcast,
                          ApplyBroadcastIfNeeded(ops));  // NOLINT
  ASSERT_EQ(bcast.size(), 2);
  auto type1 = mlir::cast<mlir::RankedTensorType>(bcast[0].getType());
  auto type2 = mlir::cast<mlir::RankedTensorType>(bcast[1].getType());
  EXPECT_THAT(type1.getShape(), ElementsAre(3, 5));
  EXPECT_THAT(type2.getShape(), ElementsAre(3, 5));
}

TEST(ApplyBroadcastIfNeeded, DifferentRanks) {
  OpBuilderUtilsBuilder builder;
  mlir::ModuleBuilder& mb = builder.get();
  mlir::OpBuilder& op_builder = mb.getOpBuilder();
  mlir::MlirOp op1 = MakeConstant(mb, 1.0f, op_builder.getF32Type(), {5});
  mlir::MlirOp op2 = MakeConstant(mb, 2.0f, op_builder.getF32Type(), {2, 3, 1});

  TT_ASSERT_OK_AND_ASSIGN(
      const auto& bcast,
      ApplyBroadcastIfNeeded(std::vector<mlir::MlirOp>{op1, op2}));  // NOLINT
  ASSERT_EQ(bcast.size(), 2);
  auto type1 = mlir::cast<mlir::RankedTensorType>(bcast[0].getType());
  auto type2 = mlir::cast<mlir::RankedTensorType>(bcast[1].getType());
  EXPECT_THAT(type1.getShape(), ElementsAre(2, 3, 5));
  EXPECT_THAT(type2.getShape(), ElementsAre(2, 3, 5));
}

TEST(ApplyBroadcastIfNeeded, DynamicShape) {
  OpBuilderUtilsBuilder builder;
  mlir::ModuleBuilder& mb = builder.get();
  mlir::OpBuilder& op_builder = mb.getOpBuilder();

  // op1 has dynamic bounded shape [?x1] with upper bound 10 and dynamic size 5
  // on dim 0
  mlir::MlirOp bound_op = MakeConstant(mb, 5, op_builder.getI32Type(), {});
  mlir::MlirOp base_op1 =
      MakeConstant(mb, 1.0f, op_builder.getF32Type(), {10, 1});
  mlir::MlirOp op1 =
      mlir::stablehlo::SetDimensionSize(base_op1, bound_op, /*dim=*/0);

  // op2 has static shape [1, 4]
  mlir::MlirOp op2 = MakeConstant(mb, 2.0f, op_builder.getF32Type(), {1, 4});

  TT_ASSERT_OK_AND_ASSIGN(
      const auto& bcast,
      ApplyBroadcastIfNeeded(std::vector<mlir::MlirOp>{op1, op2}));  // NOLINT
  ASSERT_EQ(bcast.size(), 2);

  mlir::MlirOp bcast_op1 = bcast[0];
  mlir::MlirOp bcast_op2 = bcast[1];

  auto bcast_op1_type = mlir::cast<mlir::RankedTensorType>(bcast_op1.getType());
  auto bcast_op2_type = mlir::cast<mlir::RankedTensorType>(bcast_op2.getType());

  // Both tensors should now have dynamic shape [?x4]
  EXPECT_TRUE(bcast_op1_type.isDynamicDim(0));
  EXPECT_FALSE(bcast_op1_type.isDynamicDim(1));
  EXPECT_EQ(bcast_op1_type.getDimSize(1), 4);

  EXPECT_TRUE(bcast_op2_type.isDynamicDim(0));
  EXPECT_FALSE(bcast_op2_type.isDynamicDim(1));
  EXPECT_EQ(bcast_op2_type.getDimSize(1), 4);

  // Verify bounded dimensions metadata on broadcasted op1
  auto bcast_op1_dims = GetDimensions(bcast_op1);
  EXPECT_EQ(bcast_op1_dims.size(), 2);
  EXPECT_EQ(bcast_op1_dims[0].size, 10);
  EXPECT_TRUE(bcast_op1_dims[0].boundOp.has_value());
  EXPECT_EQ(bcast_op1_dims[1].size, 4);
  EXPECT_FALSE(bcast_op1_dims[1].boundOp.has_value());

  // Verify bounded dimensions metadata on broadcasted op2
  auto bcast_op2_dims = GetDimensions(bcast_op2);
  EXPECT_EQ(bcast_op2_dims.size(), 2);
  EXPECT_EQ(bcast_op2_dims[0].size, 10);
  EXPECT_TRUE(bcast_op2_dims[0].boundOp.has_value());
  EXPECT_EQ(bcast_op2_dims[1].size, 4);
  EXPECT_FALSE(bcast_op2_dims[1].boundOp.has_value());
}

TEST(GetNumElements, StaticShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = mb.getOpBuilder();
  mlir::MlirOp op = MakeConstant(mb, 2.0f, op_builder.getF32Type(), {2, 2});
  absl::StatusOr<mlir::MlirOp> num_elements =
      GetNumElements(op, op_builder.getI32Type());
  ASSERT_TRUE(num_elements.ok());
  EXPECT_TRUE(mlir::isa<mlir::stablehlo::ConstantOp>(
      num_elements->getValue().getDefiningOp()));
}

TEST(GetNumElements, DynamicShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = mb.getOpBuilder();
  mlir::MlirOp op = MakeConstant(mb, 2.0f, op_builder.getF32Type(), {10});
  mlir::MlirOp dim_size = MakeConstant(mb, 5, op_builder.getI32Type(), {});
  mlir::MlirOp op_dyn =
      mlir::stablehlo::SetDimensionSize(op, dim_size, /*dim=*/0);
  absl::StatusOr<mlir::MlirOp> num_elements =
      GetNumElements(op_dyn, op_builder.getI32Type());
  ASSERT_TRUE(num_elements.ok());
  EXPECT_FALSE(mlir::isa<mlir::stablehlo::ConstantOp>(
      num_elements->getValue().getDefiningOp()));
}

// Helper for DynamicReshapeFromStaticDimensions tests.
struct ReshapeTestResult {
  absl::StatusOr<mlir::MlirOp> reshaped_op;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};
ReshapeTestResult BuildReshapeGraph(
    Dimensions bounded_input_shape, absl::Span<const int64_t> bound_dims,
    Dimensions static_input_shape, Dimensions static_output_shape,
    OpBuilderUtilsBuilder& op_builder_utils_builder) {
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::OpBuilder& op_builder = fb.getOpBuilder();

  // Make op of shape [bounded_input_shape]
  auto arg = mlir::func::Argument(
      fb, mlir::RankedTensorType::get(bounded_input_shape,
                                      op_builder.getF32Type()));
  auto bounded_op = arg;
  for (int64_t bound_dim : bound_dims) {
    auto size = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({}, op_builder.getI32Type()));
    bounded_op =
        mlir::stablehlo::SetDimensionSize(bounded_op, size, /*dim=*/bound_dim);
  }

  auto reshaped = ReshapeFromStaticDimensions(bounded_op, static_input_shape,
                                              static_output_shape);
  if (reshaped.ok()) {
    mlir::func::Return(fb, *reshaped);
  }
  return {reshaped, mb.build()};
}

struct DynamicReshapeParams {
  std::string test_name;
  Dimensions bounded_input_shape;
  Dimensions bound_dims;
  Dimensions static_input_shape;
  Dimensions static_output_shape;
  Dimensions expected_bounded_output_shape;
  std::vector<bool> expected_output_bounded;
};

class DynamicReshapeFromStaticDimensionsTest
    : public testing::TestWithParam<DynamicReshapeParams> {};

TEST_P(DynamicReshapeFromStaticDimensionsTest, ValidReshape) {
  const auto& params = GetParam();
  OpBuilderUtilsBuilder op_builder_utils_builder;
  ReshapeTestResult result = BuildReshapeGraph(
      params.bounded_input_shape, params.bound_dims, params.static_input_shape,
      params.static_output_shape, op_builder_utils_builder);
  TT_ASSERT_OK_AND_ASSIGN(auto reshaped_op, result.reshaped_op);

  // Check that the reshaped op has proper bounded dynamic dimensions.
  mlir::stablehlo::Dimensions reshaped_dims = GetDimensions(reshaped_op);
  for (int i = 0; i < params.expected_bounded_output_shape.size(); ++i) {
    ASSERT_EQ(reshaped_dims[i].size, params.expected_bounded_output_shape[i]);
    ASSERT_EQ(reshaped_dims[i].boundOp.has_value(),
              params.expected_output_bounded[i]);
  }
  std::cout << "module: " << DebugString(result.module.get()) << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    DynamicReshapeFromStaticDimensionsTests,
    DynamicReshapeFromStaticDimensionsTest,
    testing::ValuesIn<DynamicReshapeParams>(
        {{"Collapse",
          /*bounded_input_shape=*/{2, 13, 5},
          /*bound_dims=*/{1},
          /*static_input_shape=*/{2, 3, 5},
          /*static_output_shape=*/{6, 5},
          /*expected_bounded_output_shape=*/{26, 5},
          /*expected_output_bounded=*/{true, false}},
         {"Expand",
          /*bounded_input_shape=*/{6, 10},
          /*bound_dims=*/{1},
          /*static_input_shape=*/{6, 5},
          /*static_output_shape=*/{2, 3, 5},
          /*expected_bounded_output_shape=*/{2, 3, 10},
          /*expected_output_bounded=*/{false, false, true}},
         {"SqueezeNonDynDimBeforeBound",
          /*bounded_input_shape=*/{1, 2, 10},
          /*bound_dims=*/{2},
          /*static_input_shape=*/{1, 2, 5},
          /*static_output_shape=*/{2, 5},
          /*expected_bounded_output_shape=*/{2, 10},
          /*expected_output_bounded=*/{false, true}},
         {"SqueezeNonDynDimAfterBound",
          /*bounded_input_shape=*/{10, 2, 1},
          /*bound_dims=*/{0},
          /*static_input_shape=*/{5, 2, 1},
          /*static_output_shape=*/{5, 2},
          /*expected_bounded_output_shape=*/{10, 2},
          /*expected_output_bounded=*/{true, false}},
         {"UnsqueezeNonDynDim0",
          /*bounded_input_shape=*/{5, 10},
          /*bound_dims=*/{1},
          /*static_input_shape=*/{5, 2},
          /*static_output_shape=*/{1, 5, 2},
          /*expected_bounded_output_shape=*/{1, 5, 10},
          /*expected_output_bounded=*/{false, false, true}},
         {"SqueezeMultiple",
          /*bounded_input_shape=*/{1, 8, 1, 10, 1},
          /*bound_dims=*/{3},
          /*static_input_shape=*/{1, 8, 1, 5, 1},
          /*static_output_shape=*/{8, 5},
          /*expected_bounded_output_shape=*/{8, 10},
          /*expected_output_bounded=*/{false, true}},
         {"Flatten",
          /*bounded_input_shape=*/{1, 8, 1, 10, 1},
          /*bound_dims=*/{3},
          /*static_input_shape=*/{1, 8, 1, 5, 1},
          /*static_output_shape=*/{40},
          /*expected_bounded_output_shape=*/{80},
          /*expected_output_bounded=*/{true}},
         {"Unflatten",
          /*bounded_input_shape=*/{1, 1024},
          /*bound_dims=*/{1},
          /*static_input_shape=*/{1, 10},
          /*static_output_shape=*/{1, 1, 10, 1},
          /*expected_bounded_output_shape=*/{1, 1, 1024, 1},
          /*expected_output_bounded=*/{false, false, true, false}},
         {"CollapseMultipleBoundedDimsToMultipleBoundedDims",
          /*bounded_input_shape=*/{2, 10, 5, 10},
          /*bound_dims=*/{1, 3},
          /*static_input_shape=*/{2, 3, 5, 6},
          /*static_output_shape=*/{6, 30},
          /*expected_bounded_output_shape=*/{20, 50},
          /*expected_output_bounded=*/{true, true}},
         {"CollapseMultipleBoundedDimstoSingleBoundedDim",
          /*bounded_input_shape=*/{2, 10, 5, 10},
          /*bound_dims=*/{1, 3},
          /*static_input_shape=*/{2, 3, 5, 6},
          /*static_output_shape=*/{2, 90},
          /*expected_bounded_output_shape=*/{2, 500},
          /*expected_output_bounded=*/{false, true}},
         {"ExpandMultipleBoundedDimtoMultipleBoundedDims",
          /*bounded_input_shape=*/{2, 10, 20},
          /*bound_dims=*/{1, 2},
          /*static_input_shape=*/{2, 3, 6},
          /*static_output_shape=*/{2, 3, 1, 6, 1, 1},
          /*expected_bounded_output_shape=*/{2, 10, 1, 20, 1, 1},
          /*expected_output_bounded=*/
          {false, true, false, true, false, false}},
         {"TransposeLikeSingleBoundedDim",
          /*bounded_input_shape=*/{1, 10, 1, 6, 5},
          /*bound_dims=*/{1},
          /*static_input_shape=*/{1, 4, 1, 6, 5},
          /*static_output_shape=*/{4, 6, 1, 1, 5},
          /*expected_bounded_output_shape=*/{10, 6, 1, 1, 5},
          /*expected_output_bounded=*/{true, false, false, false, false}},
         {"TransposeLikeMultipleBoundedDims",
          /*bounded_input_shape=*/{10, 1, 1, 10, 6, 5},
          /*bound_dims=*/{0, 3},
          /*static_input_shape=*/{4, 1, 1, 6, 6, 5},
          /*static_output_shape=*/{4, 6, 6, 5, 1, 1},
          /*expected_bounded_output_shape=*/{10, 10, 6, 5, 1, 1},
          /*expected_output_bounded=*/
          {true, true, false, false, false, false}}}),
    [](const testing::TestParamInfo<
        DynamicReshapeFromStaticDimensionsTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(DynamicReshapeFromStaticDimensions, ErrorReassociationNotFound) {
  Dimensions bounded_input_shape = {2, 3, 5, 10};
  int64_t bound_dim = 3;
  Dimensions static_input_shape = {2, 3, 5, 6};
  Dimensions static_output_shape = {3, 10, 6};
  OpBuilderUtilsBuilder op_builder_utils_builder;
  ReshapeTestResult result =
      BuildReshapeGraph(bounded_input_shape, {bound_dim}, static_input_shape,
                        static_output_shape, op_builder_utils_builder);
  ASSERT_FALSE(result.reshaped_op.ok());
  EXPECT_EQ(result.reshaped_op.status().code(), error::kInvalidArgument);
  EXPECT_THAT(result.reshaped_op.status().message(),
              testing::HasSubstr("unable to determine reassociation indices"));
}

TEST(DynamicReshapeFromStaticDimensions, ErrorExpandDynamicToMultiple) {
  Dimensions bounded_input_shape = {10, 6};
  int64_t bound_dim = 0;
  Dimensions static_input_shape = {4, 6};
  Dimensions static_output_shape = {2, 2, 6};
  OpBuilderUtilsBuilder op_builder_utils_builder;
  ReshapeTestResult result =
      BuildReshapeGraph(bounded_input_shape, {bound_dim}, static_input_shape,
                        static_output_shape, op_builder_utils_builder);
  ASSERT_FALSE(result.reshaped_op.ok());
  EXPECT_EQ(result.reshaped_op.status().code(), error::kInvalidArgument);
  EXPECT_THAT(result.reshaped_op.status().message(),
              testing::HasSubstr("expands to multiple non one output dims"));
}

TEST(DynamicReshapeFromStaticDimensions, ErrorNonTransposeLikeReshape) {
  Dimensions bounded_input_shape = {10, 1, 5};
  int64_t bound_dim = 0;
  Dimensions static_input_shape = {3, 1, 5};
  Dimensions static_output_shape = {5, 3, 1};
  OpBuilderUtilsBuilder op_builder_utils_builder;
  ReshapeTestResult result =
      BuildReshapeGraph(bounded_input_shape, {bound_dim}, static_input_shape,
                        static_output_shape, op_builder_utils_builder);
  ASSERT_FALSE(result.reshaped_op.ok());
  EXPECT_EQ(result.reshaped_op.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      result.reshaped_op.status().message(),
      testing::HasSubstr(
          "reshape reassociation not supported for same sized reshapes"));
}

// Returns an f32 function argument of `shape` with `bound_dims` made bounded
// dynamic by set_dimension_size with runtime size arguments.
mlir::MlirOp MakeBoundedArg(mlir::func::FunctionBuilder& fb,
                            const Dimensions& shape,
                            absl::Span<const int64_t> bound_dims) {
  mlir::OpBuilder& op_builder = fb.getOpBuilder();
  mlir::MlirOp op = mlir::func::Argument(
      fb, mlir::RankedTensorType::get(shape, op_builder.getF32Type()));
  for (int64_t dim : bound_dims) {
    mlir::MlirOp size = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({}, op_builder.getI32Type()));
    op = mlir::stablehlo::SetDimensionSize(op, size, dim);
  }
  return op;
}

TEST(GetStaticShape, ReturnsUpperBoundsForBoundedDims) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::func::FunctionBuilder fb(op_builder_utils_builder.get(), "main");
  mlir::MlirOp op = MakeBoundedArg(fb, {2, 13, 5}, /*bound_dims=*/{1});
  ASSERT_TRUE(GetTensorTypeOrDie(op).isDynamicDim(1));
  EXPECT_THAT(GetStaticShape(op), ElementsAre(2, 13, 5));
}

TEST(ReshapeToStaticDimensions, SameShapeReturnsInput) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::func::FunctionBuilder fb(op_builder_utils_builder.get(), "main");
  mlir::MlirOp op = MakeBoundedArg(fb, {2, 13, 5}, /*bound_dims=*/{1});
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp reshaped,
                          ReshapeToStaticDimensions(op, {2, 13, 5}));
  EXPECT_EQ(reshaped.getValue(), op.getValue());
}

TEST(ReshapeToStaticDimensions, StaticShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::MlirOp op =
      MakeConstant(mb, 1.0f, mb.getOpBuilder().getF32Type(), {2, 3});
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp reshaped,
                          ReshapeToStaticDimensions(op, {6}));
  EXPECT_THAT(GetTensorTypeOrDie(reshaped).getShape(), ElementsAre(6));
}

TEST(ReshapeToStaticDimensions, CollapsePropagatesBoundedDim) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::func::FunctionBuilder fb(op_builder_utils_builder.get(), "main");
  mlir::MlirOp op = MakeBoundedArg(fb, {2, 13, 5}, /*bound_dims=*/{1});
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp reshaped,
                          ReshapeToStaticDimensions(op, {26, 5}));
  const mlir::stablehlo::Dimensions dims = GetDimensions(reshaped);
  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].size, 26);
  EXPECT_TRUE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 5);
  EXPECT_FALSE(dims[1].boundOp.has_value());
}

TEST(ReshapeToStaticDimensions, ErrorExpandBoundedDimToMultiple) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::func::FunctionBuilder fb(op_builder_utils_builder.get(), "main");
  mlir::MlirOp op = MakeBoundedArg(fb, {10, 6}, /*bound_dims=*/{0});
  absl::StatusOr<mlir::MlirOp> reshaped =
      ReshapeToStaticDimensions(op, {2, 5, 6});
  ASSERT_FALSE(reshaped.ok());
  EXPECT_EQ(reshaped.status().code(), error::kInvalidArgument);
}

// Returns the stablehlo.iota op that produces `op`, looking through any
// set_dimension_size ops applied to it.
mlir::stablehlo::IotaOp GetUnderlyingIota(mlir::MlirOp op) {
  mlir::Value value = op.getValue();
  while (auto set_size =
             value.getDefiningOp<mlir::stablehlo::SetDimensionSizeOp>()) {
    value = set_size.getOperand();
  }
  return value.getDefiningOp<mlir::stablehlo::IotaOp>();
}

TEST(Iota, StaticShapeUsesReferenceElementTypeByDefault) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::MlirOp reference =
      MakeConstant(mb, 1.0f, mb.getOpBuilder().getF32Type(), {2, 3, 4});

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp iota,
                          Iota(reference, /*iota_dimension=*/1));

  const mlir::RankedTensorType type = GetTensorTypeOrDie(iota);
  EXPECT_TRUE(type.hasStaticShape());
  EXPECT_THAT(type.getShape(), ElementsAre(2, 3, 4));
  EXPECT_TRUE(type.getElementType().isF32());
  mlir::stablehlo::IotaOp iota_op =
      iota.getValue().getDefiningOp<mlir::stablehlo::IotaOp>();
  ASSERT_TRUE(iota_op);
  EXPECT_EQ(iota_op.getIotaDimension(), 1);
}

TEST(Iota, StaticShapeUsesRequestedElementType) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::MlirOp reference =
      MakeConstant(mb, 1.0f, mb.getOpBuilder().getF32Type(), {2, 3});

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp iota,
      Iota(reference, /*iota_dimension=*/0, mb.getOpBuilder().getI32Type()));

  const mlir::RankedTensorType type = GetTensorTypeOrDie(iota);
  EXPECT_THAT(type.getShape(), ElementsAre(2, 3));
  EXPECT_TRUE(type.getElementType().isSignlessInteger(32));
}

TEST(Iota, BoundedDimsArePropagated) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp reference = MakeBoundedArg(fb, {2, 13, 5}, /*bound_dims=*/{1});

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp iota,
      Iota(reference, /*iota_dimension=*/2, fb.getOpBuilder().getI32Type()));
  mlir::func::Return(fb, iota);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  // The result is bounded in the same dimension as the reference, with the
  // reference's upper bound.
  const mlir::RankedTensorType type = GetTensorTypeOrDie(iota);
  EXPECT_FALSE(type.isDynamicDim(0));
  EXPECT_TRUE(type.isDynamicDim(1));
  EXPECT_FALSE(type.isDynamicDim(2));
  EXPECT_TRUE(type.getElementType().isSignlessInteger(32));
  EXPECT_THAT(GetStaticShape(iota), ElementsAre(2, 13, 5));
  const mlir::stablehlo::Dimensions dims = GetDimensions(iota);
  ASSERT_EQ(dims.size(), 3);
  EXPECT_FALSE(dims[0].boundOp.has_value());
  EXPECT_TRUE(dims[1].boundOp.has_value());
  EXPECT_FALSE(dims[2].boundOp.has_value());

  // The iota itself is static at the bound, and the runtime size is applied
  // with set_dimension_size.
  auto set_size =
      iota.getValue().getDefiningOp<mlir::stablehlo::SetDimensionSizeOp>();
  ASSERT_TRUE(set_size) << DebugString(module.get());
  EXPECT_EQ(set_size.getDimension(), 1);
  mlir::stablehlo::IotaOp iota_op = GetUnderlyingIota(iota);
  ASSERT_TRUE(iota_op) << DebugString(module.get());
  EXPECT_TRUE(iota_op.getType().hasStaticShape());
  EXPECT_THAT(iota_op.getType().getShape(), ElementsAre(2, 13, 5));
  EXPECT_EQ(iota_op.getIotaDimension(), 2);
}

TEST(Iota, MultipleBoundedDimsArePropagated) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp reference =
      MakeBoundedArg(fb, {4, 8, 16}, /*bound_dims=*/{0, 2});

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp iota,
                          Iota(reference, /*iota_dimension=*/0));
  mlir::func::Return(fb, iota);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  const mlir::RankedTensorType type = GetTensorTypeOrDie(iota);
  EXPECT_TRUE(type.isDynamicDim(0));
  EXPECT_FALSE(type.isDynamicDim(1));
  EXPECT_TRUE(type.isDynamicDim(2));
  EXPECT_THAT(GetStaticShape(iota), ElementsAre(4, 8, 16));
  int num_set_sizes = 0;
  module->walk([&](mlir::stablehlo::SetDimensionSizeOp) { ++num_set_sizes; });
  // Two from MakeBoundedArg on the reference, two from Iota.
  EXPECT_EQ(num_set_sizes, 4);
}

TEST(Iota, NegativeDimensionCountsFromTheEnd) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::MlirOp reference =
      MakeConstant(mb, 1.0f, mb.getOpBuilder().getF32Type(), {2, 3, 4});

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp last,
                          Iota(reference, /*iota_dimension=*/-1));
  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp first,
                          Iota(reference, /*iota_dimension=*/-3));

  mlir::stablehlo::IotaOp last_op = GetUnderlyingIota(last);
  mlir::stablehlo::IotaOp first_op = GetUnderlyingIota(first);
  ASSERT_TRUE(last_op);
  ASSERT_TRUE(first_op);
  EXPECT_EQ(last_op.getIotaDimension(), 2);
  EXPECT_EQ(first_op.getIotaDimension(), 0);
}

TEST(Iota, NegativeDimensionWithBoundedDims) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::func::FunctionBuilder fb(op_builder_utils_builder.get(), "main");
  mlir::MlirOp reference = MakeBoundedArg(fb, {2, 13, 5}, /*bound_dims=*/{1});

  TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp iota,
                          Iota(reference, /*iota_dimension=*/-2));

  mlir::stablehlo::IotaOp iota_op = GetUnderlyingIota(iota);
  ASSERT_TRUE(iota_op);
  EXPECT_EQ(iota_op.getIotaDimension(), 1);
  EXPECT_TRUE(GetTensorTypeOrDie(iota).isDynamicDim(1));
}

class IotaOutOfBoundsTest : public testing::TestWithParam<int64_t> {};

TEST_P(IotaOutOfBoundsTest, ReturnsInvalidArgument) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::MlirOp reference =
      MakeConstant(mb, 1.0f, mb.getOpBuilder().getF32Type(), {2, 3, 4});

  absl::StatusOr<mlir::MlirOp> iota = Iota(reference, GetParam());

  ASSERT_FALSE(iota.ok());
  EXPECT_EQ(iota.status().code(), error::kInvalidArgument);
  EXPECT_EQ(iota.status().message(), absl::StrCat("iota_dimension ", GetParam(),
                                                  " out of bounds for rank 3"));
}

INSTANTIATE_TEST_SUITE_P(IotaOutOfBoundsTests, IotaOutOfBoundsTest,
                         testing::Values(3, 10, -4, -10));

// Helper for Broadcast tests.
struct BroadcastTestResult {
  absl::StatusOr<mlir::MlirOp> broadcasted_op;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};
BroadcastTestResult BuildBroadcastGraph(
    Dimensions input_shape, Dimensions output_shape, Dimensions bcast_dims,
    OpBuilderUtilsBuilder& op_builder_utils_builder) {
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::OpBuilder& op_builder = fb.getOpBuilder();

  // Make op of shape [input_shape]
  auto arg = mlir::func::Argument(
      fb, mlir::RankedTensorType::get(input_shape, op_builder.getF32Type()));

  auto broadcasted = Broadcast(arg, output_shape, bcast_dims);
  if (broadcasted.ok()) {
    mlir::func::Return(fb, *broadcasted);
  }
  return {broadcasted, mb.build()};
}

struct BroadcastParams {
  std::string test_name;
  Dimensions input_shape;
  Dimensions output_shape;
  Dimensions bcast_dims;
  Dimensions expected_output_shape;
};

class BroadcastTest : public testing::TestWithParam<BroadcastParams> {};

TEST_P(BroadcastTest, ValidBroadcast) {
  const auto& params = GetParam();
  OpBuilderUtilsBuilder op_builder_utils_builder;
  BroadcastTestResult result =
      BuildBroadcastGraph(params.input_shape, params.output_shape,
                          params.bcast_dims, op_builder_utils_builder);
  TT_ASSERT_OK_AND_ASSIGN(auto broadcasted_op, result.broadcasted_op);

  auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(broadcasted_op.getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(),
              testing::ElementsAreArray(params.expected_output_shape));
  std::cout << "module: " << DebugString(result.module.get()) << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastTests, BroadcastTest,
    testing::ValuesIn<BroadcastParams>({
        {"Scalar",
         /*input_shape=*/{},
         /*output_shape=*/{2, 3},
         /*bcast_dims=*/{},
         /*expected_output_shape=*/{2, 3}},
        {"Dim3To2x3",
         /*input_shape=*/{3},
         /*output_shape=*/{2, 3},
         /*bcast_dims=*/{1},
         /*expected_output_shape=*/{2, 3}},
        {"1x3To2x3",
         /*input_shape=*/{1, 3},
         /*output_shape=*/{2, 3},
         /*bcast_dims=*/{0, 1},
         /*expected_output_shape=*/{2, 3}},
        {"8x10x128_To_1x8x2x10x128",
         /*input_shape=*/{8, 10, 128},
         /*output_shape=*/{1, 8, 2, 10, 128},
         /*bcast_dims=*/{1, 3, 4},
         /*expected_output_shape=*/{1, 8, 2, 10, 128}},
    }),
    [](const testing::TestParamInfo<BroadcastTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(BroadcastTest, InvalidRank) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  BroadcastTestResult result =
      BuildBroadcastGraph(/*input_shape=*/{2, 1}, /*output_shape=*/{2},
                          /*bcast_dims=*/{}, op_builder_utils_builder);
  ASSERT_FALSE(result.broadcasted_op.ok());
  EXPECT_EQ(result.broadcasted_op.status().code(), error::kInvalidArgument);
  EXPECT_THAT(result.broadcasted_op.status().message(),
              testing::HasSubstr("must not be more than output rank"));
}

// Helper for DynamicBroadcast tests.
struct DynamicBroadcastTestResult {
  absl::StatusOr<mlir::MlirOp> broadcasted_op;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};
DynamicBroadcastTestResult BuildDynamicBroadcastGraph(
    Dimensions bounded_input_shape, absl::Span<const int64_t> bound_dims,
    Dimensions output_shape, Dimensions bcast_dims,
    OpBuilderUtilsBuilder& op_builder_utils_builder) {
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::OpBuilder& op_builder = fb.getOpBuilder();

  // Make op of shape [bounded_input_shape]
  auto arg = mlir::func::Argument(
      fb, mlir::RankedTensorType::get(bounded_input_shape,
                                      op_builder.getF32Type()));
  auto bounded_op = arg;
  for (int64_t bound_dim : bound_dims) {
    auto size = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({}, op_builder.getI32Type()));
    bounded_op =
        mlir::stablehlo::SetDimensionSize(bounded_op, size, /*dim=*/bound_dim);
  }

  auto broadcasted = Broadcast(bounded_op, output_shape, bcast_dims);
  if (broadcasted.ok()) {
    mlir::func::Return(fb, *broadcasted);
  }
  return {broadcasted, mb.build()};
}

struct DynamicBroadcastParams {
  std::string test_name;
  Dimensions bounded_input_shape;
  Dimensions bound_dims;
  Dimensions output_shape;
  Dimensions bcast_dims;
  Dimensions expected_bounded_output_shape;
  std::vector<bool> expected_output_bounded;
};

class DynamicBroadcastTest
    : public testing::TestWithParam<DynamicBroadcastParams> {};

TEST_P(DynamicBroadcastTest, ValidBroadcast) {
  const auto& params = GetParam();
  OpBuilderUtilsBuilder op_builder_utils_builder;
  DynamicBroadcastTestResult result = BuildDynamicBroadcastGraph(
      params.bounded_input_shape, params.bound_dims, params.output_shape,
      params.bcast_dims, op_builder_utils_builder);
  TT_ASSERT_OK_AND_ASSIGN(auto broadcasted_op, result.broadcasted_op);

  // Check that the broadcasted op has proper bounded dynamic dimensions.
  mlir::stablehlo::Dimensions broadcasted_dims = GetDimensions(broadcasted_op);
  for (int i = 0; i < params.expected_bounded_output_shape.size(); ++i) {
    ASSERT_EQ(broadcasted_dims[i].size,
              params.expected_bounded_output_shape[i]);
    ASSERT_EQ(broadcasted_dims[i].boundOp.has_value(),
              params.expected_output_bounded[i]);
  }
  std::cout << "module: " << DebugString(result.module.get()) << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    DynamicBroadcastTests, DynamicBroadcastTest,
    testing::ValuesIn<DynamicBroadcastParams>({
        {"Bcast10to2x10",  // ? -> 2x?
         /*bounded_input_shape=*/{10},
         /*bound_dims=*/{0},
         /*output_shape=*/{2, 10},
         /*bcast_dims=*/{1},
         /*expected_bounded_output_shape=*/{2, 10},
         /*expected_output_bounded=*/{false, true}},
        {"Bcast1x10to2x10",  // ?x10 -> 2x10
         /*bounded_input_shape=*/{1, 10},
         /*bound_dims=*/{1},
         /*output_shape=*/{2, 10},
         /*bcast_dims=*/{0, 1},
         /*expected_bounded_output_shape=*/{2, 10},
         /*expected_output_bounded=*/{false, true}},
        {"Bcast10x1to10x5",  // 10x? -> 10x5
         /*bounded_input_shape=*/{10, 1},
         /*bound_dims=*/{0},
         /*output_shape=*/{10, 5},
         /*bcast_dims=*/{0, 1},
         /*expected_bounded_output_shape=*/{10, 5},
         /*expected_output_bounded=*/{true, false}},
        {"Bcast8x10x128to1x8x2x10x3x128",  // 8x?x128 -> 1x8x2x?x3x128
         /*bounded_input_shape=*/{8, 100, 128},
         /*bound_dims=*/{1},
         /*output_shape=*/{1, 8, 2, 10, 3, 128},
         /*bcast_dims=*/{1, 3, 5},
         /*expected_bounded_output_shape=*/{1, 8, 2, 100, 3, 128},
         /*expected_output_bounded=*/{false, false, false, true, false, false}},
    }),
    [](const testing::TestParamInfo<DynamicBroadcastTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(OpBuilderUtils, SerializeBytecode) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::func::Return(fb, {});
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();

  TT_ASSERT_OK_AND_ASSIGN(std::string bytecode,
                          SerializeBytecode(module.get()));

  // Check for the MLIR magic string that denotes bytecode.
  EXPECT_THAT(bytecode, testing::HasSubstr("\x4D\x4C\xEF\x52"));
}

TEST(OpBuilderUtils, SerializePortableArtifact) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::func::Return(fb, {});
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();

  TT_ASSERT_OK_AND_ASSIGN(std::string artifact,
                          SerializePortableArtifact(module.get()));

  // Check for the MLIR magic string that denotes bytecode.
  EXPECT_THAT(artifact, testing::HasSubstr("\x4D\x4C\xEF\x52"));
  // Check for the StableHLO_v1 producer string to indicate versioned StableHLO.
  EXPECT_THAT(artifact, testing::HasSubstr("StableHLO_v1."));
}

TEST(OpBuilderUtils, SerializePortableArtifact_CustomCall_NoVhloCustomCallV2) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::OpBuilder& op_builder = fb.getOpBuilder();

  auto tensor_type = mlir::RankedTensorType::get({10}, op_builder.getF32Type());
  const mlir::NamedAttribute call_target = op_builder.getNamedAttr(
      "call_target_name", op_builder.getStringAttr("custom_call_target"));
  const mlir::NamedAttribute has_side_effect =
      op_builder.getNamedAttr("has_side_effect", op_builder.getBoolAttr(false));
  const mlir::NamedAttribute api_version = op_builder.getNamedAttr(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          op_builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));

  auto custom_call_op = mlir::stablehlo::CustomCallOp::create(
      op_builder, fb.getLoc(), {tensor_type}, mlir::ValueRange{},
      {call_target, has_side_effect, api_version});
  mlir::MlirOp custom_call_result(fb, custom_call_op.getResult(0));
  mlir::func::Return(fb, custom_call_result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();

  TT_ASSERT_OK_AND_ASSIGN(std::string artifact,
                          SerializePortableArtifact(module.get()));

  // Check for the MLIR magic string that denotes bytecode.
  EXPECT_THAT(artifact, testing::HasSubstr("\x4D\x4C\xEF\x52"));
  // Check for the StableHLO_v1 producer string to indicate versioned StableHLO.
  EXPECT_THAT(artifact, testing::HasSubstr("StableHLO_v1."));

  // Verify that WEEK_12 compatibility requirement serializes to
  // vhlo.custom_call_v1 and avoids emitting vhlo.custom_call_v2, which causes
  // deserialization failures in StableHLO 1.17.0 (jaxlib 0.9.2).
  EXPECT_THAT(artifact, testing::Not(testing::HasSubstr("custom_call_v2")));
  EXPECT_THAT(artifact, testing::HasSubstr("custom_call_v1"));
}

TEST(OpBuilderUtils, MakeZeroSizedTensor_MlirType_DefaultShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  TT_ASSERT_OK_AND_ASSIGN(
      auto op, MakeZeroSizedTensor(builder, op_builder.getF32Type()));
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.getElementType().isF32());
  EXPECT_THAT(type.getShape(), ElementsAre(0));
}

TEST(OpBuilderUtils, MakeZeroSizedTensor_MlirType_CustomShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  TT_ASSERT_OK_AND_ASSIGN(
      auto op,
      MakeZeroSizedTensor(builder, op_builder.getI32Type(), {2, 0, 3}));
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.getElementType().isSignlessInteger(32));
  EXPECT_THAT(type.getShape(), ElementsAre(2, 0, 3));
}

TEST(OpBuilderUtils, MakeZeroSizedTensor_MlirType_InvalidShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  auto op_or = MakeZeroSizedTensor(builder, op_builder.getI32Type(), {2, 1, 3});
  EXPECT_FALSE(op_or.ok());
  EXPECT_EQ(op_or.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      op_or.status().message(),
      testing::HasSubstr("must contain at least one dimension of size 0"));
}

TEST(OpBuilderUtils, MakeZeroSizedTensor_MlirElementType_DefaultShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();

  TT_ASSERT_OK_AND_ASSIGN(auto op,
                          MakeZeroSizedTensor(builder, mlir::ElementType::F32));
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.getElementType().isF32());
  EXPECT_THAT(type.getShape(), ElementsAre(0));
}

TEST(OpBuilderUtils, MakeZeroSizedTensor_MlirElementType_CustomShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();

  TT_ASSERT_OK_AND_ASSIGN(
      auto op, MakeZeroSizedTensor(builder, mlir::ElementType::I32, {2, 0, 3}));
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.getElementType().isSignlessInteger(32));
  EXPECT_THAT(type.getShape(), ElementsAre(2, 0, 3));
}

TEST(OpBuilderUtils, MakeZeroSizedTensor_MlirElementType_InvalidShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();

  auto op_or = MakeZeroSizedTensor(builder, mlir::ElementType::I32, {2, 1, 3});
  EXPECT_FALSE(op_or.ok());
  EXPECT_EQ(op_or.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      op_or.status().message(),
      testing::HasSubstr("must contain at least one dimension of size 0"));
}

TEST(OpBuilderUtils, GetBatchDimensions) {
  EXPECT_THAT(GetBatchDimensions(0, 0), ElementsAre());
  EXPECT_THAT(GetBatchDimensions(1, 0), ElementsAre());
  EXPECT_THAT(GetBatchDimensions(2, 0), ElementsAre(1));
  EXPECT_THAT(GetBatchDimensions(2, 1), ElementsAre(0));
  EXPECT_THAT(GetBatchDimensions(3, 1), ElementsAre(0, 2));
  EXPECT_THAT(GetBatchDimensions(4, 2), ElementsAre(0, 1, 3));
}

TEST(OpBuilderUtils, SliceBatchDimensions_NoSliceNeeded) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  const mlir::MlirOp self =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {3, 4});
  const mlir::RankedTensorType index_type =
      mlir::RankedTensorType::get({3, 2}, op_builder.getI64Type());

  const mlir::MlirOp sliced = SliceBatchDimensions(self, /*dim=*/1, index_type);
  const mlir::RankedTensorType sliced_type = GetTensorTypeOrDie(sliced);
  EXPECT_THAT(sliced_type.getShape(), ElementsAre(3, 4));
}

TEST(OpBuilderUtils, SliceBatchDimensions_SliceNeeded) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  const mlir::MlirOp self =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {4, 5});
  const mlir::RankedTensorType index_type =
      mlir::RankedTensorType::get({2, 3}, op_builder.getI64Type());

  const mlir::MlirOp sliced = SliceBatchDimensions(self, /*dim=*/1, index_type);
  const mlir::RankedTensorType sliced_type = GetTensorTypeOrDie(sliced);
  EXPECT_THAT(sliced_type.getShape(), ElementsAre(2, 5));
}

TEST(OpBuilderUtils, SliceBatchDimensions_ZeroDimBatch) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  const mlir::MlirOp self =
      MakeConstant(builder, 1.0f, op_builder.getF32Type(), {3, 5});
  const mlir::RankedTensorType index_type =
      mlir::RankedTensorType::get({0, 2}, op_builder.getI64Type());

  const mlir::MlirOp sliced = SliceBatchDimensions(self, /*dim=*/1, index_type);
  const mlir::RankedTensorType sliced_type = GetTensorTypeOrDie(sliced);
  EXPECT_THAT(sliced_type.getShape(), ElementsAre(0, 5));
}

TEST(OpBuilderUtils, Squeeze_PreservesBoundedDynamism) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::OpBuilder& op_builder = fb.getOpBuilder();

  // Create bounded input tensor of shape [1, 2, ?] with bound 10 on dim 2:
  // tensor<1x2x?xf32, #stablehlo.type_extensions<bounds = [?, ?, 10]>>
  const auto bounds_attr = mlir::DenseI64ArrayAttr::get(
      op_builder.getContext(),
      {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, int64_t{10}});
  const auto encoding = mlir::stablehlo::TypeExtensionsAttr::get(
      op_builder.getContext(), bounds_attr);
  const auto input_type = mlir::RankedTensorType::get(
      {1, 2, mlir::ShapedType::kDynamic}, op_builder.getF32Type(), encoding);
  const auto input = mlir::func::Argument(fb, input_type);

  // Squeeze static dimension 0.
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp squeezed, Squeeze(input, {0}));

  const mlir::RankedTensorType result_type = GetTensorTypeOrDie(squeezed);
  EXPECT_EQ(result_type.getRank(), 2);
  EXPECT_TRUE(result_type.isDynamicDim(1));

  // Verify that the bounds attribute is preserved on the squeezed output.
  const auto result_encoding =
      mlir::dyn_cast_or_null<mlir::stablehlo::TypeExtensionsAttr>(
          result_type.getEncoding());
  ASSERT_NE(result_encoding, nullptr);
  EXPECT_THAT(result_encoding.getBounds(),
              ElementsAre(mlir::ShapedType::kDynamic, int64_t{10}));

  // Verify that GetDimensions sees the bounded dimension with bound 10.
  const mlir::stablehlo::Dimensions dims = GetDimensions(squeezed);
  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].size, 2);
  EXPECT_FALSE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 10);
  EXPECT_TRUE(dims[1].boundOp.has_value());
}

// Returns an f32 function argument of the given shape, where `bounded_dims`
// are made bounded dynamic by set_dimension_size with runtime size arguments.
mlir::MlirOp MakeBoundedArgument(mlir::func::FunctionBuilder& fb,
                                 const Dimensions& shape,
                                 const Dimensions& bounded_dims) {
  mlir::OpBuilder& op_builder = fb.getOpBuilder();
  mlir::MlirOp arg = mlir::func::Argument(
      fb, mlir::RankedTensorType::get(shape, op_builder.getF32Type()));
  for (int64_t dim : bounded_dims) {
    mlir::MlirOp size = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({}, op_builder.getI32Type()));
    arg = mlir::stablehlo::SetDimensionSize(arg, size, dim);
  }
  return arg;
}

// A convolution of operands and result with configurable dimension numbers and
// optional result_dims. Operand shapes are the bounds of their bounded
// dimensions.
struct ConvolutionLikeParams {
  std::string test_name;
  Dimensions lhs_shape;
  Dimensions lhs_bounded_dims;
  Dimensions rhs_shape;
  Dimensions rhs_bounded_dims;
  int64_t input_batch_dim = 0;
  int64_t input_feature_dim = 1;
  int64_t kernel_input_feature_dim = 1;
  int64_t kernel_output_feature_dim = 0;
  int64_t output_batch_dim = 0;
  int64_t output_feature_dim = 1;
  // Window attributes, absent if empty.
  Dimensions strides;
  // Low and high padding of each spatial dimension.
  Dimensions padding;
  Dimensions lhs_dilation;
  Dimensions rhs_dilation;
  bool window_reversal = false;
  int64_t feature_group_count = 1;
  int64_t batch_group_count = 1;
  // Optional result_dims passed to ConvolutionLike.
  Dimensions result_shape;
  Dimensions result_bounded_dims;
  // Expected result shape, with bounds for bounded dimensions.
  Dimensions expected_result_shape;
  std::vector<bool> expected_result_bounded;
  // Expected substring of the error message of unsupported bounds.
  std::string expected_error;
};

absl::StatusOr<mlir::MlirOp> CallConvolutionLike(
    mlir::func::FunctionBuilder& fb, const ConvolutionLikeParams& params,
    mlir::RankedTensorType result_type, mlir::MlirOp lhs, mlir::MlirOp rhs) {
  mlir::MlirBuilder& builder = lhs.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();
  const int64_t num_spatial_dims = params.lhs_shape.size() - 2;
  Dimensions spatial_dims;
  for (int64_t i = 0; i < num_spatial_dims; ++i) {
    spatial_dims.push_back(i + 2);
  }
  const auto dimension_numbers = mlir::stablehlo::ConvDimensionNumbersAttr::get(
      &ctx, params.input_batch_dim, params.input_feature_dim, spatial_dims,
      params.kernel_input_feature_dim, params.kernel_output_feature_dim,
      spatial_dims, params.output_batch_dim, params.output_feature_dim,
      spatial_dims);

  auto array_attr = [&](const Dimensions& values) {
    return values.empty() ? mlir::DenseI64ArrayAttr()
                          : mlir::DenseI64ArrayAttr::get(&ctx, values);
  };
  mlir::DenseIntElementsAttr padding;
  if (!params.padding.empty()) {
    padding = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get({num_spatial_dims, 2},
                                    builder.getOpBuilder().getI64Type()),
        mlir::ArrayRef<int64_t>(params.padding));
  }
  mlir::DenseBoolArrayAttr window_reversal;
  if (params.window_reversal) {
    window_reversal = mlir::DenseBoolArrayAttr::get(
        &ctx, llvm::SmallVector<bool>(num_spatial_dims, true));
  }
  std::optional<mlir::stablehlo::Dimensions> result_dims;
  if (!params.result_shape.empty()) {
    result_dims = GetDimensions(MakeBoundedArgument(
        fb, params.result_shape, params.result_bounded_dims));
  }
  return ConvolutionLike(
      result_type, lhs, rhs, dimension_numbers, params.feature_group_count,
      params.batch_group_count, array_attr(params.strides), padding,
      array_attr(params.lhs_dilation), array_attr(params.rhs_dilation),
      window_reversal, /*precision_config=*/mlir::ArrayAttr(), result_dims);
}

TEST(ConvolutionLike, StaticOperandsUseResultType) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  const ConvolutionLikeParams params = {
      .lhs_shape = {1, 4, 8},
      .rhs_shape = {4, 1, 3},
      .padding = {2, 2},
      .feature_group_count = 4,
  };
  const mlir::MlirOp lhs = MakeBoundedArgument(fb, params.lhs_shape, {});
  const mlir::MlirOp rhs = MakeBoundedArgument(fb, params.rhs_shape, {});
  const auto result_type =
      mlir::RankedTensorType::get({1, 4, 10}, fb.getOpBuilder().getF32Type());

  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp result,
      CallConvolutionLike(fb, params, result_type, lhs, rhs));

  EXPECT_TRUE(mlir::isa<mlir::stablehlo::ConvolutionOp>(
      result.getValue().getDefiningOp()));
  EXPECT_EQ(result.getType(), result_type);
}

TEST(ConvolutionLike, StaticOperandsUseStaticResultDims) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  const ConvolutionLikeParams params = {
      .lhs_shape = {1, 4, 8},
      .rhs_shape = {4, 1, 3},
      .padding = {2, 2},
      .feature_group_count = 4,
      .result_shape = {1, 4, 10},
  };
  const mlir::MlirOp lhs = MakeBoundedArgument(fb, params.lhs_shape, {});
  const mlir::MlirOp rhs = MakeBoundedArgument(fb, params.rhs_shape, {});
  const auto placeholder_result_type =
      mlir::RankedTensorType::get({1, 1, 1}, fb.getOpBuilder().getF32Type());

  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp result,
      CallConvolutionLike(fb, params, placeholder_result_type, lhs, rhs));

  EXPECT_TRUE(mlir::isa<mlir::stablehlo::ConvolutionOp>(
      result.getValue().getDefiningOp()));
  EXPECT_EQ(result.getType(), mlir::RankedTensorType::get(
                                  {1, 4, 10}, fb.getOpBuilder().getF32Type()));
}

class ConvolutionLikeBoundedTest
    : public testing::TestWithParam<ConvolutionLikeParams> {};

TEST_P(ConvolutionLikeBoundedTest, BoundsResultOfStaticConvolution) {
  const ConvolutionLikeParams& params = GetParam();
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  const mlir::MlirOp lhs =
      MakeBoundedArgument(fb, params.lhs_shape, params.lhs_bounded_dims);
  const mlir::MlirOp rhs =
      MakeBoundedArgument(fb, params.rhs_shape, params.rhs_bounded_dims);
  // Only the element type of the result type is used for bounded operands.
  const auto result_type = mlir::RankedTensorType::get(
      Dimensions(params.lhs_shape.size(), 1), fb.getOpBuilder().getF32Type());

  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp result,
      CallConvolutionLike(fb, params, result_type, lhs, rhs));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  // XLA only sees a static convolution.
  int num_convolutions = 0;
  module->walk([&](mlir::stablehlo::ConvolutionOp convolution) {
    ++num_convolutions;
    EXPECT_TRUE(convolution.getLhs().getType().hasStaticShape())
        << DebugString(convolution);
    EXPECT_TRUE(convolution.getRhs().getType().hasStaticShape())
        << DebugString(convolution);
    EXPECT_TRUE(convolution.getType().hasStaticShape())
        << DebugString(convolution);
  });
  EXPECT_EQ(num_convolutions, 1);

  EXPECT_EQ(GetElementTypeOrSelf(result), fb.getOpBuilder().getF32Type());
  const mlir::stablehlo::Dimensions result_dims = GetDimensions(result);
  ASSERT_EQ(result_dims.size(), params.expected_result_shape.size());
  for (int i = 0; i < result_dims.size(); ++i) {
    EXPECT_EQ(result_dims[i].size, params.expected_result_shape[i])
        << "dimension " << i;
    EXPECT_EQ(result_dims[i].boundOp.has_value(),
              params.expected_result_bounded[i])
        << "dimension " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    ConvolutionLikeBoundedTests, ConvolutionLikeBoundedTest,
    testing::ValuesIn<ConvolutionLikeParams>({
        {
            // Causal depthwise convolution of LFM2 and Mamba models.
            .test_name = "CausalDepthwiseBoundedLength",
            .lhs_shape = {1, 4, 68},
            .lhs_bounded_dims = {2},
            .rhs_shape = {4, 1, 3},
            .padding = {2, 2},
            .feature_group_count = 4,
            .expected_result_shape = {1, 4, 70},
            .expected_result_bounded = {false, false, true},
        },
        {
            .test_name = "StridedBoundedBatch",
            .lhs_shape = {8, 3, 10},
            .lhs_bounded_dims = {0},
            .rhs_shape = {5, 3, 3},
            .strides = {2},
            .padding = {1, 1},
            .expected_result_shape = {8, 5, 5},
            .expected_result_bounded = {true, false, false},
        },
        {
            .test_name = "StridedBoundedBatchAndLength",
            .lhs_shape = {4, 2, 9},
            .lhs_bounded_dims = {0, 2},
            .rhs_shape = {3, 2, 2},
            .strides = {3},
            .expected_result_shape = {4, 3, 3},
            .expected_result_bounded = {true, false, true},
        },
        {
            .test_name = "StridedDilatedBoundedLength",
            .lhs_shape = {2, 3, 17},
            .lhs_bounded_dims = {2},
            .rhs_shape = {4, 3, 3},
            .strides = {2},
            .padding = {1, 0},
            .rhs_dilation = {2},
            .expected_result_shape = {2, 4, 7},
            .expected_result_bounded = {false, false, true},
        },
        {
            .test_name = "NegativePaddingBoundedLength",
            .lhs_shape = {1, 1, 10},
            .lhs_bounded_dims = {2},
            .rhs_shape = {1, 1, 3},
            .padding = {-2, 1},
            .expected_result_shape = {1, 1, 7},
            .expected_result_bounded = {false, false, true},
        },
        {
            // Transposed convolution with stride 3.
            .test_name = "LhsDilatedBoundedLength",
            .lhs_shape = {1, 2, 10},
            .lhs_bounded_dims = {2},
            .rhs_shape = {5, 2, 4},
            .padding = {2, 3},
            .lhs_dilation = {3},
            .window_reversal = true,
            .expected_result_shape = {1, 5, 30},
            .expected_result_bounded = {false, false, true},
        },
        {
            .test_name = "BoundedHeightAndWidth",
            .lhs_shape = {1, 2, 6, 7},
            .lhs_bounded_dims = {2, 3},
            .rhs_shape = {3, 2, 3, 3},
            .padding = {1, 1, 1, 1},
            .expected_result_shape = {1, 3, 6, 7},
            .expected_result_bounded = {false, false, true, true},
        },
        {
            // Backward w.r.t. input of a strided dilated convolution.
            .test_name = "BackwardInputWithResultDims",
            .lhs_shape = {2, 4, 8},
            .lhs_bounded_dims = {2},
            .rhs_shape = {4, 3, 3},
            .kernel_input_feature_dim = 0,
            .kernel_output_feature_dim = 1,
            .padding = {3, 3},
            .lhs_dilation = {2},
            .rhs_dilation = {2},
            .window_reversal = true,
            .result_shape = {2, 3, 17},
            .result_bounded_dims = {2},
            .expected_result_shape = {2, 3, 17},
            .expected_result_bounded = {false, false, true},
        },
        {
            // Backward w.r.t. weight of a grouped strided convolution with
            // bounded spatial dimensions on both operands.
            .test_name = "BackwardWeightBoundedSpatial",
            .lhs_shape = {2, 4, 17},
            .lhs_bounded_dims = {2},
            .rhs_shape = {2, 6, 8},
            .rhs_bounded_dims = {2},
            .input_batch_dim = 1,
            .input_feature_dim = 0,
            .kernel_input_feature_dim = 0,
            .kernel_output_feature_dim = 1,
            .output_batch_dim = 1,
            .output_feature_dim = 0,
            .strides = {2},
            .padding = {1, 1},
            .rhs_dilation = {2},
            .batch_group_count = 2,
            .result_shape = {6, 2, 3},
            .expected_result_shape = {6, 2, 3},
            .expected_result_bounded = {false, false, false},
        },
        {
            // Backward w.r.t. weight with bounded batch (contracting feature)
            // dimension on both operands.
            .test_name = "BackwardWeightBoundedBatch",
            .lhs_shape = {8, 4, 10},
            .lhs_bounded_dims = {0},
            .rhs_shape = {8, 6, 5},
            .rhs_bounded_dims = {0},
            .input_batch_dim = 1,
            .input_feature_dim = 0,
            .kernel_input_feature_dim = 0,
            .kernel_output_feature_dim = 1,
            .output_batch_dim = 1,
            .output_feature_dim = 0,
            .rhs_dilation = {2},
            .batch_group_count = 2,
            .result_shape = {6, 2, 2},
            .expected_result_shape = {6, 2, 2},
            .expected_result_bounded = {false, false, false},
        },
        {
            // Backward w.r.t. input of a strided dilated convolution whose
            // grad_output bound (32) is looser than the one implied by the
            // input bound (17), which needs a strongly negative high padding.
            .test_name = "BackwardInputLooseBoundNegativePadding",
            .lhs_shape = {2, 4, 32},
            .lhs_bounded_dims = {2},
            .rhs_shape = {4, 3, 3},
            .kernel_input_feature_dim = 0,
            .kernel_output_feature_dim = 1,
            .padding = {1, -28},
            .lhs_dilation = {2},
            .rhs_dilation = {2},
            .window_reversal = true,
            .result_shape = {2, 3, 32},
            .result_bounded_dims = {2},
            .expected_result_shape = {2, 3, 32},
            .expected_result_bounded = {false, false, true},
        },
        {
            // Static operands only bounded through result_dims.
            .test_name = "StaticOperandsBoundedResultDims",
            .lhs_shape = {1, 4, 8},
            .rhs_shape = {4, 1, 3},
            .padding = {2, 2},
            .feature_group_count = 4,
            .result_shape = {1, 4, 10},
            .result_bounded_dims = {2},
            .expected_result_shape = {1, 4, 10},
            .expected_result_bounded = {false, false, true},
        },
    }),
    [](const testing::TestParamInfo<ConvolutionLikeBoundedTest::ParamType>&
           info) { return info.param.test_name; });

TEST(ConvolutionLike, MismatchedResultDimsReturnsNotImplementedError) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  const ConvolutionLikeParams params = {
      .lhs_shape = {1, 4, 8},
      .lhs_bounded_dims = {2},
      .rhs_shape = {4, 1, 3},
      .padding = {2, 2},
      .feature_group_count = 4,
      .result_shape = {1, 4, 9},
      .result_bounded_dims = {2},
  };
  const mlir::MlirOp lhs =
      MakeBoundedArgument(fb, params.lhs_shape, params.lhs_bounded_dims);
  const mlir::MlirOp rhs = MakeBoundedArgument(fb, params.rhs_shape, {});
  const auto result_type =
      mlir::RankedTensorType::get({1, 1, 1}, fb.getOpBuilder().getF32Type());

  const absl::StatusOr<mlir::MlirOp> result =
      CallConvolutionLike(fb, params, result_type, lhs, rhs);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kPythonNotImplementedError);
  EXPECT_THAT(
      result.status().message(),
      testing::HasSubstr(
          "expected the convolution result shape inferred at the upper bounds "
          "of the lhs operand of type tensor<1x4x?xf32, "
          "#stablehlo.bounds<?, ?, 8>> and the rhs operand of type "
          "tensor<4x1x3xf32> to be [1, 4, 9], got [1, 4, 10]; only the batch "
          "and spatial dimensions of the input (and of grad_output in "
          "convolution_backward) may be bounded dynamic"));
}

class ConvolutionLikeUnsupportedTest
    : public testing::TestWithParam<ConvolutionLikeParams> {};

TEST_P(ConvolutionLikeUnsupportedTest, ReturnsNotImplementedError) {
  const ConvolutionLikeParams& params = GetParam();
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  const mlir::MlirOp lhs =
      MakeBoundedArgument(fb, params.lhs_shape, params.lhs_bounded_dims);
  const mlir::MlirOp rhs =
      MakeBoundedArgument(fb, params.rhs_shape, params.rhs_bounded_dims);
  const auto result_type = mlir::RankedTensorType::get(
      Dimensions(params.lhs_shape.size(), 1), fb.getOpBuilder().getF32Type());

  const absl::StatusOr<mlir::MlirOp> result =
      CallConvolutionLike(fb, params, result_type, lhs, rhs);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kPythonNotImplementedError);
  EXPECT_THAT(
      result.status().message(),
      testing::AllOf(testing::HasSubstr(params.expected_error),
                     testing::EndsWith(
                         "; only the batch and spatial dimensions of the input "
                         "(and of grad_output in convolution_backward) may be "
                         "bounded dynamic")));
}

INSTANTIATE_TEST_SUITE_P(
    ConvolutionLikeUnsupportedTests, ConvolutionLikeUnsupportedTest,
    testing::ValuesIn<ConvolutionLikeParams>({
        {
            .test_name = "BoundedInputFeatureOnly",
            .lhs_shape = {1, 4, 8},
            .lhs_bounded_dims = {1},
            .rhs_shape = {4, 4, 3},
            .expected_error =
                "expected the contracted feature dimensions of a convolution "
                "to be both static or both bounded dynamic, got dimension 1 of "
                "the lhs operand of type tensor<1x?x8xf32, "
                "#stablehlo.bounds<?, 4, ?>> and dimension 1 of the rhs "
                "operand of type tensor<4x4x3xf32>",
        },
        {
            .test_name = "BoundedKernelInputFeatureOnly",
            .lhs_shape = {1, 4, 8},
            .rhs_shape = {4, 4, 3},
            .rhs_bounded_dims = {1},
            .expected_error =
                "expected the contracted feature dimensions of a convolution "
                "to be both static or both bounded dynamic, got dimension 1 of "
                "the lhs operand of type tensor<1x4x8xf32> and dimension 1 of "
                "the rhs operand of type tensor<4x?x3xf32, "
                "#stablehlo.bounds<?, 4, ?>>",
        },
        {
            .test_name = "BoundedMatchingFeatureWithFeatureGroups",
            .lhs_shape = {1, 4, 8},
            .lhs_bounded_dims = {1},
            .rhs_shape = {4, 2, 3},
            .rhs_bounded_dims = {1},
            .feature_group_count = 2,
            .expected_error =
                "expected the bounded dynamic contracted feature dimensions of "
                "a convolution to have equal upper bounds and "
                "feature_group_count 1, got upper bound 4 in dimension 1 of "
                "the lhs operand of type tensor<1x?x8xf32, "
                "#stablehlo.bounds<?, 4, ?>>, upper bound 2 in dimension 1 of "
                "the rhs operand of type tensor<4x?x3xf32, "
                "#stablehlo.bounds<?, 2, ?>> and feature_group_count 2",
        },
        {
            .test_name = "BoundedMatchingFeatureWithDifferentBounds",
            .lhs_shape = {1, 4, 8},
            .lhs_bounded_dims = {1},
            .rhs_shape = {4, 8, 3},
            .rhs_bounded_dims = {1},
            .expected_error =
                "got upper bound 4 in dimension 1 of the lhs operand of type "
                "tensor<1x?x8xf32, #stablehlo.bounds<?, 4, ?>>, upper bound 8 "
                "in dimension 1 of the rhs operand of type tensor<4x?x3xf32, "
                "#stablehlo.bounds<?, 8, ?>> and feature_group_count 1",
        },
        {
            .test_name = "BoundedKernelOutputFeature",
            .lhs_shape = {1, 4, 8},
            .rhs_shape = {4, 4, 3},
            .rhs_bounded_dims = {0},
            .expected_error =
                "expected a static size in the output feature dimension 0 of "
                "the rhs operand of a convolution, got rhs type "
                "tensor<?x4x3xf32, #stablehlo.bounds<4, ?, ?>>",
        },
        {
            .test_name = "BoundedKernelSpatialWithoutResultDims",
            .lhs_shape = {1, 4, 8},
            .rhs_shape = {4, 4, 3},
            .rhs_bounded_dims = {2},
            .expected_error =
                "expected static spatial dimensions of the rhs operand of a "
                "convolution, got a bounded dynamic size in dimension 2 of the "
                "rhs operand of type tensor<4x4x?xf32, "
                "#stablehlo.bounds<?, ?, 3>>",
        },
        {
            .test_name = "BoundedBatchWithBatchGroups",
            .lhs_shape = {8, 4, 8},
            .lhs_bounded_dims = {0},
            .rhs_shape = {4, 4, 3},
            .batch_group_count = 2,
            .expected_error =
                "expected a static size in dimension 0 of the lhs operand of a "
                "convolution with batch_group_count 2, got lhs type "
                "tensor<?x4x8xf32, #stablehlo.bounds<8, ?, ?>>",
        },
    }),
    [](const testing::TestParamInfo<ConvolutionLikeUnsupportedTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
}  // namespace torch_tpu
