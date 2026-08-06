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

#include "torch_tpu/ops/op_builder_utils.h"

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
#include "absl/types/span.h"
#include "c10/core/DefaultDtype.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "xla/xla_data.pb.h"

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
  EXPECT_EQ(default_dtype.status(), absl::OkStatus());
  return default_dtype.value();
}

// TODO(b/433265252): add tests for broadcasting on MlirOps.

TEST(OpBuilderUtils, ConvertIfIntegers_TwoOperands_Int) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  auto default_mlir_element_type = ConvertTo<mlir::ElementType>(default_dtype);
  ASSERT_EQ(default_mlir_element_type.status(), absl::OkStatus());
  auto default_mlir_type = default_mlir_element_type.value();

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::MlirOp op1 = MakeScalarConstant(builder, 1, mlir::ElementType::I32);
  mlir::MlirOp op2 = MakeScalarConstant(builder, 1, mlir::ElementType::I32);

  auto result = ConvertIfIntegers(op1, op2, default_mlir_type);
  ASSERT_TRUE(result.ok());

  mlir::RankedTensorType default_type = mlir::RankedTensorType::get(
      {}, *GetMlirType(builder.getContext(), GetDefaultMlirDType()));
  EXPECT_EQ(result->first.getType(), default_type);
  EXPECT_EQ(result->second.getType(), default_type);
}

TEST(OpBuilderUtils, ConvertIfIntegers_TwoOperands_Float) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  auto default_mlir_element_type = ConvertTo<mlir::ElementType>(default_dtype);
  ASSERT_EQ(default_mlir_element_type.status(), absl::OkStatus());
  auto default_mlir_type = default_mlir_element_type.value();

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType type =
      mlir::RankedTensorType::get({}, op_builder.getF64Type());
  mlir::MlirOp op1 = MakeScalarConstant(builder, 1, mlir::ElementType::F64);
  mlir::MlirOp op2 = MakeScalarConstant(builder, 1, mlir::ElementType::F64);

  auto result = ConvertIfIntegers(op1, op2, default_mlir_type);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->first.getType(), type);
  EXPECT_EQ(result->second.getType(), type);
}

TEST(OpBuilderUtils, ConvertIfIntegers_TwoOperands_Int_Float) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  auto default_mlir_element_type = ConvertTo<mlir::ElementType>(default_dtype);
  ASSERT_EQ(default_mlir_element_type.status(), absl::OkStatus());
  auto default_mlir_type = default_mlir_element_type.value();

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

  auto result = ConvertIfIntegers(op1, op2, default_mlir_type);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->first.getType(), float_type);
  EXPECT_EQ(result->second.getType(), float_type);
}

TEST(OpBuilderUtils, ConvertIfIntegers_TwoOperands_Float_Int) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  auto default_mlir_element_type = ConvertTo<mlir::ElementType>(default_dtype);
  ASSERT_EQ(default_mlir_element_type.status(), absl::OkStatus());
  auto default_mlir_type = default_mlir_element_type.value();

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

  auto result = ConvertIfIntegers(op1, op2, default_mlir_type);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->first.getType(), float_type);
  EXPECT_EQ(result->second.getType(), float_type);
}

TEST(OpBuilderUtils, ConvertIfIntegers_TwoIntegerOperands) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  auto default_mlir_element_type = ConvertTo<mlir::ElementType>(default_dtype);
  ASSERT_EQ(default_mlir_element_type.status(), absl::OkStatus());
  auto default_mlir_type = default_mlir_element_type.value();

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

  auto result = ConvertIfIntegers(op1, op2, default_mlir_type);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->first.getType(), float_type);
  EXPECT_EQ(result->second.getType(), float_type);
}

TEST(OpBuilderUtils, ConvertIfInteger_OneOperand_Int) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  auto default_mlir_element_type = ConvertTo<mlir::ElementType>(default_dtype);
  ASSERT_EQ(default_mlir_element_type.status(), absl::OkStatus());
  auto default_mlir_type = default_mlir_element_type.value();

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType type =
      mlir::RankedTensorType::get({}, op_builder.getI32Type());
  mlir::MlirOp op = MakeScalarConstant(builder, 1, type.getElementType());

  auto result = ConvertIfInteger(op, default_mlir_type);
  ASSERT_TRUE(result.ok());

  mlir::RankedTensorType default_type = mlir::RankedTensorType::get(
      {}, *GetMlirType(builder.getContext(), GetDefaultMlirDType()));
  EXPECT_EQ(result->getType(), default_type);
}

TEST(OpBuilderUtils, ConvertIfInteger_OneOperand_Float) {
  // Default dtype is captured at dispatch time.
  auto default_dtype = at::ScalarType::Float;
  auto default_mlir_element_type = ConvertTo<mlir::ElementType>(default_dtype);
  ASSERT_EQ(default_mlir_element_type.status(), absl::OkStatus());
  auto default_mlir_type = default_mlir_element_type.value();

  // Then we convert to it at compile time.
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::RankedTensorType type =
      mlir::RankedTensorType::get({}, op_builder.getF64Type());
  mlir::MlirOp op = MakeScalarConstant(builder, 1, type.getElementType());

  auto result = ConvertIfInteger(op, default_mlir_type);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->getType(), type);
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
  auto result = InferSize(dims1_broadcastable, dims2_broadcastible);
  ASSERT_TRUE(result.ok());
  EXPECT_THAT(result.value(), ElementsAre(2, 3));

  Dimensions dims1_non_broadcastable = {2};
  Dimensions dims2_non_broadcastable = {3};
  result = InferSize(dims1_non_broadcastable, dims2_non_broadcastable);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("must match the size of tensor"));
}

TEST(InferSize, WorksWithTwoTensors) {
  at::Tensor tensor1 = at::ones({2, 1});
  at::Tensor tensor2 = at::ones({1, 3});
  auto result = InferSize(tensor1, tensor2);
  ASSERT_TRUE(result.ok());
  EXPECT_THAT(result.value(), ElementsAre(2, 3));
}

TEST(InferSize, WorksWithMoreThanTwoTensors) {
  at::Tensor tensor1 = at::ones({4, 2, 1});
  at::Tensor tensor2 = at::ones({1, 3});
  at::Tensor tensor3 = at::ones({1, 2, 1});
  auto result = InferSize(tensor1, tensor2, tensor3);
  ASSERT_TRUE(result.ok());
  EXPECT_THAT(result.value(), ElementsAre(4, 2, 3));
}

TEST(InferSize, WorksWithDimsAndTensors) {
  at::Tensor tensor1 = at::ones({4, 2, 1});
  Dimensions dims2 = {1, 3};
  at::Tensor tensor3 = at::ones({1, 2, 1});
  auto result = InferSize(tensor1, dims2, tensor3);
  ASSERT_TRUE(result.ok());
  EXPECT_THAT(result.value(), ElementsAre(4, 2, 3));
}

TEST(BroadcastIfNeeded, Scalar) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op = MakeConstant(builder, 1.0f, op_builder.getF32Type(), {});
  Dimensions target_shape = {2, 3};
  auto result = BroadcastIfNeeded(op, target_shape);
  ASSERT_TRUE(result.ok());
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result->getType());
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
  auto result = BroadcastIfNeeded(op, target_shape);
  ASSERT_TRUE(result.ok());
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result->getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(2, 3));
}

TEST(BroadcastIfNeeded, BroadcastDim) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op = MakeConstant(builder, 1.0f, op_builder.getF32Type(), {3});
  Dimensions target_shape = {2, 3};
  auto result = BroadcastIfNeeded(op, target_shape);
  ASSERT_TRUE(result.ok());
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result->getType());
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
  auto result = BroadcastIfNeeded(op, target_shape);
  ASSERT_TRUE(result.ok());
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result->getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(2, 3));
}

TEST(BroadcastIfNeeded, StablehloDims) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::MlirOp op = MakeConstant(builder, 5, op_builder.getI32Type(), {1, 3});
  mlir::stablehlo::Dimensions target_shape = {{2}, {3}};
  auto result = BroadcastIfNeeded(op, target_shape);
  ASSERT_TRUE(result.ok());
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result->getType());
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
  auto result = BroadcastIfNeeded(op, target_shape);
  ASSERT_TRUE(result.ok());
  auto dims = GetDimensions(*result);
  EXPECT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].size, 10);  // check padded size
  EXPECT_EQ(dims[1].size, 3);   // check static size

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result->getType());
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
  auto result = BroadcastIfNeeded(op, target, {0, 2});
  ASSERT_TRUE(result.ok());
  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result->getType());
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

  auto result = BroadcastIfNeeded(input_op, target_op, {0, 2});
  ASSERT_TRUE(result.ok());

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result->getType());
  ASSERT_TRUE(result_type);
  EXPECT_EQ(result_type.getRank(), 3);
  EXPECT_TRUE(result_type.isDynamicDim(0));
  EXPECT_FALSE(result_type.isDynamicDim(1));
  EXPECT_FALSE(result_type.isDynamicDim(2));
  EXPECT_EQ(result_type.getDimSize(1), 4);
  EXPECT_EQ(result_type.getDimSize(2), 3);

  auto dims = GetDimensions(*result);
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
  auto result = BroadcastIfNeeded(input_op, target_op, {0, 1});
  ASSERT_TRUE(result.ok());

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(result->getType());
  ASSERT_TRUE(result_type);
  EXPECT_EQ(result_type.getRank(), 3);
  EXPECT_FALSE(result_type.isDynamicDim(0));
  EXPECT_FALSE(result_type.isDynamicDim(1));
  EXPECT_TRUE(result_type.isDynamicDim(2));
  EXPECT_EQ(result_type.getDimSize(0), 3);
  EXPECT_EQ(result_type.getDimSize(1), 4);

  auto dims = GetDimensions(*result);
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
  auto result = CastIfNeeded(op, mlir::ElementType::F32);
  ASSERT_TRUE(result.ok());
  mlir::RankedTensorType float_type =
      mlir::RankedTensorType::get({}, op_builder.getF32Type());
  EXPECT_EQ(result->getType(), float_type);
}

TEST(InferSize, WorksWithDimsAndScalars) {
  at::Tensor tensor1 = at::ones({4, 2, 1});
  Dimensions dims2 = {1, 3};
  at::Scalar scalar3 = 1;
  auto result = InferSize(tensor1, dims2, scalar3);
  ASSERT_TRUE(result.ok());
  EXPECT_THAT(result.value(), ElementsAre(4, 2, 3));
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

TEST(ReshapeFromStaticDimensions, StaticShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = mb.getOpBuilder();
  mlir::MlirOp op = MakeConstant(mb, 1.0f, op_builder.getF32Type(), {2, 3});
  Dimensions static_input_shape = {2, 3};
  Dimensions static_output_shape = {6};
  auto reshaped =
      ReshapeFromStaticDimensions(op, static_input_shape, static_output_shape);
  ASSERT_TRUE(reshaped.ok());
  auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(reshaped->getType());
  ASSERT_TRUE(result_type);
  EXPECT_THAT(result_type.getShape(), ElementsAre(6));
}

TEST(BroadcastIfNeeded, BroadcastLikeOtherOp) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = mb.getOpBuilder();
  mlir::MlirOp cst = MakeConstant(mb, 1.0f, op_builder.getF32Type(), {});
  mlir::MlirOp op = MakeConstant(mb, 2.0f, op_builder.getF32Type(), {2, 2});
  absl::StatusOr<mlir::MlirOp> cst_bcast = BroadcastIfNeeded(cst, op);
  ASSERT_TRUE(cst_bcast.ok());
  mlir::RankedTensorType cst_bcast_type =
      mlir::cast<mlir::RankedTensorType>(cst_bcast->getType());
  EXPECT_THAT(cst_bcast_type.getShape(), ElementsAre(2, 2));
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
  ASSERT_TRUE(result.reshaped_op.ok());

  // Check that the reshaped op has proper bounded dynamic dimensions.
  mlir::stablehlo::Dimensions reshaped_dims =
      GetDimensions(*result.reshaped_op);
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
  ASSERT_TRUE(result.broadcasted_op.ok());

  auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(result.broadcasted_op->getType());
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
  ASSERT_TRUE(result.broadcasted_op.ok());

  // Check that the broadcasted op has proper bounded dynamic dimensions.
  mlir::stablehlo::Dimensions broadcasted_dims =
      GetDimensions(*result.broadcasted_op);
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

  auto bytecode_or = SerializeBytecode(module.get());
  ASSERT_TRUE(bytecode_or.ok());
  std::string bytecode = bytecode_or.value();

  // Check for the MLIR magic string that denotes bytecode.
  EXPECT_THAT(bytecode, testing::HasSubstr("\x4D\x4C\xEF\x52"));
}

TEST(OpBuilderUtils, SerializePortableArtifact) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::ModuleBuilder& mb = op_builder_utils_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::func::Return(fb, {});
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();

  auto artifact_or = SerializePortableArtifact(module.get());
  ASSERT_TRUE(artifact_or.ok());
  std::string artifact = artifact_or.value();

  // Check for the MLIR magic string that denotes bytecode.
  EXPECT_THAT(artifact, testing::HasSubstr("\x4D\x4C\xEF\x52"));
  // Check for the StableHLO_v1 producer string to indicate versioned StableHLO.
  EXPECT_THAT(artifact, testing::HasSubstr("StableHLO_v1."));
}

TEST(OpBuilderUtils, MakeZeroSizedTensor_MlirType_DefaultShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  auto op_or = MakeZeroSizedTensor(builder, op_builder.getF32Type());
  ASSERT_TRUE(op_or.ok());
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(op_or->getType());
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.getElementType().isF32());
  EXPECT_THAT(type.getShape(), ElementsAre(0));
}

TEST(OpBuilderUtils, MakeZeroSizedTensor_MlirType_CustomShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  auto op_or = MakeZeroSizedTensor(builder, op_builder.getI32Type(), {2, 0, 3});
  ASSERT_TRUE(op_or.ok());
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(op_or->getType());
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

  auto op_or = MakeZeroSizedTensor(builder, mlir::ElementType::F32);
  ASSERT_TRUE(op_or.ok());
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(op_or->getType());
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.getElementType().isF32());
  EXPECT_THAT(type.getShape(), ElementsAre(0));
}

TEST(OpBuilderUtils, MakeZeroSizedTensor_MlirElementType_CustomShape) {
  OpBuilderUtilsBuilder op_builder_utils_builder;
  mlir::MlirBuilder& builder = op_builder_utils_builder.get();

  auto op_or = MakeZeroSizedTensor(builder, mlir::ElementType::I32, {2, 0, 3});
  ASSERT_TRUE(op_or.ok());
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(op_or->getType());
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

}  // namespace
}  // namespace torch_tpu
