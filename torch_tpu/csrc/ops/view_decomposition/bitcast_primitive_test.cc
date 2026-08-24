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

#include "torch_tpu/csrc/ops/view_decomposition/bitcast_primitive.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/view_decomposition/strided_layout.h"
#include "xla/hlo/testlib/filecheck.h"

namespace torch_tpu {
namespace {

TEST(UpdateLayoutRealToReal, RealScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::F32,
      .to_type = mlir::ElementType::F32,
  };
  EXPECT_FALSE(UpdateLayout(layout, bitcast));
}

TEST(UpdateLayoutRealToReal, RealTensorNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::F32,
      .to_type = mlir::ElementType::F32,
  };
  EXPECT_FALSE(UpdateLayout(layout, bitcast));
}

TEST(UpdateLayoutRealToReal, RealScalarSameSize) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::F32,
      .to_type = mlir::ElementType::I32,
  };
  StridedLayout expected = layout;
  // Layout is unmodified, but the bitcast is not a no-op.
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutRealToReal, RealTensorSameSize) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::F32,
      .to_type = mlir::ElementType::I32,
  };
  StridedLayout expected = layout;
  // Layout is unmodified, but the bitcast is not a no-op.
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutRealToReal, RealScalarToSmallerSize) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  layout.storage_offset = 1;
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI64,
      .to_type = mlir::ElementType::UI16,
  };
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({4});
  expected.storage_offset = 4;
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutRealToReal, RealTensorToSmallerSize) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI64,
      .to_type = mlir::ElementType::UI16,
  };
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({2, 3, 4, 4});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutRealToReal, RealTensorToLargerSize) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  layout.storage_offset = 4;
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI16,
      .to_type = mlir::ElementType::UI64,
  };
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({2, 3});
  expected.storage_offset = 1;
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutComplexToReal, ScalarViewAsReal) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kViewAsReal,
  };
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({2});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutComplexToReal, TensorViewAsReal) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kViewAsReal,
  };
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({2, 3, 4, 2});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutComplexToReal, ScalarRealPart) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kReal,
  };
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutComplexToReal, TensorRealPart) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kReal,
  };
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({2, 3, 4});
  expected.strided_dims[0].stride = 24;
  expected.strided_dims[1].stride = 8;
  expected.strided_dims[2].stride = 2;
  expected.storage_offset = 0;
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutComplexToReal, ScalarImagPart) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kImag,
  };
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({});
  expected.storage_offset = 1;
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutComplexToReal, TensorImagPart) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kImag,
  };
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({2, 3, 4});
  expected.strided_dims[0].stride = 24;
  expected.strided_dims[1].stride = 8;
  expected.strided_dims[2].stride = 2;
  expected.storage_offset = 1;
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutViewAsComplex, LastDimensionSize2) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4, 2});
  auto bitcast =
      ViewAsComplex{.complex_element_type = ComplexElementType::kComplexFloat};
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({2, 3, 4, 1});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutViewAsComplex, LastDimensionEven) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 8});
  auto bitcast =
      ViewAsComplex{.complex_element_type = ComplexElementType::kComplexFloat};
  EXPECT_TRUE(UpdateLayout(layout, bitcast));
  StridedLayout expected = MakeContiguousBaseLayout({2, 3, 4});
  EXPECT_EQ(layout, expected);
}

TEST(ViewPrimitiveShloTest, RealToRealBitcastSmallerToLarger) {
  mlir::MLIRContext ctx;
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
  mlir::OpBuilder op_builder(&ctx);
  mlir::ModuleBuilder mb(ctx, mlir::unknownLoc(ctx));

  mlir::RankedTensorType input_type =
      mlir::RankedTensorType::get({2, 4}, op_builder.getI8Type());

  std::vector<int8_t> input_values = {1, 2, 3, 4, 5, 6, 7, 8};
  auto input_attr =
      mlir::makeConstant(llvm::ArrayRef<int8_t>(input_values), input_type);
  mlir::MlirOp input = mlir::stablehlo::Constant(mb, input_attr);

  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::I8,
      .to_type = mlir::ElementType::I32,
  };

  auto result = ViewPrimitiveShlo(input, bitcast);
  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result->getType(),
            mlir::RankedTensorType::get({2}, op_builder.getI32Type()));

  std::string mlir_str = DebugString(mb.build().get());
  ASSERT_THAT(mlir_str, testing::HasSubstr("stablehlo.or"));
}

TEST(ViewPrimitiveShloTest, RealToRealBitcastSmallerToLargerUnsigned) {
  mlir::MLIRContext ctx;
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
  mlir::OpBuilder op_builder(&ctx);
  mlir::ModuleBuilder mb(ctx, mlir::unknownLoc(ctx));

  mlir::RankedTensorType input_type = mlir::RankedTensorType::get(
      {2, 2}, mlir::getElementType(ctx, mlir::ElementType::UI16));

  std::vector<uint16_t> input_values = {1, 2, 3, 4};
  auto input_attr =
      mlir::makeConstant(llvm::ArrayRef<uint16_t>(input_values), input_type);
  mlir::MlirOp input = mlir::stablehlo::Constant(mb, input_attr);

  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI16,
      .to_type = mlir::ElementType::UI32,
  };

  auto result = ViewPrimitiveShlo(input, bitcast);
  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result->getType(),
            mlir::RankedTensorType::get(
                {2}, mlir::getElementType(ctx, mlir::ElementType::UI32)));

  std::string mlir_str = DebugString(mb.build().get());
  ASSERT_THAT(mlir_str, testing::HasSubstr("stablehlo.or"));
}

TEST(ViewPrimitiveShloTest, RealToRealBitcastSmallerToLarger1D) {
  mlir::MLIRContext ctx;
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
  mlir::OpBuilder op_builder(&ctx);
  mlir::ModuleBuilder mb(ctx, mlir::unknownLoc(ctx));

  mlir::RankedTensorType input_type = mlir::RankedTensorType::get(
      {2}, mlir::getElementType(ctx, mlir::ElementType::I16));

  std::vector<int16_t> input_values = {1, 2};
  auto input_attr =
      mlir::makeConstant(llvm::ArrayRef<int16_t>(input_values), input_type);
  mlir::MlirOp input = mlir::stablehlo::Constant(mb, input_attr);

  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::I16,
      .to_type = mlir::ElementType::I32,
  };

  auto result = ViewPrimitiveShlo(input, bitcast);
  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result->getType(),
            mlir::RankedTensorType::get(
                {}, mlir::getElementType(ctx, mlir::ElementType::I32)));

  std::string mlir_str = DebugString(mb.build().get());
  absl::StatusOr<bool> filecheck_result = xla::RunFileCheck(mlir_str, R"(
    MATCH: stablehlo.bitcast_convert %{{.*}} : (tensor<2xi16>) -> tensor<i32>
  )",
                                                            {"MATCH"});
  ASSERT_TRUE(filecheck_result.ok());
  EXPECT_TRUE(*filecheck_result);
}

TEST(ViewPrimitiveShloTest, RealToRealBitcastLargerToSmaller) {
  mlir::MLIRContext ctx;
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
  mlir::OpBuilder op_builder(&ctx);
  mlir::ModuleBuilder mb(ctx, mlir::unknownLoc(ctx));

  mlir::RankedTensorType input_type = mlir::RankedTensorType::get(
      {2, 2}, mlir::getElementType(ctx, mlir::ElementType::F32));

  std::vector<float> input_values = {1.0, 2.0, 3.0, 4.0};
  auto input_attr =
      mlir::makeConstant(llvm::ArrayRef<float>(input_values), input_type);
  mlir::MlirOp input = mlir::stablehlo::Constant(mb, input_attr);

  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::F32,
      .to_type = mlir::ElementType::BF16,
  };

  auto result = ViewPrimitiveShlo(input, bitcast);
  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result->getType(),
            mlir::RankedTensorType::get(
                {2, 2, 2}, mlir::getElementType(ctx, mlir::ElementType::BF16)));

  std::string mlir_str = DebugString(mb.build().get());
  absl::StatusOr<bool> filecheck_result = xla::RunFileCheck(mlir_str, R"(
    MATCH: stablehlo.bitcast_convert %{{.*}} : (tensor<2x2xf32>) -> tensor<2x2x2xbf16>
  )",
                                                            {"MATCH"});
  ASSERT_TRUE(filecheck_result.ok());
  EXPECT_TRUE(*filecheck_result);
}

}  // namespace
}  // namespace torch_tpu
