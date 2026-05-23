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

#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"

#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

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

}  // namespace
}  // namespace torch_tpu
