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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {
using absl_testing::StatusIs;
using testing::HasSubstr;

TEST(UpdateLayoutRealToReal, RealScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::F32,
      .to_type = mlir::ElementType::F32,
  };
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutRealToReal, RealTensorNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::F32,
      .to_type = mlir::ElementType::F32,
  };
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutRealToReal, RealScalarSameSize) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::F32,
      .to_type = mlir::ElementType::I32,
  };
  StridedLayout expected = layout;
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  // Layout is unmodified, but the bitcast is not a no-op.
  EXPECT_TRUE(modified.value());
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutRealToReal, RealTensorSameSize) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::F32,
      .to_type = mlir::ElementType::I32,
  };
  StridedLayout expected = layout;
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  // Layout is unmodified, but the bitcast is not a no-op.
  EXPECT_TRUE(modified.value());
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutRealToReal, RealScalarToSmallerSize) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  layout.storage_offset = 1;
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI64,
      .to_type = mlir::ElementType::UI16,
  };
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
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
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
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
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = MakeContiguousBaseLayout({2, 3});
  expected.storage_offset = 1;
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutRealToReal, InvalidScalarToLargerSize) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI16,
      .to_type = mlir::ElementType::UI64,
  };
  EXPECT_THAT(
      UpdateLayoutImpl(layout, bitcast),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("the last dimension does not match the size ratio 4")));
}

TEST(UpdateLayoutRealToReal, InvalidTensorToLargerSizeLastDimension) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 3});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI16,
      .to_type = mlir::ElementType::UI64,
  };
  EXPECT_THAT(
      UpdateLayoutImpl(layout, bitcast),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("the last dimension does not match the size ratio 4")));
}

TEST(UpdateLayoutRealToReal, InvalidTensorToLargerSizeStrides) {
  // Input layout is contiguous (2, 3, 5) sliced as [:, :, 0:4]
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  layout.strided_dims[0].stride = 15;
  layout.strided_dims[1].stride = 5;
  layout.strided_dims[2].stride = 1;
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI16,
      .to_type = mlir::ElementType::UI64,
  };
  EXPECT_THAT(UpdateLayoutImpl(layout, bitcast),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("the stride of dimension 0 is 15 which is not "
                                 "divisible by the size ratio 4")));
}

TEST(UpdateLayoutRealToReal, InvalidTensorToLargerBadOffset) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  layout.storage_offset = 3;
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI16,
      .to_type = mlir::ElementType::UI64,
  };
  EXPECT_THAT(
      UpdateLayoutImpl(layout, bitcast),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr(
              "the storage offset 3 is not divisible by the size ratio 4")));
}

TEST(UpdateLayoutRealToReal, InvalidRealToRealFromComplex) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::COMPLEXF32,
      .to_type = mlir::ElementType::UI64,
  };
  EXPECT_THAT(
      UpdateLayoutImpl(layout, bitcast),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("real-to-real bitcasts must not have complex dtypes")));
}

TEST(UpdateLayoutRealToReal, InvalidRealToRealToComplex) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = RealToRealBitcast{
      .from_type = mlir::ElementType::UI64,
      .to_type = mlir::ElementType::COMPLEXF32,
  };
  EXPECT_THAT(
      UpdateLayoutImpl(layout, bitcast),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("real-to-real bitcasts must not have complex dtypes")));
}

TEST(UpdateLayoutComplexToReal, ScalarViewAsReal) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kViewAsReal,
  };
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = MakeContiguousBaseLayout({2});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutComplexToReal, TensorViewAsReal) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kViewAsReal,
  };
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = MakeContiguousBaseLayout({2, 3, 4, 2});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutComplexToReal, ScalarRealPart) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kReal,
  };
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = MakeContiguousBaseLayout({});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutComplexToReal, TensorRealPart) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 4});
  auto bitcast = ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kReal,
  };
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
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
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
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
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
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
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = MakeContiguousBaseLayout({2, 3, 4, 1});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutViewAsComplex, LastDimensionEven) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 8});
  auto bitcast =
      ViewAsComplex{.complex_element_type = ComplexElementType::kComplexFloat};
  auto modified = UpdateLayoutImpl(layout, bitcast);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = MakeContiguousBaseLayout({2, 3, 4});
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutViewAsComplex, InvalidScalarViewAsComplex) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto bitcast =
      ViewAsComplex{.complex_element_type = ComplexElementType::kComplexFloat};
  EXPECT_THAT(UpdateLayoutImpl(layout, bitcast),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("cannot apply view_as_complex to a scalar")));
}

TEST(UpdateLayoutViewAsComplex, InvalidTensorViewAsComplexLastSize) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 5});
  auto bitcast =
      ViewAsComplex{.complex_element_type = ComplexElementType::kComplexFloat};
  EXPECT_THAT(UpdateLayoutImpl(layout, bitcast),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("cannot view_as_complex because the last "
                                 "dimension of size 5 is not divisible by 2")));
}

TEST(UpdateLayoutViewAsComplex, InvalidTensorViewAsComplexLastStride) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 2});
  layout.strided_dims[0].stride = 12;
  layout.strided_dims[1].stride = 4;
  layout.strided_dims[2].stride = 2;
  auto bitcast =
      ViewAsComplex{.complex_element_type = ComplexElementType::kComplexFloat};
  EXPECT_THAT(UpdateLayoutImpl(layout, bitcast),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("cannot view_as_complex because the last "
                                 "dimension is not dense (stride 2 != 1)")));
}

TEST(UpdateLayoutViewAsComplex, InvalidTensorViewAsComplexStorageOffset) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 3, 2});
  layout.storage_offset = 3;
  auto bitcast =
      ViewAsComplex{.complex_element_type = ComplexElementType::kComplexFloat};
  EXPECT_THAT(UpdateLayoutImpl(layout, bitcast),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("cannot view_as_complex because the storage "
                                 "offset of 3 is not divisible by 2")));
}

TEST(UpdateLayoutViewAsComplex, InvalidTensorViewAsComplexMiddleStride) {
  StridedLayout layout = MakeContiguousBaseLayout({2, 4, 2});
  layout.strided_dims[0].stride = 12;
  layout.strided_dims[1].stride = 3;
  layout.strided_dims[2].stride = 1;
  auto bitcast =
      ViewAsComplex{.complex_element_type = ComplexElementType::kComplexFloat};
  EXPECT_THAT(UpdateLayoutImpl(layout, bitcast),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("cannot view_as_complex because stride 3 is "
                                 "not divisible by 2")));
}

}  // namespace
}  // namespace torch_tpu
