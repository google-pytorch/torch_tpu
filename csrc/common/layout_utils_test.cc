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

#include "csrc/common/layout_utils.h"

#include <cstdint>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace torch_tpu {
namespace {

TEST(LayoutUtilsTest, XlaEquivalentBitwidth) {
  EXPECT_EQ(XlaEquivalentBitwidth(mlir::ElementType::PRED), 1);
  EXPECT_EQ(TorchEquivalentBitwidth(mlir::ElementType::PRED), 8);
  EXPECT_EQ(XlaEquivalentBitwidth(mlir::ElementType::F32), 32);
  EXPECT_EQ(XlaEquivalentBitwidth(mlir::ElementType::BF16), 16);
  EXPECT_EQ(XlaEquivalentBitwidth(mlir::ElementType::I8), 8);
  EXPECT_EQ(XlaEquivalentBitwidth(mlir::ElementType::I4), 4);
}

TEST(LayoutUtilsTest, CustomLayoutTilingAndEquality) {
  CustomLayout layout1;
  layout1.minor_to_major = {1, 0};
  layout1.tiles.push_back(Indices({8, 128}));
  layout1.element_size_in_bits = 32;
  CustomLayout layout2 = layout1;
  EXPECT_EQ(layout1, layout2);
  Shape shape({100, 128}, mlir::ElementType::F32);
  shape.set_layout(layout1);
  ASSERT_TRUE(shape.layout().has_value());
  EXPECT_EQ(*shape.layout(), layout1);
  EXPECT_EQ(shape.layout()->tiles[0], Indices({8, 128}));
}

TEST(LayoutUtilsTest, MakeShapePreservesTiles) {
  const xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {16, 128}, {0, 1}, {xla::Tile({8})});

  TT_ASSERT_OK_AND_ASSIGN(const Shape shape, MakeShape(xla_shape));
  ASSERT_TRUE(shape.layout().has_value());
  EXPECT_EQ(shape.layout()->minor_to_major, Indices({0, 1}));
  ASSERT_EQ(shape.layout()->tiles.size(), 1);
  EXPECT_EQ(shape.layout()->tiles[0], Indices({8}));
}

TEST(LayoutUtilsTest, ResolveTpuLayout_StandardTensor_Success) {
  auto tensor = at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat));
  TT_ASSERT_OK_AND_ASSIGN(auto layout, ResolveTpuLayout(tensor));

  EXPECT_EQ(layout.sizes, Dimensions({2, 3}));
  EXPECT_EQ(layout.strides, Strides({3, 1}));
  EXPECT_EQ(layout.storage_offset, 0);
  EXPECT_EQ(layout.element_type, mlir::ElementType::F32);
}

TEST(LayoutUtilsTest, ResolveTpuLayout_StandardTensorZeroDim_Success) {
  auto tensor = at::empty({}, at::TensorOptions().dtype(at::kFloat));
  TT_ASSERT_OK_AND_ASSIGN(auto layout, ResolveTpuLayout(tensor));

  EXPECT_EQ(layout.sizes, Dimensions({}));
  EXPECT_EQ(layout.strides, Strides({}));
  EXPECT_EQ(layout.storage_offset, 0);
  EXPECT_EQ(layout.element_type, mlir::ElementType::F32);
}

TEST(LayoutUtilsTest, ResolveTpuLayout_FP4Tensor_Success) {
  auto tensor =
      at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2));

  TT_ASSERT_OK_AND_ASSIGN(auto layout, ResolveTpuLayout(tensor));

  EXPECT_EQ(layout.sizes, Dimensions({2, 6}));
  EXPECT_EQ(layout.strides, Strides({6, 1}));
  EXPECT_EQ(layout.storage_offset, 0);
  EXPECT_EQ(layout.element_type, mlir::ElementType::F4E2M1FN);
}

TEST(LayoutUtilsTest, ResolveTpuLayout_FP4TensorWithOffset_Success) {
  auto base_tensor =
      at::empty({10, 8}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2));
  auto tensor = base_tensor.slice(0, 1, 3).slice(1, 1, 4);

  TT_ASSERT_OK_AND_ASSIGN(auto layout, ResolveTpuLayout(tensor));

  EXPECT_EQ(layout.sizes, Dimensions({2, 6}));
  EXPECT_EQ(layout.strides, Strides({16, 1}));
  EXPECT_EQ(layout.storage_offset, 18);
  EXPECT_EQ(layout.element_type, mlir::ElementType::F4E2M1FN);
}

TEST(LayoutUtilsTest, ResolveTpuLayout_ZeroDimFP4Tensor_ReturnsError) {
  auto tensor = at::empty({}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2));
  auto layout_or = ResolveTpuLayout(tensor);
  EXPECT_EQ(layout_or.status().code(), error::kInvalidArgument);
}

TEST(LayoutUtilsTest, ResolveTpuLayout_FP4TensorNonContiguous_ReturnsError) {
  auto base_tensor =
      at::empty({10, 8}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2));
  // Slice with step=2 on the packed dimension, breaking contiguity
  auto tensor =
      base_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/8, /*step=*/2);

  auto layout_or = ResolveTpuLayout(tensor);
  EXPECT_EQ(layout_or.status().code(), error::kInvalidArgument);
}

TEST(LayoutUtilsTest, MakeXlaShapeRoundTripAndDynamism) {
  Shape src_shape({16, 128}, mlir::ElementType::BF16);
  CustomLayout layout;
  layout.minor_to_major = {0, 1};
  layout.tiles.push_back(Indices({8, 128}));
  layout.element_size_in_bits = 16;
  src_shape.set_layout(layout);

  TT_ASSERT_OK_AND_ASSIGN(const xla::Shape xla_shape, MakeXlaShape(src_shape));
  EXPECT_EQ(xla_shape.element_type(), xla::PrimitiveType::BF16);
  EXPECT_EQ(xla_shape.dimensions(), absl::Span<const int64_t>({16, 128}));
  ASSERT_TRUE(xla_shape.has_layout());
  EXPECT_EQ(xla_shape.layout().minor_to_major(),
            absl::Span<const int64_t>({0, 1}));

  TT_ASSERT_OK_AND_ASSIGN(const Shape round_trip, MakeShape(xla_shape));
  EXPECT_EQ(round_trip, src_shape);
  EXPECT_FALSE(round_trip.on_device_shape_is_dynamic());

  xla::Shape dynamic_xla_shape = xla_shape;
  dynamic_xla_shape.set_dynamic_dimension(0, true);
  TT_ASSERT_OK_AND_ASSIGN(const Shape dynamic_shape,
                          MakeShape(dynamic_xla_shape));
  EXPECT_TRUE(dynamic_shape.on_device_shape_is_dynamic());
}

TEST(LayoutUtilsTest, MakeXlaShapeScalarWithCustomLayout) {
  Shape scalar_shape({}, mlir::ElementType::BF16);
  CustomLayout layout;
  layout.element_size_in_bits = 16;
  scalar_shape.set_layout(layout);

  TT_ASSERT_OK_AND_ASSIGN(const xla::Shape xla_shape,
                          MakeXlaShape(scalar_shape));
  EXPECT_EQ(xla_shape.element_type(), xla::PrimitiveType::BF16);
  EXPECT_TRUE(xla_shape.dimensions().empty());
  ASSERT_TRUE(xla_shape.has_layout());
  EXPECT_TRUE(xla_shape.layout().minor_to_major().empty());
  EXPECT_EQ(xla_shape.layout().element_size_in_bits(), 16);

  TT_ASSERT_OK_AND_ASSIGN(const Shape round_trip, MakeShape(xla_shape));
  EXPECT_EQ(round_trip, scalar_shape);
}

TEST(LayoutUtilsTest, MakeXlaShapeInvalidDimensionsOrLayout) {
  Shape invalid_dim_shape({-5}, mlir::ElementType::F32);
  EXPECT_FALSE(MakeXlaShape(invalid_dim_shape).ok());

  Shape invalid_layout_shape({16, 128}, mlir::ElementType::F32);
  CustomLayout invalid_layout;
  invalid_layout.minor_to_major = {0};
  invalid_layout_shape.set_layout(invalid_layout);
  EXPECT_FALSE(MakeXlaShape(invalid_layout_shape).ok());
}

}  // namespace
}  // namespace torch_tpu
