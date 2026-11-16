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

#include <optional>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
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

TEST(LayoutUtilsTest, ToXlaLayoutKeepsAllFields) {
  CustomLayout layout;
  layout.minor_to_major = {1, 0};
  layout.tiles.push_back(Indices({8}));
  layout.element_size_in_bits = 32;
  EXPECT_EQ(ToXlaLayout(layout), xla::Layout({1, 0}, {xla::Tile({8})}, 32));
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

TEST(LayoutUtilsTest, ResolveTpuLayout_FP4TensorTransposed_Success) {
  auto tensor =
      at::empty({2, 4}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2)).t();

  TT_ASSERT_OK_AND_ASSIGN(auto layout, ResolveTpuLayout(tensor));

  EXPECT_EQ(layout.sizes, Dimensions({8, 2}));
  EXPECT_EQ(layout.strides, Strides({1, 8}));
  EXPECT_EQ(layout.storage_offset, 0);
  EXPECT_EQ(layout.element_type, mlir::ElementType::F4E2M1FN);

  auto unit_trailing =
      at::empty({1, 4}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2)).t();
  unit_trailing.as_strided_({4, 1}, {1, 1}, 0);
  TT_ASSERT_OK_AND_ASSIGN(auto unit_layout, ResolveTpuLayout(unit_trailing));
  EXPECT_EQ(unit_layout.sizes, Dimensions({4, 2}));
  EXPECT_EQ(unit_layout.strides, Strides({2, 1}));

  auto unit_leading_transposed =
      at::empty({4, 1}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2)).t();
  TT_ASSERT_OK_AND_ASSIGN(auto unit_leading_layout,
                          ResolveTpuLayout(unit_leading_transposed));
  EXPECT_EQ(unit_leading_layout.sizes, Dimensions({2, 4}));
  EXPECT_EQ(unit_leading_layout.strides, Strides({1, 2}));

  auto both_unit =
      at::empty({1, 1}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2)).t();
  TT_ASSERT_OK_AND_ASSIGN(auto both_unit_layout, ResolveTpuLayout(both_unit));
  EXPECT_EQ(both_unit_layout.sizes, Dimensions({1, 2}));
  EXPECT_EQ(both_unit_layout.strides, Strides({2, 1}));
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

  auto invalid_unit_stride =
      at::empty({10, 8}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2));
  invalid_unit_stride.as_strided_({1, 4}, {1, 2}, 0);
  EXPECT_EQ(ResolveTpuLayout(invalid_unit_stride).status().code(),
            error::kInvalidArgument);
}

TEST(LayoutUtilsTest, ResolveXlaLayoutsWithCustomLayout) {
  Shape default_bitwidth_shape({16, 128}, mlir::ElementType::F32);
  default_bitwidth_shape.set_layout(CustomLayout{
      .minor_to_major = {0, 1},
      .tiles = {{8, 128}},
      .element_size_in_bits = 32,
  });

  Shape custom_bitwidth_shape({16, 128}, mlir::ElementType::I8);
  custom_bitwidth_shape.set_layout(CustomLayout{
      .minor_to_major = {1, 0},
      .tiles = {{8, 128}},
      .element_size_in_bits = 4,
  });

  TT_ASSERT_OK_AND_ASSIGN(
      const auto layouts,
      ResolveXlaLayouts({default_bitwidth_shape, custom_bitwidth_shape}));
  ASSERT_EQ(layouts.size(), 2);
  EXPECT_EQ(layouts[0], xla::Layout({0, 1}, {xla::Tile({8, 128})},
                                    /*element_size_in_bits=*/0));
  EXPECT_EQ(layouts[1], xla::Layout({1, 0}, {xla::Tile({8, 128})},
                                    /*element_size_in_bits=*/4));
}

// Converts `layout`'s dimension order and tiles to a `CustomLayout`.
CustomLayout ToCustomLayout(const xla::Layout& layout) {
  CustomLayout result{.minor_to_major = Indices(layout.minor_to_major().begin(),
                                                layout.minor_to_major().end())};
  for (const xla::Tile& tile : layout.tiles()) {
    result.tiles.push_back(
        Indices(tile.dimensions().begin(), tile.dimensions().end()));
  }
  return result;
}

TEST(LayoutUtilsTest, ExtractCustomLayoutReturnsNulloptWithoutLayout) {
  const Shape shape({16, 128}, mlir::ElementType::F32);
  TT_ASSERT_OK_AND_ASSIGN(const std::optional<CustomLayout> layout,
                          ExtractCustomLayoutIfDifferentFromDefault(shape));
  EXPECT_EQ(layout, std::nullopt);
}

TEST(LayoutUtilsTest, ExtractCustomLayoutReturnsNulloptForDefaultLayout) {
  Shape shape({16, 128}, mlir::ElementType::F32);
  TT_ASSERT_OK_AND_ASSIGN(const std::vector<xla::Layout> default_layouts,
                          ResolveXlaLayouts({shape}));
  shape.set_layout(ToCustomLayout(default_layouts[0]));
  TT_ASSERT_OK_AND_ASSIGN(const std::optional<CustomLayout> layout,
                          ExtractCustomLayoutIfDifferentFromDefault(shape));
  EXPECT_EQ(layout, std::nullopt);
}

TEST(LayoutUtilsTest, ExtractCustomLayoutReturnsNonDefaultLayout) {
  Shape shape({16, 128}, mlir::ElementType::F32);
  const CustomLayout transposed{.minor_to_major = {0, 1}, .tiles = {{8, 128}}};
  shape.set_layout(transposed);
  TT_ASSERT_OK_AND_ASSIGN(const std::optional<CustomLayout> layout,
                          ExtractCustomLayoutIfDifferentFromDefault(shape));
  EXPECT_EQ(layout, transposed);
}

TEST(LayoutUtilsTest, ExtractCustomLayoutOmitsTilesForDynamicShape) {
  Shape shape({16, 128}, mlir::ElementType::F32);
  shape.set_on_device_shape_is_dynamic(true);
  shape.set_layout(CustomLayout{.minor_to_major = {0, 1}, .tiles = {{8, 128}}});
  TT_ASSERT_OK_AND_ASSIGN(const std::optional<CustomLayout> layout,
                          ExtractCustomLayoutIfDifferentFromDefault(shape));
  const CustomLayout untiled{.minor_to_major = {0, 1}};
  EXPECT_EQ(layout, untiled);
}

}  // namespace
}  // namespace torch_tpu
