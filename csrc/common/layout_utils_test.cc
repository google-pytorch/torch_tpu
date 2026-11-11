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
#include "torch/headeronly/core/MemoryFormat.h"
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

TEST(LayoutUtilsTest, EmptyLikeStrides_PreservesDenseAndRemovesGaps) {
  auto contiguous = at::empty({2, 3, 4}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(EmptyLikeStrides(contiguous), Strides({12, 4, 1}));

  auto transposed = contiguous.transpose(1, 2);
  EXPECT_EQ(EmptyLikeStrides(transposed), Strides({12, 1, 4}));

  // Sliced tensor has a stride gap; EmptyLikeStrides removes the gap while
  // preserving dimension order.
  auto sliced = at::empty({2, 3, 8}, at::TensorOptions().dtype(at::kFloat))
                    .slice(/*dim=*/2, /*start=*/0, /*end=*/4)
                    .transpose(1, 2);
  EXPECT_EQ(EmptyLikeStrides(sliced), Strides({12, 1, 4}));

  // Expanded tensor has zero stride; EmptyLikeStrides materializes dense
  // strides matching at::infer_dense_strides.
  auto expanded = at::empty({1, 3, 4}, at::TensorOptions().dtype(at::kFloat))
                      .expand({2, 3, 4});
  EXPECT_EQ(EmptyLikeStrides(expanded), Strides({12, 4, 1}));
}

TEST(LayoutUtilsTest, BinaryOpStrides_MatchesTensorIterator) {
  auto a = at::empty({2, 3, 4}, at::TensorOptions().dtype(at::kFloat))
               .transpose(1, 2);
  auto b = at::empty({2, 3, 4}, at::TensorOptions().dtype(at::kFloat))
               .transpose(1, 2);
  EXPECT_EQ(BinaryOpStrides(a, b, {2, 4, 3}), Strides({12, 1, 4}));

  // Scalar + non-contiguous tensor preserves the non-contiguous tensor strides.
  auto scalar = at::empty({}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(BinaryOpStrides(scalar, a, {2, 4, 3}), Strides({12, 1, 4}));
  EXPECT_EQ(BinaryOpStrides(a, scalar, {2, 4, 3}), Strides({12, 1, 4}));

  // Two same-shape tensors with different strides follow the first operand's
  // dimension permutation.
  auto c = at::empty({2, 4, 3}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(BinaryOpStrides(a, c, {2, 4, 3}), Strides({12, 1, 4}));
  EXPECT_EQ(BinaryOpStrides(c, a, {2, 4, 3}), Strides({12, 3, 1}));

  // Zero-element broadcast tensor returns contiguous strides.
  auto empty_transposed =
      at::empty({4, 0, 5}, at::TensorOptions().dtype(at::kFloat))
          .transpose(0, 2);
  auto bcast = at::empty({1, 1, 4}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(BinaryOpStrides(empty_transposed, bcast, {5, 0, 4}),
            Strides({4, 4, 1}));
  EXPECT_EQ(BinaryOpStrides(scalar, empty_transposed, {5, 0, 4}),
            Strides({4, 4, 1}));
  EXPECT_EQ(BinaryOpStrides(empty_transposed, scalar, {5, 0, 4}),
            Strides({4, 4, 1}));

  // 4D zero-element tensor returns contiguous strides.
  auto empty_4d =
      at::empty({4, 0, 5, 2}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(BinaryOpStrides(empty_4d, empty_4d, {4, 0, 5, 2}),
            Strides({10, 10, 2, 1}));

  // Mixed-dtype binary op where an int32 operand is expanded on a dimension
  // whose stride order conflicts with the float32 operand:
  // TensorIteratorBase::compute_types casts the int32 operand to float32 before
  // reorder_dimensions, giving it dense strides so that the expanded dimension
  // participates in dimension comparison instead of being skipped.
  auto int_expanded = at::empty({2, 1, 4}, at::TensorOptions().dtype(at::kInt))
                          .expand({2, 3, 4});
  auto float_permuted =
      at::empty({2, 4, 3}, at::TensorOptions().dtype(at::kFloat))
          .transpose(1, 2);
  EXPECT_EQ(BinaryOpStrides(int_expanded, float_permuted, {2, 3, 4},
                            /*common_dtype=*/at::kFloat),
            Strides({12, 4, 1}));

  // 4D (2, 1, 4, 5) where is_contiguous and is_channels_last both hold:
  // 1. Same-shape (2, 1, 4, 5) where both operands are ChannelsLast
  //    ((20, 1, 5, 1)) returns contiguous Strides({20, 20, 5, 1}).
  auto cl_2145 = at::empty({2, 1, 4, 5},
                           at::TensorOptions()
                               .dtype(at::kFloat)
                               .memory_format(at::MemoryFormat::ChannelsLast));
  EXPECT_EQ(cl_2145.strides(), at::IntArrayRef({20, 1, 5, 1}));
  EXPECT_EQ(BinaryOpStrides(cl_2145, cl_2145, {2, 1, 4, 5}),
            Strides({20, 20, 5, 1}));

  // 2. Same-shape (2, 1, 4, 5) where one is ChannelsLast and one is Contiguous,
  //    plus mixed-dtype promoted int32 ChannelsLast (2, 1, 4, 5) with float32
  //    Contiguous (2, 1, 4, 5) (and vice versa) returns Strides({20, 20, 5,
  //    1}).
  auto contig_2145 =
      at::empty({2, 1, 4, 5}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(contig_2145.strides(), at::IntArrayRef({20, 20, 5, 1}));
  EXPECT_EQ(BinaryOpStrides(cl_2145, contig_2145, {2, 1, 4, 5}),
            Strides({20, 20, 5, 1}));
  EXPECT_EQ(BinaryOpStrides(contig_2145, cl_2145, {2, 1, 4, 5}),
            Strides({20, 20, 5, 1}));

  auto int_cl_2145 =
      at::empty({2, 1, 4, 5}, at::TensorOptions().dtype(at::kInt).memory_format(
                                  at::MemoryFormat::ChannelsLast));
  EXPECT_EQ(BinaryOpStrides(int_cl_2145, contig_2145, {2, 1, 4, 5},
                            /*common_dtype=*/at::kFloat),
            Strides({20, 20, 5, 1}));
  EXPECT_EQ(BinaryOpStrides(contig_2145, int_cl_2145, {2, 1, 4, 5},
                            /*common_dtype=*/at::kFloat),
            Strides({20, 20, 5, 1}));

  // 3. Broadcast (2, 1, 4, 5) ChannelsLast ((20, 1, 5, 1)) + (2, 1, 4, 1)
  //    contiguous ((4, 4, 1, 1)) returns Strides({20, 1, 5, 1}).
  auto contig_2141 =
      at::empty({2, 1, 4, 1}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(contig_2141.strides(), at::IntArrayRef({4, 4, 1, 1}));
  EXPECT_EQ(BinaryOpStrides(cl_2145, contig_2141, {2, 1, 4, 5}),
            Strides({20, 1, 5, 1}));

  // 4. Broadcast (2, 4, 1) transposed to (2, 1, 4) (strides (4, 1, 1), where
  //    is_contiguous() is true) + (2, 1, 1) contiguous ((1, 1, 1)) returns
  //    Strides({4, 1, 1}).
  auto transposed_214 =
      at::empty({2, 4, 1}, at::TensorOptions().dtype(at::kFloat))
          .transpose(1, 2);
  EXPECT_EQ(transposed_214.strides(), at::IntArrayRef({4, 1, 1}));
  EXPECT_TRUE(transposed_214.is_contiguous());
  auto contig_211 = at::empty({2, 1, 1}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(BinaryOpStrides(transposed_214, contig_211, {2, 1, 4}),
            Strides({4, 1, 1}));
}

TEST(LayoutUtilsTest, UnaryOpStrides_MatchesTensorIterator) {
  auto contiguous = at::empty({2, 3, 4}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(UnaryOpStrides(contiguous), Strides({12, 4, 1}));

  auto transposed = contiguous.transpose(1, 2);
  EXPECT_EQ(UnaryOpStrides(transposed), Strides({12, 1, 4}));

  // Zero-element tensor returns contiguous strides matching TensorIterator.
  auto empty_transposed =
      at::empty({4, 0, 5}, at::TensorOptions().dtype(at::kFloat))
          .transpose(0, 2);
  EXPECT_EQ(UnaryOpStrides(empty_transposed), Strides({4, 4, 1}));
}

TEST(LayoutUtilsTest, SuggestMemoryFormatStrides_MatchesMemoryFormat) {
  auto contiguous =
      at::empty({2, 3, 4, 5}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(SuggestMemoryFormatStrides(contiguous), Strides({60, 20, 5, 1}));

  auto channels_last = at::empty(
      {2, 3, 4, 5}, at::TensorOptions()
                        .dtype(at::kFloat)
                        .memory_format(at::MemoryFormat::ChannelsLast));
  EXPECT_EQ(SuggestMemoryFormatStrides(channels_last), Strides({60, 1, 15, 3}));

  auto channels_last_3d = at::empty(
      {2, 3, 4, 5, 6}, at::TensorOptions()
                           .dtype(at::kFloat)
                           .memory_format(at::MemoryFormat::ChannelsLast3d));
  EXPECT_EQ(SuggestMemoryFormatStrides(channels_last_3d),
            Strides({360, 1, 90, 18, 3}));

  // 2D transposed tensor has suggest_memory_format() == Contiguous.
  auto transposed_2d =
      at::empty({3, 4}, at::TensorOptions().dtype(at::kFloat)).t();
  EXPECT_EQ(SuggestMemoryFormatStrides(transposed_2d), Strides({3, 1}));
}

}  // namespace
}  // namespace torch_tpu
