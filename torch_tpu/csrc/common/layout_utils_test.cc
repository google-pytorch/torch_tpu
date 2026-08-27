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

#include "torch_tpu/csrc/common/layout_utils.h"

#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "c10/core/ScalarType.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/shape.h"

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

TEST(LayoutUtilsTest, ResolveTpuLayout_StandardTensor_Success) {
  auto tensor = at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat));
  auto layout_or = ResolveTpuLayout(tensor);
  ASSERT_TRUE(layout_or.ok()) << layout_or.status();
  auto layout = layout_or.value();

  EXPECT_EQ(layout.sizes, Dimensions({2, 3}));
  EXPECT_EQ(layout.strides, Strides({3, 1}));
  EXPECT_EQ(layout.storage_offset, 0);
  EXPECT_EQ(layout.element_type, mlir::ElementType::F32);
}

TEST(LayoutUtilsTest, ResolveTpuLayout_StandardTensorZeroDim_Success) {
  auto tensor = at::empty({}, at::TensorOptions().dtype(at::kFloat));
  auto layout_or = ResolveTpuLayout(tensor);
  ASSERT_TRUE(layout_or.ok()) << layout_or.status();

  auto layout = layout_or.value();
  EXPECT_EQ(layout.sizes, Dimensions({}));
  EXPECT_EQ(layout.strides, Strides({}));
  EXPECT_EQ(layout.storage_offset, 0);
  EXPECT_EQ(layout.element_type, mlir::ElementType::F32);
}

TEST(LayoutUtilsTest, ResolveTpuLayout_FP4Tensor_Success) {
  auto tensor =
      at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2));

  auto layout_or = ResolveTpuLayout(tensor);
  ASSERT_TRUE(layout_or.ok()) << layout_or.status();
  auto layout = layout_or.value();

  EXPECT_EQ(layout.sizes, Dimensions({2, 6}));
  EXPECT_EQ(layout.strides, Strides({6, 1}));
  EXPECT_EQ(layout.storage_offset, 0);
  EXPECT_EQ(layout.element_type, mlir::ElementType::F4E2M1FN);
}

TEST(LayoutUtilsTest, ResolveTpuLayout_FP4TensorWithOffset_Success) {
  auto base_tensor =
      at::empty({10, 8}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2));
  auto tensor = base_tensor.slice(0, 1, 3).slice(1, 1, 4);

  auto layout_or = ResolveTpuLayout(tensor);
  ASSERT_TRUE(layout_or.ok()) << layout_or.status();
  auto layout = layout_or.value();

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

}  // namespace
}  // namespace torch_tpu
