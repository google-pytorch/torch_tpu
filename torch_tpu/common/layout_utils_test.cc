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

#include "torch_tpu/common/layout_utils.h"

#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "c10/core/ScalarType.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"

namespace torch_tpu {
namespace {

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
