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

#include "torch_tpu/ops/view_decomposition/view_primitive_error_utils.h"

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"

namespace torch_tpu {
namespace {

TEST(ViewPrimitiveErrorUtilsTest, GetViewPrimitiveErrorSuffix) {
  // Invalid reshape: number of elements don't match (1 != 2).
  ReshapePrimitive reshape{.base_sizes = {1}, .new_sizes = {2}};
  ViewPrimitive primitive = reshape;

  EXPECT_EQ(GetViewPrimitiveErrorSuffix(primitive),
            "; primitive=reshape(base_sizes=1, new_sizes=2); this is a "
            "TorchTPU bug");
  EXPECT_EQ(
      GetViewPrimitiveErrorSuffix(primitive, {.leading_semicolon = false}),
      "primitive=reshape(base_sizes=1, new_sizes=2); this is a "
      "TorchTPU bug");
  EXPECT_EQ(GetViewPrimitiveErrorSuffix(
                primitive, {.bug_suffix = ViewPrimitiveBugSuffix::kHide}),
            "; primitive=reshape(base_sizes=1, new_sizes=2)");
  EXPECT_EQ(GetViewPrimitiveErrorSuffix(
                primitive, {.leading_semicolon = false,
                            .bug_suffix = ViewPrimitiveBugSuffix::kHide}),
            "primitive=reshape(base_sizes=1, new_sizes=2)");
}

TEST(ViewPrimitiveErrorUtilsTest, GetUpdateLayoutBugSuffix) {
  // Invalid reshape: number of elements don't match (1 != 2).
  ReshapePrimitive reshape{.base_sizes = {1}, .new_sizes = {2}};
  ViewPrimitive primitive = reshape;
  StridedLayout layout = MakeContiguousBaseLayout({2, 3});

  EXPECT_EQ(
      GetUpdateLayoutBugSuffix(primitive, layout),
      "; calling UpdateLayout() with layout=shape: [2, 3] strides: [3, 1] "
      "storage_offset: 0 and primitive=reshape(base_sizes=1, "
      "new_sizes=2); this is a TorchTPU bug");
}

TEST(ViewPrimitiveErrorUtilsTest, GetViewPrimitiveShloErrorSuffix) {
  // Invalid reshape: number of elements don't match (1 != 2).
  ReshapePrimitive reshape{.base_sizes = {1}, .new_sizes = {2}};
  ViewPrimitive primitive = reshape;
  Dimensions shape = {2, 3};

  EXPECT_EQ(GetViewPrimitiveShloErrorSuffix(primitive, shape),
            "; calling ViewPrimitiveShlo() with input shape=[2, 3] and "
            "primitive=reshape(base_sizes=1, new_sizes=2); this is a "
            "TorchTPU bug");

  EXPECT_EQ(GetViewPrimitiveShloErrorSuffix(primitive, shape,
                                            ViewPrimitiveBugSuffix::kHide),
            "; calling ViewPrimitiveShlo() with input shape=[2, 3] and "
            "primitive=reshape(base_sizes=1, new_sizes=2)");
}

TEST(ViewPrimitiveErrorUtilsTest, GetViewPrimitiveShloErrorSuffixDynamic) {
  // Invalid reshape: number of elements don't match (1 != 2).
  ReshapePrimitive reshape{.base_sizes = {1}, .new_sizes = {2}};
  ViewPrimitive primitive = reshape;
  Dimensions shape = {2, mlir::ShapedType::kDynamic};

  EXPECT_EQ(GetViewPrimitiveShloErrorSuffix(primitive, shape),
            "; calling ViewPrimitiveShlo() with input shape=[2, dyn] and "
            "primitive=reshape(base_sizes=1, new_sizes=2); this is a "
            "TorchTPU bug");
}

}  // namespace
}  // namespace torch_tpu
