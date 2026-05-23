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

#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"

#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {
namespace {

TEST(UpdateLayoutReshape, ScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  ReshapePrimitive reshape{.base_sizes = {}, .new_sizes = {}};
  EXPECT_FALSE(UpdateLayout(layout, reshape));
}

TEST(UpdateLayoutReshape, ScalarToVector) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  ReshapePrimitive reshape{.base_sizes = {}, .new_sizes = {1, 1, 1}};
  EXPECT_TRUE(UpdateLayout(layout, reshape));
  StridedLayout expected = {
      .strided_dims =
          {
              {.size = 1, .stride = 1},
              {.size = 1, .stride = 1},
              {.size = 1, .stride = 1},
          },
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutReshape, VectorToScalar) {
  StridedLayout layout = MakeContiguousBaseLayout({1, 1, 1});
  ReshapePrimitive reshape{.base_sizes = {1, 1, 1}, .new_sizes = {}};
  EXPECT_TRUE(UpdateLayout(layout, reshape));
  StridedLayout expected = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutReshape, TensorNoOp) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  ReshapePrimitive reshape = {.base_sizes = {6, 4, 2}, .new_sizes = {6, 4, 2}};
  EXPECT_FALSE(UpdateLayout(layout, reshape));
}

TEST(UpdateLayoutReshape, TensorToTensor) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  ReshapePrimitive reshape = {.base_sizes = {6, 4, 2}, .new_sizes = {2, 3, 8}};
  EXPECT_TRUE(UpdateLayout(layout, reshape));
  StridedLayout expected = {
      .strided_dims = {{.size = 2, .stride = 27},
                       {.size = 3, .stride = 9},
                       {.size = 8, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutReshape, TensorWithOnesToTensor) {
  StridedLayout layout = {
      .strided_dims = {{.size = 1, .stride = 999},
                       {.size = 1, .stride = 999},
                       {.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1},
                       {.size = 1, .stride = 999},
                       {.size = 1, .stride = 999}},
      .storage_offset = 0,
  };
  ReshapePrimitive reshape = {.base_sizes = {1, 1, 6, 4, 2, 1, 1},
                              .new_sizes = {2, 3, 8}};
  EXPECT_TRUE(UpdateLayout(layout, reshape));
  StridedLayout expected = {
      .strided_dims = {{.size = 2, .stride = 27},
                       {.size = 3, .stride = 9},
                       {.size = 8, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutReshape, TensorToTensorWithOnes) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  ReshapePrimitive reshape = {.base_sizes = {6, 4, 2},
                              .new_sizes = {1, 2, 1, 3, 1, 8, 1}};
  EXPECT_TRUE(UpdateLayout(layout, reshape));
  StridedLayout expected = {
      .strided_dims = {{.size = 1, .stride = 1},
                       {.size = 2, .stride = 27},
                       {.size = 1, .stride = 1},
                       {.size = 3, .stride = 9},
                       {.size = 1, .stride = 1},
                       {.size = 8, .stride = 1},
                       {.size = 1, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

TEST(MergeSequentialReshape, ValidMerge) {
  Dimensions contiguous_base_shape = {54};
  ReshapePrimitive current = {.base_sizes = {54}, .new_sizes = {6, 9}};
  ReshapePrimitive to_merge = {.base_sizes = {6, 9}, .new_sizes = {27, 2}};

  auto expected_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  UpdateLayout(expected_layout, current);
  UpdateLayout(expected_layout, to_merge);

  auto merged = Merge(std::move(current), std::move(to_merge));
  auto actual_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  UpdateLayout(actual_layout, merged);

  EXPECT_EQ(actual_layout, expected_layout);
}

}  // namespace
}  // namespace torch_tpu
