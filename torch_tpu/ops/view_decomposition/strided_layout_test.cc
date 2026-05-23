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

#include "torch_tpu/ops/view_decomposition/strided_layout.h"

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/dimension_types.h"

namespace torch_tpu {
namespace {

TEST(MakeContiguousBaseLayout, ScalarLayout) {
  auto layout = MakeContiguousBaseLayout({});
  EXPECT_EQ(layout.storage_offset, 0);
  EXPECT_TRUE(layout.strided_dims.empty());
}

TEST(MakeContiguousBaseLayout, VectorLayout) {
  StridedLayout actual = MakeContiguousBaseLayout({100});
  StridedLayout expected = {
      .strided_dims =
          {
              {.size = 100, .stride = 1},
          },
      .storage_offset = 0,
  };
  EXPECT_EQ(actual, expected);
}

TEST(MakeContiguousBaseLayout, TensorLayout) {
  StridedLayout actual = MakeContiguousBaseLayout({2, 3, 4});
  StridedLayout expected = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_EQ(actual, expected);
}

TEST(StridedLayoutEquality, Equal) {
  StridedLayout lhs = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  StridedLayout rhs = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_EQ(lhs, rhs);
}

TEST(StridedLayoutEquality, UnequalRank) {
  StridedLayout lhs = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  StridedLayout rhs = {
      .strided_dims = {{.size = 2, .stride = 12}, {.size = 3, .stride = 4}},
      .storage_offset = 0,
  };
  EXPECT_NE(lhs, rhs);
}

TEST(StridedLayoutEquality, UnequalSize) {
  StridedLayout lhs = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  StridedLayout rhs = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_NE(lhs, rhs);
}

TEST(StridedLayoutEquality, UnequalStride) {
  StridedLayout lhs = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  StridedLayout rhs = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 2}},
      .storage_offset = 0,
  };
  EXPECT_NE(lhs, rhs);
}

TEST(StridedLayoutEquality, UnequalOffset) {
  StridedLayout lhs = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  StridedLayout rhs = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 1,
  };
  EXPECT_NE(lhs, rhs);
}

TEST(StridedLayoutEquality, SizeOneStrideIgnored) {
  StridedLayout lhs = {
      .strided_dims = {{.size = 1, .stride = 1234567890},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  StridedLayout rhs = {
      .strided_dims = {{.size = 1, .stride = 9876543210},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_EQ(lhs, rhs);
}

TEST(GetSizes, ReturnsDimensionSizes) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_EQ(GetSizes(layout), Dimensions({2, 3, 4}));
}

TEST(GetSizes, EmptyLayout) {
  StridedLayout layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  EXPECT_TRUE(GetSizes(layout).empty());
}

TEST(GetStrides, ReturnsDimensionStrides) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_EQ(GetStrides(layout), Strides({12, 4, 1}));
}

TEST(GetStrides, EmptyLayout) {
  StridedLayout layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  EXPECT_TRUE(GetStrides(layout).empty());
}

}  // namespace
}  // namespace torch_tpu
