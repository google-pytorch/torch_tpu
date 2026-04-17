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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {
namespace {
using absl_testing::StatusIs;
using testing::HasSubstr;

TEST(UpdateLayoutReshape, ScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto modified = UpdateLayoutImpl(
      layout, ReshapePrimitive{.base_sizes = {}, .new_sizes = {}});
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutReshape, ScalarToVector) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  auto modified = UpdateLayoutImpl(
      layout, ReshapePrimitive{.base_sizes = {}, .new_sizes = {1, 1, 1}});
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
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
  auto modified = UpdateLayoutImpl(
      layout, ReshapePrimitive{.base_sizes = {1, 1, 1}, .new_sizes = {}});
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutReshape, InvalidScalarToVector) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  EXPECT_THAT(
      UpdateLayoutImpl(layout,
                       ReshapePrimitive{.base_sizes = {}, .new_sizes = {2}}),
      StatusIs(error::kInvalidArgument,
               HasSubstr("reshape does not match the number of elements")));
}

TEST(UpdateLayoutReshape, InvalidVectorToScalar) {
  StridedLayout layout = MakeContiguousBaseLayout({2});
  EXPECT_THAT(
      UpdateLayoutImpl(layout,
                       ReshapePrimitive{.base_sizes = {2}, .new_sizes = {}}),
      StatusIs(error::kInvalidArgument,
               HasSubstr("reshape does not match the number of elements")));
}

TEST(UpdateLayoutReshape, TensorNoOp) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  ReshapePrimitive reshape = {.base_sizes = {6, 4, 2}, .new_sizes = {6, 4, 2}};
  auto modified = UpdateLayoutImpl(layout, reshape);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutReshape, TensorToTensor) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  ReshapePrimitive reshape = {.base_sizes = {6, 4, 2}, .new_sizes = {2, 3, 8}};
  auto modified = UpdateLayoutImpl(layout, reshape);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
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
  auto modified = UpdateLayoutImpl(layout, reshape);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
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
  auto modified = UpdateLayoutImpl(layout, reshape);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
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

TEST(UpdateLayoutReshape, InvalidViolatesContinguity) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  ReshapePrimitive reshape = {.base_sizes = {6, 4, 2}, .new_sizes = {24, 2}};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, reshape),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("reshape is not aligned with contiguity-like blocks")));
}

TEST(UpdateLayoutReshape, InvalidBaseRank) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  ReshapePrimitive reshape = {.base_sizes = {6, 4}, .new_sizes = {24, 1}};
  EXPECT_THAT(UpdateLayoutImpl(layout, reshape),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("reshape base sizes and layout rank must "
                                 "match")));
}

TEST(UpdateLayoutReshape, InvalidBaseSizes) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  ReshapePrimitive reshape = {.base_sizes = {6, 4, 1}, .new_sizes = {24, 1}};
  EXPECT_THAT(UpdateLayoutImpl(layout, reshape),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("reshape base sizes must match the layout")));
}

TEST(MergeSequentialReshape, ValidMerge) {
  Dimensions contiguous_base_shape = {54};
  ReshapePrimitive current = {.base_sizes = {54}, .new_sizes = {6, 9}};
  ReshapePrimitive to_merge = {.base_sizes = {6, 9}, .new_sizes = {27, 2}};

  auto expected_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  EXPECT_EQ(UpdateLayoutImpl(expected_layout, current).status(),
            absl::OkStatus());
  EXPECT_EQ(UpdateLayoutImpl(expected_layout, to_merge).status(),
            absl::OkStatus());

  auto merged = Merge(std::move(current), std::move(to_merge));
  EXPECT_EQ(merged.status(), absl::OkStatus());

  auto actual_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  EXPECT_EQ(UpdateLayoutImpl(actual_layout, merged.value()).status(),
            absl::OkStatus());

  EXPECT_EQ(actual_layout, expected_layout);
}

TEST(MergeSequentialReshape, InvalidMerge) {
  ReshapePrimitive current = {.new_sizes = {6, 9}};
  ReshapePrimitive to_merge = {.new_sizes = {27, 3}};
  auto merged = Merge(std::move(current), std::move(to_merge));
  EXPECT_THAT(
      merged.status(),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("sequential reshapes must have matching element counts")));
}

}  // namespace
}  // namespace torch_tpu
