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

#include "torch_tpu/ops/view_decomposition/unfold_primitive.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {
namespace {
using absl_testing::StatusIs;
using testing::HasSubstr;

TEST(UpdateLayoutUnfold, TensorNoOp) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 5},
                       {.size = 1, .stride = 999},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {.start_index = 0,
                            .limit_index = 5,
                            .window_stride = 999,
                            .window_size = 5};
  auto modified = UpdateLayoutImpl(layout, unfold);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutUnfold, TensorValidUnfold) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 7},
                       {.size = 1, .stride = 999},
                       {.size = 7, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {
      .start_index = 1, .limit_index = 6, .window_stride = 2, .window_size = 3};
  auto modified = UpdateLayoutImpl(layout, unfold);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = {
      .strided_dims = {{.size = 2, .stride = 7},
                       {.size = 2, .stride = 2},
                       {.size = 3, .stride = 1}},
      .storage_offset = 1,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutUnfold, UnfoldWithStrideOnLastDimension) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 14},
                       {.size = 1, .stride = 999},
                       {.size = 7, .stride = 2}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {
      .start_index = 1, .limit_index = 6, .window_stride = 2, .window_size = 3};
  auto modified = UpdateLayoutImpl(layout, unfold);
  ASSERT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = {
      .strided_dims = {{.size = 2, .stride = 14},
                       {.size = 2, .stride = 4},
                       {.size = 3, .stride = 2}},
      .storage_offset = 2,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutUnfold, InvalidScalar) {
  StridedLayout layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {
      .start_index = 0, .limit_index = 0, .window_stride = 1, .window_size = 1};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, unfold),
      StatusIs(error::kInvalidArgument,
               HasSubstr("expected input to have at least 2 dimensions")));
}

TEST(UpdateLayoutUnfold, InvalidRankOne) {
  StridedLayout layout = {
      .strided_dims = {{.size = 10, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {.start_index = 0,
                            .limit_index = 10,
                            .window_stride = 1,
                            .window_size = 1};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, unfold),
      StatusIs(error::kInvalidArgument,
               HasSubstr("expected input to have at least 2 dimensions")));
}

TEST(UpdateLayoutUnfold, InvalidPenultimateSize) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 10},
                       {.size = 2, .stride = 5},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {
      .start_index = 0, .limit_index = 5, .window_stride = 1, .window_size = 1};
  EXPECT_THAT(UpdateLayoutImpl(layout, unfold),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("expected input to have size 1 in the "
                                 "second-to-last dimension")));
}

TEST(UpdateLayoutUnfold, InvalidStartIndex) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 5},
                       {.size = 1, .stride = 999},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {
      .start_index = 5, .limit_index = 5, .window_stride = 1, .window_size = 1};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, unfold),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("start index 5 is out of bounds for dimension of size 5")));
}

TEST(UpdateLayoutUnfold, InvalidLimitIndexBelowStartIndex) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 5},
                       {.size = 1, .stride = 999},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {
      .start_index = 2, .limit_index = 1, .window_stride = 1, .window_size = 1};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, unfold),
      StatusIs(error::kInvalidArgument,
               HasSubstr("limit index cannot be less than start index")));
}

TEST(UpdateLayoutUnfold, InvalidLimitIndexOutOfBounds) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 5},
                       {.size = 1, .stride = 999},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {
      .start_index = 0, .limit_index = 6, .window_stride = 1, .window_size = 1};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, unfold),
      StatusIs(
          error::kInvalidArgument,
          HasSubstr("limit index 6 is out of bounds for dimension of size 5")));
}

TEST(UpdateLayoutUnfold, InvalidWindowSize) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 5},
                       {.size = 1, .stride = 999},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {
      .start_index = 1, .limit_index = 2, .window_stride = 1, .window_size = 2};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, unfold),
      StatusIs(error::kInvalidArgument,
               HasSubstr("window size 2 is larger than the range [1, 2)")));
}

TEST(UpdateLayoutUnfold, InvalidWindowStrideZero) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 5},
                       {.size = 1, .stride = 999},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {
      .start_index = 1, .limit_index = 5, .window_stride = 0, .window_size = 2};
  EXPECT_THAT(UpdateLayoutImpl(layout, unfold),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("window stride must be positive")));
}

TEST(UpdateLayoutUnfold, InvalidWindowStrideNegative) {
  StridedLayout layout = {
      .strided_dims = {{.size = 2, .stride = 5},
                       {.size = 1, .stride = 999},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  UnfoldPrimitive unfold = {.start_index = 1,
                            .limit_index = 5,
                            .window_stride = -1,
                            .window_size = 2};
  EXPECT_THAT(UpdateLayoutImpl(layout, unfold),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("window stride must be positive")));
}

}  // namespace
}  // namespace torch_tpu
