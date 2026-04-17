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

#include "torch_tpu/ops/view_decomposition/broadcast_primitive.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {
namespace {
using absl_testing::StatusIs;
using testing::HasSubstr;

TEST(UpdateLayoutBroadcast, ScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  BroadcastPrimitive broadcast = {.new_sizes = {}, .broadcast_dimensions = {}};
  auto modified = UpdateLayoutImpl(layout, broadcast);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutBroadcast, TensorNoOp) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  BroadcastPrimitive broadcast = {.new_sizes = {6, 4, 2},
                                  .broadcast_dimensions = {0, 1, 2}};
  auto modified = UpdateLayoutImpl(layout, broadcast);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutBroadcast, ScalarToTensor) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  BroadcastPrimitive broadcast = {.new_sizes = {2, 3, 4},
                                  .broadcast_dimensions = {}};
  auto modified = UpdateLayoutImpl(layout, broadcast);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = {
      .strided_dims = {{.size = 2, .stride = 0},
                       {.size = 3, .stride = 0},
                       {.size = 4, .stride = 0}},
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutBroadcast, TensorToTensor) {
  StridedLayout layout = {
      .strided_dims = {{.size = 1, .stride = 999},
                       {.size = 2, .stride = 1},
                       {.size = 3, .stride = 2}},
      .storage_offset = 0,
  };
  BroadcastPrimitive broadcast = {.new_sizes = {6, 3, 4, 2},
                                  .broadcast_dimensions = {0, 3, 1}};
  auto modified = UpdateLayoutImpl(layout, broadcast);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = {
      .strided_dims = {{.size = 6, .stride = 0},   // expanded
                       {.size = 3, .stride = 2},   // transposed
                       {.size = 4, .stride = 0},   // inserted
                       {.size = 2, .stride = 1}},  // transposed
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutBroadcast, InvalidRank) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  BroadcastPrimitive broadcast = {.new_sizes = {6, 4, 2, 1},
                                  .broadcast_dimensions = {0, 1, 2, 3}};
  EXPECT_THAT(UpdateLayoutImpl(layout, broadcast),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("broadcast_dimensions must have the same size "
                                 "as the input tensor")));
}

TEST(UpdateLayoutBroadcast, InvalidNegativeIndex) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  BroadcastPrimitive broadcast = {.new_sizes = {6, 4, 2},
                                  .broadcast_dimensions = {0, 1, -1}};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, broadcast),
      StatusIs(error::kInvalidArgument,
               HasSubstr("broadcast_dimensions[2] = -1 which is out of bounds "
                         "for new_sizes of size 3")));
}

TEST(UpdateLayoutBroadcast, InvalidIndexOutOfBounds) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  BroadcastPrimitive broadcast = {.new_sizes = {6, 4, 2},
                                  .broadcast_dimensions = {0, 1, 3}};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, broadcast),
      StatusIs(error::kInvalidArgument,
               HasSubstr("broadcast_dimensions[2] = 3 which is out of bounds "
                         "for new_sizes of size 3")));
}

TEST(UpdateLayoutBroadcast, InvalidDuplicateDimension) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 2, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  BroadcastPrimitive broadcast = {.new_sizes = {6, 2, 2},
                                  .broadcast_dimensions = {0, 1, 1}};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, broadcast),
      StatusIs(error::kInvalidArgument,
               HasSubstr("broadcast_dimensions has a duplicate input index")));
}

TEST(UpdateLayoutBroadcast, InvalidSizeNotOne) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  BroadcastPrimitive broadcast = {.new_sizes = {6, 4, 3},
                                  .broadcast_dimensions = {0, 1, 2}};
  EXPECT_THAT(UpdateLayoutImpl(layout, broadcast),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("cannot broadcast input dimension 2 of size 2 "
                                 "to output dimension 2 of size 3")));
}

}  // namespace
}  // namespace torch_tpu
