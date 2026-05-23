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

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {
namespace {

TEST(UpdateLayoutBroadcast, ScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  BroadcastPrimitive broadcast = {.new_sizes = {}, .broadcast_dimensions = {}};
  EXPECT_FALSE(UpdateLayout(layout, broadcast));
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
  EXPECT_FALSE(UpdateLayout(layout, broadcast));
}

TEST(UpdateLayoutBroadcast, ScalarToTensor) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  BroadcastPrimitive broadcast = {.new_sizes = {2, 3, 4},
                                  .broadcast_dimensions = {}};
  EXPECT_TRUE(UpdateLayout(layout, broadcast));
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
  EXPECT_TRUE(UpdateLayout(layout, broadcast));
  StridedLayout expected = {
      .strided_dims = {{.size = 6, .stride = 0},   // expanded
                       {.size = 3, .stride = 2},   // transposed
                       {.size = 4, .stride = 0},   // inserted
                       {.size = 2, .stride = 1}},  // transposed
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

}  // namespace
}  // namespace torch_tpu
