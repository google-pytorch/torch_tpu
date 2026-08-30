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

#include "csrc/ops/view_decomposition/unfold_primitive.h"

#include "csrc/ops/view_decomposition/strided_layout.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

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
  EXPECT_FALSE(UpdateLayout(layout, unfold));
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
  EXPECT_TRUE(UpdateLayout(layout, unfold));
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
  EXPECT_TRUE(UpdateLayout(layout, unfold));
  StridedLayout expected = {
      .strided_dims = {{.size = 2, .stride = 14},
                       {.size = 2, .stride = 4},
                       {.size = 3, .stride = 2}},
      .storage_offset = 2,
  };
  EXPECT_EQ(layout, expected);
}

}  // namespace
}  // namespace torch_tpu
