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

#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"

#include "gtest/gtest.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {
namespace {

TEST(UpdateLayoutPermute, ScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  TransposePrimitive transpose = {.permutation = {}};
  EXPECT_FALSE(UpdateLayout(layout, transpose));
}

TEST(UpdateLayoutPermute, TensorNoOp) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  TransposePrimitive transpose = {.permutation = {0, 1, 2}};
  EXPECT_FALSE(UpdateLayout(layout, transpose));
}

TEST(UpdateLayoutPermute, TensorToTensor) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  TransposePrimitive transpose = {.permutation = {1, 2, 0}};
  EXPECT_TRUE(UpdateLayout(layout, transpose));
  StridedLayout expected = {
      .strided_dims = {{.size = 4, .stride = 2},
                       {.size = 2, .stride = 1},
                       {.size = 6, .stride = 9}},
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

}  // namespace
}  // namespace torch_tpu
