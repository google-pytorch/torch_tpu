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

#include "torch_tpu/ops/view_decomposition/pad_primitive.h"

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {
namespace {

TEST(UpdateLayoutPad, ScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  PadPrimitive pad = {.pad_dims = {}};
  EXPECT_FALSE(UpdateLayout(layout, pad));
}

TEST(UpdateLayoutPad, TensorNoOp) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  PadPrimitive pad = {
      .pad_dims = {
          {.low_padding = 0, .high_padding = 0, .interior_padding = 0},
          {.low_padding = 0, .high_padding = 0, .interior_padding = 0},
          {.low_padding = 0, .high_padding = 0, .interior_padding = 0}}};
  EXPECT_FALSE(UpdateLayout(layout, pad));
}

TEST(UpdateLayoutPad, TensorToTensor) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 1,
  };
  PadPrimitive pad = {
      .pad_dims = {
          {.low_padding = 1, .high_padding = 1, .interior_padding = 1},
          {.low_padding = 1, .high_padding = 1, .interior_padding = 1},
          {.low_padding = 1, .high_padding = 1, .interior_padding = 1}}};
  EXPECT_TRUE(UpdateLayout(layout, pad));
  StridedLayout expected = {
      .strided_dims = {{.size = 13, .stride = 45},
                       {.size = 9, .stride = 5},
                       {.size = 5, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

}  // namespace
}  // namespace torch_tpu
