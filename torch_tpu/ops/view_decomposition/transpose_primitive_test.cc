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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {
namespace {
using absl_testing::StatusIs;
using testing::HasSubstr;

TEST(UpdateLayoutPermute, ScalarNoOp) {
  StridedLayout layout = MakeContiguousBaseLayout({});
  TransposePrimitive transpose = {.permutation = {}};
  auto modified = UpdateLayoutImpl(layout, transpose);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutPermute, TensorNoOp) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  TransposePrimitive transpose = {.permutation = {0, 1, 2}};
  auto modified = UpdateLayoutImpl(layout, transpose);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_FALSE(modified.value());
}

TEST(UpdateLayoutPermute, TensorToTensor) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  TransposePrimitive transpose = {.permutation = {1, 2, 0}};
  auto modified = UpdateLayoutImpl(layout, transpose);
  EXPECT_EQ(modified.status(), absl::OkStatus());
  EXPECT_TRUE(modified.value());
  StridedLayout expected = {
      .strided_dims = {{.size = 4, .stride = 2},
                       {.size = 2, .stride = 1},
                       {.size = 6, .stride = 9}},
      .storage_offset = 0,
  };
  EXPECT_EQ(layout, expected);
}

TEST(UpdateLayoutPermute, InvalidDifferentRank) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  TransposePrimitive transpose = {.permutation = {0, 1, 2, 3}};
  EXPECT_THAT(UpdateLayoutImpl(layout, transpose),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("transpose has wrong number of axes")));
}

TEST(UpdateLayoutPermute, InvalidNegativeIndex) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  TransposePrimitive transpose = {.permutation = {-1, 0, 2}};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, transpose),
      StatusIs(error::kInvalidArgument,
               HasSubstr("permutation dimension index is out of bounds")));
}

TEST(UpdateLayoutPermute, InvalidIndexTooLarge) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  TransposePrimitive transpose = {.permutation = {0, 1, 3}};
  EXPECT_THAT(
      UpdateLayoutImpl(layout, transpose),
      StatusIs(error::kInvalidArgument,
               HasSubstr("permutation dimension index is out of bounds")));
}

TEST(UpdateLayoutPermute, InvalidIndexRepeated) {
  StridedLayout layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  TransposePrimitive transpose = {.permutation = {0, 1, 0}};
  EXPECT_THAT(UpdateLayoutImpl(layout, transpose),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("transpose has duplicate axis dimension")));
}

}  // namespace
}  // namespace torch_tpu
