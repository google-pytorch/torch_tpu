/*
 * Copyright 2026 Google LLC
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

#include "csrc/eager/tensor_to_buffer.h"

#include <string>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/cleanup/cleanup.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

TEST(TensorToBufferTest, GetBufferContiguous) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  // Create a buffer and lift it to a contiguous tensor.
  TT_ASSERT_OK_AND_ASSIGN(auto dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor = MakeTensor(dummy_buffer);

  // Reading the buffer from the tensor should yield the same buffer.
  TT_ASSERT_OK_AND_ASSIGN(auto read_buffer, GetBuffer(tensor));
  EXPECT_EQ(read_buffer, dummy_buffer);
}

TEST(TensorToBufferTest, GetBufferContiguousDonated) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  // Create a buffer and lift it to a contiguous tensor.
  TT_ASSERT_OK_AND_ASSIGN(auto dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor = MakeTensor(dummy_buffer);

  // Reading the buffer from the tensor should yield the same buffer.
  // Since no view was created, the donation annotation is ignored.
  TT_ASSERT_OK_AND_ASSIGN(auto read_buffer,
                          GetBuffer(tensor, DonateBase::kDonate));
  EXPECT_EQ(read_buffer, dummy_buffer);
}

TEST(TensorToBufferTest, GetBufferNonContiguous) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  // Create a buffer and lift it to a contiguous tensor.
  TT_ASSERT_OK_AND_ASSIGN(auto dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor = MakeTensor(dummy_buffer);
  // Make the tensor no longer contiguous.
  tensor.t_();

  // Reading the buffer from the tensor should yield a deferred view.
  TT_ASSERT_OK_AND_ASSIGN(auto read_buffer, GetBuffer(tensor));
  EXPECT_NE(read_buffer, dummy_buffer);
  auto read_deferred_op = read_buffer.deferred_op();
  ASSERT_TRUE(read_deferred_op != nullptr);
  EXPECT_EQ(read_deferred_op->op_name(), OpName::kAsStrided);
  // The input should not be donated.
  EXPECT_TRUE(read_deferred_op->donated_indices().empty());
}

TEST(TensorToBufferTest, GetBufferNonContiguousDonated) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  // Create a buffer and lift it to a contiguous tensor.
  TT_ASSERT_OK_AND_ASSIGN(auto dummy_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor = MakeTensor(dummy_buffer);
  // Make the tensor no longer contiguous.
  tensor.t_();

  // Reading the buffer from the tensor should yield a deferred view.
  TT_ASSERT_OK_AND_ASSIGN(auto read_buffer,
                          GetBuffer(tensor, DonateBase::kDonate));
  EXPECT_NE(read_buffer, dummy_buffer);
  auto read_deferred_op = read_buffer.deferred_op();
  ASSERT_TRUE(read_deferred_op != nullptr);
  EXPECT_EQ(read_deferred_op->op_name(), OpName::kAsStrided);
  // The input should be donated.
  EXPECT_THAT(read_deferred_op->donated_indices(), testing::ElementsAre(0));
}

TEST(TensorToBufferTest, GetBuffers) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  // Create two buffers, one contiguous and one not.
  TT_ASSERT_OK_AND_ASSIGN(auto base_buffer_a,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(auto base_buffer_b,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor_a = MakeTensor(base_buffer_a);
  at::Tensor tensor_b = MakeTensor(base_buffer_b);
  tensor_b.t_();
  std::vector<at::Tensor> tensors = {tensor_a, tensor_b};

  TT_ASSERT_OK_AND_ASSIGN(auto read_buffers, GetBuffers(tensors));
  ASSERT_EQ(read_buffers.size(), 2);
  // a is contiguous, so reading it should yield the same buffer.
  EXPECT_EQ(read_buffers[0], base_buffer_a);

  // b is non-contiguous, so reading it should yield a deferred view.
  EXPECT_NE(read_buffers[1], base_buffer_b);
  auto read_deferred_op = read_buffers[1].deferred_op();
  ASSERT_TRUE(read_deferred_op != nullptr);
  EXPECT_EQ(read_deferred_op->op_name(), OpName::kAsStrided);
  ASSERT_EQ(read_deferred_op->inputs().size(), 1);
  EXPECT_EQ(read_deferred_op->inputs()[0], base_buffer_b);
}

TEST(TensorToBufferTest, GetBuffersInvalidDonation) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  // Create two buffers, one contiguous and one not.
  TT_ASSERT_OK_AND_ASSIGN(auto base_buffer_a,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(auto base_buffer_b,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor_a = MakeTensor(base_buffer_a);
  at::Tensor tensor_b = MakeTensor(base_buffer_b);
  tensor_b.t_();
  std::vector<at::Tensor> tensors = {tensor_a, tensor_b};

  // Pass in invalid donation indices.
  Indices donated_indices = {-1, 999};
  TT_ASSERT_OK_AND_ASSIGN(auto read_buffers,
                          GetBuffers(tensors, donated_indices));

  // Invalid donation indices should be ignored.
  ASSERT_EQ(read_buffers.size(), 2);
  // a is contiguous, so reading it should yield the same buffer.
  EXPECT_EQ(read_buffers[0], base_buffer_a);

  // b is non-contiguous, so reading it should yield a deferred view.
  EXPECT_NE(read_buffers[1], base_buffer_b);
  auto read_deferred_op = read_buffers[1].deferred_op();
  ASSERT_TRUE(read_deferred_op != nullptr);
  EXPECT_EQ(read_deferred_op->op_name(), OpName::kAsStrided);
  ASSERT_EQ(read_deferred_op->inputs().size(), 1);
  EXPECT_EQ(read_deferred_op->inputs()[0], base_buffer_b);
}

TEST(TensorToBufferTest, GetBuffersDonated) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  // Create two buffers, one contiguous and one not.
  TT_ASSERT_OK_AND_ASSIGN(auto base_buffer_a,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(auto base_buffer_b,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor_a = MakeTensor(base_buffer_a);
  at::Tensor tensor_b = MakeTensor(base_buffer_b);
  tensor_b.t_();
  std::vector<at::Tensor> tensors = {tensor_a, tensor_b};

  // Indicate that both tensors should be donated.
  Indices donated_indices = {0, 1};
  TT_ASSERT_OK_AND_ASSIGN(auto read_buffers,
                          GetBuffers(tensors, donated_indices));

  ASSERT_EQ(read_buffers.size(), 2);
  // a is contiguous, so reading it should yield the same buffer.
  // No view is constructed, so the donation is ignored.
  EXPECT_EQ(read_buffers[0], base_buffer_a);

  // b is non-contiguous, so reading it should yield a deferred view.
  // The input is donated, so the view op is marked as such.
  EXPECT_NE(read_buffers[1], base_buffer_b);
  auto read_deferred_op = read_buffers[1].deferred_op();
  ASSERT_TRUE(read_deferred_op != nullptr);
  EXPECT_EQ(read_deferred_op->op_name(), OpName::kAsStrided);
  ASSERT_EQ(read_deferred_op->inputs().size(), 1);
  EXPECT_EQ(read_deferred_op->inputs()[0], base_buffer_b);
  EXPECT_THAT(read_deferred_op->donated_indices(), testing::ElementsAre(0));
}

TEST(TensorToBufferTest, GetBuffersDonatedAliased) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  // Create one buffer with three tensors: one contiguous, and two strided
  // in the same way.
  TT_ASSERT_OK_AND_ASSIGN(auto base_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor_a = MakeTensor(base_buffer);
  at::Tensor tensor_b = MakeTensor(base_buffer);
  tensor_b.t_();
  at::Tensor tensor_c = MakeTensor(base_buffer);
  tensor_c.t_();
  std::vector<at::Tensor> tensors = {tensor_a, tensor_b, tensor_c};

  // Indicate that all tensors should be donated.
  Indices donated_indices = {0, 1, 2};
  TT_ASSERT_OK_AND_ASSIGN(auto read_buffers,
                          GetBuffers(tensors, donated_indices));

  ASSERT_EQ(read_buffers.size(), 3);
  // a is contiguous, so reading it should yield the same buffer.
  // No view is constructed, so the donation is ignored.
  EXPECT_EQ(read_buffers[0], base_buffer);

  // b is non-contiguous, so reading it should yield a deferred view.
  // The input is donated, so the view op is marked as such.
  EXPECT_NE(read_buffers[1], base_buffer);
  auto read1_deferred_op = read_buffers[1].deferred_op();
  ASSERT_TRUE(read1_deferred_op != nullptr);
  EXPECT_EQ(read1_deferred_op->op_name(), OpName::kAsStrided);
  ASSERT_EQ(read1_deferred_op->inputs().size(), 1);
  EXPECT_EQ(read1_deferred_op->inputs()[0], base_buffer);
  EXPECT_THAT(read1_deferred_op->donated_indices(), testing::ElementsAre(0));

  // c is constructed as a separate view which also wants its input to be
  // donated.
  // This is valid if and only if these views are consumed by a single output
  // operation that is executed as a single graph, with a single output to
  // donate to.
  EXPECT_NE(read_buffers[2], base_buffer);
  EXPECT_NE(read_buffers[2], read_buffers[1]);
  auto read2_deferred_op = read_buffers[2].deferred_op();
  ASSERT_TRUE(read2_deferred_op != nullptr);
  EXPECT_EQ(read2_deferred_op->op_name(), OpName::kAsStrided);
  ASSERT_EQ(read2_deferred_op->inputs().size(), 1);
  EXPECT_EQ(read2_deferred_op->inputs()[0], base_buffer);
  EXPECT_THAT(read2_deferred_op->donated_indices(), testing::ElementsAre(0));
}

}  // namespace
}  // namespace torch_tpu
