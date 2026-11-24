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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/cleanup/cleanup.h"
#include "absl/types/span.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/intrusive_ptr.h"
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
#include "xla/tsl/framework/allocator.h"

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

TEST(TensorToBufferTest, GetBufferWithoutCreatingViewReturnsContiguousBase) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  const auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });

  TT_ASSERT_OK_AND_ASSIGN(
      const DeviceBufferRef dummy_buffer,
      CreateEmptyDeviceBufferRef({2, 3}, mlir::ElementType::F32));
  const at::Tensor tensor = MakeTensor(dummy_buffer);

  TT_ASSERT_OK_AND_ASSIGN(const std::optional<DeviceBufferRef> read_buffer,
                          GetBufferWithoutCreatingView(tensor));
  EXPECT_THAT(read_buffer, testing::Optional(dummy_buffer));
  EXPECT_EQ(dummy_buffer.device_buffer_list()->num_child_ops(), 0);
}

TEST(TensorToBufferTest,
     GetBufferWithoutCreatingViewReturnsNulloptForViewsWithoutOp) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  const auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });

  TT_ASSERT_OK_AND_ASSIGN(
      const DeviceBufferRef dummy_buffer,
      CreateEmptyDeviceBufferRef({2, 3}, mlir::ElementType::F32));
  const at::Tensor base_tensor = MakeTensor(dummy_buffer);

  // Non-contiguous (transposed) view.
  at::Tensor transposed = MakeTensor(dummy_buffer);
  transposed.t_();
  TT_ASSERT_OK_AND_ASSIGN(const std::optional<DeviceBufferRef> transposed_buf,
                          GetBufferWithoutCreatingView(transposed));
  EXPECT_EQ(transposed_buf, std::nullopt);

  // Contiguous reshaped view with different dimensions.
  at::Tensor reshaped(c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(base_tensor.storage()), base_tensor.key_set(),
      base_tensor.dtype()));
  reshaped.unsafeGetTensorImpl()->set_sizes_contiguous({6});
  TT_ASSERT_OK_AND_ASSIGN(const std::optional<DeviceBufferRef> reshaped_buf,
                          GetBufferWithoutCreatingView(reshaped));
  EXPECT_EQ(reshaped_buf, std::nullopt);

  // Contiguous view with non-zero storage offset.
  at::Tensor offset_slice(c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(base_tensor.storage()), base_tensor.key_set(),
      base_tensor.dtype()));
  offset_slice.unsafeGetTensorImpl()->set_sizes_contiguous({1, 3});
  offset_slice.unsafeGetTensorImpl()->set_storage_offset(3);
  TT_ASSERT_OK_AND_ASSIGN(const std::optional<DeviceBufferRef> offset_buf,
                          GetBufferWithoutCreatingView(offset_slice));
  EXPECT_EQ(offset_buf, std::nullopt);

  // Contiguous view with reinterpreted element type.
  at::Tensor reinterpreted(c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(base_tensor.storage()), base_tensor.key_set(),
      c10::scalarTypeToTypeMeta(at::kInt)));
  reinterpreted.unsafeGetTensorImpl()->set_sizes_contiguous({2, 3});
  TT_ASSERT_OK_AND_ASSIGN(
      const std::optional<DeviceBufferRef> reinterpreted_buf,
      GetBufferWithoutCreatingView(reinterpreted));
  EXPECT_EQ(reinterpreted_buf, std::nullopt);

  // Conjugated view on a complex buffer.
  TT_ASSERT_OK_AND_ASSIGN(
      const DeviceBufferRef complex_buffer,
      CreateEmptyDeviceBufferRef({2, 3}, mlir::ElementType::COMPLEXF32));
  at::Tensor conjugated = MakeTensor(complex_buffer);
  conjugated.unsafeGetTensorImpl()->_set_conj(true);
  TT_ASSERT_OK_AND_ASSIGN(const std::optional<DeviceBufferRef> conjugated_buf,
                          GetBufferWithoutCreatingView(conjugated));
  EXPECT_EQ(conjugated_buf, std::nullopt);

  // None of the view lookups should have created a deferred child op.
  EXPECT_EQ(dummy_buffer.device_buffer_list()->num_child_ops(), 0);
  EXPECT_EQ(complex_buffer.device_buffer_list()->num_child_ops(), 0);
}

TEST(TensorToBufferTest, GetBufferWithoutCreatingViewRejectsInvalidTensors) {
  const at::Tensor undefined_tensor;  // UNINITIALIZED_TENSOR_OK
  EXPECT_FALSE(GetBufferWithoutCreatingView(undefined_tensor).ok());

  // A non-TPU tensor must still return an error even when non-contiguous /
  // non-zero offset (not std::nullopt).
  at::Tensor cpu_tensor(c10::make_intrusive<c10::TensorImpl>(
      c10::Storage::create_legacy(c10::Device(c10::DeviceType::CPU)),
      c10::DispatchKeySet(c10::DispatchKey::CPU),
      c10::scalarTypeToTypeMeta(at::kFloat)));
  cpu_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      at::IntArrayRef({2, 3}), at::IntArrayRef({1, 2}));
  cpu_tensor.unsafeGetTensorImpl()->set_storage_offset(1);
  EXPECT_FALSE(GetBufferWithoutCreatingView(cpu_tensor).ok());
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

  // Create one buffer with three aliased tensors sharing the same storage:
  // one contiguous, and two strided in the same way.
  TT_ASSERT_OK_AND_ASSIGN(auto base_buffer,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor_a = MakeTensor(base_buffer);
  auto make_alias = [](const at::Tensor& src) {
    at::Tensor alias(c10::make_intrusive<c10::TensorImpl>(
        c10::Storage(src.storage()), src.key_set(), src.dtype()));
    alias.unsafeGetTensorImpl()->set_sizes_contiguous(src.sizes());
    return alias;
  };
  at::Tensor tensor_b = make_alias(tensor_a);
  tensor_b.t_();
  at::Tensor tensor_c = make_alias(tensor_a);
  tensor_c.t_();
  ASSERT_TRUE(tensor_a.is_alias_of(tensor_b));
  ASSERT_TRUE(tensor_a.is_alias_of(tensor_c));
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

TEST(TensorToBufferTest, AssignBufferToAtTensorBumpVersion) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  TT_ASSERT_OK_AND_ASSIGN(auto buf0,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(auto buf1,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(auto buf2,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(auto buf3,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  at::Tensor tensor = MakeTensor(buf0);
  c10::Storage storage_copy = tensor.storage();
  at::Tensor view(c10::make_intrusive<c10::TensorImpl>(
      c10::TensorImpl::VIEW, std::move(storage_copy), tensor.key_set(),
      tensor.dtype()));
  view.unsafeGetTensorImpl()->set_version_counter(
      tensor.unsafeGetTensorImpl()->version_counter());
  uint32_t initial_version =
      tensor.unsafeGetTensorImpl()->version_counter().current_version();
  EXPECT_EQ(view.unsafeGetTensorImpl()->version_counter().current_version(),
            initial_version);

  // By default (bump_version = true), assigning a buffer bumps the version on
  // both the tensor and its view.
  ASSERT_TRUE(AssignBufferToAtTensor(buf1, tensor).ok());
  TT_ASSERT_OK_AND_ASSIGN(auto read_buf1, GetBuffer(tensor));
  EXPECT_EQ(read_buf1, buf1);
  EXPECT_EQ(tensor.unsafeGetTensorImpl()->version_counter().current_version(),
            initial_version + 1);
  EXPECT_EQ(view.unsafeGetTensorImpl()->version_counter().current_version(),
            initial_version + 1);

  // With bump_version = false, the buffer is updated without bumping the
  // version counter on either the tensor or its view.
  ASSERT_TRUE(AssignBufferToAtTensor(buf2, tensor,
                                     /*enable_autodonation=*/false,
                                     /*bump_version=*/false)
                  .ok());
  TT_ASSERT_OK_AND_ASSIGN(auto read_buf2, GetBuffer(tensor));
  EXPECT_EQ(read_buf2, buf2);
  EXPECT_EQ(tensor.unsafeGetTensorImpl()->version_counter().current_version(),
            initial_version + 1);
  EXPECT_EQ(view.unsafeGetTensorImpl()->version_counter().current_version(),
            initial_version + 1);

  // A subsequent assignment with bump_version = true continues to bump the
  // version counter to initial_version + 2 across views.
  ASSERT_TRUE(AssignBufferToAtTensor(buf3, tensor).ok());
  TT_ASSERT_OK_AND_ASSIGN(auto read_buf3, GetBuffer(tensor));
  EXPECT_EQ(read_buf3, buf3);
  EXPECT_EQ(tensor.unsafeGetTensorImpl()->version_counter().current_version(),
            initial_version + 2);
  EXPECT_EQ(view.unsafeGetTensorImpl()->version_counter().current_version(),
            initial_version + 2);
}

TEST(GetFreeAndTotalMemoryTest, ExcludesInUseAndReservedMemory) {
  tsl::AllocatorStats stats;
  stats.bytes_limit = 1000;
  stats.bytes_in_use = 300;
  stats.bytes_reserved = 200;

  EXPECT_THAT(internal::GetFreeAndTotalMemory(stats), testing::Pair(500, 1000));
}

TEST(GetFreeAndTotalMemoryTest, ClampsFreeMemoryAtZero) {
  tsl::AllocatorStats stats;
  stats.bytes_limit = 1000;
  stats.bytes_in_use = 800;
  stats.bytes_reserved = 300;

  EXPECT_THAT(internal::GetFreeAndTotalMemory(stats), testing::Pair(0, 1000));
}

TEST(GetFreeAndTotalMemoryTest, IsZeroWithoutLimit) {
  tsl::AllocatorStats stats;
  stats.bytes_in_use = 300;

  EXPECT_THAT(internal::GetFreeAndTotalMemory(stats), testing::Pair(0, 0));
}

}  // namespace
}  // namespace torch_tpu
