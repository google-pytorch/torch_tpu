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
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/cleanup/cleanup.h"
#include "absl/types/span.h"
#include "c10/core/Allocator.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/current_stream.h"
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

TEST(TensorToBufferTest,
     GraphCaptureTracksPreexistingStoragesAndRestoresOriginalBuffers) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  const auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef init_read_only,
                          CreateEmptyDeviceBufferRef({4}, f32_type));
  ASSERT_TRUE(init_read_only.MarkDynamic(0, 2, 8).ok());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef init_write_first,
                          CreateEmptyDeviceBufferRef({4}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef init_read_then_write,
                          CreateEmptyDeviceBufferRef({4}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef written_first_buf,
                          CreateEmptyDeviceBufferRef({4}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef written_after_read_buf,
                          CreateEmptyDeviceBufferRef({4}, f32_type));

  at::Tensor read_only = MakeTensor(init_read_only);
  at::Tensor write_first = MakeTensor(init_write_first);
  at::Tensor read_then_write = MakeTensor(init_read_then_write);

  const auto state =
      BeginGraphCaptureStorageTracking(GetCurrentDeviceStreamId());
  absl::Cleanup end_tracking = [&state] {
    EndGraphCaptureStorageTracking(*state);
  };

  // First read installs a static-shape placeholder; second read returns the
  // same placeholder without installing another.
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef placeholder_ro_1,
                          GetBuffer(read_only));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef placeholder_ro_2,
                          GetBuffer(read_only));
  EXPECT_NE(placeholder_ro_1, init_read_only);
  EXPECT_EQ(placeholder_ro_1, placeholder_ro_2);
  EXPECT_TRUE(placeholder_ro_1.is_materializing());
  EXPECT_FALSE(placeholder_ro_1.is_deferred());
  EXPECT_TRUE(placeholder_ro_1.dynamic_dimensions().empty());

  // Writing the full storage base before any read records the storage without
  // installing a placeholder.
  ASSERT_TRUE(AssignBufferAsStorageBase(written_first_buf, write_first).ok());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef read_after_write,
                          GetBuffer(write_first));
  EXPECT_EQ(read_after_write, written_first_buf);

  // Reading then writing records both the placeholder and the final write.
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef placeholder_rtw,
                          GetBuffer(read_then_write));
  EXPECT_NE(placeholder_rtw, init_read_then_write);
  ASSERT_TRUE(
      AssignBufferToAtTensor(written_after_read_buf, read_then_write).ok());

  std::move(end_tracking).Cancel();
  const CapturedGraphStorages captured = EndGraphCaptureStorageTracking(*state);

  ASSERT_EQ(captured.placeholder_to_storage.size(), 2);
  EXPECT_TRUE(captured.placeholder_to_storage.contains(placeholder_ro_1));
  EXPECT_TRUE(captured.placeholder_to_storage.contains(placeholder_rtw));

  ASSERT_EQ(captured.writes.size(), 2);
  EXPECT_EQ(captured.writes[0].storage.unsafeGetStorageImpl(),
            write_first.storage().unsafeGetStorageImpl());
  EXPECT_EQ(captured.writes[0].final_buffer, written_first_buf);
  EXPECT_FALSE(captured.writes[0].created);
  EXPECT_EQ(captured.writes[1].storage.unsafeGetStorageImpl(),
            read_then_write.storage().unsafeGetStorageImpl());
  EXPECT_EQ(captured.writes[1].final_buffer, written_after_read_buf);
  EXPECT_FALSE(captured.writes[1].created);

  // All pre-existing storages (including pre-marked dynamic bounds) are
  // restored to their pre-capture buffers.
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef restored_ro,
                          GetBuffer(read_only));
  EXPECT_EQ(restored_ro, init_read_only);
  ASSERT_EQ(restored_ro.dynamic_dimensions().size(), 1);
  EXPECT_EQ(restored_ro.dynamic_dimensions()[0].upper_bound, 8);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef restored_wf,
                          GetBuffer(write_first));
  EXPECT_EQ(restored_wf, init_write_first);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef restored_rtw,
                          GetBuffer(read_then_write));
  EXPECT_EQ(restored_rtw, init_read_then_write);
}

TEST(TensorToBufferTest,
     GraphCaptureTracksLiveCreatedStoragesAndClearsDeferredBuffers) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  const auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef deferred_live,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef deferred_freed,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));

  const auto state =
      BeginGraphCaptureStorageTracking(GetCurrentDeviceStreamId());
  absl::Cleanup end_tracking = [&state] {
    EndGraphCaptureStorageTracking(*state);
  };

  at::Tensor live_tensor = MakeTensor(deferred_live);
  {
    at::Tensor freed_tensor = MakeTensor(deferred_freed);
    (void)freed_tensor;
  }

  std::move(end_tracking).Cancel();
  const CapturedGraphStorages captured = EndGraphCaptureStorageTracking(*state);

  EXPECT_TRUE(captured.placeholder_to_storage.empty());
  ASSERT_EQ(captured.writes.size(), 1);
  EXPECT_EQ(captured.writes[0].storage.unsafeGetStorageImpl(),
            live_tensor.storage().unsafeGetStorageImpl());
  EXPECT_EQ(captured.writes[0].final_buffer, deferred_live);
  EXPECT_TRUE(captured.writes[0].created);

  ASSERT_TRUE(ClearCapturedDeferredStorages(captured).ok());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef cleared_buf,
                          GetBaseBuffer(live_tensor.storage()));
  EXPECT_NE(cleared_buf, deferred_live);
  EXPECT_THAT(cleared_buf.dimensions(), testing::ElementsAre(2, 3));
  EXPECT_EQ(cleared_buf.element_type(), f32_type);
}

TEST(TensorToBufferTest,
     AssignBufferToStorageUpdatesSlotAndAllocateGraphMempoolIdIsUnique) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  const auto cleanup_mode =
      absl::MakeCleanup([prev_mode]() { SetEagerMode(prev_mode); });
  constexpr auto f32_type = mlir::ElementType::F32;

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef buf0,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef buf1,
                          CreateEmptyDeviceBufferRef({2, 3}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef wrong_size_buf,
                          CreateEmptyDeviceBufferRef({4, 3}, f32_type));

  at::Tensor tensor = MakeTensor(buf0);
  ASSERT_TRUE(AssignBufferToStorage(buf1, tensor.storage()).ok());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef read_buf, GetBuffer(tensor));
  EXPECT_EQ(read_buf, buf1);
  EXPECT_FALSE(AssignBufferToStorage(wrong_size_buf, tensor.storage()).ok());

  const c10::MempoolId_t pool1 = AllocateGraphMempoolId();
  const c10::MempoolId_t pool2 = AllocateGraphMempoolId();
  EXPECT_NE(pool1.first, 0);
  EXPECT_EQ(pool1.second, 0);
  EXPECT_NE(pool2.first, pool1.first);
}

}  // namespace
}  // namespace torch_tpu
