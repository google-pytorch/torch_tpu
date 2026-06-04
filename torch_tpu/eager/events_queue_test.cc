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

#include "torch_tpu/eager/events_queue.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_buffer_utils.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "torch_tpu/pjrt/pjrt_utils.h"

namespace torch_tpu {
namespace {

class EventsQueueTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string device_type = "xla_cpu";
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = device_type});
    RegisterTpuAllocator();
  }
};

// A dummy MLIR op builder for testing purposes.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) {
  return DynamicMlirOpResults{};
}

TEST_F(EventsQueueTest, GetsLiveDeferredBuffers) {
  ClearEventsQueue();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Record creation of two new DataPtrs, one for buffer "a" and one for "b".
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_a = refs_or.value()[0];
  RecordNewDataPtrCreated(ref_a);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_b = refs_or.value()[0];
  RecordNewDataPtrCreated(ref_b);

  // Both a and b are live and unsynced.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(),
              testing::UnorderedElementsAre(ref_a.device_buffer_list(),
                                            ref_b.device_buffer_list()));

  // Create another DataPtr for buffer "b".
  RecordNewDataPtrCreated(ref_b);

  // Both "a" and "b" are still live and unsynced.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(),
              testing::UnorderedElementsAre(ref_a.device_buffer_list(),
                                            ref_b.device_buffer_list()));

  // Destroy one DataPtr for buffer "b".
  RecordDataPtrDestroyed(ref_b);

  // "b" still has a reference, so is still live and unsynced.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(),
              testing::UnorderedElementsAre(ref_a.device_buffer_list(),
                                            ref_b.device_buffer_list()));

  // Destroy the last DataPtr for buffer "b".
  RecordDataPtrDestroyed(ref_b);

  // "b" is no longer live, only "a" is live and unsynced.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(),
              testing::UnorderedElementsAre(ref_a.device_buffer_list()));

  // Destroy the last DataPtr for buffer "a".
  RecordDataPtrDestroyed(ref_a);
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());
}

TEST_F(EventsQueueTest, IgnoresPlaceholderBuffers) {
  ClearEventsQueue();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a placeholder and record its DataPtr creation.
  auto ref_or =
      DeviceBufferList::CreatePlaceholder(shape.dimensions(), shape.dtype());
  ASSERT_TRUE(ref_or.ok());
  auto ref = ref_or.value();
  RecordNewDataPtrCreated(ref);

  // The placeholder does not need to be synced.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());

  // Deleting the DataPtr is a no-op for the events queue.
  RecordDataPtrDestroyed(ref);
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());
}

TEST_F(EventsQueueTest, IgnoresEmptyBuffers) {
  ClearEventsQueue();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create an empty buffer and record its DataPtr creation.
  auto ref_or = CreateEmptyDeviceBufferRef(shape.dimensions(), shape.dtype());
  ASSERT_TRUE(ref_or.ok());
  auto ref = ref_or.value();
  RecordNewDataPtrCreated(ref);

  // The empty buffer does not need to be synced.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());

  // Deleting the DataPtr is a no-op for the events queue.
  RecordDataPtrDestroyed(ref);
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());
}

TEST_F(EventsQueueTest, IgnoresAlreadyMaterializedBuffers) {
  ClearEventsQueue();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);

  // Create a fully-materialized buffer (filled with zeros) and record its
  // DataPtr creation.
  auto ref_or = TpuMallocAndMemcpyHtoD(/*host_data=*/nullptr,
                                       mlir::ElementType::UI8, {1});
  ASSERT_TRUE(ref_or.ok());
  auto ref = ref_or.value();
  ASSERT_TRUE(ref.is_materialized());
  RecordNewDataPtrCreated(ref);

  // The placeholder does not need to be synced.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());

  // Deleting the DataPtr is a no-op for the events queue.
  RecordDataPtrDestroyed(ref);
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());
}

TEST_F(EventsQueueTest, ClearsBuffersAfterMaterialization) {
  ClearEventsQueue();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{1}, mlir::ElementType::UI8);

  auto input_ref_or = TpuMallocAndMemcpyHtoD(/*host_data=*/nullptr,
                                             mlir::ElementType::UI8, {1});
  ASSERT_TRUE(input_ref_or.ok());
  auto input_ref = input_ref_or.value();
  ASSERT_TRUE(input_ref.is_materialized());
  RecordNewDataPtrCreated(input_ref);

  // Record creation a deferred buffer (with a real op builder).
  auto op_builder = [](mlir::MlirBuilder& builder,
                       absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return DynamicMlirOpResults{inputs[0]};
  };

  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, std::move(op_builder), {input_ref},
      OpParamCacheKeys::Empty(), {shape});

  ASSERT_TRUE(refs_or.ok());
  auto ref = refs_or.value()[0];
  ASSERT_TRUE(ref.is_deferred());
  RecordNewDataPtrCreated(ref);

  // The deferred buffer needs to be synced.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(),
              testing::UnorderedElementsAre(ref.device_buffer_list()));

  // Materialize the buffer and wait for it to finish.
  auto materialization_status =
      Materialize(ref, MaterializationReason::kExplicitSync);
  ASSERT_TRUE(materialization_status.ok());
  ASSERT_TRUE(ref.is_materializing());
  auto await_status = ref.AwaitBuffer();
  ASSERT_TRUE(await_status.ok());
  ASSERT_TRUE(ref.is_materialized());

  // We don't need to sync the buffer anymore.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());

  // Deleting the DataPtr is a no-op for the events queue.
  RecordDataPtrDestroyed(ref);
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());
}

TEST_F(EventsQueueTest, StopsTrackingAfterClearEventsQueue) {
  ClearEventsQueue();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Record creation of a deferred buffer.
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref = refs_or.value()[0];
  RecordNewDataPtrCreated(ref);

  // Clear the events queue.
  ClearEventsQueue();

  // The buffer is no longer tracked.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());

  // Create a second data pointer to the same buffer.
  RecordNewDataPtrCreated(ref);

  // The buffer is now tracked again.
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(),
              testing::UnorderedElementsAre(ref.device_buffer_list()));

  // Delete one data pointer.
  // The first data pointer (before the clear) was forgotten, so this clears the
  // buffer from the queue.
  RecordDataPtrDestroyed(ref);
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());

  // Delete the second data pointer.
  // We'd already removed the buffer from the queue, so this is a no-op.
  RecordDataPtrDestroyed(ref);
  EXPECT_THAT(GetAllLiveUnsyncedDataPtrs(), testing::IsEmpty());
}
}  // namespace
}  // namespace torch_tpu
