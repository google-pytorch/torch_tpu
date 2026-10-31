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

#include "csrc/eager/events_queue.h"

#include <chrono>
#include <memory>
#include <ratio>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/future.h"
#include "xla/tsl/platform/env.h"

namespace torch_tpu {
namespace {

class EventsQueueTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string device_type = "tpu";
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
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Record creation of two new DataPtrs, one for buffer "a" and one for "b".
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  RecordDeferredOpCreated(ref_a.device_buffer_list());
  ref_a.device_buffer_list()->IncrementLiveDataPtrs(ref_a.index());

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  RecordDeferredOpCreated(ref_b.device_buffer_list());
  ref_b.device_buffer_list()->IncrementLiveDataPtrs(ref_b.index());

  // Both a and b are live and unsynced and will be materialized on the next
  // stream materialization or device materialization.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  InsertGraphBreak(device_index, stream_id, {}, MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);

  ASSERT_EQ(ready_work.size(), 1);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);
  EXPECT_THAT(eager_execution.must_execute,
              testing::ElementsAre(ref_a.device_buffer_list(),
                                   ref_b.device_buffer_list()));
}

TEST_F(EventsQueueTest, IgnoresPlaceholderBuffers) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a placeholder and record its DataPtr creation.
  TT_ASSERT_OK_AND_ASSIGN(auto ref, DeviceBufferList::CreatePlaceholder(
                                        shape.dimensions(), shape.dtype()));
  ref.device_buffer_list()->IncrementLiveDataPtrs(ref.index());

  // The placeholder does not need to be synced.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  InsertGraphBreak(device_index, stream_id, {}, MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);
  EXPECT_TRUE(ready_work.empty());
}

TEST_F(EventsQueueTest, IgnoresEmptyBuffers) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create an empty buffer and record its DataPtr creation.
  TT_ASSERT_OK_AND_ASSIGN(
      auto ref, CreateEmptyDeviceBufferRef(shape.dimensions(), shape.dtype()));
  ref.device_buffer_list()->IncrementLiveDataPtrs(ref.index());

  // The empty buffer does not need to be synced.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  InsertGraphBreak(device_index, stream_id, {}, MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);

  EXPECT_TRUE(ready_work.empty());
}

TEST_F(EventsQueueTest, IgnoresAlreadyMaterializedBuffers) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);

  // Create a fully-materialized buffer (filled with zeros) and record its
  // DataPtr creation.
  TT_ASSERT_OK_AND_ASSIGN(
      auto ref,
      HostToDeviceTransfer::FromZeroes(mlir::ElementType::UI8, {1}).Run());
  ASSERT_TRUE(ref.Synchronize().ok()) << ref.Synchronize();
  ref.device_buffer_list()->IncrementLiveDataPtrs(ref.index());

  // The placeholder does not need to be synced.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  InsertGraphBreak(device_index, stream_id, {}, MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);

  EXPECT_TRUE(ready_work.empty());
}

TEST_F(EventsQueueTest, ClearsBuffersAfterMaterialization) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{1}, mlir::ElementType::UI8);

  TT_ASSERT_OK_AND_ASSIGN(
      auto input_ref,
      HostToDeviceTransfer::FromZeroes(mlir::ElementType::UI8, {1}).Run());
  ASSERT_TRUE(input_ref.Synchronize().ok()) << input_ref.Synchronize();
  input_ref.device_buffer_list()->IncrementLiveDataPtrs(input_ref.index());

  // Record creation a deferred buffer (with a real op builder).
  auto op_builder = [](mlir::MlirBuilder& builder,
                       absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return DynamicMlirOpResults{inputs[0]};
  };

  TT_ASSERT_OK_AND_ASSIGN(auto refs,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, std::move(op_builder), {input_ref},
                              OpParamCacheKeys::Empty(), {shape}));
  auto ref = refs[0];
  ASSERT_TRUE(ref.is_deferred());
  RecordDeferredOpCreated(ref.device_buffer_list());

  // The deferred buffer needs to be synced.
  // Materialize the buffer and wait for it to finish.
  auto materialization_status =
      Materialize(ref, MaterializationReason::kExplicitSync);
  ASSERT_TRUE(materialization_status.ok());
  ASSERT_TRUE(ref.is_materializing());
  auto await_status = ref.AwaitBuffer();
  ASSERT_TRUE(await_status.ok());
  ASSERT_TRUE(ref.is_materialized());

  // We don't need to sync the buffer anymore.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  InsertGraphBreak(device_index, stream_id, {}, MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);

  EXPECT_TRUE(ready_work.empty());
}

TEST_F(EventsQueueTest, StopsTrackingAfterClearEventsQueue) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  // Record creation of a deferred buffer.
  TT_ASSERT_OK_AND_ASSIGN(auto refs, DeviceBufferList::CreateDeferred(
                                         OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape}));
  auto ref = refs[0];
  ref.device_buffer_list()->IncrementLiveDataPtrs(ref.index());

  // Clear the events queue.
  ClearAllStreams();

  // The buffer is no longer tracked.
  InsertGraphBreak(device_index, stream_id, {}, MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work1 = GetReadyWork(device_index, stream_id);

  EXPECT_TRUE(ready_work1.empty());

  // Create a second data pointer to the same buffer.
  ref.device_buffer_list()->IncrementLiveDataPtrs(ref.index());

  InsertGraphBreak(device_index, stream_id, {}, MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  // The buffer is live, but the DeferredOp was cleared and so does not need
  // to be materialized.
  auto ready_work2 = GetReadyWork(device_index, stream_id);
  EXPECT_TRUE(ready_work2.empty());
}

TEST_F(EventsQueueTest, DeadCodeEliminated) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Add three deferred ops to the queue, but drop all references to them.
  {
    TT_ASSERT_OK_AND_ASSIGN(
        auto refs_a,
        DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape}));
    auto list_a = refs_a[0].device_buffer_list();
    RecordDeferredOpCreated(list_a);

    TT_ASSERT_OK_AND_ASSIGN(
        auto refs_b,
        DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape}));
    auto list_b = refs_b[0].device_buffer_list();
    RecordDeferredOpCreated(list_b);

    TT_ASSERT_OK_AND_ASSIGN(
        auto refs_c,
        DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape}));
    auto list_c = refs_c[0].device_buffer_list();
    RecordDeferredOpCreated(list_c);
  }

  // Create a fourth, non-dead node and request a traversal plan for it.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_d, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_d = refs_d[0];
  auto list_d = ref_d.device_buffer_list();
  RecordDeferredOpCreated(list_d);

  const auto device_index = list_d->device_index();
  const auto stream_id = list_d->stream_id();
  InsertGraphBreak(device_index, stream_id, {list_d},
                   MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);

  ASSERT_EQ(ready_work.size(), 1);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);
  // The dead ops should be stripped.
  EXPECT_THAT(eager_execution.must_execute, testing::ElementsAre(list_d));
}

TEST_F(EventsQueueTest, DeadSideEffectsRetained) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Add three deferred ops to the queue, but drop all references to them.
  // Create a dependency chain of a -> b -> c, where c is a side-effect op.
  const DeviceBufferList* dead_a_ptr = nullptr;
  const DeviceBufferList* dead_b_ptr = nullptr;
  const DeviceBufferList* dead_c_ptr = nullptr;
  {
    TT_ASSERT_OK_AND_ASSIGN(
        auto refs_a,
        DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape}));
    auto ref_a = refs_a[0];
    auto list_a = refs_a[0].device_buffer_list();
    RecordDeferredOpCreated(list_a);
    dead_a_ptr = list_a.get();

    TT_ASSERT_OK_AND_ASSIGN(
        auto refs_b,
        DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref_a},
                                         OpParamCacheKeys::Empty(), {shape}));
    auto ref_b = refs_b[0];
    auto list_b = refs_b[0].device_buffer_list();
    RecordDeferredOpCreated(list_b);
    dead_b_ptr = list_b.get();

    TT_ASSERT_OK_AND_ASSIGN(
        auto refs_c, DeviceBufferList::CreateDeferred(
                         OpName::kDistributedAllReduce, DummyBuilder, {ref_b},
                         OpParamCacheKeys::Empty(), {shape}));
    auto list_c = refs_c[0].device_buffer_list();
    RecordDeferredOpCreated(list_c);
    dead_c_ptr = list_c.get();
  }

  // Create a fourth, non-dead node and request a snapshot for it.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_d, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_d = refs_d[0];
  auto list_d = ref_d.device_buffer_list();
  RecordDeferredOpCreated(list_d);

  const auto device_index = list_d->device_index();
  const auto stream_id = list_d->stream_id();
  InsertGraphBreak(device_index, stream_id, {list_d},
                   MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);
  ASSERT_EQ(ready_work.size(), 1);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);

  // We must still execute the dead ops.
  ASSERT_EQ(eager_execution.must_execute.size(), 4);
  EXPECT_EQ(eager_execution.must_execute[0].get(), dead_a_ptr);
  EXPECT_EQ(eager_execution.must_execute[1].get(), dead_b_ptr);
  EXPECT_EQ(eager_execution.must_execute[2].get(), dead_c_ptr);
  EXPECT_EQ(eager_execution.must_execute[3], list_d);
}

TEST_F(EventsQueueTest, MaterializationIgnoresUnusedEmptyOps) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a deferred, empty tensor.
  TT_ASSERT_OK_AND_ASSIGN(auto empty_refs, DeviceBufferList::CreateDeferred(
                                               OpName::kEmpty, DummyBuilder, {},
                                               OpParamCacheKeys::Empty(),
                                               /*output_shapes=*/{shape}));
  auto empty_ref = empty_refs[0];
  auto empty_list = empty_ref.device_buffer_list();
  RecordDeferredOpCreated(empty_list);
  empty_list->IncrementLiveDataPtrs(empty_ref.index());

  // Create a non-empty deferred tensor that does not rely on the empty tensor.
  TT_ASSERT_OK_AND_ASSIGN(
      auto non_empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto non_empty_ref = non_empty_refs[0];
  auto non_empty_list = non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(non_empty_list);
  non_empty_list->IncrementLiveDataPtrs(non_empty_ref.index());

  const auto device_index = non_empty_list->device_index();
  const auto stream_id = non_empty_list->stream_id();
  InsertGraphBreak(device_index, stream_id, {non_empty_list},
                   MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);
  ASSERT_EQ(ready_work.size(), 1);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);
  EXPECT_THAT(eager_execution.must_execute,
              testing::ElementsAre(non_empty_list));
}

TEST_F(EventsQueueTest, PrepareDeviceTraversals) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  const auto device_index = GetCurrentDeviceIndex();

  // Put three deferred ops in the queue;
  // a -> b are on device 0, with b having a live DataPtr.
  // c is on device 1.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = ref_a.device_buffer_list();
  RecordDeferredOpCreated(list_a);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);
  list_b->IncrementLiveDataPtrs(ref_b.index());

  ExchangeCurrentDeviceIndex(device_index + 1);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = ref_c.device_buffer_list();
  RecordDeferredOpCreated(list_c);
  list_c->IncrementLiveDataPtrs(ref_c.index());

  ExchangeCurrentDeviceIndex(device_index);

  // Ask for a plan to materialize the device for a and b (but not c).
  auto pending_stream_snapshots = RecordDeviceSnapshots(device_index);
  ASSERT_EQ(pending_stream_snapshots.size(), 1);

  auto ready_work =
      GetReadyWork(device_index, pending_stream_snapshots[0].stream_id);
  ASSERT_EQ(ready_work.size(), 2);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);
  // The execution will contain a and b but not c, as it is on a different
  // device.
  EXPECT_THAT(eager_execution.must_execute,
              testing::ElementsAre(list_a, list_b));
  // There will be a snapshot request for the stream after the eager execution.
  EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[1]));
}

TEST_F(EventsQueueTest, PrepareStreamTraversals) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  const auto device_index = GetCurrentDeviceIndex();

  // Put three deferred ops in the queue;
  // a -> b are on the default stream, with b having a live DataPtr.
  // c is on a non-default stream.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = ref_a.device_buffer_list();
  RecordDeferredOpCreated(list_a);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);
  list_b->IncrementLiveDataPtrs(ref_b.index());

  const auto non_default_stream_id = NextStreamId(device_index);
  const auto default_stream_id =
      ExchangeCurrentStreamId(device_index, non_default_stream_id);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = ref_c.device_buffer_list();
  RecordDeferredOpCreated(list_c);
  list_c->IncrementLiveDataPtrs(ref_c.index());

  ExchangeCurrentStreamId(device_index, default_stream_id);

  // Ask for a plan to materialize the stream for a and b (but not c).
  (void)EventSnapshot::Record(device_index, default_stream_id, {});
  auto ready_work = GetReadyWork(device_index, default_stream_id);
  ASSERT_EQ(ready_work.size(), 2);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);

  // c is not included as it is on a different stream.
  EXPECT_THAT(eager_execution.must_execute,
              testing::ElementsAre(list_a, list_b));
  // There will be a snapshot request for the stream after the eager execution.
  EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[1]));
}

TEST_F(EventsQueueTest, StreamsMaterializeSeparately) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  const auto device_index = GetCurrentDeviceIndex();

  // Put three deferred ops in the queue;
  // a -> b are on the default stream, with b having a live DataPtr.
  // c is on a non-default stream.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = ref_a.device_buffer_list();
  RecordDeferredOpCreated(list_a);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);
  list_b->IncrementLiveDataPtrs(ref_b.index());

  const auto non_default_stream_id = NextStreamId(device_index);
  const auto default_stream_id =
      ExchangeCurrentStreamId(device_index, non_default_stream_id);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = ref_c.device_buffer_list();
  RecordDeferredOpCreated(list_c);
  list_c->IncrementLiveDataPtrs(ref_c.index());

  ExchangeCurrentStreamId(device_index, default_stream_id);

  // Ask for a plan to materialize the stream for c (but not a or b)
  InsertGraphBreak(device_index, non_default_stream_id, {},
                   MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, non_default_stream_id);
  ASSERT_EQ(ready_work.size(), 1);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);

  // Only c is included as it is on a different stream from a and b.
  EXPECT_THAT(eager_execution.must_execute, testing::ElementsAre(list_c));
}

TEST_F(EventsQueueTest, RetainsFailedCompletedFutures) {
  ClearAllStreams();
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  xla::Future<void> failed_future(TT_ERROR(error::kInvalidArgument)
                                  << "Fake error");
  ASSERT_TRUE(failed_future.IsReady());

  internal::MarkStreamActive(device_index, stream_id, std::move(failed_future));

  (void)EventSnapshot::Record(device_index, stream_id);
  auto ready_work = GetReadyWork(device_index, stream_id);
  ASSERT_EQ(ready_work.size(), 1);
  ASSERT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[0]));
  const auto& snapshot_request = std::get<SnapshotRequest>(ready_work[0]);
  ASSERT_EQ(snapshot_request.futures.size(), 1);

  auto status = snapshot_request.futures[0].Await();
  EXPECT_EQ(status.code(), error::kInvalidArgument);
}

TEST_F(EventsQueueTest, PrunesSuccessfulCompletedFutures) {
  ClearAllStreams();
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  xla::Future<void> ok_future(absl::OkStatus());
  ASSERT_TRUE(ok_future.IsReady());

  internal::MarkStreamActive(device_index, stream_id, std::move(ok_future));

  (void)EventSnapshot::Record(device_index, stream_id);
  auto ready_work = GetReadyWork(device_index, stream_id);
  ASSERT_EQ(ready_work.size(), 1);
  ASSERT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[0]));
  const auto& snapshot_request = std::get<SnapshotRequest>(ready_work[0]);
  EXPECT_TRUE(snapshot_request.futures.empty());
}

TEST_F(EventsQueueTest, StreamMustMaterializeEmptyTensors) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a tensor simulating a torch.empty() call.
  TT_ASSERT_OK_AND_ASSIGN(
      auto empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kEmptyMemoryFormat, DummyBuilder,
                                       {}, OpParamCacheKeys::Empty(), {shape}));
  auto empty_ref = empty_refs[0];
  auto empty_list = empty_ref.device_buffer_list();
  RecordDeferredOpCreated(empty_list);
  empty_list->IncrementLiveDataPtrs(empty_ref.index());
  const auto device_index = empty_list->device_index();
  const auto stream_id = empty_list->stream_id();

  // A normal stream materialization does not materialize the empty tensor.
  InsertGraphBreak(device_index, stream_id, {}, MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);

  EXPECT_TRUE(ready_work.empty());

  // A stream materialization that must materialize the empty tensor does so.
  (void)EventSnapshot::Record(device_index, stream_id, {empty_list});
  ready_work = GetReadyWork(device_index, stream_id);
  ASSERT_EQ(ready_work.size(), 2);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);
  // Constant ops are listed only in must_materialize, but not must_execute.
  EXPECT_EQ(eager_execution.must_execute.size(), 0);
  EXPECT_THAT(eager_execution.must_materialize,
              testing::ElementsAre(empty_list));
  EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[1]));
}

TEST_F(EventsQueueTest, DeviceMustMaterializeEmptyTensors) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a tensor simulating a torch.empty() call.
  TT_ASSERT_OK_AND_ASSIGN(
      auto empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kEmptyMemoryFormat, DummyBuilder,
                                       {}, OpParamCacheKeys::Empty(), {shape}));
  auto empty_ref = empty_refs[0];
  auto empty_list = empty_ref.device_buffer_list();
  RecordDeferredOpCreated(empty_list);
  empty_list->IncrementLiveDataPtrs(empty_ref.index());
  const auto device_index = empty_list->device_index();
  const auto stream_id = empty_list->stream_id();

  // A normal device materialization does not materialize the empty tensor.
  RecordDeviceSnapshots(device_index);
  auto ready_work = GetReadyWork(device_index, stream_id);
  ASSERT_EQ(ready_work.size(), 1);
  EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[0]));

  // A device materialization that must materialize the empty tensor does so.
  RecordDeviceSnapshots(device_index, {empty_list});
  ready_work = GetReadyWork(device_index, stream_id);
  ASSERT_EQ(ready_work.size(), 2);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);
  // Constant ops are listed only in must_materialize, but not must_execute.
  EXPECT_TRUE(eager_execution.must_execute.empty());
  EXPECT_THAT(eager_execution.must_materialize,
              testing::ElementsAre(empty_list));
  EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[1]));
}

TEST_F(EventsQueueTest, MultipleSnapshotsPulled) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Enqueue two snapshots after single ops: [a, snapshot, b, snapshot]
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = ref_a.device_buffer_list();
  RecordDeferredOpCreated(list_a);

  (void)EventSnapshot::Record(list_a->device_index(), list_a->stream_id(), {});

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);

  (void)EventSnapshot::Record(list_b->device_index(), list_b->stream_id(), {});

  // Get the pending snapshots.
  auto ready_work = GetReadyWork(list_a->device_index(), list_a->stream_id());

  // Each snapshot request results in one pending execution and one pending
  // snapshot.
  ASSERT_EQ(ready_work.size(), 4);

  // The first pending snapshot will materialize a (only).
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& first_eager_execution = std::get<EagerExecution>(ready_work[0]);
  EXPECT_THAT(first_eager_execution.must_execute, testing::ElementsAre(list_a));
  EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[1]));

  // The second pending snapshot will materialize b (only).
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[2]));
  const auto& second_eager_execution = std::get<EagerExecution>(ready_work[2]);
  EXPECT_THAT(second_eager_execution.must_execute,
              testing::ElementsAre(list_b));
  EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[3]));
}

TEST_F(EventsQueueTest, MultipleEagerExecutionsPulled) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Enqueue a graph break and a snapshot after single ops:
  // [a, graph_break, b, snapshot]
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = ref_a.device_buffer_list();
  RecordDeferredOpCreated(list_a);

  InsertGraphBreak(list_a->device_index(), list_a->stream_id(), {list_a},
                   MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);

  (void)EventSnapshot::Record(list_b->device_index(), list_b->stream_id(), {});

  // Get the pending snapshots.
  auto ready_work = GetReadyWork(list_a->device_index(), list_a->stream_id());

  // The graph break results in separate work items for a and b, and the
  // snapshot request is a third work item.
  ASSERT_EQ(ready_work.size(), 3);

  // The first eager execution will materialize a (only).
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& first_eager_execution = std::get<EagerExecution>(ready_work[0]);
  EXPECT_THAT(first_eager_execution.must_execute, testing::ElementsAre(list_a));

  // The second eager execution will materialize b (only).
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[1]));
  const auto& second_eager_execution = std::get<EagerExecution>(ready_work[1]);
  EXPECT_THAT(second_eager_execution.must_execute,
              testing::ElementsAre(list_b));

  // The third work item is the snapshot request.
  EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[2]));
}

TEST_F(EventsQueueTest, NotPulledAfterLastSnapshot) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Enqueue one snapshots after a single op: [a, snapshot]
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = ref_a.device_buffer_list();
  RecordDeferredOpCreated(list_a);

  (void)EventSnapshot::Record(list_a->device_index(), list_a->stream_id(), {});

  // Enqueue an op after the snapshot, but don't request a second snapshot.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);

  // Get the pending snapshots.
  auto ready_work = GetReadyWork(list_a->device_index(), list_a->stream_id());

  // List b is left in the queue.
  ASSERT_EQ(ready_work.size(), 2);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);
  EXPECT_THAT(eager_execution.must_execute, testing::ElementsAre(list_a));
  EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[1]));
}

TEST_F(EventsQueueTest, SingleOpFastPathAndExtraConstantFallback) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  // 1. Single deferred op with no deferred inputs (exercises the single-op
  // fast path).
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_single,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto ref_single = refs_single[0];
  auto list_single = ref_single.device_buffer_list();
  RecordDeferredOpCreated(list_single);

  InsertGraphBreak(device_index, stream_id, {list_single},
                   MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  auto ready_work = GetReadyWork(device_index, stream_id);
  ASSERT_EQ(ready_work.size(), 1);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);
  EXPECT_THAT(eager_execution.must_execute, testing::ElementsAre(list_single));
  EXPECT_THAT(eager_execution.must_materialize,
              testing::ElementsAre(list_single));

  // 2. Single deferred op plus an explicit deferred constant in
  // must_materialize (must bypass the single-op fast path so the constant is
  // also materialized).
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_op,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto ref_op = refs_op[0];
  auto list_op = ref_op.device_buffer_list();
  RecordDeferredOpCreated(list_op);

  TT_ASSERT_OK_AND_ASSIGN(
      auto empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kEmptyMemoryFormat, DummyBuilder,
                                       {}, OpParamCacheKeys::Empty(), {shape}));
  auto empty_ref = empty_refs[0];
  auto empty_list = empty_ref.device_buffer_list();
  RecordDeferredOpCreated(empty_list);

  InsertGraphBreak(device_index, stream_id, {list_op, empty_list},
                   MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  ready_work = GetReadyWork(device_index, stream_id);
  ASSERT_EQ(ready_work.size(), 1);
  ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
  const auto& eager_execution_2 = std::get<EagerExecution>(ready_work[0]);
  EXPECT_THAT(eager_execution_2.must_execute, testing::ElementsAre(list_op));
  EXPECT_THAT(eager_execution_2.must_materialize,
              testing::ElementsAre(list_op, empty_list));
}

TEST_F(EventsQueueTest, NodeMatcherAcrossSmallAndLargeSpans) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  // Test both small span (5 <= 8, InlinedVector) and large span (10 > 8,
  // flat_hash_set).
  for (int count : {5, 10}) {
    std::vector<SharedDeviceBufferList> nodes;
    for (int i = 0; i < count; ++i) {
      TT_ASSERT_OK_AND_ASSIGN(
          auto refs,
          DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
      nodes.push_back(refs[0].device_buffer_list());
      RecordDeferredOpCreated(refs[0].device_buffer_list());
    }
    (void)EventSnapshot::Record(device_index, stream_id, nodes);
    auto ready_work = GetReadyWork(device_index, stream_id);
    ASSERT_EQ(ready_work.size(), 2);
    ASSERT_TRUE(std::holds_alternative<EagerExecution>(ready_work[0]));
    const auto& eager_execution = std::get<EagerExecution>(ready_work[0]);
    EXPECT_EQ(eager_execution.must_execute.size(), count);
    EXPECT_EQ(eager_execution.must_materialize.size(), count);
    EXPECT_THAT(eager_execution.must_execute, testing::ElementsAreArray(nodes));
    EXPECT_THAT(eager_execution.must_materialize,
                testing::ElementsAreArray(nodes));
    EXPECT_TRUE(std::holds_alternative<SnapshotRequest>(ready_work[1]));
  }
}

// Returns `duration` in milliseconds, the unit of EventSnapshot::ElapsedTime.
double ToMilliseconds(std::chrono::steady_clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

TEST(EventSnapshotTest, ElapsedTimeMeasuresCompletionTimes) {
  auto [start_promise, start_future] = xla::MakePromise<void>();
  auto [start_ok_promise, start_ok_future] = xla::MakePromise<void>();
  auto [end_promise, end_future] = xla::MakePromise<void>();
  auto [end_ok_promise, end_ok_future] = xla::MakePromise<void>();
  EventSnapshot start(std::move(start_future), std::move(start_ok_promise));
  EventSnapshot end(std::move(end_future), std::move(end_ok_promise));
  EXPECT_EQ(start.ElapsedTime(end).status().code(), error::kFailedPrecondition);

  const auto before_start = std::chrono::steady_clock::now();
  start_promise.Set();
  const auto after_start = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(start.ElapsedTime(end).status().code(), error::kFailedPrecondition);
  const auto before_end = std::chrono::steady_clock::now();
  end_promise.Set();
  const auto after_end = std::chrono::steady_clock::now();

  TT_ASSERT_OK_AND_ASSIGN(double elapsed_ms, start.ElapsedTime(end));
  EXPECT_GE(elapsed_ms, ToMilliseconds(before_end - after_start));
  EXPECT_LE(elapsed_ms, ToMilliseconds(after_end - before_start));
  TT_ASSERT_OK_AND_ASSIGN(double reverse_ms, end.ElapsedTime(start));
  EXPECT_EQ(reverse_ms, -elapsed_ms);
}

TEST(EventSnapshotTest, ElapsedTimeExcludesWorkPendingAtStart) {
  // The start event is recorded while earlier work is pending, and the end
  // event captures later work. As in the events queue, the start event also
  // releases the end event.
  auto [earlier_promise, earlier_future] = xla::MakePromise<void>();
  auto [earlier_ok_promise, earlier_ok_future] = xla::MakePromise<void>();
  EventSnapshot start(std::move(earlier_future), std::move(earlier_ok_promise));
  auto chain_future = start.OkCallback();
  auto [later_promise, later_future] = xla::MakePromise<void>();
  auto [later_ok_promise, later_ok_future] = xla::MakePromise<void>();
  EventSnapshot end(xla::JoinFutures({later_future, chain_future}),
                    std::move(later_ok_promise));
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  const auto before_earlier = std::chrono::steady_clock::now();
  earlier_promise.Set();
  const auto after_earlier = std::chrono::steady_clock::now();
  TT_ASSERT_OK_AND_ASSIGN(bool end_ready, end.Query());
  ASSERT_FALSE(end_ready);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const auto before_later = std::chrono::steady_clock::now();
  later_promise.Set();
  const auto after_later = std::chrono::steady_clock::now();

  // The time between recording the start event and `before_earlier` does not
  // count.
  TT_ASSERT_OK_AND_ASSIGN(double elapsed_ms, start.ElapsedTime(end));
  EXPECT_GE(elapsed_ms, ToMilliseconds(before_later - after_earlier));
  EXPECT_LE(elapsed_ms, ToMilliseconds(after_later - before_earlier));
}

TEST(EventSnapshotTest, ElapsedTimeIsNonNegativeForChainedPendingEvents) {
  // Two events recorded back to back while earlier work is pending: the second
  // event completes when the first event's completion callback releases it, so
  // the first event must be stamped before that callback runs.
  auto [work_promise, work_future] = xla::MakePromise<void>();
  auto [start_ok_promise, start_ok_future] = xla::MakePromise<void>();
  EventSnapshot start(std::move(work_future), std::move(start_ok_promise));
  auto chain_future = start.OkCallback();
  auto [end_ok_promise, end_ok_future] = xla::MakePromise<void>();
  EventSnapshot end(std::move(chain_future), std::move(end_ok_promise));
  TT_ASSERT_OK_AND_ASSIGN(bool end_ready, end.Query());
  ASSERT_FALSE(end_ready);

  work_promise.Set();
  TT_ASSERT_OK_AND_ASSIGN(end_ready, end.Query());
  ASSERT_TRUE(end_ready);
  TT_ASSERT_OK_AND_ASSIGN(double elapsed_ms, start.ElapsedTime(end));
  EXPECT_GE(elapsed_ms, 0.0);
}

TEST(EventSnapshotTest, ElapsedTimeReturnsEventError) {
  auto [ok_event_ok_callback_promise, ok_event_ok_callback_future] =
      xla::MakePromise<void>();
  EventSnapshot ok_event(absl::OkStatus(),
                         std::move(ok_event_ok_callback_promise));
  auto [failed_event_ok_callback_promise, failed_event_ok_callback_future] =
      xla::MakePromise<void>();
  EventSnapshot failed_event(
      xla::Future<void>(TT_ERROR(error::kInvalidArgument) << "fake error"),
      std::move(failed_event_ok_callback_promise));
  EXPECT_EQ(ok_event.ElapsedTime(failed_event).status().code(),
            error::kInvalidArgument);
}

TEST_F(EventsQueueTest, CompletesLongSnapshotChainOnSmallStack) {
  ClearAllStreams();
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  // Create a small amount of "work" which will complete when we choose.
  auto [promise, future] = xla::MakePromise<void>();
  internal::MarkStreamActive(device_index, stream_id, std::move(future));
  std::vector<xla::Future<std::shared_ptr<EventSnapshot>>> snapshot_futures;
  for (int i = 0; i < 20000; ++i) {
    snapshot_futures.push_back(
        MaterializeCurrentStream(MaterializationReason::kExplicitSync));
  }
  std::shared_ptr<EventSnapshot> last_snapshot;
  for (const auto& snapshot_future : snapshot_futures) {
    TT_ASSERT_OK_AND_ASSIGN(last_snapshot, snapshot_future.Await());
  }
  TT_ASSERT_OK_AND_ASSIGN(bool completed, last_snapshot->Query());
  ASSERT_FALSE(completed);

  // The runtime completes device work on threads with 1 MiB stacks.
  std::unique_ptr<tsl::Thread> thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions{.stack_size = 1 << 20}, "complete_transfer",
      [&promise] { promise.Set(); }));
  thread.reset();
  EXPECT_TRUE(last_snapshot->Synchronize().ok());
}

TEST_F(EventsQueueTest, OkCallbackReturnsOkAfterError) {
  ClearAllStreams();
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  // Create a small amount of "work" which will complete when we choose.
  auto [promise, future] = xla::MakePromise<void>();
  internal::MarkStreamActive(device_index, stream_id, future);

  // Create an EventSnapshot with the future and get the OK callback from it.
  auto [ok_callback_promise, created_ok_callback_future] =
      xla::MakePromise<void>();
  ASSERT_TRUE(created_ok_callback_future.IsValid());
  EventSnapshot snapshot(std::move(future), std::move(ok_callback_promise));
  auto retrieved_ok_callback = snapshot.OkCallback();
  ASSERT_TRUE(retrieved_ok_callback.IsValid());

  // The snapshot is not done, so the OK callback is not done.
  EXPECT_FALSE(created_ok_callback_future.IsReady());
  EXPECT_FALSE(retrieved_ok_callback.IsReady());

  // The event snapshot fails.
  promise.Set(TT_ERROR(error::kAborted) << "test error");

  // Awaiting the snapshot gives the failure.
  auto error = snapshot.Synchronize();
  EXPECT_EQ(error.code(), error::kAborted);

  // But the OK callback is done and OK.
  ASSERT_TRUE(created_ok_callback_future.IsReady());
  EXPECT_TRUE(created_ok_callback_future.Await().ok());
  ASSERT_TRUE(retrieved_ok_callback.IsReady());
  EXPECT_TRUE(retrieved_ok_callback.Await().ok());
}

TEST_F(EventsQueueTest, InsertedWaitBlocksHostToDeviceTransfer) {
  ClearAllStreams();
  const auto [device_index, default_stream_id] = GetCurrentDeviceStreamId();

  // Create a small amount of "work" which will complete when we choose.
  auto [promise, future] = xla::MakePromise<void>();
  internal::MarkStreamActive(device_index, default_stream_id,
                             std::move(future));

  // Get an EventSnapshot for the default stream after this "work".
  auto snapshot_future = MaterializeStream(device_index, default_stream_id,
                                           MaterializationReason::kUnknown);
  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<EventSnapshot> snapshot,
                          snapshot_future.Await());

  // Insert a stream wait for the snapshot in a different stream.
  const auto other_stream_id = NextStreamId(device_index);
  ExchangeCurrentStreamId(device_index, other_stream_id);
  absl::Cleanup cleanup = [device_index, default_stream_id] {
    ExchangeCurrentStreamId(device_index, default_stream_id);
  };
  InsertStreamWait(device_index, other_stream_id, *snapshot);

  // Schedule a transfer on the other stream.
  auto h2d_transfer =
      HostToDeviceTransfer::FromZeroes(mlir::ElementType::UI8, {});
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef transferred_ref,
      ScheduleHostToDeviceTransfer(std::move(h2d_transfer)));

  // The transfer is not yet started.
  EXPECT_TRUE(transferred_ref.is_transferring());
  EXPECT_FALSE(transferred_ref.is_materialized());

  // Complete the work for the default stream.
  promise.Set(absl::OkStatus());

  // The transfer will then start and complete.
  EXPECT_TRUE(transferred_ref.Synchronize().ok());
  EXPECT_FALSE(transferred_ref.is_transferring());
  EXPECT_TRUE(transferred_ref.is_materialized());
}

TEST_F(EventsQueueTest, InsertedWaitBlocksDeviceToHostTransfer) {
  ClearAllStreams();
  const auto [device_index, default_stream_id] = GetCurrentDeviceStreamId();

  // Create a fully-materialized DeviceBufferRef.
  std::vector<char> data = {0x01};
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef constant_ref,
      CreateConstantDeviceBufferRef(std::move(data), Dimensions(),
                                    mlir::ElementType::UI8));
  ASSERT_TRUE(Materialize(constant_ref, MaterializationReason::kUnknown).ok());
  ASSERT_TRUE(constant_ref.Synchronize().ok());

  // Create a small amount of "work" which will complete when we choose.
  auto [promise, future] = xla::MakePromise<void>();
  internal::MarkStreamActive(device_index, default_stream_id,
                             std::move(future));

  // Get an EventSnapshot for the default stream after this "work".
  auto snapshot_future = MaterializeStream(device_index, default_stream_id,
                                           MaterializationReason::kUnknown);
  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<EventSnapshot> snapshot,
                          snapshot_future.Await());

  // Insert a stream wait for the snapshot in a different stream.
  const auto other_stream_id = NextStreamId(device_index);
  ExchangeCurrentStreamId(device_index, other_stream_id);
  absl::Cleanup cleanup = [device_index, default_stream_id] {
    ExchangeCurrentStreamId(device_index, default_stream_id);
  };
  InsertStreamWait(device_index, other_stream_id, *snapshot);

  // Schedule a transfer on the other stream.
  std::vector<char> dst = {0x00};
  auto d2h_transfer = DeviceToHostTransfer::IntoBytes(constant_ref, dst.data());
  ScheduleDeviceToHostTransfer(std::move(d2h_transfer));
  auto d2h_snapshot_future = MaterializeStream(device_index, other_stream_id,
                                               MaterializationReason::kUnknown);
  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<EventSnapshot> d2h_snapshot,
                          d2h_snapshot_future.Await());

  // The transfer is not yet started.
  TT_ASSERT_OK_AND_ASSIGN(bool completed, d2h_snapshot->Query());
  ASSERT_FALSE(completed);
  EXPECT_EQ(dst[0], 0x00);

  // Complete the work for the default stream.
  promise.Set(absl::OkStatus());

  // The transfer will then start and complete.
  ASSERT_TRUE(d2h_snapshot->Synchronize().ok());
  EXPECT_EQ(dst[0], 0x01);
}

TEST(CompilationModeTest, GetCompilationMode) {
  EXPECT_EQ(GetCompilationMode(EagerMode::kInternalDeferAll),
            CompilationMode::kFastRuntime);
  EXPECT_EQ(GetCompilationMode(EagerMode::kDeferAndFuse),
            CompilationMode::kFastRuntime);
  EXPECT_EQ(GetCompilationMode(EagerMode::kInternalCompileFxGraph),
            CompilationMode::kFastRuntime);
  EXPECT_EQ(GetCompilationMode(EagerMode::kDeferNever),
            CompilationMode::kFastCompile);
  EXPECT_EQ(GetCompilationMode(EagerMode::kDeferNeverAndLaunchBlocking),
            CompilationMode::kFastCompile);
}

TEST(RepeatedOpsTest, DetectsRepeatdOps) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create three lists for three different OpNames.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_b,
      DeviceBufferList::CreateDeferred(OpName::kBaddbmmDtype, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto list_b = refs_b[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_c,
                          DeviceBufferList::CreateDeferred(
                              OpName::kCountNonzeroDimIntList, DummyBuilder, {},
                              OpParamCacheKeys::Empty(), {shape}));
  auto list_c = refs_c[0].device_buffer_list();

  // Push these ops into the queue three times, (abcabcabc)
  // For the first 5 ops, no repetition is detected.
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_c));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));

  // When we push the second repetition, it is detected.
  EXPECT_TRUE(RecordDeferredOpAndDetectRepetition(list_c));

  // The third repetition is only detected when it ends.
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));
  EXPECT_TRUE(RecordDeferredOpAndDetectRepetition(list_c));
}

TEST(RepeatedOpsTest, DoesNotDetectRepetitionsOverSnapshots) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create three lists for three different OpNames.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_b,
      DeviceBufferList::CreateDeferred(OpName::kBaddbmmDtype, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto list_b = refs_b[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(auto refs_c,
                          DeviceBufferList::CreateDeferred(
                              OpName::kCountNonzeroDimIntList, DummyBuilder, {},
                              OpParamCacheKeys::Empty(), {shape}));
  auto list_c = refs_c[0].device_buffer_list();

  // Push these ops into the queue repeatedly, but interrupt with a snapshot.
  // For the first 5 ops, no repetition is detected.
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_c));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));

  // Request an event snapshot.
  auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  (void)EventSnapshot::Record(device_index, stream_id);

  // When we push the second repetition, it is not detected, as the state was
  // reset after the snapshot.
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_c));

  // Pushing a third repetition also does not trigger detection yet.
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_c));

  // But if the loop continues further, we do detect it starting from the
  // beginning of the loop, as (cab)x2.
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_TRUE(RecordDeferredOpAndDetectRepetition(list_b));

  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_c));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_TRUE(RecordDeferredOpAndDetectRepetition(list_b));
}

TEST(RepeatedOpsTest, ReportsSuspendedGraphBreakOnResume) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  // Create two lists for two different OpNames.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_b,
      DeviceBufferList::CreateDeferred(OpName::kBaddbmmDtype, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto list_b = refs_b[0].device_buffer_list();

  // There is nothing to report before graph breaks are ever suspended.
  EXPECT_FALSE(ResumeRepeatedOpsGraphBreaks(device_index, stream_id));

  // Push (abab), suspending graph breaks before the op that completes the
  // second repetition: the repetition is detected, but not reported.
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  SuspendRepeatedOpsGraphBreaks(list_b);
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));

  // The requested graph break is reported once, on resume.
  EXPECT_TRUE(ResumeRepeatedOpsGraphBreaks(device_index, stream_id));
  EXPECT_FALSE(ResumeRepeatedOpsGraphBreaks(device_index, stream_id));

  // Once resumed, the next repetition is reported right away.
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_TRUE(RecordDeferredOpAndDetectRepetition(list_b));
}

TEST(RepeatedOpsTest, GraphBreakEndsSuspension) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  // Create two lists for two different OpNames.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_a = refs_a[0].device_buffer_list();
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_b,
      DeviceBufferList::CreateDeferred(OpName::kBaddbmmDtype, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto list_b = refs_b[0].device_buffer_list();

  // The suspension also covers the ops recorded after the op that started it.
  SuspendRepeatedOpsGraphBreaks(list_a);
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_b));

  // A graph break satisfies the requested graph break and ends the suspension,
  // so the next repetition is reported right away.
  InsertGraphBreak(device_index, stream_id, {}, MaterializationReason::kUnknown,
                   GraphSplitMode::kSplitGraph);
  // Take the graph break's work off the stream, as the ops cannot be executed.
  EXPECT_EQ(GetReadyWork(device_index, stream_id).size(), 1);
  EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list_a));
  EXPECT_TRUE(RecordDeferredOpAndDetectRepetition(list_b));
  EXPECT_FALSE(ResumeRepeatedOpsGraphBreaks(device_index, stream_id));
}

TEST(RepeatedOpsTest, ResetDropsSuspendedGraphBreak) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  TT_ASSERT_OK_AND_ASSIGN(auto refs, DeviceBufferList::CreateDeferred(
                                         OpName::kAdd, DummyBuilder, {},
                                         OpParamCacheKeys::Empty(), {shape}));
  auto list = refs[0].device_buffer_list();

  // Push (aaaa) while graph breaks are suspended, so that a graph break is
  // requested.
  SuspendRepeatedOpsGraphBreaks(list);
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list));
  }
  EXPECT_TRUE(ResumeRepeatedOpsGraphBreaks(device_index, stream_id));

  // Push (aa) while graph breaks are suspended, completing another repetition.
  // Resetting the heuristic drops the requested graph break.
  SuspendRepeatedOpsGraphBreaks(list);
  for (int i = 0; i < 2; ++i) {
    EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list));
  }
  ResetRepeatedOpsHeuristicState();
  EXPECT_FALSE(ResumeRepeatedOpsGraphBreaks(device_index, stream_id));

  // So does clearing the streams.
  SuspendRepeatedOpsGraphBreaks(list);
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(RecordDeferredOpAndDetectRepetition(list));
  }
  ClearAllStreams();
  EXPECT_FALSE(ResumeRepeatedOpsGraphBreaks(device_index, stream_id));
}
}  // namespace

}  // namespace torch_tpu
