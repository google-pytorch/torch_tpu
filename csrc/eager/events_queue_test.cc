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
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/traversal.h"
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
  EventSnapshot::Record(device_index, stream_id, {});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;

  ASSERT_EQ(traversals.size(), 1);
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.arguments(), testing::IsEmpty());
  EXPECT_THAT(traversal.execution_order(),
              testing::ElementsAre(ref_a.device_buffer_list(),
                                   ref_b.device_buffer_list()));
  EXPECT_THAT(traversal.outputs(), testing::ElementsAre(ref_a, ref_b));
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
  EventSnapshot::Record(device_index, stream_id, {});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;
  EXPECT_THAT(traversals, testing::IsEmpty());
}

TEST_F(EventsQueueTest, SyncIgnoresEmptyBuffers) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create an empty buffer and record its DataPtr creation.
  TT_ASSERT_OK_AND_ASSIGN(
      auto ref, CreateEmptyDeviceBufferRef(shape.dimensions(), shape.dtype()));
  ref.device_buffer_list()->IncrementLiveDataPtrs(ref.index());

  // The empty buffer does not need to be synced.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  EventSnapshot::Record(device_index, stream_id, {});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;
  EXPECT_THAT(traversals, testing::IsEmpty());
}

TEST_F(EventsQueueTest, IgnoresAlreadyMaterializedBuffers) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);

  // Create a fully-materialized buffer (filled with zeros) and record its
  // DataPtr creation.
  TT_ASSERT_OK_AND_ASSIGN(auto ref,
                          TpuMallocAndMemcpyHtoD(/*host_data=*/nullptr,
                                                 mlir::ElementType::UI8, {1}));
  ASSERT_TRUE(ref.is_materialized());
  ref.device_buffer_list()->IncrementLiveDataPtrs(ref.index());

  // The placeholder does not need to be synced.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  EventSnapshot::Record(device_index, stream_id, {});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;
  EXPECT_THAT(traversals, testing::IsEmpty());
}

TEST_F(EventsQueueTest, ClearsBuffersAfterMaterialization) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{1}, mlir::ElementType::UI8);

  TT_ASSERT_OK_AND_ASSIGN(auto input_ref,
                          TpuMallocAndMemcpyHtoD(/*host_data=*/nullptr,
                                                 mlir::ElementType::UI8, {1}));
  ASSERT_TRUE(input_ref.is_materialized());
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
  EventSnapshot::Record(device_index, stream_id, {});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;
  EXPECT_THAT(traversals, testing::IsEmpty());
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
  EventSnapshot::Record(device_index, stream_id, {});
  auto pending_snapshots1 =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots1.size(), 1);
  const auto& traversals1 = pending_snapshots1[0].traversals;
  EXPECT_THAT(traversals1, testing::IsEmpty());

  // Create a second data pointer to the same buffer.
  ref.device_buffer_list()->IncrementLiveDataPtrs(ref.index());

  EventSnapshot::Record(device_index, stream_id, {});
  auto pending_snapshots2 =
      PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots2.size(), 1);
  const auto& traversals2 = pending_snapshots2[0].traversals;

  // The buffer is live, but the DeferredOp was cleared and so does not need
  // to be materialized.
  EXPECT_THAT(traversals2, testing::IsEmpty());
}

TEST_F(EventsQueueTest, SingleTraversalIfPossible) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Put three deferred ops in the queue.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  auto list_a = ref_a.device_buffer_list();
  RecordDeferredOpCreated(list_a);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = ref_c.device_buffer_list();
  RecordDeferredOpCreated(list_c);

  // Ask for a plan to materialize the stream.
  const auto device_index = list_a->device_index();
  const auto stream_id = list_a->stream_id();
  EventSnapshot::Record(device_index, stream_id, {list_a, list_b, list_c});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;

  // There are no required split points, so a single traversal is returned
  // with all required nodes as outputs.
  ASSERT_EQ(traversals.size(), 1);
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.arguments(), testing::IsEmpty());
  EXPECT_THAT(traversal.execution_order(),
              testing::ElementsAre(list_a, list_b, list_c));
  EXPECT_THAT(traversal.outputs(), testing::ElementsAre(ref_a, ref_b, ref_c));
}

TEST_F(EventsQueueTest, SplitModeRespected) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Put five deferred ops in the queue:
  // [a (split after), b, c (split before), d (split both), e]
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape},
                                           OpSplitMode::kSplitAfter));
  auto list_a = refs_a[0].device_buffer_list();
  RecordDeferredOpCreated(list_a);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_b = refs_b[0].device_buffer_list();
  RecordDeferredOpCreated(list_b);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape},
                                           OpSplitMode::kSplitBefore));
  auto list_c = refs_c[0].device_buffer_list();
  RecordDeferredOpCreated(list_c);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_d, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape},
                                           OpSplitMode::kSplitBoth));
  auto list_d = refs_d[0].device_buffer_list();
  RecordDeferredOpCreated(list_d);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_e, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_e = refs_e[0].device_buffer_list();
  RecordDeferredOpCreated(list_e);

  // Ask for a plan to materialize everything.
  const auto device_index = list_a->device_index();
  const auto stream_id = list_a->stream_id();
  EventSnapshot::Record(device_index, stream_id,
                        {list_a, list_b, list_c, list_d, list_e});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;

  // Each op ends up in its own traversal.
  // a is split after, so we must split between a | b.
  // c is split before, so we must split between b | c.
  // d is split both, so we must split between c | d and between d | e.
  ASSERT_EQ(traversals.size(), 5);
  EXPECT_THAT(traversals[0]->execution_order(), testing::ElementsAre(list_a));
  EXPECT_THAT(traversals[1]->execution_order(), testing::ElementsAre(list_b));
  EXPECT_THAT(traversals[2]->execution_order(), testing::ElementsAre(list_c));
  EXPECT_THAT(traversals[3]->execution_order(), testing::ElementsAre(list_d));
  EXPECT_THAT(traversals[4]->execution_order(), testing::ElementsAre(list_e));
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
  EventSnapshot::Record(device_index, stream_id, {list_d});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;
  ASSERT_EQ(traversals.size(), 1);

  // The dead ops should be stripped.
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.execution_order(), testing::ElementsAre(list_d));
  EXPECT_THAT(traversal.outputs(), testing::ElementsAre(ref_d));
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

  // Create a fourth, non-dead node and request a traversal plan for it.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_d, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_d = refs_d[0];
  auto list_d = ref_d.device_buffer_list();
  RecordDeferredOpCreated(list_d);

  const auto device_index = list_d->device_index();
  const auto stream_id = list_d->stream_id();
  EventSnapshot::Record(device_index, stream_id, {list_d});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;

  // We get one traversal, as there were no required split points.
  ASSERT_EQ(traversals.size(), 1);

  // The side effect op is retained, and marked as an output to force it to
  // execute. This keeps its dependent inputs a and b from elimination as well.
  const Traversal& traversal = *traversals[0];
  ASSERT_EQ(traversal.execution_order().size(), 4);
  EXPECT_EQ(traversal.execution_order()[0].get(), dead_a_ptr);
  EXPECT_EQ(traversal.execution_order()[1].get(), dead_b_ptr);
  EXPECT_EQ(traversal.execution_order()[2].get(), dead_c_ptr);
  EXPECT_EQ(traversal.execution_order()[3], list_d);
  ASSERT_EQ(traversal.outputs().size(), 2);
  EXPECT_EQ(traversal.outputs()[0].device_buffer_list().get(), dead_c_ptr);
  EXPECT_EQ(traversal.outputs()[1], ref_d);
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
  EventSnapshot::Record(device_index, stream_id, {non_empty_list});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;

  // We get one traversal, as there were no required split points.
  ASSERT_EQ(traversals.size(), 1);

  // The empty tensor is not included in the traversal.
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.execution_order(),
              testing::ElementsAre(non_empty_list));
  EXPECT_THAT(traversal.outputs(), testing::ElementsAre(non_empty_ref));
}

TEST_F(EventsQueueTest, UsedEmptyOpsMaterializedOnFirstUse) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a pattern of [empty, non_empty, non_empty] where the second empty
  // tensor uses the empty tensor as an input.
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
      auto first_non_empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto first_non_empty_ref = first_non_empty_refs[0];
  auto first_non_empty_list = first_non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(first_non_empty_list);
  first_non_empty_list->IncrementLiveDataPtrs(first_non_empty_ref.index());

  TT_ASSERT_OK_AND_ASSIGN(
      auto second_non_empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {empty_ref},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto second_non_empty_ref = second_non_empty_refs[0];
  auto second_non_empty_list = second_non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(second_non_empty_list);
  second_non_empty_list->IncrementLiveDataPtrs(second_non_empty_ref.index());

  // Prepare traversals for the non-empty tensors only.
  const auto device_index = first_non_empty_list->device_index();
  const auto stream_id = first_non_empty_list->stream_id();
  EventSnapshot::Record(device_index, stream_id,
                        {first_non_empty_list, second_non_empty_list});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;

  // We get one traversal, as there were no required split points.
  ASSERT_EQ(traversals.size(), 1);

  // The empty tensor is included in the traversal, immediately before its use.
  // It is not an explicit output.
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.execution_order(),
              testing::ElementsAre(first_non_empty_list, empty_list,
                                   second_non_empty_list));
  EXPECT_THAT(traversal.outputs(),
              testing::ElementsAre(first_non_empty_ref, second_non_empty_ref));
}

TEST_F(EventsQueueTest, MaterializationAppendsExplicitEmptyOps) {
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

  // Explicitly mark the empty tensor as an output.
  EventSnapshot::Record(non_empty_list->device_index(),
                        non_empty_list->stream_id(),
                        {empty_list, non_empty_list});
  auto pending_snapshots = PrepareSnapshotMaterializations(
      non_empty_list->device_index(), non_empty_list->stream_id());

  // We get one traversal, as there were no required split points.
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;
  ASSERT_EQ(traversals.size(), 1);

  // The empty tensor is appended after the non-empty list as an output.
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.execution_order(),
              testing::ElementsAre(non_empty_list, empty_list));
  EXPECT_THAT(traversal.outputs(),
              testing::ElementsAre(non_empty_ref, empty_ref));
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

  auto pending_event_snapshots = PrepareSnapshotMaterializations(
      device_index, pending_stream_snapshots[0].stream_id);
  EXPECT_EQ(pending_event_snapshots.size(), 1);
  const auto& traversals = pending_event_snapshots[0].traversals;

  // The execution will contain a and b but not c. Only b is output as it is
  // the only executed op with a live DataPtr.
  ASSERT_EQ(traversals.size(), 1);
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.arguments(), testing::IsEmpty());
  EXPECT_THAT(traversal.execution_order(),
              testing::ElementsAre(list_a, list_b));
  EXPECT_THAT(traversal.outputs(), testing::ElementsAre(ref_b));
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
  EventSnapshot::Record(device_index, default_stream_id, {});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, default_stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;

  // The execution will contain a and b but not c. Only b is output as it is
  // the only executed op with a live DataPtr.
  ASSERT_EQ(traversals.size(), 1);
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.arguments(), testing::IsEmpty());
  EXPECT_THAT(traversal.execution_order(),
              testing::ElementsAre(list_a, list_b));
  EXPECT_THAT(traversal.outputs(), testing::ElementsAre(ref_b));
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
  EventSnapshot::Record(device_index, non_default_stream_id, {});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, non_default_stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;

  // The execution will contain c, but not a or b, even though they were
  // created first; different streams have no relative order.
  ASSERT_EQ(traversals.size(), 1);
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.arguments(), testing::IsEmpty());
  EXPECT_THAT(traversal.execution_order(), testing::ElementsAre(list_c));
  EXPECT_THAT(traversal.outputs(), testing::ElementsAre(ref_c));
}

TEST_F(EventsQueueTest, RetainsFailedCompletedFutures) {
  ClearAllStreams();
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  xla::Future<void> failed_future(TT_ERROR(error::kInvalidArgument)
                                  << "Fake error");
  ASSERT_TRUE(failed_future.IsReady());

  RecordAsyncDeviceToHost(std::move(failed_future));

  EventSnapshot::Record(device_index, stream_id);
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);

  ASSERT_EQ(pending_snapshots[0].futures.size(), 1);

  auto status = pending_snapshots[0].futures[0].Await();
  EXPECT_EQ(status.code(), error::kInvalidArgument);
}

TEST_F(EventsQueueTest, PrunesSuccessfulCompletedFutures) {
  ClearAllStreams();
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  xla::Future<void> ok_future(absl::OkStatus());
  ASSERT_TRUE(ok_future.IsReady());

  RecordAsyncDeviceToHost(std::move(ok_future));

  EventSnapshot::Record(device_index, stream_id);
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);

  EXPECT_EQ(pending_snapshots[0].futures.size(), 0);
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
  EventSnapshot::Record(device_index, stream_id, {});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);

  ASSERT_EQ(pending_snapshots.size(), 1);
  EXPECT_EQ(pending_snapshots[0].traversals.size(), 0);

  // A stream materialization that must materialize the empty tensor does so.
  EventSnapshot::Record(device_index, stream_id, {empty_list});
  pending_snapshots = PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;
  ASSERT_EQ(traversals.size(), 1);
  ASSERT_EQ(traversals[0]->execution_order().size(), 1);
  EXPECT_EQ(traversals[0]->execution_order()[0], empty_list);
  ASSERT_EQ(traversals[0]->outputs().size(), 1);
  EXPECT_EQ(traversals[0]->outputs()[0], empty_ref);
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
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  EXPECT_EQ(pending_snapshots[0].traversals.size(), 0);

  // A device materialization that must materialize the empty tensor does so.
  RecordDeviceSnapshots(device_index, {empty_list});
  pending_snapshots = PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& traversals = pending_snapshots[0].traversals;
  ASSERT_EQ(traversals.size(), 1);
  ASSERT_EQ(traversals[0]->execution_order().size(), 1);
  EXPECT_EQ(traversals[0]->execution_order()[0], empty_list);
  ASSERT_EQ(traversals[0]->outputs().size(), 1);
  EXPECT_EQ(traversals[0]->outputs()[0], empty_ref);
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

  EventSnapshot::Record(list_a->device_index(), list_a->stream_id(), {});

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);

  EventSnapshot::Record(list_b->device_index(), list_b->stream_id(), {});

  // Get the pending snapshots.
  auto pending_snapshots = PrepareSnapshotMaterializations(
      list_a->device_index(), list_a->stream_id());

  // Each snapshot request results in a pending snapshot.
  ASSERT_EQ(pending_snapshots.size(), 2);

  // The first pending snapshot will materialize a (only).
  const auto& first_snapshot = pending_snapshots[0];
  ASSERT_EQ(first_snapshot.traversals.size(), 1);
  ASSERT_EQ(first_snapshot.traversals[0]->execution_order().size(), 1);
  EXPECT_EQ(first_snapshot.traversals[0]->execution_order()[0], list_a);
  ASSERT_EQ(first_snapshot.traversals[0]->outputs().size(), 1);
  EXPECT_EQ(first_snapshot.traversals[0]->outputs()[0], ref_a);

  // The second pending snapshot will materialize b (only).
  const auto& second_snapshot = pending_snapshots[1];
  ASSERT_EQ(second_snapshot.traversals.size(), 1);
  ASSERT_EQ(second_snapshot.traversals[0]->execution_order().size(), 1);
  EXPECT_EQ(second_snapshot.traversals[0]->execution_order()[0], list_b);
  ASSERT_EQ(second_snapshot.traversals[0]->outputs().size(), 1);
  EXPECT_EQ(second_snapshot.traversals[0]->outputs()[0], ref_b);
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

  EventSnapshot::Record(list_a->device_index(), list_a->stream_id(), {});

  // Enqueue an op after the snapshot, but don't request a second snapshot.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);

  // Get the pending snapshots.
  auto pending_snapshots = PrepareSnapshotMaterializations(
      list_a->device_index(), list_a->stream_id());

  // List b is left in the queue.
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& first_snapshot = pending_snapshots[0];
  ASSERT_EQ(first_snapshot.traversals.size(), 1);
  ASSERT_EQ(first_snapshot.traversals[0]->execution_order().size(), 1);
  EXPECT_EQ(first_snapshot.traversals[0]->execution_order()[0], list_a);
  ASSERT_EQ(first_snapshot.traversals[0]->outputs().size(), 1);
  EXPECT_EQ(first_snapshot.traversals[0]->outputs()[0], ref_a);
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

  EventSnapshot::Record(device_index, stream_id, {list_single});
  auto pending_snapshots =
      PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& single_traversals = pending_snapshots[0].traversals;
  ASSERT_EQ(single_traversals.size(), 1);
  EXPECT_THAT(single_traversals[0]->execution_order(),
              testing::ElementsAre(list_single));
  EXPECT_THAT(single_traversals[0]->outputs(),
              testing::ElementsAre(ref_single));

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

  EventSnapshot::Record(device_index, stream_id, {list_op, empty_list});
  pending_snapshots = PrepareSnapshotMaterializations(device_index, stream_id);
  ASSERT_EQ(pending_snapshots.size(), 1);
  const auto& combined_traversals = pending_snapshots[0].traversals;
  ASSERT_EQ(combined_traversals.size(), 1);
  EXPECT_THAT(combined_traversals[0]->execution_order(),
              testing::ElementsAre(list_op, empty_list));
  EXPECT_THAT(combined_traversals[0]->outputs(),
              testing::ElementsAre(ref_op, empty_ref));
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
    std::vector<DeviceBufferRef> expected_outputs;
    for (int i = 0; i < count; ++i) {
      TT_ASSERT_OK_AND_ASSIGN(
          auto refs,
          DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
      nodes.push_back(refs[0].device_buffer_list());
      expected_outputs.push_back(refs[0]);
      RecordDeferredOpCreated(refs[0].device_buffer_list());
    }
    EventSnapshot::Record(device_index, stream_id, nodes);
    auto pending_snapshots =
        PrepareSnapshotMaterializations(device_index, stream_id);
    ASSERT_EQ(pending_snapshots.size(), 1);
    const auto& traversals = pending_snapshots[0].traversals;
    ASSERT_EQ(traversals.size(), 1);
    EXPECT_THAT(traversals[0]->execution_order(),
                testing::ElementsAreArray(nodes));
    EXPECT_THAT(traversals[0]->outputs(),
                testing::ElementsAreArray(expected_outputs));
  }
}

TEST_F(EventsQueueTest, CompletesLongSnapshotChainOnSmallStack) {
  ClearAllStreams();
  // Snapshots recorded while a transfer is pending wait for it, and each one
  // also waits for the snapshot before it, as while model weights load.
  auto [transfer_promise, transfer_future] = xla::MakePromise<void>();
  RecordAsyncDeviceToHost(std::move(transfer_future));
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
  xla::Promise<void> promise = std::move(transfer_promise);
  std::unique_ptr<tsl::Thread> thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions{.stack_size = 1 << 20}, "complete_transfer",
      [&promise] { promise.Set(); }));
  thread.reset();
  EXPECT_TRUE(last_snapshot->Synchronize().ok());
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
  EventSnapshot::Record(device_index, stream_id);

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

// Returns `duration` in milliseconds, the unit of EventSnapshot::ElapsedTime.
double ToMilliseconds(std::chrono::steady_clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

TEST(EventSnapshotTest, ElapsedTimeMeasuresCompletionTimes) {
  auto [start_promise, start_future] = xla::MakePromise<void>();
  auto [end_promise, end_future] = xla::MakePromise<void>();
  EventSnapshot start(std::move(start_future));
  EventSnapshot end(std::move(end_future));
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
  EventSnapshot start(std::move(earlier_future));
  auto [chain_promise, chain_future] = xla::MakePromise<void>();
  start.OkCallback(std::move(chain_promise));
  auto [later_promise, later_future] = xla::MakePromise<void>();
  EventSnapshot end(xla::JoinFutures({later_future, chain_future}));
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
  EventSnapshot start(std::move(work_future));
  auto [chain_promise, chain_future] = xla::MakePromise<void>();
  start.OkCallback(std::move(chain_promise));
  EventSnapshot end(std::move(chain_future));
  TT_ASSERT_OK_AND_ASSIGN(bool end_ready, end.Query());
  ASSERT_FALSE(end_ready);

  work_promise.Set();
  TT_ASSERT_OK_AND_ASSIGN(end_ready, end.Query());
  ASSERT_TRUE(end_ready);
  TT_ASSERT_OK_AND_ASSIGN(double elapsed_ms, start.ElapsedTime(end));
  EXPECT_GE(elapsed_ms, 0.0);
}

TEST(EventSnapshotTest, ElapsedTimeReturnsEventError) {
  EventSnapshot ok_event{xla::Future<void>(absl::OkStatus())};
  EventSnapshot failed_event{
      xla::Future<void>(TT_ERROR(error::kInvalidArgument) << "fake error")};
  EXPECT_EQ(ok_event.ElapsedTime(failed_event).status().code(),
            error::kInvalidArgument);
}
}  // namespace

}  // namespace torch_tpu
