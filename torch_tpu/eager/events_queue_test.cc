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
#include "torch_tpu/eager/traversal.h"
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
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Record creation of two new DataPtrs, one for buffer "a" and one for "b".
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_a = refs_or.value()[0];
  RecordDeferredOpCreated(ref_a.device_buffer_list());
  RecordNewDataPtrCreated(ref_a);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_b = refs_or.value()[0];
  RecordDeferredOpCreated(ref_b.device_buffer_list());
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
  ClearAllStreams();
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

TEST_F(EventsQueueTest, SyncIgnoresEmptyBuffers) {
  ClearAllStreams();
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
  ClearAllStreams();
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
  ClearAllStreams();
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
  RecordDeferredOpCreated(ref.device_buffer_list());
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
  ClearAllStreams();
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
  ClearAllStreams();

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

TEST_F(EventsQueueTest, NoTraversalIfNothingToMaterialize) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Put three deferred ops in the queue.
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto list_a = refs_or.value()[0].device_buffer_list();
  RecordDeferredOpCreated(list_a);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto list_b = refs_or.value()[0].device_buffer_list();
  RecordDeferredOpCreated(list_b);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto list_c = refs_or.value()[0].device_buffer_list();
  RecordDeferredOpCreated(list_c);

  // Ask for a traversal to materialize nothing.
  auto traversals_or = PrepareMaterializationTraversals({});
  ASSERT_TRUE(traversals_or.ok());

  // Nothing needs to be executed to materialize nothing.
  EXPECT_THAT(traversals_or.value(), testing::IsEmpty());
}

TEST_F(EventsQueueTest, MissingNodesIgnored) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Put three deferred ops in the queue.
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_a = refs_or.value()[0];
  auto list_a = ref_a.device_buffer_list();
  RecordDeferredOpCreated(list_a);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_b = refs_or.value()[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto list_c = refs_or.value()[0].device_buffer_list();
  RecordDeferredOpCreated(list_c);

  // Create a fourth node, but don't put it in the queue.
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto list_d = refs_or.value()[0].device_buffer_list();

  // Ask for a plan to materialize b and d.
  auto traversals_or = PrepareMaterializationTraversals({list_b, list_d});
  ASSERT_TRUE(traversals_or.ok());

  // a is included because it's not dead code, and is an unused by b; it needs
  // to be materialized so that it gets executed.
  // b needs to be materialized as it was an explicit materialization target.
  // c is after the last known node (b) so it doesn't need to be executed.
  // d is not in the queue, so it is ignored.
  ASSERT_EQ(traversals_or.value().size(), 1);
  const Traversal& traversal = *traversals_or.value()[0];
  EXPECT_THAT(traversal.arguments(), testing::IsEmpty());
  EXPECT_THAT(traversal.execution_order(),
              testing::ElementsAre(list_a, list_b));
  EXPECT_THAT(traversal.outputs(), testing::ElementsAre(ref_a, ref_b));
}

TEST_F(EventsQueueTest, SingleTraversalIfPossible) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Put three deferred ops in the queue.
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_a = refs_or.value()[0];
  auto list_a = ref_a.device_buffer_list();
  RecordDeferredOpCreated(list_a);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_b = refs_or.value()[0];
  auto list_b = ref_b.device_buffer_list();
  RecordDeferredOpCreated(list_b);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_c = refs_or.value()[0];
  auto list_c = ref_c.device_buffer_list();
  RecordDeferredOpCreated(list_c);

  // Ask for a plan to materialize a, b, and c.
  auto traversals_or =
      PrepareMaterializationTraversals({list_a, list_b, list_c});
  ASSERT_TRUE(traversals_or.ok());

  // There are no required split points, so a single traversal is returned
  // with all required nodes as outputs.
  ASSERT_EQ(traversals_or.value().size(), 1);
  const Traversal& traversal = *traversals_or.value()[0];
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
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                             OpParamCacheKeys::Empty(), {shape},
                                             OpSplitMode::kSplitAfter);
  ASSERT_TRUE(refs_or.ok());
  auto list_a = refs_or.value()[0].device_buffer_list();
  RecordDeferredOpCreated(list_a);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto list_b = refs_or.value()[0].device_buffer_list();
  RecordDeferredOpCreated(list_b);

  refs_or = DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                             OpParamCacheKeys::Empty(), {shape},
                                             OpSplitMode::kSplitBefore);
  ASSERT_TRUE(refs_or.ok());
  auto list_c = refs_or.value()[0].device_buffer_list();
  RecordDeferredOpCreated(list_c);

  refs_or = DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                             OpParamCacheKeys::Empty(), {shape},
                                             OpSplitMode::kSplitBoth);
  ASSERT_TRUE(refs_or.ok());
  auto list_d = refs_or.value()[0].device_buffer_list();
  RecordDeferredOpCreated(list_d);

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto list_e = refs_or.value()[0].device_buffer_list();
  RecordDeferredOpCreated(list_e);

  // Ask for a plan to materialize everything.
  auto traversals_or = PrepareMaterializationTraversals(
      {list_a, list_b, list_c, list_d, list_e});
  ASSERT_TRUE(traversals_or.ok());

  // Each op ends up in its own traversal.
  // a is split after, so we must split between a | b.
  // c is split before, so we must split between b | c.
  // d is split both, so we must split between c | d and between d | e.
  ASSERT_EQ(traversals_or.value().size(), 5);
  EXPECT_THAT(traversals_or.value()[0]->execution_order(),
              testing::ElementsAre(list_a));
  EXPECT_THAT(traversals_or.value()[1]->execution_order(),
              testing::ElementsAre(list_b));
  EXPECT_THAT(traversals_or.value()[2]->execution_order(),
              testing::ElementsAre(list_c));
  EXPECT_THAT(traversals_or.value()[3]->execution_order(),
              testing::ElementsAre(list_d));
  EXPECT_THAT(traversals_or.value()[4]->execution_order(),
              testing::ElementsAre(list_e));
}

TEST_F(EventsQueueTest, DeadCodeEliminated) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Add three deferred ops to the queue, but drop all references to them.
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  {
    refs_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs_or.ok());
    auto list_a = refs_or.value()[0].device_buffer_list();
    RecordDeferredOpCreated(list_a);

    refs_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs_or.ok());
    auto list_b = refs_or.value()[0].device_buffer_list();
    RecordDeferredOpCreated(list_b);

    refs_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs_or.ok());
    auto list_c = refs_or.value()[0].device_buffer_list();
    RecordDeferredOpCreated(list_c);

    refs_or = std::vector<DeviceBufferRef>();
  }

  // Create a fourth, non-dead node and request a traversal plan for it.
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_d = refs_or.value()[0];
  auto list_d = ref_d.device_buffer_list();
  RecordDeferredOpCreated(list_d);

  auto traversals_or = PrepareMaterializationTraversals({list_d});
  ASSERT_TRUE(traversals_or.ok());
  ASSERT_EQ(traversals_or.value().size(), 1);

  // The dead ops should be stripped.
  const Traversal& traversal = *traversals_or.value()[0];
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
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  {
    refs_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs_or.ok());
    auto ref_a = refs_or.value()[0];
    auto list_a = refs_or.value()[0].device_buffer_list();
    RecordDeferredOpCreated(list_a);
    dead_a_ptr = list_a.get();

    refs_or =
        DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref_a},
                                         OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs_or.ok());
    auto ref_b = refs_or.value()[0];
    auto list_b = refs_or.value()[0].device_buffer_list();
    RecordDeferredOpCreated(list_b);
    dead_b_ptr = list_b.get();

    refs_or = DeviceBufferList::CreateDeferred(
        OpName::kDistributedAllReduce, DummyBuilder, {ref_b},
        OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs_or.ok());
    auto list_c = refs_or.value()[0].device_buffer_list();
    RecordDeferredOpCreated(list_c);
    dead_c_ptr = list_c.get();

    refs_or = std::vector<DeviceBufferRef>();
  }

  // Create a fourth, non-dead node and request a traversal plan for it.
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref_d = refs_or.value()[0];
  auto list_d = ref_d.device_buffer_list();
  RecordDeferredOpCreated(list_d);

  auto traversals_or = PrepareMaterializationTraversals({list_d});
  ASSERT_TRUE(traversals_or.ok());

  // We get one traversal, as there were no required split points.
  ASSERT_EQ(traversals_or.value().size(), 1);

  // The side effect op is retained, and marked as an output to force it to
  // execute. This keeps its dependent inputs a and b from elimination as well.
  const Traversal& traversal = *traversals_or.value()[0];
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
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or =
      DeviceBufferList::CreateDeferred(OpName::kEmpty, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(),
                                       /*output_shapes=*/{shape});
  ASSERT_TRUE(refs_or.ok());
  auto empty_ref = refs_or.value()[0];
  auto empty_list = empty_ref.device_buffer_list();
  RecordDeferredOpCreated(empty_list);
  RecordNewDataPtrCreated(empty_ref);

  // Create a non-empty deferred tensor that does not rely on the empty tensor.
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto non_empty_ref = refs_or.value()[0];
  auto non_empty_list = non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(non_empty_list);
  RecordNewDataPtrCreated(non_empty_ref);

  auto traversals_or = PrepareMaterializationTraversals({non_empty_list});
  ASSERT_TRUE(traversals_or.ok());

  // We get one traversal, as there were no required split points.
  ASSERT_EQ(traversals_or.value().size(), 1);

  // The empty tensor is not included in the traversal.
  const Traversal& traversal = *traversals_or.value()[0];
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
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or =
      DeviceBufferList::CreateDeferred(OpName::kEmpty, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(),
                                       /*output_shapes=*/{shape});
  ASSERT_TRUE(refs_or.ok());
  auto empty_ref = refs_or.value()[0];
  auto empty_list = empty_ref.device_buffer_list();
  RecordDeferredOpCreated(empty_list);
  RecordNewDataPtrCreated(empty_ref);

  // Create a non-empty deferred tensor that does not rely on the empty tensor.
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto first_non_empty_ref = refs_or.value()[0];
  auto first_non_empty_list = first_non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(first_non_empty_list);
  RecordNewDataPtrCreated(first_non_empty_ref);

  refs_or =
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {empty_ref},
                                       OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto second_non_empty_ref = refs_or.value()[0];
  auto second_non_empty_list = second_non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(second_non_empty_list);
  RecordNewDataPtrCreated(second_non_empty_ref);

  // Prepare traversals for the non-empty tensors only.
  auto traversals_or = PrepareMaterializationTraversals(
      {first_non_empty_list, second_non_empty_list});
  ASSERT_TRUE(traversals_or.ok());

  // We get one traversal, as there were no required split points.
  ASSERT_EQ(traversals_or.value().size(), 1);

  // The empty tensor is included in the traversal, immediately before its use.
  // It is not an explicit output.
  const Traversal& traversal = *traversals_or.value()[0];
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
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or =
      DeviceBufferList::CreateDeferred(OpName::kEmpty, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(),
                                       /*output_shapes=*/{shape});
  ASSERT_TRUE(refs_or.ok());
  auto empty_ref = refs_or.value()[0];
  auto empty_list = empty_ref.device_buffer_list();
  RecordDeferredOpCreated(empty_list);
  RecordNewDataPtrCreated(empty_ref);

  // Create a non-empty deferred tensor that does not rely on the empty tensor.
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto non_empty_ref = refs_or.value()[0];
  auto non_empty_list = non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(non_empty_list);
  RecordNewDataPtrCreated(non_empty_ref);

  // Explicitly mark the empty tensor as an output.
  auto traversals_or =
      PrepareMaterializationTraversals({empty_list, non_empty_list});
  ASSERT_TRUE(traversals_or.ok());

  // We get one traversal, as there were no required split points.
  ASSERT_EQ(traversals_or.value().size(), 1);

  // The empty tensor is appended after the non-empty list as an output.
  const Traversal& traversal = *traversals_or.value()[0];
  EXPECT_THAT(traversal.execution_order(),
              testing::ElementsAre(non_empty_list, empty_list));
  EXPECT_THAT(traversal.outputs(),
              testing::ElementsAre(non_empty_ref, empty_ref));
}

TEST_F(EventsQueueTest, SideEffectsUsingPlaceholdersSkipped) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a placeholder tensor.
  auto placeholder_or =
      DeviceBufferList::CreatePlaceholder(shape.dimensions(), shape.dtype());
  ASSERT_TRUE(placeholder_or.ok());
  RecordNewDataPtrCreated(placeholder_or.value());

  // Create a side-effect op that uses the placeholder.
  auto side_effect_or = DeviceBufferList::CreateDeferred(
      OpName::kDistributedAllReduce, DummyBuilder, {placeholder_or.value()},
      OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(side_effect_or.ok());
  RecordDeferredOpCreated(side_effect_or.value()[0].device_buffer_list());
  RecordNewDataPtrCreated(side_effect_or.value()[0]);

  // Try to synchronize the side-effect op.
  auto traversals_or = PrepareMaterializationTraversals(
      {side_effect_or.value()[0].device_buffer_list()});
  ASSERT_TRUE(traversals_or.ok());

  // We should get nothing. The side-effect op was identified as part of a
  // compiled mode trace and therefore should not be materialized.
  ASSERT_EQ(traversals_or.value().size(), 0);
}

}  // namespace

}  // namespace torch_tpu
