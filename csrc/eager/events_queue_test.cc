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

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
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
#include "csrc/pjrt/pjrt_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

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
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_a = refs_a[0];
  RecordDeferredOpCreated(ref_a.device_buffer_list());
  RecordNewDataPtrCreated(ref_a);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_b = refs_b[0];
  RecordDeferredOpCreated(ref_b.device_buffer_list());
  RecordNewDataPtrCreated(ref_b);

  // Both a and b are live and unsynced and will be materialized on the next
  // stream materialization or device materialization.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareStreamTraversals(device_index, stream_id));
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
  RecordNewDataPtrCreated(ref);

  // The placeholder does not need to be synced.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareStreamTraversals(device_index, stream_id));
  EXPECT_THAT(traversals, testing::IsEmpty());
}

TEST_F(EventsQueueTest, SyncIgnoresEmptyBuffers) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create an empty buffer and record its DataPtr creation.
  TT_ASSERT_OK_AND_ASSIGN(
      auto ref, CreateEmptyDeviceBufferRef(shape.dimensions(), shape.dtype()));
  RecordNewDataPtrCreated(ref);

  // The empty buffer does not need to be synced.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareStreamTraversals(device_index, stream_id));
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
  RecordNewDataPtrCreated(ref);

  // The placeholder does not need to be synced.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareStreamTraversals(device_index, stream_id));
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
  RecordNewDataPtrCreated(input_ref);

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
  RecordNewDataPtrCreated(ref);

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
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareStreamTraversals(device_index, stream_id));
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
  RecordNewDataPtrCreated(ref);

  // Clear the events queue.
  ClearAllStreams();

  // The buffer is no longer tracked.
  TT_ASSERT_OK_AND_ASSIGN(auto traversals1,
                          PrepareStreamTraversals(device_index, stream_id));
  EXPECT_THAT(traversals1, testing::IsEmpty());

  // Create a second data pointer to the same buffer.
  RecordNewDataPtrCreated(ref);

  // The buffer is live, but the DeferredOp was cleared and so does not need
  // to be materialized.
  TT_ASSERT_OK_AND_ASSIGN(auto traversals2,
                          PrepareStreamTraversals(device_index, stream_id));
  EXPECT_THAT(traversals2, testing::IsEmpty());
}

TEST_F(EventsQueueTest, NoTraversalIfNothingToMaterialize) {
  ClearAllStreams();
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Put three deferred ops in the queue.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_a, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_a = refs_a[0].device_buffer_list();
  RecordDeferredOpCreated(list_a);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_b = refs_b[0].device_buffer_list();
  RecordDeferredOpCreated(list_b);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_c = refs_c[0].device_buffer_list();
  RecordDeferredOpCreated(list_c);

  // Ask for a traversal to materialize nothing.
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareMaterializationTraversals({}));

  // Nothing needs to be executed to materialize nothing.
  EXPECT_THAT(traversals, testing::IsEmpty());
}

TEST_F(EventsQueueTest, MissingNodesIgnored) {
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
  auto list_c = refs_c[0].device_buffer_list();
  RecordDeferredOpCreated(list_c);

  // Create a fourth node, but don't put it in the queue.
  TT_ASSERT_OK_AND_ASSIGN(auto refs_d, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto list_d = refs_d[0].device_buffer_list();

  // Ask for a plan to materialize b and d.
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareMaterializationTraversals({list_b, list_d}));

  // a is included because it's not dead code, and is an unused by b; it needs
  // to be materialized so that it gets executed.
  // b needs to be materialized as it was an explicit materialization target.
  // c is after the last known node (b) so it doesn't need to be executed.
  // d is not in the queue, so it is ignored.
  ASSERT_EQ(traversals.size(), 1);
  const Traversal& traversal = *traversals[0];
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

  // Ask for a plan to materialize a, b, and c.
  TT_ASSERT_OK_AND_ASSIGN(auto traversals, PrepareMaterializationTraversals(
                                               {list_a, list_b, list_c}));

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
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareMaterializationTraversals(
                              {list_a, list_b, list_c, list_d, list_e}));

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

  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareMaterializationTraversals({list_d}));
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

  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareMaterializationTraversals({list_d}));

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
  RecordNewDataPtrCreated(empty_ref);

  // Create a non-empty deferred tensor that does not rely on the empty tensor.
  TT_ASSERT_OK_AND_ASSIGN(
      auto non_empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto non_empty_ref = non_empty_refs[0];
  auto non_empty_list = non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(non_empty_list);
  RecordNewDataPtrCreated(non_empty_ref);

  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareMaterializationTraversals({non_empty_list}));

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
  RecordNewDataPtrCreated(empty_ref);

  // Create a non-empty deferred tensor that does not rely on the empty tensor.
  TT_ASSERT_OK_AND_ASSIGN(
      auto first_non_empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto first_non_empty_ref = first_non_empty_refs[0];
  auto first_non_empty_list = first_non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(first_non_empty_list);
  RecordNewDataPtrCreated(first_non_empty_ref);

  TT_ASSERT_OK_AND_ASSIGN(
      auto second_non_empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {empty_ref},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto second_non_empty_ref = second_non_empty_refs[0];
  auto second_non_empty_list = second_non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(second_non_empty_list);
  RecordNewDataPtrCreated(second_non_empty_ref);

  // Prepare traversals for the non-empty tensors only.
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareMaterializationTraversals(
                              {first_non_empty_list, second_non_empty_list}));

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
  RecordNewDataPtrCreated(empty_ref);

  // Create a non-empty deferred tensor that does not rely on the empty tensor.
  TT_ASSERT_OK_AND_ASSIGN(
      auto non_empty_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  auto non_empty_ref = non_empty_refs[0];
  auto non_empty_list = non_empty_ref.device_buffer_list();
  RecordDeferredOpCreated(non_empty_list);
  RecordNewDataPtrCreated(non_empty_ref);

  // Explicitly mark the empty tensor as an output.
  TT_ASSERT_OK_AND_ASSIGN(auto traversals, PrepareMaterializationTraversals(
                                               {empty_list, non_empty_list}));

  // We get one traversal, as there were no required split points.
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
  RecordNewDataPtrCreated(ref_b);

  ExchangeCurrentDeviceIndex(device_index + 1);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = ref_c.device_buffer_list();
  RecordDeferredOpCreated(list_c);
  RecordNewDataPtrCreated(ref_c);

  ExchangeCurrentDeviceIndex(device_index);

  // Ask for a plan to materialize the device for a and b (but not c).
  TT_ASSERT_OK_AND_ASSIGN(auto traversals,
                          PrepareDeviceTraversals(device_index));

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
  RecordNewDataPtrCreated(ref_b);

  const auto non_default_stream_id = NextStreamId(device_index);
  const auto default_stream_id =
      ExchangeCurrentStreamId(device_index, non_default_stream_id);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = ref_c.device_buffer_list();
  RecordDeferredOpCreated(list_c);
  RecordNewDataPtrCreated(ref_c);

  ExchangeCurrentStreamId(device_index, default_stream_id);

  // Ask for a plan to materialize the stream for a and b (but not c).
  TT_ASSERT_OK_AND_ASSIGN(
      auto traversals,
      PrepareStreamTraversals(device_index, default_stream_id));

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
  RecordNewDataPtrCreated(ref_b);

  const auto non_default_stream_id = NextStreamId(device_index);
  const auto default_stream_id =
      ExchangeCurrentStreamId(device_index, non_default_stream_id);

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {},
                                           OpParamCacheKeys::Empty(), {shape}));
  auto ref_c = refs_c[0];
  auto list_c = ref_c.device_buffer_list();
  RecordDeferredOpCreated(list_c);
  RecordNewDataPtrCreated(ref_c);

  ExchangeCurrentStreamId(device_index, default_stream_id);

  // Ask for a plan to materialize the stream for c (but not a or b)
  TT_ASSERT_OK_AND_ASSIGN(
      auto traversals,
      PrepareStreamTraversals(device_index, non_default_stream_id));

  // The execution will contain c, but not a or b, even though they were
  // created first; different streams have no relative order.
  ASSERT_EQ(traversals.size(), 1);
  const Traversal& traversal = *traversals[0];
  EXPECT_THAT(traversal.arguments(), testing::IsEmpty());
  EXPECT_THAT(traversal.execution_order(), testing::ElementsAre(list_c));
  EXPECT_THAT(traversal.outputs(), testing::ElementsAre(ref_c));
}

}  // namespace

}  // namespace torch_tpu
