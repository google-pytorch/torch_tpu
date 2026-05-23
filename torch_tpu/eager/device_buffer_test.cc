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

#include "torch_tpu/eager/device_buffer.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

using testing::SizeIs;

class SubgraphTest : public testing::Test {
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

TEST_F(SubgraphTest, SubgraphMerging) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create two independent deferred nodes.
  auto refs1_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs1_or.ok());
  auto ref1 = refs1_or.value()[0];

  auto refs2_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs2_or.ok());
  auto ref2 = refs2_or.value()[0];

  EXPECT_NE(ref1.device_buffer_list()->subgraph()->Find(),
            ref2.device_buffer_list()->subgraph()->Find());

  // Create a third node that merges the first two.
  auto refs3_or =
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {ref1, ref2},
                                       OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs3_or.ok());
  auto ref3 = refs3_or.value()[0];

  // All three should now belong to the same representative subgraph.
  auto rep1 = ref1.device_buffer_list()->subgraph()->Find();
  auto rep2 = ref2.device_buffer_list()->subgraph()->Find();
  auto rep3 = ref3.device_buffer_list()->subgraph()->Find();

  EXPECT_EQ(rep1, rep2);
  EXPECT_EQ(rep1, rep3);
}

TEST_F(SubgraphTest, GetLeafNodesInvalidPopping) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  SharedDeviceBufferList node2;
  std::shared_ptr<Subgraph> subgraph;

  {
    auto refs1_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs1_or.ok());
    SharedDeviceBufferList node1 = refs1_or.value()[0].device_buffer_list();

    auto refs2_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs2_or.ok());
    node2 = refs2_or.value()[0].device_buffer_list();

    subgraph = node1->subgraph()->Find();
    Subgraph::Merge(subgraph, node2->subgraph()->Find());
    subgraph = subgraph->Find();
    // node1 and node2 are now in the same subgraph queue.
  }
  // node1 is now out of scope and should be invalid in the queue.

  auto leaves = subgraph->GetLeafNodes();
  // node1 was invalid at the front, so it should have been popped.
  // node2 is valid, so it should be returned as a leaf.
  EXPECT_THAT(leaves, SizeIs(1));
  EXPECT_EQ(leaves[0], node2);
}

TEST_F(SubgraphTest, GetLeafNodesStopPopping) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  SharedDeviceBufferList node1;
  SharedDeviceBufferList node3;
  std::shared_ptr<Subgraph> subgraph;

  {
    auto refs1_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    node1 = refs1_or.value()[0].device_buffer_list();

    auto refs2_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    SharedDeviceBufferList node2 = refs2_or.value()[0].device_buffer_list();

    auto refs3_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    node3 = refs3_or.value()[0].device_buffer_list();

    subgraph = node1->subgraph()->Find();
    Subgraph::Merge(subgraph, node2->subgraph()->Find());
    Subgraph::Merge(subgraph, node3->subgraph()->Find());
    subgraph = subgraph->Find();
    // node1, node2, node3 are now in the same subgraph queue.
  }
  // node2 is now out of scope and should be invalid in the queue.
  // node1 and node3 are still valid.

  auto leaves = subgraph->GetLeafNodes();
  // node1 is valid at the front, so it should STOP popping there.
  // node2 (invalid) and node3 (valid) should remain in the queue.
  // Both node1 and node3 should be returned as leaves if they have no child
  // ops.
  EXPECT_THAT(leaves, SizeIs(2));
  EXPECT_EQ(leaves[0], node1);
  EXPECT_EQ(leaves[1], node3);

  // We can verify this by checking that node2 is still in the queue if we
  // could, but we can't easily inspect the private queue_. However, we can
  // check if it gets popped on the NEXT call if node1 is gone.
  leaves.clear();  // Clear references held by leaves.
  EXPECT_EQ(node1.use_count(), 1);
  node1.reset();
  leaves = subgraph->GetLeafNodes();
  // Now node2 is at the front, it should be popped. node3 is next and returned.
  EXPECT_THAT(leaves, SizeIs(1));
  EXPECT_EQ(leaves[0], node3);
}

TEST_F(SubgraphTest, DeferredOpSubgraphDereference) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  auto refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref = refs_or.value()[0];

  const DeferredOp* op = ref.deferred_op().get();
  ASSERT_NE(op, nullptr);

  // Dereference to the necessary subgraph object via the
  // subgraph pointer in the deferred op.
  EXPECT_EQ(op->subgraph(), ref.device_buffer_list()->subgraph());
}

TEST_F(SubgraphTest, MergeAll) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create two unrelated deferred nodes.
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];
  const auto* ref_a_deferred_op = ref_a.deferred_op().get();
  ASSERT_NE(ref_a_deferred_op, nullptr);

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];
  const auto* ref_b_deferred_op = ref_b.deferred_op().get();
  ASSERT_NE(ref_b_deferred_op, nullptr);

  // The subgraphs should be distinct; they haven't been merged.
  EXPECT_NE(ref_a_deferred_op->subgraph()->Find(),
            ref_b_deferred_op->subgraph()->Find());

  // Merge all subgraphs.
  auto merged_subgraph = SubgraphRegistry::GetInstance().MergeAll();

  // Both nodes should now be in this merged subgraph.
  EXPECT_EQ(ref_a_deferred_op->subgraph()->Find(), merged_subgraph);
  EXPECT_EQ(ref_b_deferred_op->subgraph()->Find(), merged_subgraph);
}

TEST_F(SubgraphTest, DistributedCollectivesInSameSubgraph) {
  ScopedPythonContextCapturer capturer(OpName::kDistributedAllReduce);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create two unrelated deferred nodes, but use the name for a distributed
  // collective.
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kDistributedAllReduce, DummyBuilder, {},
      OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];
  const auto* ref_a_deferred_op = ref_a.deferred_op().get();
  ASSERT_NE(ref_a_deferred_op, nullptr);

  // Record the representative subgraph for ref_a before creating ref_b.
  const auto* ref_a_subgraph_before =
      ref_a_deferred_op->subgraph()->Find().get();

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kDistributedAllReduce, DummyBuilder, {},
      OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];
  const auto* ref_b_deferred_op = ref_b.deferred_op().get();
  ASSERT_NE(ref_b_deferred_op, nullptr);

  const auto* ref_a_subgraph_after =
      ref_a_deferred_op->subgraph()->Find().get();
  auto* ref_b_subgraph_after = ref_b_deferred_op->subgraph()->Find().get();

  // ref_a should still have the same subgraph.
  EXPECT_EQ(ref_a_subgraph_before, ref_a_subgraph_after);

  // ref_b should have been merged into the same subgraph as ref_a.
  EXPECT_EQ(ref_b_subgraph_after, ref_a_subgraph_after);

  // Both nodes should be leaf nodes in this subgraph, and the older leaf node
  // should be first.
  std::vector<const DeviceBufferList*> actual_leaf_nodes_ptrs;
  for (const auto& leaf_node : ref_b_subgraph_after->GetLeafNodes()) {
    actual_leaf_nodes_ptrs.push_back(leaf_node.get());
  }
  EXPECT_THAT(actual_leaf_nodes_ptrs,
              testing::ElementsAre(ref_a.device_buffer_list().get(),
                                   ref_b.device_buffer_list().get()));
}

TEST_F(SubgraphTest, CollectivesMergeNonCollectives) {
  ScopedPythonContextCapturer capturer(OpName::kDistributedAllReduce);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create 3 disconnected ops in a pattern of [collective, non-collective,
  // collective]
  auto collective_refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kDistributedAllReduce, DummyBuilder, {},
      OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(collective_refs_a_or.ok());
  auto collective_ref_a = collective_refs_a_or.value()[0];
  const auto* collective_ref_a_deferred_op =
      collective_ref_a.deferred_op().get();
  ASSERT_NE(collective_ref_a_deferred_op, nullptr);
  const auto* collective_ref_a_subgraph_before =
      collective_ref_a_deferred_op->subgraph()->Find().get();

  auto non_collective_refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(non_collective_refs_b_or.ok());
  auto non_collective_ref_b = non_collective_refs_b_or.value()[0];
  const auto* non_collective_ref_b_deferred_op =
      non_collective_ref_b.deferred_op().get();
  ASSERT_NE(non_collective_ref_b_deferred_op, nullptr);

  // The collective and non-collective ops should be in different subgraphs
  // initially.
  EXPECT_NE(collective_ref_a_deferred_op->subgraph()->Find(),
            non_collective_ref_b_deferred_op->subgraph()->Find());

  auto collective_refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kDistributedAllReduce, DummyBuilder, {},
      OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(collective_refs_c_or.ok());
  auto collective_ref_c = collective_refs_c_or.value()[0];
  const auto* collective_ref_c_deferred_op =
      collective_ref_c.deferred_op().get();
  ASSERT_NE(collective_ref_c_deferred_op, nullptr);

  // Creating collective ref_c should merge all three ops into the same
  // subgraph, which should be the same as the representative subgraph for
  // ref_a before merging.
  EXPECT_EQ(collective_ref_a_deferred_op->subgraph()->Find().get(),
            collective_ref_a_subgraph_before);
  EXPECT_EQ(non_collective_ref_b_deferred_op->subgraph()->Find().get(),
            collective_ref_a_subgraph_before);
  EXPECT_EQ(collective_ref_c_deferred_op->subgraph()->Find().get(),
            collective_ref_a_subgraph_before);

  // All three nodes should be leaf nodes in this subgraph, and should be
  // returned in the order they were created.
  std::vector<const DeviceBufferList*> actual_leaf_nodes_ptrs;
  for (const auto& leaf_node :
       collective_ref_a_deferred_op->subgraph()->Find()->GetLeafNodes()) {
    actual_leaf_nodes_ptrs.push_back(leaf_node.get());
  }
  EXPECT_THAT(
      actual_leaf_nodes_ptrs,
      testing::ElementsAre(collective_ref_a.device_buffer_list().get(),
                           non_collective_ref_b.device_buffer_list().get(),
                           collective_ref_c.device_buffer_list().get()));
}

TEST_F(SubgraphTest, CollectivesAreAnchoredAndUnprunable) {
  ScopedPythonContextCapturer capturer(OpName::kDistributedAllReduce);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  std::shared_ptr<Subgraph> subgraph;
  DeviceBufferList* raw_collective_ptr = nullptr;

  {
    // Create a collective op inside a nested scope.
    auto collective_refs_or = DeviceBufferList::CreateDeferred(
        OpName::kDistributedAllReduce, DummyBuilder, {},
        OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(collective_refs_or.ok());
    auto collective_ref = collective_refs_or.value()[0];
    raw_collective_ptr = collective_ref.device_buffer_list().get();
    subgraph = collective_ref.device_buffer_list()->subgraph()->Find();
  }
  // collective_ref is now out of scope. Without AnchorSideEffect, the refcount
  // on DeviceBufferList would drop to 0 and it would be pruned by
  // GetLeafNodes().

  auto leaf_nodes = subgraph->GetLeafNodes();
  ASSERT_THAT(leaf_nodes, SizeIs(1));
  EXPECT_EQ(leaf_nodes[0].get(), raw_collective_ptr);
}

TEST_F(SubgraphTest, SideEffectingOpWithChildOpIsUnprunable) {
  ScopedPythonContextCapturer capturer(OpName::kDistributedAllReduce);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  std::shared_ptr<Subgraph> subgraph;
  DeviceBufferList* raw_collective_ptr = nullptr;

  {
    // Create a collective op inside a nested scope.
    // Collective op is side-effecting and should not be pruned even when
    // there are no references to it, or if it has a child op
    auto collective_refs_or = DeviceBufferList::CreateDeferred(
        OpName::kDistributedAllReduce, DummyBuilder, {},
        OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(collective_refs_or.ok());
    auto collective_ref = collective_refs_or.value()[0];
    raw_collective_ptr = collective_ref.device_buffer_list().get();
    subgraph = collective_ref.device_buffer_list()->subgraph()->Find();
    // Add a child op to the collective.
    auto child_refs_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {collective_ref}, OpParamCacheKeys::Empty(),
        {shape});
    ASSERT_TRUE(child_refs_or.ok());
  }

  auto leaf_nodes = subgraph->GetLeafNodes();
  ASSERT_THAT(leaf_nodes, SizeIs(1));
  EXPECT_EQ(leaf_nodes[0].get(), raw_collective_ptr);
}

TEST_F(SubgraphTest, CollectivesInInternalDeferAllArePrunable) {
  ScopedPythonContextCapturer capturer(OpName::kDistributedAllReduce);
  EagerMode old_mode = GetEagerMode();
  SetEagerMode(EagerMode::kInternalDeferAll);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  std::shared_ptr<Subgraph> subgraph;

  {
    // Create a collective op inside a nested scope.
    auto collective_refs_or = DeviceBufferList::CreateDeferred(
        OpName::kDistributedAllReduce, DummyBuilder, {},
        OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(collective_refs_or.ok());
    auto collective_ref = collective_refs_or.value()[0];
    subgraph = collective_ref.device_buffer_list()->subgraph()->Find();
  }
  // collective_ref is now out of scope. In kInternalDeferAll mode (used during
  // torch.compile tracing with fake tensors), side effects are not anchored, so
  // it should be pruned by GetLeafNodes().

  auto leaf_nodes = subgraph->GetLeafNodes();
  EXPECT_THAT(leaf_nodes, SizeIs(0));

  SetEagerMode(old_mode);
}

}  // namespace
}  // namespace torch_tpu
