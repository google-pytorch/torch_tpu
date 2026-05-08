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

class OpQueueTest : public testing::Test {
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

TEST_F(OpQueueTest, SingleGlobalQueue) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  auto refs1_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs1_or.ok());
  auto ref1 = refs1_or.value()[0];

  auto refs2_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs2_or.ok());
  auto ref2 = refs2_or.value()[0];

  // They should belong to the same global op queue from the start.
  EXPECT_EQ(ref1.device_buffer_list()->subgraph(),
            ref2.device_buffer_list()->subgraph());
}

TEST_F(OpQueueTest, GetLeafNodesInvalidPopping) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  SharedDeviceBufferList node2;
  std::shared_ptr<OpQueue> subgraph;

  {
    auto refs1_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs1_or.ok());
    SharedDeviceBufferList node1 = refs1_or.value()[0].device_buffer_list();

    auto refs2_or = DeviceBufferList::CreateDeferred(
        OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
    ASSERT_TRUE(refs2_or.ok());
    node2 = refs2_or.value()[0].device_buffer_list();

    subgraph = node1->subgraph();
    // node1 and node2 are automatically in the same global queue.
  }
  // node1 is now out of scope and should be invalid in the queue.

  auto leaves = subgraph->GetLeafNodes();
  // node1 was invalid at the front, so it should have been popped.
  // node2 is valid, so it should be returned as a leaf.
  EXPECT_THAT(leaves, SizeIs(1));
  EXPECT_EQ(leaves[0], node2);
}

TEST_F(OpQueueTest, GetLeafNodesStopPopping) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  SharedDeviceBufferList node1;
  SharedDeviceBufferList node3;
  std::shared_ptr<OpQueue> subgraph;

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

    subgraph = node1->subgraph();
    // node1, node2, node3 are now in the same global queue.
  }
  // node2 is now out of scope and should be invalid in the queue.
  // node1 and node3 are still valid.

  auto leaves = subgraph->GetLeafNodes();
  // node1 and node3 are valid. node2 is invalid but Prune removes ALL invalid
  // nodes, so it should be removed even though it's not at the front.
  // Both node1 and node3 should be returned as leaves if they have no child
  // ops.
  EXPECT_THAT(leaves, SizeIs(2));
  EXPECT_EQ(leaves[0], node1);
  EXPECT_EQ(leaves[1], node3);

  // Now reset node1 to make it invalid too.
  leaves.clear();  // Clear references held by leaves.
  EXPECT_EQ(node1.use_count(), 1);
  node1.reset();
  leaves = subgraph->GetLeafNodes();
  // Now node1 is also removed, leaving only node3.
  EXPECT_THAT(leaves, SizeIs(1));
  EXPECT_EQ(leaves[0], node3);
}

TEST_F(OpQueueTest, DeferredOpOpQueueDereference) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  auto refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  auto ref = refs_or.value()[0];

  EXPECT_NE(ref.device_buffer_list()->subgraph(), nullptr);
}

TEST_F(OpQueueTest, RetainAllNodes) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  auto refs1_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs1_or.ok());
  auto ref1 = refs1_or.value()[0];

  // Create a second node that depends on the first.
  auto refs2_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref1}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs2_or.ok());
  auto ref2 = refs2_or.value()[0];

  auto queue = ref1.device_buffer_list()->subgraph();

  // Both nodes should be in the queue.
  auto all_nodes = queue->GetAllNodes();
  EXPECT_THAT(all_nodes, SizeIs(2));
  EXPECT_EQ(all_nodes[0], ref1.device_buffer_list());
  EXPECT_EQ(all_nodes[1], ref2.device_buffer_list());
}

}  // namespace
}  // namespace torch_tpu
