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

#include "torch_tpu/eager/safe_materialization_rule.h"

#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_set.h"
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
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"

namespace torch_tpu {
namespace {

// A dummy MLIR op builder for testing purposes.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) {
  return DynamicMlirOpResults{};
}

TEST(EnforceOrderedMaterializationTest, StaleNodesDropped) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create two input buffers that don't have tensors.
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];

  // Create a node that depends on both inputs.
  auto refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a, ref_b}, OpParamCacheKeys::Empty(),
      {shape});
  ASSERT_TRUE(refs_c_or.ok());
  auto ref_c = refs_c_or.value()[0];

  // Create a traversal that requires only c to be materialized.
  auto traversal_or = Traversal::Create({ref_c});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal->SortByCreationOrder();

  absl::flat_hash_set<const DeviceBufferList*> required_outputs = {
      ref_c.device_buffer_list().get()};
  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  // a and b are not materialized; the execution plan is a single graph.
  EXPECT_THAT(materialization_nodes, testing::SizeIs(1));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_c.device_buffer_list().get()));
}

TEST(EnforceOrderedMaterializationTest, LiveNodesMaterialized) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create two input buffers and create Tensors for them.
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];
  at::Tensor tensor_a = MakeTensor(ref_a);

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];
  at::Tensor tensor_b = MakeTensor(ref_b);

  // Create a node that depends on both inputs.
  auto refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a, ref_b}, OpParamCacheKeys::Empty(),
      {shape});
  ASSERT_TRUE(refs_c_or.ok());
  auto ref_c = refs_c_or.value()[0];

  auto traversal_or = Traversal::Create({ref_c});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal->SortByCreationOrder();

  absl::flat_hash_set<const DeviceBufferList*> required_outputs = {
      ref_c.device_buffer_list().get()};
  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  // a and b are both materialized along with c; they have live tensors.
  EXPECT_THAT(materialization_nodes, testing::SizeIs(3));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_a.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_b.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_c.device_buffer_list().get()));
}

TEST(EnforceOrderedMaterializationTest, ExternalFanoutMaterialized) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create one input buffer.
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];

  // Create two output buffers that depend on the input.
  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];

  auto refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_c_or.ok());
  auto ref_c = refs_c_or.value()[0];

  // Create a traversal that requires both outputs to be materialized.
  auto traversal_or = Traversal::Create({ref_b, ref_c});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal->SortByCreationOrder();

  absl::flat_hash_set<const DeviceBufferList*> required_outputs = {
      ref_b.device_buffer_list().get(), ref_c.device_buffer_list().get()};
  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  // a is materialized because it has two separate dependencies that are
  // materialized.
  EXPECT_THAT(materialization_nodes, testing::SizeIs(3));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_a.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_b.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_c.device_buffer_list().get()));
}

TEST(EnforceOrderedMaterializationTest, InternalFanoutNotMaterialized) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a diamond-shaped graph like:
  //    b
  //   /  \
  // a     d
  //   \  /
  //    c
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];

  auto refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_c_or.ok());
  auto ref_c = refs_c_or.value()[0];

  auto refs_d_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_b, ref_c}, OpParamCacheKeys::Empty(),
      {shape});
  ASSERT_TRUE(refs_d_or.ok());
  auto ref_d = refs_d_or.value()[0];

  // Create a traversal that requires only d to be materialized.
  auto traversal_or = Traversal::Create({ref_d});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal->SortByCreationOrder();

  absl::flat_hash_set<const DeviceBufferList*> required_outputs = {
      ref_d.device_buffer_list().get()};
  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  // Only d is materialized; while a has fanout, it's fully internal to d's
  // subgraph.
  EXPECT_THAT(materialization_nodes, testing::SizeIs(1));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_d.device_buffer_list().get()));
}

TEST(EnforceOrderedMaterializationTest, DispatchOrderMaintained) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Dispatch a, b, c, d to make a graph like:
  // a -> b ----------> e
  //          c -> d /
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];

  auto refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_c_or.ok());
  auto ref_c = refs_c_or.value()[0];

  auto refs_d_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_c}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_d_or.ok());
  auto ref_d = refs_d_or.value()[0];

  auto refs_e_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_b, ref_d}, OpParamCacheKeys::Empty(),
      {shape});
  ASSERT_TRUE(refs_e_or.ok());
  auto ref_e = refs_e_or.value()[0];

  // Create a traversal that requires both d and e to be materialized.
  auto traversal_or = Traversal::Create({ref_d, ref_e});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal->SortByCreationOrder();

  absl::flat_hash_set<const DeviceBufferList*> required_outputs = {
      ref_d.device_buffer_list().get(), ref_e.device_buffer_list().get()};
  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  // b is materialized so that {a, b} is executed before {c, d}.
  // a and c are not materialized and are fused into {a, b} and {c, d}.
  // d and e are materialized because they are required outputs.
  EXPECT_THAT(materialization_nodes, testing::SizeIs(3));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_b.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_d.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_e.device_buffer_list().get()));
}

TEST(EnforceOrderedMaterializationTest, ForcedSplitHeuristicRespected) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Dispatch a, b, c, d to make a graph like:
  // a -> b -> c -> d
  // where node c is marked with OpSplitMode::kSplitBoth.
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];

  auto refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_b}, OpParamCacheKeys::Empty(), {shape},
      OpSplitMode::kSplitBoth);
  ASSERT_TRUE(refs_c_or.ok());
  auto ref_c = refs_c_or.value()[0];

  auto refs_d_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_c}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_d_or.ok());
  auto ref_d = refs_d_or.value()[0];

  // Create a traversal that requires only d to be materialized.
  auto traversal_or = Traversal::Create({ref_b, ref_d});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal->SortByCreationOrder();

  absl::flat_hash_set<const DeviceBufferList*> required_outputs = {
      ref_b.device_buffer_list().get(), ref_d.device_buffer_list().get()};
  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  // d is materialized because it is a required output.
  // Both b and c are materialized because of the forced split heuristic.
  // a is not materialized, and is fused into a graph with b.
  EXPECT_THAT(materialization_nodes, testing::SizeIs(3));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_b.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_c.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_d.device_buffer_list().get()));
}

TEST(EnforceOrderedMaterializationTest, DynamicOpSplitHeuristicRespected) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Dispatch a, b, c, d to make a graph like:
  // a -> b -> c
  // where node b has a dynamic dimension.
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];

  Shape dynamic_shape({8}, mlir::ElementType::F32,
                      {BoundedDynamicDimension{0, 2, 10}});
  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(),
      {dynamic_shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];

  auto refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_b}, OpParamCacheKeys::Empty(), {shape},
      OpSplitMode::kSplitBoth);
  ASSERT_TRUE(refs_c_or.ok());
  auto ref_c = refs_c_or.value()[0];

  // Create a traversal that requires only c to be materialized.
  auto traversal_or = Traversal::Create({ref_c});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal->SortByCreationOrder();

  absl::flat_hash_set<const DeviceBufferList*> required_outputs = {
      ref_c.device_buffer_list().get()};
  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  // c is materialized because it is a required output.
  // b is materialized because of the dynamic shape heuristic.
  // a is not materialized, and is fused into a graph with b.
  EXPECT_THAT(materialization_nodes, testing::SizeIs(2));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_b.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_c.device_buffer_list().get()));
}

TEST(EnforceOrderedMaterializationTest, NonRequiredNodesNotMaterialized) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Dispatch a, b, c to make a graph like:
  // a -> b -> c
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];

  auto refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_b}, OpParamCacheKeys::Empty(), {shape},
      OpSplitMode::kSplitBoth);
  ASSERT_TRUE(refs_c_or.ok());
  auto ref_c = refs_c_or.value()[0];

  // Create a traversal that traces the entire graph.
  auto traversal_or = Traversal::Create({ref_c});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal->SortByCreationOrder();

  // But only require b to be materialized.
  absl::flat_hash_set<const DeviceBufferList*> required_outputs = {
      ref_b.device_buffer_list().get()};

  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  // c is not materialized because it is after the last required output.
  // b is materialized because it is a required output.
  // a is not materialized, and is fused into a graph with b.
  EXPECT_THAT(materialization_nodes, testing::SizeIs(1));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_b.device_buffer_list().get()));
}

TEST(EnforceOrderedMaterializationTest, EdgesFromNonRequiredNodesConsidered) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Dispatch a, b, c to make a graph like:
  // a -> b -> c
  //  \_______/
  auto refs_a_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_a_or.ok());
  auto ref_a = refs_a_or.value()[0];

  auto refs_b_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_b_or.ok());
  auto ref_b = refs_b_or.value()[0];

  auto refs_c_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a, ref_b}, OpParamCacheKeys::Empty(),
      {shape}, OpSplitMode::kSplitBoth);
  ASSERT_TRUE(refs_c_or.ok());
  auto ref_c = refs_c_or.value()[0];

  // Create a traversal that traces the entire graph.
  auto traversal_or = Traversal::Create({ref_c});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();
  traversal->SortByCreationOrder();

  // But only require b to be materialized.
  absl::flat_hash_set<const DeviceBufferList*> required_outputs = {
      ref_b.device_buffer_list().get()};

  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  // c is not materialized because it is after the last required output.
  // b is materialized because it is a required output.
  // a is materialized because it has a live edge from c, even though c is not
  // materialized.
  EXPECT_THAT(materialization_nodes, testing::SizeIs(2));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_a.device_buffer_list().get()));
  EXPECT_THAT(materialization_nodes,
              testing::Contains(ref_b.device_buffer_list().get()));
}

}  // namespace
}  // namespace torch_tpu
