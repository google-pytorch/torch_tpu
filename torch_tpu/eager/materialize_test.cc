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

#include "torch_tpu/eager/materialize.h"

#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize_common.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/tpu_hooks.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

class MaterializeTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string device_type = "xla_cpu";
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = device_type});
    ASSERT_EQ(AddTpuHooks(), absl::OkStatus());
    RegisterTpuAllocator();
    CompilationCache::GetInstance().SetOptions({});
  }
};

TEST_F(MaterializeTest, EmptyListNoOpSuccess) {
  EXPECT_EQ(Materialize(absl::Span<const SharedDeviceBufferList>(),
                        MaterializationReason::kExplicitSync),
            absl::OkStatus());
  EXPECT_EQ(Materialize(absl::Span<const DeviceBufferRef>(),
                        MaterializationReason::kExplicitSync),
            absl::OkStatus());
}

TEST_F(MaterializeTest, MaterializedZeroSizeBufferSuccess) {
  const mlir::ElementType dtype = mlir::ElementType::F32;
  TF_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref,
                          DeviceBufferList::CreateZeroSize({0}, dtype));
  EXPECT_EQ(ref.state(), DeviceBufferRefState::kDeferred);
  EXPECT_EQ(Materialize(ref, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  EXPECT_EQ(ref.state(), DeviceBufferRefState::kMaterialized);
}

TEST_F(MaterializeTest, AddLeafNodes) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  // Create a graph of:
  // ```
  //               / -> leaf_a
  // arg -> target
  //               \ -> leaf_b
  // ```
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef arg,
      DeviceBufferList::CreateZeroSize({0}, mlir::ElementType::F32));

  const Shape shape(Dimensions{8}, mlir::ElementType::F32);

  auto builder = [shape](mlir::MlirBuilder& builder,
                         absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    if (!inputs.empty()) return DynamicMlirOpResults{inputs[0]};
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, shape.dtype(), shape.dimensions())};
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> target_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {arg},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef target_ref = target_refs[0];

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> leaf_a_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {target_ref},
                                       OpParamCacheKeys::Empty(), {shape}));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> leaf_b_refs,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {target_ref},
                                       OpParamCacheKeys::Empty(), {shape}));

  // Call AddLeafNodes on the target list.
  std::vector<SharedDeviceBufferList> actual_nodes = {
      target_ref.device_buffer_list()};
  AddLeafNodes(actual_nodes);

  // The modified list should still have the target node, plus the two leaf
  // node.
  EXPECT_EQ(actual_nodes.size(), 3);
  EXPECT_THAT(actual_nodes, testing::UnorderedElementsAre(
                                target_ref.device_buffer_list(),
                                leaf_a_refs[0].device_buffer_list(),
                                leaf_b_refs[0].device_buffer_list()));
}

TEST_F(MaterializeTest, LeafNodeMaterializationPatternSuccess) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  const Shape shape(Dimensions{8}, mlir::ElementType::F32);

  auto builder = [shape](mlir::MlirBuilder& builder,
                         absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    if (!inputs.empty()) return DynamicMlirOpResults{inputs[0]};
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, shape.dtype(), shape.dimensions())};
  };

  // Create a graph (letter indicates creation order):
  //    / -> b -> c (has Tensor)
  //  a ---> d -> e (has Tensor)
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_a,
      DeviceBufferList::CreateDeferred(OpName::kEmpty, builder,
                                       /*inputs=*/{}, OpParamCacheKeys::Empty(),
                                       {shape}));
  DeviceBufferRef ref_a = refs_a[0];

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_b = refs_b[0];

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_c,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {ref_b},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_c = refs_c[0];

  // Create a tensor for c to reflect how this would actually be used
  // (a leaf node with no tensors would ordinarily be dropped immediately).
  at::Tensor c = MakeTensor(ref_c);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_d,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_d = refs_d[0];

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_e,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {ref_d},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_e = refs_e[0];
  // Create a tensor for e to reflect how this would actually be used
  // (a leaf node with no tensors would ordinarily be dropped immediately).
  at::Tensor e = MakeTensor(ref_e);

  EXPECT_EQ(ref_a.state(), DeviceBufferRefState::kDeferred);
  EXPECT_EQ(ref_b.state(), DeviceBufferRefState::kDeferred);
  EXPECT_EQ(ref_c.state(), DeviceBufferRefState::kDeferred);
  EXPECT_EQ(ref_d.state(), DeviceBufferRefState::kDeferred);
  EXPECT_EQ(ref_e.state(), DeviceBufferRefState::kDeferred);

  // Materialize d. This should trace the entire graph from the leaf nodes
  // c and e.
  EXPECT_EQ(Materialize(ref_d, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  // a is materialized as it has fanout > 1.
  EXPECT_EQ(ref_a.state(), DeviceBufferRefState::kMaterialized);

  // b is not materialized; it is only internal to the graph of c, and has
  // neither fanout nor a live Tensor.
  EXPECT_EQ(ref_b.state(), DeviceBufferRefState::kDeferred);

  // c is materialized by SafeMaterializationRule; it was dispatched before d
  // and has a live Tensor, so it must be materialized.
  EXPECT_EQ(ref_c.state(), DeviceBufferRefState::kMaterialized);

  // d is materialized because it was the explicit target of Materialize().
  EXPECT_EQ(ref_d.state(), DeviceBufferRefState::kMaterialized);

  // e is not materialized because it was dispatched after the last required
  // node (d).
  EXPECT_EQ(ref_e.state(), DeviceBufferRefState::kDeferred);
}

TEST(MaterializeCommonTest, GetCompilationMode) {
  EXPECT_EQ(GetCompilationMode(EagerMode::kInternalDeferAll),
            CompilationMode::kFastRuntime);
  EXPECT_EQ(GetCompilationMode(EagerMode::kDeferAndFuse),
            CompilationMode::kFastRuntime);
  EXPECT_EQ(GetCompilationMode(EagerMode::kDeferNever),
            CompilationMode::kFastCompile);
  EXPECT_EQ(GetCompilationMode(EagerMode::kDeferNeverAndLaunchBlocking),
            CompilationMode::kFastCompile);
}

}  // namespace
}  // namespace torch_tpu
