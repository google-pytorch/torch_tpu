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

#include "csrc/eager/materialize.h"

#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/materialize_common.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/tpu_hooks.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

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

  static void TearDownTestSuite() {
    // Shuts down the runtime in the correct order to mimic the python bindings
    // runtime shutdown sequence (_shutdown_runtime):
    // 1. Terminate materialization worker threads so no new compilation or
    //    execution tasks are processed.
    // 2. Clear the compilation cache.
    // 3. Shut down the PjRtBackend (destroying client & devices).
    ShutDownMaterializationState();
    CompilationCache::ShutDown();
    PjrtBackend::GetInstance().Shutdown();
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
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref,
                          CreateZeroSizeDeviceBufferRef({0}, dtype));
  EXPECT_TRUE(ref.is_deferred());
  EXPECT_EQ(Materialize(ref, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  EXPECT_TRUE(ref.is_materializing());
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
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_a,
      DeviceBufferList::CreateDeferred(OpName::kEmpty, builder,
                                       /*inputs=*/{}, OpParamCacheKeys::Empty(),
                                       {shape}));
  DeviceBufferRef ref_a = refs_a[0];
  RecordDeferredOpCreated(refs_a[0].device_buffer_list());

  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_b = refs_b[0];
  RecordDeferredOpCreated(refs_b[0].device_buffer_list());

  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_c,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {ref_b},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_c = refs_c[0];
  RecordDeferredOpCreated(refs_c[0].device_buffer_list());

  // Create a tensor for c to reflect how this would actually be used
  // (a leaf node with no tensors would ordinarily be dropped immediately).
  at::Tensor c = MakeTensor(ref_c);

  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_d,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {ref_a},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_d = refs_d[0];
  RecordDeferredOpCreated(refs_d[0].device_buffer_list());

  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_e,
      DeviceBufferList::CreateDeferred(OpName::kAdd, builder, {ref_d},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_e = refs_e[0];
  RecordDeferredOpCreated(refs_e[0].device_buffer_list());
  // Create a tensor for e to reflect how this would actually be used
  // (a leaf node with no tensors would ordinarily be dropped immediately).
  at::Tensor e = MakeTensor(ref_e);

  EXPECT_TRUE(ref_a.is_deferred());
  EXPECT_TRUE(ref_b.is_deferred());
  EXPECT_TRUE(ref_c.is_deferred());
  EXPECT_TRUE(ref_d.is_deferred());
  EXPECT_TRUE(ref_e.is_deferred());

  // Materialize d. This should trace the entire graph from the leaf nodes
  // c and e.
  EXPECT_EQ(Materialize(ref_d, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  // a is materialized as it has fanout > 1.
  EXPECT_TRUE(ref_a.is_materializing());

  // b is not materialized; it is only internal to the graph of c, and has
  // neither fanout nor a live Tensor.
  EXPECT_TRUE(ref_b.is_deferred());

  // c is materialized; it was dispatched before d and has a live Tensor, so it
  // must be materialized.
  EXPECT_TRUE(ref_c.is_materializing());

  // d is materialized because it was the explicit target of Materialize().
  EXPECT_TRUE(ref_d.is_materializing());

  // e is not materialized because it was dispatched after the last required
  // node (d).
  EXPECT_TRUE(ref_e.is_deferred());
}

TEST_F(MaterializeTest, CompilerOptionsPropagateToMaterializeThread) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  const Shape shape(Dimensions{1}, mlir::ElementType::F32);
  const auto builder = [shape](mlir::MlirBuilder& builder,
                               absl::Span<mlir::MlirOp>) {
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, shape.dtype(), shape.dimensions())};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(OpName::kEmpty, builder,
                                       /*inputs=*/{}, OpParamCacheKeys::Empty(),
                                       {shape}));
  const DeviceBufferRef ref = refs[0];
  const at::Tensor t = MakeTensor(ref);

  // 1. Clear all cache entries to verify we trigger a fresh compilation.
  CompilationCache::GetInstance().EvictAll();

  // 2. Set compiler overrides on the main thread and calculate its expected
  // cache fingerprint.
  CompilerOptionOverrides overrides;
  overrides["xla_tpu_autofdo"] = "false";
  ASSERT_EQ(PushCompilerOptionOverrides(overrides), absl::OkStatus());
  const CompileOptionsKey expected_key =
      GetCompilationSpec(CompilationMode::kFastCompile).compile_options_key;

  absl::Cleanup cleanup = [] { PopCompilerOptionOverrides(); };

  // 3. Synchronously materialize/compile the buffer.
  ASSERT_EQ(Materialize(ref, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  // 4. Verify the compiled executable fingerprint in cache matches our
  // overridden key.
  const PerfStats stats = CompilationCache::GetInstance().GetCacheStats();
  ASSERT_EQ(stats.per_entry_stats.size(), 1);
  EXPECT_EQ(stats.per_entry_stats[0].key.compile_options_key(), expected_key);
}

TEST(MaterializeCommonTest, GetCompilationMode) {
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

}  // namespace
}  // namespace torch_tpu
