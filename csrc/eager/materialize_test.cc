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

#include <memory>
#include <string>
#include <utility>
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
#include "csrc/eager/materialize_common.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/tpu_hooks.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {
namespace {

class MaterializeTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string device_type = "tpu";
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

TEST_F(MaterializeTest,
       CompileAnnotatesDynamicArgumentLayoutsWithDefaultTiling) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);
  CompilationCache::GetInstance().EvictAll();

  const Shape shape_3d(Dimensions{15, 128, 128}, mlir::ElementType::UI8);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef ref_a,
                          DeviceBufferList::CreatePlaceholder(
                              shape_3d.dimensions(), shape_3d.dtype()));
  ASSERT_TRUE(ref_a.MarkDynamic(0, 2, 15).ok());
  ASSERT_TRUE(ref_a.MarkDynamic(1, 2, 128).ok());
  ASSERT_TRUE(ref_a.MarkDynamic(2, 2, 128).ok());

  MlirOpBuilder identity_builder = [](mlir::MlirBuilder& /*builder*/,
                                      absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return DynamicMlirOpResults{inputs[0]};
  };

  TT_ASSERT_OK_AND_ASSIGN(const std::vector<DeviceBufferRef> refs_b,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, std::move(identity_builder),
                              {ref_a}, OpParamCacheKeys::Empty(), {shape_3d}));
  const DeviceBufferRef ref_b = refs_b[0];

  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Traversal> tr,
                          Traversal::Create({ref_b}));

  CompilationSpec spec(std::make_unique<xla::CompileOptions>(),
                       CompileOptionsKey(12345));
  std::string mlir_text;
  ASSERT_TRUE(
      tr->Compile(
            std::move(spec), &mlir_text,
            /*use_stablehlo_bounds=*/true,
            /*argument_layouts=*/{CustomLayout{.minor_to_major = {1, 2, 0}}})
          .ok());
  EXPECT_THAT(mlir_text, testing::HasSubstr(
                             "mhlo.layout_mode = \"{1,2,0:T(8,128)(4,1)}\""));
  EXPECT_THAT(mlir_text, testing::HasSubstr("mhlo.layout_mode = \"auto\""));
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

class PinnedHostPropagationTest : public testing::Test {
 protected:
  absl::StatusOr<DeviceBufferRef> CreateRoot() {
    return CreateOp(OpName::kEmpty);
  }

  absl::StatusOr<DeviceBufferRef> CreateView(const DeviceBufferRef& input) {
    return CreateOp(OpName::kView, {input});
  }

  absl::StatusOr<DeviceBufferRef> CreateComputeOp(
      std::vector<DeviceBufferRef> inputs) {
    return CreateOp(OpName::kAdd, std::move(inputs));
  }

 private:
  absl::StatusOr<DeviceBufferRef> CreateOp(
      OpName op_name, std::vector<DeviceBufferRef> inputs = {}) {
    ScopedPythonContextCapturer capturer(op_name);
    TT_ASSIGN_OR_RETURN(
        const std::vector<DeviceBufferRef> refs,
        DeviceBufferList::CreateDeferred(
            op_name,
            [](mlir::MlirBuilder&, absl::Span<mlir::MlirOp>) {
              return DynamicMlirOpResults{};
            },
            std::move(inputs), OpParamCacheKeys::Empty(), {shape_}));
    return refs[0];
  }

  const Shape shape_{Dimensions{10}, mlir::ElementType::F32};
};

// Connectivity graph:
//
//                 root (kEmpty)
//               /      \
//         (view)        (view)
//           /              \
//       view1              view2
//          |               |
//        (view)          (view)
//          |               |
//     view1_child     view2_child
TEST_F(PinnedHostPropagationTest, TreeOfViews) {
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef root, CreateRoot());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view1, CreateView(root));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view1_child, CreateView(view1));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view2, CreateView(root));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view2_child, CreateView(view2));

  EXPECT_FALSE(root.is_pinned_host());
  EXPECT_FALSE(view1.is_pinned_host());
  EXPECT_FALSE(view1_child.is_pinned_host());
  EXPECT_FALSE(view2.is_pinned_host());
  EXPECT_FALSE(view2_child.is_pinned_host());

  // 1. Marking leaf view view1_child as pinned host propagates upstream to
  // ancestors (view1, root) and dynamically to sibling views (view2,
  // view2_child).
  view1_child.set_is_pinned_host();

  EXPECT_TRUE(view1_child.is_pinned_host());
  EXPECT_TRUE(view1.is_pinned_host());
  EXPECT_TRUE(root.is_pinned_host());
  EXPECT_TRUE(view2.is_pinned_host());
  EXPECT_TRUE(view2_child.is_pinned_host());

  // 2. New views created from pinned buffers (root or sibling view) inherit the
  // flag at creation.
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef new_view_from_root,
                          CreateView(root));
  EXPECT_TRUE(new_view_from_root.is_pinned_host());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef new_view_from_view2,
                          CreateView(view2));
  EXPECT_TRUE(new_view_from_view2.is_pinned_host());
}

// Connectivity graph:
//
//                               root (kEmpty)
//                             /   |    \       \
//                       (view)  (view)  |     (kAdd)
//                         /       |     |        \
//                     view1     view2   |   root_compute_op
//                       |         \     |
//                     (kAdd)       \   /
//                       |         (kAdd)
//              view1_compute_op     |
//                         multi_input_compute_op
TEST_F(PinnedHostPropagationTest, ComputeOpIsolation) {
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef root, CreateRoot());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view1, CreateView(root));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view1_compute_op,
                          CreateComputeOp({view1}));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view2, CreateView(root));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef root_compute_op,
                          CreateComputeOp({root}));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef multi_input_compute_op,
                          CreateComputeOp({root, view2}));

  EXPECT_FALSE(root.is_pinned_host());
  EXPECT_FALSE(view1.is_pinned_host());
  EXPECT_FALSE(view1_compute_op.is_pinned_host());
  EXPECT_FALSE(view2.is_pinned_host());
  EXPECT_FALSE(root_compute_op.is_pinned_host());
  EXPECT_FALSE(multi_input_compute_op.is_pinned_host());

  // 1. Marking a non-metadata compute op (view1_compute_op) as pinned host
  // marks itself, but does NOT propagate upstream to its parent view (view1) or
  // root.
  view1_compute_op.set_is_pinned_host();
  EXPECT_TRUE(view1_compute_op.is_pinned_host());
  EXPECT_FALSE(view1.is_pinned_host());
  EXPECT_FALSE(root.is_pinned_host());

  // 2. Marking view view1 as pinned host propagates upstream to root and
  // sibling view view2, but does NOT propagate downstream to non-metadata
  // compute ops (root_compute_op, multi_input_compute_op).
  view1.set_is_pinned_host();
  EXPECT_TRUE(view1.is_pinned_host());
  EXPECT_TRUE(root.is_pinned_host());
  EXPECT_TRUE(view2.is_pinned_host());
  EXPECT_FALSE(root_compute_op.is_pinned_host());
  EXPECT_FALSE(multi_input_compute_op.is_pinned_host());

  // 3. New compute ops created from pinned buffers do not inherit the flag.
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef new_compute_op,
                          CreateComputeOp({root}));
  EXPECT_FALSE(new_compute_op.is_pinned_host());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef new_multi_input_compute_op,
                          CreateComputeOp({root, view2}));
  EXPECT_FALSE(new_multi_input_compute_op.is_pinned_host());
}

}  // namespace
}  // namespace torch_tpu
