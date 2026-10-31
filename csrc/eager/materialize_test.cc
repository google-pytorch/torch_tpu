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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
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
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/execution_task.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/tpu_hooks.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"

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

  void SetUp() override { ClearAllStreams(); }
  void TearDown() override { ClearAllStreams(); }
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

TEST_F(MaterializeTest, OutOfOrderUnexecutedDeferred) {
  OpName op_name = OpName::kAdd;
  ScopedPythonContextCapturer capturer(op_name);

  // Create a deferred, non-constant buffer but do not register it with the
  // events queue.
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(
          op_name,
          [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
              -> absl::StatusOr<DynamicMlirOpResults> {
            return DynamicMlirOpResults{};
          },
          {}, OpParamCacheKeys::Empty(),
          {Shape{Dimensions{10}, mlir::ElementType::F32}}));
  auto never_executed_deferred = refs[0];

  // Create an op which depends on the deferred buffer and register it with the
  // events queue.
  TT_ASSERT_OK_AND_ASSIGN(
      refs, DeviceBufferList::CreateDeferred(
                op_name,
                [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
                    -> absl::StatusOr<DynamicMlirOpResults> {
                  auto ranked_tensor_type = mlir::makeTensorType(
                      builder.getContext(), {0}, mlir::ElementType::F32);
                  auto dense_elements_attr =
                      mlir::DenseElementsAttr::getFromRawBuffer(
                          ranked_tensor_type, {});
                  return DynamicMlirOpResults{
                      mlir::stablehlo::Constant(builder, dense_elements_attr)};
                },
                {never_executed_deferred}, OpParamCacheKeys::Empty(),
                {Shape{Dimensions{0}, mlir::ElementType::F32}}));
  auto depends_on_deferred = refs[0];
  RecordDeferredOpCreated(depends_on_deferred.device_buffer_list());

  // Try to materialize the dependent buffer in the background.
  // This will fail on synchronization when the execution thread detects that
  // the deferred buffer wasn't materialized first.
  Materialize(depends_on_deferred, MaterializationReason::kExplicitSync,
              GraphSplitMode::kSplitGraph,
              MaterializationBlockMode::kNonBlocking)
      .IgnoreError();
  auto finished_status = depends_on_deferred.Synchronize();
  EXPECT_EQ(finished_status.code(), error::kFailedPrecondition);
  EXPECT_NE(
      finished_status.message().find("detected an out of order execution"),
      std::string::npos);
}

TEST_F(MaterializeTest, OutOfOrderUnexecutedPending) {
  // Create a buffer in the pending state.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef pending_buffer,
                          DeviceBufferList::CreatePendingExecution(
                              Shape{Dimensions{10}, mlir::ElementType::F32},
                              device_index, stream_id));

  // Create an op which depends on the pending buffer and register it with the
  // events queue.
  OpName op_name = OpName::kAdd;
  ScopedPythonContextCapturer capturer(op_name);
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(
          op_name,
          [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
              -> absl::StatusOr<DynamicMlirOpResults> {
            auto ranked_tensor_type = mlir::makeTensorType(
                builder.getContext(), {0}, mlir::ElementType::F32);
            auto dense_elements_attr =
                mlir::DenseElementsAttr::getFromRawBuffer(ranked_tensor_type,
                                                          {});
            return DynamicMlirOpResults{
                mlir::stablehlo::Constant(builder, dense_elements_attr)};
          },
          {pending_buffer}, OpParamCacheKeys::Empty(),
          {Shape{Dimensions{0}, mlir::ElementType::F32}}));
  auto depends_on_pending = refs[0];
  RecordDeferredOpCreated(depends_on_pending.device_buffer_list());

  // Try to materialize the dependent buffer. This will fail on synchronization
  // when the execution thread detects that the pending buffer wasn't
  // materialized first.
  Materialize(depends_on_pending, MaterializationReason::kExplicitSync,
              GraphSplitMode::kSplitGraph,
              MaterializationBlockMode::kNonBlocking)
      .IgnoreError();
  auto finished_status = depends_on_pending.Synchronize();
  EXPECT_EQ(finished_status.code(), error::kFailedPrecondition);
  EXPECT_NE(
      finished_status.message().find("detected an out of order execution"),
      std::string::npos);
}

// Creates a deferred op that fills an 8-element F32 tensor, handling its
// creation as dictated by the current eager mode.
absl::StatusOr<DeviceBufferRef> CreateFillOp(const bool allow_materialization) {
  internal::DeferredOpParams params{
      .op_name = OpName::kAdd,
      .op_builder = [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp>)
          -> absl::StatusOr<DynamicMlirOpResults> {
        return DynamicMlirOpResults{
            BuildFillUninitialized(builder, mlir::ElementType::F32, {8})};
      },
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
      .output_shapes = {Shape(Dimensions{8}, mlir::ElementType::F32)},
      .allow_materialization = allow_materialization,
  };
  TT_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferRef> refs,
      internal::CreateDeferredDeviceBufferList(std::move(params)));
  TT_RET_CHECK(refs.size() == 1, error::kInternal)
      << "expected 1 output, got " << refs.size();
  return std::move(refs[0]);
}

// Waits until `ref` is no longer deferred, i.e. until its materialization has
// started. Returns false if it is still deferred after `timeout`.
bool WaitUntilNotDeferred(const DeviceBufferRef& ref,
                          const absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  while (ref.is_deferred()) {
    if (absl::Now() > deadline) {
      return false;
    }
    absl::SleepFor(absl::Milliseconds(1));
  }
  return true;
}

TEST_F(MaterializeTest, RunPostponedMaterializationExecutesOpsInDeferNever) {
  ClearAllStreams();
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kDeferNever);
  absl::Cleanup cleanup_mode = [prev_mode] { SetEagerMode(prev_mode); };

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef ref,
                          CreateFillOp(/*allow_materialization=*/false));
  EXPECT_TRUE(ref.is_deferred());

  // Running the postponed materialization executes the op.
  ASSERT_EQ(internal::RunPostponedMaterialization(), absl::OkStatus());
  ASSERT_TRUE(WaitUntilNotDeferred(ref, absl::Minutes(1)));
  EXPECT_EQ(ref.Synchronize(), absl::OkStatus());
}

TEST_F(MaterializeTest,
       RunPostponedMaterializationInsertsRepeatedOpsGraphBreakInDeferAndFuse) {
  ClearAllStreams();
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kDeferAndFuse);
  absl::Cleanup cleanup_mode = [prev_mode] { SetEagerMode(prev_mode); };

  // Create four copies of an op with materialization disallowed: the repeated
  // ops heuristic requests a graph break on the fourth one, which must be
  // postponed.
  std::vector<DeviceBufferRef> refs;
  refs.reserve(4);
  for (int i = 0; i < 4; ++i) {
    TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref,
                            CreateFillOp(/*allow_materialization=*/false));
    refs.push_back(std::move(ref));
  }

  // Sleep for 100ms. This is enough that a background materialization should
  // have started if there was a materialization attempt.
  absl::SleepFor(absl::Milliseconds(100));
  for (const DeviceBufferRef& ref : refs) {
    EXPECT_TRUE(ref.is_deferred());
  }

  // Running the postponed materialization inserts the requested graph break,
  // which executes all the ops.
  ASSERT_EQ(internal::RunPostponedMaterialization(), absl::OkStatus());
  for (const DeviceBufferRef& ref : refs) {
    ASSERT_TRUE(WaitUntilNotDeferred(ref, absl::Minutes(1)));
    EXPECT_EQ(ref.Synchronize(), absl::OkStatus());
  }
}

absl::StatusOr<DeviceBufferRef> CreateIotaOp(const bool record_on_stream = true,
                                             const int64_t size = 8) {
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  TT_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd,
          [size](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp>)
              -> absl::StatusOr<DynamicMlirOpResults> {
            const auto out_type = mlir::makeTensorType(
                builder.getContext(), {size}, mlir::ElementType::F32);
            return DynamicMlirOpResults{
                mlir::stablehlo::Iota(builder, out_type, /*iota_dimension=*/0)};
          },
          /*inputs=*/{}, OpParamCacheKeys::Empty(),
          {Shape(Dimensions{size}, mlir::ElementType::F32)}));
  if (record_on_stream) {
    RecordDeferredOpCreated(refs[0].device_buffer_list());
  }
  return std::move(refs[0]);
}

TEST_F(MaterializeTest, SnapshotAwaitsPriorTensorCoreOutputs) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  std::vector<DeviceBufferRef> refs;
  for (const int64_t size : {8, 64, 65536}) {
    TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref,
                            CreateIotaOp(/*record_on_stream=*/true, size));
    InsertGraphBreak(device_index, stream_id, {ref.device_buffer_list()},
                     MaterializationReason::kExplicitSync,
                     GraphSplitMode::kSplitGraph);
    refs.push_back(std::move(ref));
  }

  TT_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<EventSnapshot> snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph)
          .Await());
  ASSERT_EQ(snapshot->Synchronize(), absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(const bool completed, snapshot->Query());
  EXPECT_TRUE(completed);

  for (const DeviceBufferRef& ref : refs) {
    TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf, ref.AwaitBuffer());
    xla::Future<void> ready = pjrt_buf->GetReadyFuture();
    EXPECT_TRUE(ready.IsKnownReady());
    EXPECT_EQ(ready.Await(), absl::OkStatus());
  }
}

TEST_F(MaterializeTest,
       SnapshotAwaitsLastDeviceOutputAcrossBatchesAndClearAllStreams) {
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef ref1,
                          CreateIotaOp(/*record_on_stream=*/true, 65536));
  ASSERT_EQ(Materialize(ref1, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  TT_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<EventSnapshot> snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph)
          .Await());
  ASSERT_EQ(snapshot->Synchronize(), absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf1, ref1.AwaitBuffer());
  EXPECT_TRUE(pjrt_buf1->GetReadyFuture().IsKnownReady());
  EXPECT_EQ(pjrt_buf1->GetReadyFuture().Await(), absl::OkStatus());

  // Verify ClearAllStreams() drops StreamState's cached output ref.
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef tmp_ref, CreateIotaOp());
  std::optional<DeviceBufferRef> ref2 = std::move(tmp_ref);
  const DeviceBufferRef::Weak weak_ref2 = ref2->weak();
  ASSERT_EQ(Materialize(*ref2, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  ASSERT_EQ(ref2->Synchronize(), absl::OkStatus());
  ref2.reset();
  EXPECT_FALSE(weak_ref2.expired());

  ClearAllStreams();
  // `execute_thread_` may still briefly hold its ExecutionTask copy of `ref2`.
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (!weak_ref2.expired() && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_TRUE(weak_ref2.expired());
  TT_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<EventSnapshot> post_clear_snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph)
          .Await());
  ASSERT_EQ(post_clear_snapshot->Synchronize(), absl::OkStatus());
}

TEST_F(MaterializeTest, DeletedLastDeviceOutputResolvesSnapshotOk) {
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef ref, CreateIotaOp());
  ASSERT_EQ(Materialize(ref, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf, ref.AwaitBuffer());
  ASSERT_EQ(pjrt_buf->GetReadyFuture().Await(), absl::OkStatus());
  pjrt_buf->Delete();
  ASSERT_FALSE(pjrt_buf->GetReadyFuture().Await().ok());

  TT_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<EventSnapshot> snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph)
          .Await());
  EXPECT_EQ(snapshot->Synchronize(), absl::OkStatus());
}

TEST_F(MaterializeTest, CompiledExecutionPreservesPriorLastDeviceOutput) {
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef tmp_tc, CreateIotaOp());
  std::optional<DeviceBufferRef> ref_tc = std::move(tmp_tc);
  const DeviceBufferRef::Weak weak_tc = ref_tc->weak();
  ASSERT_EQ(Materialize(*ref_tc, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  ASSERT_EQ(ref_tc->Synchronize(), absl::OkStatus());
  ref_tc.reset();
  ASSERT_FALSE(weak_tc.expired());

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef unrecorded_op,
                          CreateIotaOp(/*record_on_stream=*/false));
  TT_ASSERT_OK_AND_ASSIGN(
      const auto traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{unrecorded_op}));
  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel kernel,
      traversal->Compile(GetCompilationSpec(CompilationMode::kFastCompile)));
  TT_ASSERT_OK_AND_ASSIGN(SharedLoadedExecutableWithMetadata executable,
                          kernel.fixed_shape_kernel.get());
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> compiled_outputs,
      EnqueueExecutable(std::move(executable), /*arguments=*/{},
                        {Shape(Dimensions{8}, mlir::ElementType::F32)},
                        "test_compiled"));
  ASSERT_EQ(compiled_outputs[0].Synchronize(), absl::OkStatus());
  EXPECT_FALSE(weak_tc.expired());

  TT_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<EventSnapshot> snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph)
          .Await());
  ASSERT_EQ(snapshot->Synchronize(), absl::OkStatus());
  EXPECT_TRUE(weak_tc.expired());
}

TEST_F(MaterializeTest, ExecutionTaskPreservesTraversalCorePinningMode) {
  const auto mlir_context = MakeMlirContext();
  const CompilationSpec spec =
      GetCompilationSpec(CompilationMode::kFastCompile);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef unpinned_op,
                          CreateIotaOp(/*record_on_stream=*/false));
  TT_ASSERT_OK_AND_ASSIGN(
      auto unpinned_traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{unpinned_op}));
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask unpinned_task,
      ExecutionTask::FromTraversal(std::move(unpinned_traversal), *mlir_context,
                                   spec));
  EXPECT_EQ(unpinned_task.core_pinning_mode(), CorePinningMode::kUnpinned);
  ASSERT_EQ(unpinned_task.Run(), absl::OkStatus());
  ASSERT_EQ(unpinned_op.Synchronize(), absl::OkStatus());

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef pinned_op,
                          CreateIotaOp(/*record_on_stream=*/false));
  TT_ASSERT_OK_AND_ASSIGN(
      auto pinned_traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{pinned_op}));
  pinned_traversal->SetCorePinningMode(CorePinningMode::kTensorCore);
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask pinned_task,
      ExecutionTask::FromTraversal(std::move(pinned_traversal), *mlir_context,
                                   spec));
  EXPECT_EQ(pinned_task.core_pinning_mode(), CorePinningMode::kTensorCore);
  ASSERT_EQ(pinned_task.Run(), absl::OkStatus());
  ASSERT_EQ(pinned_op.Synchronize(), absl::OkStatus());
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
