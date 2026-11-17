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

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
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
#include "csrc/pjrt/counting_pjrt_buffer_test_util.h"
#include "csrc/pjrt/pjrt_execute.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

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

  absl::StatusOr<DeviceBufferRef> MakeIota2d(const Dimensions& dims) {
    const ScopedPythonContextCapturer capturer(OpName::kAdd);
    TT_ASSIGN_OR_RETURN(
        std::vector<DeviceBufferRef> refs,
        DeviceBufferList::CreateDeferred(
            OpName::kAdd,
            [dims](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp>)
                -> absl::StatusOr<DynamicMlirOpResults> {
              const auto out_type = mlir::makeTensorType(
                  builder.getContext(), dims, mlir::ElementType::F32);
              return DynamicMlirOpResults{mlir::stablehlo::Iota(
                  builder, out_type, /*iota_dimension=*/0)};
            },
            /*inputs=*/{}, OpParamCacheKeys::Empty(),
            {Shape(dims, mlir::ElementType::F32)}));
    return std::move(refs[0]);
  }

  absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileCounting(
      const Traversal& traversal,
      const std::shared_ptr<std::atomic<int>>& on_device_shape_calls,
      const xla::CompileOptions& compile_options = xla::CompileOptions()) {
    TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client,
                        PjrtBackend::GetInstance().GetClient());
    TT_ASSIGN_OR_RETURN(ContextedModule contexted_module,
                        ContextedModule::Make([&](mlir::MLIRContext& ctx) {
                          return traversal.BuildMlirModule(ctx);
                        }));
    TT_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::PjRtLoadedExecutable> raw_executable,
        CompileAndLoadMlirModule(
            *client, std::move(contexted_module).ToMaybeOwningMlirModule(),
            compile_options));
    auto counting_executable = std::make_unique<CountingPjRtLoadedExecutable>(
        std::move(raw_executable), on_device_shape_calls);
    return LoadedExecutableWithMetadata::MakeShared(
        std::move(counting_executable));
  }

  absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileTwoOutputProducer(
      const std::shared_ptr<std::atomic<int>>& on_device_shape_calls) {
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef op0,
                        MakeIota2d(Dimensions{8, 8}));
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef op1,
                        MakeIota2d(Dimensions{16, 8}));
    TT_ASSIGN_OR_RETURN(
        const std::unique_ptr<Traversal> traversal,
        Traversal::Create(std::vector<DeviceBufferRef>{op0, op1}));
    return CompileCounting(*traversal, on_device_shape_calls);
  }

  absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileMatchingConsumer(
      const std::shared_ptr<std::atomic<int>>& on_device_shape_calls) {
    const ScopedPythonContextCapturer capturer(OpName::kAdd);
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef arg0,
                        DeviceBufferList::CreatePlaceholder(
                            Dimensions{8, 8}, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef arg1,
                        DeviceBufferList::CreatePlaceholder(
                            Dimensions{16, 8}, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(
        const std::vector<DeviceBufferRef> consumer_ops,
        DeviceBufferList::CreateDeferred(
            OpName::kAdd,
            [](mlir::MlirBuilder&, absl::Span<mlir::MlirOp> operands)
                -> absl::StatusOr<DynamicMlirOpResults> {
              return DynamicMlirOpResults{
                  mlir::stablehlo::Add(operands[0], operands[0])};
            },
            {arg0, arg1}, OpParamCacheKeys::Empty(), {out_shape0_}));
    TT_ASSIGN_OR_RETURN(
        const std::unique_ptr<Traversal> traversal,
        Traversal::Create(std::vector<DeviceBufferRef>{consumer_ops[0]}));
    return CompileCounting(*traversal, on_device_shape_calls);
  }

  const Shape out_shape0_{Dimensions{8, 8}, mlir::ElementType::F32};
  const Shape out_shape1_{Dimensions{16, 8}, mlir::ElementType::F32};
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
  EXPECT_TRUE(ref.AwaitBuffer().ok());
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
  EXPECT_TRUE(ref.AwaitBuffer().ok());

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
  Materialize(depends_on_deferred, MaterializationReason::kExplicitSync)
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
  Materialize(depends_on_pending, MaterializationReason::kExplicitSync)
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
    ASSERT_TRUE(InsertGraphBreak(device_index, stream_id,
                                 {ref.device_buffer_list()},
                                 MaterializationReason::kExplicitSync,
                                 GraphSplitMode::kSplitGraph)
                    .ok());
    refs.push_back(std::move(ref));
  }

  TT_ASSERT_OK_AND_ASSIGN(
      EventSnapshot snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  ASSERT_EQ(snapshot.Synchronize(), absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(const bool completed, snapshot.Query());
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
      EventSnapshot snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  ASSERT_EQ(snapshot.Synchronize(), absl::OkStatus());
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
      EventSnapshot post_clear_snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  ASSERT_EQ(post_clear_snapshot.Synchronize(), absl::OkStatus());
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
      EventSnapshot snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  EXPECT_EQ(snapshot.Synchronize(), absl::OkStatus());
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
      traversal->Compile(GetCompilationSpec(CompilationMode::kFastCompile),
                         UnboundedKernelPolicy::kDisallow));
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
      EventSnapshot snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  ASSERT_EQ(snapshot.Synchronize(), absl::OkStatus());
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

TEST_F(MaterializeTest, PropagatesStaticTupleAndBoundedDynamicOutputShapes) {
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  const auto on_device_shape_calls = std::make_shared<std::atomic<int>>(0);

  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata producer_executable,
      CompileTwoOutputProducer(on_device_shape_calls));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata consumer_executable,
      CompileMatchingConsumer(on_device_shape_calls));

  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef out0,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape0_, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef out1,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape1_, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef consumed_out,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape0_, device_index, stream_id));

  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask producer_task,
      ExecutionTask::FromExecutable(producer_executable, /*arguments=*/{},
                                    {out0, out1}, "test_producer"));
  ASSERT_EQ(producer_task.Run(), absl::OkStatus());
  ASSERT_EQ(producer_executable->output_device_shapes().size(), 2);
  EXPECT_EQ(out0.known_device_shape(),
            &producer_executable->output_device_shapes()[0]);
  EXPECT_EQ(out1.known_device_shape(),
            &producer_executable->output_device_shapes()[1]);
  ASSERT_NE(out0.known_device_shape(), nullptr);
  ASSERT_NE(out1.known_device_shape(), nullptr);

  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const raw_out0, out0.AwaitBuffer());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const raw_out1, out1.AwaitBuffer());
  EXPECT_EQ(*out0.known_device_shape(),
            UnwrapCountingPjRtBuffer(raw_out0)->on_device_shape());
  EXPECT_EQ(*out1.known_device_shape(),
            UnwrapCountingPjRtBuffer(raw_out1)->on_device_shape());

  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask consumer_task,
      ExecutionTask::FromExecutable(consumer_executable, {out0, out1},
                                    {consumed_out}, "test_consumer"));
  ASSERT_EQ(consumer_task.Run(), absl::OkStatus());
  ASSERT_EQ(consumed_out.Synchronize(), absl::OkStatus());
  EXPECT_EQ(on_device_shape_calls->load(std::memory_order_relaxed), 0);

  // A bounded-dynamic output (produced via runtime `SetDimensionSize`) must
  // preserve its dynamic-dimension flags in both `output_device_shapes()` and
  // `known_device_shape()` rather than turning into a static upper-bound shape.
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> size_ops,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd,
          [](mlir::MlirBuilder& builder,
             absl::Span<mlir::MlirOp>) -> absl::StatusOr<DynamicMlirOpResults> {
            const auto scalar_i32 = mlir::makeTensorType(
                builder.getContext(), {}, mlir::ElementType::I32);
            return DynamicMlirOpResults{mlir::stablehlo::Constant(
                builder, mlir::DenseElementsAttr::get(scalar_i32, int32_t{4}))};
          },
          /*inputs=*/{}, OpParamCacheKeys::Empty(),
          {Shape(Dimensions{}, mlir::ElementType::I32)}));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<Traversal> size_traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{size_ops[0]}));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata size_executable,
      CompileCounting(*size_traversal, on_device_shape_calls));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef size_buf,
                          DeviceBufferList::CreatePendingExecution(
                              Shape(Dimensions{}, mlir::ElementType::I32),
                              device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask size_task,
      ExecutionTask::FromExecutable(size_executable, /*arguments=*/{},
                                    {size_buf}, "test_size"));
  ASSERT_EQ(size_task.Run(), absl::OkStatus());

  TT_ASSERT_OK_AND_ASSIGN(
      const DeviceBufferRef size_placeholder,
      DeviceBufferList::CreatePlaceholder({}, mlir::ElementType::I32));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> bounded_dyn_ops,
      DeviceBufferList::CreateDeferred(
          OpName::kSetDimensionLogicalSize,
          [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> operands)
              -> absl::StatusOr<DynamicMlirOpResults> {
            const auto iota_type = mlir::makeTensorType(
                builder.getContext(), {8, 8}, mlir::ElementType::F32);
            mlir::MlirOp iota = mlir::stablehlo::Iota(builder, iota_type,
                                                      /*iota_dimension=*/0);
            return DynamicMlirOpResults{mlir::stablehlo::SetDimensionSize(
                iota, operands[0], /*dimension=*/0)};
          },
          {size_placeholder}, OpParamCacheKeys::Empty(), {out_shape0_}));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<Traversal> bounded_dyn_traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{bounded_dyn_ops[0]}));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata bounded_dyn_executable,
      CompileCounting(*bounded_dyn_traversal, on_device_shape_calls));
  ASSERT_EQ(bounded_dyn_executable->output_device_shapes().size(), 1);
  EXPECT_TRUE(bounded_dyn_executable->output_device_shapes()[0].is_dynamic());

  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef bounded_dyn_out,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape0_, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask bounded_dyn_task,
      ExecutionTask::FromExecutable(bounded_dyn_executable, {size_buf},
                                    {bounded_dyn_out}, "test_bounded_dyn"));
  ASSERT_EQ(bounded_dyn_task.Run(), absl::OkStatus());
  ASSERT_NE(bounded_dyn_out.known_device_shape(), nullptr);
  EXPECT_TRUE(bounded_dyn_out.known_device_shape()->is_dynamic());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const raw_dyn_out,
                          bounded_dyn_out.AwaitBuffer());
  EXPECT_EQ(*bounded_dyn_out.known_device_shape(),
            UnwrapCountingPjRtBuffer(raw_dyn_out)->on_device_shape());
  EXPECT_EQ(on_device_shape_calls->load(std::memory_order_relaxed), 0);
}

TEST_F(MaterializeTest,
       AssignsSlicesAcrossMultiBufferNodesAndHandlesEmptyOutputShapes) {
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  const auto on_device_shape_calls = std::make_shared<std::atomic<int>>(0);

  const Shape out_shape2(Dimensions{4, 8}, mlir::ElementType::F32);
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef op0,
                          MakeIota2d(Dimensions{8, 8}));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef op1,
                          MakeIota2d(Dimensions{16, 8}));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef op2,
                          MakeIota2d(Dimensions{4, 8}));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<Traversal> producer3_traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{op0, op1, op2}));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata producer3_executable,
      CompileCounting(*producer3_traversal, on_device_shape_calls));
  ASSERT_EQ(producer3_executable->output_device_shapes().size(), 3);

  // Node 0 has 1 buffer (`i = 0, node_size = 1`); node 1 has 2 buffers
  // (`i = 1, node_size = 2`), exercising `device_shapes_offset > 0`.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef node0_ref,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape0_, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> node1_refs,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd,
          [](mlir::MlirBuilder&,
             absl::Span<mlir::MlirOp>) -> absl::StatusOr<DynamicMlirOpResults> {
            return DynamicMlirOpResults{};
          },
          /*inputs=*/{}, OpParamCacheKeys::Empty(), {out_shape1_, out_shape2}));
  ASSERT_EQ(node1_refs.size(), 2);
  ASSERT_EQ(node1_refs[0].device_buffer_list()->SetAsPendingMaterialization(),
            absl::OkStatus());

  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask multi_node_task,
      ExecutionTask::FromExecutable(producer3_executable, /*arguments=*/{},
                                    {node0_ref, node1_refs[0], node1_refs[1]},
                                    "test_multi_buffer_nodes"));
  ASSERT_EQ(multi_node_task.Run(), absl::OkStatus());
  EXPECT_EQ(node0_ref.known_device_shape(),
            &producer3_executable->output_device_shapes()[0]);
  EXPECT_EQ(node0_ref.device_buffer_list()->known_device_shape(-1), nullptr);
  EXPECT_EQ(node0_ref.device_buffer_list()->known_device_shape(1), nullptr);
  EXPECT_EQ(node1_refs[0].known_device_shape(),
            &producer3_executable->output_device_shapes()[1]);
  EXPECT_EQ(node1_refs[1].known_device_shape(),
            &producer3_executable->output_device_shapes()[2]);
  EXPECT_EQ(node1_refs[0].device_buffer_list()->known_device_shape(2), nullptr);
  for (const DeviceBufferRef& ref : {node0_ref, node1_refs[0], node1_refs[1]}) {
    ASSERT_NE(ref.known_device_shape(), nullptr);
    TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const raw_buf, ref.AwaitBuffer());
    EXPECT_EQ(*ref.known_device_shape(),
              UnwrapCountingPjRtBuffer(raw_buf)->on_device_shape());
  }
  EXPECT_EQ(on_device_shape_calls->load(std::memory_order_relaxed), 0);

  // When an executable's `output_device_shapes()` is empty (e.g. gated on an
  // `output_transform`), `SetOutputNodesAsMaterialized` falls back to `{}`
  // rather than indexing out of bounds.
  const SharedLoadedExecutableWithMetadata transformed_executable =
      LoadedExecutableWithMetadata::WithDynamicExecutionMetadata(
          producer3_executable,
          UnboundedDynamicExecutionMetadata{
              .output_transform = [](PjRtBufferPointers outputs,
                                     absl::Span<const Shape>)
                  -> absl::StatusOr<PjRtBufferPointers> { return outputs; }});
  ASSERT_TRUE(transformed_executable->output_device_shapes().empty());

  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef fallback_out0,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape0_, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef fallback_out1,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape1_, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef fallback_out2,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape2, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(ExecutionTask fallback_task,
                          ExecutionTask::FromExecutable(
                              transformed_executable, /*arguments=*/{},
                              {fallback_out0, fallback_out1, fallback_out2},
                              "test_empty_output_shapes"));
  ASSERT_EQ(fallback_task.Run(), absl::OkStatus());
  EXPECT_EQ(fallback_out0.known_device_shape(), nullptr);
  EXPECT_EQ(fallback_out1.known_device_shape(), nullptr);
  EXPECT_EQ(fallback_out2.known_device_shape(), nullptr);
}

TEST_F(MaterializeTest,
       PendingPlaceholderAndErroredBuffersReportNullKnownDeviceShape) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef pending,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape0_, device_index, stream_id));
  EXPECT_EQ(pending.known_device_shape(), nullptr);
  EXPECT_EQ(pending.device_buffer_list()->known_device_shape(-1), nullptr);
  EXPECT_EQ(pending.device_buffer_list()->known_device_shape(1), nullptr);

  pending.device_buffer_list()->SetAsError(TT_ERROR(error::kInternal)
                                           << "boom");
  EXPECT_EQ(pending.known_device_shape(), nullptr);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef placeholder,
                          DeviceBufferList::CreatePlaceholder(
                              Dimensions{8, 8}, mlir::ElementType::F32));
  EXPECT_EQ(placeholder.known_device_shape(), nullptr);
}

TEST_F(MaterializeTest,
       RejectsMismatchedArgumentLayoutWithoutQueryingPjRtBuffer) {
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  const auto on_device_shape_calls = std::make_shared<std::atomic<int>>(0);

  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata producer_executable,
      CompileTwoOutputProducer(on_device_shape_calls));

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef transposed_placeholder,
                          DeviceBufferList::CreatePlaceholder(
                              Dimensions{8, 8}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> mismatched_consumer_ops,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd,
          [](mlir::MlirBuilder&, absl::Span<mlir::MlirOp> operands)
              -> absl::StatusOr<DynamicMlirOpResults> {
            return DynamicMlirOpResults{
                mlir::stablehlo::Add(operands[0], operands[0])};
          },
          {transposed_placeholder}, OpParamCacheKeys::Empty(), {out_shape0_}));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<Traversal> mismatched_consumer_traversal,
      Traversal::Create(
          std::vector<DeviceBufferRef>{mismatched_consumer_ops[0]}));
  xla::CompileOptions transposed_compile_options;
  transposed_compile_options.argument_layouts = {
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::F32, /*dimensions=*/{8, 8},
                                               /*minor_to_major=*/{0, 1})};
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata mismatched_consumer_executable,
      CompileCounting(*mismatched_consumer_traversal, on_device_shape_calls,
                      transposed_compile_options));

  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef out0,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape0_, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef out1,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape1_, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask producer_task,
      ExecutionTask::FromExecutable(producer_executable, /*arguments=*/{},
                                    {out0, out1}, "test_producer"));
  ASSERT_EQ(producer_task.Run(), absl::OkStatus());

  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef mismatched_out,
                          DeviceBufferList::CreatePendingExecution(
                              out_shape0_, device_index, stream_id));
  TT_ASSERT_OK_AND_ASSIGN(ExecutionTask mismatched_task,
                          ExecutionTask::FromExecutable(
                              mismatched_consumer_executable, {out0},
                              {mismatched_out}, "test_mismatched_consumer"));
  EXPECT_THAT(
      mismatched_task.Run(),
      absl_testing::StatusIs(
          error::kInvalidArgument,
          testing::HasSubstr("expected argument 0 to have on-device layout")));
  EXPECT_EQ(mismatched_out.known_device_shape(), nullptr);
  EXPECT_EQ(on_device_shape_calls->load(std::memory_order_relaxed), 0);
}

TEST_F(MaterializeTest,
       MultiStepBoundedDynamicTraversalPropagatesFinalOutputShapes) {
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  const auto mlir_context = MakeMlirContext();
  const CompilationSpec spec =
      GetCompilationSpec(CompilationMode::kFastCompile);

  // Materialize a bounded-dynamic input (`f32[4, 8]` with upper bound 8 on dim
  // 0) and a static input (`f32[8, 8]`). Bounded-dynamic compilation runs
  // `preamble` -> `fixed_shape_kernel` -> `postamble`, exercising multi-step
  // sequential execution and verifying that `SetOutputNodesAsMaterialized`
  // records `cached_executables.back()`'s (`postamble`'s) static `[4, 8]`
  // output shape rather than an earlier step's shape.
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef dyn_input,
                          MakeIota2d(Dimensions{4, 8}));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef static_input,
                          MakeIota2d(Dimensions{8, 8}));
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Traversal> inputs_traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{dyn_input, static_input}));
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask inputs_task,
      ExecutionTask::FromTraversal(std::move(inputs_traversal), *mlir_context,
                                   spec));
  ASSERT_EQ(inputs_task.Run(), absl::OkStatus());
  ASSERT_NE(static_input.known_device_shape(), nullptr);
  ASSERT_EQ(dyn_input.MarkDynamic(/*dimension=*/0, /*lower_bound=*/2,
                                  /*upper_bound=*/8),
            absl::OkStatus());

  const Shape dyn_shape(Dimensions{4, 8}, mlir::ElementType::F32);
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> step_outputs,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd,
          [](mlir::MlirBuilder&, absl::Span<mlir::MlirOp> operands)
              -> absl::StatusOr<DynamicMlirOpResults> {
            return DynamicMlirOpResults{
                mlir::stablehlo::Add(operands[0], operands[0]),
                mlir::stablehlo::Add(operands[1], operands[1]),
            };
          },
          {dyn_input, static_input}, OpParamCacheKeys::Empty(),
          {dyn_shape, out_shape0_}));
  ASSERT_EQ(step_outputs.size(), 2);

  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Traversal> multi_step_traversal,
                          Traversal::Create(std::vector<DeviceBufferRef>{
                              step_outputs[0], step_outputs[1]}));
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask multi_step_task,
      ExecutionTask::FromTraversal(std::move(multi_step_traversal),
                                   *mlir_context, spec));
  ASSERT_EQ(multi_step_task.Run(), absl::OkStatus());
  ASSERT_EQ(step_outputs[0].Synchronize(), absl::OkStatus());
  ASSERT_EQ(step_outputs[1].Synchronize(), absl::OkStatus());

  ASSERT_NE(step_outputs[0].known_device_shape(), nullptr);
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const raw_sliced_out,
                          step_outputs[0].AwaitBuffer());
  EXPECT_THAT(step_outputs[0].known_device_shape()->dimensions(),
              testing::ElementsAre(4, 8));
  EXPECT_EQ(*step_outputs[0].known_device_shape(),
            raw_sliced_out->on_device_shape());

  ASSERT_NE(step_outputs[1].known_device_shape(), nullptr);
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const raw_static_out,
                          step_outputs[1].AwaitBuffer());
  EXPECT_EQ(*step_outputs[1].known_device_shape(),
            raw_static_out->on_device_shape());
}

TEST_F(MaterializeTest,
       EnqueueExecutablePropagatesKnownDeviceShapesAsynchronously) {
  const auto on_device_shape_calls = std::make_shared<std::atomic<int>>(0);

  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata producer_executable,
      CompileTwoOutputProducer(on_device_shape_calls));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata consumer_executable,
      CompileMatchingConsumer(on_device_shape_calls));

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> enqueued_prod_outputs,
      EnqueueExecutable(producer_executable, /*arguments=*/{},
                        "test_enqueued_producer"));
  ASSERT_EQ(enqueued_prod_outputs.size(), 2);
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> enqueued_cons_outputs,
      EnqueueExecutable(consumer_executable, enqueued_prod_outputs,
                        {out_shape0_}, "test_enqueued_consumer"));
  ASSERT_EQ(enqueued_cons_outputs.size(), 1);
  ASSERT_EQ(enqueued_cons_outputs[0].Synchronize(), absl::OkStatus());

  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(enqueued_prod_outputs[i].known_device_shape(),
              &producer_executable->output_device_shapes()[i]);
    ASSERT_NE(enqueued_prod_outputs[i].known_device_shape(), nullptr);
    TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const raw_buf,
                            enqueued_prod_outputs[i].AwaitBuffer());
    EXPECT_EQ(*enqueued_prod_outputs[i].known_device_shape(),
              UnwrapCountingPjRtBuffer(raw_buf)->on_device_shape());
  }
  EXPECT_EQ(on_device_shape_calls->load(std::memory_order_relaxed), 0);
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
