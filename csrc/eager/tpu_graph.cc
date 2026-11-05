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

#include "csrc/eager/tpu_graph.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "csrc/common/macro_utils.h"
#include "mlir/IR/MLIRContext.h"
#if TT_TORCH_VERSION_GE(2, 12)
#include "ATen/core/GraphImplInterface.h"
#endif
#include "c10/core/Allocator.h"
#include "c10/core/AutogradState.h"
#include "c10/core/Device.h"
#include "c10/core/DeviceGuard.h"
#include "c10/core/Storage.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/utils.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/execution_task.h"
#include "csrc/eager/graph_capture_state.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/stride/stride_helper.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {

#if TT_TORCH_VERSION_GE(2, 12)  // See tpu_graph.h.

struct GraphCaptureSession {
  // The stream that the capture records operations from.
  DeviceStreamId stream;
  // Tracks the tensor storages that the capture reads and writes.
  absl_nonnull std::shared_ptr<GraphCaptureStorageState> storage;
  // Whether autograd multithreading was enabled on the owner thread before
  // the capture disabled it.
  bool autograd_multithreading_was_enabled = true;
};

namespace {

// The graph capture that the current thread owns, if any. Shared with the
// graph that began the capture. Unlike the destructors of static variables,
// the destructors of thread_local variables run when their thread exits
// (go/cpp-style#thread_local, b/172012313).
// NOLINTBEGIN(google3-runtime-global-variables)
thread_local  // CPP_THREAD_LOCAL_OK=ownership stays on the capturing thread
    std::shared_ptr<GraphCaptureSession>
        current_thread_capture_session;
// NOLINTEND(google3-runtime-global-variables)

// Ends the graph capture of `session` on its stream, and makes the current
// thread leave capture mode if it owns the capture. The tensor storage tracking
// of the capture must have ended.
void EndGraphCapture(const GraphCaptureSession& session) {
  EndStreamGraphCapture(session.stream.device_index, session.stream.stream_id);
  if (current_thread_capture_session.get() == &session) {
    ClearCurrentThreadGraphCaptureStream();
    c10::AutogradState::get_tls_state().set_multithreading_enabled(
        session.autograd_multithreading_was_enabled);
    current_thread_capture_session.reset();
  }
}

// Ends the graph capture of `session` without keeping the captured operations,
// and restores the tensor storages that existed before capture.
void AbortGraphCapture(std::shared_ptr<GraphCaptureSession> session) {
  const c10::DeviceGuard device_guard(
      c10::Device(c10::DeviceType::PrivateUse1, session->stream.device_index));
  const CapturedGraphStorages captured =
      EndGraphCaptureStorageTracking(*session->storage);
  // Drops the captured operations.
  ExtractCapturedTraversals(session->stream.device_index,
                            session->stream.stream_id, {});
  // Ends the capture before clearing the storages, so that capture does not
  // record the operations that clear them.
  EndGraphCapture(*session);
  // The callers either cannot raise or raise a more relevant error.
  ClearCapturedDeferredStorages(captured, /*aborted=*/true)
      .IgnoreError();  // IGNORE_ERROR_OK=see above
}

// Returns the DeviceBufferLists of the deferred buffers that capture wrote to
// tensor storages, in creation order.
std::vector<SharedDeviceBufferList> CollectMustMaterialize(
    const CapturedGraphStorages& captured) {
  std::vector<SharedDeviceBufferList> must_materialize;
  absl::flat_hash_set<const DeviceBufferList*> seen_lists;
  for (const CapturedStorageWrite& write : captured.writes) {
    const DeviceBufferRef& buffer = write.final_buffer;
    if (buffer.is_deferred() &&
        seen_lists.insert(buffer.device_buffer_list().get()).second) {
      must_materialize.push_back(buffer.device_buffer_list());
    }
  }
  std::sort(
      must_materialize.begin(), must_materialize.end(),
      [](const SharedDeviceBufferList& a, const SharedDeviceBufferList& b) {
        return a->creation_index() < b->creation_index();
      });
  return must_materialize;
}

// Returns the current buffer of `storage`, an input of the graph, interpreted
// with the shape and element type of `placeholder`, which stood for the
// contents of `storage` during capture.
absl::StatusOr<DeviceBufferRef> ReadStorageInput(
    const c10::Storage& storage, const DeviceBufferRef& placeholder) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer, GetBaseBuffer(storage));
  const size_t buffer_bytes = PhysicalByteSize(buffer);
  const size_t placeholder_bytes = PhysicalByteSize(placeholder);
  TT_RET_CHECK(buffer_bytes == placeholder_bytes, error::kFailedPrecondition)
      << "cannot replay a graph because an input tensor storage was resized "
      << "from " << placeholder_bytes << " to " << buffer_bytes
      << " bytes after capture";
  if (buffer.dimensions() == placeholder.dimensions() &&
      buffer.element_type() == placeholder.element_type()) {
    return buffer;
  }
  // Writing through a view of a different shape or element type can replace
  // the base buffer of a storage with one of that shape or element type.
  const absl::Span<const int64_t> sizes = placeholder.dimensions();
  return CreateViewDeviceBufferRef(
      std::move(buffer),
      TpuLayout{.sizes = CopyIntVector(sizes),
                .strides = CalculateStridesContiguous(sizes),
                .storage_offset = 0,
                .element_type = placeholder.element_type()},
      /*view_is_conj=*/false, DonateBase::kDoNotDonate);
}

}  // namespace

TpuGraphImpl::TpuGraphImpl(const at::GraphImplArgs& args)
    : keep_graph_(args.keep_graph) {}

TpuGraphImpl::~TpuGraphImpl() {
  std::shared_ptr<GraphCaptureSession> capture;
  {
    absl::MutexLock lock(mu_);
    capture = std::move(capture_);
  }
  if (capture == nullptr) {
    return;
  }
  if (current_thread_capture_session == capture) {
    AbortGraphCapture(std::move(capture));
    return;
  }
  // A destructor cannot raise, and the state of the capture belongs to its
  // owner thread, which keeps it alive.
  ABSL_LOG(WARNING) << "destroying a graph that another thread is capturing; "
                       "that thread remains in capture mode";
}

void TpuGraphImpl::capture_begin(at::MempoolId_t pool,
                                 at::GraphCaptureMode capture_mode) {
  absl::MutexLock lock(mu_);
  // Whatever the capture mode, capture records the operations that the
  // capturing thread issues on the capture stream (see tpu_graph.h).
  switch (capture_mode) {
    case at::GraphCaptureMode::Default:
    case at::GraphCaptureMode::Global:
    case at::GraphCaptureMode::ThreadLocal:
    case at::GraphCaptureMode::Relaxed:
      break;
    default:
      TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Python validates enum values.
          false, error::kInvalidArgument)
          << "invalid GraphCaptureMode value: "
          << static_cast<int>(capture_mode);
  }
  TT_CHECK_THROW(!(pool.first != 0 && pool.second != 0),
                 error::kInvalidArgument)
      << "expected at most one of pool.first and pool.second to be nonzero, "
         "got ("
      << pool.first << ", " << pool.second << ")";
  TT_CHECK_THROW(GetEagerMode() != EagerMode::kInternalCompileFxGraph,
                 error::kFailedPrecondition)
      << "cannot capture a graph while tracing an FX graph";
  TT_CHECK_THROW(capture_ == nullptr, error::kFailedPrecondition)
      << "graph capture is already in progress";
  // Like CUDAGraph, once capture_begin() assigns a pool, reset() is required
  // before capturing again, even if capture_end() failed.
  TT_CHECK_THROW(
      !capture_ended_ && mempool_id_.first == 0 && mempool_id_.second == 0,
      error::kFailedPrecondition)
      << "this Graph instance already owns a captured graph; call reset() or "
         "create a new Graph instance before capturing again";
  if (current_thread_capture_session != nullptr &&
      !IsCurrentThreadCapturingGraph()) {
    // Reconcile TLS if ClearAllStreams() cleared an earlier capture's stream.
    AbortGraphCapture(current_thread_capture_session);
  }
  TT_CHECK_THROW(!IsCurrentThreadCapturingGraph(), error::kFailedPrecondition)
      << "nested graph capture on the same thread is not supported";

  const DeviceStreamId stream = GetCurrentDeviceStreamId();
  // Stream 0 is the default stream of every device.
  TT_CHECK_THROW(stream.stream_id != 0, error::kFailedPrecondition)
      << "graphs must be captured on a non-default stream (after capture, "
         "they can be replayed on the default stream)";

  // Reserves the stream before flushing it, so that other threads cannot use
  // it until capture ends.
  TT_CHECK_THROW(
      TryReserveStreamGraphCapture(stream.device_index, stream.stream_id),
      error::kFailedPrecondition)
      << "cannot capture a graph on a stream that another thread is capturing "
         "a graph on";
  absl::Cleanup end_reservation = [stream] {
    EndStreamGraphCapture(stream.device_index, stream.stream_id);
  };
  // Flush and materialize any pending work on the capture stream before
  // starting graph capture so pre-existing deferred ops do not leak into the
  // captured graph.
  TT_ASSIGN_OR_THROW(EventSnapshot snapshot,
                     MaterializeStream(stream.device_index, stream.stream_id,
                                       MaterializationReason::kExplicitSync,
                                       GraphSplitMode::kSplitGraph));
  TT_THROW_IF_ERROR(snapshot.Synchronize());
  ActivateStreamGraphCapture(stream.device_index, stream.stream_id);
  std::move(end_reservation).Cancel();
  // Nothing below can fail.
  capture_device_index_ = stream.device_index;
  capture_stream_id_ = stream.stream_id;
  SetCurrentThreadGraphCaptureStream(stream);
  // The autograd engine runs the backward ops of TPU tensors on a per-device
  // thread unless multithreading is disabled, while only the capturing thread
  // records operations. Backward passes must run on the capturing thread for
  // capture to record the gradients they compute.
  c10::AutogradState& autograd_state = c10::AutogradState::get_tls_state();
  capture_ = std::make_shared<GraphCaptureSession>(GraphCaptureSession{
      .stream = stream,
      .storage = BeginGraphCaptureStorageTracking(stream),
      .autograd_multithreading_was_enabled =
          autograd_state.get_multithreading_enabled(),
  });
  autograd_state.set_multithreading_enabled(false);
  current_thread_capture_session = capture_;
  if (pool.first != 0 || pool.second != 0) {
    mempool_id_ = pool;
  } else {
    mempool_id_ = AllocateGraphMempoolId();
  }
  compilation_context_ = GetCompilationContext();
}

bool TpuGraphImpl::IsCapturedByCurrentThreadLocked() const {
  return capture_ != nullptr && current_thread_capture_session == capture_;
}

TpuGraphImpl::ArgSource TpuGraphImpl::ResolveCapturedBufferLocked(
    const DeviceBufferRef& buffer, const CapturedGraphStorages& captured,
    SourceMap& sources) {
  if (const auto it = sources.find(buffer); it != sources.end()) {
    return it->second;
  }
  const InputRef input_ref{.input_idx = inputs_.size()};
  if (const auto storage_it = captured.placeholder_to_storage.find(buffer);
      storage_it != captured.placeholder_to_storage.end()) {
    inputs_.push_back(
        StorageInput{.storage = storage_it->second, .placeholder = buffer});
  } else {
    return buffer;
  }
  sources.emplace(buffer, input_ref);
  return input_ref;
}

TpuGraphImpl::CapturedStage TpuGraphImpl::BuildCapturedStageLocked(
    size_t stage_idx, std::unique_ptr<Traversal> traversal,
    const CapturedGraphStorages& captured, SourceMap& sources) {
  CapturedStage stage;
  stage.arg_sources.reserve(traversal->arguments().size());
  for (const DeviceBufferRef& arg : traversal->arguments()) {
    stage.arg_sources.push_back(
        ResolveCapturedBufferLocked(arg, captured, sources));
  }

  const auto& outputs = traversal->outputs();
  stage.output_shapes.reserve(outputs.size());
  for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
    const DeviceBufferRef& out_ref = outputs[out_idx];
    stage.output_shapes.emplace_back(CopyIntVector(out_ref.dimensions()),
                                     out_ref.element_type());
    sources.emplace(
        out_ref, StageOutputRef{.stage_idx = stage_idx, .output_idx = out_idx});
  }

  stage.traversal = std::move(traversal);
  return stage;
}

void TpuGraphImpl::capture_end() {
  absl::MutexLock lock(mu_);
  if (capture_ == nullptr) {
    TT_CHECK_THROW(false, error::kFailedPrecondition)
        << "capture_end() called without a matching capture_begin()";
    return;
  }
  TT_CHECK_THROW(IsCapturedByCurrentThreadLocked(), error::kFailedPrecondition)
      << "capture must end on the thread it began on";
  // Like CUDA graph capture, capture continues after this error, so that it can
  // still end on the capture stream.
  const auto [end_device_index, end_stream_id] = GetCurrentDeviceStreamId();
  TT_CHECK_THROW(end_device_index == capture_device_index_ &&
                     end_stream_id == capture_stream_id_,
                 error::kFailedPrecondition)
      << "capture must end on the same device and stream it began on";

  if (!IsStreamCapturedByCurrentThread(capture_device_index_,
                                       capture_stream_id_)) {
    AbortGraphCapture(std::move(capture_));
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal test stream reset only.
        false, error::kFailedPrecondition)
        << "cannot end graph capture because the capture stream state was "
           "cleared";
    return;
  }

  const absl::Status capture_error =
      GetStreamGraphCaptureError(capture_device_index_, capture_stream_id_);
  if (!capture_error.ok()) {
    AbortGraphCapture(std::move(capture_));
    TT_CHECK_THROW(false, error::kFailedPrecondition)
        << "graph capture failed because of an earlier error during capture: "
        << capture_error.message();
    return;
  }

  const CapturedGraphStorages captured =
      EndGraphCaptureStorageTracking(*capture_->storage);
  // Compiles the captured operations as a full graph
  // (GraphSplitMode::kFullGraph), splitting only if an op requires a hard
  // OpSplitMode boundary.
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals =
      ExtractCapturedTraversals(capture_device_index_, capture_stream_id_,
                                CollectMustMaterialize(captured));
  // Ends the capture before clearing the storages, so that capture does not
  // record the operations that clear them.
  EndGraphCapture(*capture_);
  capture_.reset();
  TT_THROW_IF_ERROR(ClearCapturedDeferredStorages(captured));

  SourceMap sources;
  inputs_.clear();
  captured_stages_.clear();
  captured_stages_.reserve(traversals.size());
  for (size_t stage_idx = 0; stage_idx < traversals.size(); ++stage_idx) {
    captured_stages_.push_back(BuildCapturedStageLocked(
        stage_idx, std::move(traversals[stage_idx]), captured, sources));
  }

  writes_.clear();
  writes_.reserve(captured.writes.size());
  for (const CapturedStorageWrite& write : captured.writes) {
    writes_.push_back(StorageWrite{
        .source =
            ResolveCapturedBufferLocked(write.final_buffer, captured, sources),
        .target = write.storage,
        .size_bytes = PhysicalByteSize(write.final_buffer)});
  }

  if (!keep_graph_) {
    InstantiateLocked();
  }
  capture_ended_ = true;
}

void TpuGraphImpl::instantiate() {
  absl::MutexLock lock(mu_);
  TT_CHECK_THROW(capture_ended_, error::kFailedPrecondition)
      << "capture_end() must be called before instantiate()";
  if (instantiated_) {
    TT_CHECK_THROW(keep_graph_, error::kFailedPrecondition)
        << "instantiate() is intended to be called explicitly only when "
           "keep_graph=True";
  }
  InstantiateLocked();
}

void TpuGraphImpl::InstantiateLocked() {
  const CompilationSpec& compilation_spec =
      compilation_context_->compilation_specs().at(
          CompilationMode::kFastRuntime);
  std::unique_ptr<mlir::MLIRContext> mlir_context = MakeMlirContext();

  // Executables compile in the background, so starting all compilations before
  // waiting for any compiles the stages concurrently.
  std::vector<CompiledTraversal> compiled_traversals;
  compiled_traversals.reserve(captured_stages_.size());
  for (const CapturedStage& stage : captured_stages_) {
    TT_ASSIGN_OR_THROW(CompiledTraversal compiled,
                       CompileCapturedTraversal(*stage.traversal, *mlir_context,
                                                compilation_spec));
    compiled_traversals.push_back(std::move(compiled));
  }

  std::vector<CompiledStage> compiled_stages;
  compiled_stages.reserve(captured_stages_.size());
  for (size_t stage_idx = 0; stage_idx < captured_stages_.size(); ++stage_idx) {
    const CapturedStage& stage = captured_stages_[stage_idx];
    CompiledTraversal& compiled = compiled_traversals[stage_idx];
    TT_ASSIGN_OR_THROW(SharedLoadedExecutableWithMetadata executable,
                       compiled.compiled_kernel.fixed_shape_kernel.get());
    compiled_stages.push_back(CompiledStage{
        .executable = std::move(executable),
        .arg_sources = stage.arg_sources,
        .output_shapes = stage.output_shapes,
        .task_name = absl::StrCat("tpu_graph_", compiled.compilation_cache_key),
    });
  }
  compiled_stages_ = std::move(compiled_stages);

  if (!keep_graph_) {
    captured_stages_.clear();
  }
  instantiated_ = true;
}

std::vector<DeviceBufferRef> TpuGraphImpl::ReadInputsLocked() const {
  std::vector<DeviceBufferRef> buffers;
  buffers.reserve(inputs_.size());
  for (const StorageInput& input : inputs_) {
    TT_ASSIGN_OR_THROW(DeviceBufferRef buffer,
                       ReadStorageInput(input.storage, input.placeholder));
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

void TpuGraphImpl::ValidateWriteTargetsLocked() const {
  for (const StorageWrite& write : writes_) {
    TT_ASSIGN_OR_THROW(const DeviceBufferRef target_buffer,
                       GetBaseBuffer(write.target));
    const size_t target_bytes = PhysicalByteSize(target_buffer);
    TT_CHECK_THROW(target_bytes == write.size_bytes, error::kFailedPrecondition)
        << "cannot replay a graph because an output tensor storage was resized "
        << "from " << write.size_bytes << " to " << target_bytes
        << " bytes after capture";
  }
}

DeviceBufferRef TpuGraphImpl::ResolveSingleArg(
    const ArgSource& source, absl::Span<const DeviceBufferRef> inputs,
    const std::vector<std::vector<DeviceBufferRef>>& stage_outputs) {
  if (const auto* input_ref = std::get_if<InputRef>(&source)) {
    return inputs[input_ref->input_idx];
  }
  if (const auto* stage_ref = std::get_if<StageOutputRef>(&source)) {
    return stage_outputs[stage_ref->stage_idx][stage_ref->output_idx];
  }
  return std::get<DeviceBufferRef>(source);
}

std::vector<DeviceBufferRef> TpuGraphImpl::ResolveStageArgs(
    const CompiledStage& stage, absl::Span<const DeviceBufferRef> inputs,
    const std::vector<std::vector<DeviceBufferRef>>& stage_outputs) {
  std::vector<DeviceBufferRef> args;
  args.reserve(stage.arg_sources.size());
  std::vector<DeviceBufferRef> deferred_inputs;

  for (const ArgSource& source : stage.arg_sources) {
    DeviceBufferRef arg_ref = ResolveSingleArg(source, inputs, stage_outputs);
    if (arg_ref.is_deferred()) {
      deferred_inputs.push_back(arg_ref);
    }
    args.push_back(std::move(arg_ref));
  }

  if (!deferred_inputs.empty()) {
    TT_THROW_IF_ERROR(Materialize(
        deferred_inputs, MaterializationReason::kCompileModeExecution));
  }
  return args;
}

void TpuGraphImpl::ApplyWritesLocked(
    absl::Span<const DeviceBufferRef> inputs,
    const std::vector<std::vector<DeviceBufferRef>>& stage_outputs) const {
  for (const StorageWrite& write : writes_) {
    TT_THROW_IF_ERROR(AssignBufferToStorage(
        ResolveSingleArg(write.source, inputs, stage_outputs), write.target));
  }
}

void TpuGraphImpl::replay() {
  if (IsCurrentThreadCapturingGraph()) {
    TT_THROW_IF_ERROR(InvalidateCurrentThreadGraphCapture(
        TT_ERROR(error::kFailedPrecondition)
        << "cannot replay a graph while capturing a graph"));
  }
  absl::MutexLock lock(mu_);
  TT_CHECK_THROW(capture_ended_, error::kFailedPrecondition)
      << "cannot replay a graph without a preceding successful capture";
  // Like a CUDA graph, the graph replays on the current stream of the device it
  // was captured on, whatever the current device.
  const c10::DeviceGuard device_guard(
      c10::Device(c10::DeviceType::PrivateUse1, capture_device_index_));
  if (!instantiated_) {
    InstantiateLocked();
  }

  // Reads every input before any write, as the captured stages do, and checks
  // every write target before running any stage, so that a failed replay
  // leaves all tensors unchanged.
  const std::vector<DeviceBufferRef> inputs = ReadInputsLocked();
  ValidateWriteTargetsLocked();
  std::vector<std::vector<DeviceBufferRef>> stage_outputs;
  stage_outputs.reserve(compiled_stages_.size());
  for (const CompiledStage& stage : compiled_stages_) {
    std::vector<DeviceBufferRef> args =
        ResolveStageArgs(stage, inputs, stage_outputs);
    TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> outputs,
                       EnqueueExecutable(stage.executable, std::move(args),
                                         stage.output_shapes, stage.task_name));
    stage_outputs.push_back(std::move(outputs));
  }
  ApplyWritesLocked(inputs, stage_outputs);
}

void TpuGraphImpl::reset() {
  absl::MutexLock lock(mu_);
  if (capture_ != nullptr) {
    TT_CHECK_THROW(IsCapturedByCurrentThreadLocked(),
                   error::kFailedPrecondition)
        << "cannot reset a graph that another thread is capturing";
    AbortGraphCapture(std::move(capture_));
  }
  ClearCapturedGraphLocked();
}

void TpuGraphImpl::ClearCapturedGraphLocked() {
  capture_ended_ = false;
  instantiated_ = false;
  mempool_id_ = {0, 0};
  compilation_context_.reset();
  inputs_.clear();
  writes_.clear();
  captured_stages_.clear();
  compiled_stages_.clear();
}

at::MempoolId_t TpuGraphImpl::pool() const {
  absl::MutexLock lock(mu_);
  TT_CHECK_THROW(capture_ended_, error::kFailedPrecondition)
      << "pool() called without a preceding successful capture";
  return mempool_id_;
}

void TpuGraphImpl::enable_debug_mode() {
  // Debug mode has no effect, as debug_dump() is not supported yet.
}

void TpuGraphImpl::debug_dump(const std::string& /*path*/) {
  TT_WARN(WarningCategory::kTorchTpu,
          "debug_dump() is not supported on TPU yet; no graph dump produced");
}

#endif  // TT_TORCH_VERSION_GE(2, 12)

void RegisterTpuGraphImpl() {}

}  // namespace torch_tpu

#if TT_TORCH_VERSION_GE(2, 12)
namespace at {  // NON_TORCH_TPU_OK=REGISTER_GRAPH_IMPL must be in namespace at.
REGISTER_GRAPH_IMPL(TPU, torch_tpu::TpuGraphImpl)
REGISTER_GRAPH_IMPL(PRIVATEUSEONE, torch_tpu::TpuGraphImpl)
}  // namespace at
#endif  // TT_TORCH_VERSION_GE(2, 12)
