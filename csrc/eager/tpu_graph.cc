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
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "csrc/common/macro_utils.h"
#include "mlir/IR/MLIRContext.h"
#if TT_TORCH_VERSION_GE(2, 12)
#include "ATen/core/GraphImplInterface.h"
#endif
#include "ATen/core/TensorBody.h"
#include "c10/core/Allocator.h"
#include "c10/core/AutogradState.h"
#include "c10/core/Device.h"
#include "c10/core/DeviceGuard.h"
#include "c10/core/Storage.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/utils.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/device_gen_impl.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/execution_task.h"
#include "csrc/eager/graph_capture_state.h"
#include "csrc/eager/host_allocator.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/traversal.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {

#if TT_TORCH_VERSION_GE(2, 12)  // See tpu_graph.h.

namespace {

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

// Returns the strides of a contiguous tensor with the given sizes.
Strides ContiguousStrides(absl::Span<const int64_t> sizes) {
  Strides strides(sizes.size(), 1);
  for (size_t i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * std::max<int64_t>(sizes[i - 1], 1);
  }
  return strides;
}

// Returns the current buffer of `storage`, an input of the graph, interpreted
// with the shape and element type of `placeholder`, which stood for the
// contents of `storage` during capture.
absl::StatusOr<DeviceBufferRef> ReadStorageInput(
    const c10::Storage& storage, const DeviceBufferRef& placeholder) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer, GetBaseBuffer(storage));
  TT_RET_CHECK(buffer.size_bytes() == placeholder.size_bytes(),
               error::kFailedPrecondition)
      << "cannot replay a graph because an input tensor storage was resized "
      << "from " << placeholder.size_bytes() << " to " << buffer.size_bytes()
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
                .strides = ContiguousStrides(sizes),
                .storage_offset = 0,
                .element_type = placeholder.element_type()},
      /*view_is_conj=*/false, DonateBase::kDoNotDonate);
}

// Copies the current contents of `host`, a pinned CPU tensor that capture
// copied to the device, to a new buffer on the current device, with the shape
// and element type of `placeholder`, which stood for the copy during capture.
absl::StatusOr<DeviceBufferRef> ReadHostInput(
    const at::Tensor& host, const DeviceBufferRef& placeholder) {
  const void* const pinned_ptr = host.storage().data_ptr().get();
  HostToDeviceTransfer transfer = HostToDeviceTransfer::FromTensor(
      host.const_data_ptr(), pinned_ptr, placeholder.element_type(),
      placeholder.dimensions());
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer,
                      ScheduleHostToDeviceTransfer(std::move(transfer)));
  if (buffer.is_transferring()) {
    HoldPinnedPointerUntil(pinned_ptr, buffer.GetReadyFuture());
  }
  return buffer;
}

}  // namespace

TpuGraphImpl::TpuGraphImpl(const at::GraphImplArgs& args)
    : keep_graph_(args.keep_graph) {}

TpuGraphImpl::~TpuGraphImpl() { reset(); }

void TpuGraphImpl::capture_begin(at::MempoolId_t pool,
                                 at::GraphCaptureMode capture_mode) {
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
  TT_CHECK_THROW(active_capture_ == nullptr, error::kFailedPrecondition)
      << "graph capture is already in progress";
  // Like CUDAGraph, once capture_begin() assigns a pool, reset() is required
  // before capturing again, even if capture_end() failed.
  TT_CHECK_THROW(
      !capture_ended_ && mempool_id_.first == 0 && mempool_id_.second == 0,
      error::kFailedPrecondition)
      << "this Graph instance already owns a captured graph; call reset() or "
         "create a new Graph instance before capturing again";
  TT_CHECK_THROW(GetCurrentThreadGraphCapture() == nullptr,
                 error::kFailedPrecondition)
      << "nested graph capture on the same thread is not supported";

  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  // Stream 0 is the default stream of every device.
  TT_CHECK_THROW(stream_id != 0, error::kFailedPrecondition)
      << "graphs must be captured on a non-default stream (after capture, "
         "they can be replayed on the default stream)";
  capture_device_index_ = device_index;
  capture_stream_id_ = stream_id;

  absl_nonnull std::shared_ptr<GraphCaptureState> capture =
      CreateGraphCaptureStorageState(
          DeviceStreamId{.device_index = capture_device_index_,
                         .stream_id = capture_stream_id_});
  PushCurrentThreadGraphCapture(capture);
  absl::Cleanup rollback_thread_capture = [] {
    PopCurrentThreadGraphCapture();
  };

  // Until capture ends, other threads cannot use the capture stream nor the
  // tensors that capture accesses, since only this thread records operations.
  TT_THROW_IF_ERROR(BeginStreamGraphCapture(capture_device_index_,
                                            capture_stream_id_, capture));
  absl::Cleanup rollback_stream_capture = [this] {
    EndStreamGraphCapture(capture_device_index_, capture_stream_id_);
  };

  // Ensure the capture device's default RNG generator is initialized before
  // capture starts, as initializing its state tensor allocates and transfers
  // from host memory.
  GetDefaultDeviceGenerator(capture_device_index_);

  // Flush and materialize any pending work on the capture stream before
  // starting graph capture so pre-existing deferred ops do not leak into the
  // captured graph.
  TT_ASSIGN_OR_THROW(
      EventSnapshot snapshot,
      MaterializeStream(capture_device_index_, capture_stream_id_,
                        MaterializationReason::kExplicitSync,
                        GraphSplitMode::kSplitGraph));
  TT_THROW_IF_ERROR(snapshot.Synchronize());

  if (pool.first != 0 || pool.second != 0) {
    mempool_id_ = pool;
  } else {
    mempool_id_ = AllocateGraphMempoolId();
  }

  PushContextState<EagerModeContextState>(EagerMode::kInternalCompileFxGraph);
  capture->Activate();
  // The autograd engine runs the backward ops of TPU tensors on a per-device
  // thread unless multithreading is disabled, while capture tracks the tensor
  // storages accessed on the capturing thread only. Backward passes must run on
  // the capturing thread for capture to record the gradients they write.
  c10::AutogradState& autograd_state = c10::AutogradState::get_tls_state();
  autograd_multithreading_was_enabled_ =
      autograd_state.get_multithreading_enabled();
  autograd_state.set_multithreading_enabled(false);
  active_capture_ = std::move(capture);
  std::move(rollback_stream_capture).Cancel();
  std::move(rollback_thread_capture).Cancel();
}

CapturedGraphStorages TpuGraphImpl::EndCaptureOnCurrentThread() {
  active_capture_.reset();
  c10::AutogradState::get_tls_state().set_multithreading_enabled(
      autograd_multithreading_was_enabled_);
  PopContextState<EagerModeContextState>();
  return EndGraphCaptureStorageTracking();
}

TpuGraphImpl::ArgSource TpuGraphImpl::ResolveCapturedBuffer(
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
  } else if (const auto host_it =
                 captured.placeholder_to_host_tensor.find(buffer);
             host_it != captured.placeholder_to_host_tensor.end()) {
    inputs_.push_back(
        HostInput{.host = host_it->second, .placeholder = buffer});
  } else {
    return buffer;
  }
  sources.emplace(buffer, input_ref);
  return input_ref;
}

TpuGraphImpl::CapturedStage TpuGraphImpl::BuildCapturedStage(
    size_t stage_idx, std::unique_ptr<Traversal> traversal,
    const CapturedGraphStorages& captured, SourceMap& sources) {
  CapturedStage stage;
  stage.arg_sources.reserve(traversal->arguments().size());
  for (const DeviceBufferRef& arg : traversal->arguments()) {
    stage.arg_sources.push_back(ResolveCapturedBuffer(arg, captured, sources));
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
  TT_CHECK_THROW(active_capture_ != nullptr, error::kFailedPrecondition)
      << "capture_end() called without a matching capture_begin()";
  TT_CHECK_THROW(GetCurrentThreadGraphCapture() == active_capture_.get(),
                 error::kFailedPrecondition)
      << "capture must end on the thread it began on";
  // Like CUDA graph capture, capture continues after this error, so that it can
  // still end on the capture stream.
  const auto [end_device_index, end_stream_id] = GetCurrentDeviceStreamId();
  TT_CHECK_THROW(end_device_index == capture_device_index_ &&
                     end_stream_id == capture_stream_id_,
                 error::kFailedPrecondition)
      << "capture must end on the same device and stream it began on";

  const absl::Status capture_error = active_capture_->error();
  const CapturedGraphStorages captured = EndCaptureOnCurrentThread();
  // Other threads can use the capture stream again once the preexisting
  // storages are restored and the captured operations are off the stream.
  absl::Cleanup end_stream_capture = [this] {
    EndStreamGraphCapture(capture_device_index_, capture_stream_id_);
  };

  if (!capture_error.ok()) {
    ExtractCapturedTraversals(capture_device_index_, capture_stream_id_, {});
    std::move(end_stream_capture).Invoke();
    // The error below is more relevant than a failure to clear the storages.
    ClearCapturedDeferredStorages(captured)
        .IgnoreError();  // IGNORE_ERROR_OK=raises the capture error instead
    TT_CHECK_THROW(false, error::kFailedPrecondition)
        << "graph capture failed because of an earlier error during capture: "
        << capture_error.message();
  }

  compilation_context_ = GetCompilationContext();
  // Compiles the captured operations as a full graph
  // (GraphSplitMode::kFullGraph), splitting only if an op requires a hard
  // OpSplitMode boundary.
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals =
      ExtractCapturedTraversals(capture_device_index_, capture_stream_id_,
                                CollectMustMaterialize(captured));
  std::move(end_stream_capture).Invoke();
  TT_THROW_IF_ERROR(ClearCapturedDeferredStorages(captured));

  SourceMap sources;
  inputs_.clear();
  captured_stages_.clear();
  captured_stages_.reserve(traversals.size());
  for (size_t stage_idx = 0; stage_idx < traversals.size(); ++stage_idx) {
    captured_stages_.push_back(BuildCapturedStage(
        stage_idx, std::move(traversals[stage_idx]), captured, sources));
  }

  writes_.clear();
  writes_.reserve(captured.writes.size());
  for (const CapturedStorageWrite& write : captured.writes) {
    writes_.push_back(StorageWrite{
        .source = ResolveCapturedBuffer(write.final_buffer, captured, sources),
        .target = write.storage,
        .size_bytes = write.final_buffer.size_bytes()});
  }

  capture_ended_ = true;
  if (!keep_graph_) {
    instantiate();
  }
}

void TpuGraphImpl::instantiate() {
  TT_CHECK_THROW(capture_ended_, error::kFailedPrecondition)
      << "capture_end() must be called before instantiate()";
  if (instantiated_) {
    TT_CHECK_THROW(keep_graph_, error::kFailedPrecondition)
        << "instantiate() is intended to be called explicitly only when "
           "keep_graph=True";
  }

  const CompilationSpec& compilation_spec =
      compilation_context_->compilation_specs().at(
          CompilationMode::kFastRuntime);
  std::unique_ptr<mlir::MLIRContext> mlir_context = MakeMlirContext();

  // Executables compile in the background, so starting all compilations before
  // waiting for any compiles the stages concurrently.
  std::vector<CompiledTraversal> compiled_traversals;
  compiled_traversals.reserve(captured_stages_.size());
  for (CapturedStage& stage : captured_stages_) {
    std::string* const mlir_text =
        keep_graph_ && stage.mlir_text.empty() ? &stage.mlir_text : nullptr;
    TT_ASSIGN_OR_THROW(CompiledTraversal compiled,
                       CompileCapturedTraversal(*stage.traversal, *mlir_context,
                                                compilation_spec, mlir_text));
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

std::vector<DeviceBufferRef> TpuGraphImpl::ReadInputs() const {
  std::vector<DeviceBufferRef> buffers;
  buffers.reserve(inputs_.size());
  for (const Input& input : inputs_) {
    if (const auto* storage_input = std::get_if<StorageInput>(&input)) {
      TT_ASSIGN_OR_THROW(
          DeviceBufferRef buffer,
          ReadStorageInput(storage_input->storage, storage_input->placeholder));
      buffers.push_back(std::move(buffer));
    } else {
      const auto& host_input = std::get<HostInput>(input);
      TT_ASSIGN_OR_THROW(
          DeviceBufferRef buffer,
          ReadHostInput(host_input.host, host_input.placeholder));
      buffers.push_back(std::move(buffer));
    }
  }
  return buffers;
}

void TpuGraphImpl::ValidateWriteTargets() const {
  for (const StorageWrite& write : writes_) {
    TT_ASSIGN_OR_THROW(const DeviceBufferRef target_buffer,
                       GetBaseBuffer(write.target));
    TT_CHECK_THROW(target_buffer.size_bytes() == write.size_bytes,
                   error::kFailedPrecondition)
        << "cannot replay a graph because an output tensor storage was resized "
        << "from " << write.size_bytes << " to " << target_buffer.size_bytes()
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
        deferred_inputs, MaterializationReason::kCompileModeExecution,
        GraphSplitMode::kSplitGraph,
        MaterializationBlockMode::kBlockUntilStarted));
  }
  return args;
}

void TpuGraphImpl::ApplyWrites(
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
  TT_CHECK_THROW(capture_ended_, error::kFailedPrecondition)
      << "cannot replay a graph without a preceding successful capture";
  // Like a CUDA graph, the graph replays on the current stream of the device it
  // was captured on, whatever the current device.
  const c10::DeviceGuard device_guard(
      c10::Device(c10::DeviceType::PrivateUse1, capture_device_index_));
  if (!instantiated_) {
    instantiate();
  }

  // Reads every input before any write, as the captured stages do, and checks
  // every write target before running any stage, so that a failed replay
  // leaves all tensors unchanged.
  const std::vector<DeviceBufferRef> inputs = ReadInputs();
  ValidateWriteTargets();
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
  ApplyWrites(inputs, stage_outputs);
}

void TpuGraphImpl::reset() {
  if (active_capture_ != nullptr &&
      GetCurrentThreadGraphCapture() == active_capture_.get()) {
    const CapturedGraphStorages captured = EndCaptureOnCurrentThread();
    ExtractCapturedTraversals(capture_device_index_, capture_stream_id_, {});
    EndStreamGraphCapture(capture_device_index_, capture_stream_id_);
    // reset() also runs from the destructor, so it cannot throw.
    ClearCapturedDeferredStorages(captured)
        .IgnoreError();  // IGNORE_ERROR_OK=reset() cannot throw
  } else if (active_capture_ != nullptr) {
    // The capture state lives in the context state stack of the capturing
    // thread, which cannot be popped from this thread. The capture stream stays
    // reserved for that thread.
    active_capture_.reset();
    TT_WARN(WarningCategory::kTorchTpu,
            "resetting a graph that another thread is capturing; that thread "
            "remains in capture mode");
  }
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
  TT_CHECK_THROW(capture_ended_, error::kFailedPrecondition)
      << "pool() called without a preceding successful capture";
  return mempool_id_;
}

void TpuGraphImpl::enable_debug_mode() {
  // Like a CUDA graph, debug mode keeps the captured graph after capture, as
  // keep_graph=True does, so that it can be dumped.
  keep_graph_ = true;
}

void TpuGraphImpl::PopulateStageMlirTexts() {
  const CompilationSpec& compilation_spec =
      compilation_context_->compilation_specs().at(
          CompilationMode::kFastRuntime);
  std::unique_ptr<mlir::MLIRContext> mlir_context;
  for (CapturedStage& stage : captured_stages_) {
    if (!stage.mlir_text.empty()) {
      continue;
    }
    if (mlir_context == nullptr) {
      mlir_context = MakeMlirContext();
    }
    TT_THROW_IF_ERROR(CompileCapturedTraversal(*stage.traversal, *mlir_context,
                                               compilation_spec,
                                               &stage.mlir_text)
                          .status());
  }
}

void TpuGraphImpl::debug_dump(const std::string& path) {
  TT_CHECK_THROW(capture_ended_, error::kFailedPrecondition)
      << "debug_dump() called without a preceding successful capture";
  if (!keep_graph_) {
    TT_WARN(WarningCategory::kTorchTpu,
            "debug_dump() called without enable_debug_mode() or "
            "keep_graph=True; no graph dump produced");
    return;
  }

  PopulateStageMlirTexts();

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  TT_CHECK_THROW(out.is_open(), error::kInvalidArgument)
      << "failed to open debug_dump path: " << path;
  for (size_t i = 0; i < captured_stages_.size(); ++i) {
    if (i > 0) {
      out << "\n// ----- Stage " << i << " -----\n";
    }
    out << captured_stages_[i].mlir_text;
  }
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
