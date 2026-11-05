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

#ifndef TORCH_TPU_CSRC_EAGER_TPU_GRAPH_H_
#define TORCH_TPU_CSRC_EAGER_TPU_GRAPH_H_

#include <cstddef>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "csrc/common/macro_utils.h"
#if TT_TORCH_VERSION_GE(2, 12)
#include "ATen/core/GraphImplInterface.h"
#endif
#include "ATen/core/TensorBody.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/core/Storage.h"
#include "c10/core/Stream.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/shape.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/traversal.h"

namespace torch_tpu {

#if TT_TORCH_VERSION_GE(2, 12)  // PyTorch 2.12 added torch.accelerator.Graph.

// The state of a graph capture in progress (see tpu_graph.cc).
struct GraphCaptureSession;

// Implements `torch.accelerator.Graph` on TPU.
//
// Capture and compilation:
// - Records deferred ops issued by the capturing thread on a non-default
//   capture stream in any eager mode and compiles them as a full graph
//   (`GraphSplitMode::kFullGraph`, splitting only at hard `OpSplitMode`
//   boundaries). The capture stream reports `is_capturing() == True` until
//   capture ends.
// - Capture ownership is thread-local: capture must begin and end on the same
//   thread and stream (backward passes run on the capturing thread).
// - Operations incompatible with capture (e.g. replaying another graph)
//   invalidate the capture so `capture_end()` fails; `reset()` is required
//   before capturing again.
// - `pool()` currently tracks a unique pool ID (or the user-supplied pool) for
//   API parity without aliasing PJRT device buffers across graphs.
// - `enable_debug_mode()` and `debug_dump()` are stubs until follow-up changes
//   implement graph dumping.
//
// Storages and replay:
// - First reads of pre-existing tensor storages capture placeholders so every
//   replay reads their runtime contents anew; likewise, non-blocking copies
//   from pinned CPU tensors copy their runtime contents at replay time.
// - Every replay overwrites captured output storages (both pre-existing and
//   live captured-created storages) and skips `torch._assert_async()` checks.
// - Strong references to accessed storages are retained until `reset()` or
//   destruction; replay fails if any accessed storage was resized after
//   capture.
//
// Thread safety:
// - Methods on a `TpuGraphImpl` instance are serialized by `mu_`.
// - Until capture ends, other threads raise instead of using the capture
//   stream or the tensors that capture accessed.
// - Resetting a graph that another thread is capturing raises unless that
//   thread exited; destroying such a graph makes that thread end the capture
//   when it begins another capture. Once that thread exits, the next stream,
//   event, query, or tensor entry point on a non-capturing thread ends the
//   capture.
// - A capture that another thread begins during `replay()` and that accesses
//   the outputs of the graph can make the replay fail after it updated some of
//   its outputs.
class TpuGraphImpl final : public at::GraphImplInterface {
 public:
  explicit TpuGraphImpl(const at::GraphImplArgs& args = {});
  ~TpuGraphImpl() override;

  TpuGraphImpl(const TpuGraphImpl&) = delete;
  TpuGraphImpl& operator=(const TpuGraphImpl&) = delete;

  void capture_begin(at::MempoolId_t pool,
                     at::GraphCaptureMode capture_mode) override;
  void capture_end() override;
  void instantiate() override;
  void replay() override;
  void reset() override;
  at::MempoolId_t pool() const override;
  void enable_debug_mode() override;
  void debug_dump(const std::string& path) override;

 private:
  // The index of an input in `inputs_`.
  struct InputRef {
    size_t input_idx = 0;
  };

  struct StageOutputRef {
    size_t stage_idx = 0;
    size_t output_idx = 0;
  };

  // The source of a buffer used by a replay: the contents of an input at the
  // start of the replay, an output of a stage, or a constant buffer.
  using ArgSource = std::variant<InputRef, StageOutputRef, DeviceBufferRef>;

  // A tensor storage that existed before capture and was read by the graph.
  struct StorageInput {
    c10::Storage storage;
    // The placeholder that stands for the contents of `storage` in the
    // captured stages.
    DeviceBufferRef placeholder;
  };

  // A pinned CPU tensor that capture copied to the device.
  struct HostInput {
    at::Tensor host;  // UNINITIALIZED_TENSOR_OK=set on construction
    // The placeholder that stands for the contents of `host` in the captured
    // stages.
    DeviceBufferRef placeholder;
  };

  using Input = std::variant<StorageInput, HostInput>;

  // A tensor storage that every replay overwrites with the buffer from
  // `source`.
  struct StorageWrite {
    ArgSource source;
    c10::Storage target;
    // The size of the buffer from `source`.
    size_t size_bytes = 0;
  };

  struct CapturedStage {
    std::unique_ptr<Traversal> traversal;
    std::vector<ArgSource> arg_sources;
    std::vector<Shape> output_shapes;
  };

  struct CompiledStage {
    SharedLoadedExecutableWithMetadata executable;
    std::vector<ArgSource> arg_sources;
    std::vector<Shape> output_shapes;
    std::string task_name;
  };

  // Maps the buffers used during capture to their sources at replay time.
  using SourceMap = absl::flat_hash_map<DeviceBufferRef, ArgSource>;

  // Returns whether the current thread owns the capture in progress.
  bool IsCapturedByCurrentThreadLocked() const ABSL_SHARED_LOCKS_REQUIRED(mu_);
  // Clears the captured graph.
  void ClearCapturedGraphLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Compiles captured stages into PJRT executables.
  void InstantiateLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Returns the source at replay time of `buffer`, a buffer used during
  // capture. Adds the input storage or pinned CPU tensor that `buffer` stands
  // for to `inputs_` on its first use.
  ArgSource ResolveCapturedBufferLocked(const DeviceBufferRef& buffer,
                                        const CapturedGraphStorages& captured,
                                        SourceMap& sources)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  CapturedStage BuildCapturedStageLocked(size_t stage_idx,
                                         std::unique_ptr<Traversal> traversal,
                                         const CapturedGraphStorages& captured,
                                         SourceMap& sources)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Checks every write target before reading/transferring replay inputs, then
  // returns the replay input buffers (reading storage inputs first and
  // scheduling host-to-device transfers only once all inputs and write targets
  // have passed validation).
  std::vector<DeviceBufferRef> ReadInputsAndValidateTargetsLocked() const
      ABSL_SHARED_LOCKS_REQUIRED(mu_);
  static DeviceBufferRef ResolveSingleArg(
      const ArgSource& source, absl::Span<const DeviceBufferRef> inputs,
      const std::vector<std::vector<DeviceBufferRef>>& stage_outputs);
  static std::vector<DeviceBufferRef> ResolveStageArgs(
      const CompiledStage& stage, absl::Span<const DeviceBufferRef> inputs,
      const std::vector<std::vector<DeviceBufferRef>>& stage_outputs);
  void ApplyWritesLocked(
      absl::Span<const DeviceBufferRef> inputs,
      const std::vector<std::vector<DeviceBufferRef>>& stage_outputs) const
      ABSL_SHARED_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  bool keep_graph_ ABSL_GUARDED_BY(mu_) = false;
  bool capture_ended_ ABSL_GUARDED_BY(mu_) = false;
  bool instantiated_ ABSL_GUARDED_BY(mu_) = false;
  // Non-null while a capture that this graph began is in progress. The thread
  // that owns the capture shares it, so that the state of the capture outlives
  // the graph if another thread destroys the graph during capture.
  std::shared_ptr<GraphCaptureSession> capture_ ABSL_GUARDED_BY(mu_);
  c10::DeviceIndex capture_device_index_ ABSL_GUARDED_BY(mu_) = 0;
  c10::StreamId capture_stream_id_ ABSL_GUARDED_BY(mu_) = 0;
  at::MempoolId_t mempool_id_ ABSL_GUARDED_BY(mu_){0, 0};
  std::shared_ptr<const CompilationContext> compilation_context_
      ABSL_GUARDED_BY(mu_);
  std::vector<Input> inputs_ ABSL_GUARDED_BY(mu_);
  std::vector<StorageWrite> writes_ ABSL_GUARDED_BY(mu_);
  std::vector<CapturedStage> captured_stages_ ABSL_GUARDED_BY(mu_);
  std::vector<CompiledStage> compiled_stages_ ABSL_GUARDED_BY(mu_);
};

#endif  // TT_TORCH_VERSION_GE(2, 12)

// Ensures TpuGraphImpl is linked and registered with at::GraphImplRegistry
// when building against PyTorch 2.12 or later.
void RegisterTpuGraphImpl();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_TPU_GRAPH_H_
