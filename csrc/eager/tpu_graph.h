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

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "csrc/common/macro_utils.h"
#if TT_TORCH_VERSION_GE(2, 12)
#include "ATen/core/GraphImplInterface.h"
#endif
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
// Capture records the deferred ops that the capturing thread creates on the
// capture stream, in every eager mode, without running them, and compiles them
// as a full graph (`GraphSplitMode::kFullGraph`, splitting only at hard
// `OpSplitMode` boundaries). Like a CUDA graph, a graph must be captured on a
// non-default stream, and replays on the current stream. The capture stream
// reports that it is capturing (see `torch.Stream.is_capturing()`) until
// capture ends.
//
// Whatever the capture mode, capture records the operations that the thread
// that began it issues on the capture stream: issuing an operation on another
// stream during capture is an error, and capture must end on the thread and
// stream it began on. Capture ownership does not propagate to other threads,
// so the backward passes during capture run on the capturing thread. Like a
// CUDA graph capture, the capture continues if capture_end() fails because it
// is called on another stream. Like the capture of a CUDA graph, an operation
// that is not permitted during capture, such as a synchronization, invalidates
// the capture: capture_end() then fails, and the graph can capture again after
// reset(). The first read of a tensor storage that existed before capture
// stands for the storage's contents at replay time, so every replay reads its
// inputs anew. Every replay overwrites the storages that existed before capture
// and were written during capture, as well as the storages created during
// capture that were still alive when capture ended. Replays do not check the
// assertions of `torch._assert_async()`.
//
// Like a CUDA graph, which keeps reading and writing the device memory of its
// input and output tensors after they are freed, the graph holds a reference
// to every tensor storage it reads or writes until reset() or destruction, so
// that replays remain valid after the user drops these tensors. Replay fails if
// one of these storages was resized after capture.
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

  // Ends the capture that the current thread owns: the capture stream stops
  // capturing, and the current thread leaves capture mode. Requires the tensor
  // storage tracking of the capture to have ended.
  void EndCaptureOnCurrentThread();
  // Ends the capture that the current thread owns without keeping the captured
  // operations, and restores the tensor storages that existed before capture.
  void AbortCaptureOnCurrentThread();
  // Returns whether the current thread owns the capture in progress.
  bool IsCapturedByCurrentThread() const;
  // Clears the captured graph.
  void ClearCapturedGraph();
  // Returns the source at replay time of `buffer`, a buffer used during
  // capture. Adds the input storage that `buffer` stands for to `inputs_` on
  // its first use.
  ArgSource ResolveCapturedBuffer(const DeviceBufferRef& buffer,
                                  const CapturedGraphStorages& captured,
                                  SourceMap& sources);
  CapturedStage BuildCapturedStage(size_t stage_idx,
                                   std::unique_ptr<Traversal> traversal,
                                   const CapturedGraphStorages& captured,
                                   SourceMap& sources);
  // Returns the current buffers of the input storages, in the order of
  // `inputs_`.
  std::vector<DeviceBufferRef> ReadInputs() const;
  // Checks that every storage that replay overwrites still has the size it had
  // at the end of capture.
  void ValidateWriteTargets() const;
  static DeviceBufferRef ResolveSingleArg(
      const ArgSource& source, absl::Span<const DeviceBufferRef> inputs,
      const std::vector<std::vector<DeviceBufferRef>>& stage_outputs);
  static std::vector<DeviceBufferRef> ResolveStageArgs(
      const CompiledStage& stage, absl::Span<const DeviceBufferRef> inputs,
      const std::vector<std::vector<DeviceBufferRef>>& stage_outputs);
  void ApplyWrites(
      absl::Span<const DeviceBufferRef> inputs,
      const std::vector<std::vector<DeviceBufferRef>>& stage_outputs) const;

  bool keep_graph_ = false;
  bool capture_ended_ = false;
  bool instantiated_ = false;
  // Non-null while a capture that this graph began is in progress. The thread
  // that owns the capture shares it, so that the state of the capture outlives
  // the graph if another thread destroys the graph during capture.
  std::shared_ptr<GraphCaptureSession> capture_;
  c10::DeviceIndex capture_device_index_ = 0;
  c10::StreamId capture_stream_id_ = 0;
  at::MempoolId_t mempool_id_{0, 0};
  std::shared_ptr<const CompilationContext> compilation_context_;
  std::vector<StorageInput> inputs_;
  std::vector<StorageWrite> writes_;
  std::vector<CapturedStage> captured_stages_;
  std::vector<CompiledStage> compiled_stages_;
};

#endif  // TT_TORCH_VERSION_GE(2, 12)

// Ensures TpuGraphImpl is linked and registered with at::GraphImplRegistry
// when building against PyTorch 2.12 or later.
void RegisterTpuGraphImpl();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_TPU_GRAPH_H_
