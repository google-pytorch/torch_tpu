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

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
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
#include "csrc/eager/graph_capture_state.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/traversal.h"

namespace torch_tpu {

#if TT_TORCH_VERSION_GE(2, 12)  // PyTorch 2.12 added torch.accelerator.Graph.

// Implements `torch.accelerator.Graph` on TPU.
//
// Capture runs the captured region in `EagerMode::kInternalCompileFxGraph` and
// compiles the deferred ops as a full graph (`GraphSplitMode::kFullGraph`,
// splitting only at hard `OpSplitMode` boundaries). Like a CUDA graph, a graph
// must be captured on a non-default stream, and replays on
// the current stream. Capture records the operations that the capturing thread
// issues on the stream that capture began on: issuing an operation on another
// stream during capture is an error, and capture must end on the thread and
// stream it began on. Like a CUDA graph capture, the capture continues if
// capture_end() fails because it is called on another stream. Like the capture
// of a CUDA graph, an operation that is not permitted during capture, such as a
// synchronization, invalidates the capture: capture_end() then fails, and the
// graph can capture again after reset(). The first read of a tensor storage
// that existed before capture stands for the storage's contents at replay time,
// so every replay reads its inputs anew. Similarly, like a CUDA graph, every
// replay copies the contents that a pinned CPU tensor has at replay time if
// capture copied it to the device without blocking. Every replay overwrites the
// storages that existed before capture and were written during capture, as well
// as the storages created during capture that were still alive when capture
// ended.
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
    // The MLIR of the stage, which debug_dump() writes. Empty until compiled
    // while the graph is kept.
    std::string mlir_text;
  };

  struct CompiledStage {
    SharedLoadedExecutableWithMetadata executable;
    std::vector<ArgSource> arg_sources;
    std::vector<Shape> output_shapes;
    std::string task_name;
  };

  // Maps the buffers used during capture to their sources at replay time.
  using SourceMap = absl::flat_hash_map<DeviceBufferRef, ArgSource>;

  // Leaves capture mode on the current thread and returns the tensor storages
  // accessed during capture.
  CapturedGraphStorages EndCaptureOnCurrentThread();
  // Returns the source at replay time of `buffer`, a buffer used during
  // capture. Adds the input storage or pinned CPU tensor that `buffer` stands
  // for to `inputs_` on its first use.
  ArgSource ResolveCapturedBuffer(const DeviceBufferRef& buffer,
                                  const CapturedGraphStorages& captured,
                                  SourceMap& sources);
  CapturedStage BuildCapturedStage(size_t stage_idx,
                                   std::unique_ptr<Traversal> traversal,
                                   const CapturedGraphStorages& captured,
                                   SourceMap& sources);
  // Returns the buffers of the inputs, in the order of `inputs_`: the current
  // buffers of the input storages, and copies to the device of the current
  // contents of the pinned CPU tensors.
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
  // Compiles the captured stages whose MLIR is not yet known, to populate it.
  void PopulateStageMlirTexts();

  bool keep_graph_ = false;
  bool capture_ended_ = false;
  bool instantiated_ = false;
  // Whether autograd multithreading was enabled on the capturing thread before
  // capture_begin() disabled it.
  bool autograd_multithreading_was_enabled_ = true;
  // Non-null while a capture initiated on this Graph instance is in progress.
  // Also pushed onto the capturing thread's ContextManagerState stack, so
  // comparing against GetCurrentThreadGraphCapture() identifies the capturing
  // thread.
  absl_nullable std::shared_ptr<GraphCaptureState> active_capture_;
  c10::DeviceIndex capture_device_index_ = 0;
  c10::StreamId capture_stream_id_ = 0;
  at::MempoolId_t mempool_id_{0, 0};
  std::shared_ptr<const CompilationContext> compilation_context_;
  std::vector<Input> inputs_;
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
