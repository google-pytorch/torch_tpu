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

#ifndef TORCH_TPU_CSRC_EAGER_GRAPH_CAPTURE_STATE_H_
#define TORCH_TPU_CSRC_EAGER_GRAPH_CAPTURE_STATE_H_

#include <atomic>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

namespace torch_tpu {

// Base context state for an in-progress `torch.accelerator.Graph` capture on a
// Python dispatch thread.
//
// A capture state is pushed onto the thread's `ContextManagerState` stack when
// stream reservation begins (`is_active() == false` while pre-capture stream
// work is flushed) and becomes active via `Activate()` once storage tracking
// and `EagerMode::kInternalCompileFxGraph` begin.
class GraphCaptureState {
 public:
  GraphCaptureState() = default;
  virtual ~GraphCaptureState() = default;

  GraphCaptureState(const GraphCaptureState&) = delete;
  GraphCaptureState& operator=(const GraphCaptureState&) = delete;
  GraphCaptureState(GraphCaptureState&&) = delete;
  GraphCaptureState& operator=(GraphCaptureState&&) = delete;

  // Marks the capture as actively recording operations on its dispatch thread.
  void Activate() { active_.store(true, std::memory_order_release); }

  // Returns whether this capture is actively recording operations.
  [[nodiscard]] bool is_active() const {
    return active_.load(std::memory_order_acquire);
  }

  // Records `error` as the reason why the capture failed if no prior error was
  // recorded.
  void RecordError(absl::Status error) ABSL_LOCKS_EXCLUDED(mu_);

  // Returns the first error recorded by `RecordError()`, or OK.
  absl::Status error() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  std::atomic<bool> active_{false};
  mutable absl::Mutex mu_;
  absl::Status error_ ABSL_GUARDED_BY(mu_);
};

// Pushes `state` onto the current Python dispatch thread's context state stack.
void PushCurrentThreadGraphCapture(
    absl_nonnull std::shared_ptr<GraphCaptureState> state);

// Pops the innermost `GraphCaptureContextState` from the current Python
// dispatch thread's context state stack.
void PopCurrentThreadGraphCapture();

// Returns the innermost `GraphCaptureState` on the current Python dispatch
// thread (including while `is_active() == false`), or nullptr if none is
// pushed.
[[nodiscard]] GraphCaptureState* absl_nullable GetCurrentThreadGraphCapture();

// Returns whether the current thread is actively capturing a
// `torch.accelerator.Graph` (`is_active() == true`).
[[nodiscard]] bool IsCurrentThreadCapturingGraph();

// Returns `error`. If the current thread is actively capturing a graph and no
// error has invalidated the capture yet, also records `error` as the reason why
// the capture failed: like the capture of a CUDA graph after an operation that
// is not permitted during capture, such as a synchronization, the capture can
// continue but capture_end() fails.
absl::Status InvalidateCurrentThreadGraphCapture(absl::Status error);

// Returns the error that invalidated the graph capture in progress on the
// current thread, or OK.
absl::Status GetCurrentThreadGraphCaptureError();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_GRAPH_CAPTURE_STATE_H_
