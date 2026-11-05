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

#include <optional>

#include "absl/status/status.h"
#include "csrc/eager/current_stream.h"

// Tracks the `torch.accelerator.Graph` capture that the current thread owns.
//
// The state of a graph capture lives with the stream that it records
// operations from (see TryReserveStreamGraphCapture() and
// ActivateStreamGraphCapture() in events_queue.h), and the thread that began
// the capture owns it. Capture records only the operations that the owner
// thread issues on the capture stream, so this per-thread index remembers the
// stream that the current thread captures a graph on, for the code paths that
// record operations to find the capture. Every query validates the index
// against the state of the stream, so the index never refers to a capture that
// has ended.
//
// The index is per thread so that capture ownership does not propagate to
// other threads, such as the threads of the autograd engine or of at::launch().

namespace torch_tpu {

// Records that the current thread owns the graph capture in progress on
// `stream`, which the current thread began via TryReserveStreamGraphCapture()
// and ActivateStreamGraphCapture().
void SetCurrentThreadGraphCaptureStream(DeviceStreamId stream);

// Records that the current thread owns no graph capture.
void ClearCurrentThreadGraphCaptureStream();

// Returns the stream that the current thread is capturing a graph on, or
// nullopt if the current thread is not capturing a graph. A capture that an
// error invalidated is still in progress until it ends.
[[nodiscard]] std::optional<DeviceStreamId>
GetCurrentThreadGraphCaptureStream();

// Returns whether the current thread is capturing a graph.
[[nodiscard]] bool IsCurrentThreadCapturingGraph();

// Returns `error`. If the current thread is capturing a graph, also records
// `error` as the reason why the capture failed, unless an earlier error did:
// like the capture of a CUDA graph after an operation that is not permitted
// during capture, such as a synchronization, the capture continues,
// capture_end() fails, and reset() or destruction ends the capture without
// raising.
absl::Status InvalidateCurrentThreadGraphCapture(absl::Status error);

// Returns an error, which does not invalidate the capture, if `stream` is not
// `capture_stream`: capture records only the operations issued on the stream
// that it began on, so the graph would miss an operation issued on `stream`.
absl::Status ValidateGraphCaptureStream(DeviceStreamId capture_stream,
                                        DeviceStreamId stream);

// Records that a graph capture was orphaned: its graph was destroyed or reset
// during capture by a thread that could not end it. `end_orphaned_captures`
// ends the orphaned captures that the current thread can end.
void AddOrphanedGraphCapture(void (*end_orphaned_captures)());

// Records that an orphaned graph capture ended.
void RemoveOrphanedGraphCapture();

// Ends the orphaned graph captures that the current thread can end, if any, so
// that their streams and the tensors that they accessed are usable again. Does
// nothing while the current thread captures a graph, which would record the
// operations that end them, or already ends orphaned captures. Cheap if no
// capture is orphaned. Must be called without holding locks, at the entry
// points of the operations that can use the stream or the tensors of an
// orphaned capture.
void MaybeEndOrphanedGraphCaptures();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_GRAPH_CAPTURE_STATE_H_
