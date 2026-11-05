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

#include "csrc/eager/graph_capture_state.h"

#include <atomic>
#include <optional>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/events_queue.h"

namespace torch_tpu {
namespace {

// The stream that the current thread captures a graph on, if any. Validated
// against the state of the stream on every read.
thread_local  // CPP_THREAD_LOCAL_OK=ownership stays on the capturing thread
    std::optional<DeviceStreamId>
        current_thread_capture_stream;

// The number of orphaned graph captures.
ABSL_CONST_INIT std::atomic<int> orphaned_graph_captures{0};

// Ends the orphaned graph captures that the current thread can end.
ABSL_CONST_INIT std::atomic<void (*)()> end_orphaned_graph_captures{nullptr};

// Whether the current thread ends orphaned graph captures, which calls the
// entry points that end them.
ABSL_CONST_INIT thread_local  // CPP_THREAD_LOCAL_OK=per-thread reentrancy guard
    bool ending_orphaned_graph_captures = false;

}  // namespace

void SetCurrentThreadGraphCaptureStream(const DeviceStreamId stream) {
  current_thread_capture_stream = stream;
}

void ClearCurrentThreadGraphCaptureStream() {
  current_thread_capture_stream.reset();
}

std::optional<DeviceStreamId> GetCurrentThreadGraphCaptureStream() {
  if (ABSL_PREDICT_TRUE(!current_thread_capture_stream.has_value())) {
    return std::nullopt;
  }
  const DeviceStreamId stream = *current_thread_capture_stream;
  if (!IsStreamCapturedByCurrentThread(stream.device_index, stream.stream_id)) {
    // The capture ended without clearing the index, for example because the
    // state of its stream was cleared.
    current_thread_capture_stream.reset();
    return std::nullopt;
  }
  return stream;
}

bool IsCurrentThreadCapturingGraph() {
  return GetCurrentThreadGraphCaptureStream().has_value();
}

absl::Status InvalidateCurrentThreadGraphCapture(absl::Status error) {
  if (const std::optional<DeviceStreamId> stream =
          GetCurrentThreadGraphCaptureStream();
      stream.has_value()) {
    InvalidateStreamGraphCapture(stream->device_index, stream->stream_id,
                                 error);
  }
  return error;
}

absl::Status ValidateGraphCaptureStream(const DeviceStreamId capture_stream,
                                        const DeviceStreamId stream) {
  if (stream.device_index == capture_stream.device_index &&
      stream.stream_id == capture_stream.stream_id) {
    return absl::OkStatus();
  }
  // DeviceIndex is a char type, so it is cast to int to print as a number.
  return TT_ERROR(error::kFailedPrecondition)
         << "operations captured into a graph must run on the stream the "
            "capture began on (device "
         << static_cast<int>(capture_stream.device_index) << ", stream "
         << capture_stream.stream_id << "), but an operation ran on device "
         << static_cast<int>(stream.device_index) << ", stream "
         << stream.stream_id;
}

void AddOrphanedGraphCapture(void (*const end_orphaned_captures)()) {
  end_orphaned_graph_captures.store(end_orphaned_captures,
                                    std::memory_order_release);
  orphaned_graph_captures.fetch_add(1, std::memory_order_release);
}

void RemoveOrphanedGraphCapture() {
  orphaned_graph_captures.fetch_sub(1, std::memory_order_release);
}

void MaybeEndOrphanedGraphCaptures() {
  if (ABSL_PREDICT_TRUE(
          orphaned_graph_captures.load(std::memory_order_acquire) == 0) ||
      ending_orphaned_graph_captures || IsCurrentThreadCapturingGraph()) {
    return;
  }
  void (*const end_orphaned_captures)() =
      end_orphaned_graph_captures.load(std::memory_order_acquire);
  ending_orphaned_graph_captures = true;
  const absl::Cleanup done = [] { ending_orphaned_graph_captures = false; };
  end_orphaned_captures();
}

}  // namespace torch_tpu
