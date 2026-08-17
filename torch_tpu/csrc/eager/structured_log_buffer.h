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

#ifndef TORCH_TPU_EAGER_STRUCTURED_LOG_BUFFER_H_
#define TORCH_TPU_EAGER_STRUCTURED_LOG_BUFFER_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace torch_tpu {

// Represents the reason for an op materialization event.
enum class MaterializationReason {
  kUnknown,
  kCpuTransfer,
  kScalarConversion,
  kExplicitSync,
  kAutogradSync,
  kRepeatedOpDetection,
  kDebugMode,
  kDistributedOp,
  kCompileModeExecution,
};

constexpr std::string_view ToString(MaterializationReason reason) {
  switch (reason) {
    case MaterializationReason::kUnknown:
      return "unknown";
    case MaterializationReason::kCpuTransfer:
      return ".cpu()";
    case MaterializationReason::kScalarConversion:
      return "item()";
    case MaterializationReason::kExplicitSync:
      return "synchronize()";
    case MaterializationReason::kAutogradSync:
      return "autograd_post_backward";
    case MaterializationReason::kRepeatedOpDetection:
      return "repeated_op_heuristic";
    case MaterializationReason::kDebugMode:
      return "debug_mode";
    case MaterializationReason::kDistributedOp:
      return "distributed_op";
    case MaterializationReason::kCompileModeExecution:
      return "compile_mode_execution";
  }
  return "unknown";
}

// A structured log event representing a materialization event or other eager
// mode action.
struct StructuredLogEvent {
  StructuredLogEvent() = default;

  // Disable move and copy to force storing unique_ptr in containers and to
  // avoid O(N-fields) copies/moves of the (potentially large) payloads.
  StructuredLogEvent(const StructuredLogEvent&) = delete;
  StructuredLogEvent& operator=(const StructuredLogEvent&) = delete;
  StructuredLogEvent(StructuredLogEvent&&) = delete;
  StructuredLogEvent& operator=(StructuredLogEvent&&) = delete;

  // Event name, e.g. "torchtpu_eager/kMm+kAdd".
  std::string name = "";
  // Wall-clock time the event started.
  absl::Time timestamp = absl::UnixEpoch();
  // Duration from event start to event end.
  absl::Duration duration = absl::ZeroDuration();
  // Whether the PJRT executable was already in the compilation cache.
  bool cache_hit = false;
  // FX-style ATen graph with inline source locations. Populated on cache
  // miss; empty on cache hit.
  std::string aten_graph_payload = "";
  // StableHLO MLIR text with #loc debug info. Populated on cache miss;
  // empty on cache hit.
  std::string mlir_payload = "";
  // Why this materialization was triggered.
  MaterializationReason reason = MaterializationReason::kUnknown;
  // Pre-serialized JSON for the chromium_event payload.
  std::string chromium_payload = "";
  // True if compilation failed.
  bool compile_failed = false;
};

// A global thread-safe buffer for collecting structured log events.
class StructuredLogBuffer {
 public:
  // Returns the global singleton instance.
  static StructuredLogBuffer& GetInstance();

  // Returns whether logging is currently enabled.
  bool enabled() const { return enabled_; }

  // Pushes a new log event into the buffer if enabled.
  // If the buffer is full, the oldest event is evicted.
  // We take unique_ptr to avoid expensive move operations on
  // StructuredLogEvent.
  void Push(absl_nonnull std::unique_ptr<StructuredLogEvent> event);

  struct DrainResult {
    std::deque<absl_nonnull std::unique_ptr<StructuredLogEvent>> events;
    int32_t dropped_since_last_drain;
  };

  // Thread-safe drain: returns all buffered events plus the count of
  // events dropped due to overflow since the previous Drain(), then
  // clears both.
  DrainResult Drain();

  // Overrides the enable state for testing.
  void SetEnabledForTest(bool enabled) { enabled_ = enabled; }

 private:
  StructuredLogBuffer();
  friend class absl::NoDestructor<StructuredLogBuffer>;

  std::atomic<bool> enabled_;
  absl::Mutex mu_;
  std::deque<absl_nonnull std::unique_ptr<StructuredLogEvent>> events_
      ABSL_GUARDED_BY(mu_);
  int32_t dropped_since_last_drain_ ABSL_GUARDED_BY(mu_) = 0;
  static constexpr int32_t kMaxEvents = 10000;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_STRUCTURED_LOG_BUFFER_H_
