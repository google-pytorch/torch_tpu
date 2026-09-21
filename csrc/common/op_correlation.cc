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

#include "csrc/common/op_correlation.h"

#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace torch_tpu {
namespace {

// The correlation IDs of the ATen ops currently being dispatched on this
// thread, outermost first. Written from the RecordFunction enter/exit path.
thread_local std::vector<CorrelationId>  // CPP_THREAD_LOCAL_OK=dispatch
    t_op_correlation_ids;  // NOLINT(google3-runtime-global-variables)

// The correlation IDs of the ops in the program currently being executed on
// this thread. Written from the execution thread, which is a *different* thread
// from the one that writes t_op_correlation_ids -- these are two separate
// mechanisms that happen to share a file, not one stack.
thread_local std::vector<CorrelationId>  // CPP_THREAD_LOCAL_OK=launch scope
    t_launch_correlation_ids;  // NOLINT(google3-runtime-global-variables)

// Returns a reference to the process-wide atomic counter tracking how many
// correlation ID pushes have occurred across all threads. Used for diagnostic
// verification that the profiler actually received correlation callbacks.
std::atomic<int64_t>& PushCounter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

}  // namespace

void PushOpCorrelationId(uint64_t correlation_id) {
  t_op_correlation_ids.push_back(NarrowCorrelationId(correlation_id));
  PushCounter().fetch_add(1, std::memory_order_relaxed);
}

void PopOpCorrelationId() {
  // The emptiness check is load-bearing, not defensive boilerplate -- please
  // do not remove it as dead code.
  //
  // libkineto only forwards pushes to sessions that are already in its
  // `sessions_` vector, and a session is added there when the profiler is
  // configured. If a profiling session starts while an ATen op is mid-flight,
  // we receive that op's *pop* without ever having seen its *push*. Popping an
  // empty vector there would be undefined behaviour.
  if (t_op_correlation_ids.empty()) {
    return;
  }
  t_op_correlation_ids.pop_back();
}

CorrelationId GetCurrentOpCorrelationId() {
  if (t_op_correlation_ids.empty()) {
    return kNoCorrelationId;
  }
  return t_op_correlation_ids.back();
}

void ResetOpCorrelationIds() { t_op_correlation_ids.clear(); }

int64_t GetOpCorrelationPushCount() {
  return PushCounter().load(std::memory_order_relaxed);
}

void ResetOpCorrelationPushCount() {
  PushCounter().store(0, std::memory_order_relaxed);
}

absl::Span<const CorrelationId> GetCurrentLaunchCorrelationIds() {
  return absl::MakeConstSpan(t_launch_correlation_ids);
}

ScopedLaunchCorrelation::ScopedLaunchCorrelation(std::vector<CorrelationId> ids)
    : previous_(std::move(t_launch_correlation_ids)) {
  t_launch_correlation_ids = std::move(ids);
}

ScopedLaunchCorrelation::~ScopedLaunchCorrelation() {
  t_launch_correlation_ids = std::move(previous_);
}

}  // namespace torch_tpu
