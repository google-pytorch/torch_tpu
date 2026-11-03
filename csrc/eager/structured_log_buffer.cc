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

#include "csrc/eager/structured_log_buffer.h"

#include <deque>
#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/synchronization/mutex.h"
#include "csrc/common/env_vars.h"

namespace torch_tpu {

StructuredLogBuffer::StructuredLogBuffer()
    : enabled_(GetEnvOnce<kTorchTraceEnvVar>().has_value()) {}

StructuredLogBuffer& StructuredLogBuffer::GetInstance() {
  static absl::NoDestructor<StructuredLogBuffer> instance;
  return *instance;
}

void StructuredLogBuffer::Push(
    absl_nonnull std::unique_ptr<StructuredLogEvent> event) {
  if (!enabled_) return;
  absl::MutexLock lock(mu_);
  if (events_.size() >= kMaxEvents) {
    // We evict the oldest event to prevent memory bloat.
    // TODO - b/491210752: Replace drop-on-overflow with a C++-triggered drain
    // or flag configurable threshold.
    events_.pop_front();
    ++dropped_since_last_drain_;
  }
  events_.push_back(std::move(event));
}

StructuredLogBuffer::DrainResult StructuredLogBuffer::Drain() {
  if (!enabled_) return {};

  DrainResult result;
  {
    absl::MutexLock lock(mu_);
    result = {.events = std::move(events_),
              .dropped_since_last_drain = dropped_since_last_drain_};
    dropped_since_last_drain_ = 0;
  }
  // No other thread can reach the drained events, so resolve the lazy payloads
  // outside the lock, as they may wait for a compilation.
  for (auto& event : result.events) {
    if (event->lazy_mlir_payload) {
      event->mlir_payload = std::move(event->lazy_mlir_payload)();
      event->lazy_mlir_payload = nullptr;
    }
  }
  return result;
}

}  // namespace torch_tpu
