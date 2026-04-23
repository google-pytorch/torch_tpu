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

#include "torch_tpu/eager/structured_log_buffer.h"

#include <deque>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/common/env_vars.h"

namespace torch_tpu {

StructuredLogBuffer::StructuredLogBuffer()
    : enabled_(GetEnvOnce<kTorchTraceEnvVar>().has_value()) {}

StructuredLogBuffer& StructuredLogBuffer::Get() {
  static absl::NoDestructor<StructuredLogBuffer> instance;
  return *instance;
}

void StructuredLogBuffer::Push(EagerEvent event) {
  if (!enabled_) return;
  absl::MutexLock lock(mu_);
  if (events_.size() >= kMaxEvents) {
    events_.pop_front();
  }
  events_.push_back(std::move(event));
}

std::deque<EagerEvent> StructuredLogBuffer::Drain() {
  absl::MutexLock lock(mu_);
  return std::move(events_);
}

}  // namespace torch_tpu
