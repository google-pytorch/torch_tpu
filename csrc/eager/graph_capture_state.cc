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

#include <memory>
#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"

namespace torch_tpu {

void GraphCaptureState::RecordError(absl::Status error) {
  absl::MutexLock lock(mu_);
  // Keeps the first error.
  error_.Update(std::move(error));
}

absl::Status GraphCaptureState::error() const {
  absl::MutexLock lock(mu_);
  return error_;
}

void PushCurrentThreadGraphCapture(
    absl_nonnull std::shared_ptr<GraphCaptureState> state) {
  PushContextState<GraphCaptureContextState>(std::move(state));
}

void PopCurrentThreadGraphCapture() {
  PopContextState<GraphCaptureContextState>();
}

GraphCaptureState* absl_nullable GetCurrentThreadGraphCapture() {
  std::optional<GraphCaptureContextState> state =
      GetContextState<GraphCaptureContextState>();
  return state.has_value() ? state->get() : nullptr;
}

bool IsCurrentThreadCapturingGraph() {
  const GraphCaptureState* const state = GetCurrentThreadGraphCapture();
  return state != nullptr && state->is_active();
}

absl::Status InvalidateCurrentThreadGraphCapture(absl::Status error) {
  if (GraphCaptureState* const state = GetCurrentThreadGraphCapture();
      state != nullptr && state->is_active()) {
    state->RecordError(error);
  }
  return error;
}

absl::Status GetCurrentThreadGraphCaptureError() {
  const GraphCaptureState* const state = GetCurrentThreadGraphCapture();
  return state != nullptr ? state->error() : absl::OkStatus();
}

}  // namespace torch_tpu
