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

#include "csrc/internal/profiler/xprof_callback_handler.h"

#include <memory>
#include <string>
#include <utility>

#include "ATen/record_function.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "torch/csrc/profiler/orchestration/observer.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace torch_tpu {

namespace {

// Observer context for RecordFunction callbacks.
// Each RecordFunction push creates a TraceMe instance via OnFunctionEnter,
// and OnFunctionExit destroys it, effectively measuring the duration of the
// RecordFunction.
class XProfObserverContext : public at::ObserverContext {
 public:
  explicit XProfObserverContext(tsl::profiler::TraceMe tm)
      : traceme_(std::move(tm)) {}

 private:
  tsl::profiler::TraceMe traceme_;
};

// Returns true if profiling is enabled for the current thread, otherwise
// returns false.
//
// It accommodates the following profiling APIs:
//   1. TSL profiler API `tsl::profiler::ProfilerSession::Create`, and
//   2. native PyTorch profiler API `torch.profiler.profile`, and
//   3. custom TorchTPU profiler API `torch_tpu._internal.profiler.profile`.
//
// TODO(b/509670300): simplify it when the native PyTorch profiler API is fully
// integrated and available.
[[nodiscard]] bool IsProfilerEnabled() {
  // Enable profiling by default if any profiling session is active via global
  // TSL or native PyTorch API. It ensures that PyTorch events are captured even
  // if the user is not using the custom TorchTPU Python context manager.
  const auto get_default_state = [] {
    return (tsl::profiler::TraceMe::Active() ||
            torch::profiler::impl::profilerEnabled())
               ? ProfilerStatus::kEnabled
               : ProfilerStatus::kDisabled;
  };

  // Override the default state with the current thread-local context state.
  return GetContextState<ProfileContextState>(get_default_state) ==
         ProfilerStatus::kEnabled;
}

// The callback function that is called when a RecordFunction is pushed.
// It creates a TraceMe instance and returns a XProfObserverContext.
// Returns nullptr if there is no active XProf session, allowing PyTorch to
// bypass state tracking and avoid unnecessary heap allocations.
absl_nullable std::unique_ptr<at::ObserverContext> OnFunctionEnter(
    const at::RecordFunction& fn) {
  // Fast-path check: if
  //   1. no XProf session (local or remote) is active, or
  //   2. profiling is not enabled
  // return immediately to avoid the heap allocation of `XProfObserverContext`.
  if (!tsl::profiler::TraceMe::Active() || !IsProfilerEnabled()) {
    return nullptr;
  }
  return std::make_unique<XProfObserverContext>(tsl::profiler::TraceMe([&] {
    return tsl::profiler::TraceMeEncode(
        fn.name(), {{"pt_correlation_id", std::to_string(fn.handle())}});
  }));
}

// The callback function that is called when a RecordFunction is popped.
void OnFunctionExit(const at::RecordFunction& /*fn*/,
                    at::ObserverContext* /*context_ptr*/) {
  // TraceMe destructor handles the end event.
}

}  // namespace

XProfCallbackHandler::XProfCallbackHandler()
    : handle_(at::addGlobalCallback(
          // It is safe for OnFunctionEnter to return nullptr. PyTorch handles
          // null contexts natively and passes them to OnFunctionExit, which
          // safely ignores them.
          at::RecordFunctionCallback(OnFunctionEnter, OnFunctionExit)
              .needsIds(true))) {}

// Register the callback handler explicitly to avoid SIOF.
void XProfCallbackHandler::Register() {
  static absl::NoDestructor<XProfCallbackHandler> global_handler;
}

}  // namespace torch_tpu
