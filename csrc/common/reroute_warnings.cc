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

#include "csrc/common/reroute_warnings.h"

#include <atomic>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "c10/util/Exception.h"

namespace torch_tpu {
namespace {

// Buffers the warnings of every thread installing it as warning handler.
class DelayedWarningHandler final : public c10::WarningHandler {
 public:
  void process(const c10::Warning& warning) override {
    absl::MutexLock lock(mu_);
    warnings_.push_back(warning);
    has_warnings_.store(true, std::memory_order_release);
  }

  // Returns whether there may be buffered warnings, without locking.
  [[nodiscard]] bool has_warnings() const {
    return has_warnings_.load(std::memory_order_acquire);
  }

  // Returns all buffered warnings and empties the buffer.
  [[nodiscard]] std::vector<c10::Warning> GetAll() {
    absl::MutexLock lock(mu_);
    has_warnings_.store(false, std::memory_order_relaxed);
    return std::exchange(warnings_, {});
  }

 private:
  // Mirroring emptiness check so `SurfaceReroutedWarnings` can skip locking in
  // common cases.
  //
  // Invariant: `has_warnings_ == !warnings_.empty()` whenever the mutex is not
  // held, since both are only updated together under the same lock.
  std::atomic<bool> has_warnings_ = false;

  std::vector<c10::Warning> warnings_ ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;  // Protects `warnings_`.
};

// Returns the process-wide rerouted warning handler singleton.
[[nodiscard]] DelayedWarningHandler& GetDelayedWarningHandler() {
  static absl::NoDestructor<DelayedWarningHandler> handler;
  return *handler;
}

}  // namespace

void RouteWarningsOnThisThreadToDispatchThread() {
  c10::WarningUtils::set_warning_handler(&GetDelayedWarningHandler());
}

void SurfaceReroutedWarnings() {
  DelayedWarningHandler& handler = GetDelayedWarningHandler();
  // c10 warning handlers are thread-local, so `handler` is installed only on
  // threads that called `RouteWarningsOnThisThreadToDispatchThread`.
  const bool is_background_thread =
      c10::WarningUtils::get_warning_handler() == &handler;
  ABSL_CHECK(!is_background_thread)  // CRASH_OK
      << "Rerouted warnings can only be surfaced in dispatch threads as "
         "surfacing them in a background thread would just reroute them "
         "again.";

  if (handler.has_warnings()) {
    for (const c10::Warning& warning : handler.GetAll()) {
      c10::warn(warning);
    }
  }
}

}  // namespace torch_tpu
