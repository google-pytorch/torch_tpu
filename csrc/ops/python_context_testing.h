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

// Test-only utilities for python_context.h.

#ifndef TORCH_TPU_CSRC_OPS_PYTHON_CONTEXT_TESTING_H_
#define TORCH_TPU_CSRC_OPS_PYTHON_CONTEXT_TESTING_H_

#include <utility>

#include "csrc/ops/python_context.h"

namespace torch_tpu {

// Overrides the traceback of all currently alive ScopedPythonContextCapturer
// contexts for the current thread during the lifetime of this object, and
// restores the previous traceback when it is destroyed.
//
// REQUIRES: A ScopedPythonContextCapturer is alive for the current thread and
// outlives this object.
class ScopedTracebackForTesting {
 public:
  explicit ScopedTracebackForTesting(MaybeSharedTraceback traceback)
      : previous_traceback_(std::exchange(
            ScopedPythonContextCapturer::traceback_, std::move(traceback))) {}

  // This class is neither copyable nor movable.
  ScopedTracebackForTesting(const ScopedTracebackForTesting&) = delete;
  ScopedTracebackForTesting& operator=(const ScopedTracebackForTesting&) =
      delete;
  ScopedTracebackForTesting(ScopedTracebackForTesting&&) = delete;
  ScopedTracebackForTesting& operator=(ScopedTracebackForTesting&&) = delete;

  ~ScopedTracebackForTesting() {
    ScopedPythonContextCapturer::traceback_ = std::move(previous_traceback_);
  }

 private:
  MaybeSharedTraceback previous_traceback_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_PYTHON_CONTEXT_TESTING_H_
