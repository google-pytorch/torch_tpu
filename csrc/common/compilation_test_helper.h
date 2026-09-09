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

#ifndef TORCH_TPU_CSRC_COMMON_COMPILATION_TEST_HELPER_H_
#define TORCH_TPU_CSRC_COMMON_COMPILATION_TEST_HELPER_H_

#include <utility>

#include "absl/log/absl_check.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"

namespace torch_tpu {

// An RAII wrapper for managing compiler option overrides within a scope.
//
// It is particularly useful in tests to ensure that compiler option overrides
// are confined to a specific scope, so that the context state is reset when the
// scope is exited, maintaining clean states between tests.
class ScopedCompilerOptionOverrides {
 public:
  explicit ScopedCompilerOptionOverrides(CompilerOptionOverrides overrides) {
    ABSL_CHECK_OK(  // CRASH_OK
        PushCompilerOptionOverrides(std::move(overrides)));
  }

  ~ScopedCompilerOptionOverrides() { PopCompilerOptionOverrides(); }
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_COMPILATION_TEST_HELPER_H_
