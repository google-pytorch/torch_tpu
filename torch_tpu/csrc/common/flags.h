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

#ifndef TORCH_TPU_COMMON_FLAGS_H_
#define TORCH_TPU_COMMON_FLAGS_H_

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/flags/flag.h"

namespace torch_tpu {

// Returns the value of the flag at the time of the first call to this function
// for the given flag, and caches the result for subsequent calls. Hence
// absl::SetFlag() calls after the first call to this function won't affect the
// returned value of this function. Thread-safe.
//
// Example:
//   ABSL_FLAG(int, torch_tpu_internal_foo, 10, "Test flag");
//
//   const int foo = GetFlagOnce<int, &FLAGS_torch_tpu_internal_foo>();
template <typename T, const absl::Flag<T>* absl_nonnull flag_ptr>
const T& GetFlagOnce() {
  static const absl::NoDestructor<T>    //
      value(absl::GetFlag(*flag_ptr));  // GET_FLAG_OK=implementing GetFlagOnce.
  return *value;
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_FLAGS_H_
