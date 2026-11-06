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

#ifndef TORCH_TPU_CSRC_COMMON_FLAGS_H_
#define TORCH_TPU_CSRC_COMMON_FLAGS_H_

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/flags/commandlineflag.h"
#include "absl/flags/flag.h"

namespace torch_tpu {

namespace internal {

// Records that GetFlagOnce() has memoized the value of the flag identified by
// `flag_handle`. Thread-safe.
void MarkFlagMemoized(const absl::CommandLineFlag& flag_handle);

// Crashes if GetFlagOnce() has memoized the value of the flag identified by
// `flag_handle`. Thread-safe.
void CheckFlagNotMemoized(const absl::CommandLineFlag& flag_handle);

}  // namespace internal

// Returns the value of the flag at the time of the first call to this function
// for the given flag, and caches the result for subsequent calls. Thread-safe.
//
// Since the value is memoized, the flag must not be changed after the first
// call to this function: SafeSetFlag() crashes if it's called on a flag that
// has been read via GetFlagOnce(). If a flag needs to be changed at runtime,
// read it with absl::GetFlag() instead.
//
// Example:
//   ABSL_FLAG(int, torch_tpu_internal_foo, 10, "Test flag");
//
//   const int foo = GetFlagOnce<int, &FLAGS_torch_tpu_internal_foo>();
template <typename T, const absl::Flag<T>* absl_nonnull flag_ptr>
const T& GetFlagOnce() {
  static const absl::NoDestructor<T> value([] {
    internal::MarkFlagMemoized(absl::GetFlagReflectionHandle(*flag_ptr));
    return absl::GetFlag(*flag_ptr);  // GET_FLAG_OK=implementing GetFlagOnce.
  }());
  return *value;
}

// Like absl::SetFlag(), but crashes if GetFlagOnce() has already been called on
// the flag, as the new value would be silently ignored by GetFlagOnce(). Use
// this instead of absl::SetFlag().
//
// Example:
//   SafeSetFlag(&FLAGS_torch_tpu_internal_foo, 20);
template <typename T, typename V>
void SafeSetFlag(absl::Flag<T>* absl_nonnull flag, const V& value) {
  internal::CheckFlagNotMemoized(absl::GetFlagReflectionHandle(*flag));
  absl::SetFlag(flag, value);  // SET_FLAG_OK=implementing SafeSetFlag.
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_FLAGS_H_
