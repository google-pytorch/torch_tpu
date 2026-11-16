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

#include "csrc/common/flags.h"

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/commandlineflag.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"

namespace torch_tpu::internal {
namespace {

ABSL_CONST_INIT absl::Mutex memoized_flags_mutex(absl::kConstInit);

// The flags whose values have been memoized by GetFlagOnce().
absl::flat_hash_set<const absl::CommandLineFlag*>& MemoizedFlags()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(memoized_flags_mutex) {
  static absl::NoDestructor<absl::flat_hash_set<const absl::CommandLineFlag*>>
      memoized_flags;
  return *memoized_flags;
}

}  // namespace

void MarkFlagMemoized(const absl::CommandLineFlag& flag_handle) {
  absl::MutexLock lock(memoized_flags_mutex);
  MemoizedFlags().insert(&flag_handle);
}

void CheckFlagNotMemoized(const absl::CommandLineFlag& flag_handle) {
  absl::MutexLock lock(memoized_flags_mutex);
  ABSL_CHECK(!MemoizedFlags().contains(&flag_handle))  // CRASH_OK
      << "SafeSetFlag() called on --" << flag_handle.Name()
      << " after its value was memoized by GetFlagOnce(). The new value would "
         "be ignored by GetFlagOnce(). If the flag needs to be changed at "
         "runtime, read it with absl::GetFlag instead of GetFlagOnce().";
}

}  // namespace torch_tpu::internal
