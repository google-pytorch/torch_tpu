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

#include "csrc/common/env_vars.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/synchronization/mutex.h"

namespace torch_tpu::internal {
namespace {

ABSL_CONST_INIT absl::Mutex memoized_env_vars_mutex(absl::kConstInit);

// The names of the env vars whose values have been memoized by GetEnvOnce().
// The names are keyed by content rather than by address, as the kFooEnvVar
// arrays may have different addresses in different shared libraries. They
// point to the static kFooEnvVar arrays, so they never dangle.
absl::flat_hash_set<std::string_view>& MemoizedEnvVars()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(memoized_env_vars_mutex) {
  static absl::NoDestructor<absl::flat_hash_set<std::string_view>>
      memoized_env_vars;
  return *memoized_env_vars;
}

}  // namespace

void MarkEnvVarMemoized(std::string_view name) {
  absl::MutexLock lock(memoized_env_vars_mutex);
  MemoizedEnvVars().insert(name);
}

void CheckEnvVarNotMemoized(std::string_view name) {
  absl::MutexLock lock(memoized_env_vars_mutex);
  ABSL_CHECK(!MemoizedEnvVars().contains(name))  // CRASH_OK
      << "SafeSetEnv() or SafeUnsetEnv() called on " << name
      << " after its value was memoized by GetEnvOnce(). The new value would "
         "be ignored by GetEnvOnce(). If TorchTPU needs to change the env var "
         "at runtime, read it with std::getenv instead of GetEnvOnce().";
}

}  // namespace torch_tpu::internal

namespace torch_tpu {

// Replaces the "${RANK}" placeholder in the given environment variable with the
// given rank.
//
// If the environment variable is not set or does not contain the placeholder,
// this function has no effect.
template <const char* env_var_name>
static void ExpandRankInEnvVar(const int rank) {
  const char* const env_val =
      std::getenv(  // GETENV_OK=need to overwrite the env var below.
          env_var_name);
  if (env_val == nullptr || !absl::StrContains(env_val, "${RANK}")) return;
  std::string new_val(env_val);
  absl::StrReplaceAll({{"${RANK}", absl::StrCat(rank)}}, &new_val);
  SafeSetEnv<env_var_name>(new_val);
}

void EnsureRankIsExpandedInEnvVars() {
  static const bool done = [] {
    const int rank = GetIntegerEnvOnce<int, kRankEnvVar>().value_or(0);
    ExpandRankInEnvVar<kXlaFlagsEnvVar>(rank);
    ExpandRankInEnvVar<kLibtpuInitArgsEnvVar>(rank);
    return true;
  }();
  static_cast<void>(done);  // VOID_CAST_OK=unused variable, ensures the
                            // lambda is executed.
}

const std::optional<std::string>& ExpandAndGetXlaFlagsEnvVarOnce() {
  // XLA_FLAGS may reference "${RANK}", so we need to expand it before
  // accessing.
  EnsureRankIsExpandedInEnvVars();
  return GetEnvOnce<kXlaFlagsEnvVar>();
}

const std::optional<std::string>& ExpandAndGetLibtpuInitArgsEnvVarOnce() {
  // LIBTPU_INIT may reference "${RANK}", so we need to expand it before
  // accessing.
  EnsureRankIsExpandedInEnvVars();
  return GetEnvOnce<kLibtpuInitArgsEnvVar>();
}

}  // namespace torch_tpu
