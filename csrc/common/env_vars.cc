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
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/synchronization/mutex.h"
#include "csrc/common/symbol_stage.h"
#include "csrc/common/warnings.h"

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

// Records that GetEnvOnce() has memoized the value of the env var with the
// given name. Thread-safe.
void MarkEnvVarMemoized(std::string_view name) {
  absl::MutexLock lock(memoized_env_vars_mutex);
  MemoizedEnvVars().insert(name);
}

// Crashes if GetEnvOnce() has memoized the value of the env var with the given
// name. Thread-safe.
void CheckEnvVarNotMemoized(std::string_view name) {
  absl::MutexLock lock(memoized_env_vars_mutex);
  ABSL_CHECK(!MemoizedEnvVars().contains(name))  // CRASH_OK
      << "SafeSetEnv() or SafeUnsetEnv() called on " << name
      << " after its value was memoized by GetEnvOnce(). The new value would "
         "be ignored by GetEnvOnce(). If TorchTPU needs to change the env var "
         "at runtime, read it with std::getenv instead of GetEnvOnce().";
}

}  // namespace

void SafeSetEnvImpl(const char* name, const std::string& value) {
  CheckEnvVarNotMemoized(name);
  setenv(name, value.c_str(),  // SETENV_OK=implementing SafeSetEnv().
         /*overwrite=*/1);
}

void SafeUnsetEnvImpl(const char* name) {
  CheckEnvVarNotMemoized(name);
  unsetenv(name);  // SETENV_OK=implementing SafeUnsetEnv().
}

// GetEnvOnce() calls this at most once per env var, so the TT_WARN() calls
// below warn at most once per env var, like TT_WARN_ONCE() in a function
// template would.
std::optional<std::string> ReadEnvForGetEnvOnce(
    const char* name, const std::optional<SymbolStage> maybe_stage) {
  MarkEnvVarMemoized(name);
  const char* const env_var =  //
      std::getenv(name);       // GETENV_OK=implementing GetEnvOnce().
  if (env_var == nullptr) return std::nullopt;
  if (env_var[0] == '\0') return std::string();

  // The env var is set to a non-empty string. Warn the user if the env var is
  // experimental or deprecated.
  if (maybe_stage.has_value()) {
    if (maybe_stage->is_experimental()) {
      TT_WARN(WarningCategory::kExperimentalApi, "the ", name,
              " environment variable is an experimental feature and may "
              "change or be removed without notice.");
    } else if (maybe_stage->is_deprecated()) {
      const auto deprecated_since = maybe_stage->version();
      TT_WARN(WarningCategory::kDeprecatedApi, "the ", name,
              " environment variable is deprecated since TorchTPU v",
              deprecated_since, " and will be removed in a future release.");
    }
  }
  return std::string(env_var);
}

// Like ReadEnvForGetEnvOnce(), called at most once per env var by
// GetBooleanEnvOnce().
std::optional<bool> ParseBooleanEnv(const char* name,
                                    const std::optional<std::string>& env_var) {
  if (!env_var.has_value()) {
    return std::nullopt;
  }
  bool result = false;
  if (absl::SimpleAtob(*env_var, &result)) {
    return result;
  }
  TT_WARN(WarningCategory::kCorrectness, "expected environment variable ", name,
          " to be a Boolean (e.g., '1', '0', 'true', 'false'), got '", *env_var,
          "'");
  return std::nullopt;
}

// Like ReadEnvForGetEnvOnce(), called at most once per env var by
// GetIntegerEnvOnce().
void WarnInvalidIntegerEnv(const char* name, const std::string_view value) {
  TT_WARN(WarningCategory::kCorrectness, "expected environment variable ", name,
          " to be an integer, got '", value, "'");
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
    const int rank = GetIntegerEnvOnce<kRankEnvVar>().value_or(0);
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
