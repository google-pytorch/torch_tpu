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

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/synchronization/mutex.h"
#include "csrc/common/symbol_stage.h"
#include "csrc/common/warnings.h"

namespace torch_tpu::internal {

ABSL_CONST_INIT absl::Mutex env_vars_mutex(absl::kConstInit);

namespace {

// The entries of the env vars read by the GetEnvOnce() family, keyed by name.
// The names are keyed by content rather than by address, as the kFooEnvVar
// arrays may have different addresses in different shared libraries. Entries
// are never erased, and node_hash_map never moves them, so references to them
// stay valid.
absl::node_hash_map<std::string, EnvEntry>& EnvEntries()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(env_vars_mutex) {
  static absl::NoDestructor<absl::node_hash_map<std::string, EnvEntry>>
      env_entries;
  return *env_entries;
}

// Crashes if GetEnvOnce() has memoized the value of the env var with the given
// name. A valid EnvEntry doubles as the "memoized" marker, so
// ResetEnvCacheForTesting() allows changing the env var until it is read again.
void CheckEnvVarNotMemoized(const char* name)
    ABSL_LOCKS_EXCLUDED(env_vars_mutex) {
  absl::MutexLock lock(env_vars_mutex);
  const auto it = EnvEntries().find(name);
  ABSL_CHECK(it == EnvEntries().end() ||  // CRASH_OK
             it->second.Get() == nullptr)
      << "SafeSetEnv() or SafeUnsetEnv() called on " << name
      << " after its value was memoized by GetEnvOnce(). The new value would "
         "be ignored by GetEnvOnce(). If TorchTPU needs to change the env var "
         "at runtime, read it with std::getenv instead of GetEnvOnce().";
}

// Reads the env var with the given name for one of the Refresh*Env()
// functions below.
std::optional<std::string> ReadEnv(const char* name)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(env_vars_mutex) {
  const char* const env_var =  //
      std::getenv(name);       // GETENV_OK=implementing GetEnvOnce().
  if (env_var == nullptr) return std::nullopt;
  return std::string(env_var);
}

// Warns if `value`, the value of the env var with the given name, is a
// non-empty string and `stage` is experimental or deprecated.
void WarnIfExperimentalOrDeprecated(const char* name,
                                    const std::optional<std::string>& value,
                                    const SymbolStage stage) {
  if (!value.has_value() || value->empty()) return;
  if (stage.is_experimental()) {
    TT_WARN(WarningCategory::kExperimentalApi, "the ", name,
            " environment variable is an experimental feature and may "
            "change or be removed without notice.");
  } else if (stage.is_deprecated()) {
    const auto deprecated_since = stage.version();
    TT_WARN(WarningCategory::kDeprecatedApi, "the ", name,
            " environment variable is deprecated since TorchTPU v",
            deprecated_since, " and will be removed in a future release.");
  }
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

EnvEntry& GetEnvEntry(const char* name) {
  absl::MutexLock lock(env_vars_mutex);
  return EnvEntries()[name];
}

// The Refresh*Env() functions below warn after releasing env_vars_mutex, as
// warning may run arbitrary code, including Python code that reads env vars.
// Only the thread that refreshes a value warns, so each warning is emitted
// once per env var (per ResetEnvCacheForTesting() call), like TT_WARN_ONCE() in
// a function template would.

const std::optional<std::string>& RefreshStringEnv(EnvEntry& entry,
                                                   const char* name,
                                                   const SymbolStage stage) {
  std::optional<std::string> value;
  {
    absl::MutexLock lock(env_vars_mutex);
    if (const EnvEntry::Value* const set_value = entry.Get()) {
      return std::get<std::optional<std::string>>(*set_value);
    }
    value = ReadEnv(name);
    entry.SetLocked(value);
  }
  WarnIfExperimentalOrDeprecated(name, value, stage);
  return std::get<std::optional<std::string>>(*entry.Get());
}

std::optional<bool> RefreshBooleanEnv(EnvEntry& entry, const char* name,
                                      const SymbolStage stage) {
  std::optional<std::string> value;
  std::optional<bool> parsed;
  {
    absl::MutexLock lock(env_vars_mutex);
    if (const EnvEntry::Value* const set_value = entry.Get()) {
      return std::get<std::optional<bool>>(*set_value);
    }
    value = ReadEnv(name);
    if (bool result = false;
        value.has_value() && absl::SimpleAtob(*value, &result)) {
      parsed = result;
    }
    entry.SetLocked(parsed);
  }
  WarnIfExperimentalOrDeprecated(name, value, stage);
  if (value.has_value() && !parsed.has_value()) {
    TT_WARN(WarningCategory::kCorrectness, "expected environment variable ",
            name, " to be a Boolean (e.g., '1', '0', 'true', 'false'), got '",
            *value, "'");
  }
  return parsed;
}

std::optional<int64_t> RefreshIntegerEnv(EnvEntry& entry, const char* name,
                                         const SymbolStage stage,
                                         const EnvVarType type) {
  std::optional<std::string> value;
  std::optional<int64_t> parsed;
  {
    absl::MutexLock lock(env_vars_mutex);
    if (const EnvEntry::Value* const set_value = entry.Get()) {
      return std::get<std::optional<int64_t>>(*set_value);
    }
    value = ReadEnv(name);
    if (value.has_value()) {
      if (type == EnvVarType::kInt) {
        if (int result = 0; absl::SimpleAtoi(*value, &result)) parsed = result;
      } else {
        ABSL_DCHECK(type == EnvVarType::kInt64);
        if (int64_t result = 0; absl::SimpleAtoi(*value, &result)) {
          parsed = result;
        }
      }
    }
    entry.SetLocked(parsed);
  }
  WarnIfExperimentalOrDeprecated(name, value, stage);
  if (value.has_value() && !parsed.has_value()) {
    TT_WARN(WarningCategory::kCorrectness, "expected environment variable ",
            name, " to be an integer, got '", *value, "'");
  }
  return parsed;
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

// Declared in env_vars_testing.h.
void ResetEnvCacheForTesting() {
  absl::MutexLock lock(internal::env_vars_mutex);
  for (auto& [name, entry] : internal::EnvEntries()) {
    entry.InvalidateLocked();
  }
}

}  // namespace torch_tpu
