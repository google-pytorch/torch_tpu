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

#include <atomic>
#include <cstdlib>
#include <optional>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

namespace torch_tpu {
namespace {

// Whether ResetEnvCacheForTesting() has been called.
std::atomic<bool> env_cache_was_reset_for_testing = false;

// The environment variable values memoized since the last
// ResetEnvCacheForTesting() call, keyed by name.
class EnvCacheForTesting {
 public:
  void Clear() {
    absl::MutexLock lock(mutex_);
    values_.clear();
  }

  std::optional<std::string> Get(const char* name) {
    absl::MutexLock lock(mutex_);
    const auto [it, inserted] = values_.try_emplace(name);
    if (inserted) {
      const char* const value =  //
          std::getenv(name);     // GETENV_OK=testing-only reread of env vars.
      if (value != nullptr) it->second = value;
    }
    return it->second;
  }

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, std::optional<std::string>> values_
      ABSL_GUARDED_BY(mutex_);
};

EnvCacheForTesting& GetEnvCacheForTesting() {
  static absl::NoDestructor<EnvCacheForTesting> cache;
  return *cache;
}

}  // namespace

void ResetEnvCacheForTesting() {
  GetEnvCacheForTesting().Clear();
  env_cache_was_reset_for_testing.store(true, std::memory_order_release);
}

bool EnvCacheWasResetForTesting() {
  return env_cache_was_reset_for_testing.load(std::memory_order_acquire);
}

std::optional<std::string> GetEnvSinceResetForTesting(const char* name) {
  return GetEnvCacheForTesting().Get(name);
}

}  // namespace torch_tpu
