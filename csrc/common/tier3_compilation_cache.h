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

#ifndef TORCH_TPU_CSRC_COMMON_TIER3_COMPILATION_CACHE_H_
#define TORCH_TPU_CSRC_COMMON_TIER3_COMPILATION_CACHE_H_

#include <string>

#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"

namespace torch_tpu {

// Returns true if the tier-3 compilation cache is enabled for this process.
// This function is memoized, so it's cheap to call this multiple times.
[[nodiscard]] bool UsesTier3CompilationCache();

// Returns true if a local compilation backup task should be scheduled in
// parallel with a tier-3 cache read.
// This function is memoized, so it's cheap to call this multiple times.
[[nodiscard]] bool UsesLocalBackupTaskForTier3Read();

// Returns the path to the tier-3 cache file for the given key.
[[nodiscard]] std::string GetTier3CacheEntryPath(CompilationCacheKey key);

// Tries to get a loaded executable from the tier-3 compilation cache.
absl::StatusOr<SharedLoadedExecutableWithMetadata> GetFromTier3Cache(
    CompilationCacheKey key);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_TIER3_COMPILATION_CACHE_H_
