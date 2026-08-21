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

#ifndef TORCH_TPU_COMMON_COMPILATION_CACHE_UTILS_H_
#define TORCH_TPU_COMMON_COMPILATION_CACHE_UTILS_H_

#include <ostream>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/compilation.h"
#include "torch_tpu/csrc/common/fingerprint_utils.h"

namespace torch_tpu {

// The tier of the cache where an executable comes from.
enum class CacheTier {
  kUnknown,
  kTier1,
  kTier2,
  kTier3,
};

// Formats a cache tier as a human-readable string.
std::ostream& operator<<(std::ostream& os, CacheTier tier);
template <typename Sink>
void AbslStringify(Sink& sink, const CacheTier tier) {
  absl::Format(&sink, "%s", absl::FormatStreamed(tier));
}

// Returns the fingerprint of the TorchTPU binary version. The intent is to
// detect potential change in TorchTPU's behavior: whenever this changes, we
// invalidate the tier-2 and tier-3 caches to be safe.
//
// This function is memoized so that it's cheap to call this multiple times.
[[nodiscard]] FingerprintType GetTorchTpuBinaryFingerprint();

// Creates the directory recursively as needed, and sets the permissions to
// 0777 (rwxrwxrwx).
absl::Status EnsureDirExistsRecursively(const std::string& path);

// Loads a serialized executable read from the given cache.
absl::StatusOr<SharedLoadedExecutableWithMetadata> LoadSerializedExecutable(
    CacheTier tier, CompilationCacheKey key, std::string_view data);

// Writes a compiled executable to the given tier-2 or tier-3 cache file. This
// is best effort.
//
// Calling this concurrently with the same key is safe (i.e. won't produce a
// corrupted file) - the function first writes to a unique temp file in the same
// directory, and then atomically renames it to the final cache file path.
absl::Status AtomicWriteToCacheFile(
    const std::string& cache_entry_path,
    const SharedLoadedExecutableWithMetadata& executable);

// Returns the root directory of the tier-3 compilation cache, as set by the
// TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT environment variable.
//
// This function is memoized so that it's cheap to call this multiple times.
[[nodiscard]] const std::string& GetTier3CacheRootDir();

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_COMPILATION_CACHE_UTILS_H_
