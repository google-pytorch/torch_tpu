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

#ifndef TORCH_TPU_COMMON_TIER2_COMPILATION_CACHE_H_
#define TORCH_TPU_COMMON_TIER2_COMPILATION_CACHE_H_

#include <ostream>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/common/unique_file_descriptor.h"

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

// Returns true if the tier-2 compilation cache is enabled for this process.
[[nodiscard]] bool UsesTier2CompilationCache();

// Returns the fingerprint of the TorchTPU binary version. The intent is to
// detect potential change in TorchTPU's behavior: whenever this changes, we
// invalidate the tier-2 and tier-3 caches to be safe.
//
// This function is memoized so that it's cheap to call this multiple times.
[[nodiscard]] FingerprintType GetTorchTpuBinaryFingerprint();

// Returns the name of the tier-2 compilation cache as set by the
// TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE environment variable.
//
// This function is memoized so that it's cheap to call this multiple times.
[[nodiscard]] const std::string& GetTier2CacheName();

// Returns the path to the tier-2 compilation cache directory for this process.
// Precondition: `UsesTier2CompilationCache()` is true.
[[nodiscard]] const std::string& GetTier2CompilationCachePath();

// Statistics for a tier-2 cache entry.
struct Tier2CacheEntryStats {
  // When the entry's cache file was last read.
  absl::Time last_read = absl::Now();

  // These two fields are set only when the entry is read from the tier-2 cache
  // by this process. Otherwise they are zero.
  //
  // How long it took to wait for the lock before the read started.
  absl::Duration pre_read_duration;
  // How long it took to read the entry, deserialize it, and load it into a
  // PjRtLoadedExecutable.
  absl::Duration read_duration;

  // These two fields are set only when the entry is compiled and written to the
  // tier-2 cache by this process. Otherwise they are zero.
  //
  // How long it took to wait for the lock before the compilation started.
  absl::Duration pre_compile_duration;
  // How long it took to serialize and write the executable to the tier-2 cache.
  absl::Duration write_duration;
};

// Fetches a compiled executable from the tier-2 cache, where `request_start`
// is when the request for getting the executable from the tier-2 cache was
// made. Populates the output parameter `stats` with the cache entry
// statistics.
absl::StatusOr<SharedLoadedExecutableWithMetadata> GetFromTier2Cache(
    CompilationCacheKey key, absl::Time request_start,
    Tier2CacheEntryStats& stats);

// Returns the path to the tier-2 cache file for the given key.
[[nodiscard]] std::string GetTier2CacheEntryPath(CompilationCacheKey key);

// Returns the path to the tier-2 cache file for the given compact-format key
// (as produced by CompilationCacheKey::CompactFormat()).
[[nodiscard]] std::string GetTier2CacheEntryPath(std::string_view compact_key);

// Creates the directory recursively as needed, and sets the permissions to
// 0777 (rwxrwxrwx).
absl::Status EnsureDirExistsRecursively(const std::string& path);

// Loads a serialized executable read from the given cache.
absl::StatusOr<SharedLoadedExecutableWithMetadata> LoadSerializedExecutable(
    CacheTier tier, CompilationCacheKey key, std::string_view data);

// Same as above, with the key given as an opaque string (e.g. a compact-format
// cache key); the key is only used in error messages.
absl::StatusOr<SharedLoadedExecutableWithMetadata> LoadSerializedExecutable(
    CacheTier tier, std::string_view key_str, std::string_view data);

// Writes a compiled executable to the given tier-2 or tier-3 cache file. This
// is best effort.
//
// Calling this concurrently with the same key is safe (i.e. won't produce a
// corrupted file) - the function first writes to a unique temp file in the same
// directory, and then atomically renames it to the final cache file path.
absl::Status AtomicWriteToCacheFile(
    const std::string& cache_entry_path,
    const SharedLoadedExecutableWithMetadata& executable);

// RAII class for locking and unlocking a tier-2 cache entry.
class Tier2CacheEntryLock {
 public:
  // Acquires the lock for the given key. This is a blocking call.
  explicit Tier2CacheEntryLock(CompilationCacheKey key);

  // Releases the lock for the key specified in the constructor.
  ~Tier2CacheEntryLock();

  // Not copyable or movable.
  Tier2CacheEntryLock(const Tier2CacheEntryLock&) = delete;
  Tier2CacheEntryLock& operator=(const Tier2CacheEntryLock&) = delete;
  Tier2CacheEntryLock(Tier2CacheEntryLock&&) = delete;
  Tier2CacheEntryLock& operator=(Tier2CacheEntryLock&&) = delete;

  // Releases the lock for the key specified in the constructor.
  void Release();

 private:
  const CompilationCacheKey key_;
  UniqueFileDescriptor lock_fd_;  // Descriptor of the lock file.
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_TIER2_COMPILATION_CACHE_H_
