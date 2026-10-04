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

#include "csrc/common/tier3_compilation_cache.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache_utils.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/tier2_compilation_cache.h"
#include "xla/tsl/platform/env.h"

namespace torch_tpu {

constexpr std::string_view kTier3CacheFileExtension = ".bin";

bool UsesTier3CompilationCache() {
  // To use tier-3 cache, we must use tier-2 cache as well.
  static const bool result =
      UsesTier2CompilationCache() && !GetTier3CacheRootDir().empty();
  return result;
}

bool UsesLocalBackupTaskForTier3Read() {
  static const bool result = [] {
    if (!UsesTier3CompilationCache()) {
      ABSL_LOG(INFO) << "Backup compilation for tier-3 cache read is disabled "
                        "as tier-3 cache is disabled.";
      return false;
    }
    const auto backup = GetBooleanEnvOnce<
        kTorchTpuInternalTier3CompilationCacheLocalBackupTaskEnvVar>();
    if (!backup.has_value()) {
      ABSL_LOG(INFO)
          << "Backup compilation for tier-3 cache read is enabled by default.";
      return true;
    }
    const bool enabled = *backup;
    ABSL_LOG(INFO)
        << "Backup compilation for tier-3 cache read is "
        << (enabled ? "enabled" : "disabled")
        << " based on the environment variable "
        << kTorchTpuInternalTier3CompilationCacheLocalBackupTaskEnvVar
        << ", which has value \"" << (*backup ? "true" : "false") << "\".";
    return enabled;
  }();
  return result;
}

// Returns the path to the tier-3 compilation cache directory for the current
// process.
//
// This function is memoized so that it's cheap to call this multiple times.
[[nodiscard]] static const std::string& GetTier3CompilationCachePath() {
  static const absl::NoDestructor<std::string> cache_path([]() {
    ABSL_CHECK(UsesTier3CompilationCache())  // CRASH_OK
        << "Tier-3 compilation cache is not enabled.";
    std::string path = absl::StrCat(
        GetTier3CacheRootDir(), "/",
        absl::Hex(GetTorchTpuBinaryFingerprint(), absl::kZeroPad16));
    ABSL_LOG(INFO) << "Tier-3 compilation cache path for this process: "
                   << path;
    return path;
  }());
  return *cache_path;
}

// Ensures that the tier-3 cache directory exists. This function only does
// the check once.
static void EnsureTier3CacheDirExistsOnceOrDie() {
  static const bool dir_exists = []() {
    const std::string cache_path = GetTier3CompilationCachePath();
    const absl::Status status = EnsureDirExistsRecursively(cache_path);
    ABSL_CHECK(status.ok())  // CRASH_OK
        << "Failed to create tier-3 cache directory: " << cache_path
        << " with error: " << status.message();
    return true;
  }();
  static_cast<void>(dir_exists);  // VOID_CAST_OK=don't care about dummy result.
}

std::string GetTier3CacheEntryPath(const CompilationCacheKey& key) {
  EnsureTier3CacheDirExistsOnceOrDie();
  return absl::StrCat(GetTier3CompilationCachePath(), "/", key.CompactFormat(),
                      kTier3CacheFileExtension);
}

static std::optional<SharedLoadedExecutableWithMetadata>
TryLoadMatchingKeyFromPattern(tsl::Env* env, const CompilationCacheKey& key,
                              const std::string& cache_entry_path,
                              const std::string& pattern) {
  std::vector<std::string> matches;
  if (!env->GetMatchingPaths(pattern, &matches).ok() || matches.empty()) {
    return std::nullopt;
  }
  std::string data;
  for (const std::string& match_path : matches) {
    if (!tsl::ReadFileToString(env, match_path, &data).ok()) {
      continue;
    }
    auto loaded = LoadSerializedExecutable(CacheTier::kTier3, key, data);
    if (!loaded.ok()) {
      continue;
    }
    utimensat(AT_FDCWD, match_path.c_str(), nullptr, 0);
    if (link(match_path.c_str(), cache_entry_path.c_str()) != 0) {
      if (absl::Status s = AtomicWriteToCacheFile(cache_entry_path, data);
          !s.ok()) {
        ABSL_VLOG(2) << "Failed to populate tier-3 cache alias "
                     << cache_entry_path << ": " << s;
      }
    }
    utimensat(AT_FDCWD, cache_entry_path.c_str(), nullptr, 0);
    return *std::move(loaded);
  }
  return std::nullopt;
}

static std::optional<SharedLoadedExecutableWithMetadata>
TryGetFromTier3CacheByPrefix(tsl::Env* env, const CompilationCacheKey& key,
                             const std::string& cache_entry_path) {
  const std::string compact = key.CompactFormat();
  if (compact.size() <= 34) {
    return std::nullopt;
  }
  const std::string prefix = compact.substr(0, 34);
  for (const std::string& pattern : {
           absl::StrCat(GetTier3CompilationCachePath(), "/", prefix, "*",
                        kTier3CacheFileExtension),
           absl::StrCat(GetTier3CacheRootDir(), "/*/", prefix, "*",
                        kTier3CacheFileExtension),
       }) {
    if (auto loaded =
            TryLoadMatchingKeyFromPattern(env, key, cache_entry_path, pattern);
        loaded.has_value()) {
      return loaded;
    }
  }
  return std::nullopt;
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> GetFromTier3Cache(
    const CompilationCacheKey& key) {
  // Read the cache file into a char buffer.
  const std::string cache_entry_path = GetTier3CacheEntryPath(key);
  ABSL_VLOG(1) << "Reading tier-3 cache for key " << key << " at path "
               << cache_entry_path;

  tsl::Env* const env = tsl::Env::Default();
  std::string data;
  absl::Status read_status =
      tsl::ReadFileToString(env, cache_entry_path, &data);
  if (absl::IsNotFound(read_status)) {
    if (auto loaded = TryGetFromTier3CacheByPrefix(env, key, cache_entry_path);
        loaded.has_value()) {
      return *std::move(loaded);
    }
  } else if (read_status.ok()) {
    utimensat(AT_FDCWD, cache_entry_path.c_str(), nullptr, 0);
  }
  TT_RETURN_IF_ERROR(read_status)
      << "cannot read tier-3 cache file " << cache_entry_path;

  ABSL_VLOG(1) << "Read " << data.size() << " bytes from tier-3 cache for key "
               << key << " at path " << cache_entry_path;

  // Load the executable from the char buffer.
  return LoadSerializedExecutable(CacheTier::kTier3, key, data);
}

}  // namespace torch_tpu
