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

#include "torch_tpu/common/tier2_compilation_cache.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/common/unique_file_descriptor.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"

namespace torch_tpu {

// Remember to increment this whenever TorchTPU's behavior changes enough to
// invalidate the tier-2 cache.
constexpr int kTorchTpuBinaryVersion = 5;

constexpr std::string_view kTier2CacheRootDir = "/dev/shm/torch_tpu_cache";
constexpr std::string_view kLockFileExtension = ".lock";
constexpr std::string_view kTier2CacheFileExtension = ".bin";

std::ostream& operator<<(std::ostream& os, const CacheTier tier) {
  switch (tier) {
    case CacheTier::kUnknown:
      return os << "unknown";
    case CacheTier::kTier1:
      return os << "tier-1";
    case CacheTier::kTier2:
      return os << "tier-2";
    case CacheTier::kTier3:
      return os << "tier-3";
      // Deliberately omitting the default case to catch any new cache tiers as
      // a compiler error.
  }
}

// Returns the PjRt version string.
[[nodiscard]] static std::string GetPjRtVersion() {
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  ABSL_CHECK(client != nullptr)  // CRASH_OK
      << "PjRtClient must be initialized before accessing the tier-2 cache.";

  // Get the PjRt C API version.
  const auto attrib_opt = client->plugin_attributes();
  const auto attrib = attrib_opt.has_value()
                          ? *attrib_opt
                          : xla::PjRtPluginAttributes{
                                .pjrt_c_api_major_version = 0,
                                .pjrt_c_api_minor_version = 0,
                            };

  // Get the full version string.
  return absl::StrCat(client->platform_version(),
                      "\nC API version: ", attrib.pjrt_c_api_major_version, ".",
                      attrib.pjrt_c_api_minor_version);
}

FingerprintType GetTorchTpuBinaryFingerprint() {
  static const FingerprintType fingerprint = []() {
    const std::string pjrt_version = GetPjRtVersion();
    // The fingerprint has two components: the TorchTPU binary version and the
    // PjRt version.  Whenever either changes, we invalidate the tier-2 cache
    // to be safe.
    const FingerprintType fingerprint =
        FingerprintCat(kTorchTpuBinaryVersion, pjrt_version);
    ABSL_LOG(INFO) << "Tier-2 cache uses TorchTPU binary fingerprint: "
                   << absl::Hex(fingerprint, absl::kZeroPad16)
                   << "\n  based on TorchTPU binary version "
                   << kTorchTpuBinaryVersion << " and PjRt version:\n"
                   << pjrt_version;
    return fingerprint;
  }();
  return fingerprint;
}

// The value of the TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE environment
// variable that indicates that the tier-2 cache is disabled.
constexpr std::string_view kDisabledCacheNameInEnvVar = "disabled";

// The default name of the tier-2 compilation cache, used when:
// 1. the tier-3 cache is enabled, or
// 2. the TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE environment variable is
//    not set and the world size is greater than 1.
constexpr std::string_view kDefaultCacheName = "default";

const std::string& GetTier2CacheName() {
  static const absl::NoDestructor<std::string> cache_name([]() {
    const auto& tier2_cache =
        GetEnvOnce<kTorchTpuInternalTier2CompilationCacheEnvVar>();
    if (!tier2_cache.has_value() || tier2_cache->empty()) {
      // If tier-3 is enabled, we must enable tier-2 as well.
      const auto& tier3_cache_root =
          GetEnvOnce<kTorchTpuInternalTier3CompilationCacheRootEnvVar>();
      if (tier3_cache_root.has_value() && !tier3_cache_root->empty()) {
        ABSL_LOG(INFO) << "Tier-2 compilation cache is enabled with name '"
                       << kDefaultCacheName << "' because tier-3 is enabled.";
        return std::string(kDefaultCacheName);
      }
      // Decide whether to use the tier-2 cache based on the world size.
      const auto world_size_or =
          PjrtBackend::GetInstance().GetGlobalDeviceCount();
      if (world_size_or.ok()) {
        if (*world_size_or <= 1) {
          ABSL_LOG(INFO) << "Tier-2 compilation cache is disabled for world "
                            "size 1.";
          return std::string();
        } else {
          ABSL_LOG(INFO) << "Tier-2 compilation cache is enabled with name '"
                         << kDefaultCacheName << "' for world size "
                         << *world_size_or << ".";
          return std::string(kDefaultCacheName);
        }
      }
      ABSL_LOG(INFO) << "Tier-2 compilation cache is disabled because we could "
                        "not get the world size: "
                     << world_size_or.status();
      return std::string();
    }
    if (*tier2_cache == kDisabledCacheNameInEnvVar) {
      ABSL_LOG(INFO) << "Tier-2 compilation cache is disabled as requested by "
                        "the "
                     << kTorchTpuInternalTier2CompilationCacheEnvVar
                     << " environment variable.";
      return std::string();
    }
    ABSL_LOG(INFO) << "Tier-2 compilation cache is enabled with name '"
                   << *tier2_cache << "' as requested by the "
                   << kTorchTpuInternalTier2CompilationCacheEnvVar
                   << " environment variable.";
    return *tier2_cache;
  }());
  return *cache_name;
}

bool UsesTier2CompilationCache() { return !GetTier2CacheName().empty(); }

const std::string& GetTier2CompilationCachePath() {
  static const absl::NoDestructor<std::string> cache_path([]() {
    ABSL_CHECK(UsesTier2CompilationCache())  // CRASH_OK
        << "Tier-2 compilation cache is not enabled.";
    return absl::StrCat(
        kTier2CacheRootDir, "/", GetTier2CacheName(), "/",
        absl::Hex(GetTorchTpuBinaryFingerprint(), absl::kZeroPad16));
  }());
  return *cache_path;
}

std::string GetTier2CacheEntryPath(CompilationCacheKey key) {
  return GetTier2CacheEntryPath(key.CompactFormat());
}

std::string GetTier2CacheEntryPath(const std::string_view compact_key) {
  return absl::StrCat(GetTier2CompilationCachePath(), "/", compact_key,
                      kTier2CacheFileExtension);
}

// RAII class for mapping a cache file into memory.
// The mapping is created in the Make() factory and destroyed in the destructor.
// This guarantees that the mapping is destroyed even if an error/exception
// occurs.
class MappedCacheEntry {
 public:
  // Creates a MappedCacheEntry for the given key.
  // Returns an error if the cache file cannot be opened or mapped.
  //
  // The caller must hold the lock for the given key.
  static absl::StatusOr<MappedCacheEntry> Make(CompilationCacheKey key) {
    // Try to open the cache file for reading.
    const std::string path = GetTier2CacheEntryPath(key);
    UniqueFileDescriptor fd(open(path.c_str(), O_RDONLY));
    // It's normal for the cache file to be missing, in which case the mapped
    // entry will be empty. We don't want to slow down this hot path, so we
    // don't construct a detailed error message here - the error will be
    // swallowed by the caller anyway.
    TT_RET_CHECK(fd.valid(), error::kNotFound)
        << "not found";  // Deliberately trivial message.

    // Get the cache file size.
    struct stat sb;
    TT_RET_CHECK(fstat(fd.get(), &sb) != -1, error::kInternal)
        << "failed to get the size of cache file " << path;

    // Map the cache file into memory.
    auto* const data = static_cast<const char*>(  //
        mmap(nullptr,  // Let the OS pick the buffer address.
             sb.st_size,
             PROT_READ,  // Read-only.
             // MAP_POPULATE: Prefault pages since they are in RAM (tmpfs).
             MAP_SHARED | MAP_POPULATE, fd.get(), 0));
    TT_RET_CHECK(data != MAP_FAILED, error::kInternal)
        << "failed to map cache file " << path << " into memory";

    // st_atime is when the file was last read.
    return MappedCacheEntry(data, sb.st_size, absl::FromTimeT(sb.st_atime));
  }

  // Unmaps the cache file from memory.
  ~MappedCacheEntry() {
    if (data_ != nullptr) {
      munmap(const_cast<void*>(static_cast<const void*>(data_)), size_);
    }
  }

  // The class is movable but not copyable.
  MappedCacheEntry(const MappedCacheEntry&) = delete;
  MappedCacheEntry& operator=(const MappedCacheEntry&) = delete;
  MappedCacheEntry(MappedCacheEntry&& other)
      : data_(other.data_), size_(other.size_), last_read_(other.last_read_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  // We have no need for move assignment for now.
  MappedCacheEntry& operator=(MappedCacheEntry&& other) = delete;

  // Returns the mapped data, or an empty string_view if mapping failed.
  [[nodiscard]] std::string_view data() const {
    return data_ == nullptr ? "" : std::string_view(data_, size_);
  }

  // Returns the last read time of the cache file.
  [[nodiscard]] absl::Time last_read() const { return last_read_; }

 private:
  // Creates a MappedCacheEntry with the given information.
  MappedCacheEntry(const char* const data, const size_t size,
                   const absl::Time last_read)
      : data_(data), size_(size), last_read_(last_read) {}

  const char* data_ = nullptr;
  size_t size_ = 0;
  absl::Time last_read_;
};

absl::StatusOr<SharedLoadedExecutableWithMetadata> LoadSerializedExecutable(
    CacheTier tier, CompilationCacheKey key, const std::string_view data) {
  return LoadSerializedExecutable(tier, absl::StrCat(key), data);
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> LoadSerializedExecutable(
    CacheTier tier, const std::string_view key_str,
    const std::string_view data) {
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  TT_RET_CHECK(client, error::kFailedPrecondition)
      << "PjRtClient must be initialized before accessing the " << tier
      << " cache";
  TT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_executable,
      AdaptXlaError(
          client->LoadSerializedExecutable(data,
                                           /* options= */ std::nullopt,
                                           xla::LoadOptions()),
          /* context =*/absl::StrCat(
              "failed to load serialized executable from the ", tier,
              " cache for key ", key_str, ", where the serialized data has ",
              data.size(), " bytes")));
  return LoadedExecutableWithMetadata::MakeShared(std::move(pjrt_executable));
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> GetFromTier2Cache(
    CompilationCacheKey key, absl::Time request_start,
    Tier2CacheEntryStats& stats) {
  const absl::Time read_start = absl::Now();

  // Use mmap() to load the cache file into memory. This avoids making a copy of
  // the serialized executable.
  TT_ASSIGN_OR_RETURN(MappedCacheEntry mapped_entry,
                      MappedCacheEntry::Make(key));

  // Create a SharedLoadedExecutableWithMetadata from the mapped data.
  TT_ASSIGN_OR_RETURN(
      SharedLoadedExecutableWithMetadata executable,
      LoadSerializedExecutable(CacheTier::kTier2, key, mapped_entry.data()));

  stats.pre_read_duration = read_start - request_start;
  stats.read_duration = absl::Now() - read_start;
  stats.last_read = mapped_entry.last_read();

  ABSL_VLOG(2) << "Tier-2 cache HIT for key: " << key
               << "\n  Pre-read duration: " << stats.pre_read_duration
               << "\n  Read duration: " << stats.read_duration
               << "\n  Last read: " << stats.last_read;
  return executable;
  // The dtor of mapped_entry unmaps the file here.
}

absl::Status AtomicWriteToCacheFile(const std::string& cache_entry_path,
                                    const std::string& serialized_data) {
  ABSL_VLOG(1) << "Writing serialized executable to " << cache_entry_path;

  // Use tsl::Env so that we can support writing to remote files as well as
  // local files.
  tsl::Env* const env = tsl::Env::Default();

  // Create a unique temp file in the same directory as the cache file.
  // We don't use mkstemp() because it's not compatible with Colossus.
  // We must put the temp file in the same directory as the final cache file
  // so that we can atomically rename it later.
  std::string temp_file_path = absl::StrCat(cache_entry_path, ".");

  // CreateUniqueFileName appends "Hostname-ThreadID-PID-TimestampMicroseconds"
  // to the prefix. We add a suffix ".<counter>.tmp" in the unlikely event that
  // AtomicWriteToCacheFile() is called in rapid succession with the same
  // host/pid/thread/timestamp combination.
  static std::atomic<int> counter = 0;
  TT_RET_CHECK(env->CreateUniqueFileName(
                   &temp_file_path,                                 // prefix
                   absl::StrCat(".", counter.fetch_add(1), ".tmp")  // suffix
                   ),
               error::kInternal)
      << "failed to create unique temp file for " << cache_entry_path;

  {
    bool success = false;

    // Automatically clean up the temp file and the final cache file if
    // something goes wrong.
    auto cleanup = absl::MakeCleanup([&]() {
      if (!success) {
        ABSL_LOG(ERROR) << "Failed to write cache file " << cache_entry_path
                        << ". Cleaning up.";
        env->DeleteFile(temp_file_path)
            .IgnoreError();  // IGNORE_ERROR_OK=best effort
        env->DeleteFile(cache_entry_path)
            .IgnoreError();  // IGNORE_ERROR_OK=best effort
      }
    });

    std::unique_ptr<tsl::WritableFile> file;
    TT_RETURN_IF_ERROR(env->NewWritableFile(temp_file_path, &file)).SetPrepend()
        << "failed to create writable file " << temp_file_path << ": ";
    TT_RETURN_IF_ERROR(file->Append(serialized_data)).SetPrepend()
        << "failed to write to file " << temp_file_path << ": ";
    TT_RETURN_IF_ERROR(file->Close()).SetPrepend()
        << "failed to close file " << temp_file_path << ": ";

    // Atomically rename the temp file to the final cache file path.
    // If the target file already exists, RenameFile() will replace it,
    // which is what we want (last writer wins).
    TT_RETURN_IF_ERROR(env->RenameFile(temp_file_path, cache_entry_path))
            .SetPrepend()
        << "failed to rename file " << temp_file_path << " to "
        << cache_entry_path << ": ";

    success = true;
  }

  ABSL_VLOG(1) << "Successfully wrote serialized executable to "
               << cache_entry_path;
  return absl::OkStatus();
}

absl::Status AtomicWriteToCacheFile(
    const std::string& cache_entry_path,
    const SharedLoadedExecutableWithMetadata& executable) {
  ABSL_VLOG(1) << "Serializing and writing executable to file "
               << cache_entry_path;
  TT_ASSIGN_OR_RETURN(const std::string serialized,
                      executable->GetExecutable()->SerializeExecutable(),
                      _.SetPrepend()
                          << "failed to serialize executable for cache file "
                          << cache_entry_path);
  return AtomicWriteToCacheFile(cache_entry_path, serialized);
}

// Returns the path to the lock file for the given key.
[[nodiscard]] static std::string GetTier2CacheEntryLockPath(
    CompilationCacheKey key) {
  return absl::StrCat(GetTier2CompilationCachePath(), "/", key.CompactFormat(),
                      kLockFileExtension);
}

absl::Status EnsureDirExistsRecursively(const std::string& path) {
  tsl::Env* const env = tsl::Env::Default();
  return env->RecursivelyCreateDir(path);
}

// Ensures that the tier-2 cache directory exists. This function only does
// the check once.
static void EnsureTier2CacheDirExistsOnceOrDie() {
  static const bool dir_exists = []() {
    const std::string cache_path = GetTier2CompilationCachePath();
    const absl::Status status = EnsureDirExistsRecursively(cache_path);
    ABSL_CHECK(status.ok())  // CRASH_OK
        << "Failed to create tier-2 cache directory: " << cache_path
        << " with error: " << status.message();
    return true;
  }();
  static_cast<void>(dir_exists);  // VOID_CAST_OK=dummy result.
}

Tier2CacheEntryLock::Tier2CacheEntryLock(CompilationCacheKey key)
    : key_(std::move(key)) {
  ABSL_VLOG(1) << "Acquiring tier-2 cache lock for key: " << key_;

  EnsureTier2CacheDirExistsOnceOrDie();
  const std::string lock_path = GetTier2CacheEntryLockPath(key_);

  // Open the lock file, creating it if it doesn't exist.
  lock_fd_.reset(
      open(lock_path.c_str(),
           // Create an empty file if one doesn't exist; otherwise open the
           // existing file.
           O_CREAT |
               // Open for reading and writing so that we can flock() the file.
               O_RDWR,
           // Allow other processes to open and flock() the file.
           0666));
  ABSL_CHECK(lock_fd_.valid())  // CRASH_OK
      << "Failed to create tier-2 cache entry lock file: " << lock_path
      << " with error: " << strerror(errno);
  ABSL_VLOG(1) << "Created tier-2 cache entry lock file: " << lock_path;

  // Acquire an exclusive lock on the file. This is a blocking call.
  flock(lock_fd_.get(), LOCK_EX);
  ABSL_VLOG(1) << "Acquired tier-2 cache entry lock: " << lock_path;
}

Tier2CacheEntryLock::~Tier2CacheEntryLock() { Release(); }

void Tier2CacheEntryLock::Release() {
  if (lock_fd_.valid()) {
    ABSL_VLOG(1) << "Releasing tier-2 cache lock for key: " << key_;
    flock(lock_fd_.get(), LOCK_UN);
    lock_fd_.reset();
  }
}

}  // namespace torch_tpu
