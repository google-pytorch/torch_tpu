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

#include "torch_tpu/csrc/common/compilation_cache_utils.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>

#include <atomic>
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
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/compilation.h"
#include "torch_tpu/csrc/common/env_vars.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fingerprint_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"

namespace torch_tpu {

// Remember to increment this whenever TorchTPU's behavior changes enough to
// invalidate the tier-2 cache.
constexpr int kTorchTpuBinaryVersion = 5;

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

absl::StatusOr<SharedLoadedExecutableWithMetadata> LoadSerializedExecutable(
    CacheTier tier, CompilationCacheKey key, const std::string_view data) {
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
              " cache for key ", key, ", where the serialized data has ",
              data.size(), " bytes")));
  return LoadedExecutableWithMetadata::MakeShared(std::move(pjrt_executable));
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

absl::Status EnsureDirExistsRecursively(const std::string& path) {
  tsl::Env* const env = tsl::Env::Default();
  return env->RecursivelyCreateDir(path);
}

const std::string& GetTier3CacheRootDir() {
  static const absl::NoDestructor<std::string> root_dir([]() {
    const auto& tier3_cache =
        GetEnvOnce<kTorchTpuTier3CompilationCacheRootEnvVar>();
    const std::string root_dir = tier3_cache.value_or("");
    ABSL_LOG(INFO) << "Tier-3 compilation cache root directory: " << root_dir;
    return root_dir;
  }());
  return *root_dir;
}

}  // namespace torch_tpu
