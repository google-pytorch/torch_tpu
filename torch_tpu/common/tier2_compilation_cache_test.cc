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

#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/unique_file_descriptor.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {

absl::Status AtomicWriteToCacheFile(const std::string& cache_entry_path,
                                    const std::string& serialized_data);

namespace {

namespace fs = std::filesystem;

using testing::ElementsAre;
using testing::MatchesRegex;
using testing::StartsWith;

// Removes all files in the given directory, recursively.
void RemoveAllFilesInDirectoryRecursively(const std::string& directory_path) {
  if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
    return;
  }

  // Iterate over the directory (non-recursively)
  for (const auto& entry : fs::directory_iterator(directory_path)) {
    // Remove the entry (file, symlink, or subdirectory).
    std::error_code ec;  // Used to prevent exceptions from crashing the loop.
    fs::remove_all(entry.path(), ec);
    if (ec) {
      ABSL_LOG(FATAL) << "Failed to remove " << entry.path() << ": "
                      << ec.message();
    }
  }
}

// Lists all files in the given directory, non-recursively.
std::vector<std::string> ListFiles(const std::string& directory_path) {
  std::vector<std::string> files;

  if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
    return files;
  }

  // directory_iterator is non-recursive.
  try {
    for (const auto& entry : fs::directory_iterator(directory_path)) {
      // Check if it's a regular file (this filters out directories, sockets,
      // etc.)
      if (fs::is_regular_file(entry.status())) {
        files.push_back(entry.path().string());
      }
    }
  } catch (const fs::filesystem_error& e) {
    ABSL_LOG(FATAL) << "Error accessing directory: " << e.what();
  }

  return files;
}

// Returns a cache key with the given shapeless key, dimensions key, and
// a placeholder compile options key.
CompilationCacheKey MakeCacheKey(uint64_t shapeless_key, int num_dims) {
  const Dimensions dims(num_dims, 1);
  const DimensionsKey dimensions_key(dims);
  return CompilationCacheKey(
      GraphKey(ShapelessKey(shapeless_key), dimensions_key),
      GetCompilationSpec(CompilationMode::kFastCompile).compile_options_key);
}

// A test environment that initializes the PjRt client, which is required for
// GetFromTier2Cache() to work.
class TpuTestEnvironment : public testing::Environment {
 public:
  void SetUp() override {
    // This must be done before testing GetFromTier2Cache(),.
    // Use xla_cpu for testing to allow mocking multiple devices in a single
    // process without needing real TPU hardware or multiple workers.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "xla_cpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }
};

// Installs the test environment.
auto* const test_env =
    testing::AddGlobalTestEnvironment(new TpuTestEnvironment);

class Tier2CacheTest : public testing::Test {
 protected:
  Tier2CacheTest() {
    SetEnv(kTorchTpuInternalTier2CompilationCacheEnvVar,
           absl::StrCat("my_cache_", getpid()));
    cache_path_ = GetTier2CompilationCachePath();

    // Ensure the cache directory exists. Otherwise tests that run first
    // without acquiring a Tier2CacheEntryLock (like
    // ConcurrentWritesDoNotCorruptCacheFile) will fail to write.
    ABSL_CHECK_OK(EnsureDirExistsRecursively(cache_path_));

    // Clear the cache directory before each test, in case any previous tests
    // didn't clean up properly.
    RemoveAllFilesInDirectoryRecursively(cache_path_);
    const auto files = ListFiles(cache_path_);
    ABSL_CHECK(files.empty()) << "Cache directory is not empty: " << files;
  }

  std::string cache_path_;
};

class Tier2CacheEntryLockTest : public Tier2CacheTest {};

// If the lock file for a key does not exist, the Tier2CacheEntryLock ctor
// should create it.
TEST_F(Tier2CacheEntryLockTest, CreatesLockFileIfNeeded) {
  // Acquire a lock for the given key. This should create the lock file.
  const auto key = MakeCacheKey(/*shapeless_key=*/123, /*num_dims=*/0);
  Tier2CacheEntryLock lock(key);

  // The lock file should be created.
  EXPECT_THAT(ListFiles(cache_path_),
              ElementsAre(absl::StrCat(
                  cache_path_, "/",
                  "000000000000007b_5825f5f3bd962979_1e50903f6ba7baaf.lock")));
}

// If the lock file for a key already exists, the Tier2CacheEntryLock ctor
// should not create it.
TEST_F(Tier2CacheEntryLockTest, DoesNotCreateLockFileIfItExists) {
  // Create a lock file for the given key manually.
  const auto key = MakeCacheKey(/*shapeless_key=*/123, /*num_dims=*/0);
  const std::string lock_path =
      absl::StrCat(cache_path_, "/", key.CompactFormat(), ".lock");
  const UniqueFileDescriptor fd(
      open(lock_path.c_str(), O_CREAT | O_RDWR, 0666));
  ABSL_CHECK(fd.valid()) << "Failed to create lock file: " << lock_path;

  // The lock file should exist.
  EXPECT_THAT(ListFiles(cache_path_), ElementsAre(lock_path));

  // Acquire a lock for the given key. This should not create a new lock file.
  Tier2CacheEntryLock lock(key);

  // The lock file should still exist.
  EXPECT_THAT(ListFiles(cache_path_), ElementsAre(lock_path));
}

// Only one user can hold the lock for a given key at a time.
TEST_F(Tier2CacheEntryLockTest, OnlyOneUserCanHoldLockAtATime) {
  const auto key = MakeCacheKey(/*shapeless_key=*/456, /*num_dims=*/1);

  // How many threads are currently inside the lock.
  std::atomic<int> threads_inside_lock{0};

  const int num_threads = 20;
  const int iterations_per_thread = 100;

  auto worker_task = [&]() {
    for (int i = 0; i < iterations_per_thread; ++i) {
      Tier2CacheEntryLock lock(key);
      const int current_active = ++threads_inside_lock;
      ABSL_CHECK_EQ(current_active, 1)  // CRASH_OK
          << "Multiple threads acquired the lock simultaneously.";

      // Artificial delay: hold the lock longer to increase the probability of
      // overlapping with other threads if the lock mechanism is faulty.
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      --threads_inside_lock;
      // Lock is released here by dtor.
    }
  };

  // Spawn threads.
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker_task);
  }

  // Cleanup: join threads.
  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }
}

// Reads the given file as a string. Caveat: this can only read files that are
// smaller than the buffer size (1024 bytes).
[[nodiscard]] std::string ReadFile(const std::string& file_path) {
  const UniqueFileDescriptor fd(open(file_path.c_str(), O_RDONLY));
  ABSL_CHECK(fd.valid()) << "Failed to open file: " << file_path;
  char buffer[1024];
  const ssize_t bytes_read = read(fd.get(), buffer, sizeof(buffer));
  ABSL_CHECK_NE(bytes_read, -1) << "Failed to read file: " << file_path;
  return std::string(buffer, bytes_read);
}

TEST_F(Tier2CacheTest, CachePath) {
  const std::string kCachePathPrefix =
      absl::StrCat("/dev/shm/torch_tpu_cache/my_cache_", getpid(), "/");
  ASSERT_THAT(cache_path_, StartsWith(kCachePathPrefix));

  // The suffix should be a 16-character hexadecimal string.
  const std::string cache_path_suffix =
      cache_path_.substr(kCachePathPrefix.size());
  EXPECT_THAT(cache_path_suffix, MatchesRegex("[a-f0-9]{16}"));
}

// AtomicWriteToCacheFile() should write the given data to the cache file
// for the given key, if the file does not exist.
TEST_F(Tier2CacheTest, WriteToCacheEntryIfDoesNotExist) {
  const auto key = MakeCacheKey(/*shapeless_key=*/123, /*num_dims=*/0);
  Tier2CacheEntryLock lock(key);
  const std::string serialized_data = "test data";
  ASSERT_EQ(
      AtomicWriteToCacheFile(GetTier2CacheEntryPath(key), serialized_data),
      absl::OkStatus());

  // Read the data from the cache file.
  const std::string cache_file_path =
      absl::StrCat(cache_path_, "/", key.CompactFormat(), ".bin");
  const std::string data_read = ReadFile(cache_file_path);
  EXPECT_EQ(data_read, serialized_data);
}

// AtomicWriteToCacheFile() should overwrite the existing cache file for
// the given key, if it exists.
TEST_F(Tier2CacheTest, WriteToCacheEntryIfExists) {
  const auto key = MakeCacheKey(/*shapeless_key=*/123, /*num_dims=*/0);
  Tier2CacheEntryLock lock(key);

  // The first write should create the cache file.
  const std::string serialized_data = "test data";
  ASSERT_EQ(
      AtomicWriteToCacheFile(GetTier2CacheEntryPath(key), serialized_data),
      absl::OkStatus());

  // The second write should overwrite the cache file.
  const std::string serialized_data_2 = "test data 2";
  ASSERT_EQ(
      AtomicWriteToCacheFile(GetTier2CacheEntryPath(key), serialized_data_2),
      absl::OkStatus());

  // Read the data from the cache file.
  const std::string cache_file_path =
      absl::StrCat(cache_path_, "/", key.CompactFormat(), ".bin");
  const std::string data_read = ReadFile(cache_file_path);
  EXPECT_EQ(data_read, serialized_data_2);
}

// Concurrent writes to the same cache file should not corrupt it.
TEST_F(Tier2CacheTest, ConcurrentWritesDoNotCorruptCacheFile) {
  const auto key = MakeCacheKey(/*shapeless_key=*/456, /*num_dims=*/1);
  const std::string cache_file_path =
      absl::StrCat(cache_path_, "/", key.CompactFormat(), ".bin");

  const int num_threads = 20;
  const int iterations_per_thread = 100;

  auto worker_task = [&]() {
    for (int i = 0; i < iterations_per_thread; ++i) {
      const std::string serialized_data = "test data";
      // Deliberately do not acquire the lock here, as we want to test the
      // behavior of concurrent writes.
      ASSERT_EQ(
          AtomicWriteToCacheFile(GetTier2CacheEntryPath(key), serialized_data),
          absl::OkStatus());

      // Artificial delay: hold the lock longer to increase the probability of
      // overlapping with other threads.
      std::this_thread::sleep_for(std::chrono::microseconds(10));

      const std::string data_read = ReadFile(cache_file_path);
      EXPECT_EQ(data_read, serialized_data);
    }
  };

  // Spawn threads.
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker_task);
  }

  // Cleanup: join threads.
  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }
}

// GetFromTier2Cache() should return a not-found error if the cache file does
// not exist.
TEST_F(Tier2CacheTest, GetFromTier2CacheFailsIfCacheFileDoesNotExist) {
  const auto key = MakeCacheKey(/*shapeless_key=*/123, /*num_dims=*/0);
  Tier2CacheEntryStats tier2_stats;
  EXPECT_EQ(GetFromTier2Cache(key, absl::Now(), tier2_stats).status().code(),
            error::kNotFound);
}

// GetFromTier2Cache() should return an error if the cache file exists
// but has invalid data.
TEST_F(Tier2CacheTest, GetFromTier2CacheFailsIfCacheFileIsInvalid) {
  const auto key = MakeCacheKey(/*shapeless_key=*/123, /*num_dims=*/0);
  Tier2CacheEntryLock lock(key);
  const std::string serialized_data = "invalid data";
  ASSERT_EQ(
      AtomicWriteToCacheFile(GetTier2CacheEntryPath(key), serialized_data),
      absl::OkStatus());

  Tier2CacheEntryStats tier2_stats;
  const auto status_code =
      GetFromTier2Cache(key, absl::Now(), tier2_stats).status().code();

  // The error should not be kNotFound, because the file exists.
  EXPECT_NE(status_code, error::kOk);
  EXPECT_NE(status_code, error::kNotFound);
}

}  // namespace
}  // namespace torch_tpu
