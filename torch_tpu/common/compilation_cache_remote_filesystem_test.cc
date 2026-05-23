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

#include <cstdint>
#include <string>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compile_options_key.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/tier3_compilation_cache.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/tsl/platform/env.h"

namespace torch_tpu {

absl::Status AtomicWriteToCacheFile(const std::string& cache_entry_path,
                                    const std::string& serialized_data);
absl::Status EnsureDirExistsRecursively(const std::string& path);

namespace {

using testing::HasSubstr;

// A test environment that initializes the PjRt client, which is required for
// GetFromTier2Cache() to work.
class TpuTestEnvironment : public testing::Environment {
 public:
  void SetUp() override {
    // This must be done before testing GetFromTier2Cache(),.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }
};

// Installs the test environment.
auto* const test_env =
    testing::AddGlobalTestEnvironment(new TpuTestEnvironment);

class RemoteCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    test_dir_ =
        GetEnvOnce<kTorchTpuInternalTier3CompilationCacheRootEnvVar>().value_or(
            "");
    ASSERT_FALSE(test_dir_.empty())
        << "The TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT env var must "
           "be set.";
  }

  ~RemoteCacheTest() override {
    // Clean up the test file if it exists.
    if (!test_file_.empty()) {
      env_->DeleteFile(test_file_).IgnoreError();
    }
  }

  tsl::Env* const env_ = tsl::Env::Default();
  std::string test_dir_;
  std::string test_file_;
};

// Tests that AtomicWriteToCacheFile correctly writes a new file.
TEST_F(RemoteCacheTest, WritesNewCacheFile) {
  test_file_ = absl::StrCat(test_dir_, "/new_file");
  // Make test_file_ unique in case the test is run with --runs_per_test=N.
  ASSERT_TRUE(env_->CreateUniqueFileName(&test_file_, /*suffix=*/".data"));
  const std::string content = "test_data_content";

  // Clean up the file if it exists.
  env_->DeleteFile(test_file_).IgnoreError();

  // Write the file.
  ABSL_LOG(INFO) << "Writing to " << test_file_;
  absl::Status status = AtomicWriteToCacheFile(test_file_, content);
  ASSERT_EQ(status, absl::OkStatus());
  ASSERT_EQ(env_->FileExists(test_file_), absl::OkStatus());

  // Verify content.
  std::string read_content;
  status = tsl::ReadFileToString(env_, test_file_, &read_content);
  ASSERT_EQ(status, absl::OkStatus());
  EXPECT_EQ(read_content, content)
      << "Unexpected content in file " << test_file_;
}

// Tests that AtomicWriteToCacheFile correctly overwrites an existing file.
TEST_F(RemoteCacheTest, OverwritesExistingCacheFile) {
  test_file_ = absl::StrCat(test_dir_, "/existing_file");
  // Make test_file_ unique in case the test is run with --runs_per_test=N.
  ASSERT_TRUE(env_->CreateUniqueFileName(&test_file_, /*suffix=*/".data"));
  const std::string content = "test_data_content";

  // Clean up the file if it exists.
  env_->DeleteFile(test_file_).IgnoreError();

  // Write the file.
  ABSL_LOG(INFO) << "Writing to " << test_file_;
  absl::Status status = AtomicWriteToCacheFile(test_file_, content);
  ASSERT_EQ(status, absl::OkStatus());
  ASSERT_EQ(env_->FileExists(test_file_), absl::OkStatus());

  // Overwrite the file.
  ABSL_LOG(INFO) << "Overwriting " << test_file_;
  const std::string new_content = "new_test_data_content";
  status = AtomicWriteToCacheFile(test_file_, new_content);
  ASSERT_EQ(status, absl::OkStatus());
  ASSERT_EQ(env_->FileExists(test_file_), absl::OkStatus());

  // Verify new content.
  std::string read_content;
  status = tsl::ReadFileToString(env_, test_file_, &read_content);
  ASSERT_EQ(status, absl::OkStatus());
  EXPECT_EQ(read_content, new_content)
      << "Unexpected content in file " << test_file_;
}

TEST_F(RemoteCacheTest, CreatesCacheDirectory) {
  // Make test_dir_ unique in case the test is run with --runs_per_test=N.
  test_dir_ = absl::StrCat(test_dir_, "/test_dir");
  ASSERT_TRUE(env_->CreateUniqueFileName(&test_dir_, /*suffix=*/""));
  // Add two more levels of directories to test recursive creation.
  const std::string test_subdir = absl::StrCat(test_dir_, "/level1/level2");

  // Clean up test_dir_ if it exists.
  int64_t undeleted_files, undeleted_dirs;
  env_->DeleteRecursively(test_dir_, &undeleted_files, &undeleted_dirs)
      .IgnoreError();

  // Create the directory.
  ABSL_LOG(INFO) << "Creating directory " << test_subdir;
  absl::Status status = EnsureDirExistsRecursively(test_subdir);

  // Verify the directory was created.
  ASSERT_EQ(status, absl::OkStatus());
  ASSERT_EQ(env_->IsDirectory(test_subdir), absl::OkStatus());
}

TEST_F(RemoteCacheTest, GetFromTier3Cache) {
  const CompilationCacheKey key(
      GraphKey(ShapelessKey(123), DimensionsKey({10})), CompileOptionsKey(0));

  // Write a cache entry file.
  const std::string cache_entry_path = GetTier3CacheEntryPath(key);
  ABSL_LOG(INFO) << "Writing to " << cache_entry_path;
  const std::string content = "test_data_content";
  absl::Status status = AtomicWriteToCacheFile(cache_entry_path, content);
  ASSERT_EQ(status, absl::OkStatus());

  // Read back using GetFromTier3Cache.
  // We expect failure because the data is invalid, but we want to verify that
  // it *successfully read the file*.
  auto result = GetFromTier3Cache(key);
  ASSERT_NE(result.status(), absl::OkStatus());
  EXPECT_THAT(result.status().message(), HasSubstr("load serialized"));
}

}  // namespace
}  // namespace torch_tpu
