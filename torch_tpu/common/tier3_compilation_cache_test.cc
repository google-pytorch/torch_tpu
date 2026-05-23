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

#include "torch_tpu/common/tier3_compilation_cache.h"

#include <stdlib.h>

#include <cstdint>
#include <string>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compile_options_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/tier2_compilation_cache.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {

absl::Status AtomicWriteToCacheFile(const std::string& cache_entry_path,
                                    const std::string& serialized_data);

namespace {

using testing::StartsWith;

// A test environment that initializes the PjRt client, which is required for
// GetTorchTpuBinaryFingerprint() to work.
class TpuTestEnvironment : public testing::Environment {
 public:
  void SetUp() override {
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }
};

// Installs the test environment.
auto* const test_env =
    testing::AddGlobalTestEnvironment(new TpuTestEnvironment);

// Returns a cache key with the given shapeless key, dimensions key, and
// a placeholder compile options key.
CompilationCacheKey MakeCacheKey(uint64_t shapeless_key, int num_dims) {
  const Dimensions dims(num_dims, 1);
  const DimensionsKey dimensions_key(dims);
  return CompilationCacheKey(
      GraphKey(ShapelessKey(shapeless_key), dimensions_key),
      CompileOptionsKey(0));
}

class Tier3CompilationCacheTest : public testing::Test {};

TEST_F(Tier3CompilationCacheTest,
       UsesTier3CompilationCacheIsTrueWhenEnvVarIsSet) {
  EXPECT_TRUE(UsesTier3CompilationCache());
}

TEST_F(Tier3CompilationCacheTest, GetTier3CacheEntryPath) {
  const auto key = MakeCacheKey(/*shapeless_key=*/123, /*num_dims=*/2);
  const std::string path = GetTier3CacheEntryPath(key);

  // Expected path format: <root>/<fingerprint>/<key>.bin
  const std::string expected_prefix = absl::StrCat(
      GetEnvOnce<kTorchTpuInternalTier3CompilationCacheRootEnvVar>().value(),
      "/", absl::Hex(GetTorchTpuBinaryFingerprint(), absl::kZeroPad16), "/");

  ASSERT_THAT(path, StartsWith(expected_prefix));
  const std::string suffix = path.substr(expected_prefix.size());
  EXPECT_EQ(suffix, absl::StrCat(key.CompactFormat(), ".bin"));
}

// GetFromTier3Cache() should return a not-found error if the cache file does
// not exist.
TEST_F(Tier3CompilationCacheTest,
       GetFromTier3CacheFailsIfCacheFileDoesNotExist) {
  const auto key = MakeCacheKey(/*shapeless_key=*/123, /*num_dims=*/0);
  EXPECT_EQ(GetFromTier3Cache(key).status().code(), error::kNotFound);
}

// GetFromTier3Cache() should return an error if the cache file exists
// but has invalid data.
TEST_F(Tier3CompilationCacheTest, GetFromTier3CacheFailsIfCacheFileIsInvalid) {
  const auto key = MakeCacheKey(/*shapeless_key=*/124, /*num_dims=*/0);
  const std::string serialized_data = "invalid data";
  ASSERT_EQ(
      AtomicWriteToCacheFile(GetTier3CacheEntryPath(key), serialized_data),
      absl::OkStatus());

  const auto status_code = GetFromTier3Cache(key).status().code();

  // The error should not be kNotFound, because the file exists.
  EXPECT_NE(status_code, error::kOk);
  EXPECT_NE(status_code, error::kNotFound);
}

}  // namespace
}  // namespace torch_tpu
