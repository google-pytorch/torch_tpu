/*
 * Copyright 2025 Google LLC
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

// Tests for the CompilationCache that enable the tier-2 and tier-3 caches.
// Their env vars are read once per process, so these tests live in their own
// test target instead of compilation_cache_test.

#include <unistd.h>

#include <string>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/compilation_cache_test_utils.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/tier2_compilation_cache.h"
#include "csrc/common/tier3_compilation_cache.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

class TieredCompilationCacheTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Use a unique name so that entries from previous runs can't be hit.
    const std::string name = absl::StrCat("compilation_cache_tiered_test_",
                                          absl::ToUnixNanos(absl::Now()));
    SetEnv<kTorchTpuTier2CompilationCacheEnvVar>(name);
    SetEnv<kTorchTpuTier3CompilationCacheRootEnvVar>(
        absl::StrCat(testing::TempDir(), "/", name));
  }

  TieredCompilationCacheTest() {
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
    CompilationCache::Restart();
  }
};

TEST_F(TieredCompilationCacheTest, UnboundedDynamicCacheTier2Hit) {
  CompilationCache& cache = CompilationCache::GetInstance();
  const CompilationCacheKey key(GraphKey(ShapelessKey(2), DimensionsKey({})),
                                CompileOptionsKey({}), "tier2");
  ASSERT_TRUE(cache
                  .AddUnboundedDynamicCacheEntry(key, CreateDummyExecutable,
                                                 MakeDummyDynamicMetadata())
                  .ok());
  EXPECT_EQ(cache.GetCacheStats().num_compilations, 1);

  // Evict the in-memory cache, and remove the tier-3 entry so that only a
  // tier-2 hit can avoid recompilation.
  cache.EvictAll();
  ASSERT_EQ(unlink(GetTier3CacheEntryPath(key).c_str()), 0);

  ASSERT_TRUE(cache
                  .AddUnboundedDynamicCacheEntry(key, CreateDummyExecutable,
                                                 MakeDummyDynamicMetadata())
                  .ok());
  EXPECT_EQ(cache.GetCacheStats().num_compilations, 1);
}

TEST_F(TieredCompilationCacheTest, UnboundedDynamicCacheTier3Hit) {
  CompilationCache& cache = CompilationCache::GetInstance();
  const CompilationCacheKey key(GraphKey(ShapelessKey(3), DimensionsKey({})),
                                CompileOptionsKey({}), "tier3");
  ASSERT_TRUE(cache
                  .AddUnboundedDynamicCacheEntry(key, CreateDummyExecutable,
                                                 MakeDummyDynamicMetadata())
                  .ok());
  EXPECT_EQ(cache.GetCacheStats().num_compilations, 1);

  // Evict the in-memory cache, and remove the tier-2 entry so that only a
  // tier-3 hit can avoid recompilation.
  cache.EvictAll();
  ASSERT_EQ(unlink(GetTier2CacheEntryPath(key).c_str()), 0);

  ASSERT_TRUE(cache
                  .AddUnboundedDynamicCacheEntry(key, CreateDummyExecutable,
                                                 MakeDummyDynamicMetadata())
                  .ok());
  EXPECT_EQ(cache.GetCacheStats().num_compilations, 1);

  // The tier-3 hit should have backfilled the tier-2 entry. Evict the in-memory
  // cache again and remove the tier-3 entry so that only the backfilled tier-2
  // entry can avoid recompilation.
  cache.EvictAll();
  ASSERT_EQ(unlink(GetTier3CacheEntryPath(key).c_str()), 0);

  ASSERT_TRUE(cache
                  .AddUnboundedDynamicCacheEntry(key, CreateDummyExecutable,
                                                 MakeDummyDynamicMetadata())
                  .ok());
  EXPECT_EQ(cache.GetCacheStats().num_compilations, 1);
}

}  // namespace
}  // namespace torch_tpu
