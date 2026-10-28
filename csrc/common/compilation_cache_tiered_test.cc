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

#include <memory>
#include <string>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/compilation_cache_test_utils.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/tier2_compilation_cache.h"
#include "csrc/common/tier3_compilation_cache.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/pjrt/pjrt_executable.h"

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
    // Unbounded dynamic kernels are only loaded from the registry when
    // auto-loading is enabled; the env var is read once per process.
    SetEnv<kTorchTpuInternalUseUnboundedDynamicKernelsEnvVar>("1");
  }

  TieredCompilationCacheTest() {
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
    CompilationCache::Restart();
  }

  // Registers a dummy kernel under `key` and returns the lookup key that
  // resolves to it (same shapeless key, different dimensions).
  static CompilationCacheKey RegisterAndMakeLookupKey(
      const CompilationCacheKey& key) {
    RegisterDummyKernel(key);
    return CompilationCacheKey(
        GraphKey(key.graph_key().shapeless_key(), DimensionsKey({128, 128})),
        key.compile_options_key());
  }

  // Looks `lookup_key` up with shapes the dummy kernel accepts. The traced
  // computation must never be built: the registered kernel is used instead.
  static absl::Status GetOrCompileDummy(CompilationCache& cache,
                                        const CompilationCacheKey& lookup_key) {
    TT_ASSIGN_OR_RETURN(
        CompiledKernel kernel,
        cache.GetOrCompile(
            lookup_key, {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
            {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
            [](mlir::MLIRContext&)
                -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
              return TT_ERROR(error::kInternal)
                     << "The traced computation must not be built when a "
                        "compatible unbounded dynamic kernel is registered.";
            },
            std::make_unique<xla::CompileOptions>()));
    return kernel.fixed_shape_kernel.get().status();
  }
};

TEST_F(TieredCompilationCacheTest, UnboundedDynamicCacheTier2Hit) {
  CompilationCache& cache = CompilationCache::GetInstance();
  const CompilationCacheKey key(GraphKey(ShapelessKey(2), DimensionsKey({})),
                                CompileOptionsKey({}), /*id_key=*/2);
  const CompilationCacheKey lookup_key = RegisterAndMakeLookupKey(key);
  EXPECT_TRUE(GetOrCompileDummy(cache, lookup_key).ok());
  EXPECT_EQ(cache.GetCacheStats().num_compilations, 1);

  // Evict the in-memory cache, and remove the tier-3 entry so that only a
  // tier-2 hit can avoid recompilation.
  cache.EvictAll();
  ASSERT_EQ(unlink(GetTier3CacheEntryPath(key).c_str()), 0);

  EXPECT_TRUE(GetOrCompileDummy(cache, lookup_key).ok());
  EXPECT_EQ(cache.GetCacheStats().num_compilations, 1);
}

TEST_F(TieredCompilationCacheTest, UnboundedDynamicCacheTier3Hit) {
  CompilationCache& cache = CompilationCache::GetInstance();
  const CompilationCacheKey key(GraphKey(ShapelessKey(3), DimensionsKey({})),
                                CompileOptionsKey({}), /*id_key=*/3);
  const CompilationCacheKey lookup_key = RegisterAndMakeLookupKey(key);
  EXPECT_TRUE(GetOrCompileDummy(cache, lookup_key).ok());
  EXPECT_EQ(cache.GetCacheStats().num_compilations, 1);

  // Evict the in-memory cache, and remove the tier-2 entry so that the lookup
  // goes to the tier-3 cache.
  cache.EvictAll();
  ASSERT_EQ(unlink(GetTier2CacheEntryPath(key).c_str()), 0);

  EXPECT_TRUE(GetOrCompileDummy(cache, lookup_key).ok());
  // The tier-3 read races a local backup compilation, which may or may not
  // have compiled (or may still be compiling) by now.
  EXPECT_LE(cache.GetCacheStats().num_compilations, 2);

  // The lookup should have backfilled the tier-2 entry. Evict the in-memory
  // cache again and remove the tier-3 entry so that only the backfilled tier-2
  // entry can avoid recompilation.
  cache.EvictAll();
  ASSERT_EQ(unlink(GetTier3CacheEntryPath(key).c_str()), 0);

  EXPECT_TRUE(GetOrCompileDummy(cache, lookup_key).ok());
  // At most the initial compilation and the backup compilation above.
  EXPECT_LE(cache.GetCacheStats().num_compilations, 2);
}

}  // namespace
}  // namespace torch_tpu
