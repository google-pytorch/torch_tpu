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

#include "csrc/common/compilation_cache.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/container/inlined_vector.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/contain.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/flags.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/common/utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/xla.pb.h"

ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_compilation_container);

namespace torch_tpu {

// Friend class for CompilationCache, to allow using private members.
class CompilationCacheTestHelper {
 public:
  static void RestartCompilationCache() { CompilationCache::Restart(); }
};

namespace {

using testing::MatchesRegex;

CompilationCacheKey DummyKey(int key = 0) {
  return CompilationCacheKey(GraphKey(ShapelessKey(key), DimensionsKey({})),
                             CompileOptionsKey(0));
}

TEST(PerfStatsPrinterTest, EmptyPerEntry) {
  PerfStats stats;
  stats.num_cache_reqs = 10;
  stats.num_cache_hits = 5;
  EXPECT_EQ(absl::StrCat(stats),
            "num_cache_reqs=10\nnum_cache_hits=5 "
            "{50.0%}\npeak_compilation_memory_bytes=unknown\n");
}

TEST(PerfStatsPrinterTest, WithPerEntry) {
  PerfStats stats;
  stats.num_cache_reqs = 10;
  stats.num_cache_hits = 5;
  stats.per_entry_stats.push_back({
      {.compilation_duration = absl::Milliseconds(100),
       .last_read = absl::FromUnixMillis(1000),
       .read_count = 10},
      DummyKey(),
  });
  stats.per_entry_stats.push_back({
      {.compilation_duration = absl::Milliseconds(50),
       .last_read = absl::FromUnixMillis(2000),
       .read_count = 5},
      DummyKey(),
  });
  // NOLINTBEGIN
  static constexpr std::string_view kExpected = R"(num_cache_reqs=10
num_cache_hits=5 {50.0%}
peak_compilation_memory_bytes=unknown
num_compilation_events=2
sum_compilation_time=150ms
)";
  // NOLINTEND
  EXPECT_EQ(absl::StrCat(stats), kExpected);
}

TEST(CacheEntryStatsPrinterTest, Works) {
  CacheEntryStats stats;
  stats.compilation_duration = absl::Milliseconds(100);
  stats.last_read = absl::FromUnixMillis(1000);
  stats.read_count = 10;
  EXPECT_THAT(absl::StrCat(stats),
              MatchesRegex(
                  // The timestamp print-out depends on the system time zone, so
                  // we can't match it exactly.
                  "compilation_duration=100ms, last_read=.+, read_count=10"));
}

class CompilationCacheTest : public testing::Test {
 protected:
  CompilationCacheTest() {
    CompilationCacheTestHelper::RestartCompilationCache();
  }
};

TEST_F(CompilationCacheTest, DumpOnMissMode) {
  CompilationCache& cache = CompilationCache::GetInstance();
  bool initial_mode = cache.GetDumpOnCacheMissMode();
  cache.SetDumpOnCacheMissMode(!initial_mode);
  EXPECT_EQ(cache.GetDumpOnCacheMissMode(), !initial_mode);
  cache.SetDumpOnCacheMissMode(initial_mode);
  EXPECT_EQ(cache.GetDumpOnCacheMissMode(), initial_mode);
}

TEST_F(CompilationCacheTest, GetOrCompileLogsOnMiss) {
  // Use xla_cpu for unit testing as it doesn't require real hardware.
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});
  ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());

  CompilationCache& cache = CompilationCache::GetInstance();
  bool initial_mode = cache.GetDumpOnCacheMissMode();
  cache.SetDumpOnCacheMissMode(true);

  // Trigger a miss with a unique key.
  auto key = DummyKey(12345);
  std::vector<Shape> input_shapes;

  MlirComputationBuilder builder = [](mlir::MLIRContext& context) {
    return mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  };

  UniqueCompileOptions compile_options =
      GetCompilationSpec(CompilationMode::kFastCompile).xla_compile_options;

  absl::ScopedMockLog log;
  EXPECT_CALL(
      log,
      Log(absl::LogSeverity::kInfo, testing::_,
          testing::HasSubstr("Dumping StableHLO module due to cache miss")));

  log.StartCapturingLogs();
  auto result =
      cache.GetOrCompile(key, input_shapes, /*output_shapes=*/{},
                         std::move(builder), std::move(compile_options));

  // Clean up.
  cache.SetDumpOnCacheMissMode(initial_mode);
}

class CompilationCacheInitTest : public CompilationCacheTest {};

TEST_F(CompilationCacheInitTest, LazyInitialization) {
  CompilationCache::Restart();
  // Ensure we use xla_cpu for testing to avoid hardware requirements.
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});

  EXPECT_FALSE(CompilationCache::GetInstance().IsInitialized());

  CompilationCache::GetInstance().SetOptions({.cache_only = true});
  // SetOptions now triggers GetInstance(), which creates the object but NOT
  // the heavy initialization (threads).
  EXPECT_FALSE(CompilationCache::GetInstance().IsInitialized());

  CompilationCache& cache = CompilationCache::GetInstance();
  // GetInstance alone does not trigger heavy initialization anymore.
  EXPECT_FALSE(CompilationCache::GetInstance().IsInitialized());
  EXPECT_EQ(&cache, &CompilationCache::GetInstance());

  // Trigger lazy initialization via EnqueueCompilation.
  auto key = DummyKey();
  auto contexted_module_or = ContextedModule::Make([](mlir::MLIRContext& ctx) {
    return mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx)));
  });
  ASSERT_EQ(contexted_module_or.status(), absl::OkStatus());
  cache.EnqueueCompilation(key, *std::move(contexted_module_or),
                           std::make_unique<xla::CompileOptions>());
  EXPECT_TRUE(CompilationCache::GetInstance().IsInitialized());

  CompilationCache::ShutDown();
}

TEST_F(CompilationCacheInitTest, OptionsApplied) {
  CompilationCache::Restart();
  // Ensure we use xla_cpu for testing to avoid hardware requirements.
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});

  CompilationCache::GetInstance().SetOptions({.cache_only = true});
  CompilationCache& cache = CompilationCache::GetInstance();

  // Trigger a cache miss. In cache_only mode, this should return
  // FAILED_PRECONDITION.
  auto key = DummyKey();
  auto status_or = cache.GetOrCompile(
      key, /*input_shapes=*/{}, /*output_shapes=*/{},
      /*computation_builder=*/
      [](mlir::MLIRContext&) { return mlir::OwningOpRef<mlir::ModuleOp>(); },
      /*compile_options=*/std::make_unique<xla::CompileOptions>());

  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().code(), error::kFailedPrecondition);
  EXPECT_THAT(status_or.status().message(),
              testing::HasSubstr("no more compilation should happen"));

  CompilationCache::ShutDown();
}

TEST_F(CompilationCacheTest, AllowCacheModeDisabled) {
  // Use xla_cpu for unit testing as it doesn't require real hardware.
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});
  ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());

  CompilationCache& cache = CompilationCache::GetInstance();
  cache.SetAllowCacheMode(false);

  auto key = DummyKey(9999);
  std::vector<Shape> input_shapes;

  auto make_builder = []() {
    return [](mlir::MLIRContext& context)
               -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
      auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
      mlir::OpBuilder builder(&context);
      builder.setInsertionPointToEnd(module.getBody());

      auto tensor_type = mlir::RankedTensorType::get({4}, builder.getF32Type());
      auto func_type = builder.getFunctionType({tensor_type}, {tensor_type});
      auto func = builder.create<mlir::func::FuncOp>(
          mlir::UnknownLoc::get(&context), "main", func_type);

      auto* entry_block = func.addEntryBlock();
      builder.setInsertionPointToStart(entry_block);

      // Identity operation
      builder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&context),
                                           entry_block->getArgument(0));

      return mlir::OwningOpRef<mlir::ModuleOp>(module);
    };
  };

  auto builder1 = make_builder();
  auto builder2 = make_builder();

  // First compilation.
  TT_ASSERT_OK_AND_ASSIGN(
      auto result1,
      cache.GetOrCompile(key, input_shapes, /*output_shapes=*/{},
                         std::move(builder1),
                         GetCompilationSpec(CompilationMode::kFastCompile)
                             .xla_compile_options));
  TT_ASSERT_OK_AND_ASSIGN(auto exec1, result1.fixed_shape_kernel.get());

  TT_ASSERT_OK_AND_ASSIGN(
      auto result2,
      cache.GetOrCompile(key, input_shapes, /*output_shapes=*/{},
                         std::move(builder2),
                         GetCompilationSpec(CompilationMode::kFastCompile)
                             .xla_compile_options));
  TT_ASSERT_OK_AND_ASSIGN(auto exec2, result2.fixed_shape_kernel.get());

  // Verify they are different executables.
  EXPECT_NE(exec1.get(), exec2.get())
      << "Executables should be different when cache is disabled";

  // Verify cache stats.
  // If cache is disabled, we shouldn't have any hits.
  PerfStats stats = cache.GetCacheStats();
  EXPECT_EQ(stats.num_cache_hits, 0);

  // Clean up.
  cache.SetAllowCacheMode(true);
  CompilationCache::ShutDown();
}

// Must be done before running any tests.
static const bool kSetFlagDone = [] {
  absl::SetFlag(&FLAGS_torch_tpu_internal_enable_compilation_container, true);
  return true;
}();

TEST_F(CompilationCacheTest, PeakMemoryReported) {
  // Use xla_cpu for unit testing as it doesn't require real hardware.
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});
  ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());

  ASSERT_TRUE(
      (GetFlagOnce<bool,
                   &FLAGS_torch_tpu_internal_enable_compilation_container>()));
  torch_tpu::CleanUpContainer();

  CompilationCache& cache = CompilationCache::GetInstance();

  // Request compilation a few times.
  std::vector<SharedLoadedExecutableWithMetadataFuture> futures;
  for (int i = 0; i < 1; ++i) {
    MlirComputationBuilder builder = [](mlir::MLIRContext& context)
        -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
      auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
      mlir::OpBuilder builder(&context);
      builder.setInsertionPointToEnd(module.getBody());

      auto tensor_type = mlir::RankedTensorType::get({4}, builder.getF32Type());
      auto func_type = builder.getFunctionType({tensor_type}, {tensor_type});
      auto func = builder.create<mlir::func::FuncOp>(
          mlir::UnknownLoc::get(&context), "main", func_type);

      auto* entry_block = func.addEntryBlock();
      builder.setInsertionPointToStart(entry_block);

      // Identity operation
      builder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&context),
                                           entry_block->getArgument(0));

      return mlir::OwningOpRef<mlir::ModuleOp>(module);
    };

    UniqueCompileOptions compile_options =
        GetCompilationSpec(CompilationMode::kFastCompile).xla_compile_options;

    auto key = DummyKey(i + 100);
    TT_ASSERT_OK_AND_ASSIGN(auto result,
                            cache.GetOrCompile(key, {}, {}, std::move(builder),
                                               std::move(compile_options)));
    futures.push_back(std::move(result.fixed_shape_kernel));
  }

  // Wait for the compilation just to make sure we get some signal, but
  // it's OK to get metrics without waiting for the compilation to finish.
  for (auto& future : futures) {
    TT_ASSERT_OK_AND_ASSIGN(auto exec, future.get());
    (void)exec;
  }

  PerfStats stats = cache.GetCacheStats();
  ASSERT_TRUE(stats.peak_compilation_memory_bytes.has_value());
#if TT_IS_INTERNAL_TORCH_TPU
  EXPECT_GT(*stats.peak_compilation_memory_bytes, 0);
#else
  // TODO(b/538117859): fix peak compilation memory reported as 0 in OSS.
  EXPECT_GE(*stats.peak_compilation_memory_bytes, 0);
#endif
  ABSL_LOG(INFO) << "Peak compilation memory: "
                 << *stats.peak_compilation_memory_bytes;

  CompilationCache::ShutDown();
}

TEST_F(CompilationCacheTest, DynamicCacheLayoutMismatch) {
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});
  ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());

  CompilationCache& cache = CompilationCache::GetInstance();

  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  TT_ASSERT_OK_AND_ASSIGN(
      mlir::ElementType f32_type,
      torch_tpu::ConvertTo<mlir::ElementType>(builder.getF32Type()));

  Shape dynamic_shape(
      {4}, f32_type,
      absl::InlinedVector<BoundedDynamicDimension, 1>{BoundedDynamicDimension{
          .dimension = 0, .lower_bound = 1, .upper_bound = 4}});

  std::vector<Shape> input_shapes = {dynamic_shape};
  std::vector<Shape> output_shapes = {dynamic_shape};

  ShapelessKey shapeless_key(111);
  ShapeDynamismMetadata dynamism_metadata(input_shapes, output_shapes);
  DimensionsKey dimensions_key(dynamism_metadata);
  GraphKey graph_key(shapeless_key, dimensions_key);

  CompilationCacheKey key1(graph_key, CompileOptionsKey(1));
  CompilationCacheKey key2(graph_key, CompileOptionsKey(2));

  auto make_builder = []() {
    return [](mlir::MLIRContext& context)
               -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
      auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
      mlir::OpBuilder builder(&context);
      builder.setInsertionPointToEnd(module.getBody());

      auto tensor_type = mlir::RankedTensorType::get({4}, builder.getF32Type());
      auto func_type = builder.getFunctionType({tensor_type}, {tensor_type});
      auto func = builder.create<mlir::func::FuncOp>(
          mlir::UnknownLoc::get(&context), "main", func_type);

      auto* entry_block = func.addEntryBlock();
      builder.setInsertionPointToStart(entry_block);

      builder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&context),
                                           entry_block->getArgument(0));

      return mlir::OwningOpRef<mlir::ModuleOp>(module);
    };
  };

  TT_ASSERT_OK_AND_ASSIGN(
      auto result1,
      cache.GetOrCompile(key1, input_shapes, output_shapes, make_builder(),
                         GetCompilationSpec(CompilationMode::kFastCompile)
                             .xla_compile_options));
  TT_ASSERT_OK_AND_ASSIGN(auto exec1, result1.fixed_shape_kernel.get());

  TT_ASSERT_OK_AND_ASSIGN(
      auto result2,
      cache.GetOrCompile(key2, input_shapes, output_shapes, make_builder(),
                         GetCompilationSpec(CompilationMode::kFastCompile)
                             .xla_compile_options));
  TT_ASSERT_OK_AND_ASSIGN(auto exec2, result2.fixed_shape_kernel.get());

  EXPECT_NE(exec1, exec2);

  CompilationCache::ShutDown();
}

}  // namespace
}  // namespace torch_tpu
