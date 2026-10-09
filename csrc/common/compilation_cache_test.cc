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

#include <cstdint>
#include <memory>
#include <optional>
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
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache_test_utils.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/contain.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
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
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/device_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_compilation_container);

namespace torch_tpu {

// Friend class for CompilationCache, to allow using private members.
class CompilationCacheTestHelper {
 public:
  static void RestartCompilationCache() { CompilationCache::Restart(); }

  static std::optional<CacheEntryStats> GetUnboundedDynamicCacheEntryStats(
      CompilationCache& cache, const CompilationCacheKey& key) {
    absl::MutexLock lock(cache.cache_mutex_);
    auto it =
        cache.unbounded_dynamic_cache_.find(key.graph_key().shapeless_key());
    if (it == cache.unbounded_dynamic_cache_.end()) {
      return std::nullopt;
    }
    for (auto& entry : it->second) {
      if (key == entry.key) {
        return entry.cache_entry.stats();
      }
    }
    return std::nullopt;
  }
};

namespace {

using testing::ElementsAre;
using testing::IsEmpty;
using testing::MatchesRegex;

CompilationCacheKey DummyKey(int key = 0) {
  return CompilationCacheKey(GraphKey(ShapelessKey(key), DimensionsKey({})),
                             CompileOptionsKey(0));
}

TEST(DeviceIdsOfTest, ListsTheDeviceAssignmentInOrder) {
  xla::CompileOptions compile_options;
  EXPECT_THAT(DeviceIdsOf(compile_options), IsEmpty());

  // One replica with a partition per device of a group, in group order.
  xla::DeviceAssignment group(/*replica_count=*/1, /*computation_count=*/4);
  group(0, 0) = 6;
  group(0, 1) = 7;
  group(0, 2) = 2;
  group(0, 3) = 3;
  compile_options.executable_build_options.set_device_assignment(group);
  EXPECT_THAT(DeviceIdsOf(compile_options), ElementsAre(6, 7, 2, 3));
}

// The position of each device of a host of four two-core chips.
constexpr DevicePosition kHostPositions[] = {
    {0, 0, 0, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}, {1, 0, 0, 1},
    {0, 1, 0, 0}, {0, 1, 0, 1}, {1, 1, 0, 0}, {1, 1, 0, 1}};

std::vector<DevicePosition> PositionsOf(absl::Span<const int64_t> device_ids) {
  std::vector<DevicePosition> positions;
  for (const int64_t device_id : device_ids) {
    positions.push_back(kHostPositions[device_id]);
  }
  return positions;
}

std::vector<int32_t> Dims(const xla::PjRtDeviceDimensions& dims) {
  return std::vector<int32_t>(dims.begin(), dims.end());
}

TEST(OffOriginBoxOfTest, FindsTheBoxOfDevicesOffTheSliceOrigin) {
  // Both cores of the two chips at x = 1.
  std::optional<SubsliceBox> box = OffOriginBoxOf(PositionsOf({6, 7, 2, 3}), 8);
  ASSERT_TRUE(box.has_value());
  EXPECT_THAT(Dims(box->origin), ElementsAre(1, 0, 0));
  EXPECT_THAT(Dims(box->chips), ElementsAre(1, 2, 1));
  // Both cores of one chip.
  box = OffOriginBoxOf(PositionsOf({4, 5}), 8);
  ASSERT_TRUE(box.has_value());
  EXPECT_THAT(Dims(box->origin), ElementsAre(0, 1, 0));
  EXPECT_THAT(Dims(box->chips), ElementsAre(1, 1, 1));
  // One core of each of two chips.
  box = OffOriginBoxOf(PositionsOf({4, 6}), 8);
  ASSERT_TRUE(box.has_value());
  EXPECT_THAT(Dims(box->origin), ElementsAre(0, 1, 0));
  EXPECT_THAT(Dims(box->chips), ElementsAre(2, 1, 1));
}

TEST(OffOriginBoxOfTest, FindsNoBoxAtTheOriginOrForDevicesThatFillNone) {
  EXPECT_FALSE(OffOriginBoxOf({}, 8).has_value());
  // Boxes at the origin.
  EXPECT_FALSE(OffOriginBoxOf(PositionsOf({0, 1, 4, 5}), 8).has_value());
  EXPECT_FALSE(OffOriginBoxOf(PositionsOf({0, 4, 6, 2}), 8).has_value());
  // Both cores of two diagonal chips, and cores of two chips that differ.
  EXPECT_FALSE(OffOriginBoxOf(PositionsOf({4, 5, 2, 3}), 8).has_value());
  EXPECT_FALSE(OffOriginBoxOf(PositionsOf({6, 3}), 8).has_value());
  // A device listed twice, which leaves a place of the box empty.
  EXPECT_FALSE(OffOriginBoxOf(PositionsOf({2, 2, 3, 6}), 8).has_value());
  // Every device of the slice.
  EXPECT_FALSE(
      OffOriginBoxOf(PositionsOf({0, 1, 2, 3, 4, 5, 6, 7}), 8).has_value());
}

TEST(SubsliceTilingOfTest, TilesTheBoxWithProcesses) {
  const auto expect_tiling = [](const xla::PjRtDeviceDimensions& box_chips,
                                const xla::PjRtDeviceDimensions& per_process,
                                std::vector<int32_t> chips,
                                std::vector<int32_t> processes) {
    const auto [tiling_chips, tiling_processes] =
        SubsliceTilingOf(box_chips, per_process);
    EXPECT_EQ(Dims(tiling_chips), chips);
    EXPECT_EQ(Dims(tiling_processes), processes);
  };
  // One chip per process.
  expect_tiling({1, 2, 1}, {1, 1, 1}, {1, 1, 1}, {1, 2, 1});
  // Processes of 2x2x1 chips.
  expect_tiling({2, 2, 1}, {2, 2, 1}, {2, 2, 1}, {1, 1, 1});
  expect_tiling({4, 2, 1}, {2, 2, 1}, {2, 2, 1}, {2, 1, 1});
  expect_tiling({1, 2, 1}, {2, 2, 1}, {1, 2, 1}, {1, 1, 1});
  // A box that processes of 2x2x1 chips do not tile.
  expect_tiling({3, 1, 1}, {2, 2, 1}, {1, 1, 1}, {3, 1, 1});
}

TEST(PerfStatsPrinterTest, EmptyPerEntry) {
  PerfStats stats;
  stats.num_cache_reqs = 10;
  stats.num_cache_hits = 5;
  stats.num_compilations = 3;
  EXPECT_EQ(absl::StrCat(stats),
            "num_cache_reqs=10\nnum_cache_hits=5 "
            "{50.0%}\nnum_compilations=3\n"
            "peak_compilation_memory_bytes=unknown\n");
}

TEST(PerfStatsPrinterTest, WithPerEntry) {
  PerfStats stats;
  stats.num_cache_reqs = 10;
  stats.num_cache_hits = 5;
  stats.num_compilations = 2;
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
num_compilations=2
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
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
    CompilationCacheTestHelper::RestartCompilationCache();
  }
};

TEST_F(CompilationCacheTest, ExecutablesOfAllDevicesLoadWithoutOrigin) {
  TT_ASSERT_OK_AND_ASSIGN(const xla::PjRtClient* const client,
                          PjrtBackend::GetInstance().GetClient());
  std::vector<int64_t> all_devices;  // INT_VEC_OK
  for (const xla::PjRtDevice* const device : client->devices()) {
    all_devices.push_back(device->global_device_id().value());
  }

  TT_ASSERT_OK_AND_ASSIGN(xla::LoadOptions options,
                          LoadOptionsForDevices(*client, {}));
  EXPECT_FALSE(options.computation_origin.has_value());
  TT_ASSERT_OK_AND_ASSIGN(options, LoadOptionsForDevices(*client, all_devices));
  EXPECT_FALSE(options.computation_origin.has_value());
}

TEST_F(CompilationCacheTest, DumpOnMissMode) {
  CompilationCache& cache = CompilationCache::GetInstance();
  bool initial_mode = cache.GetDumpOnCacheMissMode();
  cache.SetDumpOnCacheMissMode(!initial_mode);
  EXPECT_EQ(cache.GetDumpOnCacheMissMode(), !initial_mode);
  cache.SetDumpOnCacheMissMode(initial_mode);
  EXPECT_EQ(cache.GetDumpOnCacheMissMode(), initial_mode);
}

TEST_F(CompilationCacheTest, GetOrCompileLogsOnMiss) {
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
  TT_ASSERT_OK_AND_ASSIGN(
      ContextedModule contexted_module,
      ContextedModule::Make([](mlir::MLIRContext& ctx) {
        return mlir::OwningOpRef<mlir::ModuleOp>(
            mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx)));
      }));
  cache.EnqueueCompilation(key, std::move(contexted_module),
                           std::make_unique<xla::CompileOptions>());
  EXPECT_TRUE(CompilationCache::GetInstance().IsInitialized());

  CompilationCache::ShutDown();
}

TEST_F(CompilationCacheInitTest, OptionsApplied) {
  CompilationCache::Restart();

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
  EXPECT_EQ(stats.num_compilations, 2);

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

TEST_F(CompilationCacheTest, UnboundedDynamicCache) {
  CompilationCache& cache = CompilationCache::GetInstance();

  CompilationCacheKey key(
      GraphKey(ShapelessKey(Fingerprint(std::string_view("dummykey"))),
               DimensionsKey({})),
      CompileOptionsKey({}));

  ASSERT_TRUE(cache
                  .AddUnboundedDynamicCacheEntry(key, CreateDummyExecutable,
                                                 MakeDummyDynamicMetadata())
                  .ok());

  auto initial_stats =
      CompilationCacheTestHelper::GetUnboundedDynamicCacheEntryStats(cache,
                                                                     key);
  ASSERT_TRUE(initial_stats.has_value());
  EXPECT_EQ(initial_stats->read_count, 1);
  const absl::Time before_hit = absl::Now();

  CompilationCacheKey test_key(
      GraphKey(key.graph_key().shapeless_key(), DimensionsKey({1, 2, 3})),
      CompileOptionsKey({}));
  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel result,
      cache.GetOrCompile(
          test_key, {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          [](mlir::MLIRContext& context)
              -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
            return TT_ERROR(error::kInternal)
                   << "Should not be called on a cache hit.";
          },
          std::make_unique<xla::CompileOptions>()));

  TT_ASSERT_OK_AND_ASSIGN(auto actual_exec, result.fixed_shape_kernel.get());
  EXPECT_EQ(cache.GetCacheHits(), 1);
  EXPECT_EQ(cache.GetCacheStats().num_compilations, 1);

  auto updated_stats =
      CompilationCacheTestHelper::GetUnboundedDynamicCacheEntryStats(cache,
                                                                     key);
  ASSERT_TRUE(updated_stats.has_value());
  EXPECT_EQ(updated_stats->read_count, 2);
  EXPECT_GE(updated_stats->last_read, before_hit);

  CompilationCache::ShutDown();
}

TEST_F(CompilationCacheTest, UnboundedDynamicCacheShapeMismatch) {
  CompilationCache& cache = CompilationCache::GetInstance();

  CompilationCacheKey key(
      GraphKey(ShapelessKey(Fingerprint(std::string_view("dummykey"))),
               DimensionsKey({})),
      CompileOptionsKey({}));

  ASSERT_TRUE(cache
                  .AddUnboundedDynamicCacheEntry(key, CreateDummyExecutable,
                                                 MakeDummyDynamicMetadata())
                  .ok());

  CompilationCacheKey test_key(
      GraphKey(key.graph_key().shapeless_key(), DimensionsKey({1, 2, 3})),
      CompileOptionsKey({}));
  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel result,
      cache.GetOrCompile(
          test_key,
          {Shape(Dimensions({128, 128, 256}), mlir::ElementType::F32)},
          {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          [](mlir::MLIRContext& context) {
            return mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
          },
          std::make_unique<xla::CompileOptions>()));

  EXPECT_FALSE(result.fixed_shape_kernel.get().ok());
  EXPECT_EQ(cache.GetCacheHits(), 0);
  EXPECT_EQ(cache.GetCacheMisses(), 1);
  CompilationCache::ShutDown();
}

TEST_F(CompilationCacheTest, UnboundedDynamicCacheOutputShapeMismatch) {
  CompilationCache& cache = CompilationCache::GetInstance();

  CompilationCacheKey key(
      GraphKey(ShapelessKey(Fingerprint(std::string_view("dummykey"))),
               DimensionsKey({})),
      CompileOptionsKey({}));

  ASSERT_TRUE(cache
                  .AddUnboundedDynamicCacheEntry(key, CreateDummyExecutable,
                                                 MakeDummyDynamicMetadata())
                  .ok());

  CompilationCacheKey test_key(
      GraphKey(key.graph_key().shapeless_key(), DimensionsKey({1, 2, 3})),
      CompileOptionsKey({}));
  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel result,
      cache.GetOrCompile(
          test_key, {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          {Shape(Dimensions({128, 128, 256}), mlir::ElementType::F32)},
          [](mlir::MLIRContext& context) {
            return mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
          },
          std::make_unique<xla::CompileOptions>()));

  EXPECT_FALSE(result.fixed_shape_kernel.get().ok());
  EXPECT_EQ(cache.GetCacheHits(), 0);
  EXPECT_EQ(cache.GetCacheMisses(), 1);
  CompilationCache::ShutDown();
}

TEST_F(CompilationCacheTest, UnboundedDynamicCacheMiss) {
  CompilationCache& cache = CompilationCache::GetInstance();

  CompilationCacheKey key(
      GraphKey(ShapelessKey(Fingerprint(std::string_view("dummykey"))),
               DimensionsKey({})),
      CompileOptionsKey({}));

  CompilationCacheKey test_key(
      GraphKey(key.graph_key().shapeless_key(), DimensionsKey({1, 2, 3})),
      CompileOptionsKey({}));
  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel result,
      cache.GetOrCompile(
          test_key, {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          [](mlir::MLIRContext& context) {
            return mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
          },
          std::make_unique<xla::CompileOptions>()));

  EXPECT_FALSE(result.fixed_shape_kernel.get().ok());
  EXPECT_EQ(cache.GetCacheHits(), 0);
  EXPECT_EQ(cache.GetCacheMisses(), 1);
  CompilationCache::ShutDown();
}

TEST_F(CompilationCacheTest, UnboundedDynamicCacheEvictAll) {
  CompilationCache& cache = CompilationCache::GetInstance();

  CompilationCacheKey key(
      GraphKey(ShapelessKey(Fingerprint(std::string_view("dummykey"))),
               DimensionsKey({})),
      CompileOptionsKey({}));

  ASSERT_TRUE(cache
                  .AddUnboundedDynamicCacheEntry(key, CreateDummyExecutable,
                                                 MakeDummyDynamicMetadata())
                  .ok());

  CompilationCacheKey test_key(
      GraphKey(key.graph_key().shapeless_key(), DimensionsKey({1, 2, 3})),
      CompileOptionsKey({}));

  // First lookup hits the unbounded dynamic cache.
  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel result_hit,
      cache.GetOrCompile(
          test_key, {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          [](mlir::MLIRContext& context) {
            return mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
          },
          std::make_unique<xla::CompileOptions>()));
  TT_ASSERT_OK_AND_ASSIGN(auto actual_exec,
                          result_hit.fixed_shape_kernel.get());
  EXPECT_EQ(cache.GetCacheHits(), 1);

  // Evict all cache entries (including unbounded dynamic cache).
  cache.EvictAll();
  EXPECT_FALSE(
      CompilationCacheTestHelper::GetUnboundedDynamicCacheEntryStats(cache, key)
          .has_value());

  // Subsequent lookup misses the cache and attempts compilation.
  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel result_miss,
      cache.GetOrCompile(
          test_key, {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          {Shape(Dimensions({128, 128}), mlir::ElementType::F32)},
          [](mlir::MLIRContext& context) {
            return mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
          },
          std::make_unique<xla::CompileOptions>()));
  EXPECT_FALSE(result_miss.fixed_shape_kernel.get().ok());
  EXPECT_EQ(cache.GetCacheHits(), 1);
  EXPECT_EQ(cache.GetCacheMisses(), 1);
  CompilationCache::ShutDown();
}

TEST_F(CompilationCacheTest, UnboundedDynamicCacheBackendNotInitialized) {
  PjrtBackend::GetInstance().Shutdown();
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "invalid"});
  CompilationCache& cache = CompilationCache::GetInstance();

  CompilationCacheKey key(
      GraphKey(ShapelessKey(Fingerprint(std::string_view("dummykey"))),
               DimensionsKey({})),
      CompileOptionsKey({}));

  const absl::Status status = cache.AddUnboundedDynamicCacheEntry(
      key, CreateDummyExecutable, MakeDummyDynamicMetadata());
  EXPECT_FALSE(status.ok());

  PjrtBackend::GetInstance().Shutdown();
  CompilationCache::ShutDown();
}

}  // namespace
}  // namespace torch_tpu
