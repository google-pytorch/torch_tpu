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

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache_utils.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/contain.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/flags.h"
#include "csrc/common/shape.h"
#include "csrc/common/thread_pool.h"
#include "csrc/common/tier2_compilation_cache.h"
#include "csrc/common/tier3_compilation_cache.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/internal/dynamism/dynamism_ops.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "tsl/platform/numbers.h"
#include "xla/hlo/utils/concurrency/concurrency_utils.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/xla.pb.h"

ABSL_FLAG(int32_t, torch_tpu_internal_num_compilation_threads, 0,
          "The number of threads to use for compilation. If 0, use the default "
          "number of threads based on the number of logical CPUs.");

namespace torch_tpu {

namespace {

// Returns true if the future is ready.
bool IsFutureReady(const SharedLoadedExecutableWithMetadataFuture& future) {
  return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// Returns the value of the NPROC environment variable (a positive integer), or
// 0 if it is not set or has an invalid value. This function is memoized, so
// the environment variable is only read once.
[[nodiscard]] int GetNumProcs() {
  static const int num_procs = [] {
    const auto& env_var = GetEnvOnce<kNprocEnvVar>();
    if (env_var.has_value()) {
      int num_proc;
      if (absl::SimpleAtoi(*env_var, &num_proc) && num_proc > 0) {
        return num_proc;
      }
    }
    return 0;
  }();
  return num_procs;
}

int GetNumCpus() { return sysconf(_SC_NPROCESSORS_ONLN); }

}  // namespace

// Returns the number of threads to use for compilation.
//
// `num_procs` is the value of the `NPROC` environment variable, or 0 if it is
// not set or has an invalid value.
int GetNumCompilationThreads(const int num_procs) {
  // Is the flag set?
  if (const int num_threads =
          GetFlagOnce<int32_t,
                      &FLAGS_torch_tpu_internal_num_compilation_threads>();
      num_threads != 0) {
    ABSL_CHECK_GE(num_threads, 0)  // CRASH_OK
        << "Invalid number of flag "
           "--torch_tpu_internal_num_compilation_threads: "
        << num_threads << ". Must be >= 0.";
    ABSL_VLOG(1) << "Using " << num_threads
                 << " compilation threads based on "
                    "--torch_tpu_internal_num_compilation_threads.";
    return num_threads;
  }

  int num_cpus;
  int num_hyperthreads_per_cpu;
  std::string_view method;
  // The flag is not set. Is NPROC set?
  if (num_procs > 0) {
    num_cpus = num_procs;
    num_hyperthreads_per_cpu = 4;
    method = "NPROC";
  } else {
    // NPROC is not set; use the hardware concurrency, which already accounts
    // for num_hyper-threading.
    num_cpus = GetNumCpus();
    num_hyperthreads_per_cpu =
        std::thread::hardware_concurrency() / GetNumCpus();
    method = "HW concurrency";
  }

  // Take out one entire CPU core so as to not interfere with the PyTorch main
  // thread. If the machine has only 1 CPU core, then use only 1 thread for the
  // compilation cache.
  int concurrency =
      (num_cpus > 1) ? std::max(1, num_cpus - 1) * num_hyperthreads_per_cpu : 1;

  ABSL_VLOG(1) << "Using " << concurrency << " compilation threads based on "
               << method;
  return concurrency;
}

int GetNumCompilationThreads() {
  return GetNumCompilationThreads(GetNumProcs());
}

void CompilationCache::EnsureInitialized() {
  absl::MutexLock lock(cache_mutex_);
  if (initialized_) return;

  // Spawn worker threads in the container for memory tracking, if memory
  // tracking flag is enabled.

  ScopedMemMeasuringContainer container;
  // Force initialization of XLA concurrency threads inside the container so
  // they are tracked.
  xla::concurrency::DefaultExecutor();

  compilation_pool_ = std::make_unique<torch_tpu::ThreadPool>(
      // The actual thread name will be "tf_tt_compile" in logs.
      "tt_compile", GetNumCompilationThreads());

  if (UsesLocalBackupTaskForTier3Read()) {
    backup_compilation_pool_ = std::make_unique<torch_tpu::ThreadPool>(
        // The actual thread name will be "tf_tt_compile2" in
        // logs. Unfortunately, we cannot use a longer, more
        // descriptive name, as that will cause the thread name to
        // be truncated in logs.
        "tt_compile2",
        // Use fewer threads for backup compilations.
        std::max(1, compilation_pool_->NumThreads() / 2),
        // Don't optimize for low latency - we want to leave room
        // for the main compilation pool.
        /*low_latency_hint=*/false);
  }

  initialized_ = true;
}

CompilationCache::CompilationCache() = default;

CompilationCache::~CompilationCache() {
  ABSL_VLOG(1) << "CompilationCache shutting down.";
  compilation_pool_.reset();
  backup_compilation_pool_.reset();
  EvictAll();
  absl::MutexLock lock(cache_mutex_);
  ABSL_LOG(INFO) << "CompilationCache final stats: " << perf_stats_;
}

absl::Mutex CompilationCache::cache_instance_mutex_(absl::kConstInit);

absl_nonnull std::unique_ptr<CompilationCache>&
CompilationCache::GetInstanceNoLock() {
  static absl::NoDestructor<std::unique_ptr<CompilationCache>> instance(
      // Cannot call std::make_unique here as the ctor is private.
      new CompilationCache());
  return *instance;
}

CompilationCache& CompilationCache::GetInstance() {
  absl::MutexLock lock(cache_instance_mutex_);
  auto& instance = GetInstanceNoLock();
  ABSL_CHECK(instance != nullptr)  // CRASH_OK
      << "Cannot use CompilationCache after it has been shut down.";
  return *instance;
}

void CompilationCache::ShutDown() {
  absl::MutexLock lock(cache_instance_mutex_);
  auto& instance = GetInstanceNoLock();
  ABSL_CHECK(instance != nullptr)  // CRASH_OK
      << "The CompilationCache has already been shut down. Don't shut down "
         "twice.";
  instance.reset();
}

void CompilationCache::Restart() {
  absl::MutexLock lock(cache_instance_mutex_);
  auto& instance = GetInstanceNoLock();
  // Cannot call std::make_unique here as the ctor is private.
  instance.reset(new CompilationCache());
}

bool CompilationCache::IsInitialized() const {
  absl::MutexLock lock(cache_mutex_);
  return initialized_;
}

void CompilationCache::SetOptions(
    const CompilationCacheInitializationOptions& options) {
  absl::MutexLock lock(cache_mutex_);
  options_ = options;
  // TODO(jparkerh): Consolidate cache_only_mode_ into options_.cache_only.
  cache_only_mode_ = options.cache_only;
}

// Sets the executable promise for the given key. If the promise is already
// satisfied, this function will do nothing.
static void TrySetExecutablePromise(
    CompilationCacheKey key, LoadedExecutablePromise& promise,
    const absl::StatusOr<SharedLoadedExecutableWithMetadata>& executable) {
  try {
    promise.set_value(std::move(executable));
    ABSL_VLOG(1) << "Set executable for key: " << key;
  } catch (const std::future_error& e) {
    // Check if it's the specific "already satisfied" error.
    if (e.code() == std::future_errc::promise_already_satisfied) {
      // This is expected, as another thread may have already compiled the
      // executable. If we are trying to set a status, print it here so
      // it's logged.
      ABSL_VLOG(1) << "Another thread already set the executable for key: "
                   << key;
      if (!executable.ok()) {
        ABSL_VLOG(1) << "Couldn't set " << key
                     << " with status: " << executable.status();
      }
    } else {
      ABSL_LOG(FATAL)  // CRASH_OK=TorchTPU bug
          << "Unexpected future_error: " << e.what();
    }
  }
}

void CompilationCache::EvictAll() {
  {
    auto stats = GetCacheStats();
    ABSL_VLOG(1) << "Evicted compilation state: " << stats;
    std::for_each(stats.per_entry_stats.begin(), stats.per_entry_stats.end(),
                  [](const auto& entry) { ABSL_VLOG(1) << entry; });
  }
  // Get all existing keys from the cache - these are the entries we need to
  // evict. We cannot promise to evict new entries created during the eviction
  // process, as that work may never end.
  absl::flat_hash_set<CompilationCacheKey, CompilationCacheKey::Hash>
      keys_to_evict;
  {
    absl::MutexLock lock(cache_mutex_);
    for (const auto& [key, cache_entry] : executable_cache_) {
      keys_to_evict.insert(key);
    }
  }

  // Keep evicting compiled keys until there are no more keys to evict,
  // waiting 1 second between iterations to give in-flight compilations a
  // chance to complete and update the cache.
  //
  // We chose a busy-wait loop as the alternative is much more complex.
  for (; !keys_to_evict.empty(); absl::SleepFor(absl::Seconds(1))) {
    absl::MutexLock lock(cache_mutex_);
    const auto keys = keys_to_evict;
    // Loop over a copy of keys_to_evict, as we will modify it during
    // the loop and invalidate its iterator.
    for (const auto key : keys) {
      const auto it = executable_cache_.find(key);
      if (it == executable_cache_.end()) {
        ABSL_VLOG(1) << "Key already evicted by another thread: " << key;
        keys_to_evict.erase(key);
        continue;
      }

      const auto& cache_entry = it->second;
      ABSL_VLOG_EVERY_N(1, 100)
          << "Evicting compilation future for key: " << key;
      if (IsFutureReady(cache_entry.executable_future())) {
        // The compilation is completed, so we can evict it immediately.
        executable_cache_.erase(it);
        keys_to_evict.erase(key);
        ABSL_VLOG(1) << "Evicted compilation future for key: " << key;
      } else {
        ABSL_VLOG_EVERY_N(1, 100)
            << "Waiting for compilation to complete for key: " << key;
      }
    }
    // Release the lock here to give in-flight compilations a chance to
    // complete and update the cache.
  }
  // Clear bounded dynamic cache entries.
  {
    absl::MutexLock lock(cache_mutex_);
    bounded_dynamic_cache_.clear();
  }

  ABSL_LOG(INFO) << "Compilation cache evicted.";
}

void CompilationCache::SetAllowCacheMode(bool allow) {
  absl::MutexLock lock(cache_mutex_);
  allow_cache_mode_ = allow;
  ABSL_LOG(INFO) << "CompilationCache allow-cache mode set to: " << allow;
}

void CompilationCache::SetCacheOnlyMode(bool cache_only) {
  absl::MutexLock lock(cache_mutex_);
  cache_only_mode_ = cache_only;
  ABSL_LOG(INFO) << "CompilationCache cache-only mode set to: " << cache_only;
}

void CompilationCache::SetDumpOnCacheMissMode(bool enable) {
  absl::MutexLock lock(cache_mutex_);
  dump_on_cache_miss_ = enable;
  ABSL_LOG(INFO) << "CompilationCache dump-on-miss mode set to: " << enable;
}

bool CompilationCache::GetDumpOnCacheMissMode() const {
  absl::MutexLock lock(cache_mutex_);
  return dump_on_cache_miss_;
}

int64_t CompilationCache::GetCacheRequests() const {
  absl::MutexLock lock(cache_mutex_);
  return perf_stats_.num_cache_reqs;
}

int64_t CompilationCache::GetCacheHits() const {
  absl::MutexLock lock(cache_mutex_);
  return perf_stats_.num_cache_hits;
}

int64_t CompilationCache::GetCacheMisses() const {
  absl::MutexLock lock(cache_mutex_);
  return perf_stats_.num_cache_misses();
}

PerfStats CompilationCache::GetCacheStats() const {
  absl::MutexLock lock(cache_mutex_);
  PerfStats stats = perf_stats_;

  auto peak_mem_or = ContainerPeakHostMemoryBytes();
  if (peak_mem_or.ok()) {
    stats.peak_compilation_memory_bytes = peak_mem_or.value();
  } else {
    ABSL_LOG(WARNING) << "Failed to get peak memory usage for stats: "
                      << peak_mem_or.status();
  }

  stats.per_entry_stats.reserve(executable_cache_.size());
  for (const auto& [key, cache_entry] : executable_cache_) {
    stats.per_entry_stats.push_back({
        {cache_entry.stats()},
        key,
    });
  }
  return stats;
}

const CacheEntry* CompilationCache::FindStaticCacheEntryOrNull(
    CompilationCacheKey key) const {
  const auto it = executable_cache_.find(key);
  return it == executable_cache_.end() ? nullptr : &it->second;
}

std::optional<CompilationCache::CacheLookupInternal>
CompilationCache::GetStaticCacheEntry(CompilationCacheKey key) const {
  const CacheEntry* cache_entry = FindStaticCacheEntryOrNull(key);
  if (cache_entry == nullptr) {
    return std::nullopt;
  }

  perf_stats_.num_cache_hits++;
  cache_entry->stats().read_count++;
  cache_entry->stats().last_read = absl::Now();
  ABSL_VLOG(2) << "Compilation cache STATIC HIT for key: " << key;
  return CacheLookupInternal{
      .executable_promise = cache_entry->executable_promise(),
      .executable_future = cache_entry->executable_future(),
      .needs_compilation = false,
      .dump_on_cache_miss = false,
  };
}

std::optional<CompilationCache::BoundedDynamicCache::iterator>
CompilationCache::GetBoundedDynamicCacheEntries(ShapelessKey shapeless_key) {
  if (auto it = bounded_dynamic_cache_.find(shapeless_key);
      it != bounded_dynamic_cache_.end()) {
    return it;
  }
  return std::nullopt;
}

CompilationCache::CacheLookupInternal CompilationCache::AddStaticCacheEntry(
    CompilationCacheKey key) {
  const auto [entry, inserted] = executable_cache_.insert({key, CacheEntry()});
  ABSL_CHECK(inserted)  // CRASH_OK
      << "Key already exists in static cache: " << key;
  return CacheLookupInternal{
      .executable_promise = entry->second.executable_promise(),
      .executable_future = entry->second.executable_future(),
      .needs_compilation = true,
      .dump_on_cache_miss = false};
}

CompilationCache::CacheLookupInternal
CompilationCache::AddBoundedDynamicCacheEntry(
    ShapelessKey shapeless_key, CompileOptionsKey compile_options_key,
    ShapeDynamismMetadata shape_dynamism_metadata,
    std::optional<CompilationCache::BoundedDynamicCache::iterator> dynamic_it) {
  std::vector<BoundedDynamicCacheEntry>* entries;
  if (dynamic_it.has_value()) {
    ABSL_CHECK(shapeless_key == (*dynamic_it)->first)  // CRASH_OK
        << "Shapeless key does not match dynamic iterator"
        << ToString(shapeless_key)
        << ", dynamic iterator key: " << ToString((*dynamic_it)->first);
    entries = &(*dynamic_it)->second;
  } else {
    auto [entry, inserted] = bounded_dynamic_cache_.insert(
        {shapeless_key, std::vector<BoundedDynamicCacheEntry>()});
    ABSL_CHECK(inserted)  // CRASH_OK
        << "Key already exists in bounded dynamic cache: "
        << ToString(shapeless_key);
    entries = &entry->second;
  }

  const GraphKey graph_key(shapeless_key,
                           DimensionsKey(shape_dynamism_metadata));
  CompilationCacheKey middle_executable_key(graph_key, compile_options_key);
  entries->push_back(BoundedDynamicCacheEntry{
      .middle_executable_key = middle_executable_key,
      .shape_dynamism_metadata = shape_dynamism_metadata,
  });

  auto cache_lookup = AddStaticCacheEntry(middle_executable_key);
  cache_lookup.shape_dynamism_metadata = shape_dynamism_metadata;
  return cache_lookup;
}

absl::StatusOr<CompilationCache::CacheLookupInternal>
CompilationCache::GetOrCreateCacheEntry(
    CompilationCacheKey key, const std::vector<Shape>& input_shapes,
    const std::vector<Shape>& output_shapes,
    bool skip_dynamic_lookup_and_compilation) {
  if (!allow_cache_mode_) {
    ABSL_LOG(WARNING) << "CompilationCache is disabled. key: " << key;
    // We insert an entry into the cache anyway so we can keep track of it and
    // safely evict it later.
    auto cache_lookup_or = GetStaticCacheEntry(key);
    if (cache_lookup_or.has_value()) {
      return *cache_lookup_or;
    }
    return AddStaticCacheEntry(key);
  }

  // 1. Try to find a static CacheEntry.
  if (const auto cache_lookup_or = GetStaticCacheEntry(key);
      cache_lookup_or.has_value()) {
    return *cache_lookup_or;
  }
  ABSL_VLOG(2) << "Compilation cache STATIC MISS #"
               << perf_stats_.num_cache_misses() << " for key: " << key;

  // 2. Try to find a dynamic cache entry. We hold on to the iterator to avoid
  // having to lookup the shapeless key again later.
  std::optional<BoundedDynamicCache::iterator> dynamic_it =
      skip_dynamic_lookup_and_compilation
          ? std::nullopt
          : GetBoundedDynamicCacheEntries(key.graph_key().shapeless_key());
  if (dynamic_it.has_value()) {
    for (const auto& entry : (*dynamic_it)->second) {
      if (entry.shape_dynamism_metadata.IsStaticShapeCompatible(input_shapes) &&
          entry.middle_executable_key.compile_options_key() ==
              key.compile_options_key()) {
        ABSL_VLOG(2) << "Compilation cache DYNAMIC HIT for key: " << key;

        TT_ASSIGN_OR_RETURN(
            auto cache_lookup,
            GetOrCreateCacheEntry(
                entry.middle_executable_key, input_shapes, output_shapes,
                /*skip_dynamic_lookup_and_compilation=*/true));
        cache_lookup.shape_dynamism_metadata = entry.shape_dynamism_metadata;
        return cache_lookup;
      }
    }
  }

  ABSL_VLOG(2) << "Compilation cache DYNAMIC MISS for key: " << key;

  TT_RET_CHECK(!cache_only_mode_, error::kFailedPrecondition)
      << "The user has asserted that no more compilation should happen; yet "
         "compilation is needed for key: "
      << key;

  // 3. We didn't find a compatible entry, so we need to create a new one.
  ABSL_VLOG(2) << "[TtPerf] Compilation cache MISS #"
               << perf_stats_.num_cache_misses() << " for key: " << key;

  // Do we need to create a dynamic cache entry?
  bool create_dynamic_entry =
      !skip_dynamic_lookup_and_compilation &&
      std::any_of(input_shapes.begin(), input_shapes.end(), [](const Shape& s) {
        return !s.dynamic_dimensions().empty();
      });
  if (create_dynamic_entry) {
    auto dynamism_metadata = ShapeDynamismMetadata(input_shapes, output_shapes);
    ABSL_VLOG(2) << "Creating a dynamic cache entry for key: " << key;
    return AddBoundedDynamicCacheEntry(key.graph_key().shapeless_key(),
                                       key.compile_options_key(),
                                       dynamism_metadata, dynamic_it);
  }

  ABSL_VLOG(2) << "Creating a static cache entry for key: " << key;
  // 4. Lastly, we create a static cache entry.
  return AddStaticCacheEntry(key);
}

bool CompilationCache::IsExecutableReady(CompilationCacheKey key) const {
  absl::MutexLock lock(cache_mutex_);
  if (const CacheEntry* cache_entry = FindStaticCacheEntryOrNull(key);
      cache_entry != nullptr) {
    return IsFutureReady(cache_entry->executable_future());
  }
  return false;
}

namespace {

// Returns true if the default layout is different between any input / output
// shape pair. If we are unable to retrieve layouts, or the vector sizes don't
// match, we return true.
bool AnyInducesLayoutChange(const std::vector<Shape>& input_shapes,
                            const std::vector<Shape>& output_shapes,
                            xla::PjRtClient* client) {
  if (input_shapes.size() != output_shapes.size()) {
    return true;
  }
  for (int i = 0; i < input_shapes.size(); ++i) {
    const Shape& input_shape = input_shapes[i];
    auto input_element_type =
        ConvertTo<xla::PrimitiveType>(input_shape.dtype());
    auto input_layout =
        client->GetDefaultLayout(input_element_type, input_shape.dimensions());
    const Shape& output_shape = output_shapes[i];
    auto output_element_type =
        ConvertTo<xla::PrimitiveType>(output_shape.dtype());
    auto output_layout = client->GetDefaultLayout(output_element_type,
                                                  output_shape.dimensions());
    // If we were unable to retrieve layouts, we assume the layout has changed.
    if (!input_layout.ok() || !output_layout.ok() ||
        input_layout != output_layout) {
      ABSL_VLOG(3) << "[AnyInducesLayoutChange] Input layout: " << input_layout
                   << " Output layout: " << output_layout
                   << " Input shape: " << ToString(input_shape.dimensions())
                   << " Output shape: " << ToString(output_shape.dimensions());
      return true;
    }
  }
  return false;
}

// Resolves runtime dynamic input shapes, static padded input shapes and updated
// dynamism shapes based on shape dynamism metadata.
void ResolvePaddingShapes(
    const ShapeDynamismMetadata& shape_dynamism_metadata,
    const std::vector<Shape>& input_shapes,
    std::vector<Shape>& static_runtime_input_shapes,
    std::vector<Shape>& static_padded_input_shapes,
    std::vector<Shape>& input_shapes_with_updated_dynamism) {
  absl::Span<const DimensionBounds> input_dimension_bounds =
      shape_dynamism_metadata.input_dimension_bounds();
  for (int i = 0; i < input_shapes.size(); ++i) {
    const Shape& shape = input_shapes[i];
    absl::Span<const DimensionBounds> bounds =
        input_dimension_bounds.first(shape.dimensions().size());
    input_dimension_bounds =
        input_dimension_bounds.subspan(shape.dimensions().size());
    Dimensions dimensions = shape.dimensions();
    static_runtime_input_shapes.push_back(Shape(dimensions, shape.dtype()));
    Shape dynamic_shape = Shape(dimensions, shape.dtype());
    for (int j = 0; j < bounds.size(); ++j) {
      if (bounds[j].lower != bounds[j].upper) {
        dynamic_shape.dynamic_dimensions().push_back(BoundedDynamicDimension{
            .dimension = j,
            .lower_bound = bounds[j].lower,
            .upper_bound = bounds[j].upper,
        });
      }
    }
    static_padded_input_shapes.push_back(
        Shape(GetUpperBounds(bounds), shape.dtype()));
    input_shapes_with_updated_dynamism.push_back(std::move(dynamic_shape));
  }
}

}  // namespace

// Creates a padding kernel for the dynamic kernel adapter.
absl::StatusOr<SharedLoadedExecutableWithMetadataFuture>
CompilationCache::CreatePaddingKernel(
    const ShapeDynamismMetadata& shape_dynamism_metadata,
    const std::vector<Shape>& static_runtime_input_shapes,
    const std::vector<Shape>& static_padded_input_shapes,
    std::vector<Shape> input_shapes_with_updated_dynamism,
    CompileOptionsKey compile_options_key,
    UniqueCompileOptions compile_options) {
  CompilationCacheKey padding_cache_key(
      PadModuleCacheKey(input_shapes_with_updated_dynamism),
      compile_options_key);

  MlirComputationBuilder padding_module_builder =
      [input_shapes_with_updated_dynamism =
           std::move(input_shapes_with_updated_dynamism)](
          mlir::MLIRContext& mlir_context) {
        return GetPadModule(mlir_context, input_shapes_with_updated_dynamism);
      };

  TT_ASSIGN_OR_RETURN(
      CompiledKernel padding_kernel,
      GetOrCompile(padding_cache_key, static_runtime_input_shapes,
                   static_padded_input_shapes,
                   std::move(padding_module_builder),
                   std::move(compile_options)));
  return std::move(padding_kernel.fixed_shape_kernel);
}

// Creates a slicing kernel for the dynamic kernel adapter.
absl::StatusOr<SharedLoadedExecutableWithMetadataFuture>
CompilationCache::CreateSlicingKernel(
    const ShapeDynamismMetadata& shape_dynamism_metadata,
    const std::vector<Shape>& output_shapes,
    CompileOptionsKey compile_options_key,
    UniqueCompileOptions compile_options) {
  std::vector<Dimensions> runtime_output_dims_vec;
  runtime_output_dims_vec.reserve(output_shapes.size());
  for (const auto& shape : output_shapes) {
    runtime_output_dims_vec.push_back(shape.dimensions());
  }
  std::vector<mlir::ElementType> output_dtypes;
  output_dtypes.reserve(output_shapes.size());
  for (const auto& shape : output_shapes) {
    output_dtypes.push_back(shape.dtype());
  }
  std::vector<Dimensions> padded_output_dims_vec;
  padded_output_dims_vec.reserve(runtime_output_dims_vec.size());
  absl::Span<const DimensionBounds> output_dimension_bounds =
      shape_dynamism_metadata.output_dimension_bounds();
  for (const auto& dims : runtime_output_dims_vec) {
    absl::Span<const DimensionBounds> bounds =
        output_dimension_bounds.first(dims.size());
    output_dimension_bounds = output_dimension_bounds.subspan(dims.size());
    padded_output_dims_vec.push_back(GetUpperBounds(bounds));
  }

  std::vector<Shape> runtime_output_shapes;
  runtime_output_shapes.reserve(output_shapes.size());
  for (int i = 0; i < output_shapes.size(); ++i) {
    runtime_output_shapes.push_back(
        Shape(runtime_output_dims_vec[i], output_dtypes[i]));
  }

  std::vector<Shape> padded_output_shapes;
  padded_output_shapes.reserve(output_shapes.size());
  for (int i = 0; i < output_shapes.size(); ++i) {
    padded_output_shapes.push_back(
        Shape(padded_output_dims_vec[i], output_dtypes[i]));
  }

  CompilationCacheKey slicing_cache_key(
      shape_dynamism_metadata.GetSliceModuleCacheKey(runtime_output_shapes),
      compile_options_key);

  MlirComputationBuilder slice_module_builder =
      [runtime_output_dims_vec = std::move(runtime_output_dims_vec),
       padded_output_dims_vec = std::move(padded_output_dims_vec),
       output_dtypes =
           std::move(output_dtypes)](mlir::MLIRContext& mlir_context) {
        return GetSliceModule(mlir_context, runtime_output_dims_vec,
                              padded_output_dims_vec, output_dtypes);
      };

  TT_ASSIGN_OR_RETURN(
      CompiledKernel slice_kernel,
      GetOrCompile(slicing_cache_key, padded_output_shapes, output_shapes,
                   std::move(slice_module_builder),
                   std::move(compile_options)));
  return std::move(slice_kernel.fixed_shape_kernel);
}

// Creates a dynamic kernel adapter for the given shape dynamism metadata, input
// shapes, output shapes, and compile options. If the padding operation does
// not induce a layout change, returns an adapter with only the padding kernel.
absl::StatusOr<DynamicKernelAdapter>
CompilationCache::CreateDynamicKernelAdapter(
    const ShapeDynamismMetadata& shape_dynamism_metadata,
    const std::vector<Shape>& input_shapes,
    const std::vector<Shape>& output_shapes,
    CompileOptionsKey compile_options_key,
    UniqueCompileOptions compile_options) {
  DynamicKernelAdapter adapter;
  // TODO(unda): is it possible to reuse the compile options? We make a copy
  // for now.
  // Do this first, before we move the compile options.
  auto padding_compile_options =
      std::make_unique<xla::CompileOptions>(*compile_options);

  std::vector<Shape> static_runtime_input_shapes;
  std::vector<Shape> static_padded_input_shapes;
  std::vector<Shape> input_shapes_with_updated_dynamism;

  ResolvePaddingShapes(shape_dynamism_metadata, input_shapes,
                       static_runtime_input_shapes, static_padded_input_shapes,
                       input_shapes_with_updated_dynamism);

  TT_ASSIGN_OR_RETURN(
      adapter.preamble,
      CreatePaddingKernel(shape_dynamism_metadata, static_runtime_input_shapes,
                          static_padded_input_shapes,
                          std::move(input_shapes_with_updated_dynamism),
                          compile_options_key,
                          std::move(padding_compile_options)));

  // Before doing any work for the slicing kernel, we check if we need to do it
  // at all. We know that if the padding kernel doesn't induce a layout change,
  // then the current mechanisms in the TPU backend already handle the slicing
  // for us.
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  if (!AnyInducesLayoutChange(static_runtime_input_shapes,
                              static_padded_input_shapes, client)) {
    return adapter;
  }

  ABSL_VLOG(1) << "A padding operation induces a layout change, adding a "
                  "slicing kernel to the dynamic adapter.";

  TT_ASSIGN_OR_RETURN(
      adapter.postamble,
      CreateSlicingKernel(shape_dynamism_metadata, output_shapes,
                          compile_options_key, std::move(compile_options)));
  return adapter;
}

absl::StatusOr<CompiledKernel> CompilationCache::GetOrCompile(
    const CompilationCacheKey key, const std::vector<Shape>& input_shapes,
    const std::vector<Shape>& output_shapes,
    MlirComputationBuilder computation_builder,
    UniqueCompileOptions compile_options, bool use_dynamic_adapters) {
  // Critical section for cache lookups and insertion.
  TT_ASSIGN_OR_RETURN(
      auto cache_lookup, [&]() -> absl::StatusOr<CacheLookupInternal> {
        absl::MutexLock lock(cache_mutex_);
        perf_stats_.num_cache_reqs++;
        TT_ASSIGN_OR_RETURN(auto lookup, GetOrCreateCacheEntry(
                                             key, input_shapes, output_shapes));
        lookup.dump_on_cache_miss = dump_on_cache_miss_;
        return lookup;
      }());
  // Everything we are recovering from the cache is a shared pointer, so it
  // is safe to access them without the lock.

  CompiledKernel compiled_kernel{.fixed_shape_kernel =
                                     cache_lookup.executable_future};
  CompilationCacheKey storage_key = key;
  if (cache_lookup.shape_dynamism_metadata.has_value()) {
    // Create a key for the storage of the dynamic executable.
    const GraphKey graph_key(
        key.graph_key().shapeless_key(),
        DimensionsKey(*cache_lookup.shape_dynamism_metadata));
    storage_key = CompilationCacheKey(graph_key, key.compile_options_key());
    ABSL_VLOG(2) << "Storage key for dynamic executable: " << storage_key;

    if (use_dynamic_adapters) {
      ABSL_VLOG(1) << "Found shape dynamism metadata for key: " << key;
      // Create a copy of the compile options for the adapter.
      auto adapter_compile_options =
          std::make_unique<xla::CompileOptions>(*compile_options);
      TT_ASSIGN_OR_RETURN(
          compiled_kernel.dynamic_kernel_adapter,
          CreateDynamicKernelAdapter(*cache_lookup.shape_dynamism_metadata,
                                     input_shapes, output_shapes,
                                     key.compile_options_key(),
                                     std::move(adapter_compile_options)));
    }
  }

  if (cache_lookup.needs_compilation) {
    // Only create the contexted module if we need to compile.
    auto contexted_module_or = ContextedModule::Make(computation_builder);
    if (!contexted_module_or.ok()) {
      TrySetExecutablePromise(storage_key, *cache_lookup.executable_promise,
                              contexted_module_or.status());
      return contexted_module_or.status();
    }
    if (cache_lookup.dump_on_cache_miss) {
      LogLines(absl::StrCat(
          "Dumping StableHLO module due to cache miss for key: ", storage_key,
          "\n",
          DebugString(contexted_module_or->get(),
                      DebugStringOptions::kEnableDebugInfo)));
    }
    EnqueueCompilation(storage_key, *std::move(contexted_module_or),
                       std::move(compile_options));
    ABSL_VLOG(1) << "[TtPerf] Scheduled compilation for key: " << storage_key;
  }

  return compiled_kernel;
}

void CompilationCache::EnqueueCompilation(
    CompilationCacheKey key, ContextedModule contexted_module,
    UniqueCompileOptions compile_options) {
  EnsureInitialized();
  auto executable_builder = [contexted_module = std::move(contexted_module)](
                                xla::PjRtClient& client,
                                UniqueCompileOptions options) mutable {
    return AdaptXlaError(
        client.CompileAndLoad(
            std::move(contexted_module).ToMaybeOwningMlirModule(),
            std::move(*options)),
        /* context= */ "failed to compile MLIR module");
  };

  compilation_pool_->Schedule(
      [this, key, builder = std::move(executable_builder),
       compile_options = std::move(compile_options)]() mutable {
        // Prevent compilation threads from accessing thread-local context
        // states to enforce they always rely on resolved compiler options
        // passed down from the dispatch thread.
        DisallowThisThreadToAccessContextState();
        torch_tpu::ScopedMemMeasuringContainer container;
        this->GetFromTier2OrCompile(std::move(key), std::move(builder),
                                    std::move(compile_options));
      });
}

void CompilationCache::SetExecutable(
    CompilationCacheKey key,
    absl::StatusOr<SharedLoadedExecutableWithMetadata> executable,
    CacheEntryStats stats) {
  absl::MutexLock lock(cache_mutex_);
  // If the user requested to evict the cache while we were compiling this
  // executable, the key may be missing. In this case, adding the key back to
  // the cache doesn't help as EvictAll() already set the executable future
  // to a failed state, so we just log and return.
  const auto it = executable_cache_.find(key);
  if (it == executable_cache_.end()) {
    ABSL_LOG(WARNING) << "Key already evicted when setting executable for key "
                      << key;
    return;
  }

  const CacheEntry& cache_entry = it->second;
  if (IsFutureReady(cache_entry.executable_future())) {
    // Another thread has already set the executable future.
    return;
  }

  cache_entry.stats() = std::move(stats);
  if (cache_entry.stats().compilation_duration > absl::ZeroDuration()) {
    ABSL_VLOG(1) << "Compile duration for key " << key << ": "
                 << absl::ToInt64Milliseconds(
                        cache_entry.stats().compilation_duration)
                 << " ms";
  }

  TrySetExecutablePromise(key, *cache_entry.executable_promise(),
                          std::move(executable));

  if (!allow_cache_mode_) {
    // If the cache is disabled, we remove the executable from the cache.
    // Note that this is done after the executable future is set, so a
    // user holding another shared pointer to the executable future will still
    // be able to retrieve the executable.
    executable_cache_.erase(it);
    ABSL_VLOG(1) << "Evicted key from cache because allow_cache_mode is false: "
                 << key;
  }
}

bool CompilationCache::ShouldUseTier2Cache() const {
  bool allow_cache = true;
  {
    absl::MutexLock lock(cache_mutex_);
    allow_cache = allow_cache_mode_;
  }
  return allow_cache && UsesTier2CompilationCache();
}

void CompilationCache::GetFromTier2OrCompile(
    CompilationCacheKey key, LoadedExecutableBuilder executable_builder,
    UniqueCompileOptions compile_options) {
  const bool uses_tier2 = ShouldUseTier2Cache();
  ABSL_VLOG(1) << "Compiling executable for key: " << key
               << (uses_tier2 ? absl::StrCat(" with tier-2 cache at ",
                                             GetTier2CompilationCachePath())
                              : "");

  if (!uses_tier2) {
    Compile(std::move(key), std::move(executable_builder),
            std::move(compile_options));
    return;
  }

  // Using tier-2 cache.
  const absl::Time request_start = absl::Now();

  // Check 1: if the compilation result is already in the tier-2 cache, use it.
  Tier2CacheEntryStats tier2_stats;
  auto executable_or = GetFromTier2Cache(key, request_start, tier2_stats);
  if (executable_or.ok()) {
    SetExecutable(key, std::move(executable_or),
                  {.tier2 = std::move(tier2_stats)});
    return;
  }
  ABSL_VLOG(2) << "Tier-2 cache MISS for key: " << key;

  // Otherwise, try to compile the graph.
  // Critical section for updating the tier-2 cache entry.
  {
    // Lock the tier-2 cache entry for the key to avoid multiple processes
    // doing the same compilation. This is a blocking call.
    Tier2CacheEntryLock lock(key);
    {
      // Check again if the compilation result is already in the tier-2 cache,
      // in case another thread or process has just finished the compilation and
      // released the lock.
      Tier2CacheEntryStats tier2_stats;
      auto executable_or = GetFromTier2Cache(key, request_start, tier2_stats);
      if (executable_or.ok()) {
        SetExecutable(key, std::move(executable_or),
                      {.tier2 = std::move(tier2_stats)});
        return;
      }
    }

    GetFromTier3OrCompile(std::move(key), std::move(executable_builder),
                          std::move(compile_options), lock, request_start);
  }  // End of critical section.
}

void CompilationCache::GetFromTier3OrCompile(
    CompilationCacheKey key, LoadedExecutableBuilder executable_builder,
    UniqueCompileOptions compile_options, Tier2CacheEntryLock& lock,
    const absl::Time request_start) {
  const bool uses_tier3 = UsesTier3CompilationCache();
  const auto pre_compile_duration = absl::Now() - request_start;
  absl::StatusOr<SharedLoadedExecutableWithMetadata> executable_or;
  CacheTier tier = CacheTier::kUnknown;

  const bool backup_compilation = UsesLocalBackupTaskForTier3Read();
  if (backup_compilation) {
    backup_compilation_pool_->Schedule(
        [this, key, builder = std::move(executable_builder),
         options = std::move(compile_options)]() mutable {
          // Prevent backup compilation threads from accessing thread-local
          // context states to enforce they always rely on resolved
          // compiler options passed down from the dispatch thread.
          DisallowThisThreadToAccessContextState();
          // As an optimization, if the tier-3 cache read has already
          // populated the executable, skip the backup compilation.
          if (this->IsExecutableReady(key)) {
            ABSL_VLOG(1) << "Skipping backup compilation for key: " << key
                         << " because it is already ready.";
            return;
          }
          this->Compile(key, std::move(builder), std::move(options));
        });
  }

  // If tier-3 cache is enabled, try to get the executable from it first.
  if (uses_tier3) {
    executable_or = GetFromTier3Cache(key);
    if (executable_or.ok()) {
      ABSL_VLOG(1) << "Tier-3 cache HIT for key " << key;
      tier = CacheTier::kTier3;
      SetExecutable(key, executable_or, /*stats=*/{});
    } else {
      ABSL_VLOG(1) << "Tier-3 cache MISS for key " << key
                   << " with status: " << executable_or.status();
    }
  }

  if (!executable_or.ok()) {
    // Either tier-3 cache is disabled, or the executable was not found in it.
    // Compile the graph and save the result to the tier-1 cache.
    tier = CacheTier::kTier1;
    if (!backup_compilation) {
      // Clang-tidy reports a false positive here that executable_builder and
      // compile_options are used after being moved in the backup compilation
      // above. It is wrong as that move only happens when backup_compilation is
      // true.
      Compile(key, std::move(executable_builder),  // NOLINT
              std::move(compile_options));         // NOLINT
    }
    // When the above call finishes, the tier-1 cache will contain the compiled
    // executable and its initial stats.
  }

  // Read the tier-1 cache.
  SharedLoadedExecutableWithMetadataFuture f;
  {
    absl::MutexLock lock(cache_mutex_);
    // Return early if the cache entry has been concurrently evicted.
    const CacheEntry* cache_entry = FindStaticCacheEntryOrNull(key);
    if (cache_entry == nullptr) {
      ABSL_VLOG(1)
          << "Cache entry already evicted before reading tier-1 cache for key: "
          << key;
      return;
    }
    f = cache_entry->executable_future();
  }
  // Important: the .get() must be called outside the lock region to avoid a
  // deadlock. For example, if this function (GetFromTier3OrCompile) scheduled
  // a backup compilation above, and the future is not ready yet, f.get() will
  // block until the backup compilation finishes and sets the future. However,
  // the backup compilation is running in a separate thread and won't be able
  // to set the future as the lock is still held by this thread.
  executable_or = f.get();

  if (!executable_or.ok()) {
    // The compilation failed.
    return;
  }

  // Write to tier-2 cache.
  const absl::Time write_start = absl::Now();
  const std::string tier2_entry_path = GetTier2CacheEntryPath(key);
  ABSL_VLOG(1) << "Writing to tier-2 cache for key " << key << " at path "
               << tier2_entry_path;
  const absl::Status status =
      AtomicWriteToCacheFile(tier2_entry_path, *executable_or);
  // Release the lock on the tier-2 cache entry now to unblock
  // processes that are waiting for it sooner.
  lock.Release();

  const absl::Duration write_duration = absl::Now() - write_start;
  if (!status.ok()) {
    // Writing to the tier-2 cache is best-effort - we don't need to fail
    // the user if it fails (the program will just run slower).
    ABSL_LOG(ERROR) << status;
  } else {
    ABSL_VLOG(2) << "Tier-2 cache WRITE for key: " << key
                 << "\n  Pre-compile wait: " << pre_compile_duration
                 << "\n  Write duration: " << write_duration;
    absl::MutexLock lock(cache_mutex_);
    // Skip updating stats if the cache entry has been concurrently evicted.
    const CacheEntry* cache_entry = FindStaticCacheEntryOrNull(key);
    if (cache_entry != nullptr) {
      auto& tier2_stats = cache_entry->stats().tier2;
      if (!tier2_stats.has_value()) {
        tier2_stats.emplace();
      }
      tier2_stats->pre_compile_duration = pre_compile_duration;
      tier2_stats->write_duration = write_duration;
    }
  }

  if (tier == CacheTier::kTier1 && uses_tier3) {
    // Write to tier-3 cache.
    const std::string tier3_entry_path = GetTier3CacheEntryPath(key);
    ABSL_VLOG(1) << "Writing to tier-3 cache for key " << key << " at path "
                 << tier3_entry_path;
    const absl::Status status =
        AtomicWriteToCacheFile(tier3_entry_path, *executable_or);
    if (!status.ok()) {
      // Writing to the tier-3 cache is best-effort - we don't need to fail
      // the user if it fails (the program will just run slower).
      ABSL_LOG(ERROR) << status;
    }
  }
}

void CompilationCache::Compile(CompilationCacheKey key,
                               LoadedExecutableBuilder executable_builder,
                               UniqueCompileOptions compile_options) {
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  if (client == nullptr) {
    absl::Status error = TT_ERROR(error::kFailedPrecondition)
                         << "PjRtClient must be initialized";
    ABSL_LOG(ERROR) << error.message();

    // Fulfill the promise immediately with the error status.
    SetExecutable(key, std::move(error), /*stats=*/{});
    return;
  }

  const absl::Time compile_start = absl::Now();
  absl::StatusOr<SharedLoadedExecutableWithMetadata> executable =
      torch_tpu::Compile(*client, std::move(executable_builder),
                         std::move(compile_options));

  SetExecutable(key, std::move(executable),
                {
                    .compilation_duration = absl::Now() - compile_start,
                });
}

std::string CompilationCache::HbmUsageSummary() const {
  absl::MutexLock lock(cache_mutex_);
  int64_t total_size_bytes = 0;
  int64_t num_executables = 0;
  for (const auto& [key, cache_entry] : executable_cache_) {
    if (!IsFutureReady(cache_entry.executable_future())) {
      continue;
    }
    const auto& executable = cache_entry.executable_future().get();
    if (!executable.ok() || *executable == nullptr) {
      continue;
    }
    const auto memory_stats =
        (*executable)->GetLoadedExecutable()->GetCompiledMemoryStats();
    if (!memory_stats.ok()) {
      ABSL_LOG(WARNING) << "Failed to get memory stats for key: " << std::hex
                        << key << " with status: " << memory_stats.status();
      continue;
    }
    total_size_bytes += memory_stats->generated_code_size_in_bytes;
    num_executables++;
  }
  return absl::StrCat("CompilationCache HBM usage: ",
                      tsl::strings::HumanReadableNumBytes(total_size_bytes),
                      " for ", num_executables, " executables.");
}

}  // namespace torch_tpu
