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

#ifndef TORCH_TPU_COMMON_COMPILATION_CACHE_H_
#define TORCH_TPU_COMMON_COMPILATION_CACHE_H_

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/compilation.h"
#include "torch_tpu/csrc/common/compilation_spec.h"
#include "torch_tpu/csrc/common/compile_options_key.h"
#include "torch_tpu/csrc/common/shape.h"
#include "torch_tpu/csrc/common/thread_pool.h"
#include "torch_tpu/csrc/common/tier2_compilation_cache.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"

namespace torch_tpu {

struct CompilationCacheInitializationOptions {
  bool cache_only = false;
};

// Statistics for a single entry in the tier-1 compilation cache.
struct CacheEntryStats {
  // If the entry was successfully compiled by this process, this is the time it
  // took to compile. Otherwise (e.g. if the compilation failed, or if the entry
  // was read from the tier-2 cache), it is zero.
  absl::Duration compilation_duration;
  // The last time this entry was read. The initial compilation counts as one
  // read as the executable will be used by whoever triggered the compilation.
  absl::Time last_read = absl::Now();
  // How many times this entry has been read. The initial compilation counts as
  // one read as the executable will be used by whoever triggered the
  // compilation.
  int64_t read_count = 1;
  // Set only when the entry is populated from the tier-2 cache as opposed to
  // being compiled from scratch.
  std::optional<Tier2CacheEntryStats> tier2;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const CacheEntryStats& stats) {
    absl::Format(&sink, "compilation_duration=%v, last_read=%v, read_count=%d",
                 stats.compilation_duration, stats.last_read, stats.read_count);
    // TODO(@lukeboyer): Handle tier2 if present.
  }
};

class CompilationCache;

// A cache entry, which is a future executable with its metadata.
// Invariant: executable_future is associated with executable_promise.
class CacheEntry {
 public:
  CacheEntry()
      : executable_promise_(std::make_shared<LoadedExecutablePromise>()),
        executable_future_(executable_promise_->get_future().share()) {}

  // This class is movable but not copyable.
  CacheEntry(CacheEntry&&) = default;
  CacheEntry& operator=(CacheEntry&&) = default;
  CacheEntry(const CacheEntry&) = delete;
  CacheEntry& operator=(const CacheEntry&) = delete;

  [[nodiscard]] const SharedLoadedExecutableWithMetadataFuture&
  executable_future() const {
    return executable_future_;
  }

  [[nodiscard]] const std::shared_ptr<LoadedExecutablePromise>&
  executable_promise() const {
    return executable_promise_;
  }

  [[nodiscard]] CacheEntryStats& stats() const { return stats_; }

 private:
  // Order the fields so that the promise is constructed before the future
  // and destroyed after it.
  std::shared_ptr<LoadedExecutablePromise> executable_promise_;
  SharedLoadedExecutableWithMetadataFuture executable_future_;
  mutable CacheEntryStats stats_;
};

struct BoundedDynamicCacheEntry {
  // The key for the cache entry corresponding to the middle part of the
  // bounded dynamic compilation. BoundedDynamicCacheEntries can be thought of
  // as triplets [preamble, middle, postamble] where the preamble and postamble
  // are lightweight computations.
  CompilationCacheKey middle_executable_key;
  // The shape dynamism metadata required to generate the adapters around the
  // middle executable.
  ShapeDynamismMetadata shape_dynamism_metadata;
};

// Aggregated statistics for the compilation cache.
struct PerfStats {
  // Statistics for a single entry in the compilation cache.
  struct EntryStats : public CacheEntryStats {
    CompilationCacheKey key;

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const EntryStats& stats) {
      absl::Format(&sink, "%v{%v}", stats.key,
                   static_cast<const CacheEntryStats&>(stats));
    }
  };

  // These fields refer to the tier-1 cache. E.g. a hit here means the entry was
  // found in the tier-1 cache. A miss means the entry was not found in the
  // tier-1 cache, and we had to look it up in the tier-2 cache, where it may
  // or may not be found.
  int64_t num_cache_misses() const { return num_cache_reqs - num_cache_hits; }
  int64_t num_cache_reqs = 0;
  int64_t num_cache_hits = 0;
  // Peak memory usage (host) during compilation in bytes across all concurrent
  // calls. std::nullopt represents failed to retrieve or unknown.
  std::optional<int64_t> peak_compilation_memory_bytes;

  std::vector<EntryStats> per_entry_stats;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const PerfStats& stats) {
    absl::Format(&sink, "num_cache_reqs=%d\n", stats.num_cache_reqs);
    absl::Format(&sink, "num_cache_hits=%d {%s}\n", stats.num_cache_hits,
                 PercAsStr(stats.num_cache_hits, stats.num_cache_reqs));
    if (stats.peak_compilation_memory_bytes.has_value()) {
      absl::Format(&sink, "peak_compilation_memory_bytes=%d\n",
                   *stats.peak_compilation_memory_bytes);
    } else {
      absl::Format(&sink, "peak_compilation_memory_bytes=unknown\n");
    }
    if (stats.per_entry_stats.empty()) {
      return;
    }
    absl::Duration total_comp_time = absl::ZeroDuration();
    for (const auto& entry : stats.per_entry_stats) {
      // TODO(@lukeboyer): Aggregate cache read stats as well.
      total_comp_time += entry.compilation_duration;
    }
    absl::Format(&sink, "num_compilation_events=%d\n",
                 stats.per_entry_stats.size());
    absl::Format(&sink, "sum_compilation_time=%v\n", total_comp_time);
  }
};

// An in-memory cache for compiled XLA programs. The ownership of cached
// executables is shared between the cache and the user of the executable.
// This shared ownership is needed to avoid deleting an executable still in use
// when the cache decides to evict it.
//
// This class uses a lazy initialization pattern. The singleton instance is
// created upon the first call to GetInstance(), but internal resources like
// thread pools are only allocated when they are first needed (e.g., during the
// first compilation request) or when specific options are set. This is needed
// to support lazy initialization of PjRt, as the compilation thread pool
// initialization needs Pj Rt to be initialized (and PJRT initialization is
// lazy).
class CompilationCache {
 public:
  // CompilationCache is neither copyable nor movable.
  CompilationCache(const CompilationCache&) = delete;
  CompilationCache& operator=(const CompilationCache&) = delete;
  CompilationCache(CompilationCache&&) = delete;
  CompilationCache& operator=(CompilationCache&&) = delete;

  ~CompilationCache();

  // Returns the singleton cache instance. Note that this returns the instance
  // without performing full internal initialization (e.g., creating compilation
  // thread pools). Internal resources are initialized only when they are first
  // needed.
  //
  // Cannot be called after ShutDown().
  [[nodiscard]] static CompilationCache& GetInstance()
      ABSL_LOCKS_EXCLUDED(cache_instance_mutex_);

  // Shuts down the cache and its thread pool, evicting all entries.
  //
  // Cannot be called before GetInstance() or more than once.
  static void ShutDown() ABSL_LOCKS_EXCLUDED(cache_instance_mutex_);

  // Shuts down the cache and its thread pool, evicting all entries, and then
  // creates a new instance of the compilation cache.
  //
  // Only used in testing to restore the cache to a clean state after
  // shutdown.
  static void Restart() ABSL_LOCKS_EXCLUDED(cache_instance_mutex_);

  // Returns whether the global compilation cache is initialized.
  [[nodiscard]] bool IsInitialized() const ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Sets the global compilation cache initialization options. These options
  // will be applied to the cache and used during its internal initialization.
  //
  // Calling this method after internal resources (like thread pools) have
  // already been initialized will update runtime-adjustable options (e.g.,
  // cache-only mode), but will not re-configure already established resources.
  void SetOptions(const CompilationCacheInitializationOptions& options)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Evicts all existing executables from the cache. The function waits for all
  // in-flight compilations to complete.
  void EvictAll() ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Sets the cache to allow caching or not. On initialization, the cache is
  // in allow-cache mode by default. We only need to call this function if we
  // want to disable caching (e.g. for debugging or perf analysis).
  void SetAllowCacheMode(bool allow = true) ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Sets the cache to only lookup and not compile if cache_only is true.
  // In this mode, the cache will not compile any programs, and will return
  // errors for all cache misses.
  void SetCacheOnlyMode(bool cache_only = true)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Cache statistics.
  [[nodiscard]] int64_t GetCacheRequests() const
      ABSL_LOCKS_EXCLUDED(cache_mutex_);
  [[nodiscard]] int64_t GetCacheHits() const ABSL_LOCKS_EXCLUDED(cache_mutex_);
  [[nodiscard]] int64_t GetCacheMisses() const
      ABSL_LOCKS_EXCLUDED(cache_mutex_);
  [[nodiscard]] PerfStats GetCacheStats() const
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Returns true if the executable associated with a cache key is compiled and
  // ready for execution.
  bool IsExecutableReady(CompilationCacheKey key) const
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Fetches a CompiledKernel with the right executable futures from the cache,
  // and enqueues compilations if necessary.
  absl::StatusOr<CompiledKernel> GetOrCompile(
      CompilationCacheKey key, const std::vector<Shape>& input_shapes,
      const std::vector<Shape>& output_shapes,
      MlirComputationBuilder computation_builder,
      UniqueCompileOptions compile_options, bool use_dynamic_adapters = true)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Schedules compilation of the given module associated to the key.
  //
  // Parameters:
  //  - key: The key where this entry is stored in the cache, used for tiered
  //    cache lookups.
  //  - contexted_module: The MLIR computation to compile.
  //  - compile_options: The compile options to use for the compilation.
  void EnqueueCompilation(CompilationCacheKey key,
                          ContextedModule contexted_module,
                          UniqueCompileOptions compile_options)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Sets the cache to dump the input StableHLO module for any cache misses.
  void SetDumpOnCacheMissMode(bool enable = true)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);
  bool GetDumpOnCacheMissMode() const ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Debugging function to return the total resident size of the loaded
  // executables in HBM.
  std::string HbmUsageSummary() const ABSL_LOCKS_EXCLUDED(cache_mutex_);

 private:
  friend class CompilationCacheTestHelper;

  // Protects the creation and lifecycle of the CompilationCache singleton
  // object itself. This is distinct from cache_mutex_ (a non-static member of
  // the class), which protects the internal state (cache entries, stats, etc.)
  // of a specific instance.
  static absl::Mutex cache_instance_mutex_;

  // Private constructor to enforce singleton pattern.
  CompilationCache();

  // Returns the singleton cache instance without performing full internal
  // initialization. Unlike GetInstance(), this function doesn't acquire any
  // locks.
  [[nodiscard]] static absl_nonnull std::unique_ptr<CompilationCache>&
  GetInstanceNoLock();

  // Ensures that the internal resources (like thread pools) of the compilation
  // cache are initialized. This method is idempotent and is triggered by any
  // operation that requires the compilation infrastructure (e.g.,
  // GetOrCompile, EnqueueCompilation).
  void EnsureInitialized() ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // The internal variant type for GetOrCreateCacheEntry, which returns
  // CacheEntry internal data that can be used outside of a lock, given a
  // CompilationCacheKey. If the key directly matches an entry in the
  // executable_cache_, we will return the executable_promise and
  // executable_future from the CacheEntry. Otherwise, we will check if the
  // ShapelessKey matches an entry in the bounded_dynamic_cache_. If so, we will
  // check if the input shapes are compatible with any of the bounded dynamic
  // entries, and return the internals of the middle executable cache entry
  // and the shape dynamism metadata. If an entry was created or if
  // allow_cache_mode_ is false, we will set needs_compilation to true.
  struct CacheLookupInternal {
    absl_nonnull std::shared_ptr<LoadedExecutablePromise> executable_promise;
    SharedLoadedExecutableWithMetadataFuture executable_future;
    std::optional<ShapeDynamismMetadata> shape_dynamism_metadata;
    bool needs_compilation = false;
    bool dump_on_cache_miss = false;
  };

  // Returns a pointer to the cache entry for `key` if found, otherwise returns
  // nullptr.
  [[nodiscard]] const CacheEntry* FindStaticCacheEntryOrNull(
      CompilationCacheKey key) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(cache_mutex_);

  // Retrieves a cache entry from the static cache. If the key is not found in
  // the static cache, returns nullopt.
  std::optional<CacheLookupInternal> GetStaticCacheEntry(
      CompilationCacheKey key) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(cache_mutex_);

  using BoundedDynamicCache =
      absl::flat_hash_map<ShapelessKey, std::vector<BoundedDynamicCacheEntry>,
                          ShapelessKey::Hash>;
  // Retrieves the bounded dynamic cache entries that match  this key. If the
  // key is not found in the bounded dynamic cache, returns nullopt.
  std::optional<BoundedDynamicCache::iterator> GetBoundedDynamicCacheEntries(
      ShapelessKey shapeless_key) ABSL_EXCLUSIVE_LOCKS_REQUIRED(cache_mutex_);

  // Creates a cache entry in the static cache. If the key is already in the
  // static cache, it will crash.
  CacheLookupInternal AddStaticCacheEntry(CompilationCacheKey key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(cache_mutex_);

  // Creates a new bounded dynamic cache entry for the given key and input
  // shapes. If dynamic_it is not nullopt, we check that the shapeless key
  // matches the dynamic iterator key, and we add it to the existing vector of
  // bounded dynamic cache entries. Otherwise, we create a new vector of bounded
  // dynamic cache entries.
  CacheLookupInternal AddBoundedDynamicCacheEntry(
      ShapelessKey shapeless_key, CompileOptionsKey compile_options_key,
      ShapeDynamismMetadata shape_dynamism_metadata,
      std::optional<BoundedDynamicCache::iterator> dynamic_it)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(cache_mutex_);

  // Retrieves a cache entry or create a new one as needed. This updates the
  // access count and last accessed time for the cache entry if it is found.
  //
  // Parameters:
  //  - key: The key to lookup in the cache.
  //  - input_shapes: The input shapes to the computation, in the order given by
  // a Traversal. This is used for checking compatibility with existing bounded
  // dynamic cache entries, or to create metadata for a new one.
  //  - output_shapes: The output shapes to the computation, in the order given
  // by a Traversal. This is used to create metadata for bounded dynamic cache
  // entries.
  //  - skip_dynamic_lookup_and_compilation: If true, only do lookups in the
  // static cache, and create a new static cache entry if needed.
  //
  // Returns:
  //   A `CacheLookupInternal` struct containing the cache lookup result.
  absl::StatusOr<CacheLookupInternal> GetOrCreateCacheEntry(
      CompilationCacheKey key, const std::vector<Shape>& input_shapes,
      const std::vector<Shape>& output_shapes,
      bool skip_dynamic_lookup_and_compilation = false)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(cache_mutex_);

  // Creates a dynamic kernel adapter for the given shape dynamism metadata and
  // input and output shapes.
  absl::StatusOr<DynamicKernelAdapter> CreateDynamicKernelAdapter(
      const ShapeDynamismMetadata& shape_dynamism_metadata,
      const std::vector<Shape>& input_shapes,
      const std::vector<Shape>& output_shapes,
      CompileOptionsKey compile_options_key,
      UniqueCompileOptions compile_options) ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Creates a padding kernel for the dynamic kernel adapter based on the shape
  // dynamism metadata and resolved dynamic shapes.
  absl::StatusOr<SharedLoadedExecutableWithMetadataFuture> CreatePaddingKernel(
      const ShapeDynamismMetadata& shape_dynamism_metadata,
      const std::vector<Shape>& static_runtime_input_shapes,
      const std::vector<Shape>& static_padded_input_shapes,
      std::vector<Shape> input_shapes_with_updated_dynamism,
      CompileOptionsKey compile_options_key,
      UniqueCompileOptions compile_options);

  // Creates a slicing kernel for the dynamic kernel adapter that executes slice
  // ops on the outputs if original padding operations induced layout changes.
  absl::StatusOr<SharedLoadedExecutableWithMetadataFuture> CreateSlicingKernel(
      const ShapeDynamismMetadata& shape_dynamism_metadata,
      const std::vector<Shape>& output_shapes,
      CompileOptionsKey compile_options_key,
      UniqueCompileOptions compile_options);

  // Sets the executable for the given key if it is not already set.
  // Also sets the stats for the cache entry.
  //
  // Precondition:
  //  - `key` is already in `executable_cache_`.
  //  - `cache_mutex_` is not held.
  void SetExecutable(
      CompilationCacheKey key,
      absl::StatusOr<SharedLoadedExecutableWithMetadata> executable,
      CacheEntryStats stats) ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Compiles the graph and stores the executable in the tier-1 cache.
  //
  // Precondition:
  //  - `key` is already in `executable_cache_`.
  //  - `cache_mutex_` is not held.
  // After the compilation is done successfully, the cache entry for `key` will
  // be updated with the compiled executable.
  //
  // This function is used by `GetFromTier2OrCompile()` to compile the graph
  // after a failed tier-2 cache lookup.
  void Compile(CompilationCacheKey key,
               LoadedExecutableBuilder executable_builder,
               UniqueCompileOptions compile_options)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Returns true if the cache is enabled and the tier-2 cache is configured.
  [[nodiscard]] bool ShouldUseTier2Cache() const
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Tries to get the compilation result from the tier-2 cache; if not
  // found, compiles the graph and stores the executable in the tier-1/2/3
  // caches.
  //
  // Precondition:
  //  - `key` is already in `executable_cache_` but the executable is not set.
  //  - `cache_mutex_` is not held.
  // After the compilation is done successfully, the cache entry for `key` will
  // be updated with the compiled executable.
  void GetFromTier2OrCompile(CompilationCacheKey key,
                             LoadedExecutableBuilder executable_builder,
                             UniqueCompileOptions compile_options)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Tries to get the compilation result from the tier-3 cache; if not found,
  // compiles the graph and stores the executable in the tier-1/2/3 caches.
  //
  // Precondition:
  //  - `key` is already in `executable_cache_` but the executable is not set.
  //  - `key` is not in the tier-2 cache.
  //  - `cache_mutex_` is not held.
  //  - `lock` is the lock for the tier-2 cache entry for `key` and held.
  // After the compilation is done successfully, the cache entry for `key` will
  // be updated with the compiled executable.
  void GetFromTier3OrCompile(CompilationCacheKey key,
                             LoadedExecutableBuilder executable_builder,
                             UniqueCompileOptions compile_options,
                             Tier2CacheEntryLock& lock,
                             absl::Time request_start)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Main mutex for protecting cache statistics and cache-only mode.
  mutable absl::Mutex cache_mutex_;

  // Thread pool for running compilations concurrently.
  std::unique_ptr<ThreadPool> compilation_pool_;

  // Lower-priority thread pool for running backup compilations.
  // Set only when tier-3 cache is enabled and local backup task is enabled.
  std::unique_ptr<ThreadPool> backup_compilation_pool_;

  // The cache of successfully compiled and in flight executables.
  absl::flat_hash_map<CompilationCacheKey, CacheEntry,
                      CompilationCacheKey::Hash>
      executable_cache_ ABSL_GUARDED_BY(cache_mutex_);

  // The cache of successfully compiled and in flight executables with bounded
  // dynamic shapes.
  BoundedDynamicCache bounded_dynamic_cache_ ABSL_GUARDED_BY(cache_mutex_);

  // Cache statistics.
  mutable PerfStats perf_stats_ ABSL_GUARDED_BY(cache_mutex_);
  // If true, the cache will be enabled. If false, the cache will be disabled
  // and every computation graph will be compiled from scratch (this is useful
  // for debugging and perf analysis).
  bool allow_cache_mode_ ABSL_GUARDED_BY(cache_mutex_) = true;
  // If true, the cache will only lookup and not compile. Any cache miss will
  // result in an error.
  bool cache_only_mode_ ABSL_GUARDED_BY(cache_mutex_) = false;
  // If true, the cache will dump the input StableHLO module for any cache
  // misses.
  bool dump_on_cache_miss_ ABSL_GUARDED_BY(cache_mutex_) = false;

  // The current initialization options.
  CompilationCacheInitializationOptions options_ ABSL_GUARDED_BY(cache_mutex_);

  // Whether the cache has been fully initialized (thread pools started).
  bool initialized_ ABSL_GUARDED_BY(cache_mutex_) = false;
};

// Returns the number of threads to use for compilation based on the flag value,
// the NPROC environment variable, or the hardware concurrency. Guaranteed to be
// > 0.
[[nodiscard]] int GetNumCompilationThreads();

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_COMPILATION_CACHE_H_
