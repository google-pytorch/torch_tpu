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

#include "torch_tpu/eager/prevent_graph_splits.h"

#include <algorithm>
#include <cstddef>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/traversal.h"

ABSL_FLAG(int, torch_tpu_internal_prevent_graph_splits, 0,
          "Prevent graph spilts for repeated graphs (disabled if < 1).");

namespace torch_tpu {
namespace {

// This is a quick traversal hash, faster to compute than a traversal's cache
// key, but much more prone to collisions.
inline std::size_t QuickHash(const Traversal& t) {
  size_t seed = 0;
  HashCombine(seed, t.inputs().size());
  HashCombine(seed, t.execution_order().size());
  HashCombine(seed, t.outputs().size());
  if (!t.outputs().empty()) {
    auto& out_node = t.outputs()[0].device_buffer_list();
    if (auto* deferred_op = out_node->deferred_op(); deferred_op != nullptr) {
      HashCombine(seed, static_cast<size_t>(deferred_op->op_name()));
    }
    HashCombine(seed, static_cast<size_t>(out_node->element_type(0)));
  }
  return seed;
}

// We keep track of traversal repetitions with a 2-level data struction, where
// level 1 is indexed by a "quick" traversal hash, and level 2 by the more
// expensive traversal's cache key. This is to avoid expensive cache key
// computations for traversals that don't repeat.
struct Level2RepetitionMap {
  size_t count = 0;
  enum class CompilationStatus { kNotStarted, kStarted, kFinished };
  CompilationStatus compilation_status = CompilationStatus::kNotStarted;
};

struct Level1RepetitionMap {
  size_t count = 0;
  size_t num_successes = 0;
  size_t num_failures = 0;
  bool skip = false;
  absl::flat_hash_map<CompilationCacheKey, Level2RepetitionMap,
                      CompilationCacheKey::Hash>
      map2;
};

}  // namespace

absl::StatusOr<bool> PreventGraphSplit(const Traversal& traversal) {
  const auto kThreshold =
      absl::GetFlag(FLAGS_torch_tpu_internal_prevent_graph_splits);
  if (kThreshold <= 0) {
    return false;
  }

  static absl::NoDestructor<absl::flat_hash_map<size_t, Level1RepetitionMap>>
      map1;
  static bool ongoing_compilation = false;  // Used to run only one speculative
                                            // compilation at the time, so as to
                                            // avoid interfering with other
                                            // compilations that may be
                                            // necessary to the execution.

  bool prevent_split = false;

  auto hash1 = QuickHash(traversal);
  auto& v1 = (*map1)[hash1];

  if (!v1.skip && ++v1.count >= kThreshold) {
    auto cache_key = traversal.cache_key();
    auto& v2 = v1.map2[cache_key];
    if (++v2.count >= kThreshold) {
      v1.num_successes++;
      if (!ongoing_compilation &&
          v2.compilation_status ==
              Level2RepetitionMap::CompilationStatus::kNotStarted) {
        TT_ASSIGN_OR_RETURN(auto compile_result,
                            traversal.Compile(CompilationMode::kFastRuntime));
        ongoing_compilation = true;
        v2.compilation_status =
            Level2RepetitionMap::CompilationStatus::kStarted;
        ABSL_VLOG(1) << "Started compilation for REPEATED TRAVERSAL nodes="
                     << traversal.execution_order().size()
                     << " count=" << v2.count << " hash1=" << hash1
                     << " key=" << cache_key;

      } else {
        if (v2.compilation_status !=
            Level2RepetitionMap::CompilationStatus::kFinished) {
          if (CompilationCache::GetInstance().IsExecutableReady(cache_key)) {
            ongoing_compilation = false;
            v2.compilation_status =
                Level2RepetitionMap::CompilationStatus::kFinished;
          }
        }
        if (v2.compilation_status ==
            Level2RepetitionMap::CompilationStatus::kFinished) {
          ABSL_VLOG(1) << "Not Splitting REPEATED TRAVERSAL nodes="
                       << traversal.execution_order().size()
                       << " count=" << v2.count << " hash1=" << hash1
                       << " key=" << cache_key;
          prevent_split = true;
        }
      }

    } else {
      // If we fail to identify repetitions for a given "quick" traversal hash,
      // then we stop looking for repetitions on that "quick" hash value so as
      // to avoid slowing down processing of traversals that don't repeat.
      //
      // TODO: Experiment with different values of kMaxNumFailures, which is
      // currently set to "2 * num_successes", with a minimum value derived by
      // kThreshold to be used when not enough successes have been seen yet.
      const auto kMaxNumFailures =
          std::max(2 * v1.num_successes, 3 * static_cast<size_t>(kThreshold));
      v1.num_failures++;
      if (v1.num_failures >= kMaxNumFailures) {
        ABSL_VLOG(1) << "Stopping tracking of REPEATED TRAVERSALS for hash1 "
                     << hash1 << " key=" << cache_key
                     << " v1.num_successes=" << v1.num_successes
                     << " v1.num_failures=" << v1.num_failures;
        v1.skip = true;
      }
    }
  }

  return prevent_split;
}

}  // namespace torch_tpu
