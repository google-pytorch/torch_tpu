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

#ifndef TORCH_TPU_CSRC_EAGER_SPLIT_TRAVERSAL_H_
#define TORCH_TPU_CSRC_EAGER_SPLIT_TRAVERSAL_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/traversal.h"

namespace torch_tpu {

// A configuration setting for determining if additional split points should be
// considered when splitting a traversal.
enum class GraphSplitMode {
  // No additional split points beyond those required by OpSplitMode are added.
  // This may result in large, M-to-N graphs.
  kFullGraph,
  // All outputs are split into single-output graphs, with additional graphs
  // added to prevent recomputation or out-of-order execution.
  // Exception: if no additional outputs are required to prevent recomputation
  // or out-of-order execution, no split is made, and the behavior is the
  // same as kFullGraph.
  kSplitGraph,
};

// This function splits a given traversal into multiple traversals by applying
// materialization heuristics for determining internal materialization/split
// points. The returned traversals are guaranteed to have only one output, and
// are organized according to the execution order of the original traversal.
// This function returns traversals in topological order.
[[nodiscard]] std::vector<absl_nonnull std::unique_ptr<Traversal>>
SplitTraversal(absl_nonnull std::unique_ptr<Traversal> traversal,
               absl::flat_hash_set<const DeviceBufferList*>& required_outputs);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_SPLIT_TRAVERSAL_H_
