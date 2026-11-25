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

#ifndef TORCH_TPU_CSRC_EAGER_TRAVERSAL_UTILS_H_
#define TORCH_TPU_CSRC_EAGER_TRAVERSAL_UTILS_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/traversal.h"

namespace torch_tpu {

// Enum indicating whether automatic skinny collective offload is enabled.
enum class SkinnySparseCoreOffload {
  kEnabled,
  kDisabled,
};

// Returns whether the automatic collective offload is enabled.
// See kTorchTpuInternalSkinnyScOffloadEnvVar in env_vars.h for more details.
[[nodiscard]] SkinnySparseCoreOffload GetSkinnySparseCoreOffloadMode();

// Configuration options for PrepareTraversals.
struct PrepareTraversalsOptions {
  // Whether to apply additional graph splitting. This can improve compile times
  // and avoid HBM spikes from executables with many inputs and outputs.
  GraphSplitMode graph_split_mode = GraphSplitMode::kSplitGraph;
  // The core to pin the traversals to, or kUnpinned.
  // If kUnpinned, the traversals may still be pinned depending on the value of
  // skinny_sparse_core_offload_mode.
  CorePinningMode core_pinning_mode = CorePinningMode::kUnpinned;
  // Whether to automatically pin traversals that are singular collective ops
  // ("skinny collectives") to the SparseCore. Ignored if CorePinningMode has
  // an explicit pinning.
  SkinnySparseCoreOffload skinny_sparse_core_offload_mode =
      SkinnySparseCoreOffload::kDisabled;
};

// Given a list of nodes to execute, and a list of nodes to explicitly
// materialize, returns a list of traversals that will do this work.
//
// Every node that must be materialized will be an output of a traversal, and
// every node that must be executed will either be an output, or an input to a
// different output.
//
// The number of returned traversals will depend on the OpSplitMode of the nodes
// in the must_execute list, as well as the options specified in `options`.
//
// The sequence of returned Traversals will be in a valid topological order and
// will be pinned to the appropriate core as specified by `core_pinning_mode`.
[[nodiscard]] std::vector<absl_nonnull std::unique_ptr<Traversal>>
PrepareTraversals(std::vector<SharedDeviceBufferList> must_execute,
                  absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
                  const PrepareTraversalsOptions& options);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_TRAVERSAL_UTILS_H_
