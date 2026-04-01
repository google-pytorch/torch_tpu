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

#ifndef TORCH_TPU_EAGER_SPLIT_TRAVERSAL_H_
#define TORCH_TPU_EAGER_SPLIT_TRAVERSAL_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

// FuseAll:     The entire graph for the traversal is executed as a single unit.
//              The resulting executable may be multi-output.
// SplitSafe:   the graph is split into as many traversals as necessary to
//              ensure that all tensors are executed in dispatch order, with no
//              risk of later out-of-order re-execution.
//              There will be exactly one output per split traversal, and the
//              returned traversals will be in dispatch order.
// SplitCustom: the graph is split according to the flags set in
//              split_traversal.cc.
enum class MaterializationMode { kFuseAll, kSplitSafe, kSplitCustom };

// This function splits a given traversal into multiple traversals by applying
// materialization heuristics for determining internal materialization/split
// points. The returned traversals are guaranteed to have only one output, and
// are organized according to the execution order of the original traversal.
// This function returns traversals in topological order.
absl::StatusOr<std::vector<Traversal>> SplitTraversal(
    Traversal traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& required_outputs,
    MaterializationMode mode);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_SPLIT_TRAVERSAL_H_
