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

#ifndef TORCH_TPU_CSRC_EAGER_SPLIT_UTILS_H_
#define TORCH_TPU_CSRC_EAGER_SPLIT_UTILS_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/traversal.h"

namespace torch_tpu {

// Ensures that if a tensor is used after a materialization point, it is also
// materialized before the materialization point.
//
// For example, a function like:
// ```
//   x = foo()
//   y = bar()
//   z = baz(x)
//   print(y.item())
// ```
// will have:
//   execution_order: [foo, bar, baz]
//   required_outputs: {bar}
//   materialization_points: {bar}
// This function will add a materialization point for `x = foo()`, and will
// remove any materialization points after the last required output.
void SplitAllMaterializationPoints(
    absl::Span<const SharedDeviceBufferList> execution_order,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        required_outputs,
    absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        materialization_points);

// Creates a new set of traversals, one for each split point in the original
// traversal.
//
// If `use_sorted` is true, then the execution order of the traversal will be
// partitioned into sequential chunks rather than re-traversing.
// If `use_sorted` is false, then the split points will be re-traversed for
// each returned traversal.
absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
ApplySplitPoints(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points,
    bool use_sorted = true);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_SPLIT_UTILS_H_
