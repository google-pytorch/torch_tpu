// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/eager/stale_heuristic.h"

#include <algorithm>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

void StaleHeuristic(const DeviceBufferList& node,
                    absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
                        materialization_nodes) {
  // Nodes which are live, but depend directly on a stale node, must be
  // materialized so the stale node can be freed.
  const auto deferred_op = node.deferred_op();
  ABSL_CHECK(deferred_op)  // CRASH_OK
      << "Found traversal node that's not a deferred op. This is a torch_tpu "
         "bug.";

  if (!node.is_stale() &&
      (std::any_of(deferred_op->inputs().begin(), deferred_op->inputs().end(),
                   [](const DeviceBufferRef& input) {
                     return input.device_buffer_list()->is_stale();
                   }))) {
    materialization_nodes.insert(&node);
  }
}

}  // namespace torch_tpu
