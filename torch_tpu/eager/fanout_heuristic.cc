// Copyright 2025 Google LLC
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

#include "torch_tpu/eager/fanout_heuristic.h"

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

void FanoutHeuristic(const DeviceBufferList& node,
                     absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
                         materialization_nodes) {
  if (node.num_child_ops() > 1) {
    materialization_nodes.insert(&node);
  }
}

}  // namespace torch_tpu
