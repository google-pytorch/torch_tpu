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

#include "torch_tpu/eager/forced_split_heuristic.h"

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

namespace {

bool ShouldSplitAfter(OpSplitMode mode) {
  return mode == OpSplitMode::kSplitAfter || mode == OpSplitMode::kSplitBoth;
}

bool ShouldSplitBefore(OpSplitMode mode) {
  return mode == OpSplitMode::kSplitBefore || mode == OpSplitMode::kSplitBoth;
}

}  // namespace

void ForcedSplitHeuristic(
    const DeviceBufferList& node,
    absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        materialization_nodes) {
  const auto deferred_op = node.deferred_op();
  if (deferred_op == nullptr) {
    return;
  }

  OpSplitMode split_mode = deferred_op->split_mode();
  if (ShouldSplitAfter(split_mode)) {
    materialization_nodes.insert(&node);
  }
  if (ShouldSplitBefore(split_mode)) {
    for (const auto& input : deferred_op->inputs()) {
      if (input.state() == DeviceBufferRefState::kDeferred) {
        materialization_nodes.insert(input.device_buffer_list().get());
      }
    }
  }
}

}  // namespace torch_tpu
