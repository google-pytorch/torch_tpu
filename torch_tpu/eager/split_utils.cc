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

#include "torch_tpu/eager/split_utils.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

namespace {

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
ApplySplitPointsUnsorted(
    const Traversal& traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points) {
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
  for (const auto& node : traversal.execution_order()) {
    if (!split_points.contains(node.get())) continue;
    TT_ASSIGN_OR_RETURN(auto new_traversal,
                        Traversal::Create({node}, split_points));
    traversals.push_back(std::move(new_traversal));
  }
  return traversals;
}

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
ApplySplitPointsSorted(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points) {
  std::vector<SharedDeviceBufferList> queue;
  {
    auto parts = traversal->IntoParts();
    queue = std::move(parts.execution_order);
  }
  std::vector<SharedDeviceBufferList> execution_order;
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
  for (auto& node : queue) {
    if (!split_points.contains(node.get())) {
      execution_order.push_back(std::move(node));
    } else {
      execution_order.push_back(node);  // intentional copy
      TT_ASSIGN_OR_RETURN(auto new_traversal,
                          Traversal::CreateFromExecutionOrder(execution_order));
      traversals.push_back(std::move(new_traversal));
      execution_order.clear();
    }
  }
  return traversals;
}

}  // namespace

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
ApplySplitPoints(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points,
    bool use_sorted) {
  if (use_sorted) {
    return ApplySplitPointsSorted(std::move(traversal), split_points);
  }
  return ApplySplitPointsUnsorted(*traversal, split_points);
}

}  // namespace torch_tpu
