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

#include "torch_tpu/eager/custom_split.h"

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "torch_tpu/common/flags.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/dynamic_op_split_heuristic.h"
#include "torch_tpu/eager/fanout_heuristic.h"
#include "torch_tpu/eager/forced_split_heuristic.h"
#include "torch_tpu/eager/stale_heuristic.h"
#include "torch_tpu/eager/traversal.h"
#include "tsl/profiler/lib/traceme.h"

ABSL_FLAG(bool, torch_tpu_internal_fanout_heuristic, true,
          "Use a materialization heuristic that looks at node fanout.");
ABSL_FLAG(bool, torch_tpu_internal_stale_heuristic, true,
          "Use a materialization heuristic that materializes around the stale "
          "regions of a graph.");

namespace torch_tpu {

namespace {

// Config struct indicating which materialization heuristic flags are enabled.
struct CustomHeuristics {
  void Initialize() {
    fanout = GetFlagOnce<bool, &FLAGS_torch_tpu_internal_fanout_heuristic>();
    stale = GetFlagOnce<bool, &FLAGS_torch_tpu_internal_stale_heuristic>();

    initialized = true;
  }

  // Whether the heuristics have been initialized.
  bool initialized = false;

  bool fanout = false;
  bool stale = false;
};

}  // namespace

absl::flat_hash_set<const DeviceBufferList*> CustomSplitRule(
    const Traversal& traversal) {
  tsl::profiler::TraceMe t("CustomSplitRule");
  absl::flat_hash_set<const DeviceBufferList*> split_points;
  static CustomHeuristics enabled_heuristics;
  if (!enabled_heuristics.initialized) {
    enabled_heuristics.Initialize();
  }

  for (const auto& node : traversal.execution_order()) {
    ForcedSplitHeuristic(*node, split_points);
    DynamicOpSplitHeuristic(*node, split_points);
    if (enabled_heuristics.fanout) {
      FanoutHeuristic(*node, split_points);
    }
    if (enabled_heuristics.stale) {
      StaleHeuristic(*node, split_points);
    }
  }
  return split_points;
}

}  // namespace torch_tpu
