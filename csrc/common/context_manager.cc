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

#include "csrc/common/context_manager.h"

#include <memory>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "c10/util/ThreadLocalDebugInfo.h"

namespace torch_tpu {

thread_local  // CPP_THREAD_LOCAL_OK=per dispatch-worker thread boundary
    bool allow_context_state_access = true;

void DisallowThisThreadToAccessContextState() {
  allow_context_state_access = false;
}

namespace internal {

void CrashIfContextStateAccessIsDisallowed() {
  ABSL_CHECK(allow_context_state_access)  // CRASH_OK
      << "Thread-local context state access and mutation is only allowed in "
      << "the dispatch thread. "
      << "For worker threads, please have the dispatch thread resolve "
      << "thread-local context states and explicitly pass down parameters.";
}

void PushContextStateUntyped(ContextManagerState state) {
  // Get the current top node of the shared slot to use as parent.
  const auto* const top = dynamic_cast<ContextManagerNode*>(
      c10::ThreadLocalDebugInfo::get(internal::kTpuContextSlot));

  // _push sets the new state for the current python thread.
  c10::ThreadLocalDebugInfo::_push(
      internal::kTpuContextSlot,
      std::make_shared<ContextManagerNode>(state, top));
}

absl_nonnull std::shared_ptr<const ContextManagerNode>
PopContextStateUntyped() {
  // _pop removes the last pushed state for the current python thread.
  auto node = c10::ThreadLocalDebugInfo::_pop(internal::kTpuContextSlot);
  ABSL_CHECK(node != nullptr)  // CRASH_OK
      << "No context state to pop for the current python thread. This is a bug "
         "in TorchTPU.";
  auto manager_node = std::dynamic_pointer_cast<const ContextManagerNode>(node);
  ABSL_CHECK(manager_node != nullptr)  // CRASH_OK
      << "The state in the TorchTPU context slot must be a ContextManagerNode.";
  return manager_node;
}

}  // namespace internal
}  // namespace torch_tpu
