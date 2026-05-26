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

#ifndef TORCH_TPU_COMMON_CONTEXT_MANAGER_H_
#define TORCH_TPU_COMMON_CONTEXT_MANAGER_H_

#include <memory>
#include <utility>
#include <variant>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "c10/util/ThreadLocalDebugInfo.h"
#include "torch_tpu/common/context_states.h"

namespace torch_tpu {

namespace internal {
// The ThreadLocalDebugInfo slot to use for TorchTPU context states. For now, we
// can only use the TEST_INFO slot.
// TODO(https://github.com/pytorch/pytorch/issues/56027): use a dedicated slot
// for each type of context manager.
inline const auto kTpuContextSlot = c10::DebugInfoKind::TEST_INFO;
}  // namespace internal

// The state for different types of context managers respectively.
using ContextManagerState = std::variant<
    // clang-format off
    // go/keep-sorted start
    CustomCompilerOptionsContextState,  // The `custom_compiler_options` context manager.
    DotAlgorithmContextState,           // The `dot_algorithm` context manager.
    EagerModeContextState,              // The `eager_mode` context manager.
    EnableTracebacksContextState,       // The `enable_tracebacks` context manager.
    IsSpmdSafeContextState,             // The `spmd_safe` context manager.
    LayoutContextState,                 // The `layout` context manager.
    PrecisionContextState,              // The `precision` context manager.
    ProfileContextState                 // The `profile` context manager.
    // go/keep-sorted end
    // clang-format on
    >;

// A persistent stack node for TorchTPU context managers. Each node contains a
// specific feature override and a pointer to the previous node.
class ContextManagerNode : public c10::DebugInfoBase {
 public:
  // Creates a node from the given state (delta) and parent node.
  ContextManagerNode(ContextManagerState state,
                     const ContextManagerNode* absl_nullable parent)
      : state_(std::move(state)), parent_(parent) {}

  // This class is neither copyable nor movable.
  ContextManagerNode(const ContextManagerNode&) = delete;
  ContextManagerNode& operator=(const ContextManagerNode&) = delete;
  ContextManagerNode(ContextManagerNode&&) = delete;
  ContextManagerNode& operator=(ContextManagerNode&&) = delete;

  // Accessors.
  const ContextManagerState& state() const { return state_; }
  const ContextManagerNode* absl_nullable parent() const { return parent_; }

 private:
  ContextManagerState state_;
  const ContextManagerNode* absl_nullable parent_;  // Not owned.
};

// Returns the context state of the given type from the current python
// thread's context. Returns nullptr if no context state of the given type is
// found.
template <typename T>
T GetContextState(T default_value) {
  // Traverse nodes in the TorchTPU context state slot until a state of the
  // given type is found.
  const auto* untyped_node =
      c10::ThreadLocalDebugInfo::get(internal::kTpuContextSlot);
  if (untyped_node == nullptr) {
    return default_value;
  }

  const auto* node = dynamic_cast<const ContextManagerNode*>(untyped_node);
  ABSL_CHECK(node != nullptr)  // CRASH_OK
      << "The state in the TorchTPU context slot must be a ContextManagerNode.";

  for (; node != nullptr; node = node->parent()) {
    if (const T* state = std::get_if<T>(&node->state())) {
      return *state;
    }
  }
  return default_value;
}

// Pushes the given context state onto the current python thread's state stack.
void PushContextStateUntyped(ContextManagerState state);
template <typename T>
void PushContextState(T state) {
  PushContextStateUntyped(ContextManagerState(std::move(state)));
}

// Pops the current python thread's context state stack.
absl_nonnull std::shared_ptr<const ContextManagerNode> PopContextStateUntyped();
template <typename T>
void PopContextState() {
  auto node = PopContextStateUntyped();
  const T* state = std::get_if<T>(&node->state());
  ABSL_CHECK(state != nullptr)  // CRASH_OK
      << "The popped context state is not of the expected type "
      << typeid(T).name() << ".";
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_CONTEXT_MANAGER_H_
