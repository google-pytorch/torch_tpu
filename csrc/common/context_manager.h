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

#ifndef TORCH_TPU_CSRC_COMMON_CONTEXT_MANAGER_H_
#define TORCH_TPU_CSRC_COMMON_CONTEXT_MANAGER_H_

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "c10/util/ThreadLocalDebugInfo.h"
#include "csrc/common/context_states.h"

namespace torch_tpu {
namespace internal {

// The ThreadLocalDebugInfo slot to use for TorchTPU context states. For now, we
// can only use the TEST_INFO slot.
// TODO(https://github.com/pytorch/pytorch/issues/56027): use a dedicated slot
// for each type of context manager.
inline const auto kTpuContextSlot = c10::DebugInfoKind::TEST_INFO;

// Trait that indicates whether T is one of the options for an std::variant
// type.
template <typename T, typename Variant>
inline constexpr bool is_alternative_v = false;
template <typename T, typename... Args>
inline constexpr bool is_alternative_v<T, std::variant<Args...>> =
    (std::is_same_v<T, Args> || ...);

// Crashes if the current thread is not allowed to access context states.
void CrashIfContextStateAccessIsDisallowed();

}  // namespace internal

// One-way flip to disallow the current thread to access thread-local context
// states. After the flip, any attempt to access or mutate context states leads
// to crash.
//
// Intended to be used in the background worker threads only.
void DisallowThisThreadToAccessContextState();

// The state for different types of context managers respectively.
using ContextManagerState = std::variant<
    // go/keep-sorted start
    CustomCompilerOptionsContextState,  // for `custom_compiler_options`
    EagerModeContextState,              // for `eager_mode`
    EnableTracebacksContextState,       // for `enable_tracebacks`
    IsSpmdSafeContextState,             // for `spmd_safe`
    LayoutContextState,                 // for `layout`
    PrecisionContextState,              // for `precision`
    ProfileContextState                 // for `profile`
    // go/keep-sorted end
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
// thread's context. Returns nullopt if no context state of the given type
// is found.
template <typename T>
std::optional<T> GetContextState() {
  // Statically check that T is one of the options for ContextManagerState.
  static_assert(
      internal::is_alternative_v<T, ContextManagerState>,
      "T must be one of the alternative types of ContextManagerState.");
  internal::CrashIfContextStateAccessIsDisallowed();

  // Traverse nodes in the TorchTPU context state slot until a state of the
  // given type is found.
  const auto* untyped_node =
      c10::ThreadLocalDebugInfo::get(internal::kTpuContextSlot);
  if (untyped_node == nullptr) {
    return std::nullopt;
  }

  const auto* node = dynamic_cast<const ContextManagerNode*>(untyped_node);
  ABSL_CHECK(node != nullptr)  // CRASH_OK
      << "The state in the TorchTPU context slot must be a ContextManagerNode.";

  for (; node != nullptr; node = node->parent()) {
    if (const T* state = std::get_if<T>(&node->state())) {
      return *state;
    }
  }
  return std::nullopt;
}

// Returns the context state of the given type from the current python
// thread's context. Returns get_default_value() if no context state of the
// given type is found.
template <typename T, typename GetDefaultStateT>
T GetContextState(GetDefaultStateT get_default_state) {
  auto maybe_state = GetContextState<T>();
  return maybe_state.has_value() ? std::move(maybe_state).value()
                                 : get_default_state();
}

namespace internal {

// Type-erased implementation that pushes a context state variant onto the
// underlying TorchTPU-specific `c10::ThreadLocalDebugInfo` stack.
void PushContextStateUntyped(ContextManagerState state);

// Type-erased implementation that pops and returns a node from the underlying
// TorchTPU-specific `c10::ThreadLocalDebugInfo` stack.
absl_nonnull std::shared_ptr<const ContextManagerNode> PopContextStateUntyped();

}  // namespace internal

// Pushes the given context state of the specified type T onto the current
// python thread's state stack.
template <typename T>
void PushContextState(T state) {
  static_assert(
      internal::is_alternative_v<T, ContextManagerState>,
      "T must be one of the alternative types of ContextManagerState.");
  internal::CrashIfContextStateAccessIsDisallowed();

  internal::PushContextStateUntyped(ContextManagerState(std::move(state)));
}

// Pops the current python thread's context state stack, and returns the
// context state of the specified type T.
template <typename T>
void PopContextState() {
  static_assert(
      internal::is_alternative_v<T, ContextManagerState>,
      "T must be one of the alternative types of ContextManagerState.");
  internal::CrashIfContextStateAccessIsDisallowed();

  auto node = internal::PopContextStateUntyped();
  const T* state = std::get_if<T>(&node->state());
  ABSL_CHECK(state != nullptr)  // CRASH_OK
      << "The popped context state is not of the expected type "
      << typeid(T).name() << ".";
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_CONTEXT_MANAGER_H_
