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

#include <map>
#include <memory>
#include <ostream>
#include <stack>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "c10/util/ThreadLocalDebugInfo.h"

namespace torch_tpu {

// Identifies a kind of TorchTPU context manager.
//
// This class is trivial to copy and move, and is cheap to construct and
// destroy. It is mirrored on the new c10::DebugInfoKind
// (https://github.com/pytorch/pytorch/pull/180001). We will migrate from
// this to the new c10::DebugInfoKind when it's available to TorchTPU.
class ContextManagerKind {
 private:
  // Must points to a string literal with static storage duration. The pointer
  // address is used as the identifier for the slot. The string itself is only
  // used for debugging purposes and should be descriptive of the slot's purpose
  // and ideally (but not required) be unique.
  using value_type = const std::string_view* absl_nonnull;

 public:
  ContextManagerKind() = delete;

  // Creates a ContextManagerKind with the given identity.
  explicit ContextManagerKind(value_type value);

  // Comparison operators for use as a key in maps and sets.
  constexpr bool operator==(const ContextManagerKind& other) const {
    return value_ == other.value_;
  }
  constexpr bool operator!=(const ContextManagerKind& other) const {
    return value_ != other.value_;
  }
  constexpr bool operator<(const ContextManagerKind& other) const {
    return value_ < other.value_;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ContextManagerKind& kind);

 private:
  value_type value_ = nullptr;
};

// Returns the ContextManagerKind for the given context manager state type.
// Each context manager implementation should specialize this function for its
// own state type. Different specializations must return different values.
template <typename State>
[[nodiscard]] ContextManagerKind GetContextManagerKind();

// Wrapper for a context manager state to make it storable via PyTorch's
// ThreadLocalDebugInfo API. The template parameter is the underlying type of
// the state.
template <typename State>
class ContextState : public c10::DebugInfoBase {
 public:
  explicit ContextState(State state) : state_(std::move(state)) {}

  // Accessors for the underlying state.
  const State& operator*() const { return state_; }
  const State* operator->() const { return &state_; }

 private:
  State state_;
};

// Contains the states of TorchTPU context managers for a specific
// Python thread. Conceptually, this is a collection of state stacks, one for
// each context manager kind.
class ContextStates : public c10::DebugInfoBase {
 public:
  // Holds the state of a single context manager kind.
  using SharedState = absl_nonnull std::shared_ptr<const c10::DebugInfoBase>;
  using NullableSharedState =
      absl_nullable std::shared_ptr<const c10::DebugInfoBase>;
  // The state stack for a single context manager kind.
  using StateStack = std::stack<SharedState>;
  // Maps a context manager kind to its state stack. Since we only have a small
  // number of context manager kinds, a map is very efficient.
  using StateStackByKind = std::map<ContextManagerKind, StateStack>;

  // Constructs an empty instance
  ContextStates() : stack_by_kind_(std::make_unique<StateStackByKind>()) {}

  // Pushes the given context manager state onto its stack.
  template <typename T>
  void Push(T state) const ABSL_LOCKS_EXCLUDED(mutex_) {
    const auto kind = GetContextManagerKind<T>();
    PushUntyped(kind, std::make_shared<ContextState<T>>(std::move(state)));
  }

  // Returns the top of the stack for the given state type, or the default
  // value if the stack is empty.
  template <typename T>
  T GetTop(T default_value) const ABSL_LOCKS_EXCLUDED(mutex_) {
    const auto kind = GetContextManagerKind<T>();
    const auto untyped_state = GetTopUntyped(kind);
    if (untyped_state == nullptr) {
      return default_value;
    }
    const auto state =
        std::dynamic_pointer_cast<const ContextState<T>>(untyped_state);
    ABSL_CHECK(state != nullptr)  // CRASH_OK
        << "Top of stack is not of the expected type " << typeid(T).name()
        << " for context manager kind " << kind;
    return **state;
  }

  // Pops the top of the stack for the given state type. Crashes if the stack is
  // empty, or if the top of the stack is not of the given type.
  template <typename T>
  void Pop() const ABSL_LOCKS_EXCLUDED(mutex_) {
    const auto kind = GetContextManagerKind<T>();
    const auto state = PopUntyped(kind);
    ABSL_CHECK(  // CRASH_OK
        std::dynamic_pointer_cast<const ContextState<T>>(state) != nullptr)
        << "Popped state is not of the expected type " << typeid(T).name()
        << " for context manager kind " << kind;
  }

  // Returns the instance of the context manager states for the current Python
  // thread.
  [[nodiscard]] static const ContextStates* absl_nonnull instance();

 private:
  mutable absl::Mutex mutex_;

  // Pushes the given context manager state onto the stack for the given kind.
  void PushUntyped(ContextManagerKind kind, SharedState node) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the top of the stack for the given kind, or nullptr if the stack is
  // empty.
  NullableSharedState GetTopUntyped(ContextManagerKind kind) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Pops the top of the stack for the given kind. Crashes if the stack is
  // empty.
  SharedState PopUntyped(ContextManagerKind kind) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  // We use a unique_ptr so that we can mutate the pointee without mutating
  // the ContextStates object itself.
  absl_nonnull std::unique_ptr<StateStackByKind> stack_by_kind_
      ABSL_PT_GUARDED_BY(mutex_);
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_CONTEXT_MANAGER_H_
