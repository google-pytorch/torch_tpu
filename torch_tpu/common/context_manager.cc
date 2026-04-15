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

#include "torch_tpu/common/context_manager.h"

#include <cstddef>
#include <memory>
#include <ostream>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "c10/util/ThreadLocalDebugInfo.h"

namespace torch_tpu {

ContextManagerKind::ContextManagerKind(ContextManagerKind::value_type value)
    : value_(value) {
  ABSL_CHECK(value_ != nullptr)  // CRASH_OK
      << "ContextManagerKind must be created with a non-null string_view.";
}

std::ostream& operator<<(std::ostream& os, const ContextManagerKind& kind) {
  return os << *kind.value_;
}

void ContextStates::PushUntyped(const ContextManagerKind kind,
                                SharedState node) const {
  absl::MutexLock lock(mutex_);
  (*stack_by_kind_)[kind].push(std::move(node));
}

ContextStates::NullableSharedState ContextStates::GetTopUntyped(
    const ContextManagerKind kind) const {
  absl::MutexLock lock(mutex_);
  const auto it = stack_by_kind_->find(kind);
  if (it == stack_by_kind_->end()) {
    return nullptr;
  }
  if (it->second.empty()) {
    return nullptr;
  }
  return it->second.top();
}

ContextStates::SharedState ContextStates::PopUntyped(
    const ContextManagerKind kind) const {
  absl::MutexLock lock(mutex_);
  const auto it = stack_by_kind_->find(kind);
  ABSL_CHECK(it != stack_by_kind_->end())  // CRASH_OK
      << "Popping from empty stack for context manager kind " << kind;
  ABSL_CHECK(!it->second.empty())  // CRASH_OK
      << "Popping from empty stack for context manager kind " << kind;
  auto node = std::move(it->second.top());
  it->second.pop();
  return node;
}

// Protects access to the per-python-thread context manager states.
absl::Mutex g_context_manager_states_mutex(absl::kConstInit);

// The ThreadLocalDebugInfoKind slot to use for the TorchTPU context manager
// states. For now, we can only use the TEST_INFO slot.
// TODO(https://github.com/pytorch/pytorch/issues/56027): use a dedicated slot
// for each type of context manager.
constexpr auto kTpuContextSlot = c10::DebugInfoKind::TEST_INFO;

const ContextStates* absl_nonnull ContextStates::instance() {
  // We leverage c10::ThreadLocalDebugInfo to allocate one ContextSates
  // object per Python thread, in the TEST_INFO slot.
  absl::MutexLock lock(g_context_manager_states_mutex);
  const auto* const debug_info =
      c10::ThreadLocalDebugInfo::get(kTpuContextSlot);
  if (debug_info != nullptr) {  // The instance already exists.
    const auto* const states = dynamic_cast<const ContextStates*>(debug_info);
    ABSL_CHECK(states != nullptr)  // CRASH_OK
        << "Debug info of type " << static_cast<size_t>(kTpuContextSlot)
        << " is not a ContextStates.";
    return states;
  }

  // The instance does not exist, so create it and push it onto the
  // ThreadLocalDebugInfo stack, in the TEST_INFO slot.
  auto states = std::make_shared<ContextStates>();
  c10::ThreadLocalDebugInfo::_push(kTpuContextSlot, states);
  return states.get();
}

}  // namespace torch_tpu
