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

#ifndef TORCH_TPU_CSRC_EAGER_EAGER_MODE_H_
#define TORCH_TPU_CSRC_EAGER_EAGER_MODE_H_

namespace torch_tpu {

// The op defer mode.
enum class EagerMode {
  // kDeferAndFuse defers all ops except those that cannot be deferred.  This
  // provides a higher performance eager mode.
  kDeferAndFuse,
  // kDeferNever marks all ops to be executed immediately, similarly to PyTorch
  // on CUDA eager mode. This is primarily used for debugging, as it has
  // suboptimal compile and execution performance.
  kDeferNever,
  // kDeferNeverAndLaunchBlocking marks all ops to be executed immediately and
  // waits for the compilation of one op before dispatching the next one.
  // Some expensive runtime checks are also performed in this mode. This
  // should be used only for debugging, as it has the worst execution
  // performance.
  kDeferNeverAndLaunchBlocking,
  // kInternalDeferAll attempts to defer all ops. If an op cannot be deferred,
  // it will raise a runtime exception. This can be used to enforce a "manual
  // sync" mode of execution, where explicit synchronization is required before
  // any execution will occur. This is primarily used for debugging and testing
  // and is not intended for most users.
  kInternalDeferAll,
  // kInternalCompileFxGraph fully prevents all eager mode operations from
  // being executed on-device. This is used in compilation and export tracing
  // (e.g., `torch.compile` or `torch.export`).
  kInternalCompileFxGraph,
};

// Sets the global default eager mode. Thread-safe.
void SetEagerMode(EagerMode mode);

// Returns the active eager mode for the current Python thread. Thread-safe.
//
// It resolves the mode from the thread-local context state managed via Python
// context managers, with innermost context taking precedence. If no context
// manager is active, it falls back to the global mode set by `SetEagerMode`.
[[nodiscard]] EagerMode GetEagerMode();

// Returns whether the specified eager mode is a DeferNever mode
// (kDeferNever or kDeferNeverAndLaunchBlocking).
[[nodiscard]] constexpr bool IsDeferNeverMode(EagerMode mode) {
  return mode == EagerMode::kDeferNever ||
         mode == EagerMode::kDeferNeverAndLaunchBlocking;
}

// Controls whether input buffer donation for in-place ATen operations in
// DeferNever eager mode is enabled globally. Thread-safe.
void EnableInplaceBufferDonation(bool enabled);

// Returns whether input buffer donation for in-place ATen operations in
// DeferNever eager mode is enabled for the current Python thread/context.
// Thread-safe.
//
// It resolves the state from the thread-local context state managed via Python
// context managers, with innermost context taking precedence. If no context
// manager is active, it falls back to the global setting set by
// `EnableInplaceBufferDonation`.
[[nodiscard]] bool IsInplaceBufferDonationEnabled();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_EAGER_MODE_H_
