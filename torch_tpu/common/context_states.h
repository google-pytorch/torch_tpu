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

#ifndef TORCH_TPU_COMMON_CONTEXT_STATES_H_
#define TORCH_TPU_COMMON_CONTEXT_STATES_H_

#include <memory>
#include <optional>

#include "absl/base/nullability.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_tpu/common/compilation_spec.h"
#include "xla/layout.h"

namespace torch_tpu {

// This library defines the context states for different types of context
// managers. By convention, the context state type is denoted by appending
// `ContextState` to the name of the context manager.

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
  // it will raise a runtime exception. This should be used only in
  // `torch.compile` mode.
  kInternalDeferAll,
};

// The state of the `eager_mode` context manager.
using EagerModeContextState = EagerMode;

// The state of the `precision` context manager.
using PrecisionContextState = mlir::stablehlo::Precision;

enum class ProfilerStatus {
  // By default, profiling is disabled.
  kDisabled,
  // Enables the collection of performance metrics by scoping the code region
  // under profiling in a `profiler.profile` context.
  kEnabled,
};

// The state of the `profile` context manager.
// TODO(b/509670300): remove it when the native PyTorch profiler API is fully
// integrated and available.
using ProfileContextState = ProfilerStatus;

// Whether to capture tracebacks for each dispatched op.
enum class TracebackMode {
  // Tracebacks are not captured. This is the default for eager mode, where MLIR
  // locations are not typically needed.
  kDisabled,
  // Tracebacks are captured for each dispatched op. This is the default for
  // FX compiled/export modes
  kEnabled,
};

// The state of the `enable_tracebacks` context manager.
using EnableTracebacksContextState = std::optional<TracebackMode>;

// The state of the `spmd_safe` context manager.
using IsSpmdSafeContextState = bool;

// The state of the `custom_compiler_options` context manager.
// TODO(b/502270689): switch the context state to `CompilationContext`.
using CustomCompilerOptionsContextState = CompilerOptionOverrides;

// The state of the `layout` context manager. Null means there's no active
// layout context.
using LayoutContextState = absl_nullable std::shared_ptr<const xla::Layout>;

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_CONTEXT_STATES_H_
