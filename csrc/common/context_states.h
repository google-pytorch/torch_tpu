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

#ifndef TORCH_TPU_CSRC_COMMON_CONTEXT_STATES_H_
#define TORCH_TPU_CSRC_COMMON_CONTEXT_STATES_H_

#include <memory>
#include <optional>

#include "absl/base/nullability.h"
#include "csrc/common/compilation_spec.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/layout.h"

namespace torch_tpu {

// This library defines the context states for different types of context
// managers. By convention, the context state type is denoted by appending
// `ContextState` to the name of the context manager.

enum class EagerMode;

// The state of the `eager_mode` context manager.
using EagerModeContextState = EagerMode;

// The state of the `precision` context manager. If the state is empty
// (i.e., no thread-local precision context is active), operations fall back to
// using the global precision setting configured via
// `torch.set_float32_matmul_precision`.
//
// Note: the default value for global precision is `medium` for TPU.
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
using CustomCompilerOptionsContextState =
    absl_nonnull std::shared_ptr<const CompilationContext>;

// The state of the `layout` context manager.
using LayoutContextState = absl_nonnull std::shared_ptr<const xla::Layout>;

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_CONTEXT_STATES_H_
