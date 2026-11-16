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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_GUARD_ZERO_OPERAND_CUSTOM_CALLS_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_GUARD_ZERO_OPERAND_CUSTOM_CALLS_H_

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Discardable attribute used to tag custom calls that were temporarily marked
// side-effecting by the guard pass. Its BoolAttr value records whether the op
// originally carried an explicit `has_side_effect` attribute, so the unguard
// pass can restore the op exactly.
inline constexpr llvm::StringLiteral kZeroOperandCustomCallGuardAttr =
    "torch_tpu.zero_operand_custom_call_guard";

// Creates a `func.func` pass that marks every zero-operand, side-effect-free
// `stablehlo.custom_call` as side-effecting so that MLIR CSE cannot merge
// distinct calls, e.g. the `custom_call @AllocateBuffer()` ops that JAX emits
// for `lax.empty`.
//
// Intended to bracket a CSE pass: run this immediately before CSE and
// `createUnguardZeroOperandCustomCallsPass()` immediately after it, in the same
// `func.func` pass manager, so other passes can still DCE unused calls. See
// guard_zero_operand_custom_calls.cc for details.
std::unique_ptr<mlir::Pass> createGuardZeroOperandCustomCallsPass();

// Creates a `func.func` pass that undoes
// `createGuardZeroOperandCustomCallsPass()`, restoring the original
// (side-effect-free) `has_side_effect` state so that unused calls can be
// dead-code-eliminated again.
std::unique_ptr<mlir::Pass> createUnguardZeroOperandCustomCallsPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_GUARD_ZERO_OPERAND_CUSTOM_CALLS_H_
