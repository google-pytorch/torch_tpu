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

// Protects zero-operand `stablehlo.custom_call` ops from MLIR CSE.
//
// Motivating case (b/574869112):
//
//   Pallas kernels and their JAX wrappers allocate uninitialized scratch and
//   output buffers with `lax.empty` / `jnp.empty` / `pl.empty` (for example the
//   HBM all-to-all scratch buffer of the tpu_inference fused MoE kernel, the
//   output refs of SparseCore `core_map` kernels, and `jax.empty_ref` when
//   state is discharged). On TPU, JAX lowers `lax.empty` to
//
//     %buf = stablehlo.custom_call @AllocateBuffer() {has_side_effect = false}
//         : () -> tensor<...>
//
//   which the Pallas kernel then writes into (typically via
//   input_output_aliases). Every call is semantically a *distinct* fresh
//   buffer. However, all calls of the same shape and dtype have no operands,
//   identical attributes and identical result types, and with
//   `has_side_effect = false` they report no memory effects. MLIR CSE therefore
//   treats them as pure and merges all of them into a single value. XLA then
//   has to insert copies to keep the kernels from clobbering the shared buffer
//   and keep extra buffers live. In vLLM Qwen3.5-35B-A3B prefill this merged
//   80 AllocateBuffer calls into 2, added 78 copies and 2.19 GiB peak HBM, and
//   regressed TTFT by ~10%.
//
// This is a fundamental limitation of StableHLO rather than a bug in a single
// producer: `stablehlo.custom_call` can only describe its effects with the
// `has_side_effect` boolean. `false` means "no memory effects at all" (pure,
// so CSE-able), and `true` means unscoped Allocate/Free/Read/Write (never CSE'd
// but also never removable). There is no way to express "allocates a fresh
// result: removable if unused, but not mergeable", which is what MLIR models
// with a result-scoped `MemoryEffects::Allocate` (as `memref.alloc` does).
// JAX picks `false` so that XLA can DCE unused buffers, and relies on XLA's
// HloCSE never merging zero-operand instructions; MLIR's CSE has no such rule.
//
// To get XLA's semantics inside the TorchTPU MLIR pipeline, the guard pass
// marks every zero-operand, side-effect-free custom call as side-effecting
// (tagging it with `kZeroOperandCustomCallGuardAttr`) immediately before each
// CSE run, and the unguard pass restores the original state immediately after
// it. Keeping the guarded window as small as possible means every other pass
// (notably the canonicalizer) still sees the calls as side-effect free and can
// DCE unused ones. We match on "zero operands" rather than on the
// `AllocateBuffer` target name, mirroring HloCSE's zero-operand rule and also
// covering any other fresh-value custom call targets.

#include "csrc/mlir/passes/guard_zero_operand_custom_calls.h"

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Name of the inherent `has_side_effect` attribute of `stablehlo.custom_call`.
constexpr llvm::StringLiteral kHasSideEffectAttrName = "has_side_effect";

struct GuardZeroOperandCustomCallsPass
    : public mlir::PassWrapper<GuardZeroOperandCustomCallsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GuardZeroOperandCustomCallsPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-guard-zero-operand-custom-calls";
  }
  llvm::StringRef getDescription() const override {
    return "Temporarily marks zero-operand, side-effect-free "
           "stablehlo.custom_call ops as side-effecting so that MLIR CSE does "
           "not merge distinct fresh-buffer allocations.";
  }

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    getOperation().walk([context](mlir::stablehlo::CustomCallOp op) {
      if (op->getNumOperands() != 0 || op.getHasSideEffect()) {
        return;
      }
      const bool had_explicit_attr =
          static_cast<bool>(op.getHasSideEffectAttr());
      op->setAttr(kZeroOperandCustomCallGuardAttr,
                  mlir::BoolAttr::get(context, had_explicit_attr));
      op.setHasSideEffect(true);
    });
  }
};

struct UnguardZeroOperandCustomCallsPass
    : public mlir::PassWrapper<UnguardZeroOperandCustomCallsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      UnguardZeroOperandCustomCallsPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-unguard-zero-operand-custom-calls";
  }
  llvm::StringRef getDescription() const override {
    return "Restores zero-operand stablehlo.custom_call ops marked by "
           "torch-tpu-guard-zero-operand-custom-calls to their original "
           "side-effect-free state.";
  }

  void runOnOperation() override {
    getOperation().walk([](mlir::stablehlo::CustomCallOp op) {
      auto guard =
          op->getAttrOfType<mlir::BoolAttr>(kZeroOperandCustomCallGuardAttr);
      if (!guard) {
        return;
      }
      if (guard.getValue()) {
        op.setHasSideEffect(false);
      } else {
        op->removeAttr(kHasSideEffectAttrName);
      }
      op->removeAttr(kZeroOperandCustomCallGuardAttr);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createGuardZeroOperandCustomCallsPass() {
  return std::make_unique<GuardZeroOperandCustomCallsPass>();
}

std::unique_ptr<mlir::Pass> createUnguardZeroOperandCustomCallsPass() {
  return std::make_unique<UnguardZeroOperandCustomCallsPass>();
}

}  // namespace torch_tpu
