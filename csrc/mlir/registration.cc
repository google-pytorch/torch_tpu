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

#include "csrc/mlir/registration.h"

#include <memory>

#include "csrc/mlir/passes/dummy_pass.h"
#include "csrc/mlir/passes/fold_concat_softmax.h"
#include "csrc/mlir/passes/guard_zero_operand_custom_calls.h"
#include "csrc/mlir/passes/lower_separable_gather_to_dot.h"
#include "csrc/mlir/passes/reassociate_norm_weights.h"
#include "csrc/mlir/pipeline.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "stablehlo/dialect/Register.h"

namespace torch_tpu {

void RegisterTorchTpuPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createDummyPass(); });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLowerSeparableGatherToDotPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReassociateNormWeightsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createFoldConcatSoftmaxPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createGuardZeroOperandCustomCallsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createUnguardZeroOperandCustomCallsPass();
  });
  mlir::registerPassPipeline(
      "torch-tpu-optimization-pipeline",
      "The full Torch TPU MLIR optimization pipeline run on every compiled "
      "module (see RunTorchTpuOptimizationPasses). Must be anchored on "
      "builtin.module.",
      [](mlir::OpPassManager& pm, llvm::StringRef options,
         llvm::function_ref<llvm::LogicalResult(const llvm::Twine&)>
             error_handler) -> llvm::LogicalResult {
        if (!options.empty()) {
          return error_handler(
              "torch-tpu-optimization-pipeline does not take options");
        }
        AddTorchTpuOptimizationPasses(pm);
        return llvm::success();
      },
      [](llvm::function_ref<void(const mlir::detail::PassOptions&)>) {});
}

void RegisterTorchTpuDialects(mlir::DialectRegistry& registry) {
  mlir::stablehlo::registerAllDialects(registry);
}

}  // namespace torch_tpu
