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

#include "csrc/mlir/pipeline.h"

#include <optional>

#include "absl/status/status.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/mlir/instrumentation.h"
#include "csrc/mlir/passes/fold_axial_and_split_half_rope.h"
#include "csrc/mlir/passes/fold_concat_softmax.h"
#include "csrc/mlir/passes/lower_separable_gather_to_dot.h"
#include "csrc/mlir/passes/reassociate_norm_weights.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "xla/mlir/utils/error_util.h"

namespace torch_tpu {
namespace {

void AddCleanupPasses(mlir::PassManager& pm) {
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
}

void AddShloPasses(mlir::PassManager& pm) {
  if (GetBooleanEnvOnce<kTorchTpuInternalEnableReassociateNormWeightsEnvVar>()
          .value_or(true)) {
    pm.addNestedPass<mlir::func::FuncOp>(createReassociateNormWeightsPass());
  }
  pm.addNestedPass<mlir::func::FuncOp>(createLowerSeparableGatherToDotPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloTargetIndependentOptimizationPass());
  pm.addNestedPass<mlir::func::FuncOp>(createFoldConcatSoftmaxPass());
  pm.addNestedPass<mlir::func::FuncOp>(createFoldAxialAndSplitHalfRopePass());
}

}  // namespace

void AddTorchTpuOptimizationPasses(mlir::PassManager& pm) {
  AddCleanupPasses(pm);
  AddShloPasses(pm);
  AddCleanupPasses(pm);
  pm.addPass(mlir::createSymbolDCEPass());
}

absl::Status RunTorchTpuOptimizationPasses(mlir::ModuleOp module,
                                           mlir::MLIRContext* context) {
  auto instrumentation_state = ConfigureInstrumentation(context, module);

  mlir::PassManager pm(context);
  if (instrumentation_state.has_value()) {
    context->disableMultithreading();
    instrumentation_state->Apply(pm);
  }

  AddTorchTpuOptimizationPasses(pm);

  mlir::BaseScopedDiagnosticHandler diag_handler(context);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by pipeline_test.
      mlir::succeeded(pm.run(module)), error::kInternal)
      << "Failed to run custom MLIR optimization passes: "
      << diag_handler.ConsumeStatus().message();
  return absl::OkStatus();
}

}  // namespace torch_tpu
