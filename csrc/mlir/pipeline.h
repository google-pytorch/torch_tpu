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

#ifndef TORCH_TPU_CSRC_MLIR_PIPELINE_H_
#define TORCH_TPU_CSRC_MLIR_PIPELINE_H_

#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

namespace torch_tpu {

// Adds Torch TPU optimization passes to the pass manager. `pm` must be
// anchored on `builtin.module`. Also registered as the
// `torch-tpu-optimization-pipeline` pass pipeline (see registration.h).
void AddTorchTpuOptimizationPasses(mlir::OpPassManager& pm);

// Runs all custom Torch TPU optimization passes on the given module.
absl::Status RunTorchTpuOptimizationPasses(mlir::ModuleOp module,
                                           mlir::MLIRContext* context);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PIPELINE_H_
