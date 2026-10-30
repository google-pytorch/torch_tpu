/*
 * Copyright 2025 Google LLC
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

#ifndef TORCH_TPU_CSRC_OPS_NLL_LOSS_NLL_LOSS_H_
#define TORCH_TPU_CSRC_OPS_NLL_LOSS_NLL_LOSS_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "csrc/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Gathers class weights from 1D `weight` ([C]) using `safe_target` of arbitrary
// shape ([N, d1, ..., dk]). Flattens `safe_target` to 1D before calling
// `BuildIndexSelectShlo` and reshapes the result back to `safe_target`'s shape.
mlir::MlirOp GatherWeightsShlo(mlir::MlirOp weight, mlir::MlirOp safe_target);

// How `BuildNllLossForwardOutShlo` reads `self` at each target class.
enum class NllTargetSelection {
  // Gathers one element per sample, so only those elements of `self` are read.
  kGather,
  // Sums `self` over the class dimension with every other class masked to
  // zero. All uses of `self` are then reductions, so when `self` is computed in
  // the same program (as cross_entropy does with log_softmax), XLA can fuse it
  // into those reductions instead of materializing it.
  kMaskedSum,
};

absl::StatusOr<MlirOpResults<2>> BuildNllLossForwardOutShlo(
    mlir::MlirOp self, mlir::MlirOp target, std::optional<mlir::MlirOp> weight,
    int64_t reduction, int64_t ignore_index, mlir::MlirBuilder& builder,
    NllTargetSelection target_selection = NllTargetSelection::kGather);

absl::StatusOr<mlir::MlirOp> BuildNllLossBackwardGradInputShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction, int64_t ignore_index,
    mlir::MlirOp total_weight, mlir::MlirBuilder& builder);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_NLL_LOSS_NLL_LOSS_H_
