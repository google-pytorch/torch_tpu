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

#ifndef TORCH_TPU_CSRC_OPS_NATIVE_BATCH_NORM_NATIVE_BATCH_NORM_H_
#define TORCH_TPU_CSRC_OPS_NATIVE_BATCH_NORM_NATIVE_BATCH_NORM_H_

#include <array>
#include <optional>

#include "absl/status/statusor.h"
#include "csrc/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

inline constexpr int kTorchFeaturesDimensionIndex =
    1;  // the C in an expected NCL, NCHW, or NCDHW layout.

absl::StatusOr<MlirOpResults<3>> BuildBatchNorm(
    mlir::MlirOp input, std::optional<mlir::MlirOp> weight_opt,
    std::optional<mlir::MlirOp> bias_opt,
    std::optional<mlir::MlirOp> running_mean_opt,
    std::optional<mlir::MlirOp> running_variance_opt, bool training,
    double momentum, double eps, mlir::ElementType acc_dtype);

// Builds training-mode batch norm and updates the running stats as PyTorch
// does:
//   running_mean = (1 - momentum) * running_mean + momentum * mean
//   running_var = (1 - momentum) * running_var + momentum * unbiased_var
// `momentum` is a scalar tensor. Returns (output, save_mean, save_invstd,
// new_running_mean, new_running_variance); the new running stats have the
// element types of `running_mean` and `running_variance`.
absl::StatusOr<MlirOpResults<5>> BuildBatchNormTrainingAndUpdateStats(
    mlir::MlirOp input, std::optional<mlir::MlirOp> weight_opt,
    std::optional<mlir::MlirOp> bias_opt, mlir::MlirOp running_mean,
    mlir::MlirOp running_variance, mlir::MlirOp momentum, double eps,
    mlir::ElementType acc_dtype);

absl::StatusOr<MlirOpResults<3>> BuildBatchNormBackward(
    mlir::MlirOp grad_out, mlir::MlirOp input,
    std::optional<mlir::MlirOp> weight_opt,
    std::optional<mlir::MlirOp> running_mean_opt,
    std::optional<mlir::MlirOp> running_variance_opt,
    std::optional<mlir::MlirOp> save_mean_opt,
    std::optional<mlir::MlirOp> save_invstd_opt, bool training, double eps,
    std::array<bool, 3> output_mask, mlir::ElementType acc_dtype);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_NATIVE_BATCH_NORM_NATIVE_BATCH_NORM_H_
