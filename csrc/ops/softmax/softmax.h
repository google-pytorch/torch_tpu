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

#ifndef TORCH_TPU_OPS_SOFTMAX_SOFTMAX_H_
#define TORCH_TPU_OPS_SOFTMAX_SOFTMAX_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

enum class SoftmaxMode { kSoftmax, kLogSoftmax };

// Encodes a SoftmaxMode enum as a parameter cache key string.
[[nodiscard]] inline std::string EncodeParamCacheKey(
    const SoftmaxMode softmax_mode) {
  switch (softmax_mode) {
    case SoftmaxMode::kSoftmax:
      return "softmax";
    case SoftmaxMode::kLogSoftmax:
      return "logsoftmax";
  }
}

absl::StatusOr<mlir::MlirOp> BuildSoftmaxShlo(
    mlir::MlirOp input_op, int64_t dim,
    SoftmaxMode softmax_mode = SoftmaxMode::kSoftmax);

absl::StatusOr<mlir::MlirOp> BuildSoftmaxBackwardDataShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp output_op, int64_t dim,
    mlir::stablehlo::Precision precision,
    SoftmaxMode softmax_mode = SoftmaxMode::kSoftmax);

absl::StatusOr<mlir::MlirOp> BuildMaskedSoftmaxShlo(
    mlir::MlirOp input_op, mlir::MlirOp mask_op, int64_t dim,
    std::optional<int64_t> mask_type = std::nullopt);

absl::StatusOr<mlir::MlirOp> BuildMaskedSoftmaxBackwardDataShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp output_op, mlir::MlirOp mask_op,
    int64_t dim, mlir::stablehlo::Precision precision);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SOFTMAX_SOFTMAX_H_
