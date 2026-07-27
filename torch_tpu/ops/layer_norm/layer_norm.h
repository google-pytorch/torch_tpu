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

#ifndef TORCH_TPU_OPS_LAYER_NORM_LAYER_NORM_H_
#define TORCH_TPU_OPS_LAYER_NORM_LAYER_NORM_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

struct LayerNormShloResults {
  mlir::MlirOp normalized_values;
  mlir::MlirOp mean;
  mlir::MlirOp reciprocal_std;
};

struct LayerNormBackwardShloResults {
  mlir::MlirOp grad_input;
  mlir::MlirOp grad_weight;
  mlir::MlirOp grad_bias;
};

// Welford's single reduction implementation.
absl::StatusOr<LayerNormShloResults> BuildLayerNormShlo(
    mlir::MlirOp input_op, std::optional<mlir::MlirOp> weight_op,
    std::optional<mlir::MlirOp> bias_op, int normalized_num_dims, double eps);

absl::StatusOr<LayerNormBackwardShloResults> BuildLayerNormBackwardShlo(
    mlir::MlirOp dy, mlir::MlirOp x, mlir::MlirOp mean, mlir::MlirOp rstd,
    std::optional<mlir::MlirOp> weight_op, at::IntArrayRef normalized_shape,
    bool compute_dbeta);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_LAYER_NORM_LAYER_NORM_H_
