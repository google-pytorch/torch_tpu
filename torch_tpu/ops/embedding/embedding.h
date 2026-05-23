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

#ifndef TORCH_TPU_OPS_EMBEDDING_EMBEDDING_H_
#define TORCH_TPU_OPS_EMBEDDING_EMBEDDING_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "c10/core/SymInt.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<MlirOpResults<4>> BuildEmbeddingBagShlo(
    mlir::MlirOp weight, mlir::MlirOp indices, mlir::MlirOp offsets,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    std::optional<mlir::MlirOp> per_sample_weights, bool include_last_offset,
    int64_t padding_idx);

absl::StatusOr<mlir::MlirOp> BuildEmbeddingBagBackwardShlo(
    mlir::MlirOp grad, mlir::MlirOp indices, mlir::MlirOp offsets,
    mlir::MlirOp offset2bag, mlir::MlirOp bag_size, mlir::MlirOp max_indices,
    at::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse,
    std::optional<mlir::MlirOp> per_sample_weights, int64_t padding_idx);

absl::StatusOr<mlir::MlirOp> BuildEmbeddingDenseBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp indices, at::SymInt num_weights,
    at::SymInt padding_idx, bool scale_grad_by_freq);

absl::StatusOr<mlir::MlirOp> BuildEmbeddingRenormShlo(mlir::MlirOp weight,
                                                      mlir::MlirOp indices,
                                                      double max_norm,
                                                      double norm_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_EMBEDDING_EMBEDDING_H_
