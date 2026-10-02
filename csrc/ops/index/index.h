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

#ifndef TORCH_TPU_CSRC_OPS_INDEX_INDEX_H_
#define TORCH_TPU_CSRC_OPS_INDEX_INDEX_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Builds StableHLO for tensor indexing with integer index tensors.
// `input_ops[0]` is `self`, and `input_ops[1..]` are the index tensors
// corresponding to `indexed_dims`.
absl::StatusOr<mlir::MlirOp> BuildIndexShlo(
    mlir::ArrayRef<mlir::MlirOp> input_ops, Indices indexed_dims);

// Builds StableHLO for tensor indexing with a single boolean mask at
// `dim_indexed`, given the precomputed `num_non_zero_elements` (`True` count).
// Performs dimension collapsing on `self`, mask flattening, nonzero index
// computation, and gathering within a single StableHLO lowering.
absl::StatusOr<mlir::MlirOp> BuildBooleanMaskIndexShlo(
    mlir::MlirOp self, mlir::MlirOp mask, int64_t dim_indexed,
    int64_t num_non_zero_elements);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_INDEX_INDEX_H_
