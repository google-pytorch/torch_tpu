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

#ifndef TORCH_TPU_OPS_INDEX_PUT_INDEX_PUT_H_
#define TORCH_TPU_OPS_INDEX_PUT_INDEX_PUT_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"

namespace torch_tpu {
// Builds StableHLO operations to implement the semantics of index_put operation
//
// Parameters:
//   self: The 'self' tensor to be updated.
//   indices: An ArrayRef containing the index tensors.
//   index_start_dim: The starting dimension in self indexed by the index
//                  tensors. Index tensors must be contiguous in self starting
//                  at index_start_dim.
//   index_end_dim: The ending dimension in self indexed by the index tensors.
//                  Index tensors must be contiguous in self ending at
//                  index_end_dim.
//   index_broadcast_shape: A Dimensions vector representing the common shape
//                          that all index tensors are broadcastable to.
//   values: The 'values' tensor providing the updates.
//   values_broadcast_shape: A Dimensions vector representing the shape to
//                           which the input 'values' tensor is broadcastable
//                           to. This shape is formed by concatenating
//                           `index_broadcast_shape` with the sizes of any
//                           unindexed dimensions of 'self', in their original
//                           relative order.
//   accumulate: A boolean flag. If true, the elements from the 'values' tensor
//               are added to the existing elements in the 'self' tensor at the
//               indexed locations. If false, the elements are overwritten.
//
absl::StatusOr<mlir::MlirOp> BuildIndexPutShlo(
    mlir::MlirOp self, mlir::ArrayRef<mlir::MlirOp> indices,
    int64_t index_start_dim, int64_t index_end_dim,
    const Dimensions& index_broadcast_shape, mlir::MlirOp values,
    const Dimensions& values_broadcast_shape, bool accumulate);

// Builds StableHLO operations to implement the semantics of index_put operation
// when the index is a mask tensor.
//
// Parameters:
//   self: The 'self' tensor to be updated.
//   mask: The boolean mask tensor.
//   index_start_dim: The starting dimension in self indexed by the mask tensor.
//   values: The 'values' tensor providing the updates. This must be a scalar.
//   accumulate: A boolean flag. If true, the elements from the 'values' tensor
//               are added to the existing elements in the 'self' tensor at the
//               indexed locations. If false, the elements are overwritten.
//
absl::StatusOr<mlir::MlirOp> BuildIndexPutSelectShlo(mlir::MlirOp self,
                                                     mlir::MlirOp mask,
                                                     int64_t index_start_dim,
                                                     mlir::MlirOp values,
                                                     bool accumulate);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_INDEX_PUT_INDEX_PUT_H_
