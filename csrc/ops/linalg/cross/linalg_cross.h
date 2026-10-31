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

#ifndef TORCH_TPU_CSRC_OPS_LINALG_CROSS_LINALG_CROSS_H_
#define TORCH_TPU_CSRC_OPS_LINALG_CROSS_LINALG_CROSS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Builds the StableHLO computation for the cross product of `lhs` and `rhs`
// along `dim` (which must be in `[0, rank)` and have size 3).
absl::StatusOr<mlir::MlirOp> BuildLinalgCrossShlo(mlir::MlirOp lhs,
                                                  mlir::MlirOp rhs,
                                                  int64_t dim);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_LINALG_CROSS_LINALG_CROSS_H_
