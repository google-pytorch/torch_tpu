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
#ifndef TORCH_TPU_OPS_TRIL_INDICES_TRIL_INDICES_H_
#define TORCH_TPU_OPS_TRIL_INDICES_TRIL_INDICES_H_
#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildTrilIndicesShlo(mlir::MlirBuilder& builder,
                                                  int64_t row, int64_t col,
                                                  int64_t offset,
                                                  int64_t tril_size,
                                                  mlir::Type element_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_TRIL_INDICES_TRIL_INDICES_H_
