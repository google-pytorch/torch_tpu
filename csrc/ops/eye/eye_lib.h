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

#ifndef TORCH_TPU_OPS_EYE_EYE_LIB_H_
#define TORCH_TPU_OPS_EYE_EYE_LIB_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Builds an identity matrix with shape (m, n).
absl::StatusOr<mlir::MlirOp> BuildEyeShlo(mlir::MlirBuilder& builder,
                                          mlir::ElementType element_type,
                                          int64_t m, int64_t n);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_EYE_EYE_LIB_H_
