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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_UTILS_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_UTILS_H_

#include "absl/status/statusor.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {

// Returns the Kiln dtype for `element_type` (PRED, I8 to I64, UI8 to UI64, F16,
// BF16, or F32), or an error if Kiln has no matching dtype.
absl::StatusOr<kiln::DType> ToKilnDType(mlir::ElementType element_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_UTILS_H_
