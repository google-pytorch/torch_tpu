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

#ifndef TORCH_TPU_OPS_ISIN_ISIN_H_
#define TORCH_TPU_OPS_ISIN_ISIN_H_

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Specifies whether elements are assumed to be unique to optimize search.
enum class IsUnique {
  kNo,
  kYes,
};

// Specifies whether matches should be inverted in the final predicate output.
enum class IsInverted {
  kNo,
  kYes,
};

absl::StatusOr<mlir::MlirOp> BuildIsInShlo(mlir::MlirOp elements,
                                           mlir::MlirOp test_elements,
                                           IsUnique uniqueness,
                                           IsInverted inversion);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_ISIN_ISIN_H_
