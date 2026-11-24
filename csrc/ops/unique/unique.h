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

#ifndef TORCH_TPU_CSRC_OPS_UNIQUE_UNIQUE_H_
#define TORCH_TPU_CSRC_OPS_UNIQUE_UNIQUE_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

enum class ReturnInverse { kFalse, kTrue };
enum class ReturnCounts { kFalse, kTrue };

[[nodiscard]] inline std::string EncodeParamCacheKey(
    ReturnInverse return_inverse) {
  return return_inverse == ReturnInverse::kTrue ? "true" : "false";
}

[[nodiscard]] inline std::string EncodeParamCacheKey(
    ReturnCounts return_counts) {
  return return_counts == ReturnCounts::kTrue ? "true" : "false";
}

struct BuildUniqueShloOutputs {
  mlir::MlirOp unique_values;
  mlir::MlirOp inverse_indices;
  mlir::MlirOp counts;
};

// Outputs of `BuildUniqueSortAndCountShlo`.
struct BuildUniqueSortAndCountShloOutputs {
  // I64[]: the number of unique values U.
  mlir::MlirOp count;
  // [N]: the flattened input, sorted in ascending order.
  mlir::MlirOp sorted;
  // I64 of the input's shape: for each input element, the index of its value
  // among the unique values. I64[0] unless `return_inverse` is kTrue.
  mlir::MlirOp inverse;
};

// The first phase of `_unique2`: sorts the flattened `input`, counts its unique
// values and, if `return_inverse` is kTrue, computes the inverse, which does
// not depend on the number of unique values. The other phases take `sorted` as
// input so that the input is sorted only once, and so that each output can be
// computed by a graph whose cache key does not depend on the data-dependent
// number of unique values.
absl::StatusOr<BuildUniqueSortAndCountShloOutputs> BuildUniqueSortAndCountShlo(
    mlir::MlirOp input, ReturnInverse return_inverse);

// Returns the `output_size` unique values of `sorted` (the `sorted` output of
// `BuildUniqueSortAndCountShlo`), in ascending order.
absl::StatusOr<mlir::MlirOp> BuildUniqueValuesFromSortedShlo(
    int64_t output_size, mlir::MlirOp sorted);

// Returns I64[output_size]: the number of occurrences of each unique value of
// `sorted` (the `sorted` output of `BuildUniqueSortAndCountShlo`).
absl::StatusOr<mlir::MlirOp> BuildUniqueCountsFromSortedShlo(
    int64_t output_size, mlir::MlirOp sorted);

struct BuildUniqueDimPrepareOutputs {
  mlir::MlirOp count;
  mlir::MlirOp p_indices;
  mlir::MlirOp mask;
};

absl::StatusOr<BuildUniqueDimPrepareOutputs> BuildUniqueDimPrepareShlo(
    mlir::MlirOp input, int64_t dim);

absl::StatusOr<BuildUniqueShloOutputs> BuildUniqueDimFinalizeShlo(
    int64_t output_size, mlir::MlirOp input, mlir::MlirOp p_indices,
    mlir::MlirOp mask, int64_t dim, ReturnInverse return_inverse,
    ReturnCounts return_counts);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_UNIQUE_UNIQUE_H_
