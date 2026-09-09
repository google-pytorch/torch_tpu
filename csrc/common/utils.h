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

#ifndef TORCH_TPU_CSRC_COMMON_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_UTILS_H_

// Generic utilities for torch_tpu.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "c10/util/DimVector.h"
#include "csrc/common/dimension_types.h"
#include "llvm/ADT/ArrayRef.h"
#include "torch/headeronly/util/complex.h"

// 1 if this is a Google-internal version of torch_tpu. Otherwise 0.
#ifndef TT_IS_INTERNAL_TORCH_TPU
#define TT_IS_INTERNAL_TORCH_TPU 0
#endif

#if !defined(NDEBUG) && TT_IS_INTERNAL_TORCH_TPU
// Enables error messages style check for internal debug builds.
//
// When this macro is set to 1, TorchTPU will hard crash if any error is raised
// with an error message that doesn't conform to the TorchTPU error handling
// guidelines. This should help us maintain the quality and consistency of error
// messages in the project.
#define TT_CHECKS_ERROR_FORMAT 1
#endif

namespace torch_tpu {

// always_false_v<T> is always false.
//
// This is useful for static_assert(always_false_v<T>, ...) to trigger the
// compiler to generate an error message about type T.
template <typename T>
inline constexpr bool always_false_v = false;

// Returns a copy of the array of integers.
[[nodiscard]] inline SmallInt64Vector CopyIntVector(at::IntArrayRef ints) {
  return SmallInt64Vector(ints.begin(), ints.end());
}
[[nodiscard]] inline SmallInt64Vector CopyIntVector(
    const c10::DimVector& ints) {
  return SmallInt64Vector(ints.begin(), ints.end());
}
[[nodiscard]] inline SmallInt64Vector CopyIntVector(
    absl::Span<const int64_t> ints) {
  return SmallInt64Vector(ints.begin(), ints.end());
}
[[nodiscard]] inline SmallInt64Vector CopyIntVector(
    llvm::ArrayRef<int64_t> ints) {
  return SmallInt64Vector(ints.begin(), ints.end());
}
[[nodiscard]] inline SmallInt64Vector CopyIntVector(
    const std::vector<int64_t>& ints  // INT_VEC_OK
) {
  return SmallInt64Vector(ints.begin(), ints.end());
}

// Returns a string representation of the given value that can be losslessly
// converted back to the original value. Use this when computing the computation
// cache key for a double/complex value to avoid different values mapping to the
// same cache key.
[[nodiscard]] std::string LosslessToString(double value);
[[nodiscard]] std::string LosslessToString(const c10::complex<double>& value);

// Log a very long string as multiple lines. This is to work around ABSL_LOG's
// limitation on a maximum line length.
void LogLines(std::string_view s);

// Format percentage.
std::string PercAsStr(uint64_t num, uint64_t den);

// Filter indices [0, until] using the given `predicate`.
template <typename Predicate>
Indices FilterIndices(size_t until, const Predicate& predicate) {
  Indices indices(until, 0);

  absl::c_iota(indices, 0);
  // Move indices where `predicate(i)` is `true`, first.
  auto filtered_indices_end = absl::c_stable_partition(indices, predicate);
  // Remove all indices after the last `true` index.
  indices.erase(filtered_indices_end, indices.end());

  return indices;
}

// Does the given tensor have a trivial and standard layout in memory.
bool TensorHasTrivialLayout(const at::Tensor& tensor);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_UTILS_H_
