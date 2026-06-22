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

#ifndef TORCH_TPU_UTILS_H_
#define TORCH_TPU_UTILS_H_

// Generic utilities for torch_tpu.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "c10/util/DimVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "torch/headeronly/util/complex.h"
#include "torch_tpu/common/dimension_types.h"

// 1 if this is a Google-internal version of torch_tpu. Otherwise 0.
#ifndef TT_IS_INTERNAL_TORCH_TPU
#define TT_IS_INTERNAL_TORCH_TPU 0
#endif

namespace torch_tpu {

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

// This function is deterministic within a single binary execution. However, the
// resulting seed is NOT guaranteed to be stable across different standard
// library implementations (e.g., libc++ vs. libstdc++), different compiler
// versions, or different CPU architectures.
//
// In a distributed P2P context, this is safe as long as all participating nodes
// are running identical binaries.
template <typename T>
inline void HashCombine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

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

#endif  // TORCH_TPU_UTILS_H_
