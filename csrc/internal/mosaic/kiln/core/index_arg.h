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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_INDEX_ARG_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_INDEX_ARG_H_

#include <concepts>
#include <cstdint>
#include <variant>

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"

namespace torch_tpu::kiln {

// An argument representing an index or dimension offset/bound.
class IndexArg {
 public:
  template <typename T>
    requires std::integral<T>
  /* implicit */ IndexArg(T v)  // NOLINT(google-explicit-constructor)
      : val_(static_cast<int64_t>(v)) {}
  /* implicit */ IndexArg(Scalar s)  // NOLINT(google-explicit-constructor)
      : val_(s) {
    ABSL_CHECK(s.IsInt()) << "IndexArg requires an integer scalar";  // CRASH_OK
  }

  bool IsInt() const { return std::holds_alternative<int64_t>(val_); }
  int64_t AsInt() const { return std::get<int64_t>(val_); }
  Scalar ToScalar() const {
    if (std::holds_alternative<Scalar>(val_)) {
      return std::get<Scalar>(val_);
    }
    return I32(std::get<int64_t>(val_));
  }

 private:
  std::variant<int64_t, Scalar> val_;
};

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_INDEX_ARG_H_
