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

#ifndef TORCH_TPU_COMMON_FINGERPRINT_UTILS_H_
#define TORCH_TPU_COMMON_FINGERPRINT_UTILS_H_

// Utilities for computing fingerprints.
//
// This file should only contain specializations for common STL and Abseil
// types. Specializations for TorchTPU-specific data types, such as `OpName`,
// should be added in their respective headers to prevent circular dependencies.

#include <cstddef>
#include <cstdint>
#include <map>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "tsl/platform/fingerprint.h"

namespace torch_tpu {

// Represents the fingerprint of a chunk of data.
using FingerprintType = uint64_t;  // 8 bytes

template <typename T>
[[nodiscard]] FingerprintType Fingerprint(const T& t);

namespace internal {

// Helper for computing fingerprints of various types. We use a struct instead
// of a function to allow for partial specialization.
//
// The primary template is used for non-integral types.
template <typename T,
          bool kIsSmallIntegral =
              // is_integral_v does NOT include enums. This is intentional as
              // enum numerical values are not guaranteed to be stable.
          std::is_integral_v<T> && sizeof(T) <= sizeof(FingerprintType)>
struct Fingerprint64Impl {
  static_assert(!kIsSmallIntegral,
                "The primary template should be instantiated only for "
                "non-integral types.");

  // Returns the fingerprint of the given value.
  [[nodiscard]] static FingerprintType Compute(const T& value) {
    return tsl::Fingerprint64(value);
  }
};

// Partial specialization for absl::Span.
template <typename T>
struct Fingerprint64Impl<absl::Span<T>, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(absl::Span<T> span) {
    FingerprintType fp = span.size();
    for (const T& element : span) {
      fp = tsl::FingerprintCat64(fp, Fingerprint(element));
    }
    return fp;
  }
};

// Partial specialization for std::vector.
template <typename T>
struct Fingerprint64Impl<std::vector<T>, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(const std::vector<T>& span) {
    return Fingerprint(absl::MakeConstSpan(span));
  }
};

// Partial specialization for absl::InlinedVector.
template <typename T, size_t N>
struct Fingerprint64Impl<absl::InlinedVector<T, N>,
                         /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(
      const absl::InlinedVector<T, N>& span) {
    return Fingerprint(absl::MakeConstSpan(span));
  }
};

// Partial specialization for std::map.
template <typename K, typename V>
struct Fingerprint64Impl<std::map<K, V>, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(const std::map<K, V>& m) {
    FingerprintType fp = m.size();
    for (const auto& pair : m) {
      fp = tsl::FingerprintCat64(fp, Fingerprint(pair));
    }
    return fp;
  }
};

// Specializations for floating point types.
template <>
struct Fingerprint64Impl<double, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(double value) {
    return Fingerprint(absl::bit_cast<uint64_t>(value));
  }
};

template <>
struct Fingerprint64Impl<float, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(float value) {
    return Fingerprint(absl::bit_cast<uint32_t>(value));
  }
};

// Partial specialization for std::pair.
template <typename T, typename U>
struct Fingerprint64Impl<std::pair<T, U>,  // STD_PAIR_OK=generic code.
                         /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(const std::pair<T, U>& pair) {
    return tsl::FingerprintCat64(Fingerprint(pair.first),
                                 Fingerprint(pair.second));
  }
};

// Partial specialization for std::variant.
template <typename... Ts>
struct Fingerprint64Impl<std::variant<Ts...>, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(const std::variant<Ts...>& v) {
    FingerprintType fp = v.index();
    std::visit(
        [&fp](const auto& val) {
          fp = tsl::FingerprintCat64(fp, Fingerprint(val));
        },
        v);
    return fp;
  }
};

// Partial specialization for integral types that are small enough to fit
// in a FingerprintType.
template <typename T>
struct Fingerprint64Impl<T, /*kIsSmallIntegral=*/true> {
  [[nodiscard]] static FingerprintType Compute(T value) {
    // Since casting T to FingerprintType doesn't lose information, we can
    // safely use it as the fingerprint. This is super fast.
    return static_cast<FingerprintType>(value);
  }
};

}  // namespace internal

// Returns the fingerprint of the given value.
template <typename T>
[[nodiscard]] FingerprintType Fingerprint(const T& t) {
  return internal::Fingerprint64Impl<T>::Compute(t);
}

// FingerprintCat(ts...) returns the fingerprint of the given values.
//
// The base cases.
[[nodiscard]] inline FingerprintType FingerprintCat() { return 0; }
template <typename T>
[[nodiscard]] FingerprintType FingerprintCat(const T& t) {
  return Fingerprint(t);
}
// The recursive case.
template <typename T, typename... Ts>
[[nodiscard]] FingerprintType FingerprintCat(const T& t, const Ts&... ts) {
  return tsl::FingerprintCat64(Fingerprint(t), FingerprintCat(ts...));
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_FINGERPRINT_UTILS_H_
