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

#ifndef TORCH_TPU_COMMON_FIXED_SIZE_SPAN_H_
#define TORCH_TPU_COMMON_FIXED_SIZE_SPAN_H_

#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"

namespace torch_tpu {

// Like absl::Span, but with a fixed size determined at compile time.
// This class is trivially copyable and movable.
template <typename T, int kSize>
class FixedSizeSpan : public absl::Span<T> {
 public:
  static_assert(kSize >= 0, "FixedSizeSpan must have non-negative size.");

  // Constructs from an absl::Span. Requires that the span has the expected
  // size.
  explicit FixedSizeSpan(const absl::Span<T> span)
      : absl::Span<T>(span.data(), kSize) {
    ABSL_CHECK_EQ(span.size(), kSize)  // CRASH_OK
        << "expected " << kSize << " elements, got " << span.size();
  }

  // Constructs from an initializer list. Requires that the initializer list
  // has the expected size.
  FixedSizeSpan(const std::initializer_list<T> list)
      : FixedSizeSpan(absl::Span<T>(list.begin(), list.size())) {}

  // Constructs from an std::array of the expected size.
  template <typename U>
  FixedSizeSpan(const std::array<U, kSize>& array)
      : FixedSizeSpan(absl::Span<T>(array.data(), kSize)) {}
};

// For supporting structured bindings in lambdas.
// 3. The `get` function to access the I-th element.
// This must be in the same namespace as the Array class to be found by ADL.
template <int I, typename T, int N>
T& get(FixedSizeSpan<T, N>& span) {
  static_assert(0 <= I && I < N, "Index out of bounds");
  return span[I];
}

template <int I, typename T, int N>
const T& get(const FixedSizeSpan<T, N>& span) {
  static_assert(0 <= I && I < N, "Index out of bounds");
  return span[I];
}

template <int I, typename T, int N>
T&& get(FixedSizeSpan<T, N>&& span) {
  static_assert(0 <= I && I < N, "Index out of bounds");
  return std::move(span[I]);
}

// Move elements from a FixedSizeSpan to an std::array of the same size.
//
// Since T might not be default constructible, we have to have all N elements
// ready in the aggregate initializer list when we create the std::array.
// Therefore we overload MoveToStdArray for different N.
template <typename T>
std::array<T, 0> MoveToStdArray(torch_tpu::FixedSizeSpan<T, 0> span) {
  return {};
}
template <typename T>
std::array<T, 1> MoveToStdArray(torch_tpu::FixedSizeSpan<T, 1> span) {
  return {std::move(span[0])};
}
template <typename T>
std::array<T, 2> MoveToStdArray(torch_tpu::FixedSizeSpan<T, 2> span) {
  return {std::move(span[0]), std::move(span[1])};
}
template <typename T>
std::array<T, 3> MoveToStdArray(torch_tpu::FixedSizeSpan<T, 3> span) {
  return {std::move(span[0]), std::move(span[1]), std::move(span[2])};
}
template <typename T>
std::array<T, 4> MoveToStdArray(torch_tpu::FixedSizeSpan<T, 4> span) {
  return {std::move(span[0]), std::move(span[1]), std::move(span[2]),
          std::move(span[3])};
}
template <typename T>
std::array<T, 5> MoveToStdArray(torch_tpu::FixedSizeSpan<T, 5> span) {
  return {std::move(span[0]), std::move(span[1]), std::move(span[2]),
          std::move(span[3]), std::move(span[4])};
}
template <typename T>
std::array<T, 6> MoveToStdArray(torch_tpu::FixedSizeSpan<T, 6> span) {
  return {std::move(span[0]), std::move(span[1]), std::move(span[2]),
          std::move(span[3]), std::move(span[4]), std::move(span[5])};
}
template <typename T>
std::array<T, 7> MoveToStdArray(torch_tpu::FixedSizeSpan<T, 7> span) {
  return {std::move(span[0]), std::move(span[1]), std::move(span[2]),
          std::move(span[3]), std::move(span[4]), std::move(span[5]),
          std::move(span[6])};
}
template <typename T>
std::array<T, 8> MoveToStdArray(torch_tpu::FixedSizeSpan<T, 8> span) {
  return {std::move(span[0]), std::move(span[1]), std::move(span[2]),
          std::move(span[3]), std::move(span[4]), std::move(span[5]),
          std::move(span[6]), std::move(span[7])};
}
// More overloads can be added as needed.

}  // namespace torch_tpu

// These specializations are needed for using FixedSizeSpan in structured
// binding. They must be in the `std` namespace.
namespace std {

// 1. Specialization of `std::tuple_size`
template <typename T, std::size_t N>
struct tuple_size<torch_tpu::FixedSizeSpan<T, N>>
    : std::integral_constant<std::size_t, N> {};

// 2. Specialization of `std::tuple_element`
template <std::size_t I, typename T, std::size_t N>
struct tuple_element<I, torch_tpu::FixedSizeSpan<T, N>> {
  using type = T;
};

}  // namespace std

#endif  // TORCH_TPU_COMMON_FIXED_SIZE_SPAN_H_
