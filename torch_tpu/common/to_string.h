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

#ifndef TORCH_TPU_COMMON_TO_STRING_H_
#define TORCH_TPU_COMMON_TO_STRING_H_

#include <array>
#include <cstddef>
#include <iterator>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/OptionalArrayRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// This library defines the ToString() function template and overloads that can
// convert an arbitrary type used in TorchTPU to a human-readable string useful
// for logging and error messages. Depending on the input type, ToString() may
// return either a std::string or a std::string_view.

////////////////////////////////////////////////////////////////////////////////
// Forward declarations needed for recursive definitions.

// The primary template is for types where ToString(const T&) is not defined.
template <typename T>
std::string ToString(const T& x);

template <typename T>
[[nodiscard]] std::string ToString(absl::Span<T> span);
template <typename T>
[[nodiscard]] std::string ToString(c10::OptionalArrayRef<T> value);
template <typename T>
[[nodiscard]] std::string ToString(c10::ArrayRef<T> arr);
template <typename T>
[[nodiscard]] std::string ToString(const c10::List<T>& list);
template <typename T>
[[nodiscard]] std::string ToString(const c10::IListRef<T>& list);
template <typename T>
[[nodiscard]] std::string ToString(mlir::ArrayRef<T> arr);
template <typename T>
[[nodiscard]] std::string ToString(absl::Span<T> span);
template <typename T, size_t N>
[[nodiscard]] std::string ToString(const absl::InlinedVector<T, N>& vec);
template <typename T>
[[nodiscard]] std::string ToString(const std::optional<T>& value);
template <typename T1, typename T2>
[[nodiscard]] std::string ToString(  //
    const std::pair<T1, T2>& pair);  // STD_PAIR_OK=generic code.
template <typename T>
[[nodiscard]] std::string ToString(const std::vector<T>& vec);
template <typename T, size_t kSize>
[[nodiscard]] std::string ToString(const std::array<T, kSize>& arr);
template <typename K, typename V>
[[nodiscard]] std::string ToString(const std::unordered_map<K, V>& map);
template <typename Iterator>
[[nodiscard]] std::string ToString(Iterator begin, Iterator end);

////////////////////////////////////////////////////////////////////////////////
// ToString() for PyTorch types.

[[nodiscard]] std::string ToString(c10d::ReduceOp::RedOpType reduce_op_type);
[[nodiscard]] std::string ToString(const at::Tensor& tensor,
                                   const std::string& name = "");
[[nodiscard]] std::string ToString(const at::Scalar& scalar,
                                   const std::string& name = "");

// Returns a string representation of the given type as a PyTorch dtype name
// (e.g. "float32", "int8"). This is suitable for use in user messages.
[[nodiscard]] std::string_view ToString(at::ScalarType scalar_type);

[[nodiscard]] std::string ToString(const c10d::AllgatherOptions& options);
[[nodiscard]] std::string ToString(const c10d::BroadcastOptions& options);
[[nodiscard]] std::string ToString(const c10d::ReduceScatterOptions& options);
[[nodiscard]] std::string ToString(const c10d::ScatterOptions& options);
[[nodiscard]] std::string ToString(const c10d::GatherOptions& options);
[[nodiscard]] std::string ToString(const c10d::AllToAllOptions& options);
[[nodiscard]] std::string ToString(const c10d::BarrierOptions& options);

[[nodiscard]] inline constexpr std::string_view ToString(const at::Generator&) {
  return "at::Generator";
}

[[nodiscard]] inline std::string ToString(
    const c10d::AllreduceOptions& options) {
  return ToString(options.reduceOp);
}

template <typename T>
[[nodiscard]] std::string ToString(c10::OptionalArrayRef<T> value) {
  return value.has_value() ? absl::StrCat("<", ToString(value.value()), ">")
                           : "nullopt";
}

template <typename T>
[[nodiscard]] std::string ToString(c10::ArrayRef<T> arr) {
  return ToString(absl::MakeConstSpan(arr));
}

template <typename T>
[[nodiscard]] std::string ToString(const c10::List<T>& list) {
  // List doesn't have contiguous storage, so we can't use absl::Span directly.
  return ToString(list.begin(), list.end());
}

template <typename T>
[[nodiscard]] inline std::string ToString(const c10::IListRef<T>& list) {
  // IListRef doesn't have contiguous storage, so we can't use absl::Span
  // directly.
  return ToString(list.begin(), list.end());
}

////////////////////////////////////////////////////////////////////////////////
// ToString() for MLIR types.

[[nodiscard]] std::string ToString(mlir::Type type);

template <typename T>
[[nodiscard]] std::string ToString(mlir::ArrayRef<T> arr) {
  return ToString(absl::MakeConstSpan(arr));
}

////////////////////////////////////////////////////////////////////////////////
// ToString() for absl types.

// Returns a string representation of the given span. Requires the element type
// to support ToString().
template <typename T>
[[nodiscard]] std::string ToString(absl::Span<T> span) {
  return ToString(span.begin(), span.end());
}

template <typename T, size_t N>
[[nodiscard]] std::string ToString(const absl::InlinedVector<T, N>& vec) {
  return ToString(absl::MakeConstSpan(vec));
}

////////////////////////////////////////////////////////////////////////////////
// ToString() for standard C++ types.

[[nodiscard]] inline std::string_view ToString(const bool b) {
  return b ? "true" : "false";
}

template <typename Iterator>
[[nodiscard]] std::string ToString(Iterator begin, Iterator end) {
  using V = typename std::iterator_traits<Iterator>::value_type;
  return absl::StrCat(
      "[",
      absl::StrJoin(begin, end, ", ",
                    [](std::string* out, const auto& elem) {
                      absl::StrAppend(out,
                                      ToString(static_cast<const V&>(elem)));
                    }),
      "]");
}

template <typename T>
[[nodiscard]] std::string ToString(const std::optional<T>& value) {
  // The "<>" helps to distinguish between an empty optional and a nullopt
  // for nested optionals like optional<optional<int>>.
  return value.has_value() ? absl::StrCat("<", ToString(value.value()), ">")
                           : "nullopt";
}

// Returns a string representation of the given pair.
// Both element types in the pair must support ToString().
template <typename T1, typename T2>
[[nodiscard]] std::string ToString(const std::pair<T1, T2>& pair) {
  return absl::StrCat("(", ToString(pair.first), ", ", ToString(pair.second),
                      ")");
}

// Returns a string representation of the given vector. Requires the element
// type to support ToString().
template <typename T>
[[nodiscard]] std::string ToString(const std::vector<T>& vec) {
  return ToString(absl::MakeConstSpan(vec));
}

template <typename T, size_t kSize>
[[nodiscard]] std::string ToString(const std::array<T, kSize>& arr) {
  return ToString(absl::MakeConstSpan(arr));
}

template <typename K, typename V>
[[nodiscard]] std::string ToString(const std::unordered_map<K, V>& map) {
  return ToString(map.begin(), map.end());
}

////////////////////////////////////////////////////////////////////////////////
// The primary ToString() template implementation.

// Trait: does T support absl::StrCat()?
template <typename T, typename = void>
struct supports_absl_strcat : std::false_type {};
template <typename T>
struct supports_absl_strcat<
    T, std::void_t<decltype(absl::StrCat(std::declval<const T&>()))>>
    : std::true_type {};

// Trait: does T support operator<<()?
template <typename T, typename = void>
struct supports_ostream : std::false_type {};
template <typename T>
struct supports_ostream<T, std::void_t<decltype(std::declval<std::ostream&>()
                                                << std::declval<const T&>())>>
    : std::true_type {};

// Trait: does T support .ToString()?
template <typename T, typename = void>
struct supports_tostring_method : std::false_type {};
template <typename T>
struct supports_tostring_method<
    T, std::void_t<decltype(std::declval<const T&>().ToString())>>
    : std::true_type {};

template <typename T>
inline constexpr bool always_false_v = false;

// The primary ToString() template. It delegates to absl::StrCat(), the
// .ToString() method, or operator<< depending on which is supported by T.
template <typename T>
std::string ToString(const T& x) {
  if constexpr (supports_tostring_method<T>::value) {
    // Prefer .ToString() as it's generally more efficient than
    // operator<<.
    return x.ToString();
  } else if constexpr (supports_ostream<T>::value) {
    // Next, prefer operator<< as it's generally more correct than
    // absl::StrCat(). E.g. many enum types support operator<< that
    // prints the enum name, but absl::StrCat() just prints the underlying
    // integer value.
    std::ostringstream os;
    os << x;
    return os.str();
  } else if constexpr (supports_absl_strcat<T>::value) {
    return absl::StrCat(x);
  } else {
    static_assert(always_false_v<T>,
                  "ToString() requires the type to support absl::StrCat, "
                  "operator<<, or x.ToString().");
  }
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_TO_STRING_H_
