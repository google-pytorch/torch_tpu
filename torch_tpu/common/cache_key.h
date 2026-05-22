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

#ifndef TORCH_TPU_COMMON_CACHE_KEY_H_
#define TORCH_TPU_COMMON_CACHE_KEY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Dimname.h"
#include "c10/core/Device.h"
#include "c10/core/Scalar.h"  // IWYU pragma: keep for c10::Scalar
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/Optional.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/compile_options_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

class PromotedScalar;
class MaybePromotedScalar;

enum class ScalarValue { kZero, kOne };

namespace internal {

struct PromotedScalarState {
  // Type of a function that turns a Scalar into a Tensor.
  using Promoter = absl::AnyInvocable<absl::StatusOr<at::Tensor>(
      const at::Scalar&, std::optional<at::ScalarType>) const>;

  Promoter promoter;
  at::Scalar scalar;
  bool tensor_used = false;
};

void AppendPromotedScalarPointers(
    std::vector<const PromotedScalarState* absl_nonnull>& promoted_scalars,
    const PromotedScalar& arg);

void AppendPromotedScalarPointers(
    std::vector<const PromotedScalarState* absl_nonnull>& promoted_scalars,
    const MaybePromotedScalar& arg);

}  // namespace internal

// A scalar value that is promoted to a tensor lazily.
class PromotedScalar {
 public:
  using State = internal::PromotedScalarState;

  // Type of a function that turns a Scalar into a Tensor.
  using Promoter = State::Promoter;

  // Promotes the given scalar to a tensor. We make the promoter a parameter
  // rather than a hard-coded MakeTensor() to avoid a circular dependency
  // between cache_key.h and op_dispatcher.h.
  PromotedScalar(Promoter promoter, at::Scalar scalar);

  // This class is move-only.
  PromotedScalar(PromotedScalar&& other) = default;
  PromotedScalar& operator=(PromotedScalar&& other) = default;
  PromotedScalar(const PromotedScalar&) = delete;
  PromotedScalar& operator=(const PromotedScalar&) = delete;

  // Returns the scalar value.
  [[nodiscard]] const at::Scalar& scalar() const { return state_->scalar; }

  // Returns the tensor value. Must be called at least once when the op
  // succeeds.
  absl::StatusOr<at::Tensor> GetTensor(
      std::optional<at::ScalarType> scalar_type_opt = std::nullopt);

  // Formats the scalar for logging.
  [[nodiscard]] std::string ToString() const;

  // Transforms this object to a MaybePromotedScalar object that can be
  // conditionally promoted.
  [[nodiscard]] MaybePromotedScalar AvoidPromoting(ScalarValue exclude) &&;
  [[nodiscard]] MaybePromotedScalar AvoidPromoting(ScalarValue exclude1,
                                                   ScalarValue exclude2) &&;

 private:
  friend class MaybePromotedScalar;
  friend void internal::AppendPromotedScalarPointers(
      std::vector<const State* absl_nonnull>& promoted_scalars,
      const PromotedScalar& arg);

  absl_nonnull std::unique_ptr<State> state_;
};

// A scalar value that may or may not be promoted to a tensor, depending on
// whether its value matches any of the specified excluded values.
//
// If the value matches an exclude, it is not promoted and its value is
// included in the cache key. Otherwise, it is promoted to a tensor and
// ignored in the cache key.
class MaybePromotedScalar {
 public:
  // Constructs a MaybePromotedScalar from a PromotedScalar and a value to
  // avoid promoting.
  MaybePromotedScalar(PromotedScalar s, ScalarValue exclude);

  // Constructs a MaybePromotedScalar from a PromotedScalar and two values to
  // avoid promoting.
  MaybePromotedScalar(PromotedScalar s, ScalarValue exclude1,
                      ScalarValue exclude2);

  // Returns true if the scalar value is zero.
  [[nodiscard]] bool IsZero() const;

  // Returns true if the scalar value is one.
  [[nodiscard]] bool IsOne() const;

  // Returns the tensor value. Must be called at least once when the op
  // succeeds, unless the value matches an exclude.
  absl::StatusOr<at::Tensor> GetTensor(
      std::optional<at::ScalarType> scalar_type_opt = std::nullopt);

  // Formats the scalar for logging.
  [[nodiscard]] std::string ToString() const;

  // Returns true if the scalar value matches any of the excluded values.
  [[nodiscard]] bool ValueMatchesExclude() const {
    return value_matches_exclude_;
  }

  // Returns the scalar value.
  [[nodiscard]] const at::Scalar& scalar() const {
    return promoted_scalar_.scalar();
  }

 private:
  friend void internal::AppendPromotedScalarPointers(
      std::vector<const internal::PromotedScalarState* absl_nonnull>&
          promoted_scalars,
      const MaybePromotedScalar& arg);

  // Returns the underlying state.
  [[nodiscard]] const std::unique_ptr<PromotedScalar::State>& state() const {
    return promoted_scalar_.state_;
  }

  // Returns true if the scalar value matches any of the given values.
  [[nodiscard]] bool MatchesAny(absl::Span<const ScalarValue> values) const;

  PromotedScalar promoted_scalar_;
  bool value_matches_exclude_ = false;
};

namespace internal {

// Wrapper for a value that should be ignored in the cache key computation.
template <typename T>
class IgnoredInCacheKey {
 public:
  using value_type = T;

  explicit IgnoredInCacheKey(const T& value) : value_ref_(value) {}

  [[nodiscard]] const T& value() const { return value_ref_; }

  // Supports logging.
  friend auto ToString(const IgnoredInCacheKey& value) {
    using torch_tpu::ToString;
    return ToString(value.value_ref_);
  }

 private:
  const T& value_ref_;
};

// Trait to check if T is IgnoredInCacheKey.
template <typename T>
struct is_ignored_in_cache_key : std::false_type {};
template <typename T>
struct is_ignored_in_cache_key<IgnoredInCacheKey<T>> : std::true_type {};

// Returns true if an op parameter of type T should be included in the parameter
// cache key computation.
//
// Some parameters (e.g. Tensors) are not needed in the parameter cache key
// computation as their information is already included in the cache key
// automatically.
template <typename T>
constexpr bool IncludeInCacheKey() {
  using U = std::decay_t<T>;
  // If the argument is wrapped in IgnoreInCacheKey(), it should not be
  // included in the cache key.
  if constexpr (is_ignored_in_cache_key<U>::value) {
    return false;
  }
  return !(
      // go/keep-sorted start
      std::is_same_v<U, PromotedScalar> ||                        //
      std::is_same_v<U, at::ArrayRef<at::Tensor>> ||              //
      std::is_same_v<U, at::Generator> ||                         //
      std::is_same_v<U, at::ITensorListRef> ||                    //
      std::is_same_v<U, at::Tensor> ||                            //
      std::is_same_v<U, at::TensorList> ||                        //
      std::is_same_v<U, c10d::AllToAllOptions> ||                 //
      std::is_same_v<U, c10d::AllgatherOptions> ||                //
      std::is_same_v<U, c10d::BarrierOptions> ||                  //
      std::is_same_v<U, std::vector<PromotedScalar>> ||           //
      std::is_same_v<U, std::vector<at::Tensor>> ||               //
      std::is_same_v<U, std::vector<std::vector<at::Tensor>>> ||  //
      // go/keep-sorted end
      false);
}

// FormatParamCacheKey() is a family of function templates and overloads that
// convert different op parameter types to strings, which will be used to
// compute the op cache key. They may return either a string or a
// StatusOr<std::string>.
//
// For each given value type, they must guarantee that different values
// produce different strings.
//
// FormatParamCacheKey() should be defined only for types that need to be
// included in the cache key computation.
absl::StatusOr<std::string> FormatParamCacheKey(at::Scalar value);
[[nodiscard]] std::string FormatParamCacheKey(const c10::SymInt& value);
[[nodiscard]] std::string FormatParamCacheKey(c10::SymIntArrayRef value);
[[nodiscard]] inline std::string FormatParamCacheKey(
    const at::ScalarType value) {
  return c10::toString(value);
}
[[nodiscard]] inline std::string FormatParamCacheKey(const at::Dimname value) {
  return std::string(value.symbol().toQualString());
}
[[nodiscard]] std::string FormatParamCacheKey(c10d::ReduceOp value);
[[nodiscard]] inline std::string FormatParamCacheKey(
    const mlir::ElementType value) {
  return std::string(ToShortString(value));
}
[[nodiscard]] inline std::string FormatParamCacheKey(
    const mlir::stablehlo::Precision value) {
  return std::string(mlir::stablehlo::stringifyPrecision(value));
}
[[nodiscard]] inline std::string FormatParamCacheKey(const double value) {
  return LosslessToString(value);
}
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
[[nodiscard]] inline std::string FormatParamCacheKey(const T value) {
  return absl::StrCat(value);
}
[[nodiscard]] inline std::string FormatParamCacheKey(const bool value) {
  return value ? "t" : "f";
}
[[nodiscard]] std::string FormatParamCacheKey(std::string_view value);
[[nodiscard]] inline std::string FormatParamCacheKey(const char* const value) {
  return FormatParamCacheKey(std::string_view(value));
}
[[nodiscard]] inline std::string FormatParamCacheKey(const std::string& value) {
  return FormatParamCacheKey(std::string_view(value));
}
[[nodiscard]] std::string FormatParamCacheKey(at::Layout value);
[[nodiscard]] std::string FormatParamCacheKey(at::MemoryFormat value);
[[nodiscard]] std::string FormatParamCacheKey(at::Device value);

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10::optional<at::Tensor>& value) {
  // Encode both the presence and the definedness of the tensor.
  return value.has_value() ? (value->defined() ? "t" : "u") : "";
}

[[nodiscard]] std::string FormatParamCacheKey(
    const std::optional<PromotedScalar>& value);

[[nodiscard]] std::string FormatParamCacheKey(const MaybePromotedScalar& value);

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10::optional<at::Generator>& value) {
  return value.has_value() ? "g" : "";
}

// The return type of FormatParamCacheKey(const T&). Can be either std::string
// or absl::StatusOr<std::string>.
template <typename T>
using FormattedKey = decltype(FormatParamCacheKey(std::declval<T>()));

// To guarantee that the correct overload of FormatParamCacheKey() is found,
// overloads that invoke other FormatParamCacheKey() overloads should be
// declared first and their definitions should be put after all the other
// overloads.
template <typename T>
FormattedKey<T> FormatParamCacheKey(const std::optional<T>& value);
template <typename T1, typename T2>
std::string FormatParamCacheKey(      //
    const std::pair<T1, T2>& value);  // STD_PAIR_OK=generic code.
template <typename T, std::size_t kSize>
FormattedKey<T> FormatParamCacheKey(const std::array<T, kSize>& value);
template <typename T>
FormattedKey<T> FormatParamCacheKey(const std::vector<T>& value);
template <typename T, size_t kSize, typename Allocator>
FormattedKey<T> FormatParamCacheKey(
    const absl::InlinedVector<T, kSize, Allocator>& value);
template <typename T>
FormattedKey<T> FormatParamCacheKey(at::ArrayRef<T> value);
template <typename T>
FormattedKey<T> FormatParamCacheKey(absl::Span<T> value);
template <typename K, typename V, typename Hash, typename Eq, typename Alloc>
std::string FormatParamCacheKey(
    const std::unordered_map<K, V, Hash, Eq, Alloc>& value);
template <typename T>
FormattedKey<T> FormatParamCacheKey(c10::OptionalArrayRef<T> value);

template <typename T>
FormattedKey<T> FormatParamCacheKey(const std::optional<T>& value) {
  if (!value.has_value()) {
    return "";
  }

  // Surround the formatted value with <> to distinguish between
  // a nullopt and a value of T that happens to be an empty string.
  std::string str;
  if constexpr (std::is_same_v<FormattedKey<T>, std::string>) {
    str = FormatParamCacheKey(value.value());
  } else if constexpr (std::is_same_v<FormattedKey<T>,
                                      absl::StatusOr<std::string>>) {
    TT_ASSIGN_OR_RETURN(str, FormatParamCacheKey(value.value()));
  } else {
    static_assert(false, "Unsupported return type of FormatParamCacheKey(T).");
  }
  return absl::StrCat("<", str, ">");
}

template <typename T1, typename T2>
std::string FormatParamCacheKey(const std::pair<T1, T2>& value) {
  return absl::StrCat("(", FormatParamCacheKey(value.first), ",",
                      FormatParamCacheKey(value.second), ")");
}

// Formats a range of values into a string to be used as a cache parameter key.
template <typename T, typename Result, typename Iter>
Result FormatParamCacheKeyForRange(const Iter begin, const Iter end) {
  if (begin == end) {
    return "";
  }
  std::string result = "[";
  bool first = true;
  for (Iter it = begin; it != end; ++it) {
    if (first) {
      first = false;
    } else {
      absl::StrAppend(&result, ",");
    }
    if constexpr (std::is_same_v<Result, std::string>) {
      absl::StrAppend(&result, FormatParamCacheKey(*it));
    } else if constexpr (std::is_same_v<Result, absl::StatusOr<std::string>>) {
      TT_ASSIGN_OR_RETURN(std::string str, FormatParamCacheKey(*it));
      absl::StrAppend(&result, str);
    } else {
      static_assert(false, "Unsupported Result type.");
    }
  }
  absl::StrAppend(&result, "]");
  return result;
}

template <typename T>
FormattedKey<T> FormatParamCacheKey(absl::Span<T> value) {
  return FormatParamCacheKeyForRange<T, FormattedKey<T>>(value.begin(),
                                                         value.end());
}

template <typename K, typename V, typename Hash, typename Eq, typename Alloc>
std::string FormatParamCacheKey(
    const std::unordered_map<K, V, Hash, Eq, Alloc>& value) {
  using value_type =
      typename std::unordered_map<K, V, Hash, Eq, Alloc>::value_type;
  return FormatParamCacheKeyForRange<value_type, std::string>(value.begin(),
                                                              value.end());
}

template <typename T>
FormattedKey<T> FormatParamCacheKey(const c10::OptionalArrayRef<T> value) {
  if (!value.has_value()) {
    return "";
  }
  return FormatParamCacheKey(absl::MakeConstSpan(*value));
}
template <typename T, std::size_t N>
FormattedKey<T> FormatParamCacheKey(const std::array<T, N>& value) {
  return FormatParamCacheKey(absl::MakeConstSpan(value));
}
template <typename T, size_t kSize, typename Allocator>
FormattedKey<T> FormatParamCacheKey(
    const absl::InlinedVector<T, kSize, Allocator>& value) {
  return FormatParamCacheKey(absl::MakeConstSpan(value));
}
template <typename T>
FormattedKey<T> FormatParamCacheKey(const std::vector<T>& value) {
  return FormatParamCacheKey(absl::MakeConstSpan(value));
}
template <typename T>
FormattedKey<T> FormatParamCacheKey(at::ArrayRef<T> value) {
  return FormatParamCacheKey(absl::MakeConstSpan(value));
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::AllreduceOptions& value) {
  return FormatParamCacheKey(value.reduceOp);
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::ReduceScatterOptions& value) {
  return FormatParamCacheKey(value.reduceOp);
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::BroadcastOptions& value) {
  return FormatParamCacheKey(value.rootRank);
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::ScatterOptions& value) {
  return FormatParamCacheKey(value.rootRank);
}

[[nodiscard]] inline std::string FormatParamCacheKey(
    const c10d::GatherOptions& value) {
  return FormatParamCacheKey(value.rootRank);
}

}  // namespace internal

// Returns an IgnoredInCacheKey object that wraps the given value.
template <typename T,
          // Only allow this function to be called for types that should be
          // included in the cache key.
          typename = std::enable_if_t<internal::IncludeInCacheKey<T>()>>
constexpr internal::IgnoredInCacheKey<T> IgnoreInCacheKey(
    const T& value, const std::string_view reason) {
  ABSL_CHECK(!reason.empty())  // CRASH_OK
      << "Please provide a non-empty reason for using IgnoreInCacheKey().";
  return internal::IgnoredInCacheKey<T>(value);
}

// When defining an op, any parameters that can change the compilation of the
// op without changing the shape or dtype of the inputs or outputs should be
// added to this map.
// For example, relu() does not require any parameters, but gelu() takes an
// "approximate" parameter; two gelu() calls are different if this value is
// different.
// So relu() should use {}, while gelu() should use {"approximate",
// std::string(approximate)};
//
// We chose std::map as the container type because we need the keys to be
// sorted for deterministic cache keys.
//
// We mark the class as nodiscard so that if a function returns this type, we
// will get a compiler error if the return value is ignored. This is to avoid
// wasting time computing cache keys that are not used.
class [[nodiscard]] OpParamCacheKeys {
 private:
  using Map = std::map<std::string, std::string>;

 public:
  class Builder;

  // Implement the read-only container concept.
  using value_type = Map::value_type;
  using const_iterator = Map::const_iterator;
  using size_type = Map::size_type;
  using difference_type = Map::difference_type;
  using key_type = Map::key_type;
  using mapped_type = Map::mapped_type;
  using const_reference = Map::const_reference;
  using const_pointer = Map::const_pointer;
  using const_reverse_iterator = Map::const_reverse_iterator;

  // Disable default ctor to force callers to decide what entries to include.
  OpParamCacheKeys() = delete;

  // Make OpParamCacheKeys move-only, as copying may be expensive.
  OpParamCacheKeys(const OpParamCacheKeys&) = delete;
  OpParamCacheKeys& operator=(const OpParamCacheKeys&) = delete;
  OpParamCacheKeys(OpParamCacheKeys&&);
  OpParamCacheKeys& operator=(OpParamCacheKeys&&);

  // Shorthand for making an empty OpParamCacheKeys.
  static OpParamCacheKeys Empty() { return OpParamCacheKeys(Map()); }

  // Returns a copy of the OpParamCacheKeys.
  OpParamCacheKeys Clone() const { return OpParamCacheKeys(GetMapOrDie()); }

  // Sets a parameter in the OpParamCacheKeys.
  //
  // If the formatting of the value fails, returns the error.
  // Crashes if the parameter is already in the OpParamCacheKeys as it indicates
  // a programmer error.
  template <typename T>
  absl::Status SetParam(std::string_view name, const T& value) {
    static_assert(internal::IncludeInCacheKey<T>(),
                  "This argument should not be included in the cache key "
                  "explicitly.");

    std::string name_str(name);
    const auto it = GetMapOrDie().find(name_str);
    ABSL_CHECK(it == name_to_value_.end())  // CRASH_OK
        << "Duplicate parameter name '" << name
        << "' when computing param cache keys. This is a TorchTPU bug.";

    // Rely on ADL to find the appropriate FormatParamCacheKey() overload for
    // the given value type. Some overloads return StatusOr<std::string>,
    // while others return std::string. We assign the result to a
    // StatusOr<std::string> so that we can handle both the same way.
    using internal::FormatParamCacheKey;
    absl::StatusOr<std::string> str_or = FormatParamCacheKey(value);
    TT_ASSIGN_OR_RETURN(std::string str, std::move(str_or));
    if (str.empty()) {
      // No need to add an empty string to the cache keys.
    } else {
      name_to_value_[std::move(name_str)] = std::move(str);
    }
    return absl::OkStatus();
  }

  [[nodiscard]] const_iterator begin() const { return GetMapOrDie().cbegin(); }
  [[nodiscard]] const_iterator end() const { return GetMapOrDie().cend(); }

  [[nodiscard]] size_type size() const { return GetMapOrDie().size(); }
  [[nodiscard]] bool empty() const { return GetMapOrDie().empty(); }

 private:
  // Makes an OpParamCacheKeys with the given name-to-string-key map.
  explicit OpParamCacheKeys(Map name_to_value)
      : name_to_value_(std::move(name_to_value)) {}

  // Returns the name-to-string-key map. Crashes if the object is in an invalid
  // state.
  [[nodiscard]] const Map& GetMapOrDie() const;

  // Whether the object is in a valid state. An object is initially valid,
  // and becomes invalid if it's moved from. This prevents accidental use
  // after a move.
  bool valid_ = true;
  Map name_to_value_;
};

using OpParamCacheKeysBuilder = OpParamCacheKeys::Builder;

// Builder class for OpParamCacheKeys. Supports method-chaining syntax:
//   TT_ASSIGN_OR_RETURN(auto params,
//                       *OpParamCacheKeysBuilder().SetParam(...)
//                           .SetParam(...));
class OpParamCacheKeys::Builder {
 public:
  // Creates an empty builder that doesn't contain any parameters.
  Builder() : param_keys_(OpParamCacheKeys::Empty()) {}

  // This class is move-only.
  Builder(Builder&& other) = default;
  Builder& operator=(Builder&& other) = default;
  Builder(const Builder&) = delete;
  Builder& operator=(const Builder&) = delete;

  // Creates a builder that contains the given parameters.
  explicit Builder(OpParamCacheKeys op_param_cache_keys)
      : param_keys_(std::move(op_param_cache_keys)) {}

  // If the builder is in an error state, does nothing. Otherwise, formats
  // the value as a string and:
  // 1. If the string is not empty, adds the name-string-value pair to the
  //    builder.
  // 2. If the string is empty, remove the parameter from the builder.
  // If formatting fails, the builder enters an error state.
  //
  // The formatting is done by calling FormatParamCacheKey() on the value.
  // To make this API extensible, we rely on Argument Dependent Lookup (ADL)
  // to find the appropriate FormatParamCacheKey() overload for the given
  // value type. In general, the overload for type T should be defined in
  // the same file where T is defined and in the same namespace. When that's
  // not possible (e.g. T is a c10 type, which we don't own), define the
  // overload in the torch_tpu::internal namespace in this file.
  template <typename T>
  Builder& SetParam(std::string_view name, const T& value) {
    if (first_error_.ok()) {
      first_error_.Update(param_keys_.SetParam(name, value));
    }
    return *this;
  }

  // If the builder is in an error state, returns the first error encountered.
  // Otherwise returns the cache keys accumulated so far.
  //
  // After this method is called, the builder is put into an invalid state and
  // should not be used anymore.
  absl::StatusOr<OpParamCacheKeys> operator*();

  // Returns the current error state of the builder.
  const absl::Status& status() const { return first_error_; }

 private:
  // The first error encountered during the construction of the parameter
  // cache keys, or OK if no errors were encountered.
  absl::Status first_error_ = absl::OkStatus();
  OpParamCacheKeys param_keys_;
};

namespace internal {

// Returns the name of the next argument in the given substring of a
// comma-separated list of argument names, e.g. "a, b, c".
// It also updates the substring to exclude the returned argument name.
//
// For example, if args_str is "a, b, c", the first call to this function
// will return "a" and update args_str to "b, c". The second call will return
// "b" and update args_str to "c". The third call will return "c" and update
// args_str to "".
[[nodiscard]] std::string_view ParseNextArgName(std::string_view& args_str);

// Whether to enforce at compile time that SetParam() is only called with
// arguments that should be included in the cache key.
enum class EnforceSetParamType {
  kNo,
  kYes,
};

// MakeOpParamCacheKeys() creates an OpParamCacheKeys from the given arguments
// for an op kernel. This function skips all at::Tensor-typed arguments in the
// list, as their dtypes and shapes are already encoded in the cache key
// automatically.
//
// Arguments:
//   args_str: A comma-separated list of argument names, e.g. "a, b, c".
//   args: The arguments. The number of arguments must match the number of
//       argument names in args_str.

// The base case: no arguments.
template <EnforceSetParamType kEnforceSetParamType>
inline void MakeOpParamCacheKeysImpl(OpParamCacheKeys::Builder& builder,
                                     const std::string_view args_str) {}

// The recursive case: at least one argument.
template <EnforceSetParamType kEnforceSetParamType, typename Arg,
          typename... Args>
inline void MakeOpParamCacheKeysImpl(OpParamCacheKeys::Builder& builder,
                                     std::string_view args_str, const Arg& arg,
                                     const Args&... args) {
  std::string_view arg_name = ParseNextArgName(args_str);
  if constexpr (IncludeInCacheKey<Arg>() ||
                kEnforceSetParamType == EnforceSetParamType::kYes) {
    builder.SetParam(arg_name, arg);
  }
  MakeOpParamCacheKeysImpl<kEnforceSetParamType>(builder, args_str, args...);
}

template <EnforceSetParamType kEnforceSetParamType, typename... Args>
absl::StatusOr<OpParamCacheKeys> MakeOpParamCacheKeys(
    const std::string_view args_str, const Args&... args) {
  OpParamCacheKeys::Builder builder;
  MakeOpParamCacheKeysImpl<kEnforceSetParamType>(builder, args_str, args...);
  return *std::move(builder);
}

}  // namespace internal

// Give the shapeless and dimensions keys separate types to avoid accidentally
// using the wrong key.
class ShapelessKey {
 public:
  explicit ShapelessKey(FingerprintType key) : key_(key) {}

  struct Hash {
    [[nodiscard]] inline size_t operator()(const ShapelessKey key) const {
      return key.key_;
    }
  };

  [[nodiscard]] bool operator==(const ShapelessKey rhs) const {
    return key_ == rhs.key_;
  }

  [[nodiscard]] FingerprintType key() const { return key_; }

 private:
  FingerprintType key_;
};

// Formats a shapeless key as a human-readable string.
template <typename Sink>
void AbslStringify(Sink& sink, const ShapelessKey key) {
  absl::Format(&sink, "%016x", key.key());
}

class ShapeDynamismMetadata;

class DimensionsKey {
 public:
  explicit DimensionsKey(absl::Span<const int64_t> dimensions) {
    // We keep the false parameter for backward compatibility.
    // TODO: (b/494661082) - Remove the false parameter.
    key_ = FingerprintCat(dimensions, false);
  }

  // Create a DimensionsKey from the given shape dynamism metadata.
  explicit DimensionsKey(const ShapeDynamismMetadata& shape_dynamism_metadata);

  [[nodiscard]] bool operator==(const DimensionsKey rhs) const {
    return key_ == rhs.key_;
  }

  [[nodiscard]] FingerprintType key() const { return key_; }

 private:
  FingerprintType key_;
};

// A `GraphKey` is used to identify a computation graph by
// combining its structural representation (`ShapelessKey`) and the specific
// tensor dimensions involved (`DimensionsKey`).
class GraphKey {
 public:
  struct Hash {
    [[nodiscard]] inline size_t operator()(GraphKey key) const {
      return FingerprintCat(key.shapeless_key().key(),
                            key.dimensions_key().key());
    }
  };

  GraphKey(ShapelessKey shapeless_key, DimensionsKey dimensions_key)
      : shapeless_key_(shapeless_key), dimensions_key_(dimensions_key) {}

  [[nodiscard]] friend bool operator==(GraphKey lhs, GraphKey rhs) {
    return lhs.shapeless_key_ == rhs.shapeless_key_ &&
           lhs.dimensions_key_ == rhs.dimensions_key_;
  }

  [[nodiscard]] ShapelessKey shapeless_key() const { return shapeless_key_; }

  [[nodiscard]] DimensionsKey dimensions_key() const { return dimensions_key_; }

 private:
  ShapelessKey shapeless_key_;
  DimensionsKey dimensions_key_;
};

// Formats a graph key as a human-readable string.
template <typename Sink>
void AbslStringify(Sink& sink, const GraphKey key) {
  absl::Format(&sink, "%016x_%016x", key.shapeless_key().key(),
               key.dimensions_key().key());
}

// A `CompilationCacheKey` is used to identify a compilation in the compilation
// cache. Depending on the hash/eq functions used, it may either uniquely
// identify a cached executable, or non-uniquely identify a set of executables
// that share some property.
class CompilationCacheKey {
 public:
  struct Hash {
    [[nodiscard]] inline size_t operator()(CompilationCacheKey key) const {
      const auto& graph_key = key.graph_key();
      const auto& compile_options_key = key.compile_options_key();

      return FingerprintCat(graph_key.shapeless_key().key(),
                            graph_key.dimensions_key().key(),
                            compile_options_key.key());
    }
  };

  // Creates a CompilationCacheKey from the given components.
  CompilationCacheKey(GraphKey graph_key, CompileOptionsKey compile_options_key)
      : graph_key_(graph_key), compile_options_key_(compile_options_key) {}

  // Compares two CompilationCacheKeys.
  [[nodiscard]] friend bool operator==(CompilationCacheKey lhs,
                                       CompilationCacheKey rhs) {
    return lhs.graph_key_ == rhs.graph_key_ &&
           lhs.compile_options_key_ == rhs.compile_options_key_;
  }

  // Returns a compact string representation of the key, suitable for use in a
  // file name.
  //
  // The format is "<shapeless_key>_<dimensions_key>_<compile_options_key>"
  // where each key part is formatted as a hexadecimal string of 16 digits.
  [[nodiscard]] std::string CompactFormat() const;

  [[nodiscard]] GraphKey graph_key() const { return graph_key_; }

  [[nodiscard]] CompileOptionsKey compile_options_key() const {
    return compile_options_key_;
  }

 private:
  // The fingerprint of all graph properties.
  GraphKey graph_key_;
  // The fingerprint of the compile options.
  CompileOptionsKey compile_options_key_;
};

static_assert(
    sizeof(CompilationCacheKey) <= 3 * sizeof(void*),
    "Per C++ Core Guidelines, structs whose size is at most 3 pointers are "
    "more efficient to pass by value than by reference. We pass "
    "CompilationCacheKey by value today. We need to revisit the decision if "
    "its size grows.");

// Formats a compilation cache key as a human-readable string.
std::ostream& operator<<(std::ostream& os, CompilationCacheKey key);
template <typename Sink>
void AbslStringify(Sink& sink, const CompilationCacheKey key) {
  absl::Format(&sink, "%s", absl::FormatStreamed(key));
}

// Bounds of a dimension. If the lower and upper bounds are the same, the
// dimension is static and not dynamic. If *all* dimensions are static, the
// graph is static.
//
// A valid DimensionBounds has 0 <= lower <= upper.
struct DimensionBounds {
  int64_t lower = -1;
  int64_t upper = -1;
};

Dimensions GetUpperBounds(absl::Span<const DimensionBounds> bounds);

// Returns a cache key for the pad module for the given input dynamic shapes.
GraphKey PadModuleCacheKey(absl::Span<const Shape> dynamic_shapes);

// Returns a cache key for the slice module with the given target and padded
// shapes.
GraphKey SliceModuleCacheKey(absl::Span<const Dimensions> target_shapes,
                             absl::Span<const Dimensions> padded_shapes,
                             absl::Span<const mlir::ElementType> element_types);

// The metadata necessary to check if a static graph is compatible with a
// BoundedDynamicCacheEntry.
class ShapeDynamismMetadata {
 public:
  // Create ShapeDynamismMetadata using the given input and output shapes their
  // dynamic dimension annotations.
  explicit ShapeDynamismMetadata(absl::Span<const Shape> input_shapes,
                                 absl::Span<const Shape> output_shapes);

  // Check if the static part of the given shapes is compatible with the input
  // shape dynamism bounds. This ignores dynamic annotations.
  bool IsStaticShapeCompatible(absl::Span<const Shape> shapes) const;

  [[nodiscard]] const std::vector<DimensionBounds>& input_dimension_bounds()
      const {
    return input_dimension_bounds_;
  }

  [[nodiscard]] const std::vector<DimensionBounds>& output_dimension_bounds()
      const {
    return output_dimension_bounds_;
  }

  // Returns a cache key for the slice module with the given output shapes.
  // This ignores dynamic annotations.
  GraphKey GetSliceModuleCacheKey(absl::Span<const Shape> shapes) const;

 private:
  // The lower and upper bounds of each dimension in the graph's inputs.
  std::vector<DimensionBounds> input_dimension_bounds_;
  // The lower and upper bounds of each dimension in the graphs's outputs.
  std::vector<DimensionBounds> output_dimension_bounds_;
};
// A GraphSignature holds all information necessary to uniquely identify and
// describe a graph of DeferredOps.
// This is *not* intended to be a long-lived object; it is only intended to be
// computed ephemerally to produce the fingerprint for a CompilationCacheKey.
//
// As an example, if we had this graph:
// ```python
//   a = torch.ones(2, 3, dtype=torch.float32)  # materialized
//   b = torch.ones(3, 4, dtype=torch.float32)  # materialized
//   c = a.mm(b)  # output, shape will be 2, 4 and dtype will be float32
// ```
// Then the theoretical tensors list would be [a, b, c] and the deferred ops
// list would be [mm], and so the GraphSignature would be:
// ```
// {
//   graph_output_indices: [2]  # tensors are [a, b, c], output index 2 is c
//   # a is [0, 2), b is [2, 4), c is [4, 6) sliced from tensor_dimensions
//   tensor_dimensions_starts: [0, 2, 4, 6]
//   tensor_dimensions: [2, 3, 3, 4, 2, 4]  # (2, 3) x (3, 4) = (2, 4)
//   tensor_element_types: [F32, F32, F32]  # types of a, b, and c
//   # mm's inputs are [0, 2)  in op_inputs_indices
//   op_inputs_starts: [0, 2]
//   op_inputs_indices: [0, 1]  # mm's inputs are tensors [a, b]
//   op_names: ["mm"]
//   op_param_cache_keys_starts: [0, 0]  # mm has params [0, 0), an empty span
//   op_param_cache_keys: []  # graph has no params
//   op_outputs_indices: [2]  # tensor c is the output of mm
// }
// ```
class GraphSignature {
 public:
  GraphSignature() = default;

  // Adds an input tensor to the graph returning its topological index. Asserts
  // that no ops have been added to the graph.
  int AddInput(absl::Span<const int64_t> dimensions, mlir::ElementType dtype);

  class OpSignatureBuilder {
   public:
    void AddInput(int topological_index);
    int AddOutput(absl::Span<const int64_t> dimensions,
                  mlir::ElementType dtype);

   private:
    friend class GraphSignature;
    explicit OpSignatureBuilder(GraphSignature* graph) : graph_(graph) {}

    GraphSignature* graph_;
  };

  // Adds an operation to the graph using a lambda to stream inputs and outputs.
  // donated_inputs are the indices within the streamed inputs of the tensors
  // that are aliased. Returns the topological index of the first output.
  int AddOp(OpName op_name, const OpParamCacheKeys& op_param_cache_keys,
            absl::Span<const int64_t> donated_inputs,
            absl::FunctionRef<void(OpSignatureBuilder&)> builder);

  // Specifies which tensors are graph outputs.
  void AddGraphOutput(int index);

  // Computes the key for this graph. Note that this computes the final
  // key which involves sorting some properties, so it shouldn't be called
  // before the graph is fully constructed.
  [[nodiscard]] GraphKey GetKey() const;

  int num_inputs() const { return num_inputs_; }
  int num_deferred_ops() const { return op_names_.size(); }

 private:
  int next_tensor_index_ = 0;
  int num_inputs_ = 0;
  bool has_ops_ = false;

  // Adds a tensor to the graph, returning its topological index.
  int AddTensor(absl::Span<const int64_t> dimensions, mlir::ElementType dtype);

  absl::InlinedVector<int, 8> graph_output_indices_;

  // Two graphs are equal only if they donate their root arguments in the same
  // way.
  absl::InlinedVector<int, 8> donated_indices_;

  // Two graphs are equal only if they have the same number of tensors,
  // and all tensors have the same dimensions and element types.
  absl::InlinedVector<int, 8> tensor_dimensions_starts_{0};
  std::vector<int64_t> tensor_dimensions_;  // INT_VEC_OK=many tensors' dims
  absl::InlinedVector<mlir::ElementType, 8> tensor_element_types_;

  // Two graphs are equal only if the edges in the graph are the same, which
  // we track by input indices into each DeferredOp.
  absl::InlinedVector<int, 8> op_inputs_starts_{0};
  std::vector<int> op_inputs_indices_;

  // Two graphs are equal only if all DeferredOps have matching OpNames.
  absl::InlinedVector<OpName, 8> op_names_;

  // Two graphs are equal only if all DeferredOps have the same
  // OpParamCacheKeys.
  absl::InlinedVector<int, 8> op_param_cache_keys_starts_{0};

  // The key and value of each op param cache key, sorted by key.
  std::vector<std::pair<std::string, std::string>> op_param_cache_keys_;

  // Two graphs are equal only if all DeferredOps have the same number of
  // output for each node.
  absl::InlinedVector<int, 8> op_outputs_indices_{0};

  // Keep track of which inputs are donated to avoid duplicates.
  absl::flat_hash_set<int> donated_indices_set_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_CACHE_KEY_H_
