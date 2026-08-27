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

#ifndef TORCH_TPU_OPS_MACROS_KERNEL_H_
#define TORCH_TPU_OPS_MACROS_KERNEL_H_

#include <cstddef>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/macro_utils.h"
#include "torch_tpu/csrc/common/op_name_stack.h"  // IWYU pragma: keep for ScopedOpName
#include "torch_tpu/csrc/ops/macros/logging.h"

// TT_KERNEL(op_name, param_keys, (arg1, arg2, ...), statements) wraps the
// implementation of an op kernel and takes care of logging, capturing the
// Python context, computing parameter cache keys, and attributing errors
// to the correct root op.
//
// Arguments:
//   op_name: The name of the op. E.g. OpName::kAddOut.
//   param_keys: Name of a new variable that will hold the op parameter cache
//       keys. TT_KERNEL() computes the cache keys by going through all
//       non-Tensor-typed parameters in (arg1, arg2, ...) and adding their
//       cache keys. The name `_` is special and means that computing parameter
//       cache keys is skipped (hence the variable `_` will hold an empty
//       OpParamCacheKeys object). This is useful for avoiding the cost of
//       computing the parameter cache keys when they are not needed or when
//       we want to compute them ourselves.
//   (arg1, arg2, ...): The op kernel function's exact parameter list. Always
//       pass *all* parameters of the kernel function exactly as they are,
//       including both tensors and non-tensors. Do not omit any parameters
//       or pass extra parameters.
//   statements: The actual logic of the op kernel. It can reference
//       `param_keys` to access the op parameter cache keys.
//
// Requirements:
//    - OpParamCacheKeys::Builder knows how to compute the cache keys for
//      all argument types. If you get a compiler error that
//      EncodeParamCacheKey() cannot be resolved, add an overload of
//      EncodeParamCacheKey() for that type.
//    - All argument types are printable via ToString().
//
// See go/torch-tpu-op-easy for more details.
//
// Implementation notes:
//   - The last argument ... is for the statements. We use a variadic macro
//     parameter as the statements may contain unprotected commas.
//   - When compiled in debug mode, TT_KERNEL() checks that all PromotedScalar-
//     typed arguments have their GetTensor() method called when the kernel
//     finishes successfully. This prevents bugs where the kernel uses the
//     original Scalar instead of the Tensor promoted from it. We don't do
//     this check if the kernel throws an error, as it's possible for the
//     error to happen before the kernel has a chance to call GetTensor() on
//     some or all of the PromotedScalar arguments.

#define TT_KERNEL(op_name, param_keys, args, ...)                             \
  do {                                                                        \
    ::torch_tpu::internal::ScopedOpName _scoped_op_name(op_name);             \
    TT_CHECK_AND_LOG_KERNEL_ARGS_(op_name, TT_REMOVE_PARENS_(args));          \
    TT_IF_DEBUG(                                                              \
        ::torch_tpu::internal::PromotedTensorStates _promoted_scalars;        \
        TT_COLLECT_PROMOTED_SCALAR_POINTERS_(_promoted_scalars,               \
                                             TT_REMOVE_PARENS_(args));        \
        bool _kernel_threw = false;                                           \
        absl::Cleanup _cleanup = [&_kernel_threw, &_promoted_scalars] {       \
          if (!_kernel_threw) {                                               \
            ::torch_tpu::internal::CheckTensorsUsed(_promoted_scalars);       \
          }                                                                   \
        };)                                                                   \
    ::torch_tpu::ScopedPythonContextCapturer _capturer(op_name);              \
    if constexpr (std::string_view(#param_keys) == "_") {                     \
      ::torch_tpu::internal::ConditionalStaticAssert<                         \
          std::string_view(#param_keys) == "_",                               \
          decltype(::torch_tpu::internal::NoArgAffectsCacheKey args)>();      \
      /* Use InvalidOpParamCacheKeys as the type of _, to prevent using it in \
       * ...                                                                  \
       */                                                                     \
      ::torch_tpu::internal::InvalidOpParamCacheKeys param_keys;              \
      try {                                                                   \
        __VA_ARGS__;                                                          \
      } catch (const ::torch_tpu::TtError& e) {                               \
        TT_IF_DEBUG(_kernel_threw = true;)                                    \
        ::torch_tpu::TranslateToC10ErrorAndThrow(e);                          \
      } catch (...) {                                                         \
        TT_IF_DEBUG(_kernel_threw = true;)                                    \
        throw; /* throw needed for implementing TT_KERNEL. */                 \
      }                                                                       \
    } else {                                                                  \
      TT_ASSIGN_OR_THROW(/* ERROR_COV_INFEASIBLE=in macro definition. */      \
                         ::torch_tpu::OpParamCacheKeys param_keys,            \
                         TT_MAKE_OP_PARAM_CACHE_KEYS_NO_ENFORCE_ args);       \
      try {                                                                   \
        __VA_ARGS__;                                                          \
      } catch (const ::torch_tpu::TtError& e) {                               \
        TT_IF_DEBUG(_kernel_threw = true;)                                    \
        ::torch_tpu::TranslateToC10ErrorAndThrow(e);                          \
      } catch (...) {                                                         \
        TT_IF_DEBUG(_kernel_threw = true;)                                    \
        throw; /* throw needed for implementing TT_KERNEL. */                 \
      }                                                                       \
    }                                                                         \
  } while (false)

// Collects pointers to all PromotedScalar-typed arguments into the given
// vector.
#define TT_COLLECT_PROMOTED_SCALAR_POINTERS_(scalars, ...)                    \
  ::torch_tpu::internal::CollectPromotedScalarPointers(scalars __VA_OPT__(, ) \
                                                           __VA_ARGS__)

// Invokes MakeOpParamCacheKeys() with the given arguments, with the
// stringified argument list as the first argument. E.g.
//   TT_MAKE_OP_PARAM_CACHE_KEYS(a, b, c)
// expands to
//   MakeOpParamCacheKeys("a, b, c", a, b, c)
// while
//   TT_MAKE_OP_PARAM_CACHE_KEYS()
// expands to
//   MakeOpParamCacheKeys("")
//
// Implementation note: the __VA_OPT__(,) expands to nothing if the argument
// list is empty, and expands to a comma otherwise.
#define TT_MAKE_OP_PARAM_CACHE_KEYS(...)                                       \
  (                                                                            \
      [] {                                                                     \
        static_assert(::torch_tpu::internal::ArgsAreIdentifiers(#__VA_ARGS__), \
                      "The op parameter list passed to "                       \
                      "TT_MAKE_OP_PARAM_CACHE_KEYS() must contain "            \
                      "only identifier names.");                               \
      }(),                                                                     \
      ::torch_tpu::internal::MakeOpParamCacheKeys<                             \
          ::torch_tpu::internal::EnforceSetParamType::kYes>(                   \
          #__VA_ARGS__ __VA_OPT__(, ) __VA_ARGS__))

// Like TT_MAKE_OP_PARAM_CACHE_KEYS, but does not enforce that arguments should
// be included in the cache key.
#define TT_MAKE_OP_PARAM_CACHE_KEYS_NO_ENFORCE_(...)    \
  ::torch_tpu::internal::MakeOpParamCacheKeys<          \
      ::torch_tpu::internal::EnforceSetParamType::kNo>( \
      #__VA_ARGS__ __VA_OPT__(, ) __VA_ARGS__)

namespace torch_tpu {
namespace internal {

// Variadic function CollectPromotedScalarPointers() collects pointers to all
// PromotedScalar-typed arguments into the given vector.

// Overloads for AppendPromotedScalarPointers:

// Case: PromotedScalar
inline void AppendPromotedScalarPointers(PromotedTensorStates& promoted_scalars,
                                         const PromotedScalar& arg) {
  promoted_scalars.push_back(arg.state_);
}

// Case: std::optional<PromotedScalar>
inline void AppendPromotedScalarPointers(
    PromotedTensorStates& promoted_scalars,
    const std::optional<PromotedScalar>& arg) {
  if (arg.has_value()) {
    AppendPromotedScalarPointers(promoted_scalars, *arg);
  }
}

// Case: std::vector<PromotedScalar>
void AppendPromotedScalarPointers(PromotedTensorStates& promoted_scalars,
                                  const std::vector<PromotedScalar>& arg);

// Case: IgnoredInCacheKey<T>
template <typename T>
inline void AppendPromotedScalarPointers(PromotedTensorStates& promoted_scalars,
                                         const IgnoredInCacheKey<T>& arg) {
  AppendPromotedScalarPointers(promoted_scalars, arg.value());
}

// Case: anything else (no-op)
template <typename T>
inline void AppendPromotedScalarPointers(PromotedTensorStates& promoted_scalars,
                                         const T& arg) {}

// Base case: 0 arguments to collect.
inline void CollectPromotedScalarPointers(
    PromotedTensorStates& promoted_scalars) {}

// Recursive case: collect pointers from the next argument and then call the
// function recursively with the remaining arguments.
template <typename T, typename... Args>
inline void CollectPromotedScalarPointers(
    PromotedTensorStates& promoted_scalars, const T& arg,
    const Args&... rest_args) {
  AppendPromotedScalarPointers(promoted_scalars, arg);
  CollectPromotedScalarPointers(promoted_scalars, rest_args...);
}

// Crashes if any of the given PromotedScalars was not used in the kernel.
void CheckTensorsUsed(const PromotedTensorStates& promoted_scalars);

// A type that can't be used to do anything useful.
//
// We use this type for the `_` param_keys variable in TT_KERNEL() to prevent
// the user from accidentally using it in the body of the kernel, which is
// a programmer error.
class InvalidOpParamCacheKeys {
 public:
  InvalidOpParamCacheKeys() = default;

  // Allow implicit conversion to OpParamCacheKeys to compile.
  // These functions have no implementation as they should only be used in
  // inactive code paths. E.g.
  //   TT_KERNEL(OpName::kAdd, param_keys, (size), {
  //     ... std::move(param_keys) ...
  //   });
  // expands to
  //   ...
  //   if constexpr (std::string_view("param_keys") == "_") {
  //     InvalidOpParamCacheKeys param_keys;
  //     ... std::move(param_keys) ...  // block A
  //   } else {
  //     TT_ASSIGN_OR_THROW(OpParamCacheKeys param_keys, ...);
  //     ... std::move(param_keys) ...  // block B
  //   }
  //   ...
  // Since the constexpr condition is false, block A is inactive, and thus
  // the code for InvalidOpParamCacheKeys doesn't need to be linked.
  template <typename T>
  absl::Status SetParam(std::string_view, const T&);
  operator OpParamCacheKeys&() &;    // NOLINT
  operator OpParamCacheKeys&&() &&;  // NOLINT
  OpParamCacheKeys Clone() const;
};

// This helper returns std::true_type if any of the given arguments should
// normally be included in the parameter cache key computation.
template <typename... Ts>
std::bool_constant<(... || IncludeInCacheKey<Ts>())>
ShouldIncludeSomeInCacheKey(const Ts&...);

// Checks if an argument should be ignored in the cache key.
// If not, it triggers a static_assert with the index of the argument.
template <size_t kParamIndex, typename ParamType>
struct ArgDoesNotAffectCacheKey {
  static constexpr bool value = !IncludeInCacheKey<ParamType>();
  static_assert(
      value,
      "You are using _ for the name of the parameter cache keys "
      "variable in TT_KERNEL(), which causes the parameter cache key "
      "computation to be skipped. However, the TT_KERNEL() argument "
      "list contains one or more arguments that should normally be "
      "included in parameter cache keys. If you indeed want to exclude "
      "them from cache key computation, wrap them in IgnoreInCacheKey() "
      "with a string explaining why. Otherwise, rename _ to param_keys and use "
      "it in DispatchOp().");
};

// Checks if all arguments in a pack should be ignored in the cache key.
//
// How it works:
// - It takes a pack of types `Ts...`.
// - It uses a C++20 lambda with an explicit template parameter pack
//   `<size_t... Is>` to deduce the indices from `std::index_sequence`.
// - The lambda creates an index sequence of the same size as `Ts...`.
// - It then uses a fold expression
//   `(ArgDoesNotAffectCacheKey<Is, Ts>::value && ...)`
//   to instantiate `ArgDoesNotAffectCacheKey` for each argument,
//   passing its index and type.
// - This triggers the `static_assert` inside `ArgDoesNotAffectCacheKey`
//   for any argument that should not be ignored.
template <typename... Ts>
struct NoArgAffectsCacheKey {
  explicit NoArgAffectsCacheKey(const Ts&...) {}

  static constexpr bool value = []<size_t... Is>(std::index_sequence<Is...>) {
    return (ArgDoesNotAffectCacheKey<Is, Ts>::value && ...);
  }(std::make_index_sequence<sizeof...(Ts)>{});
};

// Deduction guide for NoArgAffectsCacheKey.
//
// Why it's necessary:
// In TT_KERNEL, we use CTAD (Class Template Argument Deduction) to deduce the
// template arguments of NoArgAffectsCacheKey from the values passed to it:
//   decltype(::torch_tpu::internal::NoArgAffectsCacheKey args)
//
// While C++20 can often deduce this automatically from the constructor,
// clang warns with -Wctad-maybe-unsupported that the struct may not intend
// to support CTAD unless an explicit deduction guide is provided. This guide
// suppresses that warning by explicitly stating that CTAD is intended.
//
// How it works:
// It tells the compiler that when it sees `NoArgAffectsCacheKey` constructed
// with arguments of types `Ts...`, it should deduce the template instance as
// `NoArgAffectsCacheKey<Ts...>`.
template <typename... Ts>
NoArgAffectsCacheKey(const Ts&...) -> NoArgAffectsCacheKey<Ts...>;

// ConditionalStaticAssert() is a helper to trigger a static_assert inside a
// template only when a condition is true.
//
// This is useful when we want to perform a compile-time check in a non-template
// context (e.g. inside a non-template function) based on a macro parameter,
// but we want to avoid triggering template instantiation errors when the check
// should be skipped.
//
// How it works:
// - It takes a boolean kCondition and a type EvalStaticAssert.
// - EvalStaticAssert is expected to be a template instantiation that has a
//   `value` member, and evaluating that `value` triggers a static_assert if it
//   fails.
// - If kCondition is true, it accesses `EvalStaticAssert::value`, forcing the
//   compiler to instantiate EvalStaticAssert and thus trigger the static_assert
//   inside EvalStaticAssert.
// - If kCondition is false, it does nothing, and EvalStaticAssert is not
//   instantiated in a way that triggers its internal static_asserts.
template <bool kCondition, typename EvalStaticAssert>
constexpr void ConditionalStaticAssert() {
  if constexpr (kCondition) {
    static_cast<void>(  // VOID_CAST_OK=template meta programming.
        EvalStaticAssert::value);
  }
}

}  // namespace internal
}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_MACROS_KERNEL_H_
