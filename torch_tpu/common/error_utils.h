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

#ifndef TORCH_TPU_COMMON_ERROR_UTILS_H_
#define TORCH_TPU_COMMON_ERROR_UTILS_H_

// Utilities for reporting errors in C++ code.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/Device.h"
#include "c10/util/Optional.h"
#include "c10/util/StringUtil.h"
#include "torch_tpu/common/macro_utils.h"  // IWYU pragma: keep
#include "torch_tpu/common/status_builder.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/python_context.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

// Note that macro_utils.h is only used for the OSS build. Hence we need
// the IWYU pragma to prevent tools from removing it.

namespace torch_tpu {

namespace error {

// Aliases for absl::StatusCode enums.
// go/keep-sorted start
inline constexpr absl::StatusCode kAborted = absl::StatusCode::kAborted;
inline constexpr absl::StatusCode kAlreadyExists =
    absl::StatusCode::kAlreadyExists;
inline constexpr absl::StatusCode kCancelled = absl::StatusCode::kCancelled;
inline constexpr absl::StatusCode kDataLoss = absl::StatusCode::kDataLoss;
inline constexpr absl::StatusCode kDeadlineExceeded =
    absl::StatusCode::kDeadlineExceeded;
inline constexpr absl::StatusCode kFailedPrecondition =
    absl::StatusCode::kFailedPrecondition;
inline constexpr absl::StatusCode kIndexError =
    absl::StatusCode::kOutOfRange;  // STATUS_CODE_OK
inline constexpr absl::StatusCode kInternal = absl::StatusCode::kInternal;
inline constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;
inline constexpr absl::StatusCode kNotFound = absl::StatusCode::kNotFound;
inline constexpr absl::StatusCode kOk = absl::StatusCode::kOk;
inline constexpr absl::StatusCode kOutOfRange = absl::StatusCode::kOutOfRange;
inline constexpr absl::StatusCode kPermissionDenied =
    absl::StatusCode::kPermissionDenied;
inline constexpr absl::StatusCode kResourceExhausted =
    absl::StatusCode::kResourceExhausted;
inline constexpr absl::StatusCode kUnauthenticated =
    absl::StatusCode::kUnauthenticated;
inline constexpr absl::StatusCode kUnavailable = absl::StatusCode::kUnavailable;
inline constexpr absl::StatusCode kUnimplemented =
    absl::StatusCode::kUnimplemented;
inline constexpr absl::StatusCode kUnknown = absl::StatusCode::kUnknown;
// go/keep-sorted end

}  // namespace error

// Returns true if debug checks are enabled via the
// TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS environment variable. This function is
// memorized, so calling it is cheap and will always return the same value even
// if the environment variable is changed.
[[nodiscard]] bool GetEnableDebugChecks();

// Returns the product of two numbers, or returns an error if the product
// overflows.
absl::StatusOr<int64_t> SafeMultiply(int64_t x, int64_t y);

// Returns the number of elements in a tensor with the given dimensions.
// Returns an error if the result overflows as int64 or if a dimension is
// negative.
absl::StatusOr<int64_t> NumElements(absl::Span<const int64_t> dims);
absl::StatusOr<int64_t> NumElements(const at::Tensor& tensor);

// Safely wraps a dimension index.
//
// Parameters:
//   dim: the index of the dimension to wrap.
//   dim_bound: the number of dimensions.
//
// Fails if `dim` is not in the range `[-dim_bound, dim_bound - 1]` where
// dim_bound is non-negative.
absl::StatusOr<int64_t> SafeWrapDim(int64_t dim, int64_t dim_bound);

// Checks that `device_opt` is TPU (PrivateUse1), or crashes if it is not.
//
// `op_name` is the name of the operation that is being performed. This is used
// for error reporting purposes.
//
// If the check fails, it's a bug in the torch_tpu implementation, so we crash
// with a C++ stack trace instead of returning an error.
void CheckDeviceIsTpu(c10::optional<at::Device> device_opt,
                      std::string_view op_name);

// Returns true if the status is an XLA OOM error. This is a heuristic that
// checks the status code and message. It is not guaranteed to be 100% accurate.
// TODO(wan): use a more robust mechanism to detect XLA OOM errors.
[[nodiscard]] bool IsXlaOomError(const absl::Status& status);

// Formats the given number of items. E.g.
//   FormatCount(1, "tensor", "tensors") -> "1 tensor"
//   FormatCount(2, "tensor", "tensors") -> "2 tensors"
[[nodiscard]] inline std::string FormatCount(const int64_t count,
                                             const std::string_view singular,
                                             const std::string_view plural) {
  return absl::StrCat(count, " ", count == 1 ? singular : plural);
}

// Retrieves the status from a Status or StatusOr. Useful for contexts where we
// are interested in the Status error but not the StatusOr value.
inline const absl::Status& GetStatus(const absl::Status& status) {
  return status;
}
inline absl::Status&& GetStatus(absl::Status&& status) {
  return std::move(status);
}
template <typename T>
const absl::Status& GetStatus(const absl::StatusOr<T>& status_or) {
  return status_or.status();
}
template <typename T>
absl::Status GetStatus(absl::StatusOr<T>&& status_or) {
  return std::move(status_or).status();
}

// An exception thrown by a TT_* macro. It's a std::runtime_error with two
// additional pieces of information:
// - The original absl::Status that caused the exception.
// - The source location where the exception was thrown.
//
// This allows us to translate the TtError to a c10::Error or one of its
// subclasses, preserving the original source location.
class TtError : public std::runtime_error {
 public:
  TtError(absl::Status status, c10::SourceLocation thrown_from);

  TtError(TtError&& other) = default;
  TtError& operator=(TtError&& other) = default;
  TtError(const TtError&) = default;
  TtError& operator=(const TtError&) = default;

  // Getters.
  const absl::Status& status() const { return status_; }
  [[nodiscard]] const c10::SourceLocation& thrown_from() const {
    return thrown_from_;
  }

 private:
  absl::Status status_;  // Invariant: must not be OK.
  c10::SourceLocation thrown_from_;
};

// Translates a TtError to a c10::Error or one of its subclasses and throws it.
[[noreturn]] void TranslateToC10ErrorAndThrow(const TtError& e);

// =============================================================================
// Macros for reporting errors in torch_tpu C++ code.
// =============================================================================
//
// See go/torch-tpu-error-api for the design of these macros.
//
// When the TORCH_SHOW_CPP_STACKTRACES environment variable is set to 1,
// these macros will capture the C++ source location as an absl::Status is
// created or propagated up the call stack. This error trace is stored in the
// absl::Status's payload.
//
// When reporting an absl::Status error to the user, use
// TT_THROW_IF_ERROR(status) instead of TT_CHECK_THROW(false, ...) <<
// status.message(), as the former will preserve the C++ error trace in the
// status, if any.

// TT_ERROR(error_code) returns a StatusBuilder that can be used to construct
// an error status.
//
// Example: create an error status with a message.
//   absl::Status error = TT_ERROR(error::kInvalidArgument)
//      << "dimension size is negative: " << dim_size;
//
// Example: return an error from the current function.
//   return TT_ERROR(error::kInvalidArgument))
//      << "dimension size is negative: " << dim_size;
//
// This macro also captures the current source location and/or the stack trace
// if the TORCH_SHOW_CPP_STACKTRACES environment variable is set to 1.
//
// Implementation note:
// - We use __func__ instead of __FUNCTION__ because the former is in STL
//   and the latter is non portable.
#define TT_ERROR(error_code)                                              \
  ::torch_tpu::StatusBuilder(::torch_tpu::internal::MaybeAddCppSourceLoc( \
      ::torch_tpu::internal::SetRootOpNamePayload(                        \
          absl::Status(error_code, ""),                                   \
          ::torch_tpu::GetRootOpName(std::nullopt)),                      \
      TT_NORMALIZED_FILE, __LINE__, __func__))

// TT_SOURCE_LOCATION
//
// Returns the current C++ source location.
#define TT_SOURCE_LOCATION \
  ::c10::SourceLocation({__func__, TT_NORMALIZED_FILE, __LINE__})

// TT_RET_CHECK(cond, error_code);
//
// Evaluates `cond` as a bool, does nothing if result is true, or returns a
// Status error with the given error code if the result is false. The error
// message is empty by default, but can be set using the
// << operator. The << expressions are evaluated only if `cond` is false.
//
// This macro is modeled on the RET_CHECK macro inside Google, but with the
// following changes:
// - It allows specifying an error code.
// - It doesn't automatically include the cond expression text in the error
//   message (as that is not suitable for end users).
//
// Example:
//   TT_RET_CHECK(foo.ok(), error::kInvalidArgument)
//      << "foo is not ok: " << foo.status();
//
// Implementation note:
// - The macro cannot be simplified to
//     if (ABSL_PREDICT_FALSE(!(cond))) return TT_ERROR(error_code)
//   because that will lead to incorrect expansion of
//     if (foo)
//       TT_RET_CHECK(bar, ...) << ...;
//     else
//       Blah();
//   (the else will be associated with `if (ABSL_PREDICT_FALSE(!(cond)))`
//   instead of `if (foo)`.)
#define TT_RET_CHECK(cond, error_code) \
  if (ABSL_PREDICT_TRUE((cond)))       \
    /* do nothing */;                  \
  else                                 \
    return TT_ERROR(error_code)

// TT_ASSIGN_OR_RETURN(lhs, rexpr);
// TT_ASSIGN_OR_RETURN(lhs, rexpr, error_expr);
//
// Evaluates an expression `rexpr` that returns an `absl::StatusOr<T>`. On OK,
// moves its value into the variable defined by `lhs`; otherwise returns the
// error from the status. If there is an error, `lhs` is not evaluated; thus any
// side effects that `lhs` may have only occur in the success case.
//
// NOTE:
//   - This macro is modeled on the ASSIGN_OR_RETURN macro inside Google.
//     It has a similar interface, except that it throws instead of returning
//     an error.
//   - If lhs is parenthesized, the parentheses are removed. See examples
//     below for more details.
//   - This macro expands to multiple C++ statements, so it cannot be used
//     inside of a C++ `if` statement without braces:
//       if (cond)
//         TT_ASSIGN_OR_RETURN(lhs, rexpr);  // This is INVALID!
//
// Example: Declaring and initializing a new variable (ValueType can be anything
//          that can be initialized with assignment--including a const
//          reference, although that's discouraged by go/totw/107):
//   TT_ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(arg));
//
// Example: Assigning to an existing variable:
//   ValueType value;
//   TT_ASSIGN_OR_RETURN(value, MaybeGetValue(arg));
//
// Example: Assigning to an expression with side effects:
//   MyProto data;
//   TT_ASSIGN_OR_RETURN(*data.mutable_str(), MaybeGetValue(arg));
//   // No field "str" is added on error.
//
// Example: Initializing a map. Because of C++ preprocessor limitations,
// the type used in TT_ASSIGN_OR_RETURN cannot contain commas, so wrap the
// lhs in parentheses:
//   TT_ASSIGN_OR_RETURN((absl::flat_hash_map<Foo, Bar> my_map),
//                             GetMap());
// Or use `auto` if the type is obvious enough:
//   TT_ASSIGN_OR_RETURN(auto my_map, GetMap());
//
// Example: Assigning to structured bindings (go/totw/169). The same situation
// with comma as in map, so wrap the statement in parentheses.
//   TT_ASSIGN_OR_RETURN((auto [first, second]), GetPair());
//
// If passed, `error_expr` is evaluated to produce the returned error value. The
// expression may reference any variable visible in scope, as well as a
// `torch_tpu::StatusBuilder` object populated with the error and named by a
// single underscore `_`. The expression uses the builder to modify the status
// and is thrown directly in manner similar to TT_RETURN_IF_ERROR. For example:
//
// Example: Adjusting the error message.
//   TT_ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(op),
//       _ << " while processing op " << op.DebugString());
//
// Implementation note: when supplied with 2 arguments, the macro will delegate
// to TT_ASSIGN_OR_RETURN_2_; when supplied with 3 arguments, it will delegate
// to TT_ASSIGN_OR_RETURN_3_.
#define TT_ASSIGN_OR_RETURN(...)                                     \
  TT_GET_VARIADIC_(                                                  \
      (__VA_ARGS__, TT_ASSIGN_OR_RETURN_3_, TT_ASSIGN_OR_RETURN_2_)) \
  (__VA_ARGS__)

// Where the throwing macros can be used
// =====================================
//
// The TT_ASSIGN_OR_THROW, TT_THROW_IF_ERROR, and TT_CHECK_THROW macros may
// throw exceptions. These macros should only be used in callbacks registered
// with the PyTorch runtime. It should not be used in any other functions. In
// particular, the compilation will fail if these macros are used in a function
// that returns a Status or StatusOr - this is intended (throwing in a helper
// function may lead to undefined behavior if a caller isn't compiled with
// exceptions enabled).
//
// Caveat: Due to limitations of the C++ language, these macros cannot be used
// inside a void-returning lambda. Please convert such a lambda to a named
// function before using these macros in its body. This is rarely a
// problem as our best practice is to throw only in the top-level Python-bound
// functions.

// TT_ASSIGN_OR_THROW(lhs, rexpr);
// TT_ASSIGN_OR_THROW(lhs, rexpr, error_expr);
//
// Evaluates an expression `rexpr` that returns an `absl::StatusOr<T>`. On OK,
// moves its value into the variable defined by `lhs`; otherwise throws an
// exception with the error message from the status. If there is an error, `lhs`
// is not evaluated; thus any side effects that `lhs` may have only occur in the
// success case.
//
// NOTE:
//   - This macro is modeled on the ASSIGN_OR_RETURN macro inside Google.
//     It has a similar interface, except that it throws instead of returning
//     an error.
//   - If lhs is parenthesized, the parentheses are removed. See examples
//     below for more details.
//
// Example: Declaring and initializing a new variable (ValueType can be anything
//          that can be initialized with assignment--including a const
//          reference, although that's discouraged by go/totw/107):
//   TT_ASSIGN_OR_THROW(ValueType value, MaybeGetValue(arg));
//
// Example: Assigning to an existing variable:
//   ValueType value;
//   TT_ASSIGN_OR_THROW(value, MaybeGetValue(arg));
//
// Example: Assigning to an expression with side effects:
//   MyProto data;
//   TT_ASSIGN_OR_THROW(*data.mutable_str(), MaybeGetValue(arg));
//   // No field "str" is added on error.
//
// Example: Initializing a map. Because of C++ preprocessor limitations,
// the type used in TT_ASSIGN_OR_THROW cannot contain commas, so wrap the
// lhs in parentheses:
//   TT_ASSIGN_OR_THROW((absl::flat_hash_map<Foo, Bar> my_map),
//                             GetMap());
// Or use `auto` if the type is obvious enough:
//   TT_ASSIGN_OR_THROW(auto my_map, GetMap());
//
// Example: Assigning to structured bindings (go/totw/169). The same situation
// with comma as in map, so wrap the statement in parentheses.
//   TT_ASSIGN_OR_THROW((auto [first, second]), GetPair());
//
// If passed, `error_expr` is evaluated to produce the thrown error value. The
// expression may reference any variable visible in scope, as well as a
// `torch_tpu::StatusBuilder` object populated with the error and named by a
// single underscore `_`. The expression uses the builder to modify the status
// and is thrown directly in manner similar to TT_THROW_IF_ERROR. For example:
//
// Example: Adjusting the error message.
//   TT_ASSIGN_OR_THROW(ValueType value, MaybeGetValue(op),
//       _ << " while processing op " << op.DebugString());
//
// Caveat: this macro expands to multiple C++ statements, so it cannot be used
// inside of a C++ `if` statement without braces:
//   if (cond)
//     TT_ASSIGN_OR_THROW(lhs, rexpr);  // This is INVALID!
//
// Implementation note: when supplied with 2 arguments, the macro will delegate
// to TT_ASSIGN_OR_THROW_2_; when supplied with 3 arguments, it will delegate
// to TT_ASSIGN_OR_THROW_3_.
#define TT_ASSIGN_OR_THROW(...)                                    \
  TT_STATIC_ASSERT_RETURN_TYPE_NOT_STATUS_();                      \
  TT_GET_VARIADIC_(                                                \
      (__VA_ARGS__, TT_ASSIGN_OR_THROW_3_, TT_ASSIGN_OR_THROW_2_)) \
  (__VA_ARGS__)

// TT_RETURN_IF_ERROR(status);
//
// Evaluates an expression `status` that produces an `absl::Status`. If the
// status is not OK, returns it from the current function.
//
// This macro is modeled on the RETURN_IF_ERROR macro inside Google.
//
// For example:
//   absl::Status MultiStepFunction() {
//     TT_RETURN_IF_ERROR(Function(args...));
//     TT_RETURN_IF_ERROR(foo.Method(args...));
//     return absl::OkStatus();
//   }
//
// The macro ends with an object which allows the returned status
// to be extended with more details. Any chained expressions after the macro
// will not be evaluated unless there is an error.
//
// For example:
//   absl::Status MultiStepFunction() {
//     // Append to the error message.
//     TT_RETURN_IF_ERROR(Function(args...)) << " in MultiStepFunction";
//     return absl::OkStatus();
//   }
//
// To prepend (as opposed to append) to the error message, use SetPrepend():
//   absl::Status MultiStepFunction() {
//     // Prepend to the error message.
//     TT_RETURN_IF_ERROR(Function(args...)).SetPrepend()
//         << " in MultiStepFunction";
//     return absl::OkStatus();
//   }
//
// Implementation note: the macro captures the caller's source location so that
// the error message contains the source location of the call site of
// TT_RETURN_IF_ERROR when the TORCH_SHOW_CPP_STACKTRACES environment variable
// is set to 1.
#define TT_RETURN_IF_ERROR(status)                                          \
  TT_ELSE_BLOCKER_                                                          \
  if (auto tt_error_adapter_ = ::torch_tpu::internal::ReturnIfErrorAdaptor( \
          ::torch_tpu::GetStatus((status)))) {                              \
  } else /* NOLINT */                                                       \
    return tt_error_adapter_.Consume(TT_NORMALIZED_FILE, __LINE__, __func__)

// TT_THROW_IF_ERROR(status);
//
// Evaluates an expression `status` that produces an `absl::Status` or
// `absl::StatusOr<T>`. If the status is not OK, throws an std::runtime_error
// exception with the error message from the status.
//
// This macro is modeled on the RETURN_IF_ERROR macro inside Google, but it
// throws instead of returning an error.
//
// For example:
//   at::Tensor MultiStepFunction() {
//     ...
//     TT_THROW_IF_ERROR(foo.Method(args...));
//     ...
//     return foo;
//   }
//
// Like TT_RETURN_IF_ERROR, the macro ends with an object which allows the
// returned status to be extended with more details. Any chained expressions
// after the macro will not be evaluated unless there is an error.
//
// Implementation note: the macro captures the caller's source location
// and passes it to the c10::Error constructor later, so that the error
// message contains the source location of the call site of TT_THROW_IF_ERROR
// when the TORCH_SHOW_CPP_STACKTRACES environment variable is set to 1.
#define TT_THROW_IF_ERROR(status)                                           \
  TT_ELSE_BLOCKER_                                                          \
  if (auto tt_error_adapter_ = ::torch_tpu::internal::ReturnIfErrorAdaptor( \
          ::torch_tpu::GetStatus((status)))) {                              \
    TT_STATIC_ASSERT_RETURN_TYPE_NOT_STATUS_();                             \
  } else /* NOLINT */                                                       \
    ::torch_tpu::internal::ThrowIfErrorAdapter(                             \
        TT_NORMALIZED_FILE, __LINE__, __func__) = tt_error_adapter_.Consume()

// TT_CHECK_THROW(cond, error_code) << message;
//
// Evaluates an expression `cond` that produces a boolean. If the condition is
// false, throws an exception with the Status error code and message.
//
// This is meant to replace TORCH_CHECK in the torch_tpu codebase. It is
// better as:
//   - It is more consistent with the rest of the TT_* error macros.
//   - It allows selecting the error code.
//
// `cond` is evaluated exactly once. If the result is true, `error_code`
// and the `message` are not evaluated. Otherwise, `error_code` and `message`
// are evaluated exactly once.
#define TT_CHECK_THROW(cond, error_code)                                     \
  TT_ELSE_BLOCKER_                                                           \
  if (ABSL_PREDICT_TRUE(cond)) {                                             \
    TT_STATIC_ASSERT_RETURN_TYPE_NOT_STATUS_();                              \
  } else /* NOLINT */                                                        \
    ::torch_tpu::internal::ThrowIfErrorAdapter(TT_NORMALIZED_FILE, __LINE__, \
                                               __func__) =                   \
        ::torch_tpu::internal::ReturnIfErrorAdaptor(                         \
            ::torch_tpu::internal::SetRootOpNamePayload(                     \
                absl::Status(error_code, ""),                                \
                ::torch_tpu::GetRootOpName(std::nullopt)))                   \
            .ConsumeRequiringMessage()  // Caller must provide a message.

// Throws a TtError based on the error status.
// Use this instead of TT_CHECK_THROW when we need to control the source
// location of the error message or the type of the exception.
//
// REQUIRES: status is not OK.
#define TT_THROW_TT_ERROR_(status, source_loc)                               \
  do {                                                                       \
    const absl::Status& _status = (status);                                  \
    throw /* for implementing error API */ ::torch_tpu::TtError(_status,     \
                                                                source_loc); \
  } while (false)

// =================================================================
// == Implementation details, do not rely on anything below here. ==
// =================================================================

namespace internal {

// Special adaptor for use by TT_THROW_IF_ERROR for absl::Status
// arguments. It converts a StatusBuilder to a std::runtime_error.
//
// REQUIRES: Only used by TT_THROW_IF_ERROR implementation.
class ThrowIfErrorAdapter {
 public:
  ThrowIfErrorAdapter(const char* file, uint32_t line, const char* function)
      : file_(file), line_(line), function_(function) {}

  // This is trivially copyable and movable.
  ThrowIfErrorAdapter(const ThrowIfErrorAdapter&) = default;
  ThrowIfErrorAdapter& operator=(const ThrowIfErrorAdapter&) = default;
  ThrowIfErrorAdapter(ThrowIfErrorAdapter&&) = default;
  ThrowIfErrorAdapter& operator=(ThrowIfErrorAdapter&&) = default;

  // Converts a Status to a python exception. The weird choice of
  // operator= is due to:
  //   - A reguarlar function is incommpatible in syntax with how this adaptor
  //     is used in the TT_THROW_IF_ERROR macro.
  //   - operator= has a fairly low precedence (lower than operator<<), so
  //     the TT_THROW_IF_ERROR(expr) macro expands to
  //       ...
  //       ::torch_tpu::internal::ThrowIfErrorAdapter(
  //           TT_NORMALIZED_FILE, __LINE__, __func__) =
  //           tt_error_adapter_.Consume() << "additional " << information;
  //     which parses correctly.
  ThrowIfErrorAdapter& operator=(const absl::Status& status) {
    TT_THROW_TT_ERROR_(status, c10::SourceLocation({function_, file_, line_}));
    return *this;
  }

 private:
  const char* file_;
  uint32_t line_;
  const char* function_;
};

// Special adaptor for use by TT_RETURN_IF_ERROR for absl::Status
// arguments. This one avoids constructing a StatusBuilder on the fast path.
//
// REQUIRES: Only used by TT_RETURN_IF_ERROR implementation.
//
// The implementation is largely borrowed from the RETURN_IF_ERROR adaptor
// inside Google.
class ReturnIfErrorAdaptor {
 public:
  explicit ReturnIfErrorAdaptor(absl::Status status)
      : status_(std::move(status)) {}

  ReturnIfErrorAdaptor() = delete;
  ReturnIfErrorAdaptor(const ReturnIfErrorAdaptor&) = delete;
  ReturnIfErrorAdaptor& operator=(const ReturnIfErrorAdaptor&) = delete;

  ~ReturnIfErrorAdaptor() {
    // WARNING! WARNING! WARNING!
    //
    // We play fast and loose here and avoid destroying status_. This should be
    // safe because status_ will never own memory at destruction time. The two
    // cases to consider are:
    //  (1) OK: OkStatus() representation needs no cleanup
    //  (2) Not-OK: we take the else branch in RETURN_IF_ERROR and move
    //      status_ into StatusBuilder which leaves status_ with a MovedFromRep
    //      that needs no cleanup.
    // If the absl::Status implementation changes, leaks should be caught by
    // the various tests we have when run under lsan or debug leak checker.
  }

  explicit operator bool() const { return ABSL_PREDICT_TRUE(status_.ok()); }

  // Returns a status builder based on the original status.
  //
  // Since the original status should already have an error message, the
  // return type is StatusBuilderWithMessage instead of StatusBuilder, so
  // that we don't have to stream more messages into it.
  StatusBuilderWithMessage Consume() {
    return StatusBuilderWithMessage(std::move(status_));
  }

  // Like Consume(), but enforces at compile time that the caller to stream at
  // least one message into the returned StatusBuilder.
  StatusBuilder ConsumeRequiringMessage() {
    return StatusBuilder(std::move(status_));
  }

  // Returns a status builder that contains a copy of the original status,
  // but with the given C++ source location information appended to the
  // absl::Status payload if requested by the user.
  //
  // Since the original status should already have an error message, the
  // return type is StatusBuilderWithMessage instead of StatusBuilder, so
  // that we don't have to stream more messages into it.
  StatusBuilderWithMessage Consume(const std::string_view file, const int line,
                                   const std::string_view function) {
    return StatusBuilderWithMessage(
        MaybeAddCppSourceLoc(std::move(status_), file, line, function));
  }

 private:
  // Place the status inside a union so we can avoid generating unnecessary code
  // to call the Status destructor.
  union {
    absl::Status status_;
    char nothing_[1];
  };
};

// IsStatusOrTrait<Type>::value is true iff Type is an absl::StatusOr<>.
template <typename T>
struct IsStatusOrTrait : std::false_type {};
template <typename T>
struct IsStatusOrTrait<absl::StatusOr<T>> : std::true_type {};

// A class that can be implicitly converted to any type T where throwing
// from a function returning T is allowed by torch_tpu's local policy.
// This is used to enforce at compile time safe usage of exceptions.
class AnythingCompatibleWithThrow {
 public:
  AnythingCompatibleWithThrow() = default;

  // Allows `return AnythingCompatibleWithThrow();` to compile in any function
  // whose return type is blessed for throwing, but fail to compile otherwise.
  //
  // This version is for functions that return by reference.
  template <typename T>
  operator T&() const {
    StaticAssertCanThrow<std::remove_cv_t<std::remove_reference_t<T>>>();
    ABSL_LOG(FATAL)  // CRASH_OK
        << "This is only used for compile time checks. It should "
           "never be called at runtime.";
  }
  // This version is for functions that return by value.
  template <typename T>
  operator T() && {
    StaticAssertCanThrow<std::remove_cv_t<std::remove_reference_t<T>>>();
    ABSL_LOG(FATAL)  // CRASH_OK
        << "This is only used for compile time checks. It should "
           "never be called at runtime.";
  }

 private:
  // Statically asserts that T is a type where throwing in a function returning
  // T is allowed by torch_tpu's local policy.
  template <typename T>
  static void StaticAssertCanThrow() {
    // Extract the assertion conditions into variables with descriptive names
    // for better error messages.
    const bool return_pointer = std::is_pointer_v<T>;
    const bool return_status = std::is_same_v<T, absl::Status>;
    const bool return_status_or = IsStatusOrTrait<T>::value;
    static_assert(!return_pointer,
                  "A throwing TT_* macro cannot be used in a function that "
                  "returns a raw pointer.");
    static_assert(!return_status,
                  "A throwing TT_* macro cannot be used in a function that "
                  "returns absl::Status.");
    static_assert(!return_status_or,
                  "A throwing TT_* macro cannot be used in a function that "
                  "returns absl::StatusOr<>.");
  }
};

}  // namespace internal

// The GNU compiler emits a warning for code like:
//
//   if (foo)
//     if (bar) { } else baz;
//
// because it thinks you might want the else to bind to the first if.  This
// leads to problems with code like:
//
//   if (do_expr) RETURN_IF_ERROR(expr) << "Some message";
//
// The "switch (0) case 0:" idiom is used to suppress this.
#define TT_ELSE_BLOCKER_ \
  switch (0)             \
  case 0:                \
  default:  // NOLINT

// Internal helper for concatenating macro values.
//
// We need the *_INNER_ indirection to force the preprocessor to evaluate the
// macro arguments. E.g. if we write
//   #define CONCAT(x, y) x##y
//   #define MAKE_CONST(var) const_##var
// then
//   CONCAT(MAKE_CONST(foo), _suffix)
// will be expanded to
//   MAKE_CONST(foo)##_suffix
// instead of
//   const_foo_suffix
#define TT_CONCAT_INNER_(x, y) x##y
#define TT_CONCAT_(x, y) TT_CONCAT_INNER_(x, y)

// TT_GET_VARIADIC_((..., x3, x2)) expands to x2 if ... is a list of
// 2 elements, and x3 if ... is a list of 3 elements.
#define TT_GET_VARIADIC_(args)                            \
  /* NOLINTNEXTLINE(clang-diagnostic-pre-c++20-compat) */ \
  TT_GET_VARIADIC_HELPER_ args
// MSVC incorrectly expands variadic macros, splice together a macro call to
// work around the bug.
#define TT_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) NAME

// Statically assert that the return type of the surrounding function is
// neither absl::Status nor absl::StatusOr<>. In other words, if used
// inside a function that returns absl::Status or absl::StatusOr<>, the
// compilation will fail. Otherwise, the compilation will succeed and
// this does nothing.
//
// Implementation notes:
//
//   - When this macro is used in a function that returns absl::Status or
//     absl::StatusOr<>, the compiler will try to implicitly convert the
//     return value (i.e. AnythingCompatibleWithThrow()) to the function return
//     type. This triggers a static_assert in AnythingCompatibleWithThrow's type
//     conversion operator, and the compilation fails.
//   - Otherwise, the AnythingCompatibleWithThrow object can be implicitly
//   converted
//     to the function return type, and the compilation succeeds.
//   - The `return AnythingCompatibleWithThrow();` statement is wrapped inside a
//     `if (false)` and thus won't be executed. It is only used for the
//     compile-time check described above.
//   - The implementation triggers the clang "return-mismatch" warning.
//     Therefore we disable the warning for the duration of the macro body.
//     See
//     https://clang.llvm.org/docs/DiagnosticsReference.html#wreturn-mismatch
//     for what this warning is about.
//
// Caveat: since the implementation depends on the C++ compiler, it is
// only supported for internal builds. In the OSS builds, this macro is a
// no-op.
#if TT_IS_INTERNAL_TORCH_TPU
#define TT_STATIC_ASSERT_RETURN_TYPE_NOT_STATUS_()                 \
  do {                                                             \
    if (false) {                                                   \
      _Pragma("clang diagnostic push");                            \
      _Pragma("clang diagnostic ignored \"-Wreturn-mismatch\"");   \
      return ::torch_tpu::internal::AnythingCompatibleWithThrow(); \
      _Pragma("clang diagnostic pop");                             \
    }                                                              \
  } while (false)
#else
#define TT_STATIC_ASSERT_RETURN_TYPE_NOT_STATUS_() TT_REQUIRE_SEMICOLON_
#endif  // TT_IS_INTERNAL_TORCH_TPU

// Implements TT_ASSIGN_OR_RETURN(lhs, rexpr).
#define TT_ASSIGN_OR_RETURN_2_(lhs, rexpr) \
  TT_ASSIGN_OR_RETURN_IMPL_2_(TT_CONCAT_(_statusor_, __LINE__), lhs, rexpr)

// Implements TT_ASSIGN_OR_RETURN(lhs, rexpr). `statusor` is the name of the
// variable that will contain the result of the expression.
#define TT_ASSIGN_OR_RETURN_IMPL_2_(statusor, lhs, rexpr)                      \
  auto statusor = (rexpr);                                                     \
  TT_STATIC_ASSERT_NOT_STATUS_OR_REF_(statusor);                               \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                                    \
    return ::torch_tpu::internal::MaybeAddCppSourceLoc(                        \
        std::move(statusor).status(), TT_NORMALIZED_FILE, __LINE__, __func__); \
  }                                                                            \
  TT_REMOVE_PARENS_(lhs) = (*std::move(statusor))

// Implements TT_ASSIGN_OR_RETURN(lhs, rexpr, error_expr).
#define TT_ASSIGN_OR_RETURN_3_(lhs, rexpr, error_expr)                      \
  TT_ASSIGN_OR_RETURN_IMPL_3_(TT_CONCAT_(_statusor_, __LINE__), lhs, rexpr, \
                              error_expr)

// Implements TT_ASSIGN_OR_RETURN(lhs, rexpr, error_expr). `statusor` is the
// name of the variable that will contain the result of the expression.
//
// The error_expr can reference the special `_` variable, which is a status
// builder containing the error status. Since this status already has an
// error message, we don't have to stream more messages into it. Hence the
// type of `_` is StatusBuilderWithMessage instead of StatusBuilder.
//
// Since error_expr doesn't have to reference the `_` variable, we include
// a static_cast<void>(_) to avoid a "unused variable" warning from the C++
// compiler.
#define TT_ASSIGN_OR_RETURN_IMPL_3_(statusor, lhs, rexpr, error_expr)   \
  auto statusor = (rexpr);                                              \
  TT_STATIC_ASSERT_NOT_STATUS_OR_REF_(statusor);                        \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                             \
    ::torch_tpu::StatusBuilderWithMessage _(                            \
        ::torch_tpu::internal::MaybeAddCppSourceLoc(                    \
            std::move(statusor).status(), TT_NORMALIZED_FILE, __LINE__, \
            __func__));                                                 \
    static_cast<void>(_);                                               \
    return error_expr;                                                  \
  }                                                                     \
  TT_REMOVE_PARENS_(lhs) = (*std::move(statusor))

// Implements TT_ASSIGN_OR_THROW(lhs, rexpr).
#define TT_ASSIGN_OR_THROW_2_(lhs, rexpr) \
  TT_ASSIGN_OR_THROW_IMPL_2_(TT_CONCAT_(_statusor_, __LINE__), lhs, rexpr)

// Implements TT_ASSIGN_OR_THROW(lhs, rexpr). `statusor` is the name of the
// variable that will contain the result of the expression.
//
// Implementation note: we don't capture the caller's source location here
// because TT_THROW_TT_ERROR_ already does that.
#define TT_ASSIGN_OR_THROW_IMPL_2_(statusor, lhs, rexpr)       \
  auto statusor = (rexpr);                                     \
  TT_STATIC_ASSERT_NOT_STATUS_OR_REF_(statusor);               \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                    \
    TT_THROW_TT_ERROR_(statusor.status(), TT_SOURCE_LOCATION); \
  }                                                            \
  TT_REMOVE_PARENS_(lhs) = (*std::move(statusor))

// Implements TT_ASSIGN_OR_THROW(lhs, rexpr, error_expr).
//
// Implementation note: we don't capture the caller's source location here
// because TT_THROW_TT_ERROR_ already does that.
#define TT_ASSIGN_OR_THROW_3_(lhs, rexpr, error_expr)                      \
  TT_ASSIGN_OR_THROW_IMPL_3_(TT_CONCAT_(_statusor_, __LINE__), lhs, rexpr, \
                             error_expr)

// Implements TT_ASSIGN_OR_THROW(lhs, rexpr, error_expr). `statusor` is the name
// of the variable that will contain the result of the expression.
//
// The error_expr can reference the special `_` variable, which is a status
// builder containing the error status. Since this status already has an
// error message, we don't have to stream more messages into it. Hence the
// type of `_` is StatusBuilderWithMessage instead of StatusBuilder.
//
// Since error_expr doesn't have to reference the `_` variable, we include
// a static_cast<void>(_) to avoid a "unused variable" warning from the C++
// compiler.
#define TT_ASSIGN_OR_THROW_IMPL_3_(statusor, lhs, rexpr, error_expr)       \
  auto statusor = (rexpr);                                                 \
  TT_STATIC_ASSERT_NOT_STATUS_OR_REF_(statusor);                           \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                                \
    ::torch_tpu::StatusBuilderWithMessage _(std::move(statusor).status()); \
    static_cast<void>(_);                                                  \
    TT_THROW_TT_ERROR_(absl::Status((error_expr)), TT_SOURCE_LOCATION);    \
  }                                                                        \
  TT_REMOVE_PARENS_(lhs) = (*std::move(statusor))

// Trait: is the type a StatusOr<T&>?
template <typename T>
struct is_status_or_ref : std::false_type {};
template <typename T>
struct is_status_or_ref<absl::StatusOr<T&>> : std::true_type {};

// Statically asserts that the type of `statusor` is not a StatusOr<T&>.
#define TT_STATIC_ASSERT_NOT_STATUS_OR_REF_(statusor)                       \
  static_assert(                                                            \
      !is_status_or_ref<std::remove_cv_t<                                   \
          std::remove_reference_t<decltype(statusor)>>>::value,             \
      "TorchTPU doesn't allow returning StatusOr<T&> as it's error-prone (" \
      "it's easy to bind the value to a T object, creating a new copy). "   \
      "Try returning StatusOr<std::unique_ptr<T>> or StatusOr<T* "          \
      "absl_nonnull> instead.")

// If the input is parenthesized, removes the parentheses. Otherwise expands to
// the input unchanged.
#define TT_REMOVE_PARENS_(...)                                    \
  TT_IF_(TT_IS_PARENTHESIZED_(__VA_ARGS__), TT_REM_, TT_EMPTY_()) \
  __VA_ARGS__

// If cond is 0, expands to else_branch; if cond is 1, expands to then_branch.
#define TT_IF_(cond, then_branch, else_branch) \
  TT_CONCAT_(TT_IF_IMPL_, cond)(then_branch, else_branch)
#define TT_IF_IMPL_1(then_branch, else_branch) then_branch
#define TT_IF_IMPL_0(then_branch, else_branch) else_branch

// Expands to 1 if the input is parenthesized. Otherwise expands to 0.
// How this works:
// - if the input is parenthesized, e.g. (foo), TT_EAT_ __VA_ARGS__ will
//   expand to TT_EAT_(foo), which in turn will expand to empty.
// - if the input is not parenthesized, e.g. foo,TT_EAT_ __VA_ARGS__ will
//   expand to TT_EAT_ foo, which is not empty.
#define TT_IS_PARENTHESIZED_(...) TT_IS_EMPTY_(TT_EAT_ __VA_ARGS__)

// Expands to 1 if the input is empty. Otherwise expands to 0.
// How this works:
// - if the input is empty, the macro will expand to TT_IS_EMPTY_I_(),
//   which in turn will expand to TT_IS_EMPTY_INNER_(_), which will
//   expand to 0.
// - if the input is not empty, e.g. foo, the macro will expand to
//   TT_IS_EMPTY_I_(foo), which in turn will expand to
//   TT_IS_EMPTY_INNER_(_, foo), which will expand to 1.
#define TT_IS_EMPTY_(...) TT_IS_EMPTY_I_(__VA_ARGS__)
#define TT_IS_EMPTY_I_(...) TT_IS_EMPTY_INNER_(_ __VA_OPT__(, )##__VA_ARGS__)

#define TT_EAT_(...)
#define TT_REM_(...) __VA_ARGS__
#define TT_EMPTY_()

// Internal helpers for emptiness arguments check.
#define TT_IS_EMPTY_INNER_(...) TT_IS_EMPTY_INNER_HELPER_((__VA_ARGS__, 0, 1))

// MSVC expands variadic macros incorrectly, so we need this extra indirection
// to work around that (b/110959038).
#define TT_IS_EMPTY_INNER_HELPER_(args) TT_IS_EMPTY_INNER_I_ args
#define TT_IS_EMPTY_INNER_I_(e0, e1, is_empty, ...) is_empty

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_ERROR_UTILS_H_
