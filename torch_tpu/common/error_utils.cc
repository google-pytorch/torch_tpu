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

#include "torch_tpu/common/error_utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/WrapDimMinimal.h"
#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for IntArrayRef
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "c10/util/StringUtil.h"
#include "re2/re2.h"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/status_builder.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/eager_mode.h"
#include "xla/xla_data.pb.h"

#if !defined(NDEBUG) && TT_IS_INTERNAL_TORCH_TPU
// Enables error messages style check for internal debug builds.
//
// When this macro is set to 1, TorchTPU will hard crash if any error is raised
// with an error message that doesn't conform to the TorchTPU error handling
// guidelines. This should help us maintain the quality and consistency of error
// messages in the project.
#define TT_CHECK_ERROR_MESSAGE_FOLLOWS_GUIDELINES_ 1
#endif

namespace torch_tpu {
namespace {

#ifdef TT_CHECK_ERROR_MESSAGE_FOLLOWS_GUIDELINES_

// Pair of corresponding type names.
struct TypeNamePair {
  std::string_view torch;
  std::string_view stablehlo;
};

// Even though this array might get outdated, we should keep this check due to
// the following reasons:
//
//   1. It won't be changing too frequent
//   2. These type names doesn't collide with PyTorch names
//   3. It's not critical for TorchTPU realeases, i.e. getting this wrong
//      won't be too bad
constexpr std::array<TypeNamePair, 19> type_name_pairs{{
    // NOTE: the order in this array is important!
    //   These types are partially ordered, so that if type A comes before
    //   another type B, it means that A is NOT a substring of B.
    //
    //   For example:
    //     - f16 and bf16: since f16 is a substring of bf16, bf16 should
    //                     appear before f16.
    //     - f32 and complexf32: since f32 is a substring of complexf32,
    //                     complexf32 should appear before f32.

    // Integer types.
    {.torch = "bool", .stablehlo = "pred"},
    {.torch = "uint2", .stablehlo = "ui2"},
    {.torch = "uint4", .stablehlo = "ui4"},
    {.torch = "uint8", .stablehlo = "ui8"},
    {.torch = "uint16", .stablehlo = "ui16"},
    {.torch = "uint32", .stablehlo = "ui32"},
    {.torch = "uint64", .stablehlo = "ui64"},
    {.torch = "int2", .stablehlo = "i2"},
    {.torch = "int4", .stablehlo = "i4"},
    {.torch = "int8", .stablehlo = "i8"},
    {.torch = "int16", .stablehlo = "i16"},
    {.torch = "int32", .stablehlo = "i32"},
    {.torch = "int64", .stablehlo = "i64"},

    // Complex types.
    {.torch = "complex64", .stablehlo = "complexf32"},
    {.torch = "complex128", .stablehlo = "complexf64"},

    // Floating-point types.
    {.torch = "bfloat16", .stablehlo = "bf16"},
    {.torch = "float16", .stablehlo = "f16"},
    {.torch = "float32", .stablehlo = "f32"},
    {.torch = "float64", .stablehlo = "f64"},
}};

constexpr std::string_view kPreconditionViolatedError =
    "Error message check pre-condition violated.";

// Check: is not empty nor filled with blank spaces.
// Pre-conditions: None.
absl::Status TrimmedErrorMessageIsNotEmpty(const std::string_view message) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal error supposedly unreachable
                 // from Python.
      !absl::StripAsciiWhitespace(message).empty(), error::kInternal)
      << "The trimmed error message should not be empty. Add an appropriate "
         "error message to this error.";
  return absl::OkStatus();
}

// Check: has no leading whitespace.
// Pre-conditions: The error message must not be empty.
absl::Status HasNoLeadingWhitespace(const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kPreconditionViolatedError;

  const size_t num_leading_whitespace =
      message.size() - absl::StripLeadingAsciiWhitespace(message).size();
  TT_RET_CHECK(num_leading_whitespace == 0, error::kInternal)
      << "The error message should not start with whitespace. Remove the "
         "leading "
      << num_leading_whitespace << " whitespace character(s).";
  return absl::OkStatus();
}

// Check: has no trailing whitespace.
//
// Pre-conditions: The error message must not be empty.
absl::Status HasNoTrailingWhitespace(const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kPreconditionViolatedError;

  const size_t num_trailing_whitespace =
      message.size() - absl::StripTrailingAsciiWhitespace(message).size();
  TT_RET_CHECK(num_trailing_whitespace == 0, error::kInternal)
      << "The error message should not end with whitespace. Remove the "
         "trailing "
      << num_trailing_whitespace << " whitespace character(s).";

  return absl::OkStatus();
}

// Check: doesn't capitalize the beginning.
// Pre-conditions:
//   - The error message must not be empty.
//   - The error message must not start with whitespace.
//
// The first character of `message` should be lower-case. We intentionally allow
// other characters, e.g. punctuation and numbers.
//
// Examples (the "ones(): " prefix is just for illustration purposes):
//     [good] ones(): dimensions...
//      [bad] ones(): Dimensions...
//                    |
//                    |_ `message` starts from this character.
absl::Status StartsWithNonUppercaseLetter(const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kPreconditionViolatedError;
  ABSL_CHECK(  // CRASH_OK
      !absl::ascii_isspace(message.front()))
      << kPreconditionViolatedError;

  // Returns the first word of `message`.
  //
  // If `to_lower` parameter is set to `true`, convert the first word to
  // lower-case, and return it.
  auto get_first_word = [message](const bool to_lower = false) -> std::string {
    const size_t first_word_end = message.find_first_of(' ');
    // If there is no blank space in the whole message, first_word_end will be
    // string_view::npos. So, the expression below will return the whole trimmed
    // error message.
    const std::string_view first_word = message.substr(0, first_word_end);
    return std::string(to_lower ? absl::AsciiStrToLower(first_word)
                                : first_word);
  };

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal error supposedly unreachable
                 // from Python.
      !absl::ascii_isupper(message.front()), error::kInternal)
      << "The error message should start with a lowercase character. Replace '"
      << get_first_word() << "' with '" << get_first_word(/* to_lower= */ true)
      << "'.";

  return absl::OkStatus();
}

// Check: ends with an alphanumeric character.
//
// Pre-conditions:
//   - The error message must not be empty.
//   - The error message must not end with whitespace.
//
// The last character of `message` should be alphanumeric.
absl::Status EndsWithAlphaNum(const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kPreconditionViolatedError;
  ABSL_CHECK(  // CRASH_OK
      !absl::ascii_isspace(message.back()))
      << kPreconditionViolatedError;

  // Find the suffix of non-alphanumeric characters.
  auto last_alnum_it =
      std::find_if(message.rbegin(), message.rend(), absl::ascii_isalnum);
  const size_t non_alnum_suffix_start_pos =
      message.size() - std::distance(last_alnum_it.base(), message.end());
  const std::string_view non_alnum_suffix =
      message.substr(non_alnum_suffix_start_pos);

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal error supposedly unreachable
                 // from Python.
      non_alnum_suffix.empty(), error::kInternal)
      << "The error message should end with an alpha-numeric character. "
         "Remove the last "
      << non_alnum_suffix.size() << " characters: '" << non_alnum_suffix
      << "'.";

  return absl::OkStatus();
}

// Check: if it states the expected condition first, then it should state what
// is wrong afterwards.
// Pre-conditions: The error message must not be empty.
//
// This function is composed of 2 steps:
//
//     1. Identify whether the error message states the expected condition or
//        actual value. We do that by looking for the presence of the following
//        words:
//         - "expected ..."
//         - "... must ..."
//         - "..., got ..." (or ". Got")
//         - "..., but got ..." (or ". But got")
//
//     2. If (1) is true, check that it follows the following format:
//         - expected ..., got ...
absl::Status HasValidFormat(const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kPreconditionViolatedError;

  static LazyRE2 kExpectedConditionOrActualValuePattern = {
      R"((.*\bexpected\b.*|.*\bmust\b.*|.*[.,]\s*\b(?i:got|but got)\b.*))"};
  static LazyRE2 kExpectedGotPattern = {R"(.*\bexpected\b.*,\s*\bgot\b.*)"};

  if (RE2::FullMatch(message, *kExpectedConditionOrActualValuePattern)) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal error supposedly unreachable
                   // from Python.
        RE2::FullMatch(message, *kExpectedGotPattern), error::kInternal)
        << "The error message states an expected condition or the actual "
           "value (i.e. contains \"expected\", \"must\", or \", (but) got\" "
           "strings). Therefore, it should be in the following format: "
           "\"expected ..., got ...\".";
  }

  return absl::OkStatus();
}

// Check: shows all types as their official PyTorch dtype names.
// Pre-conditions: The error message must not be empty.
//
// Show PyTorch dtype names, as opposed to StableHLO/XLA type names.
//     - "float32" vs. "f32"
//     - "int32" vs. "si32"
//     - "int64" vs. "si64"
//     - "bool" vs. "pred"
//     - "bfloat16" vs. "bf16".
absl::Status DoesNotContainStableHloTypeName(const std::string_view message,
                                             const TypeNamePair& pair) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kPreconditionViolatedError;

  const std::string type_name_regex =
      absl::StrCat("\\b", pair.stablehlo, "\\b");

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal error supposedly
                 // unreachable from Python.
      !RE2::PartialMatch(message, type_name_regex), error::kInternal)
      << "The error message should use the PyTorch dtype names instead of "
         "the StableHLO type names, i.e. it should show '"
      << pair.torch << "' instead of '" << pair.stablehlo
      << "'. Use ToString(mlir::ElementType) for retrieving the PyTorch "
         "dtype name.";
  return absl::OkStatus();
}

// Enum specifying whether a check should be enforced or logged as warning.
//
// Checks marked as kWarn that fail are logged as warning messages. Enforced
// checks that fail will trigger a hard crash.
enum class CheckKind { kEnforce, kWarn };

// Result of checking the error message.
struct ErrorMessageChecksResult {
  // Whether the error message passed the enforced checks.
  // If true, indicates that the error message conforms to the enforced
  // guidelines (even if there are non-enforced guidelines failing), so we
  // should NOT crash.
  bool success = true;

  // The status messages (i.e. reasons) of the failing checks.
  std::vector<std::string> failure_reasons;
};

// Calls all checks, and returns their failure reasons, if any has failed. If
// there are no check failures, return a default constructed
// `ErrorMessageChecksResult{}`.
ErrorMessageChecksResult GetErrorMessageChecksResult(
    const std::string_view message) {
  ErrorMessageChecksResult result;

  // Handles the result of an error message check.
  //
  // Does nothing if the check was successful.
  // Otherwise, propagate its status by setting the `result` appropriately.
  const auto handle_check = [&result](
                                const absl::Status& status,
                                const CheckKind kind = CheckKind::kEnforce) {
    constexpr std::string_view kEnforcedPrefix = "(enforced)";

    if (!status.ok()) {
      const bool should_enforce_check = kind == CheckKind::kEnforce;

      // Given that this check has failed, set `success` depending on the value
      // of `should_enforce_check`:
      //
      //   - If it's `false` (i.e. should NOT be enforced), `success` should not
      //     change
      //
      //   - Otherwise, `success` should be false
      result.success = result.success && !should_enforce_check;

      if (should_enforce_check) {
        result.failure_reasons.push_back(
            absl::StrCat(kEnforcedPrefix, " ", status.message()));
      } else {
        result.failure_reasons.push_back(std::string(status.message()));
      }
    }
  };

  // In order to enforce vs. warn on a check, change the second argument of the
  // `handle_check`.

  const absl::Status status_not_empty = TrimmedErrorMessageIsNotEmpty(message);
  handle_check(status_not_empty, CheckKind::kEnforce);

  if (!status_not_empty.ok()) {
    return result;
  }

  handle_check(HasNoLeadingWhitespace(message), CheckKind::kEnforce);
  handle_check(HasNoTrailingWhitespace(message), CheckKind::kEnforce);

  // TODO(marcosyukio): enforce the other error message checks.
  //
  // We shall enforce these errors one-by-one, since enforcing each of them
  // might end up breaking our CI due to various errors.

  handle_check(HasValidFormat(message), CheckKind::kWarn);

  for (const auto& pair : type_name_pairs) {
    handle_check(DoesNotContainStableHloTypeName(message, pair),
                 CheckKind::kWarn);
  }

  // The subsequent checks require, as pre-condition, the message to be trimmed.
  const std::string_view trimmed_message = absl::StripAsciiWhitespace(message);

  handle_check(EndsWithAlphaNum(trimmed_message), CheckKind::kWarn);
  handle_check(StartsWithNonUppercaseLetter(trimmed_message), CheckKind::kWarn);

  return result;
}

// Formats a detailed log message explaining guideline violations.
std::string GetLogMessage(const TtError& error,
                          const std::vector<std::string>& failure_reasons) {
  const auto& source = error.thrown_from();

  return absl::StrCat(
      source.file, ":", source.line,
      ": Improper error message format in function '", source.function,
      "' (please follow go/tt-error-guide).\n", "\n", "Actual error message:\n",
      "  '", error.status().message(), "'\n", "\n", "What's wrong with it:\n",
      absl::StrJoin(failure_reasons, /* separator= */ "\n",
                    /* fmt= */ [](std::string* out, const std::string& reason) {
                      absl::StrAppend(out, "  - ", reason);
                    }));
}

void CheckErrorMessageFollowsGuidelines(const TtError& error) {
  const ErrorMessageChecksResult& result =
      GetErrorMessageChecksResult(error.status().message());

  if (result.success) {
    // If there are any error message checks failing, log them as warning.
    if (!result.failure_reasons.empty()) {
      ABSL_LOG(WARNING)
          << GetLogMessage(error, result.failure_reasons) << "\n"
          << "\n"
          << "    All error message checks (see above) are going to be "
             "enforced in\n"
          << "    the future. The \"Actual error message\" raised above fails "
             "some\n"
          << "    checks. Consider addressing the problems described.";
    }
    return;
  }

  // Crash if the error message doesn't pass the currently enforced checks.
  ABSL_LOG(FATAL)  // CRASH_OK
      << GetLogMessage(error, result.failure_reasons);
}

#endif  // TT_CHECK_ERROR_MESSAGE_FOLLOWS_GUIDELINES_

}  // namespace

bool GetEnableDebugChecks() {
  const auto& env_var = GetEnvOnce<kTorchTpuInternalEnableDebugChecksEnvVar>();
  // If the env var is set, respect the user's choice.
  if (env_var.has_value()) {
    return *env_var == "1";
  }
  // Otherwise, enable debug checks if we are in debug eager mode.
  return GetEagerMode() == EagerMode::kDeferNeverAndLaunchBlocking;
}

absl::StatusOr<int64_t> SafeMultiply(int64_t x, int64_t y) {
  const auto product_i128 = absl::int128(x) * y;  // Guaranteed to not overflow.
  const auto product_i64 = static_cast<int64_t>(product_i128);
  TT_RET_CHECK(product_i64 == product_i128, error::kInvalidArgument)
      << "product of " << x << " and " << y << " overflows as int64";
  return product_i64;
}

absl::StatusOr<int64_t> NumElements(absl::Span<const int64_t> dims) {
  int64_t n = 1;
  for (auto d : dims) {
    TT_RET_CHECK(d >= 0, error::kInvalidArgument)
        << "dimension size must be >= 0, got " << d;
    TT_ASSIGN_OR_RETURN(n, SafeMultiply(n, d),
                        _.SetOverride() << "product of dimension sizes ["
                                        << absl::StrJoin(dims, ", ")
                                        << "] overflows as int64");
  }
  return n;
}

absl::StatusOr<int64_t> NumElements(const at::Tensor& tensor) {
  return NumElements(tensor.sizes());
}

absl::StatusOr<int64_t> SafeWrapDim(int64_t dim, int64_t dim_bound) {
  try {
    return at::maybe_wrap_dim(  // MAYBE_WRAP_DIM_OK=implementing SafeWrapDim.
        dim, dim_bound);
  } catch (const c10::Error& e) {
    return TT_ERROR(error::kIndexError) << e.what_without_backtrace();
  }
}

void CheckDeviceIsTpu(const c10::optional<at::Device> device_opt,
                      const std::string_view op_name) {
  ABSL_CHECK(device_opt.has_value());       // CRASH_OK
  ABSL_CHECK_EQ(device_opt.value().type(),  // CRASH_OK
                GetPrivateUse1DeviceType());
}

bool IsXlaOomError(const absl::Status& status) {
  return absl::IsResourceExhausted(status) ||
         (absl::IsInternal(status) &&
          // Some XLA OOM errors are reported as internal errors with this
          // word in the message.
          absl::StrContains(status.message(), "allocation_size"));
}

enum class ExceptionType {
  kTtError,
  kC10Error,
};

// Returns the error message for a Status error.
// Arguments:
// - status: the error Status.
// - thrown_from: the source location where the exception is thrown.
// - exception_type: the type of the exception to be thrown.
static std::string MakeErrorMessage(const absl::Status& status,
                                    const c10::SourceLocation& thrown_from,
                                    const ExceptionType exception_type) {
  if (status.ok()) {
    return "";
  }
  const auto root_op_name = internal::GetRootOpNamePayload(status);
  const std::optional<bool> show_cpp_trace_env = TorchShowCppStacktraces();
  const bool add_cpp_trace =
      exception_type == ExceptionType::kTtError
          ?
          // When throwing a TtError to PyTorch, since TtError is not a subclass
          // of c10::Error, PyTorch won't automatically add the C++ stack trace
          // to the error message even if TORCH_SHOW_CPP_STACKTRACES is set
          // to 1. Therefore we include the C++ stack trace in the error message
          // ourselves when needed. In particular:
          //
          // - For internal torch_tpu, we always add the C++ trace unless
          //   TORCH_SHOW_CPP_STACKTRACES is set to 0.
          // - For external torch_tpu, we add the C++ trace only if
          //   TORCH_SHOW_CPP_STACKTRACES is not set.
          (TT_IS_INTERNAL_TORCH_TPU ? show_cpp_trace_env != false
                                    : show_cpp_trace_env == true)
          // When throwing a c10::Error, PyTorch will add the C++ stack trace
          // if TORCH_SHOW_CPP_STACKTRACES is set to 1. Therefore we don't need
          // to add the C++ stack trace ourselves normally. However, for
          // internal torch_tpu, we add the C++ trace when
          // TORCH_SHOW_CPP_STACKTRACES is not set (PyTorch won't add it
          // automatically in this case), so that Google developers can easily
          // debug torch_tpu bugs.
          : TT_IS_INTERNAL_TORCH_TPU && show_cpp_trace_env == std::nullopt;
  const std::string message =
      absl::StrCat(internal::GetMessageWithCppErrorTrace(status),
                   add_cpp_trace ?
                                 // Leverage c10::Error to generate a string
                                 // with the C++ stack trace.
                       c10::Error(thrown_from, "").what()
                                 : "");
  return root_op_name.empty() ? message
                              : absl::StrCat(root_op_name, "(): ", message);
}

TtError::TtError(absl::Status status, c10::SourceLocation thrown_from)
    // The std::runtime_error message is used when the TtError is passed to
    // PyTorch without being intercepted by TT_KERNEL (e.g. for ops that aren't
    // migrated to TT_KERNEL yet).
    : std::runtime_error(
          MakeErrorMessage(status, thrown_from, ExceptionType::kTtError)),
      status_(std::move(status)),
      thrown_from_(std::move(thrown_from)) {
  ABSL_CHECK(!status_.ok())  // CRASH_OK
      << "TtError must be constructed from a non-OK status.";
}

void TranslateToC10ErrorAndThrow(const TtError& e) {
#ifdef TT_CHECK_ERROR_MESSAGE_FOLLOWS_GUIDELINES_
  // Check that the error message format follows the TorchTPU guidelines
  // (go/tt-error-guide) only if we are in an internal debug build.
  CheckErrorMessageFollowsGuidelines(e);
#endif

  const std::string message =
      MakeErrorMessage(e.status(), e.thrown_from(), ExceptionType::kC10Error);

  // Map the error code to the corresponding c10::Error subclass to be
  // consistent with PyTorch CPU's error handling. These different c10::Error
  // subclasses will be translated to different Python exception types by
  // PyTorch (e.g. RuntimeError, IndexError, ValueError, etc).
  // TODO(wan): handle more error codes.
  switch (e.status().code()) {
    case error::kOutOfRange:
      throw  // Translates to Python IndexError.
          c10::IndexError(e.thrown_from(), message);
    case error::kUnimplemented:
      throw  // Translates to Python NotImplementedError.
          c10::NotImplementedError(e.thrown_from(), message);
    default:
      throw  // Translates to Python RuntimeError.
          c10::Error(e.thrown_from(), message);
  }
}

}  // namespace torch_tpu
