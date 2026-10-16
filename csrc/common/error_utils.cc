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

#include "csrc/common/error_utils.h"

#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/WrapDimMinimal.h"
#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for IntArrayRef
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "c10/util/StringUtil.h"
#include "csrc/common/context_states.h"
#include "csrc/common/device_type.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/status_builder.h"
#include "csrc/common/utils.h"
#include "csrc/eager/eager_mode.h"
#include "re2/re2.h"

namespace torch_tpu {
namespace {

// Words that are allowed to start an error message with an uppercase letter.
// This is an exception to the rule that error messages must start with a
// lowercase letter.
constexpr std::array<std::string_view, 3> kAllowedStartingWords{"TorchTPU",
                                                                "PjRt", "XLA"};

// Helper to check if a message starts with any of the allowed words.
[[nodiscard]] bool StartsWithAllowedWord(const std::string_view message) {
  return absl::c_any_of(
      kAllowedStartingWords,
      [&](const std::string_view word) { return message.starts_with(word); });
}

#if TT_CHECKS_ERROR_FORMAT

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

// Default message for checks' pre-condition violations.
constexpr std::string_view kPreconditionViolatedError =
    "Error message check pre-condition violated.";

// Check: is not empty nor filled with blank spaces.
// Pre-conditions: None.
absl::Status TrimmedErrorMessageIsNotEmpty(const std::string_view message) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
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
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
      num_leading_whitespace == 0, error::kInternal)
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
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
      num_trailing_whitespace == 0, error::kInternal)
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
// We also allow the message to start with specific allowed words (e.g.,
// "TorchTPU", "PjRt", "XLA").
//
// Examples (the "ones(): " prefix is just for illustration purposes):
//     [good] ones(): dimensions...
//     [good] ones(): TorchTPU does not support...
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

  // Allowed starting words: TorchTPU, PjRt, and XLA.
  if (StartsWithAllowedWord(message)) {
    return absl::OkStatus();
  }

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

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
      !absl::ascii_isupper(message.front()), error::kInternal)
      << "The error message should start with a lowercase character. Replace '"
      << get_first_word() << "' with '" << get_first_word(/* to_lower= */ true)
      << "'.";

  return absl::OkStatus();
}

// Returns OK if message does not start with an op or function call name (e.g.
// "foo()" or "foo():"). Pre-conditions: The error message must not be empty.
//
// TT_KERNEL automatically prefixes the op name (e.g. "op_name(): "). Error
// messages passed to TT_CHECK_THROW should not include an op name prefix.
absl::Status DoesNotStartWithOpName(const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kPreconditionViolatedError;

  static const LazyRE2 kOpNamePrefixPattern = {
      R"(^(?:[a-zA-Z_]\w*(?:\.|::))*[a-zA-Z_]\w*\s*\(\s*\)\s*:?)"};

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
      !RE2::PartialMatch(message, *kOpNamePrefixPattern), error::kInternal)
      << "The error message should not start with an op name (e.g. 'op()' or "
         "'op(): ') as that is automatically added by TT_KERNEL.";

  return absl::OkStatus();
}

// Check: ends with an alphanumeric character.
//
// Pre-conditions:
//   - The error message must not be empty.
//   - The error message must not end with whitespace.
//
// The last character of `message` should be either alphanumeric or other
// allowed characters (see kAllowedCharacters).
absl::Status EndsWithAlphaNumAndAllowedCharacters(
    const std::string_view message) {
  // Each character allowed to appear at the end must be followed by an
  // explanation.
  //   - ']': covers error messages ending with lists and shapes
  //   - ')': covers error messages ending with tuples and actual parenthesis
  //   - '}': covers error messages ending with shapes in braces
  //   - '>': covers error messages ending with templated types
  //   - "'": covers error messages ending with quoted devices or arbitrary
  //          strings (e.g. 'tpu')
  constexpr std::string_view kAllowedCharacters = "])}>'";

  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kPreconditionViolatedError;
  ABSL_CHECK(  // CRASH_OK
      !absl::ascii_isspace(message.back()))
      << kPreconditionViolatedError;

  // Find the suffix of non-alphanumeric characters.
  auto is_character_allowed = [kAllowedCharacters](const char c) {
    return absl::ascii_isalnum(c) || absl::StrContains(kAllowedCharacters, c);
  };

  auto last_allowed_char_it =
      std::find_if(message.rbegin(), message.rend(), is_character_allowed);
  const size_t not_allowed_suffix_size =
      std::distance(last_allowed_char_it.base(), message.end());

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
      not_allowed_suffix_size == 0, error::kInternal)
      << "The error message should end with either an alpha-numeric character "
         "or one of the following characters: '"
      << kAllowedCharacters << "'. Remove the last " << not_allowed_suffix_size
      << " characters: '"
      << message.substr(message.size() - not_allowed_suffix_size) << "'.";

  return absl::OkStatus();
}

// Check: if it states the expected condition first, then it should state what
// is wrong afterwards.
//
// Pre-conditions: The error message must not be empty.
//
// This function is composed of 2 steps:
//
//     1. Identify whether the error message states the expected condition or
//        actual value. We do that by looking for the presence of the following
//        words:
//         - "... expected ..."
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
      R"((.*\bexpected\b.*|.*[.,]\s*\b(?i:got|but got)\b.*))"};
  static LazyRE2 kExpectedGotPattern = {R"(^expected\b.*, got\b.*)"};

  if (RE2::FullMatch(message, *kExpectedConditionOrActualValuePattern)) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                   // unreachable from Python.
        RE2::FullMatch(message, *kExpectedGotPattern), error::kInternal)
        << "The error message states an expected condition or the actual "
           "value (i.e. contains \"expected\", or \", (but) got\" "
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

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
      !RE2::PartialMatch(message, type_name_regex), error::kInternal)
      << "The error message should use the PyTorch dtype names instead of "
         "the StableHLO type names, i.e. it should show '"
      << pair.torch << "' instead of '" << pair.stablehlo
      << "'. Use ToString(mlir::ElementType) for retrieving the PyTorch "
         "dtype name.";
  return absl::OkStatus();
}

// Check: if the sentinel is present, it must not have leading nor trailing
// empty spaces.
//
// Pre-condition: The error message must not be empty.
absl::Status HasNoSpacesSurroundingSentinel(const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kPreconditionViolatedError;

  const size_t sentinel_beg = message.find(kXlaCompilerFailedWith);
  const size_t sentinel_end = sentinel_beg + kXlaCompilerFailedWith.size();

  if (sentinel_beg == std::string_view::npos) {
    // Nothing to check if the sentinel string is not found.
    return absl::OkStatus();
  }

  const std::string_view context = message.substr(0, sentinel_beg);
  const std::string_view xla_error = message.substr(sentinel_end);

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
      !absl::StripAsciiWhitespace(context).empty(), error::kInternal)
      << "The TorchTPU context should not be empty when using the XLA error "
         "sentinel string. Make sure to add a context string before.";

  const size_t leading_spaces_beg = context.find_last_not_of(' ') + 1;
  const size_t leading_spaces_count = context.size() - leading_spaces_beg;
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
      leading_spaces_count == 0, error::kInternal)
      << "Expected no spaces before the XLA compiler error sentinel \""
      << kXlaCompilerFailedWith << "\" (it already begins with one), got "
      << leading_spaces_count << ".";

  const size_t trailing_spaces_count = xla_error.find_first_not_of(' ');
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error message checks crash and are
                 // unreachable from Python.
      trailing_spaces_count == 0, error::kInternal)
      << "Expected no spaces after the XLA compiler error sentinel \""
      << kXlaCompilerFailedWith << "\" (it already ends with one), got "
      << trailing_spaces_count << ".";

  return absl::OkStatus();
}

// Returns whether `message` has whitespace at the start, at the end, or at
// both ends.
//
// Pre-conditions:
//   - `message` must not be empty.
//
// Examples:
//   HasSurroundingWhitespaces(" foo")  -> true  (leading whitespace)
//   HasSurroundingWhitespaces("foo ")  -> true  (trailing whitespace)
//   HasSurroundingWhitespaces(" foo ") -> true  (leading and trailing)
//   HasSurroundingWhitespaces("foo")   -> false (trimmed)
//   HasSurroundingWhitespaces("f oo")  -> false (internal whitespace only)
[[nodiscard]] bool HasSurroundingWhitespaces(const std::string_view message) {
  ABSL_CHECK(!message.empty())  // CRASH_OK
      << kPreconditionViolatedError;

  return absl::ascii_isspace(message.front()) ||
         absl::ascii_isspace(message.back());
}

// Result of splitting an error message after leading context prefixes.
struct ContextPrefixSplit {
  std::optional<std::string_view> context_prefix;
  std::string_view remaining;
};

// Splits an error message after any leading context prefixes.
//
// Identifies leading context prefixes of the form "failed to ...: " and splits
// the message after the last prefix.
//
// Returns:
//   A ContextPrefixSplit, where:
//     - context_prefix: The last matched context prefix (e.g.,
//       "failed to materialize: "), or std::nullopt if no prefix is present.
//     - remaining: The substring of `message` after all context prefixes.
[[nodiscard]] ContextPrefixSplit SplitAfterContextPrefixes(
    const std::string_view message) {
  // Matches any string and splits it into two parts: zero or more leading
  // context prefixes of the form "failed to ...: ", and the remaining message.
  static const LazyRE2 kMaybeContextPrefixPattern = {
      R"(^(failed to .*?: )*(?s:(.*)))"};

  std::optional<std::string_view> context_prefix;
  std::string_view remaining;

  ABSL_CHECK(  // CRASH_OK=Full match should always work, since we can always
               // fallback to '.*'.
      RE2::FullMatch(message, *kMaybeContextPrefixPattern, &context_prefix,
                     &remaining))
      << "Could not match regex for detecting context prefixes '"
      << kMaybeContextPrefixPattern->pattern()
      << "' against message: " << message;

  return {context_prefix, remaining};
}

// Returns whether `message` starts with a context prefix.
[[nodiscard]] bool StartsWithContextPrefix(const std::string_view message) {
  return SplitAfterContextPrefixes(message).context_prefix.has_value();
}

// Removes all context prefixes from an error message.
//
// Strips all leading context prefixes of the form "failed to ...: " from the
// error message.
//
// Returns:
//   The substring of `message` after all leading context prefixes, or `message`
//   unchanged if no context prefixes are present.
[[nodiscard]] std::string_view RemoveAllContextPrefixes(
    const std::string_view message) {
  return SplitAfterContextPrefixes(message).remaining;
}

// Returns whether `message` contains an XLA compiler error by looking for the
// sentinel string `kXlaCompilerFailedWith`.
[[nodiscard]] bool HasXlaCompilerError(const std::string_view message) {
  return absl::StrContains(message, kXlaCompilerFailedWith);
}

// Strips the XLA compiler error sentinel and anything after it.
//
// Returns the substring before `kXlaCompilerFailedWith`, or the original
// message if the sentinel is not present.
[[nodiscard]] std::string_view RemoveXlaCompilerError(
    const std::string_view message) {
  const size_t pos = message.find(kXlaCompilerFailedWith);
  return pos != std::string_view::npos ? message.substr(0, pos) : message;
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

// Populates the given `ErrorMessageChecksResult` if the given `status` is an
// error status.
//
// Does nothing if `status` is not an error status.
// Otherwise, propagate its error by setting the `result` appropriately.
//
// Before appending the error status to the list of failures of `r`, we prepend:
//   - The Enforced Prefix: only if this check is being enforced.
//   - The given `context`: e.g. specifying the group this check belonged to.
//
// Returns whether `r` was modified or not, i.e. status is an error status.
bool PopulateResultIfError(ErrorMessageChecksResult& r,
                           const absl::Status& status, const CheckKind kind,
                           const std::string_view context = "") {
  constexpr std::string_view kEnforcedPrefix = "(enforced) ";

  if (!status.ok()) {
    const bool should_enforce_check = kind == CheckKind::kEnforce;

    // Given that this check has failed, set `success` depending on the value
    // of `should_enforce_check`:
    //
    //   - If it's `false` (i.e. should NOT be enforced), `success` should not
    //     change
    //
    //   - Otherwise, `success` should be false
    r.success = r.success && !should_enforce_check;

    const std::string_view enforced_prefix =
        should_enforce_check ? kEnforcedPrefix : "";

    r.failure_reasons.push_back(
        absl::StrCat(enforced_prefix, context, status.message()));
  }

  return !status.ok();
}

// Trims whitespace from the error message and checks whitespace guidelines.
//
// Performs the following checks and appends any violation status to `r`:
//   - Trimmed error message is not empty.
//   - Error message has no leading whitespace.
//   - Error message has no trailing whitespace.
//
// The leading and trailing whitespace checks are only run if the trimmed
// message is not empty.
//
// Returns the trimmed message if non-empty; otherwise returns `std::nullopt`.
[[nodiscard]] std::optional<std::string_view> TrimAndCheckMessage(
    ErrorMessageChecksResult& r, const std::string_view message,
    const std::string_view context = "") {
  const bool is_trimmed_error_empty = PopulateResultIfError(
      r, TrimmedErrorMessageIsNotEmpty(message), CheckKind::kEnforce, context);

  if (is_trimmed_error_empty) {
    return std::nullopt;
  }

  PopulateResultIfError(r, HasNoLeadingWhitespace(message), CheckKind::kEnforce,
                        context);
  PopulateResultIfError(r, HasNoTrailingWhitespace(message),
                        CheckKind::kEnforce, context);

  return absl::StripAsciiWhitespace(message);
}

// Runs checks on the entire error message.
//
// Pre-conditions:
//   - The error message should be trimmed (i.e. no surrounding whitespace).
//
// Populates `r` on any check failure.
void CheckFullMessage(ErrorMessageChecksResult& r,
                      const std::string_view message) {
  ABSL_CHECK(!HasSurroundingWhitespaces(message))  // CRASH_OK
      << kPreconditionViolatedError;

  PopulateResultIfError(r, EndsWithAlphaNumAndAllowedCharacters(message),
                        CheckKind::kEnforce);
  PopulateResultIfError(r, HasNoSpacesSurroundingSentinel(message),
                        CheckKind::kEnforce);
}

// Runs checks on the error message without any XLA compiler error.
//
// Pre-conditions:
//   - The error message should be trimmed (i.e. no surrounding whitespace).
//   - The XLA compiler error message has been removed.
//
// This function applies TorchTPU error guidelines on errors that we control.
// That is the reason for running them only after the XLA compiler error is
// removed.
//
// Populates `r` on any check failure.
void CheckMessageWithoutXlaCompilerError(ErrorMessageChecksResult& r,
                                         const std::string_view message,
                                         const std::string_view context) {
  ABSL_CHECK(!HasSurroundingWhitespaces(message))  // CRASH_OK
      << kPreconditionViolatedError;
  ABSL_CHECK(!HasXlaCompilerError(message))  // CRASH_OK
      << kPreconditionViolatedError;

  for (const auto& pair : type_name_pairs) {
    // Error messages coming from the OpenXLA project might contain StableHLO
    // type names, but we don't have control over it.
    PopulateResultIfError(r, DoesNotContainStableHloTypeName(message, pair),
                          CheckKind::kEnforce, context);
  }
}

// Checks the TorchTPU core error message formatting guidelines.
//
// Pre-conditions:
//   - The error message should be trimmed (no leading/trailing whitespace).
//   - The XLA compiler error message has been removed.
//   - Context prefixes ("failed to ...: ") should already be stripped.
//
// This function applies TorchTPU error guidelines on the core error message,
// after removing external errors and context prefixes.
//
// Populates `r` on any check failure.
void CheckTorchTpuCoreMessage(ErrorMessageChecksResult& r,
                              const std::string_view message,
                              const std::string_view context) {
  ABSL_CHECK(!HasSurroundingWhitespaces(message))  // CRASH_OK
      << kPreconditionViolatedError;
  ABSL_CHECK(!HasXlaCompilerError(message))  // CRASH_OK
      << kPreconditionViolatedError;
  ABSL_CHECK(!StartsWithContextPrefix(message))  // CRASH_OK
      << kPreconditionViolatedError;

  // Makes sure the error message after the context prefixes doesn't start with
  // an upper case letter.
  //
  // Example:
  //   failed to materialize: Expected tensor to be a 1D tensor, got 2D
  //                          |
  //                          |__ upper case letter not allowed.
  PopulateResultIfError(r, StartsWithNonUppercaseLetter(message),
                        CheckKind::kEnforce, context);
  PopulateResultIfError(r, DoesNotStartWithOpName(message), CheckKind::kEnforce,
                        context);

  // Only relevant on TorchTPU error messages.
  // - False positives might come from XLA compiler errors.
  // - Context prefixes should be removed so that we can enforce that it starts
  //   with `expected` if applicable.
  PopulateResultIfError(r, HasValidFormat(message), CheckKind::kEnforce,
                        context);
}

// Runs all checks, and returns their failure reasons.
//
// This function splits the error message checks into 3 groups, depending on
// their needed scope:
//   1. Full Error Message
//   2. No XLA Compiler Error Message: (1), but without the XLA compiler error.
//   3. TorchTPU Core Error Message: (2), but without the context prefixes.
//
// Returns the collected failures, or a default constructed
// `ErrorMessageChecksResult{}` if there are no check failures.
[[nodiscard]] ErrorMessageChecksResult GetErrorMessageChecksResult(
    const std::string_view message) {
  ErrorMessageChecksResult r;

  constexpr std::string_view kNoXlaErrorContext =
      "After removing the XLA compiler error: ";
  constexpr std::string_view kTorchTpuCoreContext =
      "On TorchTPU core error message: ";

  // Trimmed error message.
  //
  // It's reset at the start of each group, where the current error message gets
  // updated and trimmed with a call to `TrimAndCheckMessage()`.
  //
  // If `nullopt` gets assigned to variable `trimmed`, it means that the
  // processed error message is empty after trimming. Meaning that we should
  // stop running the checks.
  std::optional<std::string_view> trimmed;

  // Full Error Message
  // Following checks assume there are no leading or trailing whitespaces.
  trimmed = TrimAndCheckMessage(r, message);

  if (!trimmed.has_value()) {
    return r;
  }

  CheckFullMessage(r, *trimmed);

  // No XLA Compiler Error Message
  // Following checks assume that there is no XLA compiler error.
  trimmed = TrimAndCheckMessage(r, RemoveXlaCompilerError(*trimmed),
                                kNoXlaErrorContext);

  if (!trimmed.has_value()) {
    return r;
  }

  CheckMessageWithoutXlaCompilerError(r, *trimmed, kNoXlaErrorContext);

  // TorchTPU Core Error Message
  // Following checks assume that there are no context prefixes.
  trimmed = TrimAndCheckMessage(r, RemoveAllContextPrefixes(*trimmed),
                                kTorchTpuCoreContext);

  if (!trimmed.has_value()) {
    return r;
  }

  CheckTorchTpuCoreMessage(r, *trimmed, kTorchTpuCoreContext);

  return r;
}

// Formats a detailed log message explaining guideline violations.
[[nodiscard]] std::string GetLogMessage(
    const TtError& error, const std::vector<std::string>& failure_reasons) {
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

#endif  // TT_CHECKS_ERROR_FORMAT

}  // namespace

// Adapts an external error message (e.g., from PyTorch or OpenXLA) to conform
// to the TorchTPU error message guidelines (go/tt-error-guide).
//
// This function applies the following transformations:
// 1. Trims leading and trailing whitespace.
// 2. Lowercases the first character of the message (unless it starts with an
//    allowed starting word like "TorchTPU", "PjRt", or "XLA").
// 3. Removes trailing period characters.
std::string AdaptExternalErrorMessage(const std::string_view message) {
  // Check: trim leading and trailing whitespace.
  std::string new_message = std::string(absl::StripAsciiWhitespace(message));

  if (new_message.empty()) {
    return new_message;
  }

  // Check: error messages should start with a lowercase character.
  // We skip lowercasing if the message starts with specific allowed words.
  if (!StartsWithAllowedWord(new_message) &&
      absl::ascii_isupper(new_message[0])) {
    new_message[0] = absl::ascii_tolower(new_message[0]);
  }

  // Check: error messages should not end with punctuation (like periods).
  const size_t last_not_period_pos = new_message.find_last_not_of('.');
  if (last_not_period_pos != std::string::npos &&
      last_not_period_pos != new_message.size() - 1) {
    new_message = new_message.substr(0, last_not_period_pos + 1);
  }

  return new_message;
}

bool GetEnableDebugChecks() {
  const auto env_var = GetBooleanEnvOnce<kTorchTpuEnableDebugChecksEnvVar>();
  // If the env var is set, respect the user's choice.
  if (env_var.has_value()) {
    return *env_var;
  }
  // Otherwise, enable debug checks if we are in debug eager mode.
  return GetEagerMode() == EagerMode::kDeferNeverAndLaunchBlocking;
}

absl::StatusOr<int64_t> SafeAdd(const int64_t x, const int64_t y) {
  if (y == 0) return x;

  if (y > 0) {
    const int64_t upper_bound = std::numeric_limits<int64_t>::max() - y;
    TT_RET_CHECK(x <= upper_bound, error::kInvalidArgument)
        << "sum of " << x << " and " << y << " overflows as int64";
  } else {  // y < 0
    const int64_t lower_bound = std::numeric_limits<int64_t>::min() - y;
    TT_RET_CHECK(x >= lower_bound, error::kInvalidArgument)
        << "sum of " << x << " and " << y << " underflows as int64";
  }

  return x + y;
}

absl::StatusOr<int64_t> SafeMultiply(const int64_t x, const int64_t y) {
  const auto product_i128 = absl::int128(x) * y;  // Guaranteed to not overflow.
  const auto product_i64 = static_cast<int64_t>(product_i128);
  TT_RET_CHECK(product_i64 == product_i128, error::kInvalidArgument)
      << "product of " << x << " and " << y << " overflows as int64";
  return product_i64;
}

absl::StatusOr<uint64_t> SafeMultiply(const uint64_t x, const uint64_t y) {
  const auto product_u128 =
      absl::uint128(x) * y;  // Guaranteed to not overflow.
  const auto product_u64 = static_cast<uint64_t>(product_u128);
  TT_RET_CHECK(product_u64 == product_u128, error::kInvalidArgument)
      << "product of " << x << " and " << y << " overflows as uint64";
  return product_u64;
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
  // These `TT_RET_CHECK()` calls mirror `maybe_wrap_dim()` checks. They are
  // needed so that we can emit error messages that follow the TorchTPU
  // guidelines.

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Guaranteed by the origin of the value.
      dim_bound >= 0, error::kIndexError)
      << "expected the dimension upper bound to be >= 0, got " << dim_bound;

  if (dim_bound == 0) {
    // 0-D tensor, i.e. scalar.
    //
    // Internally, `maybe_wrap_dim()` calls itself, but setting the upper
    // bound to 1. Therefore, in this case, `dim` should be either 0 or -1.
    TT_RET_CHECK(dim == 0 || dim == -1, error::kIndexError)
        << "expected dimension to be either 0 or -1 for a shape with 0 "
           "dimensions, got "
        << dim;
  } else {
    TT_RET_CHECK(dim >= -dim_bound && dim < dim_bound, error::kIndexError)
        << "expected dimension to be in range [" << -dim_bound << ", "
        << dim_bound - 1 << "], got " << dim;
  }

  try {
    return at::maybe_wrap_dim(  // MAYBE_WRAP_DIM_OK=implementing SafeWrapDim.
        dim, dim_bound);
  } catch (const c10::Error& e) {
    return TT_ERROR(error::kIndexError)
           << AdaptExternalErrorMessage(e.what_without_backtrace());
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

namespace internal {
// Aligned to 64 bytes (cache-line size) to prevent false sharing with other
// frequently mutated global variables on the hot dispatch path.
alignas(64) std::atomic<bool> g_has_sticky_error{false};
}  // namespace internal

namespace {
absl::Mutex g_sticky_error_mutex(absl::kConstInit);
absl::NoDestructor<absl::Status> g_sticky_error
    ABSL_GUARDED_BY(g_sticky_error_mutex){absl::OkStatus()};

absl::Mutex g_assertion_mutex(absl::kConstInit);
absl::CondVar g_assertion_cv;
int64_t g_pending_assertions ABSL_GUARDED_BY(g_assertion_mutex) = 0;
}  // namespace

void SetStickyError(absl::Status status) {
  if (status.ok()) {
    return;
  }
  absl::MutexLock lock(g_sticky_error_mutex);
  if (g_sticky_error->ok()) {
    *g_sticky_error = std::move(status);
    internal::g_has_sticky_error.store(true, std::memory_order_release);
  }
}

absl::Status GetStickyError() {
  absl::MutexLock lock(g_sticky_error_mutex);
  return *g_sticky_error;
}

void IncrementPendingAssertionChecks() {
  absl::MutexLock lock(g_assertion_mutex);
  ++g_pending_assertions;
}

void DecrementPendingAssertionChecks() {
  absl::MutexLock lock(g_assertion_mutex);
  ABSL_CHECK_GT(g_pending_assertions, 0);  // CRASH_OK
  --g_pending_assertions;
  if (g_pending_assertions == 0) {
    g_assertion_cv.SignalAll();
  }
}

void WaitForPendingAssertionChecks() {
  absl::MutexLock lock(g_assertion_mutex);
  while (g_pending_assertions > 0) {
    g_assertion_cv.Wait(&g_assertion_mutex);
  }
}

void SyncAndCheckStickyError() {
  WaitForPendingAssertionChecks();
  if (ABSL_PREDICT_FALSE(HasStickyError())) {
    TT_THROW_IF_ERROR(GetStickyError());
  }
}

void ClearStickyError() {
  WaitForPendingAssertionChecks();
  absl::MutexLock lock(g_sticky_error_mutex);
  *g_sticky_error = absl::OkStatus();
  internal::g_has_sticky_error.store(false, std::memory_order_release);
}

absl::Status AdaptXlaError(const absl::Status& status,
                           const std::string_view context) {
  if (status.ok()) {
    return status;
  }

  const std::string adapted = AdaptExternalErrorMessage(status.message());
  return StatusBuilder(std::move(status)).SetOverride()
         << absl::StrCat(context, kXlaCompilerFailedWith, adapted);
}

namespace {

// Returns true if `message` indicates a VFIO collision. If a device node is
// present, writes it to `*device_path`, otherwise clears it. `device_path`
// must not be null.
bool MatchVfioCollisionMessage(std::string_view message,
                               std::string* device_path) {
  // Matches libtpu error messages and extracts the optional device path.
  // /dev/accel* is supported for future TPU generations.
  static const LazyRE2 kVfioCollisionRe = {
      R"(Couldn't open (?:iommu group|vfio container))"
      R"((?: (/dev/(?:vfio|accel)[^\s:]+))?)"};
  return RE2::PartialMatch(message, *kVfioCollisionRe, device_path);
}

// Returns true if `status` is a VFIO device-collision error, i.e. libtpu
// couldn't open a device node because another process holds it.
bool IsVfioDeviceCollisionError(const absl::Status& status) {
  std::string unused_device_path;  // Match requires a non-null destination.
  return !status.ok() &&
         MatchVfioCollisionMessage(status.message(), &unused_device_path);
}

// Upper bound on the process cmdline included in error messages.
constexpr size_t kMaxProcessNameLen = 128;

// Returns the command line for `pid` read from <proc_root>/<pid>/cmdline, or
// an empty string if unreadable.
std::string ReadProcessName(std::string_view proc_root, int pid) {
  std::ifstream cmdline_file(absl::StrCat(proc_root, "/", pid, "/cmdline"),
                             std::ios::binary);
  std::string cmdline((std::istreambuf_iterator<char>(cmdline_file)),
                      std::istreambuf_iterator<char>());
  // cmdline arguments are separated by null bytes ('\0').
  std::replace(cmdline.begin(), cmdline.end(), '\0', ' ');
  return std::string(
      absl::StripAsciiWhitespace(cmdline).substr(0, kMaxProcessNameLen));
}

// Resolves symlinks in `path`, returning `path` unchanged on failure.
// /proc/<pid>/fd symlinks point to canonical paths, so `path` must be
// resolved before comparison.
std::string CanonicalizeDevicePath(std::string_view path) {
  const std::string path_str(path);
  char resolved[PATH_MAX];
  if (realpath(path_str.c_str(), resolved) != nullptr) {
    return std::string(resolved);
  }
  return path_str;
}

// Extracts the device node path (e.g. "/dev/vfio/0") from `message`, or
// std::nullopt if none is found.
std::optional<std::string> ExtractVfioDevicePath(std::string_view message) {
  std::string device_path;
  if (MatchVfioCollisionMessage(message, &device_path) &&
      !device_path.empty()) {
    return device_path;
  }
  return std::nullopt;
}

// Information about a process holding a device node open.
struct DeviceHolder {
  int pid = 0;
  std::string name;
};

// Returns processes under `proc_root` holding `device_path` open.
std::vector<DeviceHolder> FindProcessesHoldingDevice(
    std::string_view device_path, std::string_view proc_root) {
  std::vector<DeviceHolder> holders;
  const std::string compare_path = CanonicalizeDevicePath(device_path);
  const int self_pid = getpid();

  std::unique_ptr<DIR, decltype(&closedir)> proc_dir(
      opendir(std::string(proc_root).c_str()), &closedir);
  if (proc_dir == nullptr) {
    return holders;
  }

  while (const dirent* proc_entry = readdir(proc_dir.get())) {
    int pid = 0;
    if (!absl::SimpleAtoi(proc_entry->d_name, &pid) || pid == self_pid) {
      continue;
    }
    const std::string fd_dir = absl::StrCat(proc_root, "/", pid, "/fd");
    std::unique_ptr<DIR, decltype(&closedir)> fds(opendir(fd_dir.c_str()),
                                                  &closedir);
    if (fds == nullptr) {
      continue;  // Skip processes we lack permission to inspect.
    }
    while (const dirent* fd_entry = readdir(fds.get())) {
      const std::string fd_path = absl::StrCat(fd_dir, "/", fd_entry->d_name);
      char link_target[PATH_MAX];
      const ssize_t len =
          readlink(fd_path.c_str(), link_target, sizeof(link_target) - 1);
      if (len <= 0) {
        continue;
      }
      if (std::string_view(link_target, len) == compare_path) {
        holders.push_back({pid, ReadProcessName(proc_root, pid)});
        break;
      }
    }
  }
  return holders;
}

// Formats the user-facing device-collision error message with colliding
// processes and remediation steps.
std::string FormatVfioCollisionMessage(
    std::optional<std::string_view> device_path,
    absl::Span<const DeviceHolder> holders) {
  std::string msg = "[TorchTPU] Failed to acquire a TPU device node";
  if (device_path.has_value()) {
    absl::StrAppend(&msg, " '", *device_path, "'");
  }
  absl::StrAppend(&msg,
                  " because it is already opened by another process on this "
                  "host.");
  if (!holders.empty()) {
    absl::StrAppend(&msg, "\n\nColliding process(es):\n");
    for (const DeviceHolder& holder : holders) {
      absl::StrAppend(&msg, "  - PID ", holder.pid);
      if (!holder.name.empty()) {
        absl::StrAppend(&msg, " (", holder.name, ")");
      }
      absl::StrAppend(&msg, "\n");
    }
  }
  absl::SubstituteAndAppend(
      &msg,
      "\n\nRemediation:\nWhen running multiple processes on a single TPU host, "
      "each process must be allocated a distinct device. Please set the $0 "
      "environment variable for each process, e.g.:\n  Process 0: export "
      "$0=0\n  Process 1: export $0=1",
      kTpuVisibleDevicesEnvVar);
  return msg;
}

}  // namespace

absl::Status internal::AdaptVfioDeviceCollisionError(
    absl::Status status, std::string_view proc_root) {
  if (!IsVfioDeviceCollisionError(status)) {
    return status;
  }
  const std::optional<std::string> device_path =
      ExtractVfioDevicePath(status.message());
  std::vector<DeviceHolder> holders;
  if (device_path.has_value()) {
    holders = FindProcessesHoldingDevice(*device_path, proc_root);
  }
  return StatusBuilder(std::move(status)).SetOverride()
         << FormatVfioCollisionMessage(device_path, holders);
}

absl::Status AdaptVfioDeviceCollisionError(absl::Status status) {
  return internal::AdaptVfioDeviceCollisionError(std::move(status), "/proc");
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
#if TT_CHECKS_ERROR_FORMAT
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
    case error::kPythonIndexError:
      throw  // Translates to Python IndexError.
          c10::IndexError(e.thrown_from(), message);
    case error::kPythonNotImplementedError:
      throw  // Translates to Python NotImplementedError.
          c10::NotImplementedError(e.thrown_from(), message);
    default:
      throw  // Translates to Python RuntimeError.
          c10::Error(e.thrown_from(), message);
  }
}

}  // namespace torch_tpu
