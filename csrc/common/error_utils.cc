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
#include "c10/core/DeviceGuard.h"
#include "c10/core/WrapDimMinimal.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for IntArrayRef
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "c10/util/StringUtil.h"
#include "csrc/common/device_type.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/status_builder.h"
#include "csrc/common/utils.h"
#include "csrc/eager/eager_mode.h"
#include "re2/re2.h"

namespace torch_tpu {
namespace {

// Returns whether `message` starts with one of the words that are allowed to
// start an error message with an uppercase letter. This is an exception to the
// rule that error messages must start with a lowercase letter.
//
// The allowed word must be matched as a whole word. For example, "XLA" matches
// "XLA compiler" and "XLA's", but not "XLAB".
[[nodiscard]] bool StartsWithAllowedUppercaseWord(
    const std::string_view message) {
  static const LazyRE2 kAllowedUppercaseStartingWordPattern = {
      R"(^(TorchTPU|PjRt|XLA)\b)"};
  return RE2::PartialMatch(message, *kAllowedUppercaseStartingWordPattern);
}

// The same dtype has different names in PyTorch and StableHLO. This struct
// stores the pair of names for a particular dtype.
struct DTypeNames {
  std::string_view torch;
  std::string_view stablehlo;
};

// Maps between PyTorch and StableHLO dtype names.
//
// The order doesn't affect the checks: each StableHLO type name is matched as a
// whole word (see `ValidateNoStableHloTypeName()`), so e.g. "f16" doesn't
// match "bf16".
constexpr std::array<DTypeNames, 19> kDtypeNames{{
    // go/keep-sorted start
    {.torch = "bfloat16", .stablehlo = "bf16"},
    {.torch = "bool", .stablehlo = "pred"},
    {.torch = "complex128", .stablehlo = "complexf64"},
    {.torch = "complex64", .stablehlo = "complexf32"},
    {.torch = "float16", .stablehlo = "f16"},
    {.torch = "float32", .stablehlo = "f32"},
    {.torch = "float64", .stablehlo = "f64"},
    {.torch = "int16", .stablehlo = "i16"},
    {.torch = "int2", .stablehlo = "i2"},
    {.torch = "int32", .stablehlo = "i32"},
    {.torch = "int4", .stablehlo = "i4"},
    {.torch = "int64", .stablehlo = "i64"},
    {.torch = "int8", .stablehlo = "i8"},
    {.torch = "uint16", .stablehlo = "ui16"},
    {.torch = "uint2", .stablehlo = "ui2"},
    {.torch = "uint32", .stablehlo = "ui32"},
    {.torch = "uint4", .stablehlo = "ui4"},
    {.torch = "uint64", .stablehlo = "ui64"},
    {.torch = "uint8", .stablehlo = "ui8"},
    // go/keep-sorted end
}};

// Default message for error message checks' pre-condition violations.
constexpr std::string_view kErrorMessageCheckPreconditionViolated =
    "Error message check pre-condition violated. This indicates a bug in "
    "TorchTPU's error message format checker.";

// The Validate*() functions below each check one error message guideline.
// They return a description of the violation if `message` doesn't follow the
// guideline, or std::nullopt if it does.

// Validates that the message is neither empty nor just whitespace.
std::optional<std::string> ValidateNotBlank(const std::string_view message) {
  if (absl::StripAsciiWhitespace(message).empty()) {
    return "The error message must not be whitespace only.";
  }
  return std::nullopt;
}

// Validates that the message has no leading whitespace.
// Pre-conditions: The message must not be empty.
std::optional<std::string> ValidateNoLeadingWhitespace(
    const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kErrorMessageCheckPreconditionViolated;

  const size_t num_leading_whitespace =
      message.size() - absl::StripLeadingAsciiWhitespace(message).size();
  if (num_leading_whitespace != 0) {
    return absl::StrCat(
        "The error message must not start with whitespace. Remove the ",
        num_leading_whitespace, " leading whitespace character(s).");
  }
  return std::nullopt;
}

// Validates that the message has no trailing whitespace.
// Pre-conditions: The message must not be empty.
std::optional<std::string> ValidateNoTrailingWhitespace(
    const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kErrorMessageCheckPreconditionViolated;

  const size_t num_trailing_whitespace =
      message.size() - absl::StripTrailingAsciiWhitespace(message).size();
  if (num_trailing_whitespace != 0) {
    return absl::StrCat(
        "The error message must not end with whitespace. Remove the "
        "trailing ",
        num_trailing_whitespace, " whitespace character(s).");
  }
  return std::nullopt;
}

// Validates that the message doesn't start with an uppercase letter.
// Pre-conditions:
//   - The error message must not be empty.
//   - The error message must not start with whitespace.
//
// The first character of `message` must be lower-case. We intentionally allow
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
std::optional<std::string> ValidateNoUppercaseStart(
    const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kErrorMessageCheckPreconditionViolated;
  ABSL_CHECK(  // CRASH_OK
      !absl::ascii_isspace(message.front()))
      << kErrorMessageCheckPreconditionViolated;

  if (StartsWithAllowedUppercaseWord(message)) {
    return std::nullopt;
  }

  if (absl::ascii_isupper(message.front())) {
    return "The error message must start with a lowercase character.";
  }
  return std::nullopt;
}

// Validates that the message ends with an alphanumeric or allowed character.
//
// Pre-conditions:
//   - The error message must not be empty.
//   - The error message must not end with whitespace.
//
// The last character of `message` must be either alphanumeric or other
// allowed characters (see kAllowedCharacters).
std::optional<std::string> ValidateLastChar(const std::string_view message) {
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
      << kErrorMessageCheckPreconditionViolated;
  ABSL_CHECK(  // CRASH_OK
      !absl::ascii_isspace(message.back()))
      << kErrorMessageCheckPreconditionViolated;

  const char last_char = message.back();
  if (absl::ascii_isalnum(last_char) ||
      absl::StrContains(kAllowedCharacters, last_char)) {
    return std::nullopt;
  }

  return absl::StrCat(
      "The error message must end with either an alpha-numeric character "
      "or one of the following characters: ",
      kAllowedCharacters);
}

// Validates that, if the message states the expected condition or the actual
// value, it's in the "expected ..., got ..." format.
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
std::optional<std::string> ValidateExpectedGotFormat(
    const std::string_view message) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kErrorMessageCheckPreconditionViolated;

  static LazyRE2 kExpectedConditionOrActualValuePattern = {
      R"((.*\bexpected\b.*|.*[.,]\s*\b(?i:got|but got)\b.*))"};
  static LazyRE2 kExpectedGotPattern = {R"(^expected\b.*, got\b.*)"};

  if (RE2::FullMatch(message, *kExpectedConditionOrActualValuePattern) &&
      !RE2::FullMatch(message, *kExpectedGotPattern)) {
    return "The error message states an expected condition or the actual "
           "value (i.e. contains \"expected\", or \", (but) got\" "
           "strings). Therefore, it must be in the following format: "
           "\"expected ..., got ...\".";
  }
  return std::nullopt;
}

// Validates that the message shows the given dtype by its official PyTorch
// name, as opposed to its StableHLO/XLA type name:
//     - "float32" vs. "f32"
//     - "int32" vs. "si32"
//     - "int64" vs. "si64"
//     - "bool" vs. "pred"
//     - "bfloat16" vs. "bf16".
//
// Pre-conditions: The error message must not be empty.
std::optional<std::string> ValidateNoStableHloTypeName(
    const std::string_view message, const DTypeNames& pair) {
  ABSL_CHECK(  // CRASH_OK
      !message.empty())
      << kErrorMessageCheckPreconditionViolated;

  const std::string type_name_regex =
      absl::StrCat("\\b", pair.stablehlo, "\\b");

  if (RE2::PartialMatch(message, type_name_regex)) {
    return absl::StrCat(
        "The error message must use the PyTorch dtype names instead of "
        "the StableHLO type names, e.g. it must show '",
        pair.torch, "' instead of '", pair.stablehlo,
        "'. Use ToString(mlir::ElementType) for retrieving the PyTorch "
        "dtype name.");
  }
  return std::nullopt;
}

// Validates that the XLA compiler error sentinel is preceded
// by a non-empty context and has no surrounding spaces.
//
// Pre-condition: The error message must contain the sentinel.
std::optional<std::string> ValidateXlaSentinelFormat(
    const std::string_view message) {
  const size_t sentinel_start = message.find(kXlaCompilerFailedWith);
  ABSL_CHECK(  // CRASH_OK
      sentinel_start != std::string_view::npos)
      << kErrorMessageCheckPreconditionViolated;

  const std::string_view context = message.substr(0, sentinel_start);
  const size_t sentinel_end = sentinel_start + kXlaCompilerFailedWith.size();
  const std::string_view xla_error = message.substr(sentinel_end);

  if (absl::StripAsciiWhitespace(context).empty()) {
    return "The TorchTPU context must not be empty when using the XLA error "
           "sentinel string. Make sure to add a context string before.";
  }

  // We already know that `context` contains at least one non-space character
  // (from the previous check). Hence the find will succeed.
  const size_t leading_spaces_start = context.find_last_not_of(' ') + 1;
  const size_t leading_spaces_count = context.size() - leading_spaces_start;
  if (leading_spaces_count != 0) {
    return absl::StrCat(
        "Expected no spaces before the XLA compiler error sentinel \"",
        kXlaCompilerFailedWith, "\" (it already begins with one), got ",
        leading_spaces_count, ".");
  }

  const size_t trailing_spaces_count = xla_error.find_first_not_of(' ');
  if (trailing_spaces_count != 0) {
    return absl::StrCat(
        "Expected no spaces after the XLA compiler error sentinel \"",
        kXlaCompilerFailedWith, "\" (it already ends with one), got ",
        trailing_spaces_count, ".");
  }

  return std::nullopt;
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

// Strips the XLA compiler error sentinel and anything after it.
//
// Returns the substring before `kXlaCompilerFailedWith`, or the original
// message if the sentinel is not present.
[[nodiscard]] std::string_view RemoveXlaCompilerError(
    const std::string_view message) {
  const size_t pos = message.find(kXlaCompilerFailedWith);
  return pos != std::string_view::npos ? message.substr(0, pos) : message;
}

// Checks that error messages follow the TorchTPU error handling guidelines
// (go/tt-error-guide).
//
// The checks are split into 3 groups, depending on their needed scope:
//   1. Full Error Message
//   2. No XLA Compiler Error Message: (1), but without the XLA compiler error.
//   3. TorchTPU Core Error Message: (2), but without the context prefixes.
//
// Example:
//   ErrorFormatChecker().Check(error);
class ErrorFormatChecker {
 public:
  // Checks that the message of `error` follows the guidelines. Crashes with an
  // explanation of all violations if it doesn't.
  void Check(const TtError& error) {
    violations_.clear();
    CheckMessage(error.status().message());
    if (!violations_.empty()) {
      ABSL_LOG(FATAL)  // CRASH_OK
          << ExplainViolations(error);
    }
  }

 private:
  static constexpr std::string_view kFullMessageContext = "";
  static constexpr std::string_view kNoXlaErrorContext =
      "After removing the XLA compiler error: ";
  static constexpr std::string_view kNoContextPrefixContext =
      "After removing all context prefixes: ";
  static constexpr std::string_view kNoContextPrefixAndXlaErrorContext =
      "After removing all context prefixes and the XLA compiler error: ";

  // Runs all checks on `message`, and records the violations found.
  void CheckMessage(const std::string_view message) {
    // The part of the message that is being checked.
    std::string_view to_check = message;
    context_ = kFullMessageContext;

    // Does this message contain an XLA compiler error?
    const bool has_xla_error =
        absl::StrContains(to_check, kXlaCompilerFailedWith);
    if (has_xla_error) {
      // Yes. First check that the XLA error sentinel is formatted correctly.
      AppendIfViolation(ValidateXlaSentinelFormat(to_check));
      // Remove the XLA compiler error.
      to_check = RemoveXlaCompilerError(to_check);
      context_ = kNoXlaErrorContext;
    }

    // StableHLO type names are not allowed anywhere in the TorchTPU part of the
    // message, including the context prefixes. Hence we check them before
    // removing the context prefixes. (Error messages coming from the OpenXLA
    // project might contain StableHLO type names, but we don't have control
    // over it.)
    CheckNoStableHloTypeName(to_check);

    const std::string_view no_context_prefixes =
        RemoveAllContextPrefixes(to_check);
    if (no_context_prefixes != to_check) {
      to_check = no_context_prefixes;
      context_ = has_xla_error ? kNoContextPrefixAndXlaErrorContext
                               : kNoContextPrefixContext;
    }

    CheckCoreMessage(to_check);
  }

  // Validates that `message` doesn't contain any StableHLO type name.
  void CheckNoStableHloTypeName(const std::string_view message) {
    if (message.empty()) {
      return;
    }
    for (const auto& pair : kDtypeNames) {
      AppendIfViolation(ValidateNoStableHloTypeName(message, pair));
    }
  }

  // Checks the whitespace guidelines on the error message:
  //   - Error message is not blank (i.e. empty or whitespace only).
  //   - Error message has no leading whitespace.
  //   - Error message has no trailing whitespace.
  //
  // The leading and trailing whitespace checks are only run if the message is
  // not blank.
  void CheckWhitespaceFormat(const std::string_view message) {
    if (AppendIfViolation(ValidateNotBlank(message))) {
      return;
    }

    AppendIfViolation(ValidateNoLeadingWhitespace(message));
    AppendIfViolation(ValidateNoTrailingWhitespace(message));
  }

  // Checks the TorchTPU core error message formatting guidelines.
  //
  // Pre-conditions:
  //   - The XLA compiler error message has been removed.
  //   - Context prefixes ("failed to ...: ") must already be stripped.
  //
  // This function applies TorchTPU error guidelines on the core error message,
  // after removing external errors and context prefixes.
  void CheckCoreMessage(const std::string_view message) {
    ABSL_CHECK(!absl::StrContains(message, kXlaCompilerFailedWith))  // CRASH_OK
        << kErrorMessageCheckPreconditionViolated;
    ABSL_CHECK(!StartsWithContextPrefix(message))  // CRASH_OK
        << kErrorMessageCheckPreconditionViolated;
    CheckWhitespaceFormat(message);

    // The checks below require a message without surrounding whitespace.
    // Whitespace violations (if any) have been recorded above.
    const std::string_view trimmed = absl::StripAsciiWhitespace(message);
    if (trimmed.empty()) {
      return;
    }

    // Makes sure the error message after the context prefixes doesn't start
    // with an upper case letter.
    //
    // Example:
    //   failed to materialize: Expected tensor to be a 1D tensor, got 2D
    //                          |
    //                          |__ upper case letter not allowed.
    AppendIfViolation(ValidateNoUppercaseStart(trimmed));

    AppendIfViolation(ValidateLastChar(trimmed));

    // Only relevant on TorchTPU error messages.
    // - False positives might come from XLA compiler errors.
    // - Context prefixes must be removed so that we can enforce that it
    //   starts with `expected` if applicable.
    AppendIfViolation(ValidateExpectedGotFormat(trimmed));
  }

  // If `violation` is set, records it, prefixed by the current context.
  //
  // Returns true if `violation` is set.
  bool AppendIfViolation(const std::optional<std::string>& violation) {
    if (!violation.has_value()) return false;
    violations_.push_back(absl::StrCat(context_, *violation));
    return true;
  }

  // Formats a detailed log message explaining the recorded violations.
  [[nodiscard]] std::string ExplainViolations(const TtError& error) const {
    const auto& source = error.thrown_from();
    return absl::StrCat(
        source.file, ":", source.line,
        ": Bad error message format in function '", source.function,
        "' (please follow go/tt-error-guide).\n\n"  //
        "Actual error message:\n",
        "  '", error.status().message(),
        "'\n\n"  //
        "What's wrong with it:\n",
        absl::StrJoin(violations_, /* separator= */ "\n",
                      /* fmt= */
                      [](std::string* out, const std::string& reason) {
                        absl::StrAppend(out, "  - ", reason);
                      }));
  }

  // Describes which part of the error message is being checked. Prepended to
  // the recorded violations.
  std::string_view context_ = kFullMessageContext;

  // The violations found when checking the error message against the
  // guidelines. If empty, the error message conforms to the guidelines.
  std::vector<std::string> violations_;
};

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
  if (!StartsWithAllowedUppercaseWord(new_message) &&
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
  if (device_opt->has_index() &&
      c10::impl::hasDeviceGuardImpl(GetPrivateUse1DeviceType())) {
    c10::DeviceGuard device_guard(*device_opt);
  }
}

bool IsXlaOomError(const absl::Status& status) {
  return absl::IsResourceExhausted(status) ||
         (absl::IsInternal(status) &&
          // Some XLA OOM errors are reported as internal errors with this
          // word in the message.
          absl::StrContains(status.message(), "allocation_size"));
}

absl::Status TranslateXlaTensorOomError(absl::Status status,
                                        std::string_view dtype,
                                        absl::Span<const int64_t> dims) {
  if (status.ok()) {
    return status;
  }

  TT_RET_CHECK(!IsXlaOomError(status), error::kResourceExhausted)
      << "the TPU ran out of memory while allocating tensor " << dtype << "["
      << absl::StrJoin(dims, ", ") << "]";

  // From this point, we know that the given status is not an XLA OOM error.
  return status;
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

absl::Status SyncAndReturnStickyError() {
  WaitForPendingAssertionChecks();
  if (ABSL_PREDICT_FALSE(HasStickyError())) {
    return GetStickyError();
  }
  return absl::OkStatus();
}

void SyncAndCheckStickyError() {
  TT_THROW_IF_ERROR(SyncAndReturnStickyError());
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
  if constexpr (kCheckErrorFormat) {
    // Check that the error message format follows the TorchTPU guidelines
    // (go/tt-error-guide).
    ErrorFormatChecker().Check(e);
  }

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
    case error::kResourceExhausted:
      throw  // Translates to Python torch.OutOfMemoryError.
          c10::OutOfMemoryError(e.thrown_from(), message);
    default:
      // Fallback: unhandled error codes translate to c10::Error (Python
      // RuntimeError). If a status code corresponds to a dedicated PyTorch
      // exception type (e.g. c10::OutOfMemoryError, c10::IndexError, or
      // c10::ValueError), add an explicit case above instead of relying on
      // this default fallback.
      throw  // Translates to Python RuntimeError.
          c10::Error(e.thrown_from(), message);
  }
}

}  // namespace torch_tpu
