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

// Tests for error_utils.

#include "torch_tpu/common/error_utils.h"

#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/eager/eager_mode.h"

namespace torch_tpu {
namespace {

using testing::DescribeMatcher;
using testing::ElementsAre;
using testing::Matcher;
using testing::Pair;
using testing::Pointee;
using testing::Property;
using testing::Throws;

// Per
// https://docs.pytorch.org/docs/stable/debugging_environment_variables.html,
// this disables C++ context in pytorch errors. This can be done only once per
// process as pytorch caches the value of this variable.
static const auto kInitShowCppContext =
    setenv("TORCH_SHOW_CPP_STACKTRACES", "0", /*overwrite=*/1);

// Matches a callback that throws a TtError whose .what)
// matches the given message string matcher.
MATCHER_P(ThrowsTtError, msg_matcher,
          "throws a TtError with a message that (ignoring backtrace) " +
              DescribeMatcher<const std::string&>(msg_matcher)) {
  const Matcher<const std::string&> msg_matcher_string = msg_matcher;
  return ExplainMatchResult(
      Throws<TtError>(Property("message (ignoring backtrace)", &TtError::what,
                               msg_matcher_string)),
      arg, result_listener);
}

// Tests for TT_ERROR.

TEST(Error, CanStreamMessagesIntoError) {
  const absl::Status error = TT_ERROR(error::kInternal) << "message " << 42;
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "message 42");
}

TEST(Error, CanBeReturnedFromFunction) {
  const auto test = [&]() -> absl::Status {
    return TT_ERROR(error::kInvalidArgument)
           << "dimension size is negative: " << -42;
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInvalidArgument);
  EXPECT_EQ(error.message(), "dimension size is negative: -42");
}

TEST(ErrorDeathTest, CrashesIfEmptyMessageIsStreamed) {
  const auto test = []() -> absl::Status {
    return TT_ERROR(error::kInvalidArgument) << "";
  };
  EXPECT_DEATH(test().IgnoreError(), "empty.*message.*stream");
}

// Tests for TT_RET_CHECK.

TEST(RetCheck, DoesNothingOnTrue) {
  bool cond = true;
  const auto test = [&]() -> absl::Status {
    // This should NOT return.
    TT_RET_CHECK(cond, error::kInvalidArgument) << "cond is false";
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(RetCheck, ReturnsErrorOnFalse) {
  bool cond = false;
  const auto test = [&]() -> absl::Status {
    // This should return.
    TT_RET_CHECK(cond, error::kInvalidArgument) << "Unexpected cond: " << cond;
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInvalidArgument);
  EXPECT_EQ(error.message(), "Unexpected cond: 0");
}

TEST(RetCheck, EvaluatesConditionOnceOnSuccess) {
  int count = 1;
  const auto test = [&]() -> absl::Status {
    // This should not return.
    TT_RET_CHECK(++count, error::kInvalidArgument) << "Failed";
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
  EXPECT_EQ(count, 2);  // Incremented once.
}

TEST(RetCheck, EvaluatesConditionOnceOnFailure) {
  int count = 1;
  const auto test = [&]() -> absl::Status {
    // This should return.
    TT_RET_CHECK(++count < 0, error::kInvalidArgument) << "Failed";
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInvalidArgument);
  EXPECT_EQ(error.message(), "Failed");
  EXPECT_EQ(count, 2);  // Incremented once.
}

TEST(RetCheck, EvaluatesMessageOnceOnFailure) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    // This should return.
    TT_RET_CHECK(false, error::kInvalidArgument)
        << "Failure count: " << ++count;
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInvalidArgument);
  EXPECT_EQ(error.message(), "Failure count: 1");
  EXPECT_EQ(count, 1);  // Incremented once.
}

TEST(RetCheck, SkipsEvaluatingMessageOnSuccess) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    // This should not return.
    TT_RET_CHECK(true, error::kInvalidArgument) << "Failure count: " << ++count;
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
  EXPECT_EQ(count, 0);  // Never incremented.
}

TEST(RetCheck, BehavesLikeSingleStatement) {
  int count = 0;
  int else_count = 0;
  const auto test = [&]() -> absl::Status {
    if (true)
      // If TT_RET_CHECK is expanded to multiple C++ statements, the
      // following will fail to compile as the `else` will have no matching
      // `if`:
      TT_RET_CHECK(++count, error::kInvalidArgument) << "cond is false";
    else             // Do nothing.
      ++else_count;  // NOLINT - intended for testing macro syntax.
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
  EXPECT_EQ(count, 1);  // Incremented once.
  EXPECT_EQ(else_count, 0);
}

TEST(RetCheckTest, DoesNotCrashIfEmptyMessageOnSuccess) {
  const auto test = []() -> absl::Status {
    // When the condition is true, no Status is created, so the check for
    // non-empty message is not triggered.
    TT_RET_CHECK(true, error::kInvalidArgument) << "";
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(RetCheckDeathTest, CrashesIfEmptyMessageOnFailure) {
  const auto test = []() -> absl::Status {
    TT_RET_CHECK(false, error::kInvalidArgument) << "";
    return absl::OkStatus();
  };
  EXPECT_DEATH(test().IgnoreError(), "empty.*message.*stream");
}

// Tests for TT_ASSIGN_OR_RETURN.

TEST(AssignOrReturn, AssignsToExistingVariableOnOk) {
  const auto test = [&]() -> absl::Status {
    int y = 0;
    absl::StatusOr<int> x = 1;
    TT_ASSIGN_OR_RETURN(y, x);
    EXPECT_EQ(y, 1);
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(AssignOrReturn, AssignsToNewVariableOnOk) {
  const auto test = [&]() -> absl::Status {
    absl::StatusOr<int> x = 1;
    TT_ASSIGN_OR_RETURN(int y, x);
    EXPECT_EQ(y, 1);
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(AssignOrReturn, AssignsFromConstToExistingVariableOnOk) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = 1;
    int y = 0;
    TT_ASSIGN_OR_RETURN(y, x);
    EXPECT_EQ(y, 1);
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(AssignOrReturn, AssignsFromConstToNewVariableOnOk) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = 1;
    TT_ASSIGN_OR_RETURN(int y, x);
    EXPECT_EQ(y, 1);
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(AssignOrReturn, AssignsToNewConstVariableOnOk) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = 1;
    TT_ASSIGN_OR_RETURN(const int y, x);
    EXPECT_EQ(y, 1);
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(AssignOrReturn, AssignsToExistingMoveOnlyVariableOnOk) {
  const auto test = [&]() -> absl::Status {
    absl::StatusOr<std::unique_ptr<int>> x = std::make_unique<int>(1);
    std::unique_ptr<int> y;
    TT_ASSIGN_OR_RETURN(y, std::move(x));
    EXPECT_THAT(y, Pointee(1));
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(AssignOrReturn, AssignsToNewMoveOnlyVariableOnOk) {
  const auto test = [&]() -> absl::Status {
    absl::StatusOr<std::unique_ptr<int>> x = std::make_unique<int>(1);
    TT_ASSIGN_OR_RETURN(std::unique_ptr<int> y, std::move(x));
    EXPECT_THAT(y, Pointee(1));
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(AssignOrReturn, ReturnsErrorOnFailure) {
  const auto test = [&]() -> absl::Status {
    absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_RETURN(int y, x);
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error");
}

TEST(AssignOrReturn, AcceptsParenthesizedLhs) {
  const auto test = [&]() -> absl::Status {
    const std::map<int, int> map = {{1, 2}, {3, 4}};
    const absl::StatusOr<std::map<int, int>> x = map;
    // The type std::map<int, int> contains a comma not protected by
    // parentheses. Therefore we must put the lhs in parentheses to satisfy the
    // preprocessor.
    TT_ASSIGN_OR_RETURN((const std::map<int, int> y), x);
    EXPECT_THAT(y, ElementsAre(Pair(1, 2), Pair(3, 4)));
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(AssignOrReturn, AcceptsErrorExpression) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_RETURN(int y, x, _);
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error");
}

TEST(AssignOrReturn, ErrorExpressionDoesNotHaveToReference_) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_RETURN(int y, x, TT_ERROR(error::kInternal) << "your error");
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "your error");
}

TEST(AssignOrReturn, AcceptsErrorExpressionWithAppend) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_RETURN(int y, x, _ << " has more info: " << 42);
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error has more info: 42");
}

TEST(AssignOrReturn, CanAppendEmptyMessage) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_RETURN(int y, x, _ << "");
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error");
}

TEST(AssignOrReturn, AcceptsErrorExpressionWithPrepend) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_RETURN(int y, x,
                        _.SetPrepend() << "Failed " << 42 << " times with: ");
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "Failed 42 times with: my error");
}

TEST(AssignOrReturn, CanPrependEmptyMessage) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_RETURN(int y, x, _.SetPrepend() << "");
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error");
}

TEST(AssignOrReturn, AcceptsErrorExpressionWithOverride) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_RETURN(int y, x,
                        _.SetOverride() << "Failed " << 42 << " times");
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "Failed 42 times");
}

TEST(AssignOrReturnDeathTest, CrashesIfOverriddenWithEmptyMessage) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_RETURN(int y, x, _.SetOverride() << "");
    static_cast<void>(y);
    return absl::OkStatus();
  };
  EXPECT_DEATH(test().IgnoreError(), "empty.*message.*stream");
}

TEST(AssignOrReturn, SkipsEvaluatingErrorExpressionOnOk) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    TT_ASSIGN_OR_RETURN(int y, absl::StatusOr<int>(1),
                        _ << " - with count: " << ++count);
    static_cast<void>(y);
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
  EXPECT_EQ(count, 0);
}

TEST(AssignOrReturn, EvaluatesErrorExpressionOnceOnError) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    TT_ASSIGN_OR_RETURN(
        int y, absl::StatusOr<int>(absl::Status(error::kInternal, "my error")),
        _ << " with count: " << ++count);
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error with count: 1");
  EXPECT_EQ(count, 1);  // Incremented once.
}

TEST(AssignOrReturn, CanAssignToStructuredBindings) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<std::pair<int, int>> x = std::make_pair(1, 2);
    TT_ASSIGN_OR_RETURN((auto [first, second]), x);
    EXPECT_EQ(first, 1);
    EXPECT_EQ(second, 2);
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
}

TEST(AssignOrReturn, EvaluatesExpressionOnceOnOk) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = 42;
    TT_ASSIGN_OR_RETURN(const int y, [&] {
      ++count;
      return x;
    }());
    EXPECT_EQ(y, 42);
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
  EXPECT_EQ(count, 1);  // Incremented once.
}

TEST(AssignOrReturn, EvaluatesExpressionOnceOnError) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = TT_ERROR(error::kInternal) << "my error";
    TT_ASSIGN_OR_RETURN(int y, [&] {
      ++count;
      return x;
    }());
    static_cast<void>(y);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error");
  EXPECT_EQ(count, 1);  // Incremented once.
}

// Tests for TT_ASSIGN_OR_THROW.

TEST(AssignOrThrow, AssignsToExistingVariableOnOk) {
  absl::StatusOr<int> x = 1;
  int y = 0;
  TT_ASSIGN_OR_THROW(y, x);
  EXPECT_EQ(y, 1);
}

TEST(AssignOrThrow, AssignsToNewVariableOnOk) {
  absl::StatusOr<int> x = 1;
  TT_ASSIGN_OR_THROW(int y, x);
  EXPECT_EQ(y, 1);
}

TEST(AssignOrThrow, AssignsFromConstToExistingVariableOnOk) {
  const absl::StatusOr<int> x = 1;
  int y = 0;
  TT_ASSIGN_OR_THROW(y, x);
  EXPECT_EQ(y, 1);
}

TEST(AssignOrThrow, AssignsFromConstToNewVariableOnOk) {
  const absl::StatusOr<int> x = 1;
  TT_ASSIGN_OR_THROW(int y, x);
  EXPECT_EQ(y, 1);  // NOLINT
}

TEST(AssignOrThrow, AssignsToNewConstVariableOnOk) {
  const absl::StatusOr<int> x = 1;
  TT_ASSIGN_OR_THROW(const int y, x);
  EXPECT_EQ(y, 1);  // NOLINT
}

TEST(AssignOrThrow, AssignsToExistingMoveOnlyVariableOnOk) {
  absl::StatusOr<std::unique_ptr<int>> x = std::make_unique<int>(1);
  std::unique_ptr<int> y;
  TT_ASSIGN_OR_THROW(y, std::move(x));
  ASSERT_NE(y, nullptr);
  EXPECT_EQ(*y, 1);
}

TEST(AssignOrThrow, AssignsToNewMoveOnlyVariableOnOk) {
  absl::StatusOr<std::unique_ptr<int>> x = std::make_unique<int>(1);
  TT_ASSIGN_OR_THROW(std::unique_ptr<int> y, std::move(x));
  ASSERT_NE(y, nullptr);
  EXPECT_EQ(*y, 1);
}

void AssignOrThrowError() {
  absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
  TT_ASSIGN_OR_THROW(int y, x);
  static_cast<void>(y);
}

TEST(AssignOrThrow, ThrowsRuntimeErrorOnError) {
  EXPECT_THAT(AssignOrThrowError, ThrowsTtError("my error"));
}

TEST(AssignOrThrow, AcceptsParenthesizedLhs) {
  const std::map<int, int> map = {{1, 2}, {3, 4}};
  const absl::StatusOr<std::map<int, int>> x = map;
  // The type std::map<int, int> contains a comma not protected by parentheses.
  // Therefore we must put the lhs in parentheses to satisfy the preprocessor.
  TT_ASSIGN_OR_THROW((const std::map<int, int> y), x);
  EXPECT_THAT(y, ElementsAre(Pair(1, 2), Pair(3, 4)));
}

void AssignOrThrowErrorWithExpression() {
  const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
  TT_ASSIGN_OR_THROW(int y, x, _);
  static_cast<void>(y);
}

TEST(AssignOrThrow, AcceptsErrorExpression) {
  EXPECT_THAT(AssignOrThrowErrorWithExpression, ThrowsTtError("my error"));
}

void AssignOrThrowExpressionNo_() {
  const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
  // This error expression doesn't reference the _ variable.
  TT_ASSIGN_OR_THROW(int y, x, TT_ERROR(error::kInternal) << "your error");
  static_cast<void>(y);
}

TEST(AssignOrThrow, ErrorExpressionDoesNotHaveToReference_) {
  EXPECT_THAT(AssignOrThrowExpressionNo_, ThrowsTtError("your error"));
}

void AssignOrThrowErrorWithAppend() {
  const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
  TT_ASSIGN_OR_THROW(int y, x, _ << " has more info: " << 42);
  static_cast<void>(y);
}

TEST(AssignOrThrow, AcceptsErrorExpressionWithAppend) {
  EXPECT_THAT(AssignOrThrowErrorWithAppend,
              ThrowsTtError("my error has more info: 42"));
}

void AssignOrThrowErrorWithPrepend() {
  const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
  TT_ASSIGN_OR_THROW(int y, x,
                     _.SetPrepend() << "Failed " << 42 << " times with: ");
  static_cast<void>(y);
}

TEST(AssignOrThrow, AcceptsErrorExpressionWithPrepend) {
  EXPECT_THAT(AssignOrThrowErrorWithPrepend,
              ThrowsTtError("Failed 42 times with: my error"));
}

void AssignOrThrowErrorWithOverride() {
  const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
  TT_ASSIGN_OR_THROW(int y, x, _.SetOverride() << "Failed " << 42 << " times");
  static_cast<void>(y);
}

TEST(AssignOrThrow, AcceptsErrorExpressionWithOverride) {
  EXPECT_THAT(AssignOrThrowErrorWithOverride, ThrowsTtError("Failed 42 times"));
}

void AssignOrThrowWithOk(int& count) {
  TT_ASSIGN_OR_THROW(int y, absl::StatusOr<int>(1),
                     _ << " - with count: " << ++count);
  static_cast<void>(y);
}

TEST(AssignOrThrow, SkipsEvaluatingErrorExpressionOnOk) {
  int count = 0;
  const auto test = [&]() { AssignOrThrowWithOk(count); };
  test();
  EXPECT_EQ(count, 0);
}

void AssignOrThrowWithError(int& count) {
  TT_ASSIGN_OR_THROW(
      int y, absl::StatusOr<int>(absl::Status(error::kInternal, "my error")),
      _ << " with count: " << ++count);
  static_cast<void>(y);
}

TEST(AssignOrThrow, EvaluatesErrorExpressionOnceOnError) {
  int count = 0;
  const auto test = [&]() { AssignOrThrowWithError(count); };
  EXPECT_THAT(test, ThrowsTtError("my error with count: 1"));
  EXPECT_EQ(count, 1);  // Incremented once.
}

TEST(AssignOrThrow, CanAssignToStructuredBindings) {
  const absl::StatusOr<std::pair<int, int>> x = std::make_pair(1, 2);
  TT_ASSIGN_OR_THROW((auto [first, second]), x);
  EXPECT_EQ(first, 1);
  EXPECT_EQ(second, 2);
}

TEST(AssignOrThrow, EvaluatesExpressionOnceOnOk) {
  const absl::StatusOr<int> x = 42;
  int count = 0;
  TT_ASSIGN_OR_THROW(const int y, [&] {
    ++count;
    return x;
  }());
  EXPECT_EQ(y, 42);
  EXPECT_EQ(count, 1);  // Incremented once.
}

void AssignOrThrowExpression(int& count) {
  const absl::StatusOr<int> x = TT_ERROR(error::kInternal) << "my error";
  TT_ASSIGN_OR_THROW((const int y), [&] {
    ++count;
    return x;
  }());
  static_cast<void>(y);
}

TEST(AssignOrThrow, EvaluatesExpressionOnceOnError) {
  int count = 0;
  const auto test = [&] { AssignOrThrowExpression(count); };
  EXPECT_THAT(test, ThrowsTtError("my error"));
  EXPECT_EQ(count, 1);  // Incremented once.
}

// Tests for TT_ASSIGN_OR_CRASH.

TEST(AssignOrCrash, AssignsToExistingVariableOnOk) {
  absl::StatusOr<int> x = 1;
  int y = 0;
  TT_ASSIGN_OR_CRASH(y, x);
  EXPECT_EQ(y, 1);
}

TEST(AssignOrCrash, AssignsToNewVariableOnOk) {
  absl::StatusOr<int> x = 1;
  TT_ASSIGN_OR_CRASH(int y, x);
  EXPECT_EQ(y, 1);
}

TEST(AssignOrCrash, AssignsFromConstToExistingVariableOnOk) {
  const absl::StatusOr<int> x = 1;
  int y = 0;
  TT_ASSIGN_OR_CRASH(y, x);
  EXPECT_EQ(y, 1);
}

TEST(AssignOrCrash, AssignsFromConstToNewVariableOnOk) {
  const absl::StatusOr<int> x = 1;
  TT_ASSIGN_OR_CRASH(int y, x);
  EXPECT_EQ(y, 1);
}

TEST(AssignOrCrash, AssignsToNewConstVariableOnOk) {
  const absl::StatusOr<int> x = 1;
  TT_ASSIGN_OR_CRASH(const int y, x);
  EXPECT_EQ(y, 1);
}

TEST(AssignOrCrash, AssignsToExistingMoveOnlyVariableOnOk) {
  absl::StatusOr<std::unique_ptr<int>> x = std::make_unique<int>(1);
  std::unique_ptr<int> y;
  TT_ASSIGN_OR_CRASH(y, std::move(x));
  ASSERT_NE(y, nullptr);
  EXPECT_EQ(*y, 1);
}

TEST(AssignOrCrash, AssignsToNewMoveOnlyVariableOnOk) {
  absl::StatusOr<std::unique_ptr<int>> x = std::make_unique<int>(1);
  TT_ASSIGN_OR_CRASH(std::unique_ptr<int> y, std::move(x));
  ASSERT_NE(y, nullptr);
  EXPECT_EQ(*y, 1);
}

TEST(AssignOrCrashDeathTest, CrashesOnError) {
  const auto test = [] {
    absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_CRASH(int y, x);
    static_cast<void>(y);
  };
  EXPECT_DEATH(test(), "my error");
}

TEST(AssignOrCrash, AcceptsParenthesizedLhs) {
  const std::map<int, int> map = {{1, 2}, {3, 4}};
  const absl::StatusOr<std::map<int, int>> x = map;
  TT_ASSIGN_OR_CRASH((const std::map<int, int> y), x);
  EXPECT_THAT(y, ElementsAre(Pair(1, 2), Pair(3, 4)));
}

TEST(AssignOrCrashDeathTest, AcceptsErrorExpression) {
  const auto test = [] {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_CRASH(int y, x, _);
    static_cast<void>(y);
  };
  EXPECT_DEATH(test(), "my error");
}

TEST(AssignOrCrashDeathTest, ErrorExpressionDoesNotHaveToReference_) {
  const auto test = [] {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_CRASH(int y, x, TT_ERROR(error::kInternal) << "your error");
    static_cast<void>(y);
  };
  EXPECT_DEATH(test(), "your error");
}

TEST(AssignOrCrashDeathTest, AcceptsErrorExpressionWithAppend) {
  const auto test = [] {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_CRASH(int y, x, _ << " has more info: " << 42);
    static_cast<void>(y);
  };
  EXPECT_DEATH(test(), "my error has more info: 42");
}

TEST(AssignOrCrashDeathTest, AcceptsErrorExpressionWithPrepend) {
  const auto test = [] {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_CRASH(int y, x,
                       _.SetPrepend() << "Failed " << 42 << " times with: ");
    static_cast<void>(y);
  };
  EXPECT_DEATH(test(), "Failed 42 times with: my error");
}

TEST(AssignOrCrashDeathTest, AcceptsErrorExpressionWithOverride) {
  const auto test = [] {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    TT_ASSIGN_OR_CRASH(int y, x,
                       _.SetOverride() << "Failed " << 42 << " times");
    static_cast<void>(y);
  };
  EXPECT_DEATH(test(), "Failed 42 times");
}

TEST(AssignOrCrash, SkipsEvaluatingErrorExpressionOnOk) {
  int count = 0;
  const auto test = [&] {
    TT_ASSIGN_OR_CRASH(int y, absl::StatusOr<int>(1),
                       _ << " - with count: " << ++count);
    static_cast<void>(y);
  };
  test();
  EXPECT_EQ(count, 0);
}

TEST(AssignOrCrashDeathTest, EvaluatesErrorExpressionOnceOnError) {
  int count = 0;
  const auto test = [&] {
    TT_ASSIGN_OR_CRASH(
        int y, absl::StatusOr<int>(absl::Status(error::kInternal, "my error")),
        _ << " with count: " << ++count);
    static_cast<void>(y);
  };
  EXPECT_DEATH(test(), "my error with count: 1");
}

TEST(AssignOrCrash, CanAssignToStructuredBindings) {
  const absl::StatusOr<std::pair<int, int>> x = std::make_pair(1, 2);
  TT_ASSIGN_OR_CRASH((auto [first, second]), x);
  EXPECT_EQ(first, 1);
  EXPECT_EQ(second, 2);
}

TEST(AssignOrCrash, EvaluatesExpressionOnceOnOk) {
  const absl::StatusOr<int> x = 42;
  int count = 0;
  TT_ASSIGN_OR_CRASH(const int y, [&] {
    ++count;
    return x;
  }());
  EXPECT_EQ(y, 42);
  EXPECT_EQ(count, 1);  // Incremented once.
}

TEST(AssignOrCrashDeathTest, EvaluatesExpressionOnceOnError) {
  int count = 0;
  const auto test = [&] {
    const absl::StatusOr<int> x = TT_ERROR(error::kInternal) << "my error";
    TT_ASSIGN_OR_CRASH((const int y), [&] {
      ++count;
      return x;
    }());
    static_cast<void>(y);
  };
  EXPECT_DEATH(test(), "my error");
}

// Tests for TT_RETURN_IF_ERROR.

TEST(ReturnIfError, ProceedsOnOkStatus) {
  bool early_return = true;
  const auto test = [&]() -> absl::Status {
    // This should NOT return.
    TT_RETURN_IF_ERROR(absl::OkStatus());
    early_return = false;
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
  EXPECT_FALSE(early_return);
}

TEST(ReturnIfError, ProceedsOnOkStatusOr) {
  bool early_return = true;
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = 1;
    // This should NOT return.
    TT_RETURN_IF_ERROR(x);
    early_return = false;
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
  EXPECT_FALSE(early_return);
}

TEST(ReturnIfError, ReturnsOnErrorStatus) {
  const auto test = [&]() -> absl::Status {
    // This should return.
    TT_RETURN_IF_ERROR(absl::Status(error::kInternal, "my error"));
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error");
}

TEST(ReturnIfError, ReturnsOnErrorStatusOr) {
  const auto test = [&]() -> absl::Status {
    const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
    // This should return.
    TT_RETURN_IF_ERROR(x);
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error");
}

TEST(ReturnIfError, BehavesLikeSingleStatement) {
  const auto test = [&]() -> absl::Status {
    if (true)
      // If TT_RETURN_IF_ERROR is expanded to multiple C++ statements, the
      // following will fail to compile as the `else` will have no matching
      // `if`:
      TT_RETURN_IF_ERROR(absl::Status(error::kInternal, "my error"));
    else  // Do nothing.
      ;   // NOLINT - intended for testing macro syntax.
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error");
}

TEST(ReturnIfError, CanAppendMoreInfo) {
  const auto test = [&]() -> absl::Status {
    TT_RETURN_IF_ERROR(absl::Status(error::kInternal, "my error"))
        << " has more info: " << 42;
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error has more info: 42");
}

TEST(ReturnIfError, CanPrependMoreInfo) {
  const auto test = [&]() -> absl::Status {
    TT_RETURN_IF_ERROR(absl::Status(error::kInternal, "my error")).SetPrepend()
        << "Failed " << 42 << " times with: ";
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "Failed 42 times with: my error");
}

TEST(ReturnIfError, CanOverrideMessage) {
  const auto test = [&]() -> absl::Status {
    TT_RETURN_IF_ERROR(absl::Status(error::kInternal, "my error")).SetOverride()
        << "Failed " << 42 << " times";
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "Failed 42 times");
}

TEST(ReturnIfError, EvaluatesStatusOnceOnOk) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    // This should NOT return.
    TT_RETURN_IF_ERROR([&] {
      ++count;
      return absl::OkStatus();
    }()) << "error";
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
  EXPECT_EQ(count, 1);  // Incremented once.
}

TEST(ReturnIfError, EvaluatesStatusOnceOnError) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    // This should return.
    TT_RETURN_IF_ERROR([&] {
      ++count;
      return absl::Status(error::kInternal, "my error");
    }()) << ": unexpected";
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error: unexpected");
  EXPECT_EQ(count, 1);  // Incremented once.
}

TEST(ReturnIfError, SkipsEvaluatingMessageOnOk) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    // This should NOT return.
    TT_RETURN_IF_ERROR(absl::OkStatus()) << ": unexpected " << ++count;
    return absl::OkStatus();
  };
  EXPECT_EQ(test(), absl::OkStatus());
  EXPECT_EQ(count, 0);  // Not incremented.
}

TEST(ReturnIfError, EvaluatesMessageOnceOnError) {
  int count = 0;
  const auto test = [&]() -> absl::Status {
    // This should return.
    TT_RETURN_IF_ERROR(absl::Status(error::kInternal, "my error"))
        << ": unexpected " << ++count;
    return absl::OkStatus();
  };
  const absl::Status error = test();
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "my error: unexpected 1");
  EXPECT_EQ(count, 1);  // Incremented once.
}

TEST(ErrorDeathTest, GetEnableDebugChecksEnabledByDefaultInDebugMode) {
  EXPECT_EXIT(
      {
        SetEagerMode(EagerMode::kDeferNeverAndLaunchBlocking);
        exit(GetEnableDebugChecks());
      },
      testing::ExitedWithCode(1), "");
}

TEST(GetEnableDebugChecks, IsDisabledInDefaultEagerMode) {
  SetEagerMode(EagerMode::kDeferNever);
  EXPECT_FALSE(GetEnableDebugChecks());
}

TEST(GetEnableDebugChecksDeathTest, CanBeEnabledByEnvVar) {
  EXPECT_EXIT(
      {
        SetEnv(kTorchTpuInternalEnableDebugChecksEnvVar, "1");
        exit(GetEnableDebugChecks());
      },
      testing::ExitedWithCode(1), "");
}

TEST(GetEnableDebugChecksDeathTest, CanBeDisabledByEnvVar) {
  EXPECT_EXIT(
      {
        SetEnv(kTorchTpuInternalEnableDebugChecksEnvVar, "0");
        exit(GetEnableDebugChecks());
      },
      testing::ExitedWithCode(0), "");
}

TEST(GetEnableDebugChecksDeathTest, CanBeEnabledByEnvVarInDebugMode) {
  EXPECT_EXIT(
      {
        SetEagerMode(EagerMode::kDeferNeverAndLaunchBlocking);
        SetEnv(kTorchTpuInternalEnableDebugChecksEnvVar, "1");
        exit(GetEnableDebugChecks());
      },
      testing::ExitedWithCode(1), "");
}

TEST(GetEnableDebugChecksDeathTest, CanBeDisabledByEnvVarInDebugMode) {
  EXPECT_EXIT(
      {
        SetEagerMode(EagerMode::kDeferNeverAndLaunchBlocking);
        SetEnv(kTorchTpuInternalEnableDebugChecksEnvVar, "0");
        exit(GetEnableDebugChecks());
      },
      testing::ExitedWithCode(0), "");
}

TEST(GetEnableDebugChecks, DependsOnCurrentEagerMode) {
  SetEagerMode(EagerMode::kDeferNever);
  EXPECT_FALSE(GetEnableDebugChecks());
  SetEagerMode(EagerMode::kDeferNeverAndLaunchBlocking);
  EXPECT_TRUE(GetEnableDebugChecks());
}

// Tests for TT_THROW_IF_ERROR.

TEST(ThrowIfError, ProceedsOnOkStatus) {
  // This should NOT throw.
  TT_THROW_IF_ERROR(absl::OkStatus());
}

TEST(ThrowIfError, ProceedsOnOkStatusOr) {
  // This should NOT throw.
  TT_THROW_IF_ERROR(absl::StatusOr<int>(1));
}

void ThrowIfStatus() {
  TT_THROW_IF_ERROR(absl::Status(error::kInternal, "my error"));
}

TEST(ThrowIfError, ThrowsExceptionOnErrorStatus) {
  EXPECT_THAT(ThrowIfStatus, ThrowsTtError("my error"));
}

int ThrowIfStatusInNonVoidFunction() {
  TT_THROW_IF_ERROR(absl::Status(error::kInternal, "my error"));
  return 0;
}

TEST(ThrowIfError, ThrowsExceptionOnErrorStatusInNonVoidFunction) {
  EXPECT_THAT(ThrowIfStatusInNonVoidFunction, ThrowsTtError("my error"));
}

void ThrowIfStatusOr() {
  const absl::StatusOr<int> x = absl::Status(error::kInternal, "my error");
  // This should throw.
  TT_THROW_IF_ERROR(x);
}

TEST(ThrowIfError, ThrowsExceptionOnErrorStatusOr) {
  EXPECT_THAT(ThrowIfStatusOr, ThrowsTtError("my error"));
}

void ThrowIfStatusLvalueAppend() {
  absl::Status status = absl::Status(error::kInternal, "my error");
  TT_THROW_IF_ERROR(status) << " has more info: " << 42;
}

TEST(ThrowIfError, CanAppendMoreInfoLvalue) {
  EXPECT_THAT(ThrowIfStatusLvalueAppend,
              ThrowsTtError("my error has more info: 42"));
}

void ThrowIfStatusRvalueAppend() {
  TT_THROW_IF_ERROR(absl::Status(error::kInternal, "my error"))
      << " has more info: " << 42;
}

TEST(ThrowIfError, CanAppendMoreInfo) {
  EXPECT_THAT(ThrowIfStatusRvalueAppend,
              ThrowsTtError("my error has more info: 42"));
}

void ThrowIfStatusPrepend() {
  TT_THROW_IF_ERROR(absl::Status(error::kInternal, "my error")).SetPrepend()
      << "Failed " << 42 << " times with: ";
}

TEST(ThrowIfError, CanPrependMoreInfo) {
  EXPECT_THAT(ThrowIfStatusPrepend,
              ThrowsTtError("Failed 42 times with: my error"));
}

void ThrowIfStatusOverride() {
  TT_THROW_IF_ERROR(absl::Status(error::kInternal, "my error")).SetOverride()
      << "Failed " << 42 << " times";
}

TEST(ThrowIfError, CanOverrideMessage) {
  EXPECT_THAT(ThrowIfStatusOverride, ThrowsTtError("Failed 42 times"));
}

void ThrowIfErrorBehavesLikeSingleStatement() {
  if (true)
    // If TT_THROW_IF_ERROR is expanded to multiple C++ statements, the
    // following will fail to compile as the `else` will have no matching
    // `if`:
    TT_THROW_IF_ERROR(absl::Status(error::kInternal, "my error"));
  else  // Do nothing.
    ;   // NOLINT - intended for testing macro syntax.
}

TEST(ThrowIfError, BehavesLikeSingleStatement) {
  EXPECT_THAT(ThrowIfErrorBehavesLikeSingleStatement,
              ThrowsTtError("my error"));
}

void ThrowIfErrorEvaluatesStatusOnceOnOk(int& count) {
  // This should NOT throw.
  TT_THROW_IF_ERROR([&] {
    ++count;
    return absl::OkStatus();
  }()) << "error";
}

TEST(ThrowIfError, EvaluatesStatusOnceOnOk) {
  int count = 0;
  ThrowIfErrorEvaluatesStatusOnceOnOk(count);
  EXPECT_EQ(count, 1);  // Incremented once.
}

void ThrowIfErrorEvaluatesStatusOnceOnError(int& count) {
  // This should throw.
  TT_THROW_IF_ERROR([&] {
    ++count;
    return absl::Status(error::kInternal, "my error");
  }()) << ": unexpected";
}

TEST(ThrowIfError, EvaluatesStatusOnceOnError) {
  int count = 0;
  const auto test = [&]() { ThrowIfErrorEvaluatesStatusOnceOnError(count); };
  EXPECT_THAT(test, ThrowsTtError("my error: unexpected"));
  EXPECT_EQ(count, 1);  // Incremented once.
}

void ThrowIfErrorSkipsEvaluatingMessageOnOk(int& count) {
  // This should NOT throw.
  TT_THROW_IF_ERROR(absl::OkStatus()) << ": unexpected " << ++count;
}

TEST(ThrowIfError, SkipsEvaluatingMessageOnOk) {
  int count = 0;
  ThrowIfErrorSkipsEvaluatingMessageOnOk(count);
  EXPECT_EQ(count, 0);  // Not incremented.
}

void ThrowIfErrorMessageEvaluatesOnceOnError(int& count) {
  // This should throw.
  TT_THROW_IF_ERROR(absl::Status(error::kInternal, "my error"))
      << ": unexpected " << ++count;
}

TEST(ThrowIfError, EvaluatesMessageOnceOnError) {
  int count = 0;
  const auto test = [&]() { ThrowIfErrorMessageEvaluatesOnceOnError(count); };
  EXPECT_THAT(test, ThrowsTtError("my error: unexpected 1"));
  EXPECT_EQ(count, 1);  // Incremented once.
}

// Tests for TT_CHECK_THROW.

TEST(CheckThrow, ProceedsOnTrue) {
  // This should NOT throw.
  try {
    TT_CHECK_THROW(true, error::kInternal) << "my error";
  } catch (...) {
    FAIL() << "TT_CHECK_THROW(true, ...) should not raise an exception.";
  }
}

void CheckThrowFalse() {
  TT_CHECK_THROW(false, error::kInternal) << "my error";
}

TEST(CheckThrow, ThrowsExceptionOnFalse) {
  EXPECT_THAT(CheckThrowFalse, ThrowsTtError("my error"));
}

TEST(CheckThrow, EvaluatesConditionOnceButNotErrorCodeOrMessageOnTrue) {
  int cond_count = 0, code_count = 0, message_count = 0;
  // This should NOT throw.
  TT_CHECK_THROW(++cond_count, (++code_count, error::kInternal))
      << (++message_count, "my error");
  EXPECT_EQ(cond_count, 1);     // Incremented once.
  EXPECT_EQ(code_count, 0);     // Not incremented.
  EXPECT_EQ(message_count, 0);  // Not incremented.
}

void CheckThrowEvaluateCondition(int& cond_count, int& code_count,
                                 int& message_count) {
  TT_CHECK_THROW(++cond_count < 0, (++code_count, error::kInternal))
      << (++message_count, "my error");
}

TEST(CheckThrow, EvaluatesConditionErrorCodeMessageOnceOnFalse) {
  int cond_count = 0, code_count = 0, message_count = 0;
  const auto test = [&]() {
    CheckThrowEvaluateCondition(cond_count, code_count, message_count);
  };
  EXPECT_THAT(test, ThrowsTtError("my error"));
  EXPECT_EQ(cond_count, 1);     // Incremented once.
  EXPECT_EQ(code_count, 1);     // Incremented once.
  EXPECT_EQ(message_count, 1);  // Incremented once.
}

// Tests for SafeWrapDim.
TEST(SafeWrapDim, ReturnsErrorOnInvalidNegativeDim) {
  const auto result = SafeWrapDim(/*dim=*/-2, /*dim_bound=*/1);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kOutOfRange);
  EXPECT_EQ(result.status().message(),
            "Dimension out of range (expected to be in range of [-1, 0], but "
            "got -2)");
}

TEST(SafeWrapDim, ReturnsErrorOnInvalidPositiveDim) {
  const auto result = SafeWrapDim(/*dim=*/1, /*dim_bound=*/1);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kOutOfRange);
  EXPECT_EQ(result.status().message(),
            "Dimension out of range (expected to be in range of [-1, 0], but "
            "got 1)");
}

TEST(SafeWrapDim, ReturnsErrorOnNegativeDimBound) {
  // Negative dim_bound is not supported by SafeWrapDim.
  const auto result = SafeWrapDim(/*dim=*/0, /*dim_bound=*/-1);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kOutOfRange);
  EXPECT_EQ(result.status().message(), "Rank cannot be negative but got -1");
}

TEST(SafeWrapDim, ReturnsErrorOnInvalidDimForZeroDimBound) {
  const auto result = SafeWrapDim(/*dim=*/1, /*dim_bound=*/0);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), error::kOutOfRange);
  EXPECT_EQ(result.status().message(),
            "Dimension out of range (expected to be in range of [-1, 0], but "
            "got 1)");
}

TEST(SafeWrapDim, WrapsValidNegativeDim) {
  const auto result = SafeWrapDim(/*dim=*/-1, /*dim_bound=*/3);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, 2);
}

TEST(SafeWrapDim, ReturnsValidPositiveDim) {
  const auto result = SafeWrapDim(/*dim=*/1, /*dim_bound=*/3);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, 1);
}

TEST(SafeWrapDim, ReturnsZeroOnValidDimForZeroDimBound) {
  // For 0-dimensional tensors, only dims 0 and -1 are valid.
  const auto result0 = SafeWrapDim(/*dim=*/0, /*dim_bound=*/0);
  ASSERT_TRUE(result0.ok());
  EXPECT_EQ(*result0, 0);

  const auto result1 = SafeWrapDim(/*dim=*/-1, /*dim_bound=*/0);
  ASSERT_TRUE(result1.ok());
  EXPECT_EQ(*result1, 0);
}

}  // namespace
}  // namespace torch_tpu
