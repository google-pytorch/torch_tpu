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

#include "csrc/internal/mosaic/kiln/core/error.h"

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/core/builder_scope.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

namespace torch_tpu::kiln::internal {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

struct TestScope {
  mlir::MLIRContext mlir_ctx;
  mlir::ImplicitLocOpBuilder builder{mlir::UnknownLoc::get(&mlir_ctx),
                                     &mlir_ctx};
  BuilderScope scope{&builder};
};

TEST(ErrorTest, ReportErrorRecordsOnlyTheFirstError) {
  TestScope env;

  ReportError(TT_ERROR(error::kInvalidArgument) << "first error");
  ReportError(TT_ERROR(error::kInternal) << "second error");

  EXPECT_THAT(env.scope.error(),
              StatusIs(error::kInvalidArgument, HasSubstr("first error")));
}

int CheckWithValue(bool ok, int fallback, bool* fallback_evaluated) {
  TT_KILN_RET_CHECK(ok, error::kInvalidArgument,
                    (*fallback_evaluated = true, fallback))
      << "check failed";
  return 42;
}

void CheckVoid(bool ok, bool* continued) {
  TT_KILN_RET_CHECK(ok, error::kInvalidArgument) << "void check failed";
  *continued = true;
}

TEST(ErrorTest, CheckWithValuePassingDoesNotEvaluateFallbackOrReportError) {
  TestScope env;
  bool fallback_evaluated = false;

  EXPECT_EQ(CheckWithValue(true, -1, &fallback_evaluated), 42);
  EXPECT_FALSE(fallback_evaluated);
  EXPECT_THAT(env.scope.error(), IsOk());
}

TEST(ErrorTest, CheckWithValueFailingReportsErrorAndReturnsFallback) {
  TestScope env;
  bool fallback_evaluated = false;

  EXPECT_EQ(CheckWithValue(false, -1, &fallback_evaluated), -1);
  EXPECT_TRUE(fallback_evaluated);
  EXPECT_THAT(env.scope.error(),
              StatusIs(error::kInvalidArgument, HasSubstr("check failed")));
}

TEST(ErrorTest, CheckWithValueKeepsFirstError) {
  TestScope env;
  ReportError(TT_ERROR(error::kInternal) << "first error");
  bool fallback_evaluated = false;

  EXPECT_EQ(CheckWithValue(false, -1, &fallback_evaluated), -1);
  EXPECT_THAT(env.scope.error(),
              StatusIs(error::kInternal, HasSubstr("first error")));
}

int CheckCountingEvaluations(bool ok, int* cond_evaluations,
                             int* message_evaluations) {
  TT_KILN_RET_CHECK((++*cond_evaluations, ok), error::kInvalidArgument, -1)
      << (++*message_evaluations, "check failed");
  return 42;
}

TEST(ErrorTest, CheckPassingEvaluatesCondOnceAndDoesNotEvaluateMessage) {
  TestScope env;
  int cond_evaluations = 0;
  int message_evaluations = 0;

  EXPECT_EQ(
      CheckCountingEvaluations(true, &cond_evaluations, &message_evaluations),
      42);

  EXPECT_EQ(cond_evaluations, 1);
  EXPECT_EQ(message_evaluations, 0);
}

TEST(ErrorTest, CheckFailingEvaluatesCondAndMessageOnce) {
  TestScope env;
  int cond_evaluations = 0;
  int message_evaluations = 0;

  EXPECT_EQ(
      CheckCountingEvaluations(false, &cond_evaluations, &message_evaluations),
      -1);

  EXPECT_EQ(cond_evaluations, 1);
  EXPECT_EQ(message_evaluations, 1);
}

// A `result` that fails in turn, reporting an error of its own.
int ReportErrorAndReturnFallback() {
  ReportError(TT_ERROR(error::kInternal) << "fallback error");
  return -1;
}

int CheckWithReportingResult(bool ok) {
  TT_KILN_RET_CHECK(ok, error::kInvalidArgument, ReportErrorAndReturnFallback())
      << "check failed";
  return 42;
}

TEST(ErrorTest, CheckReportsItsErrorBeforeEvaluatingResult) {
  TestScope env;

  EXPECT_EQ(CheckWithReportingResult(false), -1);
  EXPECT_THAT(env.scope.error(),
              StatusIs(error::kInvalidArgument, HasSubstr("check failed")));
}

int CheckInUnbracedIfs(bool checked, bool ok) {
  // Deliberately unbraced: the macro must work as the whole body of an `if`,
  // and leave a following `else` to that `if`.
  if (checked)
    TT_KILN_RET_CHECK(ok, error::kInvalidArgument, -1) << "check failed";

  if (checked)
    TT_KILN_RET_CHECK(ok, error::kInvalidArgument, -1) << "check failed";
  else
    return 0;
  return 42;
}

TEST(ErrorTest, CheckWorksAsUnbracedIfBody) {
  TestScope env;

  EXPECT_EQ(CheckInUnbracedIfs(false, false), 0);
  EXPECT_EQ(CheckInUnbracedIfs(true, true), 42);
  EXPECT_EQ(CheckInUnbracedIfs(true, false), -1);
}

TEST(ErrorTest, CheckVoidReturnsOnlyOnFailure) {
  TestScope env;
  bool continued = false;

  CheckVoid(true, &continued);
  EXPECT_TRUE(continued);
  EXPECT_THAT(env.scope.error(), IsOk());

  continued = false;
  CheckVoid(false, &continued);
  EXPECT_FALSE(continued);
  EXPECT_THAT(env.scope.error(), StatusIs(error::kInvalidArgument,
                                          HasSubstr("void check failed")));
}

TEST(ErrorDeathTest, ReportErrorOutsideScopeDies) {
  EXPECT_DEATH(ReportError(TT_ERROR(error::kInternal) << "boom"),
               "Kiln operations may only be called inside BuildKernel");
}

TEST(ErrorDeathTest, ReportErrorWithOkStatusDies) {
  EXPECT_DEATH(
      {
        TestScope env;
        ReportError(absl::OkStatus());
      },
      "ReportError\\(\\) requires a non-OK status");
}

}  // namespace
}  // namespace torch_tpu::kiln::internal
