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

#include "csrc/common/status_test_utils.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(StatusTestUtilsTest, AutoAssign) {
  auto fn = []() -> absl::StatusOr<int> { return 42; };
  TT_ASSERT_OK_AND_ASSIGN(auto val, fn());
  EXPECT_EQ(val, 42);
}

TEST(StatusTestUtilsTest, ExplicitType) {
  auto fn = []() -> absl::StatusOr<int> { return 10; };
  TT_ASSERT_OK_AND_ASSIGN(int val, fn());
  EXPECT_EQ(val, 10);
}

TEST(StatusTestUtilsTest, ConstRef) {
  auto fn = []() -> absl::StatusOr<std::string> { return "hello"; };
  TT_ASSERT_OK_AND_ASSIGN(const std::string& val, fn());
  EXPECT_EQ(val, "hello");
}

TEST(StatusTestUtilsTest, ExistingVariable) {
  auto fn = []() -> absl::StatusOr<int> { return 99; };
  int val = 0;
  TT_ASSERT_OK_AND_ASSIGN(val, fn());
  EXPECT_EQ(val, 99);
}

TEST(StatusTestUtilsTest, MoveOnlyType) {
  auto fn = []() -> absl::StatusOr<std::unique_ptr<int>> {
    return std::make_unique<int>(123);
  };
  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<int> ptr, fn());
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(*ptr, 123);
}

TEST(StatusTestUtilsTest, MultipleAssignmentsInSameScope) {
  auto fn1 = []() -> absl::StatusOr<int> { return 1; };
  auto fn2 = []() -> absl::StatusOr<int> { return 2; };
  TT_ASSERT_OK_AND_ASSIGN(auto a, fn1());
  TT_ASSERT_OK_AND_ASSIGN(auto b, fn2());
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);
}

TEST(StatusTestUtilsTest, FailsOnNonOkStatus) {
  EXPECT_FATAL_FAILURE(
      {
        auto fn = []() -> absl::StatusOr<int> {
          return TT_ERROR(error::kInvalidArgument)
                 << "invalid argument provided";
        };
        TT_ASSERT_OK_AND_ASSIGN(auto val, fn());
        (void)val;
      },
      "invalid argument provided");
}

TEST(StatusTestUtilsTest, FailsWithStatusCode) {
  EXPECT_FATAL_FAILURE(
      {
        auto fn = []() -> absl::StatusOr<std::string> {
          return TT_ERROR(error::kNotFound) << "resource not found";
        };
        TT_ASSERT_OK_AND_ASSIGN(auto val, fn());
        (void)val;
      },
      "NOT_FOUND");
}

TEST(StatusTestUtilsTest, ExecutionStopsOnFailure) {
  // This must be static as code in EXPECT_FATAL_FAILURE cannot reference
  // non-static local variables.
  static bool executed_after_failure;
  // Reset the flag before the test, in case the test case is run
  // multiple times.
  executed_after_failure = false;
  EXPECT_FATAL_FAILURE(
      {
        auto fn = []() -> absl::StatusOr<int> {
          return TT_ERROR(error::kInternal) << "aborted computation";
        };
        TT_ASSERT_OK_AND_ASSIGN(auto val, fn());
        executed_after_failure = true;
        (void)val;
      },
      "aborted computation");
  EXPECT_FALSE(executed_after_failure);
}

}  // namespace
}  // namespace torch_tpu
