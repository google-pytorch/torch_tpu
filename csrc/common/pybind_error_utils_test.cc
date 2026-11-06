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

#include "csrc/common/pybind_error_utils.h"

#include <exception>
#include <string>
#include <string_view>
#include <typeinfo>

#include "c10/util/Exception.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::HasSubstr;

template <class ErrorType, class FuncType>
void TestHelperRaises(const FuncType& func, const std::string_view name,
                      const std::string_view error_message_substr) {
  try {
    internal::ErrorHandlingHelper<void()>::impl(name, func);
    FAIL() << "Expected exception was not thrown";
  } catch (const ErrorType& e) {
    EXPECT_THAT(e.what(), HasSubstr(error_message_substr));
  } catch (const std::exception& e) {
    FAIL() << "Expected c10::Error, got: " << typeid(e).name()
           << " with error: " << e.what();
  }
}

void FreeFunctionThrowing() {
  TT_CHECK_THROW(false, error::kInvalidArgument) << "throwing invalid argument";
}

TEST(PyBindErrorUtilsInternalTest, ImplFreeFunctionThrowsC10Error) {
  TestHelperRaises<c10::Error>(
      FreeFunctionThrowing,  //
      /* name= */ "test_prefix",
      /* error_message_substr= */ "test_prefix(): throwing invalid argument");
}

void ThrowIndexError() {
  TT_CHECK_THROW(false, error::kPythonIndexError) << "throwing index error";
}

TEST(PyBindErrorUtilsInternalTest, ImplLambdaThrowsIndexError) {
  TestHelperRaises<c10::IndexError>(
      []() { ThrowIndexError(); },  //
      /* name= */ "test_prefix",
      /* error_message_substr= */ "test_prefix(): throwing index error");
}

struct DummyStruct {
  void ThrowingMember() const {
    TT_CHECK_THROW(false, error::kInvalidArgument) << "throwing from member";
  }
};

TEST(PyBindErrorUtilsInternalTest, ImplMemberFunctionLambdaThrowsC10Error) {
  try {
    internal::ErrorHandlingHelper<void(const DummyStruct&)>::impl(
        "DummyStruct.throwing_member",
        [](const DummyStruct& self) { self.ThrowingMember(); }, DummyStruct{});
    FAIL() << "Expected exception was not thrown";
  } catch (const c10::Error& e) {
    EXPECT_THAT(
        e.what(),
        HasSubstr("DummyStruct.throwing_member(): throwing from member"));
  } catch (const std::exception& e) {
    FAIL() << "Expected c10::Error, got: " << typeid(e).name()
           << " with error: " << e.what();
  }
}

// Test that the error message checks are run on the TorchTPU C++ functions
// bound to Python functions.
//
// Since error message style checks are only enabled in internal builds or when
// TORCH_TPU_INTERNAL_CHECK_ERROR_FORMAT is set, this test is skipped when they
// are disabled.
void ThrowInvalidMessage() {
  // "Invalid message" starts with uppercase, which is not allowed by
  // guidelines.
  TT_CHECK_THROW(false, error::kInvalidArgument) << "Invalid message";
}

TEST(PyBindErrorUtilsInternalDeathTest, ImplThrowsInvalidMessageCrashes) {
  if (!ShouldCheckErrorFormat()) {
    GTEST_SKIP() << "Error format checks are disabled in this build.";
  }
  EXPECT_DEATH(internal::ErrorHandlingHelper<void()>::impl("test_prefix",
                                                           ThrowInvalidMessage),
               "Bad error message format");
}

}  // namespace
}  // namespace torch_tpu
