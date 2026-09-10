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
#include <typeinfo>

#include "c10/util/Exception.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::HasSubstr;

void FreeFunctionThrowing() {
  TT_CHECK_THROW(false, error::kInvalidArgument) << "throwing invalid argument";
}

TEST(PyBindErrorUtilsInternalTest, ImplFreeFunctionThrowsC10Error) {
  try {
    internal::ErrorHandlingHelper<void()>::impl("test_prefix",
                                                FreeFunctionThrowing);
    FAIL() << "Expected exception was not thrown";
  } catch (const c10::Error& e) {
    // Verify exact type is c10::Error (not a subclass)
    EXPECT_EQ(typeid(e), typeid(c10::Error));
    EXPECT_THAT(e.what(),
                HasSubstr("test_prefix(): throwing invalid argument"));
  } catch (const std::exception& e) {
    FAIL() << "Expected c10::Error, got: " << e.what()
           << " (type: " << typeid(e).name() << ")";
  }
}

void ThrowIndexError() {
  TT_CHECK_THROW(false, error::kPythonIndexError) << "throwing index error";
}

TEST(PyBindErrorUtilsInternalTest, ImplLambdaThrowsIndexError) {
  try {
    internal::ErrorHandlingHelper<void()>::impl("test_prefix",
                                                []() { ThrowIndexError(); });
    FAIL() << "Expected exception was not thrown";
  } catch (const c10::IndexError& e) {
    // Verify exact type is c10::IndexError
    EXPECT_EQ(typeid(e), typeid(c10::IndexError));
    EXPECT_THAT(e.what(), HasSubstr("test_prefix(): throwing index error"));
  } catch (const std::exception& e) {
    FAIL() << "Expected c10::IndexError, got: " << e.what()
           << " (type: " << typeid(e).name() << ")";
  }
}

// Test that the error message checks are run on the TorchTPU C++ functions
// bound to Python functions.
//
// Since error message style checks are only enabled on internal debug builds,
// we need to condition the compilation of that test with this macro.
#if TT_CHECKS_ERROR_FORMAT

void ThrowInvalidMessage() {
  // "Invalid message" starts with uppercase, which is not allowed by
  // guidelines.
  TT_CHECK_THROW(false, error::kInvalidArgument) << "Invalid message";
}

TEST(PyBindErrorUtilsInternalDeathTest, ImplThrowsInvalidMessageCrashes) {
  EXPECT_DEATH(internal::ErrorHandlingHelper<void()>::impl("test_prefix",
                                                           ThrowInvalidMessage),
               "Improper error message format");
}

#endif  // TT_CHECKS_ERROR_FORMAT

}  // namespace
}  // namespace torch_tpu
