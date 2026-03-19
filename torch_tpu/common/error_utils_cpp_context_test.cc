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

// Tests the behaviors of error handling utilities when C++ context is
// requested. This cannot be merged with error_utils_test.cc as pytorch doesn't
// allow changing the C++ context mode after it's initialized.

#include <stdlib.h>

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/macro_utils.h"
#include "torch_tpu/common/status_builder.h"
#include "torch_tpu/common/testing/error_utils_test_helper.h"
#include "torch_tpu/common/utils.h"

namespace torch_tpu {
namespace {

using internal::kCppErrorTraceUrl;
using internal::NormalizeIfInsideBuildDirectory;

// Per
// https://docs.pytorch.org/docs/stable/debugging_environment_variables.html,
// this enables C++ context in pytorch errors. This can be done only once per
// process as pytorch caches the value of this variable.
static const auto kInitShowCppContext =
    setenv("TORCH_SHOW_CPP_STACKTRACES", "1", /*overwrite=*/1);

absl::Status MakeTtError(absl::StatusCode code, std::string_view message) {
  return TT_ERROR(error::kInternal) << message;
}
// The line number of the call to TT_ERROR() above.
const int kTtErrorLine = __LINE__ - 3;

// Wrapper for `NormalizeIfInsideBuildDirectory()` function.
// Returning a `std::string_view` allow us to appropriately compare the expected
// and actual strings, instead of their pointers.
std::string_view Normalize(const char* path) {
  return NormalizeIfInsideBuildDirectory(path);
}

TEST(NormalizeTorchTpuVirtualIncludesFilePath, StripsPrefix) {
  // "torch_tpu/" relative paths.
  EXPECT_EQ(Normalize("torch_tpu/foo.cc"), "torch_tpu/foo.cc");
  EXPECT_EQ(Normalize("bazel-out/k8-fastbuild/bin/torch_tpu/foo.cc"),
            "torch_tpu/foo.cc");
  EXPECT_EQ(
      Normalize("bazel-out/k8-fastbuild/bin/torch_tpu/ops/_virtual_includes/"
                "op_builder_utils/torch_tpu/ops/op_builder_utils.h"),
      "torch_tpu/ops/op_builder_utils.h");

  // "third_party/py/torch_tpu/"
  EXPECT_EQ(Normalize("third_party/py/torch_tpu/foo.cc"),
            "third_party/py/torch_tpu/foo.cc");
  EXPECT_EQ(
      Normalize("bazel-out/k8-fastbuild/bin/third_party/py/torch_tpu/foo.cc"),
      "third_party/py/torch_tpu/foo.cc");
  EXPECT_EQ(Normalize("bazel-out/k8-fastbuild/bin/third_party/py/torch_tpu/ops/"
                      "_virtual_includes/op_builder_utils/third_party/py/"
                      "torch_tpu/ops/op_builder_utils.h"),
            "third_party/py/torch_tpu/ops/op_builder_utils.h");
}

TEST(TtError, HeaderPathNormalization) {
  const absl::Status error = MakeErrorFromHeader();
  const absl::optional<absl::Cord> cpp_context =
      error.GetPayload(kCppErrorTraceUrl);
  ASSERT_TRUE(cpp_context.has_value());

  std::string trace = std::string(cpp_context.value());

#if TT_IS_INTERNAL_TORCH_TPU
  constexpr std::string_view kProjectPrefix = "third_party/py/torch_tpu";
#else
  constexpr std::string_view kProjectPrefix = "torch_tpu";
#endif  // TT_IS_INTERNAL_TORCH_TPU

  // Verify it contains the normalized path to the header.
  EXPECT_THAT(
      trace, testing::HasSubstr(absl::StrCat(
                 kProjectPrefix, "/common/testing/error_utils_test_helper.h")));

  // Parse the path from the trace.
  // Trace format: path:line: function()\n...
  size_t colon_pos = trace.find(':');
  ASSERT_NE(colon_pos, std::string::npos);
  std::string path = trace.substr(0, colon_pos);

  // Verify first segment does not end with -out.
  size_t slash_pos = path.find('/');
  std::string first_segment =
      (slash_pos == std::string::npos) ? path : path.substr(0, slash_pos);
  EXPECT_THAT(first_segment, testing::Not(testing::EndsWith("-out")));

  // Verify no virtual includes.
  EXPECT_THAT(trace, testing::Not(testing::HasSubstr("_virtual_includes")));
}

TEST(TtError, ReturnsErrorWithCppContext) {
  const absl::Status error = MakeTtError(error::kInternal, "message 42");
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "message 42");
  const absl::optional<absl::Cord> cpp_context =
      error.GetPayload(kCppErrorTraceUrl);
  ASSERT_TRUE(cpp_context.has_value());
  const auto expected_context =
      absl::StrCat(TT_NORMALIZED_FILE, ":", kTtErrorLine, ": MakeTtError()\n");
  EXPECT_EQ(cpp_context.value(), expected_context);
}

absl::Status TtRetCheckFail(absl::StatusCode code, std::string_view message) {
  TT_RET_CHECK(false, code) << message;
  return absl::OkStatus();
}
// The line number of the call to TT_RET_CHECK() above.
const int kTtRetCheckLine = __LINE__ - 4;

TEST(TtRetCheck, ReturnsErrorWithCppContext) {
  const absl::Status error = TtRetCheckFail(error::kInternal, "message 42");
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "message 42");
  const absl::optional<absl::Cord> cpp_context =
      error.GetPayload(kCppErrorTraceUrl);
  ASSERT_TRUE(cpp_context.has_value());
  const auto expected_context = absl::StrCat(
      TT_NORMALIZED_FILE, ":", kTtRetCheckLine, ": TtRetCheckFail()\n");
  EXPECT_EQ(cpp_context.value(), expected_context);
}

absl::StatusOr<int> TtRetCheckFailStatusOr(absl::StatusCode code,
                                           std::string_view message) {
  TT_RET_CHECK(false, code) << message;
  return absl::OkStatus();
}
// The line number of the call to TT_RET_CHECK() above.
const int kTtRetCheckStatusOrLine = __LINE__ - 4;

absl::Status TtAssignOrReturnFail(absl::StatusCode code,
                                  std::string_view message) {
  TT_ASSIGN_OR_RETURN(int x, TtRetCheckFailStatusOr(code, message));
  static_cast<void>(x);
  return absl::OkStatus();
}
// The line number of the call to TT_ASSIGN_OR_RETURN() above.
const int kTtAssignOrReturnLine = __LINE__ - 5;

TEST(TtAssignOrReturnTwoArgs, AppendsToCppContext) {
  const absl::Status error =
      TtAssignOrReturnFail(error::kInternal, "message 42");
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "message 42");
  const absl::optional<absl::Cord> cpp_context =
      error.GetPayload(kCppErrorTraceUrl);
  ASSERT_TRUE(cpp_context.has_value());
  const auto expected_context =
      absl::StrCat(TT_NORMALIZED_FILE, ":", kTtRetCheckStatusOrLine,
                   ": TtRetCheckFailStatusOr()\n", TT_NORMALIZED_FILE, ":",
                   kTtAssignOrReturnLine, ": TtAssignOrReturnFail()\n");
  EXPECT_EQ(cpp_context.value(), expected_context);
}

absl::Status TtAssignOrReturnFail3(absl::StatusCode code,
                                   std::string_view message) {
  TT_ASSIGN_OR_RETURN(int x, TtRetCheckFailStatusOr(code, message),
                      _ << " extra");
  static_cast<void>(x);
  return absl::OkStatus();
}
// The line number of the call to TT_ASSIGN_OR_RETURN() above.
const int kTtAssignOrReturnLine3 = __LINE__ - 5;

TEST(TtAssignOrReturnThreeArgs, AppendsToCppContext) {
  const absl::Status error =
      TtAssignOrReturnFail3(error::kInternal, "message 42");
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "message 42 extra");
  const absl::optional<absl::Cord> cpp_context =
      error.GetPayload(kCppErrorTraceUrl);
  ASSERT_TRUE(cpp_context.has_value());
  const auto expected_context =
      absl::StrCat(TT_NORMALIZED_FILE, ":", kTtRetCheckStatusOrLine,
                   ": TtRetCheckFailStatusOr()\n", TT_NORMALIZED_FILE, ":",
                   kTtAssignOrReturnLine3, ": TtAssignOrReturnFail3()\n");
  EXPECT_EQ(cpp_context.value(), expected_context);
}

absl::Status TtReturnIfErrorFail(absl::StatusCode code,
                                 std::string_view message) {
  TT_RETURN_IF_ERROR(TtRetCheckFail(code, message));
  return absl::OkStatus();
}
// The line number of the call to TT_RETURN_IF_ERROR() above.
const int kTtReturnIfErrorLine = __LINE__ - 4;

TEST(TtReturnIfError, AppendsToCppContext) {
  const absl::Status error =
      TtReturnIfErrorFail(error::kInternal, "message 42");
  EXPECT_EQ(error.code(), error::kInternal);
  EXPECT_EQ(error.message(), "message 42");
  const absl::optional<absl::Cord> cpp_context =
      error.GetPayload(kCppErrorTraceUrl);
  ASSERT_TRUE(cpp_context.has_value());
  const auto expected_context =
      absl::StrCat(TT_NORMALIZED_FILE, ":", kTtRetCheckLine,
                   ": TtRetCheckFail()\n",  //
                   TT_NORMALIZED_FILE, ":", kTtReturnIfErrorLine,
                   ": TtReturnIfErrorFail()\n");
  EXPECT_EQ(cpp_context.value(), expected_context);
}

}  // namespace
}  // namespace torch_tpu
