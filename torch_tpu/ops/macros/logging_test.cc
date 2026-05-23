// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/macros/logging.h"

#include <array>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/ones.h"
#include "absl/log/absl_log.h"
#include "c10/core/Device.h"
#include "c10/util/Optional.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

using internal::LogKernelArgs;
using testing::AllOf;
using testing::ElementsAre;
using testing::EndsWith;
using testing::HasSubstr;
using testing::IsEmpty;
using testing::StartsWith;

void Kernel2(int x, const std::string& y) {
  TT_CHECK_AND_LOG_KERNEL_ARGS_(OpName::kAdd, x, y);
}

TEST(LogKernelStart, CompilesWhenArgsAreIdentifiers) {
  int x = 1;
  std::string y = "foo";
  Kernel2(x, y);
}

TEST(LogKernelArgs, LogsNothingWithNoArgs) {
  std::ostringstream ss;
  LogKernelArgs(ss, {});
  EXPECT_EQ(ss.str(), "");
}

TEST(LogKernelArgs, LogsOneArg) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, 123);
  EXPECT_EQ(ss.str(), "arg1: 123\n");
}

TEST(LogKernelArgs, LogsMultipleArgs) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1", "arg2"}, 123, "foo");
  EXPECT_EQ(ss.str(),
            "arg1: 123\n"
            "arg2: foo\n");
}

TEST(LogKernelArgs, LogsTensor) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, at::ones(3));
  EXPECT_THAT(ss.str(), StartsWith("arg1: shape=[3]"));
}

TEST(LogKernelArgs, LogsScalar) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, at::Scalar(123.0));
  EXPECT_EQ(ss.str(), "arg1: 123\n");
}

TEST(LogKernelArgs, LogsScalarType) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, at::ScalarType::Float);
  EXPECT_EQ(ss.str(), "arg1: float32\n");
}

TEST(LogKernelArgs, LogsOptionalArrayRef) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, c10::OptionalArrayRef<int64_t>({1, 2, 3}));
  EXPECT_EQ(ss.str(), "arg1: <[1, 2, 3]>\n");
}

TEST(LogKernelArgs, LogsNullOptionalArrayRef) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, c10::OptionalArrayRef<int64_t>());
  EXPECT_EQ(ss.str(), "arg1: nullopt\n");
}

TEST(LogKernelArgs, LogsOptionalTensor) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, std::optional<at::Tensor>(at::ones(3)));
  EXPECT_THAT(ss.str(), StartsWith("arg1: <shape=[3]"));
}

TEST(LogKernelArgs, LogsVectorOfTensors) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"},
                std::vector<at::Tensor>({at::ones(3), at::ones(4)}));
  EXPECT_THAT(ss.str(), AllOf(StartsWith("arg1: [shape=[3]"),
                              HasSubstr(", shape=[4]"), EndsWith("]\n")));
}

TEST(LogKernelArgs, LogsVectorOfInts) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, std::vector<int>{1, 2, 3});
  EXPECT_EQ(ss.str(), "arg1: [1, 2, 3]\n");
}

TEST(LogKernelArgs, LogsNullopt) {
  std::ostringstream ss;
  std::optional<at::Tensor> null;
  LogKernelArgs(ss, {"arg1"}, null);
  EXPECT_THAT(ss.str(), StartsWith("arg1: nullopt\n"));
}

TEST(LogKernelArgs, LogsBool) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1", "arg2"}, true, false);
  EXPECT_EQ(ss.str(),
            "arg1: true\n"
            "arg2: false\n");
}

TEST(LogKernelArgs, LogsDevice) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, at::Device("cpu"));
  EXPECT_EQ(ss.str(), "arg1: cpu\n");
}

TEST(LogKernelArgs, LogsLayout) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, at::Layout::Strided);
  EXPECT_EQ(ss.str(), "arg1: Strided\n");
}

TEST(LogKernelArgs, LogsMemoryFormat) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1"}, at::MemoryFormat::Preserve);
  EXPECT_EQ(ss.str(), "arg1: Preserve\n");
}

TEST(LogKernelArgs, LogsVariadicArgs) {
  std::ostringstream ss;
  LogKernelArgs(ss, {"arg1", "arg2", "arg3"}, 123, "foo", at::ones(3));
  EXPECT_THAT(ss.str(), StartsWith("arg1: 123\n"
                                   "arg2: foo\n"
                                   "arg3: shape=[3]"));
}

std::vector<std::string> NullaryFunc() {
  ABSL_LOG(INFO) << __PRETTY_FUNCTION__;
  return internal::ParseArgTypesOrEmpty(__PRETTY_FUNCTION__);
}

TEST(ParseArgTypesOrEmpty, NullaryFunc) {
  EXPECT_THAT(NullaryFunc(), IsEmpty());
}

std::vector<std::string> UnaryFunc(const char* arg1) {
  ABSL_LOG(INFO) << __PRETTY_FUNCTION__;
  return internal::ParseArgTypesOrEmpty(__PRETTY_FUNCTION__);
}

TEST(ParseArgTypesOrEmpty, UnaryFunc) {
  EXPECT_THAT(UnaryFunc("foo"), ElementsAre("char *"));
}

std::vector<std::string> BinaryFunc(bool& arg1, const std::string& arg2) {
  ABSL_LOG(INFO) << __PRETTY_FUNCTION__;
  return internal::ParseArgTypesOrEmpty(__PRETTY_FUNCTION__);
}

TEST(ParseArgTypesOrEmpty, BinaryFunc) {
  bool b = true;
  EXPECT_THAT(BinaryFunc(b, "foo"), ElementsAre("bool", "std::string"));
}

std::vector<std::string> FuncWithComplexType(
    // The c10::optional part is an alias for std::optional.
    const c10::List<c10::optional<at::Tensor>>& arg1,
    // This type has a `,` in the name.
    const std::array<bool, 3>& arg2,
    // This type has an alias.
    c10::OptionalArrayRef<int64_t> arg3) {
  ABSL_LOG(INFO) << __PRETTY_FUNCTION__;
  return internal::ParseArgTypesOrEmpty(__PRETTY_FUNCTION__);
}

TEST(ParseArgTypesOrEmpty, FuncWithComplexType) {
  std::array<bool, 3> arr = {true, false, true};
  EXPECT_THAT(FuncWithComplexType(c10::List<c10::optional<at::Tensor>>(), arr,
                                  std::nullopt),
              ElementsAre("c10::List<std::optional<at::Tensor>>",
                          "std::array<bool, 3>", "at::OptionalIntArrayRef"));
}

}  // namespace
}  // namespace torch_tpu
