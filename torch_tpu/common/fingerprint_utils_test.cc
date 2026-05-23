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

#include "torch_tpu/common/fingerprint_utils.h"

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dtype.h"  // IWYU pragma: keep
#include "torch_tpu/ops/op_names.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/xla.pb.h"

namespace torch_tpu {
namespace {

TEST(Fingerprint, WorksForIntegers) {
  EXPECT_EQ(Fingerprint(1), 1);
  EXPECT_EQ(Fingerprint(1234567890), 1234567890);
  EXPECT_EQ(Fingerprint(-1), static_cast<FingerprintType>(-1));
  EXPECT_EQ(Fingerprint(std::numeric_limits<int>::max()),
            std::numeric_limits<int>::max());
  EXPECT_EQ(Fingerprint(std::numeric_limits<int>::min()),
            std::numeric_limits<int>::min());
  EXPECT_EQ(Fingerprint(std::numeric_limits<int64_t>::max()),
            std::numeric_limits<int64_t>::max());
  EXPECT_EQ(Fingerprint(std::numeric_limits<int64_t>::min()),
            std::numeric_limits<int64_t>::min());
}

TEST(Fingerprint, WorksForSpan) {
  int a[] = {1, 2, 3};
  absl::Span<int> a_span(a, 3);
  auto fp = Fingerprint(a_span);
  EXPECT_EQ(fp, 4382234674942165662ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForEmptySpan) {
  absl::Span<int> a_span;
  auto fp = Fingerprint(a_span);
  EXPECT_EQ(fp, 0)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForEnumSpan) {
  const mlir::ElementType a[] = {mlir::ElementType::F32, mlir::ElementType::F64,
                                 mlir::ElementType::BF16};
  absl::Span<const mlir::ElementType> span(a, 3);
  const auto fp = Fingerprint(span);
  EXPECT_EQ(fp, 11270043840191630410ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForVector) {
  std::vector<mlir::ElementType> a = {
      mlir::ElementType::F32, mlir::ElementType::F64, mlir::ElementType::BF16};
  const auto fp = Fingerprint(a);
  EXPECT_EQ(fp, 11270043840191630410ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForSpanSpan) {
  const mlir::ElementType a[] = {mlir::ElementType::F32, mlir::ElementType::F64,
                                 mlir::ElementType::BF16};
  const mlir::ElementType b[] = {mlir::ElementType::I16, mlir::ElementType::I2};
  absl::Span<const mlir::ElementType> span_a(a, 3), span_b(b, 2);
  std::vector<absl::Span<const mlir::ElementType>> span_span = {span_a, span_b};
  const auto fp = Fingerprint(span_span);
  EXPECT_EQ(fp, 3116560340660294610ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForBool) {
  EXPECT_EQ(Fingerprint(true), 1)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
  EXPECT_EQ(Fingerprint(false), 0)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForCString) {
  EXPECT_EQ(Fingerprint("hello"), 13009744463427800296ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForStringView) {
  EXPECT_EQ(Fingerprint(std::string_view("hello")), 13009744463427800296ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForStringObject) {
  std::string s = "hello";
  EXPECT_EQ(Fingerprint(s), 13009744463427800296ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Finterprint, WorksForPair) {
  std::pair<int, std::string> p = {1, "hi"};
  EXPECT_EQ(Fingerprint(p), 2362095114178386943ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForEmptyMap) {
  std::map<int, std::string> m;
  EXPECT_EQ(Fingerprint(m), 0)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForMap) {
  std::map<int, std::string> m = {{1, "hello"}, {2, "world"}};
  EXPECT_EQ(Fingerprint(m), 11808310844027338525ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForElementType) {
  EXPECT_EQ(Fingerprint(mlir::ElementType::F32), 13784249902793831416ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
  EXPECT_EQ(Fingerprint(mlir::ElementType::BF16), 13279827888509665365ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForOpName) {
  EXPECT_EQ(Fingerprint(OpName::kAdd), 392916950432762546ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
  EXPECT_EQ(Fingerprint(OpName::kZero_), 10492423113836821119ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForFloatingPointTypes) {
  EXPECT_EQ(Fingerprint(42.0f), 1109917696ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
  EXPECT_EQ(Fingerprint(42.0), 4631107791820423168ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(Fingerprint, WorksForVariant) {
  {
    std::variant<int, float, std::string> v = 42;
    EXPECT_EQ(Fingerprint(v), 7278683826098160691ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass. "
           "Instead, figure out why your code change caused the fingerprint to "
           "change.";
  }

  {
    std::variant<int, double, std::string> v = 42.0;
    EXPECT_EQ(Fingerprint(v), 16822203514485882311ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass. "
           "Instead, figure out why your code change caused the fingerprint to "
           "change.";
  }

  {
    xla::CompileOptions::OptionOverride override = "hello";
    EXPECT_EQ(Fingerprint(override), 9437734065348176533ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass. "
           "Instead, figure out why your code change caused the fingerprint to "
           "change.";

    override = false;
    EXPECT_EQ(Fingerprint(override), 12725806677685968135ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass. "
           "Instead, figure out why your code change caused the fingerprint to "
           "change.";
  }
}

TEST(FingerprintCat, WorksForZeroArgs) {
  EXPECT_EQ(FingerprintCat(), 0)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(FingerprintCat, WorksForOneArg) {
  EXPECT_EQ(FingerprintCat(1), 1)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
  EXPECT_EQ(FingerprintCat(1234567890), 1234567890)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

TEST(FingerprintCat, WorksForMultipleArgs) {
  EXPECT_EQ(FingerprintCat(1, std::string("hi")), 2362095114178386943ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
  EXPECT_EQ(FingerprintCat(1, 2, 3), 10629069731271099820ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
  EXPECT_EQ(FingerprintCat(1, 2, 3, 4), 6144917672450084373ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass. "
         "Instead, figure out why your code change caused the fingerprint to "
         "change.";
}

}  // namespace
}  // namespace torch_tpu
