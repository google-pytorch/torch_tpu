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

#include "csrc/common/fixed_size_span.h"

#include <array>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(FixedSizeSpanTest, ConstructFromSpan) {
  std::vector<int> values = {1, 2, 3};
  FixedSizeSpan<int, 3> span(absl::MakeSpan(values));
  EXPECT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], 1);
  EXPECT_EQ(span[1], 2);
  EXPECT_EQ(span[2], 3);
}

TEST(FixedSizeSpanTest, ConstructFromInitializerList) {
  const auto list = {10, 20, 30};
  FixedSizeSpan<const int, 3> span(list);
  EXPECT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], 10);
  EXPECT_EQ(span[1], 20);
  EXPECT_EQ(span[2], 30);
}

TEST(FixedSizeSpanDeathTest, SizeMismatch) {
  const auto construct_from_span = [] {
    std::vector<int> values = {1, 2};
    FixedSizeSpan<int, 3> span(absl::MakeSpan(values));
    static_cast<void>(span);
  };
  EXPECT_DEATH(construct_from_span(), "expected .* elements, got .*");

  const auto construct_from_initializer_list = [] {
    const auto list = {10, 20};
    FixedSizeSpan<const int, 3> span(list);
    static_cast<void>(span);
  };
  EXPECT_DEATH(construct_from_initializer_list(),
               "expected .* elements, got .*");
}

TEST(FixedSizeSpanTest, ConstructFromArray) {
  std::array<int, 2> arr = {4, 5};
  FixedSizeSpan<int, 2> span(arr);
  EXPECT_EQ(span.size(), 2);
  EXPECT_EQ(span[0], 4);
  EXPECT_EQ(span[1], 5);

  const std::array<int, 2> const_arr = {6, 7};
  FixedSizeSpan<const int, 2> const_span(const_arr);
  EXPECT_EQ(const_span.size(), 2);
  EXPECT_EQ(const_span[0], 6);
  EXPECT_EQ(const_span[1], 7);
}

TEST(FixedSizeSpanTest, StructuredBindings) {
  std::array<int, 3> arr = {7, 8, 9};
  FixedSizeSpan<int, 3> span(arr);
  auto [a, b, c] = span;
  EXPECT_EQ(a, 7);
  EXPECT_EQ(b, 8);
  EXPECT_EQ(c, 9);
}

TEST(FixedSizeSpanTest, MoveToStdArrayEmpty) {
  std::vector<int> empty;
  FixedSizeSpan<int, 0> span(absl::MakeSpan(empty));
  std::array<int, 0> arr = MoveToStdArray(span);
  EXPECT_TRUE(arr.empty());
}

TEST(FixedSizeSpanTest, MoveToStdArraySizes) {
  std::vector<int> v1 = {100};
  FixedSizeSpan<int, 1> span1(absl::MakeSpan(v1));
  std::array<int, 1> arr1 = MoveToStdArray(span1);
  EXPECT_EQ(arr1[0], 100);

  std::vector<int> v2 = {1, 2};
  FixedSizeSpan<int, 2> span2(absl::MakeSpan(v2));
  std::array<int, 2> arr2 = MoveToStdArray(span2);
  EXPECT_EQ(arr2[0], 1);
  EXPECT_EQ(arr2[1], 2);

  std::vector<int> v9 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  FixedSizeSpan<int, 9> span9(absl::MakeSpan(v9));
  std::array<int, 9> arr9 = MoveToStdArray(span9);
  EXPECT_EQ(arr9.size(), 9);
  EXPECT_EQ(arr9[0], 1);
  EXPECT_EQ(arr9[8], 9);

  std::vector<int> v12 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  FixedSizeSpan<int, 12> span12(absl::MakeSpan(v12));
  std::array<int, 12> arr12 = MoveToStdArray(span12);
  EXPECT_EQ(arr12.size(), 12);
  EXPECT_EQ(arr12[0], 1);
  EXPECT_EQ(arr12[11], 12);
}

TEST(FixedSizeSpanTest, MoveToStdArrayMoveOnly) {
  std::vector<std::unique_ptr<int>> ptrs;
  ptrs.push_back(std::make_unique<int>(42));
  ptrs.push_back(std::make_unique<int>(99));
  ptrs.push_back(std::make_unique<int>(123));

  FixedSizeSpan<std::unique_ptr<int>, 3> span(absl::MakeSpan(ptrs));
  std::array<std::unique_ptr<int>, 3> arr = MoveToStdArray(span);
  EXPECT_EQ(*arr[0], 42);
  EXPECT_EQ(*arr[1], 99);
  EXPECT_EQ(*arr[2], 123);
  EXPECT_EQ(ptrs[0], nullptr);
  EXPECT_EQ(ptrs[1], nullptr);
  EXPECT_EQ(ptrs[2], nullptr);
}

struct NonDefaultConstructible {
  explicit NonDefaultConstructible(int v) : value(v) {}
  int value;
};

TEST(FixedSizeSpanTest, MoveToStdArrayNonDefaultConstructibleEmpty) {
  std::vector<NonDefaultConstructible> empty;
  FixedSizeSpan<NonDefaultConstructible, 0> span(absl::MakeSpan(empty));
  std::array<NonDefaultConstructible, 0> arr = MoveToStdArray(span);
  EXPECT_TRUE(arr.empty());
}

TEST(FixedSizeSpanTest, MoveToStdArrayNonDefaultConstructible) {
  std::vector<NonDefaultConstructible> items;
  items.emplace_back(10);
  items.emplace_back(20);

  FixedSizeSpan<NonDefaultConstructible, 2> span(absl::MakeSpan(items));
  std::array<NonDefaultConstructible, 2> arr = MoveToStdArray(span);
  EXPECT_EQ(arr[0].value, 10);
  EXPECT_EQ(arr[1].value, 20);
}

}  // namespace
}  // namespace torch_tpu
