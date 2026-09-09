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

#include "csrc/common/constexpr_map.h"

#include <array>
#include <functional>
#include <stdexcept>
#include <string_view>

#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using namespace std::string_view_literals;

// -----------------------------------------------------------------------------
// Compile-time (constexpr) tests using static_assert
// -----------------------------------------------------------------------------

// Test empty map (N = 0)
constexpr ConstexprMap<std::string_view, int, 0> kEmptyMap{};
static_assert(kEmptyMap.empty());
static_assert(kEmptyMap.size() == 0);
static_assert(!kEmptyMap.contains("foo"sv));
static_assert(kEmptyMap.find("foo"sv) == kEmptyMap.end());
static_assert(kEmptyMap.begin() == kEmptyMap.end());
static_assert(kEmptyMap.cbegin() == kEmptyMap.cend());

// Test single element map (N = 1)
constexpr ConstexprMap<std::string_view, int, 1> kSingleMap({{"only"sv, 100}});
static_assert(!kSingleMap.empty());
static_assert(kSingleMap.size() == 1);
static_assert(kSingleMap.contains("only"sv));
static_assert(!kSingleMap.contains("other"sv));
static_assert(kSingleMap.at("only"sv) == 100);
static_assert(kSingleMap["only"sv] == 100);
static_assert(kSingleMap.find("only"sv) == kSingleMap.begin());
static_assert(kSingleMap.find("other"sv) == kSingleMap.end());

// Test multi-element map constructed out of order
constexpr ConstexprMap<std::string_view, int, 5> kMap({
    {"zebra"sv, 26},
    {"apple"sv, 1},
    {"mango"sv, 13},
    {"banana"sv, 2},
    {"cherry"sv, 3},
});

static_assert(!kMap.empty());
static_assert(kMap.size() == 5);

// Check lookup via contains
static_assert(kMap.contains("apple"sv));
static_assert(kMap.contains("banana"sv));
static_assert(kMap.contains("cherry"sv));
static_assert(kMap.contains("mango"sv));
static_assert(kMap.contains("zebra"sv));
static_assert(!kMap.contains("orange"sv));
static_assert(!kMap.contains("aardvark"sv));
static_assert(!kMap.contains("zzz"sv));

// Check element access
static_assert(kMap.at("apple"sv) == 1);
static_assert(kMap.at("banana"sv) == 2);
static_assert(kMap.at("cherry"sv) == 3);
static_assert(kMap.at("mango"sv) == 13);
static_assert(kMap.at("zebra"sv) == 26);
static_assert(kMap["apple"sv] == 1);
static_assert(kMap["zebra"sv] == 26);

// Check iteration and structured binding in constexpr
constexpr int SumValues() {
  int total = 0;
  for (const auto& [key, value] : kMap) {
    total += value;
  }
  return total;
}
static_assert(SumValues() == (1 + 2 + 3 + 13 + 26));

// Check heterogeneous lookup: string_view key queried with const char*
static_assert(kMap.contains("apple"));
static_assert(kMap.at("apple") == 1);
static_assert(kMap["banana"] == 2);
static_assert(kMap.find("zebra")->value == 26);

// Check custom comparator: std::greater
constexpr ConstexprMap<int, std::string_view, 3, std::greater<int>>
    kDescendingMap({{1, "one"sv}, {3, "three"sv}, {2, "two"sv}});

static_assert(kDescendingMap.at(3) == "three"sv);
static_assert(kDescendingMap.at(1) == "one"sv);
static_assert(kDescendingMap.contains(2));
static_assert(!kDescendingMap.contains(4));

// Check MakeConstexprMap factory function
constexpr auto kFactoryMap = MakeConstexprMap<std::string_view, int>(
    {{"x"sv, 10}, {"y"sv, 20}, {"z"sv, 30}});
static_assert(kFactoryMap.size() == 3);
static_assert(kFactoryMap.at("y"sv) == 20);

// -----------------------------------------------------------------------------
// Runtime unit tests
// -----------------------------------------------------------------------------

TEST(ConstexprMapTest, EmptyMap) {
  ConstexprMap<int, int, 0> empty_map;
  EXPECT_TRUE(empty_map.empty());
  EXPECT_EQ(empty_map.size(), 0);
  EXPECT_EQ(empty_map.begin(), empty_map.end());
  EXPECT_FALSE(empty_map.contains(1));
  EXPECT_EQ(empty_map.find(1), empty_map.end());
  EXPECT_THROW(empty_map.at(1), std::out_of_range);
  EXPECT_THROW(empty_map[1], std::out_of_range);
}

TEST(ConstexprMapTest, SingleElementMap) {
  ConstexprMap<std::string_view, int, 1> single_map({{"foo"sv, 42}});
  EXPECT_FALSE(single_map.empty());
  EXPECT_EQ(single_map.size(), 1);
  EXPECT_TRUE(single_map.contains("foo"sv));
  EXPECT_FALSE(single_map.contains("bar"sv));
  EXPECT_EQ(single_map.at("foo"sv), 42);
  EXPECT_EQ(single_map["foo"sv], 42);
  EXPECT_THROW(single_map.at("bar"sv), std::out_of_range);
  EXPECT_THROW(single_map["bar"sv], std::out_of_range);
}

TEST(ConstexprMapTest, MultiElementLookupAndOrdering) {
  ConstexprMap<int, std::string_view, 4> map(
      {{40, "forty"sv}, {10, "ten"sv}, {30, "thirty"sv}, {20, "twenty"sv}});

  EXPECT_EQ(map.size(), 4);

  EXPECT_TRUE(map.contains(10));
  EXPECT_TRUE(map.contains(20));
  EXPECT_TRUE(map.contains(30));
  EXPECT_TRUE(map.contains(40));
  EXPECT_FALSE(map.contains(25));

  EXPECT_EQ(map.at(10), "ten"sv);
  EXPECT_EQ(map.at(20), "twenty"sv);
  EXPECT_EQ(map.at(30), "thirty"sv);
  EXPECT_EQ(map.at(40), "forty"sv);

  EXPECT_THROW(map.at(50), std::out_of_range);
  EXPECT_THROW(map[50], std::out_of_range);
}

TEST(ConstexprMapTest, DuplicateKeyThrows) {
  EXPECT_THROW((ConstexprMap<int, int, 3>({{1, 10}, {2, 20}, {1, 30}})),
               std::invalid_argument);
}

}  // namespace
}  // namespace torch_tpu
