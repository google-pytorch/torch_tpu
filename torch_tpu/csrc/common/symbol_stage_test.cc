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

#include "torch_tpu/csrc/common/symbol_stage.h"

#include <string_view>

#include "gtest/gtest.h"
#include "torch_tpu/csrc/common/constexpr_map.h"

namespace torch_tpu {
namespace {

using namespace std::string_view_literals;

// -----------------------------------------------------------------------------
// Compile-time (constexpr) tests using static_assert
// -----------------------------------------------------------------------------

constexpr SymbolStage kInternalImplementationStage(
    SymbolStage::InternalImplementation());
static_assert(kInternalImplementationStage.is_internal_implementation());
static_assert(!kInternalImplementationStage.is_internal_api());
static_assert(!kInternalImplementationStage.is_experimental());
static_assert(!kInternalImplementationStage.is_stable());
static_assert(!kInternalImplementationStage.is_deprecated());
static_assert(kInternalImplementationStage.version().empty());

constexpr SymbolStage kInternalApiStage(SymbolStage::InternalApi());
static_assert(!kInternalApiStage.is_internal_implementation());
static_assert(kInternalApiStage.is_internal_api());
static_assert(!kInternalApiStage.is_experimental());
static_assert(!kInternalApiStage.is_stable());
static_assert(!kInternalApiStage.is_deprecated());
static_assert(kInternalApiStage.version().empty());

constexpr SymbolStage kExperimentalStage(SymbolStage::Experimental());
static_assert(!kExperimentalStage.is_internal_implementation());
static_assert(!kExperimentalStage.is_internal_api());
static_assert(kExperimentalStage.is_experimental());
static_assert(!kExperimentalStage.is_stable());
static_assert(!kExperimentalStage.is_deprecated());
static_assert(kExperimentalStage.version().empty());

constexpr SymbolStage kStableStage(SymbolStage::Stable());
static_assert(!kStableStage.is_internal_implementation());
static_assert(!kStableStage.is_internal_api());
static_assert(!kStableStage.is_experimental());
static_assert(kStableStage.is_stable());
static_assert(!kStableStage.is_deprecated());
static_assert(kStableStage.version().empty());

constexpr SymbolStage kDeprecatedStage(SymbolStage::Deprecated("2.13"sv));
static_assert(!kDeprecatedStage.is_internal_implementation());
static_assert(!kDeprecatedStage.is_internal_api());
static_assert(!kDeprecatedStage.is_experimental());
static_assert(!kDeprecatedStage.is_stable());
static_assert(kDeprecatedStage.is_deprecated());
static_assert(kDeprecatedStage.version() == "2.13"sv);

// Check comparisons
static_assert(kInternalImplementationStage ==
              SymbolStage(SymbolStage::InternalImplementation()));
static_assert(kInternalImplementationStage != kExperimentalStage);
static_assert(kDeprecatedStage ==
              SymbolStage(SymbolStage::Deprecated("2.13"sv)));
static_assert(kDeprecatedStage !=
              SymbolStage(SymbolStage::Deprecated("2.14"sv)));

// Check usage within ConstexprMap at compile time
constexpr ConstexprMap<std::string_view, SymbolStage, 4> kSymbolStages({
    {"my_internal_op"sv, SymbolStage(SymbolStage::InternalImplementation())},
    {"my_experimental_op"sv, SymbolStage(SymbolStage::Experimental())},
    {"my_stable_op"sv, SymbolStage(SymbolStage::Stable())},
    {"my_deprecated_op"sv, SymbolStage(SymbolStage::Deprecated("2.13"sv))},
});

static_assert(
    kSymbolStages.at("my_internal_op"sv).is_internal_implementation());
static_assert(kSymbolStages.at("my_experimental_op"sv).is_experimental());
static_assert(kSymbolStages.at("my_stable_op"sv).is_stable());
static_assert(kSymbolStages.at("my_deprecated_op"sv).is_deprecated());
static_assert(kSymbolStages.at("my_deprecated_op"sv).version() == "2.13"sv);

// -----------------------------------------------------------------------------
// Runtime unit tests
// -----------------------------------------------------------------------------

TEST(SymbolStageTest, InternalImplementationStage) {
  SymbolStage stage(SymbolStage::InternalImplementation());
  EXPECT_TRUE(stage.is_internal_implementation());
  EXPECT_FALSE(stage.is_internal_api());
  EXPECT_FALSE(stage.is_experimental());
  EXPECT_FALSE(stage.is_stable());
  EXPECT_FALSE(stage.is_deprecated());
  EXPECT_TRUE(stage.version().empty());
}

TEST(SymbolStageTest, InternalApiStage) {
  SymbolStage stage(SymbolStage::InternalApi());
  EXPECT_FALSE(stage.is_internal_implementation());
  EXPECT_TRUE(stage.is_internal_api());
  EXPECT_FALSE(stage.is_experimental());
  EXPECT_FALSE(stage.is_stable());
  EXPECT_FALSE(stage.is_deprecated());
  EXPECT_TRUE(stage.version().empty());
}

TEST(SymbolStageTest, ExperimentalStage) {
  SymbolStage stage(SymbolStage::Experimental());
  EXPECT_FALSE(stage.is_internal_implementation());
  EXPECT_FALSE(stage.is_internal_api());
  EXPECT_TRUE(stage.is_experimental());
  EXPECT_FALSE(stage.is_stable());
  EXPECT_FALSE(stage.is_deprecated());
  EXPECT_TRUE(stage.version().empty());
}

TEST(SymbolStageTest, StableStage) {
  SymbolStage stage(SymbolStage::Stable());
  EXPECT_FALSE(stage.is_internal_implementation());
  EXPECT_FALSE(stage.is_internal_api());
  EXPECT_FALSE(stage.is_experimental());
  EXPECT_TRUE(stage.is_stable());
  EXPECT_FALSE(stage.is_deprecated());
  EXPECT_TRUE(stage.version().empty());
}

TEST(SymbolStageTest, DeprecatedStage) {
  SymbolStage stage(SymbolStage::Deprecated("2.13"));
  EXPECT_FALSE(stage.is_internal_implementation());
  EXPECT_FALSE(stage.is_internal_api());
  EXPECT_FALSE(stage.is_experimental());
  EXPECT_FALSE(stage.is_stable());
  EXPECT_TRUE(stage.is_deprecated());
  EXPECT_EQ(stage.version(), "2.13");
}

TEST(SymbolStageTest, EqualityComparison) {
  EXPECT_EQ(SymbolStage(SymbolStage::InternalImplementation()),
            SymbolStage(SymbolStage::InternalImplementation()));
  EXPECT_NE(SymbolStage(SymbolStage::InternalImplementation()),
            SymbolStage(SymbolStage::Stable()));
  EXPECT_EQ(SymbolStage(SymbolStage::Deprecated("2.13")),
            SymbolStage(SymbolStage::Deprecated("2.13")));
  EXPECT_NE(SymbolStage(SymbolStage::Deprecated("2.13")),
            SymbolStage(SymbolStage::Deprecated("2.14")));
}

}  // namespace
}  // namespace torch_tpu
