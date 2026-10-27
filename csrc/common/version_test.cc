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

#include "csrc/common/version.h"

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(VersionTest, ComparesComponentByComponent) {
  EXPECT_EQ(Version(0, 1, 0), Version(0, 1, 0));
  EXPECT_EQ(Version(0, 0, 50), Version(0, 0, 50, 0));
  EXPECT_LT(Version(0, 0, 50), Version(0, 0, 50, 1));
  EXPECT_LT(Version(0, 0, 50, 1), Version(0, 0, 50, 2));
  EXPECT_LT(Version(0, 0, 50, 99), Version(0, 0, 51));
  EXPECT_LT(Version(0, 0, 99), Version(0, 1, 0));
  EXPECT_LT(Version(0, 1, 0), Version(0, 1, 1));
  EXPECT_LT(Version(0, 1, 1), Version(0, 2, 0));
  EXPECT_LT(Version(1, 99, 99), Version(2, 0, 0));
}

TEST(VersionTest, StableRanksAboveDevBuildsOfSameRelease) {
  EXPECT_LT(Version(0, 0, 46), Version(0, 0, 47, 0, 20260824));
  EXPECT_LT(Version(0, 0, 47, 0, 20260824), Version(0, 0, 47, 0, 20260825));
  EXPECT_LT(Version(0, 0, 47, 0, 20260825), Version(0, 0, 47));
  EXPECT_LT(Version(0, 0, 47), Version(0, 0, 47, 1, 20260824));
  EXPECT_LT(Version(0, 0, 47, 1, 20260824), Version(0, 0, 47, 1));
  EXPECT_LT(Version(0, 0, 47, 1), Version(0, 0, 48, 0, 20260824));
}

TEST(VersionTest, MaxIsGreaterThanOrEqualToEveryVersion) {
  EXPECT_GE(Version::Max(), Version(999, 999, 999, 999));
  EXPECT_GE(Version::Max(), Version::Max());
}

TEST(VersionDeathTest, ConstructorDiesOnNegativeComponents) {
  EXPECT_DEATH(Version(-1, 0, 0), "non-negative");
  EXPECT_DEATH(Version(0, -1, 0), "non-negative");
  EXPECT_DEATH(Version(0, 0, -1), "non-negative");
  EXPECT_DEATH(Version(0, 0, 0, -1), "non-negative");
  EXPECT_DEATH(Version(0, 0, 0, 0, -1), "non-negative");
}

TEST(VersionTest, Stringify) {
  EXPECT_EQ(absl::StrCat(Version(2, 3, 15)), "2.3.15");
  EXPECT_EQ(absl::StrCat(Version(0, 0, 50, 1)), "0.0.50.1");
  EXPECT_EQ(absl::StrCat(Version(0, 0, 47, 0, 20260824)), "0.0.47.dev20260824");
  EXPECT_EQ(absl::StrCat(Version(0, 0, 50, 1, 20261026)),
            "0.0.50.1.dev20261026");
}

}  // namespace
}  // namespace torch_tpu
