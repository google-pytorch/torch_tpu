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

#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;

TEST(VersionTest, ComparesComponentByComponent) {
  EXPECT_EQ(Version(0, 1, 0), Version(0, 1, 0));
  EXPECT_LT(Version(0, 0, 99), Version(0, 1, 0));
  EXPECT_LT(Version(0, 1, 0), Version(0, 1, 1));
  EXPECT_LT(Version(0, 1, 1), Version(0, 2, 0));
  EXPECT_LT(Version(1, 99, 99), Version(2, 0, 0));
}

TEST(VersionTest, StableRanksAboveDevBuildsOfSameRelease) {
  EXPECT_LT(Version(0, 0, 46), Version(0, 0, 47, 20260824));
  EXPECT_LT(Version(0, 0, 47, 20260824), Version(0, 0, 47, 20260825));
  EXPECT_LT(Version(0, 0, 47, 20260825), Version(0, 0, 47));
  EXPECT_LT(Version(0, 0, 47), Version(0, 0, 48, 20260824));
}

TEST(VersionTest, MaxIsGreaterThanOrEqualToEveryVersion) {
  EXPECT_GE(Version::Max(), Version(999, 999, 999));
  EXPECT_GE(Version::Max(), Version::Max());
}

TEST(VersionDeathTest, ConstructorDiesOnNegativeComponents) {
  EXPECT_DEATH(Version(-1, 0, 0), "non-negative");
  EXPECT_DEATH(Version(0, -1, 0), "non-negative");
  EXPECT_DEATH(Version(0, 0, -1), "non-negative");
  EXPECT_DEATH(Version(0, 0, 0, -1), "non-negative");
}

TEST(VersionTest, Parse) {
  EXPECT_THAT(Version::Parse("2"), IsOkAndHolds(Version(2, 0, 0)));
  EXPECT_THAT(Version::Parse("0.1"), IsOkAndHolds(Version(0, 1, 0)));
  EXPECT_THAT(Version::Parse("2.3.15"), IsOkAndHolds(Version(2, 3, 15)));
  EXPECT_THAT(Version::Parse("0.0.47.dev20260824"),
              IsOkAndHolds(Version(0, 0, 47, 20260824)));
  EXPECT_THAT(Version::Parse("0.1.dev20240617"),
              IsOkAndHolds(Version(0, 1, 0, 20240617)));
}

TEST(VersionTest, ParseIgnoresSuffix) {
  EXPECT_THAT(Version::Parse("0.0.47.dev20260824+nightly"),
              IsOkAndHolds(Version(0, 0, 47, 20260824)));
  EXPECT_THAT(Version::Parse("0.0.48rc1"), IsOkAndHolds(Version(0, 0, 48)));
}

TEST(VersionTest, ParseTimestampDevVersionBeyondInt32) {
  // Timestamp-based dev versions use 12 digits (YYYYMMDDHHMM), exceeding
  // 32-bit signed int limits (2,147,483,647).
  EXPECT_THAT(Version::Parse("0.0.47.dev202605201400"),
              IsOkAndHolds(Version(0, 0, 47, 202605201400)));
}

TEST(VersionTest, ParseRejectsMalformedVersion) {
  for (const char* version : {"", "abc", "v1.2.3", "1..2", "0.1.2.3.4"}) {
    EXPECT_THAT(Version::Parse(version).status(),
                StatusIs(error::kInvalidArgument))
        << version;
  }
}

TEST(VersionTest, ParseRejectsOverflow) {
  EXPECT_THAT(Version::Parse("99999999999.0.0").status(),
              StatusIs(error::kInvalidArgument));
}

TEST(VersionTest, Stringify) {
  EXPECT_EQ(absl::StrCat(Version(2, 3, 15)), "2.3.15");
  EXPECT_EQ(absl::StrCat(Version(0, 0, 47, 20260824)), "0.0.47.dev20260824");
}

}  // namespace
}  // namespace torch_tpu
