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

#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

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
  EXPECT_LT(Version(0, 0, 46), Version(0, 0, 47, Version::Dev(20260824)));
  EXPECT_LT(Version(0, 0, 47, Version::Dev(20260824)),
            Version(0, 0, 47, Version::Dev(20260825)));
  EXPECT_LT(Version(0, 0, 47, Version::Dev(20260825)), Version(0, 0, 47));
  EXPECT_LT(Version(0, 0, 47), Version(0, 0, 47, 1, Version::Dev(20260824)));
  EXPECT_LT(Version(0, 0, 47, 1, Version::Dev(20260824)), Version(0, 0, 47, 1));
  EXPECT_LT(Version(0, 0, 47, 1), Version(0, 0, 48, Version::Dev(20260824)));
}

TEST(VersionTest, MaxIsGreaterThanOrEqualToEveryVersion) {
  constexpr int kMaxInt = std::numeric_limits<int>::max();
  EXPECT_GT(Version::Max(), Version(999, 999, 999, 999));
  EXPECT_GT(Version::Max(), Version(kMaxInt, kMaxInt, kMaxInt, kMaxInt));
  TT_ASSERT_OK_AND_ASSIGN(const Version high_epoch,
                          Version::Parse("999!999.999.999.999.1.post999"));
  EXPECT_GT(Version::Max(), high_epoch);
  EXPECT_EQ(Version::Max(), Version::Max());
}

TEST(VersionDeathTest, ConstructorDiesOnNegativeComponents) {
  EXPECT_DEATH(Version(-1, 0, 0), "non-negative");
  EXPECT_DEATH(Version(0, -1, 0), "non-negative");
  EXPECT_DEATH(Version(0, 0, -1), "non-negative");
  EXPECT_DEATH(Version(0, 0, 0, -1), "non-negative");
  EXPECT_DEATH(Version(0, 0, 0, Version::Dev(-1)), "non-negative");
  EXPECT_DEATH(Version(0, 0, 0, 0, Version::Dev(-1)), "non-negative");
}

TEST(VersionTest, ParseReleaseAndDevVersions) {
  EXPECT_THAT(Version::Parse("2"), IsOkAndHolds(Version(2, 0, 0)));
  EXPECT_THAT(Version::Parse("0.1"), IsOkAndHolds(Version(0, 1, 0)));
  EXPECT_THAT(Version::Parse("2.3.15"), IsOkAndHolds(Version(2, 3, 15)));
  EXPECT_THAT(Version::Parse("0.0.50.0"), IsOkAndHolds(Version(0, 0, 50)));
  EXPECT_THAT(Version::Parse("0.0.50.1"), IsOkAndHolds(Version(0, 0, 50, 1)));
  EXPECT_THAT(Version::Parse("0.0.47.dev20260824"),
              IsOkAndHolds(Version(0, 0, 47, Version::Dev(20260824))));
  EXPECT_THAT(Version::Parse("0.0.50.1.dev20261026"),
              IsOkAndHolds(Version(0, 0, 50, 1, Version::Dev(20261026))));
  EXPECT_THAT(Version::Parse("0.1.dev20240617"),
              IsOkAndHolds(Version(0, 1, 0, Version::Dev(20240617))));
}

TEST(VersionTest, ParseTimestampDevVersionBeyondInt32) {
  // Timestamp-based dev versions use 12 digits (YYYYMMDDHHMM), exceeding
  // 32-bit signed int limits (2,147,483,647).
  EXPECT_THAT(Version::Parse("0.0.47.dev202605201400"),
              IsOkAndHolds(Version(0, 0, 47, Version::Dev(202605201400))));
}

TEST(VersionTest, ParsePep440NormalizationAndLocalLabels) {
  // Leading/trailing whitespace, leading 'v'/'V', and '+local' build labels.
  EXPECT_THAT(Version::Parse("  v2.3.15  "), IsOkAndHolds(Version(2, 3, 15)));
  EXPECT_THAT(Version::Parse("V0.0.50.1+tpu.v6e"),
              IsOkAndHolds(Version(0, 0, 50, 1)));
  EXPECT_THAT(Version::Parse("0.0.47.dev20260824+nightly"),
              IsOkAndHolds(Version(0, 0, 47, Version::Dev(20260824))));

  // Implicit 0 on pre/post/dev segments and PEP 440 spelling aliases.
  struct AliasCase {
    std::string_view alias;
    std::string_view canonical;
  };
  for (const auto& [alias, canonical] : std::vector<AliasCase>{
           {"1.0a", "1.0a0"},
           {"1.0.ALPHA.2", "1.0a2"},
           {"1.0-beta3", "1.0b3"},
           {"1.0c1", "1.0rc1"},
           {"1.0pre1", "1.0rc1"},
           {"1.0_preview_1", "1.0rc1"},
           {"1.0.post", "1.0.post0"},
           {"1.0-1", "1.0.post1"},
           {"1.0.rev2", "1.0.post2"},
           {"1.0r2", "1.0.post2"},
           {"1.0.dev", "1.0.dev0"},
       }) {
    TT_ASSERT_OK_AND_ASSIGN(const Version expected, Version::Parse(canonical));
    EXPECT_THAT(Version::Parse(alias), IsOkAndHolds(expected)) << alias;
  }
  EXPECT_THAT(Version::Parse("0!1.0.0"), IsOkAndHolds(Version(1, 0, 0)));
}

TEST(VersionTest, Pep440TotalOrdering) {
  // Strict ascending sequence covering dev, alpha/beta/rc pre-releases,
  // post-releases, 4+ release components, and epochs per PEP 440.
  const std::vector<std::string_view> ordered_inputs = {
      "0.0.48.dev0",
      "0.0.48.dev1",
      "0.0.48a1.dev0",
      "0.0.48a1",
      "0.0.48a1.post0.dev1",
      "0.0.48a1.post0",
      "0.0.48a1.post1",
      "0.0.48a2",
      "0.0.48b1.dev1",
      "0.0.48b1",
      "0.0.48rc1.dev0",
      "0.0.48rc1",
      "0.0.48",
      "0.0.48.post0.dev0",
      "0.0.48.post0",
      "0.0.48.post1.dev0",
      "0.0.48.post1",
      "0.0.48.0.1",
      "0.0.48.1.dev0",
      "0.0.48.1",
      "0.0.48.1.1",
      "0.0.48.1.2a1",
      "0.0.48.1.2",
      "0.0.48.2",
      "0.0.49",
      "1!0.0.1.dev0",
      "1!0.0.1",
  };
  std::vector<Version> parsed;
  parsed.reserve(ordered_inputs.size());
  for (const std::string_view input : ordered_inputs) {
    TT_ASSERT_OK_AND_ASSIGN(const Version v, Version::Parse(input));
    parsed.push_back(v);
  }
  for (size_t i = 0; i < parsed.size(); ++i) {
    EXPECT_EQ(parsed[i], parsed[i]) << ordered_inputs[i];
    for (size_t j = i + 1; j < parsed.size(); ++j) {
      EXPECT_LT(parsed[i], parsed[j])
          << ordered_inputs[i] << " vs " << ordered_inputs[j];
      EXPECT_GT(parsed[j], parsed[i])
          << ordered_inputs[j] << " vs " << ordered_inputs[i];
    }
  }
}

TEST(VersionTest, ExtraTrailingZeroReleaseComponentsCompareEqual) {
  EXPECT_THAT(Version::Parse("1.2.3.4.0"), IsOkAndHolds(Version(1, 2, 3, 4)));
  EXPECT_THAT(Version::Parse("1.2.3.0.0.0"), IsOkAndHolds(Version(1, 2, 3)));
  TT_ASSERT_OK_AND_ASSIGN(const Version five_part, Version::Parse("1.2.3.4.1"));
  EXPECT_THAT(Version::Parse("1.2.3.4.1.0"), IsOkAndHolds(five_part));
  EXPECT_THAT(Version::Parse("1.2.3.4.0.0.0.0.0"),
              IsOkAndHolds(Version(1, 2, 3, 4)));
}

TEST(VersionTest, ParseRejectsMalformedVersion) {
  for (const std::string_view version : {
           "",
           "   ",
           "abc",
           "vv1.2.3",
           "1..2",
           "1.2.",
           ".1.2",
           "-1.0.0",
           "1.0.0-",
           "1.0.0-invalid",
           "1.0.0+",
           "1.0.0+a..b",
           "1.0.0+a$b",
           "1.0.0.dev1a1",
           "1.2.3.4.5.6.7.8.9",
       }) {
    EXPECT_THAT(Version::Parse(version).status(),
                StatusIs(error::kInvalidArgument))
        << version;
  }
}

TEST(VersionTest, ParseRejectsOverflow) {
  for (const std::string_view version : {
           "99999999999!1.0.0",
           "99999999999.0.0",
           "0.99999999999.0",
           "0.0.99999999999",
           "0.0.0.99999999999",
           "0.0.0.0.99999999999",
           "0.0.50a99999999999999999999",
           "0.0.50.post99999999999999999999",
           "0.0.50.dev99999999999999999999",
       }) {
    EXPECT_THAT(Version::Parse(version).status(),
                StatusIs(error::kInvalidArgument))
        << version;
  }
}

TEST(VersionTest, Stringify) {
  EXPECT_EQ(absl::StrCat(Version(2, 3, 15)), "2.3.15");
  EXPECT_EQ(absl::StrCat(Version(0, 0, 50, 0)), "0.0.50");
  EXPECT_EQ(absl::StrCat(Version(0, 0, 50, 1)), "0.0.50.1");
  EXPECT_EQ(absl::StrCat(Version(0, 0, 47, Version::Dev(20260824))),
            "0.0.47.dev20260824");
  EXPECT_EQ(absl::StrCat(Version(0, 0, 50, 1, Version::Dev(20261026))),
            "0.0.50.1.dev20261026");
  TT_ASSERT_OK_AND_ASSIGN(const Version rc_version,
                          Version::Parse("0.0.48rc1"));
  EXPECT_EQ(absl::StrCat(rc_version), "0.0.48rc1");
  TT_ASSERT_OK_AND_ASSIGN(const Version epoch_post_dev,
                          Version::Parse("1!0.0.50.post2.dev3"));
  EXPECT_EQ(absl::StrCat(epoch_post_dev), "1!0.0.50.post2.dev3");
  TT_ASSERT_OK_AND_ASSIGN(const Version five_part, Version::Parse("1.2.3.4.1"));
  EXPECT_EQ(absl::StrCat(five_part), "1.2.3.4.1");
  TT_ASSERT_OK_AND_ASSIGN(const Version five_part_zero_subpatch,
                          Version::Parse("1.2.3.0.1"));
  EXPECT_EQ(absl::StrCat(five_part_zero_subpatch), "1.2.3.0.1");
}

}  // namespace
}  // namespace torch_tpu
