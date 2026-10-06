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

#include "csrc/common/libtpu_version.h"

#include <optional>

#include "csrc/common/error_utils.h"
#include "csrc/common/version.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

class LibtpuVersionTest : public testing::Test {
 protected:
  void SetUp() override { ResetLibtpuVersionForTesting(); }
  void TearDown() override { ResetLibtpuVersionForTesting(); }
};

TEST_F(LibtpuVersionTest, InitialValueIsUnsetInFreshProcess) {
  EXPECT_EQ(GetLibtpuVersionString(), std::nullopt);
  EXPECT_EQ(GetLibtpuVersion(), Version::Max());
}

TEST_F(LibtpuVersionTest, SettingDifferentVersionThrows) {
  SetLibtpuVersion("0.1.0");
  EXPECT_THROW(SetLibtpuVersion("0.2.0"), torch_tpu::TtError);
}

TEST_F(LibtpuVersionTest, SettingInvalidVersionThrows) {
  EXPECT_THROW(SetLibtpuVersion("0.1.2.3.4"), torch_tpu::TtError);
}

TEST_F(LibtpuVersionTest, SetAndGetVersion) {
  SetLibtpuVersion("0.1.0");
  EXPECT_EQ(GetLibtpuVersionString(), "0.1.0");
  EXPECT_EQ(GetLibtpuVersion(), Version(0, 1, 0));

  // Re-setting to identical version should succeed.
  SetLibtpuVersion("0.1.0");
  EXPECT_EQ(GetLibtpuVersionString(), "0.1.0");
  EXPECT_EQ(GetLibtpuVersion(), Version(0, 1, 0));
}

TEST_F(LibtpuVersionTest, CompareVersion) {
  SetLibtpuVersion("0.1.0");
  EXPECT_GE(GetLibtpuVersion(), Version(0, 0, 44));
  EXPECT_GE(GetLibtpuVersion(), Version(0, 0, 99));
  EXPECT_GE(GetLibtpuVersion(), Version(0, 1, 0));
  EXPECT_LT(GetLibtpuVersion(), Version(0, 1, 1));
  EXPECT_LT(GetLibtpuVersion(), Version(0, 2, 0));
}

TEST_F(LibtpuVersionTest, PaddingZerosForFirstThreeComponents) {
  SetLibtpuVersion("0.1");
  EXPECT_EQ(GetLibtpuVersion(), Version(0, 1, 0));
}

TEST_F(LibtpuVersionTest, StableVersionPrecedenceOverNightly) {
  // Stable 0.0.47 has precedence over 0.0.47 nightly pre-releases.
  SetLibtpuVersion("0.0.47");
  EXPECT_GE(GetLibtpuVersion(), Version(0, 0, 47, 20260824));
  EXPECT_GE(GetLibtpuVersion(), Version(0, 0, 47));
  EXPECT_GE(GetLibtpuVersion(), Version(0, 0, 46));
  EXPECT_LT(GetLibtpuVersion(), Version(0, 0, 48));
  EXPECT_LT(GetLibtpuVersion(), Version(0, 0, 48, 20260824));

  // Nightly 0.0.47 does not satisfy requirement for stable 0.0.47.
  ResetLibtpuVersionForTesting();
  SetLibtpuVersion("0.0.47.dev20260824+nightly");
  EXPECT_LT(GetLibtpuVersion(), Version(0, 0, 47));
  EXPECT_GE(GetLibtpuVersion(), Version(0, 0, 47, 20260824));
  EXPECT_GE(GetLibtpuVersion(), Version(0, 0, 47, 20260823));
  EXPECT_LT(GetLibtpuVersion(), Version(0, 0, 47, 20260825));
  EXPECT_GE(GetLibtpuVersion(), Version(0, 0, 46));
}

TEST_F(LibtpuVersionTest, TimestampDevVersion) {
  SetLibtpuVersion("0.0.47.dev202605201400");
  EXPECT_EQ(GetLibtpuVersion(), Version(0, 0, 47, 202605201400));
}

TEST_F(LibtpuVersionTest, SetEmptyVersion) {
  SetLibtpuVersion("");
  EXPECT_EQ(GetLibtpuVersionString(), "");
  EXPECT_EQ(GetLibtpuVersion(), Version::Max());
}

TEST_F(LibtpuVersionTest, ResetLibtpuVersionForTesting) {
  SetLibtpuVersion("0.1.0");
  EXPECT_EQ(GetLibtpuVersionString(), "0.1.0");
  ResetLibtpuVersionForTesting();
  EXPECT_EQ(GetLibtpuVersionString(), std::nullopt);
  EXPECT_EQ(GetLibtpuVersion(), Version::Max());
}

}  // namespace
}  // namespace torch_tpu
