// Copyright 2026 Google LLC
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

#include "csrc/common/device_utils.h"

#include <stdlib.h>
#include <unistd.h>

#include <cstdlib>
#include <string>

#include "absl/log/absl_check.h"
#include "csrc/common/env_vars.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(DeviceUtilsTest, IsSingleDeviceSpecifiedAcceptsValidSingleIndices) {
  EXPECT_TRUE(IsSingleDeviceSpecified("0"));
  EXPECT_TRUE(IsSingleDeviceSpecified("1"));
  EXPECT_TRUE(IsSingleDeviceSpecified("7"));
  EXPECT_TRUE(IsSingleDeviceSpecified("42"));
}

TEST(DeviceUtilsTest, IsSingleDeviceSpecifiedRejectsInvalidOrMultipleDevices) {
  EXPECT_FALSE(IsSingleDeviceSpecified(""));
  EXPECT_FALSE(IsSingleDeviceSpecified("-1"));
  EXPECT_FALSE(IsSingleDeviceSpecified("0,1"));
  EXPECT_FALSE(IsSingleDeviceSpecified("0;1"));
  EXPECT_FALSE(IsSingleDeviceSpecified("0 1"));
  EXPECT_FALSE(IsSingleDeviceSpecified(" 0"));
  EXPECT_FALSE(IsSingleDeviceSpecified("0 "));
  EXPECT_FALSE(IsSingleDeviceSpecified("abc"));
}

TEST(DeviceUtilsTest, GetSingleDeviceOrDefaultReturnsConfiguredOrDefault) {
  EXPECT_EQ(GetSingleDeviceOrDefault("3", "0"), "3");
  EXPECT_EQ(GetSingleDeviceOrDefault("0", "2"), "0");
  EXPECT_EQ(GetSingleDeviceOrDefault("", "0"), "0");
  EXPECT_EQ(GetSingleDeviceOrDefault("0,1", "0"), "0");
  EXPECT_EQ(GetSingleDeviceOrDefault("0,1,2,3", "2"), "2");
  EXPECT_EQ(GetSingleDeviceOrDefault("-1", "0"), "0");
  EXPECT_EQ(GetSingleDeviceOrDefault("invalid", "0"), "0");
}

TEST(DeviceUtilsTest, GetChipsPerHostBoundsMatchesTopologyRank) {
  EXPECT_EQ(GetChipsPerHostBounds("2,2,1"), "1,1,1");
  EXPECT_EQ(GetChipsPerHostBounds("2,2,1,2"), "1,1,1,1");
  EXPECT_EQ(GetChipsPerHostBounds(""), "1,1,1");
}

// Returns `args` after SetLibtpuInitFlag(&args, "foo", value).
[[nodiscard]] std::string WithFooFlag(std::string args, bool value) {
  SetLibtpuInitFlag(&args, "foo", value);
  return args;
}

TEST(DeviceUtilsTest, SetLibtpuInitFlagAppendsMissingFlag) {
  EXPECT_EQ(WithFooFlag("", true), "--foo=true");
  EXPECT_EQ(WithFooFlag("--bar=1", false), "--bar=1 --foo=false");
  EXPECT_EQ(WithFooFlag("--foo_bar=1", true), "--foo_bar=1 --foo=true");
}

TEST(DeviceUtilsTest, SetLibtpuInitFlagReplacesExistingFlag) {
  EXPECT_EQ(WithFooFlag("--foo=false", false), "--foo=false");
  EXPECT_EQ(WithFooFlag("--foo=true --bar", false), "--foo=false --bar");
  EXPECT_EQ(WithFooFlag("-foo=0", true), "--foo=true");
  EXPECT_EQ(WithFooFlag("--foo", false), "--foo=false");
  EXPECT_EQ(WithFooFlag("--nofoo", true), "--foo=true");
}

TEST(DeviceUtilsDeathTest, ConfigureDistributedTopologyEnvSetsEnvVars) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "0,1", /*overwrite=*/1);
        ConfigureDistributedTopologyEnv(/*rank=*/3, /*default_device=*/"1",
                                        /*topology=*/"2,1,1,2",
                                        /*sb_addrs=*/"h0:1,h1:1",
                                        /*sb_port=*/"1");
        ABSL_CHECK_STREQ(std::getenv(kCloudTpuTaskIdEnvVar), "1");
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "1");
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "1");
        ABSL_CHECK_STREQ(std::getenv(kTpuHostBoundsEnvVar), "2,1,1,2");
        ABSL_CHECK_STREQ(std::getenv(kTpuProcessBoundsEnvVar), "2,1,1,2");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar), "1,1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar),
                         "1,1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuProcessPortEnvVar), "1");
        ABSL_CHECK_STREQ(std::getenv(kTpuProcessAddressesEnvVar), "h0:1,h1:1");
        ABSL_CHECK_STREQ(std::getenv(kAllowMultipleLibtpuLoadEnvVar), "1");
        _exit(0);
      },
      testing::ExitedWithCode(0), "");
}

}  // namespace
}  // namespace torch_tpu
