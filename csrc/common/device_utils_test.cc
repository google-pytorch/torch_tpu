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

}  // namespace
}  // namespace torch_tpu
