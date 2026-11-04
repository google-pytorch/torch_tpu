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

#include <string_view>

#include "absl/status/status_matchers.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using testing::HasSubstr;

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

TEST(DeviceUtilsTest, SelectLocalDeviceReturnsTheOnlyListedDevice) {
  EXPECT_THAT(SelectLocalDevice("3", 0), IsOkAndHolds("3"));
  EXPECT_THAT(SelectLocalDevice("0", 2), IsOkAndHolds("0"));
}

TEST(DeviceUtilsTest, SelectLocalDevicePicksListedDeviceAtLocalRank) {
  EXPECT_THAT(SelectLocalDevice("0,1", 0), IsOkAndHolds("0"));
  EXPECT_THAT(SelectLocalDevice("0,1,2,3", 2), IsOkAndHolds("2"));
  EXPECT_THAT(SelectLocalDevice("4,5,6,7", 0), IsOkAndHolds("4"));
  EXPECT_THAT(SelectLocalDevice("4,5,6,7", 2), IsOkAndHolds("6"));
}

TEST(DeviceUtilsTest, SelectLocalDeviceReturnsLocalRankIfAllDevicesAreVisible) {
  EXPECT_THAT(SelectLocalDevice("", 0), IsOkAndHolds("0"));
  EXPECT_THAT(SelectLocalDevice("", 2), IsOkAndHolds("2"));
}

TEST(DeviceUtilsTest, SelectLocalDeviceFailsIfLocalRankIsNegative) {
  for (std::string_view visible_devices : {"", "3", "4,5"}) {
    EXPECT_THAT(SelectLocalDevice(visible_devices, -1),
                StatusIs(error::kInvalidArgument,
                         HasSubstr("expected local rank to be >= 0")))
        << visible_devices;
  }
}

TEST(DeviceUtilsTest, SelectLocalDeviceFailsIfVisibleDevicesIsMalformed) {
  for (std::string_view visible_devices :
       {"-1", "invalid", "4,a", "4,,5", "4, 5", "0;1"}) {
    EXPECT_THAT(SelectLocalDevice(visible_devices, 0),
                StatusIs(error::kInvalidArgument,
                         HasSubstr("expected visible devices to be a list "
                                   "of comma-separated integers >= 0")))
        << visible_devices;
  }
}

TEST(DeviceUtilsTest, SelectLocalDeviceFailsIfTooFewDevicesAreListed) {
  EXPECT_THAT(SelectLocalDevice("4,5", 2),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("out of range for visible devices")));
}

}  // namespace
}  // namespace torch_tpu
