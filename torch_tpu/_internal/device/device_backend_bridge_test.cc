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

#include "torch_tpu/_internal/device/device_backend_bridge.h"

#include <vector>

#include "absl/cleanup/cleanup.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/backend_config.h"

namespace torch_tpu {
namespace {

class DeviceBackendBridgeTest : public testing::Test {
 protected:
  void SetUp() override { Backend::ResetInstance(); }

  void TearDown() override {
    Backend::ResetInstance();
    ResetGlobalBackendConfig();
  }
};

TEST_F(DeviceBackendBridgeTest, QueryCerebroBackendDevicesAndTopology) {
  SetGlobalBackendKind(BackendKind::kCerebro);
  EXPECT_EQ(DeviceBackendBridge::GetActiveBackendKind(), "cerebro");
  EXPECT_GT(DeviceBackendBridge::GetDeviceCount(), 0);

  ASSERT_OK_AND_ASSIGN(int global_count,
                       DeviceBackendBridge::GetGlobalDeviceCount());
  EXPECT_GT(global_count, 0);

  auto devices = DeviceBackendBridge::GetAddressableDevices();
  ASSERT_FALSE(devices.empty());
  EXPECT_EQ(devices[0].kind, "TPU v5e");
  EXPECT_EQ(devices[0].coords.size(), 3);
}

TEST_F(DeviceBackendBridgeTest, QueryActiveBackendKindDefault) {
  EXPECT_EQ(DeviceBackendBridge::GetActiveBackendKind(), "cerebro");
}

}  // namespace
}  // namespace torch_tpu
