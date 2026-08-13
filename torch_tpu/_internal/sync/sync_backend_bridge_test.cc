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

#include "torch_tpu/_internal/sync/sync_backend_bridge.h"

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/backend_config.h"

namespace torch_tpu {
namespace {

class SyncBackendBridgeTest : public testing::Test {
 protected:
  void SetUp() override {
    Backend::ResetInstance();
    SetGlobalBackendKind(BackendKind::kCerebro);
  }

  void TearDown() override {
    Backend::ResetInstance();
    ResetGlobalBackendConfig();
  }
};

TEST_F(SyncBackendBridgeTest, SynchronizeAllDevicesOnCerebro) {
  EXPECT_TRUE(SyncBackendBridge::SyncAllDevices().ok());
}

TEST_F(SyncBackendBridgeTest, ClusterBarrierOnCerebro) {
  EXPECT_TRUE(SyncBackendBridge::ClusterBarrier().ok());
}

TEST_F(SyncBackendBridgeTest, GetMemoryStatsOnCerebro) {
  ASSERT_OK_AND_ASSIGN(auto stats, SyncBackendBridge::GetMemoryStats());
  EXPECT_GE(stats.bytes_in_use, 0);
}

}  // namespace
}  // namespace torch_tpu
