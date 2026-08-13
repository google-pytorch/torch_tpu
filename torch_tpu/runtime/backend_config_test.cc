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

#include "torch_tpu/runtime/backend_config.h"

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/runtime/backend.h"

namespace torch_tpu {
namespace {

class BackendConfigTest : public testing::Test {
 protected:
  void TearDown() override { ResetGlobalBackendConfig(); }
};

TEST_F(BackendConfigTest, ParseBackendKindValid) {
  ASSERT_OK_AND_ASSIGN(auto pjrt, ParseBackendKind("pjrt"));
  EXPECT_EQ(pjrt, BackendKind::kPjrt);

  ASSERT_OK_AND_ASSIGN(auto mock, ParseBackendKind("mock"));
  EXPECT_EQ(mock, BackendKind::kMock);

  ASSERT_OK_AND_ASSIGN(auto cerebro, ParseBackendKind("cerebro"));
  EXPECT_EQ(cerebro, BackendKind::kCerebro);

  EXPECT_EQ(BackendKindToString(BackendKind::kPjrt), "pjrt");
  EXPECT_EQ(BackendKindToString(BackendKind::kMock), "mock");
  EXPECT_EQ(BackendKindToString(BackendKind::kCerebro), "cerebro");
}

TEST_F(BackendConfigTest, ParseBackendKindInvalid) {
  auto invalid = ParseBackendKind("invalid_backend_kind");
  EXPECT_FALSE(invalid.ok());
}

TEST_F(BackendConfigTest, SetAndGetGlobalBackendConfig) {
  RuntimeBackendConfig config;
  config.default_backend = BackendKind::kMock;
  config.cluster_name = "test_cluster";
  config.num_nodes = 8;
  config.chips_per_node = 4;

  SetGlobalBackendConfig(config);
  auto retrieved = GetGlobalBackendConfig();
  EXPECT_EQ(retrieved.default_backend, BackendKind::kMock);
  EXPECT_EQ(retrieved.cluster_name, "test_cluster");
  EXPECT_EQ(retrieved.num_nodes, 8);
  EXPECT_EQ(retrieved.chips_per_node, 4);

  SetGlobalBackendKind(BackendKind::kPjrt);
  EXPECT_EQ(GetGlobalBackendConfig().default_backend, BackendKind::kPjrt);
}

}  // namespace
}  // namespace torch_tpu
