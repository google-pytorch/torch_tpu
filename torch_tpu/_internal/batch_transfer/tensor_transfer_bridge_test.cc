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

#include "torch_tpu/_internal/batch_transfer/tensor_transfer_bridge.h"

#include "ATen/core/TensorBody.h"
#include "ATen/ops/ones.h"
#include "ATen/ops/zeros.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/backend_config.h"

namespace torch_tpu {
namespace {

class TensorTransferBridgeTest : public testing::Test {
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

TEST_F(TensorTransferBridgeTest, SingleTensorHostDeviceRoundtrip) {
  at::Tensor host_tensor =
      at::ones({3, 3}, at::TensorOptions().dtype(at::ScalarType::Float));
  ASSERT_OK_AND_ASSIGN(auto device_tensor,
                       TensorTransferBridge::TransferHostToDevice(host_tensor));
  EXPECT_TRUE(device_tensor.defined());
  EXPECT_EQ(device_tensor.dim(), 2);
  EXPECT_EQ(device_tensor.size(0), 3);

  ASSERT_OK_AND_ASSIGN(
      auto roundtrip_host,
      TensorTransferBridge::TransferDeviceToHost(device_tensor));
  EXPECT_TRUE(roundtrip_host.defined());
  EXPECT_EQ(roundtrip_host.dim(), 2);
}

TEST_F(TensorTransferBridgeTest, BatchTransferMultipleTensors) {
  at::Tensor t1 =
      at::ones({2, 2}, at::TensorOptions().dtype(at::ScalarType::Float));
  at::Tensor t2 =
      at::zeros({4, 4}, at::TensorOptions().dtype(at::ScalarType::Float));
  std::vector<at::Tensor> host_tensors = {t1, t2};

  ASSERT_OK_AND_ASSIGN(
      auto batch_res,
      TensorTransferBridge::BatchTransferHostToDevice(host_tensors));
  EXPECT_EQ(batch_res.size(), 2);
  EXPECT_EQ(batch_res[0].size(0), 2);
  EXPECT_EQ(batch_res[1].size(0), 4);
}

}  // namespace
}  // namespace torch_tpu
