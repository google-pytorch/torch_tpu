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

#include "torch_tpu/_internal/compile/compile_backend_bridge.h"

#include <memory>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/ones.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/_internal/batch_transfer/tensor_transfer_bridge.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/backend_config.h"

namespace torch_tpu {
namespace {

constexpr const char* kSimpleModule = R"mlir(
module @jit_computation {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
)mlir";

class CompileBackendBridgeTest : public testing::Test {
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

TEST_F(CompileBackendBridgeTest, CompileAndExecuteOnCerebro) {
  ASSERT_OK_AND_ASSIGN(auto executable,
                       CompileBackendBridge::Compile(kSimpleModule));
  ASSERT_NE(executable, nullptr);

  at::Tensor host_tensor =
      at::ones({4}, at::TensorOptions().dtype(at::ScalarType::Float));
  ASSERT_OK_AND_ASSIGN(auto device_tensor,
                       TensorTransferBridge::TransferHostToDevice(host_tensor));

  std::vector<at::Tensor> inputs = {device_tensor};
  ASSERT_OK_AND_ASSIGN(auto outputs, CompileBackendBridge::ExecuteWithTensors(
                                         executable.get(), inputs));
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].size(0), 4);
}

}  // namespace
}  // namespace torch_tpu
