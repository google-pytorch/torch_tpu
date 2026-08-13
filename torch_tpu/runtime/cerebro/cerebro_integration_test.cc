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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "c10/core/ScalarType.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/backend_config.h"
#include "torch_tpu/runtime/cerebro/cerebro_backend_impl.h"
#include "torch_tpu/runtime/cerebro/cerebro_cluster_pipeline.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/eager_runtime_adapter.h"

namespace torch_tpu {
namespace {

class CerebroIntegrationTest : public testing::Test {
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

TEST_F(CerebroIntegrationTest, VerifyCerebroBackendCreationAndInstance) {
  Backend* instance = Backend::GetInstance();
  ASSERT_NE(instance, nullptr);
  EXPECT_NE(dynamic_cast<CerebroBackendImpl*>(instance), nullptr);
  EXPECT_GT(instance->GetDevices().size(), 0);

  ASSERT_OK_AND_ASSIGN(int global_count, instance->GetGlobalDeviceCount());
  EXPECT_GT(global_count, 0);
}

TEST_F(CerebroIntegrationTest, EagerTensorRoundtripOnCerebro) {
  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};  // INT_VEC_OK

  ASSERT_OK_AND_ASSIGN(auto buffer,
                       EagerRuntimeAdapter::CreateBufferFromHost(
                           host_data.data(), at::ScalarType::Float, shape));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->dtype(), at::ScalarType::Float);

  at::Tensor tensor =
      EagerRuntimeAdapter::CreateTensorFromBuffer(std::move(buffer));
  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 2);

  DeviceBuffer* extracted =
      EagerRuntimeAdapter::ExtractBufferFromTensor(tensor);
  ASSERT_NE(extracted, nullptr);
  EXPECT_EQ(extracted->dtype(), at::ScalarType::Float);
}

TEST_F(CerebroIntegrationTest, CerebroClusterPipelineOperations) {
  Backend* backend = Backend::GetInstance();
  ASSERT_NE(backend, nullptr);

  auto* cerebro_impl = dynamic_cast<CerebroBackendImpl*>(backend);
  ASSERT_NE(cerebro_impl, nullptr);

  CerebroClusterPipeline pipeline(cerebro_impl);
  EXPECT_GT(pipeline.GetClusterDeviceCount(), 0);

  ASSERT_OK_AND_ASSIGN(auto coords, pipeline.GetDeviceCoordinate(0));
  EXPECT_EQ(coords.size(), 3);

  EXPECT_TRUE(pipeline.Barrier().ok());
}

}  // namespace
}  // namespace torch_tpu
