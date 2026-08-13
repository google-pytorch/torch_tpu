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

#include "torch_tpu/runtime/eager_runtime_adapter.h"

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
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/mock_backend.h"

namespace torch_tpu {
namespace {

class EagerRuntimeAdapterTest : public testing::Test {
 protected:
  void SetUp() override {
    Backend::SetInstance(std::make_unique<MockBackend>());
  }

  void TearDown() override { Backend::ResetInstance(); }
};

TEST_F(EagerRuntimeAdapterTest, CreateBufferAndTensorRoundtrip) {
  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};  // INT_VEC_OK

  ASSERT_OK_AND_ASSIGN(auto buffer,
                       EagerRuntimeAdapter::CreateBufferFromHost(
                           host_data.data(), at::ScalarType::Float, shape));
  ASSERT_NE(buffer, nullptr);

  at::Tensor tensor =
      EagerRuntimeAdapter::CreateTensorFromBuffer(std::move(buffer));
  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 2);

  DeviceBuffer* extracted_buf =
      EagerRuntimeAdapter::ExtractBufferFromTensor(tensor);
  ASSERT_NE(extracted_buf, nullptr);
  EXPECT_EQ(extracted_buf->dtype(), at::ScalarType::Float);
}

TEST_F(EagerRuntimeAdapterTest, DeviceQueryAndSynchronization) {
  Device* default_dev = EagerRuntimeAdapter::GetDefaultDevice();
  EXPECT_NE(default_dev, nullptr);

  ASSERT_OK_AND_ASSIGN(int global_id, EagerRuntimeAdapter::GetGlobalDeviceId());
  EXPECT_EQ(global_id, 0);

  ASSERT_OK_AND_ASSIGN(int global_count,
                       EagerRuntimeAdapter::GetGlobalDeviceCount());
  EXPECT_GT(global_count, 0);

  EXPECT_TRUE(EagerRuntimeAdapter::SynchronizeDevice(0).ok());
  EXPECT_TRUE(EagerRuntimeAdapter::SynchronizeAllDevices().ok());
}

TEST_F(EagerRuntimeAdapterTest, AllocatorStatsOperations) {
  ASSERT_OK_AND_ASSIGN(auto stats, EagerRuntimeAdapter::GetAllocatorStats());
  EXPECT_GE(stats.bytes_in_use, 0);

  EXPECT_TRUE(EagerRuntimeAdapter::ClearAllocatorStats().ok());
}

}  // namespace
}  // namespace torch_tpu
