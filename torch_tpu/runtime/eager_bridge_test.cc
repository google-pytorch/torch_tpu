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

#include "torch_tpu/runtime/eager_bridge.h"

#include <memory>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/mock_backend.h"

namespace torch_tpu {
namespace {

TEST(EagerBridgeTest, WrapAndUnwrapDeviceBuffer) {
  MockDevice dev(0);
  auto mock_buf = std::make_unique<MockDeviceBuffer>(at::ScalarType::Float,
                                                     Dimensions{2, 3}, &dev);
  DeviceBuffer* raw_buf = mock_buf.get();

  at::Tensor tensor = CreateAtTensorFromTtrBuffer(std::move(mock_buf));
  ASSERT_TRUE(tensor.defined());

  EXPECT_EQ(tensor.scalar_type(), at::ScalarType::Float);
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.stride(0), 3);
  EXPECT_EQ(tensor.stride(1), 1);

  DeviceBuffer* retrieved = GetTtrBufferFromAtTensor(tensor);
  EXPECT_EQ(retrieved, raw_buf);
}

TEST(EagerBridgeTest, InPlaceAssignBufferToTensor) {
  MockDevice dev(0);
  auto buf1 = std::make_unique<MockDeviceBuffer>(at::ScalarType::Float,
                                                 Dimensions{2, 2}, &dev);

  at::Tensor tensor = CreateAtTensorFromTtrBuffer(std::move(buf1));
  EXPECT_EQ(tensor.size(0), 2);

  auto buf2 = std::make_unique<MockDeviceBuffer>(at::ScalarType::Float,
                                                 Dimensions{4, 4}, &dev);
  DeviceBuffer* raw_buf2 = buf2.get();

  absl::Status status = AssignTtrBufferToAtTensor(tensor, std::move(buf2));
  EXPECT_TRUE(status.ok());

  EXPECT_EQ(tensor.size(0), 4);
  EXPECT_EQ(tensor.size(1), 4);
  EXPECT_EQ(GetTtrBufferFromAtTensor(tensor), raw_buf2);
}

TEST(EagerBridgeTest, ExtractBufferOwnership) {
  MockDevice dev(0);
  auto mock_buf = std::make_unique<MockDeviceBuffer>(at::ScalarType::Float,
                                                     Dimensions{1, 4}, &dev);
  DeviceBuffer* raw_buf = mock_buf.get();

  at::Tensor tensor = CreateAtTensorFromTtrBuffer(std::move(mock_buf));
  EXPECT_EQ(GetTtrBufferFromAtTensor(tensor), raw_buf);

  auto extracted = ExtractTtrBufferFromAtTensor(tensor);
  EXPECT_EQ(extracted.get(), raw_buf);
  EXPECT_FALSE(tensor.defined());
}

TEST(EagerBridgeTest, ScalarTensorAndNullChecks) {
  at::Tensor empty_tensor;  // UNINITIALIZED_TENSOR_OK
  EXPECT_EQ(GetTtrBufferFromAtTensor(empty_tensor), nullptr);
  EXPECT_EQ(ExtractTtrBufferFromAtTensor(empty_tensor), nullptr);
  EXPECT_FALSE(CreateAtTensorFromTtrBuffer(nullptr).defined());

  MockDevice dev(0);
  auto scalar_buf = std::make_unique<MockDeviceBuffer>(at::ScalarType::Int,
                                                       Dimensions{}, &dev);
  at::Tensor scalar_tensor = CreateAtTensorFromTtrBuffer(std::move(scalar_buf));
  ASSERT_TRUE(scalar_tensor.defined());
  EXPECT_EQ(scalar_tensor.scalar_type(), at::ScalarType::Int);
  EXPECT_EQ(scalar_tensor.dim(), 0);
  EXPECT_EQ(scalar_tensor.numel(), 1);
}

}  // namespace
}  // namespace torch_tpu
